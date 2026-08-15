"""Deterministic, profile-bound evidence for the public-text release.

The release gate writes detailed reports that contain execution-local fields such
as timestamps and latency.  This module extracts only stable, reviewable metrics
and binds them to the exact public-text index and metadata hashes.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any


SUMMARY_SCHEMA_VERSION = 1
SUMMARY_PROFILE = "public-text-only-v1"
SUMMARY_JSON_NAME = "public-text-only.summary.json"
SUMMARY_MARKDOWN_NAME = "public-text-only.summary.md"
MAX_REPORT_BYTES = 8 * 1024 * 1024
HEX_64_RE = re.compile(r"[0-9a-f]{64}")

RAW_REPORT_NAMES = {
    "retrieval": "retrieval.json",
    "robustness": "query-robustness.json",
    "exposure": "exposure.json",
    "profile": "profile.json",
    "safety": "safety.json",
}

RETRIEVAL_METRIC_KEYS = (
    "query_count",
    "precision@5",
    "recall@5",
    "recall@10",
    "nDCG@5",
    "MRR",
    "hit@5",
    "inactive_exposure",
)
ROBUSTNESS_METRIC_KEYS = RETRIEVAL_METRIC_KEYS
ROBUSTNESS_STABILITY_KEYS = (
    "variant_count",
    "mean_top5_jaccard",
    "mean_top10_jaccard",
    "mean_rbo@10",
    "min_top5_jaccard",
    "min_top10_jaccard",
    "min_rbo@10",
    "top10_exact_match_rate",
)
EXPOSURE_KEYS = (
    "applicable_count",
    "evaluated_count",
    "unknown_count",
    "evidence_coverage",
)
PROFILE_RESULT_KEYS = (
    "profile_count",
    "profiles_with_different_top1_from_baseline",
    "distinct_profile_top1_count",
    "unknown_neutrality_failures",
    "evidence_consistency_failures",
    "adoption_suitability_certainty_phrase_hits",
    "quality_sensitivity_failures",
)
SAFETY_RESULT_KEYS = (
    "total_contracts",
    "passed_contracts",
    "failed_contracts",
)

NOT_APPLICABLE = (
    "독립 query holdout v2/v3는 full 멀티모달 artifact에 고정돼 이 프로필에는 적용하지 않으며 PASS로 세지 않습니다.",
    "이 프로필에는 image/crop vector가 없으므로 held-out image retrieval은 N/A이며 PASS로 세지 않습니다.",
)
LIMITATIONS = (
    "검색 relevance는 공고의 명시 필드로 만든 결정적 silver label이며 사람의 입양 선호 평가가 아닙니다.",
    "프로필 재정렬 결과는 고정 계약·민감도 시험이며 입양 적합성이나 최적 가중치의 증명이 아닙니다.",
    "같은 tag의 증거를 결정적으로 유지하기 위해 latency, runtime, 생성 시각과 로컬 경로는 요약에서 제외합니다.",
    "공고 상태는 고정 기준일 이후 바뀔 수 있으므로 연락 전 원문 공고에서 다시 확인해야 합니다.",
)
REPRODUCTION_COMMANDS = (
    "$referenceDate = (Get-Content docs/evaluation/public-text-only.summary.json | ConvertFrom-Json).reference_date",
    "python scripts/evaluate_retrieval.py --index data/dog_faiss.index --metas data/dog_metas.json --json-out dist/public-text-evidence/retrieval.json --markdown-out dist/public-text-evidence/retrieval.md --reference-date $referenceDate --device cpu --warmup 0",
    "python scripts/evaluate_query_robustness.py --index data/dog_faiss.index --metas data/dog_metas.json --json-out dist/public-text-evidence/query-robustness.json --markdown-out dist/public-text-evidence/query-robustness.md --reference-date $referenceDate --device cpu --no-warmup",
    "python scripts/evaluate_exposure_representation.py --metas data/dog_metas.json --retrieval-report dist/public-text-evidence/retrieval.json --json-out dist/public-text-evidence/exposure.json --markdown-out dist/public-text-evidence/exposure.md",
    "python scripts/evaluate_profile_reranking.py --metas data/dog_metas.json --json-out dist/public-text-evidence/profile.json --markdown-out dist/public-text-evidence/profile.md",
    "python scripts/evaluate_safety_contract.py --metas data/dog_metas.json --json-out dist/public-text-evidence/safety.json --markdown-out dist/public-text-evidence/safety.md",
    "python scripts/build_public_text_evaluation_summary.py --reports-dir dist/public-text-evidence --index data/dog_faiss.index --metas data/dog_metas.json --profile-marker data/release_profile.json",
)

_TOP_LEVEL_KEYS = {
    "artifacts",
    "evaluation_inputs",
    "label_policy",
    "limitations",
    "not_applicable",
    "profile",
    "reference_date",
    "reproduction_commands",
    "results",
    "schema_version",
}
_ARTIFACT_KEYS = {
    "dimension",
    "index_sha256",
    "metas_sha256",
    "unique_notice_count",
    "vector_count",
}
_INPUT_KEYS = {
    "profile_config_sha256",
    "retrieval_queries_sha256",
    "robustness_base_queries_sha256",
    "robustness_variants_sha256",
    "safety_config_sha256",
}
_LABEL_POLICY_KEYS = {
    "relevance",
    "source_fields",
    "uses_ranked_results",
    "uses_result_text",
    "uses_vlm_attributes",
    "version",
}
_RESULT_KEYS = {
    "exposure_representation",
    "profile_reranking",
    "query_robustness",
    "retrieval",
    "safety_contract",
}
_VOLATILE_KEY_PARTS = ("latency", "runtime", "duration", "elapsed")
_VOLATILE_KEYS = {"generated_at", "generated_timestamp", "machine_path"}


class PublicTextEvaluationSummaryError(ValueError):
    """Detailed reports cannot produce safe public-profile evidence."""


@dataclass(frozen=True)
class PublicTextEvaluationSummaryArtifacts:
    """Canonical JSON and its deterministically rendered human summary."""

    payload: Mapping[str, Any]
    json_bytes: bytes
    markdown_bytes: bytes


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PublicTextEvaluationSummaryError(f"{label} must be an object")
    return value


def _nested(value: Mapping[str, Any], *keys: str, label: str) -> Any:
    current: Any = value
    for key in keys:
        current = _mapping(current, label=label).get(key)
    if current is None:
        raise PublicTextEvaluationSummaryError(f"{label} is missing")
    return current


def _read_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        size = path.stat().st_size
        if size <= 0 or size > MAX_REPORT_BYTES:
            raise PublicTextEvaluationSummaryError(f"{label} size is invalid")
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8-sig"))
    except PublicTextEvaluationSummaryError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicTextEvaluationSummaryError(f"{label} cannot be read") from exc
    return _mapping(value, label=label)


def _sha256_file(path: Path, *, label: str) -> str:
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise PublicTextEvaluationSummaryError(f"{label} cannot be read") from exc
    return digest


def _sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or HEX_64_RE.fullmatch(value) is None:
        raise PublicTextEvaluationSummaryError(f"{label} is not a SHA-256")
    return value


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise PublicTextEvaluationSummaryError(f"{label} is not a valid integer")
    return value


def _number(value: Any, *, label: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PublicTextEvaluationSummaryError(f"{label} is not numeric")
    if not math.isfinite(float(value)):
        raise PublicTextEvaluationSummaryError(f"{label} is not finite")
    return value


def _selected_numbers(
    value: Any,
    keys: Sequence[str],
    *,
    label: str,
) -> dict[str, int | float]:
    source = _mapping(value, label=label)
    return {key: _number(source.get(key), label=f"{label}.{key}") for key in keys}


def _passed(value: Any, *, label: str) -> bool:
    if value is not True:
        raise PublicTextEvaluationSummaryError(f"{label} did not pass")
    return True


def _same_reference_date(reports: Mapping[str, Mapping[str, Any]]) -> str:
    values = {
        str(reports["retrieval"].get("reference_date", "")),
        str(reports["robustness"].get("reference_date", "")),
        str(reports["exposure"].get("reference_date", "")),
        str(reports["profile"].get("evaluation_reference_date", "")),
        str(reports["safety"].get("evaluation_reference_date", "")),
    }
    if len(values) != 1:
        raise PublicTextEvaluationSummaryError(
            "evaluation reports use different reference dates"
        )
    value = values.pop()
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise PublicTextEvaluationSummaryError(
            "evaluation reference date is invalid"
        ) from exc
    return value


def _require_hash_binding(
    report_value: Any,
    expected: str,
    *,
    label: str,
) -> None:
    if not isinstance(report_value, str) or report_value != expected:
        raise PublicTextEvaluationSummaryError(
            f"{label} is not bound to the staged public-text artifact"
        )


def _build_payload(
    reports: Mapping[str, Mapping[str, Any]],
    marker: Mapping[str, Any],
    *,
    index_sha256: str,
    metas_sha256: str,
) -> dict[str, Any]:
    marker_profile = marker.get("profile")
    if marker_profile != SUMMARY_PROFILE:
        raise PublicTextEvaluationSummaryError("release profile marker is invalid")
    output = _mapping(marker.get("output"), label="release profile output")
    _require_hash_binding(
        output.get("index_sha256"), index_sha256, label="profile index SHA-256"
    )
    _require_hash_binding(
        output.get("metas_sha256"), metas_sha256, label="profile metadata SHA-256"
    )

    retrieval = reports["retrieval"]
    robustness = reports["robustness"]
    exposure = reports["exposure"]
    profile = reports["profile"]
    safety = reports["safety"]

    for report, report_name in ((retrieval, "retrieval"), (robustness, "robustness")):
        _require_hash_binding(
            _nested(report, "artifacts", "index", "sha256", label=report_name),
            index_sha256,
            label=f"{report_name} index SHA-256",
        )
        _require_hash_binding(
            _nested(report, "artifacts", "metas", "sha256", label=report_name),
            metas_sha256,
            label=f"{report_name} metadata SHA-256",
        )
    _require_hash_binding(
        _nested(
            exposure,
            "inputs",
            "source_retrieval_artifact_sha256",
            "index",
            label="exposure",
        ),
        index_sha256,
        label="exposure index SHA-256",
    )
    _require_hash_binding(
        _nested(
            exposure,
            "inputs",
            "source_retrieval_artifact_sha256",
            "metas",
            label="exposure",
        ),
        metas_sha256,
        label="exposure metadata SHA-256",
    )
    _require_hash_binding(
        _nested(profile, "inputs", "dog_metas_sha256", label="profile"),
        metas_sha256,
        label="profile metadata SHA-256",
    )
    _require_hash_binding(
        _nested(safety, "inputs", "dog_metas_sha256", label="safety"),
        metas_sha256,
        label="safety metadata SHA-256",
    )

    retrieval_system = _mapping(
        _nested(
            retrieval,
            "systems",
            "hybrid_natural_graph",
            label="retrieval headline system",
        ),
        label="retrieval headline system",
    )
    robustness_summary = _mapping(
        robustness.get("robustness_summary"), label="robustness summary"
    )
    exposure_system = _mapping(
        _nested(
            exposure,
            "systems",
            "hybrid_natural_graph",
            label="exposure headline system",
        ),
        label="exposure headline system",
    )
    profile_summary = _mapping(profile.get("summary"), label="profile summary")
    safety_summary = _mapping(safety.get("summary"), label="safety summary")

    _passed(
        _nested(retrieval, "validation", "passed", label="retrieval validation"),
        label="retrieval validation",
    )
    _passed(
        _nested(robustness, "validation", "passed", label="robustness validation"),
        label="robustness validation",
    )
    _passed(
        _nested(exposure, "validation", "passed", label="exposure validation"),
        label="exposure validation",
    )
    _passed(profile_summary.get("passed"), label="profile validation")
    _passed(safety_summary.get("passed"), label="safety validation")

    label_policy_source = _mapping(
        retrieval.get("label_policy"), label="retrieval label policy"
    )
    source_fields = label_policy_source.get("source_fields")
    if (
        not isinstance(source_fields, list)
        or not source_fields
        or not all(isinstance(item, str) and item for item in source_fields)
    ):
        raise PublicTextEvaluationSummaryError("label policy source fields are invalid")
    label_policy = {
        "relevance": str(label_policy_source.get("relevance", "")),
        "source_fields": list(source_fields),
        "uses_ranked_results": label_policy_source.get("uses_ranked_results"),
        "uses_result_text": label_policy_source.get("uses_result_text"),
        "uses_vlm_attributes": label_policy_source.get("uses_vlm_attributes"),
        "version": str(label_policy_source.get("version", "")),
    }
    if (
        not label_policy["version"]
        or not label_policy["relevance"]
        or any(
            type(label_policy[key]) is not bool
            for key in (
                "uses_ranked_results",
                "uses_result_text",
                "uses_vlm_attributes",
            )
        )
    ):
        raise PublicTextEvaluationSummaryError("retrieval label policy is invalid")

    payload: dict[str, Any] = {
        "artifacts": {
            "dimension": _integer(
                output.get("dimension"), label="profile dimension", minimum=1
            ),
            "index_sha256": index_sha256,
            "metas_sha256": metas_sha256,
            "unique_notice_count": _integer(
                output.get("unique_notice_count"),
                label="profile unique notice count",
                minimum=1,
            ),
            "vector_count": _integer(
                output.get("vector_count"),
                label="profile vector count",
                minimum=1,
            ),
        },
        "evaluation_inputs": {
            "profile_config_sha256": _sha(
                _nested(profile, "inputs", "profile_config_sha256", label="profile"),
                label="profile config SHA-256",
            ),
            "retrieval_queries_sha256": _sha(
                _nested(
                    retrieval,
                    "artifacts",
                    "queries",
                    "sha256",
                    label="retrieval",
                ),
                label="retrieval query SHA-256",
            ),
            "robustness_base_queries_sha256": _sha(
                _nested(
                    robustness,
                    "artifacts",
                    "base_queries",
                    "sha256",
                    label="robustness",
                ),
                label="robustness base query SHA-256",
            ),
            "robustness_variants_sha256": _sha(
                _nested(
                    robustness,
                    "artifacts",
                    "variants",
                    "sha256",
                    label="robustness",
                ),
                label="robustness variants SHA-256",
            ),
            "safety_config_sha256": _sha(
                _nested(safety, "inputs", "config_sha256", label="safety"),
                label="safety config SHA-256",
            ),
        },
        "label_policy": label_policy,
        "limitations": list(LIMITATIONS),
        "not_applicable": list(NOT_APPLICABLE),
        "profile": SUMMARY_PROFILE,
        "reference_date": _same_reference_date(reports),
        "reproduction_commands": list(REPRODUCTION_COMMANDS),
        "results": {
            "exposure_representation": {
                "overall": _selected_numbers(
                    _nested(
                        exposure_system,
                        "query_evidence_coverage",
                        "overall",
                        label="exposure coverage",
                    ),
                    EXPOSURE_KEYS,
                    label="exposure coverage",
                ),
                "system": "hybrid_natural_graph",
                "validation_passed": True,
            },
            "profile_reranking": {
                "summary": _selected_numbers(
                    profile_summary,
                    PROFILE_RESULT_KEYS,
                    label="profile summary",
                ),
                "validation_passed": True,
            },
            "query_robustness": {
                "metrics": _selected_numbers(
                    robustness_summary.get("metrics"),
                    ROBUSTNESS_METRIC_KEYS,
                    label="robustness metrics",
                ),
                "stability": _selected_numbers(
                    robustness_summary.get("stability"),
                    ROBUSTNESS_STABILITY_KEYS,
                    label="robustness stability",
                ),
                "validation_passed": True,
            },
            "retrieval": {
                "metrics": _selected_numbers(
                    retrieval_system.get("metrics"),
                    RETRIEVAL_METRIC_KEYS,
                    label="retrieval metrics",
                ),
                "role": str(retrieval_system.get("role", "")),
                "system": "hybrid_natural_graph",
                "validation_passed": True,
            },
            "safety_contract": {
                "summary": _selected_numbers(
                    safety_summary,
                    SAFETY_RESULT_KEYS,
                    label="safety summary",
                ),
                "validation_passed": True,
            },
        },
        "schema_version": SUMMARY_SCHEMA_VERSION,
    }
    if payload["results"]["retrieval"]["role"] != "headline_natural_only":
        raise PublicTextEvaluationSummaryError("retrieval headline role is invalid")
    return payload


def _reject_volatile_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise PublicTextEvaluationSummaryError("summary key is invalid")
            lowered = key.casefold()
            if lowered in _VOLATILE_KEYS or any(
                part in lowered for part in _VOLATILE_KEY_PARTS
            ):
                raise PublicTextEvaluationSummaryError(
                    "summary contains a volatile field"
                )
            _reject_volatile_keys(child)
    elif isinstance(value, list):
        for child in value:
            _reject_volatile_keys(child)


def _exact_keys(value: Any, expected: set[str], *, label: str) -> Mapping[str, Any]:
    mapping = _mapping(value, label=label)
    if set(mapping) != expected:
        raise PublicTextEvaluationSummaryError(f"{label} schema is invalid")
    return mapping


def validate_public_text_evaluation_summary_payload(
    payload: Mapping[str, Any],
    *,
    expected_index_sha256: str | None = None,
    expected_metas_sha256: str | None = None,
) -> None:
    """Validate the compact schema and optional artifact bindings."""

    _reject_volatile_keys(payload)
    root = _exact_keys(payload, _TOP_LEVEL_KEYS, label="evaluation summary")
    if root.get("schema_version") != SUMMARY_SCHEMA_VERSION:
        raise PublicTextEvaluationSummaryError("summary schema version is invalid")
    if root.get("profile") != SUMMARY_PROFILE:
        raise PublicTextEvaluationSummaryError("summary profile is invalid")
    try:
        date.fromisoformat(str(root.get("reference_date", "")))
    except ValueError as exc:
        raise PublicTextEvaluationSummaryError(
            "summary reference date is invalid"
        ) from exc

    artifacts = _exact_keys(root.get("artifacts"), _ARTIFACT_KEYS, label="artifacts")
    index_sha = _sha(artifacts.get("index_sha256"), label="index SHA-256")
    metas_sha = _sha(artifacts.get("metas_sha256"), label="metadata SHA-256")
    if expected_index_sha256 is not None and index_sha != expected_index_sha256:
        raise PublicTextEvaluationSummaryError("summary index SHA-256 mismatch")
    if expected_metas_sha256 is not None and metas_sha != expected_metas_sha256:
        raise PublicTextEvaluationSummaryError("summary metadata SHA-256 mismatch")
    _integer(artifacts.get("dimension"), label="dimension", minimum=1)
    _integer(artifacts.get("vector_count"), label="vector count", minimum=1)
    _integer(
        artifacts.get("unique_notice_count"), label="unique notice count", minimum=1
    )

    inputs = _exact_keys(
        root.get("evaluation_inputs"), _INPUT_KEYS, label="evaluation inputs"
    )
    for key in _INPUT_KEYS:
        _sha(inputs.get(key), label=key)

    label_policy = _exact_keys(
        root.get("label_policy"), _LABEL_POLICY_KEYS, label="label policy"
    )
    if not isinstance(label_policy.get("version"), str) or not label_policy.get(
        "version"
    ):
        raise PublicTextEvaluationSummaryError("label policy version is invalid")
    if not isinstance(label_policy.get("relevance"), str) or not label_policy.get(
        "relevance"
    ):
        raise PublicTextEvaluationSummaryError("label policy relevance is invalid")
    source_fields = label_policy.get("source_fields")
    if (
        not isinstance(source_fields, list)
        or not source_fields
        or not all(isinstance(item, str) and item for item in source_fields)
    ):
        raise PublicTextEvaluationSummaryError("label policy source fields are invalid")
    for key in ("uses_ranked_results", "uses_result_text", "uses_vlm_attributes"):
        if type(label_policy.get(key)) is not bool:
            raise PublicTextEvaluationSummaryError("label policy boolean is invalid")

    if root.get("limitations") != list(LIMITATIONS):
        raise PublicTextEvaluationSummaryError("summary limitations are invalid")
    if root.get("not_applicable") != list(NOT_APPLICABLE):
        raise PublicTextEvaluationSummaryError("summary N/A scope is invalid")
    if root.get("reproduction_commands") != list(REPRODUCTION_COMMANDS):
        raise PublicTextEvaluationSummaryError(
            "summary reproduction commands are invalid"
        )

    results = _exact_keys(root.get("results"), _RESULT_KEYS, label="results")
    retrieval = _exact_keys(
        results.get("retrieval"),
        {"metrics", "role", "system", "validation_passed"},
        label="retrieval result",
    )
    if (
        retrieval.get("system") != "hybrid_natural_graph"
        or retrieval.get("role") != "headline_natural_only"
    ):
        raise PublicTextEvaluationSummaryError("retrieval result identity is invalid")
    _passed(retrieval.get("validation_passed"), label="retrieval validation")
    retrieval_metrics = _exact_keys(
        retrieval.get("metrics"), set(RETRIEVAL_METRIC_KEYS), label="retrieval metrics"
    )
    _selected_numbers(
        retrieval_metrics, RETRIEVAL_METRIC_KEYS, label="retrieval metrics"
    )

    robustness = _exact_keys(
        results.get("query_robustness"),
        {"metrics", "stability", "validation_passed"},
        label="robustness result",
    )
    _passed(robustness.get("validation_passed"), label="robustness validation")
    robustness_metrics = _exact_keys(
        robustness.get("metrics"),
        set(ROBUSTNESS_METRIC_KEYS),
        label="robustness metrics",
    )
    _selected_numbers(
        robustness_metrics, ROBUSTNESS_METRIC_KEYS, label="robustness metrics"
    )
    robustness_stability = _exact_keys(
        robustness.get("stability"),
        set(ROBUSTNESS_STABILITY_KEYS),
        label="robustness stability",
    )
    _selected_numbers(
        robustness_stability,
        ROBUSTNESS_STABILITY_KEYS,
        label="robustness stability",
    )

    exposure = _exact_keys(
        results.get("exposure_representation"),
        {"overall", "system", "validation_passed"},
        label="exposure result",
    )
    if exposure.get("system") != "hybrid_natural_graph":
        raise PublicTextEvaluationSummaryError("exposure result identity is invalid")
    _passed(exposure.get("validation_passed"), label="exposure validation")
    exposure_overall = _exact_keys(
        exposure.get("overall"), set(EXPOSURE_KEYS), label="exposure coverage"
    )
    _selected_numbers(exposure_overall, EXPOSURE_KEYS, label="exposure coverage")

    profile = _exact_keys(
        results.get("profile_reranking"),
        {"summary", "validation_passed"},
        label="profile result",
    )
    _passed(profile.get("validation_passed"), label="profile validation")
    profile_numbers = _exact_keys(
        profile.get("summary"), set(PROFILE_RESULT_KEYS), label="profile summary"
    )
    _selected_numbers(profile_numbers, PROFILE_RESULT_KEYS, label="profile summary")

    safety = _exact_keys(
        results.get("safety_contract"),
        {"summary", "validation_passed"},
        label="safety result",
    )
    _passed(safety.get("validation_passed"), label="safety validation")
    safety_numbers = _exact_keys(
        safety.get("summary"), set(SAFETY_RESULT_KEYS), label="safety summary"
    )
    _selected_numbers(safety_numbers, SAFETY_RESULT_KEYS, label="safety summary")


def canonical_summary_json_bytes(payload: Mapping[str, Any]) -> bytes:
    validate_public_text_evaluation_summary_payload(payload)
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _percentage(value: Any) -> str:
    return f"{float(value) * 100:.2f}%"


def render_public_text_evaluation_summary(payload: Mapping[str, Any]) -> bytes:
    validate_public_text_evaluation_summary_payload(payload)
    artifacts = payload["artifacts"]
    results = payload["results"]
    retrieval = results["retrieval"]["metrics"]
    robustness = results["query_robustness"]
    exposure = results["exposure_representation"]["overall"]
    profile = results["profile_reranking"]["summary"]
    safety = results["safety_contract"]["summary"]
    commands = "\n".join(str(item) for item in payload["reproduction_commands"])
    limitations = "\n".join(f"- {item}" for item in payload["limitations"])
    not_applicable = "\n".join(f"- {item}" for item in payload["not_applicable"])
    text = f"""# public-text-only 평가 요약

- 프로필: `{payload["profile"]}`
- 고정 기준일: `{payload["reference_date"]}`
- 검색 artifact: {artifacts["vector_count"]:,} vectors / {artifacts["unique_notice_count"]:,} notices / {artifacts["dimension"]} dimensions
- index SHA-256: `{artifacts["index_sha256"]}`
- metadata SHA-256: `{artifacts["metas_sha256"]}`

## 프로필 적용 결과

| 검증 | 핵심 결과 | 상태 |
|---|---|---|
| 자연어 검색 (`hybrid_natural_graph`) | P@5 {_percentage(retrieval["precision@5"])}, nDCG@5 {_percentage(retrieval["nDCG@5"])}, Hit@5 {_percentage(retrieval["hit@5"])}, inactive {_percentage(retrieval["inactive_exposure"])} | PASS |
| 질의 강건성 | {robustness["stability"]["variant_count"]} variants, mean top-5 Jaccard {robustness["stability"]["mean_top5_jaccard"]:.6f}, Hit@5 {_percentage(robustness["metrics"]["hit@5"])} | PASS |
| 근거 노출 | {exposure["evaluated_count"]}/{exposure["applicable_count"]} fields evaluated, unknown {exposure["unknown_count"]} | PASS |
| 프로필 재정렬 계약 | {profile["profiles_with_different_top1_from_baseline"]}/{profile["profile_count"]} profiles changed top-1, unknown-neutrality failures {profile["unknown_neutrality_failures"]} | PASS |
| 안전 계약 | {safety["passed_contracts"]}/{safety["total_contracts"]} contracts passed | PASS |

검색 지표의 relevance는 `{payload["label_policy"]["version"]}` 규칙으로 공고의 명시 필드만 사용해 만든 silver label입니다.

## 적용 불가

{not_applicable}

## 해석 한계

{limitations}

## 재현 명령

`environment.release.yml` 환경과 이 ZIP의 canonical `data/` artifact에서 실행합니다.

```powershell
{commands}
```
"""
    return text.replace("\r\n", "\n").encode("utf-8")


def encode_public_text_evaluation_summary(
    payload: Mapping[str, Any],
) -> PublicTextEvaluationSummaryArtifacts:
    return PublicTextEvaluationSummaryArtifacts(
        payload=dict(payload),
        json_bytes=canonical_summary_json_bytes(payload),
        markdown_bytes=render_public_text_evaluation_summary(payload),
    )


def build_public_text_evaluation_summary(
    *,
    reports_dir: Path,
    index_path: Path,
    metas_path: Path,
    profile_marker_path: Path,
) -> PublicTextEvaluationSummaryArtifacts:
    reports = {
        key: _read_json(reports_dir / filename, label=f"{key} evaluation report")
        for key, filename in RAW_REPORT_NAMES.items()
    }
    marker = _read_json(profile_marker_path, label="release profile marker")
    payload = _build_payload(
        reports,
        marker,
        index_sha256=_sha256_file(index_path, label="public-text index"),
        metas_sha256=_sha256_file(metas_path, label="public-text metadata"),
    )
    return encode_public_text_evaluation_summary(payload)


def validate_public_text_evaluation_summary_files(
    json_bytes: bytes,
    markdown_bytes: bytes,
    *,
    expected_index_sha256: str,
    expected_metas_sha256: str,
) -> Mapping[str, Any]:
    try:
        payload = json.loads(json_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicTextEvaluationSummaryError(
            "public-text evaluation summary JSON is invalid"
        ) from exc
    root = _mapping(payload, label="evaluation summary")
    validate_public_text_evaluation_summary_payload(
        root,
        expected_index_sha256=expected_index_sha256,
        expected_metas_sha256=expected_metas_sha256,
    )
    if json_bytes != canonical_summary_json_bytes(root):
        raise PublicTextEvaluationSummaryError(
            "public-text evaluation summary JSON is not canonical"
        )
    if markdown_bytes != render_public_text_evaluation_summary(root):
        raise PublicTextEvaluationSummaryError(
            "public-text evaluation summary Markdown does not match its JSON"
        )
    return root
