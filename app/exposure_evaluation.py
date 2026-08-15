"""Descriptive retrieval exposure and evidence-coverage diagnostics.

This module reads the fixed retrieval report instead of recomputing rankings.
It does not define a fairness target and does not treat any reported dimension
as a protected attribute.
"""

from __future__ import annotations

import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.profile_evaluation import portable_input_path, sha256_file
from app.retrieval_evaluation import (
    BASELINE_SYSTEM_ID,
    HEADLINE_SYSTEM_ID,
    clean_text,
    extract_public_attributes,
    is_known,
    merge_notice_metas,
    parse_reference_date,
    validate_query_specs,
)


REPORT_SCHEMA_VERSION = "exposure-representation.v1"
DEFAULT_SYSTEM_IDS = (BASELINE_SYSTEM_ID, HEADLINE_SYSTEM_ID)
GROUP_DIMENSIONS = ("region", "shelter", "size", "age", "photo_quality")
EVIDENCE_DIMENSIONS = ("color", "size", "age", "region")
UNKNOWN_GROUP = "unknown"
_SIZE_GROUPS = {"tiny", "small", "medium", "large"}
_AGE_GROUPS = {"puppy", "adult", "senior"}
_PHOTO_QUALITY_GROUPS = {"low", "medium", "high"}
_SHELTER_KEYS = ("care_name", "careNm", "shelter", "shelter_name")


def _load_json(path: Path, *, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {exc}") from exc


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _require_array(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return value


def _round_ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def _exposure_representation_ratio(
    exposure_share: float,
    representation_share: float,
) -> float | None:
    if representation_share <= 0:
        return None
    return round(exposure_share / representation_share, 6)


def _known_group(value: Any, allowed: set[str] | None = None) -> str:
    if not is_known(value):
        return UNKNOWN_GROUP
    normalized = " ".join(unicodedata.normalize("NFKC", clean_text(value)).split())
    if allowed is not None:
        normalized = normalized.casefold()
        if normalized not in allowed:
            return UNKNOWN_GROUP
    return normalized or UNKNOWN_GROUP


def _shelter_group(meta: Mapping[str, Any]) -> str:
    for key in _SHELTER_KEYS:
        value = meta.get(key)
        if is_known(value):
            return _known_group(value)
    return UNKNOWN_GROUP


def _photo_quality_group(meta: Mapping[str, Any]) -> str:
    attrs = meta.get("image_attrs")
    if isinstance(attrs, Mapping):
        value = attrs.get("photo_quality_band")
        if is_known(value):
            return _known_group(value, _PHOTO_QUALITY_GROUPS)
    return _known_group(meta.get("photo_quality_band"), _PHOTO_QUALITY_GROUPS)


def notice_groups(
    meta: Mapping[str, Any],
    attributes: Mapping[str, Any],
) -> dict[str, str]:
    """Return exactly one explicit group per diagnostic dimension."""

    return {
        "region": _known_group(attributes.get("region")),
        "shelter": _shelter_group(meta),
        "size": _known_group(attributes.get("size_hint"), _SIZE_GROUPS),
        "age": _known_group(attributes.get("age_hint"), _AGE_GROUPS),
        "photo_quality": _photo_quality_group(meta),
    }


def query_evidence_dimensions(criteria: Mapping[str, Any]) -> tuple[str, ...]:
    """Map fixed query criteria to independently checkable public fields."""

    dimensions: list[str] = []
    if criteria.get("colors_any"):
        dimensions.append("color")
    if criteria.get("sizes_any") or criteria.get("weight_kg"):
        dimensions.append("size")
    if criteria.get("age_hints_any"):
        dimensions.append("age")
    if criteria.get("regions_any"):
        dimensions.append("region")
    return tuple(dimensions)


def _evidence_is_known(
    dimension: str,
    attributes: Mapping[str, Any],
) -> bool:
    if dimension == "color":
        return bool(attributes.get("colors"))
    if dimension == "size":
        return attributes.get("weight_kg") is not None and bool(
            attributes.get("size_hint")
        )
    if dimension == "age":
        return bool(attributes.get("age_hint"))
    if dimension == "region":
        return bool(attributes.get("region"))
    raise ValueError(f"unsupported evidence dimension: {dimension}")


def _coverage_summary(
    applicable_count: int,
    evaluated_count: int,
) -> dict[str, Any]:
    unknown_count = applicable_count - evaluated_count
    if unknown_count < 0:
        raise ValueError("evaluated evidence cannot exceed applicable evidence")
    return {
        "applicable_count": applicable_count,
        "evaluated_count": evaluated_count,
        "unknown_count": unknown_count,
        "evidence_coverage": _round_ratio(evaluated_count, applicable_count),
    }


def _query_specs(query_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_queries = _require_array(query_payload.get("queries"), label="queries")
    if not raw_queries or any(not isinstance(item, Mapping) for item in raw_queries):
        raise ValueError("queries must be a non-empty array of objects")
    queries = [dict(item) for item in raw_queries]
    validate_query_specs(queries)
    ids = [clean_text(query.get("id")) for query in queries]
    if any(not query_id for query_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("query IDs must be present and unique")
    return queries


def _validate_source_artifacts(
    retrieval_report: Mapping[str, Any],
    metas_path: Path,
    queries_path: Path,
) -> None:
    artifacts = _require_mapping(
        retrieval_report.get("artifacts"),
        label="retrieval_report.artifacts",
    )
    expected = {
        "metas": sha256_file(metas_path),
        "queries": sha256_file(queries_path),
    }
    for name, expected_sha in expected.items():
        record = _require_mapping(
            artifacts.get(name),
            label=f"retrieval_report.artifacts.{name}",
        )
        actual_sha = clean_text(record.get("sha256")).lower()
        if actual_sha != expected_sha:
            raise ValueError(
                f"retrieval report {name} SHA-256 does not match current input"
            )


def _active_corpus(
    raw_metas: Sequence[Mapping[str, Any]],
    reference_date: Any,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, str]],
]:
    merged = merge_notice_metas(raw_metas)
    active_metas: dict[str, dict[str, Any]] = {}
    attributes_by_id: dict[str, dict[str, Any]] = {}
    groups_by_id: dict[str, dict[str, str]] = {}
    for dog_id in sorted(merged):
        meta = merged[dog_id]
        attributes = extract_public_attributes(meta, reference_date)
        if attributes.get("notice_status") != "active":
            continue
        active_metas[dog_id] = meta
        attributes_by_id[dog_id] = attributes
        groups_by_id[dog_id] = notice_groups(meta, attributes)
    if not active_metas:
        raise ValueError("active corpus is empty at the fixed reference date")
    return active_metas, attributes_by_id, groups_by_id


def _corpus_dimension_counts(
    groups_by_id: Mapping[str, Mapping[str, str]],
) -> dict[str, Counter[str]]:
    counts = {dimension: Counter() for dimension in GROUP_DIMENSIONS}
    for groups in groups_by_id.values():
        for dimension in GROUP_DIMENSIONS:
            counts[dimension][groups.get(dimension, UNKNOWN_GROUP)] += 1
    for dimension in GROUP_DIMENSIONS:
        counts[dimension][UNKNOWN_GROUP] += 0
    return counts


def _system_query_rows(
    retrieval_report: Mapping[str, Any],
    system_id: str,
    expected_query_ids: Sequence[str],
) -> tuple[Mapping[str, Any], dict[str, Mapping[str, Any]]]:
    systems = _require_mapping(retrieval_report.get("systems"), label="systems")
    system = _require_mapping(systems.get(system_id), label=f"systems.{system_id}")
    if system.get("status_filtered") is not True:
        raise ValueError(f"system {system_id!r} must use the fixed status filter")
    if system.get("uses_explicit_structured") is True:
        raise ValueError(f"system {system_id!r} cannot be a structured oracle")
    raw_rows = _require_array(
        system.get("queries"), label=f"systems.{system_id}.queries"
    )
    rows: dict[str, Mapping[str, Any]] = {}
    for index, raw_row in enumerate(raw_rows):
        row = _require_mapping(raw_row, label=f"systems.{system_id}.queries[{index}]")
        query_id = clean_text(row.get("id"))
        if not query_id or query_id in rows:
            raise ValueError(f"system {system_id!r} has missing or duplicate query IDs")
        rows[query_id] = row
    if set(rows) != set(expected_query_ids):
        raise ValueError(f"system {system_id!r} query IDs do not match query input")
    return system, rows


def _validated_top_ids(
    row: Mapping[str, Any],
    *,
    system_id: str,
    query_id: str,
    top_k: int,
    active_metas: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    raw_ids = _require_array(
        row.get("top_ids"),
        label=f"systems.{system_id}.{query_id}.top_ids",
    )
    if len(raw_ids) < top_k:
        raise ValueError(
            f"system {system_id!r} query {query_id!r} has fewer than {top_k} IDs"
        )
    top_ids = [clean_text(value) for value in raw_ids[:top_k]]
    if any(not dog_id for dog_id in top_ids) or len(top_ids) != len(set(top_ids)):
        raise ValueError(
            f"system {system_id!r} query {query_id!r} has blank or duplicate IDs"
        )
    missing = [dog_id for dog_id in top_ids if dog_id not in active_metas]
    if missing:
        raise ValueError(
            f"system {system_id!r} query {query_id!r} references non-active IDs: "
            + ", ".join(missing)
        )
    return top_ids


def _evaluate_system(
    system_id: str,
    system: Mapping[str, Any],
    rows_by_query: Mapping[str, Mapping[str, Any]],
    queries: Sequence[Mapping[str, Any]],
    top_k: int,
    active_metas: Mapping[str, Mapping[str, Any]],
    attributes_by_id: Mapping[str, Mapping[str, Any]],
    groups_by_id: Mapping[str, Mapping[str, str]],
) -> tuple[dict[str, Any], dict[str, Counter[str]]]:
    exposure_counts = {dimension: Counter() for dimension in GROUP_DIMENSIONS}
    evidence_applicable = Counter({dimension: 0 for dimension in EVIDENCE_DIMENSIONS})
    evidence_evaluated = Counter({dimension: 0 for dimension in EVIDENCE_DIMENSIONS})
    query_rows: list[dict[str, Any]] = []
    exposure_slots = 0
    exposed_notice_ids: set[str] = set()

    for query in queries:
        query_id = clean_text(query.get("id"))
        criteria = _require_mapping(
            query.get("criteria"),
            label=f"query {query_id}.criteria",
        )
        applicable_dimensions = query_evidence_dimensions(criteria)
        top_ids = _validated_top_ids(
            rows_by_query[query_id],
            system_id=system_id,
            query_id=query_id,
            top_k=top_k,
            active_metas=active_metas,
        )
        query_applicable = 0
        query_evaluated = 0
        for dog_id in top_ids:
            exposure_slots += 1
            exposed_notice_ids.add(dog_id)
            groups = groups_by_id[dog_id]
            for dimension in GROUP_DIMENSIONS:
                exposure_counts[dimension][groups[dimension]] += 1
            attributes = attributes_by_id[dog_id]
            for dimension in applicable_dimensions:
                evidence_applicable[dimension] += 1
                query_applicable += 1
                if _evidence_is_known(dimension, attributes):
                    evidence_evaluated[dimension] += 1
                    query_evaluated += 1
        query_rows.append(
            {
                "query_id": query_id,
                "applicable_dimensions": list(applicable_dimensions),
                "result_count": len(top_ids),
                **_coverage_summary(query_applicable, query_evaluated),
            }
        )

    total_applicable = sum(evidence_applicable.values())
    total_evaluated = sum(evidence_evaluated.values())
    evidence_by_dimension = {
        dimension: _coverage_summary(
            evidence_applicable[dimension],
            evidence_evaluated[dimension],
        )
        for dimension in EVIDENCE_DIMENSIONS
    }
    return (
        {
            "role": clean_text(system.get("role")),
            "query_count": len(queries),
            "top_k": top_k,
            "exposure_slots": exposure_slots,
            "unique_exposed_notice_count": len(exposed_notice_ids),
            "query_evidence_coverage": {
                "overall": _coverage_summary(total_applicable, total_evaluated),
                "by_dimension": evidence_by_dimension,
                "queries": query_rows,
            },
        },
        exposure_counts,
    )


def _dimension_report(
    dimension: str,
    corpus_counts: Counter[str],
    exposure_by_system: Mapping[str, Mapping[str, Counter[str]]],
    system_summaries: Mapping[str, Mapping[str, Any]],
    corpus_count: int,
) -> dict[str, Any]:
    unknown_count = corpus_counts[UNKNOWN_GROUP]
    groups = set(corpus_counts)
    for system_counts in exposure_by_system.values():
        groups.update(system_counts[dimension])

    rows: list[dict[str, Any]] = []
    ordered_groups = sorted(
        groups,
        key=lambda group: (
            group == UNKNOWN_GROUP,
            -corpus_counts[group],
            group.casefold(),
        ),
    )
    for group in ordered_groups:
        representation_share_raw = float(corpus_counts[group]) / float(corpus_count)
        representation_share = round(representation_share_raw, 6)
        exposures: dict[str, Any] = {}
        for system_id, counts in exposure_by_system.items():
            exposure_slots = int(system_summaries[system_id]["exposure_slots"])
            exposure_count = counts[dimension][group]
            exposure_share_raw = float(exposure_count) / float(exposure_slots)
            exposure_share = round(exposure_share_raw, 6)
            exposures[system_id] = {
                "exposure_count": exposure_count,
                "exposure_share": exposure_share,
                "exposure_representation_ratio": _exposure_representation_ratio(
                    exposure_share_raw,
                    representation_share_raw,
                ),
            }
        rows.append(
            {
                "group": group,
                "corpus_count": corpus_counts[group],
                "representation_share": representation_share,
                "exposure_by_system": exposures,
            }
        )
    return {
        "corpus_evidence_coverage": _coverage_summary(
            corpus_count,
            corpus_count - unknown_count,
        ),
        "groups": rows,
    }


def _invariant_failures(report: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    corpus_count = int(report["corpus"]["active_unique_notices"])
    systems = report["systems"]
    for dimension, payload in report["dimensions"].items():
        rows = payload["groups"]
        if sum(int(row["corpus_count"]) for row in rows) != corpus_count:
            failures.append(f"{dimension}: corpus group counts do not sum to corpus")
        if not any(row["group"] == UNKNOWN_GROUP for row in rows):
            failures.append(f"{dimension}: unknown group is missing")
        for system_id, summary in systems.items():
            exposure_total = sum(
                int(row["exposure_by_system"][system_id]["exposure_count"])
                for row in rows
            )
            if exposure_total != int(summary["exposure_slots"]):
                failures.append(
                    f"{dimension}/{system_id}: exposure counts do not sum to slots"
                )
    for system_id, summary in systems.items():
        coverage = summary["query_evidence_coverage"]["overall"]
        if coverage["applicable_count"] != (
            coverage["evaluated_count"] + coverage["unknown_count"]
        ):
            failures.append(f"{system_id}: evidence counts are inconsistent")
    return failures


def evaluate_exposure_representation(
    metas_path: Path,
    retrieval_report_path: Path,
    queries_path: Path,
    *,
    top_k: int = 10,
    system_ids: Sequence[str] = DEFAULT_SYSTEM_IDS,
) -> dict[str, Any]:
    """Build a deterministic report from fixed rankings and active metadata."""

    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    normalized_system_ids = tuple(clean_text(value) for value in system_ids)
    if (
        not normalized_system_ids
        or any(not value for value in normalized_system_ids)
        or len(normalized_system_ids) != len(set(normalized_system_ids))
    ):
        raise ValueError("system_ids must be present and unique")

    raw_metas_value = _load_json(metas_path, label="dog_metas")
    raw_metas = _require_array(raw_metas_value, label="dog_metas")
    if any(not isinstance(row, Mapping) for row in raw_metas):
        raise ValueError("dog_metas must contain only objects")
    retrieval_report = _require_mapping(
        _load_json(retrieval_report_path, label="retrieval_report"),
        label="retrieval_report",
    )
    query_payload = _require_mapping(
        _load_json(queries_path, label="queries"),
        label="queries",
    )
    _validate_source_artifacts(retrieval_report, metas_path, queries_path)

    reference_date_text = clean_text(retrieval_report.get("reference_date"))
    if clean_text(query_payload.get("reference_date")) != reference_date_text:
        raise ValueError("query and retrieval report reference dates do not match")
    reference_date = parse_reference_date(reference_date_text)
    queries = _query_specs(query_payload)
    query_ids = [clean_text(query.get("id")) for query in queries]

    active_metas, attributes_by_id, groups_by_id = _active_corpus(
        raw_metas,
        reference_date,
    )
    retrieval_corpus = _require_mapping(
        retrieval_report.get("corpus"),
        label="retrieval_report.corpus",
    )
    if int(retrieval_corpus.get("active_documents", -1)) != len(active_metas):
        raise ValueError("retrieval report active corpus count does not match metadata")

    corpus_counts = _corpus_dimension_counts(groups_by_id)
    system_summaries: dict[str, dict[str, Any]] = {}
    exposure_by_system: dict[str, dict[str, Counter[str]]] = {}
    for system_id in normalized_system_ids:
        system, rows_by_query = _system_query_rows(
            retrieval_report,
            system_id,
            query_ids,
        )
        summary, exposures = _evaluate_system(
            system_id,
            system,
            rows_by_query,
            queries,
            top_k,
            active_metas,
            attributes_by_id,
            groups_by_id,
        )
        system_summaries[system_id] = summary
        exposure_by_system[system_id] = exposures

    dimensions = {
        dimension: _dimension_report(
            dimension,
            corpus_counts[dimension],
            exposure_by_system,
            system_summaries,
            len(active_metas),
        )
        for dimension in GROUP_DIMENSIONS
    }
    retrieval_artifacts = _require_mapping(
        retrieval_report.get("artifacts"),
        label="retrieval_report.artifacts",
    )
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "evaluation_id": "exposure_representation.appearance_v1",
        "reference_date": reference_date_text,
        "scope": "descriptive_retrieval_exposure_not_normative_fairness",
        "method": {
            "ranking_recomputed": False,
            "ranking_logic_changed": False,
            "systems": list(normalized_system_ids),
            "query_count": len(queries),
            "top_k": top_k,
            "exposure_unit": (
                "one equal-weight Top-K slot; repeated notices across different "
                "queries count once per query"
            ),
            "representation_denominator": "unique active notices",
            "group_dimensions": list(GROUP_DIMENSIONS),
            "unknown_policy": (
                "missing or unrecognized values are retained as the explicit "
                "unknown group"
            ),
            "ratio_formula": "exposure_share / corpus_representation_share",
            "zero_representation_ratio": None,
            "query_evidence_dimensions": {
                "color": "criteria.colors_any -> public notice color",
                "size": (
                    "criteria.sizes_any or criteria.weight_kg -> public notice weight"
                ),
                "age": "criteria.age_hints_any -> public notice age",
                "region": "criteria.regions_any -> public notice region",
            },
            "active_only_is_filter_not_evidence_dimension": True,
        },
        "inputs": {
            "dog_metas": {
                "path": portable_input_path(metas_path),
                "sha256": sha256_file(metas_path),
                "rows": len(raw_metas),
            },
            "retrieval_report": {
                "path": portable_input_path(retrieval_report_path),
                "sha256": sha256_file(retrieval_report_path),
            },
            "queries": {
                "path": portable_input_path(queries_path),
                "sha256": sha256_file(queries_path),
                "count": len(queries),
            },
            "source_retrieval_artifact_sha256": {
                name: clean_text(record.get("sha256"))
                for name, record in retrieval_artifacts.items()
                if isinstance(record, Mapping)
            },
        },
        "corpus": {
            "metadata_rows": len(raw_metas),
            "active_unique_notices": len(active_metas),
        },
        "systems": system_summaries,
        "dimensions": dimensions,
        "validation": {},
        "limitations": [
            "This is a descriptive exposure diagnostic, not a normative fairness assessment.",
            "Region, shelter, size, age, and photo quality are not asserted to be protected attributes.",
            "An exposure/representation ratio above or below 1 does not by itself establish benefit, harm, bias, or an appropriate target.",
            "The result covers 12 fixed appearance queries and equal-weight Top-K slots; it does not represent all user queries or rank-discounted attention.",
            "Ratios for small corpus groups can be unstable and must be read with corpus and exposure counts.",
            "Unknown is retained as missing evidence rather than imputed as a fact.",
            "Photo quality describes image usability, not health, temperament, or adoption suitability.",
            "Notice status and metadata are fixed to the reference date and can change afterward.",
        ],
    }
    failures = _invariant_failures(report)
    report["validation"] = {
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
    }
    if failures:
        raise ValueError("exposure report invariant failure: " + "; ".join(failures))
    return report


def _percent(value: Any) -> str:
    return f"{100.0 * float(value):.2f}%"


def _ratio(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.3f}"


def _markdown_text(value: Any) -> str:
    return clean_text(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(report: Mapping[str, Any]) -> str:
    system_ids = list(report["method"]["systems"])
    lines = [
        "# 외형 검색 노출·대표성 및 증거 커버리지 진단",
        "",
        "> 이 리포트는 고정 질의의 검색 노출을 기술적으로 관찰한 결과입니다. 규범적 공정성 평가가 아니며 어떤 그룹도 보호특성으로 주장하지 않습니다.",
        "",
        f"- 기준일: `{report['reference_date']}`",
        f"- 활성 공고: {report['corpus']['active_unique_notices']}건",
        f"- 고정 질의: {report['method']['query_count']}개",
        f"- Top-K: {report['method']['top_k']}",
        f"- 시스템: {', '.join(f'`{value}`' for value in system_ids)}",
        "- 순위 재계산·핵심 랭킹 변경: 없음",
        "- 노출 단위: 각 질의의 Top-K 한 칸을 동일 가중치로 1회 계산",
        "- 비율: `Top-K 노출 점유율 / corpus 구성비`; corpus 구성비가 0이면 `N/A`",
        "",
        "## 입력 고정 정보",
        "",
        "| 입력 | 경로 | SHA-256 |",
        "|---|---|---|",
    ]
    for name in ("dog_metas", "retrieval_report", "queries"):
        record = report["inputs"][name]
        lines.append(
            f"| {name} | `{_markdown_text(record['path'])}` | `{record['sha256']}` |"
        )

    lines.extend(
        [
            "",
            "## Corpus 필드 증거 커버리지",
            "",
            "모든 활성 공고에 각 그룹 필드를 적용 가능하다고 보고, 값이 없거나 해석할 수 없으면 `unknown`으로 계산합니다.",
            "",
            "| 차원 | applicable | evaluated | unknown | coverage |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for dimension in GROUP_DIMENSIONS:
        coverage = report["dimensions"][dimension]["corpus_evidence_coverage"]
        lines.append(
            f"| {dimension} | {coverage['applicable_count']} | "
            f"{coverage['evaluated_count']} | {coverage['unknown_count']} | "
            f"{_percent(coverage['evidence_coverage'])} |"
        )

    lines.extend(
        [
            "",
            "## Top-K 질의 증거 커버리지",
            "",
            "각 결과에서 해당 질의가 요구한 색상·체중/크기·연령·지역 공개 필드만 applicable로 셉니다. `active_only`는 후보 필터이므로 증거 차원에서 제외합니다.",
            "",
            "| 시스템 | 구분 | applicable | evaluated | unknown | coverage |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for system_id in system_ids:
        evidence = report["systems"][system_id]["query_evidence_coverage"]
        rows = [("overall", evidence["overall"]), *evidence["by_dimension"].items()]
        for label, coverage in rows:
            lines.append(
                f"| `{system_id}` | {label} | {coverage['applicable_count']} | "
                f"{coverage['evaluated_count']} | {coverage['unknown_count']} | "
                f"{_percent(coverage['evidence_coverage'])} |"
            )

    for dimension in GROUP_DIMENSIONS:
        lines.extend(
            [
                "",
                f"## {dimension} 그룹별 구성비와 Top-K 노출",
                "",
            ]
        )
        header = ["그룹", "corpus n", "구성비"]
        align = ["---", "---:", "---:"]
        for system_id in system_ids:
            header.extend(
                [
                    f"{system_id} 노출 n",
                    f"{system_id} 점유율",
                    f"{system_id} 노출/구성비",
                ]
            )
            align.extend(["---:", "---:", "---:"])
        lines.extend(["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"])
        for row in report["dimensions"][dimension]["groups"]:
            values = [
                _markdown_text(row["group"]),
                str(row["corpus_count"]),
                _percent(row["representation_share"]),
            ]
            for system_id in system_ids:
                exposure = row["exposure_by_system"][system_id]
                values.extend(
                    [
                        str(exposure["exposure_count"]),
                        _percent(exposure["exposure_share"]),
                        _ratio(exposure["exposure_representation_ratio"]),
                    ]
                )
            lines.append("| " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## 해석 제한",
            "",
            "- 이 값은 기술적 노출 진단이며 규범적 공정성, 차별 또는 적정 노출 목표를 판정하지 않습니다.",
            "- 지역·보호소·크기·연령·사진 품질을 보호특성으로 간주하지 않습니다.",
            "- 비율이 1보다 크거나 작다는 사실만으로 이익·피해·편향을 뜻하지 않습니다.",
            "- 작은 그룹의 비율은 크게 흔들릴 수 있으므로 corpus 수와 실제 노출 수를 함께 봐야 합니다.",
            "- `unknown`은 값을 추정하지 않고 결측 증거로 남긴 그룹입니다.",
            "- 사진 품질은 사진의 탐색 용이성일 뿐 건강·성격·입양 적합성이 아닙니다.",
            "- 고정 12개 질의·동일 가중치 Top-K 결과이므로 전체 사용자 질의나 위치별 주목도를 대표하지 않습니다.",
            "",
            "재실행: `python scripts/evaluate_exposure_representation.py`",
            "",
        ]
    )
    return "\n".join(lines)


def serialize_report(report: Mapping[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2) + "\n"


def write_report(
    report: Mapping[str, Any],
    json_path: Path,
    markdown_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(serialize_report(report), encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
