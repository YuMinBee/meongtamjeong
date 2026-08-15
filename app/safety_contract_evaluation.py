"""Deterministic regression evaluation for the appearance-only safety contract.

This module does not measure adoption suitability.  It verifies that lifestyle
and temperament preferences stay out of appearance retrieval and reranking,
while still changing the questions prepared for a shelter.
"""

from __future__ import annotations

import json
import unicodedata
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.appearance_query import (
    APPEARANCE_SEARCH_CONDITION_KEYS,
    appearance_query_excludes_nonvisual_terms,
    appearance_search_conditions,
    normalize_appearance_query,
)
from app.inquiry_helper import InquiryPreferences, build_inquiry_materials
from app.profile_evaluation import (
    canonical_sha256,
    load_joined_metas,
    portable_input_path,
    sha256_file,
)
from app.profile_rerank import (
    APPEARANCE_CONDITION_KEYS,
    AppearanceProfile,
    ProfileRerankSettings,
    ProfileSearchRequest,
    UserProfile,
    rerank_candidates,
)


APPEARANCE_PROFILE_FIELDS = (
    "preferred_size",
    "preferred_age",
    "preferred_region",
)
LIFESTYLE_PROFILE_FIELDS = (
    "housing_type",
    "daily_absence_hours",
    "activity_level",
    "dog_experience",
    "has_children",
    "has_other_pets",
)
INQUIRY_PROFILE_FIELDS = (
    "housing_type",
    "daily_absence_hours",
    "activity_level",
    "dog_experience",
    "has_children",
    "has_other_pets",
)
RESULT_EVIDENCE_FIELDS = (
    "dog_id",
    "retrieval_score",
    "compatibility_score",
    "applicable_count",
    "evaluated_count",
    "evidence_coverage",
    "quality_score",
    "final_score",
    "matched_conditions",
    "caution_conditions",
    "unknown_conditions",
    "recommendation_reason",
)
SAFETY_CONTRACT_SCHEMA_VERSION = 2


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _require_sequence(value: Any, *, label: str) -> Sequence[Any]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or not value
    ):
        raise ValueError(f"{label} must be a non-empty array")
    return value


def _canonical_literal_query(value: Any) -> str:
    """Canonicalize spacing only, independently of the production normalizer."""

    return " ".join(unicodedata.normalize("NFKC", str(value or "")).split())


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != SAFETY_CONTRACT_SCHEMA_VERSION:
        raise ValueError(
            f"safety contract schema_version must be {SAFETY_CONTRACT_SCHEMA_VERSION}"
        )
    query_pairs = _require_sequence(config.get("query_pairs"), label="query_pairs")
    seen_pair_ids: set[str] = set()
    for index, raw_pair in enumerate(query_pairs):
        pair = _require_mapping(raw_pair, label=f"query_pairs[{index}]")
        pair_id = str(pair.get("id") or "").strip()
        if not pair_id or pair_id in seen_pair_ids:
            raise ValueError("query pair IDs must be present and unique")
        seen_pair_ids.add(pair_id)
        mixed_query = _canonical_literal_query(pair.get("mixed_query"))
        expected_query = _canonical_literal_query(pair.get("appearance_query"))
        if not mixed_query or not expected_query:
            raise ValueError(f"query pair {pair_id!r} must contain both queries")
        required_terms = _require_sequence(
            pair.get("required_appearance_terms"),
            label=f"query_pairs[{index}].required_appearance_terms",
        )
        normalized_terms = [_canonical_literal_query(term) for term in required_terms]
        if any(not term for term in normalized_terms):
            raise ValueError(
                f"query pair {pair_id!r} has an empty required appearance term"
            )
        missing_from_expected = [
            term
            for term in normalized_terms
            if term.casefold() not in expected_query.casefold()
        ]
        if missing_from_expected:
            raise ValueError(
                f"query pair {pair_id!r} expected query omits required terms: "
                + ", ".join(missing_from_expected)
            )
    candidate_ids = _require_sequence(
        config.get("candidate_ids"),
        label="candidate_ids",
    )
    normalized_ids = [str(value).strip() for value in candidate_ids]
    if any(not value for value in normalized_ids):
        raise ValueError("candidate_ids cannot contain empty values")
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError("candidate_ids cannot contain duplicates")

    profiles = _require_mapping(config.get("profiles"), label="profiles")
    _require_mapping(profiles.get("a"), label="profiles.a")
    _require_mapping(profiles.get("b"), label="profiles.b")

    inquiry_preferences = _require_mapping(
        config.get("inquiry_preferences"),
        label="inquiry_preferences",
    )
    _require_mapping(inquiry_preferences.get("a"), label="inquiry_preferences.a")
    _require_mapping(inquiry_preferences.get("b"), label="inquiry_preferences.b")
    _require_mapping(config.get("conditions_case"), label="conditions_case")


def _case(
    case_id: str,
    title: str,
    evidence: Mapping[str, Any],
    failures: Sequence[str],
) -> dict[str, Any]:
    return {
        "id": case_id,
        "title": title,
        "passed": not failures,
        "failures": list(failures),
        "evidence": dict(evidence),
    }


def _query_pair_evidence(
    query_pairs: Sequence[Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for index, raw_spec in enumerate(query_pairs, start=1):
        spec = _require_mapping(raw_spec, label=f"query_pairs[{index - 1}]")
        pair_id = str(spec.get("id") or f"query_pair_{index}")
        mixed_query = str(spec.get("mixed_query") or "")
        appearance_query = str(spec.get("appearance_query") or "")
        expected_literal = _canonical_literal_query(appearance_query)
        required_terms = [
            _canonical_literal_query(value)
            for value in spec.get("required_appearance_terms", [])
            if _canonical_literal_query(value)
        ]
        forbidden_terms = [
            str(value).strip()
            for value in spec.get("forbidden_terms", [])
            if str(value).strip()
        ]
        normalized_mixed = normalize_appearance_query(mixed_query)
        missing_required_terms = [
            term
            for term in required_terms
            if term.casefold() not in normalized_mixed.casefold()
        ]
        forbidden_hits = [
            term
            for term in forbidden_terms
            if term.casefold() in normalized_mixed.casefold()
        ]
        equivalent = normalized_mixed == expected_literal
        exclusion_detected = appearance_query_excludes_nonvisual_terms(mixed_query)
        row = {
            "id": pair_id,
            "mixed_query": mixed_query,
            "appearance_query": appearance_query,
            "normalized_mixed_query": normalized_mixed,
            "expected_literal_query": expected_literal,
            # Kept as a report-schema compatibility alias. This value is never
            # produced by normalize_appearance_query.
            "normalized_appearance_query": expected_literal,
            "equivalent": equivalent,
            "nonappearance_exclusion_detected": exclusion_detected,
            "forbidden_terms": forbidden_terms,
            "forbidden_hits": forbidden_hits,
            "required_appearance_terms": required_terms,
            "missing_required_appearance_terms": missing_required_terms,
        }
        rows.append(row)
        if not equivalent:
            failures.append(
                f"{pair_id}: mixed and appearance-only queries normalized differently"
            )
        if forbidden_terms and not exclusion_detected:
            failures.append(f"{pair_id}: no non-appearance exclusion was detected")
        if missing_required_terms:
            failures.append(
                f"{pair_id}: required appearance terms were removed: "
                + ", ".join(missing_required_terms)
            )
    return rows, failures


def _result_evidence(result: Mapping[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(result.get(key)) for key in RESULT_EVIDENCE_FIELDS}


def _profile_isolation_case(
    config: Mapping[str, Any],
    joined_metas: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    failures: list[str] = []
    candidate_ids = [str(value).strip() for value in config["candidate_ids"]]
    missing_ids = [
        notice_id for notice_id in candidate_ids if notice_id not in joined_metas
    ]

    profiles = _require_mapping(config["profiles"], label="profiles")
    profile_a = UserProfile.model_validate(profiles["a"])
    profile_b = UserProfile.model_validate(profiles["b"])
    dump_a = profile_a.model_dump(mode="json")
    dump_b = profile_b.model_dump(mode="json")
    minimal_request = ProfileSearchRequest(
        query="강아지",
        profile=AppearanceProfile.model_validate(
            {field: dump_a[field] for field in APPEARANCE_PROFILE_FIELDS}
        ),
        ranking_scope="appearance",
    )
    minimal_profile = minimal_request.profile
    minimal_dump = minimal_profile.model_dump(mode="json", exclude_unset=True)
    unexpected_minimal_fields = sorted(
        set(minimal_dump) - set(APPEARANCE_PROFILE_FIELDS)
    )
    omitted_lifestyle_fields = sorted(set(LIFESTYLE_PROFILE_FIELDS) - set(minimal_dump))
    differing_fields = sorted(
        key for key in dump_a if dump_a.get(key) != dump_b.get(key)
    )
    unexpected_differences = sorted(
        set(differing_fields) - set(LIFESTYLE_PROFILE_FIELDS)
    )
    unchanged_lifestyle_fields = sorted(
        set(LIFESTYLE_PROFILE_FIELDS) - set(differing_fields)
    )
    appearance_fields_equal = all(
        dump_a[field] == dump_b[field] for field in APPEARANCE_PROFILE_FIELDS
    )
    if not appearance_fields_equal:
        failures.append("profile appearance fields are not identical")
    if unexpected_differences:
        failures.append(
            "profiles differ outside lifestyle fields: "
            + ", ".join(unexpected_differences)
        )
    if unchanged_lifestyle_fields:
        failures.append(
            "required lifestyle fields do not differ: "
            + ", ".join(unchanged_lifestyle_fields)
        )
    if unexpected_minimal_fields:
        failures.append(
            "minimal appearance profile contains lifestyle fields: "
            + ", ".join(unexpected_minimal_fields)
        )
    if set(omitted_lifestyle_fields) != set(LIFESTYLE_PROFILE_FIELDS):
        failures.append("minimal appearance profile did not omit every lifestyle field")
    if missing_ids:
        failures.append(
            "candidate IDs missing from dog_metas: " + ", ".join(missing_ids)
        )

    score_start = float(config.get("retrieval_score_start", 0.7))
    score_step = float(config.get("retrieval_score_step", 0.003))
    retrieval_scores = {
        notice_id: round(score_start - index * score_step, 12)
        for index, notice_id in enumerate(candidate_ids)
    }
    evidence: dict[str, Any] = {
        "candidate_ids": candidate_ids,
        "missing_candidate_ids": missing_ids,
        "shared_retrieval_scores": retrieval_scores,
        "appearance_fields": {
            field: dump_a[field] for field in APPEARANCE_PROFILE_FIELDS
        },
        "appearance_fields_equal": appearance_fields_equal,
        "minimal_profile": minimal_dump,
        "minimal_profile_fields": sorted(minimal_dump),
        "minimal_profile_unexpected_fields": unexpected_minimal_fields,
        "minimal_profile_omitted_lifestyle_fields": omitted_lifestyle_fields,
        "differing_profile_fields": differing_fields,
        "required_lifestyle_fields": list(LIFESTYLE_PROFILE_FIELDS),
        "unexpected_profile_differences": unexpected_differences,
        "unchanged_lifestyle_fields": unchanged_lifestyle_fields,
        "comparison_fields": list(RESULT_EVIDENCE_FIELDS),
        "ranking_condition_keys": sorted(APPEARANCE_CONDITION_KEYS),
    }
    if missing_ids:
        evidence.update(
            {
                "profile_a_order": [],
                "profile_b_order": [],
                "minimal_profile_order": [],
                "outputs_identical": False,
            }
        )
        return _case(
            "appearance_rerank_lifestyle_isolation",
            "생활정보가 외형 범위 순위·점수·근거에 영향을 주지 않음",
            evidence,
            failures,
        )

    reference_date = datetime.fromisoformat(str(config["reference_date"]))
    settings_payload = _require_mapping(config.get("settings", {}), label="settings")
    settings = ProfileRerankSettings(
        compatibility_weight=float(settings_payload.get("compatibility_weight", 0.25)),
        quality_weight=float(settings_payload.get("quality_weight", 0.0)),
        candidate_multiplier=1,
    )
    candidates = [
        {
            **deepcopy(joined_metas[notice_id]),
            "score": retrieval_scores[notice_id],
            "_raw_meta": deepcopy(joined_metas[notice_id]),
        }
        for notice_id in candidate_ids
    ]
    common_kwargs = {
        "settings": settings,
        "topk": len(candidate_ids),
        "include_unknown_notices": True,
        "reference_date": reference_date,
        "condition_keys": APPEARANCE_CONDITION_KEYS,
    }
    results_a = rerank_candidates(candidates, profile_a, **common_kwargs)
    results_b = rerank_candidates(candidates, profile_b, **common_kwargs)
    results_minimal = rerank_candidates(candidates, minimal_profile, **common_kwargs)
    view_a = [_result_evidence(result) for result in results_a]
    view_b = [_result_evidence(result) for result in results_b]
    view_minimal = [_result_evidence(result) for result in results_minimal]
    order_a = [str(result["dog_id"]) for result in results_a]
    order_b = [str(result["dog_id"]) for result in results_b]
    order_minimal = [str(result["dog_id"]) for result in results_minimal]
    filtered_a = sorted(set(candidate_ids) - set(order_a))
    filtered_b = sorted(set(candidate_ids) - set(order_b))
    filtered_minimal = sorted(set(candidate_ids) - set(order_minimal))
    outputs_identical = view_a == view_b == view_minimal
    if filtered_a:
        failures.append(
            "profile A did not return every fixed candidate: " + ", ".join(filtered_a)
        )
    if filtered_b:
        failures.append(
            "profile B did not return every fixed candidate: " + ", ".join(filtered_b)
        )
    if filtered_minimal:
        failures.append(
            "minimal appearance profile did not return every fixed candidate: "
            + ", ".join(filtered_minimal)
        )
    if not outputs_identical:
        failures.append(
            "appearance-scope order, scores, or recommendation evidence changed "
            "with lifestyle fields or their omission"
        )

    evidence.update(
        {
            "profile_a_order": order_a,
            "profile_b_order": order_b,
            "minimal_profile_order": order_minimal,
            "profile_a_output_sha256": canonical_sha256(view_a),
            "profile_b_output_sha256": canonical_sha256(view_b),
            "minimal_profile_output_sha256": canonical_sha256(view_minimal),
            "outputs_identical": outputs_identical,
            "filtered_candidate_ids_profile_a": filtered_a,
            "filtered_candidate_ids_profile_b": filtered_b,
            "filtered_candidate_ids_minimal_profile": filtered_minimal,
            "shared_output": view_a if outputs_identical else None,
            "profile_a_output": None if outputs_identical else view_a,
            "profile_b_output": None if outputs_identical else view_b,
            "minimal_profile_output": None if outputs_identical else view_minimal,
        }
    )
    return _case(
        "appearance_rerank_lifestyle_isolation",
        "생활정보가 외형 범위 순위·점수·근거에 영향을 주지 않음",
        evidence,
        failures,
    )


def _inquiry_case(config: Mapping[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    profile_specs = _require_mapping(config["profiles"], label="profiles")
    inquiry_specs = _require_mapping(
        config["inquiry_preferences"],
        label="inquiry_preferences",
    )
    profile_a = UserProfile.model_validate(profile_specs["a"]).model_dump(mode="json")
    profile_b = UserProfile.model_validate(profile_specs["b"]).model_dump(mode="json")
    preferences_a = InquiryPreferences.model_validate(inquiry_specs["a"])
    preferences_b = InquiryPreferences.model_validate(inquiry_specs["b"])
    inquiry_a = preferences_a.model_dump(mode="json")
    inquiry_b = preferences_b.model_dump(mode="json")

    alignment_failures: list[str] = []
    for field in INQUIRY_PROFILE_FIELDS:
        if inquiry_a[field] != profile_a[field]:
            alignment_failures.append(f"A:{field}")
        if inquiry_b[field] != profile_b[field]:
            alignment_failures.append(f"B:{field}")
    if alignment_failures:
        failures.append(
            "inquiry preferences do not match their profile lifestyle fields: "
            + ", ".join(alignment_failures)
        )

    inquiry_notice = _require_mapping(
        config.get("inquiry_notice", {}),
        label="inquiry_notice",
    )
    materials_a = build_inquiry_materials(inquiry_notice, preferences_a)
    materials_b = build_inquiry_materials(inquiry_notice, preferences_b)
    questions_a = materials_a["questions"]
    questions_b = materials_b["questions"]
    only_a = [question for question in questions_a if question not in questions_b]
    only_b = [question for question in questions_b if question not in questions_a]
    if questions_a == questions_b:
        failures.append("different inquiry preferences produced identical questions")
    if not only_a or not only_b:
        failures.append(
            "each preference set must add at least one distinct shelter question"
        )

    evidence = {
        "same_notice_input": dict(inquiry_notice),
        "profile_to_inquiry_alignment_failures": alignment_failures,
        "profile_a_question_count": len(questions_a),
        "profile_b_question_count": len(questions_b),
        "profile_a_questions_sha256": canonical_sha256(questions_a),
        "profile_b_questions_sha256": canonical_sha256(questions_b),
        "questions_differ": questions_a != questions_b,
        "profile_a_unique_questions": only_a,
        "profile_b_unique_questions": only_b,
    }
    return _case(
        "inquiry_preferences_change_questions",
        "생활·성격 선호는 보호소 문의 질문을 바꿈",
        evidence,
        failures,
    )


def _condition_leakage_case(
    config: Mapping[str, Any],
    query_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    failures: list[str] = []
    spec = _require_mapping(config["conditions_case"], label="conditions_case")
    raw_conditions = _require_mapping(
        spec.get("raw_conditions"),
        label="conditions_case.raw_conditions",
    )
    expected_conditions = _require_mapping(
        spec.get("expected_conditions"),
        label="conditions_case.expected_conditions",
    )
    forbidden_keys = [
        str(value) for value in spec.get("forbidden_keys", []) if str(value)
    ]
    forbidden_terms = [
        str(value) for value in spec.get("forbidden_terms", []) if str(value)
    ]
    filtered = appearance_search_conditions(raw_conditions)
    remaining_forbidden_keys = sorted(set(filtered) & set(forbidden_keys))
    unsupported_keys = sorted(set(filtered) - set(APPEARANCE_SEARCH_CONDITION_KEYS))
    serialized_filtered = json.dumps(
        filtered,
        ensure_ascii=False,
        sort_keys=True,
    ).casefold()
    condition_term_hits = [
        term for term in forbidden_terms if term.casefold() in serialized_filtered
    ]
    query_term_hits = {
        str(row["id"]): list(row["forbidden_hits"])
        for row in query_rows
        if row["forbidden_hits"]
    }

    if filtered != dict(expected_conditions):
        failures.append(
            "filtered appearance conditions differ from expected conditions"
        )
    if remaining_forbidden_keys:
        failures.append(
            "forbidden condition keys remained: " + ", ".join(remaining_forbidden_keys)
        )
    if unsupported_keys:
        failures.append(
            "non-appearance condition keys remained: " + ", ".join(unsupported_keys)
        )
    if condition_term_hits:
        failures.append(
            "forbidden condition expressions remained: "
            + ", ".join(condition_term_hits)
        )
    if query_term_hits:
        failures.append(
            "forbidden query expressions remained after normalization: "
            + ", ".join(sorted(query_term_hits))
        )

    evidence = {
        "raw_conditions": dict(raw_conditions),
        "filtered_conditions": filtered,
        "expected_conditions": dict(expected_conditions),
        "allowed_condition_keys": sorted(APPEARANCE_SEARCH_CONDITION_KEYS),
        "remaining_forbidden_keys": remaining_forbidden_keys,
        "unsupported_keys": unsupported_keys,
        "condition_forbidden_term_hits": condition_term_hits,
        "query_forbidden_term_hits": query_term_hits,
    }
    return _case(
        "nonappearance_expression_leakage",
        "금지된 성격·생활 표현이 외형 검색 입력에 남지 않음",
        evidence,
        failures,
    )


def evaluate_safety_contract(
    config_path: Path,
    metas_path: Path,
) -> dict[str, Any]:
    """Evaluate the four fixed non-inference contracts."""

    config = _load_object(config_path, label="safety contract config")
    _validate_config(config)
    joined_metas, metadata_rows = load_joined_metas(metas_path)
    query_rows, query_failures = _query_pair_evidence(config["query_pairs"])

    cases = [
        _case(
            "appearance_query_equivalence",
            "혼합 질의와 외형 전용 질의가 같은 검색 질의로 정규화됨",
            {"pairs": query_rows},
            query_failures,
        ),
        _profile_isolation_case(config, joined_metas),
        _inquiry_case(config),
        _condition_leakage_case(config, query_rows),
    ]
    failed_cases = [case["id"] for case in cases if not case["passed"]]
    failure_details = [
        f"{case['id']}: {failure}" for case in cases for failure in case["failures"]
    ]
    candidate_ids = [str(value).strip() for value in config["candidate_ids"]]
    available_candidates = [
        joined_metas[notice_id]
        for notice_id in candidate_ids
        if notice_id in joined_metas
    ]
    return {
        "schema_version": SAFETY_CONTRACT_SCHEMA_VERSION,
        "evaluation_id": str(
            config.get("evaluation_id", "safety_contract.appearance_v1")
        ),
        "evaluation_reference_date": str(config["reference_date"]),
        "scope": "appearance_only_non_inference_contract",
        "method": {
            "contract_type": "deterministic regression, not predictive performance",
            "expected_query_policy": (
                "literal NFKC plus whitespace canonicalization; never passed through "
                "the production appearance query normalizer"
            ),
            "required_appearance_terms_must_survive": True,
            "production_functions": [
                "app.appearance_query.normalize_appearance_query",
                "app.appearance_query.appearance_search_conditions",
                "app.profile_rerank.rerank_candidates",
                "app.inquiry_helper.build_inquiry_materials",
            ],
            "ranking_condition_keys": sorted(APPEARANCE_CONDITION_KEYS),
            "lifestyle_or_temperament_scored": False,
            "adoption_suitability_evaluated": False,
        },
        "inputs": {
            "config": portable_input_path(config_path),
            "config_sha256": sha256_file(config_path),
            "dog_metas": portable_input_path(metas_path),
            "dog_metas_sha256": sha256_file(metas_path),
            "dog_metas_rows": metadata_rows,
            "unique_notices": len(joined_metas),
            "fixed_candidate_ids": candidate_ids,
            "available_candidate_count": len(available_candidates),
            "candidate_subset_sha256": canonical_sha256(available_candidates),
        },
        "summary": {
            "passed": not failed_cases,
            "total_contracts": len(cases),
            "passed_contracts": len(cases) - len(failed_cases),
            "failed_contracts": len(failed_cases),
            "failed_contract_ids": failed_cases,
            "failure_details": failure_details,
        },
        "cases": cases,
        "limitations": [
            "이 평가는 입양 적합성 성능 평가가 아니라 생활·성격 정보를 외형 검색 순위에 사용하지 않는 비추론 계약 회귀입니다.",
            "고정 질의와 고정 공고 후보에 대한 결정적 회귀이므로 모든 자연어 표현을 포괄하지 않습니다.",
            "보호소 문의 질문이 달라지는지만 확인하며, 공고 사진이나 품종으로 성격·아동 친화성·다른 동물과의 사회성을 추정하지 않습니다.",
            "실제 성격과 생활 적합성은 이 보고서로 확정할 수 없습니다.",
        ],
    }


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render a human-reviewable safety contract report."""

    summary = report["summary"]
    status = "PASS" if summary["passed"] else "FAIL"
    lines = [
        "# 외형 중심 비추론 안전 계약 평가",
        "",
        "> 이 평가는 입양 적합성 성능 평가가 아닙니다. 생활·성격 정보를 외형 검색 순위에 사용하지 않는지 확인하는 결정적 회귀 평가입니다.",
        "",
        f"- 전체 결과: **{status}**",
        (
            f"- 통과: **{summary['passed_contracts']}/{summary['total_contracts']}**, "
            f"실패: **{summary['failed_contracts']}**"
        ),
        f"- 평가 기준일: `{report['evaluation_reference_date']}`",
        f"- 설정 SHA-256: `{report['inputs']['config_sha256']}`",
        f"- dog_metas SHA-256: `{report['inputs']['dog_metas_sha256']}`",
        f"- 고정 후보: {len(report['inputs']['fixed_candidate_ids'])}개",
        "",
        "## 계약 결과",
        "",
        "| 계약 | 결과 | 실패 원인 |",
        "|---|---|---|",
    ]
    for case in report["cases"]:
        failure_text = "; ".join(case["failures"]) or "-"
        lines.append(
            f"| {_markdown_cell(case['title'])} | "
            f"{'PASS' if case['passed'] else 'FAIL'} | "
            f"{_markdown_cell(failure_text)} |"
        )

    case_by_id = {case["id"]: case for case in report["cases"]}
    query_case = case_by_id["appearance_query_equivalence"]
    lines.extend(
        [
            "",
            "## 질의 정규화 증거",
            "",
            "| 사례 | 혼합 질의 정규화 | 독립 literal 기대값 | 필수 외형어 누락 | 동일 |",
            "|---|---|---|---|---:|",
        ]
    )
    for row in query_case["evidence"]["pairs"]:
        lines.append(
            f"| {_markdown_cell(row['id'])} | "
            f"`{_markdown_cell(row['normalized_mixed_query'])}` | "
            f"`{_markdown_cell(row['expected_literal_query'])}` | "
            f"{_markdown_cell(', '.join(row['missing_required_appearance_terms']) or '-')} | "
            f"{'예' if row['equivalent'] else '아니오'} |"
        )

    profile_case = case_by_id["appearance_rerank_lifestyle_isolation"]
    profile_evidence = profile_case["evidence"]
    lines.extend(
        [
            "",
            "## 외형 범위 재정렬 격리 증거",
            "",
            (
                "- 외형 선호 동일: "
                f"**{'예' if profile_evidence['appearance_fields_equal'] else '아니오'}**"
            ),
            (
                "- 두 프로필에서 순서·점수·근거 동일: "
                f"**{'예' if profile_evidence['outputs_identical'] else '아니오'}**"
            ),
            (
                "- 최소 appearance 프로필 필드: `"
                + "`, `".join(profile_evidence["minimal_profile_fields"])
                + "`"
            ),
            (
                "- 최소 프로필에서 생략된 생활필드: `"
                + "`, `".join(
                    profile_evidence["minimal_profile_omitted_lifestyle_fields"]
                )
                + "`"
            ),
            (
                "- 프로필 A 출력 SHA-256: "
                f"`{profile_evidence.get('profile_a_output_sha256', '-')}`"
            ),
            (
                "- 프로필 B 출력 SHA-256: "
                f"`{profile_evidence.get('profile_b_output_sha256', '-')}`"
            ),
            (
                "- 최소 프로필 출력 SHA-256: "
                f"`{profile_evidence.get('minimal_profile_output_sha256', '-')}`"
            ),
            (
                "- 누락 후보 ID: "
                + (
                    ", ".join(profile_evidence["missing_candidate_ids"])
                    if profile_evidence["missing_candidate_ids"]
                    else "없음"
                )
            ),
            "",
            "고정 후보 순서:",
            "",
            "```text",
            " -> ".join(profile_evidence["profile_a_order"]) or "(결과 없음)",
            "```",
        ]
    )

    inquiry_case = case_by_id["inquiry_preferences_change_questions"]
    inquiry_evidence = inquiry_case["evidence"]
    lines.extend(
        [
            "",
            "## 문의 질문 분리 증거",
            "",
            (
                "- 두 문의 질문 목록이 다름: "
                f"**{'예' if inquiry_evidence['questions_differ'] else '아니오'}**"
            ),
            (
                f"- 프로필 A 질문 {inquiry_evidence['profile_a_question_count']}개, "
                f"고유 질문 {len(inquiry_evidence['profile_a_unique_questions'])}개"
            ),
            (
                f"- 프로필 B 질문 {inquiry_evidence['profile_b_question_count']}개, "
                f"고유 질문 {len(inquiry_evidence['profile_b_unique_questions'])}개"
            ),
            "",
            "생활·성격 선호는 순위 점수가 아니라 보호소에서 실제 관찰 여부를 확인하는 질문에만 사용됩니다.",
        ]
    )

    leakage_case = case_by_id["nonappearance_expression_leakage"]
    leakage_evidence = leakage_case["evidence"]
    lines.extend(
        [
            "",
            "## 금지 표현 누출 검사",
            "",
            "필터링된 검색 조건:",
            "",
            "```json",
            json.dumps(
                leakage_evidence["filtered_conditions"],
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            ),
            "```",
            (
                "- 남은 금지 조건 키: "
                + (
                    ", ".join(leakage_evidence["remaining_forbidden_keys"])
                    if leakage_evidence["remaining_forbidden_keys"]
                    else "없음"
                )
            ),
            (
                "- 질의 내 금지 표현 적중: "
                + (
                    json.dumps(
                        leakage_evidence["query_forbidden_term_hits"],
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    if leakage_evidence["query_forbidden_term_hits"]
                    else "없음"
                )
            ),
            "",
            "## 한계",
            "",
        ]
    )
    lines.extend(f"- {limitation}" for limitation in report["limitations"])
    if summary["failure_details"]:
        lines.extend(
            [
                "",
                "## 실패 상세",
                "",
                *[f"- {failure}" for failure in summary["failure_details"]],
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def serialize_report(report: Mapping[str, Any]) -> str:
    return (
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ).rstrip()
        + "\n"
    )


def write_report(
    report: Mapping[str, Any],
    json_path: Path,
    markdown_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(serialize_report(report), encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
