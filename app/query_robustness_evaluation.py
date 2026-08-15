"""Pure helpers for the appearance-query development regression evaluation.

Negated text changes the request, so its scores against the original qrels are
reported only as an unsupported-behavior diagnostic.  The original v1 variants
were fixed before their first measurement, but later informed query hardening;
they are therefore a development set now, not an independent holdout.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from app.appearance_query import (
    analyze_appearance_query,
    appearance_query_excludes_nonvisual_terms,
    normalize_appearance_query,
)
from app.retrieval_evaluation import (
    METRIC_KEYS,
    SILVER_LABEL_VERSION,
    clean_text,
    deduplicate_ids,
    qrels_record,
    query_metrics,
)


REPORT_SCHEMA_VERSION = "appearance-query-robustness-report.v2"
VARIANT_SCHEMA_VERSION = "appearance-query-robustness-variants.v1"
DEVELOPMENT_DESIGN_STATUS = "development_regression_after_first_measurement"
SYSTEM_ID = "appearance_natural_graph"
STABILITY_KEYS = ("top5_jaccard", "top10_jaccard", "rbo@10")
NEGATION_RE = re.compile(r"(?:아닌|않|제외|초과|미만)")

CATEGORY_POLICY: dict[str, dict[str, Any]] = {
    "synonym_paraphrase": {
        "semantic_relation": "meaning_preserving_manual_design",
        "included_in_robustness_aggregate": True,
        "description": "동일한 외형 조건을 동의어 또는 짧은 바꿔쓰기로 표현",
    },
    "word_order": {
        "semantic_relation": "meaning_preserving_manual_design",
        "included_in_robustness_aggregate": True,
        "description": "동일한 조건의 어순만 변경",
    },
    "conservative_typo": {
        "semantic_relation": "intended_meaning_preserving_one_codepoint_edit",
        "included_in_robustness_aggregate": True,
        "description": "사람이 원래 의도를 복원할 수 있는 한 글자 오타",
    },
    "lifestyle_personality_noise": {
        "semantic_relation": "same_appearance_intent_after_policy_filter",
        "included_in_robustness_aggregate": True,
        "description": "외형 모드가 제거해야 하는 생활·성격 문구를 앞에 추가",
    },
    "negation_diagnostic": {
        "semantic_relation": "changed_semantics_unsupported_diagnostic",
        "included_in_robustness_aggregate": False,
        "description": "원 질의의 한 조건을 부정한 별도 진단; 동등 질의가 아님",
    },
}
ROBUSTNESS_CATEGORIES = tuple(
    category
    for category, policy in CATEGORY_POLICY.items()
    if policy["included_in_robustness_aggregate"]
)


def _sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _levenshtein(left: str, right: str) -> int:
    """Return Unicode-codepoint edit distance without external dependencies."""

    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def validate_variant_payload(
    payload: Mapping[str, Any],
    base_query_payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Validate and expand the pre-registered one-variant-per-category design."""

    if payload.get("schema_version") != VARIANT_SCHEMA_VERSION:
        raise ValueError(f"variant schema must be {VARIANT_SCHEMA_VERSION}")
    if payload.get("base_query_set") != base_query_payload.get("schema_version"):
        raise ValueError("variant base_query_set does not match the query set")
    if payload.get("design_status") != DEVELOPMENT_DESIGN_STATUS:
        raise ValueError(
            "variant design_status must disclose post-measurement development use"
        )
    if payload.get("evaluation_role") != "development_regression_set":
        raise ValueError("variant evaluation_role must be development_regression_set")
    if payload.get("independent_holdout_required") is not True:
        raise ValueError("variant payload must require an independent holdout")
    if not clean_text(payload.get("tuning_disclosure")):
        raise ValueError("variant tuning_disclosure is required")
    if not clean_text(payload.get("tuning_policy")):
        raise ValueError("variant tuning_policy is required")

    base_queries = base_query_payload.get("queries")
    variant_queries = payload.get("queries")
    if not _sequence(base_queries) or not base_queries:
        raise ValueError("base query payload must contain a non-empty queries array")
    if not _sequence(variant_queries) or not variant_queries:
        raise ValueError("variant payload must contain a non-empty queries array")
    if not all(isinstance(row, Mapping) for row in base_queries):
        raise ValueError("every base query must be an object")
    if not all(isinstance(row, Mapping) for row in variant_queries):
        raise ValueError("every variant query entry must be an object")

    base_ids = [clean_text(row.get("id")) for row in base_queries]
    if any(not value for value in base_ids) or len(set(base_ids)) != len(base_ids):
        raise ValueError("base query IDs must be present and unique")
    variant_base_ids = [clean_text(row.get("base_id")) for row in variant_queries]
    if variant_base_ids != base_ids:
        raise ValueError("variant base IDs and order must exactly match base queries")

    base_by_id = {clean_text(row.get("id")): row for row in base_queries}
    expanded: list[dict[str, Any]] = []
    seen_texts: set[tuple[str, str]] = set()
    expected_categories = set(CATEGORY_POLICY)
    for row in variant_queries:
        base_id = clean_text(row.get("base_id"))
        variants = row.get("variants")
        if not isinstance(variants, Mapping):
            raise ValueError(f"{base_id}: variants must be an object")
        if set(variants) != expected_categories:
            raise ValueError(
                f"{base_id}: variants must contain each fixed category exactly once"
            )
        base_query = clean_text(base_by_id[base_id].get("query"))
        if not base_query:
            raise ValueError(f"{base_id}: base query text is missing")
        normalized_base = normalize_appearance_query(base_query)
        for category in CATEGORY_POLICY:
            query = clean_text(variants.get(category))
            if not query or query == base_query:
                raise ValueError(
                    f"{base_id}/{category}: distinct query text is required"
                )
            text_key = (category, query)
            if text_key in seen_texts:
                raise ValueError(
                    f"duplicate development variant text: {category}/{query}"
                )
            seen_texts.add(text_key)
            query_analysis = analyze_appearance_query(query)
            normalized_query = clean_text(query_analysis.get("normalized_query"))
            if category == "conservative_typo" and _levenshtein(base_query, query) != 1:
                raise ValueError(
                    f"{base_id}: conservative typo must be exactly one codepoint edit"
                )
            if category == "lifestyle_personality_noise":
                if not appearance_query_excludes_nonvisual_terms(query):
                    raise ValueError(
                        f"{base_id}: noise variant must trigger appearance filtering"
                    )
                if normalized_query != normalized_base:
                    raise ValueError(
                        f"{base_id}: noise variant must normalize exactly to its base"
                    )
            if category == "negation_diagnostic" and not NEGATION_RE.search(query):
                raise ValueError(f"{base_id}: negation diagnostic lacks negation text")
            unsupported_conditions = query_analysis.get("unsupported_conditions") or []
            if category == "negation_diagnostic" and not unsupported_conditions:
                raise ValueError(
                    f"{base_id}: negation diagnostic did not trigger fail-safe removal"
                )
            if category != "negation_diagnostic" and unsupported_conditions:
                raise ValueError(
                    f"{base_id}/{category}: meaning-preserving variant triggered an "
                    "unsupported-condition warning"
                )
            expanded.append(
                {
                    "id": f"{base_id}::{category}",
                    "base_id": base_id,
                    "category": category,
                    "query": query,
                    "normalized_query": normalized_query,
                    "query_policy": {
                        "typo_corrections": query_analysis.get("typo_corrections")
                        or [],
                        "synonym_normalizations": query_analysis.get(
                            "synonym_normalizations"
                        )
                        or [],
                        "unsupported_conditions": unsupported_conditions,
                        "warnings": query_analysis.get("warnings") or [],
                        "is_fully_supported": bool(
                            query_analysis.get("is_fully_supported")
                        ),
                    },
                    **CATEGORY_POLICY[category],
                }
            )
    return expanded


def top_k_jaccard(
    left: Sequence[Any],
    right: Sequence[Any],
    k: int,
) -> float:
    left_ids = set(deduplicate_ids(left)[: max(1, int(k))])
    right_ids = set(deduplicate_ids(right)[: max(1, int(k))])
    union = left_ids | right_ids
    return round(len(left_ids & right_ids) / len(union), 6) if union else 1.0


def rank_biased_overlap(
    left: Sequence[Any],
    right: Sequence[Any],
    *,
    depth: int = 10,
    persistence: float = 0.9,
) -> float:
    """Finite extrapolated RBO over the shared available prefix."""

    left_ids = deduplicate_ids(left)
    right_ids = deduplicate_ids(right)
    limit = min(max(1, int(depth)), len(left_ids), len(right_ids))
    if limit <= 0:
        return 1.0 if not left_ids and not right_ids else 0.0
    left_seen: set[str] = set()
    right_seen: set[str] = set()
    weighted = 0.0
    agreement = 0.0
    for rank in range(1, limit + 1):
        left_seen.add(left_ids[rank - 1])
        right_seen.add(right_ids[rank - 1])
        agreement = len(left_seen & right_seen) / rank
        weighted += (1.0 - persistence) * agreement * (persistence ** (rank - 1))
    return round(weighted + agreement * (persistence**limit), 6)


def rank_stability(
    baseline_ranking: Sequence[Any],
    variant_ranking: Sequence[Any],
) -> dict[str, Any]:
    baseline = deduplicate_ids(baseline_ranking)
    variant = deduplicate_ids(variant_ranking)
    return {
        "top5_jaccard": top_k_jaccard(baseline, variant, 5),
        "top10_jaccard": top_k_jaccard(baseline, variant, 10),
        "rbo@10": rank_biased_overlap(baseline, variant, depth=10),
        "top10_exact_match": baseline[:10] == variant[:10],
    }


def _mean_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "query_count": len(rows),
        **{
            key: round(
                mean(float((row.get("metrics") or {}).get(key, 0.0)) for row in rows),
                6,
            )
            if rows
            else 0.0
            for key in METRIC_KEYS
        },
    }


def _metric_delta(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, float]:
    return {
        key: round(
            float(candidate.get(key, 0.0)) - float(baseline.get(key, 0.0)),
            6,
        )
        for key in METRIC_KEYS
    }


def _mean_stability(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "variant_count": len(rows),
        **{
            f"mean_{key}": round(
                mean(float((row.get("stability") or {}).get(key, 0.0)) for row in rows),
                6,
            )
            if rows
            else 0.0
            for key in STABILITY_KEYS
        },
        **{
            f"min_{key}": round(
                min(float((row.get("stability") or {}).get(key, 0.0)) for row in rows),
                6,
            )
            if rows
            else 0.0
            for key in STABILITY_KEYS
        },
        "top10_exact_match_rate": round(
            mean(
                bool((row.get("stability") or {}).get("top10_exact_match"))
                for row in rows
            ),
            6,
        )
        if rows
        else 0.0,
    }


def _result_row(
    *,
    result_id: str,
    query: str,
    normalized_query: str,
    ranked_ids: Sequence[Any],
    relevant_ids: Iterable[Any],
    metas_by_id: Mapping[str, Mapping[str, Any]],
    reference_date: datetime,
    ranking_depth: int,
) -> dict[str, Any]:
    ranking = deduplicate_ids(ranked_ids)[: max(10, int(ranking_depth))]
    return {
        "id": result_id,
        "query": query,
        "normalized_query": normalized_query,
        "ranking_ids": ranking,
        "metrics": query_metrics(
            ranking,
            relevant_ids,
            metas_by_id,
            reference_date,
        ),
    }


def _observations(
    row: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> list[str]:
    observations: list[str] = []
    stability = row.get("stability") or {}
    metrics = row.get("metrics") or {}
    baseline_metrics = baseline.get("metrics") or {}
    if float(stability.get("top10_jaccard", 0.0)) < 1.0:
        observations.append("top10_membership_changed")
    if not bool(stability.get("top10_exact_match")):
        observations.append("top10_order_changed")
    if any(
        float(metrics.get(key, 0.0)) < float(baseline_metrics.get(key, 0.0))
        for key in ("precision@5", "nDCG@5", "MRR", "hit@5")
    ):
        observations.append("silver_metric_decreased")
    if float(metrics.get("inactive_exposure", 0.0)) > 0.0:
        observations.append("inactive_notice_exposed")
    if row.get("category") == "lifestyle_personality_noise" and row.get(
        "normalized_query"
    ) != baseline.get("normalized_query"):
        observations.append("appearance_noise_not_removed")
    return observations


def build_report(
    *,
    generated_at: str,
    reference_date: datetime,
    base_query_payload: Mapping[str, Any],
    variant_payload: Mapping[str, Any],
    qrels: Mapping[str, set[str]],
    metas_by_id: Mapping[str, Mapping[str, Any]],
    base_rankings: Mapping[str, Sequence[Any]],
    variant_rankings: Mapping[str, Sequence[Any]],
    artifacts: Mapping[str, Any],
    runtime: Mapping[str, Any],
    corpus: Mapping[str, Any],
) -> dict[str, Any]:
    variants = validate_variant_payload(variant_payload, base_query_payload)
    base_queries = base_query_payload["queries"]
    ranking_depth = max(10, int(runtime.get("evaluation_depth") or 10))

    base_rows: list[dict[str, Any]] = []
    base_by_id: dict[str, dict[str, Any]] = {}
    for query_spec in base_queries:
        base_id = clean_text(query_spec.get("id"))
        if base_id not in base_rankings:
            raise ValueError(f"missing base ranking: {base_id}")
        row = _result_row(
            result_id=base_id,
            query=clean_text(query_spec.get("query")),
            normalized_query=normalize_appearance_query(query_spec.get("query")),
            ranked_ids=base_rankings[base_id],
            relevant_ids=qrels.get(base_id, set()),
            metas_by_id=metas_by_id,
            reference_date=reference_date,
            ranking_depth=ranking_depth,
        )
        row["relevant_count"] = len(qrels.get(base_id, set()))
        base_rows.append(row)
        base_by_id[base_id] = row

    category_rows: dict[str, list[dict[str, Any]]] = {
        category: [] for category in CATEGORY_POLICY
    }
    for spec in variants:
        variant_id = spec["id"]
        base_id = spec["base_id"]
        if variant_id not in variant_rankings:
            raise ValueError(f"missing variant ranking: {variant_id}")
        row = _result_row(
            result_id=variant_id,
            query=spec["query"],
            normalized_query=spec["normalized_query"],
            ranked_ids=variant_rankings[variant_id],
            relevant_ids=qrels.get(base_id, set()),
            metas_by_id=metas_by_id,
            reference_date=reference_date,
            ranking_depth=ranking_depth,
        )
        row.update(
            {
                "base_id": base_id,
                "category": spec["category"],
                "semantic_relation": spec["semantic_relation"],
                "included_in_robustness_aggregate": spec[
                    "included_in_robustness_aggregate"
                ],
                "query_policy": spec["query_policy"],
                "relevant_count": len(qrels.get(base_id, set())),
            }
        )
        base_row = base_by_id[base_id]
        row["stability"] = rank_stability(base_row["ranking_ids"], row["ranking_ids"])
        row["metric_delta_from_base"] = _metric_delta(
            base_row["metrics"], row["metrics"]
        )
        row["observations"] = _observations(row, base_row)
        category_rows[spec["category"]].append(row)

    baseline_metrics = _mean_metrics(base_rows)
    categories: dict[str, Any] = {}
    for category, policy in CATEGORY_POLICY.items():
        rows = category_rows[category]
        metrics = _mean_metrics(rows)
        categories[category] = {
            **policy,
            "summary": {
                "metrics": metrics,
                "metric_delta_from_base_average": _metric_delta(
                    baseline_metrics, metrics
                ),
                "stability": _mean_stability(rows),
                "observation_counts": dict(
                    sorted(
                        Counter(
                            observation
                            for row in rows
                            for observation in row["observations"]
                        ).items()
                    )
                ),
            },
            "variants": rows,
        }

    robustness_rows = [
        row for category in ROBUSTNESS_CATEGORIES for row in category_rows[category]
    ]
    robustness_metrics = _mean_metrics(robustness_rows)
    worst_cases = sorted(
        robustness_rows,
        key=lambda row: (
            float(row["stability"]["top10_jaccard"]),
            float(row["stability"]["rbo@10"]),
            float(row["metric_delta_from_base"]["nDCG@5"]),
            row["id"],
        ),
    )[:12]
    worst_case_records = [
        {
            "id": row["id"],
            "base_id": row["base_id"],
            "category": row["category"],
            "query": row["query"],
            "normalized_query": row["normalized_query"],
            "stability": row["stability"],
            "metric_delta_from_base": row["metric_delta_from_base"],
            "observations": row["observations"],
        }
        for row in worst_cases
    ]
    all_rows = [
        *base_rows,
        *(row for rows in category_rows.values() for row in rows),
    ]
    inactive_values = [
        float((row.get("metrics") or {}).get("inactive_exposure", 0.0))
        for row in all_rows
    ]

    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "reference_date": reference_date.strftime("%Y-%m-%d"),
        "system": {
            "id": SYSTEM_ID,
            "description": (
                "Natural-language hybrid+graph retrieval after appearance-query "
                "normalization, with temperament graph features excluded."
            ),
            "uses_explicit_structured_hints": False,
            "negation_supported": False,
            "negated_objective_fail_safe_removal": True,
        },
        "study_design": {
            "evaluation_role": variant_payload.get("evaluation_role"),
            "design_status": variant_payload.get("design_status"),
            "tuning_disclosure": variant_payload.get("tuning_disclosure"),
            "independent_holdout_required": variant_payload.get(
                "independent_holdout_required"
            ),
            "interpretation": (
                "The unchanged v1 sentences are useful regression cases, but the "
                "observed v1 failures informed the current normalizer. These metrics "
                "are development-set results and cannot estimate unseen-query "
                "generalization without a separately frozen holdout."
            ),
        },
        "label_policy": {
            "version": SILVER_LABEL_VERSION,
            "source_fields": [
                "public_notice.color",
                "public_notice.weight",
                "public_notice.age",
                "public_notice.region",
                "public_notice.status",
            ],
            "uses_result_text": False,
            "uses_ranked_results": False,
            "uses_vlm_attributes": False,
            "relevance": "binary_all_requested_dimensions_match",
        },
        "artifacts": dict(artifacts),
        "runtime": dict(runtime),
        "corpus": dict(corpus),
        "qrels": qrels_record(base_queries, qrels),
        "baseline": {
            "summary": {"metrics": baseline_metrics},
            "queries": base_rows,
        },
        "categories": categories,
        "robustness_summary": {
            "included_categories": list(ROBUSTNESS_CATEGORIES),
            "excluded_categories": ["negation_diagnostic"],
            "metrics": robustness_metrics,
            "metric_delta_from_base_average": _metric_delta(
                baseline_metrics, robustness_metrics
            ),
            "stability": _mean_stability(robustness_rows),
            "observation_counts": dict(
                sorted(
                    Counter(
                        observation
                        for row in robustness_rows
                        for observation in row["observations"]
                    ).items()
                )
            ),
            "worst_cases": worst_case_records,
        },
        "negation_diagnostic": {
            "category": "negation_diagnostic",
            "included_in_robustness_aggregate": False,
            "interpretation": (
                "The text changes semantics. Metrics reuse base qrels only to show "
                "fail-safe removal behavior and are not relevance judgments for the "
                "negated request. The system does not claim to satisfy negation."
            ),
        },
        "inactive_exposure": {
            "evaluated_result_count": len(all_rows),
            "mean": round(mean(inactive_values), 6) if inactive_values else 0.0,
            "maximum": round(max(inactive_values), 6) if inactive_values else 0.0,
            "nonzero_result_count": sum(value > 0.0 for value in inactive_values),
        },
        "evaluation_contract": {
            "variant_count": len(variants),
            "base_query_count": len(base_rows),
            "variants_per_category": len(base_rows),
            "robustness_variant_count": len(robustness_rows),
            "performance_thresholds": None,
            "release_invariant": "inactive_exposure_must_equal_zero",
            "noise_invariant": (
                "lifestyle_personality_noise must normalize and rank exactly as base"
            ),
        },
        "limitations": [
            "Silver qrels come from public structured fields, not blinded human relevance labels.",
            "The same public metadata feeds qrels and parts of hybrid ranking, so scores are coupled.",
            "The observed v1 failures informed normalizer hardening, so this is a development regression set, not an independent test set.",
            "A separately frozen, unseen Korean holdout is required before claiming query robustness generalization.",
            "One Korean variant per category and query does not represent all user language.",
            "Synonym and word-order equivalence is manually designed, not independently annotated.",
            "Negation semantics remain unsupported; negated objective terms are removed fail-safe and excluded from every robustness aggregate.",
            "No adoption suitability, temperament, child-friendliness, or pet compatibility is measured.",
        ],
    }
    failures = report_contract_failures(report)
    report["validation"] = {
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "scope": "development-regression safety invariants only; no performance threshold",
    }
    return report


def _all_result_rows(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    baseline = report.get("baseline")
    if isinstance(baseline, Mapping) and _sequence(baseline.get("queries")):
        rows.extend(row for row in baseline["queries"] if isinstance(row, Mapping))
    categories = report.get("categories")
    if isinstance(categories, Mapping):
        for category in CATEGORY_POLICY:
            section = categories.get(category)
            if isinstance(section, Mapping) and _sequence(section.get("variants")):
                rows.extend(
                    row for row in section["variants"] if isinstance(row, Mapping)
                )
    return rows


def report_contract_failures(report: Mapping[str, Any]) -> list[str]:
    """Validate development-regression safety invariants, never quality floors."""

    failures: list[str] = []
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        failures.append("report schema_version is missing or stale")
    contract = report.get("evaluation_contract")
    if not isinstance(contract, Mapping):
        return [*failures, "evaluation_contract is missing"]
    base_count = int(contract.get("base_query_count") or 0)
    if base_count <= 0:
        failures.append("base_query_count must be positive")
    if int(contract.get("variant_count") or 0) != base_count * len(CATEGORY_POLICY):
        failures.append("development variant coverage is incomplete")
    if int(contract.get("robustness_variant_count") or 0) != base_count * len(
        ROBUSTNESS_CATEGORIES
    ):
        failures.append("robustness aggregate coverage is incomplete")
    if contract.get("performance_thresholds") is not None:
        failures.append("descriptive robustness evaluation cannot set a quality floor")

    study_design = report.get("study_design")
    if not isinstance(study_design, Mapping):
        failures.append("study_design disclosure is missing")
    else:
        if study_design.get("evaluation_role") != "development_regression_set":
            failures.append("evaluation role must disclose development-set use")
        if study_design.get("design_status") != DEVELOPMENT_DESIGN_STATUS:
            failures.append("design status does not disclose post-measurement use")
        if study_design.get("independent_holdout_required") is not True:
            failures.append("independent holdout requirement is missing")

    categories = report.get("categories")
    if not isinstance(categories, Mapping) or set(categories) != set(CATEGORY_POLICY):
        failures.append("category set is missing or changed")
        categories = {}
    base = report.get("baseline")
    base_rows = base.get("queries") if isinstance(base, Mapping) else None
    if not _sequence(base_rows) or len(base_rows) != base_count:
        failures.append("baseline query rows do not match the fixed query count")
        base_rows = []
    base_by_id = {
        clean_text(row.get("id")): row for row in base_rows if isinstance(row, Mapping)
    }
    for category, policy in CATEGORY_POLICY.items():
        section = categories.get(category)
        if not isinstance(section, Mapping):
            failures.append(f"category section is missing: {category}")
            continue
        for field in (
            "semantic_relation",
            "included_in_robustness_aggregate",
            "description",
        ):
            if section.get(field) != policy[field]:
                failures.append(f"category policy changed: {category}.{field}")
        variants = section.get("variants")
        if not _sequence(variants) or len(variants) != base_count:
            failures.append(f"category variant count is wrong: {category}")
            continue
        for row in variants:
            if not isinstance(row, Mapping):
                failures.append(f"category contains a non-object row: {category}")
                continue
            if row.get("category") != category:
                failures.append(f"variant category mismatch: {row.get('id')}")
            query_policy = row.get("query_policy")
            unsupported_conditions = (
                query_policy.get("unsupported_conditions")
                if isinstance(query_policy, Mapping)
                else None
            )
            if category == "negation_diagnostic":
                if not _sequence(unsupported_conditions):
                    failures.append(
                        f"negation warning evidence is missing: {row.get('id')}"
                    )
            elif unsupported_conditions:
                failures.append(
                    f"meaning-preserving variant has an unsupported warning: "
                    f"{row.get('id')}"
                )
            base_row = base_by_id.get(clean_text(row.get("base_id")))
            if category == "lifestyle_personality_noise" and isinstance(
                base_row, Mapping
            ):
                if row.get("normalized_query") != base_row.get("normalized_query"):
                    failures.append(f"noise was not normalized away: {row.get('id')}")
                if deduplicate_ids(row.get("ranking_ids") or []) != deduplicate_ids(
                    base_row.get("ranking_ids") or []
                ):
                    failures.append(
                        f"noise changed the appearance ranking: {row.get('id')}"
                    )

    qrels = report.get("qrels")
    if not _sequence(qrels) or len(qrels) != base_count:
        failures.append("qrels do not match the fixed base query count")
    elif any(int((row or {}).get("relevant_count") or 0) <= 0 for row in qrels):
        failures.append(
            "every base query must have at least one silver-relevant notice"
        )

    for row in _all_result_rows(report):
        if float((row.get("metrics") or {}).get("inactive_exposure", 0.0)) != 0.0:
            failures.append(f"inactive notice exposure is nonzero: {row.get('id')}")
    inactive = report.get("inactive_exposure")
    if (
        not isinstance(inactive, Mapping)
        or float(inactive.get("maximum") or 0.0) != 0.0
    ):
        failures.append("aggregate inactive exposure must equal zero")

    negation = report.get("negation_diagnostic")
    if (
        not isinstance(negation, Mapping)
        or negation.get("included_in_robustness_aggregate") is not False
    ):
        failures.append("negation must remain excluded from robustness aggregates")
    return sorted(set(failures))


def report_recalculation_failures(
    report: Mapping[str, Any],
    *,
    reference_date: datetime,
    base_query_payload: Mapping[str, Any],
    variant_payload: Mapping[str, Any],
    qrels: Mapping[str, set[str]],
    metas_by_id: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """Recalculate every metric/summary from stored rankings and current qrels."""

    failures: list[str] = []
    try:
        baseline = report.get("baseline")
        categories = report.get("categories")
        if not isinstance(baseline, Mapping) or not isinstance(categories, Mapping):
            return ["report result sections are missing"]
        base_rows = baseline.get("queries")
        if not _sequence(base_rows):
            return ["baseline query rows are missing"]
        base_rankings = {
            clean_text(row.get("id")): list(row.get("ranking_ids") or [])
            for row in base_rows
            if isinstance(row, Mapping)
        }
        variant_rankings: dict[str, list[Any]] = {}
        for category in CATEGORY_POLICY:
            section = categories.get(category)
            if not isinstance(section, Mapping) or not _sequence(
                section.get("variants")
            ):
                return [f"category rows are missing: {category}"]
            for row in section["variants"]:
                if isinstance(row, Mapping):
                    variant_rankings[clean_text(row.get("id"))] = list(
                        row.get("ranking_ids") or []
                    )
        expected = build_report(
            generated_at=clean_text(report.get("generated_at")),
            reference_date=reference_date,
            base_query_payload=base_query_payload,
            variant_payload=variant_payload,
            qrels=qrels,
            metas_by_id=metas_by_id,
            base_rankings=base_rankings,
            variant_rankings=variant_rankings,
            artifacts=report.get("artifacts") or {},
            runtime=report.get("runtime") or {},
            corpus=report.get("corpus") or {},
        )
    except (KeyError, TypeError, ValueError) as exc:
        return [f"report evidence cannot be recalculated: {exc}"]

    for field in (
        "study_design",
        "qrels",
        "baseline",
        "categories",
        "robustness_summary",
        "negation_diagnostic",
        "inactive_exposure",
        "evaluation_contract",
        "limitations",
        "validation",
    ):
        if report.get(field) != expected.get(field):
            failures.append(f"stored report section is not reproducible: {field}")
    return failures


def _number(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "0.0000"


def _md_text(value: Any) -> str:
    return clean_text(value).replace("|", "\\|").replace("\n", " ")


def report_to_markdown(report: Mapping[str, Any]) -> str:
    """Render the checked JSON evidence as a compact Korean Markdown report."""

    baseline_metrics = ((report.get("baseline") or {}).get("summary") or {}).get(
        "metrics", {}
    )
    robust = report.get("robustness_summary") or {}
    robust_metrics = robust.get("metrics") or {}
    robust_stability = robust.get("stability") or {}
    study_design = report.get("study_design") or {}
    lines = [
        "# 외형 질의 개발셋 회귀·실패 사례 평가",
        "",
        f"- 생성 시각(UTC): `{_md_text(report.get('generated_at'))}`",
        f"- 고정 기준일: `{_md_text(report.get('reference_date'))}`",
        f"- 시스템: `{SYSTEM_ID}`",
        f"- 평가 역할: `{_md_text(study_design.get('evaluation_role'))}`",
        "- 평가 경로: 일반 retrieval headline과 구분된 실제 외형 모드 경로(외형 질의 정규화 후 natural hybrid+graph 검색, temperament 그래프 신호 제외)",
        "- 판정 방식: 품질 합격선 없이 기술 통계만 공개하며, 비활성 공고 노출 0과 외형 모드의 생활·성격 문구 제거만 불변조건으로 검사",
        "- **중요:** 최초 v1 실패를 확인한 뒤 정규화 코드를 보강했으므로 아래 수치는 독립 테스트가 아닌 개발셋 회귀 결과입니다. 일반화 주장을 하려면 별도로 동결한 미관측 한국어 holdout이 필요합니다.",
        "",
        "## 요약",
        "",
        "| 범위 | 질의 수 | P@5 | R@10 | nDCG@5 | MRR | Hit@5 | 비활성 노출 | Jaccard@10 | RBO@10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| 원 질의 | {int(baseline_metrics.get('query_count') or 0)} | "
            f"{_number(baseline_metrics.get('precision@5'))} | "
            f"{_number(baseline_metrics.get('recall@10'))} | "
            f"{_number(baseline_metrics.get('nDCG@5'))} | "
            f"{_number(baseline_metrics.get('MRR'))} | "
            f"{_number(baseline_metrics.get('hit@5'))} | "
            f"{_number(baseline_metrics.get('inactive_exposure'))} | - | - |"
        ),
        (
            f"| 의미 보존 변형 합계 | {int(robust_metrics.get('query_count') or 0)} | "
            f"{_number(robust_metrics.get('precision@5'))} | "
            f"{_number(robust_metrics.get('recall@10'))} | "
            f"{_number(robust_metrics.get('nDCG@5'))} | "
            f"{_number(robust_metrics.get('MRR'))} | "
            f"{_number(robust_metrics.get('hit@5'))} | "
            f"{_number(robust_metrics.get('inactive_exposure'))} | "
            f"{_number(robust_stability.get('mean_top10_jaccard'))} | "
            f"{_number(robust_stability.get('mean_rbo@10'))} |"
        ),
        "",
        "## 변형 범주별 결과",
        "",
        "| 범주 | 집계 포함 | 수 | P@5 | nDCG@5 | Hit@5 | Jaccard@5 | Jaccard@10 | RBO@10 | Top10 완전일치 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    categories = report.get("categories") or {}
    for category in CATEGORY_POLICY:
        section = categories.get(category) or {}
        summary = section.get("summary") or {}
        metrics = summary.get("metrics") or {}
        stability = summary.get("stability") or {}
        included = "예" if section.get("included_in_robustness_aggregate") else "아니오"
        lines.append(
            f"| `{category}` | {included} | {int(metrics.get('query_count') or 0)} | "
            f"{_number(metrics.get('precision@5'))} | "
            f"{_number(metrics.get('nDCG@5'))} | "
            f"{_number(metrics.get('hit@5'))} | "
            f"{_number(stability.get('mean_top5_jaccard'))} | "
            f"{_number(stability.get('mean_top10_jaccard'))} | "
            f"{_number(stability.get('mean_rbo@10'))} | "
            f"{_number(stability.get('top10_exact_match_rate'))} |"
        )

    lines.extend(
        [
            "",
            "`negation_diagnostic`은 원 질의와 의미가 다릅니다. 부정된 외형 조건은 양성 필터로 뒤집히지 않도록 검색 신호에서 제거되지만, 부정 의미 자체를 만족시키지는 않습니다. 원 qrels 대비 수치는 fail-safe 동작 관찰값일 뿐 부정 질의의 검색 정확도가 아닙니다.",
            "",
            "## 가장 불안정한 의미 보존 변형",
            "",
            "| 변형 ID | 범주 | Jaccard@10 | RBO@10 | ΔnDCG@5 | 관찰 사항 |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in robust.get("worst_cases") or []:
        stability = row.get("stability") or {}
        delta = row.get("metric_delta_from_base") or {}
        observations = ", ".join(row.get("observations") or []) or "없음"
        lines.append(
            f"| `{_md_text(row.get('id'))}` | `{_md_text(row.get('category'))}` | "
            f"{_number(stability.get('top10_jaccard'))} | "
            f"{_number(stability.get('rbo@10'))} | "
            f"{_number(delta.get('nDCG@5'))} | {_md_text(observations)} |"
        )

    validation = report.get("validation") or {}
    lines.extend(
        [
            "",
            "## 불변조건 검증",
            "",
            f"- 결과: **{'PASS' if validation.get('passed') else 'FAIL'}**",
            f"- 비활성 공고 최대 노출률: `{_number((report.get('inactive_exposure') or {}).get('maximum'), 6)}`",
            f"- 실패: `{int(validation.get('failure_count') or 0)}`건",
            "",
            "## 재현 근거",
            "",
            "| 아티팩트 | SHA-256 |",
            "|---|---|",
        ]
    )
    artifacts = report.get("artifacts") or {}
    for name, record in artifacts.items():
        if isinstance(record, Mapping):
            lines.append(f"| `{_md_text(name)}` | `{_md_text(record.get('sha256'))}` |")

    lines.extend(["", "## 해석 한계", ""])
    for limitation in report.get("limitations") or []:
        lines.append(f"- {_md_text(limitation)}")
    lines.extend(
        [
            "",
            "이 평가는 사람의 관련성 판정이나 입양 적합성 검증이 아닙니다. v1 문장 자체는 유지했지만 그 실패가 현재 코드 개선에 사용되었으므로 독립 성능 검증으로 해석할 수 없습니다.",
            "",
        ]
    )
    return "\n".join(lines)
