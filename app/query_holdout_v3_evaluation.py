"""Pure report helpers for the frozen appearance-query holdout v3.

This module contains no query normalization or retrieval imports.  The one-shot
runner supplies stored rankings and parser observations so ``--check`` can
recalculate the evidence without invoking the evaluated system.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from app.retrieval_evaluation import (
    METRIC_KEYS,
    clean_text,
    deduplicate_ids,
    qrels_record,
    query_metrics,
)


REPORT_SCHEMA_VERSION = "appearance-query-holdout-report.v3"
HOLDOUT_SCHEMA_VERSION = "appearance-query-holdout.v3"
SYSTEM_ID = "appearance_natural_graph"
BASE_CASE_COUNT = 12
SEMANTIC_CASE_COUNT = 24
NEGATIVE_CASE_COUNT = 6
VARIANTS_PER_BASE = 2
STABILITY_KEYS = ("top5_jaccard", "top10_jaccard", "rbo@10")
DIAGNOSTIC_NAMES = {
    "negated_color_must_not_become_positive",
    "negated_region_must_not_survive_contrast",
    "lower_bound_must_not_become_upper_bound",
    "generic_query_must_not_invent_constraints",
    "unsupported_traits_must_not_invent_constraints",
    "other_species_must_not_invent_dog_appearance_constraints",
}
SYSTEM_CODE_ARTIFACTS = (
    "appearance_query_module",
    "hybrid_rag_module",
    "graph_rag_module",
    "retrieval_evaluation_module",
    "retrieval_evaluation_cli",
)
_APPEARANCE_KEY_FRAGMENTS = (
    "age",
    "coat",
    "color",
    "region",
    "size",
    "weight",
)


def _sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def validate_holdout_payload(
    payload: Mapping[str, Any],
    base_query_payload: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate the frozen v3 authoring artifact without system output."""

    if payload.get("schema_version") != HOLDOUT_SCHEMA_VERSION:
        raise ValueError(f"holdout schema must be {HOLDOUT_SCHEMA_VERSION}")
    if payload.get("base_query_set") != "data/eval_queries.appearance_v1.json":
        raise ValueError("holdout base_query_set is unexpected")

    protocol = payload.get("authoring_protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("holdout authoring_protocol is required")
    expected_protocol = {
        "independent_blind": True,
        "variants_per_base_intent": VARIANTS_PER_BASE,
        "semantic_variant_count": SEMANTIC_CASE_COUNT,
        "negative_diagnostic_count": NEGATIVE_CASE_COUNT,
        "negative_diagnostics_in_aggregate": False,
    }
    for key, expected in expected_protocol.items():
        if protocol.get(key) != expected:
            raise ValueError(f"holdout authoring protocol mismatch: {key}")
    if protocol.get("authoring_inputs") != [
        "data/eval_queries.appearance_v1.json",
        "data/dog_metas.json",
    ]:
        raise ValueError("holdout authoring inputs are unexpected")

    base_queries = base_query_payload.get("queries")
    semantic_cases = payload.get("semantic_variants")
    negative_cases = payload.get("negative_diagnostics")
    if not _sequence(base_queries) or not all(
        isinstance(row, Mapping) for row in base_queries
    ):
        raise ValueError("base query payload must contain an object array")
    if len(base_queries) != BASE_CASE_COUNT:
        raise ValueError(f"base query set must contain {BASE_CASE_COUNT} cases")
    if not _sequence(semantic_cases) or not all(
        isinstance(row, Mapping) for row in semantic_cases
    ):
        raise ValueError("semantic_variants must be an object array")
    if not _sequence(negative_cases) or not all(
        isinstance(row, Mapping) for row in negative_cases
    ):
        raise ValueError("negative_diagnostics must be an object array")
    if len(semantic_cases) != SEMANTIC_CASE_COUNT:
        raise ValueError(
            f"holdout must contain exactly {SEMANTIC_CASE_COUNT} semantic cases"
        )
    if len(negative_cases) != NEGATIVE_CASE_COUNT:
        raise ValueError(
            f"holdout must contain exactly {NEGATIVE_CASE_COUNT} negative cases"
        )

    base_ids = [clean_text(row.get("id")) for row in base_queries]
    if any(not value for value in base_ids) or len(set(base_ids)) != len(base_ids):
        raise ValueError("base query IDs must be present and unique")
    expected_semantic_base_ids = [
        base_id for base_id in base_ids for _ in range(VARIANTS_PER_BASE)
    ]
    semantic_base_ids = [clean_text(row.get("base_id")) for row in semantic_cases]
    if semantic_base_ids != expected_semantic_base_ids:
        raise ValueError(
            "semantic variants must cover every base ID twice and in base order"
        )

    base_text_by_id = {
        clean_text(row.get("id")): clean_text(row.get("query"))
        for row in base_queries
    }
    seen_ids: set[str] = set()
    seen_texts: set[str] = set(base_text_by_id.values())
    normalized_semantic: list[dict[str, Any]] = []
    for position, raw in enumerate(semantic_cases):
        row = dict(raw)
        case_id = clean_text(row.get("id"))
        base_id = clean_text(row.get("base_id"))
        query = clean_text(row.get("query"))
        if not case_id or case_id in seen_ids:
            raise ValueError(f"semantic case {position} has a duplicate or empty ID")
        if base_id not in base_text_by_id:
            raise ValueError(f"{case_id}: unknown base ID")
        if not query or query in seen_texts:
            raise ValueError(f"{case_id}: distinct semantic query text is required")
        seen_ids.add(case_id)
        seen_texts.add(query)
        normalized_semantic.append(row)

    normalized_negative: list[dict[str, Any]] = []
    for position, raw in enumerate(negative_cases):
        row = dict(raw)
        case_id = clean_text(row.get("id"))
        query = clean_text(row.get("query"))
        diagnostic = clean_text(row.get("diagnostic"))
        if not case_id or case_id in seen_ids:
            raise ValueError(f"negative case {position} has a duplicate or empty ID")
        if not query or query in seen_texts:
            raise ValueError(f"{case_id}: distinct negative query text is required")
        if diagnostic not in DIAGNOSTIC_NAMES:
            raise ValueError(f"{case_id}: unsupported negative diagnostic")
        if row.get("aggregate") is not False:
            raise ValueError(f"{case_id}: negative diagnostic must exclude aggregate")
        seen_ids.add(case_id)
        seen_texts.add(query)
        normalized_negative.append(row)
    return normalized_semantic, normalized_negative


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
    """Return finite extrapolated RBO over the shared available prefix."""

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
    holdout_ranking: Sequence[Any],
) -> dict[str, Any]:
    baseline = deduplicate_ids(baseline_ranking)
    holdout = deduplicate_ids(holdout_ranking)
    return {
        "top5_jaccard": top_k_jaccard(baseline, holdout, 5),
        "top10_jaccard": top_k_jaccard(baseline, holdout, 10),
        "rbo@10": rank_biased_overlap(baseline, holdout, depth=10),
        "top10_exact_match": baseline[:10] == holdout[:10],
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
        "case_count": len(rows),
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
    run: Mapping[str, Any],
    relevant_ids: Iterable[Any],
    metas_by_id: Mapping[str, Mapping[str, Any]],
    reference_date: datetime,
    ranking_depth: int,
) -> dict[str, Any]:
    ranking = deduplicate_ids(run.get("ranking_ids") or [])[
        : max(10, int(ranking_depth))
    ]
    return {
        "id": result_id,
        "query": query,
        "normalized_query": clean_text(run.get("normalized_query")),
        "query_policy": dict(run.get("query_policy") or {}),
        "retrieval": dict(run.get("retrieval") or {}),
        "ranking_ids": ranking,
        "metrics": query_metrics(
            ranking,
            relevant_ids,
            metas_by_id,
            reference_date,
        ),
    }


def _semantic_observations(
    row: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> list[str]:
    observations: list[str] = []
    stability = row.get("stability") or {}
    metrics = row.get("metrics") or {}
    baseline_metrics = baseline.get("metrics") or {}
    policy = row.get("query_policy") or {}
    if float(stability.get("top5_jaccard", 0.0)) < 1.0:
        observations.append("top5_membership_changed")
    if float(stability.get("top10_jaccard", 0.0)) < 1.0:
        observations.append("top10_membership_changed")
    if not bool(stability.get("top10_exact_match")):
        observations.append("top10_order_changed")
    decreased = [
        key
        for key in ("precision@5", "recall@5", "recall@10", "nDCG@5", "MRR", "hit@5")
        if float(metrics.get(key, 0.0)) < float(baseline_metrics.get(key, 0.0))
    ]
    if decreased:
        observations.append("silver_metric_decreased:" + ",".join(decreased))
    if policy.get("unsupported_conditions"):
        observations.append("unsupported_condition_signaled")
    if not clean_text(row.get("normalized_query")):
        observations.append("normalized_query_empty")
    if float(metrics.get("inactive_exposure", 0.0)) > 0.0:
        observations.append("inactive_notice_exposed")
    return observations


def _effective_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        output = {
            clean_text(key): effective
            for key, item in value.items()
            if (effective := _effective_value(item)) is not None
        }
        return output or None
    if _sequence(value) or isinstance(value, set):
        output = [
            effective
            for item in value
            if (effective := _effective_value(item)) is not None
        ]
        return output or None
    if isinstance(value, bool):
        return value if value else None
    if isinstance(value, (int, float)):
        return value
    text = clean_text(value)
    return text if text else None


def _flatten_strings(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        return [
            text
            for key, item in value.items()
            for text in (clean_text(key).lower(), *_flatten_strings(item))
            if text
        ]
    if _sequence(value) or isinstance(value, set):
        return [text for item in value for text in _flatten_strings(item)]
    text = clean_text(value).lower()
    return [text] if text else []


def _appearance_constraints(structured: Mapping[str, Any]) -> dict[str, Any]:
    return {
        clean_text(key): effective
        for key, value in structured.items()
        if any(fragment in clean_text(key).lower() for fragment in _APPEARANCE_KEY_FRAGMENTS)
        and (effective := _effective_value(value)) is not None
    }


def _numeric_bounds(value: Any, path: str = "") -> list[tuple[str, float]]:
    bounds: list[tuple[str, float]] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            child_path = f"{path}.{clean_text(key).lower()}".strip(".")
            bounds.extend(_numeric_bounds(item, child_path))
    elif _sequence(value):
        for item in value:
            bounds.extend(_numeric_bounds(item, path))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        bounds.append((path, float(value)))
    return bounds


def negative_diagnostic_failures(row: Mapping[str, Any]) -> list[str]:
    """Evaluate only the preregistered negative safety expectation."""

    diagnostic = clean_text(row.get("diagnostic"))
    retrieval = row.get("retrieval") or {}
    structured_raw = retrieval.get("structured_query") or {}
    structured = dict(structured_raw) if isinstance(structured_raw, Mapping) else {}
    tokens = _flatten_strings(structured)
    failures: list[str] = []
    if diagnostic == "negated_color_must_not_become_positive":
        if any(token in {"brown", "갈색"} for token in tokens):
            failures.append("negated_color_applied_as_positive")
    elif diagnostic == "negated_region_must_not_survive_contrast":
        if any(token in {"seoul", "서울"} for token in tokens):
            failures.append("negated_region_applied_as_positive")
    elif diagnostic == "lower_bound_must_not_become_upper_bound":
        bounds = _numeric_bounds(structured)
        has_minimum = any(
            any(term in path for term in ("min", "minimum", "lower"))
            and value >= 8.0
            for path, value in bounds
        )
        has_upper_only = any(
            any(term in path for term in ("max", "maximum", "upper"))
            and value <= 8.0
            for path, value in bounds
        )
        if has_upper_only and not has_minimum:
            failures.append("lower_bound_applied_as_upper_bound")
    elif diagnostic in {
        "generic_query_must_not_invent_constraints",
        "unsupported_traits_must_not_invent_constraints",
        "other_species_must_not_invent_dog_appearance_constraints",
    }:
        if _appearance_constraints(structured):
            failures.append("dog_appearance_constraints_invented")
    else:
        failures.append("unknown_negative_diagnostic")
    return failures


def _inactive_exposure(
    ranking: Sequence[Any],
    metas_by_id: Mapping[str, Mapping[str, Any]],
    reference_date: datetime,
) -> float:
    return query_metrics(
        ranking,
        (),
        metas_by_id,
        reference_date,
    )["inactive_exposure"]


def build_report(
    *,
    generated_at: str,
    run_started_at: str,
    reference_date: datetime,
    base_query_payload: Mapping[str, Any],
    holdout_payload: Mapping[str, Any],
    qrels: Mapping[str, set[str]],
    metas_by_id: Mapping[str, Mapping[str, Any]],
    base_runs: Mapping[str, Mapping[str, Any]],
    semantic_runs: Mapping[str, Mapping[str, Any]],
    negative_runs: Mapping[str, Mapping[str, Any]],
    artifacts: Mapping[str, Any],
    runtime: Mapping[str, Any],
    corpus: Mapping[str, Any],
    frozen_holdout_sha256: str,
) -> dict[str, Any]:
    semantic_specs, negative_specs = validate_holdout_payload(
        holdout_payload, base_query_payload
    )
    base_queries = base_query_payload["queries"]
    ranking_depth = max(10, int(runtime.get("evaluation_depth") or 10))

    base_rows: list[dict[str, Any]] = []
    base_by_id: dict[str, dict[str, Any]] = {}
    for spec in base_queries:
        base_id = clean_text(spec.get("id"))
        if base_id not in base_runs:
            raise ValueError(f"missing base run: {base_id}")
        row = _result_row(
            result_id=base_id,
            query=clean_text(spec.get("query")),
            run=base_runs[base_id],
            relevant_ids=qrels.get(base_id, set()),
            metas_by_id=metas_by_id,
            reference_date=reference_date,
            ranking_depth=ranking_depth,
        )
        row.update(
            {
                "criteria": dict(spec.get("criteria") or {}),
                "relevant_count": len(qrels.get(base_id, set())),
            }
        )
        base_rows.append(row)
        base_by_id[base_id] = row

    semantic_rows: list[dict[str, Any]] = []
    base_spec_by_id = {
        clean_text(spec.get("id")): spec for spec in base_queries
    }
    for spec in semantic_specs:
        case_id = clean_text(spec.get("id"))
        base_id = clean_text(spec.get("base_id"))
        if case_id not in semantic_runs:
            raise ValueError(f"missing semantic holdout run: {case_id}")
        row = _result_row(
            result_id=case_id,
            query=clean_text(spec.get("query")),
            run=semantic_runs[case_id],
            relevant_ids=qrels.get(base_id, set()),
            metas_by_id=metas_by_id,
            reference_date=reference_date,
            ranking_depth=ranking_depth,
        )
        row.update(
            {
                "base_id": base_id,
                "criteria": dict(base_spec_by_id[base_id].get("criteria") or {}),
                "relevant_count": len(qrels.get(base_id, set())),
                "included_in_semantic_aggregate": True,
            }
        )
        baseline = base_by_id[base_id]
        row["stability"] = rank_stability(
            baseline["ranking_ids"], row["ranking_ids"]
        )
        row["metric_delta_from_base"] = _metric_delta(
            baseline["metrics"], row["metrics"]
        )
        row["observations"] = _semantic_observations(row, baseline)
        semantic_rows.append(row)

    negative_rows: list[dict[str, Any]] = []
    for spec in negative_specs:
        case_id = clean_text(spec.get("id"))
        if case_id not in negative_runs:
            raise ValueError(f"missing negative diagnostic run: {case_id}")
        run = negative_runs[case_id]
        ranking = deduplicate_ids(run.get("ranking_ids") or [])[:ranking_depth]
        row = {
            "id": case_id,
            "query": clean_text(spec.get("query")),
            "diagnostic": clean_text(spec.get("diagnostic")),
            "included_in_semantic_aggregate": False,
            "relevance_metrics_computed": False,
            "normalized_query": clean_text(run.get("normalized_query")),
            "query_policy": dict(run.get("query_policy") or {}),
            "retrieval": dict(run.get("retrieval") or {}),
            "ranking_ids": ranking,
            "inactive_exposure": _inactive_exposure(
                ranking,
                metas_by_id,
                reference_date,
            ),
        }
        row["diagnostic_failures"] = negative_diagnostic_failures(row)
        row["diagnostic_passed"] = not row["diagnostic_failures"]
        negative_rows.append(row)

    base_metrics = _mean_metrics(base_rows)
    semantic_metrics = _mean_metrics(semantic_rows)
    failure_rows = [row for row in semantic_rows if row["observations"]]
    failure_rows.sort(
        key=lambda row: (
            float(row["stability"]["top10_jaccard"]),
            float(row["stability"]["rbo@10"]),
            float(row["metric_delta_from_base"]["nDCG@5"]),
            row["id"],
        )
    )
    all_inactive = [
        float(row["metrics"]["inactive_exposure"])
        for row in (*base_rows, *semantic_rows)
    ] + [float(row["inactive_exposure"]) for row in negative_rows]
    artifacts_copy = {
        clean_text(name): dict(record)
        for name, record in artifacts.items()
        if isinstance(record, Mapping)
    }
    system_code = {
        name: clean_text((artifacts_copy.get(name) or {}).get("sha256"))
        for name in SYSTEM_CODE_ARTIFACTS
    }

    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "run_started_at": run_started_at,
        "reference_date": reference_date.strftime("%Y-%m-%d"),
        "evaluation_design": {
            "role": "independent_one_shot_holdout",
            "one_shot": True,
            "evaluation_execution_count": 1,
            "frozen_before_system_import": True,
            "frozen_holdout_sha256": frozen_holdout_sha256,
            "author_independence": "author-independent-but-same-repo",
            "base_case_count": len(base_rows),
            "semantic_case_count": len(semantic_rows),
            "variants_per_base_intent": VARIANTS_PER_BASE,
            "negative_diagnostic_count": len(negative_rows),
            "aggregate_policy": (
                "semantic_preserving_cases_only; negative diagnostics excluded"
            ),
            "post_measurement_tuning": False,
            "post_measurement_query_replacement": False,
            "confirmatory_rerun": False,
            "performance_acceptance_threshold": None,
        },
        "label_policy": {
            "source": "base query criteria inherited by base_id",
            "qrels_kind": "silver_public_notice_fields",
            "uses_result_text": False,
            "uses_ranked_results": False,
            "uses_vlm_attributes": False,
        },
        "system": {
            "id": SYSTEM_ID,
            "description": (
                "Current appearance-mode normalization, natural structured parsing, "
                "hybrid retrieval, and graph reranking"
            ),
            "code_sha256": system_code,
            "code_bundle_sha256": clean_text(
                runtime.get("system_code_bundle_sha256")
            ),
            "appearance_query_source_inspected_by_holdout_author": False,
        },
        "artifacts": artifacts_copy,
        "runtime": dict(runtime),
        "corpus": dict(corpus),
        "qrels": qrels_record(base_queries, qrels),
        "base_queries": base_rows,
        "semantic_holdout": {
            "included_in_aggregate": True,
            "summary": {
                "base_metrics": base_metrics,
                "holdout_metrics": semantic_metrics,
                "metric_delta_from_base_average": _metric_delta(
                    base_metrics, semantic_metrics
                ),
                "rank_stability": _mean_stability(semantic_rows),
                "failure_case_count": len(failure_rows),
                "observation_counts": dict(
                    sorted(
                        Counter(
                            observation
                            for row in semantic_rows
                            for observation in row["observations"]
                        ).items()
                    )
                ),
            },
            "cases": semantic_rows,
            "failure_cases": [
                {
                    "id": row["id"],
                    "base_id": row["base_id"],
                    "query": row["query"],
                    "normalized_query": row["normalized_query"],
                    "metrics": row["metrics"],
                    "stability": row["stability"],
                    "metric_delta_from_base": row["metric_delta_from_base"],
                    "observations": row["observations"],
                }
                for row in failure_rows
            ],
        },
        "negative_diagnostics": {
            "included_in_semantic_aggregate": False,
            "relevance_metrics_computed": False,
            "summary": {
                "case_count": len(negative_rows),
                "passed_count": sum(row["diagnostic_passed"] for row in negative_rows),
                "failure_count": sum(
                    not row["diagnostic_passed"] for row in negative_rows
                ),
                "failure_counts": dict(
                    sorted(
                        Counter(
                            failure
                            for row in negative_rows
                            for failure in row["diagnostic_failures"]
                        ).items()
                    )
                ),
            },
            "cases": negative_rows,
        },
        "inactive_exposure": {
            "base_maximum": max(
                (float(row["metrics"]["inactive_exposure"]) for row in base_rows),
                default=0.0,
            ),
            "semantic_maximum": max(
                (
                    float(row["metrics"]["inactive_exposure"])
                    for row in semantic_rows
                ),
                default=0.0,
            ),
            "negative_maximum": max(
                (float(row["inactive_exposure"]) for row in negative_rows),
                default=0.0,
            ),
            "overall_maximum": max(all_inactive, default=0.0),
            "queries_with_inactive_exposure": sum(value > 0.0 for value in all_inactive),
        },
        "limitations": [
            (
                "author-independent-but-same-repo: the holdout wording was frozen "
                "without inspecting the appearance-query implementation, its tests, "
                "prior variant/holdout wording, or prior query-evaluation reports; "
                "shared repository context and inherited base intents still limit "
                "independence"
            ),
            (
                "The 24 semantic cases are a small manually authored Korean sample "
                "with two variants per base intent; no confidence interval or "
                "population-level generalization claim is valid"
            ),
            (
                "Silver qrels derive from public notice color, weight, age, region, "
                "and status fields, not independent human relevance judgments"
            ),
            (
                "Negative requests do not inherit positive qrels and are diagnostic "
                "only; their six outcomes are excluded from every semantic aggregate"
            ),
            (
                "This is one frozen one-shot execution with no post-measurement code "
                "tuning, query replacement, or confirmatory search rerun"
            ),
            (
                "Integrity PASS has no performance floor and does not establish "
                "adoption suitability, human relevance, or external replication"
            ),
        ],
    }
    failures = report_contract_failures(report)
    report["validation"] = {
        "scope": "integrity_invariants_only_not_performance_acceptance",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
    }
    return report


def report_contract_failures(report: Mapping[str, Any]) -> list[str]:
    """Validate provenance and aggregation boundaries, never quality floors."""

    failures: list[str] = []
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        failures.append("report schema is stale")
    design = report.get("evaluation_design")
    if not isinstance(design, Mapping):
        return ["evaluation_design is missing"]
    expected_design = {
        "role": "independent_one_shot_holdout",
        "one_shot": True,
        "evaluation_execution_count": 1,
        "frozen_before_system_import": True,
        "author_independence": "author-independent-but-same-repo",
        "base_case_count": BASE_CASE_COUNT,
        "semantic_case_count": SEMANTIC_CASE_COUNT,
        "variants_per_base_intent": VARIANTS_PER_BASE,
        "negative_diagnostic_count": NEGATIVE_CASE_COUNT,
        "post_measurement_tuning": False,
        "post_measurement_query_replacement": False,
        "confirmatory_rerun": False,
        "performance_acceptance_threshold": None,
    }
    for key, expected in expected_design.items():
        if design.get(key) != expected:
            failures.append(f"evaluation design mismatch: {key}")
    frozen_sha = clean_text(design.get("frozen_holdout_sha256")).lower()
    artifacts = report.get("artifacts")
    holdout_artifact = (
        artifacts.get("holdout") if isinstance(artifacts, Mapping) else None
    )
    if not frozen_sha or not isinstance(holdout_artifact, Mapping):
        failures.append("frozen holdout provenance is missing")
    elif clean_text(holdout_artifact.get("sha256")).lower() != frozen_sha:
        failures.append("frozen holdout SHA does not match its artifact")

    base_rows = report.get("base_queries")
    semantic = report.get("semantic_holdout")
    negative = report.get("negative_diagnostics")
    if not _sequence(base_rows) or len(base_rows) != BASE_CASE_COUNT:
        failures.append("base query result count is not 12")
    semantic_rows = semantic.get("cases") if isinstance(semantic, Mapping) else None
    if not _sequence(semantic_rows) or len(semantic_rows) != SEMANTIC_CASE_COUNT:
        failures.append("semantic holdout result count is not 24")
    else:
        for row in semantic_rows:
            if not isinstance(row, Mapping):
                failures.append("semantic holdout row is malformed")
            elif row.get("included_in_semantic_aggregate") is not True:
                failures.append("semantic case was excluded from its aggregate")
    negative_rows = negative.get("cases") if isinstance(negative, Mapping) else None
    if not _sequence(negative_rows) or len(negative_rows) != NEGATIVE_CASE_COUNT:
        failures.append("negative diagnostic result count is not 6")
    else:
        for row in negative_rows:
            if not isinstance(row, Mapping):
                failures.append("negative diagnostic row is malformed")
                continue
            if row.get("included_in_semantic_aggregate") is not False:
                failures.append("negative diagnostic leaked into semantic aggregate")
            if row.get("relevance_metrics_computed") is not False or "metrics" in row:
                failures.append("negative diagnostic has inherited relevance metrics")
    summary = semantic.get("summary") if isinstance(semantic, Mapping) else None
    holdout_metrics = (
        summary.get("holdout_metrics") if isinstance(summary, Mapping) else None
    )
    if not isinstance(holdout_metrics, Mapping):
        failures.append("semantic aggregate metrics are missing")
    elif int(holdout_metrics.get("query_count") or 0) != SEMANTIC_CASE_COUNT:
        failures.append("semantic aggregate query count is not 24")
    inactive = report.get("inactive_exposure") or {}
    if float(inactive.get("overall_maximum") or 0.0) > 0.0:
        failures.append("appearance retrieval exposed an inactive notice")

    runtime = report.get("runtime")
    expected_invocations = BASE_CASE_COUNT + SEMANTIC_CASE_COUNT + NEGATIVE_CASE_COUNT
    if not isinstance(runtime, Mapping):
        failures.append("runtime record is missing")
    else:
        for key in ("query_invocation_count", "retrieval_execution_count"):
            if int(runtime.get(key) or 0) != expected_invocations:
                failures.append(f"runtime {key} is not {expected_invocations}")
        if runtime.get("execution_mode") != "one_shot_no_cache_no_repeat":
            failures.append("runtime execution mode is stale")

    system = report.get("system")
    if not isinstance(system, Mapping) or system.get("id") != SYSTEM_ID:
        failures.append("evaluated system identifier is stale")
    else:
        code_hashes = system.get("code_sha256")
        if not isinstance(code_hashes, Mapping):
            failures.append("system code hashes are missing")
        else:
            for name in SYSTEM_CODE_ARTIFACTS:
                if len(clean_text(code_hashes.get(name))) != 64:
                    failures.append(f"system code hash is missing: {name}")
        if len(clean_text(system.get("code_bundle_sha256"))) != 64:
            failures.append("system code bundle hash is missing")
    return sorted(set(failures))


def report_recalculation_failures(
    report: Mapping[str, Any],
    *,
    reference_date: datetime,
    base_query_payload: Mapping[str, Any],
    holdout_payload: Mapping[str, Any],
    qrels: Mapping[str, set[str]],
    metas_by_id: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """Recompute stored metrics and diagnostics without rerunning retrieval."""

    failures: list[str] = []
    semantic_specs, negative_specs = validate_holdout_payload(
        holdout_payload, base_query_payload
    )
    if report.get("qrels") != qrels_record(base_query_payload["queries"], qrels):
        failures.append("stored qrels do not match recalculated silver qrels")

    base_rows = {
        clean_text(row.get("id")): row
        for row in report.get("base_queries") or []
        if isinstance(row, Mapping)
    }
    recalculated_base: list[dict[str, Any]] = []
    for spec in base_query_payload["queries"]:
        base_id = clean_text(spec.get("id"))
        row = base_rows.get(base_id)
        if not isinstance(row, Mapping):
            failures.append(f"missing stored base result: {base_id}")
            continue
        metrics = query_metrics(
            row.get("ranking_ids") or [],
            qrels.get(base_id, set()),
            metas_by_id,
            reference_date,
        )
        if row.get("metrics") != metrics:
            failures.append(f"stored base metrics are stale: {base_id}")
        recalculated_base.append({"metrics": metrics})

    semantic_rows = {
        clean_text(row.get("id")): row
        for row in (report.get("semantic_holdout") or {}).get("cases") or []
        if isinstance(row, Mapping)
    }
    recalculated_semantic: list[dict[str, Any]] = []
    for spec in semantic_specs:
        case_id = clean_text(spec.get("id"))
        base_id = clean_text(spec.get("base_id"))
        row = semantic_rows.get(case_id)
        base = base_rows.get(base_id)
        if not isinstance(row, Mapping) or not isinstance(base, Mapping):
            failures.append(f"missing stored semantic result: {case_id}")
            continue
        metrics = query_metrics(
            row.get("ranking_ids") or [],
            qrels.get(base_id, set()),
            metas_by_id,
            reference_date,
        )
        stability = rank_stability(
            base.get("ranking_ids") or [], row.get("ranking_ids") or []
        )
        expected_row = {
            **dict(row),
            "metrics": metrics,
            "stability": stability,
        }
        observations = _semantic_observations(expected_row, base)
        if row.get("metrics") != metrics:
            failures.append(f"stored semantic metrics are stale: {case_id}")
        if row.get("stability") != stability:
            failures.append(f"stored semantic stability is stale: {case_id}")
        if row.get("observations") != observations:
            failures.append(f"stored semantic observations are stale: {case_id}")
        recalculated_semantic.append(
            {"metrics": metrics, "stability": stability}
        )

    summary = (report.get("semantic_holdout") or {}).get("summary") or {}
    if summary.get("base_metrics") != _mean_metrics(recalculated_base):
        failures.append("stored base aggregate is stale")
    if summary.get("holdout_metrics") != _mean_metrics(recalculated_semantic):
        failures.append("stored semantic aggregate is stale")
    if summary.get("rank_stability") != _mean_stability(recalculated_semantic):
        failures.append("stored semantic stability aggregate is stale")

    negative_rows = {
        clean_text(row.get("id")): row
        for row in (report.get("negative_diagnostics") or {}).get("cases") or []
        if isinstance(row, Mapping)
    }
    for spec in negative_specs:
        case_id = clean_text(spec.get("id"))
        row = negative_rows.get(case_id)
        if not isinstance(row, Mapping):
            failures.append(f"missing stored negative diagnostic: {case_id}")
            continue
        inactive = _inactive_exposure(
            row.get("ranking_ids") or [],
            metas_by_id,
            reference_date,
        )
        diagnostic_failures = negative_diagnostic_failures(row)
        if row.get("inactive_exposure") != inactive:
            failures.append(f"stored negative inactive exposure is stale: {case_id}")
        if row.get("diagnostic_failures") != diagnostic_failures:
            failures.append(f"stored negative diagnostic is stale: {case_id}")
        if row.get("diagnostic_passed") is not (not diagnostic_failures):
            failures.append(f"stored negative status is stale: {case_id}")
        if "metrics" in row or row.get("relevance_metrics_computed") is not False:
            failures.append(f"negative diagnostic has relevance metrics: {case_id}")
    return sorted(set(failures))


def _md_text(value: Any) -> str:
    return clean_text(value).replace("|", "\\|").replace("\n", " ")


def _number(value: Any, digits: int = 6) -> str:
    return f"{float(value or 0.0):.{digits}f}"


def report_to_markdown(report: Mapping[str, Any]) -> str:
    """Render the full stored evidence and limitations."""

    design = report.get("evaluation_design") or {}
    semantic = report.get("semantic_holdout") or {}
    summary = semantic.get("summary") or {}
    base = summary.get("base_metrics") or {}
    holdout = summary.get("holdout_metrics") or {}
    delta = summary.get("metric_delta_from_base_average") or {}
    stability = summary.get("rank_stability") or {}
    system = report.get("system") or {}
    lines = [
        "# 외형 검색 독립 blind holdout 평가 v3",
        "",
        (
            "**Frozen-before-system-import one-shot 결과입니다. 측정 뒤 코드·질의 "
            "수정, 문구 교체, 확인 재실행을 하지 않았고 성능 합격선도 없습니다.**"
        ),
        "",
        f"- 기준일: `{_md_text(report.get('reference_date'))}`",
        f"- 시스템: `{_md_text(system.get('id'))}`",
        f"- 독립성: `{_md_text(design.get('author_independence'))}`",
        f"- frozen SHA-256: `{_md_text(design.get('frozen_holdout_sha256'))}`",
        (
            f"- 실행 질의: base `{int(design.get('base_case_count') or 0)}`, "
            f"semantic `{int(design.get('semantic_case_count') or 0)}`, "
            f"negative `{int(design.get('negative_diagnostic_count') or 0)}`"
        ),
        "- negative 6개는 모든 semantic aggregate와 inherited qrels에서 제외",
        "",
        "## 의미보존 aggregate",
        "",
        "| 지표 | base 12 평균 | holdout 24 평균 | 변화 |",
        "|---|---:|---:|---:|",
    ]
    for key in ("precision@5", "recall@5", "recall@10", "nDCG@5", "MRR", "hit@5"):
        lines.append(
            f"| `{key}` | {_number(base.get(key))} | "
            f"{_number(holdout.get(key))} | {_number(delta.get(key))} |"
        )
    lines.extend(
        [
            "",
            f"- 평균 top-5 Jaccard: `{_number(stability.get('mean_top5_jaccard'))}`",
            (
                f"- 평균 top-10 Jaccard: "
                f"`{_number(stability.get('mean_top10_jaccard'))}`"
            ),
            f"- 평균 RBO@10: `{_number(stability.get('mean_rbo@10'))}`",
            (
                f"- top-10 완전일치율: "
                f"`{_number(stability.get('top10_exact_match_rate'))}`"
            ),
            (
                f"- 실패·변화 사례: "
                f"`{int(summary.get('failure_case_count') or 0)}` / "
                f"`{int(design.get('semantic_case_count') or 0)}`"
            ),
            "",
            "## 의미보존 질의별 결과",
            "",
            "| ID | base | 문장 | 정규화 | P@5 | nDCG@5 | J@10 | 관찰 |",
            "|---|---|---|---|---:|---:|---:|---|",
        ]
    )
    for row in semantic.get("cases") or []:
        metrics = row.get("metrics") or {}
        row_stability = row.get("stability") or {}
        observations = ", ".join(row.get("observations") or []) or "없음"
        lines.append(
            f"| `{_md_text(row.get('id'))}` | `{_md_text(row.get('base_id'))}` | "
            f"{_md_text(row.get('query'))} | "
            f"{_md_text(row.get('normalized_query'))} | "
            f"{_number(metrics.get('precision@5'))} | "
            f"{_number(metrics.get('nDCG@5'))} | "
            f"{_number(row_stability.get('top10_jaccard'))} | "
            f"{_md_text(observations)} |"
        )

    lines.extend(["", "## 모든 실패·변화 사례", ""])
    failure_cases = semantic.get("failure_cases") or []
    if not failure_cases:
        lines.append("- 기록된 순위 변화·점수 하락·지원 경고 없음")
    else:
        for row in failure_cases:
            observations = ", ".join(row.get("observations") or [])
            lines.append(
                f"- `{_md_text(row.get('id'))}` — {_md_text(row.get('query'))}: "
                f"{_md_text(observations)}"
            )

    negatives = report.get("negative_diagnostics") or {}
    negative_summary = negatives.get("summary") or {}
    lines.extend(
        [
            "",
            "## 부정 진단 — aggregate 제외",
            "",
            (
                "positive silver qrels를 상속하지 않으며, 사전 정의된 오적용 "
                "진단만 기록합니다."
            ),
            (
                f"- 진단 PASS `{int(negative_summary.get('passed_count') or 0)}` / "
                f"`{int(negative_summary.get('case_count') or 0)}`, 실패 "
                f"`{int(negative_summary.get('failure_count') or 0)}`"
            ),
            "",
            "| ID | 문장 | 정규화 | 진단 | 결과 | 실패 |",
            "|---|---|---|---|---:|---|",
        ]
    )
    for row in negatives.get("cases") or []:
        failures = ", ".join(row.get("diagnostic_failures") or []) or "없음"
        lines.append(
            f"| `{_md_text(row.get('id'))}` | {_md_text(row.get('query'))} | "
            f"{_md_text(row.get('normalized_query'))} | "
            f"`{_md_text(row.get('diagnostic'))}` | "
            f"{'PASS' if row.get('diagnostic_passed') else 'FAIL'} | "
            f"{_md_text(failures)} |"
        )

    inactive = report.get("inactive_exposure") or {}
    validation = report.get("validation") or {}
    lines.extend(
        [
            "",
            "## 무결성·비활성 노출",
            "",
            f"- 무결성: **{'PASS' if validation.get('passed') else 'FAIL'}**",
            f"- 범위: `{_md_text(validation.get('scope'))}`",
            f"- 무결성 실패: `{int(validation.get('failure_count') or 0)}`건",
            f"- base 최대: `{_number(inactive.get('base_maximum'))}`",
            f"- semantic 최대: `{_number(inactive.get('semantic_maximum'))}`",
            f"- negative 최대: `{_number(inactive.get('negative_maximum'))}`",
            f"- 전체 최대: `{_number(inactive.get('overall_maximum'))}`",
            (
                f"- 비활성 노출 질의 수: "
                f"`{int(inactive.get('queries_with_inactive_exposure') or 0)}`"
            ),
            "",
            "## 시스템 코드 SHA-256",
            "",
            f"- bundle: `{_md_text(system.get('code_bundle_sha256'))}`",
            "",
            "| 코드 아티팩트 | SHA-256 |",
            "|---|---|",
        ]
    )
    for name, sha256 in (system.get("code_sha256") or {}).items():
        lines.append(f"| `{_md_text(name)}` | `{_md_text(sha256)}` |")

    lines.extend(["", "## 전체 아티팩트 SHA-256", "", "| 아티팩트 | SHA-256 |"])
    lines.append("|---|---|")
    for name, record in (report.get("artifacts") or {}).items():
        if isinstance(record, Mapping) and clean_text(record.get("sha256")):
            lines.append(
                f"| `{_md_text(name)}` | `{_md_text(record.get('sha256'))}` |"
            )

    lines.extend(["", "## 한계", ""])
    for limitation in report.get("limitations") or []:
        lines.append(f"- {_md_text(limitation)}")
    lines.extend(
        [
            "",
            (
                "이 평가는 사람의 관련성 판정, 입양 적합성 평가, 외부 저자에 의한 "
                "완전 독립 검증이 아닙니다."
            ),
            "",
        ]
    )
    return "\n".join(lines)
