"""Pure helpers for the frozen appearance-query holdout evaluation.

This module deliberately contains no query-normalization or retrieval logic.
The one-shot runner passes stored rankings and query-policy observations into
these helpers so ``--check`` can recalculate evidence without running the
search system again.
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


REPORT_SCHEMA_VERSION = "appearance-query-holdout-report.v2"
HOLDOUT_SCHEMA_VERSION = "appearance-query-holdout.v2"
SYSTEM_ID = "appearance_natural_graph"
SEMANTIC_CASE_COUNT = 12
STABILITY_KEYS = ("top5_jaccard", "top10_jaccard", "rbo@10")
NEGATION_EXPECTATIONS = {
    "must_not_apply_all_negated_constraints_as_positive",
    "must_not_apply_negated_region_as_positive",
}


def _sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def validate_holdout_payload(
    payload: Mapping[str, Any],
    base_query_payload: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate the pre-run frozen holdout without consulting system output."""

    if payload.get("schema_version") != HOLDOUT_SCHEMA_VERSION:
        raise ValueError(f"holdout schema must be {HOLDOUT_SCHEMA_VERSION}")
    if payload.get("source_query_set") != "data/eval_queries.appearance_v1.json":
        raise ValueError("holdout source_query_set is unexpected")
    if payload.get("label_policy") != "inherit_source_qrels_by_base_query_id":
        raise ValueError("holdout label policy must inherit source qrels")

    protocol = payload.get("freeze_protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("holdout freeze_protocol is required")
    if protocol.get("authored_before_first_system_run") is not True:
        raise ValueError("holdout must declare pre-run authorship")
    if protocol.get("independence_scope") != "author-independent-but-same-repo":
        raise ValueError("holdout independence scope is missing")

    base_queries = base_query_payload.get("queries")
    semantic_cases = payload.get("semantic_preserving_cases")
    negative_cases = payload.get("negation_diagnostics")
    if not _sequence(base_queries) or not all(
        isinstance(row, Mapping) for row in base_queries
    ):
        raise ValueError("base query payload must contain an object array")
    if not _sequence(semantic_cases) or not all(
        isinstance(row, Mapping) for row in semantic_cases
    ):
        raise ValueError("semantic_preserving_cases must be an object array")
    if not _sequence(negative_cases) or not all(
        isinstance(row, Mapping) for row in negative_cases
    ):
        raise ValueError("negation_diagnostics must be an object array")
    if len(semantic_cases) != SEMANTIC_CASE_COUNT:
        raise ValueError(
            f"holdout must contain exactly {SEMANTIC_CASE_COUNT} semantic cases"
        )
    if not negative_cases:
        raise ValueError("holdout must contain separate negation diagnostics")

    base_ids = [clean_text(row.get("id")) for row in base_queries]
    if any(not value for value in base_ids) or len(set(base_ids)) != len(base_ids):
        raise ValueError("base query IDs must be present and unique")
    semantic_base_ids = [
        clean_text(row.get("base_query_id")) for row in semantic_cases
    ]
    if semantic_base_ids != base_ids:
        raise ValueError(
            "semantic holdout cases must cover base IDs exactly once and in order"
        )

    base_text_by_id = {
        clean_text(row.get("id")): clean_text(row.get("query"))
        for row in base_queries
    }
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()
    normalized_semantic: list[dict[str, Any]] = []
    for position, raw in enumerate(semantic_cases):
        row = dict(raw)
        case_id = clean_text(row.get("id"))
        base_id = clean_text(row.get("base_query_id"))
        query = clean_text(row.get("query"))
        perturbations = row.get("perturbations")
        if not case_id or case_id in seen_ids:
            raise ValueError(f"semantic case {position} has a duplicate or empty ID")
        if (
            not query
            or query == base_text_by_id.get(base_id)
            or query in seen_texts
        ):
            raise ValueError(f"{case_id}: distinct holdout query text is required")
        if not _sequence(perturbations) or not perturbations:
            raise ValueError(f"{case_id}: perturbations must be a non-empty array")
        if any(not clean_text(value) for value in perturbations):
            raise ValueError(f"{case_id}: perturbation names must be non-empty")
        seen_ids.add(case_id)
        seen_texts.add(query)
        normalized_semantic.append(row)

    normalized_negative: list[dict[str, Any]] = []
    for position, raw in enumerate(negative_cases):
        row = dict(raw)
        case_id = clean_text(row.get("id"))
        base_id = clean_text(row.get("negated_base_query_id"))
        query = clean_text(row.get("query"))
        expectation = clean_text(row.get("diagnostic_expectation"))
        if not case_id or case_id in seen_ids:
            raise ValueError(f"negative case {position} has a duplicate or empty ID")
        if base_id not in base_text_by_id:
            raise ValueError(f"{case_id}: unknown negated base query")
        if not query or query in seen_texts:
            raise ValueError(f"{case_id}: distinct negative query text is required")
        if expectation not in NEGATION_EXPECTATIONS:
            raise ValueError(f"{case_id}: unsupported diagnostic expectation")
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


def _negative_observations(
    row: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> list[str]:
    observations: list[str] = []
    policy = row.get("query_policy") or {}
    stability = row.get("rank_stability_to_negated_base") or {}
    if not bool(policy.get("unsupported_conditions")):
        observations.append("negation_not_signaled_by_query_policy")
    if row.get("normalized_query") == baseline.get("normalized_query"):
        observations.append("normalized_to_negated_base")
    if bool(stability.get("top10_exact_match")):
        observations.append("top10_matches_negated_base")
    if not clean_text(row.get("normalized_query")):
        observations.append("normalized_query_empty")
    return observations


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
    for spec in semantic_specs:
        case_id = clean_text(spec.get("id"))
        base_id = clean_text(spec.get("base_query_id"))
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
                "base_query_id": base_id,
                "perturbations": list(spec.get("perturbations") or []),
                "criteria": dict(
                    next(
                        base.get("criteria") or {}
                        for base in base_queries
                        if clean_text(base.get("id")) == base_id
                    )
                ),
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
        base_id = clean_text(spec.get("negated_base_query_id"))
        if case_id not in negative_runs:
            raise ValueError(f"missing negative diagnostic run: {case_id}")
        run = negative_runs[case_id]
        ranking = deduplicate_ids(run.get("ranking_ids") or [])[:ranking_depth]
        baseline = base_by_id[base_id]
        row = {
            "id": case_id,
            "query": clean_text(spec.get("query")),
            "negated_base_query_id": base_id,
            "diagnostic_expectation": clean_text(
                spec.get("diagnostic_expectation")
            ),
            "included_in_semantic_aggregate": False,
            "relevance_metrics_computed": False,
            "normalized_query": clean_text(run.get("normalized_query")),
            "query_policy": dict(run.get("query_policy") or {}),
            "retrieval": dict(run.get("retrieval") or {}),
            "ranking_ids": ranking,
            "rank_stability_to_negated_base": rank_stability(
                baseline["ranking_ids"], ranking
            ),
        }
        row["fail_safe_signal_present"] = bool(
            (row.get("query_policy") or {}).get("unsupported_conditions")
        )
        row["observations"] = _negative_observations(row, baseline)
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

    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "run_started_at": run_started_at,
        "reference_date": reference_date.strftime("%Y-%m-%d"),
        "evaluation_design": {
            "role": "independent_one_shot_holdout",
            "one_shot": True,
            "evaluation_execution_count": 1,
            "frozen_before_run": True,
            "frozen_holdout_sha256": frozen_holdout_sha256,
            "author_independence": "author-independent-but-same-repo",
            "semantic_case_count": len(semantic_rows),
            "negation_diagnostic_count": len(negative_rows),
            "aggregate_policy": (
                "semantic_preserving_cases_only; negation diagnostics excluded"
            ),
            "post_measurement_tuning": False,
            "performance_acceptance_threshold": None,
        },
        "label_policy": {
            "source": "base query criteria inherited by base_query_id",
            "qrels_kind": "silver_public_notice_fields",
            "uses_result_text": False,
            "uses_ranked_results": False,
            "uses_vlm_attributes": False,
        },
        "system": {
            "id": SYSTEM_ID,
            "description": (
                "Current appearance-mode normalization plus natural structured "
                "parsing, hybrid retrieval, and graph reranking"
            ),
        },
        "artifacts": dict(artifacts),
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
                    "base_query_id": row["base_query_id"],
                    "query": row["query"],
                    "normalized_query": row["normalized_query"],
                    "stability": row["stability"],
                    "metric_delta_from_base": row["metric_delta_from_base"],
                    "observations": row["observations"],
                }
                for row in failure_rows
            ],
        },
        "negation_diagnostics": {
            "included_in_semantic_aggregate": False,
            "relevance_metrics_computed": False,
            "summary": {
                "case_count": len(negative_rows),
                "fail_safe_signal_count": sum(
                    bool(row["fail_safe_signal_present"]) for row in negative_rows
                ),
                "observation_counts": dict(
                    sorted(
                        Counter(
                            observation
                            for row in negative_rows
                            for observation in row["observations"]
                        ).items()
                    )
                ),
            },
            "cases": negative_rows,
        },
        "inactive_exposure": {
            "maximum": max(
                [
                    float(row["metrics"]["inactive_exposure"])
                    for row in (*base_rows, *semantic_rows)
                ],
                default=0.0,
            )
        },
        "limitations": [
            (
                "author-independent-but-same-repo: the holdout author did not inspect "
                "the excluded v1 variants or appearance-query implementation, but "
                "shared repository context and base qrels can still align assumptions"
            ),
            (
                "The 12 semantic cases are a small manually authored sample with one "
                "case per base intent; no confidence interval or population claim is valid"
            ),
            (
                "Silver qrels come from public notice color, weight, age, region, and "
                "status fields rather than independent human relevance judgments"
            ),
            (
                "Negated requests change semantics, so they are diagnostics only and "
                "are not scored against inherited positive qrels"
            ),
            (
                "This is one frozen one-shot run; no code or query tuning and no "
                "confirmatory rerun followed measurement"
            ),
            (
                "The blinded appearance-query module is invoked at runtime but its "
                "content hash is intentionally omitted from author-visible provenance"
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
        "frozen_before_run": True,
        "author_independence": "author-independent-but-same-repo",
        "semantic_case_count": SEMANTIC_CASE_COUNT,
        "post_measurement_tuning": False,
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
    negative = report.get("negation_diagnostics")
    if not _sequence(base_rows) or len(base_rows) != SEMANTIC_CASE_COUNT:
        failures.append("base query result count is not 12")
    semantic_rows = semantic.get("cases") if isinstance(semantic, Mapping) else None
    if not _sequence(semantic_rows) or len(semantic_rows) != SEMANTIC_CASE_COUNT:
        failures.append("semantic holdout result count is not 12")
    else:
        for row in semantic_rows:
            if not isinstance(row, Mapping):
                failures.append("semantic holdout row is malformed")
                continue
            if row.get("included_in_semantic_aggregate") is not True:
                failures.append("semantic case was excluded from its aggregate")
    negative_rows = negative.get("cases") if isinstance(negative, Mapping) else None
    if not _sequence(negative_rows) or not negative_rows:
        failures.append("negation diagnostics are missing")
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
        failures.append("semantic aggregate query count is not 12")
    if float((report.get("inactive_exposure") or {}).get("maximum") or 0.0) > 0.0:
        failures.append("appearance retrieval exposed an inactive notice")
    system = report.get("system")
    if not isinstance(system, Mapping) or system.get("id") != SYSTEM_ID:
        failures.append("evaluated system identifier is stale")
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
    """Recompute stored metrics and stability without rerunning retrieval."""

    failures: list[str] = []
    semantic_specs, negative_specs = validate_holdout_payload(
        holdout_payload, base_query_payload
    )
    expected_qrels = qrels_record(base_query_payload["queries"], qrels)
    if report.get("qrels") != expected_qrels:
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
        base_id = clean_text(spec.get("base_query_id"))
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
        if row.get("metrics") != metrics:
            failures.append(f"stored semantic metrics are stale: {case_id}")
        if row.get("stability") != stability:
            failures.append(f"stored semantic stability is stale: {case_id}")
        recalculated_semantic.append({"metrics": metrics, "stability": stability})

    summary = (report.get("semantic_holdout") or {}).get("summary") or {}
    if summary.get("base_metrics") != _mean_metrics(recalculated_base):
        failures.append("stored base aggregate is stale")
    if summary.get("holdout_metrics") != _mean_metrics(recalculated_semantic):
        failures.append("stored semantic aggregate is stale")
    if summary.get("rank_stability") != _mean_stability(recalculated_semantic):
        failures.append("stored semantic stability aggregate is stale")

    negative_rows = {
        clean_text(row.get("id")): row
        for row in (report.get("negation_diagnostics") or {}).get("cases") or []
        if isinstance(row, Mapping)
    }
    for spec in negative_specs:
        case_id = clean_text(spec.get("id"))
        base_id = clean_text(spec.get("negated_base_query_id"))
        row = negative_rows.get(case_id)
        base = base_rows.get(base_id)
        if not isinstance(row, Mapping) or not isinstance(base, Mapping):
            failures.append(f"missing stored negative diagnostic: {case_id}")
            continue
        stability = rank_stability(
            base.get("ranking_ids") or [], row.get("ranking_ids") or []
        )
        if row.get("rank_stability_to_negated_base") != stability:
            failures.append(f"stored negative stability is stale: {case_id}")
        if "metrics" in row or row.get("relevance_metrics_computed") is not False:
            failures.append(f"negative diagnostic has relevance metrics: {case_id}")
    return sorted(set(failures))


def _md_text(value: Any) -> str:
    return clean_text(value).replace("|", "\\|").replace("\n", " ")


def _number(value: Any, digits: int = 3) -> str:
    return f"{float(value or 0.0):.{digits}f}"


def report_to_markdown(report: Mapping[str, Any]) -> str:
    """Render the JSON evidence without adding claims absent from the report."""

    design = report.get("evaluation_design") or {}
    semantic = report.get("semantic_holdout") or {}
    summary = semantic.get("summary") or {}
    base = summary.get("base_metrics") or {}
    holdout = summary.get("holdout_metrics") or {}
    delta = summary.get("metric_delta_from_base_average") or {}
    stability = summary.get("rank_stability") or {}
    lines = [
        "# 외형 검색 독립 holdout 평가 v2",
        "",
        (
            "**이 결과는 frozen-before-run one-shot 측정입니다. 측정 뒤 코드·질의 "
            "튜닝이나 확인용 재실행을 하지 않았으며, 성능 합격선을 두지 않았습니다.**"
        ),
        "",
        f"- 기준일: `{_md_text(report.get('reference_date'))}`",
        f"- 시스템: `{_md_text((report.get('system') or {}).get('id'))}`",
        f"- 독립성 범위: `{_md_text(design.get('author_independence'))}`",
        f"- 고정 holdout SHA-256: `{_md_text(design.get('frozen_holdout_sha256'))}`",
        f"- 의미보존 질의: `{int(design.get('semantic_case_count') or 0)}`개",
        (
            f"- 부정 진단: `{int(design.get('negation_diagnostic_count') or 0)}`개 "
            "(의미보존 aggregate에서 제외)"
        ),
        "- 검증 PASS는 파일·집계 불변조건만 뜻하며 성능 합격을 뜻하지 않음",
        "",
        "## 의미보존 aggregate",
        "",
        "| 지표 | base 평균 | holdout 평균 | 변화 |",
        "|---|---:|---:|---:|",
    ]
    for key in ("precision@5", "recall@5", "recall@10", "nDCG@5", "MRR", "hit@5"):
        lines.append(
            f"| `{key}` | {_number(base.get(key), 6)} | "
            f"{_number(holdout.get(key), 6)} | {_number(delta.get(key), 6)} |"
        )
    lines.extend(
        [
            "",
            (
                f"- 평균 top-5 Jaccard: "
                f"`{_number(stability.get('mean_top5_jaccard'), 6)}`"
            ),
            (
                f"- 평균 top-10 Jaccard: "
                f"`{_number(stability.get('mean_top10_jaccard'), 6)}`"
            ),
            f"- 평균 RBO@10: `{_number(stability.get('mean_rbo@10'), 6)}`",
            (
                f"- top-10 완전일치율: "
                f"`{_number(stability.get('top10_exact_match_rate'), 6)}`"
            ),
            "",
            "## 의미보존 질의별 결과",
            "",
            "| ID | holdout 문장 | 정규화 문장 | P@5 | nDCG@5 | top10 Jaccard | 관찰 |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    for row in semantic.get("cases") or []:
        metrics = row.get("metrics") or {}
        row_stability = row.get("stability") or {}
        observations = ", ".join(row.get("observations") or []) or "없음"
        lines.append(
            f"| `{_md_text(row.get('id'))}` | {_md_text(row.get('query'))} | "
            f"{_md_text(row.get('normalized_query'))} | "
            f"{_number(metrics.get('precision@5'), 6)} | "
            f"{_number(metrics.get('nDCG@5'), 6)} | "
            f"{_number(row_stability.get('top10_jaccard'), 6)} | "
            f"{_md_text(observations)} |"
        )

    negatives = report.get("negation_diagnostics") or {}
    lines.extend(
        [
            "",
            "## 부정 질의 진단 — aggregate 제외",
            "",
            (
                "부정 질의는 원 qrels와 의미가 다르므로 precision/recall을 계산하지 "
                "않았습니다. 아래는 fail-safe 신호와 원 긍정 질의 대비 순위만 진단합니다."
            ),
            "",
            "| ID | 문장 | 정규화 문장 | fail-safe 신호 | top10 Jaccard | 관찰 |",
            "|---|---|---|---:|---:|---|",
        ]
    )
    for row in negatives.get("cases") or []:
        row_stability = row.get("rank_stability_to_negated_base") or {}
        observations = ", ".join(row.get("observations") or []) or "없음"
        lines.append(
            f"| `{_md_text(row.get('id'))}` | {_md_text(row.get('query'))} | "
            f"{_md_text(row.get('normalized_query'))} | "
            f"{'있음' if row.get('fail_safe_signal_present') else '없음'} | "
            f"{_number(row_stability.get('top10_jaccard'), 6)} | "
            f"{_md_text(observations)} |"
        )

    lines.extend(["", "## 실패·변화 사례", ""])
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

    validation = report.get("validation") or {}
    lines.extend(
        [
            "",
            "## 무결성 검증",
            "",
            f"- 결과: **{'PASS' if validation.get('passed') else 'FAIL'}**",
            f"- 범위: `{_md_text(validation.get('scope'))}`",
            f"- 실패: `{int(validation.get('failure_count') or 0)}`건",
            (
                f"- 비활성 공고 최대 노출률: "
                f"`{_number((report.get('inactive_exposure') or {}).get('maximum'), 6)}`"
            ),
            "",
            "## 재현 근거",
            "",
            "| 아티팩트 | SHA-256 |",
            "|---|---|",
        ]
    )
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
