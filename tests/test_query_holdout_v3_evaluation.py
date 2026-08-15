from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pytest

from app.query_holdout_v3_evaluation import (
    DIAGNOSTIC_NAMES,
    SYSTEM_CODE_ARTIFACTS,
    build_report,
    negative_diagnostic_failures,
    rank_biased_overlap,
    rank_stability,
    report_contract_failures,
    report_recalculation_failures,
    report_to_markdown,
    top_k_jaccard,
    validate_holdout_payload,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_QUERIES = ROOT / "data" / "eval_queries.appearance_v1.json"
HOLDOUT = ROOT / "data" / "eval_query_holdout.appearance_v3.json"
HOLDOUT_SHA256 = ROOT / "data" / "eval_query_holdout.appearance_v3.sha256"
REFERENCE_DATE = datetime(2026, 7, 26)
FROZEN_SHA = "a" * 64


def _base_payload() -> dict:
    return {
        "schema_version": "appearance-query-set.v1",
        "reference_date": "2026-07-26",
        "queries": [
            {
                "id": f"q-{index:02d}",
                "query": f"base query {index}",
                "criteria": {"active_only": True},
            }
            for index in range(12)
        ],
    }


def _holdout_payload() -> dict:
    semantic = [
        {
            "id": f"v-{base_index:02d}-{variant_index}",
            "base_id": f"q-{base_index:02d}",
            "query": f"semantic query {base_index} {variant_index}",
        }
        for base_index in range(12)
        for variant_index in range(2)
    ]
    negative = [
        {
            "id": f"n-{index}",
            "query": f"negative query {index}",
            "diagnostic": diagnostic,
            "aggregate": False,
        }
        for index, diagnostic in enumerate(sorted(DIAGNOSTIC_NAMES))
    ]
    return {
        "schema_version": "appearance-query-holdout.v3",
        "reference_date": "2026-07-26",
        "authoring_protocol": {
            "independent_blind": True,
            "authoring_inputs": [
                "data/eval_queries.appearance_v1.json",
                "data/dog_metas.json",
            ],
            "variants_per_base_intent": 2,
            "semantic_variant_count": 24,
            "negative_diagnostic_count": 6,
            "negative_diagnostics_in_aggregate": False,
        },
        "base_query_set": "data/eval_queries.appearance_v1.json",
        "semantic_variants": semantic,
        "negative_diagnostics": negative,
    }


def _metas() -> dict[str, dict]:
    return {
        "a": {
            "desertionNo": "a",
            "process_state": "보호중",
            "notice_end": "2026-08-01",
        },
        "b": {
            "desertionNo": "b",
            "process_state": "보호중",
            "notice_end": "2026-08-01",
        },
    }


def _run(ranking: list[str], structured: dict | None = None) -> dict:
    return {
        "normalized_query": "normalized",
        "query_policy": {
            "typo_corrections": [],
            "synonym_normalizations": [],
            "unsupported_conditions": [],
            "warnings": [],
            "is_fully_supported": True,
        },
        "ranking_ids": ranking,
        "retrieval": {
            "search_query": "normalized",
            "structured_query": structured or {},
            "status_filter": "active_only",
        },
    }


def _report() -> tuple[dict, dict, dict, dict[str, set[str]], dict]:
    base = _base_payload()
    holdout = _holdout_payload()
    qrels = {row["id"]: {"a"} for row in base["queries"]}
    base_runs = {row["id"]: _run(["a", "b"]) for row in base["queries"]}
    semantic_runs = {
        row["id"]: _run(["b", "a"]) for row in holdout["semantic_variants"]
    }
    negative_runs = {
        row["id"]: _run(
            ["a", "b"],
            {"coat_color": ["brown"]}
            if row["diagnostic"] == "negated_color_must_not_become_positive"
            else {},
        )
        for row in holdout["negative_diagnostics"]
    }
    artifacts = {
        "holdout": {"sha256": FROZEN_SHA},
        **{name: {"sha256": "b" * 64} for name in SYSTEM_CODE_ARTIFACTS},
    }
    report = build_report(
        generated_at="2026-07-26T00:01:00+00:00",
        run_started_at="2026-07-26T00:00:00+00:00",
        reference_date=REFERENCE_DATE,
        base_query_payload=base,
        holdout_payload=holdout,
        qrels=qrels,
        metas_by_id=_metas(),
        base_runs=base_runs,
        semantic_runs=semantic_runs,
        negative_runs=negative_runs,
        artifacts=artifacts,
        runtime={
            "evaluation_depth": 10,
            "query_invocation_count": 42,
            "retrieval_execution_count": 42,
            "execution_mode": "one_shot_no_cache_no_repeat",
            "system_code_bundle_sha256": "c" * 64,
        },
        corpus={"documents": 2},
        frozen_holdout_sha256=FROZEN_SHA,
    )
    return report, base, holdout, qrels, _metas()


def test_frozen_real_holdout_has_expected_shape_and_digest() -> None:
    base = json.loads(BASE_QUERIES.read_text(encoding="utf-8"))
    holdout_bytes = HOLDOUT.read_bytes()
    holdout = json.loads(holdout_bytes)
    sidecar_tokens = HOLDOUT_SHA256.read_text(encoding="utf-8").split()

    semantic, negative = validate_holdout_payload(holdout, base)

    assert len(semantic) == 24
    assert len(negative) == 6
    assert all(row["aggregate"] is False for row in negative)
    assert hashlib.sha256(holdout_bytes).hexdigest() == sidecar_tokens[0]
    assert Path(sidecar_tokens[1]).name == HOLDOUT.name


def test_holdout_validation_rejects_missing_variant_and_negative_leak() -> None:
    holdout = _holdout_payload()
    holdout["semantic_variants"].pop()
    with pytest.raises(ValueError, match="exactly 24"):
        validate_holdout_payload(holdout, _base_payload())

    holdout = _holdout_payload()
    holdout["negative_diagnostics"][0]["aggregate"] = True
    with pytest.raises(ValueError, match="exclude aggregate"):
        validate_holdout_payload(holdout, _base_payload())


def test_rank_stability_is_bounded_and_order_sensitive() -> None:
    assert top_k_jaccard(["a", "b"], ["a", "b"], 5) == 1.0
    assert rank_biased_overlap(["a", "b"], ["a", "b"]) == 1.0
    reordered = rank_stability(["a", "b", "c"], ["b", "a", "c"])
    assert reordered["top5_jaccard"] == 1.0
    assert reordered["top10_jaccard"] == 1.0
    assert 0.0 < reordered["rbo@10"] < 1.0
    assert reordered["top10_exact_match"] is False


def test_negative_diagnostics_detect_only_preregistered_misapplication() -> None:
    color_row = {
        "diagnostic": "negated_color_must_not_become_positive",
        "retrieval": {"structured_query": {"coat_color": ["brown"]}},
    }
    assert negative_diagnostic_failures(color_row) == [
        "negated_color_applied_as_positive"
    ]

    contrast_row = {
        "diagnostic": "negated_region_must_not_survive_contrast",
        "retrieval": {"structured_query": {"regions": ["경기"]}},
    }
    assert negative_diagnostic_failures(contrast_row) == []

    inverted_weight = {
        "diagnostic": "lower_bound_must_not_become_upper_bound",
        "retrieval": {"structured_query": {"weight_kg": {"max": 8}}},
    }
    assert negative_diagnostic_failures(inverted_weight) == [
        "lower_bound_applied_as_upper_bound"
    ]


def test_report_recalculates_evidence_and_excludes_negative_qrels() -> None:
    report, base, holdout, qrels, metas = _report()

    assert report["validation"]["passed"] is True
    assert report["semantic_holdout"]["summary"]["holdout_metrics"][
        "query_count"
    ] == 24
    assert report["semantic_holdout"]["summary"]["failure_case_count"] == 24
    assert report["negative_diagnostics"]["summary"]["failure_count"] == 1
    assert all(
        "metrics" not in row
        and row["relevance_metrics_computed"] is False
        and row["included_in_semantic_aggregate"] is False
        for row in report["negative_diagnostics"]["cases"]
    )
    assert report["inactive_exposure"]["overall_maximum"] == 0.0
    assert report_contract_failures(report) == []
    assert (
        report_recalculation_failures(
            report,
            reference_date=REFERENCE_DATE,
            base_query_payload=base,
            holdout_payload=holdout,
            qrels=qrels,
            metas_by_id=metas,
        )
        == []
    )


def test_recalculation_detects_tampered_semantic_metric() -> None:
    report, base, holdout, qrels, metas = _report()
    tampered = copy.deepcopy(report)
    tampered["semantic_holdout"]["cases"][0]["metrics"]["precision@5"] = 0.999

    failures = report_recalculation_failures(
        tampered,
        reference_date=REFERENCE_DATE,
        base_query_payload=base,
        holdout_payload=holdout,
        qrels=qrels,
        metas_by_id=metas,
    )

    assert any("semantic metrics" in failure for failure in failures)


def test_markdown_discloses_one_shot_independence_silver_and_all_hashes() -> None:
    markdown = report_to_markdown(_report()[0])

    assert "Frozen-before-system-import one-shot" in markdown
    assert "author-independent-but-same-repo" in markdown
    assert "Silver qrels" in markdown
    assert "negative 6개" in markdown
    assert "시스템 코드 SHA-256" in markdown
    assert "모든 실패·변화 사례" in markdown
