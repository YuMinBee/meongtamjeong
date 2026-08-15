from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime
from pathlib import Path

from app.query_holdout_evaluation import (
    build_report,
    rank_stability,
    report_contract_failures,
    report_recalculation_failures,
    validate_holdout_payload,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_QUERIES_PATH = ROOT / "data" / "eval_queries.appearance_v1.json"
HOLDOUT_PATH = ROOT / "data" / "eval_query_holdout.appearance_v2.json"
HOLDOUT_SHA_PATH = ROOT / "data" / "eval_query_holdout.appearance_v2.json.sha256"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _run(query: str, dog_id: str) -> dict:
    return {
        "normalized_query": query,
        "query_policy": {
            "typo_corrections": [],
            "synonym_normalizations": [],
            "unsupported_conditions": [],
            "warnings": [],
            "is_fully_supported": True,
        },
        "retrieval": {
            "search_query": query,
            "structured_query": {},
            "status_filter": "active_only",
        },
        "ranking_ids": [dog_id],
    }


def _synthetic_report() -> tuple[dict, dict, dict, dict[str, set[str]], dict]:
    base_payload = _load(BASE_QUERIES_PATH)
    holdout_payload = _load(HOLDOUT_PATH)
    semantic, negative = validate_holdout_payload(holdout_payload, base_payload)
    dog_ids = {
        row["id"]: f"dog-{index}"
        for index, row in enumerate(base_payload["queries"], start=1)
    }
    qrels = {base_id: {dog_id} for base_id, dog_id in dog_ids.items()}
    metas = {
        dog_id: {
            "desertionNo": dog_id,
            "process_state": "보호중",
            "notice_start": "20260701",
            "notice_end": "20260801",
        }
        for dog_id in dog_ids.values()
    }
    base_runs = {
        row["id"]: _run(row["query"], dog_ids[row["id"]])
        for row in base_payload["queries"]
    }
    semantic_runs = {
        row["id"]: _run(row["query"], dog_ids[row["base_query_id"]])
        for row in semantic
    }
    negative_runs = {
        row["id"]: _run(row["query"], dog_ids[row["negated_base_query_id"]])
        for row in negative
    }
    frozen_sha = hashlib.sha256(HOLDOUT_PATH.read_bytes()).hexdigest()
    report = build_report(
        generated_at="2026-07-26T01:00:01+00:00",
        run_started_at="2026-07-26T01:00:00+00:00",
        reference_date=datetime(2026, 7, 26),
        base_query_payload=base_payload,
        holdout_payload=holdout_payload,
        qrels=qrels,
        metas_by_id=metas,
        base_runs=base_runs,
        semantic_runs=semantic_runs,
        negative_runs=negative_runs,
        artifacts={"holdout": {"sha256": frozen_sha}},
        runtime={"evaluation_depth": 10},
        corpus={"notices": len(metas)},
        frozen_holdout_sha256=frozen_sha,
    )
    return report, base_payload, holdout_payload, qrels, metas


def test_frozen_holdout_digest_and_coverage() -> None:
    base_payload = _load(BASE_QUERIES_PATH)
    holdout_payload = _load(HOLDOUT_PATH)
    semantic, negative = validate_holdout_payload(holdout_payload, base_payload)

    expected = HOLDOUT_SHA_PATH.read_text(encoding="utf-8").split()[0]
    assert hashlib.sha256(HOLDOUT_PATH.read_bytes()).hexdigest() == expected
    assert len(semantic) == 12
    assert [row["base_query_id"] for row in semantic] == [
        row["id"] for row in base_payload["queries"]
    ]
    assert len(negative) == 3


def test_rank_stability_is_order_sensitive_but_membership_aware() -> None:
    stability = rank_stability(["a", "b", "c"], ["b", "a", "c"])

    assert stability["top5_jaccard"] == 1.0
    assert stability["top10_jaccard"] == 1.0
    assert stability["rbo@10"] < 1.0
    assert stability["top10_exact_match"] is False


def test_report_excludes_negation_from_semantic_aggregate() -> None:
    report, base_payload, holdout_payload, qrels, metas = _synthetic_report()

    assert report["validation"]["passed"] is True
    assert report["semantic_holdout"]["summary"]["holdout_metrics"][
        "query_count"
    ] == 12
    assert report["negation_diagnostics"]["included_in_semantic_aggregate"] is False
    assert all(
        "metrics" not in row
        for row in report["negation_diagnostics"]["cases"]
    )
    assert (
        report_recalculation_failures(
            report,
            reference_date=datetime(2026, 7, 26),
            base_query_payload=base_payload,
            holdout_payload=holdout_payload,
            qrels=qrels,
            metas_by_id=metas,
        )
        == []
    )


def test_contract_detects_negation_aggregate_leak() -> None:
    report, *_ = _synthetic_report()
    tampered = copy.deepcopy(report)
    tampered["negation_diagnostics"]["cases"][0][
        "included_in_semantic_aggregate"
    ] = True

    assert any(
        "negative diagnostic leaked" in failure
        for failure in report_contract_failures(tampered)
    )
