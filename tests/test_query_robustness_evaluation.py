from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path

import pytest

from app.appearance_query import normalize_appearance_query
from app.query_robustness_evaluation import (
    CATEGORY_POLICY,
    build_report,
    rank_biased_overlap,
    rank_stability,
    report_contract_failures,
    report_recalculation_failures,
    report_to_markdown,
    top_k_jaccard,
    validate_variant_payload,
)
from app.retrieval_evaluation import build_silver_qrels, merge_notice_metas
from scripts.evaluate_query_robustness import (
    DEFAULT_JSON_OUT,
    DEFAULT_MARKDOWN_OUT,
    MODULE_PATH,
    SCRIPT_PATH,
    check_tracked_report,
    parse_args,
    report_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_QUERIES = ROOT / "data" / "eval_queries.appearance_v1.json"
VARIANTS = ROOT / "data" / "eval_query_variants.appearance_v1.json"
REFERENCE_DATE = datetime(2026, 7, 26)


def _base_payload() -> dict:
    return {
        "schema_version": "appearance-query-set.v1",
        "reference_date": "2026-07-26",
        "queries": [
            {
                "id": "q",
                "query": "갈색 소형견",
                "structured": {
                    "coat_color": ["brown"],
                    "body_size_hint": ["small"],
                },
                "criteria": {
                    "colors_any": ["brown"],
                    "sizes_any": ["small"],
                    "active_only": True,
                },
            }
        ],
    }


def _variant_payload() -> dict:
    return {
        "schema_version": "appearance-query-robustness-variants.v1",
        "base_query_set": "appearance-query-set.v1",
        "design_status": "development_regression_after_first_measurement",
        "evaluation_role": "development_regression_set",
        "independent_holdout_required": True,
        "tuning_disclosure": "Observed v1 failures informed the normalizer.",
        "tuning_policy": "Use an unseen holdout for generalization claims.",
        "queries": [
            {
                "base_id": "q",
                "variants": {
                    "synonym_paraphrase": "갈색의 작은 강아지",
                    "word_order": "소형견 중 갈색인 아이",
                    "conservative_typo": "갈색 소형겐",
                    "lifestyle_personality_noise": "차분한 갈색 소형견",
                    "negation_diagnostic": "갈색이 아닌 소형견",
                },
            }
        ],
    }


def _metas() -> list[dict]:
    return [
        {
            "desertionNo": "a",
            "colorCd": "갈색",
            "weight": "4(Kg)",
            "age": "2020(년생)",
            "processState": "보호중",
            "noticeEdt": "2026-08-01",
        },
        {
            "desertionNo": "b",
            "colorCd": "갈색",
            "weight": "5(Kg)",
            "age": "2021(년생)",
            "processState": "보호중",
            "noticeEdt": "2026-08-01",
        },
        {
            "desertionNo": "d",
            "colorCd": "검정",
            "weight": "11(Kg)",
            "age": "2022(년생)",
            "processState": "보호중",
            "noticeEdt": "2026-08-01",
        },
        {
            "desertionNo": "closed",
            "colorCd": "갈색",
            "weight": "4(Kg)",
            "age": "2020(년생)",
            "processState": "종료(입양)",
            "noticeEdt": "2026-07-20",
        },
    ]


def _rankings() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    base = {"q": ["a", "b", "d"]}
    variants = {
        "q::synonym_paraphrase": ["b", "a", "d"],
        "q::word_order": ["a", "d", "b"],
        "q::conservative_typo": ["a", "b", "d"],
        "q::lifestyle_personality_noise": ["a", "b", "d"],
        "q::negation_diagnostic": ["d", "b", "a"],
    }
    return base, variants


def _report() -> tuple[dict, dict, dict, dict[str, set[str]], dict]:
    base_payload = _base_payload()
    variant_payload = _variant_payload()
    metas = _metas()
    qrels = build_silver_qrels(
        metas,
        base_payload["queries"],
        REFERENCE_DATE,
    )
    base_rankings, variant_rankings = _rankings()
    report = build_report(
        generated_at="2026-07-26T00:00:00+00:00",
        reference_date=REFERENCE_DATE,
        base_query_payload=base_payload,
        variant_payload=variant_payload,
        qrels=qrels,
        metas_by_id=merge_notice_metas(metas),
        base_rankings=base_rankings,
        variant_rankings=variant_rankings,
        artifacts={},
        runtime={"evaluation_depth": 10},
        corpus={"documents": 4},
    )
    return report, base_payload, variant_payload, qrels, merge_notice_metas(metas)


def test_development_variant_design_is_complete_and_separates_negation() -> None:
    base = json.loads(BASE_QUERIES.read_text(encoding="utf-8"))
    variants = json.loads(VARIANTS.read_text(encoding="utf-8"))

    expanded = validate_variant_payload(variants, base)

    assert len(expanded) == 12 * len(CATEGORY_POLICY)
    assert all(
        sum(row["category"] == category for row in expanded) == 12
        for category in CATEGORY_POLICY
    )
    negations = [row for row in expanded if row["category"] == "negation_diagnostic"]
    assert negations
    assert all(not row["included_in_robustness_aggregate"] for row in negations)
    assert all(row["query_policy"]["unsupported_conditions"] for row in negations)
    assert all(row["query_policy"]["warnings"] for row in negations)
    assert all(row["query_policy"]["is_fully_supported"] is False for row in negations)
    meaning_preserving = [
        row for row in expanded if row["category"] != "negation_diagnostic"
    ]
    assert all(
        row["query_policy"]["unsupported_conditions"] == []
        for row in meaning_preserving
    )
    noise = [
        row for row in expanded if row["category"] == "lifestyle_personality_noise"
    ]
    base_by_id = {row["id"]: row for row in base["queries"]}
    assert all(
        row["normalized_query"]
        == normalize_appearance_query(base_by_id[row["base_id"]]["query"])
        for row in noise
    )


def test_variant_validation_rejects_tuned_or_mislabeled_inputs() -> None:
    payload = _variant_payload()
    payload["queries"][0]["variants"]["conservative_typo"] = "검정 대형견"
    with pytest.raises(ValueError, match="one codepoint"):
        validate_variant_payload(payload, _base_payload())

    payload = _variant_payload()
    payload["queries"][0]["variants"]["lifestyle_personality_noise"] = (
        "차분한 검정 대형견"
    )
    with pytest.raises(ValueError, match="normalize exactly"):
        validate_variant_payload(payload, _base_payload())

    payload = _variant_payload()
    payload["queries"][0]["variants"]["negation_diagnostic"] = "갈색 소형견 부탁해"
    with pytest.raises(ValueError, match="negation"):
        validate_variant_payload(payload, _base_payload())


def test_rank_stability_metrics_are_bounded_and_order_sensitive() -> None:
    assert top_k_jaccard(["a", "b"], ["a", "b"], 5) == 1.0
    assert rank_biased_overlap(["a", "b"], ["a", "b"]) == 1.0
    reordered = rank_stability(["a", "b", "c"], ["b", "a", "c"])
    assert reordered["top5_jaccard"] == 1.0
    assert reordered["top10_jaccard"] == 1.0
    assert 0.0 < reordered["rbo@10"] < 1.0
    assert reordered["top10_exact_match"] is False


def test_report_recalculates_all_evidence_without_quality_threshold() -> None:
    report, base, variants, qrels, metas_by_id = _report()

    assert report["validation"]["passed"] is True
    assert report["evaluation_contract"]["performance_thresholds"] is None
    assert report["robustness_summary"]["metrics"]["query_count"] == 4
    assert (
        report["categories"]["negation_diagnostic"]["included_in_robustness_aggregate"]
        is False
    )
    assert report["inactive_exposure"]["maximum"] == 0.0
    assert (
        report_recalculation_failures(
            report,
            reference_date=REFERENCE_DATE,
            base_query_payload=base,
            variant_payload=variants,
            qrels=qrels,
            metas_by_id=metas_by_id,
        )
        == []
    )

    tampered = copy.deepcopy(report)
    tampered["baseline"]["queries"][0]["metrics"]["precision@5"] = 0.999
    failures = report_recalculation_failures(
        tampered,
        reference_date=REFERENCE_DATE,
        base_query_payload=base,
        variant_payload=variants,
        qrels=qrels,
        metas_by_id=metas_by_id,
    )
    assert any("baseline" in failure for failure in failures)


def test_inactive_exposure_and_noise_changes_fail_closed() -> None:
    report, *_ = _report()
    tampered = copy.deepcopy(report)
    noise = tampered["categories"]["lifestyle_personality_noise"]["variants"][0]
    noise["ranking_ids"] = ["b", "a", "d"]
    assert any(
        "noise changed" in failure for failure in report_contract_failures(tampered)
    )

    tampered = copy.deepcopy(report)
    tampered["baseline"]["queries"][0]["metrics"]["inactive_exposure"] = 0.1
    assert any("inactive" in failure for failure in report_contract_failures(tampered))


def test_markdown_states_silver_and_negation_limits() -> None:
    markdown = report_to_markdown(_report()[0])
    assert "품질 합격선 없이 기술 통계" in markdown
    assert "부정 질의의 검색 정확도가 아닙니다" in markdown
    assert "독립 테스트가 아닌 개발셋 회귀 결과" in markdown
    assert "미관측 한국어 holdout" in markdown
    assert "사람의 관련성 판정이나 입양 적합성 검증이 아닙니다" in markdown


def test_tracked_report_is_current_and_recalculates() -> None:
    assert DEFAULT_JSON_OUT.exists()
    assert DEFAULT_MARKDOWN_OUT.exists()
    report = json.loads(DEFAULT_JSON_OUT.read_text(encoding="utf-8"))
    assert {
        "evaluation_module",
        "evaluation_cli",
        "appearance_query_module",
        "hybrid_rag_module",
        "graph_rag_module",
        "retrieval_evaluation_module",
        "retrieval_evaluation_cli",
    }.issubset(report["artifacts"])
    assert check_tracked_report(parse_args(["--check"])) == []


def test_check_rejects_stale_retrieval_code_hash(tmp_path: Path) -> None:
    report = json.loads(DEFAULT_JSON_OUT.read_text(encoding="utf-8"))
    report["artifacts"]["evaluation_module"] = report_artifact(MODULE_PATH)
    report["artifacts"]["evaluation_cli"] = report_artifact(SCRIPT_PATH)
    report["artifacts"]["hybrid_rag_module"]["sha256"] = "0" * 64
    json_out = tmp_path / "report.json"
    markdown_out = tmp_path / "report.md"
    json_out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_out.write_text(report_to_markdown(report), encoding="utf-8")
    args = parse_args(
        [
            "--check",
            "--json-out",
            str(json_out),
            "--markdown-out",
            str(markdown_out),
        ]
    )

    failures = check_tracked_report(args)

    assert "tracked report artifact hash is stale: hybrid_rag_module" in failures
