from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

import app.safety_contract_evaluation as safety_evaluation
from app.safety_contract_evaluation import (
    evaluate_safety_contract,
    render_markdown,
    serialize_report,
    sha256_file,
)
from scripts.evaluate_safety_contract import main


ROOT = Path(__file__).resolve().parents[1]
TRACKED_CONFIG = ROOT / "data" / "eval_safety_contract.appearance_v1.json"
TRACKED_METAS = ROOT / "data" / "dog_metas.json"


def _synthetic_rows() -> list[dict[str, object]]:
    common: dict[str, object] = {
        "processState": "보호중",
        "noticeEdt": "20260831",
        "age": "2022(년생)",
        "careAddr": "서울특별시 중구",
        "behavior_evidence_source": "shelter_reported",
    }
    return [
        {
            **common,
            "desertionNo": "D-001",
            "weight": "4.0(Kg)",
            "photo_quality_score": 0.9,
            "housing_types": ["apartment"],
            "activity_level": "low",
            "max_absence_hours": 10,
            "recommended_experience": "none",
            "children_compatible": True,
            "other_pets_compatible": True,
        },
        {
            **common,
            "desertionNo": "D-002",
            "weight": "16.0(Kg)",
            "photo_quality_score": 0.5,
            "housing_types": ["house"],
            "activity_level": "high",
            "max_absence_hours": 1,
            "recommended_experience": "experienced",
            "children_compatible": False,
            "other_pets_compatible": False,
        },
        {
            **common,
            "desertionNo": "D-003",
            "weight": "7.0(Kg)",
            "photo_quality_score": 0.7,
            "housing_types": ["other"],
            "activity_level": "medium",
            "max_absence_hours": 4,
            "recommended_experience": "some",
        },
    ]


def _write_inputs(
    tmp_path: Path,
    *,
    missing_candidate: bool = False,
) -> tuple[Path, Path]:
    config = json.loads(TRACKED_CONFIG.read_text(encoding="utf-8"))
    config["candidate_ids"] = ["D-001", "D-002", "D-003"]
    if missing_candidate:
        config["candidate_ids"].append("D-MISSING")
    config_path = tmp_path / "config.json"
    metas_path = tmp_path / "metas.json"
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metas_path.write_text(
        json.dumps(_synthetic_rows(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return config_path, metas_path


def _cases_by_id(report: dict[str, object]) -> dict[str, dict[str, object]]:
    return {case["id"]: case for case in report["cases"]}  # type: ignore[index]


def test_safety_contract_records_all_four_passing_contracts(
    tmp_path: Path,
) -> None:
    config_path, metas_path = _write_inputs(tmp_path)

    report = evaluate_safety_contract(config_path, metas_path)
    cases = _cases_by_id(report)

    assert report["summary"] == {
        "passed": True,
        "total_contracts": 4,
        "passed_contracts": 4,
        "failed_contracts": 0,
        "failed_contract_ids": [],
        "failure_details": [],
    }
    assert report["schema_version"] == 2
    assert report["inputs"]["config_sha256"] == sha256_file(config_path)
    assert report["inputs"]["dog_metas_sha256"] == sha256_file(metas_path)
    assert report["inputs"]["fixed_candidate_ids"] == ["D-001", "D-002", "D-003"]

    query_evidence = cases["appearance_query_equivalence"]["evidence"]
    assert all(row["equivalent"] for row in query_evidence["pairs"])
    assert all(
        row["missing_required_appearance_terms"] == []
        for row in query_evidence["pairs"]
    )
    assert all(
        row["expected_literal_query"] == row["appearance_query"]
        for row in query_evidence["pairs"]
    )

    profile_evidence = cases["appearance_rerank_lifestyle_isolation"]["evidence"]
    assert profile_evidence["appearance_fields_equal"] is True
    assert profile_evidence["outputs_identical"] is True
    assert profile_evidence["profile_a_order"] == profile_evidence["profile_b_order"]
    assert (
        profile_evidence["profile_a_order"] == profile_evidence["minimal_profile_order"]
    )
    assert (
        profile_evidence["profile_a_output_sha256"]
        == profile_evidence["profile_b_output_sha256"]
        == profile_evidence["minimal_profile_output_sha256"]
    )
    assert set(profile_evidence["minimal_profile_fields"]) == {
        "preferred_size",
        "preferred_age",
        "preferred_region",
    }
    assert set(profile_evidence["minimal_profile_omitted_lifestyle_fields"]) == {
        "housing_type",
        "daily_absence_hours",
        "activity_level",
        "dog_experience",
        "has_children",
        "has_other_pets",
    }
    assert profile_evidence["minimal_profile_unexpected_fields"] == []
    assert len(profile_evidence["shared_output"]) == 3
    assert profile_evidence["missing_candidate_ids"] == []

    inquiry_evidence = cases["inquiry_preferences_change_questions"]["evidence"]
    assert inquiry_evidence["profile_to_inquiry_alignment_failures"] == []
    assert inquiry_evidence["questions_differ"] is True
    assert inquiry_evidence["profile_a_unique_questions"]
    assert inquiry_evidence["profile_b_unique_questions"]
    assert any(
        "선호하는 활동량은 낮은 편" in question
        for question in inquiry_evidence["profile_a_unique_questions"]
    )
    assert any(
        "선호하는 활동량은 높은 편" in question
        for question in inquiry_evidence["profile_b_unique_questions"]
    )

    leakage_evidence = cases["nonappearance_expression_leakage"]["evidence"]
    assert leakage_evidence["remaining_forbidden_keys"] == []
    assert leakage_evidence["unsupported_keys"] == []
    assert leakage_evidence["condition_forbidden_term_hits"] == []
    assert leakage_evidence["query_forbidden_term_hits"] == {}


def test_missing_fixed_candidate_is_a_contract_failure(tmp_path: Path) -> None:
    config_path, metas_path = _write_inputs(tmp_path, missing_candidate=True)

    report = evaluate_safety_contract(config_path, metas_path)
    profile_case = _cases_by_id(report)["appearance_rerank_lifestyle_isolation"]

    assert report["summary"]["passed"] is False
    assert report["summary"]["failed_contracts"] == 1
    assert report["summary"]["failed_contract_ids"] == [
        "appearance_rerank_lifestyle_isolation"
    ]
    assert profile_case["evidence"]["missing_candidate_ids"] == ["D-MISSING"]
    assert any(
        "D-MISSING" in failure for failure in report["summary"]["failure_details"]
    )


def test_inquiry_activity_level_must_match_profile(tmp_path: Path) -> None:
    config_path, metas_path = _write_inputs(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["inquiry_preferences"]["b"]["activity_level"] = "low"
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = evaluate_safety_contract(config_path, metas_path)
    inquiry_case = _cases_by_id(report)["inquiry_preferences_change_questions"]

    assert inquiry_case["passed"] is False
    assert inquiry_case["evidence"]["profile_to_inquiry_alignment_failures"] == [
        "B:activity_level"
    ]
    assert any("B:activity_level" in failure for failure in inquiry_case["failures"])


def test_query_contract_fails_when_production_normalizer_drops_appearance_terms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, metas_path = _write_inputs(tmp_path)
    monkeypatch.setattr(
        safety_evaluation,
        "normalize_appearance_query",
        lambda _value: "강아지",
    )

    report = evaluate_safety_contract(config_path, metas_path)
    query_case = _cases_by_id(report)["appearance_query_equivalence"]

    assert query_case["passed"] is False
    assert report["summary"]["passed"] is False
    assert any(
        "required appearance terms were removed" in failure
        for failure in query_case["failures"]
    )


def test_query_contract_rejects_required_term_missing_from_literal_expectation(
    tmp_path: Path,
) -> None:
    config_path, metas_path = _write_inputs(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["query_pairs"][0]["required_appearance_terms"].append("검정")
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected query omits required terms"):
        evaluate_safety_contract(config_path, metas_path)


def test_query_contract_rejects_legacy_schema(tmp_path: Path) -> None:
    config_path, metas_path = _write_inputs(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["schema_version"] = 1
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schema_version must be 2"):
        evaluate_safety_contract(config_path, metas_path)


def test_markdown_explicitly_disclaims_adoption_suitability(
    tmp_path: Path,
) -> None:
    config_path, metas_path = _write_inputs(tmp_path)
    report = evaluate_safety_contract(config_path, metas_path)

    markdown = render_markdown(report)
    serialized = serialize_report(report)

    assert "입양 적합성 성능 평가가 아닙니다" in markdown
    assert "비추론" in markdown
    assert report["inputs"]["config_sha256"] in markdown
    assert report["inputs"]["dog_metas_sha256"] in markdown
    assert serialized.endswith("\n")
    assert json.loads(serialized) == report


def test_cli_writes_json_and_markdown_and_returns_contract_status(
    tmp_path: Path,
) -> None:
    config_path, metas_path = _write_inputs(tmp_path)
    json_out = tmp_path / "report.json"
    markdown_out = tmp_path / "report.md"
    args = [
        "--config",
        str(config_path),
        "--metas",
        str(metas_path),
        "--json-out",
        str(json_out),
        "--markdown-out",
        str(markdown_out),
    ]

    assert main(args) == 0
    assert json.loads(json_out.read_text(encoding="utf-8"))["summary"]["passed"]
    assert "입양 적합성 성능 평가가 아닙니다" in markdown_out.read_text(
        encoding="utf-8"
    )
    assert main([*args, "--check"]) == 0

    failing_config = json.loads(config_path.read_text(encoding="utf-8"))
    failing_config["candidate_ids"] = [
        *deepcopy(failing_config["candidate_ids"]),
        "D-MISSING",
    ]
    config_path.write_text(
        json.dumps(failing_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    assert main(args) == 1
    assert (
        json.loads(json_out.read_text(encoding="utf-8"))["summary"]["passed"] is False
    )


def test_tracked_safety_contract_passes_current_metadata() -> None:
    report = evaluate_safety_contract(TRACKED_CONFIG, TRACKED_METAS)

    assert report["summary"]["passed"] is True
    assert report["summary"]["passed_contracts"] == 4
    assert report["inputs"]["available_candidate_count"] == len(
        report["inputs"]["fixed_candidate_ids"]
    )
