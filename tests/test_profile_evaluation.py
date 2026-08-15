from __future__ import annotations

import json
from pathlib import Path

import pytest

import app.profile_evaluation as profile_evaluation_module
from app.profile_evaluation import (
    derive_silver_label,
    evaluate_profile_reranking,
    join_notice_rows,
    ranking_metrics,
    raw_appearance,
    render_markdown,
    serialize_report,
    write_report,
)
from app.profile_rerank import UserProfile
from scripts.evaluate_profile_reranking import main as profile_evaluation_main


def _profile(size: str, age: str, region: str) -> dict[str, object]:
    return {
        "housing_type": "apartment",
        "daily_absence_hours": 8,
        "activity_level": "low",
        "dog_experience": "none",
        "preferred_size": size,
        "preferred_age": age,
        "preferred_region": region,
        "has_children": True,
        "has_other_pets": True,
    }


def _notice(
    notice_id: str,
    *,
    weight: str,
    age: str,
    region: str,
    row_type: str = "text",
) -> dict[str, object]:
    return {
        "desertionNo": notice_id,
        "weight": weight,
        "age": age,
        "org_name": region,
        "process_state": "보호중",
        "notice_end": "20261231",
        "type": row_type,
    }


def test_join_notice_rows_is_deterministic_and_prefers_text_source() -> None:
    rows = [
        {
            "desertionNo": "dog-1",
            "type": "image",
            "weight": "99(Kg)",
            "image_url": "https://example.test/dog-1.jpg",
        },
        {
            "desertionNo": "dog-1",
            "type": "text",
            "weight": "7(Kg)",
            "age": "2022(년생)",
        },
        {"desertionNo": "dog-2", "type": "text", "weight": "10(Kg)"},
    ]

    joined = join_notice_rows(reversed(rows))

    assert list(joined) == ["dog-1", "dog-2"]
    assert joined["dog-1"]["weight"] == "7(Kg)"
    assert joined["dog-1"]["age"] == "2022(년생)"
    assert joined["dog-1"]["image_url"].endswith("dog-1.jpg")


def test_silver_label_uses_only_raw_notice_weight_age_and_region() -> None:
    profile = UserProfile.model_validate(_profile("small", "adult", "서울"))
    derived_only = {
        "desertionNo": "derived-only",
        "size": "small",
        "age_group": "adult",
        "region": "서울",
        "vlm_attrs": {"body_size_hint": "small"},
        "activity_level": "low",
        "children_compatible": True,
    }

    appearance = raw_appearance(derived_only, reference_year=2026)
    label = derive_silver_label(derived_only, profile, reference_year=2026)

    assert appearance["size"] == "unknown"
    assert appearance["age_group"] == "unknown"
    assert appearance["region"] == "서울"
    assert label["condition_states"] == {
        "preferred_size": "unknown",
        "preferred_age": "unknown",
        "preferred_region": "matched",
    }
    assert label["relevance_grade"] == pytest.approx(1 / 3, abs=1e-6)
    assert label["binary_relevant"] is False


def test_ranking_metrics_uses_binary_precision_and_graded_ndcg() -> None:
    labels = {
        "a": {"binary_relevant": False, "relevance_grade": 0.0},
        "b": {"binary_relevant": True, "relevance_grade": 1.0},
        "c": {"binary_relevant": False, "relevance_grade": 2 / 3},
        "d": {"binary_relevant": True, "relevance_grade": 1.0},
        "e": {"binary_relevant": False, "relevance_grade": 1 / 3},
    }

    metrics = ranking_metrics(["a", "b", "c", "d", "e"], labels, k=5)

    assert metrics["precision_at_5"] == pytest.approx(0.4)
    assert metrics["mrr"] == pytest.approx(0.5)
    assert 0 < metrics["ndcg_at_5"] < 1


def test_actual_reranker_changes_same_pool_by_appearance_profile(
    tmp_path: Path,
) -> None:
    rows = [
        _notice("distractor", weight="11(Kg)", age="2022(년생)", region="부산"),
        _notice("large-1", weight="22(Kg)", age="2022(년생)", region="경기도"),
        _notice("small-1", weight="5(Kg)", age="2022(년생)", region="서울특별시"),
        _notice("small-2", weight="7(Kg)", age="2023(년생)", region="서울특별시"),
        _notice("large-2", weight="25(Kg)", age="2023(년생)", region="경기도"),
        _notice("partial", weight="10(Kg)", age="2026(년생)", region="서울특별시"),
        _notice("small-3", weight="6(Kg)", age="2021(년생)", region="서울특별시"),
        _notice("large-3", weight="21(Kg)", age="2021(년생)", region="경기도"),
        {
            "desertionNo": "small-1",
            "type": "image",
            "vlm_attrs": {"body_size_hint": "large"},
        },
    ]
    config = {
        "schema_version": 1,
        "evaluation_id": "test-profile-eval",
        "reference_date": "2026-07-23",
        "baseline_score_start": 0.6,
        "baseline_score_step": 0.002,
        "settings": {"compatibility_weight": 0.25, "quality_weight": 0.0},
        "quality_weight_sensitivity": {
            "control_quality_weight": 0.0,
            "production_default_quality_weight": 0.05,
            "top_k": 5,
        },
        "candidate_ids": [
            "distractor",
            "large-1",
            "small-1",
            "small-2",
            "large-2",
            "partial",
            "small-3",
            "large-3",
        ],
        "profiles": [
            {
                "id": "small-seoul",
                "label": "A: 서울 소형 성견",
                "profile": _profile("small", "adult", "서울"),
            },
            {
                "id": "large-gyeonggi",
                "label": "B: 경기 대형 성견",
                "profile": _profile("large", "adult", "경기"),
            },
        ],
    }
    metas_path = tmp_path / "dog_metas.json"
    config_path = tmp_path / "profiles.json"
    metas_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")

    report = evaluate_profile_reranking(config_path, metas_path)

    first, second = report["profiles"]
    assert report["schema_version"] == 2
    assert first["baseline_order"] == second["baseline_order"]
    assert first["reranked_order"][0].startswith("small-")
    assert second["reranked_order"][0].startswith("large-")
    assert first["reranked_order"][0] != second["reranked_order"][0]
    assert first["metric_delta"]["precision_at_5"] > 0
    assert second["metric_delta"]["precision_at_5"] >= 0
    assert report["method"]["lifestyle_or_behavior_scored"] is False
    assert report["method"]["evaluation_type"] == (
        "deterministic_rerank_contract_not_performance"
    )
    assert report["summary"]["passed"] is True
    assert report["summary"]["failure_count"] == 0
    assert report["summary"]["unknown_neutrality_failures"] == 0
    assert report["summary"]["evidence_consistency_failures"] == 0
    assert report["summary"]["silver_score_mismatches"] == 0
    assert report["summary"]["adoption_suitability_certainty_phrase_hits"] == 0
    sensitivity = report["quality_weight_sensitivity"]
    assert sensitivity["summary"]["passed"] is True
    assert sensitivity["settings"]["control_quality_weight"] == 0.0
    assert sensitivity["settings"]["production_default_quality_weight"] == 0.05
    assert sensitivity["settings"]["production_default_matches_code_default"] is True
    assert sensitivity["quality_coverage"]["known_count"] == 0
    assert sensitivity["quality_coverage"]["unknown_count"] == len(
        config["candidate_ids"]
    )

    markdown = render_markdown(report)
    assert "독립적인 검색 성능·사용자 효용 평가가 아닙니다" in markdown
    assert "동일 후보군의 프로필 A/B 순서 변화" in markdown
    assert "사람이 판정한 입양 적합도 평가가 아니며" in markdown
    assert "입양 적합도를 확정하는 문구 탐지: **0건**" in markdown
    assert "사진 품질 가중치 결정적 민감도 부록" in markdown
    assert "검색 성능 향상, 사용자 효용 또는 최적 가중치를 주장" in markdown

    json_out = tmp_path / "report.json"
    markdown_out = tmp_path / "report.md"
    write_report(report, json_out, markdown_out)
    assert json_out.read_text(encoding="utf-8") == serialize_report(report)
    assert markdown_out.read_text(encoding="utf-8") == markdown


def test_repository_fixed_pool_quality_weight_sensitivity_appendix() -> None:
    root = Path(__file__).resolve().parents[1]
    report = evaluate_profile_reranking(
        root / "data" / "eval_profiles.appearance_v1.json",
        root / "data" / "dog_metas.json",
    )
    sensitivity = report["quality_weight_sensitivity"]
    summary = sensitivity["summary"]
    coverage = sensitivity["quality_coverage"]

    assert report["method"]["quality_weight"] == 0.0
    assert sensitivity["evaluation_type"] == (
        "deterministic_sensitivity_contract_not_performance"
    )
    assert sensitivity["settings"]["control_quality_weight"] == 0.0
    assert sensitivity["settings"]["production_default_quality_weight"] == 0.05
    assert sensitivity["settings"]["production_default_matches_code_default"] is True
    assert summary["passed"] is True
    assert summary["profiles_with_top1_change"] == 0
    assert summary["profiles_with_top_k_membership_change"] == 0
    assert summary["profiles_with_top_k_position_change"] == 1
    assert summary["mean_profile_mean_absolute_rank_shift"] == pytest.approx(0.5)
    assert summary["max_absolute_rank_shift_across_profiles"] == 3
    assert summary["unknown_observations_with_rank_change"] == 0
    assert summary["unknown_observations_with_direct_score_change"] == 0
    assert summary["unknown_observations_with_relative_context_change"] == 2

    assert coverage["candidate_count"] == 12
    assert coverage["known_count"] == 11
    assert coverage["unknown_count"] == 1
    assert coverage["known_rate"] == pytest.approx(11 / 12, abs=1e-6)
    assert coverage["unknown_candidate_ids"] == ["441570202600371"]
    assert coverage["known_quality_bonus_at_production_default_stats"] == {
        "min": 0.03366,
        "mean": 0.045838,
        "max": 0.04906,
    }

    profiles = {profile["profile_id"]: profile for profile in sensitivity["profiles"]}
    first = profiles["profile_a_small_adult_seoul"]
    second = profiles["profile_b_large_adult_gyeonggi"]
    assert first["top1"]["changed"] is False
    assert first["top_k_membership"]["changed_candidate_count"] == 0
    assert first["top_k_position"]["changed_candidate_count"] == 3
    assert first["top_k_position"]["changed_notice_ids"] == [
        "411310202600444",
        "411315202600532",
        "411315202600531",
    ]
    assert first["top_k_position"]["changed_slot_count"] == 3
    assert first["all_candidate_rank_change"]["mean_absolute_rank_change"] == 0.5
    assert first["all_candidate_rank_change"]["max_absolute_rank_change"] == 2
    assert second["top1"]["changed"] is False
    assert second["top_k_membership"]["changed_candidate_count"] == 0
    assert second["top_k_position"]["changed_candidate_count"] == 0
    assert second["top_k_position"]["changed_notice_ids"] == []
    assert second["top_k_position"]["changed_slot_count"] == 0
    assert second["all_candidate_rank_change"]["mean_absolute_rank_change"] == 0.5
    assert second["all_candidate_rank_change"]["max_absolute_rank_change"] == 3

    first_unknown = first["unknown_quality_candidates"][0]
    second_unknown = second["unknown_quality_candidates"][0]
    assert first_unknown["notice_id"] == "441570202600371"
    assert first_unknown["control"]["rank"] == 6
    assert first_unknown["production_default"]["rank"] == 6
    assert first_unknown["score_delta"] == 0.0
    assert first_unknown["direct_score_neutral"] is True
    assert first_unknown["control"]["margin_to_higher"] == pytest.approx(0.012)
    assert first_unknown["production_default"]["margin_to_higher"] == pytest.approx(
        0.059015
    )
    assert first_unknown["control"]["margin_over_lower"] == pytest.approx(0.065334)
    assert first_unknown["production_default"]["margin_over_lower"] == pytest.approx(
        0.019594
    )
    assert second_unknown["control"]["rank"] == 5
    assert second_unknown["production_default"]["rank"] == 5
    assert second_unknown["score_delta"] == 0.0
    assert second_unknown["direct_score_neutral"] is True
    assert second_unknown["control"]["margin_to_higher"] == pytest.approx(0.081333)
    assert second_unknown["production_default"]["margin_to_higher"] == pytest.approx(
        0.129083
    )
    assert second_unknown["control"]["margin_over_lower"] == pytest.approx(0.065334)
    assert second_unknown["production_default"]["margin_over_lower"] == pytest.approx(
        0.021969
    )


def test_profile_cli_returns_nonzero_when_contract_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        _notice(
            f"dog-{index}",
            weight=f"{4 + index}(Kg)",
            age="2022(년생)",
            region="서울특별시",
        )
        for index in range(5)
    ]
    config = {
        "schema_version": 1,
        "evaluation_id": "profile-cli-contract",
        "reference_date": "2026-07-26",
        "settings": {"compatibility_weight": 0.25, "quality_weight": 0.0},
        "candidate_ids": [f"dog-{index}" for index in range(5)],
        "profiles": [
            {
                "id": "a",
                "profile": _profile("small", "adult", "서울"),
            },
            {
                "id": "b",
                "profile": _profile("large", "adult", "경기"),
            },
        ],
    }
    metas_path = tmp_path / "metas.json"
    config_path = tmp_path / "config.json"
    json_out = tmp_path / "report.json"
    markdown_out = tmp_path / "report.md"
    metas_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
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

    assert profile_evaluation_main(args) == 0
    monkeypatch.setattr(
        profile_evaluation_module,
        "_unknown_neutrality_probe",
        lambda *_args: {
            "passed": False,
            "retrieval_score": 0.5,
            "compatibility_score": 0.1,
            "final_score": 0.6,
            "score_delta": 0.1,
            "unknown_condition_count": 3,
        },
    )

    assert profile_evaluation_main(args) == 1
    failed_report = json.loads(json_out.read_text(encoding="utf-8"))
    assert failed_report["summary"]["passed"] is False
    assert failed_report["summary"]["failure_count"] == 2
