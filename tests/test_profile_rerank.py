from __future__ import annotations

import ast
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from app import profile_rerank
from app.notice_status import classify_notice as shared_classify_notice
from app.profile_rerank import (
    ProfileRerankSettings,
    ProfileSearchRequest,
    UserProfile,
    normalize_dog,
    rerank_candidates,
)


REQUIRED_PROFILE_FIELDS = {
    "housing_type",
    "daily_absence_hours",
    "activity_level",
    "dog_experience",
    "preferred_size",
    "preferred_age",
    "preferred_region",
    "has_children",
    "has_other_pets",
}

REQUIRED_RESULT_FIELDS = {
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
}


@pytest.fixture
def reference_date() -> datetime:
    return datetime(2026, 7, 17, 12, 0, 0)


@pytest.fixture
def small_profile_payload() -> dict[str, object]:
    return {
        "housing_type": "apartment",
        "daily_absence_hours": 8,
        "activity_level": "low",
        "dog_experience": "none",
        "preferred_size": "small",
        "preferred_age": "adult",
        "preferred_region": "서울",
        "has_children": True,
        "has_other_pets": False,
    }


@pytest.fixture
def small_profile(small_profile_payload: dict[str, object]) -> UserProfile:
    return UserProfile.model_validate(small_profile_payload)


@pytest.fixture
def large_profile() -> UserProfile:
    return UserProfile(
        housing_type="house",
        daily_absence_hours=2,
        activity_level="high",
        dog_experience="experienced",
        preferred_size="large",
        preferred_age="puppy",
        preferred_region="부산",
        has_children=False,
        has_other_pets=True,
    )


@pytest.fixture
def small_active_candidate() -> dict[str, object]:
    return {
        "score": 0.52,
        "desertionNo": "small-active",
        "age": "2022(년생)",
        "weight": "7(Kg)",
        "sexCd": "F",
        "neuterYn": "Y",
        "careAddr": "서울특별시 마포구",
        "kindCd": "[개] 믹스견",
        "merged_desc": "차분하고 아이와 생활 가능한 개체",
        "vlm_attrs": {
            "body_size_hint": "small",
            "photo_quality_score": 0.8,
            "coat_color": ["white"],
        },
        "housing_types": ["apartment"],
        "activity_level": "low",
        "max_absence_hours": 10,
        "recommended_experience": "none",
        "children_compatible": True,
        "other_pets_compatible": False,
        "processState": "보호중",
        "noticeEdt": "20261231",
        "last_verified_at": "2026-07-17T03:00:00",
        "detail_url": "https://example.test/small-active",
    }


@pytest.fixture
def large_active_candidate() -> dict[str, object]:
    return {
        "score": 0.52,
        "desertionNo": "large-active",
        "age": "2026(년생)",
        "weight": "25(Kg)",
        "sexCd": "M",
        "neuterYn": "N",
        "careAddr": "부산광역시 해운대구",
        "kindCd": "[개] 대형 믹스견",
        "desc": "활발하고 다른 반려동물과 생활 가능한 개체",
        "vlm_attrs": {
            "body_size_hint": "large",
            "photo_quality_score": 0.8,
            "coat_color": ["black"],
        },
        "housing_types": ["house"],
        "activity_level": "high",
        "max_absence_hours": 4,
        "recommended_experience": "experienced",
        "children_compatible": False,
        "other_pets_compatible": True,
        "processState": "보호중",
        "noticeEdt": "20261231",
        "detail_url": "https://example.test/large-active",
    }


def test_user_profile_and_search_request_contract(
    small_profile_payload: dict[str, object],
) -> None:
    payload = dict(small_profile_payload)
    payload["preferred_region"] = "  서울  "
    profile = UserProfile.model_validate(payload)

    dumped = profile.model_dump(mode="json")
    assert set(dumped) == REQUIRED_PROFILE_FIELDS
    assert dumped["preferred_region"] == "서울"
    assert isinstance(dumped["has_children"], bool)
    assert isinstance(dumped["has_other_pets"], bool)

    without_region = dict(small_profile_payload)
    without_region.pop("preferred_region")
    assert UserProfile.model_validate(without_region).preferred_region is None

    with pytest.raises(ValidationError):
        UserProfile.model_validate({**small_profile_payload, "unexpected": "value"})
    with pytest.raises(ValidationError):
        UserProfile.model_validate({**small_profile_payload, "has_children": "true"})
    with pytest.raises(ValidationError):
        UserProfile.model_validate({**small_profile_payload, "daily_absence_hours": -1})
    with pytest.raises(ValidationError):
        UserProfile.model_validate({**small_profile_payload, "daily_absence_hours": 25})
    with pytest.raises(ValidationError):
        UserProfile.model_validate(
            {**small_profile_payload, "daily_absence_hours": True}
        )
    with pytest.raises(ValidationError):
        UserProfile.model_validate(
            {**small_profile_payload, "daily_absence_hours": float("inf")}
        )

    request = ProfileSearchRequest(
        query="  차분한 소형견  ",
        profile=profile,
        topk=2,
    )
    assert request.query == "차분한 소형견"
    assert request.topk == 2

    conditions_only = ProfileSearchRequest(
        conditions={"body_size_hint": ["small"]},
        profile=profile,
    )
    assert conditions_only.query == ""
    assert conditions_only.conditions == {"body_size_hint": ["small"]}

    with pytest.raises(ValidationError):
        ProfileSearchRequest(profile=profile)
    with pytest.raises(ValidationError):
        ProfileSearchRequest(conditions={"keywords": []}, profile=profile)
    with pytest.raises(ValidationError):
        ProfileSearchRequest(conditions={"unsupported": "value"}, profile=profile)
    with pytest.raises(ValidationError):
        ProfileSearchRequest(conditions={"face_visible": "maybe"}, profile=profile)
    with pytest.raises(ValidationError):
        ProfileSearchRequest(conditions={"coat_color": {}}, profile=profile)
    with pytest.raises(ValidationError):
        ProfileSearchRequest(conditions={"sex": ["X"]}, profile=profile)
    with pytest.raises(ValidationError):
        ProfileSearchRequest(query="소형견", profile=profile, topk=21)


def test_normalize_dog_maps_public_aliases_to_canonical_evidence(
    small_active_candidate: dict[str, object],
    reference_date: datetime,
) -> None:
    dog = normalize_dog(small_active_candidate, reference_date=reference_date)

    assert dog.dog_id == "small-active"
    assert dog.age == pytest.approx(4.0)
    assert dog.age_group == "adult"
    assert dog.weight == pytest.approx(7.0)
    assert dog.size == "small"
    assert dog.sex == "female"
    assert dog.neutered == "yes"
    assert dog.region == "서울"
    assert dog.breed == "[개] 믹스견"
    assert dog.mixed_breed is True
    assert dog.description == "차분하고 아이와 생활 가능한 개체"
    assert dog.vlm_attributes["body_size_hint"] == "small"
    assert dog.vlm_attributes["coat_color"] == ["white"]
    assert dog.photo_quality_score == pytest.approx(0.8)
    assert dog.notice_status == "active"
    assert dog.notice_end == "20261231"
    assert dog.last_verified_at == "2026-07-17T03:00:00"
    assert dog.source_url == "https://example.test/small-active"


def test_same_candidates_reverse_order_for_different_profiles(
    small_profile: UserProfile,
    large_profile: UserProfile,
    small_active_candidate: dict[str, object],
    large_active_candidate: dict[str, object],
    reference_date: datetime,
) -> None:
    settings = ProfileRerankSettings(compatibility_weight=0.25, quality_weight=0.05)
    candidates = [small_active_candidate, large_active_candidate]

    small_results = rerank_candidates(
        candidates,
        small_profile,
        settings,
        topk=2,
        reference_date=reference_date,
    )
    large_results = rerank_candidates(
        candidates,
        large_profile,
        settings,
        topk=2,
        reference_date=reference_date,
    )

    assert [item["dog_id"] for item in small_results] == [
        "small-active",
        "large-active",
    ]
    assert [item["dog_id"] for item in large_results] == [
        "large-active",
        "small-active",
    ]

    small_by_id = {item["dog_id"]: item for item in small_results}
    large_by_id = {item["dog_id"]: item for item in large_results}
    assert (
        small_by_id["small-active"]["compatibility_score"]
        > small_by_id["large-active"]["compatibility_score"]
    )
    assert (
        large_by_id["large-active"]["compatibility_score"]
        > large_by_id["small-active"]["compatibility_score"]
    )
    assert {item["retrieval_score"] for item in small_results} == {0.52}
    assert {item["retrieval_score"] for item in large_results} == {0.52}


def test_unknown_evidence_is_neutral_and_explicitly_reported(
    small_profile: UserProfile,
    reference_date: datetime,
) -> None:
    candidate = {
        "score": 0.4,
        "desertionNo": "active-with-unknown-evidence",
        "processState": "보호중",
        "noticeEdt": "20261231",
    }

    result = rerank_candidates(
        [candidate],
        small_profile,
        ProfileRerankSettings(compatibility_weight=0.25, quality_weight=0.05),
        topk=1,
        reference_date=reference_date,
    )[0]

    assert REQUIRED_RESULT_FIELDS <= set(result)
    assert result["retrieval_score"] == pytest.approx(0.4)
    assert result["compatibility_score"] == pytest.approx(0.0)
    assert result["applicable_count"] == 8
    assert result["evaluated_count"] == 0
    assert result["evidence_coverage"] == pytest.approx(0.0)
    assert result["quality_score"] is None
    assert result["final_score"] == pytest.approx(0.4)
    assert result["matched_conditions"] == []
    assert result["caution_conditions"] == []
    assert len(result["unknown_conditions"]) == 8
    assert all(
        message in result["recommendation_reason"]
        for message in result["unknown_conditions"]
    )
    assert "확인된 조건 0/8" in result["recommendation_reason"]


def test_evidence_coverage_counts_only_conditions_supported_by_notice_evidence(
    small_profile: UserProfile,
    reference_date: datetime,
) -> None:
    candidate = {
        "score": 0.4,
        "desertionNo": "active-with-size-only",
        "weight": "7(Kg)",
        "processState": "보호중",
        "noticeEdt": "20261231",
    }

    result = rerank_candidates(
        [candidate],
        small_profile,
        ProfileRerankSettings(compatibility_weight=0.25, quality_weight=0.05),
        topk=1,
        reference_date=reference_date,
    )[0]

    assert result["applicable_count"] == 8
    assert result["evaluated_count"] == 1
    assert result["evidence_coverage"] == pytest.approx(1 / 8)
    assert result["compatibility_score"] == pytest.approx(1 / 8)
    assert result["final_score"] == pytest.approx(0.4 + 0.25 * (1 / 8))
    assert len(result["matched_conditions"]) == 1
    assert result["caution_conditions"] == []
    assert len(result["unknown_conditions"]) == 7
    assert "확인된 조건 1/8" in result["recommendation_reason"]


def test_sparse_evidence_does_not_outrank_broad_evidence(
    small_profile: UserProfile,
    small_active_candidate: dict[str, object],
    reference_date: datetime,
) -> None:
    sparse = {
        "score": 0.5,
        "desertionNo": "sparse",
        "weight": "7(Kg)",
        "processState": "보호중",
        "noticeEdt": "20261231",
    }
    broad = deepcopy(small_active_candidate)
    broad.update({"score": 0.5, "desertionNo": "broad"})

    results = rerank_candidates(
        [sparse, broad],
        small_profile,
        ProfileRerankSettings(compatibility_weight=0.25, quality_weight=0),
        topk=2,
        reference_date=reference_date,
    )

    assert [result["dog_id"] for result in results] == ["broad", "sparse"]
    assert results[0]["compatibility_score"] > results[1]["compatibility_score"]


def test_verified_weight_takes_precedence_over_vlm_size_observation(
    reference_date: datetime,
) -> None:
    dog = normalize_dog(
        {
            "desertionNo": "size-conflict",
            "weight": "25(Kg)",
            "vlm_attrs": {"body_size_hint": "small"},
        },
        reference_date=reference_date,
    )

    assert dog.weight == pytest.approx(25.0)
    assert dog.size == "large"
    assert dog.vlm_attributes["body_size_hint"] == "small"


def test_closed_and_expired_are_excluded_while_unknown_policy_is_configurable(
    small_profile: UserProfile,
    reference_date: datetime,
) -> None:
    candidates = [
        {
            "score": 0.5,
            "desertionNo": "active",
            "processState": "보호중",
            "noticeEdt": "20261231",
        },
        {
            "score": 0.5,
            "desertionNo": "adoption-waiting",
            "processState": "입양대기",
            "noticeEdt": "20261231",
        },
        {
            "score": 0.9,
            "desertionNo": "closed",
            "processState": "종료(입양)",
            "noticeEdt": "20261231",
        },
        {
            "score": 0.9,
            "desertionNo": "expired",
            "processState": "보호중",
            "noticeEdt": "20260716",
        },
        {"score": 0.5, "desertionNo": "unknown-status"},
    ]

    default_results = rerank_candidates(
        candidates,
        small_profile,
        topk=10,
        reference_date=reference_date,
    )
    default_ids = {item["dog_id"] for item in default_results}
    assert default_ids == {"active", "adoption-waiting", "unknown-status"}
    assert "closed" not in default_ids
    assert "expired" not in default_ids

    unknown_result = next(
        item for item in default_results if item["dog_id"] == "unknown-status"
    )
    assert unknown_result["meta"]["notice_status"] == "unknown"
    assert any(
        "공고 활성 상태" in message for message in unknown_result["unknown_conditions"]
    )

    conservative_results = rerank_candidates(
        candidates,
        small_profile,
        topk=10,
        include_unknown_notices=False,
        reference_date=reference_date,
    )
    assert {item["dog_id"] for item in conservative_results} == {
        "active",
        "adoption-waiting",
    }


def test_normalization_reuses_shared_notice_classifier() -> None:
    assert profile_rerank.classify_notice is shared_classify_notice


def test_notice_status_age_and_activity_are_conservative(
    reference_date: datetime,
) -> None:
    assert (
        shared_classify_notice(
            {"processState": "종료(입양대기)", "noticeEdt": "20261231"},
            reference_date=reference_date,
        )
        == "closed"
    )
    assert (
        shared_classify_notice(
            {"processState": "입양 가능", "noticeEdt": "20261231"},
            reference_date=reference_date,
        )
        == "active"
    )
    assert (
        shared_classify_notice({"active": False}, reference_date=reference_date)
        == "closed"
    )
    assert (
        shared_classify_notice({"searchable": True}, reference_date=reference_date)
        == "active"
    )
    assert (
        shared_classify_notice(
            {"process_state": "Unknown", "notice_status": "closed"},
            reference_date=reference_date,
        )
        == "closed"
    )
    assert (
        shared_classify_notice(
            {"notice_end": "Unknown", "noticeEdt": "20260716"},
            reference_date=reference_date,
        )
        == "expired"
    )
    for closed_state in ("입양완료", "반환", "종료", "자연사", "안락사"):
        assert (
            shared_classify_notice(
                {"processState": closed_state, "noticeEdt": "20261231"},
                reference_date=reference_date,
            )
            == "closed"
        )

    assert normalize_dog({"age": 3}, reference_date=reference_date).age_group == "adult"
    assert (
        normalize_dog({"age": "36개월"}, reference_date=reference_date).age_group
        == "adult"
    )
    assert (
        normalize_dog({"age": "10살"}, reference_date=reference_date).age_group
        == "senior"
    )
    assert (
        normalize_dog({"age": "성견"}, reference_date=reference_date).age_group
        == "adult"
    )
    combined_age = normalize_dog(
        {"age": "2024(60일미만)(년생)"},
        reference_date=reference_date,
    )
    assert combined_age.age == pytest.approx(2.0)
    assert combined_age.age_group == "adult"
    fallback = normalize_dog(
        {
            "age": "Unknown",
            "age_text": "3살",
            "breed": "미상",
            "kindCd": "[개] 믹스견",
        },
        reference_date=reference_date,
    )
    assert fallback.age_group == "adult"
    assert fallback.breed == "[개] 믹스견"
    invalid_numbers = normalize_dog(
        {"age": -1, "weight": "-7(Kg)", "description": "Unknown"},
        reference_date=reference_date,
    )
    assert invalid_numbers.age is None
    assert invalid_numbers.age_group == "unknown"
    assert invalid_numbers.weight is None
    assert invalid_numbers.size == "unknown"
    assert invalid_numbers.description == ""
    assert (
        normalize_dog({"age": "2027(년생)"}, reference_date=reference_date).age_group
        == "unknown"
    )
    assert (
        normalize_dog(
            {"has_child_experience": False},
            reference_date=reference_date,
        ).children_compatible
        is None
    )
    assert (
        normalize_dog(
            {"desc": "차분하고 얌전하지만 사람을 좋아함"},
            reference_date=reference_date,
        ).activity_level_hint
        is None
    )
    assert (
        normalize_dog(
            {"desc": "공고에 활동량 많음으로 명시"},
            reference_date=reference_date,
        ).activity_level_hint
        == "high"
    )


def test_settings_load_env_and_compute_candidate_depth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PROFILE_COMPATIBILITY_WEIGHT", "0.4")
    monkeypatch.setenv("PROFILE_QUALITY_WEIGHT", "0.2")
    monkeypatch.setenv("PROFILE_CANDIDATE_MULTIPLIER", "3")

    settings = ProfileRerankSettings.from_env()
    assert settings.compatibility_weight == pytest.approx(0.4)
    assert settings.quality_weight == pytest.approx(0.2)
    assert settings.candidate_multiplier == 3
    assert settings.candidate_count(2) == 6
    assert settings.candidate_count(2, maximum=5) == 5

    with pytest.raises(ValueError):
        ProfileRerankSettings(compatibility_weight=-0.1)
    with pytest.raises(ValueError):
        ProfileRerankSettings(quality_weight=-0.1)
    with pytest.raises(ValueError):
        ProfileRerankSettings(candidate_multiplier=0)
    with pytest.raises(ValueError):
        ProfileRerankSettings(compatibility_weight=float("nan"))
    with pytest.raises(ValueError):
        ProfileRerankSettings(quality_weight=float("inf"))


def test_exact_score_formula_sorting_and_reason_consistency(
    small_profile: UserProfile,
    small_active_candidate: dict[str, object],
    large_active_candidate: dict[str, object],
    reference_date: datetime,
) -> None:
    settings = ProfileRerankSettings(compatibility_weight=0.4, quality_weight=0.2)
    results = rerank_candidates(
        [large_active_candidate, small_active_candidate],
        small_profile,
        settings,
        topk=2,
        reference_date=reference_date,
    )

    assert results[0]["dog_id"] == "small-active"
    assert results[0]["final_score"] >= results[1]["final_score"]

    for result in results:
        assert REQUIRED_RESULT_FIELDS <= set(result)
        assert isinstance(result["retrieval_score"], float)
        assert isinstance(result["compatibility_score"], float)
        assert isinstance(result["applicable_count"], int)
        assert isinstance(result["evaluated_count"], int)
        assert isinstance(result["evidence_coverage"], float)
        assert 0 <= result["evaluated_count"] <= result["applicable_count"]
        expected_coverage = (
            result["evaluated_count"] / result["applicable_count"]
            if result["applicable_count"]
            else 0.0
        )
        assert result["evidence_coverage"] == pytest.approx(expected_coverage)
        assert result["quality_score"] is None or isinstance(
            result["quality_score"], float
        )
        assert isinstance(result["final_score"], float)
        assert all(
            isinstance(value, str)
            for field in (
                "matched_conditions",
                "caution_conditions",
                "unknown_conditions",
            )
            for value in result[field]
        )

        quality = result["quality_score"] or 0.0
        expected = (
            result["retrieval_score"]
            + settings.compatibility_weight * result["compatibility_score"]
            + settings.quality_weight * quality
        )
        assert result["final_score"] == pytest.approx(expected, abs=1e-6)

        all_evidence = (
            result["matched_conditions"]
            + result["caution_conditions"]
            + result["unknown_conditions"]
        )
        assert all(
            message in result["recommendation_reason"] for message in all_evidence
        )
        assert (
            f"확인된 조건 {result['evaluated_count']}/{result['applicable_count']}"
            in result["recommendation_reason"]
        )
        assert not (
            set(result["matched_conditions"])
            & set(result["caution_conditions"])
            & set(result["unknown_conditions"])
        )


def test_sorting_uses_unrounded_final_score(
    small_profile: UserProfile,
    reference_date: datetime,
) -> None:
    candidates = [
        {
            "score": 0.5000003,
            "desertionNo": "z-lower",
            "processState": "보호중",
        },
        {
            "score": 0.5000004,
            "desertionNo": "a-higher",
            "processState": "보호중",
        },
    ]

    results = rerank_candidates(
        candidates,
        small_profile,
        ProfileRerankSettings(compatibility_weight=0, quality_weight=0),
        topk=2,
        reference_date=reference_date,
    )

    assert [item["dog_id"] for item in results] == ["a-higher", "z-lower"]
    assert [item["final_score"] for item in results] == [0.5, 0.5]


def test_raw_meta_is_merged_and_candidate_fields_take_precedence(
    large_profile: UserProfile,
    reference_date: datetime,
) -> None:
    raw_meta = {
        "desertionNo": "merged-dog",
        "age_group": "puppy",
        "size": "small",
        "region": "서울",
        "photo_quality_score": 0.9,
        "housing_types": ["house"],
        "activity_level": "high",
        "max_absence_hours": 4,
        "recommended_experience": "experienced",
        "other_pets_compatible": True,
        "processState": "보호중",
        "noticeSdt": "20260701",
        "noticeEdt": "20261231",
        "noticeNo": "서울-2026-001",
        "last_verified_at": "2026-07-17T12:25:15+09:00",
        "detail_url": "https://example.org/notices/merged-dog",
        "image_url": "https://example.org/images/merged-dog.jpg",
        "care_name": "테스트 보호소",
        "care_tel": "02-0000-0000",
        "care_addr": "서울시 테스트구",
        "org_name": "테스트구청",
        "happen_place": "테스트 공원",
    }
    candidate = {
        "score": 0.5,
        "desertionNo": "merged-dog",
        "size": "large",
        "region": "부산",
        "_raw_meta": deepcopy(raw_meta),
    }

    result = rerank_candidates(
        [candidate],
        large_profile,
        ProfileRerankSettings(compatibility_weight=0.25, quality_weight=0.05),
        topk=1,
        reference_date=reference_date,
    )[0]

    assert result["dog_id"] == "merged-dog"
    assert result["retrieval_score"] == pytest.approx(0.5)
    assert result["quality_score"] == pytest.approx(0.9)
    assert result["meta"]["size"] == "large"
    assert result["meta"]["region"] == "부산"
    assert result["meta"]["notice_status"] == "active"
    assert result["meta"]["notice_start"] == "20260701"
    assert result["meta"]["notice_no"] == "서울-2026-001"
    assert result["meta"]["image_url"].endswith("merged-dog.jpg")
    assert result["meta"]["care_name"] == "테스트 보호소"
    assert result["meta"]["care_tel"] == "02-0000-0000"
    assert result["meta"]["care_addr"] == "서울시 테스트구"
    assert result["meta"]["org_name"] == "테스트구청"
    assert result["meta"]["happen_place"] == "테스트 공원"
    assert "_raw_meta" not in result["meta"]


def _post_routes(tree: ast.Module) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    routes: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call) or not isinstance(
                decorator.func, ast.Attribute
            ):
                continue
            if decorator.func.attr != "post" or not decorator.args:
                continue
            path_arg = decorator.args[0]
            if isinstance(path_arg, ast.Constant) and isinstance(path_arg.value, str):
                routes[path_arg.value] = node
    return routes


def test_main_source_keeps_profile_and_legacy_api_route_contracts() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "app" / "main.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    routes = _post_routes(tree)

    assert "/search/profile" in routes
    assert "/search/text" in routes
    assert "/recommend_with_image" in routes

    profile_route = routes["/search/profile"]
    assert profile_route.name == "search_profile"
    assert profile_route.args.args
    assert ast.unparse(profile_route.args.args[0].annotation) == "ProfileSearchRequest"

    profile_source = ast.unparse(profile_route)
    assert "PROFILE_RERANK_SETTINGS.candidate_count" in profile_source
    assert "hybrid_search" in profile_source
    assert "rerank_candidates" in profile_source
    assert "PROFILE_RESULT_DISCLAIMER" in profile_source
    for response_key in (
        "retrieval",
        "profile",
        "candidate_count",
        "count",
        "weights",
        "notice_policy",
        "disclaimer",
        "results",
    ):
        assert repr(response_key) in profile_source

    text_route = routes["/search/text"]
    image_route = routes["/recommend_with_image"]
    assert text_route.name == "search_text"
    assert ast.unparse(text_route.args.args[0].annotation) == "TextQuery"
    assert image_route.name == "recommend_with_image"
    assert isinstance(image_route, ast.AsyncFunctionDef)
    image_args = {argument.arg for argument in image_route.args.args}
    assert {"profile", "ref_image", "topk"} <= image_args
