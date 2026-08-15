from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.inquiry_helper import InquiryPreferences, build_inquiry_materials


META = {
    "care_name": "해봄동물보호센터",
    "notice_no": "서울-중구-2026-00123",
    "breed": "믹스견",
}


def test_build_inquiry_materials_is_deterministic() -> None:
    preferences = {
        "desired_temperaments": ["active", "calm", "active"],
        "housing_type": "apartment",
        "daily_absence_hours": 6,
        "dog_experience": "some",
        "has_children": False,
        "has_other_pets": True,
    }

    first = build_inquiry_materials(META, preferences)
    second = build_inquiry_materials(dict(reversed(list(META.items()))), preferences)

    assert first == second
    assert first["questions"] == list(dict.fromkeys(first["questions"]))


def test_selected_conditions_become_observation_questions() -> None:
    result = build_inquiry_materials(
        META,
        {
            "desired_temperaments": [
                "calm",
                "friendly",
                "active",
                "independent",
            ],
            "activity_level": "high",
            "housing_type": "apartment",
            "daily_absence_hours": 8,
            "dog_experience": "none",
            "has_children": True,
            "has_other_pets": True,
            "additional_question": "차량 이동 중 관찰된 반응이 있나요?",
        },
    )
    joined_questions = "\n".join(result["questions"])

    assert "주변 자극" in joined_questions
    assert "낯선 사람" in joined_questions
    assert "산책이나 놀이" in joined_questions
    assert "선호하는 활동량은 높은 편" in joined_questions
    assert "실제로 관찰된 활동량" in joined_questions
    assert "사람이 곁에 없는 상황" in joined_questions
    assert "약 8시간" in joined_questions
    assert "실내 보호 공간" in joined_questions
    assert "초보 보호자" in joined_questions
    assert "어린이와 같은 공간" in joined_questions
    assert "다른 개나 고양이" in joined_questions
    assert "알려주세요?" not in joined_questions
    assert "차량 이동 중" in joined_questions
    assert "해봄동물보호센터" in result["phone_script"]
    assert "서울-중구-2026-00123" in result["email_subject"]
    assert "믹스견" in result["email_body"]


@pytest.mark.parametrize(
    ("activity_level", "expected_preference"),
    [
        ("low", "낮은 편"),
        ("medium", "보통"),
        ("high", "높은 편"),
    ],
)
def test_activity_level_only_changes_observation_question(
    activity_level: str,
    expected_preference: str,
) -> None:
    result = build_inquiry_materials(META, {"activity_level": activity_level})
    activity_questions = [
        question for question in result["questions"] if "선호하는 활동량" in question
    ]

    assert len(activity_questions) == 1
    assert expected_preference in activity_questions[0]
    assert "실제로 관찰된 활동량" in activity_questions[0]
    assert "적합" not in activity_questions[0]
    assert "추천" not in activity_questions[0]


def test_output_does_not_claim_personality_or_compatibility() -> None:
    result = build_inquiry_materials(
        META,
        {"desired_temperaments": ["calm", "friendly", "independent"]},
    )
    all_text = "\n".join(
        [
            *result["questions"],
            result["phone_script"],
            result["email_subject"],
            result["email_body"],
        ]
    )

    assert "이 개는 차분" not in all_text
    assert "친화적입니다" not in all_text
    assert "잘 지냅니다" not in all_text
    assert "적합합니다" not in all_text
    assert "성격 또는 생활 적합성을 단정하지 않고" in all_text
    assert "실제로 관찰" in all_text
    assert "입양 신청이나 입양 확정을 뜻하지 않습니다" in all_text


def test_empty_preferences_still_produce_minimal_materials() -> None:
    result = build_inquiry_materials({}, None)

    assert set(result) == {
        "questions",
        "phone_script",
        "email_subject",
        "email_body",
        "disclaimer",
        "privacy_notice",
    }
    assert len(result["questions"]) == 2
    assert "현재 보호 중인지" in result["questions"][0]
    assert result["email_subject"] == "[입양 문의]"
    assert "검색 결과에서 선택한 유기견 공고" in result["email_body"]
    assert "어린이와 같은 공간" not in result["email_body"]
    assert "다른 개나 고양이" not in result["email_body"]


def test_questions_are_deduplicated_after_normalization() -> None:
    base = build_inquiry_materials(META, {})
    duplicate = base["questions"][0].rstrip("?") + "  "
    preferences = InquiryPreferences(
        desired_temperaments=["calm", "calm"],
        additional_question=duplicate,
    )

    result = build_inquiry_materials(META, preferences)
    keys = [question.rstrip("?？.!。！").casefold() for question in result["questions"]]

    assert len(keys) == len(set(keys))
    assert sum("현재 보호 중인지" in question for question in result["questions"]) == 1
    assert sum("주변 자극" in question for question in result["questions"]) == 1


def test_unknown_meta_sentinels_are_not_used_as_contact_facts() -> None:
    result = build_inquiry_materials(
        {
            "desertionNo": "D1",
            "care_name": "unknown",
            "breed": "Unknown",
        }
    )
    combined = "\n".join(
        [result["phone_script"], result["email_subject"], result["email_body"]]
    )

    assert result["email_subject"] == "[입양 문의] [D1]"
    assert "unknown" not in combined.casefold()


@pytest.mark.parametrize(
    "private_value",
    [
        "제 전화번호는 010-1234-5678입니다.",
        "해외에서는 +82 10-1234-5678로 연락해 주세요.",
        "표기는 +82 (0)10 1234 5678입니다.",
        "전각 번호 ０１０－１２３４－５６７８로 답 주세요.",
        "답장은 adopter@example.com으로 주세요.",
        "식별번호 990101-1234567도 적어둘게요.",
    ],
)
def test_additional_question_rejects_obvious_personal_identifiers(
    private_value: str,
) -> None:
    with pytest.raises(ValidationError, match="개인 연락처나 식별번호"):
        InquiryPreferences(additional_question=private_value)


def test_templates_warn_that_inquiry_is_not_an_adoption_commitment() -> None:
    result = build_inquiry_materials(META, {})

    assert result["disclaimer"] in result["phone_script"]
    assert result["disclaimer"] in result["email_body"]
    assert "불필요한 개인정보" in result["privacy_notice"]
    assert "공고가 현재 유효하다면 상담 절차" in result["email_body"]
