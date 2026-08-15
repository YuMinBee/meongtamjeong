"""Deterministic shelter-inquiry materials.

This module turns a user's lifestyle and temperament *preferences* into
questions for a shelter.  It deliberately does not infer a dog's personality
from a photo, breed label, or sparse notice metadata.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from enum import Enum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    field_validator,
)


class DesiredTemperament(str, Enum):
    calm = "calm"
    friendly = "friendly"
    active = "active"
    independent = "independent"


class InquiryHousingType(str, Enum):
    apartment = "apartment"
    house = "house"
    other = "other"


class InquiryActivityLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class InquiryDogExperience(str, Enum):
    none = "none"
    some = "some"
    experienced = "experienced"


_TEMPERAMENT_ORDER = {
    DesiredTemperament.calm.value: 0,
    DesiredTemperament.friendly.value: 1,
    DesiredTemperament.active.value: 2,
    DesiredTemperament.independent.value: 3,
}
_EMAIL_LIKE_RE = re.compile(
    r"(?i)[a-z0-9.!#$%&'*+/=?^_`{|}~-]+"
    r"@[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?"
)
_PHONE_LIKE_RE = re.compile(
    r"(?<!\d)(?:"
    r"(?:\+?82[-.\s]?(?:\(0\)[-.\s]?)?0?\d{1,2})"
    r"|(?:0\d{1,2})"
    r")[-.\s]?\d{3,4}[-.\s]?\d{4}(?!\d)"
)
_RESIDENT_ID_LIKE_RE = re.compile(r"(?<!\d)\d{6}[-.\s]?[1-4]\d{6}(?!\d)")
INQUIRY_DISCLAIMER = (
    "이 문구는 보호 여부와 상담 가능성을 확인하기 위한 문의 초안이며, "
    "입양 신청이나 입양 확정을 뜻하지 않습니다."
)
INQUIRY_PRIVACY_NOTICE = (
    "복사한 문구를 보내기 전에 이름, 개인 전화번호, 상세 주소 등 불필요한 "
    "개인정보가 포함되지 않았는지 확인해 주세요."
)


class InquiryPreferences(BaseModel):
    """Normalized, non-identifying inputs used to prepare shelter questions."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    desired_temperaments: tuple[DesiredTemperament, ...] = ()
    housing_type: InquiryHousingType | None = None
    daily_absence_hours: float | None = Field(
        default=None,
        ge=0,
        le=24,
        allow_inf_nan=False,
    )
    activity_level: InquiryActivityLevel | None = None
    dog_experience: InquiryDogExperience | None = None
    has_children: StrictBool | None = None
    has_other_pets: StrictBool | None = None
    additional_question: str | None = Field(default=None, max_length=500)

    @field_validator("desired_temperaments", mode="before")
    @classmethod
    def normalize_temperaments(cls, value: Any) -> tuple[Any, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            values: list[Any] = [
                part.strip() for part in value.split(",") if part.strip()
            ]
        else:
            try:
                values = list(value)
            except TypeError:
                values = [value]

        normalized: list[Any] = []
        seen: set[str] = set()
        for item in values:
            raw = item.value if isinstance(item, Enum) else item
            key = str(raw).strip().lower()
            if key and key not in seen:
                seen.add(key)
                normalized.append(key)
        normalized.sort(key=lambda item: (_TEMPERAMENT_ORDER.get(item, 99), item))
        return tuple(normalized)

    @field_validator("daily_absence_hours", mode="before")
    @classmethod
    def reject_boolean_hours(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("daily_absence_hours는 숫자여야 합니다.")
        return value

    @field_validator("additional_question")
    @classmethod
    def normalize_additional_question(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = _clean_text(value, max_length=500)
        # Normalize compatibility characters before looking for identifiers so
        # full-width digits/symbols cannot bypass the server-side privacy gate.
        privacy_scan = unicodedata.normalize("NFKC", normalized)
        privacy_scan = "".join(
            character
            for character in privacy_scan
            if unicodedata.category(character) != "Cf"
        )
        if any(
            pattern.search(privacy_scan)
            for pattern in (
                _EMAIL_LIKE_RE,
                _PHONE_LIKE_RE,
                _RESIDENT_ID_LIKE_RE,
            )
        ):
            raise ValueError("추가 질문에는 개인 연락처나 식별번호를 입력하지 마세요.")
        return normalized or None


_BASE_QUESTIONS = (
    "현재 보호 중인지와 방문 상담이 가능한 시간은 언제인가요?",
    "보호 기간 동안 반복해서 관찰된 행동과 아직 확인하지 못한 행동은 무엇인가요?",
)

_TEMPERAMENT_QUESTIONS = {
    DesiredTemperament.calm: (
        "보호 공간에서 쉴 때와 주변 자극이 있을 때 각각 어떤 행동이 "
        "반복해서 관찰되나요?"
    ),
    DesiredTemperament.friendly: (
        "낯선 사람과 익숙한 보호자를 만났을 때 각각 어떤 반응이 관찰되나요?"
    ),
    DesiredTemperament.active: (
        "산책이나 놀이 시간의 활동량과 활동 후 안정되는 모습은 어떻게 관찰되나요?"
    ),
}

_ACTIVITY_LEVEL_QUESTIONS = {
    InquiryActivityLevel.low: (
        "선호하는 활동량은 낮은 편입니다. 보호 기간의 산책·놀이·휴식 "
        "상황에서 실제로 관찰된 활동량은 어느 정도였나요?"
    ),
    InquiryActivityLevel.medium: (
        "선호하는 활동량은 보통입니다. 보호 기간의 산책·놀이·휴식 "
        "상황에서 실제로 관찰된 활동량은 어느 정도였나요?"
    ),
    InquiryActivityLevel.high: (
        "선호하는 활동량은 높은 편입니다. 보호 기간의 산책·놀이·휴식 "
        "상황에서 실제로 관찰된 활동량은 어느 정도였나요?"
    ),
}

_HOUSING_QUESTIONS = {
    InquiryHousingType.apartment: (
        "실내 보호 공간에서 반복적인 짖음이나 생활 소음에 반응한 기록이 있나요?"
    ),
    InquiryHousingType.house: (
        "실내와 실외 보호 공간에서 행동 차이 또는 문·울타리 주변의 반응이 "
        "관찰된 기록이 있나요?"
    ),
    InquiryHousingType.other: (
        "현재 생활 공간에서 안전을 위해 사용 중인 설비나 특별한 관리 방법이 있나요?"
    ),
}

_EXPERIENCE_QUESTIONS = {
    InquiryDogExperience.none: (
        "초보 보호자가 알 수 있도록 목줄 착용, 접촉, 급식, 이동 과정에서 "
        "관찰된 어려움이나 필요한 관리가 있나요?"
    ),
    InquiryDogExperience.some: (
        "목줄 착용, 접촉, 급식, 이동 과정에서 반복해서 관찰된 어려움이나 "
        "필요한 관리가 있나요?"
    ),
    InquiryDogExperience.experienced: (
        "일상 관리 중 별도의 경험이나 준비가 필요한 행동이 실제로 관찰됐나요?"
    ),
}

_ALONE_OBSERVATION = (
    "사람이 곁에 없는 상황을 관찰한 기록이 있다면, 관찰한 시간과 "
    "짖음·배회·휴식 등의 행동을 알려주실 수 있나요?"
)

_CHILDREN_OBSERVATION = (
    "어린이와 같은 공간에 있었던 실제 관찰 기록이 있나요? 기록이 없다면 "
    "아직 확인되지 않은 정보인지도 알려주실 수 있나요?"
)

_OTHER_PETS_OBSERVATION = (
    "다른 개나 고양이와 같은 공간에 있었던 실제 관찰 기록이 있나요? "
    "있다면 동물 종류와 당시 상황도 알려주실 수 있나요?"
)

_SHELTER_KEYS = (
    "care_name",
    "careNm",
    "shelter_name",
    "shelter",
    "careName",
)
_NOTICE_KEYS = (
    "notice_no",
    "noticeNo",
    "desertionNo",
    "desertion_no",
)
_BREED_KEYS = (
    "breed",
    "breed_name",
    "breed_full_name",
    "kindCd",
    "breed_code",
)
_UNKNOWN_TEXTS = {
    "unknown",
    "none",
    "null",
    "n/a",
    "na",
    "-",
    "미상",
    "알 수 없음",
    "정보 없음",
}


def _clean_text(value: Any, *, max_length: int = 200) -> str:
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return ""
    cleaned = re.sub(r"\s+", " ", str(value)).strip()
    return cleaned[:max_length]


def _first_text(meta: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = _clean_text(meta.get(key))
        if value and value.casefold() not in _UNKNOWN_TEXTS:
            return value
    return ""


def _as_question(value: str) -> str:
    value = _clean_text(value, max_length=500)
    if not value:
        return ""
    if value.endswith(("?", "？")):
        return value
    return value.rstrip(".!。！") + "?"


def _question_key(value: str) -> str:
    return re.sub(r"[\s?？.!。！]+$", "", value).casefold()


def _deduplicate_questions(candidates: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        question = _as_question(candidate)
        key = _question_key(question)
        if question and key not in seen:
            seen.add(key)
            result.append(question)
    return result


def _normalize_preferences(
    preferences: InquiryPreferences | Mapping[str, Any] | None,
) -> InquiryPreferences:
    if preferences is None:
        return InquiryPreferences()
    if isinstance(preferences, InquiryPreferences):
        return preferences
    if isinstance(preferences, Mapping):
        return InquiryPreferences.model_validate(dict(preferences))
    raise TypeError("preferences는 InquiryPreferences 또는 매핑이어야 합니다.")


def _build_questions(preferences: InquiryPreferences) -> list[str]:
    candidates = list(_BASE_QUESTIONS)

    for temperament in preferences.desired_temperaments:
        question = _TEMPERAMENT_QUESTIONS.get(temperament)
        if question:
            candidates.append(question)

    if preferences.activity_level is not None:
        candidates.append(_ACTIVITY_LEVEL_QUESTIONS[preferences.activity_level])

    if preferences.housing_type is not None:
        candidates.append(_HOUSING_QUESTIONS[preferences.housing_type])

    if preferences.dog_experience is not None:
        candidates.append(_EXPERIENCE_QUESTIONS[preferences.dog_experience])

    if (
        preferences.daily_absence_hours is not None
        and preferences.daily_absence_hours > 0
    ):
        hours = f"{preferences.daily_absence_hours:g}"
        candidates.append(
            f"하루 평균 약 {hours}시간 집을 비울 예정입니다. 사람이 곁에 없는 "
            "상황을 관찰한 기록이 있다면, 관찰한 시간과 짖음·배회·휴식 등의 "
            "행동을 알려주실 수 있나요?"
        )
    elif DesiredTemperament.independent in preferences.desired_temperaments:
        candidates.append(_ALONE_OBSERVATION)

    if preferences.has_children is True:
        candidates.append(_CHILDREN_OBSERVATION)
    if preferences.has_other_pets is True:
        candidates.append(_OTHER_PETS_OBSERVATION)
    if preferences.additional_question:
        candidates.append(preferences.additional_question)

    return _deduplicate_questions(candidates)


def _notice_reference(notice_no: str, breed: str) -> str:
    details: list[str] = []
    if notice_no:
        details.append(f"공고번호 {notice_no}")
    if breed:
        details.append(f"공고 품종 표기 {breed}")
    if not details:
        return "검색 결과에서 선택한 유기견 공고"
    return ", ".join(details)


def _numbered_questions(questions: list[str]) -> str:
    return "\n".join(
        f"{index}. {question}" for index, question in enumerate(questions, start=1)
    )


def build_inquiry_materials(
    meta: Mapping[str, Any] | None,
    preferences: InquiryPreferences | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build deterministic, JSON-friendly phone and email inquiry materials."""

    safe_meta = meta if isinstance(meta, Mapping) else {}
    normalized = _normalize_preferences(preferences)

    shelter = _first_text(safe_meta, _SHELTER_KEYS)
    notice_no = _first_text(safe_meta, _NOTICE_KEYS)
    breed = _first_text(safe_meta, _BREED_KEYS)
    reference = _notice_reference(notice_no, breed)
    questions = _build_questions(normalized)
    numbered = _numbered_questions(questions)

    shelter_intro = f"{shelter} 담당자님께 " if shelter else ""
    phone_script = (
        f"안녕하세요. {shelter_intro}공고를 보고 상담 가능 여부를 문의드립니다.\n"
        f"문의 대상은 {reference}입니다.\n\n"
        f"{INQUIRY_DISCLAIMER}\n\n"
        "확인하고 싶은 내용은 다음과 같습니다.\n"
        f"{numbered}\n\n"
        "사진이나 품종 표기만으로 성격 또는 생활 적합성을 단정하지 않고, "
        "보호 기간에 실제로 관찰된 내용이 있는지만 확인하려고 합니다."
    )

    subject_parts = ["입양 문의"]
    if notice_no:
        subject_parts.append(notice_no)
    if breed:
        subject_parts.append(breed)
    email_subject = "[" + "] [".join(subject_parts) + "]"

    greeting = (
        f"안녕하세요, {shelter} 담당자님." if shelter else "안녕하세요, 담당자님."
    )
    email_body = (
        f"{greeting}\n\n"
        f"{reference}를 보고 문의드립니다.\n"
        "현재 보호 여부와 방문 상담 가능 시간을 포함해 아래 내용을 확인하고 "
        "싶습니다.\n\n"
        f"{INQUIRY_DISCLAIMER}\n\n"
        f"{numbered}\n\n"
        "사진이나 품종 표기만으로 성격 또는 생활 적합성을 단정하지 않고, "
        "보호 기간에 실제로 관찰된 내용이 있는지만 확인하려고 합니다.\n"
        "확인되지 않은 항목은 확인되지 않았다고 알려주셔도 됩니다.\n\n"
        "공고가 현재 유효하다면 상담 절차도 함께 안내 부탁드립니다.\n"
        "감사합니다."
    )

    return {
        "questions": questions,
        "phone_script": phone_script,
        "email_subject": email_subject,
        "email_body": email_body,
        "disclaimer": INQUIRY_DISCLAIMER,
        "privacy_notice": INQUIRY_PRIVACY_NOTICE,
    }
