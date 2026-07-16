from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    field_validator,
    model_validator,
)

from app.dog_attributes import normalize_vlm_attrs
from app.graph_rag import infer_age_hint, infer_region, infer_size_from_weight
from app.hybrid_rag import resolve_photo_quality_score
from app.notice_status import classify_notice


UNKNOWN = "unknown"
PROFILE_RESULT_DISCLAIMER = (
    "이 결과는 입양 적합성을 확정하지 않으며, 실제 성격과 생활 적합성은 "
    "보호소 방문 및 상담을 통해 확인해야 합니다."
)
_CONDITION_ENUM_VALUES = {
    "coat_color": {"white", "black", "brown", "tan", "cream", "gray", "spotted"},
    "fur_length": {"short", "medium", "long", "curly", "fluffy"},
    "ear_shape": {"upright", "floppy", "semi_upright", "folded"},
    "body_size_hint": {"tiny", "small", "medium", "large"},
    "sex": {"M", "F"},
    "age_hint": {"puppy", "adult", "senior"},
}
_FREE_TEXT_CONDITION_KEYS = {"personality", "keywords"}


class HousingType(str, Enum):
    apartment = "apartment"
    house = "house"
    other = "other"


class ActivityLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class DogExperience(str, Enum):
    none = "none"
    some = "some"
    experienced = "experienced"


class PreferredSize(str, Enum):
    small = "small"
    medium = "medium"
    large = "large"
    any = "any"


class PreferredAge(str, Enum):
    puppy = "puppy"
    adult = "adult"
    senior = "senior"
    any = "any"


class UserProfile(BaseModel):
    """Typed lifestyle information used only for post-retrieval reranking."""

    model_config = ConfigDict(extra="forbid")

    housing_type: HousingType
    daily_absence_hours: float = Field(ge=0, le=24, allow_inf_nan=False)
    activity_level: ActivityLevel
    dog_experience: DogExperience
    preferred_size: PreferredSize
    preferred_age: PreferredAge
    preferred_region: Optional[str] = Field(default=None, max_length=100)
    has_children: StrictBool
    has_other_pets: StrictBool

    @field_validator("daily_absence_hours", mode="before")
    @classmethod
    def reject_boolean_hours(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("daily_absence_hours는 숫자여야 합니다.")
        return value

    @field_validator("preferred_region")
    @classmethod
    def normalize_region(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class ProfileSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(default="", max_length=2000)
    conditions: Optional[Dict[str, Any]] = None
    profile: UserProfile
    topk: int = Field(default=5, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        return value.strip()

    @field_validator("conditions")
    @classmethod
    def validate_conditions(
        cls, value: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if value is None:
            return None
        allowed = {
            "coat_color",
            "fur_length",
            "ear_shape",
            "body_size_hint",
            "sex",
            "age_hint",
            "personality",
            "face_visible",
            "whole_body_visible",
            "min_photo_quality",
            "keywords",
        }
        unsupported = sorted(set(value) - allowed)
        if unsupported:
            raise ValueError("지원하지 않는 검색 조건입니다: " + ", ".join(unsupported))

        normalized: Dict[str, Any] = {}
        for key, raw_value in value.items():
            if key in _CONDITION_ENUM_VALUES or key in _FREE_TEXT_CONDITION_KEYS:
                if isinstance(raw_value, str):
                    raw_items = [raw_value]
                elif isinstance(raw_value, list):
                    raw_items = raw_value
                else:
                    raise ValueError(f"{key}는 문자열 또는 문자열 배열이어야 합니다.")
                if len(raw_items) > 20 or any(
                    not isinstance(item, str) for item in raw_items
                ):
                    raise ValueError(f"{key}는 최대 20개의 문자열만 허용합니다.")
                items = [item.strip() for item in raw_items if item.strip()]
                if any(len(item) > 100 for item in items):
                    raise ValueError(f"{key}의 각 값은 100자 이하여야 합니다.")
                if key in _CONDITION_ENUM_VALUES:
                    items = [
                        item.upper() if key == "sex" else item.lower() for item in items
                    ]
                    invalid = sorted(set(items) - _CONDITION_ENUM_VALUES[key])
                    if invalid:
                        raise ValueError(
                            f"{key}에 지원하지 않는 값이 있습니다: "
                            + ", ".join(invalid)
                        )
                normalized[key] = list(dict.fromkeys(items))
                continue

            if key in {"face_visible", "whole_body_visible"}:
                if raw_value is not None and not isinstance(raw_value, bool):
                    raise ValueError(f"{key}는 boolean 또는 null이어야 합니다.")
                normalized[key] = raw_value
                continue

            if key == "min_photo_quality":
                if raw_value is None:
                    normalized[key] = None
                    continue
                quality = _parse_float(raw_value)
                if quality is None or not 0 <= quality <= 1:
                    raise ValueError("min_photo_quality는 0 이상 1 이하여야 합니다.")
                normalized[key] = quality
        return normalized

    @model_validator(mode="after")
    def require_query_or_conditions(self) -> "ProfileSearchRequest":
        def has_value(value: Any) -> bool:
            if value is None:
                return False
            if isinstance(value, str):
                return bool(value.strip())
            if isinstance(value, (list, tuple, set)):
                return any(has_value(item) for item in value)
            # False is meaningful for visibility filters and zero is meaningful
            # for min_photo_quality.
            return True

        if not self.query and not any(
            has_value(value) for value in (self.conditions or {}).values()
        ):
            raise ValueError("query 또는 conditions 중 하나는 반드시 입력해야 합니다.")
        return self


class NormalizedDog(BaseModel):
    """Canonical, conservative view of fields that can support reranking."""

    model_config = ConfigDict(extra="forbid")

    dog_id: str
    age: Optional[float] = None
    age_group: Literal["puppy", "adult", "senior", "unknown"] = UNKNOWN
    weight: Optional[float] = None
    size: Literal["small", "medium", "large", "unknown"] = UNKNOWN
    sex: Literal["male", "female", "unknown"] = UNKNOWN
    neutered: Literal["yes", "no", "unknown"] = UNKNOWN
    region: str = UNKNOWN
    breed: str = UNKNOWN
    mixed_breed: Optional[bool] = None
    description: str = ""
    vlm_attributes: Dict[str, Any] = Field(default_factory=dict)
    photo_quality_score: Optional[float] = Field(default=None, ge=0, le=1)
    notice_status: Literal["active", "closed", "expired", "unknown"] = UNKNOWN
    notice_end: Optional[str] = None
    source_url: str = ""

    # These hints are populated only when the notice explicitly provides them.
    housing_types: List[Literal["apartment", "house", "other"]] = Field(
        default_factory=list
    )
    activity_level_hint: Optional[Literal["low", "medium", "high"]] = None
    max_absence_hours: Optional[float] = Field(default=None, ge=0)
    recommended_experience: Optional[Literal["none", "some", "experienced"]] = None
    children_compatible: Optional[bool] = None
    other_pets_compatible: Optional[bool] = None


@dataclass(frozen=True)
class ProfileRerankSettings:
    compatibility_weight: float = 0.25
    quality_weight: float = 0.05
    candidate_multiplier: int = 5

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.compatibility_weight)
            or self.compatibility_weight < 0
        ):
            raise ValueError("PROFILE_COMPATIBILITY_WEIGHT must be >= 0")
        if not math.isfinite(self.quality_weight) or self.quality_weight < 0:
            raise ValueError("PROFILE_QUALITY_WEIGHT must be >= 0")
        if self.candidate_multiplier < 1:
            raise ValueError("PROFILE_CANDIDATE_MULTIPLIER must be >= 1")

    @classmethod
    def from_env(cls) -> "ProfileRerankSettings":
        return cls(
            compatibility_weight=float(
                os.getenv("PROFILE_COMPATIBILITY_WEIGHT", "0.25")
            ),
            quality_weight=float(os.getenv("PROFILE_QUALITY_WEIGHT", "0.05")),
            candidate_multiplier=int(os.getenv("PROFILE_CANDIDATE_MULTIPLIER", "5")),
        )

    def candidate_count(self, topk: int, maximum: int = 50) -> int:
        return min(maximum, max(topk, topk * self.candidate_multiplier))


@dataclass(frozen=True)
class ConditionAssessment:
    key: str
    status: Literal["matched", "caution", "unknown"]
    message: str
    score: Optional[float]


@dataclass(frozen=True)
class CompatibilityResult:
    score: float
    evaluated_count: int
    matched_conditions: List[str]
    caution_conditions: List[str]
    unknown_conditions: List[str]


_NUMBER_RE = re.compile(r"(-?[0-9]+(?:[.,][0-9]+)?)")
_YEAR_RE = re.compile(r"(?:19|20)\d{2}")
_AGE_RE = re.compile(r"(-?[0-9]+(?:[.,][0-9]+)?)\s*(?:살|세)")
_MONTH_RE = re.compile(r"(-?[0-9]+(?:[.,][0-9]+)?)\s*개월")
_ABSENCE_RE = re.compile(
    r"혼자\s*([0-9]+(?:[.,][0-9]+)?)\s*시간[^.\n]*(?:가능|괜찮|잘\s*지냄)"
)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _has_known_value(value: Any) -> bool:
    if value in (None, "", [], {}):
        return False
    if isinstance(value, str):
        return value.strip().lower() not in {
            "-",
            "--",
            "unknown",
            "none",
            "null",
            "n/a",
            "미상",
            "알 수 없음",
            "정보 없음",
        }
    return True


def _first_text(meta: Dict[str, Any], *keys: str) -> str:
    for key in keys:
        raw_value = meta.get(key)
        if _has_known_value(raw_value):
            return _clean_text(raw_value)
    return ""


def _first_value(meta: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = meta.get(key)
        if _has_known_value(value):
            return value
    return None


def _parse_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    match = _NUMBER_RE.search(_clean_text(value).replace(",", "."))
    if not match:
        return None
    try:
        parsed = float(match.group(1).replace(",", "."))
        return parsed if math.isfinite(parsed) else None
    except ValueError:
        return None


def _parse_optional_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    lowered = _clean_text(value).lower()
    if lowered in {"true", "yes", "y", "1", "가능", "적합", "함께 가능"}:
        return True
    if lowered in {"false", "no", "n", "0", "불가", "부적합", "어려움"}:
        return False
    return None


def _parse_age_years(
    value: Any, reference_year: Optional[int] = None
) -> Optional[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        parsed = float(value)
        if not math.isfinite(parsed) or parsed < 0:
            return None
        return parsed
    text = _clean_text(value)
    if not text:
        return None
    year_match = _YEAR_RE.search(text)
    if year_match and "년생" in text:
        if reference_year is None:
            reference_year = datetime.now().year
        years = reference_year - int(year_match.group(0))
        return float(years) if years >= 0 else None
    age_match = _AGE_RE.search(text)
    if age_match:
        parsed = float(age_match.group(1).replace(",", "."))
        return parsed if parsed >= 0 else None
    month_match = _MONTH_RE.search(text)
    if month_match:
        parsed = float(month_match.group(1).replace(",", "."))
        return parsed / 12.0 if parsed >= 0 else None
    if "60일미만" in text:
        return 2.0 / 12.0
    if year_match:
        if reference_year is None:
            reference_year = datetime.now().year
        years = reference_year - int(year_match.group(0))
        return float(years) if years >= 0 else None
    return None


def _normalize_size(value: Any) -> str:
    lowered = _clean_text(value).lower()
    if lowered in {"tiny", "small", "소형", "소형견", "초소형"}:
        return "small"
    if lowered in {"medium", "중형", "중형견"}:
        return "medium"
    if lowered in {"large", "대형", "대형견"}:
        return "large"
    return UNKNOWN


def _normalize_sex(value: Any) -> str:
    lowered = _clean_text(value).lower()
    if lowered in {"m", "male", "수", "수컷", "남아"}:
        return "male"
    if lowered in {"f", "female", "암", "암컷", "여아"}:
        return "female"
    return UNKNOWN


def _normalize_neutered(value: Any) -> str:
    lowered = _clean_text(value).lower()
    if lowered in {"y", "yes", "true", "1", "중성화", "완료"}:
        return "yes"
    if lowered in {"n", "no", "false", "0", "미중성화", "안함"}:
        return "no"
    return UNKNOWN


def _normalize_activity(value: Any) -> Optional[str]:
    lowered = _clean_text(value).lower()
    aliases = {
        "low": {"low", "낮음", "낮은"},
        "medium": {"medium", "보통", "중간"},
        "high": {"high", "높음", "높은"},
    }
    for normalized, values in aliases.items():
        if lowered in values:
            return normalized
    return None


def _infer_activity_from_explicit_text(text: str) -> Optional[str]:
    lowered = text.lower()
    if any(token in lowered for token in ("활동량 많", "운동량 많")):
        return "high"
    if any(token in lowered for token in ("활동량 적", "운동량 적")):
        return "low"
    if any(token in lowered for token in ("활동량 보통", "운동량 보통")):
        return "medium"
    return None


def _normalize_experience(value: Any) -> Optional[str]:
    lowered = _clean_text(value).lower()
    if lowered in {"none", "초보", "초보 가능", "첫 반려견 가능"}:
        return "none"
    if lowered in {"some", "약간", "일부 경험", "경험 권장"}:
        return "some"
    if lowered in {"experienced", "경험자", "숙련", "경험자 필요", "경험자 권장"}:
        return "experienced"
    return None


def _infer_experience_from_explicit_text(text: str) -> Optional[str]:
    lowered = text.lower()
    if any(token in lowered for token in ("경험자 필요", "경험자 권장", "숙련 보호자")):
        return "experienced"
    if any(
        token in lowered for token in ("초보 가능", "초보자 가능", "첫 반려견 가능")
    ):
        return "none"
    return None


def _normalize_housing_types(value: Any) -> List[str]:
    values: Iterable[Any]
    if isinstance(value, (list, tuple, set)):
        values = value
    elif value in (None, ""):
        values = []
    else:
        values = re.split(r"[,/|]", _clean_text(value))
    normalized: List[str] = []
    for item in values:
        lowered = _clean_text(item).lower()
        if lowered in {"apartment", "아파트", "원룸", "오피스텔", "실내"}:
            candidate = "apartment"
        elif lowered in {"house", "주택", "단독주택", "마당"}:
            candidate = "house"
        elif lowered in {"other", "기타"}:
            candidate = "other"
        else:
            continue
        if candidate not in normalized:
            normalized.append(candidate)
    return normalized


def _explicit_children_compatibility(meta: Dict[str, Any], text: str) -> Optional[bool]:
    explicit = _parse_optional_bool(
        _first_value(meta, "children_compatible", "child_friendly")
    )
    if explicit is not None:
        return explicit
    lowered = text.lower()
    if any(
        token in lowered
        for token in ("아이와 생활 가능", "아동과 생활 가능", "어린이 친화")
    ):
        return True
    if any(
        token in lowered
        for token in ("아이 없는 가정", "아동과 생활 어려움", "어린이 동거 불가")
    ):
        return False
    return None


def _explicit_pet_compatibility(meta: Dict[str, Any], text: str) -> Optional[bool]:
    explicit = _parse_optional_bool(
        _first_value(
            meta, "other_pets_compatible", "pet_friendly", "other_animal_compatible"
        )
    )
    if explicit is not None:
        return explicit
    lowered = text.lower()
    if any(
        token in lowered
        for token in ("다른 동물과 생활 가능", "다른 반려동물과 생활 가능", "합사 가능")
    ):
        return True
    if any(
        token in lowered
        for token in ("다른 동물 없는 가정", "합사 불가", "다른 반려동물과 생활 어려움")
    ):
        return False
    return None


def normalize_dog(
    meta: Dict[str, Any],
    reference_year: Optional[int] = None,
    reference_date: Optional[datetime] = None,
) -> NormalizedDog:
    if reference_year is None and reference_date is not None:
        reference_year = reference_date.year
    raw_attrs = meta.get("vlm_attrs") if isinstance(meta.get("vlm_attrs"), dict) else {}
    if not raw_attrs and isinstance(meta.get("vlm_attributes"), dict):
        raw_attrs = meta["vlm_attributes"]
    vlm_attributes = normalize_vlm_attrs(raw_attrs) if raw_attrs else {}
    raw_age = _first_value(meta, "age", "age_text")
    age = _parse_age_years(raw_age, reference_year=reference_year)
    explicit_age_group = _clean_text(meta.get("age_group")).lower()
    if explicit_age_group in {"puppy", "adult", "senior"}:
        age_group = explicit_age_group
    elif age is not None:
        if age <= 1:
            age_group = "puppy"
        elif age >= 8:
            age_group = "senior"
        else:
            age_group = "adult"
    else:
        raw_age_text = _clean_text(raw_age)
        inferred_age_group = (
            ""
            if _YEAR_RE.search(raw_age_text)
            else infer_age_hint(raw_age, reference_year=reference_year)
        )
        if not inferred_age_group and raw_age_text.lower() in {
            "adult",
            "성견",
        }:
            inferred_age_group = "adult"
        age_group = inferred_age_group or UNKNOWN

    raw_weight = _first_value(meta, "weight", "weight_kg")
    weight = _parse_float(raw_weight)
    if weight is not None and weight <= 0:
        weight = None
    explicit_size = _normalize_size(
        _first_value(meta, "size", "size_hint", "body_size_hint")
        or vlm_attributes.get("body_size_hint")
    )
    inferred_size = _normalize_size(infer_size_from_weight(weight))
    size = explicit_size if explicit_size != UNKNOWN else inferred_size

    breed = (
        _first_text(meta, "breed_name", "kindCd", "breed", "breed_code", "breedCd")
        or UNKNOWN
    )
    explicit_mixed = _parse_optional_bool(
        _first_value(meta, "mixed_breed", "is_mixed", "isMixed")
    )
    if explicit_mixed is None:
        lowered_breed = breed.lower()
        if "믹스" in lowered_breed or "mix" in lowered_breed:
            explicit_mixed = True
        elif "순종" in lowered_breed or "purebred" in lowered_breed:
            explicit_mixed = False

    description = _first_text(
        meta,
        "merged_desc",
        "desc_full",
        "desc",
        "specialMark",
        "vlm_desc",
    )
    behavior_text = " ".join(
        part
        for part in (
            _first_text(meta, "behavior_notes"),
            _first_text(meta, "desc", "specialMark"),
        )
        if part
    )

    if not description:
        description = _first_text(meta, "description")

    explicit_activity = _normalize_activity(
        _first_value(meta, "activity_level", "activity_level_hint")
    )
    activity_hint = explicit_activity or _infer_activity_from_explicit_text(
        behavior_text
    )

    absence_hours = _parse_float(
        _first_value(meta, "max_absence_hours", "daily_absence_tolerance_hours")
    )
    if absence_hours is None:
        match = _ABSENCE_RE.search(behavior_text)
        if match:
            absence_hours = float(match.group(1).replace(",", "."))
    if absence_hours is not None and not 0 <= absence_hours <= 24:
        absence_hours = None

    experience = _normalize_experience(
        _first_value(meta, "recommended_experience", "experience_level")
    ) or _infer_experience_from_explicit_text(behavior_text)

    quality = _parse_float(meta.get("photo_quality_score"))
    if quality is None:
        quality = _parse_float(resolve_photo_quality_score(meta))
    if quality is not None:
        quality = max(0.0, min(1.0, float(quality)))

    notice_status = classify_notice(meta, reference_date=reference_date)
    explicit_notice_status = _clean_text(meta.get("notice_status")).lower()
    if notice_status == UNKNOWN and explicit_notice_status in {
        "active",
        "closed",
        "expired",
        "unknown",
    }:
        notice_status = explicit_notice_status

    notice_end = _first_text(meta, "notice_end", "noticeEdt") or None
    source_url = _first_text(meta, "detail_url", "source_url", "url")
    dog_id = _first_text(meta, "desertionNo", "desertion_no", "dog_id") or UNKNOWN

    return NormalizedDog(
        dog_id=dog_id,
        age=age,
        age_group=age_group,
        weight=weight,
        size=size,
        sex=_normalize_sex(_first_value(meta, "sex", "sexCd")),
        neutered=_normalize_neutered(
            _first_value(meta, "neuter", "neuterYn", "neutered")
        ),
        region=infer_region(meta) or _first_text(meta, "region") or UNKNOWN,
        breed=breed,
        mixed_breed=explicit_mixed,
        description=description,
        vlm_attributes=vlm_attributes,
        photo_quality_score=quality,
        notice_status=notice_status,
        notice_end=notice_end,
        source_url=source_url,
        housing_types=_normalize_housing_types(
            _first_value(meta, "housing_types", "housing_type", "suitable_housing")
        ),
        activity_level_hint=activity_hint,
        max_absence_hours=absence_hours,
        recommended_experience=experience,
        children_compatible=_explicit_children_compatibility(meta, behavior_text),
        other_pets_compatible=_explicit_pet_compatibility(meta, behavior_text),
    )


def compare_size(
    profile: UserProfile, dog: NormalizedDog
) -> Optional[ConditionAssessment]:
    desired = profile.preferred_size.value
    if desired == "any":
        return None
    labels = {"small": "소형견", "medium": "중형견", "large": "대형견"}
    if dog.size == UNKNOWN:
        return ConditionAssessment(
            "size",
            "unknown",
            "선호 크기와 일치하는지는 공고 정보만으로 확인할 수 없습니다.",
            None,
        )
    if dog.size == desired:
        return ConditionAssessment(
            "size", "matched", f"선호한 {labels[desired]} 조건과 일치합니다.", 1.0
        )
    return ConditionAssessment(
        "size",
        "caution",
        f"공고상 크기는 {labels[dog.size]}으로, 선호한 {labels[desired]}과 다릅니다.",
        0.0,
    )


def compare_age(
    profile: UserProfile, dog: NormalizedDog
) -> Optional[ConditionAssessment]:
    desired = profile.preferred_age.value
    if desired == "any":
        return None
    labels = {"puppy": "어린 개체", "adult": "성견", "senior": "노령견"}
    if dog.age_group == UNKNOWN:
        return ConditionAssessment(
            "age",
            "unknown",
            "선호 연령대와 일치하는지는 공고 정보만으로 확인할 수 없습니다.",
            None,
        )
    if dog.age_group == desired:
        return ConditionAssessment(
            "age", "matched", f"선호한 {labels[desired]} 조건과 일치합니다.", 1.0
        )
    return ConditionAssessment(
        "age",
        "caution",
        f"공고상 연령대는 {labels[dog.age_group]}으로, 선호한 {labels[desired]}과 다릅니다.",
        0.0,
    )


def _region_key(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()


def compare_region(
    profile: UserProfile, dog: NormalizedDog
) -> Optional[ConditionAssessment]:
    desired = profile.preferred_region
    if not desired:
        return None
    if dog.region == UNKNOWN:
        return ConditionAssessment(
            "region",
            "unknown",
            "희망 지역과의 일치 여부는 공고 정보만으로 확인할 수 없습니다.",
            None,
        )
    desired_key = _region_key(desired)
    actual_key = _region_key(dog.region)
    if desired_key in actual_key or actual_key in desired_key:
        return ConditionAssessment("region", "matched", "희망 지역과 일치합니다.", 1.0)
    return ConditionAssessment(
        "region",
        "caution",
        f"공고 지역은 {dog.region}으로 희망 지역 {desired}과 다릅니다.",
        0.0,
    )


def compare_housing_type(
    profile: UserProfile, dog: NormalizedDog
) -> ConditionAssessment:
    desired = profile.housing_type.value
    if not dog.housing_types:
        return ConditionAssessment(
            "housing_type",
            "unknown",
            "주거 형태와의 생활 적합성은 공고 정보만으로 확인할 수 없습니다.",
            None,
        )
    if desired in dog.housing_types:
        return ConditionAssessment(
            "housing_type", "matched", "입력한 주거 형태와 공고 조건이 일치합니다.", 1.0
        )
    return ConditionAssessment(
        "housing_type",
        "caution",
        "공고에 명시된 주거 조건과 입력한 주거 형태가 다릅니다.",
        0.0,
    )


def compare_activity_hint(
    profile: UserProfile, dog: NormalizedDog
) -> ConditionAssessment:
    desired = profile.activity_level.value
    if dog.activity_level_hint is None:
        return ConditionAssessment(
            "activity_level",
            "unknown",
            "원하는 활동 수준과 맞는지는 공고 정보만으로 확인할 수 없습니다.",
            None,
        )
    if dog.activity_level_hint == desired:
        return ConditionAssessment(
            "activity_level",
            "matched",
            "공고에 명시된 활동 수준과 선호가 일치합니다.",
            1.0,
        )
    return ConditionAssessment(
        "activity_level", "caution", "공고에 명시된 활동 수준과 선호가 다릅니다.", 0.0
    )


def compare_absence_hours(
    profile: UserProfile, dog: NormalizedDog
) -> ConditionAssessment:
    if dog.max_absence_hours is None:
        return ConditionAssessment(
            "daily_absence_hours",
            "unknown",
            "하루 부재 시간 동안의 생활 적합성은 공고 정보만으로 확인할 수 없습니다.",
            None,
        )
    if profile.daily_absence_hours <= dog.max_absence_hours:
        return ConditionAssessment(
            "daily_absence_hours",
            "matched",
            "공고에 명시된 혼자 지낼 수 있는 시간 범위와 맞습니다.",
            1.0,
        )
    return ConditionAssessment(
        "daily_absence_hours",
        "caution",
        "입력한 부재 시간이 공고에 명시된 혼자 지낼 수 있는 시간보다 깁니다.",
        0.0,
    )


def compare_experience_level(
    profile: UserProfile, dog: NormalizedDog
) -> ConditionAssessment:
    if dog.recommended_experience is None:
        return ConditionAssessment(
            "dog_experience",
            "unknown",
            "양육 경험 수준과의 적합성은 공고 정보만으로 확인할 수 없습니다.",
            None,
        )
    rank = {"none": 0, "some": 1, "experienced": 2}
    if rank[profile.dog_experience.value] >= rank[dog.recommended_experience]:
        return ConditionAssessment(
            "dog_experience",
            "matched",
            "공고에 명시된 양육 경험 조건을 충족합니다.",
            1.0,
        )
    return ConditionAssessment(
        "dog_experience",
        "caution",
        "공고에서는 현재 입력보다 높은 양육 경험 수준을 권장합니다.",
        0.0,
    )


def compare_children(
    profile: UserProfile, dog: NormalizedDog
) -> Optional[ConditionAssessment]:
    if not profile.has_children:
        return None
    if dog.children_compatible is None:
        return ConditionAssessment(
            "has_children",
            "unknown",
            "아동과의 생활 가능 여부는 보호소 상담이 필요합니다.",
            None,
        )
    if dog.children_compatible:
        return ConditionAssessment(
            "has_children",
            "matched",
            "공고에 아동과 생활 가능하다고 명시되어 있습니다.",
            1.0,
        )
    return ConditionAssessment(
        "has_children",
        "caution",
        "공고에 아동과의 생활이 어렵다고 명시되어 있습니다.",
        0.0,
    )


def compare_other_pets(
    profile: UserProfile, dog: NormalizedDog
) -> Optional[ConditionAssessment]:
    if not profile.has_other_pets:
        return None
    if dog.other_pets_compatible is None:
        return ConditionAssessment(
            "has_other_pets",
            "unknown",
            "다른 반려동물과의 생활 가능 여부는 보호소 상담이 필요합니다.",
            None,
        )
    if dog.other_pets_compatible:
        return ConditionAssessment(
            "has_other_pets",
            "matched",
            "공고에 다른 반려동물과 생활 가능하다고 명시되어 있습니다.",
            1.0,
        )
    return ConditionAssessment(
        "has_other_pets",
        "caution",
        "공고에 다른 반려동물과의 생활이 어렵다고 명시되어 있습니다.",
        0.0,
    )


def calculate_compatibility(
    profile: UserProfile, dog: NormalizedDog
) -> CompatibilityResult:
    assessments: Sequence[Optional[ConditionAssessment]] = (
        compare_size(profile, dog),
        compare_age(profile, dog),
        compare_region(profile, dog),
        compare_housing_type(profile, dog),
        compare_activity_hint(profile, dog),
        compare_absence_hours(profile, dog),
        compare_experience_level(profile, dog),
        compare_children(profile, dog),
        compare_other_pets(profile, dog),
    )
    present = [assessment for assessment in assessments if assessment is not None]
    known_scores = [
        assessment.score for assessment in present if assessment.score is not None
    ]
    # In an additive formula, zero is the neutral contribution when every
    # requested condition is unknown. Unknown items never enter the divisor.
    score = sum(known_scores) / len(known_scores) if known_scores else 0.0
    return CompatibilityResult(
        score=max(0.0, min(1.0, score)),
        evaluated_count=len(known_scores),
        matched_conditions=[a.message for a in present if a.status == "matched"],
        caution_conditions=[a.message for a in present if a.status == "caution"],
        unknown_conditions=[a.message for a in present if a.status == "unknown"],
    )


def build_recommendation_reason(compatibility: CompatibilityResult) -> str:
    messages = (
        compatibility.matched_conditions
        + compatibility.caution_conditions
        + compatibility.unknown_conditions
    )
    if not messages:
        return "기존 검색 상위 후보입니다. 실제 생활 적합성은 보호소에서 확인해 주세요."
    return " ".join(messages)


def _clamp_score(value: Any) -> float:
    parsed = _parse_float(value)
    if parsed is None:
        return 0.0
    return max(0.0, min(1.0, parsed))


def rerank_candidates(
    candidates: Sequence[Dict[str, Any]],
    profile: UserProfile,
    settings: Optional[ProfileRerankSettings] = None,
    *,
    topk: int = 5,
    include_unknown_notices: bool = True,
    reference_date: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    settings = settings or ProfileRerankSettings.from_env()
    reranked: List[tuple[float, float, str, Dict[str, Any]]] = []

    for candidate in candidates:
        raw_meta = (
            candidate.get("_raw_meta")
            if isinstance(candidate.get("_raw_meta"), dict)
            else {}
        )
        merged_meta = dict(raw_meta)
        merged_meta.update(
            {key: value for key, value in candidate.items() if key != "_raw_meta"}
        )
        dog = normalize_dog(merged_meta, reference_date=reference_date)
        if dog.notice_status in {"closed", "expired"}:
            continue
        if dog.notice_status == UNKNOWN and not include_unknown_notices:
            continue

        retrieval_score = _clamp_score(candidate.get("score"))
        compatibility = calculate_compatibility(profile, dog)
        if dog.notice_status == UNKNOWN:
            compatibility = CompatibilityResult(
                score=compatibility.score,
                evaluated_count=compatibility.evaluated_count,
                matched_conditions=compatibility.matched_conditions,
                caution_conditions=compatibility.caution_conditions,
                unknown_conditions=compatibility.unknown_conditions
                + ["공고 활성 상태는 보호소 또는 원문 공고에서 다시 확인해야 합니다."],
            )
        quality_score = dog.photo_quality_score

        final_score = (
            retrieval_score + settings.compatibility_weight * compatibility.score
        )
        if quality_score is not None:
            final_score += settings.quality_weight * quality_score

        normalized_meta = dog.model_dump()
        result = {
            "dog_id": dog.dog_id,
            "source_url": dog.source_url,
            "retrieval_score": round(retrieval_score, 6),
            "compatibility_score": round(compatibility.score, 6),
            "quality_score": round(quality_score, 6)
            if quality_score is not None
            else None,
            "final_score": round(final_score, 6),
            "matched_conditions": compatibility.matched_conditions,
            "caution_conditions": compatibility.caution_conditions,
            "unknown_conditions": compatibility.unknown_conditions,
            "recommendation_reason": build_recommendation_reason(compatibility),
            "meta": normalized_meta,
        }
        reranked.append(
            (
                final_score,
                retrieval_score,
                dog.dog_id,
                result,
            )
        )

    reranked.sort(key=lambda item: item[:3], reverse=True)
    return [item[3] for item in reranked[: max(1, topk)]]
