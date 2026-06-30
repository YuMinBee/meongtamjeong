import json
from typing import Any, Dict, List, Optional


COLOR_KO = {
    "white": "흰색",
    "black": "검은색",
    "brown": "갈색",
    "tan": "황갈색",
    "cream": "크림색",
    "ivory": "아이보리색",
    "gray": "회색",
    "grey": "회색",
    "yellow": "황색",
    "gold": "금색",
    "red": "붉은색",
    "spotted": "반점",
    "mixed": "혼합색",
}

FUR_LENGTH_KO = {
    "short": "단모",
    "medium": "중간 길이 털",
    "long": "장모",
    "curly": "곱슬털",
    "fluffy": "복슬복슬한 털",
    "unknown": "털 길이 불확실",
}

EAR_SHAPE_KO = {
    "upright": "선 귀",
    "floppy": "처진 귀",
    "semi_upright": "반쯤 선 귀",
    "folded": "접힌 귀",
    "hidden": "가려진 귀",
    "unknown": "귀 모양 불확실",
}

BODY_SIZE_KO = {
    "tiny": "초소형",
    "small": "소형",
    "medium": "중형",
    "large": "대형",
    "unknown": "크기 불확실",
}


ALLOWED_COAT_COLORS = {
    "white",
    "black",
    "brown",
    "tan",
    "cream",
    "ivory",
    "gray",
    "yellow",
    "gold",
    "red",
    "spotted",
    "mixed",
}
ALLOWED_FUR_LENGTHS = {"short", "medium", "long", "curly", "fluffy", "unknown"}
ALLOWED_EAR_SHAPES = {"upright", "floppy", "semi_upright", "folded", "hidden", "unknown"}
ALLOWED_BODY_SIZE_HINTS = {"tiny", "small", "medium", "large", "unknown"}

COLOR_ALIASES = {
    "grey": "gray",
    "grayish": "gray",
    "ivory_white": "ivory",
    "off_white": "ivory",
    "beige": "tan",
    "blonde": "yellow",
    "golden": "gold",
    "spot": "spotted",
    "spots": "spotted",
    "patchy": "spotted",
}
FUR_LENGTH_ALIASES = {
    "straight": "short",
    "smooth": "short",
    "wire": "medium",
    "wavy": "curly",
}
EAR_SHAPE_ALIASES = {
    "pricked": "upright",
    "erect": "upright",
    "drop": "floppy",
    "dropped": "floppy",
    "half_upright": "semi_upright",
    "semi-erect": "semi_upright",
}
BODY_SIZE_ALIASES = {
    "mini": "tiny",
    "miniature": "tiny",
    "smallish": "small",
    "mid": "medium",
    "big": "large",
}

def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [clean_text(item) for item in value if clean_text(item)]
    if isinstance(value, str):
        separators = [",", "/", "|"]
        values = [value]
        for separator in separators:
            if separator in value:
                values = value.split(separator)
                break
        return [clean_text(item) for item in values if clean_text(item)]
    return [clean_text(value)] if clean_text(value) else []


def _as_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    text = clean_text(value).lower()
    if text in {"true", "yes", "y", "1", "visible", "보임"}:
        return True
    if text in {"false", "no", "n", "0", "hidden", "안보임", "안 보임"}:
        return False
    return None


def _as_float(value: Any) -> Optional[float]:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, score))


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_choice(value: Any, allowed: set, aliases: Dict[str, str], default: str = "unknown") -> str:
    text = clean_text(value).lower().replace(" ", "_")
    if not text:
        return default
    normalized = aliases.get(text, text)
    return normalized if normalized in allowed else default


def _normalize_colors(value: Any) -> List[str]:
    colors: List[str] = []
    for item in _as_list(value):
        color = _normalize_choice(item, ALLOWED_COAT_COLORS, COLOR_ALIASES, default="")
        if color and color not in colors:
            colors.append(color)
    return colors


def _normalize_dog_count(value: Any) -> int:
    count = _as_int(value)
    if count is None or count < 1:
        return 1
    return min(count, 10)

def _label(mapping: Dict[str, str], value: str) -> str:
    text = clean_text(value)
    return mapping.get(text.lower(), text)


def extract_json_object(text: str) -> str:
    raw = clean_text(text)
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    try:
        json.loads(raw)
        return raw
    except json.JSONDecodeError:
        pass

    for start, char in enumerate(raw):
        if char != "{":
            continue
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(raw)):
            current = raw[index]
            if in_string:
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    in_string = False
                continue
            if current == '"':
                in_string = True
            elif current == "{":
                depth += 1
            elif current == "}":
                depth -= 1
                if depth == 0:
                    return raw[start : index + 1]
    raise ValueError("No JSON object found in VLM output")


def normalize_vlm_attrs(attrs: Any) -> Dict[str, Any]:
    if not isinstance(attrs, dict):
        return {}

    normalized: Dict[str, Any] = {
        "coat_color": _normalize_colors(attrs.get("coat_color")),
        "fur_length": _normalize_choice(
            attrs.get("fur_length"),
            ALLOWED_FUR_LENGTHS,
            FUR_LENGTH_ALIASES,
        ),
        "ear_shape": _normalize_choice(
            attrs.get("ear_shape"),
            ALLOWED_EAR_SHAPES,
            EAR_SHAPE_ALIASES,
        ),
        "body_size_hint": _normalize_choice(
            attrs.get("body_size_hint"),
            ALLOWED_BODY_SIZE_HINTS,
            BODY_SIZE_ALIASES,
        ),
        "face_visible": _as_bool(attrs.get("face_visible")),
        "whole_body_visible": _as_bool(attrs.get("whole_body_visible")),
        "dog_count": _normalize_dog_count(attrs.get("dog_count")),
        "photo_quality_score": _as_float(attrs.get("photo_quality_score")),
        "uncertainty": _as_list(attrs.get("uncertainty")),
        "evidence_ko": clean_text(attrs.get("evidence_ko")),
    }
    return normalized

def parse_vlm_attrs_output(text: str) -> Dict[str, Any]:
    payload = json.loads(extract_json_object(text))
    return normalize_vlm_attrs(payload)


def format_vlm_attrs_for_embedding(attrs: Any) -> str:
    normalized = normalize_vlm_attrs(attrs)
    if not normalized:
        return ""

    parts: List[str] = []
    colors = [_label(COLOR_KO, color) for color in normalized["coat_color"]]
    if colors:
        parts.append("털색 " + " ".join(colors + normalized["coat_color"]))

    fur_length = normalized["fur_length"]
    if fur_length and fur_length != "unknown":
        parts.append(f"털길이 {_label(FUR_LENGTH_KO, fur_length)} {fur_length}")

    ear_shape = normalized["ear_shape"]
    if ear_shape and ear_shape != "unknown":
        parts.append(f"귀모양 {_label(EAR_SHAPE_KO, ear_shape)} {ear_shape}")

    body_size = normalized["body_size_hint"]
    if body_size and body_size != "unknown":
        parts.append(f"몸집 {_label(BODY_SIZE_KO, body_size)} {body_size}")

    if normalized["face_visible"] is True:
        parts.append("얼굴이 잘 보이는 사진")
    elif normalized["face_visible"] is False:
        parts.append("얼굴이 잘 안 보이는 사진")

    if normalized["whole_body_visible"] is True:
        parts.append("전신이 보이는 사진")
    elif normalized["whole_body_visible"] is False:
        parts.append("전신이 일부만 보이는 사진")

    if normalized["dog_count"]:
        parts.append(f"사진 속 강아지 {normalized['dog_count']}마리")

    uncertainty = normalized["uncertainty"]
    if uncertainty:
        parts.append("불확실 " + " ".join(uncertainty))

    evidence = normalized["evidence_ko"]
    if evidence:
        parts.append("근거 " + evidence)

    return "시각 속성: " + ", ".join(parts) if parts else ""


def summarize_vlm_attrs_ko(attrs: Any) -> str:
    normalized = normalize_vlm_attrs(attrs)
    if not normalized:
        return ""

    chunks: List[str] = []
    colors = [_label(COLOR_KO, color) for color in normalized["coat_color"]]
    if colors:
        chunks.append("털색 " + "/".join(colors))
    if normalized["fur_length"] != "unknown":
        chunks.append(_label(FUR_LENGTH_KO, normalized["fur_length"]))
    if normalized["ear_shape"] != "unknown":
        chunks.append(_label(EAR_SHAPE_KO, normalized["ear_shape"]))
    if normalized["body_size_hint"] != "unknown":
        chunks.append(_label(BODY_SIZE_KO, normalized["body_size_hint"]))
    if normalized["face_visible"] is False:
        chunks.append("얼굴 확인 어려움")
    elif normalized["face_visible"] is True:
        chunks.append("얼굴 확인 가능")
    if normalized["whole_body_visible"] is False:
        chunks.append("전신 일부만 보임")
    elif normalized["whole_body_visible"] is True:
        chunks.append("전신 확인 가능")
    if normalized["photo_quality_score"] is not None:
        chunks.append(f"사진품질 {normalized['photo_quality_score']:.2f}")
    if normalized["evidence_ko"]:
        chunks.append("근거: " + normalized["evidence_ko"])
    return "; ".join(chunks)


def build_photo_advice(attrs: Any) -> List[str]:
    normalized = normalize_vlm_attrs(attrs)
    if not normalized:
        return []

    advice: List[str] = []
    score = normalized["photo_quality_score"]
    if score is not None and score < 0.55:
        advice.append("사진이 흐리거나 정보가 부족해 더 밝고 선명한 사진 재촬영을 권장합니다.")
    elif score is not None and score < 0.7:
        advice.append("검색 품질 향상을 위해 더 선명한 사진을 추가하면 좋습니다.")

    if normalized["face_visible"] is False:
        advice.append("얼굴이 잘 보이는 정면 사진을 추가하세요.")
    if normalized["whole_body_visible"] is False:
        advice.append("체형과 크기를 확인할 수 있는 전신 사진을 추가하세요.")
    if normalized["dog_count"] and normalized["dog_count"] > 1:
        advice.append("여러 마리가 함께 보이면 개체별 단독 사진을 추가하세요.")
    if normalized["uncertainty"]:
        advice.append("불확실한 속성이 있어 다른 각도의 사진을 추가하면 공고 품질이 좋아집니다.")
    return advice