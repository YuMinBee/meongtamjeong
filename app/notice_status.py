from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

CLOSED_STATE_TOKENS = (
    "종료",
    "입양",
    "반환",
    "자연사",
    "안락사",
    "기증",
)
HARD_CLOSED_STATE_TOKENS = tuple(
    token for token in CLOSED_STATE_TOKENS if token != "입양"
)
ACTIVE_STATE_TOKENS = (
    "보호중",
    "공고중",
    "보호",
)
UNKNOWN_TEXT_VALUES = {
    "",
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


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def first_known_text(meta: Dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = clean_text(meta.get(key))
        if value.lower() not in UNKNOWN_TEXT_VALUES:
            return value
    return ""


def parse_notice_date(value: Any) -> Optional[datetime]:
    text = clean_text(value)
    if not text:
        return None
    for fmt in ("%Y%m%d", "%Y-%m-%d", "%Y.%m.%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def notice_state(meta: Dict[str, Any]) -> str:
    return first_known_text(
        meta,
        "process_state",
        "processState",
        "status",
        "notice_status",
    )


def notice_end_date(meta: Dict[str, Any]) -> Optional[datetime]:
    return parse_notice_date(first_known_text(meta, "notice_end", "noticeEdt"))


def explicit_searchable(meta: Dict[str, Any]) -> Optional[bool]:
    for key in ("active", "is_active", "searchable"):
        if key not in meta:
            continue
        value = meta.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value)
        lowered = clean_text(value).lower()
        if lowered in {"true", "yes", "y", "1", "active", "검색가능"}:
            return True
        if lowered in {"false", "no", "n", "0", "inactive", "검색불가"}:
            return False
    return None


def classify_notice(
    meta: Dict[str, Any], reference_date: Optional[datetime] = None
) -> str:
    reference_date = reference_date or datetime.now()
    state = notice_state(meta)
    compact_state = state.replace(" ", "")
    lowered_state = compact_state.lower()
    notice_end = notice_end_date(meta)

    if lowered_state in {"closed", "inactive"}:
        return "closed"
    if lowered_state == "expired":
        return "expired"
    if state and any(token in state for token in HARD_CLOSED_STATE_TOKENS):
        return "closed"

    # Public data can use "입양대기" or "입양 가능" while the notice is open.
    adoption_open = any(
        token in compact_state for token in ("입양대기", "입양가능", "입양문의")
    )
    if "입양" in state and not adoption_open:
        return "closed"
    if notice_end and notice_end.date() < reference_date.date():
        return "expired"

    searchable = explicit_searchable(meta)
    if searchable is False:
        return "closed"
    if (
        lowered_state == "active"
        or adoption_open
        or (state and any(token in state for token in ACTIVE_STATE_TOKENS))
    ):
        return "active"
    if searchable is True:
        return "active"
    if notice_end and notice_end.date() >= reference_date.date():
        return "active"
    return "unknown"


def is_searchable_notice(
    meta: Dict[str, Any],
    reference_date: Optional[datetime] = None,
    include_unknown: bool = True,
) -> bool:
    status = classify_notice(meta, reference_date=reference_date)
    return status == "active" or (include_unknown and status == "unknown")


def notice_filter_reason(
    meta: Dict[str, Any], reference_date: Optional[datetime] = None
) -> str:
    status = classify_notice(meta, reference_date=reference_date)
    if status == "closed":
        return f"closed:{notice_state(meta)}"
    if status == "expired":
        end = notice_end_date(meta)
        return f"expired:{end:%Y%m%d}" if end else "expired"
    if status == "unknown":
        return "unknown_status"
    return "active"
