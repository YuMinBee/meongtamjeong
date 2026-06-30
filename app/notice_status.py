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
ACTIVE_STATE_TOKENS = (
    "보호중",
    "공고중",
    "보호",
)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


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
    return clean_text(meta.get("process_state") or meta.get("processState") or meta.get("status"))


def notice_end_date(meta: Dict[str, Any]) -> Optional[datetime]:
    return parse_notice_date(meta.get("notice_end") or meta.get("noticeEdt"))


def classify_notice(meta: Dict[str, Any], reference_date: Optional[datetime] = None) -> str:
    reference_date = reference_date or datetime.now()
    state = notice_state(meta)
    notice_end = notice_end_date(meta)

    if state and any(token in state for token in CLOSED_STATE_TOKENS):
        return "closed"
    if notice_end and notice_end.date() < reference_date.date():
        return "expired"
    if state and any(token in state for token in ACTIVE_STATE_TOKENS):
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


def notice_filter_reason(meta: Dict[str, Any], reference_date: Optional[datetime] = None) -> str:
    status = classify_notice(meta, reference_date=reference_date)
    if status == "closed":
        return f"closed:{notice_state(meta)}"
    if status == "expired":
        end = notice_end_date(meta)
        return f"expired:{end:%Y%m%d}" if end else "expired"
    if status == "unknown":
        return "unknown_status"
    return "active"