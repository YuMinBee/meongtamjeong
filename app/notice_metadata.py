"""Lossless, conservative normalization for public animal notices.

Breed values from the public API are shelter-reported labels.  They are useful
for display and retrieval, but they are not proof of ancestry or individual
behaviour.  This module deliberately exposes that provenance and only marks a
notice as mixed when the source label says so explicitly.
"""

from __future__ import annotations

import ipaddress
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional
from urllib.parse import urlsplit


PUBLIC_NOTICE_BREED_SOURCE = "public_notice_reported"
_VERIFIED_BREED_FLAG_SOURCES = {
    "manual_verified",
    "shelter_verified",
    "genetic_verified",
}
_MIXED_BREED_MARKERS = ("\ubbf9\uc2a4", "mixed", "mix")
_BREED_PREFIX_RE = re.compile(r"^\s*\[[^\]]+\]\s*")
_LIST_SPLIT_RE = re.compile(r"[,;|\n]+")
_UNKNOWN_BREED_LABELS = {
    "",
    "-",
    "--",
    "unknown",
    "none",
    "null",
    "n/a",
    "정보 없음",
    "알 수 없음",
    "\ubbf8\uc0c1",
    "\uc815\ubcf4\uc5c6\uc74c",
}


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_http_url(value: Any, *, max_length: int = 2048) -> str:
    """Return a display/link-safe absolute HTTP(S) URL or an empty string."""

    candidate = clean_text(value)
    if (
        not candidate
        or len(candidate) > max_length
        or any(ord(character) < 32 for character in candidate)
        or any(character.isspace() for character in candidate)
        or "\\" in candidate
    ):
        return ""
    try:
        parsed = urlsplit(candidate)
        hostname = parsed.hostname
    except ValueError:
        return ""
    if (
        parsed.scheme.casefold() not in {"http", "https"}
        or not parsed.netloc
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        return ""
    normalized_host = hostname.casefold().rstrip(".")
    if normalized_host == "localhost" or normalized_host.endswith(
        (".localhost", ".local", ".internal", ".lan", ".home")
    ):
        return ""
    try:
        address = ipaddress.ip_address(normalized_host)
    except ValueError:
        address = None
    if address is not None and not address.is_global:
        return ""
    return candidate


def first_text(record: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = clean_text(record.get(key))
        if value:
            return value
    return ""


def _optional_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    lowered = clean_text(value).lower()
    if lowered in {"true", "yes", "y", "1"}:
        return True
    if lowered in {"false", "no", "n", "0"}:
        return False
    return None


def is_explicit_mixed_breed(*labels: Any) -> bool:
    """Return true only when a source label explicitly says mixed/mix."""

    combined = " ".join(
        clean_text(label).lower() for label in labels if clean_text(label)
    )
    return any(marker in combined for marker in _MIXED_BREED_MARKERS)


def _normalized_breed_name(value: Any) -> str:
    name = _BREED_PREFIX_RE.sub("", clean_text(value)).strip()
    return "" if name.lower() in _UNKNOWN_BREED_LABELS else name


def _is_unknown_breed_label(value: Any) -> bool:
    return not _normalized_breed_name(value)


def normalize_breed_fields(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize new and legacy breed fields without asserting pure ancestry."""

    raw_kind_code = first_text(record, "kindCd")
    breed_code = first_text(record, "breed_code", "breedCd")
    if breed_code.lower() in _UNKNOWN_BREED_LABELS:
        breed_code = ""
    if not breed_code and raw_kind_code.isdigit():
        breed_code = raw_kind_code

    breed_full_name = first_text(record, "breed_full_name", "kindFullNm")
    breed_name = _normalized_breed_name(first_text(record, "breed_name", "kindNm"))
    if _is_unknown_breed_label(breed_full_name):
        breed_full_name = ""
    if not breed_name and breed_full_name:
        breed_name = _normalized_breed_name(breed_full_name)

    legacy_breed = first_text(record, "breed")
    if not breed_name:
        for candidate in (raw_kind_code, legacy_breed):
            if candidate and not candidate.isdigit():
                breed_name = _normalized_breed_name(candidate)
                if breed_name:
                    break

    explicit_mixed = _optional_bool(
        record.get("mixed_breed", record.get("is_mixed", record.get("isMixed")))
    )
    declared_breed_source = first_text(record, "breed_source").lower()
    has_public_notice_breed_field = any(
        first_text(record, key) for key in ("kindCd", "kindNm", "kindFullNm", "breedCd")
    )
    reported_mixed = is_explicit_mixed_breed(
        breed_name, breed_full_name, legacy_breed, raw_kind_code
    )
    if reported_mixed:
        # Conflicting explicit metadata is not resolved by guessing.
        explicit_mixed = None if explicit_mixed is False else True
    elif (
        has_public_notice_breed_field
        or declared_breed_source == PUBLIC_NOTICE_BREED_SOURCE
        or declared_breed_source not in _VERIFIED_BREED_FLAG_SOURCES
    ):
        # A specific shelter-reported label is not proof of pure ancestry, and
        # an unprovenanced boolean is not sufficient evidence either.
        explicit_mixed = None

    normalized_legacy_breed = _normalized_breed_name(legacy_breed)
    breed = (
        breed_name
        or (normalized_legacy_breed if not normalized_legacy_breed.isdigit() else "")
        or breed_code
    )
    breed_source_label = first_text(
        record, "breed_source_label", "kindNm", "kindFullNm"
    )
    if _is_unknown_breed_label(breed_source_label):
        breed_source_label = ""
    if not breed_source_label and breed_name:
        breed_source_label = breed_name
    breed_source = first_text(record, "breed_source")
    if (
        not breed_source
        and has_public_notice_breed_field
        and (breed_name or breed_code)
    ):
        breed_source = PUBLIC_NOTICE_BREED_SOURCE

    return {
        "breed": breed or "Unknown",
        "breed_code": breed_code,
        "breed_name": breed_name,
        "breed_full_name": breed_full_name,
        "breed_source_label": breed_source_label,
        "breed_source": breed_source,
        "mixed_breed": explicit_mixed,
    }


def _iter_image_candidates(record: Mapping[str, Any]) -> Iterable[Any]:
    # Keep the legacy primary image stable; additional source images are
    # append-only metadata and must not silently replace it.
    yield record.get("image_url")

    image_urls = record.get("image_urls")
    if isinstance(image_urls, (list, tuple, set)):
        yield from image_urls
    elif image_urls:
        yield image_urls

    for key in (
        "popfile",
        "popfile1",
        "popfile2",
        "popfile3",
        "fileName",
        "thumb",
        "image",
        "img",
    ):
        yield record.get(key)

    known_keys = {
        "image_urls",
        "image_url",
        "url",
        "popfile",
        "popfile1",
        "popfile2",
        "popfile3",
        "fileName",
        "thumb",
        "image",
        "img",
    }
    for key, value in record.items():
        if key in known_keys:
            continue
        lowered = str(key).lower()
        if any(token in lowered for token in ("popfile", "image", "img", "thumb")):
            yield value


def extract_image_urls(record: Mapping[str, Any]) -> List[str]:
    urls: List[str] = []
    for candidate in _iter_image_candidates(record):
        value = normalize_http_url(candidate)
        if value and value not in urls:
            urls.append(value)
    # Some legacy records used `url` for an image, while others use it for the
    # notice page. Treat it as an image only when no image-specific field exists.
    fallback_url = normalize_http_url(record.get("url"))
    if not urls and fallback_url:
        urls.append(fallback_url)
    return urls


def split_reported_items(value: Any) -> List[str]:
    values: Iterable[Any]
    if isinstance(value, (list, tuple, set)):
        values = value
    elif value in (None, ""):
        values = ()
    else:
        values = _LIST_SPLIT_RE.split(clean_text(value))

    normalized: List[str] = []
    for item in values:
        text = clean_text(item)
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def normalize_additional_notice_fields(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Return source fields that older cache builders dropped."""

    images = extract_image_urls(record)
    health_checks = split_reported_items(record.get("health_checks"))
    if not health_checks:
        health_checks = split_reported_items(record.get("healthChk"))
    vaccinations = split_reported_items(record.get("vaccinations"))
    if not vaccinations:
        vaccinations = split_reported_items(record.get("vaccinationChk"))
    return {
        **normalize_breed_fields(record),
        "color": first_text(record, "color", "colorCd"),
        "happen_date": first_text(record, "happen_date", "happenDt"),
        "image_url": images[0] if images else "",
        "image_urls": images,
        "upstream_updated_at": first_text(record, "upstream_updated_at", "updTm"),
        "health_checks": health_checks,
        "vaccinations": vaccinations,
        "safety_health_note": first_text(record, "safety_health_note", "sfeHealth"),
        "safety_social_note": first_text(record, "safety_social_note", "sfeSoci"),
    }
