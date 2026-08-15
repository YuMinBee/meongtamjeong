"""Provider-neutral source contract for abandoned-animal notice collection.

Providers return source records and source provenance only. Filtering,
normalization, and indexing remain repository-owned so an adapter cannot
silently redefine which notices are searchable.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

import requests

from app.species import species_config


NOTICE_SOURCE_SCHEMA_VERSION = "notice-source-record.v1"
NATIONAL_ANIMAL_API_BASE = (
    "https://apis.data.go.kr/1543061/abandonmentPublicService_v2/abandonmentPublic_v2"
)
_ID_KEYS = ("notice_id", "desertionNo", "desertion_no", "noticeNo", "notice_no")
_STATUS_KEYS = (
    "source_status",
    "process_state",
    "processState",
    "status",
    "notice_status",
)
_PIPELINE_ID_KEYS = ("desertionNo", "desertion_no")
_PIPELINE_STATUS_KEYS = (
    "process_state",
    "processState",
    "status",
    "notice_status",
)


def _clean_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _first_text(record: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = _clean_text(record.get(key))
        if value:
            return value
    return ""


def _aware_iso_seconds(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("provider request timestamp must include a timezone")
    return value.isoformat(timespec="seconds")


@dataclass(frozen=True)
class NoticeFetchRequest:
    """Transport-neutral collection request passed to every provider."""

    species: str
    start: datetime
    end: datetime
    rows: int
    max_pages: int

    def __post_init__(self) -> None:
        if not _clean_text(self.species):
            raise ValueError("provider request species is required")
        _aware_iso_seconds(self.start)
        _aware_iso_seconds(self.end)
        if self.end < self.start:
            raise ValueError("provider request end must not precede start")
        if self.rows <= 0:
            raise ValueError("provider request rows must be positive")
        if self.max_pages <= 0:
            raise ValueError("provider request max_pages must be positive")

    @property
    def retrieved_at(self) -> str:
        """Use the common collection clock for deterministic provenance."""

        return _aware_iso_seconds(self.end)


@dataclass(frozen=True)
class NoticeProvenance:
    """Minimum non-secret provenance attached to one source record."""

    provider_id: str
    source_record_id: str
    retrieved_at: str
    source_uri: str = ""

    def __post_init__(self) -> None:
        if not _clean_text(self.provider_id):
            raise ValueError("provider provenance provider_id is required")
        if not _clean_text(self.source_record_id):
            raise ValueError("provider provenance source_record_id is required")
        timestamp = _clean_text(self.retrieved_at)
        try:
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(
                "provider provenance retrieved_at must be ISO-8601"
            ) from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("provider provenance retrieved_at must include a timezone")

    def as_dict(self) -> dict[str, str]:
        return {
            "schema_version": NOTICE_SOURCE_SCHEMA_VERSION,
            "provider_id": self.provider_id,
            "source_record_id": self.source_record_id,
            "retrieved_at": self.retrieved_at,
            "source_uri": self.source_uri,
        }


@dataclass(frozen=True)
class NoticeSourceRecord:
    """Canonical envelope around a provider's lossless source mapping."""

    notice_id: str
    source_status: str
    source_record: Mapping[str, Any]
    provenance: NoticeProvenance

    def __post_init__(self) -> None:
        if not _clean_text(self.notice_id):
            raise ValueError("provider record notice_id is required")
        if not isinstance(self.source_record, Mapping):
            raise TypeError("provider record source_record must be a mapping")
        if not _clean_text(self.source_status):
            object.__setattr__(self, "source_status", "unknown")

    def normalization_input(self) -> dict[str, Any]:
        """Materialize aliases consumed by the existing normalization pipeline."""

        record = copy.deepcopy(dict(self.source_record))
        if not _first_text(record, _PIPELINE_ID_KEYS):
            record["desertionNo"] = self.notice_id
        if not _first_text(record, _PIPELINE_STATUS_KEYS):
            record["process_state"] = self.source_status
        return record


@dataclass(frozen=True)
class NoticeProviderResult:
    """One provider fetch with transport diagnostics kept out of records."""

    records: tuple[NoticeSourceRecord, ...]
    pages_fetched: int
    raw_items: int
    diagnostics: Mapping[str, int] = field(default_factory=dict)


@runtime_checkable
class NoticeProvider(Protocol):
    """Small contract implemented by public API and offline import adapters."""

    provider_id: str

    def fetch(self, request: NoticeFetchRequest) -> NoticeProviderResult: ...


def validate_provider_result(
    provider: NoticeProvider,
    result: NoticeProviderResult,
) -> None:
    """Fail closed when an adapter violates the collection contract."""

    provider_id = _clean_text(getattr(provider, "provider_id", ""))
    if not provider_id:
        raise ValueError("notice provider provider_id is required")
    if not isinstance(result, NoticeProviderResult):
        raise TypeError("notice provider must return NoticeProviderResult")
    if result.pages_fetched < 0 or result.raw_items < 0:
        raise ValueError("notice provider counts must be nonnegative")
    if result.raw_items < len(result.records):
        raise ValueError("notice provider raw_items cannot be smaller than records")
    if not isinstance(result.diagnostics, Mapping):
        raise TypeError("notice provider diagnostics must be a mapping")
    for key, value in result.diagnostics.items():
        if (
            not _clean_text(key)
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            raise ValueError(
                "notice provider diagnostic values must be nonnegative integers"
            )
    for index, record in enumerate(result.records):
        if not isinstance(record, NoticeSourceRecord):
            raise TypeError(f"notice provider record {index} has an invalid type")
        if record.provenance.provider_id != provider_id:
            raise ValueError(
                f"notice provider record {index} provenance provider_id mismatch"
            )


class NationalAnimalProtectionNoticeProvider:
    """Default adapter for the existing national animal notice API."""

    provider_id = "kr-national-animal-protection-information-system"

    def __init__(
        self,
        *,
        api_key: str,
        session: requests.Session,
        api_base: str = NATIONAL_ANIMAL_API_BASE,
    ) -> None:
        self._api_key = api_key
        self._session = session
        self._api_base = api_base

    def fetch(self, request: NoticeFetchRequest) -> NoticeProviderResult:
        records: list[NoticeSourceRecord] = []
        pages_fetched = 0
        raw_items = 0
        skipped_non_object = 0
        skipped_missing_id = 0

        for page in range(1, request.max_pages + 1):
            params = {
                "serviceKey": self._api_key,
                "numOfRows": request.rows,
                "pageNo": page,
                "_type": "json",
                "upkind": species_config(request.species)["upkind"],
                "bgnde": request.start.strftime("%Y%m%d"),
                "endde": request.end.strftime("%Y%m%d"),
            }
            try:
                response = self._session.get(
                    self._api_base,
                    params=params,
                    timeout=30,
                )
                response.raise_for_status()
            except requests.RequestException as exc:
                raise RuntimeError(
                    f"public animal API request failed on page {page}: "
                    f"{exc.__class__.__name__}"
                ) from None

            data = response.json()
            body = (
                data.get("response", {}).get("body", {})
                if isinstance(data, Mapping)
                else {}
            )
            raw_page_items = body.get("items", {}) if isinstance(body, Mapping) else {}
            page_items = (
                raw_page_items.get("item", [])
                if isinstance(raw_page_items, Mapping)
                else []
            )
            if isinstance(page_items, Mapping):
                page_items = [page_items]
            if not isinstance(page_items, list) or not page_items:
                break

            pages_fetched += 1
            raw_items += len(page_items)
            for item in page_items:
                if not isinstance(item, Mapping):
                    skipped_non_object += 1
                    continue
                source_record = copy.deepcopy(dict(item))
                notice_id = _first_text(source_record, _ID_KEYS)
                if not notice_id:
                    skipped_missing_id += 1
                    continue
                source_status = _first_text(source_record, _STATUS_KEYS) or "unknown"
                records.append(
                    NoticeSourceRecord(
                        notice_id=notice_id,
                        source_status=source_status,
                        source_record=source_record,
                        provenance=NoticeProvenance(
                            provider_id=self.provider_id,
                            source_record_id=notice_id,
                            retrieved_at=request.retrieved_at,
                            # Deliberately excludes serviceKey and all query params.
                            source_uri=self._api_base,
                        ),
                    )
                )

        result = NoticeProviderResult(
            records=tuple(records),
            pages_fetched=pages_fetched,
            raw_items=raw_items,
            diagnostics={
                "skipped_non_object": skipped_non_object,
                "skipped_missing_id": skipped_missing_id,
            },
        )
        validate_provider_result(self, result)
        return result


class LocalJsonNoticeProvider:
    """Offline adapter for provider-contract JSON fixtures and exports."""

    provider_id = "local-json"

    def __init__(self, path: Path) -> None:
        self._path = path

    def fetch(self, request: NoticeFetchRequest) -> NoticeProviderResult:
        source_bytes = self._path.read_bytes()
        payload = json.loads(source_bytes.decode("utf-8-sig"))
        source_uri = "local-json:sha256:" + hashlib.sha256(source_bytes).hexdigest()
        if isinstance(payload, Mapping):
            items = payload.get("items")
        elif isinstance(payload, list):
            items = payload
        else:
            items = None
        if not isinstance(items, list):
            raise ValueError(
                "local JSON provider input must be an array or contain an items array"
            )

        records: list[NoticeSourceRecord] = []
        skipped_species = 0
        for index, item in enumerate(items):
            if not isinstance(item, Mapping):
                raise ValueError(f"local JSON provider item {index} must be an object")
            source_record = copy.deepcopy(dict(item))
            record_species = _clean_text(source_record.get("species")).lower()
            if record_species and record_species != request.species.lower():
                skipped_species += 1
                continue
            notice_id = _first_text(source_record, _ID_KEYS)
            if not notice_id:
                raise ValueError(f"local JSON provider item {index} requires notice_id")
            source_status = _first_text(source_record, _STATUS_KEYS) or "unknown"
            records.append(
                NoticeSourceRecord(
                    notice_id=notice_id,
                    source_status=source_status,
                    source_record=source_record,
                    provenance=NoticeProvenance(
                        provider_id=self.provider_id,
                        source_record_id=notice_id,
                        retrieved_at=request.retrieved_at,
                        # Content identity is reproducible without exposing a
                        # local filename that may contain a person's name.
                        source_uri=source_uri,
                    ),
                )
            )

        result = NoticeProviderResult(
            records=tuple(records),
            pages_fetched=1 if items else 0,
            raw_items=len(items),
            diagnostics={"skipped_species": skipped_species},
        )
        validate_provider_result(self, result)
        return result
