from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from app.notice_provider import (
    LocalJsonNoticeProvider,
    NationalAnimalProtectionNoticeProvider,
    NoticeFetchRequest,
    NoticeProvider,
    validate_provider_result,
)
from scripts import fetch_live_dogs
from scripts.sync_active_index import (
    select_active_fresh_records,
    validate_fresh_payload_contract,
)


FIXED_NOW = datetime(2026, 7, 26, 12, 0, tzinfo=timezone(timedelta(hours=9)))


def request() -> NoticeFetchRequest:
    return NoticeFetchRequest(
        species="dog",
        start=FIXED_NOW - timedelta(days=30),
        end=FIXED_NOW,
        rows=100,
        max_pages=2,
    )


class FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class FakeSession:
    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self._payloads = payloads
        self.calls: list[dict[str, Any]] = []

    def get(
        self,
        url: str,
        *,
        params: dict[str, Any],
        timeout: int,
    ) -> FakeResponse:
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        return FakeResponse(self._payloads[len(self.calls) - 1])


def api_payload(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {"response": {"body": {"items": {"item": items}}}}


def test_national_provider_satisfies_contract_without_leaking_key() -> None:
    session = FakeSession(
        [
            api_payload(
                [
                    {
                        "desertionNo": "PUBLIC-SYNTH-001",
                        "processState": "보호중",
                        "popfile1": "https://example.invalid/public-synth.jpg",
                    }
                ]
            ),
            api_payload([]),
        ]
    )
    secret = "test-service-key-not-for-output"
    provider = NationalAnimalProtectionNoticeProvider(
        api_key=secret,
        session=session,  # type: ignore[arg-type]
    )

    assert isinstance(provider, NoticeProvider)
    result = provider.fetch(request())
    validate_provider_result(provider, result)

    assert result.pages_fetched == 1
    assert result.raw_items == 1
    assert len(result.records) == 1
    record = result.records[0]
    assert record.notice_id == "PUBLIC-SYNTH-001"
    assert record.source_status == "보호중"
    assert record.provenance.provider_id == provider.provider_id
    assert record.provenance.retrieved_at == FIXED_NOW.isoformat(timespec="seconds")
    assert secret not in json.dumps(
        record.provenance.as_dict(),
        ensure_ascii=False,
    )
    assert "?" not in record.provenance.source_uri

    first_call = session.calls[0]
    assert first_call["params"]["serviceKey"] == secret
    assert first_call["params"]["upkind"] == "417000"
    assert first_call["params"]["bgnde"] == "20260626"
    assert first_call["params"]["endde"] == "20260726"
    assert first_call["timeout"] == 30


def write_local_payload(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "notice_id": "LOCAL-SYNTH-001",
                        "source_status": "active",
                        "species": "dog",
                        "notice_no": "SYNTH-001",
                        "breed": "합성 믹스견",
                        "mixed_breed": True,
                        "image_urls": ["https://example.invalid/local-synth-dog.jpg"],
                        "desc": "테스트 전용 합성 공고",
                        "notice_start": "20260701",
                        "notice_end": "20991231",
                        "detail_url": (
                            "https://example.invalid/notices/LOCAL-SYNTH-001"
                        ),
                    },
                    {
                        "notice_id": "LOCAL-SYNTH-CLOSED-001",
                        "source_status": "closed",
                        "species": "dog",
                        "image_url": ("https://example.invalid/local-synth-closed.jpg"),
                        "notice_end": "20260720",
                    },
                    {
                        "notice_id": "LOCAL-SYNTH-CAT-001",
                        "source_status": "active",
                        "species": "cat",
                        "image_url": "https://example.invalid/local-synth-cat.jpg",
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_local_json_provider_runs_through_real_collection_path(
    tmp_path: Path,
) -> None:
    source = tmp_path / "synthetic-notices.json"
    write_local_payload(source)
    provider = LocalJsonNoticeProvider(source)

    result = provider.fetch(request())
    validate_provider_result(provider, result)
    assert result.raw_items == 3
    assert result.diagnostics == {"skipped_species": 1}
    assert [record.notice_id for record in result.records] == [
        "LOCAL-SYNTH-001",
        "LOCAL-SYNTH-CLOSED-001",
    ]

    payload = fetch_live_dogs.fetch_live_animals(
        api_key="",
        species="dog",
        years=1,
        rows=100,
        max_pages=2,
        provider=provider,
        now=FIXED_NOW,
    )

    assert payload["provider"] == "local-json"
    assert payload["stats"] == {
        "pages_fetched": 1,
        "raw_items": 3,
        "provider_records": 2,
        "kept_items": 1,
        "skipped_closed": 1,
        "skipped_no_image": 0,
        "skipped_duplicate": 0,
    }
    assert payload["provider_diagnostics"] == {"skipped_species": 1}
    item = payload["items"][0]
    assert item["desertionNo"] == "LOCAL-SYNTH-001"
    assert item["process_state"] == "active"
    assert item["image_url"] == "https://example.invalid/local-synth-dog.jpg"
    assert item["detail_url"] == ("https://example.invalid/notices/LOCAL-SYNTH-001")
    assert item["source_status"] == "active"
    provenance = item["source_provenance"]
    assert provenance == {
        "schema_version": "notice-source-record.v1",
        "provider_id": "local-json",
        "source_record_id": "LOCAL-SYNTH-001",
        "retrieved_at": FIXED_NOW.isoformat(timespec="seconds"),
        "source_uri": provenance["source_uri"],
    }
    assert provenance["source_uri"] == (
        "local-json:sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()
    )
    assert source.name not in provenance["source_uri"]
    validate_fresh_payload_contract(payload, payload["items"])
    active, _ = select_active_fresh_records(
        payload["items"],
        reference_date=FIXED_NOW,
    )
    assert list(active) == ["LOCAL-SYNTH-001"]


def test_local_json_provider_rejects_record_without_notice_id(
    tmp_path: Path,
) -> None:
    source = tmp_path / "invalid.json"
    source.write_text(
        json.dumps({"items": [{"source_status": "active"}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires notice_id"):
        LocalJsonNoticeProvider(source).fetch(request())


def test_fetch_cli_preserves_default_and_validates_local_input() -> None:
    defaults = fetch_live_dogs.parse_args([])
    assert defaults.provider == "national-api"
    assert defaults.input_json is None

    local = fetch_live_dogs.parse_args(
        [
            "--provider",
            "local-json",
            "--input-json",
            "data/examples/notice-provider.synthetic.json",
        ]
    )
    assert local.provider == "local-json"
    assert local.input_json == Path("data/examples/notice-provider.synthetic.json")

    with pytest.raises(SystemExit):
        fetch_live_dogs.parse_args(["--provider", "local-json"])


def test_tracked_local_example_is_synthetic_and_contract_compatible() -> None:
    path = Path("data/examples/notice-provider.synthetic.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    serialized = json.dumps(payload, ensure_ascii=False)

    assert "example.invalid" in serialized
    assert "care_tel" not in serialized
    assert "care_addr" not in serialized
    result = LocalJsonNoticeProvider(path).fetch(request())
    validate_provider_result(LocalJsonNoticeProvider(path), result)
    assert len(result.records) == 1
