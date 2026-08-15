from __future__ import annotations

import pytest
import requests
from urllib.parse import parse_qs, urlsplit

from app.notice_metadata import (
    PUBLIC_NOTICE_BREED_SOURCE,
    extract_image_urls,
    normalize_additional_notice_fields,
    normalize_breed_fields,
    normalize_http_url,
)
from scripts.fetch_live_dogs import build_live_dog_meta
from scripts import fetch_live_dogs


def test_v2_public_notice_fields_are_preserved_without_breed_overclaim() -> None:
    raw = {
        "kindCd": "000114",
        "kindNm": "믹스견",
        "kindFullNm": "[개] 믹스견",
        "colorCd": "흰색, 갈색",
        "happenDt": "20260722",
        "popfile1": "https://example.test/one.jpg",
        "popfile2": "https://example.test/two.jpg",
        "popfile3": "https://example.test/three.jpg",
        "updTm": "2026-07-23 09:30:00",
        "healthChk": "심장사상충 검사, 기초검진",
        "vaccinationChk": "종합백신; 광견병",
        "sfeHealth": "기관 원문 건강 메모",
        "sfeSoci": "기관 원문 사회성 메모",
    }

    normalized = normalize_additional_notice_fields(raw)

    assert normalized["breed"] == "믹스견"
    assert normalized["breed_code"] == "000114"
    assert normalized["breed_name"] == "믹스견"
    assert normalized["breed_full_name"] == "[개] 믹스견"
    assert normalized["breed_source_label"] == "믹스견"
    assert normalized["breed_source"] == PUBLIC_NOTICE_BREED_SOURCE
    assert normalized["mixed_breed"] is True
    assert normalized["color"] == "흰색, 갈색"
    assert normalized["happen_date"] == "20260722"
    assert normalized["image_url"] == "https://example.test/one.jpg"
    assert normalized["image_urls"] == [
        "https://example.test/one.jpg",
        "https://example.test/two.jpg",
        "https://example.test/three.jpg",
    ]
    assert normalized["upstream_updated_at"] == "2026-07-23 09:30:00"
    assert normalized["health_checks"] == ["심장사상충 검사", "기초검진"]
    assert normalized["vaccinations"] == ["종합백신", "광견병"]
    assert normalized["safety_health_note"] == "기관 원문 건강 메모"
    assert normalized["safety_social_note"] == "기관 원문 사회성 메모"


def test_specific_reported_breed_does_not_imply_purebred() -> None:
    normalized = normalize_breed_fields(
        {
            "kindCd": "000128",
            "kindNm": "말티즈",
            "kindFullNm": "[개] 말티즈",
        }
    )

    assert normalized["breed"] == "말티즈"
    assert normalized["mixed_breed"] is None
    assert normalized["breed_source"] == PUBLIC_NOTICE_BREED_SOURCE


def test_numeric_code_is_not_mapped_to_an_unverified_breed_name() -> None:
    normalized = normalize_breed_fields({"kindCd": "000114"})

    assert normalized["breed"] == "000114"
    assert normalized["breed_code"] == "000114"
    assert normalized["breed_name"] == ""
    assert normalized["mixed_breed"] is None


def test_placeholder_breed_labels_remain_unknown() -> None:
    normalized = normalize_breed_fields(
        {
            "kindCd": "[개] 미상",
            "kindNm": "정보 없음",
            "kindFullNm": "[개] 알 수 없음",
            "breed_code": "Unknown",
        }
    )

    assert normalized["breed"] == "Unknown"
    assert normalized["breed_code"] == ""
    assert normalized["breed_name"] == ""
    assert normalized["breed_full_name"] == ""
    assert normalized["breed_source_label"] == ""
    assert normalized["breed_source"] == ""
    assert normalized["mixed_breed"] is None


def test_legacy_breed_without_provenance_is_not_relabelled_as_public() -> None:
    normalized = normalize_breed_fields({"breed": "말티즈"})

    assert normalized["breed"] == "말티즈"
    assert normalized["breed_source"] == ""
    assert normalized["mixed_breed"] is None


def test_conflicting_mixed_metadata_stays_unknown() -> None:
    normalized = normalize_breed_fields({"kindNm": "믹스견", "mixed_breed": False})

    assert normalized["mixed_breed"] is None


def test_specific_public_label_does_not_trust_unverified_mixed_flag() -> None:
    for claimed_value in (True, False):
        normalized = normalize_breed_fields(
            {
                "kindCd": "000128",
                "kindNm": "말티즈",
                "mixed_breed": claimed_value,
            }
        )

        assert normalized["mixed_breed"] is None


def test_image_urls_are_deduplicated_in_source_order() -> None:
    record = {
        "image_urls": ["https://example.test/two.jpg"],
        "image_url": "https://example.test/one.jpg",
        "popfile1": "https://example.test/one.jpg",
        "popfile2": "https://example.test/two.jpg",
    }

    assert extract_image_urls(record) == [
        "https://example.test/one.jpg",
        "https://example.test/two.jpg",
    ]


def test_explicit_image_field_wins_over_generic_notice_url() -> None:
    record = {
        "url": "https://example.test/notices/123",
        "popfile1": "https://example.test/images/123.jpg",
    }

    assert extract_image_urls(record) == [
        "https://example.test/images/123.jpg",
    ]


def test_empty_canonical_lists_do_not_hide_reported_health_fields() -> None:
    normalized = normalize_additional_notice_fields(
        {
            "health_checks": [],
            "healthChk": "기초검진",
            "vaccinations": [],
            "vaccinationChk": "종합백신",
        }
    )

    assert normalized["health_checks"] == ["기초검진"]
    assert normalized["vaccinations"] == ["종합백신"]


def test_live_cache_builder_exposes_fields_additively() -> None:
    meta = build_live_dog_meta(
        {
            "desertionNo": "123",
            "upKindCd": "417000",
            "kindCd": "000114",
            "kindNm": "믹스견",
            "colorCd": "검정",
            "popfile1": "https://example.test/dog.jpg",
            "processState": "보호중",
        }
    )

    assert meta["desertionNo"] == "123"
    assert meta["upkind"] == "417000"
    assert meta["breed_source_label"] == "믹스견"
    assert meta["mixed_breed"] is True
    assert meta["color"] == "검정"
    assert meta["image_url"] == "https://example.test/dog.jpg"
    assert meta["image_urls"] == ["https://example.test/dog.jpg"]


def test_public_api_failure_does_not_expose_service_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "secret-key-that-must-not-appear"

    class FailingSession:
        def get(self, *_args: object, **_kwargs: object) -> object:
            raise requests.exceptions.ProxyError(f"failed URL?serviceKey={secret}")

    monkeypatch.setattr(fetch_live_dogs, "build_session", FailingSession)

    with pytest.raises(RuntimeError) as exc_info:
        fetch_live_dogs.fetch_live_animals(
            api_key="not-a-real-test-api-key",
            species="dog",
            years=1,
            rows=10,
            max_pages=1,
        )

    assert secret not in str(exc_info.value)
    assert "ProxyError" in str(exc_info.value)


def test_fetch_lookback_days_overrides_legacy_years() -> None:
    assert fetch_live_dogs.resolve_lookback_days(years=3) == 1095
    assert fetch_live_dogs.resolve_lookback_days(years=3, days=60) == 60


@pytest.mark.parametrize(
    ("years", "days"),
    [(0, None), (-1, None), (3, 0), (3, -1)],
)
def test_fetch_lookback_rejects_nonpositive_windows(
    years: int,
    days: int | None,
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        fetch_live_dogs.resolve_lookback_days(years=years, days=days)


@pytest.mark.parametrize(
    "value",
    [
        "javascript:alert(1)",
        "data:text/html,unsafe",
        "//example.test/relative",
        "https://user:password@example.test/private",
        "https://example.test/a b",
        "https://example.test\\@evil.test/path",
        "https://example.test/path\nnext",
    ],
)
def test_normalize_http_url_rejects_unsafe_or_ambiguous_links(value: str) -> None:
    assert normalize_http_url(value) == ""


def test_normalize_http_url_preserves_absolute_public_links() -> None:
    value = "https://example.test/notices/123?source=open#detail"

    assert normalize_http_url(value) == value


def test_image_url_normalization_drops_local_network_and_credential_targets() -> None:
    assert extract_image_urls(
        {
            "image_urls": [
                "http://127.0.0.1/private.jpg",
                "http://192.168.0.3/private.jpg",
                "https://localhost/private.jpg",
                "https://user:secret@example.test/private.jpg",
                "https://example.test/public.jpg",
            ]
        }
    ) == ["https://example.test/public.jpg"]


def test_live_cache_fallback_detail_url_encodes_provider_notice_id() -> None:
    meta = build_live_dog_meta(
        {
            "desertionNo": "SYNTH&A=1",
            "detail_url": "javascript:alert(1)",
        }
    )
    parsed = urlsplit(meta["detail_url"])

    assert parsed.scheme == "https"
    assert parsed.hostname == "www.animal.go.kr"
    assert parse_qs(parsed.query) == {
        "desertionNo": ["SYNTH&A=1"],
        "menuNo": ["1000000055"],
    }
