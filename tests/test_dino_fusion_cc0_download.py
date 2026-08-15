from __future__ import annotations

from experiments.dino_fusion.download_cc0_dogs import (
    candidate_rejection_reason,
    is_explicit_cc0,
    source_page_url,
    strip_html,
    title_series_key,
)


def extmetadata(**overrides: str) -> dict:
    values = {
        "LicenseShortName": "CC0",
        "LicenseUrl": "http://creativecommons.org/publicdomain/zero/1.0/deed.en",
        "UsageTerms": "Creative Commons Zero, Public Domain Dedication",
        **overrides,
    }
    return {key: {"value": value} for key, value in values.items()}


def page(
    *,
    title: str = "File:Friendly dog.jpg",
    mime: str = "image/jpeg",
    categories: str = "CC-Zero|Photographs of dogs",
) -> dict:
    metadata = extmetadata()
    metadata["Categories"] = {"value": categories}
    return {
        "pageid": 7,
        "title": title,
        "imageinfo": [
            {
                "mime": mime,
                "width": 1200,
                "height": 900,
                "url": "https://upload.wikimedia.org/original.jpg",
                "thumburl": "https://upload.wikimedia.org/thumb.jpg",
                "extmetadata": metadata,
            }
        ],
    }


def test_explicit_cc0_requires_name_url_and_usage_terms() -> None:
    assert is_explicit_cc0(extmetadata()) is True
    assert is_explicit_cc0(extmetadata(LicenseShortName="CC BY 4.0")) is False
    assert is_explicit_cc0(extmetadata(LicenseUrl="https://example.com")) is False
    assert is_explicit_cc0(extmetadata(UsageTerms="Public domain")) is False


def test_candidate_filter_rejects_non_jpeg_and_non_photo_titles() -> None:
    assert candidate_rejection_reason(page()) == ""
    assert candidate_rejection_reason(page(mime="image/png")) == "not_jpeg"
    assert (
        candidate_rejection_reason(page(title="File:Dog anatomy diagram.jpg"))
        == "non_photo_title"
    )
    assert (
        candidate_rejection_reason(page(categories="CC-Zero|Paintings of dogs"))
        == "non_photo_category"
    )


def test_metadata_text_and_source_page_are_safe_and_readable() -> None:
    assert strip_html('<a href="/wiki/User:A">Alice</a> &amp; Bob') == "Alice & Bob"
    assert source_page_url("File:Dog at Nørre Vorupør.jpg").startswith(
        "https://commons.wikimedia.org/wiki/File:Dog_at_N%C3%B8rre_Vorup%C3%B8r.jpg"
    )
    assert title_series_key("File:Boxer dog in Iran 04.jpg") == title_series_key(
        "File:Boxer dog in Iran 08.JPG"
    )
