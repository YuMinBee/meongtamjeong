from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

import pytest

from scripts import reuse_image_enrichment as reuse


@pytest.fixture
def workdir() -> Iterator[Path]:
    runtime_root = Path(__file__).resolve().parent / ".runtime"
    directory = runtime_root / f"reuse-image-{uuid4().hex}"
    directory.mkdir(parents=True)
    try:
        yield directory
    finally:
        shutil.rmtree(directory)
        try:
            runtime_root.rmdir()
        except OSError:
            pass


def test_reuses_only_image_attrs_and_preserves_fresh_wrapper_fields() -> None:
    fresh_payload = {
        "fetched_at": "2026-07-25T23:35:45+09:00",
        "stats": {"kept_items": 2, "skipped_closed": 9},
        "source_note": "must stay",
        "items": [
            {
                "desertionNo": "A",
                "desc": "fresh description",
                "process_state": "보호중",
                "last_verified_at": "fresh timestamp",
                "image_url": "https://fresh.test/a.jpg",
                "detail_url": "https://fresh.test/a",
                "vlm_attrs": {"fresh": "preserved"},
            },
            {
                "desertionNo": "B",
                "desc": "no reusable attrs",
            },
        ],
    }
    existing = [
        {
            "desertionNo": "A",
            "desc": "stale description",
            "process_state": "종료",
            "last_verified_at": "stale timestamp",
            "image_url": "https://fresh.test/a.jpg",
            "detail_url": "https://stale.test/a",
            "image_attrs": {"photo_quality_score": 0.8, "target_detected": False},
            "vlm_attrs": {"stale": "must not copy"},
            "vlm_desc": "must not copy",
        }
    ]
    originals = deepcopy((fresh_payload, existing))

    records, stats = reuse.reuse_image_attrs(fresh_payload["items"], existing)
    output = reuse.build_output_payload(fresh_payload, records, stats)

    assert output["fetched_at"] == fresh_payload["fetched_at"]
    assert output["stats"] == fresh_payload["stats"]
    assert output["source_note"] == "must stay"
    assert output["items"][0] == {
        **fresh_payload["items"][0],
        "image_attrs": {
            "photo_quality_score": 0.8,
            "target_detected": False,
        },
    }
    assert "vlm_desc" not in output["items"][0]
    assert output["items"][0]["vlm_attrs"] == {"fresh": "preserved"}
    assert output["items"][1] == fresh_payload["items"][1]
    assert output[reuse.REUSE_STATS_KEY] == stats
    assert stats["matched_fresh_records"] == 1
    assert stats["enriched_fresh_records"] == 1
    assert stats["unmatched_fresh_records"] == 1
    assert stats["image_url_mismatch_unique_ids"] == 0
    assert (fresh_payload, existing) == originals


def test_duplicate_vector_attrs_merge_deterministically_from_most_complete() -> None:
    richest = {
        "type": "image",
        "desertionNo": "A",
        "image_url": "https://example.test/a.jpg",
        "image_attrs": {
            "detector": "preferred",
            "confidence": 0.9,
            "sharpness": 1.0,
            "target_detected": True,
            "blank": "",
            "nested": {"primary": 1, "conflict": "preferred"},
        },
    }
    supplement = {
        "type": "crop_image",
        "desertionNo": "A",
        "image_url": "https://example.test/a.jpg",
        "image_attrs": {
            "detector": "other",
            "blank": "filled",
            "zero_is_evidence": 0,
            "false_is_evidence": False,
            "nested": {"secondary": 2, "conflict": "other"},
        },
    }
    empty = {
        "type": "text",
        "desertionNo": "A",
        "image_url": "https://example.test/a.jpg",
        "image_attrs": {"detector": "unknown", "blank": None},
    }

    forward, forward_stats = reuse.build_image_attrs_index([richest, supplement, empty])
    reverse, reverse_stats = reuse.build_image_attrs_index([empty, supplement, richest])

    assert forward == reverse
    assert forward_stats == reverse_stats
    assert forward["A"].source_image_url == "https://example.test/a.jpg"
    assert forward["A"].image_attrs == {
        "blank": "filled",
        "confidence": 0.9,
        "detector": "preferred",
        "false_is_evidence": False,
        "nested": {
            "conflict": "preferred",
            "primary": 1,
            "secondary": 2,
        },
        "sharpness": 1.0,
        "target_detected": True,
        "zero_is_evidence": 0,
    }
    assert forward_stats["existing_duplicate_id_rows"] == 2
    assert forward_stats["existing_ids_merged_from_multiple_rows"] == 1


def test_fresh_image_attrs_win_conflicts_and_only_empty_values_are_filled() -> None:
    fresh = [
        {
            "desertionNo": "A",
            "image_url": "https://example.test/a.jpg",
            "image_attrs": {
                "detector": "new",
                "quality": "",
                "nested": {"fresh": 1, "missing": None},
            },
        }
    ]
    existing = [
        {
            "desertionNo": "A",
            "image_url": "https://example.test/a.jpg",
            "image_attrs": {
                "detector": "old",
                "quality": 0.75,
                "nested": {"fresh": 999, "missing": 2, "old": 3},
            },
        }
    ]
    originals = deepcopy((fresh, existing))

    output, stats = reuse.reuse_image_attrs(fresh, existing)

    assert output[0]["image_attrs"] == {
        "detector": "new",
        "nested": {"fresh": 1, "missing": 2, "old": 3},
        "quality": 0.75,
    }
    assert stats["filled_attribute_values"] == 3
    assert (fresh, existing) == originals


def test_non_object_fresh_image_attrs_are_not_overwritten() -> None:
    fresh = [
        {
            "desertionNo": "A",
            "image_url": "https://example.test/a.jpg",
            "image_attrs": "fresh opaque value",
        }
    ]
    existing = [
        {
            "desertionNo": "A",
            "image_url": "https://example.test/a.jpg",
            "image_attrs": {"score": 0.8},
        }
    ]

    output, stats = reuse.reuse_image_attrs(fresh, existing)

    assert output == fresh
    assert stats["matched_fresh_records"] == 1
    assert stats["enriched_fresh_records"] == 0
    assert stats["fresh_records_with_protected_non_object_image_attrs"] == 1


def test_list_cache_gets_items_wrapper_and_reports_missing_and_duplicate_ids() -> None:
    fresh = [
        {"desertionNo": "A", "image_url": "https://example.test/a.jpg"},
        {"desertion_no": " A ", "image_url": "https://example.test/a.jpg"},
        {"desertionNo": ""},
    ]
    existing = [
        {
            "desertionNo": "A",
            "image_url": "https://example.test/a.jpg",
            "image_attrs": {"score": 0.4},
        },
        {
            "desertionNo": "A",
            "image_url": "https://example.test/a.jpg",
            "image_attrs": {"band": "medium"},
        },
        {"type": "missing-id", "image_attrs": {"ignored": True}},
    ]

    records, stats = reuse.reuse_image_attrs(fresh, existing)
    output = reuse.build_output_payload(fresh, records, stats)

    assert list(output) == ["items", reuse.REUSE_STATS_KEY]
    assert [item.get("image_attrs") for item in output["items"]] == [
        {"band": "medium", "score": 0.4},
        {"band": "medium", "score": 0.4},
        None,
    ]
    assert stats["fresh_unique_ids"] == 1
    assert stats["fresh_missing_id_records"] == 1
    assert stats["fresh_duplicate_id_records"] == 1
    assert stats["existing_unique_ids"] == 1
    assert stats["existing_missing_id_rows"] == 1
    assert stats["existing_duplicate_id_rows"] == 1
    assert stats["matched_fresh_records"] == 2
    assert stats["matched_unique_ids"] == 1


def test_url_mismatch_and_existing_url_conflict_skip_reuse() -> None:
    fresh = [
        {"desertionNo": "MISMATCH", "image_url": "https://new.test/dog.jpg"},
        {"desertionNo": "CONFLICT", "image_url": "https://same.test/dog.jpg"},
    ]
    existing = [
        {
            "desertionNo": "MISMATCH",
            "image_url": "https://old.test/dog.jpg",
            "image_attrs": {"score": 0.8},
        },
        {
            "desertionNo": "CONFLICT",
            "image_url": "https://same.test/dog.jpg",
            "image_attrs": {"score": 0.7},
        },
        {
            "desertionNo": "CONFLICT",
            "image_url": "https://other.test/dog.jpg",
            "image_attrs": {"band": "high"},
        },
    ]

    forward, forward_stats = reuse.reuse_image_attrs(fresh, existing)
    reverse, reverse_stats = reuse.reuse_image_attrs(fresh, list(reversed(existing)))

    assert forward == fresh
    assert reverse == fresh
    assert forward_stats == reverse_stats
    assert forward_stats["image_url_mismatch_fresh_records"] == 1
    assert forward_stats["image_url_mismatch_unique_ids"] == 1
    assert forward_stats["existing_ids_with_conflicting_source_image_urls"] == 1
    assert forward_stats["existing_ids_with_reusable_image_attrs"] == 1


def test_cli_writes_atomic_utf8_without_bom_and_rejects_input_overwrite(
    workdir: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fresh_path = workdir / "fresh.json"
    metas_path = workdir / "metas.json"
    output_path = workdir / "output.json"
    fresh_bytes = json.dumps(
        {
            "fetched_at": "지금",
            "stats": {"kept": 1},
            "items": [
                {
                    "desertionNo": "한글-ID",
                    "desc": "순함",
                    "image_url": "https://example.test/korean.jpg",
                }
            ],
        },
        ensure_ascii=False,
    ).encode("utf-8")
    fresh_path.write_bytes(fresh_bytes)
    metas_path.write_text(
        json.dumps(
            [
                {
                    "desertionNo": "한글-ID",
                    "image_url": "https://example.test/korean.jpg",
                    "image_attrs": {"photo_quality_band": "좋음"},
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    assert (
        reuse.main(
            [
                "--fresh-cache",
                str(fresh_path),
                "--existing-metas",
                str(metas_path),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )

    raw = output_path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert raw.endswith(b"\n")
    payload = json.loads(raw.decode("utf-8"))
    assert payload["fetched_at"] == "지금"
    assert payload["stats"] == {"kept": 1}
    assert payload["items"][0]["desc"] == "순함"
    assert payload["items"][0]["image_attrs"] == {"photo_quality_band": "좋음"}
    assert not list(workdir.glob(".*.tmp"))

    assert (
        reuse.main(
            [
                "--fresh-cache",
                str(fresh_path),
                "--existing-metas",
                str(metas_path),
                "--output",
                str(fresh_path),
            ]
        )
        == 2
    )
    assert fresh_path.read_bytes() == fresh_bytes
    assert "must not overwrite an input file" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("payload", "label"),
    [
        ({"not_items": []}, "fresh cache"),
        ([{"ok": True}, "not-an-object"], "fresh cache"),
    ],
)
def test_fresh_cache_validation_rejects_invalid_shapes(
    workdir: Path, payload: object, label: str
) -> None:
    path = workdir / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=label):
        reuse.load_fresh_cache(path)


def test_existing_metas_requires_vector_row_list(workdir: Path) -> None:
    path = workdir / "metas.json"
    path.write_text('{"items": []}', encoding="utf-8")

    with pytest.raises(ValueError, match="JSON list of vector metadata rows"):
        reuse.load_existing_metas(path)
