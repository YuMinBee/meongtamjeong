from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
import shutil
from uuid import uuid4

import pytest

from scripts import merge_dog_metadata as merge


@pytest.fixture
def workdir() -> Iterator[Path]:
    runtime_root = Path(__file__).resolve().parent / ".runtime"
    directory = runtime_root / f"merge-{uuid4().hex}"
    directory.mkdir(parents=True)
    try:
        yield directory
    finally:
        shutil.rmtree(directory)
        try:
            runtime_root.rmdir()
        except OSError:
            pass


def test_merge_preserves_rows_and_applies_non_empty_source_precedence() -> None:
    metas = [
        {
            "type": "vector-a",
            "desertionNo": "A",
            "priority": "vector-a",
            "region": "vector-region",
            "nested": {"vector": 1, "shared": "vector"},
        },
        {"type": 7, "desertion_no": 2, "priority": "vector-b"},
        {"desertionNo": "C", "priority": "vector-c"},
        {"type": "no-id", "priority": "unmatched"},
    ]
    cache = [
        {
            "type": "cache-type-must-not-win",
            "desertionNo": "A",
            "priority": "cache-a",
            "region": "cache-region",
            "nested": {"cache": 2, "shared": "cache"},
        },
        {
            "type": "cache-type-must-not-win",
            "desertionNo": "2",
            "priority": "cache-b",
        },
        {
            "type": "cache-type-must-not-be-added",
            "desertionNo": "C",
            "priority": "cache-c",
        },
    ]
    enriched = [
        {
            "desertionNo": "A",
            "priority": "enriched-a",
            "region": "Unknown",
            "nested": {"enriched": 3, "shared": "enriched"},
        },
        {"desertionNo": "2", "priority": " unknown "},
        {"desertionNo": "C", "priority": "   "},
    ]
    originals = deepcopy((metas, cache, enriched))

    merged, stats = merge.merge_dog_metadata(metas, cache, enriched)

    assert len(merged) == len(metas)
    assert [merge.record_id(item) for item in merged] == [
        merge.record_id(item) for item in metas
    ]
    assert [item.get("type") for item in merged] == [item.get("type") for item in metas]
    assert [("type" in item) for item in merged] == [("type" in item) for item in metas]
    assert [("desertionNo" in item, "desertion_no" in item) for item in merged] == [
        ("desertionNo" in item, "desertion_no" in item) for item in metas
    ]
    assert merged[0]["desertionNo"] == "A"
    assert merged[1]["desertion_no"] == 2

    assert merged[0]["priority"] == "enriched-a"
    assert merged[0]["region"] == "cache-region"
    assert merged[0]["nested"] == {
        "vector": 1,
        "cache": 2,
        "enriched": 3,
        "shared": "enriched",
    }
    assert merged[1]["priority"] == "cache-b"
    assert merged[2]["priority"] == "cache-c"
    assert merged[3] == metas[3]
    assert stats["metas_total"] == 4
    assert stats["matched_vectors"] == 3
    assert stats["vectors_missing_id"] == 1
    assert (metas, cache, enriched) == originals


def test_run_continues_when_optional_cache_is_missing(
    workdir: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    metas_path = workdir / "metas.json"
    missing_cache_path = workdir / "missing-cache.json"
    enriched_path = workdir / "enriched.json"
    output_path = workdir / "output.json"
    metas_path.write_text(
        '[{"type": "dog", "desertionNo": "A", "priority": "meta"}]',
        encoding="utf-8",
    )
    enriched_path.write_text(
        '{"items": [{"desertionNo": "A", "priority": "enriched"}]}',
        encoding="utf-8",
    )
    args = merge.parse_args(
        [
            "--metas",
            str(metas_path),
            "--cache",
            str(missing_cache_path),
            "--enriched",
            str(enriched_path),
            "--output",
            str(output_path),
        ]
    )

    stats = merge.run(args)

    output = merge.load_metas(output_path)
    assert output[0]["priority"] == "enriched"
    assert stats["cache"]["file_missing"] is True
    assert stats["cache"]["records"] == 0
    assert stats["enriched"]["file_missing"] is False
    assert "[WARN] cache file not found" in capsys.readouterr().err


def test_required_metas_missing_is_an_error(
    workdir: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing_metas_path = workdir / "missing-metas.json"

    with pytest.raises(FileNotFoundError, match="required metas file not found"):
        merge.load_metas(missing_metas_path)

    exit_code = merge.main(
        [
            "--metas",
            str(missing_metas_path),
            "--cache",
            str(workdir / "missing-cache.json"),
            "--enriched",
            str(workdir / "missing-enriched.json"),
            "--output",
            str(workdir / "unused-output.json"),
        ]
    )

    assert exit_code == 1
    assert "[ERROR] required metas file not found" in capsys.readouterr().err
