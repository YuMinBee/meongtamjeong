from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.notice_status import classify_notice as shared_classify_notice
from scripts import check_contest_readiness as readiness


REFERENCE_DATE = datetime(2026, 7, 17)


@pytest.fixture
def workdir() -> Iterator[Path]:
    runtime_root = Path(__file__).resolve().parent / ".runtime"
    directory = runtime_root / f"readiness-{uuid4().hex}"
    directory.mkdir(parents=True)
    try:
        yield directory
    finally:
        shutil.rmtree(directory)
        try:
            runtime_root.rmdir()
        except OSError:
            pass


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def test_cache_overlay_reports_deterministic_coverage_and_status(
    workdir: Path,
) -> None:
    metas_path = workdir / "metas.json"
    cache_path = workdir / "cache.json"
    metas = [
        {"desertionNo": "A", "type": "image"},
        {"desertionNo": "A", "type": "text"},
        {"desertion_no": "B", "type": "image"},
    ]
    cache = {
        "fetched_at": "2026-07-17T09:00:00",
        "items": [
            {
                "desertionNo": "A",
                "processState": "active",
                "noticeEdt": "2026-12-31",
                "region": "Seoul",
                "careNm": "Shelter A",
                "vlm_attrs": {"photo_quality_score": 0.8},
            },
            {
                "desertion_no": "B",
                "status": "active",
                "notice_end": "2026-12-31",
                "region_name": "Busan",
                "care_name": "Shelter B",
            },
        ],
    }
    write_json(metas_path, metas)
    write_json(cache_path, cache)
    original_metas = metas_path.read_bytes()
    original_cache = cache_path.read_bytes()

    first = readiness.build_report(
        metas_path,
        cache_path=cache_path,
        reference_date=REFERENCE_DATE,
        strict=True,
    )
    second = readiness.build_report(
        metas_path,
        cache_path=cache_path,
        reference_date=REFERENCE_DATE,
        strict=True,
    )

    assert readiness.classify_notice is shared_classify_notice
    assert first == second
    assert first["ready"] is True
    assert first["metas"]["row_count"] == 3
    assert first["metas"]["unique_id_count"] == 2
    assert first["metas"]["duplicate_id_count"] == 1
    assert first["metas"]["coverage"]["status"]["count"] == 0
    assert first["cache"]["row_count"] == 2
    assert first["effective"]["cache_matched_row_count"] == 3
    assert first["effective"]["cache_matched_unique_id_count"] == 2
    assert first["effective"]["coverage"]["status"] == {"count": 3, "ratio": 1.0}
    assert first["effective"]["coverage"]["notice_end"]["count"] == 3
    assert first["effective"]["coverage"]["region"]["count"] == 3
    assert first["effective"]["coverage"]["shelter"]["count"] == 3
    assert first["effective"]["coverage"]["last_verified"]["count"] == 3
    assert first["effective"]["coverage"]["photo_quality"]["count"] == 2
    assert first["effective"]["status_distribution"] == {
        "active": 3,
        "closed": 0,
        "expired": 0,
        "unknown": 0,
    }
    assert metas_path.read_bytes() == original_metas
    assert cache_path.read_bytes() == original_cache


def test_strict_cli_writes_report_before_nonzero_exit(workdir: Path, capsys) -> None:
    metas_path = workdir / "metas.json"
    output_path = workdir / "report.json"
    write_json(
        metas_path,
        [{"desertionNo": "known", "type": "image"}, {"type": "text"}],
    )
    original = metas_path.read_bytes()

    exit_code = readiness.main(
        [
            "--metas",
            str(metas_path),
            "--output",
            str(output_path),
            "--reference-date",
            "2026-07-17",
            "--strict",
        ]
    )

    assert exit_code == 1
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert json.loads(capsys.readouterr().out) == report
    assert report["ready"] is False
    assert report["strict"]["fatal_issues"] == [
        "metas_missing_id",
        "status_coverage_zero",
        "unknown_notice_status",
    ]
    assert metas_path.read_bytes() == original
    assert not list(workdir.glob(f".{output_path.name}.*.tmp"))

    write_json(
        metas_path,
        [
            {"dog_id": "active", "status": "active", "orgNm": "Seoul"},
            {"dog_id": "closed", "status": "closed", "careAddr": "Busan"},
        ],
    )
    inactive = readiness.build_report(
        metas_path, reference_date=REFERENCE_DATE, strict=True
    )
    assert "inactive_notice_status" in inactive["strict"]["fatal_issues"]
    assert inactive["metas"]["unique_id_count"] == 2
    assert inactive["effective"]["coverage"]["region"]["count"] == 2

    write_json(
        metas_path,
        [
            {"dog_id": "end-only", "noticeEdt": "20261231"},
            {"dog_id": "bad-state", "status": "garbage"},
        ],
    )
    classifier_aligned = readiness.build_report(
        metas_path, reference_date=REFERENCE_DATE, strict=True
    )
    assert classifier_aligned["effective"]["coverage"]["status"] == {
        "count": 1,
        "ratio": 0.5,
    }
    assert classifier_aligned["effective"]["status_distribution"] == {
        "active": 1,
        "closed": 0,
        "expired": 0,
        "unknown": 1,
    }


def test_optional_index_match_is_unknown_without_faiss_and_fatal_on_mismatch(
    workdir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metas_path = workdir / "metas.json"
    index_path = workdir / "dogs.index"
    write_json(metas_path, [{"desertionNo": "A", "status": "active"}])
    index_path.write_bytes(b"fake")

    def unavailable() -> object:
        raise ModuleNotFoundError("faiss")

    monkeypatch.setattr(readiness, "load_faiss", unavailable)
    unknown = readiness.build_report(
        metas_path,
        index_path=index_path,
        reference_date=REFERENCE_DATE,
        strict=True,
    )
    assert unknown["ready"] is False
    assert unknown["index"] == {
        "requested": True,
        "path": str(index_path),
        "exists": True,
        "size_bytes": 4,
        "faiss_available": False,
        "verification": "unknown",
        "ntotal": None,
        "matches_meta_rows": None,
    }
    assert "faiss_unavailable" in unknown["strict"]["fatal_issues"]

    non_strict_unknown = readiness.build_report(
        metas_path,
        index_path=index_path,
        reference_date=REFERENCE_DATE,
        strict=False,
    )
    assert non_strict_unknown["ready"] is True
    assert "faiss_unavailable" in non_strict_unknown["strict"]["warnings"]

    fake_faiss = SimpleNamespace(read_index=lambda _path: SimpleNamespace(ntotal=2))
    monkeypatch.setattr(readiness, "load_faiss", lambda: fake_faiss)
    mismatch = readiness.build_report(
        metas_path,
        index_path=index_path,
        reference_date=REFERENCE_DATE,
        strict=True,
    )
    assert mismatch["ready"] is False
    assert mismatch["index"]["verification"] == "mismatch"
    assert mismatch["index"]["ntotal"] == 2
    assert mismatch["index"]["matches_meta_rows"] is False
    assert "index_meta_mismatch" in mismatch["strict"]["fatal_issues"]
