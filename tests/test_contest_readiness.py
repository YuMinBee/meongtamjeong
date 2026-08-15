from __future__ import annotations

import hashlib
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
ROOT = Path(__file__).resolve().parents[1]


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


def test_tracked_snapshot_manifest_has_readable_official_provider() -> None:
    manifest = json.loads(
        (ROOT / "data" / "snapshot_manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["source"]["provider"] == "농림축산식품부 농림축산검역본부"


class FakeIndex:
    def __init__(
        self,
        vectors: list[list[float]],
        *,
        dimension: int = readiness.EXPECTED_CLIP_DIMENSION,
        metric_type: int = 1,
        reconstruction_failures: set[int] | None = None,
    ) -> None:
        self._vectors = vectors
        self.ntotal = len(vectors)
        self.d = dimension
        self.metric_type = metric_type
        self._reconstruction_failures = reconstruction_failures or set()

    def reconstruct(self, position: int) -> list[float]:
        if position in self._reconstruction_failures:
            raise RuntimeError("cannot reconstruct")
        return self._vectors[position]


def fake_faiss(index: FakeIndex) -> SimpleNamespace:
    return SimpleNamespace(
        METRIC_L2=1,
        read_index=lambda _path: index,
    )


def public_text_row(dog_id: str, text: str = "공고 설명") -> dict[str, object]:
    return {
        "desertionNo": dog_id,
        "type": "text",
        "embedding_source": "public_notice_text",
        "desc_full": text,
        "embedding_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "status": "active",
    }


def image_row(dog_id: str, row_type: str) -> dict[str, object]:
    image_url = f"https://public.test/{dog_id}.jpg"
    return {
        "desertionNo": dog_id,
        "type": row_type,
        "embedding_source": (
            "full_image" if row_type == "image" else "dog_crop"
        ),
        "image_url": image_url,
        "embedding_image_url": image_url,
        "status": "active",
    }


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
    assert first["effective"]["unique_notices"]["count"] == 2
    assert first["effective"]["unique_notices"]["coverage"]["status"] == {
        "count": 2,
        "ratio": 1.0,
    }
    assert first["effective"]["freshness"] == {
        "enabled": False,
        "max_age_days": None,
        "reference_date": "2026-07-17",
        "unique_notice_count": 2,
        "timestamp_present_count": 2,
        "parsed_count": 2,
        "fresh_count": 2,
        "stale_count": 0,
        "missing_count": 0,
        "invalid_count": 0,
        "future_count": 0,
        "oldest_verified_date": "2026-07-17",
        "newest_verified_date": "2026-07-17",
    }
    assert first["effective"]["status_distribution"] == {
        "active": 3,
        "closed": 0,
        "expired": 0,
        "unknown": 0,
    }
    assert metas_path.read_bytes() == original_metas
    assert cache_path.read_bytes() == original_cache


def test_freshness_is_counted_per_unique_notice_and_strictly_gated(
    workdir: Path,
) -> None:
    metas_path = workdir / "metas.json"
    records = [
        {
            "dog_id": "fresh-boundary",
            "type": "image",
            "status": "active",
            "last_verified_at": "2026-07-10",
        },
        {
            "dog_id": "fresh-boundary",
            "type": "text",
            "status": "active",
            "last_verified_at": "2026-07-09T23:59:59+09:00",
        },
        {
            "dog_id": "fresh-zulu",
            "status": "active",
            "fetched_at": "2026-07-17T00:01:02Z",
        },
        {
            "dog_id": "stale",
            "status": "active",
            "last_verified": "2026-07-09T23:59:59",
        },
        {"dog_id": "missing", "status": "active"},
        {
            "dog_id": "invalid",
            "status": "active",
            "verified_at": "17/07/2026",
        },
        {
            "dog_id": "future",
            "status": "active",
            "last_verified_at": "2026-07-18",
        },
    ]
    write_json(metas_path, records)

    strict_report = readiness.build_report(
        metas_path,
        reference_date=REFERENCE_DATE,
        strict=True,
        max_age_days=7,
    )

    assert strict_report["effective"]["row_count"] == 7
    assert strict_report["effective"]["unique_id_count"] == 6
    assert strict_report["effective"]["duplicate_id_count"] == 1
    assert strict_report["effective"]["unique_notices"]["count"] == 6
    assert strict_report["effective"]["unique_notices"]["coverage"][
        "last_verified"
    ] == {"count": 5, "ratio": 0.833333}
    assert strict_report["effective"]["freshness"] == {
        "enabled": True,
        "max_age_days": 7,
        "reference_date": "2026-07-17",
        "unique_notice_count": 6,
        "timestamp_present_count": 5,
        "parsed_count": 4,
        "fresh_count": 2,
        "stale_count": 1,
        "missing_count": 1,
        "invalid_count": 1,
        "future_count": 1,
        "oldest_verified_date": "2026-07-09",
        "newest_verified_date": "2026-07-18",
    }
    assert strict_report["ready"] is False
    assert strict_report["strict"]["fatal_issues"] == [
        "freshness_future",
        "freshness_invalid",
        "freshness_missing",
        "freshness_stale",
    ]

    warning_report = readiness.build_report(
        metas_path,
        reference_date=REFERENCE_DATE,
        strict=False,
        max_age_days=7,
    )
    assert warning_report["ready"] is True
    assert warning_report["strict"]["fatal_issues"] == []
    assert set(warning_report["strict"]["warnings"]) >= {
        "freshness_future",
        "freshness_invalid",
        "freshness_missing",
        "freshness_stale",
    }


def test_freshness_is_opt_in_and_cache_fetched_at_is_effective(
    workdir: Path,
) -> None:
    metas_path = workdir / "metas.json"
    cache_path = workdir / "cache.json"
    write_json(
        metas_path,
        [
            {"dog_id": "A", "type": "image", "status": "active"},
            {"dog_id": "A", "type": "text", "status": "active"},
        ],
    )

    legacy = readiness.build_report(
        metas_path,
        reference_date=REFERENCE_DATE,
        strict=True,
    )
    assert legacy["ready"] is True
    assert "freshness_missing" not in legacy["strict"]["fatal_issues"]
    assert legacy["effective"]["freshness"]["enabled"] is False
    assert legacy["effective"]["freshness"]["missing_count"] == 1

    write_json(
        cache_path,
        {
            "fetched_at": "2026-07-17T23:59:59+09:00",
            "items": [{"dog_id": "A", "status": "active"}],
        },
    )
    cached = readiness.build_report(
        metas_path,
        cache_path=cache_path,
        reference_date=REFERENCE_DATE,
        strict=True,
        max_age_days=0,
    )
    assert cached["ready"] is True
    assert cached["effective"]["freshness"]["unique_notice_count"] == 1
    assert cached["effective"]["freshness"]["fresh_count"] == 1
    assert cached["effective"]["freshness"]["stale_count"] == 0


def test_max_age_days_cli_validation_and_exit_code(
    workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    metas_path = workdir / "metas.json"
    output_path = workdir / "report.json"
    write_json(
        metas_path,
        [
            {
                "dog_id": "old",
                "status": "active",
                "last_verified_at": "2026-07-16",
            }
        ],
    )

    exit_code = readiness.main(
        [
            "--metas",
            str(metas_path),
            "--output",
            str(output_path),
            "--reference-date",
            "2026-07-17",
            "--max-age-days",
            "0",
            "--strict",
        ]
    )
    assert exit_code == 1
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert json.loads(capsys.readouterr().out) == report
    assert report["strict"]["fatal_issues"] == ["freshness_stale"]

    with pytest.raises(SystemExit) as negative:
        readiness.parse_args(["--max-age-days", "-1"])
    assert negative.value.code == 2
    with pytest.raises(SystemExit) as fractional:
        readiness.parse_args(["--max-age-days", "1.5"])
    assert fractional.value.code == 2
    with pytest.raises(ValueError, match="nonnegative integer"):
        readiness.build_report(
            metas_path,
            reference_date=REFERENCE_DATE,
            max_age_days=-1,
        )
    with pytest.raises(ValueError, match="nonnegative integer"):
        readiness.build_report(
            metas_path,
            reference_date=REFERENCE_DATE,
            max_age_days=True,
        )


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
    assert "artifact_integrity" not in non_strict_unknown

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


def test_strict_index_artifact_integrity_accepts_sync_contract(
    workdir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metas_path = workdir / "metas.json"
    index_path = workdir / "dogs.index"
    records = [
        image_row("A", "image"),
        image_row("A", "crop_image"),
        public_text_row("A"),
    ]
    write_json(metas_path, records)
    index_path.write_bytes(b"fake")
    unit_vector = [1.0] + [0.0] * (readiness.EXPECTED_CLIP_DIMENSION - 1)
    index = FakeIndex([unit_vector.copy() for _ in records])
    monkeypatch.setattr(readiness, "load_faiss", lambda: fake_faiss(index))

    report = readiness.build_report(
        metas_path,
        index_path=index_path,
        reference_date=REFERENCE_DATE,
        strict=True,
    )

    assert report["ready"] is True
    assert report["index"]["verification"] == "match"
    integrity = report["artifact_integrity"]
    assert integrity["verification"] == "pass"
    assert integrity["metadata"]["passed"] is True
    assert integrity["metadata"]["row_type_distribution"] == {
        "crop_image": 1,
        "image": 1,
        "text": 1,
    }
    assert integrity["vectors"]["passed"] is True
    assert integrity["vectors"]["dimension"] == 512
    assert integrity["vectors"]["metric_type"] == 1
    assert integrity["vectors"]["norm_min"] == 1.0
    assert integrity["vectors"]["norm_max"] == 1.0


def test_strict_index_artifact_integrity_rejects_metadata_violations(
    workdir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metas_path = workdir / "metas.json"
    index_path = workdir / "dogs.index"
    bad_image = image_row("A", "image")
    bad_image.update(
        {
            "embedding_source": "dog_crop",
            "embedding_image_url": "https://stale.test/A.jpg",
            "vlm_attrs": {},
        }
    )
    bad_text = public_text_row("A")
    bad_text["embedding_text_sha256"] = "not-the-public-text-hash"
    no_text = image_row("B", "crop_image")
    blank_id_text = public_text_row("")
    unknown_type = {
        "desertionNo": "C",
        "type": "legacy_image",
        "embedding_source": "full_image",
        "status": "active",
    }
    records = [bad_image, bad_text, no_text, blank_id_text, unknown_type]
    write_json(metas_path, records)
    index_path.write_bytes(b"fake")
    unit_vector = [1.0] + [0.0] * (readiness.EXPECTED_CLIP_DIMENSION - 1)
    index = FakeIndex([unit_vector.copy() for _ in records])
    monkeypatch.setattr(readiness, "load_faiss", lambda: fake_faiss(index))

    report = readiness.build_report(
        metas_path,
        index_path=index_path,
        reference_date=REFERENCE_DATE,
        strict=True,
    )

    assert report["ready"] is False
    integrity = report["artifact_integrity"]
    assert integrity["verification"] == "fail"
    assert integrity["vectors"]["passed"] is True
    metadata = integrity["metadata"]
    assert metadata["blank_id_row_count"] == 1
    assert metadata["invalid_row_type_count"] == 1
    assert metadata["invalid_provenance_count"] == 1
    assert metadata["invalid_text_hash_count"] == 1
    assert metadata["invalid_image_source_count"] == 1
    assert metadata["forbidden_derived_field_occurrence_count"] == 1
    assert metadata["unique_ids_missing_public_text_count"] == 3
    assert metadata["samples"]["unique_ids_missing_public_text"] == ["A", "B", "C"]
    assert set(report["strict"]["fatal_issues"]) >= {
        "artifact_blank_id",
        "artifact_forbidden_derived_field",
        "artifact_image_source_invalid",
        "artifact_provenance_invalid",
        "artifact_public_text_missing",
        "artifact_row_type_invalid",
        "artifact_text_hash_invalid",
        "metas_missing_id",
    }


def test_strict_index_artifact_integrity_rejects_vector_violations(
    workdir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metas_path = workdir / "metas.json"
    index_path = workdir / "dogs.index"
    records = [public_text_row(dog_id) for dog_id in "ABCDE"]
    write_json(metas_path, records)
    index_path.write_bytes(b"fake")
    dimension = readiness.EXPECTED_CLIP_DIMENSION - 1
    normal = [1.0] + [0.0] * (dimension - 1)
    zero = [0.0] * dimension
    non_unit = [2.0] + [0.0] * (dimension - 1)
    nonfinite = [float("nan")] + [0.0] * (dimension - 1)
    index = FakeIndex(
        [normal, zero, non_unit, nonfinite, normal],
        dimension=dimension,
        metric_type=0,
        reconstruction_failures={4},
    )
    monkeypatch.setattr(readiness, "load_faiss", lambda: fake_faiss(index))

    report = readiness.build_report(
        metas_path,
        index_path=index_path,
        reference_date=REFERENCE_DATE,
        strict=True,
    )

    assert report["ready"] is False
    integrity = report["artifact_integrity"]
    assert integrity["metadata"]["passed"] is True
    vectors = integrity["vectors"]
    assert vectors["passed"] is False
    assert vectors["dimension"] == dimension
    assert vectors["metric_type"] == 0
    assert vectors["reconstruction_error_count"] == 1
    assert vectors["nonfinite_vector_count"] == 1
    assert vectors["zero_norm_vector_count"] == 1
    assert vectors["non_unit_norm_vector_count"] == 1
    assert set(report["strict"]["fatal_issues"]) >= {
        "artifact_index_dimension_invalid",
        "artifact_index_metric_invalid",
        "artifact_vector_nonfinite",
        "artifact_vector_not_normalized",
        "artifact_vector_reconstruction_failed",
        "artifact_vector_zero",
    }
