from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.public_text_release import (
    PUBLIC_TEXT_METADATA_FIELDS,
    PUBLIC_TEXT_RELEASE_PROFILE,
    PublicTextReleaseError,
    derive_public_text_release,
    deterministic_json_bytes,
    public_notice_text,
    public_text_sha256,
    validate_public_text_release_payload,
    validate_public_text_snapshot_manifest,
)
from scripts import build_public_text_release as build_cli


class FakeIndex:
    def __init__(
        self,
        dimension: int,
        *,
        vectors: list[list[float]] | None = None,
        metric_type: int = 1,
    ) -> None:
        self.d = dimension
        self.metric_type = metric_type
        self.vectors = [np.asarray(row, dtype=np.float32) for row in vectors or []]

    @property
    def ntotal(self) -> int:
        return len(self.vectors)

    def reconstruct(self, position: int) -> np.ndarray:
        return self.vectors[position].copy()

    def add(self, matrix: np.ndarray) -> None:
        for row in np.asarray(matrix, dtype=np.float32):
            self.vectors.append(row.copy())


class FakeFaiss:
    METRIC_L2 = 1

    @staticmethod
    def IndexFlatL2(dimension: int) -> FakeIndex:
        return FakeIndex(dimension)

    @staticmethod
    def serialize_index(index: FakeIndex) -> np.ndarray:
        payload = json.dumps(
            {
                "d": index.d,
                "metric_type": index.metric_type,
                "vectors": [row.tolist() for row in index.vectors],
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return np.frombuffer(payload, dtype=np.uint8).copy()

    @staticmethod
    def deserialize_index(value: np.ndarray) -> FakeIndex:
        payload = json.loads(
            np.asarray(value, dtype=np.uint8).tobytes().decode("utf-8")
        )
        return FakeIndex(
            int(payload["d"]),
            vectors=payload["vectors"],
            metric_type=int(payload["metric_type"]),
        )


def _unit_vector(position: int) -> list[float]:
    row = [0.0] * 512
    row[position % 512] = 1.0
    return row


def _text_meta(notice_id: str, **extra: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "desertionNo": notice_id,
        "type": "text",
        "embedding_source": "public_notice_text",
        "desc": f"공개 설명 {notice_id}",
        "sex": "M",
        "age": "2024(년생)",
        "weight": "5(Kg)",
        "neuter": "Y",
        "color": "흰색",
        "detail_url": f"https://www.animal.go.kr/notice/{notice_id}",
        "image_url": f"https://photos.example/{notice_id}.jpg",
        "image_urls": [f"https://photos.example/{notice_id}.jpg"],
        "image_attrs": {
            "crop_path": f"data/image_crops/{notice_id}.jpg",
            "photo_quality_score": 0.9,
        },
        "vlm_desc": "사진에서 생성한 설명",
    }
    row.update(extra)
    row["desc_full"] = public_notice_text(row)
    row["embedding_text_sha256"] = public_text_sha256(row["desc_full"])
    return row


def _source_payload() -> tuple[bytes, bytes, bytes]:
    first = _text_meta("dog-a")
    second = _text_meta("dog-b", sex="F", color="검정")
    second["desc_full"] = public_notice_text(second)
    second["embedding_text_sha256"] = public_text_sha256(second["desc_full"])
    metas = [
        {
            **first,
            "type": "image",
            "embedding_source": "full_image",
        },
        first,
        {
            **first,
            "type": "crop_image",
            "embedding_source": "dog_crop",
        },
        second,
    ]
    index = FakeIndex(512, vectors=[_unit_vector(i) for i in range(len(metas))])
    manifest = {
        "schema_version": 1,
        "generated_at": "2026-07-26T00:00:00+09:00",
        "source": {
            "provider": "public-provider",
            "active_notices": 2,
            "dataset_url": "https://data.example/public-dataset",
            "source_image_url": "https://photos.example/leak.jpg",
            "api_key": "must-not-survive",
        },
        "embedding": {
            "model": "OpenAI CLIP ViT-B/32",
            "vector_types": {"image": 1, "crop_image": 1, "text": 2},
        },
        "readiness": {"strict_passed": True},
    }
    return (
        FakeFaiss.serialize_index(index).tobytes(),
        deterministic_json_bytes(metas),
        deterministic_json_bytes(manifest),
    )


def test_derivation_keeps_only_verified_public_text_and_is_deterministic() -> None:
    index_bytes, metas_bytes, manifest_bytes = _source_payload()

    first = derive_public_text_release(
        index_bytes,
        metas_bytes,
        source_manifest_bytes=manifest_bytes,
        faiss_module=FakeFaiss,
    )
    second = derive_public_text_release(
        index_bytes,
        metas_bytes,
        source_manifest_bytes=manifest_bytes,
        faiss_module=FakeFaiss,
    )

    assert first.index_bytes == second.index_bytes
    assert first.metas_bytes == second.metas_bytes
    assert first.snapshot_manifest_bytes == second.snapshot_manifest_bytes
    assert first.release_profile_bytes == second.release_profile_bytes
    assert first.derivation_report_bytes == second.derivation_report_bytes
    assert first.vector_count == 2
    assert first.unique_notice_count == 2
    assert first.excluded_vector_counts == {"image": 1, "crop_image": 1}

    output_index = FakeFaiss.deserialize_index(
        np.frombuffer(first.index_bytes, dtype=np.uint8)
    )
    assert output_index.ntotal == 2
    assert np.array_equal(output_index.reconstruct(0), np.asarray(_unit_vector(1)))
    assert np.array_equal(output_index.reconstruct(1), np.asarray(_unit_vector(3)))

    output_metas = json.loads(first.metas_bytes)
    assert [row["desertionNo"] for row in output_metas] == ["dog-a", "dog-b"]
    assert all(set(row) <= PUBLIC_TEXT_METADATA_FIELDS for row in output_metas)
    for row in output_metas:
        assert row["type"] == "text"
        assert row["embedding_source"] == "public_notice_text"
        assert "image_attrs" not in row
        assert "image_url" not in row
        assert "image_urls" not in row
        assert "vlm_desc" not in row

    manifest = json.loads(first.snapshot_manifest_bytes)
    assert manifest["release_profile"]["id"] == PUBLIC_TEXT_RELEASE_PROFILE
    assert manifest["source"] == {
        "active_notices": 2,
        "dataset_url": "https://data.example/public-dataset",
        "provider": "public-provider",
    }
    assert manifest["embedding"]["vector_types"] == {
        "crop_image": 0,
        "image": 0,
        "text": 2,
    }
    assert manifest["artifacts"]["data/dog_faiss.index"]["sha256"] == (
        first.index_sha256
    )
    assert manifest["artifacts"]["data/dog_metas.json"]["sha256"] == (
        first.metas_sha256
    )
    assert (
        validate_public_text_snapshot_manifest(first.snapshot_manifest_bytes)[
            "remote_photo_references"
        ]
        == 0
    )

    validated = validate_public_text_release_payload(
        first.index_bytes,
        first.metas_bytes,
        first.release_profile_bytes,
        faiss_module=FakeFaiss,
    )
    assert validated == {
        "profile": PUBLIC_TEXT_RELEASE_PROFILE,
        "vectors": 2,
        "unique_notices": 2,
        "dimension": 512,
        "visual_vectors": 0,
        "remote_photo_reference_fields": 0,
    }


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (
            lambda rows: rows[1].update(embedding_source="full_image"),
            "invalid embedding provenance",
        ),
        (
            lambda rows: rows[1].update(desc_full="stale generated text"),
            "does not match canonical public text",
        ),
        (
            lambda rows: rows[1].update(embedding_text_sha256="0" * 64),
            "invalid public-text SHA-256",
        ),
        (
            lambda rows: rows[3].update(desertionNo="dog-a"),
            "duplicate public-text row",
        ),
    ],
)
def test_derivation_rejects_unprovenanced_or_inconsistent_text(
    mutation: Any,
    error: str,
) -> None:
    index_bytes, metas_bytes, manifest_bytes = _source_payload()
    rows = json.loads(metas_bytes)
    mutation(rows)

    with pytest.raises(PublicTextReleaseError, match=error):
        derive_public_text_release(
            index_bytes,
            deterministic_json_bytes(rows),
            source_manifest_bytes=manifest_bytes,
            faiss_module=FakeFaiss,
        )


def test_derivation_requires_one_public_text_vector_for_every_notice() -> None:
    index_bytes, metas_bytes, manifest_bytes = _source_payload()
    rows = json.loads(metas_bytes)
    rows[3]["type"] = "image"
    rows[3]["embedding_source"] = "full_image"

    with pytest.raises(PublicTextReleaseError, match="without a public-text vector"):
        derive_public_text_release(
            index_bytes,
            deterministic_json_bytes(rows),
            source_manifest_bytes=manifest_bytes,
            faiss_module=FakeFaiss,
        )


def test_validation_rejects_disallowed_field_in_derived_metadata() -> None:
    index_bytes, metas_bytes, manifest_bytes = _source_payload()
    artifacts = derive_public_text_release(
        index_bytes,
        metas_bytes,
        source_manifest_bytes=manifest_bytes,
        faiss_module=FakeFaiss,
    )
    rows = json.loads(artifacts.metas_bytes)
    rows[0]["image_url"] = "https://photos.example/leak.jpg"

    with pytest.raises(PublicTextReleaseError, match="disallowed fields"):
        validate_public_text_release_payload(
            artifacts.index_bytes,
            deterministic_json_bytes(rows),
            artifacts.release_profile_bytes,
            faiss_module=FakeFaiss,
        )


@pytest.mark.parametrize("field", ("source_image_url", "api_key"))
def test_manifest_validation_rejects_photo_or_secret_fields(field: str) -> None:
    index_bytes, metas_bytes, manifest_bytes = _source_payload()
    artifacts = derive_public_text_release(
        index_bytes,
        metas_bytes,
        source_manifest_bytes=manifest_bytes,
        faiss_module=FakeFaiss,
    )
    manifest = json.loads(artifacts.snapshot_manifest_bytes)
    manifest["source"][field] = "https://photos.example/leak.jpg"

    with pytest.raises(
        PublicTextReleaseError,
        match="photo-reference or secret field",
    ):
        validate_public_text_snapshot_manifest(deterministic_json_bytes(manifest))


def test_build_cli_writes_staging_artifacts_and_check_only_does_not_write(
    tmp_path: Path,
) -> None:
    index_bytes, metas_bytes, manifest_bytes = _source_payload()
    source = tmp_path / "source"
    source.mkdir()
    index_path = source / "dog_faiss.index"
    metas_path = source / "dog_metas.json"
    manifest_path = source / "snapshot_manifest.json"
    index_path.write_bytes(index_bytes)
    metas_path.write_bytes(metas_bytes)
    manifest_path.write_bytes(manifest_bytes)

    checked_dir = tmp_path / "checked"
    checked, checked_paths = build_cli.build_public_text_release(
        index_path=index_path,
        metas_path=metas_path,
        manifest_path=manifest_path,
        output_dir=checked_dir,
        check_only=True,
        faiss_module=FakeFaiss,
    )
    assert checked.vector_count == 2
    assert not checked_dir.exists()
    assert not any(path.exists() for path in checked_paths.values())

    output_dir = tmp_path / "output"
    written, paths = build_cli.build_public_text_release(
        index_path=index_path,
        metas_path=metas_path,
        manifest_path=manifest_path,
        output_dir=output_dir,
        faiss_module=FakeFaiss,
    )
    assert paths["index"].read_bytes() == written.index_bytes
    assert paths["metas"].read_bytes() == written.metas_bytes
    assert paths["profile"].read_bytes() == written.release_profile_bytes
    assert paths["manifest"].read_bytes() == written.snapshot_manifest_bytes
    assert paths["report"].read_bytes() == written.derivation_report_bytes
