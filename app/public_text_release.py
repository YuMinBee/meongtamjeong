"""Deterministic public-text-only release artifact derivation.

This module creates a rights-exposure-minimizing distribution profile.  It is
not a legal determination. Only CLIP rows whose metadata records public-notice
text provenance and passes canonical text/hash validation are retained. Image/crop vectors,
remote photo references, image-analysis metadata, and all unrecognized fields
are excluded from the derived artifacts.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.notice_metadata import first_text


PUBLIC_TEXT_RELEASE_PROFILE = "public-text-only-v1"
PUBLIC_TEXT_RELEASE_PURPOSE = "rights_exposure_minimization_not_a_legal_determination"
EXPECTED_CLIP_DIMENSION = 512
ALLOWED_SOURCE_ROW_TYPES = frozenset({"text", "image", "crop_image"})
PHOTO_FILE_SUFFIXES = frozenset(
    {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)

# A positive allowlist is intentional.  A denylist could silently retain a new
# detector, VLM, photo-quality, crop, or remote-photo field added in the future.
PUBLIC_TEXT_METADATA_FIELDS = frozenset(
    {
        "age",
        "breed",
        "breed_code",
        "breed_full_name",
        "breed_name",
        "breed_source",
        "breed_source_label",
        "care_addr",
        "care_name",
        "care_tel",
        "color",
        "desc",
        "desc_full",
        "desertionNo",
        "detail_url",
        "embedding_source",
        "embedding_text_sha256",
        "happen_date",
        "happen_place",
        "health_checks",
        "last_verified_at",
        "mixed_breed",
        "neuter",
        "notice_end",
        "notice_no",
        "notice_start",
        "org_name",
        "process_state",
        "safety_health_note",
        "safety_social_note",
        "sex",
        "species",
        "type",
        "upkind",
        "upstream_updated_at",
        "vaccinations",
        "weight",
    }
)
PUBLIC_TEXT_MANIFEST_SOURCE_FIELDS = frozenset(
    {
        "active_notices",
        "closed_items_skipped",
        "dataset_url",
        "duplicate_items_skipped",
        "fetched_at",
        "lookback_days",
        "no_image_items_skipped",
        "pages_fetched",
        "provider",
        "raw_items",
        "source_cache_distributed",
        "source_cache_sha256",
        "species",
        "upkind",
    }
)
FORBIDDEN_MANIFEST_KEY_FRAGMENTS = (
    "api_key",
    "credential",
    "image_url",
    "password",
    "photo_url",
    "secret",
    "source_image_url",
    "token",
)


class PublicTextReleaseError(ValueError):
    """The source snapshot cannot produce the declared release profile."""


@dataclass(frozen=True)
class PublicTextReleaseArtifacts:
    """In-memory deterministic artifacts ready for staging or packaging."""

    index_bytes: bytes
    metas_bytes: bytes
    snapshot_manifest_bytes: bytes
    release_profile_bytes: bytes
    derivation_report_bytes: bytes
    vector_count: int
    unique_notice_count: int
    dimension: int
    source_vector_count: int
    excluded_vector_counts: Mapping[str, int]
    source_index_sha256: str
    source_metas_sha256: str
    index_sha256: str
    metas_sha256: str

    def safe_summary(self) -> dict[str, Any]:
        return {
            "profile": PUBLIC_TEXT_RELEASE_PROFILE,
            "vectors": self.vector_count,
            "unique_notices": self.unique_notice_count,
            "dimension": self.dimension,
            "source_vectors": self.source_vector_count,
            "excluded_vectors": dict(self.excluded_vector_counts),
            "index_sha256": self.index_sha256,
            "metas_sha256": self.metas_sha256,
        }


def clean_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def public_notice_text(record: Mapping[str, Any]) -> str:
    """Build the same public-fact-only CLIP text used by active index sync."""

    desc = first_text(record, "desc", "specialMark")
    fields = (
        ("설명", desc),
        ("성별", first_text(record, "sexCd", "sex")),
        ("나이", first_text(record, "age")),
        ("체중", first_text(record, "weight")),
        ("중성화", first_text(record, "neuterYn", "neuter")),
        ("색상", first_text(record, "color", "colorCd")),
    )
    return ", ".join(f"{label} {value}" for label, value in fields if value)


def public_text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def deterministic_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _load_json_bytes(data: bytes, *, label: str) -> Any:
    try:
        return json.loads(data.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicTextReleaseError(f"{label} must be valid UTF-8 JSON") from exc


def _load_source_metas(data: bytes) -> list[dict[str, Any]]:
    value = _load_json_bytes(data, label="source metadata")
    if not isinstance(value, list) or not value:
        raise PublicTextReleaseError("source metadata must be a non-empty array")
    if not all(isinstance(row, Mapping) for row in value):
        raise PublicTextReleaseError("every source metadata row must be an object")
    return [dict(row) for row in value]


def _load_source_manifest(data: bytes | None) -> dict[str, Any]:
    if data is None:
        return {}
    value = _load_json_bytes(data, label="source snapshot manifest")
    if not isinstance(value, Mapping):
        raise PublicTextReleaseError("source snapshot manifest must be an object")
    return copy.deepcopy(dict(value))


def _iter_manifest_keys(value: Any) -> Sequence[str]:
    keys: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            keys.append(clean_text(key).lower())
            keys.extend(_iter_manifest_keys(child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            keys.extend(_iter_manifest_keys(child))
    return keys


def validate_public_text_snapshot_manifest(data: bytes) -> dict[str, Any]:
    """Validate that the derived manifest itself carries no photo URL or secret field."""

    value = _load_json_bytes(data, label="derived snapshot manifest")
    if not isinstance(value, Mapping):
        raise PublicTextReleaseError("derived snapshot manifest must be an object")
    source = value.get("source")
    if not isinstance(source, Mapping):
        raise PublicTextReleaseError("derived manifest source contract is missing")
    for key in _iter_manifest_keys(value):
        if any(fragment in key for fragment in FORBIDDEN_MANIFEST_KEY_FRAGMENTS):
            raise PublicTextReleaseError(
                "derived manifest contains a photo-reference or secret field"
            )
    if set(source) - PUBLIC_TEXT_MANIFEST_SOURCE_FIELDS:
        raise PublicTextReleaseError(
            "derived manifest source contains disallowed fields"
        )
    serialized = data.decode("utf-8-sig")
    if re.search(
        r"https?://[^\s\"']+\.(?:bmp|gif|jpe?g|png|tiff?|webp)(?:[?\"']|$)",
        serialized,
        flags=re.IGNORECASE,
    ):
        raise PublicTextReleaseError("derived manifest contains a remote photo URL")
    release_profile = value.get("release_profile")
    if (
        not isinstance(release_profile, Mapping)
        or clean_text(release_profile.get("id")) != PUBLIC_TEXT_RELEASE_PROFILE
    ):
        raise PublicTextReleaseError("derived manifest release profile is invalid")
    embedding = value.get("embedding")
    vector_types = (
        embedding.get("vector_types") if isinstance(embedding, Mapping) else None
    )
    if not isinstance(vector_types, Mapping) or {
        "crop_image": int(vector_types.get("crop_image") or 0),
        "image": int(vector_types.get("image") or 0),
    } != {"crop_image": 0, "image": 0}:
        raise PublicTextReleaseError("derived manifest retained visual vectors")
    return {
        "profile": PUBLIC_TEXT_RELEASE_PROFILE,
        "source_fields": sorted(source),
        "remote_photo_references": 0,
        "secret_fields": 0,
        "visual_vectors": 0,
    }


def _sanitize_text_meta(meta: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(meta[key])
        for key in sorted(PUBLIC_TEXT_METADATA_FIELDS)
        if key in meta
    }


def _sanitize_manifest_source(source_manifest: Mapping[str, Any]) -> dict[str, Any]:
    source = source_manifest.get("source")
    if not isinstance(source, Mapping):
        return {}
    return {
        key: copy.deepcopy(source[key])
        for key in sorted(PUBLIC_TEXT_MANIFEST_SOURCE_FIELDS)
        if key in source
    }


def _selection_fingerprint(selected: Sequence[tuple[int, str]]) -> str:
    material = "\n".join(f"{position}\t{notice_id}" for position, notice_id in selected)
    return sha256_bytes((material + "\n").encode("utf-8"))


def _validate_text_meta(meta: Mapping[str, Any], *, position: int) -> str:
    notice_id = clean_text(meta.get("desertionNo"))
    if not notice_id:
        raise PublicTextReleaseError(f"text metadata row {position} has no desertionNo")
    if clean_text(meta.get("embedding_source")) != "public_notice_text":
        raise PublicTextReleaseError(
            f"text metadata row {position} has invalid embedding provenance"
        )

    expected_text = public_notice_text(meta)
    if not expected_text:
        raise PublicTextReleaseError(
            f"text metadata row {position} has empty canonical public text"
        )
    if clean_text(meta.get("desc_full")) != expected_text:
        raise PublicTextReleaseError(
            f"text metadata row {position} does not match canonical public text"
        )
    expected_hash = public_text_sha256(expected_text)
    if clean_text(meta.get("embedding_text_sha256")).lower() != expected_hash:
        raise PublicTextReleaseError(
            f"text metadata row {position} has an invalid public-text SHA-256"
        )
    return notice_id


def _load_faiss_module(faiss_module: Any | None) -> Any:
    if faiss_module is not None:
        return faiss_module
    try:
        return importlib.import_module("faiss")
    except ImportError as exc:
        raise PublicTextReleaseError(
            "faiss is required to derive the public-text-only release"
        ) from exc


def _deserialize_index(faiss_module: Any, data: bytes) -> Any:
    try:
        return faiss_module.deserialize_index(np.frombuffer(data, dtype=np.uint8))
    except Exception as exc:
        raise PublicTextReleaseError("source FAISS index cannot be read") from exc


def _serialize_index(faiss_module: Any, index: Any) -> bytes:
    try:
        value = faiss_module.serialize_index(index)
        return np.asarray(value, dtype=np.uint8).tobytes()
    except Exception as exc:
        raise PublicTextReleaseError(
            "derived FAISS index cannot be serialized"
        ) from exc


def _validate_source_index(
    index: Any, metas: Sequence[Mapping[str, Any]], faiss: Any
) -> tuple[int, int]:
    try:
        total = int(index.ntotal)
        dimension = int(index.d)
    except (AttributeError, TypeError, ValueError) as exc:
        raise PublicTextReleaseError("source FAISS index metadata is invalid") from exc
    if total != len(metas):
        raise PublicTextReleaseError(
            f"source index/metas mismatch: vectors={total} rows={len(metas)}"
        )
    if dimension != EXPECTED_CLIP_DIMENSION:
        raise PublicTextReleaseError(
            f"source index dimension must be {EXPECTED_CLIP_DIMENSION}, got {dimension}"
        )
    metric = getattr(index, "metric_type", None)
    expected_metric = getattr(faiss, "METRIC_L2", metric)
    if metric is not None and metric != expected_metric:
        raise PublicTextReleaseError("source FAISS index must use L2 distance")
    return total, dimension


def _reconstruct_vector(index: Any, position: int, dimension: int) -> np.ndarray:
    try:
        vector = np.asarray(index.reconstruct(position), dtype=np.float32).reshape(-1)
    except Exception as exc:
        raise PublicTextReleaseError(
            f"source FAISS vector {position} cannot be reconstructed"
        ) from exc
    if vector.shape != (dimension,):
        raise PublicTextReleaseError(
            f"source FAISS vector {position} has dimension {vector.size}, expected {dimension}"
        )
    if not np.isfinite(vector).all():
        raise PublicTextReleaseError(
            f"source FAISS vector {position} contains non-finite values"
        )
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise PublicTextReleaseError(
            f"source FAISS vector {position} has an invalid norm"
        )
    return vector.copy()


def _derived_manifest(
    source_manifest: Mapping[str, Any],
    *,
    source_manifest_sha256: str | None,
    source_index_sha256: str,
    source_metas_sha256: str,
    source_vectors: int,
    text_vectors: int,
    unique_notices: int,
    index_bytes: bytes,
    metas_bytes: bytes,
    release_profile_bytes: bytes,
    derivation_report_bytes: bytes,
) -> dict[str, Any]:
    source_section = _sanitize_manifest_source(source_manifest)
    original_embedding = source_manifest.get("embedding")
    original_vector_types = (
        copy.deepcopy(original_embedding.get("vector_types"))
        if isinstance(original_embedding, Mapping)
        else {}
    )
    readiness = copy.deepcopy(source_manifest.get("readiness"))
    if not isinstance(readiness, Mapping):
        readiness = {}
    readiness = dict(readiness)
    readiness.update(
        {
            "photo_quality_notice_coverage": 0.0,
            "forbidden_derived_field_occurrence_count": 0,
            "unique_ids_missing_public_text_count": 0,
        }
    )
    return {
        "schema_version": 2,
        "generated_at": clean_text(source_manifest.get("generated_at")),
        "release_profile": {
            "id": PUBLIC_TEXT_RELEASE_PROFILE,
            "purpose": PUBLIC_TEXT_RELEASE_PURPOSE,
            "legal_determination": False,
            "photos_included": False,
            "remote_photo_references_included": False,
            "visual_vectors_included": False,
            "photo_quality_included": False,
        },
        "source": source_section,
        "source_snapshot": {
            "manifest_sha256": source_manifest_sha256 or "",
            "index_sha256": source_index_sha256,
            "metas_sha256": source_metas_sha256,
            "vectors": source_vectors,
            "vector_types": original_vector_types,
        },
        "enrichment": {
            "policy": "excluded_from_public_text_only_release",
            "retained_crop_vectors": 0,
            "full_image_vectors": 0,
            "notices_with_photo_quality": 0,
        },
        "embedding": {
            "model": clean_text(
                original_embedding.get("model")
                if isinstance(original_embedding, Mapping)
                else ""
            )
            or "OpenAI CLIP ViT-B/32",
            "dimension": EXPECTED_CLIP_DIMENSION,
            "vectors": text_vectors,
            "vector_types": {
                "crop_image": 0,
                "image": 0,
                "text": text_vectors,
            },
        },
        "readiness": readiness,
        "artifacts": {
            "data/dog_faiss.index": {
                "size_bytes": len(index_bytes),
                "sha256": sha256_bytes(index_bytes),
            },
            "data/dog_metas.json": {
                "rows": text_vectors,
                "size_bytes": len(metas_bytes),
                "sha256": sha256_bytes(metas_bytes),
            },
            "data/release_profile.json": {
                "size_bytes": len(release_profile_bytes),
                "sha256": sha256_bytes(release_profile_bytes),
            },
            "data/public_text_release_report.json": {
                "size_bytes": len(derivation_report_bytes),
                "sha256": sha256_bytes(derivation_report_bytes),
            },
        },
        "counts": {
            "unique_notices": unique_notices,
            "text_vectors": text_vectors,
        },
        "limitations": [
            "This profile minimizes distributed photo-related artifacts; it is not a legal determination.",
            "Image uploads search public-notice text vectors rather than an image/crop vector index.",
            "Photo-quality scoring and remote notice-photo display are disabled.",
            "Notice status can change after the source snapshot and must be rechecked before contact.",
        ],
    }


def derive_public_text_release(
    source_index_bytes: bytes,
    source_metas_bytes: bytes,
    *,
    source_manifest_bytes: bytes | None = None,
    faiss_module: Any | None = None,
) -> PublicTextReleaseArtifacts:
    """Derive deterministic public-text-only artifacts from one source snapshot."""

    if not source_index_bytes:
        raise PublicTextReleaseError("source FAISS index is empty")
    if not source_metas_bytes:
        raise PublicTextReleaseError("source metadata is empty")

    faiss = _load_faiss_module(faiss_module)
    metas = _load_source_metas(source_metas_bytes)
    source_manifest = _load_source_manifest(source_manifest_bytes)
    source_index = _deserialize_index(faiss, source_index_bytes)
    source_total, dimension = _validate_source_index(source_index, metas, faiss)

    source_types: Counter[str] = Counter()
    all_notice_ids: set[str] = set()
    selected_rows: list[dict[str, Any]] = []
    selected_vectors: list[np.ndarray] = []
    selected_positions: list[tuple[int, str]] = []
    selected_ids: set[str] = set()

    for position, meta in enumerate(metas):
        row_type = clean_text(meta.get("type"))
        if row_type not in ALLOWED_SOURCE_ROW_TYPES:
            raise PublicTextReleaseError(
                f"source metadata row {position} has unsupported type {row_type!r}"
            )
        source_types[row_type] += 1
        notice_id = clean_text(meta.get("desertionNo"))
        if not notice_id:
            raise PublicTextReleaseError(
                f"source metadata row {position} has no desertionNo"
            )
        all_notice_ids.add(notice_id)
        if row_type != "text":
            continue

        validated_id = _validate_text_meta(meta, position=position)
        if validated_id in selected_ids:
            raise PublicTextReleaseError(
                f"source metadata has duplicate public-text row for {validated_id}"
            )
        sanitized = _sanitize_text_meta(meta)
        if set(sanitized) - PUBLIC_TEXT_METADATA_FIELDS:
            raise AssertionError(
                "public-text metadata sanitizer violated its allowlist"
            )
        selected_ids.add(validated_id)
        selected_rows.append(sanitized)
        selected_vectors.append(_reconstruct_vector(source_index, position, dimension))
        selected_positions.append((position, validated_id))

    if not selected_rows:
        raise PublicTextReleaseError(
            "source snapshot has no public-notice text vectors"
        )
    missing_text_ids = sorted(all_notice_ids - selected_ids)
    if missing_text_ids:
        raise PublicTextReleaseError(
            "source snapshot has notice IDs without a public-text vector: "
            f"count={len(missing_text_ids)}"
        )

    matrix = np.ascontiguousarray(np.vstack(selected_vectors), dtype=np.float32)
    try:
        derived_index = faiss.IndexFlatL2(dimension)
        derived_index.add(matrix)
    except Exception as exc:
        raise PublicTextReleaseError("derived FAISS index cannot be built") from exc
    if int(derived_index.ntotal) != len(selected_rows):
        raise PublicTextReleaseError("derived FAISS index and metadata counts differ")

    index_bytes = _serialize_index(faiss, derived_index)
    metas_bytes = deterministic_json_bytes(selected_rows)
    source_index_hash = sha256_bytes(source_index_bytes)
    source_metas_hash = sha256_bytes(source_metas_bytes)
    index_hash = sha256_bytes(index_bytes)
    metas_hash = sha256_bytes(metas_bytes)
    source_manifest_hash = (
        sha256_bytes(source_manifest_bytes)
        if source_manifest_bytes is not None
        else None
    )

    excluded_counts = {
        key: int(source_types.get(key, 0)) for key in ("image", "crop_image")
    }
    release_profile = {
        "schema_version": 1,
        "profile": PUBLIC_TEXT_RELEASE_PROFILE,
        "purpose": PUBLIC_TEXT_RELEASE_PURPOSE,
        "legal_determination": False,
        "runtime_policy": {
            "graph_overlay_enabled": False,
            "photo_quality_enabled": False,
            "remote_notice_photo_display_enabled": False,
            "visual_asset_routes_enabled": False,
        },
        "artifact_contract": {
            "allowed_vector_types": ["text"],
            "allowed_embedding_sources": ["public_notice_text"],
            "metadata_allowlist": sorted(PUBLIC_TEXT_METADATA_FIELDS),
            "photos_included": False,
            "remote_photo_references_included": False,
        },
        "source": {
            "index_sha256": source_index_hash,
            "metas_sha256": source_metas_hash,
            "manifest_sha256": source_manifest_hash or "",
            "vector_count": source_total,
        },
        "output": {
            "dimension": dimension,
            "index_sha256": index_hash,
            "metas_sha256": metas_hash,
            "unique_notice_count": len(selected_ids),
            "vector_count": len(selected_rows),
        },
    }
    release_profile_bytes = deterministic_json_bytes(release_profile)

    derivation_report = {
        "schema_version": 1,
        "profile": PUBLIC_TEXT_RELEASE_PROFILE,
        "purpose": PUBLIC_TEXT_RELEASE_PURPOSE,
        "legal_determination": False,
        "selection": {
            "rule": (
                "type=text; embedding_source=public_notice_text; canonical desc_full "
                "and embedding_text_sha256 required"
            ),
            "selection_sha256": _selection_fingerprint(selected_positions),
            "selected_vectors": len(selected_rows),
            "unique_notices": len(selected_ids),
            "excluded_vector_counts": excluded_counts,
            "metadata_allowlist": sorted(PUBLIC_TEXT_METADATA_FIELDS),
        },
        "source_artifacts": {
            "index": {
                "sha256": source_index_hash,
                "size_bytes": len(source_index_bytes),
                "vectors": source_total,
            },
            "metas": {
                "sha256": source_metas_hash,
                "size_bytes": len(source_metas_bytes),
                "rows": len(metas),
            },
            "snapshot_manifest": {
                "sha256": source_manifest_hash or "",
                "size_bytes": len(source_manifest_bytes or b""),
            },
        },
        "output_artifacts": {
            "index": {
                "sha256": index_hash,
                "size_bytes": len(index_bytes),
                "vectors": len(selected_rows),
                "dimension": dimension,
            },
            "metas": {
                "sha256": metas_hash,
                "size_bytes": len(metas_bytes),
                "rows": len(selected_rows),
            },
            "release_profile": {
                "sha256": sha256_bytes(release_profile_bytes),
                "size_bytes": len(release_profile_bytes),
            },
        },
        "validation": {
            "passed": True,
            "image_vectors_in_output": 0,
            "crop_vectors_in_output": 0,
            "remote_photo_reference_fields_in_output": 0,
            "unrecognized_metadata_fields_in_output": 0,
        },
    }
    derivation_report_bytes = deterministic_json_bytes(derivation_report)

    manifest = _derived_manifest(
        source_manifest,
        source_manifest_sha256=source_manifest_hash,
        source_index_sha256=source_index_hash,
        source_metas_sha256=source_metas_hash,
        source_vectors=source_total,
        text_vectors=len(selected_rows),
        unique_notices=len(selected_ids),
        index_bytes=index_bytes,
        metas_bytes=metas_bytes,
        release_profile_bytes=release_profile_bytes,
        derivation_report_bytes=derivation_report_bytes,
    )
    snapshot_manifest_bytes = deterministic_json_bytes(manifest)
    validate_public_text_snapshot_manifest(snapshot_manifest_bytes)

    # Decode our own output once before returning.  This catches accidental
    # non-JSON values or serializer regressions without trusting the ZIP step.
    for label, payload in (
        ("derived metadata", metas_bytes),
        ("derived snapshot manifest", snapshot_manifest_bytes),
        ("release profile", release_profile_bytes),
        ("derivation report", derivation_report_bytes),
    ):
        _load_json_bytes(payload, label=label)

    return PublicTextReleaseArtifacts(
        index_bytes=index_bytes,
        metas_bytes=metas_bytes,
        snapshot_manifest_bytes=snapshot_manifest_bytes,
        release_profile_bytes=release_profile_bytes,
        derivation_report_bytes=derivation_report_bytes,
        vector_count=len(selected_rows),
        unique_notice_count=len(selected_ids),
        dimension=dimension,
        source_vector_count=source_total,
        excluded_vector_counts=excluded_counts,
        source_index_sha256=source_index_hash,
        source_metas_sha256=source_metas_hash,
        index_sha256=index_hash,
        metas_sha256=metas_hash,
    )


def validate_public_text_release_payload(
    index_bytes: bytes,
    metas_bytes: bytes,
    release_profile_bytes: bytes,
    *,
    faiss_module: Any | None = None,
) -> dict[str, Any]:
    """Fail closed when an already-derived artifact violates the profile."""

    faiss = _load_faiss_module(faiss_module)
    metas = _load_source_metas(metas_bytes)
    profile = _load_json_bytes(release_profile_bytes, label="release profile")
    if not isinstance(profile, Mapping):
        raise PublicTextReleaseError("release profile must be an object")
    if clean_text(profile.get("profile")) != PUBLIC_TEXT_RELEASE_PROFILE:
        raise PublicTextReleaseError("release profile marker is invalid")
    index = _deserialize_index(faiss, index_bytes)
    total, dimension = _validate_source_index(index, metas, faiss)

    seen: set[str] = set()
    for position, meta in enumerate(metas):
        extra = set(meta) - PUBLIC_TEXT_METADATA_FIELDS
        if extra:
            raise PublicTextReleaseError(
                f"derived metadata row {position} contains disallowed fields"
            )
        if clean_text(meta.get("type")) != "text":
            raise PublicTextReleaseError(
                f"derived metadata row {position} is not a text vector"
            )
        notice_id = _validate_text_meta(meta, position=position)
        if notice_id in seen:
            raise PublicTextReleaseError(
                f"derived metadata has duplicate public-text row for {notice_id}"
            )
        seen.add(notice_id)
        _reconstruct_vector(index, position, dimension)

    output = profile.get("output")
    if not isinstance(output, Mapping):
        raise PublicTextReleaseError("release profile output contract is missing")
    if int(output.get("vector_count") or -1) != total:
        raise PublicTextReleaseError("release profile vector count is stale")
    if clean_text(output.get("index_sha256")) != sha256_bytes(index_bytes):
        raise PublicTextReleaseError("release profile index SHA-256 is stale")
    if clean_text(output.get("metas_sha256")) != sha256_bytes(metas_bytes):
        raise PublicTextReleaseError("release profile metadata SHA-256 is stale")
    return {
        "profile": PUBLIC_TEXT_RELEASE_PROFILE,
        "vectors": total,
        "unique_notices": len(seen),
        "dimension": dimension,
        "visual_vectors": 0,
        "remote_photo_reference_fields": 0,
    }


def validate_public_text_metadata_rows(
    metas: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate the runtime metadata half of the public-text-only contract."""

    if not metas:
        raise PublicTextReleaseError("public-text-only metadata is empty")
    seen: set[str] = set()
    for position, meta in enumerate(metas):
        if not isinstance(meta, Mapping):
            raise PublicTextReleaseError(
                f"public-text-only metadata row {position} is not an object"
            )
        extra = set(meta) - PUBLIC_TEXT_METADATA_FIELDS
        if extra:
            raise PublicTextReleaseError(
                f"public-text-only metadata row {position} contains disallowed fields"
            )
        if clean_text(meta.get("type")) != "text":
            raise PublicTextReleaseError(
                f"public-text-only metadata row {position} is not a text vector"
            )
        notice_id = _validate_text_meta(meta, position=position)
        if notice_id in seen:
            raise PublicTextReleaseError(
                f"public-text-only metadata has duplicate row for {notice_id}"
            )
        seen.add(notice_id)
    return {
        "profile": PUBLIC_TEXT_RELEASE_PROFILE,
        "vectors": len(metas),
        "unique_notices": len(seen),
        "visual_vectors": 0,
        "remote_photo_reference_fields": 0,
    }
