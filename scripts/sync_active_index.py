"""Synchronize a FAISS index with an authoritative active-notice cache.

Existing vectors whose notice IDs remain active are retained byte-for-byte
while their metadata is rebuilt from the fresh notice. Notices absent from the
fresh active set are removed. New notices receive original-image and public
notice text embeddings; an image failure never prevents a text-only result.

Heavy runtime dependencies are imported only by the CLI. Core synchronization
accepts injected encoders, downloaders, and quality evaluators for fast tests.
No VLM output or inferred personality/social traits are copied or generated.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote
from uuid import uuid4

import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.notice_metadata import (  # noqa: E402
    first_text,
    normalize_additional_notice_fields,
)
from app.notice_status import classify_notice  # noqa: E402
from app.public_image_download import (  # noqa: E402
    ALLOWED_PUBLIC_IMAGE_FORMATS as ALLOWED_PUBLIC_IMAGE_FORMATS,
    DEFAULT_ALLOWED_IMAGE_HOSTS,
    DEFAULT_IMAGE_MAX_BYTES as DEFAULT_IMAGE_MAX_BYTES,
    DEFAULT_IMAGE_MAX_PIXELS as DEFAULT_IMAGE_MAX_PIXELS,
    IMAGE_DOWNLOAD_CHUNK_BYTES as IMAGE_DOWNLOAD_CHUNK_BYTES,
    NONPUBLIC_IMAGE_HOST_SUFFIXES as NONPUBLIC_IMAGE_HOST_SUFFIXES,
    PublicImageDownloader,
    parse_public_image_hostname,
)


DATA_DIR = BASE_DIR / "data"
DEFAULT_INDEX = DATA_DIR / "dog_faiss.index"
DEFAULT_METAS = DATA_DIR / "dog_metas.json"
DEFAULT_MIN_ACTIVE_NOTICES = 1
DEFAULT_MAX_REMOVED_ID_FRACTION = 0.5
ID_KEYS = ("desertionNo", "desertion_no")
VECTOR_PROVENANCE_KEYS = (
    "embedding_source",
    "embedding_model",
    "embedding_version",
)

# These fields are generated or commonly used as unsupported behavioural
# guesses. Public source fields such as desc and safety_social_note remain.
FORBIDDEN_DERIVED_KEYS = frozenset(
    {
        "vlm_attrs",
        "vlm_desc",
        "vlm_attr_text",
        "merged_desc",
        "photo_advice",
        "personality",
        "temperament",
        "activity_level",
        "child_friendly",
        "pet_friendly",
        "aggression",
    }
)


class Encoder(Protocol):
    dimension: int

    def encode_image(self, image: Any) -> np.ndarray | None: ...

    def encode_text(self, text: str) -> np.ndarray | None: ...


ImageDownloader = Callable[[str], Any | None]
QualityEvaluator = Callable[[Any], Mapping[str, Any]]


@dataclass(frozen=True)
class SyncResult:
    vectors: np.ndarray
    metas: list[dict[str, Any]]
    report: dict[str, Any]


def clean_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def record_id(record: Mapping[str, Any]) -> str:
    for key in ID_KEYS:
        value = clean_text(record.get(key))
        if value:
            return value
    return ""


def _canonicalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    if isinstance(value, tuple):
        return [_canonicalize(item) for item in value]
    return copy.deepcopy(value)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            _canonicalize(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _useful_leaf_count(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return int(bool(value.strip()))
    if isinstance(value, Mapping):
        return sum(_useful_leaf_count(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_useful_leaf_count(item) for item in value)
    return 1


def load_fresh_cache(path: Path) -> tuple[Any, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict):
        records = payload.get("items")
    elif isinstance(payload, list):
        records = payload
    else:
        records = None
    if not isinstance(records, list):
        raise ValueError(
            "fresh cache must be a JSON list or an object containing an items list"
        )
    invalid = [
        index for index, item in enumerate(records) if not isinstance(item, dict)
    ]
    if invalid:
        preview = ", ".join(str(index) for index in invalid[:5])
        raise ValueError(
            f"fresh cache contains non-object records at indices {preview}"
        )
    validate_fresh_payload_contract(payload, records)
    return payload, records


def validate_fresh_payload_contract(
    payload: Any,
    records: Sequence[Mapping[str, Any]],
) -> None:
    """Reject wrapper metadata that makes an authoritative refresh ambiguous."""

    if not isinstance(payload, Mapping):
        return

    if "include_closed" in payload:
        include_closed = payload.get("include_closed")
        if not isinstance(include_closed, bool):
            raise ValueError("fresh cache include_closed must be a boolean")
        if include_closed:
            raise ValueError(
                "fresh cache include_closed=true is not allowed for active-only sync"
            )

    if "stats" not in payload:
        return
    stats = payload.get("stats")
    if not isinstance(stats, Mapping):
        raise ValueError("fresh cache stats must be an object")
    if "kept_items" not in stats:
        return

    kept_items = stats.get("kept_items")
    if (
        isinstance(kept_items, bool)
        or not isinstance(kept_items, int)
        or kept_items < 0
    ):
        raise ValueError("fresh cache stats.kept_items must be a nonnegative integer")
    if kept_items != len(records):
        raise ValueError(
            "fresh cache stats.kept_items does not match items length: "
            f"kept_items={kept_items} items={len(records)}"
        )


def load_metas(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError("existing metas must be a JSON list")
    invalid = [
        index for index, item in enumerate(payload) if not isinstance(item, dict)
    ]
    if invalid:
        preview = ", ".join(str(index) for index in invalid[:5])
        raise ValueError(
            f"existing metas contains non-object records at indices {preview}"
        )
    return payload


def parse_reference_date(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise argparse.ArgumentTypeError("reference date must be YYYY-MM-DD") from exc


def parse_nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a nonnegative integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a nonnegative integer")
    return parsed


def parse_fraction(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be between 0 and 1") from exc
    if not np.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return parsed


def reference_date_from_payload(payload: Any) -> datetime | None:
    if not isinstance(payload, Mapping):
        return None
    text = clean_text(payload.get("fetched_at"))
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def select_active_fresh_records(
    records: Sequence[Mapping[str, Any]],
    *,
    reference_date: datetime,
    include_unknown: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Select one deterministic authoritative active record per notice ID."""

    status_counts: Counter[str] = Counter()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing_id = 0

    for record in records:
        status = classify_notice(dict(record), reference_date=reference_date)
        status_counts[status] += 1
        dog_id = record_id(record)
        if not dog_id:
            missing_id += 1
            continue
        if status == "active" or (status == "unknown" and include_unknown):
            grouped[dog_id].append(copy.deepcopy(dict(record)))

    selected: dict[str, dict[str, Any]] = {}
    duplicate_rows = 0
    for dog_id in sorted(grouped):
        candidates = sorted(
            grouped[dog_id],
            key=lambda item: (
                -_useful_leaf_count(item),
                canonical_json_bytes(item),
            ),
        )
        selected[dog_id] = candidates[0]
        duplicate_rows += len(candidates) - 1

    stats = {
        "fresh_input_records": len(records),
        "fresh_missing_id_records": missing_id,
        "fresh_duplicate_active_id_records": duplicate_rows,
        "fresh_status_records": {
            key: status_counts.get(key, 0)
            for key in ("active", "closed", "expired", "unknown")
        },
        "active_unique_ids": len(selected),
        "include_unknown": include_unknown,
        "reference_date": reference_date.date().isoformat(),
    }
    return selected, stats


def evaluate_refresh_safety_gates(
    fresh_records: Sequence[Mapping[str, Any]],
    existing_metas: Sequence[Mapping[str, Any]],
    *,
    reference_date: datetime,
    include_unknown: bool,
    min_active_notices: int,
    max_removed_id_fraction: float,
) -> dict[str, Any]:
    """Fail closed before encoding when a refresh looks incomplete."""

    active, _ = select_active_fresh_records(
        fresh_records,
        reference_date=reference_date,
        include_unknown=include_unknown,
    )
    existing_ids = {record_id(meta) for meta in existing_metas if record_id(meta)}
    removed_ids = existing_ids - set(active)
    removed_fraction = len(removed_ids) / len(existing_ids) if existing_ids else 0.0
    active_passed = len(active) >= min_active_notices
    removed_passed = removed_fraction <= max_removed_id_fraction
    gate_report = {
        "policy": "fail_closed_before_encoding",
        "min_active_notices": {
            "configured_minimum": min_active_notices,
            "observed_active_unique_ids": len(active),
            "passed": active_passed,
        },
        "max_removed_id_fraction": {
            "configured_maximum": max_removed_id_fraction,
            "observed_removed_fraction": removed_fraction,
            "observed_removed_existing_unique_ids": len(removed_ids),
            "observed_existing_unique_ids": len(existing_ids),
            "passed": removed_passed,
        },
    }
    if not active_passed:
        raise ValueError(
            "fresh cache failed min-active-notices safety gate: "
            f"active={len(active)} minimum={min_active_notices}"
        )
    if not removed_passed:
        raise ValueError(
            "fresh cache failed max-removed-id-fraction safety gate: "
            f"removed={len(removed_ids)} existing={len(existing_ids)} "
            f"fraction={removed_fraction:.6f} "
            f"maximum={max_removed_id_fraction:.6f}"
        )
    return gate_report


def build_public_notice_text(record: Mapping[str, Any]) -> str:
    """Build CLIP text only from public notice facts, excluding breed/VLM."""

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


def sanitize_notice_metadata(record: Mapping[str, Any]) -> dict[str, Any]:
    """Create authoritative metadata without derived behavioural/VLM fields."""

    dog_id = record_id(record)
    if not dog_id:
        raise ValueError("notice metadata requires desertionNo")

    sanitized = {
        key: copy.deepcopy(value)
        for key, value in record.items()
        if key not in FORBIDDEN_DERIVED_KEYS
        and key not in {"type", "embedding_source", "desertion_no", "desc_full"}
    }
    sanitized.update(copy.deepcopy(normalize_additional_notice_fields(record)))
    sanitized["desertionNo"] = dog_id

    detail_url = first_text(record, "detail_url")
    if not detail_url:
        detail_url = (
            "https://www.animal.go.kr/front/awtis/public/publicDtl.do"
            f"?desertionNo={quote(dog_id)}&menuNo=1000000055"
        )
    sanitized["detail_url"] = detail_url
    sanitized["desc_full"] = build_public_notice_text(record)
    return sanitized


def validate_existing_vectors(
    vectors: Any, metas: Sequence[Mapping[str, Any]]
) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("existing vectors must be a 2-dimensional matrix")
    if array.shape[0] != len(metas):
        raise ValueError(
            f"index/metas mismatch: vectors={array.shape[0]} metas={len(metas)}"
        )
    if array.shape[1] <= 0:
        raise ValueError("index dimension must be positive")
    if not np.isfinite(array).all():
        raise ValueError("existing vectors contain non-finite values")
    return np.ascontiguousarray(array, dtype=np.float32)


def extract_index_vectors(index: Any) -> np.ndarray:
    try:
        total = int(index.ntotal)
        dimension = int(index.d)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("invalid FAISS index: missing ntotal or dimension") from exc
    if dimension <= 0:
        raise ValueError("FAISS index dimension must be positive")
    if total == 0:
        return np.empty((0, dimension), dtype=np.float32)

    rows: list[np.ndarray] = []
    for position in range(total):
        try:
            row = np.asarray(index.reconstruct(position), dtype=np.float32).reshape(-1)
        except Exception as exc:
            raise ValueError(
                f"FAISS index cannot reconstruct vector {position}"
            ) from exc
        if row.shape != (dimension,):
            raise ValueError(
                f"FAISS vector {position} has dimension {row.size}, "
                f"expected {dimension}"
            )
        rows.append(row)
    return np.ascontiguousarray(np.vstack(rows), dtype=np.float32)


def _validated_new_vector(
    value: Any, *, expected_dimension: int, label: str
) -> np.ndarray | None:
    if value is None:
        return None
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.shape != (expected_dimension,):
        raise ValueError(
            f"{label} encoder dimension mismatch: "
            f"got {vector.size}, expected {expected_dimension}"
        )
    if not np.isfinite(vector).all():
        raise ValueError(f"{label} encoder returned non-finite values")
    if float(np.linalg.norm(vector)) == 0.0:
        raise ValueError(f"{label} encoder returned a zero vector")
    return vector


def public_text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stored_image_source_url(meta: Mapping[str, Any]) -> str:
    return first_text(meta, "embedding_image_url", "image_url")


def _text_row_matches(meta: Mapping[str, Any], text: str) -> bool:
    if not text:
        return False
    expected_hash = public_text_sha256(text)
    stored_hash = clean_text(meta.get("embedding_text_sha256"))
    if stored_hash:
        return stored_hash == expected_hash
    return clean_text(meta.get("desc_full")) == text


def _warning_codes(stats: Mapping[str, Any]) -> list[str]:
    warnings: list[str] = []
    conditions = (
        ("fresh_missing_ids", stats.get("fresh_missing_id_records")),
        ("fresh_duplicate_ids", stats.get("fresh_duplicate_active_id_records")),
        ("image_vectors_refreshed", stats.get("refresh_image_required_unique_ids")),
        ("text_vectors_refreshed", stats.get("refresh_text_required_unique_ids")),
        ("new_image_failures", stats.get("new_image_failures")),
        ("new_text_failures", stats.get("new_text_failures")),
        ("refreshed_image_failures", stats.get("refreshed_image_failures")),
        ("refreshed_text_failures", stats.get("refreshed_text_failures")),
        ("new_ids_without_vectors", stats.get("new_ids_without_vectors")),
        (
            "refreshes_without_replacement_vectors",
            stats.get("refreshes_without_replacement_vectors"),
        ),
    )
    for code, count in conditions:
        if isinstance(count, int) and count > 0:
            warnings.append(code)
    return warnings


def synchronize_active_vectors(
    existing_vectors: Any,
    existing_metas: Sequence[Mapping[str, Any]],
    fresh_records: Sequence[Mapping[str, Any]],
    *,
    reference_date: datetime,
    encoder: Encoder | None,
    image_downloader: ImageDownloader | None,
    quality_evaluator: QualityEvaluator | None = None,
    include_unknown: bool = False,
    text_only: bool = False,
) -> SyncResult:
    """Return synchronized vectors, metadata, and a deterministic report."""

    source_vectors = validate_existing_vectors(existing_vectors, existing_metas)
    dimension = int(source_vectors.shape[1])
    active, fresh_stats = select_active_fresh_records(
        fresh_records,
        reference_date=reference_date,
        include_unknown=include_unknown,
    )

    existing_ids = {record_id(meta) for meta in existing_metas if record_id(meta)}
    new_ids = sorted(set(active) - existing_ids)
    removed_ids = sorted(existing_ids - set(active))
    if encoder is not None and int(encoder.dimension) != dimension:
        raise ValueError(
            "encoder/index dimension mismatch: "
            f"encoder={encoder.dimension} index={dimension}"
        )

    output_vectors: list[np.ndarray] = []
    output_metas: list[dict[str, Any]] = []
    retained_ids: set[str] = set()
    retained_types: Counter[str] = Counter()
    current_image_ids: set[str] = set()
    current_text_ids: set[str] = set()
    removed_stale_rows = 0
    removed_changed_image_rows = 0
    removed_changed_crop_rows = 0
    removed_changed_text_rows = 0
    removed_profile_excluded_image_rows = 0
    removed_unrecognized_rows = 0
    existing_missing_id_rows = 0

    for position, existing_meta in enumerate(existing_metas):
        dog_id = record_id(existing_meta)
        if not dog_id:
            existing_missing_id_rows += 1
            continue
        fresh_record = active.get(dog_id)
        if fresh_record is None:
            removed_stale_rows += 1
            continue

        latest = sanitize_notice_metadata(fresh_record)
        row_type = clean_text(existing_meta.get("type"))
        if text_only and row_type in {"image", "crop_image"}:
            removed_profile_excluded_image_rows += 1
            continue
        fresh_image_url = first_text(latest, "image_url")
        fresh_text = clean_text(latest.get("desc_full"))
        if row_type == "text":
            if not _text_row_matches(existing_meta, fresh_text):
                removed_changed_text_rows += 1
                continue
        elif row_type in {"image", "crop_image"}:
            stored_image_url = _stored_image_source_url(existing_meta)
            if (
                not stored_image_url
                or not fresh_image_url
                or stored_image_url != fresh_image_url
            ):
                if row_type == "image":
                    removed_changed_image_rows += 1
                else:
                    removed_changed_crop_rows += 1
                continue
        else:
            removed_unrecognized_rows += 1
            continue

        latest["type"] = row_type
        for key in VECTOR_PROVENANCE_KEYS:
            if key in existing_meta:
                latest[key] = copy.deepcopy(existing_meta[key])
        if row_type == "text":
            latest["embedding_source"] = (
                clean_text(latest.get("embedding_source")) or "public_notice_text"
            )
            latest["embedding_text_sha256"] = public_text_sha256(fresh_text)
            current_text_ids.add(dog_id)
        else:
            latest["embedding_image_url"] = fresh_image_url
            if not clean_text(latest.get("embedding_source")):
                latest["embedding_source"] = (
                    "full_image" if row_type == "image" else "dog_crop"
                )
            if row_type == "image":
                current_image_ids.add(dog_id)

        output_vectors.append(source_vectors[position].copy())
        output_metas.append(latest)
        retained_ids.add(dog_id)
        retained_types[row_type] += 1

    existing_active_ids = existing_ids.intersection(active)
    refresh_image_ids = (
        set()
        if text_only
        else {
            dog_id
            for dog_id in existing_active_ids
            if dog_id not in current_image_ids
        }
    )
    refresh_text_ids = {
        dog_id for dog_id in existing_active_ids if dog_id not in current_text_ids
    }
    requested_image_ids = (
        set() if text_only else set(new_ids).union(refresh_image_ids)
    )
    requested_text_ids = set(new_ids).union(refresh_text_ids)
    generation_ids = sorted(requested_image_ids.union(requested_text_ids))
    if generation_ids and encoder is None:
        raise ValueError(
            "an encoder is required when vectors must be added or refreshed"
        )

    generated: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    quality_failures = 0
    new_ids_without_vectors: list[str] = []
    refreshes_without_vectors: list[str] = []
    refreshed_success_ids: set[str] = set()

    for dog_id in generation_ids:
        record = active[dog_id]
        common_meta = sanitize_notice_metadata(record)
        added_for_id = 0
        category = "new" if dog_id in new_ids else "refreshed"

        if dog_id in requested_image_ids:
            image = None
            image_url = first_text(common_meta, "image_url")
            if not image_url:
                failures[f"{category}_image_missing_urls"] += 1
            elif image_downloader is None:
                failures[f"{category}_image_download_failures"] += 1
            else:
                try:
                    image = image_downloader(image_url)
                except Exception:
                    image = None
                if image is None:
                    failures[f"{category}_image_download_failures"] += 1

            if image is not None:
                if quality_evaluator is not None:
                    try:
                        quality = dict(quality_evaluator(image))
                    except Exception:
                        quality = {}
                        quality_failures += 1
                    if quality:
                        attrs = common_meta.get("image_attrs")
                        image_attrs = (
                            copy.deepcopy(dict(attrs))
                            if isinstance(attrs, Mapping)
                            else {}
                        )
                        image_attrs.update(copy.deepcopy(quality))
                        common_meta["image_attrs"] = image_attrs
                        if "photo_quality_score" not in quality:
                            quality_failures += 1

                try:
                    raw_image_vector = encoder.encode_image(image) if encoder else None
                except Exception:
                    raw_image_vector = None
                image_vector = _validated_new_vector(
                    raw_image_vector,
                    expected_dimension=dimension,
                    label=f"image:{dog_id}",
                )
                if image_vector is None:
                    failures[f"{category}_image_encoding_failures"] += 1
                else:
                    image_meta = copy.deepcopy(common_meta)
                    image_meta["type"] = "image"
                    image_meta["embedding_source"] = "full_image"
                    image_meta["embedding_image_url"] = image_url
                    output_vectors.append(image_vector)
                    output_metas.append(image_meta)
                    generated[f"{category}_image_vectors"] += 1
                    added_for_id += 1

        if dog_id in requested_text_ids:
            text = clean_text(common_meta.get("desc_full"))
            if text:
                try:
                    raw_text_vector = encoder.encode_text(text) if encoder else None
                except Exception as exc:
                    failures[f"{category}_text_failures"] += 1
                    raise ValueError(
                        "public text embedding failed for notice "
                        f"{dog_id}: {exc.__class__.__name__}"
                    ) from None
                text_vector = _validated_new_vector(
                    raw_text_vector,
                    expected_dimension=dimension,
                    label=f"text:{dog_id}",
                )
                if text_vector is None:
                    failures[f"{category}_text_failures"] += 1
                    raise ValueError(
                        "public text embedding failed for notice "
                        f"{dog_id}: encoder returned no vector"
                    )
                else:
                    text_meta = copy.deepcopy(common_meta)
                    text_meta["type"] = "text"
                    text_meta["embedding_source"] = "public_notice_text"
                    text_meta["embedding_text_sha256"] = public_text_sha256(text)
                    output_vectors.append(text_vector)
                    output_metas.append(text_meta)
                    generated[f"{category}_text_vectors"] += 1
                    added_for_id += 1
            else:
                failures[f"{category}_text_failures"] += 1
                raise ValueError(
                    f"public text embedding failed for notice {dog_id}: "
                    "public notice text is empty"
                )

        if category == "new" and added_for_id == 0:
            new_ids_without_vectors.append(dog_id)
        elif category == "refreshed":
            if added_for_id:
                refreshed_success_ids.add(dog_id)
            else:
                refreshes_without_vectors.append(dog_id)

    if output_vectors:
        vectors = np.ascontiguousarray(np.vstack(output_vectors), dtype=np.float32)
    else:
        vectors = np.empty((0, dimension), dtype=np.float32)
    if vectors.shape[0] != len(output_metas):
        raise AssertionError("internal index/metas synchronization failure")

    output_ids = {record_id(meta) for meta in output_metas if record_id(meta)}
    output_text_ids = {
        record_id(meta)
        for meta in output_metas
        if clean_text(meta.get("type")) == "text" and record_id(meta)
    }
    active_ids_without_vectors = sorted(set(active) - output_ids)
    active_ids_without_public_text_vectors = sorted(set(active) - output_text_ids)
    if active_ids_without_vectors:
        raise ValueError(
            "active notices lost all vectors during synchronization: "
            + ", ".join(active_ids_without_vectors[:20])
        )
    if active_ids_without_public_text_vectors:
        raise ValueError(
            "active notices are missing public text vectors after synchronization: "
            + ", ".join(active_ids_without_public_text_vectors[:20])
        )

    removed_existing_vectors = (
        existing_missing_id_rows
        + removed_stale_rows
        + removed_changed_image_rows
        + removed_changed_crop_rows
        + removed_changed_text_rows
        + removed_profile_excluded_image_rows
        + removed_unrecognized_rows
    )
    new_image_failures = (
        failures["new_image_missing_urls"]
        + failures["new_image_download_failures"]
        + failures["new_image_encoding_failures"]
    )
    refreshed_image_failures = (
        failures["refreshed_image_missing_urls"]
        + failures["refreshed_image_download_failures"]
        + failures["refreshed_image_encoding_failures"]
    )
    stats: dict[str, Any] = {
        **fresh_stats,
        "existing_vectors": int(source_vectors.shape[0]),
        "existing_unique_ids": len(existing_ids),
        "existing_missing_id_rows": existing_missing_id_rows,
        "retained_vectors": sum(retained_types.values()),
        "retained_unique_ids": len(retained_ids),
        "retained_image_vectors": retained_types["image"],
        "retained_crop_vectors": retained_types["crop_image"],
        "retained_text_vectors": retained_types["text"],
        "removed_existing_vectors": removed_existing_vectors,
        "removed_stale_vectors": removed_stale_rows,
        "removed_changed_image_vectors": removed_changed_image_rows,
        "removed_changed_crop_vectors": removed_changed_crop_rows,
        "removed_changed_text_vectors": removed_changed_text_rows,
        "removed_profile_excluded_image_vectors": (
            removed_profile_excluded_image_rows
        ),
        "removed_unrecognized_vectors": removed_unrecognized_rows,
        "removed_existing_unique_ids": len(removed_ids),
        "refresh_required_unique_ids": len(refresh_image_ids.union(refresh_text_ids)),
        "refresh_image_required_unique_ids": len(refresh_image_ids),
        "refresh_text_required_unique_ids": len(refresh_text_ids),
        "refreshed_unique_ids": len(refreshed_success_ids),
        "refreshed_image_vectors": generated["refreshed_image_vectors"],
        "refreshed_text_vectors": generated["refreshed_text_vectors"],
        "refreshed_image_missing_urls": failures["refreshed_image_missing_urls"],
        "refreshed_image_download_failures": failures[
            "refreshed_image_download_failures"
        ],
        "refreshed_image_encoding_failures": failures[
            "refreshed_image_encoding_failures"
        ],
        "refreshed_image_failures": refreshed_image_failures,
        "refreshed_text_failures": failures["refreshed_text_failures"],
        "refreshes_without_replacement_vectors": len(refreshes_without_vectors),
        "refreshes_without_replacement_vectors_sample": (
            refreshes_without_vectors[:20]
        ),
        "new_unique_ids": len(new_ids),
        "new_image_vectors": generated["new_image_vectors"],
        "new_text_vectors": generated["new_text_vectors"],
        "new_image_missing_urls": failures["new_image_missing_urls"],
        "new_image_download_failures": failures["new_image_download_failures"],
        "new_image_encoding_failures": failures["new_image_encoding_failures"],
        "new_image_failures": new_image_failures,
        "new_text_failures": failures["new_text_failures"],
        "photo_quality_failures": quality_failures,
        "new_ids_without_vectors": len(new_ids_without_vectors),
        "new_ids_without_vectors_sample": new_ids_without_vectors[:20],
        "active_ids_without_vectors": len(active_ids_without_vectors),
        "active_ids_without_public_text_vectors": len(
            active_ids_without_public_text_vectors
        ),
        "output_vectors": int(vectors.shape[0]),
        "output_unique_ids": len(output_ids),
        "dimension": dimension,
    }
    report = {
        "schema_version": 1,
        "release_profile": "public-text-only-v1" if text_only else "full",
        "policy": {
            "existing_active_vectors": (
                "retained_only_when_embedding_source_provenance_matches"
            ),
            "notice_metadata": "fresh_cache_authoritative",
            "missing_fresh_ids": "removed",
            "changed_text": "old_text_row_removed_and_reembedded",
            "changed_image_url": (
                "old_image_and_crop_rows_removed; original_image_reembedded"
            ),
            "unchanged_crop": "retained; crops_are_never_regenerated",
            "new_image_source": "original_notice_image",
            "new_text_source": "public_notice_fields_without_reported_breed",
            "image_failure": "text_only_allowed_when_public_text_vector_succeeds",
            "public_text_failure": "abort_without_writing_outputs",
            "active_notice_vector_coverage": "at_least_one_vector_and_one_text_vector",
            "vlm_or_personality_inference": "prohibited",
            "visual_vectors": "excluded" if text_only else "retained_or_refreshed",
        },
        "stats": stats,
        "warnings": _warning_codes(stats),
        "content_hashes": {
            "vectors_float32_sha256": hashlib.sha256(
                vectors.tobytes(order="C")
            ).hexdigest(),
            "metas_canonical_sha256": canonical_sha256(output_metas),
        },
    }
    return SyncResult(vectors=vectors, metas=output_metas, report=report)


class ClipEncoder:
    """Lazy runtime wrapper matching the repository's current CLIP contract."""

    def __init__(self, model_name: str = "ViT-B/32", device: str = "auto"):
        clip = importlib.import_module("clip")
        torch = importlib.import_module("torch")
        resolved_device = (
            "cuda"
            if device == "auto" and bool(torch.cuda.is_available())
            else ("cpu" if device == "auto" else device)
        )
        model, preprocess = clip.load(model_name, device=resolved_device)
        self._clip = clip
        self._torch = torch
        self._model = model
        self._preprocess = preprocess
        self._device = resolved_device
        projection = getattr(model, "text_projection", None)
        shape = getattr(projection, "shape", ())
        self.dimension = int(shape[-1]) if shape else 512
        self.model_name = model_name

    def encode_image(self, image: Any) -> np.ndarray | None:
        tensor = self._preprocess(image.convert("RGB")).unsqueeze(0).to(self._device)
        with self._torch.no_grad():
            vector = self._model.encode_image(tensor)
            vector = vector / vector.norm(dim=-1, keepdim=True)
        return vector.cpu().numpy()[0]

    def encode_text(self, text: str) -> np.ndarray | None:
        tokens = self._clip.tokenize([text], truncate=True).to(self._device)
        with self._torch.no_grad():
            vector = self._model.encode_text(tokens)
            vector = vector / vector.norm(dim=-1, keepdim=True)
        return vector.cpu().numpy()[0]


def default_quality_evaluator(image: Any) -> Mapping[str, Any]:
    module = importlib.import_module("scripts.enrich_image_crops")
    return module.compute_photo_quality(image)


def safe_read_faiss_index(path: Path, read_index: Callable[[str], Any]) -> Any:
    try:
        return read_index(str(path))
    except RuntimeError as exc:
        if "Illegal byte sequence" not in str(exc):
            raise
        safe_path = Path(tempfile.gettempdir()) / f"sync-read-{uuid4().hex}.index"
        try:
            shutil.copyfile(path, safe_path)
            return read_index(str(safe_path))
        finally:
            safe_path.unlink(missing_ok=True)


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    content = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_index_atomic(
    index: Any,
    path: Path,
    write_index: Callable[[Any, str], None],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    safe_temporary: Path | None = None
    try:
        try:
            write_index(index, str(temporary))
        except RuntimeError as exc:
            if "Illegal byte sequence" not in str(exc):
                raise
            safe_temporary = (
                Path(tempfile.gettempdir()) / f"sync-write-{uuid4().hex}.index"
            )
            write_index(index, str(safe_temporary))
            shutil.copyfile(safe_temporary, temporary)
        with temporary.open("rb+") as file:
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        if safe_temporary is not None:
            safe_temporary.unlink(missing_ok=True)


def validate_output_paths(
    *,
    input_paths: Sequence[Path],
    output_paths: Sequence[Path],
) -> None:
    resolved_inputs = {path.resolve() for path in input_paths}
    resolved_outputs = [path.resolve() for path in output_paths]
    if len(set(resolved_outputs)) != len(resolved_outputs):
        raise ValueError("index, metas, and report outputs must use distinct paths")
    collisions = resolved_inputs.intersection(resolved_outputs)
    if collisions:
        raise ValueError("output paths must not overwrite input files")


def artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retain vectors for current active notices, remove stale IDs, and "
            "embed only new notices."
        )
    )
    parser.add_argument("--fresh-cache", type=Path, required=True)
    parser.add_argument("--existing-index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--existing-metas", type=Path, default=DEFAULT_METAS)
    parser.add_argument("--index-out", type=Path, required=True)
    parser.add_argument("--metas-out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--reference-date", type=parse_reference_date)
    parser.add_argument("--include-unknown", action="store_true")
    parser.add_argument(
        "--min-active-notices",
        type=parse_nonnegative_int,
        default=DEFAULT_MIN_ACTIVE_NOTICES,
        help=(
            "Abort before encoding when the fresh cache contains fewer active "
            "unique notices than this value."
        ),
    )
    parser.add_argument(
        "--max-removed-id-fraction",
        type=parse_fraction,
        default=DEFAULT_MAX_REMOVED_ID_FRACTION,
        help=(
            "Abort before encoding when the fraction of existing unique notice "
            "IDs absent from the active fresh set exceeds this value."
        ),
    )
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--text-only",
        action="store_true",
        help=(
            "Do not retain, download, or embed notice images. Produce an "
            "intermediate index containing public-notice text vectors only."
        ),
    )
    parser.add_argument("--image-timeout", type=float, default=15.0)
    parser.add_argument(
        "--allowed-image-host",
        action="append",
        type=parse_public_image_hostname,
        dest="allowed_image_hosts",
        help=(
            "Exact public hostname allowed for image downloads. Repeat for multiple "
            "hosts. Defaults to openapi.animal.go.kr. Redirects are not followed."
        ),
    )
    return parser.parse_args(argv)


def run(
    args: argparse.Namespace,
    *,
    faiss_module: Any | None = None,
    encoder: Encoder | None = None,
    image_downloader: ImageDownloader | None = None,
    quality_evaluator: QualityEvaluator | None = None,
) -> dict[str, Any]:
    input_paths = [
        args.fresh_cache,
        args.existing_index,
        args.existing_metas,
    ]
    output_paths = [args.index_out, args.metas_out, args.report_out]
    validate_output_paths(input_paths=input_paths, output_paths=output_paths)

    fresh_payload, fresh_records = load_fresh_cache(args.fresh_cache)
    existing_metas = load_metas(args.existing_metas)
    reference_date = args.reference_date or reference_date_from_payload(fresh_payload)
    if reference_date is None:
        raise ValueError(
            "--reference-date is required when fresh cache fetched_at is missing "
            "or invalid"
        )

    safety_gates = evaluate_refresh_safety_gates(
        fresh_records,
        existing_metas,
        reference_date=reference_date,
        include_unknown=bool(args.include_unknown),
        min_active_notices=args.min_active_notices,
        max_removed_id_fraction=args.max_removed_id_fraction,
    )

    faiss = faiss_module or importlib.import_module("faiss")
    existing_index = safe_read_faiss_index(
        args.existing_index,
        faiss.read_index,
    )
    metric = getattr(existing_index, "metric_type", None)
    expected_metric = getattr(faiss, "METRIC_L2", metric)
    if metric is not None and metric != expected_metric:
        raise ValueError("existing FAISS index must use L2 distance")
    existing_vectors = extract_index_vectors(existing_index)

    runtime_encoder = encoder or ClipEncoder(args.clip_model, args.device)
    text_only = bool(args.text_only)
    allowed_image_hosts = (
        ()
        if text_only
        else tuple(args.allowed_image_hosts or DEFAULT_ALLOWED_IMAGE_HOSTS)
    )
    downloader = (
        None
        if text_only
        else image_downloader
        or PublicImageDownloader(
            args.image_timeout,
            allowed_hosts=allowed_image_hosts,
        )
    )
    evaluator = (
        None
        if text_only
        else quality_evaluator or default_quality_evaluator
    )
    result = synchronize_active_vectors(
        existing_vectors,
        existing_metas,
        fresh_records,
        reference_date=reference_date,
        encoder=runtime_encoder,
        image_downloader=downloader,
        quality_evaluator=evaluator,
        include_unknown=bool(args.include_unknown),
        text_only=text_only,
    )

    output_index = faiss.IndexFlatL2(result.vectors.shape[1])
    if result.vectors.shape[0]:
        output_index.add(result.vectors)
    if int(output_index.ntotal) != len(result.metas):
        raise ValueError("generated FAISS index and metadata row counts differ")
    if int(output_index.d) != result.vectors.shape[1]:
        raise ValueError("generated FAISS index dimension differs from vectors")

    write_index_atomic(output_index, args.index_out, faiss.write_index)
    write_json_atomic(args.metas_out, result.metas)

    report = copy.deepcopy(result.report)
    report["safety_gates"] = safety_gates
    report["source"] = {
        "fresh_fetched_at": (
            clean_text(fresh_payload.get("fetched_at"))
            if isinstance(fresh_payload, Mapping)
            else ""
        ),
        "reference_date": reference_date.date().isoformat(),
        "allowed_image_hosts": list(allowed_image_hosts),
        "image_redirects_followed": False,
    }
    report["artifacts"] = {
        "inputs": {
            "fresh_cache": artifact_record(args.fresh_cache),
            "existing_index": artifact_record(args.existing_index),
            "existing_metas": artifact_record(args.existing_metas),
        },
        "outputs": {
            "index": artifact_record(args.index_out),
            "metas": artifact_record(args.metas_out),
        },
    }
    write_json_atomic(args.report_out, report)

    print(
        json.dumps(
            {
                "output_vectors": report["stats"]["output_vectors"],
                "retained_vectors": report["stats"]["retained_vectors"],
                "refreshed_unique_ids": report["stats"]["refreshed_unique_ids"],
                "refreshed_image_vectors": report["stats"]["refreshed_image_vectors"],
                "refreshed_text_vectors": report["stats"]["refreshed_text_vectors"],
                "new_unique_ids": report["stats"]["new_unique_ids"],
                "release_profile": report["release_profile"],
                "warnings": report["warnings"],
                "report_sha256": sha256_file(args.report_out),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        run(args)
    except (OSError, ValueError, json.JSONDecodeError, ImportError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
