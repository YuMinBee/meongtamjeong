"""Build a deterministic contest-readiness report for vector metadata."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import os
import sys
from collections import Counter
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
from uuid import uuid4


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DEFAULT_METAS_PATH = DATA_DIR / "dog_metas.json"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.notice_status import (  # noqa: E402
    classify_notice,
    notice_end_date,
)


ID_KEYS = ("desertionNo", "desertion_no", "dog_id")
UNKNOWN_TEXT = {"", "-", "--", "unknown", "none", "null", "n/a", "na"}
REGION_KEYS = (
    "region",
    "region_name",
    "sido",
    "sido_name",
    "upr_name",
    "uprNm",
    "org_name",
    "orgNm",
    "care_addr",
    "careAddr",
    "happen_place",
    "happenPlace",
)
SHELTER_KEYS = ("care_name", "careNm", "shelter", "shelter_name")
LAST_VERIFIED_KEYS = (
    "last_verified_at",
    "last_verified",
    "verified_at",
    "fetched_at",
)
FRESHNESS_ISSUE_CODES = {
    "missing": "freshness_missing",
    "invalid": "freshness_invalid",
    "future": "freshness_future",
    "stale": "freshness_stale",
}
EXPECTED_CLIP_DIMENSION = 512
CLIP_NORM_ABS_TOLERANCE = 0.05
ALLOWED_ROW_PROVENANCE = {
    "image": "full_image",
    "crop_image": "dog_crop",
    "text": "public_notice_text",
}
FORBIDDEN_DERIVED_KEYS = frozenset(
    {
        "activity_level",
        "aggression",
        "child_friendly",
        "merged_desc",
        "personality",
        "pet_friendly",
        "photo_advice",
        "temperament",
        "vlm_attr_text",
        "vlm_attributes",
        "vlm_attrs",
        "vlm_desc",
    }
)
ARTIFACT_SAMPLE_LIMIT = 20


def is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in UNKNOWN_TEXT
    if isinstance(value, float) and math.isnan(value):
        return False
    if isinstance(value, Mapping):
        return bool(value) and any(is_present(item) for item in value.values())
    if isinstance(value, (list, tuple, set, frozenset)):
        return bool(value) and any(is_present(item) for item in value)
    return True


def record_id(record: Mapping[str, Any]) -> str:
    for key in ID_KEYS:
        value = record.get(key)
        if is_present(value):
            return str(value).strip()
    return ""


def load_records(
    path: Path, *, allow_items_object: bool
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata: dict[str, Any] = {}
    if isinstance(payload, list):
        records = payload
        root_kind = "list"
    elif allow_items_object and isinstance(payload, dict):
        records = payload.get("items")
        root_kind = "items_object"
        metadata["fetched_at"] = payload.get("fetched_at")
    else:
        records = None
        root_kind = type(payload).__name__
    if not isinstance(records, list):
        expected = "list or object with items list" if allow_items_object else "list"
        raise ValueError(f"expected JSON {expected}")
    if any(not isinstance(item, dict) for item in records):
        raise ValueError("all records must be JSON objects")
    metadata["root_kind"] = root_kind
    return records, metadata


def merge_non_empty(
    target: Mapping[str, Any], source: Mapping[str, Any]
) -> dict[str, Any]:
    merged = copy.deepcopy(dict(target))
    for key, value in source.items():
        if key in ID_KEYS or key == "type" or not is_present(value):
            continue
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = merge_non_empty(current, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def overlay_cache(
    metas: Sequence[Mapping[str, Any]],
    cache: Sequence[Mapping[str, Any]],
    fetched_at: Any = None,
) -> tuple[list[dict[str, Any]], int, int]:
    cache_by_id: dict[str, dict[str, Any]] = {}
    for record in cache:
        dog_id = record_id(record)
        if not dog_id:
            continue
        prepared = copy.deepcopy(dict(record))
        if is_present(fetched_at) and not any(
            is_present(prepared.get(key)) for key in LAST_VERIFIED_KEYS
        ):
            prepared["last_verified_at"] = fetched_at
        cache_by_id[dog_id] = merge_non_empty(cache_by_id.get(dog_id, {}), prepared)

    matched_rows = 0
    matched_ids: set[str] = set()
    effective: list[dict[str, Any]] = []
    for meta in metas:
        copied = copy.deepcopy(dict(meta))
        dog_id = record_id(meta)
        cached = cache_by_id.get(dog_id)
        if cached is not None:
            copied = merge_non_empty(copied, cached)
            matched_rows += 1
            matched_ids.add(dog_id)
        effective.append(copied)
    return effective, matched_rows, len(matched_ids)


def _has_any(record: Mapping[str, Any], keys: Sequence[str]) -> bool:
    return any(is_present(record.get(key)) for key in keys)


def _has_photo_quality(record: Mapping[str, Any]) -> bool:
    candidates = [record.get("photo_quality_score")]
    for key in ("image_attrs", "vlm_attrs"):
        nested = record.get(key)
        if isinstance(nested, Mapping):
            candidates.extend(
                [
                    nested.get("photo_quality_score"),
                    nested.get("crop_photo_quality_score"),
                ]
            )
    for value in candidates:
        if isinstance(value, bool):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number) and 0 <= number <= 1:
            return True
    return False


def unique_notice_records(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse vector rows into one effective record per known notice ID."""
    unique_by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        dog_id = record_id(record)
        if not dog_id:
            continue
        unique_by_id[dog_id] = merge_non_empty(
            unique_by_id.get(dog_id, {}),
            record,
        )
        if not record_id(unique_by_id[dog_id]):
            unique_by_id[dog_id]["dog_id"] = dog_id
    return list(unique_by_id.values())


def _status_distribution(
    records: Sequence[Mapping[str, Any]], reference_date: datetime
) -> dict[str, int]:
    statuses = Counter(
        classify_notice(dict(item), reference_date=reference_date) for item in records
    )
    return {
        key: statuses.get(key, 0) for key in ("active", "closed", "expired", "unknown")
    }


def coverage(
    records: Sequence[Mapping[str, Any]], reference_date: datetime
) -> dict[str, dict[str, float | int]]:
    total = len(records)
    counts = {
        "status": sum(
            1
            for item in records
            if classify_notice(dict(item), reference_date=reference_date) != "unknown"
        ),
        "notice_end": sum(1 for item in records if notice_end_date(dict(item))),
        "region": sum(1 for item in records if _has_any(item, REGION_KEYS)),
        "shelter": sum(1 for item in records if _has_any(item, SHELTER_KEYS)),
        "last_verified": sum(
            1 for item in records if _has_any(item, LAST_VERIFIED_KEYS)
        ),
        "photo_quality": sum(1 for item in records if _has_photo_quality(item)),
    }
    return {
        key: {
            "count": count,
            "ratio": round(count / total, 6) if total else 0.0,
        }
        for key, count in counts.items()
    }


def analyze_records(
    records: Sequence[Mapping[str, Any]], reference_date: datetime
) -> dict[str, Any]:
    ids = [record_id(item) for item in records]
    known_ids = [dog_id for dog_id in ids if dog_id]
    id_counts = Counter(known_ids)
    unique_records = unique_notice_records(records)
    return {
        "row_count": len(records),
        "unique_id_count": len(id_counts),
        "missing_id_count": len(records) - len(known_ids),
        "duplicate_id_count": sum(count - 1 for count in id_counts.values()),
        "duplicate_unique_id_count": sum(count > 1 for count in id_counts.values()),
        "coverage": coverage(records, reference_date),
        "status_distribution": _status_distribution(records, reference_date),
        "unique_notices": {
            "count": len(unique_records),
            "coverage": coverage(unique_records, reference_date),
            "status_distribution": _status_distribution(unique_records, reference_date),
        },
    }


def _parse_verification_timestamp(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, time.min)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith(("Z", "z")):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _verification_timestamp_value(record: Mapping[str, Any]) -> Any:
    return next(
        (record.get(key) for key in LAST_VERIFIED_KEYS if is_present(record.get(key))),
        None,
    )


def analyze_freshness(
    records: Sequence[Mapping[str, Any]],
    reference_date: datetime,
    max_age_days: Optional[int],
) -> dict[str, Any]:
    """Classify last-verification dates once per unique notice."""
    records_by_id: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        dog_id = record_id(record)
        if dog_id:
            records_by_id.setdefault(dog_id, []).append(record)
    counts = Counter(
        {
            "missing": 0,
            "invalid": 0,
            "future": 0,
            "stale": 0,
            "fresh": 0,
        }
    )
    parsed_dates: list[date] = []
    for notice_rows in records_by_id.values():
        values = []
        for record in notice_rows:
            value = _verification_timestamp_value(record)
            if value is not None:
                values.append(value)
        if not values:
            counts["missing"] += 1
            continue
        valid_dates = [
            parsed.date()
            for value in values
            if (parsed := _parse_verification_timestamp(value)) is not None
        ]
        if not valid_dates:
            counts["invalid"] += 1
            continue
        verified_date = max(valid_dates)
        parsed_dates.append(verified_date)
        age_days = (reference_date.date() - verified_date).days
        if age_days < 0:
            counts["future"] += 1
        elif max_age_days is not None and age_days > max_age_days:
            counts["stale"] += 1
        else:
            counts["fresh"] += 1

    return {
        "enabled": max_age_days is not None,
        "max_age_days": max_age_days,
        "reference_date": reference_date.date().isoformat(),
        "unique_notice_count": len(records_by_id),
        "timestamp_present_count": len(records_by_id) - counts["missing"],
        "parsed_count": len(parsed_dates),
        "fresh_count": counts["fresh"],
        "stale_count": counts["stale"],
        "missing_count": counts["missing"],
        "invalid_count": counts["invalid"],
        "future_count": counts["future"],
        "oldest_verified_date": min(parsed_dates).isoformat() if parsed_dates else None,
        "newest_verified_date": max(parsed_dates).isoformat() if parsed_dates else None,
    }


def load_faiss() -> Any:
    return importlib.import_module("faiss")


def inspect_index(
    index_path: Optional[Path],
    meta_rows: int,
    strict: bool,
    add_issue: Any,
    *,
    loaded_index: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "requested": index_path is not None,
        "path": str(index_path) if index_path is not None else None,
        "exists": None,
        "size_bytes": None,
        "faiss_available": None,
        "verification": "not_requested",
        "ntotal": None,
        "matches_meta_rows": None,
    }
    if index_path is None:
        return result
    result["exists"] = index_path.is_file()
    if not result["exists"]:
        result["verification"] = "missing"
        add_issue("index_file_missing", "fatal", f"index file not found: {index_path}")
        return result
    result["size_bytes"] = index_path.stat().st_size
    try:
        faiss = load_faiss()
    except (ImportError, ModuleNotFoundError):
        result["faiss_available"] = False
        result["verification"] = "unknown"
        add_issue(
            "faiss_unavailable",
            "fatal" if strict else "warning",
            "FAISS is unavailable; index row alignment is unknown",
        )
        return result
    result["faiss_available"] = True
    try:
        index = faiss.read_index(str(index_path))
        ntotal = int(index.ntotal)
    except Exception as exc:
        result["verification"] = "error"
        result["error_type"] = type(exc).__name__
        add_issue("index_read_error", "fatal", "FAISS index could not be read")
        return result
    if loaded_index is not None:
        loaded_index["faiss"] = faiss
        loaded_index["index"] = index
    result["ntotal"] = ntotal
    result["matches_meta_rows"] = ntotal == meta_rows
    result["verification"] = "match" if ntotal == meta_rows else "mismatch"
    if ntotal != meta_rows:
        add_issue(
            "index_meta_mismatch",
            "fatal",
            f"index ntotal {ntotal} does not match metadata rows {meta_rows}",
        )
    return result


def _sample_append(samples: list[Any], value: Any) -> None:
    if len(samples) < ARTIFACT_SAMPLE_LIMIT:
        samples.append(value)


def _forbidden_derived_paths(value: Any, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, nested in sorted(value.items(), key=lambda item: str(item[0])):
            key = str(raw_key)
            path = f"{prefix}.{key}" if prefix else key
            if key.casefold() in FORBIDDEN_DERIVED_KEYS:
                paths.append(path)
            paths.extend(_forbidden_derived_paths(nested, path))
    elif isinstance(value, (list, tuple)):
        for position, nested in enumerate(value):
            path = f"{prefix}[{position}]"
            paths.extend(_forbidden_derived_paths(nested, path))
    return paths


def _inspect_artifact_metadata(
    records: Sequence[Mapping[str, Any]], add_issue: Any
) -> dict[str, Any]:
    blank_id_rows: list[int] = []
    invalid_row_types: list[dict[str, Any]] = []
    invalid_provenance: list[dict[str, Any]] = []
    invalid_text_hashes: list[dict[str, Any]] = []
    invalid_image_sources: list[dict[str, Any]] = []
    forbidden_fields: list[dict[str, Any]] = []
    unique_ids: set[str] = set()
    public_text_ids: set[str] = set()
    row_types: Counter[str] = Counter()
    forbidden_occurrences = 0

    for position, record in enumerate(records):
        dog_id = record_id(record)
        if dog_id:
            unique_ids.add(dog_id)
        else:
            _sample_append(blank_id_rows, position)

        row_type = str(record.get("type") or "").strip()
        row_types[row_type or "<blank>"] += 1
        expected_provenance = ALLOWED_ROW_PROVENANCE.get(row_type)
        if expected_provenance is None:
            _sample_append(
                invalid_row_types,
                {"row": position, "id": dog_id, "type": row_type},
            )
        else:
            actual_provenance = str(record.get("embedding_source") or "").strip()
            provenance_valid = actual_provenance == expected_provenance
            if not provenance_valid:
                _sample_append(
                    invalid_provenance,
                    {
                        "row": position,
                        "id": dog_id,
                        "type": row_type,
                        "actual": actual_provenance,
                        "expected": expected_provenance,
                    },
                )

            if row_type == "text":
                text = str(record.get("desc_full") or "").strip()
                stored_hash = str(record.get("embedding_text_sha256") or "").strip()
                expected_hash = (
                    hashlib.sha256(text.encode("utf-8")).hexdigest() if text else ""
                )
                hash_valid = bool(text) and stored_hash == expected_hash
                if not hash_valid:
                    _sample_append(
                        invalid_text_hashes,
                        {"row": position, "id": dog_id},
                    )
                if dog_id and provenance_valid and hash_valid:
                    public_text_ids.add(dog_id)
            elif row_type in {"image", "crop_image"}:
                current_url = str(record.get("image_url") or "").strip()
                source_url = str(record.get("embedding_image_url") or "").strip()
                if not current_url or not source_url or source_url != current_url:
                    _sample_append(
                        invalid_image_sources,
                        {"row": position, "id": dog_id, "type": row_type},
                    )

        for path in _forbidden_derived_paths(record):
            forbidden_occurrences += 1
            _sample_append(
                forbidden_fields,
                {"row": position, "id": dog_id, "path": path},
            )

    missing_public_text_ids = sorted(unique_ids - public_text_ids)
    invalid_row_type_count = sum(
        count
        for row_type, count in row_types.items()
        if row_type not in ALLOWED_ROW_PROVENANCE
    )
    invalid_provenance_count = sum(
        1
        for record in records
        if (
            (row_type := str(record.get("type") or "").strip())
            in ALLOWED_ROW_PROVENANCE
            and str(record.get("embedding_source") or "").strip()
            != ALLOWED_ROW_PROVENANCE[row_type]
        )
    )
    invalid_text_hash_count = sum(
        1
        for record in records
        if str(record.get("type") or "").strip() == "text"
        and (
            not (text := str(record.get("desc_full") or "").strip())
            or str(record.get("embedding_text_sha256") or "").strip()
            != hashlib.sha256(text.encode("utf-8")).hexdigest()
        )
    )
    invalid_image_source_count = sum(
        1
        for record in records
        if str(record.get("type") or "").strip() in {"image", "crop_image"}
        and (
            not (current_url := str(record.get("image_url") or "").strip())
            or not (source_url := str(record.get("embedding_image_url") or "").strip())
            or source_url != current_url
        )
    )

    counts = {
        "blank_id_row_count": len(records)
        - sum(bool(record_id(row)) for row in records),
        "invalid_row_type_count": invalid_row_type_count,
        "invalid_provenance_count": invalid_provenance_count,
        "invalid_text_hash_count": invalid_text_hash_count,
        "invalid_image_source_count": invalid_image_source_count,
        "forbidden_derived_field_occurrence_count": forbidden_occurrences,
        "unique_ids_missing_public_text_count": len(missing_public_text_ids),
    }
    issue_specs = (
        (
            "artifact_blank_id",
            counts["blank_id_row_count"],
            "artifact metadata contains rows without an ID",
        ),
        (
            "artifact_row_type_invalid",
            counts["invalid_row_type_count"],
            "artifact metadata contains an unsupported or blank row type",
        ),
        (
            "artifact_provenance_invalid",
            counts["invalid_provenance_count"],
            "artifact row provenance does not match its row type",
        ),
        (
            "artifact_text_hash_invalid",
            counts["invalid_text_hash_count"],
            "one or more public text hashes do not match desc_full",
        ),
        (
            "artifact_image_source_invalid",
            counts["invalid_image_source_count"],
            "one or more image embedding source URLs do not match image_url",
        ),
        (
            "artifact_forbidden_derived_field",
            counts["forbidden_derived_field_occurrence_count"],
            "artifact metadata contains prohibited VLM or inferred behaviour fields",
        ),
        (
            "artifact_public_text_missing",
            counts["unique_ids_missing_public_text_count"],
            "one or more notice IDs have no valid public notice text row",
        ),
    )
    for code, count, message in issue_specs:
        if count:
            add_issue(code, "fatal", f"{message} ({count})")

    passed = not any(counts.values())
    return {
        "passed": passed,
        "row_count": len(records),
        "unique_id_count": len(unique_ids),
        "row_type_distribution": dict(sorted(row_types.items())),
        **counts,
        "samples": {
            "blank_id_rows": blank_id_rows,
            "invalid_row_types": invalid_row_types,
            "invalid_provenance": invalid_provenance,
            "invalid_text_hashes": invalid_text_hashes,
            "invalid_image_sources": invalid_image_sources,
            "forbidden_derived_fields": forbidden_fields,
            "unique_ids_missing_public_text": missing_public_text_ids[
                :ARTIFACT_SAMPLE_LIMIT
            ],
        },
    }


def _inspect_artifact_vectors(
    faiss: Any,
    index: Any,
    add_issue: Any,
) -> dict[str, Any]:
    expected_metric = int(getattr(faiss, "METRIC_L2", 1))
    try:
        actual_metric = int(index.metric_type)
    except (AttributeError, TypeError, ValueError):
        actual_metric = None
    try:
        dimension = int(index.d)
    except (AttributeError, TypeError, ValueError):
        dimension = None
    try:
        vector_count = int(index.ntotal)
    except (AttributeError, TypeError, ValueError):
        vector_count = 0

    metric_valid = actual_metric == expected_metric
    dimension_valid = dimension == EXPECTED_CLIP_DIMENSION
    reconstruction_errors: list[int] = []
    nonfinite_vectors: list[int] = []
    zero_norm_vectors: list[int] = []
    non_unit_vectors: list[dict[str, Any]] = []
    finite_norms: list[float] = []
    vector_violations: Counter[str] = Counter()

    for position in range(max(vector_count, 0)):
        try:
            raw_vector = index.reconstruct(position)
            values = [float(value) for value in raw_vector]
            if dimension is None or len(values) != dimension:
                raise ValueError("vector dimension does not match index.d")
        except Exception:
            vector_violations["reconstruction"] += 1
            _sample_append(reconstruction_errors, position)
            continue
        if not all(math.isfinite(value) for value in values):
            vector_violations["nonfinite"] += 1
            _sample_append(nonfinite_vectors, position)
            continue
        norm = math.sqrt(math.fsum(value * value for value in values))
        finite_norms.append(norm)
        if norm <= 1e-12:
            vector_violations["zero"] += 1
            _sample_append(zero_norm_vectors, position)
        elif abs(norm - 1.0) > CLIP_NORM_ABS_TOLERANCE:
            vector_violations["non_unit"] += 1
            _sample_append(
                non_unit_vectors,
                {"row": position, "norm": round(norm, 8)},
            )

    counts = {
        "reconstruction_error_count": vector_violations["reconstruction"],
        "nonfinite_vector_count": vector_violations["nonfinite"],
        "zero_norm_vector_count": vector_violations["zero"],
        "non_unit_norm_vector_count": vector_violations["non_unit"],
    }
    if not metric_valid:
        add_issue(
            "artifact_index_metric_invalid",
            "fatal",
            "FAISS index must use L2 distance",
        )
    if not dimension_valid:
        add_issue(
            "artifact_index_dimension_invalid",
            "fatal",
            f"FAISS index dimension must be {EXPECTED_CLIP_DIMENSION}",
        )
    vector_issue_specs = (
        (
            "artifact_vector_reconstruction_failed",
            counts["reconstruction_error_count"],
            "one or more FAISS vectors could not be reconstructed",
        ),
        (
            "artifact_vector_nonfinite",
            counts["nonfinite_vector_count"],
            "one or more FAISS vectors contain non-finite values",
        ),
        (
            "artifact_vector_zero",
            counts["zero_norm_vector_count"],
            "one or more FAISS vectors have zero norm",
        ),
        (
            "artifact_vector_not_normalized",
            counts["non_unit_norm_vector_count"],
            "one or more CLIP vectors are outside the normalization tolerance",
        ),
    )
    for code, count, message in vector_issue_specs:
        if count:
            add_issue(code, "fatal", f"{message} ({count})")

    passed = metric_valid and dimension_valid and not any(counts.values())
    return {
        "checked": True,
        "passed": passed,
        "expected_metric_type": expected_metric,
        "metric_type": actual_metric,
        "expected_dimension": EXPECTED_CLIP_DIMENSION,
        "dimension": dimension,
        "vector_count": vector_count,
        "norm_abs_tolerance": CLIP_NORM_ABS_TOLERANCE,
        "norm_min": round(min(finite_norms), 8) if finite_norms else None,
        "norm_max": round(max(finite_norms), 8) if finite_norms else None,
        "norm_mean": (
            round(math.fsum(finite_norms) / len(finite_norms), 8)
            if finite_norms
            else None
        ),
        **counts,
        "samples": {
            "reconstruction_error_rows": reconstruction_errors,
            "nonfinite_vector_rows": nonfinite_vectors,
            "zero_norm_vector_rows": zero_norm_vectors,
            "non_unit_norm_vectors": non_unit_vectors,
        },
    }


def inspect_artifact_integrity(
    records: Sequence[Mapping[str, Any]],
    loaded_index: Mapping[str, Any],
    add_issue: Any,
) -> dict[str, Any]:
    metadata = _inspect_artifact_metadata(records, add_issue)
    faiss = loaded_index.get("faiss")
    index = loaded_index.get("index")
    if faiss is None or index is None:
        vectors = {
            "checked": False,
            "passed": False,
            "reason": "FAISS index was not available for semantic inspection",
        }
    else:
        vectors = _inspect_artifact_vectors(faiss, index, add_issue)
    passed = metadata["passed"] and vectors["passed"]
    return {
        "enabled": True,
        "verification": "pass" if passed else "fail",
        "passed": passed,
        "policy": {
            "allowed_row_provenance": ALLOWED_ROW_PROVENANCE,
            "forbidden_derived_keys": sorted(FORBIDDEN_DERIVED_KEYS),
            "expected_index_metric": "L2",
            "expected_clip_dimension": EXPECTED_CLIP_DIMENSION,
            "clip_norm_abs_tolerance": CLIP_NORM_ABS_TOLERANCE,
        },
        "metadata": metadata,
        "vectors": vectors,
    }


def _file_input(path: Optional[Path], requested: bool) -> dict[str, Any]:
    return {
        "requested": requested,
        "path": str(path) if path is not None else None,
        "exists": path.is_file() if path is not None else None,
        "size_bytes": path.stat().st_size
        if path is not None and path.is_file()
        else None,
    }


def build_report(
    metas_path: Path,
    *,
    cache_path: Optional[Path] = None,
    index_path: Optional[Path] = None,
    reference_date: datetime,
    strict: bool = False,
    max_age_days: Optional[int] = None,
) -> dict[str, Any]:
    if max_age_days is not None:
        if (
            isinstance(max_age_days, bool)
            or not isinstance(max_age_days, int)
            or max_age_days < 0
        ):
            raise ValueError("max_age_days must be a nonnegative integer")

    issues: list[dict[str, str]] = []

    def add_issue(code: str, severity: str, message: str) -> None:
        if not any(issue["code"] == code for issue in issues):
            issues.append({"code": code, "severity": severity, "message": message})

    metas: list[dict[str, Any]] = []
    metas_meta: dict[str, Any] = {}
    if not metas_path.is_file():
        add_issue("metas_file_missing", "fatal", f"metas file not found: {metas_path}")
    else:
        try:
            metas, metas_meta = load_records(metas_path, allow_items_object=False)
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            metas_meta["error_type"] = type(exc).__name__
            add_issue("metas_invalid", "fatal", "metas JSON is invalid")

    cache: list[dict[str, Any]] = []
    cache_meta: dict[str, Any] = {}
    if cache_path is not None:
        if not cache_path.is_file():
            add_issue(
                "cache_file_missing", "fatal", f"cache file not found: {cache_path}"
            )
        else:
            try:
                cache, cache_meta = load_records(cache_path, allow_items_object=True)
            except (json.JSONDecodeError, OSError, ValueError) as exc:
                cache_meta["error_type"] = type(exc).__name__
                add_issue("cache_invalid", "fatal", "cache JSON is invalid")

    effective, matched_rows, matched_ids = overlay_cache(
        metas, cache, cache_meta.get("fetched_at")
    )
    metas_stats = analyze_records(metas, reference_date)
    cache_stats = analyze_records(cache, reference_date)
    effective_stats = analyze_records(effective, reference_date)
    freshness_stats = analyze_freshness(effective, reference_date, max_age_days)

    if metas_stats["row_count"] == 0:
        add_issue("metas_empty", "fatal", "metadata has no rows")
    if metas_stats["missing_id_count"]:
        add_issue("metas_missing_id", "fatal", "metadata contains rows without an ID")
    if cache_path is not None and cache_stats["missing_id_count"]:
        add_issue("cache_missing_id", "fatal", "cache contains rows without an ID")
    if effective_stats["coverage"]["status"]["count"] == 0:
        add_issue("status_coverage_zero", "fatal", "status coverage is zero")
    if effective_stats["status_distribution"]["unknown"]:
        add_issue(
            "unknown_notice_status",
            "fatal" if strict else "warning",
            "one or more metadata rows have unknown notice status",
        )
    inactive_rows = (
        effective_stats["status_distribution"]["closed"]
        + effective_stats["status_distribution"]["expired"]
    )
    if inactive_rows:
        add_issue(
            "inactive_notice_status",
            "fatal" if strict else "warning",
            "one or more metadata rows are closed or expired",
        )
    for field, value in effective_stats["coverage"].items():
        if field != "status" and value["count"] == 0:
            add_issue(
                f"{field}_coverage_zero",
                "warning",
                f"{field} coverage is zero",
            )
    if max_age_days is not None:
        freshness_messages = {
            "missing": "one or more unique notices have no verification timestamp",
            "invalid": "one or more unique notices have an invalid verification timestamp",
            "future": "one or more unique notices were verified after the reference date",
            "stale": (
                "one or more unique notices exceed the configured maximum age "
                f"of {max_age_days} days"
            ),
        }
        for category, code in FRESHNESS_ISSUE_CODES.items():
            if freshness_stats[f"{category}_count"]:
                add_issue(
                    code,
                    "fatal" if strict else "warning",
                    freshness_messages[category],
                )

    loaded_index: dict[str, Any] = {}
    index_info = inspect_index(
        index_path,
        metas_stats["row_count"],
        strict,
        add_issue,
        loaded_index=loaded_index,
    )
    artifact_integrity = (
        inspect_artifact_integrity(metas, loaded_index, add_issue)
        if strict and index_path is not None
        else None
    )
    issues.sort(key=lambda item: (item["severity"] != "fatal", item["code"]))
    fatal_codes = [item["code"] for item in issues if item["severity"] == "fatal"]
    warning_codes = [item["code"] for item in issues if item["severity"] == "warning"]
    report = {
        "schema_version": 1,
        "reference_date": reference_date.date().isoformat(),
        "ready": not fatal_codes,
        "inputs": {
            "metas": {**_file_input(metas_path, True), **metas_meta},
            "cache": {**_file_input(cache_path, cache_path is not None), **cache_meta},
        },
        "metas": metas_stats,
        "cache": cache_stats,
        "effective": {
            **effective_stats,
            "cache_matched_row_count": matched_rows,
            "cache_matched_unique_id_count": matched_ids,
            "freshness": freshness_stats,
        },
        "index": index_info,
        "strict": {
            "enabled": strict,
            "passed": not fatal_codes,
            "fatal_issues": fatal_codes,
            "warnings": warning_codes,
        },
        "issues": issues,
    }
    if artifact_integrity is not None:
        report["artifact_integrity"] = artifact_integrity
    return report


def report_text(report: Mapping[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def parse_reference_date(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise argparse.ArgumentTypeError("reference date must be YYYY-MM-DD") from exc


def parse_nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "max age days must be a nonnegative integer"
        ) from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("max age days must be a nonnegative integer")
    return parsed


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report static metadata readiness before contest submission."
    )
    parser.add_argument("--metas", type=Path, default=DEFAULT_METAS_PATH)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--max-age-days",
        type=parse_nonnegative_int,
        help=(
            "Maximum age of last_verified_at/fetched_at in days, measured "
            "against --reference-date."
        ),
    )
    parser.add_argument(
        "--reference-date",
        type=parse_reference_date,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Fixed YYYY-MM-DD date for deterministic notice classification.",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    reference_date = (
        args.reference_date
        if isinstance(args.reference_date, datetime)
        else parse_reference_date(args.reference_date)
    )
    report = build_report(
        args.metas,
        cache_path=args.cache,
        index_path=args.index,
        reference_date=reference_date,
        strict=bool(args.strict),
        max_age_days=getattr(args, "max_age_days", None),
    )
    content = report_text(report)
    if args.output is not None:
        input_paths = [
            path.resolve()
            for path in (args.metas, args.cache, args.index)
            if path is not None
        ]
        if args.output.resolve() in input_paths:
            raise ValueError("output path must not overwrite an input file")
        write_atomic(args.output, content)
    print(content, end="")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = run(args)
    except (OSError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    return 1 if args.strict and not report["ready"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
