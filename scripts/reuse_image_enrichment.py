"""Reuse deterministic image enrichment when refreshing public notice data.

The fresh notice cache is authoritative for every field except
``image_attrs``. Existing vector metadata is consulted only for that field;
VLM output and all notice text, status, timestamps, and URLs remain untouched.

When several vector rows share a notice ID, the most complete ``image_attrs``
object wins conflicts and less complete rows fill only its empty values. Ties
are resolved by canonical JSON order, so input row order cannot change the
result.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4


ID_KEYS = ("desertionNo", "desertion_no")
REUSE_STATS_KEY = "image_attrs_reuse_stats"
EMPTY_TEXT_VALUES = frozenset(
    {
        "",
        "-",
        "--",
        "unknown",
        "none",
        "null",
        "n/a",
        "na",
        "미상",
        "알 수 없음",
        "정보 없음",
    }
)


@dataclass(frozen=True)
class ReusableImageEnrichment:
    source_image_url: str
    image_attrs: dict[str, Any]


def is_empty_value(value: Any) -> bool:
    """Return whether a value carries no useful image-enrichment evidence."""

    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in EMPTY_TEXT_VALUES
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, Mapping):
        return not value or all(is_empty_value(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return not value or all(is_empty_value(item) for item in value)
    # Boolean false and numeric zero are meaningful detector outputs.
    return False


def record_id(record: Mapping[str, Any]) -> str:
    """Read a normalized notice ID without changing the source record."""

    for key in ID_KEYS:
        value = record.get(key)
        if not is_empty_value(value):
            return str(value).strip()
    return ""


def source_image_url(record: Mapping[str, Any]) -> str:
    """Return the exact primary source URL used for image enrichment."""

    image_url = str(record.get("image_url") or "").strip()
    if image_url.startswith(("http://", "https://")):
        return image_url
    image_urls = record.get("image_urls")
    if isinstance(image_urls, (list, tuple)):
        for value in image_urls:
            text = str(value or "").strip()
            if text.startswith(("http://", "https://")):
                return text
    for key in (
        "popfile",
        "popfile1",
        "fileName",
        "thumb",
        "image",
        "img",
    ):
        value = str(record.get(key) or "").strip()
        if value.startswith(("http://", "https://")):
            return value
    return ""


def useful_leaf_count(value: Any) -> int:
    """Count populated leaves for deterministic completeness ranking."""

    if is_empty_value(value):
        return 0
    if isinstance(value, Mapping):
        return sum(useful_leaf_count(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(useful_leaf_count(item) for item in value)
    return 1


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


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _canonicalize(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def fill_empty_values(
    target: Mapping[str, Any], source: Mapping[str, Any]
) -> tuple[dict[str, Any], int]:
    """Fill empty target values from source while preserving useful values."""

    result = copy.deepcopy(dict(target))
    filled_leaves = 0

    for key in sorted(source, key=str):
        source_value = source[key]
        if is_empty_value(source_value):
            continue

        current_value = result.get(key)
        if isinstance(source_value, Mapping) and isinstance(current_value, Mapping):
            merged, nested_filled = fill_empty_values(current_value, source_value)
            if nested_filled:
                result[key] = merged
                filled_leaves += nested_filled
            continue

        if key not in result or is_empty_value(current_value):
            result[key] = _canonicalize(source_value)
            filled_leaves += useful_leaf_count(source_value)

    return result, filled_leaves


def _id_stats(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    seen: set[str] = set()
    missing = 0
    duplicates = 0
    for record in records:
        dog_id = record_id(record)
        if not dog_id:
            missing += 1
            continue
        if dog_id in seen:
            duplicates += 1
        else:
            seen.add(dog_id)
    return {
        "unique_ids": len(seen),
        "missing_id_records": missing,
        "duplicate_id_records": duplicates,
    }


def build_image_attrs_index(
    existing_metas: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, ReusableImageEnrichment], dict[str, int]]:
    """Build URL-bound image enrichment, skipping ambiguous provenance."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    urls_by_id: dict[str, set[str]] = defaultdict(set)
    attrs_rows_missing_url: set[str] = set()
    rows_with_attrs = 0

    for meta in existing_metas:
        dog_id = record_id(meta)
        if not dog_id:
            continue
        image_url = source_image_url(meta)
        if image_url:
            urls_by_id[dog_id].add(image_url)
        attrs = meta.get("image_attrs")
        if not isinstance(attrs, Mapping) or is_empty_value(attrs):
            continue
        grouped[dog_id].append(copy.deepcopy(dict(attrs)))
        rows_with_attrs += 1
        if not image_url:
            attrs_rows_missing_url.add(dog_id)

    index: dict[str, ReusableImageEnrichment] = {}
    multirow_ids = 0
    conflicting_url_ids = 0
    missing_url_ids = 0
    for dog_id in sorted(grouped):
        candidates = sorted(
            grouped[dog_id],
            key=lambda attrs: (-useful_leaf_count(attrs), _canonical_json(attrs)),
        )
        if len(candidates) > 1:
            multirow_ids += 1

        source_urls = urls_by_id.get(dog_id, set())
        if len(source_urls) > 1:
            conflicting_url_ids += 1
            continue
        if len(source_urls) != 1 or dog_id in attrs_rows_missing_url:
            missing_url_ids += 1
            continue

        merged: dict[str, Any] = {}
        for attrs in candidates:
            merged, _ = fill_empty_values(merged, attrs)
        index[dog_id] = ReusableImageEnrichment(
            source_image_url=next(iter(source_urls)),
            image_attrs=_canonicalize(merged),
        )

    base_stats = _id_stats(existing_metas)
    return index, {
        "existing_meta_rows": len(existing_metas),
        "existing_unique_ids": base_stats["unique_ids"],
        "existing_missing_id_rows": base_stats["missing_id_records"],
        "existing_duplicate_id_rows": base_stats["duplicate_id_records"],
        "existing_rows_with_image_attrs": rows_with_attrs,
        "existing_ids_with_image_attrs": len(grouped),
        "existing_ids_with_reusable_image_attrs": len(index),
        "existing_ids_merged_from_multiple_rows": multirow_ids,
        "existing_ids_with_conflicting_source_image_urls": conflicting_url_ids,
        "existing_ids_missing_source_image_url": missing_url_ids,
    }


def reuse_image_attrs(
    fresh_records: Sequence[Mapping[str, Any]],
    existing_metas: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Copy only reusable ``image_attrs`` into deep-copied fresh records."""

    attrs_index, existing_stats = build_image_attrs_index(existing_metas)
    fresh_stats = _id_stats(fresh_records)
    output: list[dict[str, Any]] = []
    matched_records = 0
    enriched_records = 0
    filled_values = 0
    protected_non_object_attrs = 0
    fresh_missing_image_url_records = 0
    mismatch_records = 0
    matched_ids: set[str] = set()
    enriched_ids: set[str] = set()
    mismatch_ids: set[str] = set()

    for fresh_record in fresh_records:
        updated = copy.deepcopy(dict(fresh_record))
        dog_id = record_id(fresh_record)
        reusable = attrs_index.get(dog_id)

        if reusable is not None:
            fresh_url = source_image_url(fresh_record)
            if not fresh_url:
                fresh_missing_image_url_records += 1
                output.append(updated)
                continue
            if fresh_url != reusable.source_image_url:
                mismatch_records += 1
                mismatch_ids.add(dog_id)
                output.append(updated)
                continue
            matched_records += 1
            matched_ids.add(dog_id)
            current_attrs = fresh_record.get("image_attrs")
            if not isinstance(current_attrs, Mapping) and not is_empty_value(
                current_attrs
            ):
                protected_non_object_attrs += 1
                output.append(updated)
                continue
            target = dict(current_attrs) if isinstance(current_attrs, Mapping) else {}
            merged_attrs, changed = fill_empty_values(
                target,
                reusable.image_attrs,
            )
            if changed:
                updated["image_attrs"] = _canonicalize(merged_attrs)
                enriched_records += 1
                enriched_ids.add(dog_id)
                filled_values += changed

        output.append(updated)

    stats = {
        "schema_version": 1,
        "fresh_records": len(fresh_records),
        "fresh_unique_ids": fresh_stats["unique_ids"],
        "fresh_missing_id_records": fresh_stats["missing_id_records"],
        "fresh_duplicate_id_records": fresh_stats["duplicate_id_records"],
        **existing_stats,
        "matched_fresh_records": matched_records,
        "matched_unique_ids": len(matched_ids),
        "unmatched_fresh_records": len(fresh_records) - matched_records,
        "enriched_fresh_records": enriched_records,
        "enriched_unique_ids": len(enriched_ids),
        "filled_attribute_values": filled_values,
        "fresh_records_missing_image_url_for_reuse": fresh_missing_image_url_records,
        "image_url_mismatch_fresh_records": mismatch_records,
        "image_url_mismatch_unique_ids": len(mismatch_ids),
        "fresh_records_with_protected_non_object_image_attrs": (
            protected_non_object_attrs
        ),
    }
    return output, stats


def _validate_records(payload: Any, label: str, *, allow_wrapper: bool) -> list[dict]:
    if isinstance(payload, list):
        records = payload
    elif allow_wrapper and isinstance(payload, dict):
        records = payload.get("items")
    else:
        records = None

    if not isinstance(records, list):
        expected = (
            "a JSON list or an object containing an items list"
            if allow_wrapper
            else "a JSON list of vector metadata rows"
        )
        raise ValueError(f"{label} must be {expected}")

    invalid = [
        index for index, item in enumerate(records) if not isinstance(item, dict)
    ]
    if invalid:
        preview = ", ".join(str(index) for index in invalid[:5])
        raise ValueError(f"{label} contains non-object records at indices {preview}")
    return records


def load_fresh_cache(path: Path) -> tuple[Any, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return payload, _validate_records(payload, "fresh cache", allow_wrapper=True)


def load_existing_metas(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return _validate_records(payload, "existing metas", allow_wrapper=False)


def build_output_payload(
    fresh_payload: Any,
    enriched_records: list[dict[str, Any]],
    stats: Mapping[str, Any],
) -> dict[str, Any]:
    """Preserve a source wrapper, or add one when the source was a bare list."""

    if isinstance(fresh_payload, dict):
        output = copy.deepcopy(fresh_payload)
    else:
        output = {}
    output["items"] = enriched_records
    output[REUSE_STATS_KEY] = copy.deepcopy(dict(stats))
    return output


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write UTF-8 JSON without BOM, replacing the destination atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy only deterministic image_attrs from existing vector metadata "
            "into a freshly fetched notice cache."
        )
    )
    parser.add_argument("--fresh-cache", type=Path, required=True)
    parser.add_argument("--existing-metas", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, int]:
    input_paths = {
        args.fresh_cache.resolve(),
        args.existing_metas.resolve(),
    }
    if args.output.resolve() in input_paths:
        raise ValueError("output path must not overwrite an input file")

    fresh_payload, fresh_records = load_fresh_cache(args.fresh_cache)
    existing_metas = load_existing_metas(args.existing_metas)
    enriched_records, stats = reuse_image_attrs(fresh_records, existing_metas)
    output_payload = build_output_payload(fresh_payload, enriched_records, stats)
    write_json_atomic(args.output, output_payload)
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True))
    return stats


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run(parse_args(argv))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
