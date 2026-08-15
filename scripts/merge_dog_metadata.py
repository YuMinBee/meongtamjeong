"""Merge notice/cache metadata into vector metadata without rebuilding vectors.

The output keeps the input vector metadata list in exactly the same order and
with exactly the same ``type`` values.  Notice metadata is joined by
``desertionNo`` (or ``desertion_no``), with the enriched cache taking
precedence over the base cache.  Empty/unknown source values never replace a
useful value that is already present.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.notice_metadata import normalize_additional_notice_fields  # noqa: E402

DATA_DIR = BASE_DIR / "data"

DEFAULT_METAS_PATH = DATA_DIR / "dog_metas.json"
DEFAULT_CACHE_PATH = DATA_DIR / "local_dog_cache.json"
DEFAULT_ENRICHED_PATH = DATA_DIR / "local_dog_cache_enriched.json"
DEFAULT_OUTPUT_PATH = DATA_DIR / "dog_metas.enriched.json"

ID_KEYS = ("desertionNo", "desertion_no")
PROTECTED_VECTOR_KEYS = frozenset(("type", "desertionNo", "desertion_no"))

# These placeholders carry no additional information and should not erase a
# concrete value from dog_metas or a lower-priority cache.
EMPTY_TEXT_VALUES = frozenset(
    (
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
    )
)

# Cache files have existed in both public-API camelCase and application
# snake_case forms.  Keep the original source keys, while also projecting the
# useful values onto the canonical names consumed by the current application.
CANONICAL_FIELD_ALIASES: Mapping[str, Sequence[str]] = {
    "desertionNo": ("desertionNo", "desertion_no"),
    "notice_no": ("notice_no", "noticeNo"),
    "species": ("species", "animal_species"),
    "upkind": ("upkind", "upkindCd", "upKindCd"),
    "region": ("region", "region_name", "sido", "sido_name", "upr_name", "uprNm"),
    "org_name": ("org_name", "orgNm"),
    "care_name": ("care_name", "careNm"),
    "care_tel": ("care_tel", "careTel"),
    "care_addr": ("care_addr", "careAddr"),
    "happen_place": ("happen_place", "happenPlace"),
    "process_state": ("process_state", "processState", "status"),
    "notice_start": ("notice_start", "noticeSdt"),
    "notice_end": ("notice_end", "noticeEdt"),
    "breed": ("breed", "kindNm", "kindFullNm", "kindCd"),
    "breed_code": ("breed_code", "breedCd"),
    "breed_name": ("breed_name", "kindNm"),
    "breed_full_name": ("breed_full_name", "kindFullNm"),
    "breed_source_label": ("breed_source_label", "kindNm", "kindFullNm"),
    "breed_source": ("breed_source",),
    "mixed_breed": ("mixed_breed", "is_mixed", "isMixed"),
    "color": ("color", "colorCd"),
    "happen_date": ("happen_date", "happenDt"),
    "sex": ("sex", "sexCd"),
    "age": ("age",),
    "weight": ("weight",),
    "neuter": ("neuter", "neuterYn"),
    "desc": ("desc", "specialMark"),
    "image_url": (
        "image_url",
        "url",
        "popfile",
        "popfile1",
        "popfile2",
        "popfile3",
        "fileName",
        "thumb",
        "image",
        "img",
    ),
    "image_urls": ("image_urls",),
    "upstream_updated_at": ("upstream_updated_at", "updTm"),
    "health_checks": ("health_checks", "healthChk"),
    "vaccinations": ("vaccinations", "vaccinationChk"),
    "safety_health_note": ("safety_health_note", "sfeHealth"),
    "safety_social_note": ("safety_social_note", "sfeSoci"),
    "detail_url": ("detail_url",),
    "active": ("active", "is_active", "searchable"),
    "vlm_desc": ("vlm_desc",),
    "merged_desc": ("merged_desc",),
    "vlm_attrs": ("vlm_attrs",),
    "vlm_attr_text": ("vlm_attr_text",),
    "photo_advice": ("photo_advice",),
    "image_attrs": ("image_attrs", "photo_attrs"),
    "photo_quality_score": ("photo_quality_score",),
    "photo_quality_band": ("photo_quality_band",),
}


def is_empty_value(value: Any) -> bool:
    """Return whether *value* has no useful metadata information."""

    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in EMPTY_TEXT_VALUES
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, Mapping):
        return not value or all(is_empty_value(item) for item in value.values())
    if isinstance(value, (list, tuple, set, frozenset)):
        return not value or all(is_empty_value(item) for item in value)
    # False and numeric zero are meaningful metadata values.
    return False


def record_id(record: Mapping[str, Any]) -> str:
    """Read and normalize a record identifier from either supported key."""

    for key in ID_KEYS:
        value = record.get(key)
        if is_empty_value(value):
            continue
        return str(value).strip()
    return ""


def first_non_empty(record: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = record.get(key)
        if not is_empty_value(value):
            return value
    return None


def with_canonical_fields(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Copy a source record and add canonical aliases for known fields."""

    prepared = copy.deepcopy(dict(record))
    for canonical, aliases in CANONICAL_FIELD_ALIASES.items():
        value = first_non_empty(record, aliases)
        if not is_empty_value(value):
            prepared[canonical] = copy.deepcopy(value)

    # Apply the same conservative public-notice semantics as the live/cache
    # builders. This also collects every distinct source image URL while the
    # vector row itself continues to point at only the first image.
    normalized_notice = normalize_additional_notice_fields(record)
    for canonical, value in normalized_notice.items():
        if canonical == "mixed_breed":
            # Keep an explicit unknown for specific labels and contradictions;
            # a reported breed name is not evidence of pure ancestry.
            has_breed_evidence = any(
                key in record
                for key in ("mixed_breed", "is_mixed", "isMixed")
            ) or any(
                not is_empty_value(record.get(key))
                for key in (
                    "breed",
                    "breed_code",
                    "breed_name",
                    "breed_full_name",
                    "breedCd",
                    "kindCd",
                    "kindNm",
                    "kindFullNm",
                )
            )
            if has_breed_evidence:
                prepared[canonical] = value
        elif not is_empty_value(value):
            prepared[canonical] = copy.deepcopy(value)
    return prepared


def merge_non_empty(
    target: Mapping[str, Any],
    source: Mapping[str, Any],
    protected_keys: Iterable[str] = (),
) -> Tuple[Dict[str, Any], Set[str]]:
    """Overlay non-empty source values and report changed top-level keys.

    Dictionaries are merged recursively so an enriched ``image_attrs`` object
    can add photo fields without dropping useful detector fields already found
    in the base cache.
    """

    result = copy.deepcopy(dict(target))
    protected = set(protected_keys)
    changed: Set[str] = set()

    for key, source_value in source.items():
        if key in protected:
            continue

        current_value = result.get(key)
        if key == "mixed_breed" and source_value is None:
            if key not in result or current_value is not None:
                result[key] = None
                changed.add(key)
            continue

        if is_empty_value(source_value):
            continue

        if key == "image_urls":
            source_items = (
                list(source_value)
                if isinstance(source_value, (list, tuple, set, frozenset))
                else [source_value]
            )
            current_items = (
                list(current_value)
                if isinstance(current_value, (list, tuple, set, frozenset))
                else ([current_value] if not is_empty_value(current_value) else [])
            )
            merged_items: List[Any] = []
            for item in source_items + current_items:
                if not is_empty_value(item) and item not in merged_items:
                    merged_items.append(copy.deepcopy(item))
            if current_value != merged_items:
                result[key] = merged_items
                changed.add(key)
            continue

        if isinstance(current_value, Mapping) and isinstance(source_value, Mapping):
            merged_value, nested_changes = merge_non_empty(current_value, source_value)
            if nested_changes:
                result[key] = merged_value
                changed.add(key)
            continue

        if key not in result or current_value != source_value:
            result[key] = copy.deepcopy(source_value)
            changed.add(key)

    return result, changed


def _validate_record_list(
    payload: Any, path: Path, label: str, require_list_root: bool
) -> List[Dict[str, Any]]:
    if require_list_root:
        records = payload
    elif isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = payload.get("items")
    else:
        records = None

    if not isinstance(records, list):
        expected = (
            "a JSON list"
            if require_list_root
            else "a JSON list or an object containing an items list"
        )
        raise ValueError(f"{label} must be {expected}: {path}")

    invalid = [
        index for index, item in enumerate(records) if not isinstance(item, dict)
    ]
    if invalid:
        preview = ", ".join(str(index) for index in invalid[:5])
        raise ValueError(
            f"{label} contains non-object records at indices {preview}: {path}"
        )
    return records


def load_metas(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"required metas file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _validate_record_list(payload, path, "metas", require_list_root=True)


def load_optional_cache(
    path: Optional[Path], label: str
) -> Tuple[List[Dict[str, Any]], bool]:
    if path is None or not path.exists():
        shown_path = str(path) if path is not None else "<disabled>"
        print(
            f"[WARN] {label} file not found; continuing without it: {shown_path}",
            file=sys.stderr,
        )
        return [], True

    payload = json.loads(path.read_text(encoding="utf-8"))
    records = _validate_record_list(payload, path, label, require_list_root=False)
    return records, False


def build_source_index(
    records: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    """Index source records, combining duplicate IDs in encounter order."""

    index: Dict[str, Dict[str, Any]] = {}
    missing_id = 0
    duplicates = 0

    for record in records:
        dog_id = record_id(record)
        if not dog_id:
            missing_id += 1
            continue

        prepared = with_canonical_fields(record)
        if dog_id in index:
            duplicates += 1
            index[dog_id], _ = merge_non_empty(
                index[dog_id], prepared, protected_keys=("type",)
            )
        else:
            index[dog_id] = prepared

    return index, {
        "records": len(records),
        "unique_ids": len(index),
        "missing_id_records": missing_id,
        "duplicate_id_records": duplicates,
    }


def merge_dog_metadata(
    metas: Sequence[Mapping[str, Any]],
    cache_records: Sequence[Mapping[str, Any]],
    enriched_records: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Merge cache metadata into vector metadata while preserving vector rows."""

    cache_index, cache_stats = build_source_index(cache_records)
    enriched_index, enriched_stats = build_source_index(enriched_records)
    merged_metas: List[Dict[str, Any]] = []

    stats: Dict[str, Any] = {
        "metas_total": len(metas),
        "vectors_missing_id": 0,
        "vectors_without_notice_match": 0,
        "matched_vectors": 0,
        "matched_unique_ids": 0,
        "updated_vectors": 0,
        "updated_top_level_fields": 0,
        "cache": dict(cache_stats),
        "enriched": dict(enriched_stats),
    }
    stats["cache"].update(
        {"matched_vectors": 0, "updated_vectors": 0, "updated_top_level_fields": 0}
    )
    stats["enriched"].update(
        {"matched_vectors": 0, "updated_vectors": 0, "updated_top_level_fields": 0}
    )

    matched_ids: Set[str] = set()
    for meta in metas:
        merged = copy.deepcopy(dict(meta))
        original_type_present = "type" in meta
        original_type = copy.deepcopy(meta.get("type"))
        dog_id = record_id(meta)
        changed_for_vector: Set[str] = set()
        matched = False

        if not dog_id:
            stats["vectors_missing_id"] += 1
        else:
            for source_name, source_index in (
                ("cache", cache_index),
                ("enriched", enriched_index),
            ):
                source = source_index.get(dog_id)
                if source is None:
                    continue
                matched = True
                matched_ids.add(dog_id)
                stats[source_name]["matched_vectors"] += 1
                merged, changed = merge_non_empty(
                    merged,
                    source,
                    protected_keys=PROTECTED_VECTOR_KEYS,
                )
                if changed:
                    stats[source_name]["updated_vectors"] += 1
                    stats[source_name]["updated_top_level_fields"] += len(changed)
                    changed_for_vector.update(changed)

        if matched:
            stats["matched_vectors"] += 1
        else:
            stats["vectors_without_notice_match"] += 1
        if changed_for_vector:
            stats["updated_vectors"] += 1
            stats["updated_top_level_fields"] += len(changed_for_vector)

        # Defensive restoration plus postcondition checks make the vector-row
        # invariants explicit even if future cache schemas add these keys.
        if original_type_present:
            merged["type"] = original_type
        else:
            merged.pop("type", None)
        for key in ID_KEYS:
            if key in meta:
                merged[key] = copy.deepcopy(meta[key])
            else:
                merged.pop(key, None)
        merged_metas.append(merged)

    stats["matched_unique_ids"] = len(matched_ids)
    cache_matched_ids = matched_ids.intersection(cache_index)
    enriched_matched_ids = matched_ids.intersection(enriched_index)
    stats["cache"]["unmatched_unique_ids"] = len(cache_index) - len(cache_matched_ids)
    stats["enriched"]["unmatched_unique_ids"] = len(enriched_index) - len(
        enriched_matched_ids
    )

    if len(merged_metas) != len(metas):
        raise AssertionError("vector metadata count changed during merge")
    for index, (original, merged) in enumerate(zip(metas, merged_metas)):
        if original.get("type") != merged.get("type") or ("type" in original) != (
            "type" in merged
        ):
            raise AssertionError(f"vector metadata type changed at index {index}")
        if record_id(original) != record_id(merged):
            raise AssertionError(
                f"vector metadata order/identifier changed at index {index}"
            )

    return merged_metas, stats


def write_output(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(list(records), ensure_ascii=False, indent=2) + "\n"
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge live/enriched notice metadata into dog vector metadata without changing vector rows."
    )
    parser.add_argument(
        "--metas",
        type=Path,
        default=DEFAULT_METAS_PATH,
        help="Input dog_metas JSON list.",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help="Optional base notice cache JSON.",
    )
    parser.add_argument(
        "--enriched",
        type=Path,
        default=DEFAULT_ENRICHED_PATH,
        help="Optional VLM/image-enriched notice cache JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSON path (default: data/dog_metas.enriched.json).",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    metas = load_metas(args.metas)
    cache_records, cache_missing = load_optional_cache(args.cache, "cache")
    enriched_records, enriched_missing = load_optional_cache(
        args.enriched, "enriched cache"
    )

    merged, stats = merge_dog_metadata(metas, cache_records, enriched_records)
    stats["cache"]["file_missing"] = cache_missing
    stats["enriched"]["file_missing"] = enriched_missing
    stats["output"] = str(args.output)
    write_output(args.output, merged)

    print(
        "[INFO] "
        f"metas={stats['metas_total']} matched_vectors={stats['matched_vectors']} "
        f"updated_vectors={stats['updated_vectors']} fields={stats['updated_top_level_fields']}"
    )
    print(
        "[INFO] "
        f"cache_records={stats['cache']['records']} cache_ids={stats['cache']['unique_ids']} "
        f"enriched_records={stats['enriched']['records']} enriched_ids={stats['enriched']['unique_ids']}"
    )
    print(f"[DONE] saved: {args.output}")
    print(json.dumps({"stats": stats}, ensure_ascii=False, sort_keys=True))
    return stats


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        run(args)
    except (FileNotFoundError, json.JSONDecodeError, ValueError, OSError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
