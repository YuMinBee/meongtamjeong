"""Build a deterministic contest-readiness report for vector metadata."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime
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
    statuses = Counter(
        classify_notice(dict(item), reference_date=reference_date) for item in records
    )
    return {
        "row_count": len(records),
        "unique_id_count": len(id_counts),
        "missing_id_count": len(records) - len(known_ids),
        "duplicate_id_count": sum(count - 1 for count in id_counts.values()),
        "duplicate_unique_id_count": sum(count > 1 for count in id_counts.values()),
        "coverage": coverage(records, reference_date),
        "status_distribution": {
            key: statuses.get(key, 0)
            for key in ("active", "closed", "expired", "unknown")
        },
    }


def load_faiss() -> Any:
    return importlib.import_module("faiss")


def inspect_index(
    index_path: Optional[Path], meta_rows: int, strict: bool, add_issue: Any
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
) -> dict[str, Any]:
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

    index_info = inspect_index(index_path, metas_stats["row_count"], strict, add_issue)
    issues.sort(key=lambda item: (item["severity"] != "fatal", item["code"]))
    fatal_codes = [item["code"] for item in issues if item["severity"] == "fatal"]
    warning_codes = [item["code"] for item in issues if item["severity"] == "warning"]
    return {
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
