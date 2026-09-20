"""CPU-only completeness, overlap, checksum and image-integrity validation."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import zipfile

from PIL import Image

from experiments.composed_retrieval.download import DATA, ROOT, sha256
from experiments.composed_retrieval.prepare import (
    ARTIFACTS, TASKS, load_json, manifests, vg_metadata, write_json,
)


def check_query_manifest(manifest: dict) -> dict:
    assets = {asset["key"]: asset for asset in manifest["assets"]}
    if len(assets) != len(manifest["assets"]):
        raise ValueError("duplicate asset keys")
    query_ids = set()
    group_splits = {}
    for query in manifest["queries"]:
        if query["id"] in query_ids:
            raise ValueError("duplicate query ID")
        query_ids.add(query["id"])
        if query["reference"] not in assets:
            raise ValueError("missing reference")
        if not query["text"].strip():
            raise ValueError("empty condition")
        if query["group"] in group_splits and group_splits[query["group"]] != query["split"]:
            raise ValueError("reference group leaks across splits")
        group_splits[query["group"]] = query["split"]
        candidates = query["candidates"] if "candidates" in query else list(assets)
        if not set(query["positives"]) <= set(candidates):
            raise ValueError("positive missing from gallery")
        if len(set(candidates)) != len(candidates):
            raise ValueError("duplicate candidate slot IDs")
        if "candidate_assets" in query:
            if len(query["candidate_assets"]) != len(candidates):
                raise ValueError("slot/asset length mismatch")
            if not set(query["candidate_assets"]) <= set(assets):
                raise ValueError("candidate asset missing")
    duplicates = manifest.get("duplicate_candidate_audit", [])
    labeled_sets = {}
    if manifest.get("dataset") == "circo":
        for query in manifest["queries"]:
            labeled_sets.setdefault(query["split"], set()).update([query["reference"], *query["positives"]])
        if labeled_sets.get("development", set()) & labeled_sets.get("audit", set()):
            raise ValueError("CIRCO labeled images leak across development/audit splits")
    return {"queries": len(query_ids), "assets": len(assets),
            "splits": dict(Counter(query["split"] for query in manifest["queries"])),
            "independence_groups": len(group_splits),
            "reference_groups": len({query.get("reference_group", query["group"]) for query in manifest["queries"]}),
            "duplicate_candidate_queries": len(duplicates),
            "target_duplicated_queries": sum(row["target_occurrences"] > 1 for row in duplicates)}


def inspect_image(path: Path) -> dict:
    try:
        data = path.read_bytes()
        import hashlib
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            rgb = image.convert("RGB")
            pixel = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            return {"path": str(path.relative_to(ROOT)), "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(), "decoded_sha256": pixel,
                    "size": list(image.size)}
    except Exception as error:
        return {"path": str(path.relative_to(ROOT)), "error": str(error)}


def validate_archives(*, deep: bool) -> list[dict]:
    reports = []
    for name in ("annotations_trainval2017.zip", "val2017.zip", "unlabeled2017.zip", "image_data.json.zip"):
        path = DATA / name
        if not path.exists():
            reports.append({"file": name, "complete": False, "error": "not downloaded"})
            continue
        entry = {"file": name, "bytes": path.stat().st_size}
        source_path = path.with_suffix(path.suffix + ".source.json")
        if source_path.exists():
            source = load_json(source_path)
            entry["source"] = source
            if source["size"] != path.stat().st_size:
                entry.update(complete=False, error="archive length mismatch")
                reports.append(entry)
                continue
        try:
            with zipfile.ZipFile(path) as archive:
                images = [info for info in archive.infolist() if info.filename.endswith(".jpg")]
                entry["image_members"] = len(images)
                if deep:
                    bad = archive.testzip()
                    if bad:
                        raise ValueError(f"CRC mismatch: {bad}")
                    entry["crc_verified"] = True
            if deep:
                entry["sha256"] = sha256(path)
                saved_hash = path.with_suffix(path.suffix + ".sha256")
                if saved_hash.exists() and saved_hash.read_text().strip() != entry["sha256"]:
                    raise ValueError("archive checksum changed")
                saved_hash.write_text(entry["sha256"] + "\n")
            entry["complete"] = True
        except Exception as error:
            entry.update(complete=False, error=str(error))
        reports.append(entry)
        print("archive", name, entry.get("complete"), flush=True)
    return reports


def run(*, deep: bool) -> dict:
    manifests()
    summary = {"created_at": datetime.now(timezone.utc).isoformat(), "gpu_used": False,
               "deep_integrity_checked": deep, "datasets": {}, "archives": [], "errors": []}
    source_assets = {}
    required_members = {}
    all_manifests = {}
    for dataset in [*(f"genecis_{task}" for task in TASKS), "circo"]:
        path = ARTIFACTS / f"manifest_{dataset}.json"
        if not path.exists():
            summary["errors"].append(f"missing manifest: {dataset}")
            continue
        manifest = load_json(path)
        all_manifests[dataset] = manifest
        summary["datasets"][dataset] = {**check_query_manifest(manifest), "manifest_sha256": sha256(path)}
        for asset in manifest["assets"]:
            if "path" in asset:
                source_assets[asset["source_id"]] = ROOT / asset["path"]
            else:
                required_members.setdefault(asset["zip"], set()).add(asset["member"])
    for relative, required in required_members.items():
        path = ROOT / relative
        if path.exists():
            with zipfile.ZipFile(path) as archive:
                missing = required - set(archive.namelist())
            if missing:
                summary["errors"].append(f"{relative}: {len(missing)} missing image members")
        else:
            summary["errors"].append(f"missing archive: {relative}")
    training_path = ARTIFACTS / "training_manifest.json"
    if training_path.exists():
        training = load_json(training_path)
        by_split = {split: {row["image_id"] for row in training["records"] if row["split"] == split}
                    for split in ("train", "validation")}
        vg_ids = {int(row["coco_id"]) for row in vg_metadata() if row.get("coco_id") is not None}
        selected_ids = by_split["train"] | by_split["validation"]
        overlap = {"train_validation": len(by_split["train"] & by_split["validation"]),
                   "training_pool_vs_all_visual_genome": len(selected_ids & vg_ids)}
        if "circo" in all_manifests:
            circo_ids = {int(asset["source_id"].split(":")[1]) for asset in all_manifests["circo"]["assets"]}
            overlap["training_pool_vs_circo_gallery"] = len(selected_ids & circo_ids)
        object_ids = {int(asset["source_id"].split(":")[1])
                      for name, manifest in all_manifests.items() if name.endswith("_object")
                      for asset in manifest["assets"]}
        overlap["training_pool_vs_genecis_coco_images"] = len(selected_ids & object_ids)
        summary["overlap_audit"] = overlap
        if any(overlap.values()):
            summary["errors"].append("source-ID overlap detected")
        summary["training"] = {"images": len(training["records"]),
                               "splits": {key: len(value) for key, value in by_split.items()},
                               "captions": sum(len(row["captions"]) for row in training["records"]),
                               "manifest_sha256": sha256(training_path)}
        for row in training["records"]:
            source_assets[f"training:{row['image_id']}"] = ROOT / row["path"]
    else:
        summary["errors"].append("missing training manifest")
    missing = [str(path.relative_to(ROOT)) for path in source_assets.values() if not path.exists()]
    summary["source_files"] = {"required": len(source_assets), "missing": len(missing)}
    if missing:
        summary["errors"].append(f"missing {len(missing)} loose image files")
        write_json(ARTIFACTS / "missing_images.json", missing)
    if deep and not missing:
        with ThreadPoolExecutor(4) as pool:
            inspections = list(pool.map(inspect_image, source_assets.values()))
        failures = [row for row in inspections if "error" in row]
        write_json(ARTIFACTS / "image_integrity.json", inspections)
        summary["source_files"]["decode_failures"] = len(failures)
        if failures:
            summary["errors"].append(f"{len(failures)} image decode failures")
        # Check decoded-byte duplicates across the independently sourced VG/training pools.
        training_pixels = {row["decoded_sha256"] for row in inspections
                           if "decoded_sha256" in row and "train2017" in row["path"]}
        vg_pixels = {row["decoded_sha256"] for row in inspections
                     if "decoded_sha256" in row and "train2017" not in row["path"]}
        duplicates = len(training_pixels & vg_pixels)
        summary["overlap_audit"]["training_pool_vs_vg_decoded_duplicates"] = duplicates
        if duplicates:
            summary["errors"].append("decoded-image duplicates across training and VG")
    summary["archives"] = validate_archives(deep=deep)
    if any(not entry["complete"] for entry in summary["archives"]):
        summary["errors"].append("archive downloads/integrity incomplete")
    summary["cirr"] = {"status": "annotations_only_raw_images_pending_official_access",
                       "source_manifest": str((DATA / "cirr_sources.json").relative_to(ROOT))}
    summary["ready_for_gpu_experiments"] = not summary["errors"] and deep
    write_json(ARTIFACTS / "data_readiness.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deep", action="store_true")
    args = parser.parse_args()
    result = run(deep=args.deep)
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
