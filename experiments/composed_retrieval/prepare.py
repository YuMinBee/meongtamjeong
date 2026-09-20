"""Build source-grounded manifests, preserving official query galleries."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import random
import time
import zipfile

from experiments.composed_retrieval.download import DATA, ROOT, S3, fetch, sha256

ARTIFACTS = Path(__file__).parent / "artifacts"
TASKS = ("focus_attribute", "change_attribute", "focus_object", "change_object")
VG_METADATA = "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def vg_metadata() -> list[dict]:
    archive = fetch(VG_METADATA, DATA / "image_data.json.zip")
    with zipfile.ZipFile(archive) as handle:
        return json.loads(handle.read("image_data.json"))


def download_jobs(jobs: list[tuple[str, Path]], report: Path, *, workers: int = 24) -> None:
    started = time.monotonic()
    failures = []
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(fetch, url, path): (url, path) for url, path in jobs}
        for count, future in enumerate(as_completed(futures), 1):
            url, path = futures[future]
            try:
                future.result()
            except Exception as error:
                failures.append({"url": url, "path": str(path), "error": str(error)})
            if count % 250 == 0 or count == len(jobs):
                print(f"{report.stem}: {count}/{len(jobs)}, failures={len(failures)}, "
                      f"elapsed={time.monotonic() - started:.1f}s", flush=True)
    write_json(report, {"attempted": len(jobs), "successful": len(jobs) - len(failures),
                        "failures": failures, "elapsed_seconds": time.monotonic() - started})
    if failures:
        raise RuntimeError(f"{len(failures)} downloads failed; see {report}")


def prepare_vg(*, workers: int = 24) -> None:
    metas = {str(row["image_id"]): row for row in vg_metadata()}
    ids = set()
    for task in TASKS[:2]:
        for query in load_json(DATA / "annotations" / "genecis" / f"{task}.json"):
            for image in [query["reference"], query["target"], *query["gallery"]]:
                ids.add(str(image["image_id"]))
    jobs = [(metas[image_id]["url"].replace("http://", "https://"),
             DATA / "vg" / f"{image_id}.jpg") for image_id in sorted(ids)]
    download_jobs(jobs, DATA / "vg_download_report.json", workers=workers)


def prepare_training(*, workers: int = 16) -> None:
    excluded = {int(row["coco_id"]) for row in vg_metadata() if row.get("coco_id") is not None}
    with zipfile.ZipFile(DATA / "annotations_trainval2017.zip") as handle:
        captions = json.loads(handle.read("annotations/captions_train2017.json"))
    by_id = defaultdict(list)
    for annotation in captions["annotations"]:
        by_id[int(annotation["image_id"])].append(annotation["caption"])
    eligible = sorted(set(by_id) - excluded)
    random.Random(20260916).shuffle(eligible)
    if len(eligible) < 6000:
        raise ValueError("not enough disjoint COCO training images")
    selected = eligible[:6000]
    records = [{"image_id": image_id, "path": str((DATA / "train2017" / f"{image_id:012d}.jpg").relative_to(ROOT)),
                "captions": by_id[image_id], "split": "train" if index < 5000 else "validation"}
               for index, image_id in enumerate(selected)]
    manifest = {"source": "COCO2017 train captions", "seed": 20260916,
                "excluded_vg_coco_ids": len(excluded), "eligible_images": len(eligible),
                "vg_metadata_sha256": sha256(DATA / "image_data.json.zip"),
                "annotation_archive_sha256": sha256(DATA / "annotations_trainval2017.zip"),
                "records": records}
    write_json(ARTIFACTS / "training_manifest.json", manifest)
    jobs = [(f"{S3}/train2017/{row['image_id']:012d}.jpg", ROOT / row["path"]) for row in records]
    download_jobs(jobs, DATA / "training_download_report.json", workers=workers)


def asset_for_genecis(image: dict) -> dict:
    if "instance_bbox" in image:
        image_id = str(image["image_id"])
        bbox = list(image["instance_bbox"])
        token = hashlib.sha256(json.dumps(bbox, separators=(",", ":")).encode()).hexdigest()[:16]
        return {"key": f"vg:{image_id}:{token}", "source_id": f"vg:{image_id}",
                "path": str((DATA / "vg" / f"{image_id}.jpg").relative_to(ROOT)),
                "bbox": bbox, "preprocessing": "genecis_v0_dilation_0.7_square_pad"}
    image_id = int(image["val_image_id"])
    return {"key": f"cocoval:{image_id:012d}", "source_id": f"coco:{image_id}",
            "zip": str((DATA / "val2017.zip").relative_to(ROOT)),
            "member": f"val2017/{image_id:012d}.jpg", "preprocessing": "full"}


def genecis_manifest(task: str) -> dict:
    annotation = DATA / "annotations" / "genecis" / f"{task}.json"
    assets = {}
    queries = []
    duplicate_queries = []
    for index, raw in enumerate(load_json(annotation)):
        reference, target = [asset_for_genecis(raw[key]) for key in ("reference", "target")]
        gallery = [target] + [asset_for_genecis(image) for image in raw["gallery"]]
        for asset in [reference, *gallery]:
            assets[asset["key"]] = asset
        query_id = f"{task}:{index}"
        candidate_keys = [image["key"] for image in gallery]
        if len(candidate_keys) != len(set(candidate_keys)):
            duplicate_queries.append({"query_id": query_id,
                                      "repeated_entries": len(candidate_keys) - len(set(candidate_keys)),
                                      "target_occurrences": candidate_keys.count(target["key"])})
        # Preserve every official candidate slot, including annotation duplicates.
        # Pseudorandom stable slot IDs prevent positive-first ordering from deciding ties.
        slots = [(hashlib.sha256(f"{query_id}:slot:{slot}".encode()).hexdigest(), key)
                 for slot, key in enumerate(candidate_keys)]
        positive_slot = slots[0][0]
        slots.sort()
        queries.append({"id": f"{task}:{index}", "group": reference["source_id"],
                        "reference": reference["key"], "text": raw["condition"],
                        "candidates": [slot[0] for slot in slots],
                        "candidate_assets": [slot[1] for slot in slots],
                        "positives": [positive_slot],
                        "split": "external"})
    return {"dataset": f"genecis_{task}", "annotation_sha256": sha256(annotation),
            "protocol": "official_v0", "assets": sorted(assets.values(), key=lambda a: a["key"]),
            "queries": queries, "duplicate_candidate_audit": duplicate_queries,
            "tie_policy": "stable_sha256_candidate_slot_order"}


def circo_query_records(rows: list[dict], keys: set[str]) -> list[dict]:
    """Keep all queries linked through any reference/positive image in one split."""
    parent = {}

    def root(image_id: int) -> int:
        parent.setdefault(image_id, image_id)
        if parent[image_id] != image_id:
            parent[image_id] = root(parent[image_id])
        return parent[image_id]

    for row in rows:
        for image_id in row["gt_img_ids"]:
            left, right = root(int(row["reference_img_id"])), root(int(image_id))
            parent[max(left, right)] = min(left, right)
    queries = []
    for raw in rows:
        group = f"circo-component:{root(int(raw['reference_img_id']))}"
        # Split depends on annotation connectivity, never model scores.
        split = "development" if hashlib.sha256(group.encode()).digest()[0] % 2 == 0 else "audit"
        reference = f"coco:{raw['reference_img_id']:012d}"
        positives = [f"coco:{int(image_id):012d}" for image_id in raw["gt_img_ids"]]
        if reference not in keys or not set(positives) <= keys:
            raise ValueError("CIRCO reference or positive absent from full gallery")
        queries.append({"id": f"circo:{raw['id']}", "group": group, "reference": reference,
                        "reference_group": f"coco:{raw['reference_img_id']}",
                        "text": raw["relative_caption"], "shared_concept": raw["shared_concept"],
                        "positives": positives, "split": split,
                        "semantic_aspects": raw["semantic_aspects"]})
    return queries


def circo_manifest() -> dict:
    annotation = DATA / "annotations" / "circo" / "val.json"
    archive_path = DATA / "unlabeled2017.zip"
    with zipfile.ZipFile(archive_path) as archive:
        names = sorted(name for name in archive.namelist() if name.endswith(".jpg"))
    if len(names) != 123403:
        raise ValueError(f"incomplete CIRCO gallery: {len(names)} != 123403")
    assets = [{"key": f"coco:{int(Path(name).stem):012d}",
               "source_id": f"coco:{int(Path(name).stem)}",
               "zip": str(archive_path.relative_to(ROOT)), "member": name,
               "preprocessing": "full"} for name in names]
    queries = circo_query_records(load_json(annotation), {asset["key"] for asset in assets})
    return {"dataset": "circo", "annotation_sha256": sha256(annotation),
            "protocol": "local_labeled_image_component_split_of_official_validation_full_gallery",
            "split_rule": "sha256_parity_of_reference_positive_connected_component",
            "gallery_size": len(assets), "assets": assets, "queries": queries}


def manifests() -> None:
    for task in TASKS:
        manifest = genecis_manifest(task)
        write_json(ARTIFACTS / f"manifest_genecis_{task}.json", manifest)
        print(task, len(manifest["queries"]), "queries", len(manifest["assets"]), "assets")
    if (DATA / "unlabeled2017.zip").exists():
        manifest = circo_manifest()
        write_json(ARTIFACTS / "manifest_circo.json", manifest)
        print("circo", len(manifest["queries"]), "queries", len(manifest["assets"]), "assets")
    else:
        print("CIRCO archive not complete yet; rerun manifests after download", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", choices=("vg", "training", "manifests"))
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    if args.target == "manifests":
        manifests()
    else:
        {"vg": prepare_vg, "training": prepare_training}[args.target](workers=args.workers)


if __name__ == "__main__":
    main()
