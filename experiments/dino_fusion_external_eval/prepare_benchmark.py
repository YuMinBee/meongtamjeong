"""Join PetFinder descriptions, structured labels, and archived dog photos.

The text table and photo archive were published separately.  Their PetFinder
IDs make it possible to build a small real multimodal holdout without treating
image-derived guesses as personality ground truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import re
import sys
import zipfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.dino_fusion.behavior import NEGATIVE, POSITIVE, UNKNOWN  # noqa: E402
from experiments.dino_fusion_external_eval.compatibility import (  # noqa: E402
    COMPATIBILITY_AXES,
    parse_label,
    split_for_group,
)


DEFAULT_CSV = (
    ROOT
    / "tmp"
    / "dino_fusion_external_eval"
    / "petfinder_text"
    / "allDogDescriptions.csv"
)
DEFAULT_IMAGE_ZIP = (
    ROOT
    / "tmp"
    / "dino_fusion_external_eval"
    / "petfinder_images"
    / "1xx.zip"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "artifacts"
SCHEMA_VERSION = "petfinder-dino-clip-external.v1"
DESCRIPTION_MIN_CHARS = 30
SPLIT_SEED = "petfinder-external-v1"
_IMAGE_MEMBER = re.compile(r"(?:^|/)(\d+)-(\d+)\.(?:jpe?g|png)$", re.IGNORECASE)
_HTML_TAG = re.compile(r"<[^>]+>")


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def clean_description(value: Any) -> str:
    text = html.unescape(_HTML_TAG.sub(" ", _clean(value)))
    return re.sub(r"\s+", " ", text).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(payload: Any, path: Path, *, pretty: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2 if pretty else None,
                separators=None if pretty else (",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def image_members_by_pet(image_zip: Path) -> dict[str, list[str]]:
    members: dict[str, list[tuple[int, str]]] = defaultdict(list)
    with zipfile.ZipFile(image_zip) as archive:
        for member in archive.namelist():
            match = _IMAGE_MEMBER.search(member)
            if match:
                members[match.group(1)].append((int(match.group(2)), member))
    return {
        pet_id: [member for _, member in sorted(rows)]
        for pet_id, rows in members.items()
    }


def _appearance(row: Mapping[str, Any]) -> dict[str, str]:
    fields = (
        "breed_primary",
        "breed_secondary",
        "color_primary",
        "color_secondary",
        "color_tertiary",
        "age",
        "sex",
        "size",
        "coat",
    )
    return {
        key: "" if _clean(row.get(key)).lower() == "na" else _clean(row.get(key))
        for key in fields
    }


def load_text_records(csv_path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    seen: set[str] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "id",
            "org_id",
            "species",
            "description",
            *(axis.source_column for axis in COMPATIBILITY_AXES),
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"PetFinder CSV is missing columns: {sorted(missing)}")
        for row in reader:
            pet_id = _clean(row.get("id"))
            organization_id = _clean(row.get("org_id"))
            description = clean_description(row.get("description"))
            if _clean(row.get("species")).lower() != "dog":
                exclusions["not_dog"] += 1
                continue
            if not pet_id or pet_id in seen:
                exclusions["missing_or_duplicate_id"] += 1
                continue
            if not organization_id:
                exclusions["missing_organization"] += 1
                continue
            if len(description) <= DESCRIPTION_MIN_CHARS:
                exclusions["short_or_missing_description"] += 1
                continue
            labels = [parse_label(row.get(axis.source_column)) for axis in COMPATIBILITY_AXES]
            seen.add(pet_id)
            records.append(
                {
                    "pet_id": pet_id,
                    "organization_id": organization_id,
                    "split": split_for_group(organization_id, seed=SPLIT_SEED),
                    "name": _clean(row.get("name")),
                    "description": description,
                    "labels": labels,
                    "appearance": _appearance(row),
                    "source_url": _clean(row.get("url")),
                }
            )
    return records, dict(sorted(exclusions.items()))


def _label_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts: dict[str, Any] = {}
    for axis_index, axis in enumerate(COMPATIBILITY_AXES):
        values = [int(record["labels"][axis_index]) for record in records]
        counts[axis.key] = {
            "positive": values.count(POSITIVE),
            "negative": values.count(NEGATIVE),
            "unknown": values.count(UNKNOWN),
        }
    return counts


def _split_stats(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        rows = [record for record in records if record["split"] == split]
        output[split] = {
            "records": len(rows),
            "organizations": len({record["organization_id"] for record in rows}),
            "labels": _label_counts(rows),
        }
    return output


def prepare(
    *,
    csv_path: Path,
    image_zip: Path,
    output_dir: Path,
    minimum_photos: int = 4,
) -> dict[str, Any]:
    if minimum_photos < 2:
        raise ValueError("minimum_photos must be at least two")
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    if not image_zip.is_file():
        raise FileNotFoundError(image_zip)

    text_records, exclusions = load_text_records(csv_path)
    image_members = image_members_by_pet(image_zip)
    multimodal_records: list[dict[str, Any]] = []
    for record in text_records:
        members = image_members.get(str(record["pet_id"]), [])
        if len(members) < minimum_photos:
            continue
        multimodal_records.append({**record, "image_members": members})

    text_path = output_dir / "text_records.json"
    multimodal_path = output_dir / "multimodal_records.json"
    write_json_atomic(text_records, text_path)
    write_json_atomic(multimodal_records, multimodal_path)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "method": {
            "join_key": "PetFinder pet ID",
            "split_group": "organization_id",
            "split_seed": SPLIT_SEED,
            "split_fractions": {"train": 0.70, "validation": 0.15, "test": 0.15},
            "description_minimum_characters_exclusive": DESCRIPTION_MIN_CHARS,
            "minimum_photos": minimum_photos,
            "query_photo_index": 0,
            "gallery_photo_index": 1,
        },
        "sources": {
            "text": {
                "path": str(csv_path.resolve()),
                "sha256": sha256_file(csv_path),
                "upstream": "https://github.com/the-pudding/data/tree/master/dog-shelters",
            },
            "images": {
                "path": str(image_zip.resolve()),
                "sha256": sha256_file(image_zip),
                "upstream": "https://huggingface.co/datasets/drzraf/petfinder-dogs",
                "license_status": "not stated by the dataset card; do not redistribute",
            },
        },
        "outputs": {
            "text_records": text_path.name,
            "text_records_sha256": sha256_file(text_path),
            "multimodal_records": multimodal_path.name,
            "multimodal_records_sha256": sha256_file(multimodal_path),
        },
        "text_dataset": {
            "records": len(text_records),
            "exclusions": exclusions,
            "splits": _split_stats(text_records),
        },
        "multimodal_dataset": {
            "records": len(multimodal_records),
            "organizations": len(
                {record["organization_id"] for record in multimodal_records}
            ),
            "splits": _split_stats(multimodal_records),
        },
        "axes": [
            {"key": axis.key, "source_column": axis.source_column}
            for axis in COMPATIBILITY_AXES
        ],
        "limitations": [
            "Structured fields are shelter-provided labels, not lab observations.",
            "Photos and structured records come from different collection dates.",
            "Cat-compatible positives are sparse in the multimodal test split.",
            "Image archive licensing is unstated, so all photos remain local and ignored.",
        ],
    }
    write_json_atomic(manifest, output_dir / "benchmark_manifest.json", pretty=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--image-zip", type=Path, default=DEFAULT_IMAGE_ZIP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--minimum-photos", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = prepare(
        csv_path=args.csv,
        image_zip=args.image_zip,
        output_dir=args.output_dir,
        minimum_photos=args.minimum_photos,
    )
    print(json.dumps(manifest["multimodal_dataset"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
