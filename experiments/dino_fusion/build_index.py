"""Build an isolated DINOv3 image index without modifying production files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.public_image_download import (  # noqa: E402
    DEFAULT_ALLOWED_IMAGE_HOSTS,
    PublicImageDownloader,
    parse_public_image_hostname,
)
from experiments.dino_fusion.core import (  # noqa: E402
    DEFAULT_DINO_MODEL_ID,
    DinoV3Encoder,
    clean_text,
    normalize_rows,
    notice_id,
    vector_modality,
)


SCHEMA_VERSION = "dino-fusion-index.v1"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "artifacts"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_faiss_atomic(index: Any, path: Path, faiss: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    safe_temporary: Path | None = None
    try:
        try:
            faiss.write_index(index, str(temporary))
        except RuntimeError as exc:
            if "Illegal byte sequence" not in str(exc):
                raise
            safe_temporary = (
                Path(tempfile.gettempdir()) / f"dino-write-{uuid4().hex}.index"
            )
            faiss.write_index(index, str(safe_temporary))
            shutil.copyfile(safe_temporary, temporary)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        if safe_temporary is not None:
            safe_temporary.unlink(missing_ok=True)


def select_visual_rows(
    metas: Sequence[Mapping[str, Any]],
    *,
    sources: set[str],
    limit: int = 0,
) -> list[tuple[int, Mapping[str, Any]]]:
    selected: list[tuple[int, Mapping[str, Any]]] = []
    for index, meta in enumerate(metas):
        if vector_modality(meta) not in sources or not notice_id(meta):
            continue
        selected.append((index, meta))
        if limit > 0 and len(selected) >= limit:
            break
    return selected


def resolve_crop_path(meta: Mapping[str, Any]) -> Path | None:
    attrs = meta.get("image_attrs")
    value = clean_text(attrs.get("crop_path")) if isinstance(attrs, Mapping) else ""
    if not value:
        return None
    candidate = Path(value)
    candidate = candidate if candidate.is_absolute() else ROOT / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        return None
    return resolved if resolved.is_file() else None


def full_image_url(meta: Mapping[str, Any]) -> str:
    return clean_text(meta.get("embedding_image_url") or meta.get("image_url"))


def load_row_image(
    meta: Mapping[str, Any], downloader: PublicImageDownloader
) -> tuple[Any | None, str]:
    modality = vector_modality(meta)
    if modality == "crop_image":
        crop_path = resolve_crop_path(meta)
        if crop_path is None:
            return None, "missing_crop"
        try:
            with Image.open(crop_path) as source:
                return source.convert("RGB").copy(), str(crop_path.relative_to(ROOT))
        except (OSError, ValueError):
            return None, "invalid_crop"
    url = full_image_url(meta)
    if not url:
        return None, "missing_image_url"
    image = downloader(url)
    return (image, url) if image is not None else (None, "download_failed")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metas", type=Path, default=ROOT / "data/dog_metas.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-id", default=DEFAULT_DINO_MODEL_ID)
    parser.add_argument("--revision")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--source",
        action="append",
        choices=("full_image", "crop_image"),
        dest="sources",
    )
    parser.add_argument(
        "--allowed-image-host",
        action="append",
        type=parse_public_image_hostname,
        dest="allowed_image_hosts",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace, *, encoder: Any | None = None) -> dict[str, Any]:
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.limit < 0:
        raise ValueError("limit must be non-negative")
    sources = set(args.sources or ("full_image", "crop_image"))
    output_dir = args.output_dir.resolve()
    index_path = output_dir / "dino.index"
    metas_path = output_dir / "dino_metas.json"
    manifest_path = output_dir / "dino_manifest.json"
    outputs = (index_path, metas_path, manifest_path)
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"outputs already exist ({names}); pass --overwrite")

    raw_metas = json.loads(args.metas.read_text(encoding="utf-8"))
    if not isinstance(raw_metas, list):
        raise ValueError("metadata must be a JSON list")
    selected = select_visual_rows(raw_metas, sources=sources, limit=args.limit)
    if not selected:
        raise ValueError("no matching visual rows")

    runtime_encoder = encoder or DinoV3Encoder(
        args.model_id,
        device=args.device,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    allowed_hosts = tuple(args.allowed_image_hosts or DEFAULT_ALLOWED_IMAGE_HOSTS)
    downloader = PublicImageDownloader(allowed_hosts=allowed_hosts)
    vectors: list[np.ndarray] = []
    output_metas: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
    batch_images: list[Any] = []
    batch_rows: list[tuple[int, Mapping[str, Any], str]] = []

    def flush_batch() -> None:
        if not batch_images:
            return
        encoded = normalize_rows(runtime_encoder.encode_batch(batch_images))
        if encoded.shape[0] != len(batch_rows):
            raise ValueError("encoder returned a different batch size")
        if encoded.shape[1] != int(runtime_encoder.dimension):
            raise ValueError("encoder dimension does not match its declaration")
        for vector, (source_index, meta, source_ref) in zip(encoded, batch_rows):
            modality = vector_modality(meta)
            vectors.append(vector)
            output_metas.append(
                {
                    "notice_id": notice_id(meta),
                    "desertionNo": notice_id(meta),
                    "type": "image" if modality == "full_image" else "crop_image",
                    "embedding_source": f"{runtime_encoder.model_family}_{modality}",
                    "source_meta_index": source_index,
                    "source_ref": source_ref,
                }
            )
        for image in batch_images:
            close = getattr(image, "close", None)
            if callable(close):
                close()
        batch_images.clear()
        batch_rows.clear()

    try:
        for source_index, meta in selected:
            image, source_ref = load_row_image(meta, downloader)
            if image is None:
                failures[source_ref] += 1
                continue
            batch_images.append(image)
            batch_rows.append((source_index, meta, source_ref))
            if len(batch_images) >= args.batch_size:
                flush_batch()
        flush_batch()
    finally:
        downloader.close()
        for image in batch_images:
            close = getattr(image, "close", None)
            if callable(close):
                close()

    if not vectors:
        raise RuntimeError("DINO index build produced no vectors")
    matrix = normalize_rows(np.stack(vectors))
    faiss = __import__("faiss")
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    if int(index.ntotal) != len(output_metas):
        raise RuntimeError("generated index/metadata row mismatch")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_faiss_atomic(index, index_path, faiss)
    write_json_atomic(metas_path, output_metas)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "training_required": False,
        "model": {
            "id": runtime_encoder.model_id,
            "family": runtime_encoder.model_family,
            "requested_revision": runtime_encoder.requested_revision,
            "resolved_revision": runtime_encoder.resolved_revision,
            "dimension": int(runtime_encoder.dimension),
            "pooling": runtime_encoder.pooling,
            "frozen": True,
        },
        "source": {
            "metadata": str(args.metas.resolve()),
            "metadata_sha256": sha256_file(args.metas),
            "selected_rows": len(selected),
            "modalities": sorted(sources),
        },
        "output": {
            "vectors": int(index.ntotal),
            "failures": dict(sorted(failures.items())),
            "index": index_path.name,
            "index_sha256": sha256_file(index_path),
            "metas": metas_path.name,
            "metas_sha256": sha256_file(metas_path),
        },
    }
    write_json_atomic(manifest_path, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    report = run(parse_args(argv))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
