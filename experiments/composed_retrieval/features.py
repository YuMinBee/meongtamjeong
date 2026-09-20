"""Resumable frozen features; read COCO directly from verified ZIP archives."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import re
import threading
import time
import zipfile

import numpy as np
from PIL import Image

from experiments.composed_retrieval.download import ROOT, sha256
from experiments.composed_retrieval.prepare import ARTIFACTS, TASKS, load_json, write_json

CACHE = ARTIFACTS / "features"
_local = threading.local()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def canonical_provenance(value):
    value = dict(value)
    # Callable repr addresses vary across Python processes, not preprocessing.
    value["clip_preprocess"] = re.sub(r" at 0x[0-9A-Fa-f]+", "", value["clip_preprocess"])
    return value


def image_for_asset(asset):
    if "zip" in asset:
        if not hasattr(_local, "archives"):
            _local.archives = {}
        path = str(ROOT / asset["zip"])
        if path not in _local.archives:
            _local.archives[path] = zipfile.ZipFile(path)
        payload = _local.archives[path].read(asset["member"])
    else:
        payload = (ROOT / asset["path"]).read_bytes()
    with Image.open(io.BytesIO(payload)) as source:
        image = source.convert("RGB")
    if "bbox" in asset:
        x, y, width, height = asset["bbox"]
        left, top = max(0, x - .7 * width), max(0, y - .7 * height)
        right = min(image.width, left + 1.7 * width)
        bottom = min(image.height, top + 1.7 * height)
        image = image.crop((left, top, right, bottom))
        side = max(image.size)
        square = Image.new("RGB", (side, side))
        square.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
        image = square
    pixel_hash = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
    return image, hashlib.sha256(payload).hexdigest(), pixel_hash


def catalog():
    training = load_json(ARTIFACTS / "training_manifest.json")
    manifests = [load_json(ARTIFACTS / f"manifest_{name}.json") for name in
                 ["circo", *[f"genecis_{task}" for task in TASKS]]]
    assets = {}
    texts = set()
    for row in training["records"]:
        key = f"train:{row['image_id']}"
        assets[key] = {"key": key, "path": row["path"], "preprocessing": "full"}
        texts.update(row["captions"])
    # Training first so a restart can start training without re-encoding the gallery.
    for manifest in manifests:
        for asset in manifest["assets"]:
            if asset["key"] in assets and assets[asset["key"]] != asset:
                raise ValueError(f"Conflicting asset {asset['key']}")
            assets[asset["key"]] = asset
        texts.update(q["text"] for q in manifest["queries"])
    return list(assets.values()), sorted(texts)


def encode(*, batch_size=64, chunk_size=2048, workers=6):
    import torch
    from experiments.dino_fusion.core import ClipEncoder, DinoEncoder, normalize_rows

    readiness = load_json(ARTIFACTS / "data_readiness.json")
    if not readiness["ready_for_gpu_experiments"]:
        raise RuntimeError("Data readiness check failed")
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("This run requires the authorized CUDA GPU")
    CACHE.mkdir(parents=True, exist_ok=True)
    assets, texts = catalog()
    clip = ClipEncoder(device="cuda")
    dino = DinoEncoder(device="cuda", local_files_only=True)
    clip_weights = Path.home() / ".cache" / "clip" / "ViT-B-32.pt"
    provenance = {
        "format": 1, "assets_hash": digest(assets), "texts_hash": digest(texts),
        "clip": clip.model_name, "clip_weights_sha256": sha256(clip_weights),
        "clip_preprocess": str(clip._preprocess), "clip_tokenization": "truncate=True; raw condition/caption",
        "dino": dino.model_id, "dino_revision": dino.resolved_revision,
        "dino_processor": dino._processor.to_dict(), "pooling": dino.pooling,
        "image_count": len(assets), "text_count": len(texts),
        "batch_size": batch_size, "chunk_size": chunk_size,
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
        "precision": "CLIP native fp16, DINO float32, normalized cache float32",
        "training_manifest_sha256": sha256(ARTIFACTS / "training_manifest.json"),
        "protocol_sha256": sha256(Path(__file__).with_name("PROTOCOL.md")),
    }
    provenance = canonical_provenance(provenance)
    signature = digest(provenance)
    index_path = CACHE / "index.json"
    if index_path.exists():
        previous = load_json(index_path)
        if digest(canonical_provenance(previous["provenance"])) != signature:
            raise RuntimeError("Feature provenance changed; use a separate cache directory")
        # Preserve identifiers of legacy chunks generated before repr normalization.
        signature = previous["signature"]
    else:
        write_json(index_path, {"signature": signature, "provenance": provenance,
                                "image_keys": [a["key"] for a in assets], "texts": texts})
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        def prepare(asset):
            image, byte_hash, pixel_hash = image_for_asset(asset)
            return image, clip._preprocess(image), byte_hash, pixel_hash

        for start in range(0, len(assets), chunk_size):
            path = CACHE / f"images_{start:07d}.npz"
            if path.exists():
                with np.load(path) as saved:
                    if str(saved["signature"]) != signature:
                        raise RuntimeError(f"Stale chunk {path}")
                continue
            clip_rows, dino_rows, hashes = [], [], []
            for offset in range(start, min(start + chunk_size, len(assets)), batch_size):
                batch = assets[offset:min(offset + batch_size, start + chunk_size, len(assets))]
                prepared = list(pool.map(prepare, batch))
                images = [row[0] for row in prepared]
                tensors = torch.stack([row[1] for row in prepared]).to("cuda")
                with torch.inference_mode():
                    encoded = clip._model.encode_image(tensors)
                clip_rows.append(normalize_rows(encoded.float().cpu().numpy()))
                dino_rows.append(dino.encode_batch(images))
                hashes.extend([[row[2], row[3]] for row in prepared])
            temporary = path.with_suffix(".partial")
            with temporary.open("wb") as handle:
                np.savez(handle, clip=np.concatenate(clip_rows), dino=np.concatenate(dino_rows),
                         content_hashes=np.array(hashes), signature=signature)
            temporary.replace(path)
            print(f"features images {min(start + chunk_size, len(assets))}/{len(assets)} "
                  f"elapsed={time.monotonic() - started:.1f}s", flush=True)
    for start in range(0, len(texts), chunk_size):
        path = CACHE / f"texts_{start:07d}.npz"
        if path.exists():
            with np.load(path) as saved:
                if str(saved["signature"]) != signature:
                    raise RuntimeError(f"Stale chunk {path}")
            continue
        rows = [clip.encode_text_batch(texts[i:min(i + batch_size, start + chunk_size, len(texts))])
                for i in range(start, min(start + chunk_size, len(texts)), batch_size)]
        temporary = path.with_suffix(".partial")
        with temporary.open("wb") as handle:
            np.savez(handle, clip=np.concatenate(rows), signature=signature)
        temporary.replace(path)
        print(f"features texts {min(start + chunk_size, len(texts))}/{len(texts)}", flush=True)
    if not (CACHE / "complete.json").exists():
        write_json(CACHE / "complete.json", {"signature": signature,
                                             "elapsed_seconds_this_invocation": time.monotonic() - started})


def load_features():
    index = load_json(CACHE / "index.json")
    if load_json(CACHE / "complete.json")["signature"] != index["signature"]:
        raise RuntimeError("Feature cache is incomplete")
    images = {name: [] for name in ("clip", "dino")}
    texts = []
    for path in sorted(CACHE.glob("images_*.npz")):
        with np.load(path) as chunk:
            if str(chunk["signature"]) != index["signature"]:
                raise RuntimeError(f"Stale cache {path}")
            for name in images:
                images[name].append(chunk[name])
    for path in sorted(CACHE.glob("texts_*.npz")):
        with np.load(path) as chunk:
            if str(chunk["signature"]) != index["signature"]:
                raise RuntimeError(f"Stale cache {path}")
            texts.append(chunk["clip"])
    result = {name: np.concatenate(rows) for name, rows in images.items()}
    result["text"] = np.concatenate(texts)
    if len(result["clip"]) != len(index["image_keys"]) or len(result["text"]) != len(index["texts"]):
        raise RuntimeError("Feature row count mismatch")
    return index, result


if __name__ == "__main__":
    encode()
