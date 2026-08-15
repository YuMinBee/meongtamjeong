"""Search the dog corpus with CLIP text mapped into DINOv3 visual space."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.dino_fusion.alignment import (  # noqa: E402
    blend_embeddings,
    project_embeddings,
    projection_head_from_checkpoint,
)
from app.notice_metadata import normalize_breed_fields, normalize_http_url  # noqa: E402
from experiments.dino_fusion.build_index import sha256_file  # noqa: E402
from experiments.dino_fusion.core import (  # noqa: E402
    ClipEncoder,
    DinoEncoder,
    clean_text,
    collapse_visual_hits,
    safe_read_faiss_index,
)


DEFAULT_DINO_DIR = Path(__file__).resolve().parent / "artifacts" / "dinov3"
DEFAULT_FLOW_DIR = Path(__file__).resolve().parent / "artifacts" / "dinode_flow"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_json_list(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    if not isinstance(payload, list) or not all(
        isinstance(item, Mapping) for item in payload
    ):
        raise ValueError(f"expected a JSON list of objects: {path}")
    return [dict(item) for item in payload]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--architecture", default="flow")
    parser.add_argument(
        "--training-report",
        type=Path,
        default=DEFAULT_FLOW_DIR / "alignment_training_report.json",
    )
    parser.add_argument(
        "--dino-index", type=Path, default=DEFAULT_DINO_DIR / "dino.index"
    )
    parser.add_argument(
        "--dino-metas", type=Path, default=DEFAULT_DINO_DIR / "dino_metas.json"
    )
    parser.add_argument(
        "--dino-manifest",
        type=Path,
        default=DEFAULT_DINO_DIR / "dino_manifest.json",
    )
    parser.add_argument("--clip-metas", type=Path, default=ROOT / "data/dog_metas.json")
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument("--text-weight", type=float, default=0.2)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    query_text = clean_text(args.text)
    if not query_text:
        raise ValueError("text query cannot be empty")
    if args.topk <= 0:
        raise ValueError("topk must be positive")
    if not 0.0 <= args.text_weight <= 1.0:
        raise ValueError("text-weight must be between zero and one")
    if args.image is not None and not args.image.is_file():
        raise FileNotFoundError(f"query image not found: {args.image}")

    torch = __import__("torch")
    faiss = __import__("faiss")
    device = (
        "cuda"
        if args.device == "auto" and bool(torch.cuda.is_available())
        else ("cpu" if args.device == "auto" else args.device)
    )
    dino_index = safe_read_faiss_index(args.dino_index, faiss.read_index)
    dino_metas = load_json_list(args.dino_metas)
    clip_metas = load_json_list(args.clip_metas)
    dino_manifest = load_json(args.dino_manifest)
    dino_model = dino_manifest.get("model") or {}
    if int(dino_index.ntotal) != len(dino_metas):
        raise ValueError("DINO index/metadata row mismatch")

    training_report = load_json(args.training_report)
    architecture = clean_text(args.architecture)
    details = (training_report.get("architectures") or {}).get(architecture) or {}
    checkpoint_path = args.training_report.parent / clean_text(
        details.get("checkpoint")
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")
    expected_hash = clean_text(details.get("checkpoint_sha256"))
    if sha256_file(checkpoint_path) != expected_hash:
        raise ValueError(f"checkpoint hash mismatch: {checkpoint_path.name}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    head = projection_head_from_checkpoint(checkpoint).to(device)
    head.load_state_dict(checkpoint["state_dict"])
    head.eval()

    started = time.perf_counter()
    clip_encoder = ClipEncoder(args.clip_model, device=device)
    clip_model_init_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    clip_text = clip_encoder.encode_text(query_text)
    with torch.inference_mode():
        mapped = (
            project_embeddings(head, torch.as_tensor(clip_text, device=device))
            .detach()
            .float()
            .cpu()
            .numpy()
        )
    text_encode_ms = (time.perf_counter() - started) * 1000

    query_vector = mapped
    image_encode_ms: float | None = None
    if args.image is not None:
        model_id = clean_text(dino_model.get("id"))
        revision = clean_text(dino_model.get("resolved_revision")) or None
        dino_encoder = DinoEncoder(
            model_id,
            device=device,
            revision=revision,
            local_files_only=args.local_files_only,
        )
        with Image.open(args.image) as source:
            image = source.convert("RGB").copy()
        try:
            started = time.perf_counter()
            dino_image = dino_encoder.encode(image)
            image_encode_ms = (time.perf_counter() - started) * 1000
        finally:
            image.close()
        query_vector = blend_embeddings(
            dino_image, mapped, text_weight=args.text_weight
        )

    if query_vector.shape[1] != int(dino_index.d):
        raise ValueError("aligned query/index dimension mismatch")
    scores, indices = dino_index.search(query_vector, int(dino_index.ntotal))
    ranking = collapse_visual_hits(
        scores[0].tolist(),
        indices[0].tolist(),
        dino_metas,
        score_kind="similarity",
    )[: args.topk]
    for result in ranking:
        vector_index = int(result["vector_index"])
        source_index = dino_metas[vector_index].get("source_meta_index")
        if not isinstance(source_index, int) or not 0 <= source_index < len(clip_metas):
            raise ValueError("DINO result has invalid source metadata index")
        meta = clip_metas[source_index]
        breed = normalize_breed_fields(meta)
        result["animal"] = {
            "breed": breed["breed"],
            "breed_source": breed["breed_source"],
            "color": clean_text(meta.get("color") or meta.get("colorCd")),
            "age": clean_text(meta.get("age") or meta.get("ageCd")),
            "weight": clean_text(meta.get("weight")),
            "sex": clean_text(meta.get("sex") or meta.get("sexCd")),
            "region": clean_text(meta.get("orgNm") or meta.get("region")),
            "notice_status": clean_text(
                meta.get("processState") or meta.get("notice_status")
            ),
            "image_url": normalize_http_url(
                meta.get("image_url") or meta.get("popfile")
            ),
        }
    return {
        "query": {
            "text": query_text,
            "image": str(args.image.resolve()) if args.image else None,
            "mode": "dino_image_plus_aligned_text" if args.image else "aligned_text",
        },
        "shared_space": {
            "text_encoder": args.clip_model,
            "visual_encoder": dino_model.get("id"),
            "head": architecture,
            "checkpoint": checkpoint_path.name,
            "checkpoint_sha256": expected_hash,
            "text_weight": args.text_weight if args.image else 1.0,
        },
        "latency_ms": {
            "clip_model_cold_start": round(clip_model_init_ms, 3),
            "clip_text_plus_flow_projection": round(text_encode_ms, 3),
            "dino_image": round(image_encode_ms, 3)
            if image_encode_ms is not None
            else None,
        },
        "corpus_vectors": int(dino_index.ntotal),
        "results": ranking,
        "limitations": [
            "The head was trained on controlled color, size, and age descriptions.",
            "Free-form language and relevance still require a larger human-labeled test.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(run(parse_args(argv)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
