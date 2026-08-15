"""Run image retrieval with CLIP, DINOv3, and scale-free late fusion."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.dino_fusion.core import (  # noqa: E402
    ClipImageEncoder,
    DEFAULT_DINO_MODEL_ID,
    DinoV3Encoder,
    collapse_visual_hits,
    reciprocal_rank_fusion,
    safe_read_faiss_index,
)


DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"


def load_json_list(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(
        isinstance(item, Mapping) for item in payload
    ):
        raise ValueError(f"expected a JSON list of objects: {path}")
    return [dict(item) for item in payload]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--clip-index", type=Path, default=ROOT / "data/dog_faiss.index"
    )
    parser.add_argument(
        "--clip-metas", type=Path, default=ROOT / "data/dog_metas.json"
    )
    parser.add_argument(
        "--dino-index", type=Path, default=DEFAULT_ARTIFACT_DIR / "dino.index"
    )
    parser.add_argument(
        "--dino-metas",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "dino_metas.json",
    )
    parser.add_argument("--dino-model-id", default=DEFAULT_DINO_MODEL_ID)
    parser.add_argument("--dino-revision")
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--candidate-depth", type=int, default=200)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--clip-weight", type=float, default=1.0)
    parser.add_argument("--dino-weight", type=float, default=1.0)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.candidate_depth <= 0 or args.topk <= 0:
        raise ValueError("candidate depth and topk must be positive")
    faiss = __import__("faiss")
    clip_index = safe_read_faiss_index(args.clip_index, faiss.read_index)
    dino_index = safe_read_faiss_index(args.dino_index, faiss.read_index)
    clip_metas = load_json_list(args.clip_metas)
    dino_metas = load_json_list(args.dino_metas)
    if int(clip_index.ntotal) != len(clip_metas):
        raise ValueError("CLIP index/metadata row mismatch")
    if int(dino_index.ntotal) != len(dino_metas):
        raise ValueError("DINO index/metadata row mismatch")

    with Image.open(args.image) as source:
        image = source.convert("RGB").copy()
    try:
        clip_encoder = ClipImageEncoder(args.clip_model, device=args.device)
        dino_encoder = DinoV3Encoder(
            args.dino_model_id,
            device=args.device,
            revision=args.dino_revision,
            local_files_only=args.local_files_only,
        )
        clip_query = clip_encoder.encode(image)
        dino_query = dino_encoder.encode(image)
    finally:
        image.close()

    if clip_query.shape[1] != int(clip_index.d):
        raise ValueError("CLIP query/index dimension mismatch")
    if dino_query.shape[1] != int(dino_index.d):
        raise ValueError("DINO query/index dimension mismatch")

    clip_depth = min(int(clip_index.ntotal), args.candidate_depth * 3)
    dino_depth = min(int(dino_index.ntotal), args.candidate_depth)
    clip_distances, clip_indices = clip_index.search(clip_query, clip_depth)
    dino_scores, dino_indices = dino_index.search(dino_query, dino_depth)
    clip_ranking = collapse_visual_hits(
        clip_distances[0].tolist(),
        clip_indices[0].tolist(),
        clip_metas,
        score_kind="distance",
    )[: args.candidate_depth]
    dino_ranking = collapse_visual_hits(
        dino_scores[0].tolist(),
        dino_indices[0].tolist(),
        dino_metas,
        score_kind="similarity",
    )[: args.candidate_depth]
    fused = reciprocal_rank_fusion(
        {"clip": clip_ranking, "dino": dino_ranking},
        weights={"clip": args.clip_weight, "dino": args.dino_weight},
        rrf_k=args.rrf_k,
        limit=args.topk,
    )
    return {
        "query_image": str(args.image.resolve()),
        "training_required": False,
        "fusion": {
            "method": "weighted_reciprocal_rank_fusion",
            "rrf_k": args.rrf_k,
            "weights": {"clip": args.clip_weight, "dino": args.dino_weight},
        },
        "candidate_counts": {
            "clip": len(clip_ranking),
            "dino": len(dino_ranking),
        },
        "results": fused,
    }


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(run(parse_args(argv)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
