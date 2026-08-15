"""Evaluate frozen CLIP, DINO, and late fusion on prior held-out queries.

Only query photos whose payload hash still matches the repository's previous
held-out execution are evaluated. Images are decoded in memory and never saved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.heldout_image_evaluation import (  # noqa: E402
    ImageDownloadError,
    SafePublicImageDownloader,
    aggregate_rank_metrics,
    rank_record,
)
from app.graph_rag import infer_size_from_weight  # noqa: E402
from app.retrieval_evaluation import normalize_public_colors  # noqa: E402
from experiments.dino_fusion.build_index import write_json_atomic  # noqa: E402
from experiments.dino_fusion.core import (  # noqa: E402
    ClipImageEncoder,
    DinoEncoder,
    clean_text,
    collapse_visual_hits,
    reciprocal_rank_fusion,
    safe_read_faiss_index,
    normalize_rows,
)


DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
DEFAULT_ALLOWED_HOSTS = ("openapi.animal.go.kr",)
REPORT_SCHEMA_VERSION = "dino-fusion-pilot.v1"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def metas_by_notice(metas: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for meta in metas:
        dog_id = clean_text(meta.get("desertionNo") or meta.get("desertion_no"))
        if not dog_id:
            continue
        current = merged.setdefault(dog_id, {})
        for key, value in meta.items():
            if key not in current or not current.get(key):
                current[key] = value
    return merged


def secondary_url(meta: Mapping[str, Any]) -> str:
    urls = meta.get("image_urls")
    if not isinstance(urls, list):
        return ""
    primary = clean_text(meta.get("embedding_image_url") or meta.get("image_url"))
    return next(
        (clean_text(url) for url in urls[1:] if clean_text(url) != primary),
        "",
    )


def mean_latency(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return round(mean(values), 3) if values else None


def paired_rank_comparison(
    query_rows: Sequence[Mapping[str, Any]],
    *,
    left: str,
    right: str,
) -> dict[str, int]:
    """Count per-query left-system rank wins, ties, and losses."""

    counts: Counter[str] = Counter()
    for query in query_rows:
        systems = query.get("systems") or {}
        left_rank = (systems.get(left) or {}).get("rank")
        right_rank = (systems.get(right) or {}).get("rank")
        left_value = int(left_rank) if isinstance(left_rank, int) else 10**9
        right_value = int(right_rank) if isinstance(right_rank, int) else 10**9
        if left_value < right_value:
            counts["wins"] += 1
        elif left_value == right_value:
            counts["ties"] += 1
        else:
            counts["losses"] += 1
    return {key: counts[key] for key in ("wins", "ties", "losses")}


def appearance_prompts(meta: Mapping[str, Any]) -> dict[str, str]:
    """Build conservative bilingual visual prompts from public color/weight."""

    raw_color = clean_text(meta.get("color") or meta.get("colorCd"))
    lowered = raw_color.casefold()
    colors: list[str] = []
    color_rules = (
        ("black", ("검", "흑", "black")),
        ("white", ("흰", "백", "white")),
        ("brown", ("갈", "brown")),
        ("cream", ("크림", "cream", "ivory")),
        ("gray", ("회", "gray", "grey")),
        ("tan", ("황", "노랑", "tan", "yellow")),
    )
    for label, terms in color_rules:
        if any(term in lowered for term in terms):
            colors.append(label)
    size = clean_text(infer_size_from_weight(meta.get("weight")))
    size_en = {
        "tiny": "very small",
        "small": "small",
        "medium": "medium-sized",
        "large": "large",
    }.get(size, "")
    size_ko = {
        "tiny": "아주 작은",
        "small": "작은",
        "medium": "중간 크기의",
        "large": "큰",
    }.get(size, "")

    english = "a photo of a"
    if size_en:
        english += f" {size_en}"
    english += " dog"
    if colors:
        english += f" with {' and '.join(colors)} fur"
    korean_parts = []
    if raw_color:
        korean_parts.append(f"{raw_color.replace('&', '과')} 털의")
    if size_ko:
        korean_parts.append(size_ko)
    korean_parts.append("강아지 사진")
    return {"english": english, "korean": " ".join(korean_parts)}


def _dcg(relevances: Sequence[int]) -> float:
    import math

    return sum(value / math.log2(rank + 1) for rank, value in enumerate(relevances, 1))


def attribute_relevance_summary(
    query_rows: Sequence[Mapping[str, Any]],
    *,
    system: str,
    metas: Mapping[str, Mapping[str, Any]],
    candidate_notice_ids: set[str],
) -> dict[str, float | int]:
    """Score coarse color+size relevance independently of exact notice ID."""

    per_query: list[dict[str, float]] = []
    for query in query_rows:
        target = metas.get(clean_text(query.get("notice_id"))) or {}
        query_colors = set(
            normalize_public_colors(target.get("color") or target.get("colorCd"))
        )
        query_size = clean_text(infer_size_from_weight(target.get("weight")))
        if not query_colors or not query_size:
            continue

        def relevant(dog_id: str) -> bool:
            candidate = metas.get(dog_id) or {}
            candidate_colors = set(
                normalize_public_colors(
                    candidate.get("color") or candidate.get("colorCd")
                )
            )
            candidate_size = clean_text(
                infer_size_from_weight(candidate.get("weight"))
            )
            return bool(query_colors.intersection(candidate_colors)) and (
                query_size == candidate_size
            )

        relevant_count = sum(relevant(dog_id) for dog_id in candidate_notice_ids)
        top_ids = [
            clean_text(value)
            for value in ((query.get("systems") or {}).get(system) or {}).get(
                "top_ids", []
            )
        ]
        top_relevance = [int(relevant(dog_id)) for dog_id in top_ids[:10]]
        top_relevance.extend([0] * (10 - len(top_relevance)))
        ideal = [1] * min(10, relevant_count)
        ideal.extend([0] * (10 - len(ideal)))
        ideal_dcg = _dcg(ideal)
        per_query.append(
            {
                "precision@5": sum(top_relevance[:5]) / 5,
                "precision@10": sum(top_relevance) / 10,
                "hit@5": float(any(top_relevance[:5])),
                "hit@10": float(any(top_relevance)),
                "nDCG@10": _dcg(top_relevance) / ideal_dcg if ideal_dcg else 0.0,
            }
        )
    metric_keys = ("precision@5", "precision@10", "hit@5", "hit@10", "nDCG@10")
    return {
        "query_count": len(per_query),
        **{
            key: round(mean(row[key] for row in per_query), 6) if per_query else 0.0
            for key in metric_keys
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--heldout-report",
        type=Path,
        default=ROOT / "docs/evaluation/heldout_image_retrieval.appearance_v1.json",
    )
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
    parser.add_argument(
        "--dino-manifest",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "dino_manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "pilot_report.json",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--clip-weight", type=float, default=1.0)
    parser.add_argument("--dino-weight", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=12.0)
    parser.add_argument("--allowed-host", action="append", dest="allowed_hosts")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.limit < 0 or args.topk <= 0:
        raise ValueError("limit must be non-negative and topk must be positive")
    faiss = __import__("faiss")
    heldout = load_json(args.heldout_report)
    clip_metas = load_json(args.clip_metas)
    dino_metas = load_json(args.dino_metas)
    dino_manifest = load_json(args.dino_manifest)
    if not isinstance(clip_metas, list) or not isinstance(dino_metas, list):
        raise ValueError("index metadata must be JSON lists")
    clip_index = safe_read_faiss_index(args.clip_index, faiss.read_index)
    dino_index = safe_read_faiss_index(args.dino_index, faiss.read_index)
    if int(clip_index.ntotal) != len(clip_metas):
        raise ValueError("CLIP index/metadata row mismatch")
    if int(dino_index.ntotal) != len(dino_metas):
        raise ValueError("DINO index/metadata row mismatch")

    dino_model = dino_manifest.get("model") or {}
    dino_model_id = clean_text(dino_model.get("id"))
    dino_revision = clean_text(dino_model.get("resolved_revision")) or None
    if not dino_model_id:
        raise ValueError("DINO manifest is missing the model id")
    clip_encoder = ClipImageEncoder(args.clip_model, device=args.device)
    dino_encoder = DinoEncoder(
        dino_model_id,
        device=args.device,
        revision=dino_revision,
        local_files_only=args.local_files_only,
    )
    if int(clip_index.d) <= 0 or int(dino_index.d) <= 0:
        raise ValueError("invalid index dimension")
    if int(dino_index.d) != int(dino_encoder.dimension):
        raise ValueError("DINO encoder/index dimension mismatch")

    source_indices = {
        int(meta["source_meta_index"])
        for meta in dino_metas
        if isinstance(meta.get("source_meta_index"), int)
    }
    if len(source_indices) != len(dino_metas):
        raise ValueError("DINO metadata source indices are missing or duplicated")
    for source_index in source_indices:
        if not 0 <= source_index < len(clip_metas):
            raise ValueError("DINO source metadata index is out of range")

    current_by_id = metas_by_notice(clip_metas)
    sample_by_id = {
        clean_text(row.get("notice_id")): row for row in heldout.get("sample", [])
    }
    previous_queries = list(heldout.get("queries") or [])
    if args.limit:
        previous_queries = previous_queries[: args.limit]
    dino_notice_ids = {
        clean_text(meta.get("notice_id") or meta.get("desertionNo"))
        for meta in dino_metas
    }

    downloader = SafePublicImageDownloader(
        timeout=args.timeout,
        allowed_hosts=tuple(args.allowed_hosts or DEFAULT_ALLOWED_HOSTS),
    )
    failures: Counter[str] = Counter()
    system_rows: dict[str, list[dict[str, Any]]] = {
        "clip_all_visual": [],
        "clip_crop_only": [],
        "dino_crop_only": [],
        "clip_dino_crop_rrf": [],
        "clip_text_en_crop_only": [],
        "clip_text_ko_crop_only": [],
        "clip_image_text_en_crop_only": [],
        "clip_image_text_ko_crop_only": [],
        "dino_image_clip_text_en_rrf": [],
        "dino_image_clip_text_ko_rrf": [],
    }
    query_rows: list[dict[str, Any]] = []
    attempted = 0
    downloaded = 0

    try:
        for previous in previous_queries:
            dog_id = clean_text(previous.get("notice_id"))
            if dog_id not in dino_notice_ids:
                failures["target_missing_dino_crop"] += 1
                continue
            meta = current_by_id.get(dog_id) or {}
            sample = sample_by_id.get(dog_id) or {}
            url = secondary_url(meta)
            if not url:
                failures["missing_secondary_url"] += 1
                continue
            expected_url_hash = clean_text(sample.get("secondary_url_sha256"))
            actual_url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()
            if expected_url_hash and actual_url_hash != expected_url_hash:
                failures["secondary_url_changed"] += 1
                continue
            attempted += 1
            try:
                downloaded_image = downloader.fetch(url)
            except ImageDownloadError as exc:
                failures[f"download_{exc.reason}"] += 1
                continue
            try:
                if downloaded_image.payload_sha256 != clean_text(
                    previous.get("query_payload_sha256")
                ):
                    failures["secondary_payload_changed"] += 1
                    continue
                downloaded += 1
                started = time.perf_counter()
                clip_query = clip_encoder.encode(downloaded_image.image)
                clip_encode_ms = (time.perf_counter() - started) * 1000
                started = time.perf_counter()
                dino_query = dino_encoder.encode(downloaded_image.image)
                dino_encode_ms = (time.perf_counter() - started) * 1000
                prompts = appearance_prompts(meta)
                started = time.perf_counter()
                clip_text_en_query = clip_encoder.encode_text(prompts["english"])
                clip_text_en_ms = (time.perf_counter() - started) * 1000
                started = time.perf_counter()
                clip_text_ko_query = clip_encoder.encode_text(prompts["korean"])
                clip_text_ko_ms = (time.perf_counter() - started) * 1000

                clip_distances, clip_indices = clip_index.search(
                    clip_query, int(clip_index.ntotal)
                )
                dino_scores, dino_indices = dino_index.search(
                    dino_query, int(dino_index.ntotal)
                )
                clip_all = collapse_visual_hits(
                    clip_distances[0].tolist(),
                    clip_indices[0].tolist(),
                    clip_metas,
                    score_kind="distance",
                )
                filtered_pairs = [
                    (score, index)
                    for score, index in zip(
                        clip_distances[0].tolist(), clip_indices[0].tolist()
                    )
                    if int(index) in source_indices
                ]
                clip_crop = collapse_visual_hits(
                    [pair[0] for pair in filtered_pairs],
                    [pair[1] for pair in filtered_pairs],
                    clip_metas,
                    score_kind="distance",
                )

                def clip_crop_ranking(query_vector: Any) -> list[dict[str, Any]]:
                    distances, indices = clip_index.search(
                        query_vector, int(clip_index.ntotal)
                    )
                    pairs = [
                        (score, index)
                        for score, index in zip(
                            distances[0].tolist(), indices[0].tolist()
                        )
                        if int(index) in source_indices
                    ]
                    return collapse_visual_hits(
                        [pair[0] for pair in pairs],
                        [pair[1] for pair in pairs],
                        clip_metas,
                        score_kind="distance",
                    )

                clip_text_en_crop = clip_crop_ranking(clip_text_en_query)
                clip_text_ko_crop = clip_crop_ranking(clip_text_ko_query)
                clip_image_text_en = clip_crop_ranking(
                    normalize_rows((2.0 * clip_query) + clip_text_en_query)
                )
                clip_image_text_ko = clip_crop_ranking(
                    normalize_rows((2.0 * clip_query) + clip_text_ko_query)
                )
                dino_crop = collapse_visual_hits(
                    dino_scores[0].tolist(),
                    dino_indices[0].tolist(),
                    dino_metas,
                    score_kind="similarity",
                )
                fused = reciprocal_rank_fusion(
                    {"clip": clip_crop, "dino": dino_crop},
                    weights={
                        "clip": args.clip_weight,
                        "dino": args.dino_weight,
                    },
                    rrf_k=args.rrf_k,
                )
                dino_text_en_fused = reciprocal_rank_fusion(
                    {"dino_image": dino_crop, "clip_text": clip_text_en_crop},
                    weights={
                        "dino_image": args.dino_weight,
                        "clip_text": args.clip_weight,
                    },
                    rrf_k=args.rrf_k,
                )
                dino_text_ko_fused = reciprocal_rank_fusion(
                    {"dino_image": dino_crop, "clip_text": clip_text_ko_crop},
                    weights={
                        "dino_image": args.dino_weight,
                        "clip_text": args.clip_weight,
                    },
                    rrf_k=args.rrf_k,
                )
                rankings = {
                    "clip_all_visual": clip_all,
                    "clip_crop_only": clip_crop,
                    "dino_crop_only": dino_crop,
                    "clip_dino_crop_rrf": fused,
                    "clip_text_en_crop_only": clip_text_en_crop,
                    "clip_text_ko_crop_only": clip_text_ko_crop,
                    "clip_image_text_en_crop_only": clip_image_text_en,
                    "clip_image_text_ko_crop_only": clip_image_text_ko,
                    "dino_image_clip_text_en_rrf": dino_text_en_fused,
                    "dino_image_clip_text_ko_rrf": dino_text_ko_fused,
                }
                records = {
                    system: rank_record(dog_id, ranking, top_k=args.topk)
                    for system, ranking in rankings.items()
                }
                for system, record in records.items():
                    system_rows[system].append(record)
                query_rows.append(
                    {
                        "notice_id": dog_id,
                        "sample_order": previous.get("sample_order"),
                        "payload_sha256": downloaded_image.payload_sha256,
                        "clip_encode_ms": round(clip_encode_ms, 3),
                        "dino_encode_ms": round(dino_encode_ms, 3),
                        "clip_text_en_ms": round(clip_text_en_ms, 3),
                        "clip_text_ko_ms": round(clip_text_ko_ms, 3),
                        "prompts": prompts,
                        "systems": records,
                    }
                )
            finally:
                downloaded_image.image.close()
    finally:
        downloader.close()

    systems: dict[str, Any] = {}
    for system, rows in system_rows.items():
        systems[system] = aggregate_rank_metrics(
            rows,
            attempted_count=attempted,
            downloaded_count=downloaded,
            duplicate_count=0,
        )
    systems["clip_all_visual"]["mean_encode_ms"] = mean_latency(
        query_rows, "clip_encode_ms"
    )
    systems["clip_crop_only"]["mean_encode_ms"] = mean_latency(
        query_rows, "clip_encode_ms"
    )
    systems["dino_crop_only"]["mean_encode_ms"] = mean_latency(
        query_rows, "dino_encode_ms"
    )
    systems["clip_dino_crop_rrf"]["mean_encode_ms"] = round(
        (mean_latency(query_rows, "clip_encode_ms") or 0.0)
        + (mean_latency(query_rows, "dino_encode_ms") or 0.0),
        3,
    )
    for language in ("en", "ko"):
        text_key = f"clip_text_{language}_ms"
        text_ms = mean_latency(query_rows, text_key) or 0.0
        systems[f"clip_text_{language}_crop_only"]["mean_encode_ms"] = round(
            text_ms, 3
        )
        systems[f"clip_image_text_{language}_crop_only"][
            "mean_encode_ms"
        ] = round((mean_latency(query_rows, "clip_encode_ms") or 0.0) + text_ms, 3)
        systems[f"dino_image_clip_text_{language}_rrf"][
            "mean_encode_ms"
        ] = round((mean_latency(query_rows, "dino_encode_ms") or 0.0) + text_ms, 3)
    attribute_relevance = {
        system: attribute_relevance_summary(
            query_rows,
            system=system,
            metas=current_by_id,
            candidate_notice_ids=dino_notice_ids,
        )
        for system in system_rows
    }
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "training_required": False,
        "scope": {
            "query_source": "previously verified distinct secondary photos",
            "query_storage": "memory_only",
            "candidate_corpus": "notices with local Faster R-CNN crops",
            "comparison": [
                "production CLIP full+crop visual corpus",
                "same crop-only corpus with CLIP",
                "same crop-only corpus with DINO",
                "equal-weight crop-only CLIP+DINO RRF",
                "English/Korean CLIP text-to-crop retrieval",
                "CLIP image+text vector averaging on the crop corpus",
                "DINO image + CLIP text rank fusion on the crop corpus",
            ],
            "text_prompt_policy": (
                "bilingual prompts derived only from public color and weight-based size; "
                "candidate text vectors are excluded"
            ),
        },
        "models": {
            "clip": {"id": args.clip_model, "frozen": True},
            "dino": dino_model,
        },
        "corpus": {
            "clip_vectors": int(clip_index.ntotal),
            "dino_crop_vectors": int(dino_index.ntotal),
            "dino_crop_notices": len(dino_notice_ids),
            "previous_evaluable_queries": len(previous_queries),
            "target_covered_queries": attempted,
            "payload_verified_queries": downloaded,
        },
        "fusion": {
            "method": "weighted_reciprocal_rank_fusion",
            "rrf_k": args.rrf_k,
            "weights": {"clip": args.clip_weight, "dino": args.dino_weight},
        },
        "systems": systems,
        "attribute_relevance": attribute_relevance,
        "paired_comparisons": {
            "dino_crop_only_vs_clip_crop_only": paired_rank_comparison(
                query_rows,
                left="dino_crop_only",
                right="clip_crop_only",
            ),
            "dino_crop_only_vs_clip_all_visual": paired_rank_comparison(
                query_rows,
                left="dino_crop_only",
                right="clip_all_visual",
            ),
            "clip_dino_crop_rrf_vs_dino_crop_only": paired_rank_comparison(
                query_rows,
                left="clip_dino_crop_rrf",
                right="dino_crop_only",
            ),
            "dino_text_en_rrf_vs_dino_image": paired_rank_comparison(
                query_rows,
                left="dino_image_clip_text_en_rrf",
                right="dino_crop_only",
            ),
            "dino_text_ko_rrf_vs_dino_image": paired_rank_comparison(
                query_rows,
                left="dino_image_clip_text_ko_rrf",
                right="dino_crop_only",
            ),
            "clip_image_text_en_vs_clip_image": paired_rank_comparison(
                query_rows,
                left="clip_image_text_en_crop_only",
                right="clip_crop_only",
            ),
            "clip_image_text_ko_vs_clip_image": paired_rank_comparison(
                query_rows,
                left="clip_image_text_ko_crop_only",
                right="clip_crop_only",
            ),
        },
        "failures": dict(sorted(failures.items())),
        "queries": query_rows,
        "limitations": [
            "DINOv2 is an infrastructure fallback; this is not a DINOv3/DINOde result.",
            "The pilot evaluates only notices with existing local detector crops.",
            "The query set was selected by a prior CLIP held-out run and is small.",
            "RRF weights are defaults and were not tuned on an independent validation set.",
            "Prompts are metadata-derived controls, not independently authored user queries.",
            "Exact-notice rank does not measure the relevance of other visually valid dogs.",
        ],
    }
    write_json_atomic(args.output, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    summary = {
        "schema_version": report["schema_version"],
        "corpus": report["corpus"],
        "systems": report["systems"],
        "attribute_relevance": report["attribute_relevance"],
        "paired_comparisons": report["paired_comparisons"],
        "failures": report["failures"],
        "output": str(args.output),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
