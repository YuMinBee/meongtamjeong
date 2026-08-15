from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import clip
import faiss
import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.hybrid_rag import (  # noqa: E402
    BM25Index,
    build_hybrid_documents,
    build_search_query_text,
    parse_structured_query,
    rank_hybrid_documents,
    vector_hits_to_doc_modality_scores,
)
from app.notice_status import is_searchable_notice  # noqa: E402

DATA_DIR = BASE_DIR / "data"
DEFAULT_QUERIES = DATA_DIR / "eval_queries.sample.json"
DEFAULT_OUTPUT = DATA_DIR / "image_attr_eval_report.json"


def read_faiss_index(path: Path) -> Any:
    try:
        return faiss.read_index(str(path))
    except RuntimeError as exc:
        if "Illegal byte sequence" not in str(exc):
            raise
        safe_dir = Path(tempfile.gettempdir()) / "dog_faiss_safe"
        safe_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
        safe_path = safe_dir / f"{path.stem}-{digest}{path.suffix}"
        if (
            not safe_path.exists()
            or safe_path.stat().st_size != path.stat().st_size
            or safe_path.stat().st_mtime < path.stat().st_mtime
        ):
            shutil.copy2(path, safe_path)
        return faiss.read_index(str(safe_path))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def encode_text(model: Any, device: str, text: str) -> np.ndarray:
    tokens = clip.tokenize([text], truncate=True).to(device)
    with torch.no_grad():
        feat = model.encode_text(tokens)
        feat /= feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().astype("float32")


def result_text(item: Dict[str, Any]) -> str:
    doc = item.get("doc") or {}
    meta = doc.get("meta") or {}
    bits = [doc.get("text", "")]
    for key in (
        "breed",
        "breed_name",
        "sex",
        "age",
        "weight",
        "desc",
        "desc_full",
        "specialMark",
        "vlm_desc",
        "merged_desc",
        "vlm_attr_text",
    ):
        bits.append(clean_text(meta.get(key)))
    return " ".join(bit for bit in bits if clean_text(bit)).lower()


def matches_groups(item: Dict[str, Any], groups: Iterable[Iterable[str]]) -> bool:
    text = result_text(item)
    normalized_groups = [list(group) for group in groups or []]
    if not normalized_groups:
        return False
    for group in normalized_groups:
        terms = [clean_text(term).lower() for term in group if clean_text(term)]
        if terms and not any(term in text for term in terms):
            return False
    return True


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def doc_has_crop_vector(item: Dict[str, Any], metas: List[Dict[str, Any]]) -> bool:
    doc = item.get("doc") or {}
    for vector_index in doc.get("vector_indices") or []:
        if isinstance(vector_index, int) and 0 <= vector_index < len(metas):
            if metas[vector_index].get("type") == "crop_image":
                return True
    return False


def image_attr_coverage(metas: List[Dict[str, Any]]) -> Dict[str, Any]:
    doc_ids = set()
    covered = set()
    detected = set()
    scored = set()
    for meta in metas:
        doc_id = clean_text(meta.get("desertionNo")) or str(len(doc_ids))
        doc_ids.add(doc_id)
        attrs = meta.get("image_attrs") if isinstance(meta.get("image_attrs"), dict) else {}
        if attrs:
            covered.add(doc_id)
        if attrs.get("target_detected") is True or attrs.get("dog_detected") is True:
            detected.add(doc_id)
        if isinstance(attrs.get("photo_quality_score"), (int, float)):
            scored.add(doc_id)
    total = len(doc_ids) or 1
    return {
        "docs": len(doc_ids),
        "image_attrs_docs": len(covered),
        "image_attrs_coverage": round(len(covered) / total, 4),
        "detected_docs": len(detected),
        "detected_rate": round(len(detected) / total, 4),
        "photo_quality_docs": len(scored),
        "photo_quality_coverage": round(len(scored) / total, 4),
    }


def evaluate_index(
    label: str,
    index_path: Path,
    metas_path: Path,
    queries: List[Dict[str, Any]],
    model: Any,
    device: str,
    topk: int,
    rerank_depth: int,
) -> Dict[str, Any]:
    index = read_faiss_index(index_path)
    metas = load_json(metas_path)
    if index.ntotal != len(metas):
        raise ValueError(f"{label}: index({index.ntotal}) != metas({len(metas)})")

    docs, vector_to_doc = build_hybrid_documents(metas)
    bm25 = BM25Index(docs)
    query_reports: List[Dict[str, Any]] = []
    total_time_ms = 0.0

    for query in queries:
        raw_query = clean_text(query.get("query"))
        structured = parse_structured_query(raw_query)
        query_text = build_search_query_text(raw_query, structured)
        query_vec = encode_text(model, device, query_text or raw_query)

        started = time.perf_counter()
        vector_depth = min(index.ntotal, max(rerank_depth * 3, topk * 30))
        distances, indices = index.search(query_vec, vector_depth)
        vector_scores, vector_score_details = vector_hits_to_doc_modality_scores(
            distances[0].tolist(),
            indices[0].tolist(),
            vector_to_doc,
            metas,
        )
        ranked = rank_hybrid_documents(
            docs,
            bm25,
            structured,
            query_text or raw_query,
            vector_scores,
            vector_score_details=vector_score_details,
            topk=topk,
            rerank_depth=rerank_depth,
            filter_inactive_notices=True,
            include_unknown_notices=True,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        total_time_ms += elapsed_ms

        groups = query.get("match_groups") or []
        hits = [item for item in ranked if matches_groups(item, groups)]
        photo_scores = [float((item.get("score_parts") or {}).get("photo_quality")) for item in ranked if isinstance((item.get("score_parts") or {}).get("photo_quality"), (int, float))]
        visual_scores = [float((item.get("score_parts") or {}).get("visual_attrs")) for item in ranked if isinstance((item.get("score_parts") or {}).get("visual_attrs"), (int, float))]
        searchable = [item for item in ranked if is_searchable_notice((item.get("doc") or {}).get("meta") or {}, include_unknown=True)]
        query_reports.append(
            {
                "id": clean_text(query.get("id")),
                "query": raw_query,
                "hit_at_k": bool(hits),
                "top1_match": bool(ranked and matches_groups(ranked[0], groups)),
                "match_count_at_k": len(hits),
                "result_count": len(ranked),
                "searchable_count": len(searchable),
                "avg_photo_quality_at_k": round(mean(photo_scores), 4),
                "avg_visual_attr_score_at_k": round(mean(visual_scores), 4),
                "crop_candidate_count_at_k": sum(1 for item in ranked if doc_has_crop_vector(item, metas)),
                "elapsed_ms": round(elapsed_ms, 2),
                "top_desertionNo": clean_text(((ranked[0].get("doc") or {}).get("meta") or {}).get("desertionNo")) if ranked else "",
            }
        )

    query_count = len(query_reports) or 1
    return {
        "label": label,
        "index": str(index_path),
        "metas": str(metas_path),
        "vectors": int(index.ntotal),
        "metas_count": len(metas),
        "docs_count": len(docs),
        "topk": topk,
        "rerank_depth": rerank_depth,
        "hit_at_k": round(sum(1 for item in query_reports if item["hit_at_k"]) / query_count, 4),
        "top1_match": round(sum(1 for item in query_reports if item["top1_match"]) / query_count, 4),
        "avg_match_count_at_k": round(mean([float(item["match_count_at_k"]) for item in query_reports]), 4),
        "avg_photo_quality_at_k": round(mean([float(item["avg_photo_quality_at_k"]) for item in query_reports]), 4),
        "avg_visual_attr_score_at_k": round(mean([float(item["avg_visual_attr_score_at_k"]) for item in query_reports]), 4),
        "avg_crop_candidate_count_at_k": round(mean([float(item["crop_candidate_count_at_k"]) for item in query_reports]), 4),
        "avg_latency_ms": round(total_time_ms / query_count, 2),
        "image_attr_coverage": image_attr_coverage(metas),
        "queries": query_reports,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline and image-attribute enhanced search indexes.")
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--baseline-index", type=Path, default=DATA_DIR / "dog_faiss.index")
    parser.add_argument("--baseline-metas", type=Path, default=DATA_DIR / "dog_metas.json")
    parser.add_argument("--candidate-index", type=Path, default=DATA_DIR / "dog_faiss_image_attrs.index")
    parser.add_argument("--candidate-metas", type=Path, default=DATA_DIR / "dog_metas_image_attrs.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--rerank-depth", type=int, default=80)
    parser.add_argument("--clip-model", default="ViT-B/32")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load(args.clip_model, device=device)
    model.eval()
    queries = load_json(args.queries)

    baseline = evaluate_index("baseline", args.baseline_index, args.baseline_metas, queries, model, device, args.topk, args.rerank_depth)
    candidate = evaluate_index("candidate", args.candidate_index, args.candidate_metas, queries, model, device, args.topk, args.rerank_depth)
    report = {
        "device": device,
        "queries": str(args.queries),
        "baseline": baseline,
        "candidate": candidate,
        "delta": {
            "hit_at_k": round(candidate["hit_at_k"] - baseline["hit_at_k"], 4),
            "top1_match": round(candidate["top1_match"] - baseline["top1_match"], 4),
            "avg_photo_quality_at_k": round(candidate["avg_photo_quality_at_k"] - baseline["avg_photo_quality_at_k"], 4),
            "avg_visual_attr_score_at_k": round(candidate["avg_visual_attr_score_at_k"] - baseline["avg_visual_attr_score_at_k"], 4),
            "avg_latency_ms": round(candidate["avg_latency_ms"] - baseline["avg_latency_ms"], 2),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "baseline": {k: baseline[k] for k in ("hit_at_k", "top1_match", "avg_latency_ms")}, "candidate": {k: candidate[k] for k in ("hit_at_k", "top1_match", "avg_latency_ms")}, "delta": report["delta"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
