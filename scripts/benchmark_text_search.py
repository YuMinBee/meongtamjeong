import argparse
import json
import math
import time
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List

import clip
import faiss
import torch


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
LEGACY_DIR = BASE_DIR / "archive" / "legacy_dog_embedding"
DEFAULT_QUERY_PATH = DATA_DIR / "eval_queries.sample.json"

DEFAULT_QUERIES = [
    "흰색 소형견",
    "검은색 강아지",
    "귀가 쫑긋 선 강아지",
    "갈색 털 중형견",
    "복슬복슬한 강아지",
    "대형견",
    "작고 마른 체형의 강아지",
    "리트리버 같은 강아지",
]


def percentile(values: List[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * ratio
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    lower_value = ordered[lower]
    upper_value = ordered[upper]
    return lower_value + (upper_value - lower_value) * (index - lower)


def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
    values_ms = [value * 1000.0 for value in values]
    return {
        "avg_ms": round(mean(values_ms), 3),
        "p50_ms": round(median(values_ms), 3),
        "p95_ms": round(percentile(values_ms, 0.95), 3),
        "min_ms": round(min(values_ms), 3),
        "max_ms": round(max(values_ms), 3),
    }


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def load_queries(path: Path) -> List[str]:
    if not path.exists():
        return DEFAULT_QUERIES

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"queries must be a list: {path}")

    queries: List[str] = []
    for item in payload:
        if isinstance(item, str):
            query = clean_text(item)
        elif isinstance(item, dict):
            query = clean_text(item.get("query"))
        else:
            query = ""
        if query:
            queries.append(query)

    return queries or DEFAULT_QUERIES


@torch.no_grad()
def embed_text(model: Any, device: str, query: str):
    tokens = clip.tokenize([query], truncate=True).to(device)
    vector = model.encode_text(tokens)
    vector = vector / vector.norm(dim=-1, keepdim=True)
    return vector.detach().cpu().numpy().astype("float32")


def benchmark_system(
    *,
    label: str,
    index_path: Path,
    metas_path: Path,
    model: Any,
    device: str,
    queries: List[str],
    topk: int,
    warmup: int,
    repeat: int,
) -> Dict[str, Any]:
    load_started = time.perf_counter()
    index = faiss.read_index(str(index_path))
    metas = json.loads(metas_path.read_text(encoding="utf-8"))
    load_elapsed = time.perf_counter() - load_started

    metadata_warning = ""
    if index.ntotal != len(metas):
        metadata_warning = f"Index({index.ntotal}) != Metas({len(metas)})"

    embed_times: List[float] = []
    search_times: List[float] = []
    total_times: List[float] = []
    per_query: List[Dict[str, Any]] = []

    for query in queries:
        for _ in range(max(0, warmup)):
            vector = embed_text(model=model, device=device, query=query)
            index.search(vector, topk)

        query_embed_times: List[float] = []
        query_search_times: List[float] = []
        query_total_times: List[float] = []

        for _ in range(max(1, repeat)):
            started = time.perf_counter()
            vector = embed_text(model=model, device=device, query=query)
            embedded = time.perf_counter()
            index.search(vector, topk)
            finished = time.perf_counter()

            embed_elapsed = embedded - started
            search_elapsed = finished - embedded
            total_elapsed = finished - started

            embed_times.append(embed_elapsed)
            search_times.append(search_elapsed)
            total_times.append(total_elapsed)
            query_embed_times.append(embed_elapsed)
            query_search_times.append(search_elapsed)
            query_total_times.append(total_elapsed)

        per_query.append(
            {
                "query": query,
                "embed": summarize(query_embed_times),
                "search": summarize(query_search_times),
                "total": summarize(query_total_times),
            }
        )

    return {
        "label": label,
        "index_path": str(index_path),
        "metas_path": str(metas_path),
        "index_size": index.ntotal,
        "metas_count": len(metas),
        "metadata_warning": metadata_warning,
        "load_ms": round(load_elapsed * 1000.0, 3),
        "embed": summarize(embed_times),
        "search": summarize(search_times),
        "total": summarize(total_times),
        "queries": per_query,
    }


def print_summary(result: Dict[str, Any]) -> None:
    print("=" * 72)
    print(
        f"[{result['label']}] index_size={result['index_size']} metas={result['metas_count']} "
        f"load={result['load_ms']}ms"
    )
    if result["metadata_warning"]:
        print(f"  warning: {result['metadata_warning']}")
    print(
        "  embed  avg={avg_ms}ms p50={p50_ms}ms p95={p95_ms}ms".format(
            **result["embed"]
        )
    )
    print(
        "  search avg={avg_ms}ms p50={p50_ms}ms p95={p95_ms}ms".format(
            **result["search"]
        )
    )
    print(
        "  total  avg={avg_ms}ms p50={p50_ms}ms p95={p95_ms}ms".format(
            **result["total"]
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="레거시/현재 텍스트 검색 추론 속도를 비교합니다."
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=DEFAULT_QUERY_PATH,
        help="질의 목록 JSON 경로. 없으면 기본 샘플 질의를 사용합니다.",
    )
    parser.add_argument("--topk", type=int, default=20, help="FAISS 검색 topk")
    parser.add_argument("--warmup", type=int, default=3, help="질의별 워밍업 횟수")
    parser.add_argument("--repeat", type=int, default=20, help="질의별 실측 반복 횟수")
    parser.add_argument(
        "--clip-model",
        default="ViT-B/32",
        help="CLIP 모델 이름",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_DIR / "benchmark_text_search.json",
        help="결과 JSON 저장 경로",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queries = load_queries(args.queries)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] device={device}")
    print(f"[INFO] queries={len(queries)} repeat={args.repeat} warmup={args.warmup}")
    model, _ = clip.load(args.clip_model, device=device)
    model.eval()

    systems = [
        {
            "label": "legacy",
            "index_path": LEGACY_DIR / "dog_faiss.index",
            "metas_path": LEGACY_DIR / "dog_metas.json",
        },
        {
            "label": "current",
            "index_path": DATA_DIR / "dog_faiss.index",
            "metas_path": DATA_DIR / "dog_metas.json",
        },
    ]

    results: List[Dict[str, Any]] = []
    for system in systems:
        result = benchmark_system(
            label=system["label"],
            index_path=system["index_path"],
            metas_path=system["metas_path"],
            model=model,
            device=device,
            queries=queries,
            topk=args.topk,
            warmup=args.warmup,
            repeat=args.repeat,
        )
        results.append(result)
        print_summary(result)

    output_payload = {
        "device": device,
        "clip_model": args.clip_model,
        "queries": queries,
        "topk": args.topk,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[DONE] saved -> {args.output}")


if __name__ == "__main__":
    main()
