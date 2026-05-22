import argparse
import csv
import html
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import clip
import faiss
import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.query_expansion import expand_query_text


DATA_DIR = BASE_DIR / "data"
DEFAULT_BASE_PATH = DATA_DIR / "local_dog_cache.json"
DEFAULT_ENRICHED_PATH = DATA_DIR / "local_dog_cache_enriched.json"
DEFAULT_QUERY_PATH = DATA_DIR / "eval_queries.sample.json"
DEFAULT_HTML_PATH = DATA_DIR / "search_eval_report.html"
DEFAULT_JSON_PATH = DATA_DIR / "search_eval_report.json"
DEFAULT_SCORE_CSV_PATH = DATA_DIR / "search_eval_score_sheet.csv"


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def parse_weight_kg(text: str) -> Optional[float]:
    digits = []
    number = ""
    for char in clean_text(text):
        if char.isdigit() or char == ".":
            number += char
        elif number:
            digits.append(number)
            number = ""
    if number:
        digits.append(number)
    if not digits:
        return None
    try:
        return float(digits[0])
    except ValueError:
        return None


def size_label(weight_text: str) -> str:
    kg = parse_weight_kg(weight_text)
    if kg is None:
        return "체형 미상"
    if kg < 10:
        return "소형견"
    if kg < 25:
        return "중형견"
    return "대형견"


def load_items(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    items = payload.get("items") or []
    return [item for item in items if isinstance(item, dict)]


def load_queries(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"queries must be a list: {path}")

    queries: List[Dict[str, Any]] = []
    for index, item in enumerate(payload, start=1):
        if isinstance(item, str):
            query = clean_text(item)
            if not query:
                continue
            queries.append({"id": f"q{index}", "query": query, "goal": "", "match_groups": []})
            continue

        if not isinstance(item, dict):
            continue

        query = clean_text(item.get("query"))
        if not query:
            continue

        groups = []
        for group in item.get("match_groups") or []:
            if not isinstance(group, list):
                continue
            tokens = [clean_text(token).lower() for token in group if clean_text(token)]
            if tokens:
                groups.append(tokens)

        queries.append(
            {
                "id": clean_text(item.get("id")) or f"q{index}",
                "query": query,
                "goal": clean_text(item.get("goal")),
                "match_groups": groups,
            }
        )
    return queries


def build_doc_text(item: Dict[str, Any], *, use_enriched_desc: bool) -> str:
    desc = clean_text(item.get("merged_desc")) if use_enriched_desc else clean_text(item.get("desc"))
    parts = [
        clean_text(item.get("breed_name")) or clean_text(item.get("breed")) or clean_text(item.get("breed_code")),
        size_label(clean_text(item.get("weight"))),
        clean_text(item.get("sex")),
        clean_text(item.get("age")),
        clean_text(item.get("weight")),
        clean_text(item.get("neuter")),
        desc,
    ]
    return " ".join(part for part in parts if part)


def align_docs(
    base_items: List[Dict[str, Any]],
    enriched_items: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    base_map = {
        clean_text(item.get("desertionNo")): item
        for item in base_items
        if clean_text(item.get("desertionNo"))
    }
    enriched_map = {
        clean_text(item.get("desertionNo")): item
        for item in enriched_items
        if clean_text(item.get("desertionNo"))
    }

    common_ids = [did for did in enriched_map if did in base_map]
    common_ids.sort(reverse=True)

    base_docs: List[Dict[str, Any]] = []
    enriched_docs: List[Dict[str, Any]] = []

    for desertion_no in common_ids:
        base_item = base_map[desertion_no]
        enriched_item = enriched_map[desertion_no]

        base_docs.append(
            {
                "desertionNo": desertion_no,
                "breed": clean_text(base_item.get("breed_name")) or clean_text(base_item.get("breed")) or clean_text(base_item.get("breed_code")),
                "sex": clean_text(base_item.get("sex")) or "Unknown",
                "age": clean_text(base_item.get("age")) or "Unknown",
                "weight": clean_text(base_item.get("weight")) or "Unknown",
                "desc": clean_text(base_item.get("desc")),
                "image_url": clean_text(base_item.get("image_url")),
                "detail_url": clean_text(base_item.get("detail_url")),
                "search_text": build_doc_text(base_item, use_enriched_desc=False),
            }
        )
        enriched_docs.append(
            {
                "desertionNo": desertion_no,
                "breed": clean_text(enriched_item.get("breed_name")) or clean_text(enriched_item.get("breed")) or clean_text(enriched_item.get("breed_code")),
                "sex": clean_text(enriched_item.get("sex")) or "Unknown",
                "age": clean_text(enriched_item.get("age")) or "Unknown",
                "weight": clean_text(enriched_item.get("weight")) or "Unknown",
                "desc": clean_text(enriched_item.get("merged_desc")) or clean_text(enriched_item.get("desc")),
                "base_desc": clean_text(enriched_item.get("desc")),
                "vlm_desc": clean_text(enriched_item.get("vlm_desc")),
                "image_url": clean_text(enriched_item.get("image_url")),
                "detail_url": clean_text(enriched_item.get("detail_url")),
                "search_text": build_doc_text(enriched_item, use_enriched_desc=True),
            }
        )

    return base_docs, enriched_docs


@torch.no_grad()
def encode_texts(model: Any, device: str, texts: List[str], batch_size: int = 64) -> np.ndarray:
    vectors: List[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        tokens = clip.tokenize(batch, truncate=True).to(device)
        encoded = model.encode_text(tokens)
        encoded = encoded / encoded.norm(dim=-1, keepdim=True)
        vectors.append(encoded.detach().cpu().numpy().astype("float32"))
    return np.vstack(vectors)


def build_index(vectors: np.ndarray) -> faiss.Index:
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def matches_groups(text: str, groups: List[List[str]]) -> bool:
    if not groups:
        return False
    lowered = clean_text(text).lower()
    for group in groups:
        if not any(token in lowered for token in group):
            return False
    return True


def search_docs(
    *,
    index: faiss.Index,
    query_vector: np.ndarray,
    docs: List[Dict[str, Any]],
    topk: int,
    groups: List[List[str]],
) -> List[Dict[str, Any]]:
    scores, indices = index.search(query_vector, topk)
    results: List[Dict[str, Any]] = []

    for rank, (score, idx) in enumerate(zip(scores[0].tolist(), indices[0].tolist()), start=1):
        if idx < 0:
            continue
        doc = docs[idx]
        result = {
            "rank": rank,
            "score": round(float(score), 4),
            "is_hit": matches_groups(doc.get("search_text", ""), groups),
            "desertionNo": doc.get("desertionNo"),
            "breed": doc.get("breed"),
            "sex": doc.get("sex"),
            "age": doc.get("age"),
            "weight": doc.get("weight"),
            "desc": doc.get("desc"),
            "base_desc": doc.get("base_desc", ""),
            "vlm_desc": doc.get("vlm_desc", ""),
            "image_url": doc.get("image_url"),
            "detail_url": doc.get("detail_url"),
        }
        results.append(result)
    return results


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    hit_count = sum(1 for item in results if item["is_hit"])
    return {
        "top1_hit": bool(results and results[0]["is_hit"]),
        "hit_count": hit_count,
        "hit_rate": round(hit_count / len(results), 3) if results else 0.0,
    }


def build_query_report(
    *,
    model: Any,
    device: str,
    base_index: faiss.Index,
    enriched_index: faiss.Index,
    base_docs: List[Dict[str, Any]],
    enriched_docs: List[Dict[str, Any]],
    queries: List[Dict[str, Any]],
    topk: int,
) -> List[Dict[str, Any]]:
    expanded_queries = [expand_query_text(query["query"]) for query in queries]
    query_vectors = encode_texts(model, device, expanded_queries, batch_size=32)
    reports: List[Dict[str, Any]] = []

    for query, expanded_query, vector in zip(queries, expanded_queries, query_vectors):
        query_vector = vector.reshape(1, -1)
        base_results = search_docs(
            index=base_index,
            query_vector=query_vector,
            docs=base_docs,
            topk=topk,
            groups=query["match_groups"],
        )
        enriched_results = search_docs(
            index=enriched_index,
            query_vector=query_vector,
            docs=enriched_docs,
            topk=topk,
            groups=query["match_groups"],
        )

        reports.append(
            {
                "id": query["id"],
                "query": query["query"],
                "goal": query["goal"],
                "expanded_query": expanded_query,
                "match_groups": query["match_groups"],
                "baseline": {
                    "summary": summarize_results(base_results),
                    "results": base_results,
                },
                "enriched": {
                    "summary": summarize_results(enriched_results),
                    "results": enriched_results,
                },
            }
        )

    return reports


def build_summary(query_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    baseline_top1 = [report["baseline"]["summary"]["top1_hit"] for report in query_reports]
    enriched_top1 = [report["enriched"]["summary"]["top1_hit"] for report in query_reports]
    baseline_rate = [report["baseline"]["summary"]["hit_rate"] for report in query_reports]
    enriched_rate = [report["enriched"]["summary"]["hit_rate"] for report in query_reports]

    better = 0
    tied = 0
    worse = 0
    for report in query_reports:
        base_hit = report["baseline"]["summary"]["hit_count"]
        enriched_hit = report["enriched"]["summary"]["hit_count"]
        if enriched_hit > base_hit:
            better += 1
        elif enriched_hit < base_hit:
            worse += 1
        else:
            tied += 1

    return {
        "query_count": len(query_reports),
        "baseline_top1_hit_rate": round(mean(baseline_top1), 3) if baseline_top1 else 0.0,
        "enriched_top1_hit_rate": round(mean(enriched_top1), 3) if enriched_top1 else 0.0,
        "baseline_avg_hit_rate_at_k": round(mean(baseline_rate), 3) if baseline_rate else 0.0,
        "enriched_avg_hit_rate_at_k": round(mean(enriched_rate), 3) if enriched_rate else 0.0,
        "enriched_better_queries": better,
        "tied_queries": tied,
        "enriched_worse_queries": worse,
    }


def result_card(result: Dict[str, Any]) -> str:
    hit_badge = "<span class='badge hit'>hit</span>" if result["is_hit"] else "<span class='badge miss'>miss</span>"
    image_html = ""
    image_url = clean_text(result.get("image_url"))
    if image_url:
        image_html = f"<img src='{html.escape(image_url)}' alt='dog image' loading='lazy' />"

    detail_url = clean_text(result.get("detail_url"))
    link_html = ""
    if detail_url:
        link_html = f"<a href='{html.escape(detail_url)}' target='_blank' rel='noreferrer'>상세 보기</a>"

    desc = html.escape(clean_text(result.get("desc")) or "(설명 없음)")
    return f"""
    <article class="card {'hit-card' if result['is_hit'] else ''}">
      {image_html}
      <div class="card-body">
        <div class="card-head">
          <strong>#{result['rank']} {html.escape(clean_text(result.get('breed')) or 'Unknown')}</strong>
          <span>score {result['score']}</span>
        </div>
        <div class="meta">
          <span>{html.escape(clean_text(result.get('sex')) or 'Unknown')}</span>
          <span>{html.escape(clean_text(result.get('age')) or 'Unknown')}</span>
          <span>{html.escape(clean_text(result.get('weight')) or 'Unknown')}</span>
          {hit_badge}
        </div>
        <p>{desc}</p>
        <div class="links">
          <span>공고번호 {html.escape(clean_text(result.get('desertionNo')))}</span>
          {link_html}
        </div>
      </div>
    </article>
    """


def build_html(summary: Dict[str, Any], reports: List[Dict[str, Any]], topk: int) -> str:
    sections: List[str] = []
    for report in reports:
        groups = ", ".join(" / ".join(group) for group in report["match_groups"]) or "수동 평가 전용"
        baseline_summary = report["baseline"]["summary"]
        enriched_summary = report["enriched"]["summary"]
        sections.append(
            f"""
            <section class="query-section">
              <div class="query-head">
                <div>
                  <h2>{html.escape(report['id'])}. {html.escape(report['query'])}</h2>
                  <p>{html.escape(report['goal'] or '설명 기반 후보 비교')}</p>
                  <p class="match-groups">match groups: {html.escape(groups)}</p>
                </div>
                <div class="query-stats">
                  <div>
                    <strong>Baseline</strong>
                    <span>top1 hit {baseline_summary['top1_hit']}</span>
                    <span>hit@{topk} {baseline_summary['hit_count']}/{topk}</span>
                  </div>
                  <div>
                    <strong>Enriched</strong>
                    <span>top1 hit {enriched_summary['top1_hit']}</span>
                    <span>hit@{topk} {enriched_summary['hit_count']}/{topk}</span>
                  </div>
                </div>
              </div>
              <div class="compare-grid">
                <div>
                  <h3>Baseline</h3>
                  <div class="card-grid">
                    {''.join(result_card(item) for item in report['baseline']['results'])}
                  </div>
                </div>
                <div>
                  <h3>Enriched</h3>
                  <div class="card-grid">
                    {''.join(result_card(item) for item in report['enriched']['results'])}
                  </div>
                </div>
              </div>
            </section>
            """
        )

    return f"""<!doctype html>
<html lang="ko">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Search Eval Report</title>
    <style>
      :root {{
        --bg: #f5efe5;
        --panel: rgba(255, 251, 246, 0.96);
        --ink: #231813;
        --muted: #6d6157;
        --line: rgba(35, 24, 19, 0.12);
        --accent: #c86e3d;
        --hit: #1f8f55;
        --miss: #8d8d8d;
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        color: var(--ink);
        font-family: "Gill Sans", "Noto Sans KR", sans-serif;
        background:
          radial-gradient(circle at top left, rgba(255,255,255,0.9), transparent 30%),
          linear-gradient(135deg, #f7f0e8 0%, #ead9c9 100%);
      }}
      main {{
        width: min(1500px, calc(100vw - 28px));
        margin: 20px auto 36px;
      }}
      .hero, .query-section {{
        border: 1px solid var(--line);
        border-radius: 28px;
        background: var(--panel);
        box-shadow: 0 18px 48px rgba(35, 24, 19, 0.08);
      }}
      .hero {{
        padding: 26px 28px;
      }}
      h1 {{
        margin: 0 0 10px;
        font-size: clamp(2rem, 3vw, 3.2rem);
        line-height: 0.95;
        letter-spacing: -0.04em;
      }}
      .hero p {{
        margin: 6px 0 0;
        color: var(--muted);
      }}
      .summary-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 12px;
        margin-top: 18px;
      }}
      .summary-card {{
        padding: 16px 18px;
        border-radius: 20px;
        border: 1px solid var(--line);
        background: rgba(255,255,255,0.82);
      }}
      .summary-card strong {{
        display: block;
        font-size: 1.2rem;
        margin-bottom: 4px;
      }}
      .query-section {{
        margin-top: 18px;
        padding: 22px;
      }}
      .query-head {{
        display: flex;
        justify-content: space-between;
        gap: 18px;
        align-items: start;
      }}
      .query-head h2 {{
        margin: 0 0 6px;
        font-size: 1.5rem;
      }}
      .query-head p {{
        margin: 4px 0;
        color: var(--muted);
      }}
      .query-stats {{
        display: grid;
        grid-template-columns: repeat(2, minmax(160px, 1fr));
        gap: 10px;
      }}
      .query-stats > div {{
        padding: 12px 14px;
        border-radius: 18px;
        border: 1px solid var(--line);
        background: rgba(255,255,255,0.78);
        display: grid;
        gap: 4px;
      }}
      .compare-grid {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 18px;
        margin-top: 18px;
      }}
      .compare-grid h3 {{
        margin: 0 0 12px;
      }}
      .card-grid {{
        display: grid;
        gap: 12px;
      }}
      .card {{
        display: grid;
        grid-template-columns: 140px 1fr;
        gap: 14px;
        overflow: hidden;
        border-radius: 22px;
        border: 1px solid var(--line);
        background: rgba(255,255,255,0.82);
      }}
      .hit-card {{
        border-color: rgba(31, 143, 85, 0.35);
      }}
      .card img {{
        width: 100%;
        height: 100%;
        max-height: 160px;
        object-fit: cover;
        background: #e8ddcf;
      }}
      .card-body {{
        padding: 14px 16px 14px 0;
        display: grid;
        gap: 8px;
      }}
      .card-head {{
        display: flex;
        justify-content: space-between;
        gap: 10px;
      }}
      .meta, .links {{
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        color: var(--muted);
        font-size: 0.92rem;
      }}
      .badge {{
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 700;
      }}
      .badge.hit {{
        background: rgba(31, 143, 85, 0.12);
        color: var(--hit);
      }}
      .badge.miss {{
        background: rgba(141, 141, 141, 0.12);
        color: var(--miss);
      }}
      a {{
        color: var(--accent);
        text-decoration: none;
      }}
      @media (max-width: 1100px) {{
        .compare-grid {{
          grid-template-columns: 1fr;
        }}
      }}
      @media (max-width: 760px) {{
        .query-head {{
          display: grid;
        }}
        .query-stats {{
          grid-template-columns: 1fr;
        }}
        .card {{
          grid-template-columns: 1fr;
        }}
        .card img {{
          max-height: 240px;
        }}
        .card-body {{
          padding: 0 14px 14px;
        }}
      }}
    </style>
  </head>
  <body>
    <main>
      <section class="hero">
        <h1>Search Eval Report</h1>
        <p>보강 전 설명 기반 검색 결과와 VLM 보강 설명 기반 검색 결과를 같은 질의셋으로 비교한 리포트입니다.</p>
        <p>hit 판정은 match groups 기반 휴리스틱이라, 최종 평가는 사람 검토와 함께 보는 것을 권장합니다.</p>
        <div class="summary-grid">
          <div class="summary-card">
            <strong>{summary['baseline_top1_hit_rate']}</strong>
            <span>baseline top1 hit rate</span>
          </div>
          <div class="summary-card">
            <strong>{summary['enriched_top1_hit_rate']}</strong>
            <span>enriched top1 hit rate</span>
          </div>
          <div class="summary-card">
            <strong>{summary['baseline_avg_hit_rate_at_k']}</strong>
            <span>baseline avg hit@{topk}</span>
          </div>
          <div class="summary-card">
            <strong>{summary['enriched_avg_hit_rate_at_k']}</strong>
            <span>enriched avg hit@{topk}</span>
          </div>
          <div class="summary-card">
            <strong>{summary['enriched_better_queries']}</strong>
            <span>enriched better queries</span>
          </div>
          <div class="summary-card">
            <strong>{summary['tied_queries']}</strong>
            <span>tied queries</span>
          </div>
        </div>
      </section>
      {''.join(sections)}
    </main>
  </body>
</html>
"""


def write_score_sheet(path: Path, reports: List[Dict[str, Any]], topk: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "query_id",
                "query",
                f"baseline_hit_count_at_{topk}",
                f"enriched_hit_count_at_{topk}",
                "baseline_top1_hit",
                "enriched_top1_hit",
                "reviewer_baseline_score_1to5",
                "reviewer_enriched_score_1to5",
                "winner",
            ]
        )
        for report in reports:
            writer.writerow(
                [
                    report["id"],
                    report["query"],
                    report["baseline"]["summary"]["hit_count"],
                    report["enriched"]["summary"]["hit_count"],
                    report["baseline"]["summary"]["top1_hit"],
                    report["enriched"]["summary"]["top1_hit"],
                    "",
                    "",
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="보강 전/후 검색 결과를 side-by-side 리포트로 생성합니다."
    )
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE_PATH, help="원본 공고 JSON")
    parser.add_argument("--enriched", type=Path, default=DEFAULT_ENRICHED_PATH, help="보강 공고 JSON")
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERY_PATH, help="평가 질의 JSON")
    parser.add_argument("--topk", type=int, default=5, help="질의별 비교할 상위 결과 수")
    parser.add_argument("--clip-model", default="ViT-B/32", help="CLIP 모델 이름")
    parser.add_argument("--html-out", type=Path, default=DEFAULT_HTML_PATH, help="HTML 리포트 저장 경로")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_PATH, help="JSON 리포트 저장 경로")
    parser.add_argument(
        "--score-sheet-out",
        type=Path,
        default=DEFAULT_SCORE_CSV_PATH,
        help="수동 평가용 CSV 저장 경로",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queries = load_queries(args.queries)
    base_items = load_items(args.base)
    enriched_items = load_items(args.enriched)
    base_docs, enriched_docs = align_docs(base_items, enriched_items)
    if not base_docs or not enriched_docs:
        raise SystemExit("[ERROR] 비교 가능한 공고가 없습니다.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}")
    print(f"[INFO] aligned_docs={len(base_docs)} queries={len(queries)}")

    model, _ = clip.load(args.clip_model, device=device)
    model.eval()

    base_vectors = encode_texts(model, device, [doc["search_text"] for doc in base_docs])
    enriched_vectors = encode_texts(model, device, [doc["search_text"] for doc in enriched_docs])
    base_index = build_index(base_vectors)
    enriched_index = build_index(enriched_vectors)

    reports = build_query_report(
        model=model,
        device=device,
        base_index=base_index,
        enriched_index=enriched_index,
        base_docs=base_docs,
        enriched_docs=enriched_docs,
        queries=queries,
        topk=args.topk,
    )
    summary = build_summary(reports)

    html_text = build_html(summary, reports, args.topk)
    args.html_out.parent.mkdir(parents=True, exist_ok=True)
    args.html_out.write_text(html_text, encoding="utf-8")

    payload = {
        "base_path": str(args.base),
        "enriched_path": str(args.enriched),
        "queries_path": str(args.queries),
        "aligned_docs": len(base_docs),
        "topk": args.topk,
        "summary": summary,
        "reports": reports,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_score_sheet(args.score_sheet_out, reports, args.topk)

    print(f"[DONE] html -> {args.html_out}")
    print(f"[DONE] json -> {args.json_out}")
    print(f"[DONE] score sheet -> {args.score_sheet_out}")


if __name__ == "__main__":
    main()
