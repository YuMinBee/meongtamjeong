import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional


VALID_WINNERS = {"baseline", "enriched", "tie"}


def parse_score(value: str) -> Optional[float]:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def load_review_rows(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    current: Dict[str, str] = {}

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[query]"):
            if current:
                rows.append(current)
            body = line[len("[query]"):].strip()
            query_id, _, query = body.partition("|")
            current = {
                "query_id": query_id.strip(),
                "query": query.strip(),
                "reviewer_baseline_score_1to5": "",
                "reviewer_enriched_score_1to5": "",
                "winner": "",
            }
            continue
        if "=" not in line or not current:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        mapping = {
            "baseline_score": "reviewer_baseline_score_1to5",
            "enriched_score": "reviewer_enriched_score_1to5",
            "winner": "winner",
        }
        if key in mapping:
            current[mapping[key]] = value
    if current:
        rows.append(current)
    return rows


def load_rows(path: Path) -> List[Dict[str, str]]:
    if path.suffix.lower() == ".txt":
        return load_review_rows(path)
    return load_csv_rows(path)


def summarize(rows: List[Dict[str, str]]) -> Dict[str, object]:
    baseline_scores: List[float] = []
    enriched_scores: List[float] = []
    winner_counts = {"baseline": 0, "enriched": 0, "tie": 0, "unfilled": 0}
    score_deltas: List[float] = []
    unscored_queries: List[str] = []

    for row in rows:
        query_id = (row.get("query_id") or "").strip()
        query = (row.get("query") or "").strip()
        baseline = parse_score(row.get("reviewer_baseline_score_1to5", ""))
        enriched = parse_score(row.get("reviewer_enriched_score_1to5", ""))
        winner = (row.get("winner") or "").strip().lower()

        if baseline is not None:
            baseline_scores.append(baseline)
        if enriched is not None:
            enriched_scores.append(enriched)
        if baseline is not None and enriched is not None:
            score_deltas.append(enriched - baseline)
        if baseline is None or enriched is None or winner not in VALID_WINNERS:
            winner_counts["unfilled"] += 1
            unscored_queries.append(f"{query_id}:{query}")
            continue
        winner_counts[winner] += 1

    return {
        "query_count": len(rows),
        "scored_query_count": len(rows) - winner_counts["unfilled"],
        "baseline_avg_score": round(mean(baseline_scores), 3) if baseline_scores else None,
        "enriched_avg_score": round(mean(enriched_scores), 3) if enriched_scores else None,
        "avg_score_delta": round(mean(score_deltas), 3) if score_deltas else None,
        "winner_counts": winner_counts,
        "unscored_queries": unscored_queries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="수동 평가 CSV를 집계합니다.")
    parser.add_argument("csv_path", type=Path, help="search_eval score sheet CSV 경로")
    parser.add_argument(
        "--json",
        action="store_true",
        help="결과를 JSON으로 출력합니다.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.csv_path)
    summary = summarize(rows)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    print(f"queries={summary['query_count']} scored={summary['scored_query_count']}")
    print(f"baseline_avg_score={summary['baseline_avg_score']}")
    print(f"enriched_avg_score={summary['enriched_avg_score']}")
    print(f"avg_score_delta={summary['avg_score_delta']}")
    print(f"winner_counts={summary['winner_counts']}")
    if summary['unscored_queries']:
        print("unscored_queries=")
        for item in summary['unscored_queries']:
            print(f"- {item}")


if __name__ == "__main__":
    main()
