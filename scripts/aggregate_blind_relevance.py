"""Aggregate complete blinded relevance CSVs into nDCG and query wins."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.blind_relevance_evaluation import (  # noqa: E402
    aggregate_judgments,
    load_json,
    load_judgments,
    render_result_markdown,
    sha256_file,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate one or more complete reviewer CSVs and aggregate blinded "
            "graded relevance into system nDCG and query-level wins."
        )
    )
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument(
        "--labels",
        type=Path,
        nargs="+",
        required=True,
        help="CSV files exported by the task HTML.",
    )
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        key = load_json(args.key)
        if not isinstance(key, dict):
            raise ValueError("key must contain a JSON object")
        judgments, _source_rows = load_judgments(args.labels, key)
        report = aggregate_judgments(key, judgments)
        report["label_artifacts"] = [
            {
                "file": path.name,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in args.labels
        ]
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if args.markdown_out:
            args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
            args.markdown_out.write_text(
                render_result_markdown(report),
                encoding="utf-8",
            )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"blind relevance aggregation failed: {exc}", file=sys.stderr)
        return 1

    sample = report["sample"]
    print(
        "blind relevance aggregation complete: "
        f"reviewers={sample['reviewers']} queries={sample['queries']} "
        f"judgments={sample['judgments']}"
    )
    for system_id, result in report["systems"].items():
        metric = next(key for key in result if key.startswith("nDCG@"))
        print(f"{system_id}: {metric}={result[metric]:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
