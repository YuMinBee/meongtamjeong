"""Generate a local blinded human relevance task from a retrieval report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.blind_relevance_evaluation import (  # noqa: E402
    build_blind_task,
    load_json,
    write_task_artifacts,
)


DEFAULT_REPORT = ROOT / "docs" / "evaluation" / "retrieval_eval.appearance_v1.json"
DEFAULT_METAS = ROOT / "data" / "dog_metas.json"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "reports" / "blind_relevance"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pool top-k results from multiple retrieval systems, hide system "
            "names and ranks, and create a local HTML/CSV relevance task."
        )
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--metas", type=Path, default=DEFAULT_METAS)
    parser.add_argument(
        "--systems",
        nargs="+",
        help=(
            "System IDs to compare. Defaults to baseline_system and "
            "headline_system from the report contract."
        ),
    )
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--image-policy",
        choices=("prefer-local", "local-only", "remote", "none"),
        default="local-only",
        help=(
            "prefer-local uses repo crop files first and public image URLs only "
            "as fallback; local-only makes no remote image requests."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    html_output = args.output_dir / "task.html"
    key_output = args.output_dir / "task_key.json"
    csv_output = args.output_dir / "labels_template.csv"
    try:
        report = load_json(args.report)
        metas = load_json(args.metas)
        if not isinstance(report, dict):
            raise ValueError("report must contain a JSON object")
        if not isinstance(metas, list):
            raise ValueError("metas must contain a JSON array")
        task, key = build_blind_task(
            report,
            metas,
            report_path=args.report.resolve(),
            metas_path=args.metas.resolve(),
            systems=args.systems,
            depth=args.depth,
            seed=args.seed,
            html_output=html_output.resolve(),
            project_root=ROOT,
            image_policy=args.image_policy,
        )
        write_task_artifacts(
            task,
            key,
            html_output=html_output,
            key_output=key_output,
            csv_output=csv_output,
            allow_remote_images=args.image_policy in {"prefer-local", "remote"},
        )
    except (OSError, ValueError, TypeError) as exc:
        print(f"blind task generation failed: {exc}", file=sys.stderr)
        return 1

    candidate_count = sum(len(query["candidates"]) for query in task["queries"])
    missing_images = sum(
        not candidate["image_src"]
        for query in task["queries"]
        for candidate in query["candidates"]
    )
    print(
        "blind relevance task generated: "
        f"task={task['task_id']} queries={len(task['queries'])} "
        f"query_candidates={candidate_count} missing_images={missing_images}"
    )
    print(f"visible task: {html_output}")
    print(f"unblinding key (keep from reviewers): {key_output}")
    print(f"CSV template: {csv_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
