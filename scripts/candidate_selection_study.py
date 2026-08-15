"""Generate, aggregate, and verify the local candidate-selection pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.candidate_selection_study import (  # noqa: E402
    DEFAULT_QUERY_IDS,
    DEFAULT_TIMEOUT_SECONDS,
    aggregate_trials,
    build_study,
    check_study_integrity,
    load_trial_csvs,
    render_result_markdown,
    sha256_file,
    write_study_artifacts,
)
from app.blind_relevance_evaluation import load_json  # noqa: E402


DEFAULT_REPORT = ROOT / "docs" / "evaluation" / "retrieval_eval.appearance_v1.json"
DEFAULT_METAS = ROOT / "data" / "dog_metas.json"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "reports" / "candidate_selection"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a local-only, blinded pilot measuring the time to select "
            "three shelter-visit candidates."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    generate.add_argument("--metas", type=Path, default=DEFAULT_METAS)
    generate.add_argument("--project-root", type=Path, default=ROOT)
    generate.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    generate.add_argument(
        "--systems",
        nargs=2,
        metavar=("BASELINE", "MEONGTAMJEONG"),
        help=(
            "Two retrieval report system IDs. Defaults to the report's "
            "baseline_system and headline_system."
        ),
    )
    generate.add_argument(
        "--query-ids",
        nargs=4,
        default=list(DEFAULT_QUERY_IDS),
        metavar=("Q1", "Q2", "Q3", "Q4"),
    )
    generate.add_argument("--depth", type=int, default=10)
    generate.add_argument("--seed", type=int, default=20260726)
    generate.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
    )

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--key", type=Path, required=True)
    aggregate.add_argument("--labels", type=Path, nargs="+", required=True)
    aggregate.add_argument("--project-root", type=Path, default=ROOT)
    aggregate.add_argument("--task-html", type=Path)
    aggregate.add_argument("--json-out", type=Path, required=True)
    aggregate.add_argument("--markdown-out", type=Path)

    check = subparsers.add_parser("check")
    check.add_argument("--key", type=Path, required=True)
    check.add_argument("--project-root", type=Path, default=ROOT)
    check.add_argument("--task-html", type=Path)
    check.add_argument("--labels", type=Path, nargs="+")
    check.add_argument("--result", type=Path)
    check.add_argument("--markdown", type=Path)
    return parser


def _load_key(path: Path) -> dict[str, object]:
    key = load_json(path)
    if not isinstance(key, dict):
        raise ValueError("key must contain a JSON object")
    return key


def _task_html_path(key_path: Path, explicit: Path | None) -> Path:
    return explicit if explicit is not None else key_path.with_name("task.html")


def _label_artifacts(paths: Sequence[Path]) -> list[dict[str, object]]:
    return [
        {
            "artifact_index": index,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for index, path in enumerate(paths, start=1)
    ]


def _generate(args: argparse.Namespace) -> int:
    report = load_json(args.report)
    metas = load_json(args.metas)
    if not isinstance(report, dict):
        raise ValueError("report must contain a JSON object")
    if not isinstance(metas, list):
        raise ValueError("metas must contain a JSON array")
    task, key = build_study(
        report,
        metas,
        report_path=args.report.resolve(),
        metas_path=args.metas.resolve(),
        project_root=args.project_root.resolve(),
        systems=args.systems,
        query_ids=args.query_ids,
        depth=args.depth,
        seed=args.seed,
        timeout_seconds=args.timeout_seconds,
    )
    html_output = args.output_dir / "task.html"
    key_output = args.output_dir / "task_key.json"
    write_study_artifacts(
        task,
        key,
        html_output=html_output,
        key_output=key_output,
    )
    print(
        "candidate selection study generated: "
        f"study={task['study_id']} tasks={len(task['tasks'])} "
        f"candidates={len(task['candidates'])}"
    )
    print(f"visible local task: {html_output}")
    print(f"unblinding key (do not give to participants): {key_output}")
    print(
        "No human result exists yet; this command does not produce a performance claim."
    )
    return 0


def _aggregate(args: argparse.Namespace) -> int:
    key = _load_key(args.key)
    task_html = _task_html_path(args.key, args.task_html)
    failures = check_study_integrity(
        key,
        project_root=args.project_root,
        task_html=task_html,
    )
    if failures:
        raise ValueError("; ".join(failures))
    rows = load_trial_csvs(args.labels, key)
    report = aggregate_trials(
        key,
        rows,
        source_key_sha256=sha256_file(args.key),
        label_artifacts=_label_artifacts(args.labels),
    )
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
    print(
        "candidate selection aggregation complete: "
        f"participants={report['sample']['participants']} "
        f"trials={report['sample']['trials']}"
    )
    print(
        "Descriptive small-sample pilot only; do not present it as "
        "population, adoption, temperament, or safety performance."
    )
    return 0


def _check(args: argparse.Namespace) -> int:
    key = _load_key(args.key)
    task_html = _task_html_path(args.key, args.task_html)
    failures = check_study_integrity(
        key,
        project_root=args.project_root,
        task_html=task_html,
    )
    if args.result is not None and not args.labels:
        raise ValueError("--result requires --labels")
    if args.markdown is not None and args.result is None:
        raise ValueError("--markdown requires --result")

    if args.labels:
        rows = load_trial_csvs(args.labels, key)
        if args.result is not None:
            expected = aggregate_trials(
                key,
                rows,
                source_key_sha256=sha256_file(args.key),
                label_artifacts=_label_artifacts(args.labels),
            )
            actual = load_json(args.result)
            if actual != expected:
                failures.append("result JSON does not match validated inputs")
            if args.markdown is not None:
                expected_markdown = render_result_markdown(expected)
                try:
                    actual_markdown = args.markdown.read_text(encoding="utf-8")
                except OSError as exc:
                    failures.append(f"result Markdown cannot be read: {exc}")
                else:
                    if actual_markdown != expected_markdown:
                        failures.append(
                            "result Markdown does not match validated result"
                        )
    if failures:
        print("candidate selection check failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    if args.labels:
        participant_count = len(
            {row["participant_code"] for row in load_trial_csvs(args.labels, key)}
        )
        print(
            "candidate selection check passed: "
            f"validated_participants={participant_count}"
        )
    else:
        print(
            "candidate selection preflight passed: sources, schedule, "
            "blinding artifact, and hashes are internally consistent"
        )
        print("No human result was checked or claimed.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "generate":
            return _generate(args)
        if args.command == "aggregate":
            return _aggregate(args)
        if args.command == "check":
            return _check(args)
        raise ValueError(f"unsupported command: {args.command}")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"candidate selection study failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
