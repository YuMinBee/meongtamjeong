"""Prepare, aggregate, and verify the external public-portal pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.blind_relevance_evaluation import load_json, sha256_file  # noqa: E402
from app.portal_candidate_selection_study import (  # noqa: E402
    BLIND_LABEL_CSV_COLUMNS,
    TRIAL_CSV_COLUMNS,
    aggregate_portal_trials,
    build_blind_label_rows,
    check_protocol_sources,
    freeze_protocol,
    load_blind_labels,
    load_trial_csvs,
    prepare_trial_sheets,
    render_result_markdown,
    validate_protocol,
    validate_template,
    write_csv_rows,
)


DEFAULT_TEMPLATE = (
    ROOT / "docs" / "evaluation" / "portal_candidate_selection.protocol.v1.json"
)
DEFAULT_SNAPSHOT = ROOT / "data" / "dog_metas.json"
DEFAULT_INDEX = ROOT / "data" / "dog_faiss.index"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "reports" / "portal_candidate_selection"


def _json_object(path: Path, label: str) -> dict[str, Any]:
    value = load_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _artifact(path: Path, index: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if index is not None:
        result["artifact_index"] = index
    return result


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a PII-free, pre-registered end-to-end candidate search pilot. "
            "This tool does not browse systems or fabricate human observations."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-template")
    validate.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    freeze.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    freeze.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    freeze.add_argument("--project-root", type=Path, default=ROOT)
    freeze.add_argument("--study-date", required=True)
    freeze.add_argument("--source-commit", required=True)
    freeze.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_DIR / "protocol.json"
    )

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--protocol", type=Path, required=True)
    prepare.add_argument("--participants", type=int, default=4)
    prepare.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    blind = subparsers.add_parser("blind")
    blind.add_argument("--protocol", type=Path, required=True)
    blind.add_argument("--trials", type=Path, nargs="+", required=True)
    blind.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_DIR / "blind_labels.csv"
    )

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--protocol", type=Path, required=True)
    aggregate.add_argument("--trials", type=Path, nargs="+", required=True)
    aggregate.add_argument("--labels", type=Path, required=True)
    aggregate.add_argument("--json-out", type=Path, required=True)
    aggregate.add_argument("--markdown-out", type=Path)

    check = subparsers.add_parser("check")
    check.add_argument("--protocol", type=Path, required=True)
    check.add_argument("--project-root", type=Path, default=ROOT)
    check.add_argument("--trials", type=Path, nargs="+")
    check.add_argument("--labels", type=Path)
    check.add_argument("--result", type=Path)
    check.add_argument("--markdown", type=Path)
    return parser


def _validate_template(args: argparse.Namespace) -> int:
    validate_template(_json_object(args.template, "template"))
    print("portal pilot template is valid; no human result is present or claimed")
    return 0


def _freeze(args: argparse.Namespace) -> int:
    template = _json_object(args.template, "template")
    protocol = freeze_protocol(
        template,
        template_path=args.template.resolve(),
        snapshot_path=args.snapshot.resolve(),
        index_path=args.index.resolve(),
        project_root=args.project_root.resolve(),
        study_date=args.study_date,
        source_commit=args.source_commit,
    )
    _write_json(args.output, protocol)
    print(f"frozen portal pilot protocol: {protocol['protocol_id']}")
    print(f"write-once study protocol: {args.output}")
    print("Freeze before the first participant; this command creates no result.")
    return 0


def _prepare(args: argparse.Namespace) -> int:
    protocol = _json_object(args.protocol, "protocol")
    sheets = prepare_trial_sheets(protocol, args.participants)
    for participant, rows in sheets.items():
        write_csv_rows(
            args.output_dir / f"{participant}.csv",
            TRIAL_CSV_COLUMNS,
            rows,
        )
    print(
        f"prepared {len(sheets)} assigned-code sheets in {args.output_dir}; "
        "do not replace P codes with names or email addresses"
    )
    return 0


def _validated_inputs(
    protocol_path: Path,
    trial_paths: Sequence[Path],
    labels_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    protocol = _json_object(protocol_path, "protocol")
    rows = load_trial_csvs(trial_paths, protocol)
    expected_labels = build_blind_label_rows(protocol, rows)
    labels = load_blind_labels(labels_path, expected_labels)
    return protocol, rows, labels


def _blind(args: argparse.Namespace) -> int:
    protocol = _json_object(args.protocol, "protocol")
    rows = load_trial_csvs(args.trials, protocol)
    label_rows = build_blind_label_rows(protocol, rows)
    write_csv_rows(args.output, BLIND_LABEL_CSV_COLUMNS, label_rows)
    serialized = json.dumps(label_rows, ensure_ascii=False)
    hidden_values = {
        "official_portal",
        "meongtamjeong",
        *(str(row["participant_code"]) for row in rows),
    }
    if any(value in serialized for value in hidden_values):
        raise ValueError(
            "blind sheet unexpectedly contains a source or participant code"
        )
    print(f"blind relevance sheet prepared: rows={len(label_rows)} path={args.output}")
    print("Give only this sheet to an independent labeler; no result exists yet.")
    return 0


def _expected_report(
    protocol_path: Path,
    trial_paths: Sequence[Path],
    labels_path: Path,
) -> dict[str, Any]:
    protocol, rows, labels = _validated_inputs(protocol_path, trial_paths, labels_path)
    return aggregate_portal_trials(
        protocol,
        rows,
        labels,
        protocol_sha256=sha256_file(protocol_path),
        trial_artifacts=[
            _artifact(path, index) for index, path in enumerate(trial_paths, start=1)
        ],
        label_artifact=_artifact(labels_path),
    )


def _aggregate(args: argparse.Namespace) -> int:
    report = _expected_report(args.protocol, args.trials, args.labels)
    _write_json(args.json_out, report)
    if args.markdown_out is not None:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(render_result_markdown(report), encoding="utf-8")
    print(
        "portal pilot aggregation complete: "
        f"participants={report['sample']['participants']} "
        f"scheduled_trials={report['sample']['scheduled_trials']}"
    )
    print(
        "Descriptive 3-5 person search pilot only; do not claim adoption or "
        "population impact."
    )
    return 0


def _check(args: argparse.Namespace) -> int:
    protocol = _json_object(args.protocol, "protocol")
    validate_protocol(protocol)
    source_failures = check_protocol_sources(protocol, args.project_root)
    if source_failures:
        raise ValueError("; ".join(source_failures))
    if args.result is not None and (not args.trials or args.labels is None):
        raise ValueError("--result requires --trials and --labels")
    if args.markdown is not None and args.result is None:
        raise ValueError("--markdown requires --result")
    if args.trials and args.labels is None:
        load_trial_csvs(args.trials, protocol)
    elif args.labels is not None:
        if not args.trials:
            raise ValueError("--labels requires --trials")
        expected = _expected_report(args.protocol, args.trials, args.labels)
        if args.result is not None:
            if load_json(args.result) != expected:
                raise ValueError("result JSON does not match validated inputs")
            if args.markdown is not None and args.markdown.read_text(
                encoding="utf-8"
            ) != render_result_markdown(expected):
                raise ValueError("result Markdown does not match validated inputs")
    print("portal pilot check passed")
    if not args.trials:
        print("No human trials or result were checked or claimed.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate-template":
            return _validate_template(args)
        if args.command == "freeze":
            return _freeze(args)
        if args.command == "prepare":
            return _prepare(args)
        if args.command == "blind":
            return _blind(args)
        if args.command == "aggregate":
            return _aggregate(args)
        if args.command == "check":
            return _check(args)
        raise ValueError(f"unsupported command: {args.command}")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"portal candidate selection study failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
