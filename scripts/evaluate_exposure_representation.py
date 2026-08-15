"""Generate descriptive exposure, representation, and evidence diagnostics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.exposure_evaluation import (  # noqa: E402
    DEFAULT_SYSTEM_IDS,
    evaluate_exposure_representation,
    render_markdown,
    serialize_report,
    write_report,
)


DEFAULT_METAS = ROOT / "data" / "dog_metas.json"
DEFAULT_RETRIEVAL_REPORT = (
    ROOT / "docs" / "evaluation" / "retrieval_eval.appearance_v1.json"
)
DEFAULT_QUERIES = ROOT / "data" / "eval_queries.appearance_v1.json"
DEFAULT_JSON_OUT = (
    ROOT / "docs" / "evaluation" / "exposure_representation.appearance_v1.json"
)
DEFAULT_MARKDOWN_OUT = (
    ROOT / "docs" / "evaluation" / "exposure_representation.appearance_v1.md"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize corpus representation, fixed Top-K exposure, and public-field "
            "evidence coverage. This is descriptive, not a normative fairness test."
        )
    )
    parser.add_argument("--metas", type=Path, default=DEFAULT_METAS)
    parser.add_argument(
        "--retrieval-report",
        type=Path,
        default=DEFAULT_RETRIEVAL_REPORT,
    )
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--system",
        action="append",
        dest="systems",
        help=(
            "Retrieval report system ID to include. Repeat for multiple systems; "
            "defaults to clip_active and hybrid_natural_graph."
        ),
    )
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN_OUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail when tracked outputs differ from a fresh deterministic evaluation.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    systems = tuple(args.systems) if args.systems else DEFAULT_SYSTEM_IDS
    try:
        report = evaluate_exposure_representation(
            args.metas.resolve(),
            args.retrieval_report.resolve(),
            args.queries.resolve(),
            top_k=args.top_k,
            system_ids=systems,
        )
    except (OSError, ValueError, TypeError) as exc:
        print(f"exposure evaluation failed: {exc}", file=sys.stderr)
        return 1

    expected_json = serialize_report(report)
    expected_markdown = render_markdown(report)
    stale: list[str] = []
    if args.check:
        if (
            not args.json_out.exists()
            or args.json_out.read_text(encoding="utf-8") != expected_json
        ):
            stale.append(str(args.json_out))
        if (
            not args.markdown_out.exists()
            or args.markdown_out.read_text(encoding="utf-8") != expected_markdown
        ):
            stale.append(str(args.markdown_out))
    else:
        write_report(report, args.json_out, args.markdown_out)

    print(
        "exposure evaluation: "
        f"active={report['corpus']['active_unique_notices']} "
        f"queries={report['method']['query_count']} "
        f"top_k={report['method']['top_k']} "
        f"systems={','.join(report['method']['systems'])} "
        f"status={'PASS' if report['validation']['passed'] else 'FAIL'}"
    )
    if stale:
        print("stale exposure outputs: " + ", ".join(stale))
    return 0 if report["validation"]["passed"] and not stale else 1


if __name__ == "__main__":
    raise SystemExit(main())
