from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.safety_contract_evaluation import (  # noqa: E402
    evaluate_safety_contract,
    render_markdown,
    serialize_report,
    write_report,
)


DEFAULT_CONFIG = ROOT / "data" / "eval_safety_contract.appearance_v1.json"
DEFAULT_METAS = ROOT / "data" / "dog_metas.json"
DEFAULT_JSON_OUT = ROOT / "docs" / "evaluation" / "safety_contract.appearance_v1.json"
DEFAULT_MARKDOWN_OUT = ROOT / "docs" / "evaluation" / "safety_contract.appearance_v1.md"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the deterministic appearance-only non-inference safety "
            "contract. This is not an adoption-suitability benchmark."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--metas", type=Path, default=DEFAULT_METAS)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN_OUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail when report files do not match a fresh evaluation.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = evaluate_safety_contract(
            args.config.resolve(),
            args.metas.resolve(),
        )
    except (OSError, ValueError, TypeError) as exc:
        print(f"safety contract evaluation failed: {exc}", file=sys.stderr)
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

    summary = report["summary"]
    print(
        "safety contract evaluation: "
        f"passed={summary['passed_contracts']}/{summary['total_contracts']} "
        f"failed={summary['failed_contracts']} "
        f"status={'PASS' if summary['passed'] else 'FAIL'}"
    )
    if stale:
        print("stale safety contract outputs: " + ", ".join(stale))
    return 0 if summary["passed"] and not stale else 1


if __name__ == "__main__":
    raise SystemExit(main())
