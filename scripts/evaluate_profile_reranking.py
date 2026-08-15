from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.profile_evaluation import (  # noqa: E402
    evaluate_profile_reranking,
    render_markdown,
    serialize_report,
    write_report,
)


DEFAULT_CONFIG = ROOT / "data" / "eval_profiles.appearance_v1.json"
DEFAULT_METAS = ROOT / "data" / "dog_metas.json"
DEFAULT_JSON_OUT = ROOT / "docs" / "evaluation" / "profile_rerank.appearance_v1.json"
DEFAULT_MARKDOWN_OUT = ROOT / "docs" / "evaluation" / "profile_rerank.appearance_v1.md"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the deterministic appearance-only reranking contract on "
            "fixed notice IDs; this is not a performance benchmark."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--metas", type=Path, default=DEFAULT_METAS)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN_OUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail when tracked report files do not match a fresh evaluation.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = evaluate_profile_reranking(
        args.config.resolve(),
        args.metas.resolve(),
    )
    expected_json = serialize_report(report)
    expected_markdown = render_markdown(report)
    if args.check:
        stale: list[str] = []
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
        if stale:
            print("stale profile evaluation outputs: " + ", ".join(stale))
            return 1
    else:
        write_report(report, args.json_out, args.markdown_out)

    summary = report["summary"]
    print(
        "profile rerank evaluation: "
        f"profiles={summary['profile_count']} "
        f"mean_P@5_delta={summary['mean_precision_at_5_delta']:.3f} "
        f"mean_nDCG@5_delta={summary['mean_ndcg_at_5_delta']:.3f} "
        f"neutrality_failures={summary['unknown_neutrality_failures']} "
        f"certainty_hits={summary['adoption_suitability_certainty_phrase_hits']} "
        f"status={'PASS' if summary['passed'] else 'FAIL'}"
    )
    if not summary["passed"]:
        for failure in summary["failure_details"]:
            print(f"profile rerank contract failed: {failure}", file=sys.stderr)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
