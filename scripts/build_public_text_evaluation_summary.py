"""Build deterministic public-text evaluation evidence from gate reports."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.public_text_evaluation_summary import (  # noqa: E402
    SUMMARY_JSON_NAME,
    SUMMARY_MARKDOWN_NAME,
    PublicTextEvaluationSummaryError,
    build_public_text_evaluation_summary,
)


DEFAULT_REPORTS_DIR = ROOT / "dist" / "public-text-release-gate"
DEFAULT_INDEX = DEFAULT_REPORTS_DIR / "dog_faiss.index"
DEFAULT_METAS = DEFAULT_REPORTS_DIR / "dog_metas.json"
DEFAULT_PROFILE_MARKER = DEFAULT_REPORTS_DIR / "release_profile.json"


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary.write(data)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--metas", type=Path, default=DEFAULT_METAS)
    parser.add_argument(
        "--profile-marker",
        type=Path,
        default=DEFAULT_PROFILE_MARKER,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="output directory (default: reports directory)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate and render in memory without writing files",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        artifacts = build_public_text_evaluation_summary(
            reports_dir=args.reports_dir,
            index_path=args.index,
            metas_path=args.metas,
            profile_marker_path=args.profile_marker,
        )
        output_dir = args.output_dir or args.reports_dir
        if not args.check_only:
            _write_atomic(output_dir / SUMMARY_JSON_NAME, artifacts.json_bytes)
            _write_atomic(
                output_dir / SUMMARY_MARKDOWN_NAME,
                artifacts.markdown_bytes,
            )
    except PublicTextEvaluationSummaryError as exc:
        print(
            json.dumps(
                {"ok": False, "error": str(exc)},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1

    print(
        json.dumps(
            {
                "checked_only": bool(args.check_only),
                "ok": True,
                "profile": artifacts.payload["profile"],
                "reference_date": artifacts.payload["reference_date"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
