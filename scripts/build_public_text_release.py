"""Build a deterministic public-text-only-v1 staging directory."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.public_text_release import (  # noqa: E402
    PUBLIC_TEXT_RELEASE_PROFILE,
    PublicTextReleaseArtifacts,
    PublicTextReleaseError,
    derive_public_text_release,
    validate_public_text_release_payload,
    validate_public_text_snapshot_manifest,
)


DEFAULT_INDEX = ROOT / "data" / "dog_faiss.index"
DEFAULT_METAS = ROOT / "data" / "dog_metas.json"
DEFAULT_MANIFEST = ROOT / "data" / "snapshot_manifest.json"
DEFAULT_OUTPUT_DIR = ROOT / "dist" / "public-text-only"
OUTPUT_NAMES = {
    "index": "dog_faiss.index",
    "metas": "dog_metas.json",
    "manifest": "snapshot_manifest.json",
    "profile": "release_profile.json",
    "report": "public_text_release_report.json",
}


def _read_required(path: Path, *, label: str) -> bytes:
    try:
        value = path.read_bytes()
    except OSError as exc:
        raise PublicTextReleaseError(f"{label} cannot be read: {path}") from exc
    if not value:
        raise PublicTextReleaseError(f"{label} is empty: {path}")
    return value


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


def _output_paths(output_dir: Path) -> dict[str, Path]:
    return {key: output_dir / name for key, name in OUTPUT_NAMES.items()}


def _validate_distinct_paths(
    inputs: Sequence[Path],
    outputs: Sequence[Path],
) -> None:
    input_paths = {path.resolve(strict=False) for path in inputs}
    for output in outputs:
        if output.resolve(strict=False) in input_paths:
            raise PublicTextReleaseError(
                "staging output must not overwrite a source artifact"
            )


def build_public_text_release(
    *,
    index_path: Path,
    metas_path: Path,
    manifest_path: Path,
    output_dir: Path,
    check_only: bool = False,
    faiss_module: Any | None = None,
) -> tuple[PublicTextReleaseArtifacts, dict[str, Path]]:
    paths = _output_paths(output_dir)
    _validate_distinct_paths(
        (index_path, metas_path, manifest_path),
        tuple(paths.values()),
    )
    artifacts = derive_public_text_release(
        _read_required(index_path, label="source FAISS index"),
        _read_required(metas_path, label="source metadata"),
        source_manifest_bytes=_read_required(
            manifest_path,
            label="source snapshot manifest",
        ),
        faiss_module=faiss_module,
    )
    validate_public_text_release_payload(
        artifacts.index_bytes,
        artifacts.metas_bytes,
        artifacts.release_profile_bytes,
        faiss_module=faiss_module,
    )
    validate_public_text_snapshot_manifest(artifacts.snapshot_manifest_bytes)
    if check_only:
        return artifacts, paths

    payloads = {
        "index": artifacts.index_bytes,
        "metas": artifacts.metas_bytes,
        "manifest": artifacts.snapshot_manifest_bytes,
        "profile": artifacts.release_profile_bytes,
        "report": artifacts.derivation_report_bytes,
    }
    for key in ("index", "metas", "manifest", "profile", "report"):
        _write_atomic(paths[key], payloads[key])
    return artifacts, paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--metas", type=Path, default=DEFAULT_METAS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="derive and validate in memory without writing staging artifacts",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    faiss_module: Any | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    try:
        artifacts, paths = build_public_text_release(
            index_path=args.index,
            metas_path=args.metas,
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            check_only=bool(args.check_only),
            faiss_module=faiss_module,
        )
    except PublicTextReleaseError as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "profile": PUBLIC_TEXT_RELEASE_PROFILE,
                    "error": str(exc),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1

    print(
        json.dumps(
            {
                "ok": True,
                "checked_only": bool(args.check_only),
                **artifacts.safe_summary(),
                "outputs": (
                    {}
                    if args.check_only
                    else {key: str(path) for key, path in sorted(paths.items())}
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
