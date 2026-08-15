"""Fail-closed provenance checks for the distributed CLIP/FAISS snapshot.

These checks prove that the current artifacts match the repository's declared
provenance records and that the query encoder package came from the pinned
official CLIP commit. They deliberately do not claim that every retained
vector has a cryptographically verifiable encoder fingerprint.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any, Callable, Mapping


EXPECTED_CLIP_MODEL = "ViT-B/32"
EXPECTED_DECLARED_MODEL = "OpenAI CLIP ViT-B/32"
EXPECTED_CLIP_DIMENSION = 512
EXPECTED_CLIP_COMMIT = "dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1"
EXPECTED_CLIP_REPOSITORY = "https://github.com/openai/CLIP.git"


class ProvenanceError(ValueError):
    """Raised when artifact or encoder provenance cannot be verified."""


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(chunk_size):
                digest.update(chunk)
    except OSError as exc:
        raise ProvenanceError(
            f"provenance artifact cannot be read: {path.name}"
        ) from exc
    return digest.hexdigest()


def _load_object(path: Path, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProvenanceError(f"{label} cannot be read") from exc
    if not isinstance(payload, Mapping):
        raise ProvenanceError(f"{label} must contain an object")
    return payload


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProvenanceError(f"{label} is missing")
    return value


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ProvenanceError(f"{label} does not match the pinned snapshot")


def _artifact_sha(
    artifacts: Mapping[str, Any],
    key: str,
    *,
    label: str,
) -> str:
    record = _mapping(artifacts.get(key), label)
    value = str(record.get("sha256") or "").strip().lower()
    if len(value) != 64:
        raise ProvenanceError(f"{label} SHA-256 is missing")
    return value


def _lock_commit(lock_path: Path) -> str:
    try:
        lines = lock_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ProvenanceError("requirements lock cannot be read") from exc
    declarations = [
        line.strip() for line in lines if line.strip().lower().startswith("clip @")
    ]
    expected = f"clip @ git+{EXPECTED_CLIP_REPOSITORY}@{EXPECTED_CLIP_COMMIT}"
    if declarations != [expected]:
        raise ProvenanceError("requirements lock does not pin the official CLIP commit")
    return EXPECTED_CLIP_COMMIT


def verify_declared_index_provenance(
    *,
    manifest_path: Path,
    sync_report_path: Path,
    lock_path: Path,
    index_path: Path,
    metas_path: Path,
    requested_clip_model: str,
    metadata_row_count: int,
) -> dict[str, Any]:
    """Verify artifact-bound, repository-declared index provenance.

    This function is intentionally pure Python and does not import CLIP, Torch,
    FAISS, NumPy, or any network client.
    """

    _require_equal(
        requested_clip_model,
        EXPECTED_CLIP_MODEL,
        "requested CLIP model",
    )
    if metadata_row_count <= 0:
        raise ProvenanceError("metadata row count must be positive")

    manifest = _load_object(manifest_path, "snapshot manifest")
    sync_report = _load_object(sync_report_path, "active index sync report")
    embedding = _mapping(manifest.get("embedding"), "manifest embedding")
    _require_equal(
        embedding.get("model"),
        EXPECTED_DECLARED_MODEL,
        "manifest embedding model",
    )
    _require_equal(
        embedding.get("dimension"),
        EXPECTED_CLIP_DIMENSION,
        "manifest embedding dimension",
    )
    _require_equal(
        embedding.get("vectors"),
        metadata_row_count,
        "manifest vector count",
    )
    submission_runtime = _mapping(
        embedding.get("submission_runtime"),
        "manifest submission runtime",
    )
    _require_equal(
        submission_runtime.get("clip_source_commit"),
        EXPECTED_CLIP_COMMIT,
        "manifest CLIP source commit",
    )

    index_sha = _sha256_file(index_path)
    metas_sha = _sha256_file(metas_path)
    sync_sha = _sha256_file(sync_report_path)
    manifest_artifacts = _mapping(
        manifest.get("artifacts"),
        "manifest artifacts",
    )
    _require_equal(
        _artifact_sha(
            manifest_artifacts,
            "data/dog_faiss.index",
            label="manifest index artifact",
        ),
        index_sha,
        "manifest index artifact SHA-256",
    )
    _require_equal(
        _artifact_sha(
            manifest_artifacts,
            "data/dog_metas.json",
            label="manifest metadata artifact",
        ),
        metas_sha,
        "manifest metadata artifact SHA-256",
    )
    _require_equal(
        _artifact_sha(
            manifest_artifacts,
            "data/active_index_sync_report.json",
            label="manifest sync report artifact",
        ),
        sync_sha,
        "manifest sync report artifact SHA-256",
    )

    sync_artifacts = _mapping(sync_report.get("artifacts"), "sync artifacts")
    sync_outputs = _mapping(sync_artifacts.get("outputs"), "sync output artifacts")
    _require_equal(
        _artifact_sha(sync_outputs, "index", label="sync output index"),
        index_sha,
        "sync output index SHA-256",
    )
    _require_equal(
        _artifact_sha(sync_outputs, "metas", label="sync output metadata"),
        metas_sha,
        "sync output metadata SHA-256",
    )

    promotion = _mapping(
        sync_report.get("promotion_validation"),
        "sync promotion validation",
    )
    stats = _mapping(sync_report.get("stats"), "sync stats")
    for source, prefix in ((promotion, "sync promotion"), (stats, "sync stats")):
        _require_equal(
            source.get("dimension"),
            EXPECTED_CLIP_DIMENSION,
            f"{prefix} dimension",
        )
    _require_equal(
        promotion.get("vector_count"),
        metadata_row_count,
        "sync promotion vector count",
    )
    _require_equal(
        stats.get("output_vectors"),
        metadata_row_count,
        "sync output vector count",
    )

    runtime = _mapping(
        sync_report.get("runtime_provenance"),
        "sync runtime provenance",
    )
    incremental = _mapping(
        runtime.get("incremental_embedding"),
        "sync incremental embedding provenance",
    )
    _require_equal(
        incremental.get("model"),
        EXPECTED_DECLARED_MODEL,
        "sync incremental embedding model",
    )
    sync_submission = _mapping(
        runtime.get("submission_runtime"),
        "sync submission runtime",
    )
    _require_equal(
        sync_submission.get("clip_source_commit"),
        EXPECTED_CLIP_COMMIT,
        "sync CLIP source commit",
    )
    lock_commit = _lock_commit(lock_path)

    return {
        "declared_index_provenance_consistent": True,
        "declared_model": EXPECTED_DECLARED_MODEL,
        "requested_model": EXPECTED_CLIP_MODEL,
        "declared_dimension": EXPECTED_CLIP_DIMENSION,
        "declared_vector_count": metadata_row_count,
        "declared_clip_source_commit": EXPECTED_CLIP_COMMIT,
        "requirements_lock_commit": lock_commit,
        "index_sha256": index_sha,
        "metas_sha256": metas_sha,
        "sync_report_sha256": sync_sha,
        "retained_vector_count": int(stats.get("retained_vectors") or 0),
    }


def verify_index_shape(index: Any, metadata_row_count: int) -> dict[str, int]:
    """Validate the loaded FAISS shape before any downloader is constructed."""

    try:
        dimension = int(index.d)
        vector_count = int(index.ntotal)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ProvenanceError("FAISS index shape cannot be read") from exc
    _require_equal(
        dimension,
        EXPECTED_CLIP_DIMENSION,
        "FAISS index dimension",
    )
    _require_equal(
        vector_count,
        metadata_row_count,
        "FAISS index vector count",
    )
    return {"index_dimension": dimension, "index_vector_count": vector_count}


def verify_installed_clip_commit(
    clip_module: Any,
    *,
    distribution_getter: Callable[[str], Any] = importlib.metadata.distribution,
) -> dict[str, Any]:
    """Verify the imported query encoder belongs to the pinned VCS install."""

    try:
        distribution = distribution_getter("clip")
        direct_url_text = distribution.read_text("direct_url.json")
        direct_url = json.loads(direct_url_text or "")
    except (
        importlib.metadata.PackageNotFoundError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ) as exc:
        raise ProvenanceError(
            "installed CLIP direct_url provenance is unavailable"
        ) from exc
    if not isinstance(direct_url, Mapping):
        raise ProvenanceError("installed CLIP direct_url provenance is invalid")
    _require_equal(
        direct_url.get("url"),
        EXPECTED_CLIP_REPOSITORY,
        "installed CLIP repository",
    )
    vcs_info = _mapping(direct_url.get("vcs_info"), "installed CLIP VCS provenance")
    _require_equal(vcs_info.get("vcs"), "git", "installed CLIP VCS")
    _require_equal(
        vcs_info.get("commit_id"),
        EXPECTED_CLIP_COMMIT,
        "installed CLIP commit",
    )

    module_file = getattr(clip_module, "__file__", None)
    if not module_file:
        raise ProvenanceError("imported CLIP module path is unavailable")
    try:
        module_path = Path(module_file).resolve()
        distribution_root = Path(distribution.locate_file("")).resolve()
        module_path.relative_to(distribution_root)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ProvenanceError(
            "imported CLIP module is outside the pinned distribution"
        ) from exc

    return {
        "query_encoder_commit_verified": True,
        "query_encoder_repository": EXPECTED_CLIP_REPOSITORY,
        "query_encoder_commit": EXPECTED_CLIP_COMMIT,
    }
