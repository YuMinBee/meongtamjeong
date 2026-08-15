"""Plan, execute, or verify held-out secondary-photo retrieval evaluation.

``--plan`` and ``--check`` never load CLIP/FAISS or access the network.
Only the explicit ``--run-network`` mode downloads public notice images.
Downloaded bytes and decoded images remain in memory and are never written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import tempfile
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence, TypeVar


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.heldout_image_evaluation import (  # noqa: E402
    DEFAULT_MAX_SAMPLES,
    DEFAULT_REFERENCE_DATE,
    DEFAULT_SEED,
    REPORT_SCHEMA_VERSION,
    HeldoutCandidate,
    HttpxSessionAdapter,
    ImageDownloadError,
    SafePublicImageDownloader,
    aggregate_rank_metrics,
    build_candidate_plan,
    build_plan_report,
    candidate_vector_indices,
    clean_text,
    indexed_visual_source_urls,
    parse_reference_date,
    rank_notice_scores,
    rank_record,
    report_failures,
    report_to_markdown,
)
from app.index_provenance import (  # noqa: E402
    ProvenanceError,
    verify_declared_index_provenance,
    verify_index_shape,
    verify_installed_clip_commit,
)


DATA_DIR = BASE_DIR / "data"
DEFAULT_INDEX = DATA_DIR / "dog_faiss.index"
DEFAULT_METAS = DATA_DIR / "dog_metas.json"
DEFAULT_MANIFEST = DATA_DIR / "snapshot_manifest.json"
DEFAULT_SYNC_REPORT = DATA_DIR / "active_index_sync_report.json"
DEFAULT_REQUIREMENTS_LOCK = BASE_DIR / "requirements.lock.txt"
DEFAULT_JSON_OUT = (
    BASE_DIR / "docs" / "evaluation" / "heldout_image_retrieval.appearance_v1.json"
)
DEFAULT_MARKDOWN_OUT = (
    BASE_DIR / "docs" / "evaluation" / "heldout_image_retrieval.appearance_v1.md"
)
DEFAULT_ALLOWED_HOSTS = ("openapi.animal.go.kr",)
MAX_DOWNLOAD_WORKERS = 4


_DownloadToken = TypeVar("_DownloadToken")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def report_artifact(path: Path, **extra: Any) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        display_path = resolved.relative_to(BASE_DIR).as_posix()
    except ValueError:
        display_path = str(resolved)
    return {
        "path": display_path,
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
        **extra,
    }


def load_metas(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    if not isinstance(payload, list) or not all(
        isinstance(item, Mapping) for item in payload
    ):
        raise ValueError("metas must contain an array of objects")
    return [dict(item) for item in payload]


def build_plan(args: argparse.Namespace) -> tuple[Any, list[dict[str, Any]]]:
    metas = load_metas(args.metas)
    reference_date = parse_reference_date(args.reference_date)
    plan = build_candidate_plan(
        metas,
        reference_date=reference_date,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    return plan, metas


def base_report(
    args: argparse.Namespace,
    plan: Any,
    metas: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    manifest_path = Path(getattr(args, "manifest", DEFAULT_MANIFEST))
    sync_report_path = Path(getattr(args, "sync_report", DEFAULT_SYNC_REPORT))
    requirements_lock_path = Path(
        getattr(args, "requirements_lock", DEFAULT_REQUIREMENTS_LOCK)
    )
    return build_plan_report(
        plan,
        reference_date=parse_reference_date(args.reference_date).strftime("%Y-%m-%d"),
        seed=args.seed,
        max_samples=args.max_samples,
        artifacts={
            "index": report_artifact(args.index),
            "metas": report_artifact(args.metas, rows=len(metas)),
            "snapshot_manifest": report_artifact(manifest_path),
            "active_index_sync_report": report_artifact(sync_report_path),
            "requirements_lock": report_artifact(requirements_lock_path),
        },
        clip_model=args.clip_model,
    )


def verify_declared_provenance(
    args: argparse.Namespace,
    metadata_row_count: int,
) -> dict[str, Any]:
    """Bind the requested run to the repository's declared snapshot artifacts."""

    return verify_declared_index_provenance(
        manifest_path=Path(getattr(args, "manifest", DEFAULT_MANIFEST)),
        sync_report_path=Path(getattr(args, "sync_report", DEFAULT_SYNC_REPORT)),
        lock_path=Path(getattr(args, "requirements_lock", DEFAULT_REQUIREMENTS_LOCK)),
        index_path=args.index,
        metas_path=args.metas,
        requested_clip_model=args.clip_model,
        metadata_row_count=metadata_row_count,
    )


def load_heavy_runtime() -> dict[str, Any]:
    """Import packages omitted from lightweight CI only for a measured run."""

    import clip
    import faiss
    import numpy as np
    import torch

    return {
        "clip": clip,
        "faiss": faiss,
        "numpy": np,
        "torch": torch,
    }


def read_faiss_index(faiss_module: Any, path: Path) -> Any:
    """Read FAISS across Windows paths that can trigger an encoding error."""

    try:
        return faiss_module.read_index(str(path))
    except RuntimeError as exc:
        if "Illegal byte sequence" not in str(exc):
            raise
        safe_dir = Path(tempfile.gettempdir()) / "meong-heldout-faiss"
        safe_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
        safe_path = safe_dir / f"{path.stem}-{digest}{path.suffix}"
        if (
            not safe_path.exists()
            or safe_path.stat().st_size != path.stat().st_size
            or safe_path.stat().st_mtime < path.stat().st_mtime
        ):
            shutil.copy2(path, safe_path)
        return faiss_module.read_index(str(safe_path))


def resolve_device(torch_module: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch_module.cuda.is_available() else "cpu"


def extract_normalized_candidate_matrix(
    index: Any,
    vector_indices: Sequence[int],
    numpy_module: Any,
) -> Any:
    if not vector_indices:
        raise ValueError("visual candidate index is empty")
    vectors = []
    for vector_index in vector_indices:
        vector = numpy_module.asarray(
            index.reconstruct(int(vector_index)),
            dtype=numpy_module.float32,
        ).reshape(-1)
        if vector.size != int(index.d):
            raise ValueError("reconstructed vector dimension mismatch")
        if not numpy_module.isfinite(vector).all():
            raise ValueError("candidate vector contains non-finite values")
        norm = float(numpy_module.linalg.norm(vector))
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError("candidate vector has invalid norm")
        vectors.append(vector / norm)
    return numpy_module.ascontiguousarray(
        numpy_module.vstack(vectors),
        dtype=numpy_module.float32,
    )


def encode_image(
    runtime: Mapping[str, Any],
    model: Any,
    preprocess: Any,
    image: Any,
    device: str,
) -> Any:
    numpy_module = runtime["numpy"]
    torch_module = runtime["torch"]
    tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
    with torch_module.no_grad():
        features = model.encode_image(tensor)
        norm = features.norm(dim=-1, keepdim=True)
        if bool((norm <= 0).any()):
            raise ValueError("query embedding has zero norm")
        features = features / norm
    array = features.cpu().numpy().astype("float32").reshape(-1)
    if not numpy_module.isfinite(array).all():
        raise ValueError("query embedding contains non-finite values")
    return array


class _ThreadLocalDownloaderPool:
    """Own one downloader/session per executor thread and close every instance."""

    def __init__(self, factory: Callable[[], Any]) -> None:
        self._factory = factory
        self._local = threading.local()
        self._created: list[Any] = []
        self._lock = threading.Lock()

    def fetch(self, url: str) -> tuple[Any | None, str | None]:
        downloader = getattr(self._local, "downloader", None)
        if downloader is None:
            downloader = self._factory()
            self._local.downloader = downloader
            with self._lock:
                self._created.append(downloader)
        try:
            return downloader.fetch(url), None
        except ImageDownloadError as exc:
            return None, exc.reason

    def close(self) -> None:
        for downloader in self._created:
            close = getattr(downloader, "close", None)
            if callable(close):
                close()


def _ordered_parallel_downloads(
    items: Sequence[tuple[_DownloadToken, str]],
    *,
    downloader_factory: Callable[[], Any],
    workers: int,
) -> Iterator[tuple[_DownloadToken, Any | None, str | None]]:
    """Download fixed-size batches while yielding outcomes in input order.

    At most ``workers`` decoded images can exist at once because the next batch
    is not submitted until every result in the current batch has been yielded.
    """

    parsed_workers = int(workers)
    if not 1 <= parsed_workers <= MAX_DOWNLOAD_WORKERS:
        raise ValueError(
            f"download workers must be between 1 and {MAX_DOWNLOAD_WORKERS}"
        )
    if not items:
        return
    pool = _ThreadLocalDownloaderPool(downloader_factory)
    try:
        with ThreadPoolExecutor(max_workers=parsed_workers) as executor:
            for offset in range(0, len(items), parsed_workers):
                batch = items[offset : offset + parsed_workers]
                futures = [executor.submit(pool.fetch, url) for _, url in batch]
                for (token, _url), future in zip(batch, futures, strict=True):
                    downloaded, reason = future.result()
                    yield token, downloaded, reason
    finally:
        pool.close()


def _close_downloaded_image(downloaded: Any) -> None:
    close = getattr(getattr(downloaded, "image", None), "close", None)
    if callable(close):
        close()


def _end_to_end_metrics(
    metrics: dict[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    attempted = int(metrics.get("attempted_count") or 0)
    for key in ("hit@1", "hit@5", "hit@10"):
        hits = sum(int(row.get(key) or 0) for row in rows)
        metrics[f"end_to_end_{key}"] = round(hits / attempted, 6) if attempted else 0.0


def evaluate_network(
    args: argparse.Namespace,
    plan: Any,
    metas: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run the explicitly authorized network/model evaluation."""

    declared_provenance = verify_declared_provenance(args, len(metas))
    runtime = load_heavy_runtime()
    query_encoder_provenance = verify_installed_clip_commit(runtime["clip"])
    index = read_faiss_index(runtime["faiss"], args.index)
    index_shape = verify_index_shape(index, len(metas))
    reference_date = parse_reference_date(args.reference_date)
    vector_indices, vector_counts = candidate_vector_indices(
        metas,
        reference_date=reference_date,
    )
    candidate_matrix = extract_normalized_candidate_matrix(
        index,
        vector_indices,
        runtime["numpy"],
    )
    device = resolve_device(runtime["torch"], args.device)
    model, preprocess = runtime["clip"].load(args.clip_model, device=device)
    model.eval()
    download_workers = int(getattr(args, "download_workers", MAX_DOWNLOAD_WORKERS))
    use_env_proxy = bool(getattr(args, "use_env_proxy", False))
    download_client = clean_text(getattr(args, "download_client", "requests"))
    if download_client not in {"requests", "httpx"}:
        raise ValueError("download client must be requests or httpx")

    def downloader_factory() -> SafePublicImageDownloader:
        session = HttpxSessionAdapter() if download_client == "httpx" else None
        return SafePublicImageDownloader(
            timeout=args.timeout,
            max_bytes=args.max_bytes,
            max_pixels=args.max_pixels,
            max_redirects=args.max_redirects,
            retries=args.retries,
            allowed_hosts=args.allowed_host,
            session=session,
            use_env_proxy=use_env_proxy,
        )

    all_indexed_source_urls, source_audit = indexed_visual_source_urls(
        metas,
        reference_date=reference_date,
    )
    audit_all_sources = bool(getattr(args, "audit_all_indexed_sources", False))
    indexed_source_urls = (
        all_indexed_source_urls
        if audit_all_sources
        else tuple(sorted({candidate.indexed_url for candidate in plan.sample}))
    )
    failure_reasons: Counter[str] = Counter()
    primary_sha_by_id: dict[str, str] = {}
    source_sha_by_url: dict[str, str] = {}
    indexed_sha_counts: Counter[str] = Counter()
    # Phase one establishes the exact payload SHA of every active visual
    # vector's public source, including sources outside the query sample.
    # Images are released after each iteration; only hashes remain.
    primary_requests = tuple(
        (source_url, source_url) for source_url in indexed_source_urls
    )
    for source_url, downloaded, reason in _ordered_parallel_downloads(
        primary_requests,
        downloader_factory=downloader_factory,
        workers=download_workers,
    ):
        if downloaded is None:
            failure_reasons[f"indexed_source_{reason or 'unknown'}"] += 1
            continue
        try:
            source_sha_by_url[source_url] = downloaded.payload_sha256
            indexed_sha_counts[downloaded.payload_sha256] += 1
        finally:
            _close_downloaded_image(downloaded)
    for candidate in plan.sample:
        primary_sha = source_sha_by_url.get(candidate.indexed_url)
        if primary_sha:
            primary_sha_by_id[candidate.notice_id] = primary_sha

    rows: list[dict[str, Any]] = []
    query_sha_to_id: dict[str, str] = {}
    secondary_downloaded_count = 0
    embedded_count = 0
    duplicate_count = 0
    secondary_requests: list[tuple[tuple[int, HeldoutCandidate, str], str]] = []
    for sample_order, candidate in enumerate(plan.sample, start=1):
        primary_sha = primary_sha_by_id.get(candidate.notice_id)
        if not primary_sha:
            failure_reasons["primary_unavailable"] += 1
            continue
        secondary_requests.append(
            ((sample_order, candidate, primary_sha), candidate.secondary_url)
        )
    for token, downloaded, reason in _ordered_parallel_downloads(
        secondary_requests,
        downloader_factory=downloader_factory,
        workers=download_workers,
    ):
        sample_order, candidate, primary_sha = token
        if downloaded is None:
            failure_reasons[f"secondary_{reason or 'unknown'}"] += 1
            continue
        try:
            secondary_downloaded_count += 1
            query_sha = downloaded.payload_sha256
            if query_sha in indexed_sha_counts:
                duplicate_count += 1
                failure_reasons["query_matches_indexed_primary_bytes"] += 1
                continue
            if query_sha in query_sha_to_id:
                duplicate_count += 1
                failure_reasons["duplicate_secondary_query_bytes"] += 1
                continue
            query_sha_to_id[query_sha] = candidate.notice_id
            try:
                query_vector = encode_image(
                    runtime,
                    model,
                    preprocess,
                    downloaded.image,
                    device,
                )
            except (MemoryError, OSError, RuntimeError, ValueError) as exc:
                failure_reasons[f"embedding_{type(exc).__name__}"] += 1
                continue
            embedded_count += 1
            scores = candidate_matrix @ query_vector
            ranking = rank_notice_scores(scores.tolist(), vector_indices, metas)
            record = rank_record(candidate.notice_id, ranking, top_k=10)
            if record["rank"] is None:
                failure_reasons["ground_truth_missing_from_candidates"] += 1
                continue
            record.update(
                {
                    "sample_order": sample_order,
                    "query_payload_sha256": query_sha,
                    "indexed_payload_sha256": primary_sha,
                    "query_bytes": downloaded.byte_count,
                    "query_format": downloaded.image_format,
                    "query_width": downloaded.width,
                    "query_height": downloaded.height,
                    "region": candidate.region,
                    "size": candidate.size,
                }
            )
            rows.append(record)
        finally:
            _close_downloaded_image(downloaded)

    metrics = aggregate_rank_metrics(
        rows,
        attempted_count=len(plan.sample),
        downloaded_count=secondary_downloaded_count,
        duplicate_count=duplicate_count,
    )
    metrics["primary_downloaded_count"] = len(primary_sha_by_id)
    metrics["payload_audit_source_count"] = len(indexed_source_urls)
    metrics["payload_audit_downloaded_count"] = len(source_sha_by_url)
    metrics["payload_audit_coverage"] = (
        round(len(source_sha_by_url) / len(indexed_source_urls), 6)
        if indexed_source_urls
        else 0.0
    )
    metrics["embedded_count"] = embedded_count
    metrics["primary_download_coverage"] = (
        round(len(primary_sha_by_id) / len(plan.sample), 6) if plan.sample else 0.0
    )
    _end_to_end_metrics(metrics, rows)
    source_coverage_complete = bool(indexed_source_urls) and (
        len(source_sha_by_url) == len(indexed_source_urls)
    )
    declared_coverage_complete = source_coverage_complete and (
        not audit_all_sources
        or int(source_audit.get("missing_source_url_rows") or 0) == 0
    )
    coverage_status = "complete" if declared_coverage_complete else "partial"
    payload_leakage_audit_complete = bool(rows) and (
        not audit_all_sources or declared_coverage_complete
    )
    execution_status = (
        "completed" if rows and payload_leakage_audit_complete else "incomplete"
    )

    report = base_report(args, plan, metas)
    report["runtime"] = {
        "clip_model": args.clip_model,
        "device": device,
        "network_executed": True,
        "declared_index_provenance_consistent": declared_provenance[
            "declared_index_provenance_consistent"
        ],
        "query_encoder_commit_verified": query_encoder_provenance[
            "query_encoder_commit_verified"
        ],
        "index_encoder_verified": False,
        "index_encoder_provenance": (
            "artifact-bound declaration only; no per-vector encoder fingerprint"
        ),
        "declared_clip_source_commit": declared_provenance[
            "declared_clip_source_commit"
        ],
        "query_encoder_commit": query_encoder_provenance["query_encoder_commit"],
        "retained_vector_count": declared_provenance["retained_vector_count"],
        **index_shape,
        "torch_version": clean_text(getattr(runtime["torch"], "__version__", "")),
        "faiss_metric_type": int(getattr(index, "metric_type", -1)),
        "similarity": "cosine over normalized reconstructed index vectors",
        "timeout_seconds": args.timeout,
        "max_image_bytes": args.max_bytes,
        "max_image_pixels": args.max_pixels,
        "max_redirects": args.max_redirects,
        "retries": args.retries,
        "allowed_hosts": sorted(args.allowed_host),
        "download_client": download_client,
        "download_workers": download_workers,
        "environment_proxy_enabled": use_env_proxy,
        "environment_proxy_values_persisted": False,
        "audit_all_indexed_sources": audit_all_sources,
        "transport_policy": (
            "upgrade initial HTTP to HTTPS; reject HTTPS-to-HTTP redirects"
        ),
    }
    report["candidate_index"] = {
        **vector_counts,
        **{f"source_{key}": value for key, value in source_audit.items()},
        "unique_notices": len(
            {
                clean_text(metas[index].get("desertionNo"))
                or clean_text(metas[index].get("desertion_no"))
                for index in vector_indices
            }
        ),
        "text_vectors_in_ranking": 0,
    }
    report["metrics"] = metrics
    report["failure_reasons"] = dict(sorted(failure_reasons.items()))
    report["queries"] = rows
    report["execution"] = {
        "status": execution_status,
        "coverage_status": coverage_status,
        "metrics_available": bool(rows),
        "network_requested": True,
        "images_persisted": False,
        "payload_leakage_audit_complete": payload_leakage_audit_complete,
        "payload_leakage_audit_scope": (
            "all_active_visual_sources"
            if audit_all_sources
            else "successfully_downloaded_sampled_primary_sources"
        ),
        "primary_sha_scope": (
            "every evaluable query has its own primary SHA; secondary payloads "
            "matching any successfully audited source in the declared "
            "payload_leakage_audit_scope are excluded"
        ),
    }
    invariant_failures = report_failures(
        report,
        expected_sample_sha256=plan.sample_sha256,
    )
    report["validation"] = {
        "passed": not invariant_failures,
        "failure_count": len(invariant_failures),
        "failures": invariant_failures,
    }
    return report


def write_report(
    report: Mapping[str, Any],
    json_out: Path,
    markdown_out: Path,
) -> None:
    markdown = report_to_markdown(report)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_out.write_text(markdown, encoding="utf-8")


def check_report(args: argparse.Namespace) -> list[str]:
    """Check hashes, deterministic sample, schema, and Markdown without network."""

    failures: list[str] = []
    if not args.json_out.exists():
        return [f"JSON report is missing: {args.json_out}"]
    if not args.markdown_out.exists():
        failures.append(f"Markdown report is missing: {args.markdown_out}")
    try:
        report = load_json(args.json_out)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return [f"JSON report cannot be read: {exc}"]
    if not isinstance(report, Mapping):
        return ["JSON report must contain an object"]
    if args.markdown_out.exists():
        expected_markdown = report_to_markdown(report)
        if args.markdown_out.read_text(encoding="utf-8") != expected_markdown:
            failures.append("Markdown report does not match JSON")

    plan, metas = build_plan(args)
    try:
        verify_declared_provenance(args, len(metas))
    except ProvenanceError as exc:
        failures.append(f"declared index provenance is invalid: {exc}")
    expected_plan_report = base_report(args, plan, metas)
    failures.extend(
        report_failures(
            report,
            expected_sample_sha256=plan.sample_sha256,
        )
    )
    execution = report.get("execution")
    execution_status = (
        clean_text(execution.get("status")) if isinstance(execution, Mapping) else ""
    )
    if execution_status == "plan_only" and not bool(
        getattr(args, "allow_plan_only_check", False)
    ):
        failures.append(
            "tracked report is plan-only; run --run-network or explicitly use "
            "--allow-plan-only-check for local plan development"
        )
    for field in (
        "scope",
        "sample_policy",
        "corpus",
        "exclusion_reasons",
        "distributions",
        "sample",
        "limitations",
    ):
        if report.get(field) != expected_plan_report.get(field):
            failures.append(f"plan field is stale or altered: {field}")
    if clean_text(report.get("reference_date")) != parse_reference_date(
        args.reference_date
    ).strftime("%Y-%m-%d"):
        failures.append("reference_date is stale")
    runtime = report.get("runtime")
    if not isinstance(runtime, Mapping):
        failures.append("runtime is missing")
    elif clean_text(runtime.get("clip_model")) != args.clip_model:
        failures.append("clip_model setting is stale")
    corpus = report.get("corpus")
    if not isinstance(corpus, Mapping):
        failures.append("corpus is missing")
    else:
        for key in (
            "metadata_rows",
            "active_notices",
            "visual_indexed_notices",
            "eligible_pairs",
            "selected_pairs",
        ):
            if corpus.get(key) != plan.corpus.get(key):
                failures.append(f"corpus field is stale: {key}")
    artifacts = report.get("artifacts")
    expected_artifacts = {
        "index": args.index,
        "metas": args.metas,
        "snapshot_manifest": Path(getattr(args, "manifest", DEFAULT_MANIFEST)),
        "active_index_sync_report": Path(
            getattr(args, "sync_report", DEFAULT_SYNC_REPORT)
        ),
        "requirements_lock": Path(
            getattr(args, "requirements_lock", DEFAULT_REQUIREMENTS_LOCK)
        ),
    }
    if not isinstance(artifacts, Mapping):
        failures.append("artifacts are missing")
    else:
        for name, path in expected_artifacts.items():
            record = artifacts.get(name)
            if not isinstance(record, Mapping):
                failures.append(f"artifact is missing: {name}")
            elif clean_text(record.get("sha256")) != sha256_file(path):
                failures.append(f"artifact hash is stale: {name}")
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        failures.append("report schema is stale")
    return sorted(set(failures))


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _download_worker_count(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= MAX_DOWNLOAD_WORKERS:
        raise argparse.ArgumentTypeError(
            f"value must be between 1 and {MAX_DOWNLOAD_WORKERS}"
        )
    return parsed


def _nonnegative_small_int(value: str) -> int:
    parsed = int(value)
    if not 0 <= parsed <= 3:
        raise argparse.ArgumentTypeError("value must be between 0 and 3")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive finite number")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate held-out secondary public-notice photos against only "
            "primary full-image and crop CLIP vectors."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--plan",
        action="store_true",
        help="write a deterministic sample plan without network/model access",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="alias for --plan",
    )
    mode.add_argument(
        "--check",
        action="store_true",
        help="verify the tracked report without network/model access",
    )
    mode.add_argument(
        "--run-network",
        action="store_true",
        help="explicitly download public images and run CLIP/FAISS evaluation",
    )
    parser.add_argument(
        "--allow-plan-only-check",
        action="store_true",
        help=(
            "allow --check to validate a non-measured plan locally; CI and "
            "release checks must omit this flag"
        ),
    )
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--metas", type=Path, default=DEFAULT_METAS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--sync-report",
        type=Path,
        default=DEFAULT_SYNC_REPORT,
    )
    parser.add_argument(
        "--requirements-lock",
        type=Path,
        default=DEFAULT_REQUIREMENTS_LOCK,
    )
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN_OUT)
    parser.add_argument("--reference-date", default=DEFAULT_REFERENCE_DATE)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument(
        "--max-samples",
        type=_positive_int,
        default=DEFAULT_MAX_SAMPLES,
    )
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--download-workers",
        type=_download_worker_count,
        default=MAX_DOWNLOAD_WORKERS,
        help="parallel image downloads; constrained to 1-4",
    )
    parser.add_argument(
        "--download-client",
        choices=("requests", "httpx"),
        default="requests",
        help="streaming client used only by explicit network execution",
    )
    parser.add_argument(
        "--use-env-proxy",
        action="store_true",
        help=(
            "opt in to HTTP(S) proxy environment variables; proxy values are "
            "never written to reports"
        ),
    )
    parser.add_argument("--timeout", type=_positive_float, default=12.0)
    parser.add_argument(
        "--max-bytes",
        type=_positive_int,
        default=12 * 1024 * 1024,
    )
    parser.add_argument(
        "--max-pixels",
        type=_positive_int,
        default=25_000_000,
    )
    parser.add_argument(
        "--max-redirects",
        type=_nonnegative_small_int,
        default=3,
    )
    parser.add_argument(
        "--retries",
        type=_nonnegative_small_int,
        choices=(0, 1),
        default=1,
    )
    parser.add_argument(
        "--allowed-host",
        action="append",
        default=None,
        help=(
            "exact public image host; repeat to allow more. "
            "Defaults to openapi.animal.go.kr"
        ),
    )
    parser.add_argument(
        "--audit-all-indexed-sources",
        action="store_true",
        help=(
            "download every active visual source for a corpus-wide payload SHA "
            "audit; off by default because it can require many slow requests"
        ),
    )
    args = parser.parse_args(argv)
    if args.allow_plan_only_check and not args.check:
        parser.error("--allow-plan-only-check requires --check")
    args.allowed_host = tuple(args.allowed_host or DEFAULT_ALLOWED_HOSTS)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.check:
        failures = check_report(args)
        if failures:
            for failure in failures:
                print(
                    f"held-out image evaluation check failed: {failure}",
                    file=sys.stderr,
                )
            return 1
        print("held-out image evaluation check: PASS")
        return 0

    plan, metas = build_plan(args)
    if args.run_network:
        report = evaluate_network(args, plan, metas)
    else:
        report = base_report(args, plan, metas)
        failures = report_failures(
            report,
            expected_sample_sha256=plan.sample_sha256,
        )
        report["validation"] = {
            "passed": not failures,
            "failure_count": len(failures),
            "failures": failures,
        }
    write_report(report, args.json_out, args.markdown_out)
    status = clean_text((report.get("execution") or {}).get("status"))
    print(
        "held-out image evaluation: "
        f"status={status} eligible={plan.corpus['eligible_pairs']} "
        f"selected={plan.corpus['selected_pairs']} "
        f"sample_sha256={plan.sample_sha256}"
    )
    return 0 if not report_failures(report) else 1


if __name__ == "__main__":
    raise SystemExit(main())
