"""Run or verify the frozen one-shot appearance-query holdout v3."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.query_holdout_v3_evaluation import (  # noqa: E402
    REPORT_SCHEMA_VERSION,
    SYSTEM_CODE_ARTIFACTS,
    SYSTEM_ID,
    build_report,
    report_contract_failures,
    report_recalculation_failures,
    report_to_markdown,
    validate_holdout_payload,
)
from app.retrieval_evaluation import (  # noqa: E402
    artifact_record,
    build_silver_qrels,
    clean_text,
    merge_notice_metas,
    notice_id,
    parse_reference_date,
    validate_nonempty_qrels,
    validate_query_specs,
)
from scripts.evaluate_retrieval import (  # noqa: E402
    corpus_summary,
    encode_text,
    load_heavy_runtime,
    prepare_reference_docs,
    read_faiss_index,
    resolve_device,
)


DATA_DIR = BASE_DIR / "data"
DEFAULT_INDEX = DATA_DIR / "dog_faiss.index"
DEFAULT_METAS = DATA_DIR / "dog_metas.json"
DEFAULT_QUERIES = DATA_DIR / "eval_queries.appearance_v1.json"
DEFAULT_HOLDOUT = DATA_DIR / "eval_query_holdout.appearance_v3.json"
DEFAULT_HOLDOUT_SHA256 = DATA_DIR / "eval_query_holdout.appearance_v3.sha256"
DEFAULT_JSON_OUT = (
    BASE_DIR / "docs" / "evaluation" / "query_holdout.appearance_v3.json"
)
DEFAULT_MARKDOWN_OUT = (
    BASE_DIR / "docs" / "evaluation" / "query_holdout.appearance_v3.md"
)
DEFAULT_RUN_MARKER = (
    BASE_DIR / "docs" / "evaluation" / "query_holdout.appearance_v3.run.json"
)
MODULE_PATH = BASE_DIR / "app" / "query_holdout_v3_evaluation.py"
SCRIPT_PATH = Path(__file__).resolve()
APPEARANCE_QUERY_PATH = BASE_DIR / "app" / "appearance_query.py"
HYBRID_RAG_PATH = BASE_DIR / "app" / "hybrid_rag.py"
GRAPH_RAG_PATH = BASE_DIR / "app" / "graph_rag.py"
RETRIEVAL_EVALUATION_PATH = BASE_DIR / "app" / "retrieval_evaluation.py"
RETRIEVAL_CLI_PATH = BASE_DIR / "scripts" / "evaluate_retrieval.py"
EXPECTED_QUERY_EXECUTIONS = 12 + 24 + 6


def load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must contain a JSON object")
    return dict(payload)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def report_artifact(path: Path, **extra: Any) -> dict[str, Any]:
    record = artifact_record(path, **extra)
    try:
        record["path"] = path.resolve().relative_to(BASE_DIR).as_posix()
    except ValueError:
        record["path"] = str(path.resolve())
    return record


def frozen_holdout_sha256(holdout_path: Path, sidecar_path: Path) -> str:
    """Verify the preregistered digest before importing the evaluated system."""

    tokens = sidecar_path.read_text(encoding="utf-8").strip().split()
    if not tokens:
        raise ValueError("holdout SHA-256 sidecar is empty")
    expected = tokens[0].lower()
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ValueError("holdout SHA-256 sidecar has an invalid digest")
    actual = clean_text(artifact_record(holdout_path).get("sha256")).lower()
    if actual != expected:
        raise ValueError(
            f"frozen holdout SHA-256 mismatch: {actual!r} != {expected!r}"
        )
    if len(tokens) > 1 and Path(tokens[1]).name != holdout_path.name:
        raise ValueError("holdout SHA-256 sidecar filename is unexpected")
    return actual


def _load_inputs(
    args: argparse.Namespace,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    datetime,
    str,
]:
    frozen_sha = frozen_holdout_sha256(args.holdout, args.holdout_sha256)
    base_payload = load_json_object(args.queries, label="base query file")
    base_queries = base_payload.get("queries")
    if not isinstance(base_queries, list) or not all(
        isinstance(row, Mapping) for row in base_queries
    ):
        raise ValueError("base query file must contain a queries array")
    base_queries = [dict(row) for row in base_queries]
    validate_query_specs(base_queries)
    base_payload["queries"] = base_queries

    holdout_payload = load_json_object(args.holdout, label="holdout file")
    semantic_cases, negative_cases = validate_holdout_payload(
        holdout_payload, base_payload
    )
    reference_date = parse_reference_date(
        args.reference_date
        or holdout_payload.get("reference_date")
        or base_payload.get("reference_date")
    )
    return (
        base_payload,
        holdout_payload,
        base_queries,
        semantic_cases,
        negative_cases,
        reference_date,
        frozen_sha,
    )


def _query_policy_record(analysis: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "typo_corrections": list(analysis.get("typo_corrections") or []),
        "synonym_normalizations": list(
            analysis.get("synonym_normalizations") or []
        ),
        "unsupported_conditions": list(
            analysis.get("unsupported_conditions") or []
        ),
        "warnings": list(analysis.get("warnings") or []),
        "is_fully_supported": bool(analysis.get("is_fully_supported")),
    }


def appearance_retrieval(
    *,
    runtime: Mapping[str, Any],
    model: Any,
    device: str,
    index: Any,
    metas: Sequence[Mapping[str, Any]],
    docs: Sequence[Mapping[str, Any]],
    vector_to_doc: Mapping[int, int],
    bm25: Any,
    graph: Any,
    normalized_query: str,
    excluded_graph_feature_prefixes: Sequence[str],
    evaluation_depth: int,
    rerank_depth: int,
    vector_depth: int,
) -> tuple[list[str], dict[str, Any]]:
    """Mirror the current ``ranking_scope=appearance`` retrieval path."""

    structured = runtime["parse_structured_query"](normalized_query)
    search_query = (
        runtime["build_search_query_text"](normalized_query, structured)
        or normalized_query
    )
    query_vec = encode_text(runtime, model, device, search_query)
    distances, indices = index.search(
        query_vec,
        min(int(index.ntotal), vector_depth),
    )
    vector_scores, vector_details = runtime["vector_hits_to_doc_modality_scores"](
        distances[0].tolist(),
        indices[0].tolist(),
        vector_to_doc,
        metas,
    )
    graph_candidates = graph.candidate_doc_scores(
        structured,
        search_query,
        limit=0,
        excluded_feature_prefixes=excluded_graph_feature_prefixes,
    )
    candidate_limit = max(
        evaluation_depth,
        min(rerank_depth, evaluation_depth * 4),
    )
    ranked = runtime["rank_hybrid_documents"](
        docs=docs,
        bm25=bm25,
        structured=structured,
        query_text=search_query,
        vector_scores=vector_scores,
        vector_score_details=vector_details,
        topk=candidate_limit,
        rerank_depth=max(rerank_depth, candidate_limit),
        strict_filters=False,
        extra_candidate_indices=graph_candidates.keys(),
        extra_candidate_scores=graph_candidates,
        filter_inactive_notices=True,
        include_unknown_notices=False,
    )
    ranked = runtime["rerank_with_graph"](
        ranked=ranked,
        graph=graph,
        structured=structured,
        query_text=search_query,
        topk=evaluation_depth,
        excluded_feature_prefixes=excluded_graph_feature_prefixes,
    )
    ranking: list[str] = []
    for item in ranked:
        doc = item.get("doc") or {}
        dog_id = notice_id(doc.get("meta") or {}) or clean_text(doc.get("doc_id"))
        if dog_id and dog_id not in ranking:
            ranking.append(dog_id)
    return ranking, {
        "search_query": search_query,
        "structured_query": structured,
        "status_filter": "active_only",
        "include_unknown_notices": False,
        "graph_enabled": True,
        "candidate_documents": len(ranked),
    }


def _pre_run_code_artifacts() -> dict[str, dict[str, Any]]:
    """Hash evaluated code bytes without exposing source to the holdout author."""

    return {
        "appearance_query_module": report_artifact(APPEARANCE_QUERY_PATH),
        "hybrid_rag_module": report_artifact(HYBRID_RAG_PATH),
        "graph_rag_module": report_artifact(GRAPH_RAG_PATH),
        "retrieval_evaluation_module": report_artifact(
            RETRIEVAL_EVALUATION_PATH
        ),
        "retrieval_evaluation_cli": report_artifact(RETRIEVAL_CLI_PATH),
    }


def system_code_bundle_sha256(
    artifacts: Mapping[str, Mapping[str, Any]],
) -> str:
    lines = [
        f"{name}:{clean_text((artifacts.get(name) or {}).get('sha256'))}"
        for name in SYSTEM_CODE_ARTIFACTS
    ]
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()


def evaluate(
    args: argparse.Namespace,
    *,
    run_started_at: str,
    attempt_id: str,
) -> dict[str, Any]:
    """Perform the sole system execution after all freeze checks pass."""

    (
        base_payload,
        holdout_payload,
        base_queries,
        semantic_cases,
        negative_cases,
        reference_date,
        frozen_sha,
    ) = _load_inputs(args)
    metas = json.loads(args.metas.read_text(encoding="utf-8"))
    if not isinstance(metas, list) or not all(
        isinstance(meta, Mapping) for meta in metas
    ):
        raise ValueError("metas file must contain an array of objects")
    qrels = build_silver_qrels(metas, base_queries, reference_date)
    validate_nonempty_qrels(qrels)
    code_artifacts = _pre_run_code_artifacts()
    code_bundle_sha = system_code_bundle_sha256(code_artifacts)

    # Deliberately imported only after the holdout digest and qrels are fixed.
    from app.appearance_query import (
        APPEARANCE_GRAPH_EXCLUDED_FEATURE_PREFIXES,
        analyze_appearance_query,
        normalize_appearance_query,
    )

    runtime = load_heavy_runtime()
    index = read_faiss_index(runtime["faiss"], args.index)
    if int(index.ntotal) != len(metas):
        raise ValueError(
            f"index/metas mismatch: {index.ntotal} vectors != {len(metas)} rows"
        )
    docs, vector_to_doc = runtime["build_hybrid_documents"](metas)
    notices = merge_notice_metas(metas)
    reference_docs = prepare_reference_docs(
        docs,
        reference_date,
        include_unknown_notices=False,
    )
    bm25 = runtime["BM25Index"](reference_docs)
    graph = runtime["build_dog_graph"](reference_docs)
    device = resolve_device(runtime["torch"], args.device)
    model, _ = runtime["clip"].load(args.clip_model, device=device)
    model.eval()

    evaluation_depth = max(10, min(50, int(args.evaluation_depth)))
    rerank_depth = max(evaluation_depth, int(args.rerank_depth))
    vector_depth = min(
        int(index.ntotal),
        max(evaluation_depth * 30, rerank_depth * 3),
    )
    query_invocation_count = 0
    retrieval_execution_count = 0
    normalized_queries: set[str] = set()

    def retrieve(raw_query: str) -> dict[str, Any]:
        nonlocal query_invocation_count, retrieval_execution_count
        query_invocation_count += 1
        analysis = analyze_appearance_query(raw_query)
        normalized = clean_text(analysis.get("normalized_query"))
        direct_normalized = clean_text(normalize_appearance_query(raw_query))
        if normalized != direct_normalized:
            raise RuntimeError("appearance query analysis and normalization disagree")
        normalized_queries.add(normalized)
        ranking, retrieval = appearance_retrieval(
            runtime=runtime,
            model=model,
            device=device,
            index=index,
            metas=metas,
            docs=reference_docs,
            vector_to_doc=vector_to_doc,
            bm25=bm25,
            graph=graph,
            normalized_query=normalized,
            excluded_graph_feature_prefixes=(
                APPEARANCE_GRAPH_EXCLUDED_FEATURE_PREFIXES
            ),
            evaluation_depth=evaluation_depth,
            rerank_depth=rerank_depth,
            vector_depth=vector_depth,
        )
        retrieval_execution_count += 1
        return {
            "normalized_query": normalized,
            "query_policy": _query_policy_record(analysis),
            "ranking_ids": list(ranking),
            "retrieval": dict(retrieval),
        }

    base_runs = {
        clean_text(row.get("id")): retrieve(clean_text(row.get("query")))
        for row in base_queries
    }
    semantic_runs = {
        clean_text(row.get("id")): retrieve(clean_text(row.get("query")))
        for row in semantic_cases
    }
    negative_runs = {
        clean_text(row.get("id")): retrieve(clean_text(row.get("query")))
        for row in negative_cases
    }
    if query_invocation_count != EXPECTED_QUERY_EXECUTIONS:
        raise RuntimeError("unexpected query invocation count")
    if retrieval_execution_count != EXPECTED_QUERY_EXECUTIONS:
        raise RuntimeError("unexpected retrieval execution count")

    artifacts = {
        "index": report_artifact(
            args.index,
            vectors=int(index.ntotal),
            dimension=int(index.d),
        ),
        "metas": report_artifact(args.metas, rows=len(metas)),
        "base_queries": report_artifact(args.queries, count=len(base_queries)),
        "holdout": report_artifact(
            args.holdout,
            semantic_cases=len(semantic_cases),
            negative_diagnostics=len(negative_cases),
            frozen_before_system_import=True,
        ),
        "holdout_sha256": report_artifact(args.holdout_sha256),
        "evaluation_module": report_artifact(MODULE_PATH),
        "evaluation_cli": report_artifact(SCRIPT_PATH),
        **code_artifacts,
    }
    return build_report(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        run_started_at=run_started_at,
        reference_date=reference_date,
        base_query_payload=base_payload,
        holdout_payload=holdout_payload,
        qrels=qrels,
        metas_by_id=notices,
        base_runs=base_runs,
        semantic_runs=semantic_runs,
        negative_runs=negative_runs,
        artifacts=artifacts,
        runtime={
            "attempt_id": attempt_id,
            "device": device,
            "clip_model": args.clip_model,
            "evaluation_depth": evaluation_depth,
            "rerank_depth": rerank_depth,
            "vector_depth": vector_depth,
            "warmup": False,
            "include_unknown_notices": False,
            "appearance_query_normalization": True,
            "excluded_graph_feature_prefixes": list(
                APPEARANCE_GRAPH_EXCLUDED_FEATURE_PREFIXES
            ),
            "query_invocation_count": query_invocation_count,
            "retrieval_execution_count": retrieval_execution_count,
            "unique_normalized_queries": len(normalized_queries),
            "execution_mode": "one_shot_no_cache_no_repeat",
            "system_code_bundle_sha256": code_bundle_sha,
        },
        corpus={
            "vectors": int(index.ntotal),
            **corpus_summary(notices, reference_date),
        },
        frozen_holdout_sha256=frozen_sha,
    )


def _check_marker(
    args: argparse.Namespace,
    report: Mapping[str, Any],
    frozen_sha: str,
) -> list[str]:
    failures: list[str] = []
    if not args.run_marker.exists():
        return ["one-shot run marker is missing"]
    try:
        marker = load_json_object(args.run_marker, label="run marker")
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        return [f"one-shot run marker cannot be read: {exc}"]
    if marker.get("schema_version") != "appearance-query-holdout-run.v3":
        failures.append("one-shot run marker schema is stale")
    if marker.get("status") != "completed":
        failures.append("one-shot run marker is not completed")
    if int(marker.get("execution_count") or 0) != 1:
        failures.append("one-shot run marker execution count is not 1")
    if clean_text(marker.get("frozen_holdout_sha256")).lower() != frozen_sha:
        failures.append("one-shot run marker frozen SHA is stale")
    if clean_text(marker.get("attempt_id")) != clean_text(
        (report.get("runtime") or {}).get("attempt_id")
    ):
        failures.append("one-shot run marker attempt ID is stale")
    expected_report_sha = artifact_record(args.json_out).get("sha256")
    expected_markdown_sha = artifact_record(args.markdown_out).get("sha256")
    if marker.get("report_sha256") != expected_report_sha:
        failures.append("one-shot run marker report SHA is stale")
    if marker.get("markdown_sha256") != expected_markdown_sha:
        failures.append("one-shot run marker Markdown SHA is stale")
    return failures


def check_tracked_report(args: argparse.Namespace) -> list[str]:
    """Check hashes and recalculate evidence without rerunning search."""

    failures: list[str] = []
    if not args.json_out.exists():
        return [f"tracked JSON report is missing: {args.json_out}"]
    if not args.markdown_out.exists():
        return [f"tracked Markdown report is missing: {args.markdown_out}"]
    try:
        report = load_json_object(args.json_out, label="tracked JSON report")
        (
            base_payload,
            holdout_payload,
            base_queries,
            semantic_cases,
            negative_cases,
            reference_date,
            frozen_sha,
        ) = _load_inputs(args)
        metas = json.loads(args.metas.read_text(encoding="utf-8"))
        if not isinstance(metas, list) or not all(
            isinstance(meta, Mapping) for meta in metas
        ):
            return ["metas file must contain an array of objects"]
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        return [f"tracked report inputs cannot be read: {exc}"]

    expected_markdown = report_to_markdown(report)
    if args.markdown_out.read_text(encoding="utf-8") != expected_markdown:
        failures.append("tracked Markdown report does not match its JSON report")
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        failures.append("tracked report schema is stale")
    if clean_text(report.get("reference_date")) != reference_date.strftime("%Y-%m-%d"):
        failures.append("tracked report reference_date is stale")
    if (
        clean_text(
            (report.get("evaluation_design") or {}).get("frozen_holdout_sha256")
        ).lower()
        != frozen_sha
    ):
        failures.append("tracked frozen holdout SHA-256 is stale")
    system = report.get("system")
    if not isinstance(system, Mapping) or system.get("id") != SYSTEM_ID:
        failures.append("tracked report system is stale")

    expected_artifacts = {
        "index": args.index,
        "metas": args.metas,
        "base_queries": args.queries,
        "holdout": args.holdout,
        "holdout_sha256": args.holdout_sha256,
        "evaluation_module": MODULE_PATH,
        "evaluation_cli": SCRIPT_PATH,
        "appearance_query_module": APPEARANCE_QUERY_PATH,
        "hybrid_rag_module": HYBRID_RAG_PATH,
        "graph_rag_module": GRAPH_RAG_PATH,
        "retrieval_evaluation_module": RETRIEVAL_EVALUATION_PATH,
        "retrieval_evaluation_cli": RETRIEVAL_CLI_PATH,
    }
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, Mapping):
        failures.append("tracked report artifact records are missing")
    else:
        for name, path in expected_artifacts.items():
            record = artifacts.get(name)
            if not isinstance(record, Mapping):
                failures.append(f"tracked report artifact is missing: {name}")
                continue
            if clean_text(record.get("sha256")) != artifact_record(path).get("sha256"):
                failures.append(f"tracked report artifact hash is stale: {name}")

    runtime = report.get("runtime")
    expected_runtime = {
        "clip_model": args.clip_model,
        "evaluation_depth": max(10, min(50, int(args.evaluation_depth))),
        "rerank_depth": max(
            max(10, min(50, int(args.evaluation_depth))),
            int(args.rerank_depth),
        ),
        "warmup": False,
        "include_unknown_notices": False,
        "appearance_query_normalization": True,
        "query_invocation_count": EXPECTED_QUERY_EXECUTIONS,
        "retrieval_execution_count": EXPECTED_QUERY_EXECUTIONS,
        "execution_mode": "one_shot_no_cache_no_repeat",
    }
    if not isinstance(runtime, Mapping):
        failures.append("tracked report runtime settings are missing")
    else:
        for key, expected in expected_runtime.items():
            if runtime.get(key) != expected:
                failures.append(f"tracked report runtime setting is stale: {key}")

    if isinstance(artifacts, Mapping):
        current_bundle = system_code_bundle_sha256(artifacts)
        if clean_text((system or {}).get("code_bundle_sha256")) != current_bundle:
            failures.append("tracked system code bundle SHA is stale")
        if clean_text((runtime or {}).get("system_code_bundle_sha256")) != current_bundle:
            failures.append("tracked runtime code bundle SHA is stale")

    qrels = build_silver_qrels(metas, base_queries, reference_date)
    try:
        validate_nonempty_qrels(qrels)
    except ValueError as exc:
        failures.append(str(exc))
    notices = merge_notice_metas(metas)
    failures.extend(report_contract_failures(report))
    failures.extend(
        report_recalculation_failures(
            report,
            reference_date=reference_date,
            base_query_payload=base_payload,
            holdout_payload=holdout_payload,
            qrels=qrels,
            metas_by_id=notices,
        )
    )
    if len(base_queries) + len(semantic_cases) + len(negative_cases) != (
        EXPECTED_QUERY_EXECUTIONS
    ):
        failures.append("input query execution count is stale")
    failures.extend(_check_marker(args, report, frozen_sha))
    return sorted(set(failures))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen blind appearance-query v3 holdout exactly once, or "
            "verify its stored evidence without rerunning retrieval."
        )
    )
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--metas", type=Path, default=DEFAULT_METAS)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument(
        "--holdout-sha256",
        type=Path,
        default=DEFAULT_HOLDOUT_SHA256,
    )
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN_OUT)
    parser.add_argument("--run-marker", type=Path, default=DEFAULT_RUN_MARKER)
    parser.add_argument(
        "--reference-date",
        default=None,
        help="Fixed YYYY-MM-DD date; defaults to the holdout/base query value.",
    )
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--evaluation-depth", type=int, default=50)
    parser.add_argument("--rerank-depth", type=int, default=80)
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Verify hashes, Markdown, marker, invariants, and recalculated metrics "
            "from stored rankings without importing normalization or running search."
        ),
    )
    return parser.parse_args(argv)


def _new_attempt_marker(frozen_sha: str, started_at: str) -> dict[str, Any]:
    attempt_id = hashlib.sha256(
        f"{frozen_sha}\n{started_at}\none-shot-v3\n".encode()
    ).hexdigest()[:24]
    return {
        "schema_version": "appearance-query-holdout-run.v3",
        "attempt_id": attempt_id,
        "status": "started",
        "execution_count": 1,
        "started_at": started_at,
        "frozen_holdout_sha256": frozen_sha,
        "scope": {
            "base_queries": 12,
            "semantic_queries": 24,
            "negative_queries": 6,
            "retrieval_executions": EXPECTED_QUERY_EXECUTIONS,
        },
    }


def _refuse_repeat(args: argparse.Namespace) -> list[str]:
    return [
        f"one-shot output already exists: {path}"
        for path in (args.json_out, args.markdown_out, args.run_marker)
        if path.exists()
    ]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.check:
        failures = check_tracked_report(args)
        if failures:
            for failure in failures:
                print(f"query holdout v3 check failed: {failure}", file=sys.stderr)
            return 1
        print("query holdout v3 evaluation check: PASS")
        return 0

    repeat_failures = _refuse_repeat(args)
    if repeat_failures:
        for failure in repeat_failures:
            print(f"query holdout v3 run refused: {failure}", file=sys.stderr)
        return 2
    try:
        frozen_sha = frozen_holdout_sha256(args.holdout, args.holdout_sha256)
        started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        marker = _new_attempt_marker(frozen_sha, started_at)
        _write_json(args.run_marker, marker)
        report = evaluate(
            args,
            run_started_at=started_at,
            attempt_id=clean_text(marker.get("attempt_id")),
        )
        _write_json(args.json_out, report)
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(report_to_markdown(report), encoding="utf-8")
        marker.update(
            {
                "status": "completed",
                "completed_at": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
                "report_sha256": artifact_record(args.json_out).get("sha256"),
                "markdown_sha256": artifact_record(args.markdown_out).get("sha256"),
                "integrity_validation_passed": bool(
                    (report.get("validation") or {}).get("passed")
                ),
            }
        )
        _write_json(args.run_marker, marker)
    except Exception as exc:  # pragma: no cover - protects the one-shot record
        if args.run_marker.exists():
            try:
                marker = load_json_object(args.run_marker, label="run marker")
                marker.update(
                    {
                        "status": "failed",
                        "failed_at": datetime.now(timezone.utc).isoformat(
                            timespec="seconds"
                        ),
                        "error_type": type(exc).__name__,
                        "error": clean_text(exc),
                    }
                )
                _write_json(args.run_marker, marker)
            except Exception:
                pass
        print(f"query holdout v3 one-shot failed: {exc}", file=sys.stderr)
        return 1

    summary = report["semantic_holdout"]["summary"]
    print(
        json.dumps(
            {
                "json": str(args.json_out),
                "markdown": str(args.markdown_out),
                "run_marker": str(args.run_marker),
                "reference_date": report["reference_date"],
                "frozen_holdout_sha256": report["evaluation_design"][
                    "frozen_holdout_sha256"
                ],
                "system_code_bundle_sha256": report["system"][
                    "code_bundle_sha256"
                ],
                "query_invocation_count": report["runtime"][
                    "query_invocation_count"
                ],
                "retrieval_execution_count": report["runtime"][
                    "retrieval_execution_count"
                ],
                "base_metrics": summary["base_metrics"],
                "holdout_metrics": summary["holdout_metrics"],
                "metric_delta": summary["metric_delta_from_base_average"],
                "rank_stability": summary["rank_stability"],
                "negative_diagnostics": report["negative_diagnostics"]["summary"],
                "inactive_exposure": report["inactive_exposure"],
                "validation": report["validation"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["validation"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
