"""Compare raw CLIP retrieval with the current hybrid+graph pipeline.

Heavy ML dependencies are imported only after argument and input validation, so
the evaluation helpers and CLI help remain usable in a lightweight CI job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Mapping, Sequence


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.notice_status import classify_notice  # noqa: E402
from app.retrieval_evaluation import (  # noqa: E402
    BASELINE_SYSTEM_ID,
    DIAGNOSTIC_SYSTEM_ID,
    HEADLINE_SYSTEM_ID,
    ORACLE_SYSTEM_ID,
    REPORT_SCHEMA_VERSION,
    SILVER_LABEL_VERSION,
    aggregate_query_results,
    artifact_record,
    build_query_result,
    build_silver_qrels,
    clean_text,
    extract_public_attributes,
    merge_notice_metas,
    metric_delta,
    notice_id,
    parse_reference_date,
    qrels_record,
    report_contract_failures,
    report_to_markdown,
    validate_nonempty_qrels,
    validate_query_specs,
)


DATA_DIR = BASE_DIR / "data"
DEFAULT_INDEX = DATA_DIR / "dog_faiss.index"
DEFAULT_METAS = DATA_DIR / "dog_metas.json"
DEFAULT_QUERIES = DATA_DIR / "eval_queries.appearance_v1.json"
DEFAULT_JSON_OUT = (
    BASE_DIR / "docs" / "evaluation" / "retrieval_eval.appearance_v1.json"
)
DEFAULT_MARKDOWN_OUT = (
    BASE_DIR / "docs" / "evaluation" / "retrieval_eval.appearance_v1.md"
)

HYBRID_BM25_SYSTEM_ID = "hybrid_bm25_natural"
NATURAL_CONDITIONS_SYSTEM_ID = "hybrid_natural_conditions"
SYSTEM_ORDER = (
    DIAGNOSTIC_SYSTEM_ID,
    BASELINE_SYSTEM_ID,
    HYBRID_BM25_SYSTEM_ID,
    NATURAL_CONDITIONS_SYSTEM_ID,
    HEADLINE_SYSTEM_ID,
    ORACLE_SYSTEM_ID,
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_query_payload(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = load_json(path)
    if not isinstance(payload, Mapping):
        raise ValueError("query file must contain an object")
    queries = payload.get("queries")
    if not isinstance(queries, list) or not all(
        isinstance(query, Mapping) for query in queries
    ):
        raise ValueError("query file must contain a queries array")
    normalized = [dict(query) for query in queries]
    validate_query_specs(normalized)
    return dict(payload), normalized


def report_artifact(path: Path, **extra: Any) -> dict[str, Any]:
    """Record a portable repo-relative path while hashing the real file."""

    record = artifact_record(path, **extra)
    try:
        record["path"] = path.resolve().relative_to(BASE_DIR).as_posix()
    except ValueError:
        record["path"] = str(path.resolve())
    return record


def load_heavy_runtime() -> dict[str, Any]:
    """Load packages omitted from lightweight CI only when evaluation runs."""

    import clip
    import faiss
    import torch

    from app.graph_rag import build_dog_graph, rerank_with_graph
    from app.hybrid_rag import (
        BM25Index,
        build_hybrid_documents,
        build_search_query_text,
        merge_structured_query,
        parse_structured_query,
        rank_hybrid_documents,
        vector_hits_to_doc_modality_scores,
        vector_hits_to_doc_scores,
    )

    return {
        "clip": clip,
        "faiss": faiss,
        "torch": torch,
        "BM25Index": BM25Index,
        "build_dog_graph": build_dog_graph,
        "build_hybrid_documents": build_hybrid_documents,
        "build_search_query_text": build_search_query_text,
        "merge_structured_query": merge_structured_query,
        "parse_structured_query": parse_structured_query,
        "rank_hybrid_documents": rank_hybrid_documents,
        "rerank_with_graph": rerank_with_graph,
        "vector_hits_to_doc_modality_scores": vector_hits_to_doc_modality_scores,
        "vector_hits_to_doc_scores": vector_hits_to_doc_scores,
    }


def read_faiss_index(faiss_module: Any, path: Path) -> Any:
    try:
        return faiss_module.read_index(str(path))
    except RuntimeError as exc:
        if "Illegal byte sequence" not in str(exc):
            raise
        safe_dir = Path(tempfile.gettempdir()) / "dog_faiss_safe"
        safe_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
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


def encode_text(
    runtime: Mapping[str, Any],
    model: Any,
    device: str,
    text: str,
) -> Any:
    clip_module = runtime["clip"]
    torch_module = runtime["torch"]
    tokens = clip_module.tokenize([text], truncate=True).to(device)
    with torch_module.no_grad():
        features = model.encode_text(tokens)
        features /= features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().astype("float32")


def _doc_id(docs: Sequence[Mapping[str, Any]], doc_index: int) -> str:
    if not 0 <= doc_index < len(docs):
        return ""
    doc = docs[doc_index]
    return clean_text(doc.get("doc_id")) or notice_id(doc.get("meta") or {})


def _mean_repeat(
    retrieve: Callable[[], tuple[list[str], dict[str, Any]]],
    repeat: int,
) -> tuple[list[str], dict[str, Any], float]:
    latencies: list[float] = []
    ranking: list[str] = []
    details: dict[str, Any] = {}
    for _ in range(max(1, repeat)):
        started = time.perf_counter()
        current_ranking, current_details = retrieve()
        latencies.append((time.perf_counter() - started) * 1000.0)
        if not ranking:
            ranking = current_ranking
            details = current_details
    return ranking, details, mean(latencies)


def clip_only_retrieval(
    runtime: Mapping[str, Any],
    model: Any,
    device: str,
    index: Any,
    docs: Sequence[Mapping[str, Any]],
    vector_to_doc: Mapping[int, int],
    raw_query: str,
    evaluation_depth: int,
    vector_depth: int,
    *,
    status_filtered: bool = False,
) -> tuple[list[str], dict[str, Any]]:
    """Raw query -> CLIP/FAISS, optionally on the fixed searchable corpus."""

    query_vec = encode_text(runtime, model, device, raw_query)
    distances, indices = index.search(
        query_vec,
        min(int(index.ntotal), vector_depth),
    )
    vector_scores = runtime["vector_hits_to_doc_scores"](
        distances[0].tolist(),
        indices[0].tolist(),
        vector_to_doc,
    )
    ordered = sorted(
        vector_scores,
        key=lambda doc_index: (
            -float(vector_scores[doc_index]),
            _doc_id(docs, doc_index),
        ),
    )
    ranking: list[str] = []
    for doc_index in ordered:
        if status_filtered:
            meta = docs[doc_index].get("meta") or {}
            if not bool(meta.get("active")):
                continue
        dog_id = _doc_id(docs, doc_index)
        if dog_id and dog_id not in ranking:
            ranking.append(dog_id)
        if len(ranking) >= evaluation_depth:
            break
    return ranking, {
        "search_query": raw_query,
        "candidate_documents": len(vector_scores),
        "status_filter": "fixed_searchable_corpus" if status_filtered else "none",
        "status_filtered": status_filtered,
        "uses_explicit_structured": False,
    }


def hybrid_stage_retrieval(
    runtime: Mapping[str, Any],
    model: Any,
    device: str,
    index: Any,
    metas: Sequence[Mapping[str, Any]],
    docs: Sequence[Mapping[str, Any]],
    vector_to_doc: Mapping[int, int],
    bm25: Any,
    graph: Any,
    query: Mapping[str, Any],
    reference_date: datetime,
    evaluation_depth: int,
    rerank_depth: int,
    vector_depth: int,
    include_unknown_notices: bool,
    *,
    structured_source: str,
    use_graph: bool,
) -> tuple[list[str], dict[str, Any]]:
    """Run one production-backed hybrid ablation stage."""

    raw_query = clean_text(query.get("query"))
    natural_parsed = runtime["parse_structured_query"](raw_query)
    explicit = query.get("structured")
    uses_explicit = structured_source == "oracle" and isinstance(explicit, Mapping)
    if structured_source == "none":
        structured: dict[str, Any] = {}
        search_query = raw_query
    elif structured_source == "natural":
        structured = natural_parsed
        search_query = runtime["build_search_query_text"](raw_query, structured)
    elif structured_source == "oracle":
        structured = dict(natural_parsed)
        if isinstance(explicit, Mapping):
            structured = runtime["merge_structured_query"](
                structured,
                dict(explicit),
            )
        search_query = runtime["build_search_query_text"](raw_query, structured)
    else:
        raise ValueError(f"unsupported structured source: {structured_source}")

    search_query = search_query or raw_query
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
    graph_candidates = (
        graph.candidate_doc_scores(structured, search_query, limit=0)
        if use_graph
        else {}
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
        extra_candidate_indices=graph_candidates.keys() if use_graph else None,
        extra_candidate_scores=graph_candidates if use_graph else None,
        # ``docs`` carries statuses normalized to the fixed reference date.
        filter_inactive_notices=True,
        include_unknown_notices=include_unknown_notices,
    )
    if use_graph:
        ranked = runtime["rerank_with_graph"](
            ranked=ranked,
            graph=graph,
            structured=structured,
            query_text=search_query,
            topk=candidate_limit,
        )

    ranking: list[str] = []
    for item in ranked:
        meta = (item.get("doc") or {}).get("meta") or {}
        dog_id = notice_id(meta) or clean_text((item.get("doc") or {}).get("doc_id"))
        if dog_id and dog_id not in ranking:
            ranking.append(dog_id)
        if len(ranking) >= evaluation_depth:
            break
    return ranking, {
        "search_query": search_query,
        "natural_parsed_query": natural_parsed,
        "structured_query": structured,
        "structured_source": structured_source,
        "uses_explicit_structured": uses_explicit,
        "graph_enabled": use_graph,
        "candidate_documents": len(ranked),
        "status_filter": (
            f"active+unknown@{reference_date:%Y-%m-%d}"
            if include_unknown_notices
            else f"active@{reference_date:%Y-%m-%d}"
        ),
        "status_filtered": True,
    }


def hybrid_graph_retrieval(
    runtime: Mapping[str, Any],
    model: Any,
    device: str,
    index: Any,
    metas: Sequence[Mapping[str, Any]],
    docs: Sequence[Mapping[str, Any]],
    vector_to_doc: Mapping[int, int],
    bm25: Any,
    graph: Any,
    query: Mapping[str, Any],
    reference_date: datetime,
    evaluation_depth: int,
    rerank_depth: int,
    vector_depth: int,
    include_unknown_notices: bool,
) -> tuple[list[str], dict[str, Any]]:
    """Compatibility wrapper for the former explicit-structured evaluation."""

    return hybrid_stage_retrieval(
        runtime,
        model,
        device,
        index,
        metas,
        docs,
        vector_to_doc,
        bm25,
        graph,
        query,
        reference_date,
        evaluation_depth,
        rerank_depth,
        vector_depth,
        include_unknown_notices,
        structured_source="oracle",
        use_graph=True,
    )


def prepare_reference_docs(
    docs: Sequence[Mapping[str, Any]],
    reference_date: datetime,
    include_unknown_notices: bool,
) -> list[dict[str, Any]]:
    """Freeze notice status so app filtering cannot drift with wall-clock time."""

    prepared: list[dict[str, Any]] = []
    for doc in docs:
        copied = dict(doc)
        meta = dict(doc.get("meta") or {})
        status = classify_notice(meta, reference_date=reference_date)
        searchable = status == "active" or (
            include_unknown_notices and status == "unknown"
        )
        frozen_state = "active" if searchable else "closed"
        for key in ("process_state", "processState", "status", "notice_status"):
            meta[key] = frozen_state
        for key in ("notice_end", "noticeEdt"):
            meta[key] = ""
        meta["active"] = searchable
        copied["meta"] = meta
        prepared.append(copied)
    return prepared


def corpus_summary(
    notices: Mapping[str, Mapping[str, Any]],
    reference_date: datetime,
) -> dict[str, Any]:
    statuses: dict[str, int] = {}
    coverage = {
        "color": 0,
        "weight": 0,
        "age": 0,
        "region": 0,
    }
    for meta in notices.values():
        attrs = extract_public_attributes(meta, reference_date)
        status = clean_text(attrs.get("notice_status")) or "unknown"
        statuses[status] = statuses.get(status, 0) + 1
        coverage["color"] += int(bool(attrs.get("colors")))
        coverage["weight"] += int(attrs.get("weight_kg") is not None)
        coverage["age"] += int(bool(attrs.get("age_hint")))
        coverage["region"] += int(bool(attrs.get("region")))
    return {
        "documents": len(notices),
        "active_documents": statuses.get("active", 0),
        "notice_status": statuses,
        "public_field_coverage_count": coverage,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    query_payload, queries = load_query_payload(args.queries)
    reference_date = parse_reference_date(
        args.reference_date or query_payload.get("reference_date")
    )
    metas = load_json(args.metas)
    if not isinstance(metas, list) or not all(
        isinstance(meta, Mapping) for meta in metas
    ):
        raise ValueError("metas file must contain an array of objects")

    qrels = build_silver_qrels(metas, queries, reference_date)
    if not args.allow_empty_qrels:
        validate_nonempty_qrels(qrels)

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
        args.include_unknown_notices,
    )
    bm25 = runtime["BM25Index"](reference_docs)
    graph = runtime["build_dog_graph"](reference_docs)

    device = resolve_device(runtime["torch"], args.device)
    model, _ = runtime["clip"].load(args.clip_model, device=device)
    model.eval()
    for _ in range(max(0, args.warmup)):
        encode_text(runtime, model, device, "갈색 소형견")

    evaluation_depth = max(10, min(50, int(args.evaluation_depth)))
    rerank_depth = max(evaluation_depth, int(args.rerank_depth))
    vector_depth = min(
        int(index.ntotal),
        max(evaluation_depth * 30, rerank_depth * 3),
    )

    system_queries: dict[str, list[dict[str, Any]]] = {
        system_id: [] for system_id in SYSTEM_ORDER
    }

    def append_result(
        system_id: str,
        query: Mapping[str, Any],
        ranked_ids: Sequence[str],
        relevant: set[str],
        details: Mapping[str, Any],
        latency_ms: float,
    ) -> None:
        result = build_query_result(
            query,
            ranked_ids,
            relevant,
            notices,
            reference_date,
            latency_ms,
            args.top_ids,
        )
        result["retrieval"] = dict(details)
        system_queries[system_id].append(result)

    for query in queries:
        query_id = clean_text(query.get("id"))
        relevant = qrels[query_id]
        raw_query = clean_text(query.get("query"))

        for system_id, status_filtered in (
            (DIAGNOSTIC_SYSTEM_ID, False),
            (BASELINE_SYSTEM_ID, True),
        ):
            ranked_ids, details, latency = _mean_repeat(
                lambda raw_query=raw_query, status_filtered=status_filtered: (
                    clip_only_retrieval(
                        runtime,
                        model,
                        device,
                        index,
                        reference_docs,
                        vector_to_doc,
                        raw_query,
                        evaluation_depth,
                        vector_depth,
                        status_filtered=status_filtered,
                    )
                ),
                args.repeat,
            )
            append_result(
                system_id,
                query,
                ranked_ids,
                relevant,
                details,
                latency,
            )

        hybrid_stages = (
            (HYBRID_BM25_SYSTEM_ID, "none", False),
            (NATURAL_CONDITIONS_SYSTEM_ID, "natural", False),
            (HEADLINE_SYSTEM_ID, "natural", True),
            (ORACLE_SYSTEM_ID, "oracle", True),
        )
        for system_id, structured_source, use_graph in hybrid_stages:
            ranked_ids, details, latency = _mean_repeat(
                lambda query=query, structured_source=structured_source, use_graph=use_graph: (
                    hybrid_stage_retrieval(
                        runtime,
                        model,
                        device,
                        index,
                        metas,
                        reference_docs,
                        vector_to_doc,
                        bm25,
                        graph,
                        query,
                        reference_date,
                        evaluation_depth,
                        rerank_depth,
                        vector_depth,
                        args.include_unknown_notices,
                        structured_source=structured_source,
                        use_graph=use_graph,
                    )
                ),
                args.repeat,
            )
            append_result(
                system_id,
                query,
                ranked_ids,
                relevant,
                details,
                latency,
            )

    system_metrics = {
        system_id: aggregate_query_results(rows)
        for system_id, rows in system_queries.items()
    }
    system_definitions = {
        DIAGNOSTIC_SYSTEM_ID: {
            "role": "inactive_diagnostic_only",
            "description": (
                "Raw natural-language CLIP/FAISS retrieval without notice-status "
                "filtering. It is not the performance baseline."
            ),
            "uses_explicit_structured": False,
            "status_filtered": False,
        },
        BASELINE_SYSTEM_ID: {
            "role": "status_matched_baseline",
            "description": (
                "Raw natural-language CLIP/FAISS retrieval on the same fixed "
                "searchable corpus as every candidate system."
            ),
            "uses_explicit_structured": False,
            "status_filtered": True,
        },
        HYBRID_BM25_SYSTEM_ID: {
            "role": "ablation",
            "description": (
                "Raw natural-language CLIP+BM25 production fusion with condition "
                "and graph signals disabled; the shared visual-metadata term remains."
            ),
            "uses_explicit_structured": False,
            "status_filtered": True,
        },
        NATURAL_CONDITIONS_SYSTEM_ID: {
            "role": "ablation",
            "description": (
                "Natural-language parsed conditions with CLIP+BM25 production "
                "fusion and no graph reranking."
            ),
            "uses_explicit_structured": False,
            "status_filtered": True,
        },
        HEADLINE_SYSTEM_ID: {
            "role": "headline_natural_only",
            "description": (
                "End-to-end natural-language parser, CLIP, BM25, condition scoring, "
                "and graph reranking. Evaluation JSON structured hints are ignored."
            ),
            "uses_explicit_structured": False,
            "status_filtered": True,
        },
        ORACLE_SYSTEM_ID: {
            "role": "oracle_upper_bound_not_headline",
            "description": (
                "Explicit evaluation structured hints merged into parsed conditions "
                "before hybrid+graph retrieval. This is an oracle upper bound only."
            ),
            "uses_explicit_structured": True,
            "status_filtered": True,
        },
    }
    systems = {
        system_id: {
            **system_definitions[system_id],
            "metrics": system_metrics[system_id],
            "queries": system_queries[system_id],
        }
        for system_id in SYSTEM_ORDER
    }
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reference_date": reference_date.strftime("%Y-%m-%d"),
        "label_policy": {
            "version": SILVER_LABEL_VERSION,
            "source_fields": [
                "public_notice.color",
                "public_notice.weight",
                "public_notice.age",
                "public_notice.region",
                "public_notice.status",
            ],
            "uses_result_text": False,
            "uses_ranked_results": False,
            "uses_vlm_attributes": False,
            "relevance": "binary_all_requested_dimensions_match",
        },
        "artifacts": {
            "index": report_artifact(
                args.index,
                vectors=int(index.ntotal),
                dimension=int(index.d),
            ),
            "metas": report_artifact(args.metas, rows=len(metas)),
            "queries": report_artifact(
                args.queries,
                count=len(queries),
                schema_version=query_payload.get("schema_version"),
            ),
        },
        "runtime": {
            "device": device,
            "clip_model": args.clip_model,
            "repeat": max(1, int(args.repeat)),
            "warmup": max(0, int(args.warmup)),
            "evaluation_depth": evaluation_depth,
            "rerank_depth": rerank_depth,
            "vector_depth": vector_depth,
            "top_ids": max(1, int(args.top_ids)),
            "include_unknown_notices": args.include_unknown_notices,
        },
        "corpus": {
            "vectors": int(index.ntotal),
            **corpus_summary(notices, reference_date),
        },
        "qrels": qrels_record(queries, qrels),
        "evaluation_contract": {
            "headline_system": HEADLINE_SYSTEM_ID,
            "baseline_system": BASELINE_SYSTEM_ID,
            "oracle_system": ORACLE_SYSTEM_ID,
            "diagnostic_system": DIAGNOSTIC_SYSTEM_ID,
            "system_order": list(SYSTEM_ORDER),
            "status_filtered_systems": [
                system_id
                for system_id in SYSTEM_ORDER
                if system_id != DIAGNOSTIC_SYSTEM_ID
            ],
            "headline_uses_query_structured": False,
            "oracle_is_upper_bound_only": True,
        },
        "systems": systems,
        "delta": metric_delta(
            system_metrics[BASELINE_SYSTEM_ID],
            system_metrics[HEADLINE_SYSTEM_ID],
        ),
        "diagnostic_inactive_delta": metric_delta(
            system_metrics[DIAGNOSTIC_SYSTEM_ID],
            system_metrics[BASELINE_SYSTEM_ID],
        ),
        "oracle_delta_from_headline": metric_delta(
            system_metrics[HEADLINE_SYSTEM_ID],
            system_metrics[ORACLE_SYSTEM_ID],
        ),
        "ablation_deltas_from_baseline": {
            system_id: metric_delta(
                system_metrics[BASELINE_SYSTEM_ID],
                system_metrics[system_id],
            )
            for system_id in SYSTEM_ORDER
            if system_id not in {DIAGNOSTIC_SYSTEM_ID, BASELINE_SYSTEM_ID}
        },
        "stage_deltas": {
            current: {
                "from_system": previous,
                "delta": metric_delta(
                    system_metrics[previous],
                    system_metrics[current],
                ),
            }
            for previous, current in zip(
                (
                    BASELINE_SYSTEM_ID,
                    HYBRID_BM25_SYSTEM_ID,
                    NATURAL_CONDITIONS_SYSTEM_ID,
                    HEADLINE_SYSTEM_ID,
                ),
                (
                    HYBRID_BM25_SYSTEM_ID,
                    NATURAL_CONDITIONS_SYSTEM_ID,
                    HEADLINE_SYSTEM_ID,
                    ORACLE_SYSTEM_ID,
                ),
            )
        },
    }
    invariant_failures = report_contract_failures(report)
    report["validation"] = {
        "passed": not invariant_failures,
        "failure_count": len(invariant_failures),
        "failures": invariant_failures,
    }
    return report


def _unit_interval(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return parsed


def check_tracked_report(args: argparse.Namespace) -> list[str]:
    """Check tracked report provenance and invariants without loading ML models."""

    failures: list[str] = []
    if not args.json_out.exists():
        return [f"tracked JSON report is missing: {args.json_out}"]
    if not args.markdown_out.exists():
        failures.append(f"tracked Markdown report is missing: {args.markdown_out}")
    try:
        report = load_json(args.json_out)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return [f"tracked JSON report cannot be read: {exc}"]
    if not isinstance(report, Mapping):
        return ["tracked JSON report must contain an object"]
    if args.markdown_out.exists():
        expected_markdown = report_to_markdown(report)
        if args.markdown_out.read_text(encoding="utf-8") != expected_markdown:
            failures.append("tracked Markdown report does not match its JSON report")

    query_payload, queries = load_query_payload(args.queries)
    expected_date = parse_reference_date(
        args.reference_date or query_payload.get("reference_date")
    ).strftime("%Y-%m-%d")
    if clean_text(report.get("schema_version")) != REPORT_SCHEMA_VERSION:
        failures.append(
            "report schema is stale: "
            f"{clean_text(report.get('schema_version'))!r} != {REPORT_SCHEMA_VERSION!r}"
        )
    if clean_text(report.get("reference_date")) != expected_date:
        failures.append(
            f"report reference_date {report.get('reference_date')!r} != {expected_date!r}"
        )

    runtime = report.get("runtime")
    if not isinstance(runtime, Mapping):
        failures.append("report runtime settings are missing")
    else:
        expected_runtime = {
            "clip_model": args.clip_model,
            "evaluation_depth": max(10, min(50, int(args.evaluation_depth))),
            "rerank_depth": max(
                max(10, min(50, int(args.evaluation_depth))),
                int(args.rerank_depth),
            ),
            "top_ids": max(1, int(args.top_ids)),
            "include_unknown_notices": args.include_unknown_notices,
        }
        for key, expected in expected_runtime.items():
            if runtime.get(key) != expected:
                failures.append(
                    f"report runtime setting is stale: {key}="
                    f"{runtime.get(key)!r} != {expected!r}"
                )

    artifacts = report.get("artifacts")
    expected_artifacts = {
        "index": args.index,
        "metas": args.metas,
        "queries": args.queries,
    }
    if not isinstance(artifacts, Mapping):
        failures.append("report artifacts are missing")
    else:
        for name, path in expected_artifacts.items():
            record = artifacts.get(name)
            if not isinstance(record, Mapping):
                failures.append(f"report artifact is missing: {name}")
                continue
            actual_hash = artifact_record(path).get("sha256")
            if clean_text(record.get("sha256")) != actual_hash:
                failures.append(f"report artifact hash is stale: {name}")

    qrels = report.get("qrels")
    if not isinstance(qrels, Sequence) or len(qrels) != len(queries):
        failures.append("report qrels do not match the fixed query count")
    failures.extend(
        report_contract_failures(
            report,
            min_headline_precision_at_5=args.min_headline_precision_at_5,
            min_headline_hit_at_5=args.min_headline_hit_at_5,
            max_headline_inactive_exposure=args.max_headline_inactive_exposure,
        )
    )
    return failures


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate status-matched CLIP against natural-only hybrid+graph search, "
            "with explicit structured input reported only as an oracle upper bound."
        )
    )
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--metas", type=Path, default=DEFAULT_METAS)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=DEFAULT_MARKDOWN_OUT,
    )
    parser.add_argument(
        "--reference-date",
        default=None,
        help="Fixed YYYY-MM-DD date; defaults to the query-set value.",
    )
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--evaluation-depth", type=int, default=50)
    parser.add_argument("--rerank-depth", type=int, default=80)
    parser.add_argument("--top-ids", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--include-unknown-notices",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--allow-empty-qrels",
        action="store_true",
        help="Keep zero-relevance queries instead of failing the run.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Check tracked report input hashes, schema, and deterministic "
            "invariants without loading ML models or rewriting outputs."
        ),
    )
    parser.add_argument(
        "--min-headline-precision-at-5",
        type=_unit_interval,
        default=0.70,
        help=("Release floor for natural-only headline precision@5 (default 0.70)."),
    )
    parser.add_argument(
        "--min-headline-hit-at-5",
        type=_unit_interval,
        default=0.90,
        help="Release floor for natural-only headline hit@5 (default 0.90).",
    )
    parser.add_argument(
        "--max-headline-inactive-exposure",
        type=_unit_interval,
        default=0.0,
        help="Maximum allowed natural-only headline inactive exposure (default 0).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.check:
        failures = check_tracked_report(args)
        if failures:
            for failure in failures:
                print(f"retrieval evaluation check failed: {failure}", file=sys.stderr)
            return 1
        print("retrieval evaluation check: PASS")
        return 0

    report = evaluate(args)
    failures = report_contract_failures(
        report,
        min_headline_precision_at_5=args.min_headline_precision_at_5,
        min_headline_hit_at_5=args.min_headline_hit_at_5,
        max_headline_inactive_exposure=args.max_headline_inactive_exposure,
    )
    report["validation"] = {
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "release_criteria": {
            "min_headline_precision_at_5": args.min_headline_precision_at_5,
            "min_headline_hit_at_5": args.min_headline_hit_at_5,
            "max_headline_inactive_exposure": args.max_headline_inactive_exposure,
        },
    }
    markdown = report_to_markdown(report)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.markdown_out.write_text(markdown, encoding="utf-8")

    baseline = report["systems"][BASELINE_SYSTEM_ID]["metrics"]
    candidate = report["systems"][HEADLINE_SYSTEM_ID]["metrics"]
    print(
        json.dumps(
            {
                "json": str(args.json_out),
                "markdown": str(args.markdown_out),
                "reference_date": report["reference_date"],
                "index_sha256": report["artifacts"]["index"]["sha256"],
                "metas_sha256": report["artifacts"]["metas"]["sha256"],
                BASELINE_SYSTEM_ID: baseline,
                HEADLINE_SYSTEM_ID: candidate,
                "delta": report["delta"],
                "validation": report["validation"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
