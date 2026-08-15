"""Generate or verify the appearance-query development regression report."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.appearance_query import (  # noqa: E402
    APPEARANCE_GRAPH_EXCLUDED_FEATURE_PREFIXES,
    normalize_appearance_query,
)
from app.query_robustness_evaluation import (  # noqa: E402
    REPORT_SCHEMA_VERSION,
    SYSTEM_ID,
    build_report,
    report_contract_failures,
    report_recalculation_failures,
    report_to_markdown,
    validate_variant_payload,
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
DEFAULT_VARIANTS = DATA_DIR / "eval_query_variants.appearance_v1.json"
DEFAULT_JSON_OUT = (
    BASE_DIR / "docs" / "evaluation" / "query_robustness.appearance_v1.json"
)
DEFAULT_MARKDOWN_OUT = (
    BASE_DIR / "docs" / "evaluation" / "query_robustness.appearance_v1.md"
)
MODULE_PATH = BASE_DIR / "app" / "query_robustness_evaluation.py"
SCRIPT_PATH = Path(__file__).resolve()
APPEARANCE_QUERY_PATH = BASE_DIR / "app" / "appearance_query.py"
HYBRID_RAG_PATH = BASE_DIR / "app" / "hybrid_rag.py"
GRAPH_RAG_PATH = BASE_DIR / "app" / "graph_rag.py"
RETRIEVAL_EVALUATION_PATH = BASE_DIR / "app" / "retrieval_evaluation.py"
RETRIEVAL_CLI_PATH = BASE_DIR / "scripts" / "evaluate_retrieval.py"


def load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must contain a JSON object")
    return dict(payload)


def report_artifact(path: Path, **extra: Any) -> dict[str, Any]:
    record = artifact_record(path, **extra)
    try:
        record["path"] = path.resolve().relative_to(BASE_DIR).as_posix()
    except ValueError:
        record["path"] = str(path.resolve())
    return record


def _load_inputs(
    args: argparse.Namespace,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    datetime,
]:
    base_payload = load_json_object(args.queries, label="base query file")
    base_queries = base_payload.get("queries")
    if not isinstance(base_queries, list) or not all(
        isinstance(row, Mapping) for row in base_queries
    ):
        raise ValueError("base query file must contain a queries array")
    base_queries = [dict(row) for row in base_queries]
    validate_query_specs(base_queries)
    base_payload["queries"] = base_queries

    variant_payload = load_json_object(args.variants, label="variant file")
    variants = validate_variant_payload(variant_payload, base_payload)
    reference_date = parse_reference_date(
        args.reference_date or base_payload.get("reference_date")
    )
    return base_payload, variant_payload, base_queries, variants, reference_date


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
    raw_query: str,
    evaluation_depth: int,
    rerank_depth: int,
    vector_depth: int,
) -> tuple[list[str], str]:
    """Mirror the retrieval portion of ``ranking_scope=appearance``."""

    normalized_query = normalize_appearance_query(raw_query)
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
        excluded_feature_prefixes=APPEARANCE_GRAPH_EXCLUDED_FEATURE_PREFIXES,
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
        excluded_feature_prefixes=APPEARANCE_GRAPH_EXCLUDED_FEATURE_PREFIXES,
    )
    ranking: list[str] = []
    for item in ranked:
        doc = item.get("doc") or {}
        dog_id = notice_id(doc.get("meta") or {}) or clean_text(doc.get("doc_id"))
        if dog_id and dog_id not in ranking:
            ranking.append(dog_id)
    return ranking, normalized_query


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    (
        base_payload,
        variant_payload,
        base_queries,
        variants,
        reference_date,
    ) = _load_inputs(args)
    metas = json.loads(args.metas.read_text(encoding="utf-8"))
    if not isinstance(metas, list) or not all(
        isinstance(meta, Mapping) for meta in metas
    ):
        raise ValueError("metas file must contain an array of objects")
    qrels = build_silver_qrels(metas, base_queries, reference_date)
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
        include_unknown_notices=False,
    )
    bm25 = runtime["BM25Index"](reference_docs)
    graph = runtime["build_dog_graph"](reference_docs)
    device = resolve_device(runtime["torch"], args.device)
    model, _ = runtime["clip"].load(args.clip_model, device=device)
    model.eval()
    if args.warmup:
        encode_text(runtime, model, device, "갈색 소형견")

    evaluation_depth = max(10, min(50, int(args.evaluation_depth)))
    rerank_depth = max(evaluation_depth, int(args.rerank_depth))
    vector_depth = min(
        int(index.ntotal),
        max(evaluation_depth * 30, rerank_depth * 3),
    )
    ranking_cache: dict[str, list[str]] = {}

    def retrieve(query: str) -> list[str]:
        normalized = normalize_appearance_query(query)
        if normalized not in ranking_cache:
            ranking, actual_normalized = appearance_retrieval(
                runtime=runtime,
                model=model,
                device=device,
                index=index,
                metas=metas,
                docs=reference_docs,
                vector_to_doc=vector_to_doc,
                bm25=bm25,
                graph=graph,
                raw_query=query,
                evaluation_depth=evaluation_depth,
                rerank_depth=rerank_depth,
                vector_depth=vector_depth,
            )
            if actual_normalized != normalized:
                raise RuntimeError("appearance query normalization changed during run")
            ranking_cache[normalized] = ranking
        return list(ranking_cache[normalized])

    base_rankings = {
        clean_text(row.get("id")): retrieve(clean_text(row.get("query")))
        for row in base_queries
    }
    variant_rankings = {row["id"]: retrieve(row["query"]) for row in variants}
    artifacts = {
        "index": report_artifact(
            args.index,
            vectors=int(index.ntotal),
            dimension=int(index.d),
        ),
        "metas": report_artifact(args.metas, rows=len(metas)),
        "base_queries": report_artifact(args.queries, count=len(base_queries)),
        "variants": report_artifact(args.variants, count=len(variants)),
        "evaluation_module": report_artifact(MODULE_PATH),
        "evaluation_cli": report_artifact(SCRIPT_PATH),
        "appearance_query_module": report_artifact(APPEARANCE_QUERY_PATH),
        "hybrid_rag_module": report_artifact(HYBRID_RAG_PATH),
        "graph_rag_module": report_artifact(GRAPH_RAG_PATH),
        "retrieval_evaluation_module": report_artifact(RETRIEVAL_EVALUATION_PATH),
        "retrieval_evaluation_cli": report_artifact(RETRIEVAL_CLI_PATH),
    }
    return build_report(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        reference_date=reference_date,
        base_query_payload=base_payload,
        variant_payload=variant_payload,
        qrels=qrels,
        metas_by_id=notices,
        base_rankings=base_rankings,
        variant_rankings=variant_rankings,
        artifacts=artifacts,
        runtime={
            "device": device,
            "clip_model": args.clip_model,
            "evaluation_depth": evaluation_depth,
            "rerank_depth": rerank_depth,
            "vector_depth": vector_depth,
            "warmup": bool(args.warmup),
            "include_unknown_notices": False,
            "appearance_query_normalization": True,
            "excluded_graph_feature_prefixes": list(
                APPEARANCE_GRAPH_EXCLUDED_FEATURE_PREFIXES
            ),
            "unique_normalized_queries": len(ranking_cache),
        },
        corpus={
            "vectors": int(index.ntotal),
            **corpus_summary(notices, reference_date),
        },
    )


def check_tracked_report(args: argparse.Namespace) -> list[str]:
    """Fail closed on stale inputs/code and recalculate stored evidence."""

    failures: list[str] = []
    if not args.json_out.exists():
        return [f"tracked JSON report is missing: {args.json_out}"]
    if not args.markdown_out.exists():
        failures.append(f"tracked Markdown report is missing: {args.markdown_out}")
    try:
        report = load_json_object(args.json_out, label="tracked JSON report")
        (
            base_payload,
            variant_payload,
            base_queries,
            variants,
            reference_date,
        ) = _load_inputs(args)
        metas = json.loads(args.metas.read_text(encoding="utf-8"))
        if not isinstance(metas, list) or not all(
            isinstance(meta, Mapping) for meta in metas
        ):
            return ["metas file must contain an array of objects"]
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        return [f"tracked report inputs cannot be read: {exc}"]

    if args.markdown_out.exists():
        expected_markdown = report_to_markdown(report)
        if args.markdown_out.read_text(encoding="utf-8") != expected_markdown:
            failures.append("tracked Markdown report does not match its JSON report")
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        failures.append("tracked report schema is stale")
    if clean_text(report.get("reference_date")) != reference_date.strftime("%Y-%m-%d"):
        failures.append("tracked report reference_date is stale")
    system = report.get("system")
    if not isinstance(system, Mapping) or system.get("id") != SYSTEM_ID:
        failures.append("tracked report system is stale")

    expected_artifacts = {
        "index": args.index,
        "metas": args.metas,
        "base_queries": args.queries,
        "variants": args.variants,
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
        "include_unknown_notices": False,
        "appearance_query_normalization": True,
        "excluded_graph_feature_prefixes": list(
            APPEARANCE_GRAPH_EXCLUDED_FEATURE_PREFIXES
        ),
    }
    if not isinstance(runtime, Mapping):
        failures.append("tracked report runtime settings are missing")
    else:
        for key, expected in expected_runtime.items():
            if runtime.get(key) != expected:
                failures.append(f"tracked report runtime setting is stale: {key}")

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
            variant_payload=variant_payload,
            qrels=qrels,
            metas_by_id=notices,
        )
    )
    contract = report.get("evaluation_contract") or {}
    if int(contract.get("variant_count") or 0) != len(variants):
        failures.append("tracked report variant count is stale")
    return sorted(set(failures))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate retained appearance-query development variants against the "
            "existing silver qrels and report rank stability without a quality floor."
        )
    )
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--metas", type=Path, default=DEFAULT_METAS)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--variants", type=Path, default=DEFAULT_VARIANTS)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN_OUT)
    parser.add_argument(
        "--reference-date",
        default=None,
        help="Fixed YYYY-MM-DD date; defaults to the base query-set value.",
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
        "--warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Verify hashes, Markdown, invariants, and recomputed metrics from "
            "stored rankings without loading CLIP/FAISS."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.check:
        failures = check_tracked_report(args)
        if failures:
            for failure in failures:
                print(f"query robustness check failed: {failure}", file=sys.stderr)
            return 1
        print("query robustness evaluation check: PASS")
        return 0

    report = evaluate(args)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.markdown_out.write_text(report_to_markdown(report), encoding="utf-8")
    summary = report["robustness_summary"]
    print(
        json.dumps(
            {
                "json": str(args.json_out),
                "markdown": str(args.markdown_out),
                "reference_date": report["reference_date"],
                "robustness_metrics": summary["metrics"],
                "rank_stability": summary["stability"],
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
