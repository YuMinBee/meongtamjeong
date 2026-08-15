from __future__ import annotations

import ast
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pytest
import scripts.evaluate_retrieval as retrieval_cli

from app.retrieval_evaluation import (
    BASELINE_SYSTEM_ID,
    DIAGNOSTIC_SYSTEM_ID,
    HEADLINE_SYSTEM_ID,
    ORACLE_SYSTEM_ID,
    REPORT_SCHEMA_VERSION,
    aggregate_query_results,
    artifact_record,
    build_query_result,
    build_silver_qrels,
    extract_public_attributes,
    merge_notice_metas,
    metric_delta,
    parse_reference_date,
    query_metrics,
    report_to_markdown,
    report_contract_failures,
    validate_nonempty_qrels,
    validate_query_specs,
)
from scripts.evaluate_retrieval import (
    DEFAULT_JSON_OUT,
    DEFAULT_MARKDOWN_OUT,
    SYSTEM_ORDER,
    check_tracked_report,
    hybrid_stage_retrieval,
    parse_args,
    prepare_reference_docs,
    report_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
QUERY_SET = ROOT / "data" / "eval_queries.appearance_v1.json"
CLI_PATH = ROOT / "scripts" / "evaluate_retrieval.py"
REPORT_JSON = ROOT / "docs" / "evaluation" / "retrieval_eval.appearance_v1.json"
REPORT_MARKDOWN = ROOT / "docs" / "evaluation" / "retrieval_eval.appearance_v1.md"


def active_meta(
    dog_id: str,
    *,
    color: str = "갈색",
    weight: str = "5(Kg)",
    age: str = "2024(년생)",
    region: str = "서울특별시",
    notice_end: str = "20260731",
) -> dict[str, object]:
    return {
        "desertionNo": dog_id,
        "type": "image",
        "color": color,
        "weight": weight,
        "age": age,
        "org_name": region,
        "process_state": "보호중",
        "notice_end": notice_end,
    }


def test_extract_public_attributes_uses_reference_date_and_notice_fields() -> None:
    meta = active_meta(
        "dog-1",
        color="갈색&흰색",
        weight="7.2(Kg)",
        age="2026(년생)",
        region="경기도 수원시",
    )
    attrs = extract_public_attributes(meta, datetime(2026, 7, 25))

    assert attrs == {
        "dog_id": "dog-1",
        "colors": ("brown", "white"),
        "weight_kg": 7.2,
        "size_hint": "small",
        "age_hint": "puppy",
        "region": "경기",
        "notice_status": "active",
    }


def test_silver_qrels_never_use_description_result_text_or_vlm_fields() -> None:
    relevant = {
        **active_meta("right", color="흰색", weight="4(Kg)"),
        "desc": "검정색 대형견",
        "desc_full": "경기도의 검정색 대형견",
        "vlm_attrs": {
            "coat_color": ["black"],
            "body_size_hint": "large",
        },
    }
    decoy = {
        **active_meta("wrong", color="검정색", weight="24(Kg)"),
        "desc": "흰색 소형견",
        "desc_full": "서울의 흰색 소형견",
        "vlm_attrs": {
            "coat_color": ["white"],
            "body_size_hint": "small",
        },
    }
    query = {
        "id": "white-small-seoul",
        "query": "서울의 흰색 소형견",
        "criteria": {
            "colors_any": ["white"],
            "sizes_any": ["small"],
            "regions_any": ["서울"],
        },
    }

    qrels = build_silver_qrels(
        [relevant, decoy],
        [query],
        datetime(2026, 7, 25),
    )

    assert qrels == {"white-small-seoul": {"right"}}


def test_qrels_use_fixed_reference_date_and_exclude_expired_notices() -> None:
    query = {
        "id": "brown",
        "query": "갈색 강아지",
        "criteria": {"colors_any": ["brown"]},
    }
    expired = active_meta("expired", notice_end="20260724")
    open_notice = active_meta("open", notice_end="20260726")

    july_23 = build_silver_qrels(
        [expired, open_notice],
        [query],
        parse_reference_date("2026-07-23"),
    )
    july_25 = build_silver_qrels(
        [expired, open_notice],
        [query],
        parse_reference_date("2026-07-25"),
    )

    assert july_23["brown"] == {"expired", "open"}
    assert july_25["brown"] == {"open"}


def test_merge_notice_metas_collapses_modalities_without_losing_public_fields() -> None:
    metas = [
        {
            "desertionNo": "same",
            "type": "text",
            "color": "흰색",
            "weight": "Unknown",
        },
        {
            "desertionNo": "same",
            "type": "image",
            "weight": "3(Kg)",
            "org_name": "서울특별시",
        },
    ]

    merged = merge_notice_metas(metas)

    assert list(merged) == ["same"]
    assert merged["same"]["color"] == "흰색"
    assert merged["same"]["weight"] == "3(Kg)"
    assert merged["same"]["org_name"] == "서울특별시"


def test_query_metrics_cover_required_measures_and_inactive_exposure() -> None:
    reference_date = datetime(2026, 7, 25)
    ranked = ["expired", "r2", "x", "r1", "y", "z"]
    relevant = {"r1", "r2", "r3"}
    metas = {dog_id: active_meta(dog_id) for dog_id in ranked}
    metas["expired"] = active_meta("expired", notice_end="20260724")

    metrics = query_metrics(ranked, relevant, metas, reference_date)
    ideal_dcg = 1.0 + (1.0 / 1.584962500721156) + 0.5
    actual_dcg = (1.0 / 1.584962500721156) + (1.0 / 2.321928094887362)

    assert metrics["precision@5"] == 0.4
    assert metrics["recall@5"] == pytest.approx(2 / 3, abs=1e-6)
    assert metrics["recall@10"] == pytest.approx(2 / 3, abs=1e-6)
    assert metrics["nDCG@5"] == pytest.approx(
        actual_dcg / ideal_dcg,
        abs=1e-6,
    )
    assert metrics["MRR"] == 0.5
    assert metrics["hit@5"] == 1.0
    assert metrics["inactive_exposure"] == pytest.approx(1 / 6, abs=1e-6)


def test_query_result_and_aggregate_keep_relevant_count_top_ids_and_latency() -> None:
    reference_date = datetime(2026, 7, 25)
    query = {
        "id": "q1",
        "query": "갈색 소형견",
        "criteria": {"colors_any": ["brown"]},
    }
    metas = {
        "a": active_meta("a"),
        "b": active_meta("b"),
    }
    first = build_query_result(
        query,
        ["a", "a", "b"],
        {"a"},
        metas,
        reference_date,
        latency_ms=10.0,
        top_ids_limit=2,
    )
    second = {
        **first,
        "latency_ms": 20.0,
        "metrics": {
            **first["metrics"],
            "precision@5": 0.0,
            "hit@5": 0.0,
        },
    }

    summary = aggregate_query_results([first, second])

    assert first["relevant_count"] == 1
    assert first["top_ids"] == ["a", "b"]
    assert summary["query_count"] == 2
    assert summary["precision@5"] == 0.1
    assert summary["hit@5"] == 0.5
    assert summary["latency_ms"] == 15.0
    assert summary["latency_ms_p50"] == 15.0
    assert summary["latency_ms_p95"] == 19.5


def test_metric_delta_and_markdown_report_are_consistent() -> None:
    baseline_metrics = {
        "query_count": 1,
        "precision@5": 0.2,
        "recall@5": 0.1,
        "recall@10": 0.2,
        "nDCG@5": 0.3,
        "MRR": 0.25,
        "hit@5": 1.0,
        "inactive_exposure": 0.2,
        "latency_ms": 4.0,
        "latency_ms_p50": 4.0,
        "latency_ms_p95": 4.0,
    }
    candidate_metrics = {
        **baseline_metrics,
        "precision@5": 0.4,
        "inactive_exposure": 0.0,
        "latency_ms": 7.0,
        "latency_ms_p50": 7.0,
        "latency_ms_p95": 7.0,
    }
    query_row = {
        "id": "q",
        "relevant_count": 2,
        "top_ids": ["one", "two"],
        "metrics": {"precision@5": 0.4},
    }
    delta = metric_delta(baseline_metrics, candidate_metrics)
    systems = {
        system_id: {
            "role": "test",
            "uses_explicit_structured": system_id == ORACLE_SYSTEM_ID,
            "status_filtered": system_id != DIAGNOSTIC_SYSTEM_ID,
            "metrics": baseline_metrics
            if system_id == BASELINE_SYSTEM_ID
            else candidate_metrics,
            "queries": [
                {
                    **query_row,
                    "retrieval": {
                        "uses_explicit_structured": system_id == ORACLE_SYSTEM_ID,
                        "structured_source": "natural",
                        "structured_query": {"coat_color": ["white"]},
                        "natural_parsed_query": {"coat_color": ["white"]},
                    },
                }
            ],
        }
        for system_id in SYSTEM_ORDER
    }
    report = {
        "reference_date": "2026-07-25",
        "artifacts": {
            name: {"path": name, "sha256": "a" * 64}
            for name in ("index", "metas", "queries")
        },
        "corpus": {"documents": 10, "active_documents": 8},
        "evaluation_contract": {
            "headline_system": HEADLINE_SYSTEM_ID,
            "baseline_system": BASELINE_SYSTEM_ID,
            "oracle_system": ORACLE_SYSTEM_ID,
            "diagnostic_system": DIAGNOSTIC_SYSTEM_ID,
            "system_order": list(SYSTEM_ORDER),
            "status_filtered_systems": list(SYSTEM_ORDER[1:]),
        },
        "systems": systems,
        "qrels": [{"id": "q", "relevant_count": 2}],
        "delta": delta,
    }

    markdown = report_to_markdown(report)

    assert delta["precision@5"] == 0.2
    assert delta["inactive_exposure"] == -0.2
    assert delta["latency_ms"] == 3.0
    assert "2026-07-25" in markdown
    assert "검색 결과 텍스트·순위·설명·VLM 출력은 정답 라벨" in markdown
    assert "`generated_at`과 latency 값은 결정적 산출물이 아닙니다" in markdown
    assert "사람 검수 정답이 아닌 시스템 간 회귀 비교용 근사치" in markdown
    assert HEADLINE_SYSTEM_ID in markdown
    assert "상한선이며 대표 성능이 아님" in markdown
    assert "`one, two`" in markdown


def test_query_set_is_fixed_and_covers_all_public_label_dimensions() -> None:
    payload = json.loads(QUERY_SET.read_text(encoding="utf-8"))
    queries = payload["queries"]
    validate_query_specs(queries)

    criteria_keys = {key for query in queries for key in query["criteria"]}
    assert payload["reference_date"] == "2026-07-26"
    assert len(queries) >= 10
    assert {
        "colors_any",
        "sizes_any",
        "age_hints_any",
        "regions_any",
        "weight_kg",
    }.issubset(criteria_keys)


def test_validation_rejects_bad_queries_and_empty_qrels() -> None:
    with pytest.raises(ValueError, match="unsupported criteria"):
        validate_query_specs(
            [
                {
                    "id": "bad",
                    "query": "query",
                    "criteria": {"description_contains": "white"},
                }
            ]
        )
    with pytest.raises(ValueError, match="no relevant documents"):
        validate_nonempty_qrels({"empty": set()})
    with pytest.raises(ValueError, match="reference_date is required"):
        parse_reference_date("")


def test_artifact_record_includes_full_sha256() -> None:
    record = artifact_record(QUERY_SET, rows=1)
    expected = hashlib.sha256(QUERY_SET.read_bytes()).hexdigest()

    assert record["path"] == str(QUERY_SET)
    assert record["sha256"] == expected
    assert record["bytes"] == QUERY_SET.stat().st_size
    assert record["rows"] == 1


def test_cli_defaults_to_tracked_docs_path_and_portable_artifact_names() -> None:
    assert DEFAULT_JSON_OUT == (
        ROOT / "docs" / "evaluation" / "retrieval_eval.appearance_v1.json"
    )
    assert DEFAULT_MARKDOWN_OUT == (
        ROOT / "docs" / "evaluation" / "retrieval_eval.appearance_v1.md"
    )
    assert report_artifact(QUERY_SET)["path"] == (
        "data/eval_queries.appearance_v1.json"
    )


def test_committed_report_is_auditable_and_matches_input_hashes() -> None:
    report = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    markdown = REPORT_MARKDOWN.read_text(encoding="utf-8")

    for record in report["artifacts"].values():
        artifact_path = ROOT / record["path"]
        assert (
            hashlib.sha256(artifact_path.read_bytes()).hexdigest() == (record["sha256"])
        )
    assert report["schema_version"] == REPORT_SCHEMA_VERSION
    assert report["reference_date"] == "2026-07-26"
    assert report["evaluation_contract"]["headline_system"] == HEADLINE_SYSTEM_ID
    assert report["evaluation_contract"]["oracle_system"] == ORACLE_SYSTEM_ID
    assert report["systems"][HEADLINE_SYSTEM_ID]["uses_explicit_structured"] is False
    assert report["systems"][ORACLE_SYSTEM_ID]["uses_explicit_structured"] is True
    assert report["validation"]["passed"] is True
    assert report["label_policy"]["uses_result_text"] is False
    assert report["label_policy"]["uses_ranked_results"] is False
    assert all(item["relevant_count"] > 0 for item in report["qrels"])
    assert all(
        item["relevant_count"] == len(item["relevant_ids"]) for item in report["qrels"]
    )
    assert "inactive_exposure" in report["delta"]
    assert "JSON의 `generated_at`과 latency 값은 결정적 산출물이 아닙니다" in markdown


def test_report_contract_rejects_oracle_headline_and_release_regression() -> None:
    query = {
        "id": "q",
        "retrieval": {
            "uses_explicit_structured": False,
            "structured_source": "natural",
            "structured_query": {"coat_color": ["white"]},
            "natural_parsed_query": {"coat_color": ["white"]},
        },
    }
    metrics = {
        "query_count": 1,
        "precision@5": 0.4,
        "hit@5": 1.0,
        "inactive_exposure": 0.0,
    }
    systems = {
        system_id: {
            "uses_explicit_structured": system_id == ORACLE_SYSTEM_ID,
            "status_filtered": system_id != DIAGNOSTIC_SYSTEM_ID,
            "metrics": metrics,
            "queries": [query],
        }
        for system_id in SYSTEM_ORDER
    }
    report = {
        "evaluation_contract": {
            "headline_system": HEADLINE_SYSTEM_ID,
            "baseline_system": BASELINE_SYSTEM_ID,
            "oracle_system": ORACLE_SYSTEM_ID,
            "diagnostic_system": DIAGNOSTIC_SYSTEM_ID,
            "status_filtered_systems": list(SYSTEM_ORDER[1:]),
        },
        "systems": systems,
        "qrels": [{"id": "q"}],
    }

    assert report_contract_failures(report) == []
    assert any(
        "precision@5" in failure
        for failure in report_contract_failures(
            report,
            min_headline_precision_at_5=0.5,
        )
    )
    report["systems"][HEADLINE_SYSTEM_ID]["uses_explicit_structured"] = True
    assert any(
        "headline system" in failure for failure in report_contract_failures(report)
    )


def test_natural_stage_ignores_explicit_structured_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Row(list[float]):
        def tolist(self) -> list[float]:
            return list(self)

    class FakeIndex:
        ntotal = 1

        def search(self, _query: object, _depth: int) -> tuple[list[Row], list[Row]]:
            return [Row([0.9])], [Row([0])]

    class FakeGraph:
        def candidate_doc_scores(
            self,
            _structured: object,
            _query: str,
            *,
            limit: int,
        ) -> dict[int, float]:
            assert limit == 0
            return {}

    captured: list[dict[str, object]] = []

    def rank_hybrid_documents(**kwargs: object) -> list[dict[str, object]]:
        captured.append(dict(kwargs["structured"]))  # type: ignore[arg-type]
        docs = kwargs["docs"]  # type: ignore[assignment]
        return [{"doc": docs[0], "doc_index": 0, "score": 0.5}]  # type: ignore[index]

    runtime = {
        "parse_structured_query": lambda _query: {"coat_color": ["parsed"]},
        "merge_structured_query": lambda base, override: {**base, **override},
        "build_search_query_text": lambda raw, _structured: raw,
        "vector_hits_to_doc_modality_scores": lambda *_args: ({0: 0.9}, {0: {}}),
        "rank_hybrid_documents": rank_hybrid_documents,
        "rerank_with_graph": lambda **kwargs: kwargs["ranked"],
    }
    monkeypatch.setattr(retrieval_cli, "encode_text", lambda *_args: object())
    meta = {"desertionNo": "dog", "active": True, "process_state": "active"}
    docs = [{"doc_id": "dog", "meta": meta}]
    query = {
        "id": "q",
        "query": "흰색 강아지",
        "structured": {"coat_color": ["oracle"]},
    }
    common = (
        runtime,
        object(),
        "cpu",
        FakeIndex(),
        [meta],
        docs,
        {0: 0},
        object(),
        FakeGraph(),
        query,
        datetime(2026, 7, 26),
        10,
        10,
        1,
        False,
    )

    _, natural = hybrid_stage_retrieval(
        *common,
        structured_source="natural",
        use_graph=False,
    )
    _, oracle = hybrid_stage_retrieval(
        *common,
        structured_source="oracle",
        use_graph=True,
    )

    assert captured == [
        {"coat_color": ["parsed"]},
        {"coat_color": ["oracle"]},
    ]
    assert natural["uses_explicit_structured"] is False
    assert oracle["uses_explicit_structured"] is True


def test_lightweight_check_detects_stale_artifact_hash(tmp_path: Path) -> None:
    index_path = tmp_path / "index.bin"
    metas_path = tmp_path / "metas.json"
    queries_path = tmp_path / "queries.json"
    json_out = tmp_path / "report.json"
    markdown_out = tmp_path / "report.md"
    index_path.write_bytes(b"index-v1")
    metas_path.write_text("[]", encoding="utf-8")
    queries_path.write_text(
        json.dumps(
            {
                "reference_date": "2026-07-26",
                "queries": [
                    {
                        "id": "q",
                        "query": "흰색 강아지",
                        "criteria": {"colors_any": ["white"]},
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    metrics = {
        "query_count": 1,
        "precision@5": 0.70,
        "hit@5": 0.90,
        "inactive_exposure": 0.0,
    }
    query_row = {
        "id": "q",
        "retrieval": {
            "uses_explicit_structured": False,
            "structured_source": "natural",
            "structured_query": {"coat_color": ["white"]},
            "natural_parsed_query": {"coat_color": ["white"]},
        },
    }
    systems = {
        system_id: {
            "uses_explicit_structured": system_id == ORACLE_SYSTEM_ID,
            "status_filtered": system_id != DIAGNOSTIC_SYSTEM_ID,
            "metrics": metrics,
            "queries": [query_row],
        }
        for system_id in SYSTEM_ORDER
    }
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "reference_date": "2026-07-26",
        "runtime": {
            "clip_model": "ViT-B/32",
            "evaluation_depth": 50,
            "rerank_depth": 80,
            "top_ids": 10,
            "include_unknown_notices": True,
        },
        "artifacts": {
            "index": artifact_record(index_path),
            "metas": artifact_record(metas_path),
            "queries": artifact_record(queries_path),
        },
        "evaluation_contract": {
            "headline_system": HEADLINE_SYSTEM_ID,
            "baseline_system": BASELINE_SYSTEM_ID,
            "oracle_system": ORACLE_SYSTEM_ID,
            "diagnostic_system": DIAGNOSTIC_SYSTEM_ID,
            "status_filtered_systems": list(SYSTEM_ORDER[1:]),
        },
        "systems": systems,
        "qrels": [{"id": "q"}],
    }
    json_out.write_text(json.dumps(report), encoding="utf-8")
    markdown_out.write_text(report_to_markdown(report), encoding="utf-8")
    args = parse_args(
        [
            "--check",
            "--index",
            str(index_path),
            "--metas",
            str(metas_path),
            "--queries",
            str(queries_path),
            "--json-out",
            str(json_out),
            "--markdown-out",
            str(markdown_out),
        ]
    )

    assert check_tracked_report(args) == []
    index_path.write_bytes(b"index-v2")
    assert "report artifact hash is stale: index" in check_tracked_report(args)


def test_cli_keeps_heavy_dependencies_inside_functions() -> None:
    tree = ast.parse(CLI_PATH.read_text(encoding="utf-8"))
    module_imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            module_imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module_imports.add((node.module or "").split(".")[0])

    assert {"clip", "faiss", "torch", "numpy"}.isdisjoint(module_imports)


def test_reference_docs_keep_notice_filter_stable_after_reference_date() -> None:
    docs = [
        {"doc_id": "expired", "meta": active_meta("expired", notice_end="20260724")},
        {"doc_id": "open", "meta": active_meta("open", notice_end="20260726")},
        {"doc_id": "unknown", "meta": {"desertionNo": "unknown"}},
    ]

    prepared = prepare_reference_docs(
        docs,
        datetime(2026, 7, 25),
        include_unknown_notices=True,
    )

    assert [item["meta"]["process_state"] for item in prepared] == [
        "closed",
        "active",
        "active",
    ]
    # Frozen fields no longer drift when app helpers are called on a later day.
    assert [
        extract_public_attributes(item["meta"], datetime(2030, 1, 1))["notice_status"]
        for item in prepared
    ] == ["closed", "active", "active"]
