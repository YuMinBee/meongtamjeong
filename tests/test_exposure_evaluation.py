from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.exposure_evaluation import (
    DEFAULT_SYSTEM_IDS,
    GROUP_DIMENSIONS,
    REPORT_SCHEMA_VERSION,
    UNKNOWN_GROUP,
    evaluate_exposure_representation,
    notice_groups,
    query_evidence_dimensions,
    render_markdown,
    serialize_report,
)
from app.profile_evaluation import sha256_file
from app.retrieval_evaluation import extract_public_attributes, parse_reference_date
from scripts.evaluate_exposure_representation import main


ROOT = Path(__file__).resolve().parents[1]
METAS = ROOT / "data" / "dog_metas.json"
QUERIES = ROOT / "data" / "eval_queries.appearance_v1.json"
RETRIEVAL_REPORT = ROOT / "docs" / "evaluation" / "retrieval_eval.appearance_v1.json"
REPORT_JSON = (
    ROOT / "docs" / "evaluation" / "exposure_representation.appearance_v1.json"
)
REPORT_MARKDOWN = (
    ROOT / "docs" / "evaluation" / "exposure_representation.appearance_v1.md"
)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _active_meta(
    dog_id: str,
    *,
    row_type: str = "text",
    color: str = "갈색",
    weight: str = "2(Kg)",
    age: str = "2026(년생)",
    region: str = "서울특별시",
    shelter: str = "보호소 A",
    quality: str | None = None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "desertionNo": dog_id,
        "type": row_type,
        "color": color,
        "weight": weight,
        "age": age,
        "org_name": region,
        "care_name": shelter,
        "process_state": "보호중",
        "notice_end": "20260831",
    }
    if quality is not None:
        meta["image_attrs"] = {"photo_quality_band": quality}
    return meta


def _fixture_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    metas_path = tmp_path / "metas.json"
    queries_path = tmp_path / "queries.json"
    retrieval_path = tmp_path / "retrieval.json"
    metas = [
        _active_meta("a", row_type="image", quality="high"),
        _active_meta("a", row_type="text"),
        _active_meta(
            "b",
            color="검정",
            weight="10(Kg)",
            age="2020(년생)",
            region="경기도 수원시",
            shelter="보호소 B",
        ),
        _active_meta(
            "c",
            color="Unknown",
            weight="Unknown",
            age="Unknown",
            region="Unknown",
            shelter="Unknown",
        ),
    ]
    queries = {
        "schema_version": "appearance-query-set.v1",
        "reference_date": "2026-07-26",
        "queries": [
            {
                "id": "q1",
                "query": "서울의 갈색 소형견",
                "criteria": {
                    "colors_any": ["brown"],
                    "sizes_any": ["tiny"],
                    "regions_any": ["서울"],
                    "active_only": True,
                },
            },
            {
                "id": "q2",
                "query": "성견",
                "criteria": {
                    "age_hints_any": ["adult"],
                    "active_only": True,
                },
            },
        ],
    }
    _write_json(metas_path, metas)
    _write_json(queries_path, queries)
    retrieval = {
        "schema_version": "retrieval-evaluation.v2",
        "reference_date": "2026-07-26",
        "artifacts": {
            "metas": {"path": "metas.json", "sha256": sha256_file(metas_path)},
            "queries": {
                "path": "queries.json",
                "sha256": sha256_file(queries_path),
            },
            "index": {"path": "index", "sha256": "0" * 64},
        },
        "corpus": {"active_documents": 3},
        "systems": {
            "clip_active": {
                "role": "status_matched_baseline",
                "status_filtered": True,
                "uses_explicit_structured": False,
                "queries": [
                    {"id": "q1", "top_ids": ["a", "b"]},
                    {"id": "q2", "top_ids": ["b", "c"]},
                ],
            },
            "hybrid_natural_graph": {
                "role": "headline_natural_only",
                "status_filtered": True,
                "uses_explicit_structured": False,
                "queries": [
                    {"id": "q1", "top_ids": ["b", "a"]},
                    {"id": "q2", "top_ids": ["a", "c"]},
                ],
            },
        },
    }
    _write_json(retrieval_path, retrieval)
    return metas_path, retrieval_path, queries_path


def _group(report: dict[str, Any], dimension: str, name: str) -> dict[str, Any]:
    return next(
        row for row in report["dimensions"][dimension]["groups"] if row["group"] == name
    )


def test_group_normalization_and_query_evidence_mapping() -> None:
    reference_date = parse_reference_date("2026-07-26")
    meta = _active_meta(
        "a",
        shelter="  보호소　A  ",
        quality="HIGH",
    )
    attributes = extract_public_attributes(meta, reference_date)

    assert notice_groups(meta, attributes) == {
        "region": "서울",
        "shelter": "보호소 A",
        "size": "tiny",
        "age": "puppy",
        "photo_quality": "high",
    }
    assert query_evidence_dimensions(
        {
            "colors_any": ["brown"],
            "weight_kg": {"max": 8},
            "age_hints_any": ["adult"],
            "regions_any": ["서울"],
            "active_only": True,
        }
    ) == ("color", "size", "age", "region")


def test_exposure_counts_unique_notices_unknown_and_ratios(tmp_path: Path) -> None:
    metas, retrieval, queries = _fixture_inputs(tmp_path)
    report = evaluate_exposure_representation(
        metas,
        retrieval,
        queries,
        top_k=2,
    )

    assert report["schema_version"] == REPORT_SCHEMA_VERSION
    assert report["corpus"] == {"metadata_rows": 4, "active_unique_notices": 3}
    baseline = report["systems"]["clip_active"]
    assert baseline["exposure_slots"] == 4
    assert baseline["unique_exposed_notice_count"] == 3
    assert baseline["query_evidence_coverage"]["overall"] == {
        "applicable_count": 8,
        "evaluated_count": 7,
        "unknown_count": 1,
        "evidence_coverage": 0.875,
    }

    seoul = _group(report, "region", "서울")
    gyeonggi = _group(report, "region", "경기")
    unknown = _group(report, "region", UNKNOWN_GROUP)
    assert seoul["representation_share"] == pytest.approx(1 / 3, abs=1e-6)
    assert seoul["exposure_by_system"]["clip_active"] == {
        "exposure_count": 1,
        "exposure_share": 0.25,
        "exposure_representation_ratio": pytest.approx(0.75, abs=1e-6),
    }
    assert gyeonggi["exposure_by_system"]["clip_active"]["exposure_count"] == 2
    assert gyeonggi["exposure_by_system"]["clip_active"][
        "exposure_representation_ratio"
    ] == pytest.approx(1.5, abs=1e-6)
    assert unknown["corpus_count"] == 1
    for dimension in GROUP_DIMENSIONS:
        assert _group(report, dimension, UNKNOWN_GROUP)
        rows = report["dimensions"][dimension]["groups"]
        assert sum(row["corpus_count"] for row in rows) == 3
        for system_id in DEFAULT_SYSTEM_IDS:
            assert (
                sum(
                    row["exposure_by_system"][system_id]["exposure_count"]
                    for row in rows
                )
                == 4
            )

    markdown = render_markdown(report)
    assert "규범적 공정성 평가가 아니며" in markdown
    assert "보호특성으로 주장하지 않습니다" in markdown
    assert "unknown" in markdown


def test_invalid_fixed_ranking_or_artifact_hash_fails_closed(tmp_path: Path) -> None:
    metas, retrieval, queries = _fixture_inputs(tmp_path)
    payload = json.loads(retrieval.read_text(encoding="utf-8"))
    payload["systems"]["clip_active"]["queries"][0]["top_ids"] = ["a", "a"]
    _write_json(retrieval, payload)
    with pytest.raises(ValueError, match="duplicate IDs"):
        evaluate_exposure_representation(
            metas,
            retrieval,
            queries,
            top_k=2,
        )

    metas, retrieval, queries = _fixture_inputs(tmp_path)
    payload = json.loads(retrieval.read_text(encoding="utf-8"))
    payload["artifacts"]["metas"]["sha256"] = "f" * 64
    _write_json(retrieval, payload)
    with pytest.raises(ValueError, match="SHA-256"):
        evaluate_exposure_representation(
            metas,
            retrieval,
            queries,
            top_k=2,
        )


def test_cli_check_is_byte_deterministic(tmp_path: Path) -> None:
    metas, retrieval, queries = _fixture_inputs(tmp_path)
    json_out = tmp_path / "report.json"
    markdown_out = tmp_path / "report.md"
    common_args = [
        "--metas",
        str(metas),
        "--retrieval-report",
        str(retrieval),
        "--queries",
        str(queries),
        "--top-k",
        "2",
        "--json-out",
        str(json_out),
        "--markdown-out",
        str(markdown_out),
    ]

    assert main(common_args) == 0
    first_json = json_out.read_bytes()
    assert main([*common_args, "--check"]) == 0
    assert json_out.read_bytes() == first_json
    json_out.write_text("{}\n", encoding="utf-8")
    assert main([*common_args, "--check"]) == 1


def test_committed_exposure_report_matches_fixed_artifacts() -> None:
    report = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    markdown = REPORT_MARKDOWN.read_text(encoding="utf-8")

    assert report["schema_version"] == REPORT_SCHEMA_VERSION
    assert report["scope"] == "descriptive_retrieval_exposure_not_normative_fairness"
    assert report["inputs"]["dog_metas"]["sha256"] == sha256_file(METAS)
    assert report["inputs"]["queries"]["sha256"] == sha256_file(QUERIES)
    assert report["inputs"]["retrieval_report"]["sha256"] == sha256_file(
        RETRIEVAL_REPORT
    )
    assert report["corpus"]["active_unique_notices"] == 1516
    assert report["validation"] == {
        "passed": True,
        "failure_count": 0,
        "failures": [],
    }
    assert serialize_report(report) == REPORT_JSON.read_text(encoding="utf-8")
    assert render_markdown(report) == markdown

    expected_corpus_groups = {
        "size": {"tiny": 526, "small": 547, "medium": 325, "large": 118},
        "age": {"puppy": 808, "adult": 627, "senior": 81},
        "photo_quality": {"high": 719, "medium": 477, "low": 32},
    }
    for dimension, expected in expected_corpus_groups.items():
        for group_name, count in expected.items():
            assert _group(report, dimension, group_name)["corpus_count"] == count
    assert _group(report, "region", UNKNOWN_GROUP)["corpus_count"] == 0
    assert _group(report, "photo_quality", UNKNOWN_GROUP)["corpus_count"] == 288
    assert _group(report, "size", UNKNOWN_GROUP)["corpus_count"] == 0
    assert (
        _group(report, "size", UNKNOWN_GROUP)["exposure_by_system"][
            "hybrid_natural_graph"
        ]["exposure_representation_ratio"]
        is None
    )
    assert report["systems"]["hybrid_natural_graph"]["exposure_slots"] == 120
    for dimension in GROUP_DIMENSIONS:
        rows = report["dimensions"][dimension]["groups"]
        assert sum(row["corpus_count"] for row in rows) == 1516
        for system_id in DEFAULT_SYSTEM_IDS:
            assert (
                sum(
                    row["exposure_by_system"][system_id]["exposure_count"]
                    for row in rows
                )
                == 120
            )
    assert "규범적 공정성 평가가 아니며" in markdown
    assert "보호특성으로 주장하지 않습니다" in markdown
