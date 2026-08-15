from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from app.blind_relevance_evaluation import (
    LABEL_COLUMNS,
    aggregate_judgments,
    build_blind_task,
    load_judgments,
    render_result_markdown,
    render_task_html,
    write_label_template,
)
from scripts import aggregate_blind_relevance as aggregate_cli
from scripts import generate_blind_relevance_task as generate_cli


def retrieval_report() -> dict[str, object]:
    query_1 = {"id": "q1", "query": "갈색 소형견", "top_ids": ["d1", "d2"]}
    query_2 = {"id": "q2", "query": "흰색 성견", "top_ids": ["d3", "d1"]}
    return {
        "schema_version": "retrieval-evaluation.v2",
        "reference_date": "2026-07-26",
        "evaluation_contract": {
            "baseline_system": "baseline-secret",
            "headline_system": "candidate-secret",
        },
        "systems": {
            "baseline-secret": {
                "role": "baseline",
                "queries": [query_1, query_2],
            },
            "candidate-secret": {
                "role": "headline",
                "queries": [
                    {**query_1, "top_ids": ["d2", "d3"]},
                    {**query_2, "top_ids": ["d1", "d2"]},
                ],
            },
        },
    }


def notice_metas(project_root: Path) -> list[dict[str, object]]:
    crop = project_root / "images" / "d1.jpg"
    crop.parent.mkdir(parents=True, exist_ok=True)
    crop.write_bytes(b"not-a-real-jpeg")
    return [
        {
            "desertionNo": "d1",
            "type": "image",
            "breed_name": "믹스견",
            "color": "갈색",
            "age": "2024(년생)",
            "weight": "5(Kg)",
            "org_name": "서울특별시",
            "care_tel": "010-1111-2222",
            "happen_place": "민감한 상세 발견 장소",
            "image_url": "https://openapi.animal.go.kr/d1.jpg",
            "image_attrs": {"crop_path": "images/d1.jpg"},
        },
        {
            "desertionNo": "d2",
            "type": "image",
            "breed_name": "진도견",
            "image_url": "https://openapi.animal.go.kr/d2.jpg",
        },
        {
            "desertionNo": "d3",
            "type": "image",
            "breed_name": "믹스견",
            "image_url": "https://untrusted.example/d3.jpg",
            "image_urls": ["javascript:alert(1)"],
        },
    ]


def make_task(
    tmp_path: Path,
    *,
    seed: int = 7,
    image_policy: str = "prefer-local",
) -> tuple[dict[str, object], dict[str, object]]:
    report_path = tmp_path / "retrieval.json"
    report_path.write_text(
        json.dumps(retrieval_report(), ensure_ascii=False),
        encoding="utf-8",
    )
    return build_blind_task(
        retrieval_report(),
        notice_metas(tmp_path),
        report_path=report_path,
        metas_path=None,
        systems=None,
        depth=2,
        seed=seed,
        html_output=tmp_path / "out" / "task.html",
        project_root=tmp_path,
        image_policy=image_policy,
    )


def completed_judgments(
    key: dict[str, object],
    reviewer_grades: dict[str, dict[str, int]],
) -> dict[str, dict[tuple[str, str], int]]:
    output: dict[str, dict[tuple[str, str], int]] = {}
    for reviewer, grades_by_notice in reviewer_grades.items():
        output[reviewer] = {}
        for query in key["queries"]:  # type: ignore[index]
            query_id = query["query_id"]
            for candidate in query["candidates"]:
                output[reviewer][(query_id, candidate["candidate_code"])] = (
                    grades_by_notice[candidate["notice_id"]]
                )
    return output


def test_task_is_deterministic_and_visible_artifact_is_blinded(tmp_path: Path) -> None:
    task, key = make_task(tmp_path)
    repeated_task, repeated_key = make_task(tmp_path)

    assert task == repeated_task
    assert key == repeated_key
    assert key["source_report"]["path"] == "retrieval.json"  # type: ignore[index]
    visible = json.dumps(task, ensure_ascii=False)
    assert "baseline-secret" not in visible
    assert "candidate-secret" not in visible
    assert "system_ranks" not in visible
    assert '"d1"' not in visible
    assert "010-1111-2222" not in visible
    assert "민감한 상세 발견 장소" not in visible

    key_text = json.dumps(key, ensure_ascii=False)
    assert "baseline-secret" in key_text
    assert "candidate-secret" in key_text
    assert "system_ranks" in key_text


def test_task_prefers_local_images_and_rejects_unsafe_remote_urls(
    tmp_path: Path,
) -> None:
    task, _key = make_task(tmp_path)
    candidates = {
        item["candidate_code"]: item
        for query in task["queries"]  # type: ignore[index]
        for item in query["candidates"]
    }
    all_sources = [candidate["image_src"] for candidate in candidates.values()]

    assert any(source.endswith("images/d1.jpg") for source in all_sources)
    assert "https://openapi.animal.go.kr/d2.jpg" in all_sources
    assert "javascript:alert(1)" not in all_sources
    assert "https://untrusted.example/d3.jpg" not in all_sources
    assert "" in all_sources

    local_task, _key = make_task(tmp_path, image_policy="local-only")
    assert local_task["task_id"] != task["task_id"]
    local_html = render_task_html(local_task, allow_remote_images=False)
    assert "connect-src 'none'" in local_html
    assert (
        "http:"
        not in local_html.split("Content-Security-Policy", 1)[1].split('">', 1)[0]
    )
    assert "baseline-secret" not in local_html


def test_label_loader_requires_complete_unique_pseudonymous_responses(
    tmp_path: Path,
) -> None:
    task, key = make_task(tmp_path)
    template = tmp_path / "labels.csv"
    write_label_template(task, template)

    rows = list(csv.DictReader(template.open(encoding="utf-8-sig", newline="")))
    for row in rows:
        row["reviewer_id"] = "reviewer-01"
        row["relevance"] = "1"
    with template.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    judgments, sources = load_judgments([template], key)
    assert len(judgments["reviewer-01"]) == len(rows)
    assert len(sources) == len(rows)

    incomplete = tmp_path / "incomplete.csv"
    with incomplete.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows[:-1])
    with pytest.raises(ValueError, match="complete the full task"):
        load_judgments([incomplete], key)

    rows[0]["reviewer_id"] = "real name"
    invalid = tmp_path / "invalid.csv"
    with invalid.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="pseudonymous"):
        load_judgments([invalid], key)


def test_aggregation_reports_consensus_ndcg_pairwise_wins_and_sample_size(
    tmp_path: Path,
) -> None:
    _task, key = make_task(tmp_path)
    judgments = completed_judgments(
        key,
        {
            "r1": {"d1": 2, "d2": 1, "d3": 0},
            "r2": {"d1": 2, "d2": 1, "d3": 0},
        },
    )

    report = aggregate_judgments(key, judgments)
    ideal = 3.0 + 1.0 / math.log2(3)
    baseline_q1 = report["systems"]["baseline-secret"]["queries"][0]["nDCG@2"]
    candidate_q1 = report["systems"]["candidate-secret"]["queries"][0]["nDCG@2"]

    assert baseline_q1 == 1.0
    assert candidate_q1 == pytest.approx(1.0 / ideal, abs=1e-6)
    assert report["sample"] == {
        "reviewers": 2,
        "reviewer_ids": ["r1", "r2"],
        "queries": 2,
        "unique_query_candidates": 6,
        "judgments": 12,
        "grade_distribution": {"0": 4, "1": 4, "2": 4},
    }
    assert (
        report["task_source"]["source_report"]["sha256"]
        == key["source_report"]["sha256"]
    )
    assert report["protocol"]["depth"] == 2
    pair = report["pairwise_preference"][0]
    assert pair["system_a_wins"] == 1
    assert pair["system_b_wins"] == 1
    assert pair["ties"] == 0
    assert pair["system_a_preference_rate_excluding_ties"] == 0.5
    markdown = render_result_markdown(report)
    assert "not a direct list preference vote" not in markdown
    assert "Blind human relevance evaluation" in markdown
    assert "`baseline-secret`" in markdown


def test_generator_and_aggregator_cli_round_trip(tmp_path: Path) -> None:
    assert generate_cli.build_parser().parse_args([]).image_policy == "local-only"
    report_path = tmp_path / "retrieval.json"
    metas_path = tmp_path / "metas.json"
    output_dir = tmp_path / "task"
    report_path.write_text(
        json.dumps(retrieval_report(), ensure_ascii=False),
        encoding="utf-8",
    )
    metas_path.write_text(
        json.dumps(notice_metas(tmp_path), ensure_ascii=False),
        encoding="utf-8",
    )

    assert (
        generate_cli.main(
            [
                "--report",
                str(report_path),
                "--metas",
                str(metas_path),
                "--output-dir",
                str(output_dir),
                "--depth",
                "2",
                "--image-policy",
                "none",
            ]
        )
        == 0
    )
    assert (output_dir / "task.html").is_file()
    assert (output_dir / "task_key.json").is_file()
    assert (output_dir / "labels_template.csv").is_file()
    assert "baseline-secret" not in (output_dir / "task.html").read_text(
        encoding="utf-8"
    )

    key = json.loads((output_dir / "task_key.json").read_text(encoding="utf-8"))
    labels = output_dir / "reviewer.csv"
    template_rows = list(
        csv.DictReader(
            (output_dir / "labels_template.csv").open(
                encoding="utf-8-sig",
                newline="",
            )
        )
    )
    for row in template_rows:
        row["reviewer_id"] = "r1"
        row["relevance"] = "2"
    with labels.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows(template_rows)

    json_out = tmp_path / "result.json"
    markdown_out = tmp_path / "result.md"
    assert (
        aggregate_cli.main(
            [
                "--key",
                str(output_dir / "task_key.json"),
                "--labels",
                str(labels),
                "--json-out",
                str(json_out),
                "--markdown-out",
                str(markdown_out),
            ]
        )
        == 0
    )
    result = json.loads(json_out.read_text(encoding="utf-8"))
    assert result["task_id"] == key["task_id"]
    assert result["sample"]["reviewers"] == 1
    assert result["label_artifacts"][0]["file"] == "reviewer.csv"
    assert len(result["label_artifacts"][0]["sha256"]) == 64
    assert markdown_out.is_file()
