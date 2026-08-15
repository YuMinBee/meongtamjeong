from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path

import pytest

from app.portal_candidate_selection_study import (
    BLIND_LABEL_CSV_COLUMNS,
    TRIAL_CSV_COLUMNS,
    aggregate_portal_trials,
    build_blind_label_rows,
    check_protocol_sources,
    freeze_protocol,
    load_blind_labels,
    load_trial_csvs,
    prepare_trial_sheets,
    render_result_markdown,
    validate_protocol,
    validate_template,
    write_csv_rows,
)
from scripts import portal_candidate_selection_study as cli


ROOT = Path(__file__).resolve().parents[1]
SOURCE_TEMPLATE = (
    ROOT / "docs" / "evaluation" / "portal_candidate_selection.protocol.v1.json"
)


def _frozen(tmp_path: Path) -> tuple[dict[str, object], Path, Path, Path]:
    template = json.loads(SOURCE_TEMPLATE.read_text(encoding="utf-8"))
    template_path = tmp_path / "docs" / "protocol.template.json"
    template_path.parent.mkdir(parents=True)
    template_path.write_text(json.dumps(template, ensure_ascii=False), encoding="utf-8")
    snapshot_path = tmp_path / "data" / "dog_metas.json"
    snapshot_path.parent.mkdir(parents=True)
    snapshot_path.write_text("[]\n", encoding="utf-8")
    index_path = tmp_path / "data" / "dog_faiss.index"
    index_path.write_bytes(b"synthetic-test-index")
    protocol = freeze_protocol(
        template,
        template_path=template_path,
        snapshot_path=snapshot_path,
        index_path=index_path,
        project_root=tmp_path,
        study_date="2026-08-01",
        source_commit="abcdef1234567890",
    )
    return protocol, template_path, snapshot_path, index_path


def _official_url(notice_id: str) -> str:
    return (
        "https://www.animal.go.kr/front/awtis/public/publicDtl.do"
        f"?desertionNo={notice_id}&menuNo=1000000055"
    )


def _synthetic_sheets(
    protocol: dict[str, object], participant_count: int = 4
) -> dict[str, list[dict[str, str]]]:
    sheets = prepare_trial_sheets(protocol, participant_count)
    candidate_seed = 411000000000000
    for participant_offset, rows in enumerate(sheets.values(), start=1):
        for trial_offset, row in enumerate(rows, start=1):
            row.update(
                {
                    "outcome": "completed",
                    "duration_ms": (
                        "100000" if row["system_id"] == "official_portal" else "50000"
                    ),
                    "failure_stage": "none",
                }
            )
            for candidate_offset in range(1, 4):
                notice_id = str(
                    candidate_seed
                    + participant_offset * 100
                    + trial_offset * 10
                    + candidate_offset
                )
                row[f"candidate_{candidate_offset}_notice_id"] = notice_id
                row[f"candidate_{candidate_offset}_url"] = _official_url(notice_id)
                row[f"candidate_{candidate_offset}_active_status"] = "active"
    return sheets


def _write_sheets(
    tmp_path: Path, sheets: dict[str, list[dict[str, str]]]
) -> list[Path]:
    paths: list[Path] = []
    for participant, rows in sheets.items():
        path = tmp_path / f"{participant}.csv"
        write_csv_rows(path, TRIAL_CSV_COLUMNS, rows)
        paths.append(path)
    return paths


def test_template_freeze_and_counterbalanced_sheets_are_result_free(
    tmp_path: Path,
) -> None:
    template = json.loads(SOURCE_TEMPLATE.read_text(encoding="utf-8"))
    validate_template(template)
    protocol, template_path, snapshot_path, index_path = _frozen(tmp_path)
    repeated = freeze_protocol(
        template,
        template_path=template_path,
        snapshot_path=snapshot_path,
        index_path=index_path,
        project_root=tmp_path,
        study_date="2026-08-01",
        source_commit="abcdef1234567890",
    )
    assert protocol == repeated
    validate_protocol(protocol)
    assert check_protocol_sources(protocol, tmp_path) == []
    assert protocol["human_results"] is None
    assert protocol["registration_status"] == "frozen_before_first_participant"
    assert protocol["protocol_id"].startswith("portal-e2e-appearance-v1-")

    sheets = prepare_trial_sheets(protocol, 4)
    assert list(sheets) == ["P01", "P02", "P03", "P04"]
    assert {rows[0]["sequence_id"] for rows in sheets.values()} == {
        "S1",
        "S2",
        "S3",
        "S4",
    }
    for rows in sheets.values():
        assert len(rows) == 4
        assert {row["system_id"] for row in rows} == {
            "official_portal",
            "meongtamjeong",
        }
        assert [row["system_id"] for row in rows].count("official_portal") == 2
        assert all(row["outcome"] == "" for row in rows)

    tampered = deepcopy(protocol)
    tampered["design"]["time_cap_seconds"] = 60  # type: ignore[index]
    with pytest.raises(ValueError, match="design hash mismatch"):
        validate_protocol(tampered)

    snapshot_path.write_text("[{}]\n", encoding="utf-8")
    assert "source_snapshot SHA-256 mismatch" in check_protocol_sources(
        protocol, tmp_path
    )


def test_trial_loader_separates_failures_and_requires_active_official_candidates(
    tmp_path: Path,
) -> None:
    protocol, _template_path, _snapshot_path, _index_path = _frozen(tmp_path)
    sheets = _synthetic_sheets(protocol)
    rows_by_participant = list(sheets.values())
    rows_by_participant[0][0].update(
        {
            "outcome": "network_failure",
            "duration_ms": "1200",
            "failure_stage": "search",
            "candidate_1_notice_id": "",
            "candidate_1_url": "",
            "candidate_1_active_status": "",
            "candidate_2_notice_id": "",
            "candidate_2_url": "",
            "candidate_2_active_status": "",
            "candidate_3_notice_id": "",
            "candidate_3_url": "",
            "candidate_3_active_status": "",
        }
    )
    rows_by_participant[1][0].update(
        {
            "outcome": "technical_failure",
            "duration_ms": "800",
            "failure_stage": "notice_detail",
            "candidate_1_notice_id": "",
            "candidate_1_url": "",
            "candidate_1_active_status": "",
            "candidate_2_notice_id": "",
            "candidate_2_url": "",
            "candidate_2_active_status": "",
            "candidate_3_notice_id": "",
            "candidate_3_url": "",
            "candidate_3_active_status": "",
        }
    )
    rows_by_participant[2][0].update(
        {
            "outcome": "time_cap",
            "duration_ms": "300000",
            "failure_stage": "search",
            "candidate_3_notice_id": "",
            "candidate_3_url": "",
            "candidate_3_active_status": "",
        }
    )
    rows_by_participant[3][0].update(
        {
            "outcome": "participant_stopped",
            "duration_ms": "10000",
            "failure_stage": "active_verification",
            "candidate_3_active_status": "inactive",
        }
    )
    paths = _write_sheets(tmp_path, sheets)
    loaded = load_trial_csvs(paths, protocol)
    assert {row["outcome"] for row in loaded} == {
        "completed",
        "network_failure",
        "technical_failure",
        "time_cap",
        "participant_stopped",
    }
    blind_rows = build_blind_label_rows(protocol, loaded)
    report = aggregate_portal_trials(
        protocol,
        loaded,
        {row["candidate_ref"]: "relevant" for row in blind_rows},
        protocol_sha256="a" * 64,
        trial_artifacts=[],
        label_artifact={"sha256": "b" * 64, "bytes": 1},
    )
    portal = report["conditions"]["official_portal"]
    meong = report["conditions"]["meongtamjeong"]
    assert portal["outcome_counts"]["network_failure"] == 1
    assert portal["outcome_counts"]["time_cap"] == 1
    assert meong["outcome_counts"]["technical_failure"] == 1
    assert meong["outcome_counts"]["participant_stopped"] == 1
    assert portal["completion_rate_all_scheduled"] == 0.75
    assert meong["completion_rate_all_scheduled"] == 0.75
    assert portal["capped_duration_ms_excluding_external_failures"]["count"] == 7
    assert meong["capped_duration_ms_excluding_external_failures"]["count"] == 7
    assert (
        report["paired_participant_comparison"][
            "excluded_due_to_network_or_technical_failure"
        ]
        == 2
    )

    invalid = _synthetic_sheets(protocol)
    first = next(iter(invalid.values()))[0]
    first["candidate_1_active_status"] = "unknown"
    invalid_paths = _write_sheets(tmp_path / "inactive", invalid)
    with pytest.raises(ValueError, match="three active candidates"):
        load_trial_csvs(invalid_paths, protocol)

    wrong_host = _synthetic_sheets(protocol)
    first = next(iter(wrong_host.values()))[0]
    first["candidate_1_url"] = (
        "https://example.com/publicDtl.do?desertionNo=411000000000111"
    )
    wrong_paths = _write_sheets(tmp_path / "wrong-host", wrong_host)
    with pytest.raises(ValueError, match="official animal.go.kr"):
        load_trial_csvs(wrong_paths, protocol)

    pii = _synthetic_sheets(protocol)
    pii_rows = next(iter(pii.values()))
    for row in pii_rows:
        row["participant_code"] = "real-name@example.com"
    pii_paths = _write_sheets(tmp_path / "pii", pii)
    with pytest.raises(ValueError, match="assigned P plus digits"):
        load_trial_csvs(pii_paths, protocol)


def test_blind_labels_hide_source_and_aggregate_descriptive_metrics(
    tmp_path: Path,
) -> None:
    protocol, _template_path, _snapshot_path, _index_path = _frozen(tmp_path)
    sheets = _synthetic_sheets(protocol)
    paths = _write_sheets(tmp_path, sheets)
    loaded = load_trial_csvs(paths, protocol)
    blind_rows = build_blind_label_rows(protocol, loaded)
    blind_text = json.dumps(blind_rows, ensure_ascii=False)
    assert "official_portal" not in blind_text
    assert "meongtamjeong" not in blind_text
    assert not any(code in blind_text for code in sheets)
    assert len(blind_rows) == 48

    for index, row in enumerate(blind_rows):
        row["relevance_label"] = (
            "unknown" if index == 0 else "not_relevant" if index == 1 else "relevant"
        )
    labels_path = tmp_path / "blind_labels.csv"
    write_csv_rows(labels_path, BLIND_LABEL_CSV_COLUMNS, blind_rows)
    expected_rows = build_blind_label_rows(protocol, loaded)
    labels = load_blind_labels(labels_path, expected_rows)
    report = aggregate_portal_trials(
        protocol,
        loaded,
        labels,
        protocol_sha256="a" * 64,
        trial_artifacts=[
            {"artifact_index": index, "sha256": "b" * 64, "bytes": 10}
            for index in range(1, 5)
        ],
        label_artifact={"sha256": "c" * 64, "bytes": 20},
    )
    assert report["sample"]["participants"] == 4
    assert report["sample"]["balanced_four_participant_schedule_achieved"] is True
    assert (
        report["conditions"]["official_portal"]["completion_rate_all_scheduled"] == 1.0
    )
    assert report["conditions"]["meongtamjeong"]["completion_rate_all_scheduled"] == 1.0
    assert report["paired_participant_comparison"]["meongtamjeong_faster"] == 4
    assert (
        report["paired_participant_comparison"][
            "meongtamjeong_minus_official_portal_ms"
        ]["median"]
        == -50000.0
    )
    serialized = json.dumps(report, ensure_ascii=False)
    assert '"P01"' not in serialized
    assert "inferential_statistics" in serialized
    markdown = render_result_markdown(report)
    assert "입양·문의·성격·건강 성과" in markdown
    assert "네트워크 실패" in markdown


def test_cli_freeze_prepare_blind_aggregate_and_check_round_trip(
    tmp_path: Path,
) -> None:
    protocol, template_path, snapshot_path, index_path = _frozen(tmp_path)
    protocol_path = tmp_path / "run" / "protocol.json"
    assert (
        cli.main(
            [
                "freeze",
                "--template",
                str(template_path),
                "--snapshot",
                str(snapshot_path),
                "--index",
                str(index_path),
                "--project-root",
                str(tmp_path),
                "--study-date",
                "2026-08-01",
                "--source-commit",
                "abcdef1234567890",
                "--output",
                str(protocol_path),
            ]
        )
        == 0
    )
    assert json.loads(protocol_path.read_text(encoding="utf-8")) == protocol
    trial_dir = tmp_path / "run" / "trials"
    assert (
        cli.main(
            [
                "prepare",
                "--protocol",
                str(protocol_path),
                "--participants",
                "4",
                "--output-dir",
                str(trial_dir),
            ]
        )
        == 0
    )
    filled = _synthetic_sheets(protocol)
    trial_paths: list[Path] = []
    for participant, rows in filled.items():
        path = trial_dir / f"{participant}.csv"
        write_csv_rows(path, TRIAL_CSV_COLUMNS, rows)
        trial_paths.append(path)
    labels_path = tmp_path / "run" / "blind_labels.csv"
    assert (
        cli.main(
            [
                "blind",
                "--protocol",
                str(protocol_path),
                "--trials",
                *(str(path) for path in trial_paths),
                "--output",
                str(labels_path),
            ]
        )
        == 0
    )
    with labels_path.open(encoding="utf-8-sig", newline="") as stream:
        label_rows = list(csv.DictReader(stream))
    for row in label_rows:
        row["relevance_label"] = "relevant"
    write_csv_rows(labels_path, BLIND_LABEL_CSV_COLUMNS, label_rows)
    result_path = tmp_path / "run" / "result.json"
    markdown_path = tmp_path / "run" / "result.md"
    common = [
        "--protocol",
        str(protocol_path),
        "--trials",
        *(str(path) for path in trial_paths),
        "--labels",
        str(labels_path),
    ]
    assert (
        cli.main(
            [
                "aggregate",
                *common,
                "--json-out",
                str(result_path),
                "--markdown-out",
                str(markdown_path),
            ]
        )
        == 0
    )
    assert (
        cli.main(
            [
                "check",
                *common,
                "--project-root",
                str(tmp_path),
                "--result",
                str(result_path),
                "--markdown",
                str(markdown_path),
            ]
        )
        == 0
    )
