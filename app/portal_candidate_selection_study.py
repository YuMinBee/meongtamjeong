"""Pre-registered end-to-end pilot comparing the public portal and Meongtamjeong.

This module is deliberately separate from ``candidate_selection_study``. The
existing study compares frozen retrieval rankings inside one local UI; this
study measures a manual, live end-to-end task across two products. It never
contacts either product or creates human results on its own.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import date
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence
from urllib.parse import parse_qs, urlparse

from app.blind_relevance_evaluation import clean_text, sha256_file
from app.candidate_selection_study import canonical_sha256


TEMPLATE_SCHEMA_VERSION = "portal-candidate-selection-template.v1"
PROTOCOL_SCHEMA_VERSION = "portal-candidate-selection-protocol.v1"
RESULT_SCHEMA_VERSION = "portal-candidate-selection-result.v1"

TRIAL_CSV_COLUMNS = (
    "protocol_id",
    "participant_code",
    "sequence_id",
    "trial_index",
    "task_id",
    "system_id",
    "outcome",
    "duration_ms",
    "failure_stage",
    "candidate_1_notice_id",
    "candidate_1_url",
    "candidate_1_active_status",
    "candidate_2_notice_id",
    "candidate_2_url",
    "candidate_2_active_status",
    "candidate_3_notice_id",
    "candidate_3_url",
    "candidate_3_active_status",
)

BLIND_LABEL_CSV_COLUMNS = (
    "protocol_id",
    "candidate_ref",
    "task_id",
    "task_prompt",
    "notice_id",
    "canonical_notice_url",
    "relevance_label",
)

SYSTEM_IDS = ("official_portal", "meongtamjeong")
OUTCOMES = (
    "completed",
    "time_cap",
    "participant_stopped",
    "network_failure",
    "technical_failure",
)
FAILURE_STAGES = (
    "none",
    "system_launch",
    "task_setup",
    "search",
    "notice_detail",
    "active_verification",
)
ACTIVE_STATUSES = ("active", "inactive", "unknown")
RELEVANCE_LABELS = ("relevant", "not_relevant", "unknown")

_PARTICIPANT_RE = re.compile(r"^P[0-9]{2,4}$")
_NOTICE_ID_RE = re.compile(r"^[0-9]{15}$")
_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")


def _portable_path(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"source must stay inside project root: {path}") from exc


def _design(value: Mapping[str, Any]) -> Mapping[str, Any]:
    design = value.get("design")
    if not isinstance(design, Mapping):
        raise ValueError("protocol needs a design object")
    return design


def _task_map(design: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    tasks = design.get("tasks")
    if not isinstance(tasks, Sequence) or isinstance(tasks, (str, bytes)):
        raise ValueError("design.tasks must be an array")
    output: dict[str, Mapping[str, Any]] = {}
    for task in tasks:
        if not isinstance(task, Mapping):
            raise ValueError("each task must be an object")
        task_id = clean_text(task.get("task_id"))
        if not task_id or task_id in output:
            raise ValueError("task IDs must be non-empty and unique")
        if not clean_text(task.get("prompt")):
            raise ValueError(f"task {task_id!r} needs a prompt")
        criteria = task.get("criteria")
        if not isinstance(criteria, Mapping) or criteria.get("active_only") is not True:
            raise ValueError(f"task {task_id!r} must set criteria.active_only=true")
        output[task_id] = task
    if len(output) != 4:
        raise ValueError("the external portal pilot requires exactly four tasks")
    return output


def _sequence_map(design: Mapping[str, Any]) -> dict[str, list[Mapping[str, Any]]]:
    raw_sequences = design.get("sequences")
    if not isinstance(raw_sequences, Mapping):
        raise ValueError("design.sequences must be an object")
    expected_sequence_ids = {"S1", "S2", "S3", "S4"}
    if set(raw_sequences) != expected_sequence_ids:
        raise ValueError("sequences must be exactly S1-S4")
    tasks = _task_map(design)
    output: dict[str, list[Mapping[str, Any]]] = {}
    task_system_counts: Counter[tuple[str, str]] = Counter()
    for sequence_id in sorted(expected_sequence_ids):
        trials = raw_sequences[sequence_id]
        if not isinstance(trials, Sequence) or isinstance(trials, (str, bytes)):
            raise ValueError(f"sequence {sequence_id} must be an array")
        if len(trials) != 4 or not all(isinstance(row, Mapping) for row in trials):
            raise ValueError(f"sequence {sequence_id} needs four trials")
        normalized = [row for row in trials if isinstance(row, Mapping)]
        if [row.get("trial_index") for row in normalized] != [1, 2, 3, 4]:
            raise ValueError(f"sequence {sequence_id} indexes must be 1-4")
        task_ids = [clean_text(row.get("task_id")) for row in normalized]
        systems = [clean_text(row.get("system_id")) for row in normalized]
        if set(task_ids) != set(tasks):
            raise ValueError(f"sequence {sequence_id} must use each task once")
        if any(system not in SYSTEM_IDS for system in systems):
            raise ValueError(f"sequence {sequence_id} has an unknown system")
        if any(systems.count(system) != 2 for system in SYSTEM_IDS):
            raise ValueError(f"sequence {sequence_id} needs two trials per system")
        for task_id, system_id in zip(task_ids, systems, strict=True):
            task_system_counts[(task_id, system_id)] += 1
        output[sequence_id] = normalized
    if any(
        task_system_counts[(task_id, system_id)] != 2
        for task_id in tasks
        for system_id in SYSTEM_IDS
    ):
        raise ValueError("the four sequences must balance every task across systems")
    return output


def validate_template(template: Mapping[str, Any]) -> None:
    """Validate the immutable, result-free pre-registration template."""

    if clean_text(template.get("schema_version")) != TEMPLATE_SCHEMA_VERSION:
        raise ValueError("unsupported portal pilot template schema")
    if clean_text(template.get("registration_status")) != "template_no_human_results":
        raise ValueError("template must explicitly state that it has no human results")
    if any(key in template for key in ("results", "observed_metrics", "participants")):
        raise ValueError("pre-registration template must not contain human results")
    base_id = clean_text(template.get("protocol_id_base"))
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{7,80}", base_id):
        raise ValueError("protocol_id_base is invalid")
    design = _design(template)
    if int(design.get("participant_min", 0)) != 3:
        raise ValueError("participant_min must remain 3")
    if int(design.get("participant_target", 0)) != 4:
        raise ValueError("participant_target must remain 4")
    if int(design.get("participant_max", 0)) != 5:
        raise ValueError("participant_max must remain 5")
    if int(design.get("selection_count", 0)) != 3:
        raise ValueError("selection_count must remain 3")
    cap = int(design.get("time_cap_seconds", 0))
    if not 60 <= cap <= 900:
        raise ValueError("time_cap_seconds must be between 60 and 900")
    systems = design.get("systems")
    if not isinstance(systems, Sequence) or isinstance(systems, (str, bytes)):
        raise ValueError("design.systems must be an array")
    system_ids = {
        clean_text(item.get("system_id"))
        for item in systems
        if isinstance(item, Mapping)
    }
    if system_ids != set(SYSTEM_IDS):
        raise ValueError("systems must be official_portal and meongtamjeong")
    portal = next(
        item
        for item in systems
        if isinstance(item, Mapping)
        and clean_text(item.get("system_id")) == "official_portal"
    )
    parsed = urlparse(clean_text(portal.get("entry_url")))
    if parsed.scheme != "https" or parsed.hostname not in {
        "animal.go.kr",
        "www.animal.go.kr",
    }:
        raise ValueError("official_portal entry_url must use the official HTTPS host")
    _task_map(design)
    _sequence_map(design)
    if design.get("active_status_source") != "official_notice_detail_at_selection":
        raise ValueError("active status must be checked on the official notice detail")
    if design.get("blind_relevance_source_hidden") is not True:
        raise ValueError("blind relevance must hide the selecting system")


def freeze_protocol(
    template: Mapping[str, Any],
    *,
    template_path: Path,
    snapshot_path: Path,
    index_path: Path,
    project_root: Path,
    study_date: str,
    source_commit: str,
) -> dict[str, Any]:
    """Freeze mutable provenance before the first participant starts."""

    validate_template(template)
    try:
        date.fromisoformat(study_date)
    except ValueError as exc:
        raise ValueError("study_date must be ISO YYYY-MM-DD") from exc
    if not _COMMIT_RE.fullmatch(source_commit):
        raise ValueError("source_commit must be a 7-40 character Git SHA")
    try:
        source_template = json.loads(template_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"template file cannot be loaded: {exc}") from exc
    if source_template != template:
        raise ValueError("template object does not match template_path content")
    if not snapshot_path.is_file():
        raise ValueError("snapshot_path must be an existing file")
    if not index_path.is_file():
        raise ValueError("index_path must be an existing file")

    design = deepcopy(dict(_design(template)))
    material = {
        "template_sha256": sha256_file(template_path),
        "design_sha256": canonical_sha256(design),
        "snapshot_sha256": sha256_file(snapshot_path),
        "snapshot_bytes": snapshot_path.stat().st_size,
        "index_sha256": sha256_file(index_path),
        "index_bytes": index_path.stat().st_size,
        "study_date": study_date,
        "source_commit": source_commit.lower(),
    }
    protocol_id = (
        clean_text(template.get("protocol_id_base"))
        + "-"
        + canonical_sha256(material)[:16]
    )
    protocol = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "registration_status": "frozen_before_first_participant",
        "protocol_id": protocol_id,
        "study_date": study_date,
        "source_commit": source_commit.lower(),
        "source_template": {
            "path": _portable_path(template_path, project_root),
            "sha256": material["template_sha256"],
        },
        "source_snapshot": {
            "path": _portable_path(snapshot_path, project_root),
            "sha256": material["snapshot_sha256"],
            "bytes": material["snapshot_bytes"],
        },
        "source_index": {
            "path": _portable_path(index_path, project_root),
            "sha256": material["index_sha256"],
            "bytes": material["index_bytes"],
        },
        "protocol_id_base": clean_text(template.get("protocol_id_base")),
        "protocol_id_material": material,
        "design": design,
        "human_results": None,
    }
    validate_protocol(protocol)
    return protocol


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    """Validate a frozen protocol and its tamper-evident identifier."""

    if clean_text(protocol.get("schema_version")) != PROTOCOL_SCHEMA_VERSION:
        raise ValueError("unsupported frozen portal pilot protocol schema")
    if (
        clean_text(protocol.get("registration_status"))
        != "frozen_before_first_participant"
    ):
        raise ValueError("protocol was not frozen before participation")
    if protocol.get("human_results") is not None:
        raise ValueError("frozen protocol must not embed human results")
    design = _design(protocol)
    template_view = {
        "schema_version": TEMPLATE_SCHEMA_VERSION,
        "registration_status": "template_no_human_results",
        "protocol_id_base": protocol.get("protocol_id_base"),
        "design": design,
    }
    validate_template(template_view)
    material = protocol.get("protocol_id_material")
    if not isinstance(material, Mapping):
        raise ValueError("protocol_id_material is missing")
    if clean_text(material.get("design_sha256")) != canonical_sha256(design):
        raise ValueError("frozen design hash mismatch")
    if clean_text(material.get("study_date")) != clean_text(protocol.get("study_date")):
        raise ValueError("study date does not match frozen material")
    if clean_text(material.get("source_commit")) != clean_text(
        protocol.get("source_commit")
    ):
        raise ValueError("source commit does not match frozen material")
    for source_name, prefix in (
        ("source_template", "template"),
        ("source_snapshot", "snapshot"),
        ("source_index", "index"),
    ):
        source = protocol.get(source_name)
        if not isinstance(source, Mapping):
            raise ValueError(f"{source_name} metadata is missing")
        if clean_text(source.get("sha256")) != clean_text(
            material.get(f"{prefix}_sha256")
        ):
            raise ValueError(f"{source_name} hash does not match frozen material")
        material_bytes = material.get(f"{prefix}_bytes")
        if material_bytes is not None and int(source.get("bytes", -1)) != int(
            material_bytes
        ):
            raise ValueError(f"{source_name} size does not match frozen material")
    expected_id = (
        clean_text(protocol.get("protocol_id_base"))
        + "-"
        + canonical_sha256(dict(material))[:16]
    )
    if clean_text(protocol.get("protocol_id")) != expected_id:
        raise ValueError("protocol_id does not match frozen material")


def check_protocol_sources(
    protocol: Mapping[str, Any], project_root: Path
) -> list[str]:
    """Check that the frozen template and search snapshot still match disk."""

    validate_protocol(protocol)
    root = project_root.resolve()
    failures: list[str] = []
    for source_name in ("source_template", "source_snapshot", "source_index"):
        source = protocol.get(source_name)
        if not isinstance(source, Mapping):
            failures.append(f"{source_name} metadata is missing")
            continue
        portable = clean_text(source.get("path"))
        candidate = (root / portable).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            failures.append(f"{source_name} path escapes project root")
            continue
        if not candidate.is_file():
            failures.append(f"{source_name} file is missing: {portable}")
            continue
        if sha256_file(candidate) != clean_text(source.get("sha256")):
            failures.append(f"{source_name} SHA-256 mismatch")
        expected_bytes = source.get("bytes")
        if expected_bytes is not None and candidate.stat().st_size != int(
            expected_bytes
        ):
            failures.append(f"{source_name} byte count mismatch")
    return failures


def prepare_trial_sheets(
    protocol: Mapping[str, Any], participant_count: int
) -> dict[str, list[dict[str, str]]]:
    """Create pre-filled, PII-free CSV rows for three to five participants."""

    validate_protocol(protocol)
    if not 3 <= participant_count <= 5:
        raise ValueError("participant_count must be between 3 and 5")
    sequences = _sequence_map(_design(protocol))
    sequence_order = ("S1", "S2", "S3", "S4", "S1")
    output: dict[str, list[dict[str, str]]] = {}
    for offset in range(participant_count):
        participant = f"P{offset + 1:02d}"
        sequence_id = sequence_order[offset]
        rows: list[dict[str, str]] = []
        for trial in sequences[sequence_id]:
            row = {column: "" for column in TRIAL_CSV_COLUMNS}
            row.update(
                {
                    "protocol_id": clean_text(protocol.get("protocol_id")),
                    "participant_code": participant,
                    "sequence_id": sequence_id,
                    "trial_index": str(trial["trial_index"]),
                    "task_id": clean_text(trial.get("task_id")),
                    "system_id": clean_text(trial.get("system_id")),
                    "failure_stage": "none",
                }
            )
            rows.append(row)
        output[participant] = rows
    return output


def write_csv_rows(
    path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _parse_duration(value: str, cap_ms: int, location: str) -> int:
    if not re.fullmatch(r"[0-9]+", value):
        raise ValueError(f"{location}: duration_ms must be an integer")
    duration = int(value)
    if not 0 <= duration <= cap_ms:
        raise ValueError(f"{location}: duration_ms must be between 0 and {cap_ms}")
    return duration


def canonical_notice_url(notice_id: str, raw_url: str) -> str:
    """Validate an official detail URL and remove tracking differences."""

    if not _NOTICE_ID_RE.fullmatch(notice_id):
        raise ValueError("notice ID must contain exactly 15 digits")
    parsed = urlparse(raw_url)
    if parsed.scheme != "https" or parsed.hostname not in {
        "animal.go.kr",
        "www.animal.go.kr",
    }:
        raise ValueError("candidate URL must use the official animal.go.kr HTTPS host")
    if not parsed.path.endswith("/front/awtis/public/publicDtl.do"):
        raise ValueError("candidate URL must be an official public notice detail page")
    if parse_qs(parsed.query).get("desertionNo") != [notice_id]:
        raise ValueError("candidate URL desertionNo must equal candidate notice ID")
    return (
        "https://www.animal.go.kr/front/awtis/public/publicDtl.do"
        f"?desertionNo={notice_id}&menuNo=1000000055"
    )


def _candidate_slots(raw: Mapping[str, Any], location: str) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    for index in range(1, 4):
        notice_id = clean_text(raw.get(f"candidate_{index}_notice_id"))
        url = clean_text(raw.get(f"candidate_{index}_url"))
        status = clean_text(raw.get(f"candidate_{index}_active_status"))
        present = [bool(notice_id), bool(url), bool(status)]
        if any(present) and not all(present):
            raise ValueError(f"{location}: candidate {index} fields must be all set")
        if not any(present):
            continue
        if status not in ACTIVE_STATUSES:
            raise ValueError(f"{location}: candidate {index} active status is invalid")
        try:
            canonical_url = canonical_notice_url(notice_id, url)
        except ValueError as exc:
            raise ValueError(f"{location}: candidate {index}: {exc}") from exc
        candidates.append(
            {
                "notice_id": notice_id,
                "canonical_notice_url": canonical_url,
                "active_status": status,
            }
        )
    notice_ids = [row["notice_id"] for row in candidates]
    if len(notice_ids) != len(set(notice_ids)):
        raise ValueError(f"{location}: candidate notice IDs must be unique")
    return candidates


def load_trial_csvs(
    paths: Sequence[Path], protocol: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Load complete manual trial sheets and enforce the frozen schedule."""

    validate_protocol(protocol)
    if not paths:
        raise ValueError("at least one trial CSV is required")
    if not 3 <= len(paths) <= 5:
        raise ValueError("the pilot requires one CSV for each of 3-5 participants")
    design = _design(protocol)
    sequences = _sequence_map(design)
    tasks = _task_map(design)
    cap_ms = int(design["time_cap_seconds"]) * 1000
    output: list[dict[str, Any]] = []
    seen_participants: set[str] = set()
    sequence_counts: Counter[str] = Counter()

    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != TRIAL_CSV_COLUMNS:
                raise ValueError(f"{path}: trial CSV columns do not match the template")
            source_rows = list(reader)
        if len(source_rows) != 4:
            raise ValueError(f"{path}: each participant needs exactly four rows")
        participants = {clean_text(row.get("participant_code")) for row in source_rows}
        sequence_ids = {clean_text(row.get("sequence_id")) for row in source_rows}
        if len(participants) != 1 or len(sequence_ids) != 1:
            raise ValueError(f"{path}: participant and sequence must be consistent")
        participant = next(iter(participants))
        sequence_id = next(iter(sequence_ids))
        if not _PARTICIPANT_RE.fullmatch(participant):
            raise ValueError(
                f"{path}: participant must use an assigned P plus digits code"
            )
        if participant in seen_participants:
            raise ValueError(f"{path}: duplicate participant code")
        if sequence_id not in sequences:
            raise ValueError(f"{path}: unknown sequence_id")
        seen_participants.add(participant)
        sequence_counts[sequence_id] += 1
        expected = {int(row["trial_index"]): row for row in sequences[sequence_id]}
        seen_indexes: set[int] = set()
        for line_number, raw in enumerate(source_rows, start=2):
            location = f"{path}:{line_number}"
            if clean_text(raw.get("protocol_id")) != clean_text(
                protocol.get("protocol_id")
            ):
                raise ValueError(f"{location}: protocol_id does not match")
            index_raw = clean_text(raw.get("trial_index"))
            if not re.fullmatch(r"[1-4]", index_raw):
                raise ValueError(f"{location}: trial_index must be 1-4")
            trial_index = int(index_raw)
            if trial_index in seen_indexes:
                raise ValueError(f"{location}: duplicate trial_index")
            seen_indexes.add(trial_index)
            scheduled = expected[trial_index]
            for field in ("task_id", "system_id"):
                if clean_text(raw.get(field)) != clean_text(scheduled.get(field)):
                    raise ValueError(f"{location}: {field} does not match schedule")
            task_id = clean_text(scheduled.get("task_id"))
            system_id = clean_text(scheduled.get("system_id"))
            outcome = clean_text(raw.get("outcome"))
            if outcome not in OUTCOMES:
                raise ValueError(f"{location}: outcome is invalid")
            failure_stage = clean_text(raw.get("failure_stage"))
            if failure_stage not in FAILURE_STAGES:
                raise ValueError(f"{location}: failure_stage is invalid")
            duration = _parse_duration(
                clean_text(raw.get("duration_ms")), cap_ms, location
            )
            candidates = _candidate_slots(raw, location)
            is_three_active = len(candidates) == 3 and all(
                row["active_status"] == "active" for row in candidates
            )
            if outcome == "completed":
                if duration == 0 or failure_stage != "none" or not is_three_active:
                    raise ValueError(
                        f"{location}: completed requires positive time, no failure, "
                        "and three active candidates"
                    )
            else:
                if failure_stage == "none":
                    raise ValueError(
                        f"{location}: incomplete outcome needs a controlled failure_stage"
                    )
                if is_three_active:
                    raise ValueError(
                        f"{location}: three active candidates must be marked completed"
                    )
            if outcome == "time_cap" and duration != cap_ms:
                raise ValueError(f"{location}: time_cap duration must equal the cap")

            output.append(
                {
                    "participant_code": participant,
                    "sequence_id": sequence_id,
                    "trial_index": trial_index,
                    "task_id": task_id,
                    "task_prompt": clean_text(tasks[task_id].get("prompt")),
                    "system_id": system_id,
                    "outcome": outcome,
                    "duration_ms": duration,
                    "failure_stage": failure_stage,
                    "candidates": candidates,
                }
            )
        if seen_indexes != {1, 2, 3, 4}:
            raise ValueError(f"{path}: trial indexes must be exactly 1-4")

    participant_count = len(seen_participants)
    counts = sorted(sequence_counts.values(), reverse=True)
    if participant_count == 3 and (len(sequence_counts) != 3 or counts != [1, 1, 1]):
        raise ValueError("three participants must use three distinct sequences")
    if participant_count == 4 and any(
        sequence_counts.get(key, 0) != 1 for key in sequences
    ):
        raise ValueError("four participants must use S1-S4 exactly once")
    if participant_count == 5 and sorted(sequence_counts.values()) != [1, 1, 1, 2]:
        raise ValueError("five participants must use every sequence plus one repeat")
    return output


def _candidate_ref(protocol_id: str, task_id: str, notice_id: str) -> str:
    digest = hashlib.sha256(
        f"{protocol_id}\0{task_id}\0{notice_id}".encode("utf-8")
    ).hexdigest()
    return "R-" + digest[:12].upper()


def build_blind_label_rows(
    protocol: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, str]]:
    """Create a system- and participant-blind sheet for final candidates."""

    validate_protocol(protocol)
    protocol_id = clean_text(protocol.get("protocol_id"))
    unique: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        if clean_text(row.get("outcome")) != "completed":
            continue
        task_id = clean_text(row.get("task_id"))
        prompt = clean_text(row.get("task_prompt"))
        for candidate in row.get("candidates") or []:
            if not isinstance(candidate, Mapping):
                continue
            notice_id = clean_text(candidate.get("notice_id"))
            key = (task_id, notice_id)
            unique[key] = {
                "protocol_id": protocol_id,
                "candidate_ref": _candidate_ref(protocol_id, task_id, notice_id),
                "task_id": task_id,
                "task_prompt": prompt,
                "notice_id": notice_id,
                "canonical_notice_url": clean_text(
                    candidate.get("canonical_notice_url")
                ),
                "relevance_label": "",
            }
    if not unique:
        raise ValueError(
            "no completed final candidates are available for blind labeling"
        )
    seed = int(_design(protocol).get("blind_label_seed", 0))
    return sorted(
        unique.values(),
        key=lambda row: hashlib.sha256(
            f"{seed}\0{row['candidate_ref']}".encode("utf-8")
        ).hexdigest(),
    )


def load_blind_labels(
    path: Path,
    expected_rows: Sequence[Mapping[str, str]],
) -> dict[str, str]:
    """Validate completed blind labels without exposing selection source."""

    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != BLIND_LABEL_CSV_COLUMNS:
            raise ValueError(f"{path}: blind label columns do not match the template")
        raw_rows = list(reader)
    expected = {clean_text(row.get("candidate_ref")): row for row in expected_rows}
    if len(raw_rows) != len(expected):
        raise ValueError(
            f"{path}: blind label row count does not match final candidates"
        )
    labels: dict[str, str] = {}
    fixed_fields = BLIND_LABEL_CSV_COLUMNS[:-1]
    for line_number, raw in enumerate(raw_rows, start=2):
        location = f"{path}:{line_number}"
        candidate_ref = clean_text(raw.get("candidate_ref"))
        if candidate_ref not in expected or candidate_ref in labels:
            raise ValueError(f"{location}: unknown or duplicate candidate_ref")
        for field in fixed_fields:
            if clean_text(raw.get(field)) != clean_text(
                expected[candidate_ref].get(field)
            ):
                raise ValueError(f"{location}: {field} was changed after blinding")
        label = clean_text(raw.get("relevance_label"))
        if label not in RELEVANCE_LABELS:
            raise ValueError(f"{location}: relevance_label is invalid")
        labels[candidate_ref] = label
    if set(labels) != set(expected):
        raise ValueError(f"{path}: blind labels are incomplete")
    return labels


def _round(value: float) -> float:
    return round(float(value), 6)


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _numeric_summary(values: Sequence[int | float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
        }
    numeric = [float(value) for value in values]
    return {
        "count": len(numeric),
        "min": _round(min(numeric)),
        "p25": _round(_percentile(numeric, 0.25)),
        "median": _round(median(numeric)),
        "p75": _round(_percentile(numeric, 0.75)),
        "max": _round(max(numeric)),
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return _round(numerator / denominator) if denominator else None


def aggregate_portal_trials(
    protocol: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    labels: Mapping[str, str],
    *,
    protocol_sha256: str,
    trial_artifacts: Sequence[Mapping[str, Any]],
    label_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Aggregate descriptive end-to-end metrics; never infer adoption outcomes."""

    validate_protocol(protocol)
    participants = sorted({clean_text(row.get("participant_code")) for row in rows})
    if not 3 <= len(participants) <= 5 or len(rows) != len(participants) * 4:
        raise ValueError("validated rows must contain four trials for 3-5 participants")
    cap_ms = int(_design(protocol)["time_cap_seconds"]) * 1000
    protocol_id = clean_text(protocol.get("protocol_id"))

    def label_for(task_id: str, notice_id: str) -> str:
        candidate_ref = _candidate_ref(protocol_id, task_id, notice_id)
        if candidate_ref not in labels:
            raise ValueError(f"missing blind label for {candidate_ref}")
        return labels[candidate_ref]

    by_system: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_participant_system: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    by_task_system: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    sequence_counts: Counter[str] = Counter()
    for row in rows:
        system_id = clean_text(row.get("system_id"))
        participant = clean_text(row.get("participant_code"))
        task_id = clean_text(row.get("task_id"))
        by_system[system_id].append(row)
        by_participant_system[(participant, system_id)].append(row)
        by_task_system[(task_id, system_id)].append(row)
    for participant in participants:
        first = next(
            row
            for row in rows
            if clean_text(row.get("participant_code")) == participant
        )
        sequence_counts[clean_text(first.get("sequence_id"))] += 1

    condition_results: dict[str, Any] = {}
    for system_id in SYSTEM_IDS:
        system_rows = by_system[system_id]
        completed = [row for row in system_rows if row.get("outcome") == "completed"]
        performance_rows = [
            row
            for row in system_rows
            if row.get("outcome") not in {"network_failure", "technical_failure"}
        ]
        capped_durations = [
            int(row["duration_ms"]) if row.get("outcome") == "completed" else cap_ms
            for row in performance_rows
        ]
        selection_labels: list[str] = []
        three_relevant = 0
        fully_judged_trials = 0
        unique_pairs: set[tuple[str, str]] = set()
        for row in completed:
            trial_labels: list[str] = []
            for candidate in row.get("candidates") or []:
                task_id = clean_text(row.get("task_id"))
                notice_id = clean_text(candidate.get("notice_id"))
                unique_pairs.add((task_id, notice_id))
                trial_labels.append(label_for(task_id, notice_id))
            selection_labels.extend(trial_labels)
            if "unknown" not in trial_labels:
                fully_judged_trials += 1
                if trial_labels.count("relevant") == 3:
                    three_relevant += 1
        known = [label for label in selection_labels if label != "unknown"]
        outcome_counts = Counter(clean_text(row.get("outcome")) for row in system_rows)
        condition_results[system_id] = {
            "scheduled_trials": len(system_rows),
            "outcome_counts": {
                outcome: outcome_counts.get(outcome, 0) for outcome in OUTCOMES
            },
            "completion_rate_all_scheduled": _rate(len(completed), len(system_rows)),
            "technically_evaluable_trials": len(performance_rows),
            "completion_rate_technically_evaluable": _rate(
                len(completed), len(performance_rows)
            ),
            "capped_duration_ms_excluding_external_failures": _numeric_summary(
                capped_durations
            ),
            "completed_duration_ms": _numeric_summary(
                [int(row["duration_ms"]) for row in completed]
            ),
            "final_candidate_selection_events": len(selection_labels),
            "unique_task_notice_pairs": len(unique_pairs),
            "blind_relevance": {
                "relevant": selection_labels.count("relevant"),
                "not_relevant": selection_labels.count("not_relevant"),
                "unknown": selection_labels.count("unknown"),
                "known_label_count": len(known),
                "relevant_rate_among_known": _rate(
                    selection_labels.count("relevant"), len(known)
                ),
                "fully_judged_completed_trials": fully_judged_trials,
                "three_of_three_relevant_trials": three_relevant,
                "three_of_three_relevant_rate_among_fully_judged": _rate(
                    three_relevant, fully_judged_trials
                ),
            },
            "active_final_candidate_events": len(completed) * 3,
        }

    paired_detail: list[dict[str, Any]] = []
    excluded_participants = 0
    portal_faster = meong_faster = ties = 0
    for participant in participants:
        condition_medians: dict[str, float] = {}
        eligible = True
        for system_id in SYSTEM_IDS:
            participant_rows = by_participant_system[(participant, system_id)]
            performance_rows = [
                row
                for row in participant_rows
                if row.get("outcome") not in {"network_failure", "technical_failure"}
            ]
            if len(performance_rows) != 2:
                eligible = False
                break
            condition_medians[system_id] = float(
                median(
                    int(row["duration_ms"])
                    if row.get("outcome") == "completed"
                    else cap_ms
                    for row in performance_rows
                )
            )
        if not eligible:
            excluded_participants += 1
            continue
        delta = (
            condition_medians["meongtamjeong"] - condition_medians["official_portal"]
        )
        if delta < 0:
            meong_faster += 1
        elif delta > 0:
            portal_faster += 1
        else:
            ties += 1
        participant_ref = (
            "P-"
            + hashlib.sha256(f"{protocol_id}\0{participant}".encode("utf-8"))
            .hexdigest()[:8]
            .upper()
        )
        paired_detail.append(
            {
                "participant_ref": participant_ref,
                "official_portal_median_capped_duration_ms": _round(
                    condition_medians["official_portal"]
                ),
                "meongtamjeong_median_capped_duration_ms": _round(
                    condition_medians["meongtamjeong"]
                ),
                "meongtamjeong_minus_official_portal_ms": _round(delta),
            }
        )

    task_results: list[dict[str, Any]] = []
    for task_id in _task_map(_design(protocol)):
        for system_id in SYSTEM_IDS:
            task_rows = by_task_system[(task_id, system_id)]
            completed = [row for row in task_rows if row.get("outcome") == "completed"]
            task_results.append(
                {
                    "task_id": task_id,
                    "system_id": system_id,
                    "scheduled_trials": len(task_rows),
                    "completed_trials": len(completed),
                    "completion_rate": _rate(len(completed), len(task_rows)),
                }
            )

    deltas = [
        float(row["meongtamjeong_minus_official_portal_ms"]) for row in paired_detail
    ]
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "study_status": "completed_descriptive_external_portal_pilot",
        "protocol_id": protocol_id,
        "source_protocol": {
            "sha256": protocol_sha256,
            "schema_version": clean_text(protocol.get("schema_version")),
        },
        "source_trial_artifacts": [dict(item) for item in trial_artifacts],
        "source_blind_label_artifact": dict(label_artifact),
        "sample": {
            "participants": len(participants),
            "scheduled_trials": len(rows),
            "sequence_counts": {
                sequence_id: sequence_counts.get(sequence_id, 0)
                for sequence_id in ("S1", "S2", "S3", "S4")
            },
            "balanced_four_participant_schedule_achieved": (
                len(participants) == 4
                and all(
                    sequence_counts.get(sequence_id, 0) == 1
                    for sequence_id in ("S1", "S2", "S3", "S4")
                )
            ),
        },
        "conditions": condition_results,
        "paired_participant_comparison": {
            "eligible_participants": len(paired_detail),
            "excluded_due_to_network_or_technical_failure": excluded_participants,
            "meongtamjeong_faster": meong_faster,
            "official_portal_faster": portal_faster,
            "ties": ties,
            "meongtamjeong_minus_official_portal_ms": _numeric_summary(deltas),
            "participants_detail": paired_detail,
        },
        "task_results": task_results,
        "analysis_contract": {
            "primary_task": "time_to_confirm_three_active_candidates",
            "time_cap_and_participant_stop_are_capped": True,
            "network_and_technical_failures_excluded_from_capped_time": True,
            "all_failure_types_reported_separately": True,
            "blind_relevance_unknown_is_not_counted_as_not_relevant": True,
            "inferential_statistics": False,
            "claim_boundary": (
                "Small descriptive search pilot only; it does not measure adoption, "
                "shelter contact, temperament, health, safety, or population effects."
            ),
        },
    }


def render_result_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact, claim-bounded result summary."""

    sample = report["sample"]
    paired = report["paired_participant_comparison"]
    lines = [
        "# 공식 포털 대 멍탐정 후보 탐색 파일럿 결과",
        "",
        "> 3~5명의 기술적 파일럿입니다. 입양·문의·성격·건강 성과나 모집단 효과를 측정하지 않습니다.",
        "",
        f"- 프로토콜: `{report['protocol_id']}`",
        f"- 참여자: {sample['participants']}명",
        f"- 예정 시행: {sample['scheduled_trials']}회",
        f"- 4인 완전 균형: {sample['balanced_four_participant_schedule_achieved']}",
        "",
        "## 조건별 기술 통계",
        "",
        "| 조건 | 완료/예정 | time cap | 중도포기 | 네트워크 실패 | 기술 실패 | capped 중앙(ms)* | blind 관련성(known) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for system_id in SYSTEM_IDS:
        row = report["conditions"][system_id]
        outcomes = row["outcome_counts"]
        relevance = row["blind_relevance"]
        lines.append(
            f"| `{system_id}` | {outcomes['completed']}/{row['scheduled_trials']} | "
            f"{outcomes['time_cap']} | {outcomes['participant_stopped']} | "
            f"{outcomes['network_failure']} | {outcomes['technical_failure']} | "
            f"{row['capped_duration_ms_excluding_external_failures']['median']} | "
            f"{relevance['relevant']}/{relevance['known_label_count']} |"
        )
    lines.extend(
        [
            "",
            "\\* time cap과 참여자 중단은 cap으로 포함하고 네트워크·기술 실패는 시간 비교에서 제외하되 별도 공개합니다.",
            "",
            "## 참여자 단위 비교",
            "",
            f"- paired 가능: {paired['eligible_participants']}명",
            f"- 외부 실패로 paired 제외: {paired['excluded_due_to_network_or_technical_failure']}명",
            f"- 멍탐정이 더 빠름: {paired['meongtamjeong_faster']}명",
            f"- 공식 포털이 더 빠름: {paired['official_portal_faster']}명",
            f"- 동률: {paired['ties']}명",
            f"- 멍탐정 - 공식 포털 중앙 차이(ms): {paired['meongtamjeong_minus_official_portal_ms']['median']}",
            "",
            "## 해석 제한",
            "",
            "관련성 평가는 선택 출처를 숨긴 공식 공고 링크로 수행합니다. `unknown`은 오답으로 간주하지 않습니다. 종료 후보는 완료 후보로 인정하지 않으며, 실제 보호소 문의와 입양 성과는 평가 범위 밖입니다.",
            "",
        ]
    )
    return "\n".join(lines)
