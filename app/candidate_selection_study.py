"""Local, blinded pilot for selecting three shelter-visit candidates.

The visible artifact contains only opaque condition and candidate codes.  A
separate key keeps the mapping to retrieval systems, public notice IDs, and
field-derived evidence labels.  No browser event is sent to a server.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

from app.blind_relevance_evaluation import (
    clean_text,
    merge_notice_metadata,
    resolve_system_ids,
    sha256_file,
)


TASK_SCHEMA_VERSION = "candidate-selection-task.v1"
KEY_SCHEMA_VERSION = "candidate-selection-key.v1"
RESULT_SCHEMA_VERSION = "candidate-selection-result.v1"
CSV_COLUMNS = (
    "study_id",
    "participant_code",
    "sequence_id",
    "trial_index",
    "trial_token",
    "task_id",
    "condition_code",
    "duration_ms",
    "first_selection_ms",
    "selected_candidate_codes",
    "selection_toggle_count",
    "hidden_ms",
    "timed_out",
    "completed",
)
DEFAULT_QUERY_IDS = (
    "brown-small",
    "jeju-white-puppy",
    "gyeongnam-black-small",
    "senior-under-8kg",
)
DEFAULT_TIMEOUT_SECONDS = 180
MAX_LOCAL_IMAGE_BYTES = 2 * 1024 * 1024
_PARTICIPANT_RE = re.compile(r"^P[0-9]{2,4}$")
_SAFE_IMAGE_MIME = {
    ".gif": "image/gif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _portable_source_path(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    root = project_root.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"source must stay inside project root: {path}") from exc


def _query_rows(system: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = system.get("queries")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        raise ValueError("each selected system needs a queries array")
    output: dict[str, Mapping[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ValueError("query rows must be objects")
        query_id = clean_text(raw.get("id"))
        top_ids = raw.get("top_ids")
        if not query_id:
            raise ValueError("query row is missing id")
        if query_id in output:
            raise ValueError(f"duplicate query id: {query_id}")
        if not isinstance(top_ids, Sequence) or isinstance(
            top_ids, (str, bytes, bytearray)
        ):
            raise ValueError(f"query {query_id!r} is missing top_ids")
        output[query_id] = raw
    return output


def _deduplicated_ids(values: Sequence[Any], depth: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = clean_text(raw)
        if value and value not in seen:
            result.append(value)
            seen.add(value)
        if len(result) >= depth:
            break
    return result


def _qrels_by_id(report: Mapping[str, Any]) -> dict[str, set[str]]:
    raw_qrels = report.get("qrels")
    if not isinstance(raw_qrels, Sequence) or isinstance(
        raw_qrels,
        (str, bytes, bytearray),
    ):
        raise ValueError("retrieval report needs qrels for evidence guardrails")
    output: dict[str, set[str]] = {}
    for raw in raw_qrels:
        if not isinstance(raw, Mapping):
            raise ValueError("qrel rows must be objects")
        query_id = clean_text(raw.get("id"))
        relevant = raw.get("relevant_ids")
        if (
            not query_id
            or not isinstance(relevant, Sequence)
            or isinstance(
                relevant,
                (str, bytes, bytearray),
            )
        ):
            raise ValueError("each qrel needs id and relevant_ids")
        if query_id in output:
            raise ValueError(f"duplicate qrel id: {query_id}")
        output[query_id] = {
            clean_text(value) for value in relevant if clean_text(value)
        }
    return output


def _opaque_candidate_code(study_id: str, notice_id: str) -> str:
    digest = hashlib.sha256(f"{study_id}\0{notice_id}".encode()).hexdigest()
    return "C-" + digest[:12].upper()


def _opaque_task_id(study_id: str, query_id: str) -> str:
    digest = hashlib.sha256(f"{study_id}\0{query_id}".encode()).hexdigest()
    return "T-" + digest[:10].upper()


def _trial_token(
    study_id: str,
    sequence_id: str,
    trial_index: int,
    task_id: str,
    condition_code: str,
) -> str:
    material = f"{study_id}\0{sequence_id}\0{trial_index}\0{task_id}\0{condition_code}"
    return hashlib.sha256(material.encode()).hexdigest()[:24]


def _local_image_data_uri(
    meta: Mapping[str, Any],
    *,
    project_root: Path,
) -> str:
    attrs = meta.get("image_attrs")
    if not isinstance(attrs, Mapping):
        return ""
    raw_path = clean_text(attrs.get("crop_path"))
    if not raw_path:
        return ""
    path = Path(raw_path)
    if not path.is_absolute():
        path = project_root / path
    try:
        resolved = path.resolve()
        resolved.relative_to(project_root.resolve())
    except (OSError, ValueError):
        return ""
    mime = _SAFE_IMAGE_MIME.get(resolved.suffix.lower())
    if not mime or not resolved.is_file():
        return ""
    size = resolved.stat().st_size
    if not 0 < size <= MAX_LOCAL_IMAGE_BYTES:
        return ""
    encoded = base64.b64encode(resolved.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _display_metadata(meta: Mapping[str, Any]) -> list[str]:
    values = (
        ("품종 표기", meta.get("breed_name") or meta.get("breed")),
        ("색상", meta.get("color")),
        ("나이", meta.get("age")),
        ("체중", meta.get("weight")),
        ("지역", meta.get("region") or meta.get("org_name")),
    )
    output = [
        f"{label}: {clean_text(value)}" for label, value in values if clean_text(value)
    ]
    return output or ["공고 사실 정보 없음"]


def _condition_code_map(seed: int, material_sha256: str) -> dict[str, str]:
    parity = (
        int(
            hashlib.sha256(f"{seed}\0{material_sha256}".encode()).hexdigest(),
            16,
        )
        % 2
    )
    if parity:
        return {"baseline": "B", "meongtamjeong": "A"}
    return {"baseline": "A", "meongtamjeong": "B"}


def _sequence_blueprint() -> dict[str, list[tuple[int, str]]]:
    return {
        "S1": [
            (0, "baseline"),
            (1, "meongtamjeong"),
            (2, "meongtamjeong"),
            (3, "baseline"),
        ],
        "S2": [
            (1, "meongtamjeong"),
            (2, "baseline"),
            (3, "baseline"),
            (0, "meongtamjeong"),
        ],
        "S3": [
            (2, "baseline"),
            (3, "meongtamjeong"),
            (0, "meongtamjeong"),
            (1, "baseline"),
        ],
        "S4": [
            (3, "meongtamjeong"),
            (0, "baseline"),
            (1, "baseline"),
            (2, "meongtamjeong"),
        ],
    }


def build_study(
    report: Mapping[str, Any],
    metas: Sequence[Mapping[str, Any]],
    *,
    report_path: Path,
    metas_path: Path,
    project_root: Path,
    systems: Sequence[str] | None = None,
    query_ids: Sequence[str] = DEFAULT_QUERY_IDS,
    depth: int = 10,
    seed: int = 20260726,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a local-only visible task and its separate unblinding key."""

    if len(query_ids) != 4 or len(set(query_ids)) != 4:
        raise ValueError("exactly four distinct query IDs are required")
    if not 3 <= depth <= 50:
        raise ValueError("depth must be between 3 and 50")
    if not 30 <= timeout_seconds <= 600:
        raise ValueError("timeout_seconds must be between 30 and 600")

    selected_systems = resolve_system_ids(report, systems)
    if len(selected_systems) != 2:
        raise ValueError("candidate selection study compares exactly two systems")
    baseline_system, meong_system = selected_systems
    report_systems = report.get("systems")
    if not isinstance(report_systems, Mapping):
        raise ValueError("report needs systems")
    rows_by_role = {
        "baseline": _query_rows(report_systems[baseline_system]),
        "meongtamjeong": _query_rows(report_systems[meong_system]),
    }
    qrels = _qrels_by_id(report)
    meta_by_id = merge_notice_metadata(metas)

    report_hash = sha256_file(report_path)
    metas_hash = sha256_file(metas_path)
    try:
        source_report_payload = json.loads(report_path.read_text(encoding="utf-8"))
        source_metas_payload = json.loads(metas_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"study source cannot be loaded: {exc}") from exc
    if source_report_payload != report:
        raise ValueError("report object does not match report_path content")
    if source_metas_payload != list(metas):
        raise ValueError("metas object does not match metas_path content")
    study_material = {
        "source_report_sha256": report_hash,
        "source_metas_sha256": metas_hash,
        "systems": selected_systems,
        "query_ids": list(query_ids),
        "depth": depth,
        "seed": seed,
        "timeout_seconds": timeout_seconds,
        "candidate_pool": "union_top_k_then_local_image_required",
        "image_policy": "embedded_local_crop_only",
    }
    material_hash = canonical_sha256(study_material)
    study_id = "selection-" + material_hash[:16]
    role_to_code = _condition_code_map(seed, material_hash)

    visible_candidates: dict[str, dict[str, Any]] = {}
    visible_tasks: list[dict[str, Any]] = []
    key_tasks: list[dict[str, Any]] = []
    task_ids: list[str] = []

    for query_id in query_ids:
        if query_id not in qrels:
            raise ValueError(f"query {query_id!r} is missing qrels")
        missing_roles = [
            role for role, rows in rows_by_role.items() if query_id not in rows
        ]
        if missing_roles:
            raise ValueError(
                f"query {query_id!r} is missing from: {', '.join(missing_roles)}"
            )
        baseline_row = rows_by_role["baseline"][query_id]
        meong_row = rows_by_role["meongtamjeong"][query_id]
        query_text = clean_text(baseline_row.get("query"))
        if not query_text or clean_text(meong_row.get("query")) != query_text:
            raise ValueError(f"query text differs or is empty for {query_id!r}")

        rankings_by_role = {
            "baseline": _deduplicated_ids(baseline_row["top_ids"], depth),
            "meongtamjeong": _deduplicated_ids(meong_row["top_ids"], depth),
        }
        raw_pool = list(
            dict.fromkeys(
                rankings_by_role["baseline"] + rankings_by_role["meongtamjeong"]
            )
        )
        eligible_notice_ids: list[str] = []
        excluded_missing_local_image: list[str] = []
        image_by_notice: dict[str, str] = {}
        for notice_id in raw_pool:
            image_data = _local_image_data_uri(
                meta_by_id.get(notice_id, {}),
                project_root=project_root,
            )
            if not image_data:
                excluded_missing_local_image.append(notice_id)
                continue
            eligible_notice_ids.append(notice_id)
            image_by_notice[notice_id] = image_data

        confirmed_count = sum(
            notice_id in qrels[query_id] for notice_id in eligible_notice_ids
        )
        if len(eligible_notice_ids) < 3:
            raise ValueError(
                "local crop preflight failed for "
                f"query {query_id!r}: raw_pool={len(raw_pool)}, "
                f"usable_local_crops={len(eligible_notice_ids)}, "
                f"missing_or_invalid={len(excluded_missing_local_image)}. "
                "Network fallback is intentionally disabled; generate local crops "
                "first as documented in CANDIDATE_SELECTION_PROTOCOL.md."
            )
        if confirmed_count < 3:
            raise ValueError(
                "candidate evidence preflight failed for "
                f"query {query_id!r}: usable_local_crops={len(eligible_notice_ids)}, "
                f"evidence_confirmed={confirmed_count}. At least three are required; "
                "do not replace the task after seeing human outcomes."
            )

        code_by_notice = {
            notice_id: _opaque_candidate_code(study_id, notice_id)
            for notice_id in eligible_notice_ids
        }
        for notice_id, code in code_by_notice.items():
            visible_candidates.setdefault(
                code,
                {
                    "candidate_code": code,
                    "image_data": image_by_notice[notice_id],
                    "metadata": _display_metadata(meta_by_id.get(notice_id, {})),
                },
            )

        orders: dict[str, list[str]] = {}
        system_ranks: dict[str, dict[str, int | None]] = {}
        pool_codes = set(code_by_notice.values())
        for role, ranked_notice_ids in rankings_by_role.items():
            ranked_codes = [
                code_by_notice[notice_id]
                for notice_id in ranked_notice_ids
                if notice_id in code_by_notice
            ]
            tail_codes = sorted(pool_codes - set(ranked_codes))
            condition_code = role_to_code[role]
            orders[condition_code] = ranked_codes + tail_codes
        for notice_id, code in code_by_notice.items():
            system_ranks[code] = {
                role: (
                    rankings_by_role[role].index(notice_id) + 1
                    if notice_id in rankings_by_role[role]
                    else None
                )
                for role in ("baseline", "meongtamjeong")
            }

        task_id = _opaque_task_id(study_id, query_id)
        task_ids.append(task_id)
        visible_tasks.append(
            {
                "task_id": task_id,
                "query": query_text,
                "condition_orders": orders,
            }
        )
        key_tasks.append(
            {
                "task_id": task_id,
                "query_id": query_id,
                "query": query_text,
                "raw_pool_count": len(raw_pool),
                "eligible_pool_count": len(eligible_notice_ids),
                "excluded_missing_local_image_count": len(excluded_missing_local_image),
                "excluded_missing_local_image_notice_ids": (
                    excluded_missing_local_image
                ),
                "evidence_confirmed_pool_count": confirmed_count,
                "candidates": [
                    {
                        "candidate_code": code_by_notice[notice_id],
                        "notice_id": notice_id,
                        "evidence_confirmed": notice_id in qrels[query_id],
                        "system_ranks": system_ranks[code_by_notice[notice_id]],
                    }
                    for notice_id in eligible_notice_ids
                ],
            }
        )

    visible_sequences: dict[str, list[dict[str, Any]]] = {}
    key_sequences: dict[str, list[dict[str, Any]]] = {}
    for sequence_id, blueprint in _sequence_blueprint().items():
        visible_trials: list[dict[str, Any]] = []
        key_trials: list[dict[str, Any]] = []
        for trial_index, (task_position, role) in enumerate(blueprint, start=1):
            task_id = task_ids[task_position]
            condition_code = role_to_code[role]
            token = _trial_token(
                study_id,
                sequence_id,
                trial_index,
                task_id,
                condition_code,
            )
            visible_trials.append(
                {
                    "trial_index": trial_index,
                    "task_id": task_id,
                    "condition_code": condition_code,
                    "trial_token": token,
                }
            )
            key_trials.append(
                {
                    **visible_trials[-1],
                    "role": role,
                    "system_id": (
                        baseline_system if role == "baseline" else meong_system
                    ),
                }
            )
        visible_sequences[sequence_id] = visible_trials
        key_sequences[sequence_id] = key_trials

    task = {
        "schema_version": TASK_SCHEMA_VERSION,
        "study_id": study_id,
        "timeout_ms": timeout_seconds * 1000,
        "selection_count": 3,
        "privacy": {
            "network_requests": False,
            "remote_analytics": False,
            "persistent_browser_storage": False,
            "free_text": False,
        },
        "instructions": [
            "진행자가 사전 배정한 P와 숫자 2~4자리 코드만 입력하세요.",
            "각 과제에서 먼저 확인하고 싶은 후보를 정확히 3개 선택하세요.",
            "사진이나 공고 정보만으로 성격·건강·입양 적합성을 단정하지 마세요.",
            "결과는 브라우저에서 CSV로만 저장되며 서버로 전송되지 않습니다.",
        ],
        "candidates": visible_candidates,
        "tasks": visible_tasks,
        "sequences": visible_sequences,
    }
    visible_text = json.dumps(task, ensure_ascii=False, separators=(",", ":"))
    leaked_systems = [
        system_id for system_id in selected_systems if system_id in visible_text
    ]
    if leaked_systems:
        raise ValueError("visible task leaks a selected system ID")
    key = {
        "schema_version": KEY_SCHEMA_VERSION,
        "study_id": study_id,
        "study_material": study_material,
        "study_material_sha256": material_hash,
        "source_report": {
            "path": _portable_source_path(report_path, project_root),
            "sha256": report_hash,
            "schema_version": clean_text(report.get("schema_version")),
            "reference_date": clean_text(report.get("reference_date")),
        },
        "source_metas": {
            "path": _portable_source_path(metas_path, project_root),
            "sha256": metas_hash,
        },
        "protocol": {
            "participant_count_target": "3-5; four is balanced",
            "task_count_per_participant": 4,
            "selection_count": 3,
            "timeout_ms": timeout_seconds * 1000,
            "candidate_pool": "union_top_k_then_local_image_required",
            "tail_order": "candidates absent from a system top-k follow in opaque-code order",
            "image_policy": "embedded_local_crop_only",
            "same_candidate_pool_between_conditions": True,
            "system_names_hidden_from_visible_task": True,
            "field_qrels_are_guardrails_not_adoption_suitability": True,
        },
        "condition_roles": {
            role_to_code["baseline"]: {
                "role": "baseline",
                "system_id": baseline_system,
            },
            role_to_code["meongtamjeong"]: {
                "role": "meongtamjeong",
                "system_id": meong_system,
            },
        },
        "tasks": key_tasks,
        "sequences": key_sequences,
        "visible_task_sha256": canonical_sha256(task),
        "artifacts": {},
    }
    validate_key_structure(key)
    return task, key


def _json_for_script(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def render_task_html(task: Mapping[str, Any]) -> str:
    payload = _json_for_script(task)
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'none'; form-action 'none'; base-uri 'none'">
  <title>보호소 방문 후보 3개 선택 파일럿</title>
  <style>
    :root{{--ink:#17221d;--muted:#607068;--line:#d9e1dc;--brand:#175b47;--soft:#f3f7f4}}
    *{{box-sizing:border-box}} body{{margin:0;background:#f6f7f5;color:var(--ink);font-family:system-ui,-apple-system,"Noto Sans KR",sans-serif}}
    main{{max-width:1080px;margin:auto;padding:28px}} .panel{{background:white;border:1px solid var(--line);border-radius:18px;padding:22px;box-shadow:0 14px 34px #19352a12}}
    h1,h2,p{{margin-top:0}} .muted{{color:var(--muted);line-height:1.65}} label{{display:block;margin:12px 0 6px;font-weight:700}}
    input,select,button{{font:inherit}} input,select{{width:100%;padding:11px;border:1px solid var(--line);border-radius:10px}}
    button{{border:0;border-radius:10px;padding:11px 16px;background:var(--brand);color:white;font-weight:800;cursor:pointer}} button:disabled{{opacity:.45;cursor:not-allowed}}
    .toolbar{{display:flex;gap:10px;align-items:center;justify-content:space-between;flex-wrap:wrap;margin-bottom:18px}}
    .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px}} .card{{border:2px solid transparent;border-radius:14px;background:var(--soft);padding:10px;cursor:pointer;text-align:left;color:var(--ink)}}
    .card.selected{{border-color:var(--brand);box-shadow:0 0 0 3px #175b4720}} .card img{{width:100%;aspect-ratio:4/3;object-fit:cover;border-radius:10px;background:#e6ebe7}}
    .card ul{{margin:10px 0 0;padding-left:18px;font-size:.84rem;line-height:1.55}} .code{{font-size:.72rem;color:var(--muted);margin-top:8px}}
    .hidden{{display:none!important}} .progress{{font-weight:800}} .privacy{{font-size:.82rem;color:var(--muted);margin-top:18px}} .danger{{color:#9d2f28}}
  </style>
</head>
<body>
<main>
  <section class="panel" id="setup">
    <h1>보호소 방문 후보 3개 선택 파일럿</h1>
    <p class="muted">두 정렬 방식의 후보 탐색 시간을 비교하는 소규모 로컬 과제입니다. 입양 적합성·성격·안전을 평가하지 않습니다.</p>
    <ul id="instructions"></ul>
    <label for="participant">진행자 배정 참여자 코드</label>
    <input id="participant" minlength="3" maxlength="5" pattern="P[0-9]{{2,4}}" autocomplete="off" autocapitalize="characters" spellcheck="false" placeholder="예: P01">
    <label for="sequence">진행자가 배정한 순서표</label>
    <select id="sequence"></select>
    <p class="danger hidden" id="setupError">진행자가 배정한 P + 숫자 2~4자리 코드와 순서표를 확인하세요.</p>
    <button id="begin">연습 화면 열기</button>
  </section>
  <section class="panel hidden" id="practice">
    <h2>연습</h2><p class="muted">아래 네 칸 중 세 칸을 눌러 선택한 뒤 다음으로 이동하세요. 연습 기록은 저장되지 않습니다.</p>
    <div class="cards" id="practiceCards"></div>
    <button id="finishPractice" disabled>과제 안내로 이동</button>
  </section>
  <section class="panel hidden" id="ready">
    <div class="progress" id="readyProgress"></div>
    <h2 id="readyQuery"></h2>
    <p class="danger hidden" id="trialStartError" role="alert">카드 이미지를 준비하지 못해 측정을 시작하지 않았습니다. 다시 시도하거나 진행자에게 알려주세요.</p>
    <p class="muted">결과가 나타난 뒤 공고 사실을 확인하고 먼저 살펴볼 후보 3개를 고르세요. 제한시간은 화면에 표시하지 않습니다.</p>
    <button id="startTrial">과제 시작</button>
  </section>
  <section class="panel hidden" id="trial">
    <div class="toolbar"><div><div class="progress" id="trialProgress"></div><h2 id="trialQuery"></h2></div><div id="selectedCount">0 / 3 선택</div></div>
    <div class="cards" id="cards"></div>
    <div class="toolbar" style="margin-top:18px"><span class="muted">정확히 3개를 선택하면 확정할 수 있습니다.</span><button id="confirm" disabled>3개 후보 확정</button></div>
  </section>
  <section class="panel hidden" id="done">
    <h2>로컬 기록 완료</h2>
    <p class="muted">기록은 아직 이 브라우저 메모리에만 있습니다. CSV를 저장해 진행자에게 전달하세요. 이름이나 이메일을 파일명에 넣지 마세요.</p>
    <button id="export">CSV 저장</button>
    <p class="privacy">원격 요청·분석·브라우저 영구 저장소를 사용하지 않았습니다.</p>
  </section>
</main>
<script>
"use strict";
const study={payload};
const byId=id=>document.getElementById(id);
const setup=byId("setup"),practice=byId("practice"),ready=byId("ready"),trial=byId("trial"),done=byId("done");
const participant=byId("participant"),sequence=byId("sequence"),setupError=byId("setupError");
const startTrialButton=byId("startTrial"),trialStartError=byId("trialStartError"),cards=byId("cards");
const taskMap=new Map(study.tasks.map(item=>[item.task_id,item]));
let participantCode="",sequenceId="",schedule=[],trialPosition=0,rows=[];
let selected=new Set(),startedAt=0,firstSelection=null,toggleCount=0,hiddenMs=0,hiddenAt=null,timeoutHandle=null,active=false;
let trialStartPending=false;
const esc=value=>String(value).replace(/[&<>"']/g,ch=>({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}}[ch]));
byId("instructions").innerHTML=study.instructions.map(item=>`<li>${{esc(item)}}</li>`).join("");
sequence.innerHTML=Object.keys(study.sequences).map(id=>`<option value="${{id}}">${{id}}</option>`).join("");
const practiceSelected=new Set();
byId("practiceCards").innerHTML=["연습 A","연습 B","연습 C","연습 D"].map((label,index)=>`<button type="button" class="card" data-practice="${{index}}"><strong>${{label}}</strong></button>`).join("");
byId("practiceCards").addEventListener("click",event=>{{
  const card=event.target.closest("[data-practice]"); if(!card)return;
  const code=card.dataset.practice;
  if(practiceSelected.has(code)){{practiceSelected.delete(code);card.classList.remove("selected")}}
  else if(practiceSelected.size<3){{practiceSelected.add(code);card.classList.add("selected")}}
  byId("finishPractice").disabled=practiceSelected.size!==3;
}});
byId("begin").addEventListener("click",()=>{{
  const candidate=participant.value.trim(),seq=sequence.value;
  if(!/^P[0-9]{{2,4}}$/.test(candidate)||!study.sequences[seq]){{setupError.classList.remove("hidden");return}}
  participantCode=candidate;sequenceId=seq;schedule=study.sequences[seq];setup.classList.add("hidden");practice.classList.remove("hidden");
}});
byId("finishPractice").addEventListener("click",()=>{{practice.classList.add("hidden");showReady()}});
function showReady(){{
  const item=schedule[trialPosition],task=taskMap.get(item.task_id);
  ready.classList.remove("hidden");trial.classList.add("hidden");
  startTrialButton.disabled=false;trialStartError.classList.add("hidden");
  byId("readyProgress").textContent=`과제 ${{trialPosition+1}} / ${{schedule.length}}`;
  byId("readyQuery").textContent=task.query;
}}
async function decodeTrialImages(images){{
  if(images.length===0||images.some(image=>typeof image.decode!=="function"))throw new Error("trial image decode unavailable");
  await Promise.all(images.map(image=>image.decode()));
  if(images.some(image=>!image.complete||image.naturalWidth===0))throw new Error("trial image decode failed");
}}
const nextAnimationFrame=()=>new Promise(resolve=>requestAnimationFrame(resolve));
async function renderTrial(){{
  if(active||trialStartPending)return;
  trialStartPending=true;startTrialButton.disabled=true;trialStartError.classList.add("hidden");
  const item=schedule[trialPosition],task=taskMap.get(item.task_id),order=task.condition_orders[item.condition_code];
  selected=new Set();firstSelection=null;toggleCount=0;hiddenMs=0;hiddenAt=null;
  byId("trialProgress").textContent=`과제 ${{trialPosition+1}} / ${{schedule.length}}`;
  byId("trialQuery").textContent=task.query;
  cards.innerHTML=order.map(code=>{{
    const candidate=study.candidates[code];
    return `<button type="button" class="card" data-code="${{esc(code)}}" aria-pressed="false"><img alt="후보 공고 사진" src="${{candidate.image_data}}"><div class="code">${{esc(code)}}</div><ul>${{candidate.metadata.map(value=>`<li>${{esc(value)}}</li>`).join("")}}</ul></button>`;
  }}).join("");
  updateSelection();
  try{{
    const images=Array.from(cards.querySelectorAll("img"));
    if(images.length!==order.length)throw new Error("trial image count mismatch");
    await decodeTrialImages(images);
    ready.classList.add("hidden");trial.classList.remove("hidden");
    await nextAnimationFrame();
    await nextAnimationFrame();
    startedAt=performance.now();active=true;trialStartPending=false;
    timeoutHandle=setTimeout(()=>finishTrial(false),study.timeout_ms);
  }}catch(error){{
    active=false;trialStartPending=false;cards.innerHTML="";
    trial.classList.add("hidden");ready.classList.remove("hidden");startTrialButton.disabled=false;
    trialStartError.textContent="카드 이미지를 준비하지 못해 측정을 시작하지 않았습니다. 다시 시도하거나 진행자에게 알려주세요.";
    trialStartError.classList.remove("hidden");
  }}
}}
startTrialButton.addEventListener("click",renderTrial);
cards.addEventListener("click",event=>{{
  if(!active)return;const card=event.target.closest("[data-code]");if(!card)return;
  const code=card.dataset.code;
  if(selected.has(code)){{selected.delete(code);card.classList.remove("selected");card.setAttribute("aria-pressed","false");toggleCount++}}
  else if(selected.size<study.selection_count){{selected.add(code);card.classList.add("selected");card.setAttribute("aria-pressed","true");toggleCount++;if(firstSelection===null)firstSelection=performance.now()-startedAt}}
  updateSelection();
}});
function updateSelection(){{
  byId("selectedCount").textContent=`${{selected.size}} / ${{study.selection_count}} 선택`;
  byId("confirm").disabled=selected.size!==study.selection_count;
}}
byId("confirm").addEventListener("click",()=>finishTrial(true));
document.addEventListener("visibilitychange",()=>{{
  if(!active)return;
  if(document.hidden&&hiddenAt===null)hiddenAt=performance.now();
  else if(!document.hidden&&hiddenAt!==null){{hiddenMs+=performance.now()-hiddenAt;hiddenAt=null}}
}});
function finishTrial(completed){{
  if(!active)return;active=false;clearTimeout(timeoutHandle);
  const now=performance.now();if(hiddenAt!==null){{hiddenMs+=now-hiddenAt;hiddenAt=null}}
  const duration=completed?Math.min(Math.round(now-startedAt),study.timeout_ms):study.timeout_ms;
  const item=schedule[trialPosition];
  rows.push({{
    study_id:study.study_id,participant_code:participantCode,sequence_id:sequenceId,
    trial_index:item.trial_index,trial_token:item.trial_token,task_id:item.task_id,condition_code:item.condition_code,
    duration_ms:duration,first_selection_ms:firstSelection===null?"":Math.min(Math.round(firstSelection),duration),
    selected_candidate_codes:Array.from(selected).join("|"),selection_toggle_count:toggleCount,
    hidden_ms:Math.min(Math.round(hiddenMs),duration),timed_out:String(!completed),completed:String(completed)
  }});
  trialPosition++;trial.classList.add("hidden");
  if(trialPosition<schedule.length)showReady();else done.classList.remove("hidden");
}}
function csvCell(value){{const text=String(value??"");return /[",\\r\\n]/.test(text)?`"${{text.replaceAll('"','""')}}"`:text}}
byId("export").addEventListener("click",()=>{{
  const columns={json.dumps(list(CSV_COLUMNS))};
  const lines=[columns.join(","),...rows.map(row=>columns.map(column=>csvCell(row[column])).join(","))];
  const blob=new Blob(["\\ufeff"+lines.join("\\r\\n")+"\\r\\n"],{{type:"text/csv;charset=utf-8"}});
  const url=URL.createObjectURL(blob),link=document.createElement("a");link.href=url;link.download=`${{study.study_id}}-${{participantCode}}.csv`;link.click();URL.revokeObjectURL(url);
}});
</script>
</body>
</html>
"""


def write_study_artifacts(
    task: Mapping[str, Any],
    key: dict[str, Any],
    *,
    html_output: Path,
    key_output: Path,
) -> None:
    html = render_task_html(task)
    html_output.parent.mkdir(parents=True, exist_ok=True)
    key_output.parent.mkdir(parents=True, exist_ok=True)
    html_output.write_text(html, encoding="utf-8")
    key["artifacts"] = {
        "task_html": {
            "file": html_output.name,
            "sha256": sha256_file(html_output),
            "bytes": html_output.stat().st_size,
        }
    }
    key_output.write_text(
        json.dumps(key, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def validate_key_structure(key: Mapping[str, Any]) -> None:
    if clean_text(key.get("schema_version")) != KEY_SCHEMA_VERSION:
        raise ValueError("unsupported candidate selection key schema")
    material = key.get("study_material")
    if not isinstance(material, Mapping):
        raise ValueError("key is missing study_material")
    material_hash = canonical_sha256(material)
    if material_hash != clean_text(key.get("study_material_sha256")):
        raise ValueError("study material hash mismatch")
    if clean_text(key.get("study_id")) != "selection-" + material_hash[:16]:
        raise ValueError("study_id does not match study material")
    visible_hash = clean_text(key.get("visible_task_sha256"))
    if not re.fullmatch(r"[0-9a-f]{64}", visible_hash):
        raise ValueError("visible task SHA-256 is missing or malformed")

    conditions = key.get("condition_roles")
    if not isinstance(conditions, Mapping) or len(conditions) != 2:
        raise ValueError("key needs exactly two condition codes")
    roles = {
        clean_text(value.get("role"))
        for value in conditions.values()
        if isinstance(value, Mapping)
    }
    if roles != {"baseline", "meongtamjeong"}:
        raise ValueError("condition roles must be baseline and meongtamjeong")

    tasks = key.get("tasks")
    if not isinstance(tasks, Sequence) or isinstance(tasks, (str, bytes, bytearray)):
        raise ValueError("key needs tasks")
    if len(tasks) != 4:
        raise ValueError("key needs exactly four tasks")
    task_ids: set[str] = set()
    pools: dict[str, set[str]] = {}
    for task in tasks:
        if not isinstance(task, Mapping):
            raise ValueError("task rows must be objects")
        task_id = clean_text(task.get("task_id"))
        candidates = task.get("candidates")
        if not task_id or task_id in task_ids:
            raise ValueError("task IDs must be present and unique")
        if not isinstance(candidates, Sequence) or isinstance(
            candidates,
            (str, bytes, bytearray),
        ):
            raise ValueError(f"task {task_id} needs candidates")
        if any(not isinstance(item, Mapping) for item in candidates):
            raise ValueError(f"task {task_id} candidate rows must be objects")
        codes = [clean_text(item.get("candidate_code")) for item in candidates]
        if len(codes) < 3 or any(not code for code in codes):
            raise ValueError(f"task {task_id} needs at least three candidate codes")
        if len(set(codes)) != len(codes):
            raise ValueError(f"task {task_id} has duplicate candidate codes")
        if sum(bool(item.get("evidence_confirmed")) for item in candidates) < 3:
            raise ValueError(
                f"task {task_id} needs three evidence-confirmed candidates"
            )
        task_ids.add(task_id)
        pools[task_id] = set(codes)

    sequences = key.get("sequences")
    if not isinstance(sequences, Mapping) or set(sequences) != {
        "S1",
        "S2",
        "S3",
        "S4",
    }:
        raise ValueError("key needs balanced sequences S1-S4")
    task_role_counts: Counter[tuple[str, str]] = Counter()
    position_role_counts: Counter[tuple[int, str]] = Counter()
    tokens: set[str] = set()
    for sequence_id, trials in sequences.items():
        if not isinstance(trials, Sequence) or len(trials) != 4:
            raise ValueError(f"{sequence_id} needs four trials")
        sequence_tasks: set[str] = set()
        sequence_roles: Counter[str] = Counter()
        for expected_index, trial in enumerate(trials, start=1):
            if not isinstance(trial, Mapping):
                raise ValueError("sequence trials must be objects")
            trial_index = trial.get("trial_index")
            task_id = clean_text(trial.get("task_id"))
            condition_code = clean_text(trial.get("condition_code"))
            role = clean_text(trial.get("role"))
            token = clean_text(trial.get("trial_token"))
            if trial_index != expected_index:
                raise ValueError(f"{sequence_id} trial indexes are not contiguous")
            if task_id not in task_ids or task_id in sequence_tasks:
                raise ValueError(f"{sequence_id} must use each task exactly once")
            condition = conditions.get(condition_code)
            if (
                not isinstance(condition, Mapping)
                or clean_text(condition.get("role")) != role
            ):
                raise ValueError("condition code and role mapping disagree")
            if clean_text(trial.get("system_id")) != clean_text(
                condition.get("system_id")
            ):
                raise ValueError("sequence system ID and condition mapping disagree")
            expected_token = _trial_token(
                clean_text(key.get("study_id")),
                clean_text(sequence_id),
                expected_index,
                task_id,
                condition_code,
            )
            if token != expected_token or token in tokens:
                raise ValueError("trial token mismatch or duplicate")
            tokens.add(token)
            sequence_tasks.add(task_id)
            sequence_roles[role] += 1
            task_role_counts[(task_id, role)] += 1
            position_role_counts[(expected_index, role)] += 1
        if sequence_tasks != task_ids or sequence_roles != {
            "baseline": 2,
            "meongtamjeong": 2,
        }:
            raise ValueError(f"{sequence_id} is not internally balanced")
    for task_id in task_ids:
        for role in ("baseline", "meongtamjeong"):
            if task_role_counts[(task_id, role)] != 2:
                raise ValueError("task-condition allocation is not balanced")
    for position in range(1, 5):
        for role in ("baseline", "meongtamjeong"):
            if position_role_counts[(position, role)] != 2:
                raise ValueError("serial-position allocation is not balanced")


def _safe_source_path(project_root: Path, portable: str) -> Path:
    if not portable or Path(portable).is_absolute():
        raise ValueError("source paths in key must be relative")
    root = project_root.resolve()
    resolved = (root / portable).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("source path escapes project root") from exc
    return resolved


def check_study_integrity(
    key: Mapping[str, Any],
    *,
    project_root: Path,
    task_html: Path | None = None,
) -> list[str]:
    failures: list[str] = []
    try:
        validate_key_structure(key)
    except (KeyError, TypeError, ValueError) as exc:
        return [str(exc)]
    source_paths: dict[str, Path] = {}
    for field in ("source_report", "source_metas"):
        source = key.get(field)
        if not isinstance(source, Mapping):
            failures.append(f"{field} is missing")
            continue
        try:
            path = _safe_source_path(project_root, clean_text(source.get("path")))
            actual = sha256_file(path)
        except (OSError, ValueError) as exc:
            failures.append(f"{field} cannot be verified: {exc}")
            continue
        source_paths[field] = path
        if actual != clean_text(source.get("sha256")):
            failures.append(f"{field} SHA-256 does not match current source")
    if not failures and set(source_paths) == {"source_report", "source_metas"}:
        try:
            report = json.loads(
                source_paths["source_report"].read_text(encoding="utf-8")
            )
            metas = json.loads(source_paths["source_metas"].read_text(encoding="utf-8"))
            material = key["study_material"]
            if not isinstance(report, Mapping):
                raise ValueError("source report must contain an object")
            if not isinstance(metas, list) or any(
                not isinstance(item, Mapping) for item in metas
            ):
                raise ValueError("source metas must contain an object array")
            expected_task, expected_key = build_study(
                report,
                metas,
                report_path=source_paths["source_report"],
                metas_path=source_paths["source_metas"],
                project_root=project_root,
                systems=list(material["systems"]),
                query_ids=list(material["query_ids"]),
                depth=int(material["depth"]),
                seed=int(material["seed"]),
                timeout_seconds=int(material["timeout_seconds"]),
            )
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            failures.append(f"study cannot be rebuilt from sources: {exc}")
        else:
            actual_without_artifacts = dict(key)
            actual_without_artifacts.pop("artifacts", None)
            expected_without_artifacts = dict(expected_key)
            expected_without_artifacts.pop("artifacts", None)
            if actual_without_artifacts != expected_without_artifacts:
                failures.append(
                    "study key does not match a deterministic source rebuild"
                )
            if canonical_sha256(expected_task) != clean_text(
                key.get("visible_task_sha256")
            ):
                failures.append("visible task hash does not match source rebuild")
    if task_html is not None:
        artifact = (key.get("artifacts") or {}).get("task_html")
        if not isinstance(artifact, Mapping):
            failures.append("task_html artifact is missing from key")
        elif not task_html.is_file():
            failures.append("task_html file is missing")
        else:
            if sha256_file(task_html) != clean_text(artifact.get("sha256")):
                failures.append("task_html SHA-256 mismatch")
            if task_html.stat().st_size != artifact.get("bytes"):
                failures.append("task_html byte count mismatch")
    return failures


def _parse_bool(value: str, *, field: str, location: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"{location}: {field} must be exactly true or false")


def _parse_int(
    value: str,
    *,
    field: str,
    location: str,
    minimum: int,
    maximum: int,
    allow_empty: bool = False,
) -> int | None:
    if allow_empty and value == "":
        return None
    if not re.fullmatch(r"\d+", value):
        raise ValueError(f"{location}: {field} must be an integer")
    parsed = int(value)
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{location}: {field} must be between {minimum} and {maximum}")
    return parsed


def load_trial_csvs(
    paths: Sequence[Path],
    key: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Load structurally complete participant exports and reject tampering."""

    validate_key_structure(key)
    if not paths:
        raise ValueError("at least one participant CSV is required")
    study_id = clean_text(key.get("study_id"))
    timeout_ms = int((key.get("protocol") or {}).get("timeout_ms"))
    tasks = {
        clean_text(task.get("task_id")): task
        for task in key["tasks"]
        if isinstance(task, Mapping)
    }
    conditions = key["condition_roles"]
    sequences = key["sequences"]
    output: list[dict[str, Any]] = []
    seen_participants: set[str] = set()

    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
                raise ValueError(
                    f"{path}: columns must be exactly {', '.join(CSV_COLUMNS)}"
                )
            source_rows = list(reader)
        if len(source_rows) != 4:
            raise ValueError(f"{path}: each participant must complete four trials")
        participant_values = {
            clean_text(row.get("participant_code")) for row in source_rows
        }
        sequence_values = {clean_text(row.get("sequence_id")) for row in source_rows}
        if len(participant_values) != 1 or len(sequence_values) != 1:
            raise ValueError(f"{path}: participant and sequence must be consistent")
        participant = next(iter(participant_values))
        sequence_id = next(iter(sequence_values))
        if not _PARTICIPANT_RE.fullmatch(participant):
            raise ValueError(
                f"{path}: use the assigned participant code P followed by 2-4 digits"
            )
        if participant in seen_participants:
            raise ValueError(f"{path}: duplicate participant code")
        if sequence_id not in sequences:
            raise ValueError(f"{path}: unknown sequence_id")
        seen_participants.add(participant)

        expected_trials = {
            int(trial["trial_index"]): trial for trial in sequences[sequence_id]
        }
        seen_indexes: set[int] = set()
        for line_number, raw in enumerate(source_rows, start=2):
            location = f"{path}:{line_number}"
            if clean_text(raw.get("study_id")) != study_id:
                raise ValueError(f"{location}: study_id does not match key")
            trial_index_raw = clean_text(raw.get("trial_index"))
            trial_index = _parse_int(
                trial_index_raw,
                field="trial_index",
                location=location,
                minimum=1,
                maximum=4,
            )
            assert trial_index is not None
            if trial_index in seen_indexes:
                raise ValueError(f"{location}: duplicate trial_index")
            seen_indexes.add(trial_index)
            expected = expected_trials[trial_index]
            for field in ("trial_token", "task_id", "condition_code"):
                if clean_text(raw.get(field)) != clean_text(expected.get(field)):
                    raise ValueError(f"{location}: {field} does not match schedule")

            duration = _parse_int(
                clean_text(raw.get("duration_ms")),
                field="duration_ms",
                location=location,
                minimum=0,
                maximum=timeout_ms,
            )
            first_selection = _parse_int(
                clean_text(raw.get("first_selection_ms")),
                field="first_selection_ms",
                location=location,
                minimum=0,
                maximum=timeout_ms,
                allow_empty=True,
            )
            toggles = _parse_int(
                clean_text(raw.get("selection_toggle_count")),
                field="selection_toggle_count",
                location=location,
                minimum=0,
                maximum=10000,
            )
            hidden_ms = _parse_int(
                clean_text(raw.get("hidden_ms")),
                field="hidden_ms",
                location=location,
                minimum=0,
                maximum=timeout_ms,
            )
            assert (
                duration is not None and toggles is not None and hidden_ms is not None
            )
            if first_selection is not None and first_selection > duration:
                raise ValueError(f"{location}: first_selection_ms exceeds duration_ms")
            if hidden_ms > duration:
                raise ValueError(f"{location}: hidden_ms exceeds duration_ms")

            timed_out = _parse_bool(
                clean_text(raw.get("timed_out")),
                field="timed_out",
                location=location,
            )
            completed = _parse_bool(
                clean_text(raw.get("completed")),
                field="completed",
                location=location,
            )
            selected_raw = clean_text(raw.get("selected_candidate_codes"))
            selected = selected_raw.split("|") if selected_raw else []
            if len(selected) != len(set(selected)):
                raise ValueError(f"{location}: selected candidates contain duplicates")
            task_id = clean_text(expected.get("task_id"))
            allowed = {
                clean_text(item.get("candidate_code"))
                for item in tasks[task_id]["candidates"]
            }
            if any(code not in allowed for code in selected):
                raise ValueError(f"{location}: selected candidate is not in task pool")
            if len(selected) > 3:
                raise ValueError(
                    f"{location}: no more than three candidates are allowed"
                )
            if completed != (not timed_out and len(selected) == 3):
                raise ValueError(
                    f"{location}: completed/timed_out/selection count are inconsistent"
                )
            if timed_out and duration != timeout_ms:
                raise ValueError(f"{location}: timed-out duration must equal the cap")
            if first_selection is None and selected:
                raise ValueError(
                    f"{location}: selected candidates require first_selection_ms"
                )
            if toggles < len(selected):
                raise ValueError(
                    f"{location}: selection_toggle_count is smaller than selection count"
                )

            task_candidate_map = {
                clean_text(item.get("candidate_code")): bool(
                    item.get("evidence_confirmed")
                )
                for item in tasks[task_id]["candidates"]
            }
            condition_code = clean_text(expected.get("condition_code"))
            condition = conditions[condition_code]
            confirmed = sum(task_candidate_map[code] for code in selected)
            output.append(
                {
                    "participant_code": participant,
                    "sequence_id": sequence_id,
                    "trial_index": trial_index,
                    "task_id": task_id,
                    "query_id": clean_text(tasks[task_id].get("query_id")),
                    "condition_code": condition_code,
                    "role": clean_text(condition.get("role")),
                    "system_id": clean_text(condition.get("system_id")),
                    "duration_ms": duration,
                    "first_selection_ms": first_selection,
                    "selected_count": len(selected),
                    "evidence_confirmed_count": confirmed,
                    "selection_toggle_count": toggles,
                    "hidden_ms": hidden_ms,
                    "timed_out": timed_out,
                    "completed": completed,
                }
            )
        if seen_indexes != {1, 2, 3, 4}:
            raise ValueError(f"{path}: trial indexes must be exactly 1-4")
    return output


def _round(value: float) -> float:
    return round(float(value), 6)


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
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


def aggregate_trials(
    key: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    source_key_sha256: str,
    label_artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate descriptive pilot metrics without inferential claims."""

    validate_key_structure(key)
    if not rows:
        raise ValueError("at least one validated participant is required")
    participants = sorted({clean_text(row.get("participant_code")) for row in rows})
    if not 3 <= len(participants) <= 5:
        raise ValueError(
            "a completed candidate-selection pilot requires 3-5 participants"
        )
    if len(rows) != len(participants) * 4:
        raise ValueError("validated rows must contain four trials per participant")

    by_role: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_task_role: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    by_participant_role: dict[
        tuple[str, str],
        list[Mapping[str, Any]],
    ] = defaultdict(list)
    sequence_counts: Counter[str] = Counter()
    for row in rows:
        role = clean_text(row.get("role"))
        participant = clean_text(row.get("participant_code"))
        task_id = clean_text(row.get("task_id"))
        by_role[role].append(row)
        by_task_role[(task_id, role)].append(row)
        by_participant_role[(participant, role)].append(row)
    for participant in participants:
        participant_rows = [
            row
            for row in rows
            if clean_text(row.get("participant_code")) == participant
        ]
        sequence_counts[clean_text(participant_rows[0].get("sequence_id"))] += 1

    condition_results: dict[str, Any] = {}
    for role in ("baseline", "meongtamjeong"):
        role_rows = by_role[role]
        completed = [row for row in role_rows if bool(row.get("completed"))]
        selected_total = sum(int(row.get("selected_count", 0)) for row in role_rows)
        confirmed_total = sum(
            int(row.get("evidence_confirmed_count", 0)) for row in role_rows
        )
        confirmed_three = sum(
            bool(row.get("completed"))
            and int(row.get("evidence_confirmed_count", 0)) == 3
            for row in role_rows
        )
        system_id = clean_text(role_rows[0].get("system_id"))
        condition_results[role] = {
            "system_id": system_id,
            "participant_count": len(participants),
            "trial_count": len(role_rows),
            "completed_trials": len(completed),
            "timed_out_trials": sum(bool(row.get("timed_out")) for row in role_rows),
            "completion_rate": _rate(len(completed), len(role_rows)),
            "three_of_three_evidence_confirmed_trials": confirmed_three,
            "three_of_three_evidence_confirmed_rate_among_completed": _rate(
                confirmed_three,
                len(completed),
            ),
            "evidence_confirmed_rate_across_selected": _rate(
                confirmed_total,
                selected_total,
            ),
            "capped_duration_ms": _numeric_summary(
                [int(row["duration_ms"]) for row in role_rows]
            ),
            "completed_duration_ms": _numeric_summary(
                [int(row["duration_ms"]) for row in completed]
            ),
            "first_selection_ms": _numeric_summary(
                [
                    int(row["first_selection_ms"])
                    for row in role_rows
                    if row.get("first_selection_ms") is not None
                ]
            ),
            "selection_toggle_count": _numeric_summary(
                [int(row["selection_toggle_count"]) for row in role_rows]
            ),
            "hidden_ms": _numeric_summary([int(row["hidden_ms"]) for row in role_rows]),
        }

    paired_rows: list[dict[str, Any]] = []
    faster = slower = ties = 0
    for participant in participants:
        baseline_rows = by_participant_role[(participant, "baseline")]
        meong_rows = by_participant_role[(participant, "meongtamjeong")]
        if len(baseline_rows) != 2 or len(meong_rows) != 2:
            raise ValueError("each participant needs two trials per condition")
        baseline_median = float(
            median(int(row["duration_ms"]) for row in baseline_rows)
        )
        meong_median = float(median(int(row["duration_ms"]) for row in meong_rows))
        delta = meong_median - baseline_median
        if delta < 0:
            faster += 1
        elif delta > 0:
            slower += 1
        else:
            ties += 1
        participant_ref = (
            "P-"
            + hashlib.sha256(f"{key['study_id']}\0{participant}".encode())
            .hexdigest()[:8]
            .upper()
        )
        paired_rows.append(
            {
                "participant_ref": participant_ref,
                "baseline_median_capped_duration_ms": _round(baseline_median),
                "meongtamjeong_median_capped_duration_ms": _round(meong_median),
                "meongtamjeong_minus_baseline_ms": _round(delta),
                "relative_delta_from_baseline": (
                    _round(delta / baseline_median) if baseline_median else None
                ),
            }
        )
    paired_deltas = [
        float(row["meongtamjeong_minus_baseline_ms"]) for row in paired_rows
    ]

    tasks_by_id = {
        clean_text(task.get("task_id")): task
        for task in key["tasks"]
        if isinstance(task, Mapping)
    }
    task_results: list[dict[str, Any]] = []
    for task_id, task in tasks_by_id.items():
        for role in ("baseline", "meongtamjeong"):
            task_rows = by_task_role[(task_id, role)]
            completed = [row for row in task_rows if bool(row.get("completed"))]
            task_results.append(
                {
                    "query_id": clean_text(task.get("query_id")),
                    "role": role,
                    "trial_count": len(task_rows),
                    "completed_trials": len(completed),
                    "completion_rate": _rate(len(completed), len(task_rows)),
                    "three_of_three_evidence_confirmed_trials": sum(
                        bool(row.get("completed"))
                        and int(row.get("evidence_confirmed_count", 0)) == 3
                        for row in task_rows
                    ),
                    "capped_duration_ms": _numeric_summary(
                        [int(row["duration_ms"]) for row in task_rows]
                    ),
                }
            )

    baseline_result = condition_results["baseline"]
    meong_result = condition_results["meongtamjeong"]
    guard = {
        "descriptive_time_reduction_observed": (
            _numeric_summary(paired_deltas)["median"] is not None
            and float(_numeric_summary(paired_deltas)["median"]) < 0
        ),
        "meongtamjeong_completion_rate_not_lower": (
            float(meong_result["completion_rate"] or 0)
            >= float(baseline_result["completion_rate"] or 0)
        ),
        "meongtamjeong_three_of_three_rate_not_lower": (
            float(
                meong_result["three_of_three_evidence_confirmed_rate_among_completed"]
                or 0
            )
            >= float(
                baseline_result[
                    "three_of_three_evidence_confirmed_rate_among_completed"
                ]
                or 0
            )
        ),
        "claim_boundary": (
            "Descriptive pilot only. Time must be interpreted with completion "
            "and evidence-confirmed selection guardrails."
        ),
    }
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "study_id": clean_text(key.get("study_id")),
        "study_status": "completed_small_local_human_pilot",
        "source_key": {
            "sha256": source_key_sha256,
            "schema_version": clean_text(key.get("schema_version")),
        },
        "source_artifacts": [dict(item) for item in label_artifacts],
        "protocol": {
            **dict(key.get("protocol") or {}),
            "primary_unit": "participant-level median capped duration",
            "inferential_statistics": False,
            "system_identity_blinded_during_task": True,
        },
        "sample": {
            "participants": len(participants),
            "trials": len(rows),
            "sequence_counts": {
                sequence_id: sequence_counts.get(sequence_id, 0)
                for sequence_id in ("S1", "S2", "S3", "S4")
            },
            "balanced_four_participant_schedule_achieved": (
                len(participants) == 4
                and all(
                    sequence_counts.get(sequence_id, 0) == 1
                    for sequence_id in sequence_counts
                )
            ),
        },
        "conditions": condition_results,
        "paired_participant_comparison": {
            "participants": len(paired_rows),
            "meongtamjeong_faster": faster,
            "baseline_faster": slower,
            "ties": ties,
            "meongtamjeong_minus_baseline_ms": _numeric_summary(paired_deltas),
            "participants_detail": paired_rows,
        },
        "task_pool_audit": [
            {
                "query_id": clean_text(task.get("query_id")),
                "raw_union_top_k": int(task.get("raw_pool_count", 0)),
                "local_image_eligible": int(task.get("eligible_pool_count", 0)),
                "excluded_missing_or_invalid_local_image": int(
                    task.get("excluded_missing_local_image_count", 0)
                ),
                "evidence_confirmed_in_pool": int(
                    task.get("evidence_confirmed_pool_count", 0)
                ),
            }
            for task in key["tasks"]
        ],
        "task_by_condition": task_results,
        "interpretation_guard": guard,
        "limitations": [
            "Three to five convenience-sample participants form a usability pilot, not population-level evidence.",
            "No p-value, confidence interval, or causal population claim is supported by this sample.",
            "The baseline is the recorded internal retrieval baseline, not the official public portal or general web search.",
            "The same fixed, local-image-eligible candidate pool isolates ordering and does not measure corpus recall.",
            "Candidates without a usable local crop are excluded; missing local images can change the evaluated pool, so per-task exclusion counts must be reported.",
            "Timing starts after cards render, so model, network, query-formulation, and public-site latency are excluded.",
            "Field-derived qrels only check explicit notice criteria; they do not measure temperament, health, safety, or adoption suitability.",
            "Counterbalancing reduces but cannot remove practice, fatigue, device, and project-familiarity effects.",
            "Structural CSV validation detects malformed or inconsistent exports but cannot prove that a participant followed instructions.",
            "Notice metadata and status reflect the fixed source snapshot; shelter confirmation remains necessary before a visit.",
        ],
    }


def render_result_markdown(report: Mapping[str, Any]) -> str:
    sample = report["sample"]
    paired = report["paired_participant_comparison"]
    lines = [
        "# 보호소 방문 후보 3개 선택 시간 파일럿",
        "",
        "> 소수 로컬 참여자의 기술적·사용성 파일럿입니다. 입양 성공, 성격, 안전 또는 모집단 효과를 측정하지 않습니다.",
        "",
        f"- Study: `{report['study_id']}`",
        f"- 참여자: {sample['participants']}명",
        f"- 완료된 과제 행: {sample['trials']}건",
        f"- 4인 완전 균형 일정: {'예' if sample['balanced_four_participant_schedule_achieved'] else '아니오'}",
        "",
        "## 조건별 기술 통계",
        "",
        "| 조건 | 과제 | 완료율 | 3/3 공고사실 확인율* | capped 중앙시간(ms) |",
        "|---|---:|---:|---:|---:|",
    ]
    for role in ("baseline", "meongtamjeong"):
        row = report["conditions"][role]
        evidence_rate = row["three_of_three_evidence_confirmed_rate_among_completed"]
        lines.append(
            f"| `{role}` | {row['trial_count']} | "
            f"{row['completion_rate'] if row['completion_rate'] is not None else 'n/a'} | "
            f"{evidence_rate if evidence_rate is not None else 'n/a'} | "
            f"{row['capped_duration_ms']['median']} |"
        )
    lines.extend(
        [
            "",
            "\\* 공고에 명시된 질의 조건만 확인한 값이며 입양 적합성 판정이 아닙니다.",
            "",
            "## 참여자 단위 paired 비교",
            "",
            f"- 멍탐정이 더 빠름: {paired['meongtamjeong_faster']}명",
            f"- baseline이 더 빠름: {paired['baseline_faster']}명",
            f"- 동률: {paired['ties']}명",
            f"- 멍탐정 - baseline 중앙 차이(ms): {paired['meongtamjeong_minus_baseline_ms']['median']}",
            "",
            "## 로컬 후보 pool 감사",
            "",
            "| Query | 원 합집합 | 로컬 사진 후보 | 사진 누락 제외 | 공고사실 확인 후보 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in report["task_pool_audit"]:
        lines.append(
            f"| `{row['query_id']}` | {row['raw_union_top_k']} | "
            f"{row['local_image_eligible']} | "
            f"{row['excluded_missing_or_invalid_local_image']} | "
            f"{row['evidence_confirmed_in_pool']} |"
        )
    lines.extend(
        [
            "",
            "## 과제·조건별 공개",
            "",
            "| Query | 조건 | 시행 | 완료 | capped 중앙시간(ms) |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in report["task_by_condition"]:
        lines.append(
            f"| `{row['query_id']}` | `{row['role']}` | {row['trial_count']} | "
            f"{row['completed_trials']} | {row['capped_duration_ms']['median']} |"
        )
    lines.extend(["", "## 해석 제한", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.append("")
    return "\n".join(lines)
