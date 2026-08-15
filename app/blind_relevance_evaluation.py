"""Helpers for local, blinded human relevance evaluation.

The visible task contains a shuffled pool of candidates and never records
system names or original ranks.  A separate key maps opaque candidate codes
back to each system's ranking for later aggregation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse


TASK_SCHEMA_VERSION = "blind-relevance-task.v1"
KEY_SCHEMA_VERSION = "blind-relevance-key.v1"
RESULT_SCHEMA_VERSION = "blind-relevance-result.v1"
LABEL_COLUMNS = (
    "task_id",
    "reviewer_id",
    "query_id",
    "candidate_code",
    "relevance",
)
GRADES = {
    0: "무관",
    1: "부분 일치",
    2: "잘 일치",
}
_SAFE_REVIEWER_RE = re.compile(r"^[^\s,\r\n]{1,40}$")
_ALLOWED_IMAGE_HOSTS = {
    "animal.go.kr",
    "www.animal.go.kr",
    "openapi.animal.go.kr",
}


def clean_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _notice_id(meta: Mapping[str, Any], fallback: str) -> str:
    for field in ("desertionNo", "desertion_no"):
        value = clean_text(meta.get(field))
        if value:
            return value
    return fallback


def merge_notice_metadata(
    metas: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Collapse vector-level rows without exposing contact/location details."""

    merged: dict[str, dict[str, Any]] = {}
    for position, raw in enumerate(metas):
        if not isinstance(raw, Mapping):
            continue
        dog_id = _notice_id(raw, f"row-{position}")
        current = merged.setdefault(dog_id, {})
        prefer = clean_text(raw.get("type")).lower() == "image"
        for field in (
            "breed",
            "breed_name",
            "color",
            "age",
            "weight",
            "org_name",
            "region",
            "image_url",
            "image_urls",
            "image_attrs",
        ):
            value = raw.get(field)
            if value in (None, "", [], {}):
                continue
            if field not in current or prefer:
                current[field] = value
    return merged


def resolve_system_ids(
    report: Mapping[str, Any],
    requested: Sequence[str] | None = None,
) -> list[str]:
    systems = report.get("systems")
    if not isinstance(systems, Mapping) or len(systems) < 2:
        raise ValueError("retrieval report needs at least two systems")

    if requested:
        selected = [clean_text(value) for value in requested if clean_text(value)]
    else:
        contract = report.get("evaluation_contract")
        selected = []
        if isinstance(contract, Mapping):
            for field in ("baseline_system", "headline_system"):
                value = clean_text(contract.get(field))
                if value and value not in selected:
                    selected.append(value)
        if len(selected) < 2:
            selected = list(systems)[:2]

    if len(selected) < 2:
        raise ValueError("select at least two distinct systems")
    if len(set(selected)) != len(selected):
        raise ValueError("system selection contains duplicates")
    missing = [system_id for system_id in selected if system_id not in systems]
    if missing:
        raise ValueError("unknown systems: " + ", ".join(missing))
    return selected


def _query_rows_by_id(system: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = system.get("queries")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        raise ValueError("each system needs a queries array")
    output: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("query rows must be objects")
        query_id = clean_text(row.get("id"))
        if not query_id:
            raise ValueError("query row is missing id")
        if query_id in output:
            raise ValueError(f"duplicate query id: {query_id}")
        top_ids = row.get("top_ids")
        if not isinstance(top_ids, Sequence) or isinstance(
            top_ids, (str, bytes, bytearray)
        ):
            raise ValueError(f"query {query_id!r} is missing top_ids")
        output[query_id] = row
    return output


def _deduplicated_ids(values: Iterable[Any], depth: int) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        dog_id = clean_text(value)
        if dog_id and dog_id not in seen:
            output.append(dog_id)
            seen.add(dog_id)
        if len(output) >= depth:
            break
    return output


def _safe_remote_image(meta: Mapping[str, Any]) -> str:
    values: list[Any] = [meta.get("image_url")]
    image_urls = meta.get("image_urls")
    if isinstance(image_urls, Sequence) and not isinstance(
        image_urls, (str, bytes, bytearray)
    ):
        values.extend(image_urls)
    for value in values:
        url = clean_text(value)
        parsed = urlparse(url)
        if (
            parsed.scheme in {"http", "https"}
            and parsed.hostname in _ALLOWED_IMAGE_HOSTS
        ):
            return url
    return ""


def _local_crop(meta: Mapping[str, Any], project_root: Path) -> Path | None:
    attrs = meta.get("image_attrs")
    if not isinstance(attrs, Mapping):
        return None
    raw_path = clean_text(attrs.get("crop_path"))
    if not raw_path:
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        path = project_root / path
    try:
        resolved = path.resolve()
        resolved.relative_to(project_root.resolve())
    except (OSError, ValueError):
        return None
    if resolved.is_file():
        return resolved
    return None


def _image_source(
    meta: Mapping[str, Any],
    *,
    image_policy: str,
    html_output: Path,
    project_root: Path,
) -> str:
    if image_policy == "none":
        return ""
    local = _local_crop(meta, project_root)
    if image_policy in {"prefer-local", "local-only"} and local is not None:
        try:
            relative = os.path.relpath(local, html_output.parent)
            return Path(relative).as_posix()
        except ValueError:
            return local.as_uri()
    if image_policy == "local-only":
        return ""
    return _safe_remote_image(meta)


def _display_metadata(meta: Mapping[str, Any]) -> list[str]:
    values = (
        ("품종", meta.get("breed_name") or meta.get("breed")),
        ("색상", meta.get("color")),
        ("나이", meta.get("age")),
        ("체중", meta.get("weight")),
        ("지역", meta.get("region") or meta.get("org_name")),
    )
    return [
        f"{label}: {clean_text(value)}" for label, value in values if clean_text(value)
    ]


def _opaque_code(task_id: str, query_id: str, dog_id: str) -> str:
    raw = f"{task_id}\0{query_id}\0{dog_id}".encode("utf-8")
    return "C-" + hashlib.sha256(raw).hexdigest()[:12].upper()


def build_blind_task(
    report: Mapping[str, Any],
    metas: Sequence[Mapping[str, Any]],
    *,
    report_path: Path,
    metas_path: Path | None = None,
    systems: Sequence[str] | None,
    depth: int,
    seed: int,
    html_output: Path,
    project_root: Path,
    image_policy: str = "local-only",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a visible task and a separate unblinding key."""

    if not 1 <= depth <= 50:
        raise ValueError("depth must be between 1 and 50")
    if image_policy not in {"prefer-local", "local-only", "remote", "none"}:
        raise ValueError(f"unsupported image policy: {image_policy}")
    selected = resolve_system_ids(report, systems)
    report_systems = report["systems"]
    rows_by_system = {
        system_id: _query_rows_by_id(report_systems[system_id])
        for system_id in selected
    }
    query_sets = [set(rows) for rows in rows_by_system.values()]
    query_ids = set.intersection(*query_sets)
    if not query_ids:
        raise ValueError("selected systems have no shared queries")
    if any(query_ids != query_set for query_set in query_sets):
        raise ValueError("selected systems must contain the same query IDs")

    meta_by_id = merge_notice_metadata(metas)
    report_hash = sha256_file(report_path)
    if metas_path is not None:
        metas_hash = sha256_file(metas_path)
    else:
        metas_hash = hashlib.sha256(
            json.dumps(
                metas,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    try:
        portable_report_path = (
            report_path.resolve().relative_to(project_root.resolve()).as_posix()
        )
    except ValueError:
        portable_report_path = report_path.name
    if metas_path is not None:
        try:
            portable_metas_path = (
                metas_path.resolve().relative_to(project_root.resolve()).as_posix()
            )
        except ValueError:
            portable_metas_path = metas_path.name
    else:
        portable_metas_path = ""
    task_material = json.dumps(
        {
            "report_sha256": report_hash,
            "metas_sha256": metas_hash,
            "systems": selected,
            "depth": depth,
            "seed": seed,
            "image_policy": image_policy,
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    task_id = "blind-" + hashlib.sha256(task_material.encode("utf-8")).hexdigest()[:16]
    rng = random.Random(seed)
    visible_queries: list[dict[str, Any]] = []
    key_queries: list[dict[str, Any]] = []

    first_system = selected[0]
    for query_id in sorted(query_ids):
        source_row = rows_by_system[first_system][query_id]
        query_text = clean_text(source_row.get("query"))
        if not query_text:
            raise ValueError(f"query {query_id!r} has no text")
        for system_id in selected[1:]:
            compared_text = clean_text(rows_by_system[system_id][query_id].get("query"))
            if compared_text != query_text:
                raise ValueError(
                    f"query text differs across systems for query {query_id!r}"
                )
        rankings: dict[str, list[str]] = {}
        pooled: list[str] = []
        seen: set[str] = set()
        for system_id in selected:
            ranking = _deduplicated_ids(
                rows_by_system[system_id][query_id]["top_ids"],
                depth,
            )
            if not ranking:
                raise ValueError(
                    f"system {system_id!r}, query {query_id!r} has no candidates"
                )
            rankings[system_id] = ranking
            for dog_id in ranking:
                if dog_id not in seen:
                    pooled.append(dog_id)
                    seen.add(dog_id)
        rng.shuffle(pooled)

        visible_candidates: list[dict[str, Any]] = []
        key_candidates: list[dict[str, Any]] = []
        for dog_id in pooled:
            meta = meta_by_id.get(dog_id, {})
            code = _opaque_code(task_id, query_id, dog_id)
            visible_candidates.append(
                {
                    "candidate_code": code,
                    "image_src": _image_source(
                        meta,
                        image_policy=image_policy,
                        html_output=html_output,
                        project_root=project_root,
                    ),
                    "metadata": _display_metadata(meta),
                }
            )
            system_ranks = {
                system_id: ranking.index(dog_id) + 1
                for system_id, ranking in rankings.items()
                if dog_id in ranking
            }
            key_candidates.append(
                {
                    "candidate_code": code,
                    "notice_id": dog_id,
                    "system_ranks": system_ranks,
                }
            )
        visible_queries.append(
            {
                "query_id": query_id,
                "query": query_text,
                "candidates": visible_candidates,
            }
        )
        key_queries.append(
            {
                "query_id": query_id,
                "query": query_text,
                "candidates": key_candidates,
            }
        )

    visible = {
        "schema_version": TASK_SCHEMA_VERSION,
        "task_id": task_id,
        "instructions": {
            "0": "질의와 무관",
            "1": "일부 조건만 일치하거나 판단이 애매함",
            "2": "질의와 잘 일치",
        },
        "queries": visible_queries,
    }
    key = {
        "schema_version": KEY_SCHEMA_VERSION,
        "task_id": task_id,
        "source_report": {
            "path": portable_report_path,
            "sha256": report_hash,
            "schema_version": clean_text(report.get("schema_version")),
            "reference_date": clean_text(report.get("reference_date")),
        },
        "source_metas": {
            "path": portable_metas_path,
            "sha256": metas_hash,
        },
        "protocol": {
            "depth": depth,
            "seed": seed,
            "grades": {str(key): value for key, value in GRADES.items()},
            "pooling": "union_of_each_system_top_k",
            "candidate_order": "deterministically_shuffled_per_query",
            "image_policy": image_policy,
            "systems_and_original_ranks_hidden_from_visible_task": True,
        },
        "systems": [
            {
                "system_id": system_id,
                "role": clean_text(report_systems[system_id].get("role")),
            }
            for system_id in selected
        ],
        "queries": key_queries,
    }
    return visible, key


def _json_for_script(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def render_task_html(task: Mapping[str, Any], *, allow_remote_images: bool) -> str:
    payload = _json_for_script(task)
    image_sources = "https: http: data:" if allow_remote_images else "data:"
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta http-equiv="Content-Security-Policy"
        content="default-src 'none'; img-src 'self' file: {image_sources}; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'none'; form-action 'none'">
  <title>블라인드 검색 관련성 평가</title>
  <style>
    :root {{ color-scheme: light; --ink:#18211c; --muted:#5e6b63; --line:#dce6df; --accent:#196b45; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:system-ui,-apple-system,"Noto Sans KR",sans-serif; color:var(--ink); background:#f5f8f6; }}
    main {{ max-width:1100px; margin:0 auto; padding:28px 18px 60px; }}
    h1 {{ margin:0 0 8px; font-size:clamp(25px,4vw,38px); }}
    .intro,.toolbar {{ background:white; border:1px solid var(--line); border-radius:16px; padding:16px; }}
    .intro p {{ margin:6px 0; color:var(--muted); }}
    .toolbar {{ position:sticky; top:8px; z-index:3; margin:16px 0; display:flex; gap:12px; align-items:end; flex-wrap:wrap; box-shadow:0 6px 24px #163c2614; }}
    label {{ display:grid; gap:5px; font-size:14px; color:var(--muted); }}
    input {{ min-width:190px; padding:10px 12px; border:1px solid #b8c8be; border-radius:9px; font:inherit; }}
    button {{ border:0; border-radius:9px; padding:11px 15px; font:inherit; font-weight:700; cursor:pointer; }}
    #export {{ background:var(--accent); color:white; }}
    #export:disabled {{ opacity:.45; cursor:not-allowed; }}
    .progress {{ margin-left:auto; font-weight:700; }}
    section {{ margin:26px 0 38px; }}
    section h2 {{ margin:0 0 14px; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(230px,1fr)); gap:14px; }}
    article {{ overflow:hidden; background:white; border:1px solid var(--line); border-radius:14px; }}
    img,.placeholder {{ width:100%; aspect-ratio:4/3; object-fit:cover; background:#e6ece8; }}
    .placeholder {{ display:grid; place-items:center; color:var(--muted); }}
    .body {{ padding:13px; }}
    .code {{ font:12px ui-monospace,monospace; color:var(--muted); }}
    ul {{ min-height:54px; margin:8px 0 12px; padding-left:18px; font-size:13px; color:var(--muted); }}
    fieldset {{ border:0; padding:0; margin:0; display:grid; grid-template-columns:repeat(3,1fr); gap:6px; }}
    fieldset label {{ display:block; text-align:center; color:var(--ink); }}
    fieldset input {{ position:absolute; opacity:0; pointer-events:none; }}
    fieldset span {{ display:block; padding:9px 4px; border:1px solid #b8c8be; border-radius:8px; cursor:pointer; }}
    fieldset input:checked + span {{ color:white; background:var(--accent); border-color:var(--accent); }}
    .warning {{ color:#8a4512; font-weight:650; }}
  </style>
</head>
<body>
<main>
  <div class="intro">
    <h1>블라인드 검색 관련성 평가</h1>
    <p>시스템 이름과 원래 순위는 숨겨져 있습니다. 질의에 대한 외형·공개 공고 조건의 관련성만 0/1/2로 판단하세요.</p>
    <p class="warning">입양 적합성, 성격, 공격성, 아동·다른 동물 친화성을 사진으로 추정하지 마세요. 이름·이메일 대신 임의의 검수자 코드를 사용하세요.</p>
  </div>
  <div class="toolbar">
    <label>검수자 코드<input id="reviewer" maxlength="40" placeholder="예: reviewer-01"></label>
    <button id="export" disabled>CSV 저장</button>
    <div class="progress" id="progress"></div>
  </div>
  <div id="queries"></div>
</main>
<script>
const task={payload};
const answers=new Map();
const root=document.getElementById("queries");
const exportButton=document.getElementById("export");
const reviewer=document.getElementById("reviewer");
const total=task.queries.reduce((n,q)=>n+q.candidates.length,0);
function esc(value){{return String(value).replace(/[&<>"']/g,c=>({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}}[c]));}}
function update(){{
  document.getElementById("progress").textContent=`${{answers.size}} / ${{total}}`;
  exportButton.disabled=answers.size!==total || !/^[^\\s,\\r\\n]{{1,40}}$/.test(reviewer.value);
}}
for(const query of task.queries){{
  const section=document.createElement("section");
  section.innerHTML=`<h2>${{esc(query.query)}}</h2><div class="grid"></div>`;
  const grid=section.querySelector(".grid");
  for(const candidate of query.candidates){{
    const key=`${{query.query_id}}\\u0000${{candidate.candidate_code}}`;
    const card=document.createElement("article");
    const visual=candidate.image_src
      ? `<img loading="lazy" alt="후보 사진" referrerpolicy="no-referrer" src="${{esc(candidate.image_src)}}">`
      : `<div class="placeholder">로컬 사진 없음</div>`;
    const metadata=candidate.metadata.map(value=>`<li>${{esc(value)}}</li>`).join("");
    card.innerHTML=`${{visual}}<div class="body"><div class="code">${{esc(candidate.candidate_code)}}</div><ul>${{metadata}}</ul><fieldset aria-label="관련성 점수">
      ${{[0,1,2].map(value=>`<label><input type="radio" name="${{esc(key)}}" value="${{value}}"><span>${{value}}</span></label>`).join("")}}
    </fieldset></div>`;
    for(const input of card.querySelectorAll("input")) input.addEventListener("change",()=>{{answers.set(key,input.value); update();}});
    grid.appendChild(card);
  }}
  root.appendChild(section);
}}
reviewer.addEventListener("input",update);
exportButton.addEventListener("click",()=>{{
  const quote=value=>`"${{String(value).replaceAll('"','""')}}"`;
  const rows=[["task_id","reviewer_id","query_id","candidate_code","relevance"]];
  for(const query of task.queries) for(const candidate of query.candidates){{
    const key=`${{query.query_id}}\\u0000${{candidate.candidate_code}}`;
    rows.push([task.task_id,reviewer.value,query.query_id,candidate.candidate_code,answers.get(key)]);
  }}
  const csv="\\ufeff"+rows.map(row=>row.map(quote).join(",")).join("\\r\\n");
  const url=URL.createObjectURL(new Blob([csv],{{type:"text/csv;charset=utf-8"}}));
  const link=document.createElement("a"); link.href=url; link.download=`${{task.task_id}}-${{reviewer.value}}.csv`; link.click();
  setTimeout(()=>URL.revokeObjectURL(url),0);
}});
update();
</script>
</body>
</html>
"""


def write_label_template(task: Mapping[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(LABEL_COLUMNS)
        for query in task["queries"]:
            for candidate in query["candidates"]:
                writer.writerow(
                    (
                        task["task_id"],
                        "",
                        query["query_id"],
                        candidate["candidate_code"],
                        "",
                    )
                )


def write_task_artifacts(
    task: Mapping[str, Any],
    key: Mapping[str, Any],
    *,
    html_output: Path,
    key_output: Path,
    csv_output: Path,
    allow_remote_images: bool,
) -> None:
    for path in (html_output, key_output, csv_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    html_output.write_text(
        render_task_html(task, allow_remote_images=allow_remote_images),
        encoding="utf-8",
    )
    key_output.write_text(
        json.dumps(key, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_label_template(task, csv_output)


def _expected_rows(
    key: Mapping[str, Any],
) -> tuple[dict[tuple[str, str], Mapping[str, Any]], dict[str, str]]:
    candidates: dict[tuple[str, str], Mapping[str, Any]] = {}
    query_text: dict[str, str] = {}
    for query in key.get("queries") or []:
        if not isinstance(query, Mapping):
            raise ValueError("key contains an invalid query")
        query_id = clean_text(query.get("query_id"))
        query_text[query_id] = clean_text(query.get("query"))
        for candidate in query.get("candidates") or []:
            if not isinstance(candidate, Mapping):
                raise ValueError("key contains an invalid candidate")
            code = clean_text(candidate.get("candidate_code"))
            item_key = (query_id, code)
            if not query_id or not code or item_key in candidates:
                raise ValueError("key contains missing or duplicate candidate codes")
            candidates[item_key] = candidate
    if not candidates:
        raise ValueError("key has no candidates")
    return candidates, query_text


def load_judgments(
    paths: Sequence[Path],
    key: Mapping[str, Any],
) -> tuple[dict[str, dict[tuple[str, str], int]], list[dict[str, Any]]]:
    expected, _ = _expected_rows(key)
    task_id = clean_text(key.get("task_id"))
    by_reviewer: dict[str, dict[tuple[str, str], int]] = {}
    source_rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != LABEL_COLUMNS:
                raise ValueError(
                    f"{path}: columns must be exactly {', '.join(LABEL_COLUMNS)}"
                )
            for line_number, row in enumerate(reader, start=2):
                row_task = clean_text(row.get("task_id"))
                reviewer = clean_text(row.get("reviewer_id"))
                query_id = clean_text(row.get("query_id"))
                code = clean_text(row.get("candidate_code"))
                raw_grade = clean_text(row.get("relevance"))
                if row_task != task_id:
                    raise ValueError(
                        f"{path}:{line_number}: task_id does not match key"
                    )
                if not _SAFE_REVIEWER_RE.fullmatch(reviewer):
                    raise ValueError(
                        f"{path}:{line_number}: use a 1-40 character pseudonymous "
                        "reviewer_id without spaces or commas"
                    )
                item_key = (query_id, code)
                if item_key not in expected:
                    raise ValueError(
                        f"{path}:{line_number}: unknown query/candidate pair"
                    )
                if raw_grade not in {"0", "1", "2"}:
                    raise ValueError(
                        f"{path}:{line_number}: relevance must be 0, 1, or 2"
                    )
                reviewer_rows = by_reviewer.setdefault(reviewer, {})
                if item_key in reviewer_rows:
                    raise ValueError(
                        f"{path}:{line_number}: duplicate judgment for {reviewer}"
                    )
                reviewer_rows[item_key] = int(raw_grade)
                source_rows.append(
                    {
                        "reviewer_id": reviewer,
                        "query_id": query_id,
                        "candidate_code": code,
                        "relevance": int(raw_grade),
                        "source": str(path),
                    }
                )
    if not by_reviewer:
        raise ValueError("no judgments found")
    missing = {
        reviewer: sorted(set(expected) - set(rows))
        for reviewer, rows in by_reviewer.items()
        if set(rows) != set(expected)
    }
    if missing:
        details = ", ".join(
            f"{reviewer}={len(items)} missing" for reviewer, items in missing.items()
        )
        raise ValueError("every reviewer must complete the full task: " + details)
    return by_reviewer, source_rows


def _dcg(grades: Sequence[float]) -> float:
    return sum(
        (2.0**grade - 1.0) / math.log2(rank + 1)
        for rank, grade in enumerate(grades, start=1)
    )


def _round(value: float) -> float:
    return round(float(value), 6)


def aggregate_judgments(
    key: Mapping[str, Any],
    judgments: Mapping[str, Mapping[tuple[str, str], int]],
) -> dict[str, Any]:
    expected, query_text = _expected_rows(key)
    if not judgments:
        raise ValueError("at least one reviewer is required")
    reviewers = sorted(judgments)
    consensus = {
        item_key: mean(float(judgments[reviewer][item_key]) for reviewer in reviewers)
        for item_key in expected
    }
    systems = [
        clean_text(system.get("system_id"))
        for system in key.get("systems") or []
        if isinstance(system, Mapping)
    ]
    if len(systems) < 2 or len(set(systems)) != len(systems):
        raise ValueError("key needs at least two distinct systems")
    depth = int((key.get("protocol") or {}).get("depth") or 0)
    if depth <= 0:
        raise ValueError("key protocol depth must be positive")

    by_system: dict[str, list[dict[str, Any]]] = {system: [] for system in systems}
    for query in key["queries"]:
        query_id = clean_text(query.get("query_id"))
        candidate_rows = query["candidates"]
        pool_grades = [
            consensus[(query_id, clean_text(candidate["candidate_code"]))]
            for candidate in candidate_rows
        ]
        ideal_dcg = _dcg(sorted(pool_grades, reverse=True)[:depth])
        for system in systems:
            ranked: list[tuple[int, str]] = []
            for candidate in candidate_rows:
                ranks = candidate.get("system_ranks") or {}
                if system in ranks and int(ranks[system]) <= depth:
                    ranked.append(
                        (int(ranks[system]), clean_text(candidate["candidate_code"]))
                    )
            ranked.sort()
            grades = [consensus[(query_id, code)] for _, code in ranked]
            score = _dcg(grades) / ideal_dcg if ideal_dcg else 0.0
            by_system[system].append(
                {
                    "query_id": query_id,
                    "query": query_text[query_id],
                    f"nDCG@{depth}": _round(score),
                    "judged_results": len(grades),
                }
            )

    system_results: dict[str, Any] = {}
    metric_key = f"nDCG@{depth}"
    for system in systems:
        rows = by_system[system]
        system_results[system] = {
            metric_key: _round(mean(row[metric_key] for row in rows)),
            "query_count": len(rows),
            "judged_results": sum(row["judged_results"] for row in rows),
            "queries": rows,
        }

    pairwise: list[dict[str, Any]] = []
    for first_index, first in enumerate(systems):
        first_by_query = {
            row["query_id"]: float(row[metric_key]) for row in by_system[first]
        }
        for second in systems[first_index + 1 :]:
            second_by_query = {
                row["query_id"]: float(row[metric_key]) for row in by_system[second]
            }
            first_wins = second_wins = ties = 0
            for query_id in sorted(first_by_query):
                delta = first_by_query[query_id] - second_by_query[query_id]
                if abs(delta) <= 1e-9:
                    ties += 1
                elif delta > 0:
                    first_wins += 1
                else:
                    second_wins += 1
            decisive = first_wins + second_wins
            pairwise.append(
                {
                    "system_a": first,
                    "system_b": second,
                    "query_count": first_wins + second_wins + ties,
                    "system_a_wins": first_wins,
                    "system_b_wins": second_wins,
                    "ties": ties,
                    "system_a_preference_rate_excluding_ties": (
                        _round(first_wins / decisive) if decisive else None
                    ),
                    "preference_definition": (
                        f"query-level {metric_key} win; not a direct list preference vote"
                    ),
                }
            )

    all_grades = [
        grade
        for reviewer_rows in judgments.values()
        for grade in reviewer_rows.values()
    ]
    distribution = Counter(all_grades)
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "task_id": clean_text(key.get("task_id")),
        "task_source": {
            "key_schema_version": clean_text(key.get("schema_version")),
            "source_report": dict(key.get("source_report") or {}),
            "source_metas": dict(key.get("source_metas") or {}),
            "task_protocol": dict(key.get("protocol") or {}),
        },
        "protocol": {
            "depth": depth,
            "grade_scale": {str(key): value for key, value in GRADES.items()},
            "consensus": "arithmetic_mean_per_candidate",
            "gain": "2^grade-1",
            "aggregation": f"macro_mean_query_{metric_key}",
            "pairwise_preference": f"query-level {metric_key} wins excluding ties",
            "complete_task_required_per_reviewer": True,
        },
        "sample": {
            "reviewers": len(reviewers),
            "reviewer_ids": reviewers,
            "queries": len(key["queries"]),
            "unique_query_candidates": len(expected),
            "judgments": len(all_grades),
            "grade_distribution": {
                str(grade): distribution.get(grade, 0) for grade in GRADES
            },
        },
        "systems": system_results,
        "pairwise_preference": pairwise,
        "limitations": [
            "The candidate pool contains only the compared systems' top-k results.",
            "System preference is inferred from query-level nDCG wins, not collected as a direct preference vote.",
            "Small reviewer/query samples must not be presented as population-level evidence.",
            "Relevance labels do not measure adoption suitability, temperament, or safety.",
        ],
    }


def render_result_markdown(report: Mapping[str, Any]) -> str:
    sample = report["sample"]
    metric = next(
        key
        for system in report["systems"].values()
        for key in system
        if key.startswith("nDCG@")
    )
    lines = [
        "# Blind human relevance evaluation",
        "",
        f"- Task: `{report['task_id']}`",
        f"- Reviewers: {sample['reviewers']}",
        f"- Queries: {sample['queries']}",
        f"- Unique query-candidates: {sample['unique_query_candidates']}",
        f"- Judgments: {sample['judgments']}",
        "",
        "## Provenance",
        "",
        f"- Retrieval report SHA-256: `{report['task_source']['source_report'].get('sha256', '')}`",
        f"- Metadata SHA-256: `{report['task_source']['source_metas'].get('sha256', '')}`",
    ]
    for artifact in report.get("label_artifacts") or []:
        lines.append(f"- Label `{artifact['file']}` SHA-256: `{artifact['sha256']}`")
    lines.extend(
        [
            "",
            "## System results",
            "",
            f"| System | {metric} | Queries | Judged results |",
            "|---|---:|---:|---:|",
        ]
    )
    for system_id, result in report["systems"].items():
        lines.append(
            f"| `{system_id}` | {result[metric]:.6f} | "
            f"{result['query_count']} | {result['judged_results']} |"
        )
    lines.extend(
        [
            "",
            "## Pairwise relevance-derived preference",
            "",
            "| A | B | A wins | B wins | Ties | A rate (ties excluded) |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in report["pairwise_preference"]:
        rate = row["system_a_preference_rate_excluding_ties"]
        rate_text = "n/a" if rate is None else f"{rate:.6f}"
        lines.append(
            f"| `{row['system_a']}` | `{row['system_b']}` | "
            f"{row['system_a_wins']} | {row['system_b_wins']} | "
            f"{row['ties']} | {rate_text} |"
        )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.append("")
    return "\n".join(lines)
