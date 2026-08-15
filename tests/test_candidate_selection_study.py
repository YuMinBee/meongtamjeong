from __future__ import annotations

import csv
import json
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from app.candidate_selection_study import (
    CSV_COLUMNS,
    aggregate_trials,
    build_study,
    check_study_integrity,
    load_trial_csvs,
    render_result_markdown,
    render_task_html,
    validate_key_structure,
    write_study_artifacts,
)
from scripts import candidate_selection_study as cli


QUERY_IDS = ("q1", "q2", "q3", "q4")


def _report() -> dict[str, object]:
    baseline_queries: list[dict[str, object]] = []
    meong_queries: list[dict[str, object]] = []
    qrels: list[dict[str, object]] = []
    for query_index, query_id in enumerate(QUERY_IDS, start=1):
        notice_ids = [f"notice-{query_index}-{index}" for index in range(1, 6)]
        query = f"외형 조건 {query_index}"
        baseline_queries.append({"id": query_id, "query": query, "top_ids": notice_ids})
        meong_queries.append(
            {
                "id": query_id,
                "query": query,
                "top_ids": [
                    notice_ids[2],
                    notice_ids[1],
                    notice_ids[0],
                    notice_ids[4],
                    notice_ids[3],
                ],
            }
        )
        qrels.append({"id": query_id, "relevant_ids": notice_ids[:3]})
    return {
        "schema_version": "retrieval-evaluation.test",
        "reference_date": "2026-07-26",
        "evaluation_contract": {
            "baseline_system": "baseline-secret",
            "headline_system": "meong-secret",
        },
        "systems": {
            "baseline-secret": {
                "role": "baseline",
                "queries": baseline_queries,
            },
            "meong-secret": {
                "role": "headline",
                "queries": meong_queries,
            },
        },
        "qrels": qrels,
    }


def _metas(project_root: Path) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    image_dir = project_root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for query_index in range(1, 5):
        for candidate_index in range(1, 6):
            notice_id = f"notice-{query_index}-{candidate_index}"
            image = image_dir / f"{notice_id}.jpg"
            image.write_bytes(f"local-image-{notice_id}".encode())
            output.append(
                {
                    "desertionNo": notice_id,
                    "type": "image",
                    "breed_name": "믹스견",
                    "color": "갈색",
                    "age": "2023(년생)",
                    "weight": "5(Kg)",
                    "region": "서울특별시",
                    "care_tel": "010-1111-2222",
                    "happen_place": "민감한 상세 장소",
                    "image_url": f"https://example.invalid/{notice_id}.jpg",
                    "image_attrs": {
                        "crop_path": image.relative_to(project_root).as_posix()
                    },
                }
            )
    return output


def _build(tmp_path: Path) -> tuple[dict[str, object], dict[str, object], Path, Path]:
    report = _report()
    metas = _metas(tmp_path)
    report_path = tmp_path / "retrieval.json"
    metas_path = tmp_path / "metas.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False),
        encoding="utf-8",
    )
    metas_path.write_text(
        json.dumps(metas, ensure_ascii=False),
        encoding="utf-8",
    )
    task, key = build_study(
        report,
        metas,
        report_path=report_path,
        metas_path=metas_path,
        project_root=tmp_path,
        query_ids=QUERY_IDS,
        depth=5,
        seed=7,
        timeout_seconds=60,
    )
    return task, key, report_path, metas_path


def _task_by_id(key: dict[str, object]) -> dict[str, dict[str, object]]:
    return {task["task_id"]: task for task in key["tasks"]}  # type: ignore[index]


def _rows_for_participant(
    key: dict[str, object],
    *,
    participant: str,
    sequence_id: str,
    baseline_duration: int = 1000,
    meong_duration: int = 800,
) -> list[dict[str, str]]:
    tasks = _task_by_id(key)
    rows: list[dict[str, str]] = []
    for trial in key["sequences"][sequence_id]:  # type: ignore[index]
        task = tasks[trial["task_id"]]
        confirmed = [
            candidate["candidate_code"]
            for candidate in task["candidates"]
            if candidate["evidence_confirmed"]
        ][:3]
        duration = baseline_duration if trial["role"] == "baseline" else meong_duration
        rows.append(
            {
                "study_id": str(key["study_id"]),
                "participant_code": participant,
                "sequence_id": sequence_id,
                "trial_index": str(trial["trial_index"]),
                "trial_token": trial["trial_token"],
                "task_id": trial["task_id"],
                "condition_code": trial["condition_code"],
                "duration_ms": str(duration),
                "first_selection_ms": "200",
                "selected_candidate_codes": "|".join(confirmed),
                "selection_toggle_count": "3",
                "hidden_ms": "0",
                "timed_out": "false",
                "completed": "true",
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def test_build_is_local_only_deterministic_blinded_and_balanced(
    tmp_path: Path,
) -> None:
    task, key, _report_path, _metas_path = _build(tmp_path)
    repeated_task, repeated_key, _report_path, _metas_path = _build(tmp_path)

    assert task == repeated_task
    assert key == repeated_key
    validate_key_structure(key)
    visible = json.dumps(task, ensure_ascii=False)
    html = render_task_html(task)
    assert "baseline-secret" not in visible
    assert "meong-secret" not in visible
    assert "notice-1-1" not in visible
    assert "010-1111-2222" not in visible
    assert "민감한 상세 장소" not in visible
    assert "https://" not in html
    assert "http://" not in html
    assert "connect-src 'none'" in html
    assert "localStorage" not in html
    assert "sessionStorage" not in html
    assert 'pattern="P[0-9]{2,4}"' in html
    assert "/^P[0-9]{2,4}$/" in html
    assert "async function decodeTrialImages(images)" in html
    assert "await Promise.all(images.map(image=>image.decode()))" in html
    assert "trial image decode failed" in html
    trial_source = html[html.index("async function renderTrial()") :]
    decode_at = trial_source.index("await decodeTrialImages(images)")
    show_at = trial_source.index('trial.classList.remove("hidden")')
    first_frame_at = trial_source.index("await nextAnimationFrame()")
    second_frame_at = trial_source.index(
        "await nextAnimationFrame()", first_frame_at + 1
    )
    timer_at = trial_source.index("startedAt=performance.now()")
    timeout_at = trial_source.index("timeoutHandle=setTimeout")
    assert (
        decode_at < show_at < first_frame_at < second_frame_at < timer_at < timeout_at
    )
    assert all(
        candidate["image_data"].startswith("data:image/jpeg;base64,")
        for candidate in task["candidates"].values()
    )
    for visible_task in task["tasks"]:
        orders = list(visible_task["condition_orders"].values())
        assert len(orders) == 2
        assert set(orders[0]) == set(orders[1])

    key_text = json.dumps(key, ensure_ascii=False)
    assert "baseline-secret" in key_text
    assert "meong-secret" in key_text
    assert "notice-1-1" in key_text
    for sequence in key["sequences"].values():
        assert len(sequence) == 4
        assert {trial["task_id"] for trial in sequence} == {
            task["task_id"] for task in key["tasks"]
        }
        assert [trial["role"] for trial in sequence].count("baseline") == 2
        assert [trial["role"] for trial in sequence].count("meongtamjeong") == 2


def test_browser_timer_waits_for_decode_and_paint_and_fails_closed(
    tmp_path: Path,
) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the generated-browser timing test")

    task, _key, _report_path, _metas_path = _build(tmp_path)
    html = render_task_html(task)
    script = html.split("<script>", 1)[1].split("</script>", 1)[0]
    harness = r"""
const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync(0, "utf8");

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function makeClassList(hidden) {
  const values = new Set(hidden ? ["hidden"] : []);
  return {
    add(value) { values.add(value); },
    remove(value) { values.delete(value); },
    contains(value) { return values.has(value); },
  };
}

class FakeElement {
  constructor(id, imageFactory, hidden) {
    this.id = id;
    this.imageFactory = imageFactory;
    this.classList = makeClassList(hidden);
    this.listeners = new Map();
    this.images = [];
    this.attributes = new Map();
    this._innerHTML = "";
    this.textContent = "";
    this.value = "";
    this.disabled = false;
  }
  addEventListener(type, handler) { this.listeners.set(type, handler); }
  fire(type, event = {}) {
    const handler = this.listeners.get(type);
    if (!handler) throw new Error(`missing ${type} handler for ${this.id}`);
    return handler(event);
  }
  set innerHTML(value) {
    this._innerHTML = value;
    if (this.id === "cards") {
      const count = (value.match(/<img\b/g) || []).length;
      this.images = Array.from({length: count}, () => this.imageFactory());
    }
  }
  get innerHTML() { return this._innerHTML; }
  querySelectorAll(selector) { return selector === "img" ? this.images : []; }
  setAttribute(name, value) { this.attributes.set(name, value); }
}

async function flushMicrotasks() {
  for (let index = 0; index < 6; index += 1) await Promise.resolve();
}

async function runScenario(rejectDecode) {
  const imageControls = [];
  const frameCallbacks = [];
  const timeoutCalls = [];
  let nowCalls = 0;

  function makeImage() {
    let resolveDecode;
    let rejectImage;
    const control = {decodeCalls: 0};
    const image = {
      complete: false,
      naturalWidth: 0,
      decode() {
        control.decodeCalls += 1;
        return new Promise((resolve, reject) => {
          resolveDecode = resolve;
          rejectImage = reject;
        });
      },
    };
    control.resolve = () => {
      image.complete = true;
      image.naturalWidth = 320;
      resolveDecode();
    };
    control.reject = () => rejectImage(new Error("synthetic decode failure"));
    imageControls.push(control);
    return image;
  }

  const hiddenIds = new Set([
    "practice", "ready", "trial", "done", "setupError", "trialStartError",
  ]);
  const ids = [
    "setup", "practice", "ready", "trial", "done", "participant",
    "sequence", "setupError", "trialStartError", "instructions",
    "practiceCards", "finishPractice", "begin", "startTrial", "cards",
    "readyProgress", "readyQuery", "trialProgress", "trialQuery",
    "selectedCount", "confirm", "export",
  ];
  const elements = new Map(
    ids.map(id => [id, new FakeElement(id, makeImage, hiddenIds.has(id))]),
  );
  const documentListeners = new Map();
  const document = {
    hidden: false,
    getElementById(id) {
      const element = elements.get(id);
      if (!element) throw new Error(`unknown element ${id}`);
      return element;
    },
    addEventListener(type, handler) { documentListeners.set(type, handler); },
    createElement(id) { return new FakeElement(id, makeImage, false); },
  };
  const context = {
    document,
    performance: {now() { nowCalls += 1; return 1000; }},
    requestAnimationFrame(callback) { frameCallbacks.push(callback); return 1; },
    setTimeout(callback, delay) {
      timeoutCalls.push({callback, delay});
      return timeoutCalls.length;
    },
    clearTimeout() {},
    Blob: class {},
    URL: {createObjectURL() { return "blob:test"; }, revokeObjectURL() {}},
    console,
  };
  vm.createContext(context);
  vm.runInContext(source, context, {filename: "task-inline.js"});

  elements.get("participant").value = "P01";
  elements.get("sequence").value = "S1";
  elements.get("begin").fire("click");
  elements.get("finishPractice").fire("click");
  const startPromise = elements.get("startTrial").fire("click");

  assert(startPromise && typeof startPromise.then === "function", "start must be async");
  assert(imageControls.length > 0, "trial cards must contain images");
  assert(
    imageControls.every(control => control.decodeCalls === 1),
    "every trial image must be decoded exactly once",
  );
  assert(nowCalls === 0 && timeoutCalls.length === 0, "timer started before decode");
  assert(
    !elements.get("ready").classList.contains("hidden") &&
      elements.get("trial").classList.contains("hidden"),
    "trial became visible before decode completed",
  );

  if (rejectDecode) {
    imageControls[0].reject();
    await startPromise;
    assert(nowCalls === 0 && timeoutCalls.length === 0, "decode failure started timer");
    assert(elements.get("trial").classList.contains("hidden"), "failed trial visible");
    assert(!elements.get("ready").classList.contains("hidden"), "ready screen hidden");
    assert(
      !elements.get("trialStartError").classList.contains("hidden"),
      "decode failure was not shown",
    );
    assert(elements.get("startTrial").disabled === false, "retry was not enabled");
    assert(elements.get("cards").innerHTML === "", "failed cards were retained");
    return;
  }

  imageControls.forEach(control => control.resolve());
  await flushMicrotasks();
  assert(frameCallbacks.length === 1, "first animation frame was not requested");
  assert(nowCalls === 0 && timeoutCalls.length === 0, "timer started before paint frames");
  assert(elements.get("ready").classList.contains("hidden"), "ready still visible");
  assert(!elements.get("trial").classList.contains("hidden"), "trial not visible");

  frameCallbacks.shift()(0);
  await flushMicrotasks();
  assert(frameCallbacks.length === 1, "second animation frame was not requested");
  assert(nowCalls === 0 && timeoutCalls.length === 0, "timer started after one frame");

  frameCallbacks.shift()(16);
  await startPromise;
  assert(nowCalls === 1, "performance timer did not start after two frames");
  assert(timeoutCalls.length === 1, "timeout did not start with performance timer");
}

(async () => {
  await runScenario(false);
  await runScenario(true);
  process.stdout.write("browser timing gate: PASS\n");
})().catch(error => {
  console.error(error.stack || error);
  process.exitCode = 1;
});
"""
    completed = subprocess.run(
        [node, "-e", harness],
        input=script,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert completed.stdout.strip() == "browser timing gate: PASS"


def test_integrity_check_detects_source_and_html_changes(tmp_path: Path) -> None:
    task, key, _report_path, metas_path = _build(tmp_path)
    html_path = tmp_path / "out" / "task.html"
    key_path = tmp_path / "out" / "task_key.json"
    write_study_artifacts(
        task,
        key,
        html_output=html_path,
        key_output=key_path,
    )
    assert (
        check_study_integrity(
            key,
            project_root=tmp_path,
            task_html=html_path,
        )
        == []
    )

    tampered_key = deepcopy(key)
    tampered_key["tasks"][0]["candidates"][0]["notice_id"] = "changed-notice"
    failures = check_study_integrity(
        tampered_key,
        project_root=tmp_path,
        task_html=html_path,
    )
    assert "study key does not match a deterministic source rebuild" in failures

    html_path.write_text("tampered", encoding="utf-8")
    failures = check_study_integrity(
        key,
        project_root=tmp_path,
        task_html=html_path,
    )
    assert "task_html SHA-256 mismatch" in failures

    metas_path.write_text("[]", encoding="utf-8")
    failures = check_study_integrity(key, project_root=tmp_path)
    assert "source_metas SHA-256 does not match current source" in failures


def test_clean_clone_without_ignored_crops_fails_fast_without_network_fallback(
    tmp_path: Path,
) -> None:
    report = _report()
    metas = _metas(tmp_path)
    for item in metas:
        item["image_attrs"] = {"crop_path": "ignored-crops/not-present.jpg"}
    report_path = tmp_path / "retrieval.json"
    metas_path = tmp_path / "metas.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    metas_path.write_text(json.dumps(metas, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="Network fallback is intentionally disabled"):
        build_study(
            report,
            metas,
            report_path=report_path,
            metas_path=metas_path,
            project_root=tmp_path,
            query_ids=QUERY_IDS,
            depth=5,
            timeout_seconds=60,
        )


def test_csv_loader_accepts_complete_exports_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    _task, key, _report_path, _metas_path = _build(tmp_path)
    rows = _rows_for_participant(key, participant="P01", sequence_id="S1")
    labels = tmp_path / "P01.csv"
    _write_csv(labels, rows)

    loaded = load_trial_csvs([labels], key)
    assert len(loaded) == 4
    assert {row["role"] for row in loaded} == {"baseline", "meongtamjeong"}
    assert all(row["evidence_confirmed_count"] == 3 for row in loaded)

    mutations = [
        ("trial_token", "wrong", "does not match schedule"),
        ("task_id", "T-WRONG", "does not match schedule"),
        ("duration_ms", "60001", "must be between"),
        ("completed", "false", "are inconsistent"),
        (
            "selected_candidate_codes",
            "C-NOT-IN-POOL",
            "selected candidate is not in task pool",
        ),
    ]
    for field, value, message in mutations:
        tampered_rows = deepcopy(rows)
        tampered_rows[0][field] = value
        tampered = tmp_path / f"tampered-{field}.csv"
        _write_csv(tampered, tampered_rows)
        with pytest.raises(ValueError, match=message):
            load_trial_csvs([tampered], key)

    duplicate_rows = deepcopy(rows)
    duplicate_rows[1]["trial_index"] = duplicate_rows[0]["trial_index"]
    duplicate = tmp_path / "duplicate.csv"
    _write_csv(duplicate, duplicate_rows)
    with pytest.raises(ValueError, match="duplicate trial_index"):
        load_trial_csvs([duplicate], key)

    incomplete = tmp_path / "incomplete.csv"
    _write_csv(incomplete, rows[:-1])
    with pytest.raises(ValueError, match="complete four trials"):
        load_trial_csvs([incomplete], key)

    invalid_codes = (
        "Alice",
        "YuMin",
        "alice@example.com",
        "real name",
        "P1",
        "P00000",
        "p01",
    )
    for index, invalid_code in enumerate(invalid_codes):
        invalid_rows = deepcopy(rows)
        for row in invalid_rows:
            row["participant_code"] = invalid_code
        invalid_participant = tmp_path / f"invalid-participant-{index}.csv"
        _write_csv(invalid_participant, invalid_rows)
        with pytest.raises(ValueError, match="assigned participant code"):
            load_trial_csvs([invalid_participant], key)

    for valid_code in ("P00", "P0000"):
        valid_rows = deepcopy(rows)
        for row in valid_rows:
            row["participant_code"] = valid_code
        valid_participant = tmp_path / f"valid-{valid_code}.csv"
        _write_csv(valid_participant, valid_rows)
        assert len(load_trial_csvs([valid_participant], key)) == 4

    timed_out_rows = deepcopy(rows)
    timed_out_rows[0].update(
        {
            "duration_ms": "60000",
            "first_selection_ms": "",
            "selected_candidate_codes": "",
            "selection_toggle_count": "0",
            "timed_out": "true",
            "completed": "false",
        }
    )
    timed_out = tmp_path / "timed-out.csv"
    _write_csv(timed_out, timed_out_rows)
    loaded_timeout = load_trial_csvs([timed_out], key)
    assert loaded_timeout[0]["timed_out"] is True
    assert loaded_timeout[0]["duration_ms"] == 60000


def test_aggregation_uses_participant_paired_capped_time_and_guardrails(
    tmp_path: Path,
) -> None:
    _task, key, _report_path, _metas_path = _build(tmp_path)
    label_paths: list[Path] = []
    for index, sequence_id in enumerate(("S1", "S2", "S3", "S4"), start=1):
        path = tmp_path / f"P{index:02}.csv"
        _write_csv(
            path,
            _rows_for_participant(
                key,
                participant=f"P{index:02}",
                sequence_id=sequence_id,
                baseline_duration=1000 + index,
                meong_duration=800 + index,
            ),
        )
        label_paths.append(path)
    loaded = load_trial_csvs(label_paths, key)
    report = aggregate_trials(
        key,
        loaded,
        source_key_sha256="a" * 64,
        label_artifacts=[
            {"artifact_index": index, "sha256": "b" * 64, "bytes": 100}
            for index in range(1, 5)
        ],
    )

    assert report["sample"]["participants"] == 4
    assert report["sample"]["trials"] == 16
    assert report["sample"]["balanced_four_participant_schedule_achieved"] is True
    assert report["conditions"]["baseline"]["completion_rate"] == 1.0
    assert (
        report["conditions"]["meongtamjeong"][
            "three_of_three_evidence_confirmed_rate_among_completed"
        ]
        == 1.0
    )
    paired = report["paired_participant_comparison"]
    assert paired["meongtamjeong_faster"] == 4
    assert paired["baseline_faster"] == 0
    assert paired["meongtamjeong_minus_baseline_ms"]["median"] == -200.0
    assert len(report["task_pool_audit"]) == 4
    assert all(
        row["local_image_eligible"] == 5
        and row["excluded_missing_or_invalid_local_image"] == 0
        for row in report["task_pool_audit"]
    )
    serialized = json.dumps(report, ensure_ascii=False)
    assert '"P01"' not in serialized
    assert "small_local_human_pilot" in report["study_status"]
    markdown = render_result_markdown(report)
    assert "모집단 효과를 측정하지 않습니다" in markdown
    assert "입양 적합성 판정이 아닙니다" in markdown

    with pytest.raises(ValueError, match="requires 3-5 participants"):
        aggregate_trials(
            key,
            loaded[:4],
            source_key_sha256="a" * 64,
            label_artifacts=[],
        )


def test_cli_generate_aggregate_and_check_round_trip(tmp_path: Path) -> None:
    report_path = tmp_path / "retrieval.json"
    metas_path = tmp_path / "metas.json"
    report_path.write_text(
        json.dumps(_report(), ensure_ascii=False),
        encoding="utf-8",
    )
    metas_path.write_text(
        json.dumps(_metas(tmp_path), ensure_ascii=False),
        encoding="utf-8",
    )
    output_dir = tmp_path / "study"
    generate_args = [
        "generate",
        "--report",
        str(report_path),
        "--metas",
        str(metas_path),
        "--project-root",
        str(tmp_path),
        "--output-dir",
        str(output_dir),
        "--query-ids",
        *QUERY_IDS,
        "--depth",
        "5",
        "--timeout-seconds",
        "60",
    ]
    assert cli.main(generate_args) == 0
    html_path = output_dir / "task.html"
    key_path = output_dir / "task_key.json"
    assert html_path.is_file()
    assert key_path.is_file()
    html = html_path.read_text(encoding="utf-8")
    assert "baseline-secret" not in html
    assert "meong-secret" not in html
    assert (
        cli.main(
            [
                "check",
                "--key",
                str(key_path),
                "--project-root",
                str(tmp_path),
            ]
        )
        == 0
    )

    key = json.loads(key_path.read_text(encoding="utf-8"))
    labels: list[Path] = []
    for index, sequence_id in enumerate(("S1", "S2", "S3"), start=1):
        label = tmp_path / f"participant-{index}.csv"
        _write_csv(
            label,
            _rows_for_participant(
                key,
                participant=f"P{index:02}",
                sequence_id=sequence_id,
            ),
        )
        labels.append(label)
    result_path = tmp_path / "result.json"
    markdown_path = tmp_path / "result.md"
    aggregate_args = [
        "aggregate",
        "--key",
        str(key_path),
        "--labels",
        *(str(path) for path in labels),
        "--project-root",
        str(tmp_path),
        "--json-out",
        str(result_path),
        "--markdown-out",
        str(markdown_path),
    ]
    assert cli.main(aggregate_args) == 0
    assert (
        cli.main(
            [
                "check",
                "--key",
                str(key_path),
                "--labels",
                *(str(path) for path in labels),
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

    tampered = json.loads(result_path.read_text(encoding="utf-8"))
    tampered["sample"]["participants"] = 999
    result_path.write_text(
        json.dumps(tampered, ensure_ascii=False),
        encoding="utf-8",
    )
    assert (
        cli.main(
            [
                "check",
                "--key",
                str(key_path),
                "--labels",
                *(str(path) for path in labels),
                "--project-root",
                str(tmp_path),
                "--result",
                str(result_path),
            ]
        )
        == 1
    )
