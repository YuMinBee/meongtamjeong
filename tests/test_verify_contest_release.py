from __future__ import annotations

import json
import subprocess
import zipfile
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import package_release
from scripts import verify_contest_release as gate


class RecordingRunner:
    def __init__(
        self,
        *,
        failures: dict[str, int] | None = None,
        dirty: bool = False,
        secret: str = "",
    ) -> None:
        self.failures = failures or {}
        self.dirty = dirty
        self.secret = secret
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def __call__(self, argv: list[str], **kwargs: object) -> object:
        self.calls.append((list(argv), dict(kwargs)))
        command_text = " ".join(argv)
        return_code = next(
            (code for marker, code in self.failures.items() if marker in command_text),
            0,
        )
        stdout = b""
        if argv[:2] == ["git", "status"] and self.dirty:
            stdout = (
                f"?? private-{self.secret}.txt\n" if self.secret else "?? changed.py\n"
            ).encode()
        elif kwargs.get("stdout") == subprocess.PIPE and self.secret:
            stdout = f"API_KEY={self.secret}\n".encode()
        if (
            return_code == 0
            and any(item.endswith("package_release.py") for item in argv)
            and "--output" in argv
        ):
            cwd = Path(str(kwargs["cwd"]))
            output = cwd / argv[argv.index("--output") + 1]
            tag = argv[argv.index("--require-tag") + 1]
            profile = (
                argv[argv.index("--profile") + 1]
                if "--profile" in argv
                else gate.FULL_RELEASE_PROFILE
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            manifest = {
                "commit_sha": "a" * 40,
                "files": [],
                "profile": profile,
                "project": "meongtamjeong",
                "release_tag": tag,
                "schema_version": 1,
                "tag_object_id": "b" * 40,
            }
            with zipfile.ZipFile(output, "w") as archive:
                archive.writestr(
                    gate.RELEASE_MANIFEST_PATH,
                    json.dumps(manifest, sort_keys=True) + "\n",
                )
        return SimpleNamespace(
            returncode=return_code,
            stdout=stdout,
            stderr=f"secret={self.secret}".encode(),
        )


def _step(report: dict[str, object], name: str) -> dict[str, object]:
    steps = report["steps"]
    assert isinstance(steps, list)
    return next(item for item in steps if item["name"] == name)


def test_public_text_gate_tests_match_packaged_test_allowlist() -> None:
    assert (
        frozenset(gate.PUBLIC_TEXT_TESTS) == package_release.PUBLIC_TEXT_RETAINED_TESTS
    )


def test_fixed_suite_uses_current_date_and_required_argv_lists(tmp_path: Path) -> None:
    runner = RecordingRunner()
    report = gate.run_release_gate(
        repo=tmp_path,
        reference_date=date(2026, 8, 26),
        runner=runner,
    )

    commands = [call[0] for call in runner.calls]
    readiness = next(
        argv for argv in commands if "check_contest_readiness.py" in " ".join(argv)
    )
    assert readiness[-9:] == [
        "--metas",
        "data/dog_metas.json",
        "--index",
        "data/dog_faiss.index",
        "--reference-date",
        "2026-08-26",
        "--max-age-days",
        "7",
        "--strict",
    ]
    evaluation_commands = [
        argv for argv in commands if Path(argv[1]).name.startswith("evaluate_")
    ]
    assert len(evaluation_commands) == 8
    assert any(
        Path(argv[1]).name == "evaluate_query_robustness.py"
        for argv in evaluation_commands
    )
    assert any(
        Path(argv[1]).name == "evaluate_query_holdout.py"
        for argv in evaluation_commands
    )
    assert any(
        Path(argv[1]).name == "evaluate_query_holdout_v3.py"
        for argv in evaluation_commands
    )
    assert all(argv[-1] == "--check" for argv in evaluation_commands)
    assert any(
        argv[-2:]
        == ["scripts/portal_candidate_selection_study.py", "validate-template"]
        for argv in commands
    )
    assert any(argv[-3:] == ["-m", "pip", "check"] for argv in commands)
    assert any(argv[-1] == "scripts/check_dependency_snapshot.py" for argv in commands)
    assert any(
        argv[-6:] == ["-m", "ruff", "check", "app", "scripts", "tests"]
        for argv in commands
    )
    assert any(argv[-3:] == ["-m", "pytest", "-q"] for argv in commands)
    assert any(
        "smoke_full_runtime.py" in argument for argv in commands for argument in argv
    )
    history = next(
        argv
        for argv in commands
        if any(item.endswith("scan_git_history_secrets.py") for item in argv)
    )
    package = next(
        argv
        for argv in commands
        if any(item.endswith("package_release.py") for item in argv)
    )
    assert history[-1] == "--strict-user-refs"
    assert package[-1] == "--check-only"
    assert "--allow-dirty" not in package
    assert any(argv[:2] == ["git", "status"] for argv in commands)
    assert report["ready"] is False
    assert report["mode"] == "preflight"
    assert report["release_identity"] is None

    for _, kwargs in runner.calls:
        assert kwargs["shell"] is False
        assert kwargs["stderr"] == subprocess.DEVNULL
        assert kwargs["stdin"] == subprocess.DEVNULL


def test_default_date_comes_from_local_calendar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gate, "current_local_date", lambda: date(2026, 11, 3))
    runner = RecordingRunner()
    report = gate.run_release_gate(repo=tmp_path, runner=runner)

    assert report["reference_date"] == "2026-11-03"
    readiness = runner.calls[0][0]
    index = readiness.index("--reference-date")
    assert readiness[index + 1] == "2026-11-03"


def test_failures_do_not_short_circuit_and_sensitive_output_is_never_reported(
    tmp_path: Path,
) -> None:
    secret = "do-not-print-this-value"
    runner = RecordingRunner(
        failures={
            "check_contest_readiness.py": 2,
            "scan_git_history_secrets.py": 1,
            "package_release.py": 1,
        },
        dirty=True,
        secret=secret,
    )
    report = gate.run_release_gate(
        repo=tmp_path,
        reference_date=date(2026, 8, 26),
        runner=runner,
    )
    serialized = json.dumps(report, ensure_ascii=False)

    assert len(runner.calls) == len(gate.build_steps(reference_date=date(2026, 8, 26)))
    assert report["ready"] is False
    assert _step(report, "contest_readiness")["exit_code"] == 2
    assert _step(report, "git_history_secret_scan")["exit_code"] == 1
    assert _step(report, "release_package_check")["exit_code"] == 1
    assert _step(report, "git_clean")["exit_code"] == 1
    assert secret not in serialized
    assert "stderr" not in serialized
    assert "API_KEY" not in serialized


def test_skip_slow_is_preflight_only_and_never_ready(tmp_path: Path) -> None:
    runner = RecordingRunner()
    report = gate.run_release_gate(
        repo=tmp_path,
        reference_date=date(2026, 10, 10),
        skip_slow=True,
        runner=runner,
    )
    commands = [" ".join(argv) for argv, _ in runner.calls]

    assert report["mode"] == "preflight"
    assert report["preflight_passed"] is True
    assert report["ready"] is False
    assert _step(report, "pytest")["status"] == "skipped"
    assert _step(report, "full_runtime_smoke")["status"] == "skipped"
    assert not any("-m pytest" in command for command in commands)
    assert not any("smoke_full_runtime.py" in command for command in commands)
    assert _step(report, "release_package_check")["status"] == "passed"
    assert _step(report, "git_history_secret_scan")["status"] == "passed"


def test_required_tag_builds_and_verifies_archive_identity(tmp_path: Path) -> None:
    runner = RecordingRunner()
    tag = "contest-2026-final"
    report = gate.run_release_gate(
        repo=tmp_path,
        reference_date=date(2026, 11, 3),
        required_tag=tag,
        runner=runner,
    )
    commands = [argv for argv, _ in runner.calls]
    package = next(argv for argv in commands if "package_release.py" in " ".join(argv))

    assert package[package.index("--require-tag") + 1] == tag
    assert "--check-only" not in package
    assert (
        package[package.index("--output") + 1]
        == gate.RELEASE_ARCHIVE_PATHS[gate.FULL_RELEASE_PROFILE]
    )
    archive_check = next(
        argv for argv in commands if "verify_release_archive.py" in " ".join(argv)
    )
    assert archive_check[archive_check.index("--expected-tag") + 1] == tag
    assert report["required_tag_enforced"] is True
    assert report["mode"] == "release"
    assert report["ready"] is True
    assert report["release_identity"] == {
        "archive_path": gate.RELEASE_ARCHIVE_PATHS[gate.FULL_RELEASE_PROFILE],
        "archive_sha256": report["release_identity"]["archive_sha256"],
        "commit_sha": "a" * 40,
        "profile": gate.FULL_RELEASE_PROFILE,
        "tag": tag,
        "tag_object_id": "b" * 40,
    }
    assert len(report["release_identity"]["archive_sha256"]) == 64
    assert _step(report, "release_archive_runtime")["status"] == "passed"
    assert _step(report, "release_archive_identity")["status"] == "passed"


def test_public_text_gate_uses_staged_artifacts_and_marks_heldout_na(
    tmp_path: Path,
) -> None:
    runner = RecordingRunner()
    report = gate.run_release_gate(
        repo=tmp_path,
        reference_date=date(2026, 7, 26),
        profile=gate.PUBLIC_TEXT_RELEASE_PROFILE,
        runner=runner,
    )

    commands = [argv for argv, _ in runner.calls]
    command_text = [" ".join(argv) for argv in commands]
    assert any("build_public_text_release.py" in value for value in command_text)
    assert any(
        "evaluate_retrieval.py" in value
        and "dist/public-text-release-gate/dog_faiss.index" in value
        for value in command_text
    )
    assert any(
        "evaluate_query_robustness.py" in value
        and "dist/public-text-release-gate/dog_metas.json" in value
        for value in command_text
    )
    summary = next(
        argv
        for argv in commands
        if "build_public_text_evaluation_summary.py" in " ".join(argv)
    )
    package = next(argv for argv in commands if "package_release.py" in " ".join(argv))
    assert package[package.index("--profile") + 1] == "public-text-only"
    assert package[package.index("--public-evidence-dir") + 1] == gate.PUBLIC_TEXT_STAGE
    assert commands.index(summary) < commands.index(package)
    pytest_argv = next(argv for argv in commands if argv[1:4] == ["-m", "pytest", "-q"])
    assert tuple(pytest_argv[4:]) == gate.PUBLIC_TEXT_TESTS
    assert any(
        argv[-2:]
        == ["scripts/portal_candidate_selection_study.py", "validate-template"]
        for argv in commands
    )
    smoke = next(
        argv
        for argv in commands
        if any(item == "scripts/smoke_full_runtime.py" for item in argv)
    )
    assert smoke[smoke.index("--profile") + 1] == "public-text-only"

    heldout = _step(report, "evaluation_heldout_image_retrieval")
    assert heldout["status"] == "not_applicable"
    assert heldout["exit_code"] is None
    assert "no image/crop vectors" in heldout["message"]
    assert report["profile"] == "public-text-only"
    for name in (
        "evaluation_query_holdout_v2_historical_integrity",
        "evaluation_query_holdout_v3_independent",
    ):
        holdout = _step(report, name)
        assert holdout["status"] == "not_applicable"
        assert "full multimodal artifact" in holdout["message"]
    assert report["summary"]["not_applicable"] == 3
    assert report["ready"] is False
    assert report["mode"] == "preflight"
    assert not any(
        "evaluate_heldout_image_retrieval.py" in value for value in command_text
    )
    assert not any("evaluate_query_holdout" in value for value in command_text)


def test_timeout_and_missing_executable_fail_closed(tmp_path: Path) -> None:
    calls = 0

    def runner(argv: list[str], **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise subprocess.TimeoutExpired(argv, timeout=10, output=b"secret")
        if calls == 2:
            raise FileNotFoundError("secret executable detail")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"secret")

    report = gate.run_release_gate(
        repo=tmp_path,
        reference_date=date(2026, 8, 26),
        runner=runner,
    )

    assert report["ready"] is False
    assert _step(report, "contest_readiness")["exit_code"] == 124
    assert _step(report, "evaluation_retrieval")["exit_code"] == 127
    serialized = json.dumps(report)
    assert "secret executable detail" not in serialized
    assert 'b"secret"' not in serialized
    assert calls == len(gate.build_steps(reference_date=date(2026, 8, 26)))


def test_json_output_must_not_modify_non_dist_repository_paths(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="only under dist"):
        gate._safe_json_target(tmp_path, tmp_path / "docs" / "gate.json")

    assert (
        gate._safe_json_target(tmp_path, tmp_path / "dist" / "gate.json")
        == (tmp_path / "dist" / "gate.json").resolve()
    )


def test_main_emits_json_only_and_skip_slow_exits_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "never-echo-this"
    runner = RecordingRunner(secret=secret)
    monkeypatch.setattr(gate.subprocess, "run", runner)
    monkeypatch.setattr(gate, "current_local_date", lambda: date(2026, 8, 26))

    exit_code = gate.main(
        [
            "--repo",
            str(tmp_path),
            "--skip-slow",
            "--json-out",
            str(tmp_path / "dist" / "release-gate.json"),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    written = json.loads(
        (tmp_path / "dist" / "release-gate.json").read_text(encoding="utf-8")
    )

    assert exit_code == 1
    assert captured.err == ""
    assert payload == written
    assert payload["ready"] is False
    assert payload["mode"] == "preflight"
    assert secret not in captured.out


def test_atomic_json_write_failure_is_a_gate_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(gate.subprocess, "run", RecordingRunner())
    monkeypatch.setattr(
        gate,
        "_write_atomic",
        lambda path, content: (_ for _ in ()).throw(OSError("private detail")),
    )

    exit_code = gate.main(
        [
            "--repo",
            str(tmp_path),
            "--json-out",
            str(tmp_path / "dist" / "release-gate.json"),
        ]
    )
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert report["ready"] is False
    assert _step(report, "json_output")["status"] == "failed"
    assert "private detail" not in json.dumps(report)
