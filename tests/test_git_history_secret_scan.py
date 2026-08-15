from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts import scan_git_history_secrets as history_scan


SECRET_VALUE = "history-secret-value-that-must-never-print"


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "History Scan Test")
    return repo


def _commit_all(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _write_allowlist(repo: Path, entries: list[dict[str, object]]) -> Path:
    path = repo / "security" / "git-history-secret-allowlist.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"schema_version": 1, "entries": entries},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _allowlist_entry(
    *,
    blob: str,
    path: str,
    line: int = 1,
    kind: str = "environment-secret",
    ref_scope: str = "user",
) -> dict[str, object]:
    return {
        "blob": blob,
        "path": path,
        "line": line,
        "kind": kind,
        "ref_scope": ref_scope,
        "rationale": "Reviewed synthetic test fixture; no real credential is present.",
        "reviewed_at": "2026-01-01",
    }


def test_clean_history_has_no_findings(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    (repo / "settings.env.example").write_text(
        "ANIMAL_API_KEY=your-api-key\n",
        encoding="utf-8",
    )
    _commit_all(repo, "clean example")

    result = history_scan.scan_git_history(repo)

    assert result.clean is True
    assert result.findings == ()
    assert result.stats.reachable_blobs == 1
    assert result.stats.scanned_blobs == 1
    assert result.stats.skipped_large_blobs == 0


def test_deleted_historical_env_is_still_scanned(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    env_file = repo / ".env"
    env_file.write_text(
        f"ANIMAL_API_KEY={SECRET_VALUE}\n",
        encoding="utf-8",
    )
    secret_commit = _commit_all(repo, "temporarily tracked env")
    env_file.unlink()
    (repo / "README.md").write_text("clean current tree\n", encoding="utf-8")
    _commit_all(repo, "remove env")

    result = history_scan.scan_git_history(repo)

    assert result.clean is False
    assert len(result.findings) == 1
    finding = result.findings[0]
    assert finding.commit == secret_commit[:12]
    assert len(finding.blob) == 12
    assert finding.path == ".env"
    assert finding.line == 1
    assert finding.kind == "environment-secret"
    assert finding.ref_scope == "user"
    serialized = json.dumps(result.as_dict(), ensure_ascii=False)
    assert SECRET_VALUE not in serialized


def test_env_example_keeps_literal_semantics_when_blob_is_scanned_once(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path)
    identifier_shaped_value = "".join(
        ("Synthetic", "Credential", "Identifier", "123456789")
    )
    env_file = repo / ".env.example"
    env_file.write_text(
        f"ANIMAL_API_KEY={identifier_shaped_value}\n",
        encoding="utf-8",
    )
    secret_commit = _commit_all(repo, "historical env example")

    result = history_scan.scan_git_history(repo)

    assert result.clean is False
    assert len(result.findings) == 1
    finding = result.findings[0]
    assert finding.commit == secret_commit[:12]
    assert finding.path == ".env.example"
    assert finding.kind == "environment-secret"
    assert identifier_shaped_value not in json.dumps(result.as_dict())


def test_same_blob_in_many_commits_is_scanned_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _init_repo(tmp_path)
    secret_data = f"ANIMAL_API_KEY={SECRET_VALUE}\n".encode()
    (repo / "config.txt").write_bytes(secret_data)
    _commit_all(repo, "add config")
    (repo / "notes.txt").write_text("first note\n", encoding="utf-8")
    _commit_all(repo, "leave config unchanged")
    (repo / "notes.txt").write_text("second note\n", encoding="utf-8")
    _commit_all(repo, "still leave config unchanged")

    original = history_scan.scan_secrets
    calls_for_secret_blob = 0

    def counting_scan(path: str, data: bytes):
        nonlocal calls_for_secret_blob
        if data == secret_data:
            calls_for_secret_blob += 1
        return original(path, data)

    monkeypatch.setattr(history_scan, "scan_secrets", counting_scan)
    result = history_scan.scan_git_history(repo)

    assert calls_for_secret_blob == 1
    assert len(result.findings) == 1
    assert result.stats.finding_blobs == 1
    assert result.stats.scanned_blobs == result.stats.reachable_blobs


def test_cli_json_never_contains_secret_value_and_is_deterministic(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = _init_repo(tmp_path)
    (repo / ".env").write_text(
        f"ANIMAL_API_KEY={SECRET_VALUE}\n",
        encoding="utf-8",
    )
    _commit_all(repo, "secret for scanner test")
    output = tmp_path / "reports" / "history.json"

    first_exit = history_scan.main(
        [
            "--repo",
            str(repo),
            "--json-out",
            str(output),
            "--strict",
        ]
    )
    first_io = capsys.readouterr()
    first_file = output.read_text(encoding="utf-8")
    second_exit = history_scan.main(
        [
            "--repo",
            str(repo),
            "--json-out",
            str(output),
            "--strict",
        ]
    )
    second_io = capsys.readouterr()
    second_file = output.read_text(encoding="utf-8")

    assert first_exit == second_exit == 1
    combined = (
        first_io.out
        + first_io.err
        + second_io.out
        + second_io.err
        + first_file
        + second_file
    )
    assert SECRET_VALUE not in combined
    assert first_io.err == second_io.err == ""
    assert first_io.out == second_io.out
    assert first_file == second_file
    payload = json.loads(first_file)
    assert set(payload["findings"][0]) == {
        "blob",
        "commit",
        "kind",
        "line",
        "path",
        "ref_scope",
    }


def test_large_blob_is_skipped_with_explicit_stats(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    (repo / "large.bin").write_bytes(b"\0" + b"x" * 127)
    _commit_all(repo, "large binary")

    result = history_scan.scan_git_history(repo, max_blob_bytes=64)

    assert result.clean is True
    assert result.stats.reachable_blobs == 1
    assert result.stats.scanned_blobs == 0
    assert result.stats.skipped_large_blobs == 1
    assert result.stats.skipped_large_bytes == 128


def test_binary_blob_is_checked_for_known_token_patterns(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    fake_token = b"AKIA" + (b"A" * 16)
    (repo / "weights.bin").write_bytes(b"\0model-prefix:" + fake_token)
    _commit_all(repo, "binary fixture")

    result = history_scan.scan_git_history(repo)

    assert len(result.findings) == 1
    assert result.findings[0].kind == "aws-access-key"
    assert result.stats.binary_blobs_scanned == 1
    assert fake_token.decode() not in json.dumps(result.as_dict())


def test_json_output_inside_git_metadata_is_rejected(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = _init_repo(tmp_path)
    (repo / "README.md").write_text("clean\n", encoding="utf-8")
    _commit_all(repo, "initial")

    exit_code = history_scan.main(
        [
            "--repo",
            str(repo),
            "--json-out",
            str(repo / ".git" / "scan.json"),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert captured.out == ""
    assert SECRET_VALUE not in captured.err
    assert not (repo / ".git" / "scan.json").exists()


def test_blob_reachable_from_direct_tree_ref_keeps_safe_path(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path)
    (repo / "only-in-tree.env").write_text(
        f"ANIMAL_API_KEY={SECRET_VALUE}\n",
        encoding="utf-8",
    )
    _git(repo, "add", "only-in-tree.env")
    tree_id = _git(repo, "write-tree")
    _git(repo, "update-ref", "refs/codex/test/direct-tree", tree_id)

    result = history_scan.scan_git_history(repo)

    assert len(result.findings) == 1
    assert result.findings[0].commit is None
    assert result.findings[0].path == "only-in-tree.env"
    assert result.findings[0].ref_scope == "tool"
    assert result.stats.tool_ref_roots == 1
    assert result.stats.tool_only_reachable_blobs == 1
    assert SECRET_VALUE not in json.dumps(result.as_dict())


def test_strict_user_refs_ignores_tool_only_finding(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = _init_repo(tmp_path)
    (repo / "README.md").write_text("clean user history\n", encoding="utf-8")
    _commit_all(repo, "clean user history")
    (repo / "tool-only.env").write_text(
        f"ANIMAL_API_KEY={SECRET_VALUE}\n",
        encoding="utf-8",
    )
    _git(repo, "add", "tool-only.env")
    tree_id = _git(repo, "write-tree")
    _git(repo, "update-ref", "refs/codex/test/tool-only", tree_id)

    result = history_scan.scan_git_history(repo)

    assert result.strict_ready is False
    assert result.strict_user_ready is True
    assert result.as_dict()["user_refs_ok"] is True
    exit_code = history_scan.main(["--repo", str(repo), "--strict-user-refs"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert SECRET_VALUE not in captured.out + captured.err


def test_strict_user_refs_still_blocks_user_finding(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = _init_repo(tmp_path)
    (repo / ".env").write_text(
        f"ANIMAL_API_KEY={SECRET_VALUE}\n",
        encoding="utf-8",
    )
    _commit_all(repo, "user secret")

    result = history_scan.scan_git_history(repo)

    assert result.strict_user_ready is False
    exit_code = history_scan.main(["--repo", str(repo), "--strict-user-refs"])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert SECRET_VALUE not in captured.out + captured.err


def test_exact_reviewed_allowlist_entry_is_suppressed(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    (repo / ".env").write_text(
        f"ANIMAL_API_KEY={SECRET_VALUE}\n",
        encoding="utf-8",
    )
    _commit_all(repo, "synthetic finding")
    blob = _git(repo, "rev-parse", "HEAD:.env")
    _write_allowlist(repo, [_allowlist_entry(blob=blob, path=".env")])

    result = history_scan.scan_git_history(repo)

    assert result.strict_ready is True
    assert result.findings == ()
    assert len(result.suppressed_findings) == 1
    assert result.suppressed_findings[0].finding.blob == blob[:12]
    assert result.stale_allowlist_entries == ()
    assert result.stats.suppressed_findings == 1


def test_allowlist_match_is_exact_and_mismatch_is_stale(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    (repo / ".env").write_text(
        f"ANIMAL_API_KEY={SECRET_VALUE}\n",
        encoding="utf-8",
    )
    _commit_all(repo, "synthetic finding")
    blob = _git(repo, "rev-parse", "HEAD:.env")
    _write_allowlist(
        repo,
        [_allowlist_entry(blob=blob, path=".env", line=2)],
    )

    result = history_scan.scan_git_history(repo)

    assert len(result.findings) == 1
    assert result.suppressed_findings == ()
    assert len(result.stale_allowlist_entries) == 1
    assert result.strict_ready is False


def test_stale_allowlist_entry_makes_strict_cli_fail(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = _init_repo(tmp_path)
    (repo / "README.md").write_text("clean\n", encoding="utf-8")
    _commit_all(repo, "clean history")
    blob = _git(repo, "rev-parse", "HEAD:README.md")
    _write_allowlist(
        repo,
        [_allowlist_entry(blob=blob, path="README.md")],
    )

    exit_code = history_scan.main(["--repo", str(repo), "--strict"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert SECRET_VALUE not in captured.out + captured.err
    payload = json.loads(captured.out)
    assert payload["stats"]["stale_allowlist_entries"] == 1
    assert payload["findings"] == []


def test_malformed_allowlist_is_rejected(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    (repo / "README.md").write_text("clean\n", encoding="utf-8")
    _commit_all(repo, "clean history")
    path = repo / "security" / "git-history-secret-allowlist.json"
    path.parent.mkdir()
    path.write_text(
        '{"schema_version":1,"entries":[],"entries":[]}',
        encoding="utf-8",
    )

    with pytest.raises(
        history_scan.HistoryScanError,
        match="duplicate JSON key",
    ):
        history_scan.scan_git_history(repo)


def test_tool_allowlist_is_inactive_when_tool_refs_are_absent(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path)
    (repo / "README.md").write_text("clean\n", encoding="utf-8")
    _commit_all(repo, "clean history")
    _write_allowlist(
        repo,
        [
            _allowlist_entry(
                blob="0" * 40,
                path="tool-only-fixture.txt",
                ref_scope="tool",
            )
        ],
    )

    result = history_scan.scan_git_history(repo)

    assert result.strict_ready is True
    assert result.stale_allowlist_entries == ()
    assert len(result.inactive_tool_allowlist_entries) == 1
    assert result.stats.tool_ref_roots == 0
