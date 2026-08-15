"""Scan every reachable Git-history blob for likely secrets.

The scanner deliberately reports only safe metadata.  Matched values, blob
contents, and Git stderr are never included in normal output or exceptions.
Each reachable blob object is scanned at most once, even when the same bytes
appear in many commits or paths.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

if __package__:
    from scripts.package_release import SecretFinding, scan_secrets
else:
    from package_release import SecretFinding, scan_secrets


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ALLOWLIST = Path("security") / "git-history-secret-allowlist.json"
DEFAULT_MAX_BLOB_BYTES = 64 * 1024 * 1024
SHORT_SHA_LENGTH = 12
_OBJECT_ID_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_KIND_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_TOOL_REF_PREFIX = b"refs/codex/"
_ALLOWLIST_ENTRY_FIELDS = {
    "blob",
    "kind",
    "line",
    "path",
    "rationale",
    "ref_scope",
    "reviewed_at",
}
_MAX_ALLOWLIST_BYTES = 1024 * 1024


class HistoryScanError(RuntimeError):
    """The Git history could not be scanned safely."""


@dataclass(frozen=True)
class HistorySecretFinding:
    """Safe metadata for one finding in one unique historical blob."""

    commit: str | None
    blob: str
    path: str | None
    line: int
    kind: str
    ref_scope: str

    def as_dict(self) -> dict[str, object]:
        return {
            "blob": self.blob,
            "commit": self.commit,
            "kind": self.kind,
            "line": self.line,
            "path": self.path,
            "ref_scope": self.ref_scope,
        }


@dataclass(frozen=True)
class SuppressedHistoryFinding:
    """A finding suppressed by one exact, reviewed allowlist entry."""

    finding: HistorySecretFinding
    rationale: str
    reviewed_at: str

    def as_dict(self) -> dict[str, object]:
        return {
            **self.finding.as_dict(),
            "rationale": self.rationale,
            "reviewed_at": self.reviewed_at,
        }


@dataclass(frozen=True)
class AllowlistEntry:
    """A validated exact-match exception; no matched value is retained."""

    blob: str
    path: str
    line: int
    kind: str
    ref_scope: str
    rationale: str
    reviewed_at: str

    @property
    def key(self) -> tuple[str, str, int, str, str]:
        return (
            self.blob,
            self.path,
            self.line,
            self.kind,
            self.ref_scope,
        )

    def safe_dict(self, *, status: str) -> dict[str, object]:
        return {
            "blob": _short_sha(self.blob),
            "kind": self.kind,
            "line": self.line,
            "path": self.path,
            "rationale": self.rationale,
            "ref_scope": self.ref_scope,
            "reviewed_at": self.reviewed_at,
            "status": status,
        }


@dataclass(frozen=True)
class _LocatedFinding:
    """Internal full-SHA finding used only for exact matching."""

    blob: str
    commit: str | None
    path: str | None
    line: int
    kind: str
    ref_scope: str

    @property
    def key(self) -> tuple[str, str, int, str, str] | None:
        if self.path is None:
            return None
        return (
            self.blob,
            self.path,
            self.line,
            self.kind,
            self.ref_scope,
        )

    def safe_finding(self) -> HistorySecretFinding:
        return HistorySecretFinding(
            commit=_short_sha(self.commit) if self.commit is not None else None,
            blob=_short_sha(self.blob),
            path=self.path,
            line=self.line,
            kind=self.kind,
            ref_scope=self.ref_scope,
        )


@dataclass(frozen=True)
class HistoryScanStats:
    """Non-sensitive counters that describe scan coverage."""

    reachable_objects: int
    reachable_commits: int
    reachable_blobs: int
    scanned_blobs: int
    scanned_bytes: int
    binary_blobs_scanned: int
    skipped_large_blobs: int
    skipped_large_bytes: int
    finding_blobs: int
    detected_findings: int
    findings: int
    suppressed_findings: int
    allowlist_entries: int
    stale_allowlist_entries: int
    inactive_tool_allowlist_entries: int
    max_blob_bytes: int
    user_ref_roots: int
    tool_ref_roots: int
    user_reachable_blobs: int
    tool_reachable_blobs: int
    tool_only_reachable_blobs: int
    shared_reachable_blobs: int
    unknown_scope_reachable_blobs: int

    def as_dict(self) -> dict[str, object]:
        return {
            "binary_blobs_scanned": self.binary_blobs_scanned,
            "deduplicated_by_blob_sha": True,
            "detected_findings": self.detected_findings,
            "finding_blobs": self.finding_blobs,
            "findings": self.findings,
            "allowlist_entries": self.allowlist_entries,
            "inactive_tool_allowlist_entries": (self.inactive_tool_allowlist_entries),
            "max_blob_bytes": self.max_blob_bytes,
            "reachable_blobs": self.reachable_blobs,
            "reachable_commits": self.reachable_commits,
            "reachable_objects": self.reachable_objects,
            "scanned_blobs": self.scanned_blobs,
            "scanned_bytes": self.scanned_bytes,
            "shared_reachable_blobs": self.shared_reachable_blobs,
            "skipped_large_blobs": self.skipped_large_blobs,
            "skipped_large_bytes": self.skipped_large_bytes,
            "stale_allowlist_entries": self.stale_allowlist_entries,
            "suppressed_findings": self.suppressed_findings,
            "tool_only_reachable_blobs": self.tool_only_reachable_blobs,
            "tool_reachable_blobs": self.tool_reachable_blobs,
            "tool_ref_roots": self.tool_ref_roots,
            "unknown_scope_reachable_blobs": (self.unknown_scope_reachable_blobs),
            "user_reachable_blobs": self.user_reachable_blobs,
            "user_ref_roots": self.user_ref_roots,
        }


@dataclass(frozen=True)
class HistoryScanResult:
    """Deterministic, value-free result of a Git-history scan."""

    findings: tuple[HistorySecretFinding, ...]
    suppressed_findings: tuple[SuppressedHistoryFinding, ...]
    stale_allowlist_entries: tuple[AllowlistEntry, ...]
    inactive_tool_allowlist_entries: tuple[AllowlistEntry, ...]
    stats: HistoryScanStats

    @property
    def clean(self) -> bool:
        return not self.findings and not self.stale_allowlist_entries

    @property
    def strict_ready(self) -> bool:
        return self.clean and self.stats.skipped_large_blobs == 0

    @property
    def strict_user_ready(self) -> bool:
        """Whether user-controlled refs are clean enough for a release.

        Local ``refs/codex/*`` snapshots are still scanned and reported, but
        they are not part of a pushed contest release. Mixed blobs remain
        blocking because they are also reachable from a user-controlled ref.
        Oversized unscanned blobs stay conservative and block both modes.
        """

        blocking_scopes = {"mixed", "user"}
        return (
            not any(finding.ref_scope in blocking_scopes for finding in self.findings)
            and not any(
                entry.ref_scope in blocking_scopes
                for entry in self.stale_allowlist_entries
            )
            and self.stats.skipped_large_blobs == 0
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "findings": [finding.as_dict() for finding in self.findings],
            "inactive_tool_allowlist_entries": [
                entry.safe_dict(status="inactive-tool-scope")
                for entry in self.inactive_tool_allowlist_entries
            ],
            "ok": self.strict_ready,
            "user_refs_ok": self.strict_user_ready,
            "stale_allowlist_entries": [
                entry.safe_dict(status="stale")
                for entry in self.stale_allowlist_entries
            ],
            "stats": self.stats.as_dict(),
            "suppressed_findings": [
                finding.as_dict() for finding in self.suppressed_findings
            ],
        }


def _run_git(
    repo: Path,
    *args: str,
    input_data: bytes | None = None,
) -> bytes:
    """Run Git without ever relaying its stderr."""

    environment = os.environ.copy()
    environment.update(
        {
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_PAGER": "cat",
            "LC_ALL": "C",
        }
    )
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=False,
            input=input_data,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
        )
    except FileNotFoundError as exc:
        raise HistoryScanError("Git executable was not found") from exc
    except OSError as exc:
        raise HistoryScanError("Git could not be executed") from exc
    if completed.returncode != 0:
        raise HistoryScanError(
            f"Git command failed with exit code {completed.returncode}"
        )
    return completed.stdout


def _resolve_repo(repo: Path) -> Path:
    try:
        root = repo.expanduser().resolve(strict=True)
    except OSError as exc:
        raise HistoryScanError("repository path is unavailable") from exc
    if not root.is_dir():
        raise HistoryScanError("repository path is not a directory")
    inside = _run_git(root, "rev-parse", "--is-inside-work-tree").strip()
    bare = _run_git(root, "rev-parse", "--is-bare-repository").strip()
    if inside != b"true" and bare != b"true":
        raise HistoryScanError("path is not a Git repository")
    return root


def _validate_object_id(raw: bytes) -> str:
    try:
        object_id = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise HistoryScanError("Git returned an invalid object identifier") from exc
    if not _OBJECT_ID_RE.fullmatch(object_id):
        raise HistoryScanError("Git returned an invalid object identifier")
    return object_id


def _reachable_object_ids(repo: Path) -> list[str]:
    raw = _run_git(
        repo,
        "rev-list",
        "--objects",
        "--all",
        "--no-object-names",
    )
    object_ids: set[str] = set()
    for line in raw.splitlines():
        if line:
            object_ids.add(_validate_object_id(line.strip()))
    return sorted(object_ids)


def _ref_roots(repo: Path) -> tuple[set[str], set[str]]:
    """Split all refs into standard/user roots and Codex-internal roots."""

    raw = _run_git(
        repo,
        "for-each-ref",
        "--format=%(objectname)%00%(refname)",
    )
    user_roots: set[str] = set()
    tool_roots: set[str] = set()
    for line in raw.splitlines():
        pieces = line.split(b"\0", 1)
        if len(pieces) != 2:
            raise HistoryScanError("Git returned invalid ref metadata")
        object_id = _validate_object_id(pieces[0])
        refname = pieces[1]
        if not refname.startswith(b"refs/"):
            raise HistoryScanError("Git returned invalid ref metadata")
        if refname.startswith(_TOOL_REF_PREFIX):
            tool_roots.add(object_id)
        else:
            user_roots.add(object_id)
    return user_roots, tool_roots


def _reachable_from_roots(repo: Path, roots: set[str]) -> set[str]:
    if not roots:
        return set()
    request = ("\n".join(sorted(roots)) + "\n").encode("ascii")
    raw = _run_git(
        repo,
        "rev-list",
        "--objects",
        "--no-object-names",
        "--stdin",
        input_data=request,
    )
    result: set[str] = set()
    for line in raw.splitlines():
        if line:
            result.add(_validate_object_id(line.strip()))
    return result


def _blob_ref_scope(
    blob: str,
    *,
    user_objects: set[str],
    tool_objects: set[str],
) -> str:
    in_user = blob in user_objects
    in_tool = blob in tool_objects
    if in_user and in_tool:
        return "mixed"
    if in_tool:
        return "tool"
    if in_user:
        return "user"
    return "unknown"


def _object_info(
    repo: Path,
    object_ids: Sequence[str],
) -> dict[str, tuple[str, int]]:
    if not object_ids:
        return {}
    request = ("\n".join(object_ids) + "\n").encode("ascii")
    raw = _run_git(
        repo,
        "cat-file",
        "--batch-check=%(objectname) %(objecttype) %(objectsize)",
        input_data=request,
    )
    result: dict[str, tuple[str, int]] = {}
    for line in raw.splitlines():
        pieces = line.split(b" ")
        if len(pieces) != 3:
            raise HistoryScanError("Git returned invalid object metadata")
        object_id = _validate_object_id(pieces[0])
        try:
            object_type = pieces[1].decode("ascii")
            size = int(pieces[2])
        except (UnicodeDecodeError, ValueError) as exc:
            raise HistoryScanError("Git returned invalid object metadata") from exc
        if size < 0 or object_type not in {"blob", "commit", "tag", "tree"}:
            raise HistoryScanError("Git returned invalid object metadata")
        result[object_id] = (object_type, size)
    if set(result) != set(object_ids):
        raise HistoryScanError("Git object enumeration was incomplete")
    return result


def _scan_unique_blobs(
    repo: Path,
    blobs: Sequence[tuple[str, int]],
    *,
    max_blob_bytes: int,
    scan_paths: Mapping[str, str],
) -> tuple[
    dict[str, tuple[SecretFinding, ...]],
    int,
    int,
    int,
    int,
    int,
]:
    findings_by_blob: dict[str, tuple[SecretFinding, ...]] = {}
    scanned_blobs = 0
    scanned_bytes = 0
    binary_blobs = 0
    skipped_blobs = 0
    skipped_bytes = 0

    for object_id, expected_size in blobs:
        if expected_size > max_blob_bytes:
            skipped_blobs += 1
            skipped_bytes += expected_size
            continue
        data = _run_git(repo, "cat-file", "blob", object_id)
        if len(data) != expected_size:
            raise HistoryScanError("Git blob size changed during the scan")
        scanned_blobs += 1
        scanned_bytes += expected_size
        if b"\0" in data:
            binary_blobs += 1
        blob_findings = tuple(
            scan_secrets(scan_paths.get(object_id, "<historical-blob>"), data)
        )
        if blob_findings:
            findings_by_blob[object_id] = blob_findings

    return (
        findings_by_blob,
        scanned_blobs,
        scanned_bytes,
        binary_blobs,
        skipped_blobs,
        skipped_bytes,
    )


def _parse_tree(
    raw: bytes,
    target_blobs: set[str],
) -> list[tuple[str, str]]:
    matches: list[tuple[str, str]] = []
    for entry in raw.split(b"\0"):
        if not entry:
            continue
        try:
            metadata, path_bytes = entry.split(b"\t", 1)
            _mode, object_type, object_id_bytes = metadata.split(b" ", 2)
        except ValueError as exc:
            raise HistoryScanError("Git returned invalid tree metadata") from exc
        if object_type != b"blob":
            continue
        object_id = _validate_object_id(object_id_bytes)
        if object_id not in target_blobs:
            continue
        path = path_bytes.decode("utf-8", errors="replace")
        matches.append((object_id, path))
    return matches


def _direct_root_trees(repo: Path) -> set[str]:
    """Return trees referenced directly, including peeled annotated tags."""

    raw = _run_git(
        repo,
        "for-each-ref",
        "--format=%(objectname)%00%(objecttype)%00%(*objectname)%00%(*objecttype)",
    )
    trees: set[str] = set()
    for line in raw.splitlines():
        pieces = line.split(b"\0")
        if len(pieces) != 4:
            raise HistoryScanError("Git returned invalid ref metadata")
        direct_id, direct_type, peeled_id, peeled_type = pieces
        if direct_type == b"tree":
            trees.add(_validate_object_id(direct_id))
        if peeled_type == b"tree":
            trees.add(_validate_object_id(peeled_id))
    return trees


def _finding_locations(
    repo: Path,
    commits: Sequence[str],
    direct_trees: Sequence[str],
    target_blobs: set[str],
) -> dict[str, tuple[tuple[str | None, str | None], ...]]:
    """Return every distinct historical path with one representative commit."""

    candidates: dict[str, dict[str, str | None]] = {}
    if not target_blobs:
        return {}
    for commit in sorted(commits):
        tree = _run_git(
            repo,
            "ls-tree",
            "-r",
            "-z",
            "--full-tree",
            commit,
        )
        for blob, path in _parse_tree(tree, target_blobs):
            by_path = candidates.setdefault(blob, {})
            current = by_path.get(path)
            if current is None or commit < current:
                by_path[path] = commit
    for tree_id in sorted(direct_trees):
        tree = _run_git(
            repo,
            "ls-tree",
            "-r",
            "-z",
            "--full-tree",
            tree_id,
        )
        for blob, path in _parse_tree(tree, target_blobs):
            candidates.setdefault(blob, {}).setdefault(path, None)

    result: dict[str, tuple[tuple[str | None, str | None], ...]] = {}
    for blob in sorted(target_blobs):
        by_path = candidates.get(blob)
        if not by_path:
            result[blob] = ((None, None),)
            continue
        result[blob] = tuple(
            (commit, path)
            for path, commit in sorted(by_path.items(), key=lambda item: item[0])
        )
    return result


def _history_scan_path(
    locations: Sequence[tuple[str | None, str | None]],
) -> str:
    """Choose one conservative real path while still scanning each blob once."""

    paths = sorted({path for _commit, path in locations if path})
    for path in paths:
        basename = PurePosixPath(path).name.casefold()
        if (
            basename == ".env"
            or basename.startswith(".env.")
            or basename.endswith(".env")
        ):
            return path
    return paths[0] if paths else "<historical-blob>"


def _validate_history_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\0" in value or "\\" in value:
        raise HistoryScanError("allowlist contains an invalid path")
    path = PurePosixPath(value)
    if path.is_absolute() or str(path) != value:
        raise HistoryScanError("allowlist contains an invalid path")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise HistoryScanError("allowlist contains an invalid path")
    return value


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise HistoryScanError("allowlist contains a duplicate JSON key")
        result[key] = value
    return result


def _validated_allowlist_entry(raw: object) -> AllowlistEntry:
    if not isinstance(raw, dict) or set(raw) != _ALLOWLIST_ENTRY_FIELDS:
        raise HistoryScanError("allowlist entry has invalid fields")

    blob = raw["blob"]
    if not isinstance(blob, str) or not _OBJECT_ID_RE.fullmatch(blob):
        raise HistoryScanError("allowlist entry has an invalid blob SHA")
    path = _validate_history_path(raw["path"])
    line = raw["line"]
    if isinstance(line, bool) or not isinstance(line, int) or line < 1:
        raise HistoryScanError("allowlist entry has an invalid line")
    kind = raw["kind"]
    if not isinstance(kind, str) or not _KIND_RE.fullmatch(kind):
        raise HistoryScanError("allowlist entry has an invalid finding kind")
    ref_scope = raw["ref_scope"]
    if ref_scope not in {"mixed", "tool", "user"}:
        raise HistoryScanError("allowlist entry has an invalid ref scope")
    rationale = raw["rationale"]
    if (
        not isinstance(rationale, str)
        or not 10 <= len(rationale) <= 500
        or any(ord(character) < 32 for character in rationale)
    ):
        raise HistoryScanError("allowlist entry has an invalid rationale")
    reviewed_at = raw["reviewed_at"]
    if not isinstance(reviewed_at, str):
        raise HistoryScanError("allowlist entry has an invalid review date")
    try:
        reviewed_date = date.fromisoformat(reviewed_at)
    except ValueError as exc:
        raise HistoryScanError("allowlist entry has an invalid review date") from exc
    if reviewed_date.isoformat() != reviewed_at or reviewed_date > date.today():
        raise HistoryScanError("allowlist entry has an invalid review date")
    return AllowlistEntry(
        blob=blob,
        path=path,
        line=line,
        kind=kind,
        ref_scope=ref_scope,
        rationale=rationale,
        reviewed_at=reviewed_at,
    )


def _load_allowlist(
    repo: Path,
    requested_path: Path | None,
) -> tuple[AllowlistEntry, ...]:
    candidate = requested_path or DEFAULT_ALLOWLIST
    if not candidate.is_absolute():
        candidate = repo / candidate
    if candidate.is_symlink():
        raise HistoryScanError("allowlist cannot be a symbolic link")
    try:
        resolved = candidate.resolve(strict=False)
        resolved.relative_to(repo)
    except (OSError, ValueError) as exc:
        raise HistoryScanError("allowlist must stay inside the repository") from exc
    if not resolved.exists():
        if requested_path is None:
            return ()
        raise HistoryScanError("requested allowlist does not exist")
    if not resolved.is_file():
        raise HistoryScanError("allowlist is not a regular file")
    try:
        data = resolved.read_bytes()
    except OSError as exc:
        raise HistoryScanError("allowlist could not be read") from exc
    if len(data) > _MAX_ALLOWLIST_BYTES:
        raise HistoryScanError("allowlist is too large")
    if scan_secrets(DEFAULT_ALLOWLIST.as_posix(), data):
        raise HistoryScanError("allowlist contains secret-like content")
    try:
        document = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
        )
    except HistoryScanError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HistoryScanError("allowlist is not valid UTF-8 JSON") from exc
    if not isinstance(document, dict) or set(document) != {
        "entries",
        "schema_version",
    }:
        raise HistoryScanError("allowlist document has invalid fields")
    if document["schema_version"] != 1:
        raise HistoryScanError("allowlist schema version is unsupported")
    raw_entries = document["entries"]
    if not isinstance(raw_entries, list):
        raise HistoryScanError("allowlist entries must be a list")
    entries = tuple(_validated_allowlist_entry(raw) for raw in raw_entries)
    keys = [entry.key for entry in entries]
    if len(keys) != len(set(keys)):
        raise HistoryScanError("allowlist contains duplicate entries")
    return tuple(sorted(entries, key=lambda entry: entry.key))


def _apply_allowlist(
    findings: Sequence[_LocatedFinding],
    entries: Sequence[AllowlistEntry],
    *,
    tool_ref_roots: int,
) -> tuple[
    tuple[HistorySecretFinding, ...],
    tuple[SuppressedHistoryFinding, ...],
    tuple[AllowlistEntry, ...],
    tuple[AllowlistEntry, ...],
]:
    by_key = {entry.key: entry for entry in entries}
    matched: set[tuple[str, str, int, str, str]] = set()
    active: list[HistorySecretFinding] = []
    suppressed: list[SuppressedHistoryFinding] = []
    for finding in findings:
        entry = by_key.get(finding.key) if finding.key is not None else None
        safe_finding = finding.safe_finding()
        if entry is None:
            active.append(safe_finding)
            continue
        matched.add(entry.key)
        suppressed.append(
            SuppressedHistoryFinding(
                finding=safe_finding,
                rationale=entry.rationale,
                reviewed_at=entry.reviewed_at,
            )
        )

    unmatched = [entry for entry in entries if entry.key not in matched]
    inactive_tool = tuple(
        entry
        for entry in unmatched
        if entry.ref_scope == "tool" and tool_ref_roots == 0
    )
    stale = tuple(entry for entry in unmatched if entry not in inactive_tool)
    return tuple(active), tuple(suppressed), stale, inactive_tool


def _short_sha(object_id: str) -> str:
    return object_id[:SHORT_SHA_LENGTH]


def scan_git_history(
    repo: Path,
    *,
    max_blob_bytes: int = DEFAULT_MAX_BLOB_BYTES,
    allowlist_path: Path | None = None,
) -> HistoryScanResult:
    """Scan every reachable blob once and return value-free finding metadata."""

    if max_blob_bytes <= 0:
        raise HistoryScanError("max blob bytes must be greater than zero")
    root = _resolve_repo(repo)
    allowlist = _load_allowlist(root, allowlist_path)
    object_ids = _reachable_object_ids(root)
    object_info = _object_info(root, object_ids)
    user_roots, tool_roots = _ref_roots(root)
    user_objects = _reachable_from_roots(root, user_roots)
    tool_objects = _reachable_from_roots(root, tool_roots)
    commits = sorted(
        object_id
        for object_id, (object_type, _size) in object_info.items()
        if object_type == "commit"
    )
    blobs = sorted(
        (
            (object_id, size)
            for object_id, (object_type, size) in object_info.items()
            if object_type == "blob"
        ),
        key=lambda item: item[0],
    )
    blob_ids = {blob for blob, _size in blobs}
    direct_trees = sorted(_direct_root_trees(root))
    all_locations = _finding_locations(
        root,
        commits,
        direct_trees,
        blob_ids,
    )
    (
        raw_findings,
        scanned_blobs,
        scanned_bytes,
        binary_blobs,
        skipped_blobs,
        skipped_bytes,
    ) = _scan_unique_blobs(
        root,
        blobs,
        max_blob_bytes=max_blob_bytes,
        scan_paths={blob: _history_scan_path(all_locations[blob]) for blob in blob_ids},
    )
    locations = {blob: all_locations[blob] for blob in raw_findings}

    located_findings: list[_LocatedFinding] = []
    for blob, blob_findings in raw_findings.items():
        ref_scope = _blob_ref_scope(
            blob,
            user_objects=user_objects,
            tool_objects=tool_objects,
        )
        for commit, path in locations[blob]:
            for finding in blob_findings:
                located_findings.append(
                    _LocatedFinding(
                        commit=commit,
                        blob=blob,
                        path=path,
                        line=finding.line,
                        kind=finding.kind,
                        ref_scope=ref_scope,
                    )
                )
    located_findings.sort(
        key=lambda item: (
            item.path or "",
            item.commit or "",
            item.blob,
            item.line,
            item.kind,
            item.ref_scope,
        )
    )
    (
        safe_findings,
        suppressed_findings,
        stale_entries,
        inactive_tool_entries,
    ) = _apply_allowlist(
        located_findings,
        allowlist,
        tool_ref_roots=len(tool_roots),
    )

    user_blob_ids = blob_ids.intersection(user_objects)
    tool_blob_ids = blob_ids.intersection(tool_objects)
    shared_blob_ids = user_blob_ids.intersection(tool_blob_ids)
    unknown_blob_ids = blob_ids.difference(user_blob_ids, tool_blob_ids)
    stats = HistoryScanStats(
        reachable_objects=len(object_ids),
        reachable_commits=len(commits),
        reachable_blobs=len(blobs),
        scanned_blobs=scanned_blobs,
        scanned_bytes=scanned_bytes,
        binary_blobs_scanned=binary_blobs,
        skipped_large_blobs=skipped_blobs,
        skipped_large_bytes=skipped_bytes,
        finding_blobs=len(raw_findings),
        detected_findings=len(located_findings),
        findings=len(safe_findings),
        suppressed_findings=len(suppressed_findings),
        allowlist_entries=len(allowlist),
        stale_allowlist_entries=len(stale_entries),
        inactive_tool_allowlist_entries=len(inactive_tool_entries),
        max_blob_bytes=max_blob_bytes,
        user_ref_roots=len(user_roots),
        tool_ref_roots=len(tool_roots),
        user_reachable_blobs=len(user_blob_ids),
        tool_reachable_blobs=len(tool_blob_ids),
        tool_only_reachable_blobs=len(tool_blob_ids.difference(user_blob_ids)),
        shared_reachable_blobs=len(shared_blob_ids),
        unknown_scope_reachable_blobs=len(unknown_blob_ids),
    )
    return HistoryScanResult(
        findings=safe_findings,
        suppressed_findings=suppressed_findings,
        stale_allowlist_entries=stale_entries,
        inactive_tool_allowlist_entries=inactive_tool_entries,
        stats=stats,
    )


def _git_directory(repo: Path) -> Path:
    raw = _run_git(repo, "rev-parse", "--absolute-git-dir").strip()
    try:
        decoded = os.fsdecode(raw)
        return Path(decoded).resolve(strict=True)
    except (OSError, TypeError) as exc:
        raise HistoryScanError("Git directory is unavailable") from exc


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _write_json_safely(repo: Path, output: Path, text: str) -> None:
    try:
        target = output.expanduser().resolve(strict=False)
        if target.exists() and (target.is_symlink() or not target.is_file()):
            raise HistoryScanError("JSON output path is not a regular file")
        git_directory = _git_directory(repo)
        if _is_within(target, git_directory):
            raise HistoryScanError("JSON output cannot be written inside Git metadata")
        target.parent.mkdir(parents=True, exist_ok=True)
        parent = target.parent.resolve(strict=True)
    except HistoryScanError:
        raise
    except OSError as exc:
        raise HistoryScanError("JSON output path is unavailable") from exc

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(text)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, target)
        temporary_path = None
    except OSError as exc:
        raise HistoryScanError("JSON output could not be written") from exc
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Scan each reachable Git blob once for likely secrets without "
            "printing matched values."
        )
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=BASE_DIR,
        help="Git repository to scan (default: this project)",
    )
    parser.add_argument(
        "--max-blob-bytes",
        type=int,
        default=DEFAULT_MAX_BLOB_BYTES,
        help=(
            "skip individual blobs larger than this many bytes "
            f"(default: {DEFAULT_MAX_BLOB_BYTES})"
        ),
    )
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=None,
        help=(
            "exact-match reviewed allowlist inside the repository "
            f"(default: {DEFAULT_ALLOWLIST.as_posix()} when present)"
        ),
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="atomically write the same value-free JSON report to this path",
    )
    strictness = parser.add_mutually_exclusive_group()
    strictness.add_argument(
        "--strict",
        action="store_true",
        help=(
            "return exit code 1 for findings, stale allowlist entries, or skipped blobs"
        ),
    )
    strictness.add_argument(
        "--strict-user-refs",
        action="store_true",
        help=(
            "return exit code 1 for findings reachable from user-controlled refs, "
            "user/mixed stale allowlist entries, or any skipped blob; local "
            "refs/codex findings remain visible but do not block the release"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = scan_git_history(
            args.repo,
            max_blob_bytes=args.max_blob_bytes,
            allowlist_path=args.allowlist,
        )
        report = json.dumps(
            result.as_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        if args.json_out is not None:
            _write_json_safely(
                _resolve_repo(args.repo),
                args.json_out,
                report + "\n",
            )
    except HistoryScanError as exc:
        print(
            json.dumps(
                {"error": str(exc), "ok": False},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(report)
    if args.strict and not result.strict_ready:
        return 1
    if args.strict_user_refs and not result.strict_user_ready:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
