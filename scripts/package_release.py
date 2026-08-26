"""Create a deterministic, secret-scanned contest submission archive.

Only regular files recorded in Git's index are eligible.  Their bytes are read
from the working tree so intentional, uncommitted edits to already tracked
files are packaged.  Release-eligible untracked files fail closed so newly
implemented source cannot be omitted silently; ignored and explicitly excluded
local artifacts remain outside the trust boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import posixpath
import re
import stat
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Sequence
from urllib.parse import urlsplit


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

DEFAULT_OUTPUT = BASE_DIR / "dist" / "meongtamjeong-contest.zip"
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
REGULAR_GIT_MODES = {"100644", "100755"}
MODEL_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors"}
CONTEST_ENV_PATH = ".env.contest.example"
CONTEST_ARTIFACT_KEYS = ("INDEX_PATH", "METAS_PATH")
FULL_RELEASE_PROFILE = "full"
PUBLIC_TEXT_PACKAGE_PROFILE = "public-text-only"
RELEASE_PROFILE_CHOICES = (FULL_RELEASE_PROFILE, PUBLIC_TEXT_PACKAGE_PROFILE)
RELEASE_MANIFEST_PATH = "release/manifest.json"
RELEASE_MANIFEST_SCHEMA_VERSION = 1
PUBLIC_TEXT_SOURCE_INDEX = "data/dog_faiss.index"
PUBLIC_TEXT_SOURCE_METAS = "data/dog_metas.json"
PUBLIC_TEXT_SOURCE_MANIFEST = "data/snapshot_manifest.json"
PUBLIC_TEXT_PROFILE_MARKER = "data/release_profile.json"
PUBLIC_TEXT_DERIVATION_REPORT = "data/public_text_release_report.json"
PUBLIC_TEXT_README = "README.md"
PUBLIC_TEXT_DATA_CARD = "DATA_CARD.md"
PUBLIC_TEXT_MODEL_CARD = "MODEL_CARD.md"
PUBLIC_TEXT_CONTRIBUTING = "CONTRIBUTING.md"
PUBLIC_TEXT_RELEASE_GUIDE = "docs/public-text-only-release.md"
PUBLIC_TEXT_PROVIDER_GUIDE = "docs/provider-adapter.md"
PUBLIC_TEXT_PROVIDER_EXAMPLE = "data/examples/notice-provider.synthetic.json"
PUBLIC_TEXT_EVALUATION_SUMMARY_JSON = "docs/evaluation/public-text-only.summary.json"
PUBLIC_TEXT_EVALUATION_SUMMARY_MARKDOWN = "docs/evaluation/public-text-only.summary.md"
PUBLIC_TEXT_PORTAL_PROTOCOL_GUIDE = (
    "docs/evaluation/PORTAL_CANDIDATE_SELECTION_PROTOCOL.md"
)
PUBLIC_TEXT_PORTAL_PROTOCOL_TEMPLATE = (
    "docs/evaluation/portal_candidate_selection.protocol.v1.json"
)
PUBLIC_TEXT_PORTAL_TRIAL_TEMPLATE = (
    "docs/evaluation/templates/portal_candidate_selection_trials.csv"
)
PUBLIC_TEXT_PORTAL_BLIND_LABEL_TEMPLATE = (
    "docs/evaluation/templates/portal_candidate_selection_blind_labels.csv"
)
PUBLIC_TEXT_SAMPLE_PREFIX = "assets/samples/"
PUBLIC_TEXT_EXCLUDED_SOURCE_PREFIXES = (".github/",)
PUBLIC_TEXT_RETAINED_DOCS = frozenset(
    {
        "docs/SBOM.md",
        "docs/dependency-report.json",
        "docs/dependency-report.md",
        PUBLIC_TEXT_PORTAL_BLIND_LABEL_TEMPLATE,
        PUBLIC_TEXT_PORTAL_PROTOCOL_GUIDE,
        PUBLIC_TEXT_PORTAL_PROTOCOL_TEMPLATE,
        PUBLIC_TEXT_PORTAL_TRIAL_TEMPLATE,
        PUBLIC_TEXT_EVALUATION_SUMMARY_JSON,
        PUBLIC_TEXT_EVALUATION_SUMMARY_MARKDOWN,
        PUBLIC_TEXT_PROVIDER_GUIDE,
        PUBLIC_TEXT_RELEASE_GUIDE,
    }
)
PUBLIC_TEXT_RETAINED_TESTS = frozenset(
    {
        "tests/test_appearance_query.py",
        "tests/test_dependency_snapshot.py",
        "tests/test_generate_sbom.py",
        "tests/test_package_release.py",
        "tests/test_public_text_evaluation_summary.py",
        "tests/test_notice_metadata.py",
        "tests/test_notice_provider.py",
        "tests/test_portal_candidate_selection_study.py",
        "tests/test_public_text_release.py",
        "tests/test_public_text_runtime.py",
        "tests/test_smoke_full_runtime.py",
        "tests/test_verify_contest_release.py",
        "tests/test_verify_release_archive.py",
    }
)
PUBLIC_TEXT_REQUIRED_PROFILE_NEUTRAL_TOOLING = frozenset(
    {
        "app/notice_provider.py",
        "app/portal_candidate_selection_study.py",
        PUBLIC_TEXT_PORTAL_BLIND_LABEL_TEMPLATE,
        PUBLIC_TEXT_PORTAL_PROTOCOL_GUIDE,
        PUBLIC_TEXT_PORTAL_PROTOCOL_TEMPLATE,
        PUBLIC_TEXT_PORTAL_TRIAL_TEMPLATE,
        PUBLIC_TEXT_PROVIDER_EXAMPLE,
        PUBLIC_TEXT_PROVIDER_GUIDE,
        "scripts/fetch_live_dogs.py",
        "scripts/portal_candidate_selection_study.py",
        "tests/test_appearance_query.py",
        "tests/test_notice_metadata.py",
        "tests/test_notice_provider.py",
        "tests/test_portal_candidate_selection_study.py",
    }
)
COMMON_REQUIRED_RELEASE_PATHS = frozenset(
    {
        ".env.contest.example",
        ".env.example",
        "CONTRIBUTING.md",
        "DATA_CARD.md",
        "LICENSE",
        "MODEL_CARD.md",
        "NOTICE",
        "README.md",
        "SECURITY.md",
        "SBOM.cdx.json",
        "SBOM.spdx.json",
        "THIRD_PARTY_NOTICES.md",
        "app/appearance_query.py",
        "app/graph_rag.py",
        "app/hybrid_rag.py",
        "app/inquiry_helper.py",
        "app/main.py",
        "app/notice_metadata.py",
        "app/notice_status.py",
        "app/profile_demo.html",
        "app/profile_rerank.py",
        "app/public_text_release.py",
        "data/dog_faiss.index",
        "data/dog_metas.json",
        "data/snapshot_manifest.json",
        "docs/dependency-report.json",
        "docs/dependency-report.md",
        "docs/SBOM.md",
        "environment.release.yml",
        "environment.yml",
        "pytest.ini",
        "requirements-dev.txt",
        "requirements.lock.txt",
        "requirements.txt",
        "scripts/check_contest_readiness.py",
        "scripts/check_dependency_snapshot.py",
        "scripts/generate_dependency_report.py",
        "scripts/generate_sbom.py",
        "scripts/package_release.py",
        "scripts/smoke_full_runtime.py",
        "scripts/verify_contest_release.py",
        "scripts/verify_release_archive.py",
        "tests/test_package_release.py",
        "tests/test_dependency_snapshot.py",
        "tests/test_generate_sbom.py",
        "tests/test_smoke_full_runtime.py",
        "tests/test_verify_contest_release.py",
        "tests/test_verify_release_archive.py",
    }
)
FULL_REQUIRED_RELEASE_PATHS = frozenset(
    {
        "data/active_index_sync_report.json",
        "data/eval_query_holdout.appearance_v3.sha256",
        "docs/contest-development-report-draft.md",
        "docs/contest-evidence-matrix.md",
        "docs/contest-release-runbook.md",
        "docs/demo-video-script.md",
        "docs/evaluation/profile_rerank.appearance_v1.json",
        "docs/evaluation/profile_rerank.appearance_v1.md",
        "docs/evaluation/query_holdout.appearance_v3.json",
        "docs/evaluation/query_holdout.appearance_v3.md",
        "docs/evaluation/query_holdout.appearance_v3.run.json",
        "docs/evaluation/retrieval_eval.appearance_v1.json",
        "docs/evaluation/retrieval_eval.appearance_v1.md",
        "docs/evaluation/safety_contract.appearance_v1.json",
        "docs/evaluation/safety_contract.appearance_v1.md",
        "tests/test_api_compat.py",
        "tests/test_appearance_query.py",
        "tests/test_inquiry_helper.py",
        "tests/test_profile_rerank.py",
    }
)
PUBLIC_TEXT_REQUIRED_RELEASE_PATHS = frozenset(
    {
        PUBLIC_TEXT_PROVIDER_EXAMPLE,
        PUBLIC_TEXT_PROVIDER_GUIDE,
        PUBLIC_TEXT_DERIVATION_REPORT,
        PUBLIC_TEXT_EVALUATION_SUMMARY_JSON,
        PUBLIC_TEXT_EVALUATION_SUMMARY_MARKDOWN,
        PUBLIC_TEXT_PROFILE_MARKER,
        PUBLIC_TEXT_RELEASE_GUIDE,
        "app/notice_provider.py",
        "app/public_text_evaluation_summary.py",
        "scripts/build_public_text_evaluation_summary.py",
        "scripts/build_public_text_release.py",
        "tests/test_appearance_query.py",
        "tests/test_public_text_evaluation_summary.py",
        "tests/test_public_text_runtime.py",
    }
)
# These reports and their runners are tied to the full multimodal artifact.
# Keeping only part of the frozen evaluation in the text-only ZIP would make
# its default check command unusable and invite profile-confused claims.
PUBLIC_TEXT_FULL_PROFILE_HOLDOUT_PATHS = frozenset(
    {
        "app/query_holdout_evaluation.py",
        "app/query_holdout_v3_evaluation.py",
        "scripts/evaluate_query_holdout.py",
        "scripts/evaluate_query_holdout_v3.py",
        "tests/test_query_holdout_evaluation.py",
        "tests/test_query_holdout_v3_evaluation.py",
    }
)
PUBLIC_TEXT_FULL_PROFILE_HOLDOUT_PREFIXES = (
    "data/eval_query_holdout.appearance_v2.",
    "data/eval_query_holdout.appearance_v3.",
    "docs/evaluation/query_holdout.appearance_v2.",
    "docs/evaluation/query_holdout.appearance_v3.",
)
PUBLIC_TEXT_STALE_ARTIFACTS = frozenset(
    {
        "data/active_index_sync_report.json",
        "docs/evaluation/exposure_representation.appearance_v1.json",
        "docs/evaluation/exposure_representation.appearance_v1.md",
        "docs/evaluation/heldout_image_retrieval.appearance_v1.json",
        "docs/evaluation/heldout_image_retrieval.appearance_v1.md",
        "docs/evaluation/profile_rerank.appearance_v1.json",
        "docs/evaluation/profile_rerank.appearance_v1.md",
        "docs/evaluation/query_robustness.appearance_v1.json",
        "docs/evaluation/query_robustness.appearance_v1.md",
        "docs/evaluation/retrieval_eval.appearance_v1.json",
        "docs/evaluation/retrieval_eval.appearance_v1.md",
        "docs/evaluation/safety_contract.appearance_v1.json",
        "docs/evaluation/safety_contract.appearance_v1.md",
    }
)
PUBLIC_TEXT_PHOTO_SUFFIXES = frozenset(
    {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)
WINDOWS_RESERVED_NAMES = {
    "aux",
    "clock$",
    "con",
    "nul",
    "prn",
    *(f"com{number}" for number in range(1, 10)),
    *(f"lpt{number}" for number in range(1, 10)),
}

_SECRET_NAME_PATTERN = r"""
    (?:[a-z0-9]+[_-])*
    (?:
        api[_-]?key
        |access[_-]?token
        |auth[_-]?token
        |client[_-]?secret
        |private[_-]?key
        |secret[_-]?key
        |service[_-]?key
        |password
        |passwd
        |authorization
    )
"""
_NAMED_SECRET_RE = re.compile(
    rf"""(?ix)
    \b(?:{_SECRET_NAME_PATTERN})\b
    \s*[:=]\s*
    (?:[rubf]{{0,2}})?
    (?P<quote>["'])
    (?P<value>.*?)
    (?P=quote)
    """
)
_ENV_SECRET_RE = re.compile(
    rf"""(?ix)
    ^\s*
    (?P<name>{_SECRET_NAME_PATTERN})
    \s*=\s*(?P<value>[^#\r\n]*?)\s*$
    """
)
_KNOWN_TOKEN_PATTERNS = (
    (
        "private-key",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    ),
    ("aws-access-key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b")),
    ("google-api-key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("openai-token", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    (
        "slack-token",
        re.compile(r"\bxox(?:a|b|p|r|s)-[A-Za-z0-9-]{20,}\b"),
    ),
    (
        "bearer-token",
        re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{20,}\b"),
    ),
)
_PLACEHOLDER_EXACT = {
    "",
    "-",
    "change-me",
    "changeme",
    "dummy",
    "example",
    "none",
    "null",
    "placeholder",
    "redacted",
    "replace-me",
    "test",
    "xxx",
}
_PLACEHOLDER_FRAGMENTS = (
    "change-me",
    "dummy",
    "example",
    "not-a-real",
    "placeholder",
    "redacted",
    "replace-me",
    "sample",
    "test-",
    "your-",
    "your_",
)


class PackagingError(RuntimeError):
    """A release archive could not be produced safely."""


class UntrackedFilesError(PackagingError):
    """Release-eligible untracked files would otherwise be omitted."""

    def __init__(self, count: int) -> None:
        self.count = count
        super().__init__(
            f"{count} release-eligible untracked file(s) found; "
            "stage/commit first, then build the release again"
        )


@dataclass(frozen=True)
class SecretFinding:
    """A finding that deliberately excludes the matched secret value."""

    path: str
    line: int
    kind: str


class SecretScanError(PackagingError):
    """One or more potential secrets were found in eligible tracked files."""

    def __init__(self, findings: Sequence[SecretFinding]) -> None:
        self.findings = tuple(findings)
        locations = ", ".join(
            f"{finding.path}:{finding.line} ({finding.kind})"
            for finding in self.findings[:10]
        )
        suffix = "" if len(self.findings) <= 10 else ", ..."
        super().__init__(
            f"potential secrets found at {locations}{suffix}; "
            "secret values were not displayed"
        )


@dataclass(frozen=True)
class TrackedFile:
    """A validated regular file selected from Git's index."""

    archive_path: str
    mode: str
    data: bytes


@dataclass(frozen=True)
class PackageResult:
    """Safe summary suitable for CLI output."""

    checked_only: bool
    file_count: int
    excluded_count: int
    total_bytes: int
    output: str | None
    sha256: str | None
    commit_sha: str | None
    profile: str = FULL_RELEASE_PROFILE
    profile_summary: dict[str, object] | None = None
    release_tag: str | None = None
    tag_object_id: str | None = None
    release_manifest: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "checked_only": self.checked_only,
            "file_count": self.file_count,
            "excluded_count": self.excluded_count,
            "total_bytes": self.total_bytes,
            "output": self.output,
            "sha256": self.sha256,
            "commit_sha": self.commit_sha,
            "profile": self.profile,
            "profile_summary": self.profile_summary,
            "release_tag": self.release_tag,
            "tag_object_id": self.tag_object_id,
            "release_manifest": self.release_manifest,
            "source": (
                "clean HEAD working tree"
                if self.commit_sha
                else "git-index paths + current working-tree bytes"
            ),
        }


def _run_git(repo: Path, *args: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise PackagingError("Git executable was not found") from exc
    except subprocess.CalledProcessError as exc:
        # Git's stderr can contain a path but should never be allowed to echo
        # arbitrary file contents or credentials through this CLI.
        raise PackagingError(
            f"Git command failed with exit code {exc.returncode}"
        ) from exc
    return completed.stdout


def _git_differs(repo: Path, *, cached: bool) -> bool:
    args = ["git", "-C", str(repo), "diff", "--quiet"]
    if cached:
        args.insert(-1, "--cached")
    args.append("--")
    try:
        completed = subprocess.run(
            args,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise PackagingError("Git executable was not found") from exc
    if completed.returncode == 0:
        return False
    if completed.returncode == 1:
        return True
    raise PackagingError(
        f"Git cleanliness check failed with exit code {completed.returncode}"
    )


def _require_clean_head(
    repo: Path, *, required_tag: str | None = None
) -> tuple[str, str | None]:
    try:
        commit_sha = _run_git(repo, "rev-parse", "--verify", "HEAD^{commit}")
        commit = commit_sha.decode("ascii").strip().lower()
    except (PackagingError, UnicodeDecodeError) as exc:
        raise PackagingError("release requires an existing HEAD commit") from exc
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise PackagingError("Git returned an invalid HEAD commit ID")
    if _git_differs(repo, cached=False) or _git_differs(repo, cached=True):
        raise PackagingError(
            "release requires tracked files and the Git index to match HEAD; "
            "commit the intended changes first"
        )

    tag_object_id: str | None = None
    if required_tag is not None:
        tag = required_tag.strip()
        if not tag or tag.startswith("-") or any(ch.isspace() for ch in tag):
            raise PackagingError("release tag name is invalid")
        try:
            tag_type = _run_git(repo, "cat-file", "-t", f"refs/tags/{tag}")
            tag_object = _run_git(repo, "rev-parse", "--verify", f"refs/tags/{tag}")
            tagged_sha = _run_git(
                repo,
                "rev-parse",
                "--verify",
                f"refs/tags/{tag}^{{commit}}",
            )
            tagged_commit = tagged_sha.decode("ascii").strip().lower()
        except (PackagingError, UnicodeDecodeError) as exc:
            raise PackagingError(
                f"required release tag was not found: {tag!r}"
            ) from exc
        if tag_type.decode("ascii").strip() != "tag":
            raise PackagingError(f"required release tag must be annotated: {tag!r}")
        tag_object_id = tag_object.decode("ascii").strip().lower()
        if not re.fullmatch(r"[0-9a-f]{40}", tag_object_id):
            raise PackagingError("Git returned an invalid annotated tag object ID")
        if tagged_commit != commit:
            raise PackagingError(f"required release tag {tag!r} does not point to HEAD")
    return commit, tag_object_id


def _validate_archive_name(name: str) -> None:
    if not name or "\x00" in name or "\\" in name:
        raise PackagingError("unsafe archive path")
    if name.startswith("/") or re.match(r"^[A-Za-z]:", name):
        raise PackagingError(f"unsafe archive path: {name!r}")
    path = PurePosixPath(name)
    if str(path) != name or any(part in {"", ".", ".."} for part in path.parts):
        raise PackagingError(f"unsafe archive path: {name!r}")
    for part in path.parts:
        if part.endswith((" ", ".")):
            raise PackagingError(f"unsafe archive path: {name!r}")
        stem = part.split(".", 1)[0].casefold()
        if stem in WINDOWS_RESERVED_NAMES:
            raise PackagingError(f"unsafe archive path: {name!r}")


def _exclusion_reason(name: str) -> str | None:
    path = PurePosixPath(name)
    lowered_parts = tuple(part.casefold() for part in path.parts)
    basename = lowered_parts[-1]
    if ".git" in lowered_parts:
        return "git-metadata"
    if "__pycache__" in lowered_parts or basename.endswith((".pyc", ".pyo")):
        return "python-cache"
    if path.suffix.casefold() in MODEL_SUFFIXES:
        return "model-weight"
    if basename == ".env" or (
        basename.startswith(".env.") and not basename.endswith(".example")
    ):
        return "environment-file"
    return None


def _parse_git_index(raw: bytes) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    seen: set[str] = set()
    for encoded in raw.split(b"\0"):
        if not encoded:
            continue
        try:
            metadata, path_bytes = encoded.split(b"\t", 1)
            mode_bytes, _object_id, stage_bytes = metadata.split(b" ", 2)
            name = path_bytes.decode("utf-8")
            mode = mode_bytes.decode("ascii")
            stage = stage_bytes.decode("ascii")
        except (UnicodeDecodeError, ValueError) as exc:
            raise PackagingError("Git index contains an invalid path entry") from exc
        if stage != "0":
            raise PackagingError(f"unmerged Git index entry: {name!r}")
        _validate_archive_name(name)
        if name in seen:
            raise PackagingError(f"duplicate Git index path: {name!r}")
        seen.add(name)
        entries.append((name, mode))
    return sorted(entries)


def _parse_untracked_paths(raw: bytes) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for encoded in raw.split(b"\0"):
        if not encoded:
            continue
        try:
            name = encoded.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PackagingError(
                "Git worktree contains an invalid untracked path"
            ) from exc
        try:
            _validate_archive_name(name)
        except PackagingError as exc:
            raise PackagingError(
                "Git worktree contains an unsafe untracked path; refusing release"
            ) from exc
        if name in seen:
            raise PackagingError(
                "Git returned a duplicate untracked path; refusing release"
            )
        seen.add(name)
        paths.append(name)
    return sorted(paths)


def _reject_release_eligible_untracked_files(
    root: Path,
    *,
    output_resolved: Path | None,
) -> None:
    """Fail when an untracked regular file could be omitted from the release."""

    raw = _run_git(root, "ls-files", "--others", "--exclude-standard", "-z")
    names = _parse_untracked_paths(raw)
    candidate_count = 0

    for name in names:
        if _exclusion_reason(name) is not None:
            continue

        source = root.joinpath(*PurePosixPath(name).parts)
        try:
            source_mode = source.lstat().st_mode
        except OSError as exc:
            raise PackagingError(
                "an untracked path could not be inspected safely; refusing release"
            ) from exc
        if stat.S_ISLNK(source_mode):
            raise PackagingError(
                "an untracked symbolic link was found; refusing release"
            )
        if not stat.S_ISREG(source_mode):
            raise PackagingError(
                "an untracked non-regular path was found; refusing release"
            )

        try:
            source_resolved = source.resolve(strict=True)
            source_resolved.relative_to(root)
        except (OSError, ValueError) as exc:
            raise PackagingError(
                "an untracked path escapes the repository; refusing release"
            ) from exc

        # A stale archive at the requested destination is generated output, not
        # source waiting to be staged.  Only a validated regular file qualifies.
        if output_resolved is not None and source_resolved == output_resolved:
            continue
        candidate_count += 1

    if candidate_count:
        raise UntrackedFilesError(candidate_count)


def _is_placeholder(value: str) -> bool:
    # JavaScript assignments commonly end in ``;``. Keep that syntax outside
    # the value so ``apiKey = '';`` is treated like the empty placeholder it is.
    # A Python keyword argument can likewise end in a comma on its own line.
    normalized = (
        value.strip()
        .removesuffix(";")
        .strip()
        .removesuffix(",")
        .strip()
        .strip("\"'")
        .strip()
        .casefold()
    )
    if normalized in _PLACEHOLDER_EXACT:
        return True
    if normalized.startswith(("$", "${", "<", "__")):
        return True
    return any(fragment in normalized for fragment in _PLACEHOLDER_FRAGMENTS)


def _is_environment_file(path: str) -> bool:
    basename = PurePosixPath(path).name.casefold()
    return (
        basename == ".env" or basename.startswith(".env.") or basename.endswith(".env")
    )


def _is_code_reference(path: str, value: str) -> bool:
    normalized = (
        value.strip()
        .removesuffix(";")
        .strip()
        .removesuffix(",")
        .strip()
        .casefold()
    )
    if any(
        marker in normalized
        for marker in (
            "args.",
            "getenv(",
            "os.environ",
            "process.env",
            "request.",
            "settings.",
            ".value.",
            ".value.trim(",
        )
    ):
        return True
    # In source code, an unquoted identifier is a runtime value, not an
    # embedded credential. Do not apply this exemption to .env-style files:
    # there the same text is a literal configuration value and must fail
    # closed.
    return not _is_environment_file(path) and bool(
        re.fullmatch(r"[a-z_$][a-z0-9_$]*(?:\.[a-z_$][a-z0-9_$]*)*", normalized)
    )


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def scan_secrets(path: str, data: bytes) -> list[SecretFinding]:
    """Return safe metadata for likely secrets without retaining their values."""

    binary = b"\0" in data
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="ignore")

    findings: set[SecretFinding] = set()
    for kind, pattern in _KNOWN_TOKEN_PATTERNS:
        for match in pattern.finditer(text):
            findings.add(SecretFinding(path, _line_number(text, match.start()), kind))

    if binary:
        return sorted(findings, key=lambda item: (item.path, item.line, item.kind))

    for match in _NAMED_SECRET_RE.finditer(text):
        value = match.group("value")
        if not _is_placeholder(value):
            findings.add(
                SecretFinding(path, _line_number(text, match.start()), "named-secret")
            )

    for line_number, line in enumerate(text.splitlines(), start=1):
        match = _ENV_SECRET_RE.match(line)
        if (
            match
            and not _is_placeholder(match.group("value"))
            and not _is_code_reference(path, match.group("value"))
        ):
            findings.add(SecretFinding(path, line_number, "environment-secret"))

    return sorted(findings, key=lambda item: (item.path, item.line, item.kind))


def collect_tracked_files(
    repo: Path,
    *,
    output: Path | None = None,
) -> tuple[list[TrackedFile], int]:
    """Read validated Git-index paths from the current working tree."""

    root = repo.resolve(strict=True)
    if not root.is_dir():
        raise PackagingError("repository path is not a directory")
    raw = _run_git(root, "ls-files", "--cached", "--stage", "-z")
    entries = _parse_git_index(raw)
    output_resolved = output.resolve(strict=False) if output is not None else None
    _reject_release_eligible_untracked_files(
        root,
        output_resolved=output_resolved,
    )
    selected: list[TrackedFile] = []
    excluded_count = 0

    for name, mode in entries:
        if _exclusion_reason(name) is not None:
            excluded_count += 1
            continue
        source = root.joinpath(*PurePosixPath(name).parts)
        source_resolved = source.resolve(strict=False)
        try:
            source_resolved.relative_to(root)
        except ValueError as exc:
            raise PackagingError(f"tracked path escapes repository: {name!r}") from exc
        if output_resolved is not None and source_resolved == output_resolved:
            excluded_count += 1
            continue
        if mode not in REGULAR_GIT_MODES:
            raise PackagingError(
                f"tracked path is not a regular file: {name!r} (mode {mode})"
            )
        if source.is_symlink():
            raise PackagingError(f"tracked path is a symbolic link: {name!r}")
        if not source.is_file():
            raise PackagingError(f"tracked file is missing: {name!r}")
        data = source.read_bytes()
        selected.append(TrackedFile(name, mode, data))

    if not selected:
        raise PackagingError("no eligible tracked files found")
    return selected, excluded_count


def _is_public_text_sample_photo(path: str) -> bool:
    pure = PurePosixPath(path)
    return (
        len(pure.parts) >= 3
        and tuple(part.casefold() for part in pure.parts[:2]) == ("assets", "samples")
        and pure.suffix.casefold() in PUBLIC_TEXT_PHOTO_SUFFIXES
    )


def _is_public_text_sample_artifact(path: str) -> bool:
    return path.casefold().startswith(PUBLIC_TEXT_SAMPLE_PREFIX)


def _is_public_text_full_profile_holdout(path: str) -> bool:
    normalized = path.casefold()
    return (
        normalized in PUBLIC_TEXT_FULL_PROFILE_HOLDOUT_PATHS
        or normalized.startswith(PUBLIC_TEXT_FULL_PROFILE_HOLDOUT_PREFIXES)
    )


def _iter_json_strings(value: object) -> Sequence[str]:
    values: list[str] = []
    if isinstance(value, str):
        values.append(value)
    elif isinstance(value, dict):
        for child in value.values():
            values.extend(_iter_json_strings(child))
    elif isinstance(value, list):
        for child in value:
            values.extend(_iter_json_strings(child))
    return values


def _validate_public_text_provider_example(item: TrackedFile) -> None:
    """Require the distributed provider fixture to use non-routable URLs only."""

    try:
        payload = json.loads(item.data.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PackagingError(
            "public-text-only provider example must be valid UTF-8 JSON"
        ) from exc
    if not isinstance(payload, (dict, list)):
        raise PackagingError("public-text-only provider example has an invalid root")

    urls = [
        value.strip()
        for value in _iter_json_strings(payload)
        if re.match(r"(?i)^https?://", value.strip())
    ]
    if not urls:
        raise PackagingError(
            "public-text-only provider example must contain a synthetic URL"
        )
    for value in urls:
        parsed = urlsplit(value)
        hostname = (parsed.hostname or "").casefold()
        if (
            parsed.scheme.casefold() not in {"http", "https"}
            or not hostname.endswith(".invalid")
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise PackagingError(
                "public-text-only provider example contains a routable or credentialed URL"
            )


def _public_text_readme(artifacts: object) -> bytes:
    summary = artifacts.safe_summary()
    vector_count = int(summary["vectors"])
    unique_notices = int(summary["unique_notices"])
    dimension = int(summary["dimension"])
    text = f"""# 멍탐정 — public-text-only 배포본

이 압축 파일은 `{summary["profile"]}` 프로필입니다. 공고 사진 관련 권리 노출을
줄이기 위한 배포 대안이며, 사진이나 데이터의 권리 상태가 안전하다는 법적 판단이
아닙니다.

## 포함된 검색 데이터

- 활성 공고 {unique_notices:,}건의 공개 공고 텍스트 벡터 {vector_count:,}개
- 임베딩 차원 {dimension}
- 벡터 유형: `text`만 포함
- 메타데이터: 고정 allowlist를 통과한 공개 공고 텍스트 필드만 포함

공고 사진 파일, 사진/crop 벡터, 원격 공고 사진 URL, 사진 품질 점수, 샘플 사진과
그 출처 manifest는 포함하지 않습니다. 런타임에서도 로컬 graph overlay와 공고
사진·crop·image-audit 제공 경로를 닫습니다.

## 검색 범위와 한계

자연어 검색과 외형 프로필 재정렬은 텍스트 벡터와 공고의 명시 필드를 사용합니다.
사용자 업로드 이미지 검색 API는 호환성을 위해 유지되지만, 업로드 임베딩을 공고
텍스트 벡터와 비교합니다. 따라서 원본 full 프로필의 사진/crop 검색 품질이나 사진
품질 기반 재정렬 성능을 이 배포본의 성능으로 주장하지 않습니다. 성격·공격성·아동
친화성·다른 동물과의 사회성도 사진이나 품종만으로 추정하지 않습니다.

원본 full 프로필용 시각 평가 리포트는 이 번들과 입력 artifact가 달라 제외했습니다.
full 프로젝트와 평가 자료는 원 저장소에서 별도로 확인해야 합니다:
https://github.com/YuMinBee/meongtamjeong

이 text-only artifact에 직접 적용한 정량 결과와 silver-label 한계는
[public-text-only evaluation summary](docs/evaluation/public-text-only.summary.md)에
고정 index·metadata SHA-256와 함께 제공합니다. 실행 시각·latency·로컬 경로는 같은
tag의 증거를 흔들 수 있어 요약에서 제외했습니다.

세부 파생·검증 계약은 [public-text-only release guide](docs/public-text-only-release.md)를
참고하십시오.

## 프로필 독립 오픈소스 도구

- [공고 Provider Adapter](docs/provider-adapter.md): 다른 기관의 공개 공고를 같은
  상태 필터·정규화·동기화 경로에 연결합니다.
- [공식 포털 후보 탐색 파일럿](docs/evaluation/PORTAL_CANDIDATE_SELECTION_PROTOCOL.md):
  결과를 만들지 않는 사전 고정 프로토콜, 교차 배정과 CSV 템플릿을 제공합니다.

`data/examples/notice-provider.synthetic.json`도 계약 시험용으로 포함합니다. 그 안의
`.invalid` URL은 인터넷에서 연결될 수 없는 합성 문자열이며 실제 공고·사진 참조가
아닙니다. 위의 원격 사진 참조 제외 계약은 canonical 운영 검색 artifact인
`dog_metas.json`, `snapshot_manifest.json`, FAISS index에 적용됩니다. 사람 파일럿의
실제 참여자 자료나 결과는 이 ZIP에 포함하거나 성과로 주장하지 않습니다.

## 실행

Conda 환경을 활성화하고 루트에서 다음 값을 설정한 뒤 실행합니다.

```powershell
conda env create -f environment.release.yml
conda activate meongtamjeong-release
$env:APP_ENV = "contest"
$keyBytes = New-Object byte[] 32
$rng = [Security.Cryptography.RandomNumberGenerator]::Create()
$rng.GetBytes($keyBytes)
$rng.Dispose()
$env:API_KEY = [Convert]::ToBase64String($keyBytes)
$env:INDEX_PATH = "data/dog_faiss.index"
$env:METAS_PATH = "data/dog_metas.json"
$env:RELEASE_PROFILE_PATH = "data/release_profile.json"
$env:RELEASE_PROFILE = "public-text-only"
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

`GET /health`에서 `release_profile`, `vector_modalities`, 공고 사진 및 visual asset
경로 비활성화 상태를 확인할 수 있습니다. 공고 상태는 스냅샷 이후 바뀔 수 있으므로
연락 전 실제 공고 링크에서 다시 확인해야 합니다.

```powershell
curl.exe -H "x-api-key: $env:API_KEY" http://127.0.0.1:8000/health
```

이 ZIP에서 유지한 profile-applicable 테스트와 실제 런타임 smoke는 다음 명령으로
검증합니다.

```powershell
python -m pytest -q
python scripts/smoke_full_runtime.py --profile public-text-only --index data/dog_faiss.index --metas data/dog_metas.json --release-profile-marker data/release_profile.json
```

이 프로필은 새 ZIP의 노출만 최소화합니다. 현재 또는 과거 Git 원격 이력에 이미
공개된 샘플 사진·사진 파생 인덱스의 권리 상태나 회수 조치를 해결하거나 판정하지
않습니다. GitHub URL 자체를 제출하려면 별도의 text-only clean branch/repository
또는 권리 확인이 필요합니다.
"""
    return text.replace("\r\n", "\n").encode("utf-8")


def _public_text_data_card(artifacts: object) -> bytes:
    summary = artifacts.safe_summary()
    text = f"""# Data Card — public-text-only-v1

이 문서는 이 압축 파일에 실제 포함된 `public-text-only-v1` artifact만 설명합니다.
공고 사진 관련 권리 노출을 줄이기 위한 배포 대안이며 법적 판단이 아닙니다.

- `data/dog_metas.json`: 활성 공고 {int(summary["unique_notices"]):,}건에 대응하는
  allowlist 기반 공개 공고 텍스트 메타 {int(summary["vectors"]):,}행
- `data/dog_faiss.index`: `public_notice_text` provenance를 검증한
  {int(summary["vectors"]):,}개 text 벡터, {int(summary["dimension"])}차원
- `data/snapshot_manifest.json`: source 필드 allowlist와 파생 artifact 해시
- `data/release_profile.json`: 런타임 fail-closed 정책과 canonical SHA-256
- `data/public_text_release_report.json`: 선택·제외 건수와 검증 계약

canonical 운영 검색 artifact에는 사진 파일, 실제 원격 공고 사진 URL, image/crop
벡터, VLM·detector 속성, crop 경로와 사진 품질 점수를 포함하지 않습니다. 별도
provider 계약 fixture의 `.invalid` URL은 연결 불가능한 합성 예시이며 운영 메타가
아닙니다. 공고 상태와 연락처는 스냅샷 이후 달라질 수 있고, 품종 표기는 기관의
공고 표기이지 혈통 또는 행동 특성의 검증이 아닙니다.
"""
    return text.replace("\r\n", "\n").encode("utf-8")


def _public_text_model_card(artifacts: object) -> bytes:
    summary = artifacts.safe_summary()
    text = f"""# Model Card — public-text-only-v1

이 배포본은 CLIP ViT-B/32의 {int(summary["dimension"])}차원 공간에 저장된 공개 공고
텍스트 벡터 {int(summary["vectors"]):,}개를 검색합니다. image/crop 벡터와 사진 품질
신호는 포함하거나 점수에 사용하지 않습니다.

사용자 업로드 이미지 API는 호환성을 위해 유지되지만 이미지 임베딩을 공고 텍스트
벡터와 비교하므로, full 멀티모달 인덱스의 사진 검색 성능을 이 배포본에 귀속하지
않습니다. 이 기능은 보호소 방문 전 외형 후보 탐색을 돕는 도구이며 입양 적합성,
성격, 공격성, 아동 친화성 또는 다른 동물과의 사회성을 판정하지 않습니다.

이 ZIP에서 제외된 full-profile 평가 수치는 이 text-only artifact의 성능 근거가
아닙니다. 재현 시 `release_profile.json`과 `/health`의 profile·modality를 먼저
확인해야 합니다.
"""
    return text.replace("\r\n", "\n").encode("utf-8")


def _public_text_contributing() -> bytes:
    text = """# Contributing — public-text-only-v1

이 번들의 canonical index·metadata·manifest·profile marker를 직접 편집하지 마세요.
원본 저장소에서 `scripts/build_public_text_release.py`로 결정론적으로 다시 파생하고,
allowlist·provenance·canonical text/hash 검증 실패를 우회하지 않아야 합니다.

이 ZIP에는 profile-applicable 테스트만 남아 있으므로 `python -m pytest -q`를
실행합니다. 실제 CPU 경로는 다음 명령으로 확인합니다.

```powershell
python scripts/smoke_full_runtime.py --profile public-text-only --index data/dog_faiss.index --metas data/dog_metas.json --release-profile-marker data/release_profile.json
```

사진·원격 사진 URL·image/crop 벡터를 다시 넣는 변경은 이 배포 프로필의 범위를
벗어나며 full 프로필에서 별도로 검토해야 합니다.

Provider adapter와 결과 없는 공식 포털 파일럿 프로토콜은 profile-neutral 도구로
함께 배포합니다. 합성 provider fixture의 URL은 `.invalid`로 유지하고, 사람 파일럿의
원본 CSV나 결과를 source tree에 추가하지 마십시오. full artifact에 고정된 query
holdout v2/v3 runner·입력·리포트·테스트는 이 text-only ZIP의 성능 근거가 아니므로
의도적으로 제외합니다.
"""
    return text.replace("\r\n", "\n").encode("utf-8")


def _public_text_release_guide(artifacts: object) -> bytes:
    summary = artifacts.safe_summary()
    text = f"""# public-text-only-v1 release guide

## 목적과 비법률적 성격

이 프로필은 새 배포물의 공고 사진 관련 권리 노출을 최소화합니다. 권리 안전성이나
법적 이용 가능성을 판정하지 않습니다. 현재·과거 Git 원격 이력에 공개된 사진 및
사진 파생 artifact의 권리 상태나 회수 조치도 해결하지 않습니다.

## 결정론적 파생 계약

- source row는 `type=text`, `embedding_source=public_notice_text`여야 합니다.
- canonical 공개 공고 텍스트와 `desc_full`, SHA-256가 일치해야 합니다.
- 공고마다 검증된 text row가 정확히 하나 있어야 합니다.
- 출력은 {int(summary["unique_notices"]):,}개 공고, {int(summary["vectors"]):,}개
  {int(summary["dimension"])}차원 text vector입니다.
- metadata와 manifest source는 고정 allowlist 밖의 필드를 제거합니다.

## 런타임 계약

marker/index/metas SHA-256·vector count·dimension·metadata allowlist가 어긋나면
시작하지 않습니다. graph overlay, 원격 공고 사진, 사진 품질 점수와 공고
gallery/crop/audit route는 비활성화합니다. 사용자 업로드 이미지 검색은 유지하지만
공고 text vector와 비교하므로 full 시각 검색 품질을 주장하지 않습니다.

## 평가 해석

full 멀티모달 artifact에서 만든 retrieval·robustness·exposure·profile·safety·held-out
리포트는 이 ZIP에서 제외합니다. 특히 held-out image retrieval은 visual vector가 없는
프로필에 적용 불가(N/A)이며 PASS로 세지 않습니다. 공고 상태는 연락 전 원문에서
다시 확인해야 합니다.

대신 이 profile의 고정 index·metadata에서 release gate가 다시 계산한 안정 지표만
[public-text-only evaluation summary](evaluation/public-text-only.summary.md)에
결정론적으로 축약해 포함합니다. raw 리포트의 실행 시각·latency·로컬 경로는
포함하지 않으며, summary JSON의 artifact SHA-256가 canonical 데이터와 다르면
패키징이 실패합니다.

full artifact에 고정된 독립 query holdout v2/v3의 runner·입력·sidecar·리포트·테스트도
부분적으로 남기지 않고 모두 제외합니다. text-only 성능으로 재사용하거나 PASS로
세지 않습니다.

## 함께 배포하는 profile-neutral 도구

[공고 Provider Adapter](provider-adapter.md)와
[공식 포털 후보 탐색 파일럿](evaluation/PORTAL_CANDIDATE_SELECTION_PROTOCOL.md)은
검색 artifact 유형과 무관한 재사용 도구라 포함합니다. Provider 합성 fixture의 모든
URL은 연결 불가능한 `.invalid`이며 canonical 운영 metadata가 아닙니다. 포털 파일럿은
결과 없는 템플릿만 제공하며 실제 사람 자료나 성과를 포함하지 않습니다. 파일럿을
실행한다면 이 ZIP의 text-only index·metadata와 실행 commit을 첫 참여자 전에 새로
freeze해야 합니다.

GitHub URL을 출품물로 쓰려면 별도 text-only clean branch/repository를 만들거나
공고 사진·사진 파생 artifact의 재배포 범위를 서면으로 확인해야 합니다.
"""
    return text.replace("\r\n", "\n").encode("utf-8")


def _is_public_text_full_profile_document(path: str) -> bool:
    lowered = path.casefold()
    return lowered.startswith("docs/") and path not in PUBLIC_TEXT_RETAINED_DOCS


def _public_text_evaluation_files(
    evidence_dir: Path | None,
    *,
    index_sha256: str,
    metas_sha256: str,
) -> tuple[TrackedFile, TrackedFile]:
    if evidence_dir is None:
        raise PackagingError(
            "public-text-only profile requires gate-generated evaluation evidence"
        )
    try:
        from app.public_text_evaluation_summary import (
            MAX_REPORT_BYTES,
            SUMMARY_JSON_NAME,
            SUMMARY_MARKDOWN_NAME,
            PublicTextEvaluationSummaryError,
            validate_public_text_evaluation_summary_files,
        )

        root = evidence_dir.resolve(strict=True)
        if not root.is_dir():
            raise PackagingError(
                "public-text-only evaluation evidence path is not a directory"
            )
        json_bytes = (root / SUMMARY_JSON_NAME).read_bytes()
        markdown_bytes = (root / SUMMARY_MARKDOWN_NAME).read_bytes()
        if (
            not json_bytes
            or not markdown_bytes
            or len(json_bytes) > MAX_REPORT_BYTES
            or len(markdown_bytes) > MAX_REPORT_BYTES
        ):
            raise PackagingError("public-text-only evaluation evidence size is invalid")
        validate_public_text_evaluation_summary_files(
            json_bytes,
            markdown_bytes,
            expected_index_sha256=index_sha256,
            expected_metas_sha256=metas_sha256,
        )
    except PackagingError:
        raise
    except (OSError, PublicTextEvaluationSummaryError) as exc:
        raise PackagingError(
            "public-text-only evaluation evidence is missing or invalid"
        ) from exc
    return (
        TrackedFile(
            PUBLIC_TEXT_EVALUATION_SUMMARY_JSON,
            "100644",
            json_bytes,
        ),
        TrackedFile(
            PUBLIC_TEXT_EVALUATION_SUMMARY_MARKDOWN,
            "100644",
            markdown_bytes,
        ),
    )


def _is_public_text_excluded_source_path(path: str) -> bool:
    return path.casefold().startswith(PUBLIC_TEXT_EXCLUDED_SOURCE_PREFIXES)


def _is_public_text_inapplicable_test(path: str) -> bool:
    return (
        path.casefold().startswith("tests/") and path not in PUBLIC_TEXT_RETAINED_TESTS
    )


def _validate_public_text_markdown_links(files: Sequence[TrackedFile]) -> None:
    paths = {item.archive_path for item in files}
    for item in files:
        if not item.archive_path.casefold().endswith(".md"):
            continue
        try:
            text = item.data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise PackagingError(
                "public-text-only Markdown is not valid UTF-8"
            ) from exc
        for match in re.finditer(r"!?\[[^\]]*\]\(([^)]+)\)", text):
            target = match.group(1).strip().strip("<>")
            if not target or target.startswith("#"):
                continue
            if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", target):
                continue
            target = target.split("#", 1)[0].split("?", 1)[0]
            resolved = posixpath.normpath(
                posixpath.join(str(PurePosixPath(item.archive_path).parent), target)
            )
            if resolved == "." or resolved.startswith("../"):
                raise PackagingError(
                    "public-text-only Markdown link escapes the archive"
                )
            if resolved not in paths and not any(
                path.startswith(resolved.rstrip("/") + "/") for path in paths
            ):
                raise PackagingError(
                    "public-text-only Markdown contains a broken local link"
                )


def _apply_public_text_profile(
    files: Sequence[TrackedFile],
    *,
    evidence_dir: Path | None = None,
    faiss_module: object | None = None,
) -> tuple[list[TrackedFile], int, dict[str, object]]:
    """Replace canonical data with deterministic public-text-only artifacts."""

    files_by_name = {item.archive_path: item for item in files}
    required = (
        PUBLIC_TEXT_SOURCE_INDEX,
        PUBLIC_TEXT_SOURCE_METAS,
        PUBLIC_TEXT_SOURCE_MANIFEST,
    )
    missing = [name for name in required if name not in files_by_name]
    if missing:
        raise PackagingError(
            "public-text-only profile requires tracked source artifact(s): "
            + ", ".join(missing)
        )
    missing_tooling = sorted(
        PUBLIC_TEXT_REQUIRED_PROFILE_NEUTRAL_TOOLING - files_by_name.keys()
    )
    if missing_tooling:
        raise PackagingError(
            "public-text-only profile requires complete profile-neutral tooling: "
            + ", ".join(missing_tooling)
        )
    _validate_public_text_provider_example(files_by_name[PUBLIC_TEXT_PROVIDER_EXAMPLE])

    try:
        from app.public_text_release import (
            PublicTextReleaseError,
            derive_public_text_release,
            validate_public_text_release_payload,
            validate_public_text_snapshot_manifest,
        )

        artifacts = derive_public_text_release(
            files_by_name[PUBLIC_TEXT_SOURCE_INDEX].data,
            files_by_name[PUBLIC_TEXT_SOURCE_METAS].data,
            source_manifest_bytes=files_by_name[PUBLIC_TEXT_SOURCE_MANIFEST].data,
            faiss_module=faiss_module,
        )
        validate_public_text_release_payload(
            artifacts.index_bytes,
            artifacts.metas_bytes,
            artifacts.release_profile_bytes,
            faiss_module=faiss_module,
        )
        validate_public_text_snapshot_manifest(artifacts.snapshot_manifest_bytes)
    except PublicTextReleaseError as exc:
        raise PackagingError(
            f"public-text-only artifact derivation failed: {exc}"
        ) from exc

    replaced_paths = {
        PUBLIC_TEXT_README,
        PUBLIC_TEXT_DATA_CARD,
        PUBLIC_TEXT_MODEL_CARD,
        PUBLIC_TEXT_CONTRIBUTING,
        PUBLIC_TEXT_RELEASE_GUIDE,
        PUBLIC_TEXT_SOURCE_INDEX,
        PUBLIC_TEXT_SOURCE_METAS,
        PUBLIC_TEXT_SOURCE_MANIFEST,
        PUBLIC_TEXT_PROFILE_MARKER,
        PUBLIC_TEXT_DERIVATION_REPORT,
        PUBLIC_TEXT_EVALUATION_SUMMARY_JSON,
        PUBLIC_TEXT_EVALUATION_SUMMARY_MARKDOWN,
    }
    selected: list[TrackedFile] = []
    profile_excluded = 0
    for item in files:
        if item.archive_path in replaced_paths:
            continue
        if _is_public_text_full_profile_holdout(item.archive_path):
            profile_excluded += 1
            continue
        if item.archive_path in PUBLIC_TEXT_STALE_ARTIFACTS:
            profile_excluded += 1
            continue
        if _is_public_text_full_profile_document(item.archive_path):
            profile_excluded += 1
            continue
        if _is_public_text_excluded_source_path(item.archive_path):
            profile_excluded += 1
            continue
        if _is_public_text_inapplicable_test(item.archive_path):
            profile_excluded += 1
            continue
        if _is_public_text_sample_artifact(item.archive_path):
            profile_excluded += 1
            continue
        selected.append(item)

    source_modes = {
        name: files_by_name[name].mode
        for name in (
            PUBLIC_TEXT_SOURCE_INDEX,
            PUBLIC_TEXT_SOURCE_METAS,
            PUBLIC_TEXT_SOURCE_MANIFEST,
        )
    }
    selected.extend(
        (
            TrackedFile(
                PUBLIC_TEXT_README,
                files_by_name.get(
                    PUBLIC_TEXT_README,
                    TrackedFile(PUBLIC_TEXT_README, "100644", b""),
                ).mode,
                _public_text_readme(artifacts),
            ),
            TrackedFile(
                PUBLIC_TEXT_DATA_CARD,
                files_by_name.get(
                    PUBLIC_TEXT_DATA_CARD,
                    TrackedFile(PUBLIC_TEXT_DATA_CARD, "100644", b""),
                ).mode,
                _public_text_data_card(artifacts),
            ),
            TrackedFile(
                PUBLIC_TEXT_MODEL_CARD,
                files_by_name.get(
                    PUBLIC_TEXT_MODEL_CARD,
                    TrackedFile(PUBLIC_TEXT_MODEL_CARD, "100644", b""),
                ).mode,
                _public_text_model_card(artifacts),
            ),
            TrackedFile(
                PUBLIC_TEXT_CONTRIBUTING,
                files_by_name.get(
                    PUBLIC_TEXT_CONTRIBUTING,
                    TrackedFile(PUBLIC_TEXT_CONTRIBUTING, "100644", b""),
                ).mode,
                _public_text_contributing(),
            ),
            TrackedFile(
                PUBLIC_TEXT_RELEASE_GUIDE,
                files_by_name.get(
                    PUBLIC_TEXT_RELEASE_GUIDE,
                    TrackedFile(PUBLIC_TEXT_RELEASE_GUIDE, "100644", b""),
                ).mode,
                _public_text_release_guide(artifacts),
            ),
            TrackedFile(
                PUBLIC_TEXT_SOURCE_INDEX,
                source_modes[PUBLIC_TEXT_SOURCE_INDEX],
                artifacts.index_bytes,
            ),
            TrackedFile(
                PUBLIC_TEXT_SOURCE_METAS,
                source_modes[PUBLIC_TEXT_SOURCE_METAS],
                artifacts.metas_bytes,
            ),
            TrackedFile(
                PUBLIC_TEXT_SOURCE_MANIFEST,
                source_modes[PUBLIC_TEXT_SOURCE_MANIFEST],
                artifacts.snapshot_manifest_bytes,
            ),
            TrackedFile(
                PUBLIC_TEXT_PROFILE_MARKER,
                "100644",
                artifacts.release_profile_bytes,
            ),
            TrackedFile(
                PUBLIC_TEXT_DERIVATION_REPORT,
                "100644",
                artifacts.derivation_report_bytes,
            ),
        )
    )
    selected.extend(
        _public_text_evaluation_files(
            evidence_dir,
            index_sha256=artifacts.index_sha256,
            metas_sha256=artifacts.metas_sha256,
        )
    )
    selected.sort(key=lambda item: item.archive_path)

    archive_paths = [item.archive_path for item in selected]
    if len(archive_paths) != len(set(archive_paths)):
        raise PackagingError(
            "public-text-only profile produced duplicate archive paths"
        )
    if any(_is_public_text_sample_artifact(path) for path in archive_paths):
        raise PackagingError("public-text-only profile retained a sample artifact")
    if any(path in PUBLIC_TEXT_STALE_ARTIFACTS for path in archive_paths):
        raise PackagingError(
            "public-text-only profile retained a stale evaluation artifact"
        )
    if any(_is_public_text_full_profile_holdout(path) for path in archive_paths):
        raise PackagingError(
            "public-text-only profile retained a full-profile query holdout artifact"
        )
    if any(_is_public_text_full_profile_document(path) for path in archive_paths):
        raise PackagingError(
            "public-text-only profile retained a full-profile document"
        )
    if any(_is_public_text_excluded_source_path(path) for path in archive_paths):
        raise PackagingError(
            "public-text-only profile retained incompatible source metadata"
        )
    if any(_is_public_text_inapplicable_test(path) for path in archive_paths):
        raise PackagingError("public-text-only profile retained an inapplicable test")
    public_readme = next(
        item.data for item in selected if item.archive_path == PUBLIC_TEXT_README
    )
    forbidden_evaluation_links = {
        path.encode("utf-8")
        for path in PUBLIC_TEXT_STALE_ARTIFACTS
        if path.startswith("docs/evaluation/")
    }
    forbidden_evaluation_links.update(
        prefix.encode("utf-8")
        for prefix in PUBLIC_TEXT_FULL_PROFILE_HOLDOUT_PREFIXES
        if prefix.startswith("docs/evaluation/")
    )
    if b"3,746" in public_readme or any(
        marker in public_readme for marker in forbidden_evaluation_links
    ):
        raise PackagingError("public-text-only README retained a full-profile claim")
    _validate_public_text_markdown_links(selected)
    return selected, profile_excluded, artifacts.safe_summary()


def _apply_release_profile(
    files: Sequence[TrackedFile],
    *,
    profile: str,
    public_evidence_dir: Path | None = None,
    faiss_module: object | None = None,
) -> tuple[list[TrackedFile], int, dict[str, object] | None]:
    if profile == FULL_RELEASE_PROFILE:
        return list(files), 0, None
    if profile == PUBLIC_TEXT_PACKAGE_PROFILE:
        return _apply_public_text_profile(
            files,
            evidence_dir=public_evidence_dir,
            faiss_module=faiss_module,
        )
    raise PackagingError(f"unsupported release profile: {profile!r}")


def _scan_files(files: Sequence[TrackedFile]) -> None:
    findings: list[SecretFinding] = []
    for item in files:
        findings.extend(scan_secrets(item.archive_path, item.data))
    if findings:
        raise SecretScanError(findings)


def required_release_paths(profile: str) -> frozenset[str]:
    """Return the mandatory review surface for one final submission profile."""

    if profile == FULL_RELEASE_PROFILE:
        return COMMON_REQUIRED_RELEASE_PATHS | FULL_REQUIRED_RELEASE_PATHS
    if profile == PUBLIC_TEXT_PACKAGE_PROFILE:
        return COMMON_REQUIRED_RELEASE_PATHS | PUBLIC_TEXT_REQUIRED_RELEASE_PATHS
    raise PackagingError(f"unsupported release profile: {profile!r}")


def _validate_required_release_paths(
    files: Sequence[TrackedFile], *, profile: str
) -> None:
    names = {item.archive_path for item in files}
    missing = sorted(required_release_paths(profile) - names)
    if missing:
        raise PackagingError(
            "tagged release is missing mandatory submission file(s): "
            + ", ".join(missing)
        )
    if RELEASE_MANIFEST_PATH in names:
        raise PackagingError(
            f"{RELEASE_MANIFEST_PATH} is generated during packaging and must not be tracked"
        )


def _release_manifest_file(
    files: Sequence[TrackedFile],
    *,
    profile: str,
    commit_sha: str,
    required_tag: str,
    tag_object_id: str,
) -> TrackedFile:
    payload_files = [
        {
            "path": item.archive_path,
            "sha256": hashlib.sha256(item.data).hexdigest(),
            "size": len(item.data),
        }
        for item in sorted(files, key=lambda value: value.archive_path)
    ]
    manifest = {
        "commit_sha": commit_sha,
        "files": payload_files,
        "profile": profile,
        "project": "meongtamjeong",
        "release_tag": required_tag,
        "schema_version": RELEASE_MANIFEST_SCHEMA_VERSION,
        "tag_object_id": tag_object_id,
    }
    data = (
        json.dumps(
            manifest,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    return TrackedFile(RELEASE_MANIFEST_PATH, "100644", data)


def _validate_contest_artifact_paths(files: Sequence[TrackedFile]) -> None:
    """Require contest example artifact paths to exist in the release archive."""

    files_by_name = {item.archive_path: item for item in files}
    contest_env = files_by_name.get(CONTEST_ENV_PATH)
    if contest_env is None:
        return
    try:
        text = contest_env.data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise PackagingError(f"{CONTEST_ENV_PATH} is not valid UTF-8") from exc

    configured: dict[str, str] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if name not in CONTEST_ARTIFACT_KEYS:
            continue
        if name in configured:
            raise PackagingError(
                f"duplicate {name} in {CONTEST_ENV_PATH}:{line_number}"
            )
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1].strip()
        configured[name] = value

    missing_keys = [key for key in CONTEST_ARTIFACT_KEYS if not configured.get(key)]
    if missing_keys:
        raise PackagingError(
            f"{CONTEST_ENV_PATH} is missing required artifact path(s): "
            + ", ".join(missing_keys)
        )

    archive_names = set(files_by_name)
    for key in CONTEST_ARTIFACT_KEYS:
        value = configured[key]
        archive_path = value[2:] if value.startswith("./") else value
        try:
            _validate_archive_name(archive_path)
        except PackagingError as exc:
            raise PackagingError(
                f"{CONTEST_ENV_PATH} {key} has an unsafe artifact path"
            ) from exc
        if archive_path not in archive_names:
            raise PackagingError(
                f"{CONTEST_ENV_PATH} {key} target is not included in release"
            )


def _zip_info(item: TrackedFile) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(item.archive_path, FIXED_ZIP_TIMESTAMP)
    info.create_system = 3
    permissions = 0o755 if item.mode == "100755" else 0o644
    info.external_attr = (stat.S_IFREG | permissions) << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    info.extra = b""
    info.comment = b""
    return info


def _write_zip(path: Path, files: Sequence[TrackedFile]) -> None:
    with zipfile.ZipFile(
        path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as archive:
        for item in files:
            archive.writestr(_zip_info(item), item.data, compresslevel=9)


def validate_zip(path: Path, expected: Sequence[TrackedFile]) -> None:
    """Validate paths, modes, bytes, CRCs, and secrets in the completed ZIP."""

    expected_by_name = {item.archive_path: item for item in expected}
    with zipfile.ZipFile(path, mode="r") as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise PackagingError("archive contains duplicate paths")
        if names != sorted(expected_by_name):
            raise PackagingError("archive file list does not match validated input")
        for info in infos:
            _validate_archive_name(info.filename)
            if _exclusion_reason(info.filename) is not None:
                raise PackagingError(
                    f"archive contains excluded path: {info.filename!r}"
                )
            unix_mode = (info.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(unix_mode) or not stat.S_ISREG(unix_mode):
                raise PackagingError(
                    f"archive entry is not a regular file: {info.filename!r}"
                )
            data = archive.read(info)
            expected_item = expected_by_name.get(info.filename)
            if expected_item is None or data != expected_item.data:
                raise PackagingError(
                    f"archive content verification failed: {info.filename!r}"
                )
            findings = scan_secrets(info.filename, data)
            if findings:
                raise SecretScanError(findings)
        bad_entry = archive.testzip()
        if bad_entry is not None:
            raise PackagingError(f"archive CRC verification failed: {bad_entry!r}")


def package_release(
    repo: Path,
    *,
    output: Path | None = None,
    check_only: bool = False,
    require_clean: bool = False,
    required_tag: str | None = None,
    profile: str = FULL_RELEASE_PROFILE,
    public_evidence_dir: Path | None = None,
    faiss_module: object | None = None,
) -> PackageResult:
    """Validate tracked working-tree files and optionally build an atomic ZIP."""

    if profile not in RELEASE_PROFILE_CHOICES:
        raise PackagingError(f"unsupported release profile: {profile!r}")
    if profile != PUBLIC_TEXT_PACKAGE_PROFILE and public_evidence_dir is not None:
        raise PackagingError(
            "--public-evidence-dir is valid only with --profile public-text-only"
        )
    root = repo.resolve(strict=True)
    target = (output or (root / "dist" / "meongtamjeong-contest.zip")).resolve(
        strict=False
    )
    files, excluded_count = collect_tracked_files(root, output=target)
    clean_required = require_clean or profile == PUBLIC_TEXT_PACKAGE_PROFILE
    commit_sha: str | None = None
    tag_object_id: str | None = None
    if profile == PUBLIC_TEXT_PACKAGE_PROFILE:
        commit_sha, tag_object_id = _require_clean_head(root, required_tag=required_tag)
    files, profile_excluded, profile_summary = _apply_release_profile(
        files,
        profile=profile,
        public_evidence_dir=public_evidence_dir,
        faiss_module=faiss_module,
    )
    excluded_count += profile_excluded
    _validate_contest_artifact_paths(files)
    _scan_files(files)
    if commit_sha is None and (clean_required or required_tag is not None):
        commit_sha, tag_object_id = _require_clean_head(root, required_tag=required_tag)
    normalized_tag: str | None = None
    if required_tag is not None:
        if commit_sha is None or tag_object_id is None:  # pragma: no cover
            raise PackagingError("tagged release requires a verified annotated tag")
        normalized_tag = required_tag.strip()
        _validate_required_release_paths(files, profile=profile)
        files = sorted(
            [
                *files,
                _release_manifest_file(
                    files,
                    profile=profile,
                    commit_sha=commit_sha,
                    required_tag=normalized_tag,
                    tag_object_id=tag_object_id,
                ),
            ],
            key=lambda item: item.archive_path,
        )
        _scan_files(files)
    total_bytes = sum(len(item.data) for item in files)
    if check_only:
        return PackageResult(
            checked_only=True,
            file_count=len(files),
            excluded_count=excluded_count,
            total_bytes=total_bytes,
            output=None,
            sha256=None,
            commit_sha=commit_sha,
            profile=profile,
            profile_summary=profile_summary,
            release_tag=normalized_tag,
            tag_object_id=tag_object_id,
            release_manifest=(RELEASE_MANIFEST_PATH if normalized_tag else None),
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        _write_zip(temporary_path, files)
        validate_zip(temporary_path, files)
        digest = hashlib.sha256(temporary_path.read_bytes()).hexdigest()
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return PackageResult(
        checked_only=False,
        file_count=len(files),
        excluded_count=excluded_count,
        total_bytes=total_bytes,
        output=str(target),
        sha256=digest,
        commit_sha=commit_sha,
        profile=profile,
        profile_summary=profile_summary,
        release_tag=normalized_tag,
        tag_object_id=tag_object_id,
        release_manifest=(RELEASE_MANIFEST_PATH if normalized_tag else None),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a deterministic contest ZIP from tracked working-tree files "
            "after exclusion and secret checks."
        )
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=BASE_DIR,
        help="Git repository root (default: this project)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output ZIP path (default: <repo>/dist/meongtamjeong-contest.zip)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate candidates without creating or changing a ZIP",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help=(
            "development-only escape hatch: allow tracked bytes that differ "
            "from HEAD (the CLI fails closed by default)"
        ),
    )
    parser.add_argument(
        "--require-tag",
        help="also require this exact Git tag to resolve to the clean HEAD commit",
    )
    parser.add_argument(
        "--profile",
        choices=RELEASE_PROFILE_CHOICES,
        default=FULL_RELEASE_PROFILE,
        help=(
            "release content profile; public-text-only derives a rights-exposure-"
            "minimizing text-only artifact set and always requires clean HEAD"
        ),
    )
    parser.add_argument(
        "--public-evidence-dir",
        type=Path,
        help=(
            "directory containing gate-generated public-text summary JSON/Markdown; "
            "required with --profile public-text-only"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.profile == PUBLIC_TEXT_PACKAGE_PROFILE and args.allow_dirty:
            raise PackagingError(
                "--allow-dirty is not permitted with --profile public-text-only"
            )
        result = package_release(
            args.repo,
            output=args.output,
            check_only=args.check_only,
            require_clean=not args.allow_dirty,
            required_tag=args.require_tag,
            profile=args.profile,
            public_evidence_dir=args.public_evidence_dir,
        )
    except PackagingError as exc:
        print(
            json.dumps(
                {"ok": False, "error": str(exc)},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {"ok": True, **result.as_dict()},
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
