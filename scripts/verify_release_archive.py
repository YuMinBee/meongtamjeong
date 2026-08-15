"""Verify and smoke-test a tagged deterministic release ZIP.

The verifier treats archive names and manifest values as untrusted.  It first
validates the complete ZIP and manifest without extracting, then writes only
validated regular files into a private temporary directory and invokes the
packaged runtime smoke test with the current Python interpreter.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import unicodedata
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.package_release import (  # noqa: E402
    FIXED_ZIP_TIMESTAMP,
    PUBLIC_TEXT_PACKAGE_PROFILE,
    RELEASE_MANIFEST_PATH,
    RELEASE_MANIFEST_SCHEMA_VERSION,
    RELEASE_PROFILE_CHOICES,
    required_release_paths,
)


DEFAULT_ARCHIVE = BASE_DIR / "dist" / "meongtamjeong-contest.zip"
DEFAULT_TIMEOUT_SECONDS = 300
MAX_ARCHIVE_ENTRIES = 20_000
MAX_MEMBER_BYTES = 2 * 1024 * 1024 * 1024
MAX_TOTAL_BYTES = 8 * 1024 * 1024 * 1024
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_COMPRESSION_RATIO = 10_000
COPY_CHUNK_BYTES = 1024 * 1024
_HEX_40_RE = re.compile(r"[0-9a-f]{40}")
_HEX_64_RE = re.compile(r"[0-9a-f]{64}")
_MANIFEST_KEYS = {
    "commit_sha",
    "files",
    "profile",
    "project",
    "release_tag",
    "schema_version",
    "tag_object_id",
}
_MANIFEST_FILE_KEYS = {"path", "sha256", "size"}
_WINDOWS_RESERVED_NAMES = {
    "aux",
    "clock$",
    "con",
    "nul",
    "prn",
    *(f"com{number}" for number in range(1, 10)),
    *(f"lpt{number}" for number in range(1, 10)),
}
_REMOVED_PYTHON_ENV_NAMES = {
    "PYTHONHOME",
    "PYTHONINSPECT",
    "PYTHONPATH",
    "PYTHONSTARTUP",
}

Runner = Callable[..., object]


class ArchiveVerificationError(RuntimeError):
    """A release archive failed closed without exposing untrusted values."""


@dataclass(frozen=True)
class ArchiveVerificationResult:
    """Value-safe successful verification summary."""

    archive_sha256: str
    commit_sha: str
    tag_object_id: str
    profile: str
    entry_count: int
    payload_file_count: int
    total_uncompressed_bytes: int
    runtime_smoke_passed: bool = True

    def as_dict(self) -> dict[str, object]:
        return {
            "archive_sha256": self.archive_sha256,
            "commit_sha": self.commit_sha,
            "entry_count": self.entry_count,
            "payload_file_count": self.payload_file_count,
            "profile": self.profile,
            "runtime_smoke_passed": self.runtime_smoke_passed,
            "tag_object_id": self.tag_object_id,
            "total_uncompressed_bytes": self.total_uncompressed_bytes,
        }


@dataclass(frozen=True)
class _ManifestFile:
    path: str
    sha256: str
    size: int


@dataclass(frozen=True)
class _InspectedArchive:
    infos: tuple[zipfile.ZipInfo, ...]
    manifest_files: Mapping[str, _ManifestFile]
    commit_sha: str
    tag_object_id: str
    total_uncompressed_bytes: int


def _validate_expected_tag(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or value.startswith("-")
        or len(value) > 255
        or any(character.isspace() or ord(character) < 32 for character in value)
    ):
        raise ArchiveVerificationError("expected release tag is invalid")
    return value


def _validate_profile(value: str) -> str:
    if value not in RELEASE_PROFILE_CHOICES:
        raise ArchiveVerificationError("expected release profile is invalid")
    return value


def _validate_archive_name(name: str) -> None:
    if (
        not isinstance(name, str)
        or not name
        or "\x00" in name
        or "\\" in name
        or name.startswith("/")
        or re.match(r"^[A-Za-z]:", name)
        or unicodedata.normalize("NFC", name) != name
    ):
        raise ArchiveVerificationError("archive contains an unsafe path")
    path = PurePosixPath(name)
    if str(path) != name or any(part in {"", ".", ".."} for part in path.parts):
        raise ArchiveVerificationError("archive contains an unsafe path")
    for part in path.parts:
        if (
            part.endswith((" ", "."))
            or ":" in part
            or any(ord(character) < 32 or ord(character) == 127 for character in part)
        ):
            raise ArchiveVerificationError("archive contains an unsafe path")
        if part.split(".", 1)[0].casefold() in _WINDOWS_RESERVED_NAMES:
            raise ArchiveVerificationError("archive contains an unsafe path")


def _validate_archive_path(path: Path) -> Path:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise ArchiveVerificationError("release archive is unavailable") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ArchiveVerificationError("release archive must be a regular file")
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise ArchiveVerificationError("release archive is unavailable") from exc


def _validate_entry_set(infos: Sequence[zipfile.ZipInfo]) -> None:
    if not infos or len(infos) > MAX_ARCHIVE_ENTRIES:
        raise ArchiveVerificationError("archive entry count is invalid")
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        raise ArchiveVerificationError("archive contains duplicate paths")
    if names != sorted(names):
        raise ArchiveVerificationError("archive entry order is not deterministic")

    canonical_names: set[str] = set()
    for name in names:
        _validate_archive_name(name)
        canonical = name.casefold()
        if canonical in canonical_names:
            raise ArchiveVerificationError("archive contains colliding paths")
        canonical_names.add(canonical)

    for canonical in canonical_names:
        parts = canonical.split("/")
        if any(
            "/".join(parts[:position]) in canonical_names
            for position in range(1, len(parts))
        ):
            raise ArchiveVerificationError("archive contains colliding paths")


def _validate_entry_metadata(
    archive: zipfile.ZipFile,
    infos: Sequence[zipfile.ZipInfo],
) -> int:
    if archive.comment:
        raise ArchiveVerificationError("archive metadata is not deterministic")
    total_bytes = 0
    for info in infos:
        if info.is_dir() or info.flag_bits & 0x1:
            raise ArchiveVerificationError("archive entry is not a regular file")
        unix_mode = (info.external_attr >> 16) & 0xFFFF
        if (
            info.create_system != 3
            or not stat.S_ISREG(unix_mode)
            or stat.S_IMODE(unix_mode) not in {0o644, 0o755}
        ):
            raise ArchiveVerificationError("archive entry is not a regular file")
        if (
            info.date_time != FIXED_ZIP_TIMESTAMP
            or info.compress_type != zipfile.ZIP_DEFLATED
            or info.extra
            or info.comment
        ):
            raise ArchiveVerificationError("archive metadata is not deterministic")
        if (
            info.file_size < 0
            or info.compress_size < 0
            or info.file_size > MAX_MEMBER_BYTES
        ):
            raise ArchiveVerificationError("archive entry size is invalid")
        total_bytes += info.file_size
        if total_bytes > MAX_TOTAL_BYTES:
            raise ArchiveVerificationError("archive total size is invalid")
        if info.file_size > 0 and (
            info.compress_size <= 0
            or info.file_size > max(1, info.compress_size) * MAX_COMPRESSION_RATIO
        ):
            raise ArchiveVerificationError("archive compression ratio is unsafe")
    return total_bytes


def _read_member_bytes(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    limit: int,
) -> bytes:
    if info.file_size > limit:
        raise ArchiveVerificationError("release manifest is too large")
    output = bytearray()
    try:
        with archive.open(info, mode="r") as source:
            while chunk := source.read(min(COPY_CHUNK_BYTES, limit + 1)):
                output.extend(chunk)
                if len(output) > limit:
                    raise ArchiveVerificationError("release manifest is too large")
    except ArchiveVerificationError:
        raise
    except Exception as exc:
        raise ArchiveVerificationError("archive CRC verification failed") from exc
    if len(output) != info.file_size:
        raise ArchiveVerificationError("archive entry size verification failed")
    return bytes(output)


def _canonical_manifest_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _parse_manifest(
    raw: bytes,
    *,
    expected_tag: str,
    expected_profile: str,
) -> tuple[dict[str, _ManifestFile], str, str]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArchiveVerificationError("release manifest is invalid") from exc
    if not isinstance(payload, Mapping) or set(payload) != _MANIFEST_KEYS:
        raise ArchiveVerificationError("release manifest schema is invalid")
    if raw != _canonical_manifest_bytes(payload):
        raise ArchiveVerificationError("release manifest is not deterministic")
    if (
        type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != RELEASE_MANIFEST_SCHEMA_VERSION
        or payload.get("project") != "meongtamjeong"
    ):
        raise ArchiveVerificationError("release manifest identity is invalid")
    release_tag = payload.get("release_tag")
    profile = payload.get("profile")
    if not isinstance(release_tag, str) or not hmac.compare_digest(
        release_tag, expected_tag
    ):
        raise ArchiveVerificationError("release manifest tag mismatch")
    if not isinstance(profile, str) or not hmac.compare_digest(
        profile, expected_profile
    ):
        raise ArchiveVerificationError("release manifest profile mismatch")
    commit_sha = payload.get("commit_sha")
    tag_object_id = payload.get("tag_object_id")
    if not isinstance(commit_sha, str) or not _HEX_40_RE.fullmatch(commit_sha):
        raise ArchiveVerificationError("release manifest commit ID is invalid")
    if not isinstance(tag_object_id, str) or not _HEX_40_RE.fullmatch(tag_object_id):
        raise ArchiveVerificationError("release manifest tag object ID is invalid")

    rows = payload.get("files")
    if not isinstance(rows, list):
        raise ArchiveVerificationError("release manifest file list is invalid")
    files: dict[str, _ManifestFile] = {}
    paths: list[str] = []
    canonical_paths: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != _MANIFEST_FILE_KEYS:
            raise ArchiveVerificationError("release manifest file record is invalid")
        path = row.get("path")
        sha256 = row.get("sha256")
        size = row.get("size")
        if not isinstance(path, str):
            raise ArchiveVerificationError("release manifest file path is invalid")
        _validate_archive_name(path)
        if path == RELEASE_MANIFEST_PATH:
            raise ArchiveVerificationError("release manifest cannot list itself")
        canonical = path.casefold()
        if path in files or canonical in canonical_paths:
            raise ArchiveVerificationError("release manifest contains duplicate paths")
        if not isinstance(sha256, str) or not _HEX_64_RE.fullmatch(sha256):
            raise ArchiveVerificationError("release manifest file hash is invalid")
        if type(size) is not int or size < 0 or size > MAX_MEMBER_BYTES:
            raise ArchiveVerificationError("release manifest file size is invalid")
        files[path] = _ManifestFile(path=path, sha256=sha256, size=size)
        paths.append(path)
        canonical_paths.add(canonical)
    if paths != sorted(paths):
        raise ArchiveVerificationError(
            "release manifest file order is not deterministic"
        )
    return files, commit_sha, tag_object_id


def _hash_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        with archive.open(info, mode="r") as source:
            while chunk := source.read(COPY_CHUNK_BYTES):
                size += len(chunk)
                if size > info.file_size or size > MAX_MEMBER_BYTES:
                    raise ArchiveVerificationError(
                        "archive entry size verification failed"
                    )
                digest.update(chunk)
    except ArchiveVerificationError:
        raise
    except Exception as exc:
        raise ArchiveVerificationError("archive CRC verification failed") from exc
    if size != info.file_size:
        raise ArchiveVerificationError("archive entry size verification failed")
    return digest.hexdigest(), size


def _validate_mandatory_paths(
    names: set[str],
    *,
    profile: str,
    mandatory_paths: Iterable[str],
) -> None:
    supplied: set[str] = set()
    for value in mandatory_paths:
        if not isinstance(value, str):
            raise ArchiveVerificationError("supplied mandatory path is invalid")
        try:
            _validate_archive_name(value)
        except ArchiveVerificationError as exc:
            raise ArchiveVerificationError(
                "supplied mandatory path is invalid"
            ) from exc
        supplied.add(value)
    try:
        required = set(required_release_paths(profile))
    except Exception as exc:
        raise ArchiveVerificationError("release mandatory path policy failed") from exc
    required.update(supplied)
    required.add(RELEASE_MANIFEST_PATH)
    if not required.issubset(names):
        raise ArchiveVerificationError("archive is missing mandatory release files")


def _inspect_archive(
    archive: zipfile.ZipFile,
    *,
    expected_tag: str,
    expected_profile: str,
    mandatory_paths: Iterable[str],
) -> _InspectedArchive:
    infos = tuple(archive.infolist())
    _validate_entry_set(infos)
    total_bytes = _validate_entry_metadata(archive, infos)
    info_by_name = {info.filename: info for info in infos}
    manifest_info = info_by_name.get(RELEASE_MANIFEST_PATH)
    if manifest_info is None:
        raise ArchiveVerificationError("release manifest is missing")
    manifest_raw = _read_member_bytes(
        archive,
        manifest_info,
        limit=MAX_MANIFEST_BYTES,
    )
    manifest_files, commit_sha, tag_object_id = _parse_manifest(
        manifest_raw,
        expected_tag=expected_tag,
        expected_profile=expected_profile,
    )

    payload_names = set(info_by_name) - {RELEASE_MANIFEST_PATH}
    if set(manifest_files) != payload_names:
        raise ArchiveVerificationError(
            "release manifest does not cover the archive exactly"
        )
    _validate_mandatory_paths(
        set(info_by_name),
        profile=expected_profile,
        mandatory_paths=mandatory_paths,
    )
    for path, expected in manifest_files.items():
        info = info_by_name[path]
        if info.file_size != expected.size:
            raise ArchiveVerificationError("release manifest file size mismatch")
        actual_sha256, actual_size = _hash_member(archive, info)
        if actual_size != expected.size or not hmac.compare_digest(
            actual_sha256, expected.sha256
        ):
            raise ArchiveVerificationError("release manifest file hash mismatch")
    try:
        bad_entry = archive.testzip()
    except Exception as exc:
        raise ArchiveVerificationError("archive CRC verification failed") from exc
    if bad_entry is not None:
        raise ArchiveVerificationError("archive CRC verification failed")
    return _InspectedArchive(
        infos=infos,
        manifest_files=manifest_files,
        commit_sha=commit_sha,
        tag_object_id=tag_object_id,
        total_uncompressed_bytes=total_bytes,
    )


def _safe_target(root: Path, archive_name: str) -> Path:
    target = root.joinpath(*PurePosixPath(archive_name).parts)
    try:
        target.relative_to(root)
    except ValueError as exc:  # pragma: no cover - guarded by path validation
        raise ArchiveVerificationError("archive extraction target is unsafe") from exc
    return target


def _extract_validated_archive(
    archive: zipfile.ZipFile,
    inspected: _InspectedArchive,
    destination: Path,
) -> None:
    root = destination.resolve(strict=True)
    for info in inspected.infos:
        target = _safe_target(root, info.filename)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            parent = target.parent.resolve(strict=True)
            parent.relative_to(root)
            digest = hashlib.sha256()
            size = 0
            with archive.open(info, mode="r") as source, target.open("xb") as output:
                while chunk := source.read(COPY_CHUNK_BYTES):
                    size += len(chunk)
                    if size > info.file_size:
                        raise ArchiveVerificationError(
                            "archive extraction size mismatch"
                        )
                    output.write(chunk)
                    digest.update(chunk)
        except ArchiveVerificationError:
            raise
        except Exception as exc:
            raise ArchiveVerificationError("archive extraction failed") from exc
        if size != info.file_size:
            raise ArchiveVerificationError("archive extraction size mismatch")
        expected = inspected.manifest_files.get(info.filename)
        if expected is not None and (
            size != expected.size
            or not hmac.compare_digest(digest.hexdigest(), expected.sha256)
        ):
            raise ArchiveVerificationError("archive extraction hash mismatch")


def _child_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key.upper() not in _REMOVED_PYTHON_ENV_NAMES
    }
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def _smoke_command(python_executable: str, profile: str) -> list[str]:
    command = [
        python_executable,
        "scripts/smoke_full_runtime.py",
        "--profile",
        profile,
        "--index",
        "data/dog_faiss.index",
        "--metas",
        "data/dog_metas.json",
    ]
    if profile == PUBLIC_TEXT_PACKAGE_PROFILE:
        command.extend(
            [
                "--release-profile-marker",
                "data/release_profile.json",
            ]
        )
    return command


def _run_runtime_smoke(
    extracted_root: Path,
    *,
    profile: str,
    runner: Runner,
    python_executable: str,
    timeout_seconds: int,
) -> None:
    if (
        not isinstance(python_executable, str)
        or not python_executable
        or type(timeout_seconds) is not int
        or timeout_seconds <= 0
    ):
        raise ArchiveVerificationError("runtime smoke configuration is invalid")
    command = _smoke_command(python_executable, profile)
    try:
        completed = runner(
            command,
            cwd=extracted_root,
            env=_child_environment(),
            shell=False,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise ArchiveVerificationError("runtime smoke timed out") from exc
    except (FileNotFoundError, OSError) as exc:
        raise ArchiveVerificationError("runtime smoke could not start") from exc
    except Exception as exc:
        raise ArchiveVerificationError("runtime smoke runner failed") from exc
    returncode = (
        completed if type(completed) is int else getattr(completed, "returncode", None)
    )
    if type(returncode) is not int:
        raise ArchiveVerificationError("runtime smoke result is invalid")
    if returncode != 0:
        raise ArchiveVerificationError("runtime smoke failed")


def _sha256_open_file(handle: Any) -> str:
    digest = hashlib.sha256()
    handle.seek(0)
    while chunk := handle.read(COPY_CHUNK_BYTES):
        digest.update(chunk)
    handle.seek(0)
    return digest.hexdigest()


def verify_release_archive(
    archive_path: Path,
    *,
    expected_tag: str,
    expected_profile: str,
    mandatory_paths: Iterable[str] = (),
    runner: Runner = subprocess.run,
    python_executable: str | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    temp_parent: Path | None = None,
) -> ArchiveVerificationResult:
    """Verify one archive and run its packaged runtime smoke test."""

    tag = _validate_expected_tag(expected_tag)
    profile = _validate_profile(expected_profile)
    path = _validate_archive_path(Path(archive_path))
    interpreter = python_executable if python_executable is not None else sys.executable
    try:
        with path.open("rb") as raw_archive:
            archive_sha256 = _sha256_open_file(raw_archive)
            with zipfile.ZipFile(raw_archive, mode="r") as archive:
                inspected = _inspect_archive(
                    archive,
                    expected_tag=tag,
                    expected_profile=profile,
                    mandatory_paths=mandatory_paths,
                )
                with tempfile.TemporaryDirectory(
                    prefix="meongtamjeong-release-verify-",
                    dir=temp_parent,
                ) as temporary:
                    extracted_root = Path(temporary)
                    _extract_validated_archive(
                        archive,
                        inspected,
                        extracted_root,
                    )
                    _run_runtime_smoke(
                        extracted_root,
                        profile=profile,
                        runner=runner,
                        python_executable=interpreter,
                        timeout_seconds=timeout_seconds,
                    )
    except ArchiveVerificationError:
        raise
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise ArchiveVerificationError("release archive is invalid") from exc
    except Exception as exc:
        raise ArchiveVerificationError("release archive verification failed") from exc
    return ArchiveVerificationResult(
        archive_sha256=archive_sha256,
        commit_sha=inspected.commit_sha,
        tag_object_id=inspected.tag_object_id,
        profile=profile,
        entry_count=len(inspected.infos),
        payload_file_count=len(inspected.manifest_files),
        total_uncompressed_bytes=inspected.total_uncompressed_bytes,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", nargs="?", type=Path)
    parser.add_argument("--archive", dest="archive_option", type=Path)
    parser.add_argument("--expected-tag", required=True)
    parser.add_argument(
        "--expected-profile",
        "--profile",
        dest="expected_profile",
        choices=RELEASE_PROFILE_CHOICES,
        required=True,
    )
    parser.add_argument(
        "--mandatory-path",
        "--require-path",
        dest="mandatory_paths",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.archive is not None and args.archive_option is not None:
        print(
            json.dumps(
                {"error": "archive path was supplied more than once", "ok": False},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    archive_path = args.archive_option or args.archive or DEFAULT_ARCHIVE
    try:
        result = verify_release_archive(
            archive_path,
            expected_tag=args.expected_tag,
            expected_profile=args.expected_profile,
            mandatory_paths=args.mandatory_paths,
            timeout_seconds=args.timeout_seconds,
        )
    except ArchiveVerificationError as exc:
        print(
            json.dumps(
                {"error": str(exc), "ok": False},
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
