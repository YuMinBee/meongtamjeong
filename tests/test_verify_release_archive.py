from __future__ import annotations

import hashlib
import json
import stat
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import pytest

from scripts import package_release
from scripts import verify_release_archive as verifier


TAG = "contest-2026-final"
COMMIT_SHA = "1" * 40
TAG_OBJECT_ID = "2" * 40


def _default_payload(
    profile: str,
    *,
    extra: dict[str, bytes] | None = None,
) -> dict[str, bytes]:
    payload = {
        path: f"fixture:{path}\n".encode()
        for path in package_release.required_release_paths(profile)
    }
    payload.update(extra or {})
    return payload


def _manifest(
    payload: dict[str, bytes],
    *,
    profile: str,
) -> dict[str, object]:
    return {
        "commit_sha": COMMIT_SHA,
        "files": [
            {
                "path": path,
                "sha256": hashlib.sha256(data).hexdigest(),
                "size": len(data),
            }
            for path, data in sorted(payload.items())
        ],
        "profile": profile,
        "project": "meongtamjeong",
        "release_tag": TAG,
        "schema_version": 1,
        "tag_object_id": TAG_OBJECT_ID,
    }


def _canonical_json(payload: dict[str, object]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()


def _zip_info(
    name: str,
    *,
    mode: int = stat.S_IFREG | 0o644,
    timestamp: tuple[int, int, int, int, int, int] = (
        1980,
        1,
        1,
        0,
        0,
        0,
    ),
) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, timestamp)
    info.create_system = 3
    info.external_attr = mode << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    info.extra = b""
    info.comment = b""
    return info


def _write_archive(
    path: Path,
    *,
    profile: str = package_release.FULL_RELEASE_PROFILE,
    payload: dict[str, bytes] | None = None,
    manifest_mutator: Callable[[dict[str, object]], None] | None = None,
    names: list[str] | None = None,
    mode_overrides: dict[str, int] | None = None,
    timestamp_overrides: dict[str, tuple[int, int, int, int, int, int]] | None = None,
    archive_comment: bytes = b"",
    canonical_manifest: bool = True,
) -> tuple[dict[str, bytes], dict[str, object]]:
    files = dict(payload or _default_payload(profile))
    manifest = _manifest(files, profile=profile)
    if manifest_mutator is not None:
        manifest_mutator(manifest)
    manifest_bytes = (
        _canonical_json(manifest)
        if canonical_manifest
        else json.dumps(manifest, ensure_ascii=False, indent=2).encode()
    )
    contents = {
        **files,
        package_release.RELEASE_MANIFEST_PATH: manifest_bytes,
    }
    entry_names = names or sorted(contents)
    with zipfile.ZipFile(
        path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as archive:
        archive.comment = archive_comment
        for name in entry_names:
            archive.writestr(
                _zip_info(
                    name,
                    mode=(mode_overrides or {}).get(
                        name,
                        stat.S_IFREG | 0o644,
                    ),
                    timestamp=(timestamp_overrides or {}).get(
                        name,
                        (1980, 1, 1, 0, 0, 0),
                    ),
                ),
                contents[name],
                compresslevel=9,
            )
    return files, manifest


class InspectingRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, object]]] = []
        self.extracted_root: Path | None = None

    def __call__(self, argv: list[str], **kwargs: object) -> object:
        self.calls.append((list(argv), dict(kwargs)))
        root = Path(kwargs["cwd"])
        self.extracted_root = root
        assert (root / "scripts" / "smoke_full_runtime.py").is_file()
        assert (root / package_release.RELEASE_MANIFEST_PATH).is_file()
        return SimpleNamespace(returncode=0)


def test_full_archive_verifies_extracts_and_runs_isolated_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_path = tmp_path / "release.zip"
    extra_path = "review/required.txt"
    _write_archive(
        archive_path,
        payload=_default_payload(
            package_release.FULL_RELEASE_PROFILE,
            extra={extra_path: b"required\n"},
        ),
    )
    monkeypatch.setenv("PYTHONPATH", "untrusted-import-path")
    monkeypatch.setenv("pythonhome", "untrusted-home")
    monkeypatch.setenv("PYTHONSTARTUP", "untrusted-startup")
    runner = InspectingRunner()

    result = verifier.verify_release_archive(
        archive_path,
        expected_tag=TAG,
        expected_profile=package_release.FULL_RELEASE_PROFILE,
        mandatory_paths=[extra_path],
        runner=runner,
        python_executable=sys.executable,
        temp_parent=tmp_path,
    )

    assert result.runtime_smoke_passed is True
    assert result.commit_sha == COMMIT_SHA
    assert result.tag_object_id == TAG_OBJECT_ID
    assert result.entry_count == result.payload_file_count + 1
    assert len(result.archive_sha256) == 64
    assert len(runner.calls) == 1
    argv, kwargs = runner.calls[0]
    assert argv == [
        sys.executable,
        "scripts/smoke_full_runtime.py",
        "--profile",
        "full",
        "--index",
        "data/dog_faiss.index",
        "--metas",
        "data/dog_metas.json",
    ]
    assert kwargs["shell"] is False
    assert kwargs["check"] is False
    assert kwargs["stdin"] == subprocess.DEVNULL
    assert kwargs["stdout"] == subprocess.DEVNULL
    assert kwargs["stderr"] == subprocess.DEVNULL
    environment = kwargs["env"]
    assert isinstance(environment, dict)
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert all(
        key.upper() not in {"PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP"}
        for key in environment
    )
    assert runner.extracted_root is not None
    assert not runner.extracted_root.exists()


def test_public_text_archive_uses_profile_marker_smoke_argument(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "public.zip"
    _write_archive(
        archive_path,
        profile=package_release.PUBLIC_TEXT_PACKAGE_PROFILE,
    )
    runner = InspectingRunner()

    verifier.verify_release_archive(
        archive_path,
        expected_tag=TAG,
        expected_profile=package_release.PUBLIC_TEXT_PACKAGE_PROFILE,
        runner=runner,
    )

    argv = runner.calls[0][0]
    assert argv[argv.index("--profile") + 1] == "public-text-only"
    assert argv[-2:] == [
        "--release-profile-marker",
        "data/release_profile.json",
    ]


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda value: value.__setitem__("schema_version", 2),
            "identity",
        ),
        (
            lambda value: value.__setitem__("project", "other-project"),
            "identity",
        ),
        (
            lambda value: value.__setitem__("release_tag", "other-tag"),
            "tag mismatch",
        ),
        (
            lambda value: value.__setitem__("profile", "public-text-only"),
            "profile mismatch",
        ),
        (
            lambda value: value.__setitem__("commit_sha", "A" * 40),
            "commit ID",
        ),
        (
            lambda value: value.__setitem__("tag_object_id", "short"),
            "tag object ID",
        ),
        (
            lambda value: value.__setitem__("unexpected", True),
            "schema",
        ),
    ],
)
def test_manifest_identity_fails_closed(
    tmp_path: Path,
    mutator: Callable[[dict[str, object]], None],
    message: str,
) -> None:
    archive_path = tmp_path / "bad-manifest.zip"
    _write_archive(archive_path, manifest_mutator=mutator)

    with pytest.raises(verifier.ArchiveVerificationError, match=message):
        verifier.verify_release_archive(
            archive_path,
            expected_tag=TAG,
            expected_profile=package_release.FULL_RELEASE_PROFILE,
            runner=InspectingRunner(),
        )


def test_manifest_must_be_canonical_json(tmp_path: Path) -> None:
    archive_path = tmp_path / "pretty-manifest.zip"
    _write_archive(archive_path, canonical_manifest=False)

    with pytest.raises(verifier.ArchiveVerificationError, match="deterministic"):
        verifier.verify_release_archive(
            archive_path,
            expected_tag=TAG,
            expected_profile=package_release.FULL_RELEASE_PROFILE,
            runner=InspectingRunner(),
        )


@pytest.mark.parametrize("failure_kind", ["missing", "extra", "hash", "size"])
def test_manifest_must_cover_every_payload_with_exact_hash_and_size(
    tmp_path: Path,
    failure_kind: str,
) -> None:
    archive_path = tmp_path / f"{failure_kind}.zip"

    def mutate(manifest: dict[str, object]) -> None:
        rows = manifest["files"]
        assert isinstance(rows, list)
        if failure_kind == "missing":
            rows.pop()
        elif failure_kind == "extra":
            rows.append(
                {
                    "path": "unpacked/ghost.txt",
                    "sha256": "0" * 64,
                    "size": 0,
                }
            )
            rows.sort(key=lambda row: row["path"])
        elif failure_kind == "hash":
            rows[0]["sha256"] = "0" * 64
        else:
            rows[0]["size"] += 1

    _write_archive(archive_path, manifest_mutator=mutate)

    with pytest.raises(verifier.ArchiveVerificationError):
        verifier.verify_release_archive(
            archive_path,
            expected_tag=TAG,
            expected_profile=package_release.FULL_RELEASE_PROFILE,
            runner=InspectingRunner(),
        )


def test_common_and_supplied_mandatory_paths_are_required(tmp_path: Path) -> None:
    archive_path = tmp_path / "missing-common.zip"
    payload = _default_payload(package_release.FULL_RELEASE_PROFILE)
    payload.pop("README.md")
    _write_archive(archive_path, payload=payload)

    with pytest.raises(verifier.ArchiveVerificationError, match="mandatory"):
        verifier.verify_release_archive(
            archive_path,
            expected_tag=TAG,
            expected_profile=package_release.FULL_RELEASE_PROFILE,
            runner=InspectingRunner(),
        )

    complete_path = tmp_path / "missing-supplied.zip"
    _write_archive(complete_path)
    with pytest.raises(verifier.ArchiveVerificationError, match="mandatory"):
        verifier.verify_release_archive(
            complete_path,
            expected_tag=TAG,
            expected_profile=package_release.FULL_RELEASE_PROFILE,
            mandatory_paths=["review/not-present.txt"],
            runner=InspectingRunner(),
        )


@pytest.mark.parametrize(
    "unsafe_name",
    [
        "../escape.txt",
        "/absolute.txt",
        "folder\\backslash.txt",
        "C:/drive.txt",
        "folder/../escape.txt",
        "folder/trailing. ",
    ],
)
def test_unsafe_archive_paths_are_rejected_without_echo(
    tmp_path: Path,
    unsafe_name: str,
) -> None:
    archive_path = tmp_path / "unsafe.zip"
    payload = _default_payload(
        package_release.FULL_RELEASE_PROFILE,
        extra={unsafe_name: b"unsafe\n"},
    )
    _write_archive(archive_path, payload=payload)

    with pytest.raises(verifier.ArchiveVerificationError) as captured:
        verifier.verify_release_archive(
            archive_path,
            expected_tag=TAG,
            expected_profile=package_release.FULL_RELEASE_PROFILE,
            runner=InspectingRunner(),
        )

    assert unsafe_name not in str(captured.value)


def test_case_collisions_and_file_prefix_collisions_are_rejected(
    tmp_path: Path,
) -> None:
    for suffix, extras in (
        ("case", {"review/A.txt": b"a", "review/a.txt": b"b"}),
        ("prefix", {"review": b"a", "review/nested.txt": b"b"}),
    ):
        archive_path = tmp_path / f"{suffix}.zip"
        _write_archive(
            archive_path,
            payload=_default_payload(
                package_release.FULL_RELEASE_PROFILE,
                extra=extras,
            ),
        )
        with pytest.raises(verifier.ArchiveVerificationError, match="colliding"):
            verifier.verify_release_archive(
                archive_path,
                expected_tag=TAG,
                expected_profile=package_release.FULL_RELEASE_PROFILE,
                runner=InspectingRunner(),
            )


def test_duplicate_zip_names_are_rejected(tmp_path: Path) -> None:
    archive_path = tmp_path / "duplicate.zip"
    payload = _default_payload(package_release.FULL_RELEASE_PROFILE)
    normal_names = sorted(
        [*payload, package_release.RELEASE_MANIFEST_PATH, "README.md"]
    )
    with pytest.warns(UserWarning, match="Duplicate name"):
        _write_archive(
            archive_path,
            payload=payload,
            names=normal_names,
        )

    with pytest.raises(verifier.ArchiveVerificationError, match="duplicate"):
        verifier.verify_release_archive(
            archive_path,
            expected_tag=TAG,
            expected_profile=package_release.FULL_RELEASE_PROFILE,
            runner=InspectingRunner(),
        )


@pytest.mark.parametrize(
    "mode",
    [
        stat.S_IFLNK | 0o777,
        stat.S_IFDIR | 0o755,
        stat.S_IFREG | 0o600,
    ],
)
def test_nonregular_or_nondeterministic_modes_are_rejected(
    tmp_path: Path,
    mode: int,
) -> None:
    archive_path = tmp_path / "bad-mode.zip"
    _write_archive(
        archive_path,
        mode_overrides={"README.md": mode},
    )

    with pytest.raises(verifier.ArchiveVerificationError, match="regular"):
        verifier.verify_release_archive(
            archive_path,
            expected_tag=TAG,
            expected_profile=package_release.FULL_RELEASE_PROFILE,
            runner=InspectingRunner(),
        )


def test_nondeterministic_order_timestamp_and_comment_are_rejected(
    tmp_path: Path,
) -> None:
    payload = _default_payload(package_release.FULL_RELEASE_PROFILE)
    all_names = sorted([*payload, package_release.RELEASE_MANIFEST_PATH])
    cases = (
        {
            "path": tmp_path / "order.zip",
            "names": list(reversed(all_names)),
        },
        {
            "path": tmp_path / "timestamp.zip",
            "timestamp_overrides": {"README.md": (2026, 7, 26, 0, 0, 0)},
        },
        {
            "path": tmp_path / "comment.zip",
            "archive_comment": b"comment",
        },
    )
    for case in cases:
        archive_path = case.pop("path")
        _write_archive(archive_path, payload=payload, **case)
        with pytest.raises(verifier.ArchiveVerificationError, match="deterministic"):
            verifier.verify_release_archive(
                archive_path,
                expected_tag=TAG,
                expected_profile=package_release.FULL_RELEASE_PROFILE,
                runner=InspectingRunner(),
            )


def test_corrupt_member_crc_fails_before_runner(tmp_path: Path) -> None:
    archive_path = tmp_path / "crc.zip"
    _write_archive(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        info = archive.getinfo("README.md")
        header_offset = info.header_offset
        compressed_size = info.compress_size
    raw = bytearray(archive_path.read_bytes())
    name_length = int.from_bytes(raw[header_offset + 26 : header_offset + 28], "little")
    extra_length = int.from_bytes(
        raw[header_offset + 28 : header_offset + 30], "little"
    )
    data_offset = header_offset + 30 + name_length + extra_length
    raw[data_offset + max(0, compressed_size // 2)] ^= 0xFF
    archive_path.write_bytes(raw)
    runner = InspectingRunner()

    with pytest.raises(verifier.ArchiveVerificationError, match="CRC"):
        verifier.verify_release_archive(
            archive_path,
            expected_tag=TAG,
            expected_profile=package_release.FULL_RELEASE_PROFILE,
            runner=runner,
        )

    assert runner.calls == []


@pytest.mark.parametrize("failure", ["returncode", "timeout", "missing"])
def test_runtime_failures_are_value_safe(
    tmp_path: Path,
    failure: str,
) -> None:
    archive_path = tmp_path / "runner.zip"
    _write_archive(archive_path)
    secret = "do-not-echo-this-value"

    def runner(argv: list[str], **kwargs: object) -> object:
        assert kwargs["stdout"] == subprocess.DEVNULL
        assert kwargs["stderr"] == subprocess.DEVNULL
        if failure == "timeout":
            raise subprocess.TimeoutExpired(argv, 1, output=secret)
        if failure == "missing":
            raise FileNotFoundError(secret)
        return SimpleNamespace(returncode=7, stdout=secret, stderr=secret)

    with pytest.raises(verifier.ArchiveVerificationError) as captured:
        verifier.verify_release_archive(
            archive_path,
            expected_tag=TAG,
            expected_profile=package_release.FULL_RELEASE_PROFILE,
            runner=runner,
        )

    assert secret not in str(captured.value)


def test_malformed_zip_and_invalid_supplied_values_fail_closed(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "not-a-zip.zip"
    archive_path.write_bytes(b"not a zip")
    with pytest.raises(verifier.ArchiveVerificationError):
        verifier.verify_release_archive(
            archive_path,
            expected_tag=TAG,
            expected_profile=package_release.FULL_RELEASE_PROFILE,
            runner=InspectingRunner(),
        )

    valid_path = tmp_path / "valid.zip"
    _write_archive(valid_path)
    with pytest.raises(verifier.ArchiveVerificationError, match="tag"):
        verifier.verify_release_archive(
            valid_path,
            expected_tag=" bad tag ",
            expected_profile=package_release.FULL_RELEASE_PROFILE,
            runner=InspectingRunner(),
        )
    with pytest.raises(verifier.ArchiveVerificationError, match="mandatory"):
        verifier.verify_release_archive(
            valid_path,
            expected_tag=TAG,
            expected_profile=package_release.FULL_RELEASE_PROFILE,
            mandatory_paths=["../secret-name"],
            runner=InspectingRunner(),
        )
