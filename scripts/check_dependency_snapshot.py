"""Verify that the active Python environment exactly matches the frozen inventory.

The public JSON result deliberately contains package names and stable error codes
only. Versions, local paths, and direct URLs never appear in command output.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, TextIO
from urllib.parse import urlsplit


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = BASE_DIR / "docs" / "dependency-report.json"
DEFAULT_LOCK = BASE_DIR / "requirements.lock.txt"
EXPECTED_COMPONENT_COUNT = 75
BOOTSTRAP_PACKAGES = frozenset({"pip", "setuptools", "wheel"})
CLIP_PACKAGE = "clip"
CLIP_ORIGIN = "https://github.com/openai/clip.git"
MAX_SNAPSHOT_BYTES = 4 * 1024 * 1024

_NAME_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.!+_-]*$")
_REVISION_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_PIN_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s]+)$")
_DIRECT_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s+@\s+(\S+)$")


class SnapshotValidationError(ValueError):
    """A frozen report or lock is malformed or internally inconsistent."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class ActiveInventoryError(ValueError):
    """Installed distribution metadata cannot form an unambiguous inventory."""


@dataclass(frozen=True)
class VcsSource:
    origin: str
    revision: str


@dataclass(frozen=True)
class ExpectedPackage:
    version: str
    vcs: VcsSource | None = None


@dataclass(frozen=True)
class ExpectedSnapshot:
    packages: Mapping[str, ExpectedPackage]


@dataclass(frozen=True)
class InstalledPackage:
    version: str
    vcs_state: str = "not-applicable"
    vcs_origin: str = ""
    vcs_revision: str = ""
    vcs_kind: str = ""


@dataclass(frozen=True)
class SnapshotCheckResult:
    expected_components: int
    installed_components: int
    issues: Mapping[str, tuple[str, ...]]

    @property
    def ok(self) -> bool:
        return not any(self.issues.values())

    def to_payload(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "expected_components": self.expected_components,
            "installed_components": self.installed_components,
            "issues": {
                key: list(names) for key, names in sorted(self.issues.items()) if names
            },
        }


def canonicalize_package_name(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("invalid package name")
    candidate = value.strip()
    if not candidate or not _NAME_RE.fullmatch(candidate):
        raise ValueError("invalid package name")
    return re.sub(r"[-_.]+", "-", candidate).lower()


def _validated_version(value: Any) -> str:
    if not isinstance(value, str) or not _VERSION_RE.fullmatch(value):
        raise ValueError("invalid package version")
    return value


def _json_without_duplicate_keys(text: str) -> Any:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SnapshotValidationError("malformed_dependency_report")
            result[key] = value
        return result

    try:
        return json.loads(text, object_pairs_hook=unique_object)
    except SnapshotValidationError:
        raise
    except (json.JSONDecodeError, TypeError, UnicodeError) as exc:
        raise SnapshotValidationError("malformed_dependency_report") from exc


def _canonical_vcs_origin(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(ord(character) < 32 or character.isspace() for character in value)
    ):
        raise ValueError("invalid VCS origin")
    candidate = value[4:] if value.startswith("git+") else value
    try:
        parsed = urlsplit(candidate)
    except ValueError as exc:
        raise ValueError("invalid VCS origin") from exc
    if (
        parsed.scheme.lower() != "https"
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("invalid VCS origin")
    path = parsed.path.rstrip("/").casefold()
    if not path or path == "/":
        raise ValueError("invalid VCS origin")
    return f"https://{parsed.hostname.casefold()}{path}"


def _parse_vcs_requirement(value: Any) -> VcsSource:
    if not isinstance(value, str) or not value.startswith("git+"):
        raise ValueError("invalid VCS requirement")
    without_prefix = value[4:]
    origin_text, separator, revision = without_prefix.rpartition("@")
    if not separator or not _REVISION_RE.fullmatch(revision):
        raise ValueError("invalid VCS requirement")
    return VcsSource(
        origin=_canonical_vcs_origin(origin_text),
        revision=revision.lower(),
    )


def _parse_lock(lock_text: str) -> dict[str, tuple[str | None, VcsSource | None]]:
    pins: dict[str, tuple[str | None, VcsSource | None]] = {}
    for raw_line in lock_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        pinned_match = _PIN_RE.fullmatch(line)
        direct_match = _DIRECT_RE.fullmatch(line)
        try:
            if pinned_match:
                name = canonicalize_package_name(pinned_match.group(1))
                version = _validated_version(pinned_match.group(2))
                pin = (version, None)
            elif direct_match:
                name = canonicalize_package_name(direct_match.group(1))
                pin = (None, _parse_vcs_requirement(direct_match.group(2)))
            else:
                raise ValueError("unsupported lock entry")
        except ValueError as exc:
            raise SnapshotValidationError("malformed_dependency_lock") from exc
        if name in pins:
            raise SnapshotValidationError("malformed_dependency_lock")
        pins[name] = pin
    if not pins:
        raise SnapshotValidationError("malformed_dependency_lock")
    return pins


def load_expected_snapshot(report_text: str, lock_text: str) -> ExpectedSnapshot:
    """Parse and cross-check the frozen 75-component report and exact lock."""

    report = _json_without_duplicate_keys(report_text)
    if (
        not isinstance(report, dict)
        or report.get("schema_version") != 1
        or report.get("report_type")
        != "installed-environment-python-dependency-inventory"
    ):
        raise SnapshotValidationError("malformed_dependency_report")

    dependencies = report.get("dependencies")
    summary = report.get("summary")
    if not isinstance(dependencies, list) or not isinstance(summary, dict):
        raise SnapshotValidationError("malformed_dependency_report")
    if (
        len(dependencies) != EXPECTED_COMPONENT_COUNT
        or type(summary.get("components")) is not int
        or summary.get("components") != EXPECTED_COMPONENT_COUNT
        or type(summary.get("installed_environment_components")) is not int
        or summary.get("installed_environment_components") != EXPECTED_COMPONENT_COUNT
        or report.get("missing_dependencies") != []
        or report.get("unknown_license_dependencies") != []
    ):
        raise SnapshotValidationError("inconsistent_dependency_report")

    packages: dict[str, ExpectedPackage] = {}
    try:
        for entry in dependencies:
            if not isinstance(entry, dict) or entry.get("installed") is not True:
                raise ValueError("invalid dependency entry")
            name = canonicalize_package_name(entry.get("canonical_name"))
            if entry.get("canonical_name") != name:
                raise ValueError("noncanonical dependency name")
            if canonicalize_package_name(entry.get("name")) != name:
                raise ValueError("dependency name mismatch")
            if canonicalize_package_name(entry.get("installed_name")) != name:
                raise ValueError("installed name mismatch")
            if name in packages:
                raise ValueError("duplicate dependency")
            packages[name] = ExpectedPackage(
                version=_validated_version(entry.get("installed_version"))
            )
    except ValueError as exc:
        raise SnapshotValidationError("malformed_dependency_report") from exc

    if set(packages) != {entry.get("canonical_name") for entry in dependencies}:
        raise SnapshotValidationError("inconsistent_dependency_report")
    if not BOOTSTRAP_PACKAGES.issubset(packages) or CLIP_PACKAGE not in packages:
        raise SnapshotValidationError("inconsistent_dependency_report")

    clip_entries = [
        entry for entry in dependencies if entry.get("canonical_name") == CLIP_PACKAGE
    ]
    if len(clip_entries) != 1:
        raise SnapshotValidationError("inconsistent_dependency_report")
    try:
        report_clip_origin = _canonical_vcs_origin(clip_entries[0].get("origin_url"))
        report_clip_source = _parse_vcs_requirement(clip_entries[0].get("source_url"))
    except ValueError as exc:
        raise SnapshotValidationError("malformed_dependency_report") from exc
    if (
        report_clip_origin != CLIP_ORIGIN
        or report_clip_source.origin != CLIP_ORIGIN
        or report_clip_source.origin != report_clip_origin
    ):
        raise SnapshotValidationError("inconsistent_dependency_report")
    packages[CLIP_PACKAGE] = ExpectedPackage(
        version=packages[CLIP_PACKAGE].version,
        vcs=report_clip_source,
    )

    pins = _parse_lock(lock_text)
    expected_lock_names = set(packages) - BOOTSTRAP_PACKAGES
    if set(pins) != expected_lock_names or BOOTSTRAP_PACKAGES.intersection(pins):
        raise SnapshotValidationError("inconsistent_dependency_lock")
    for name, package in packages.items():
        if name in BOOTSTRAP_PACKAGES:
            continue
        locked_version, locked_vcs = pins[name]
        if package.vcs is None:
            if locked_vcs is not None or locked_version != package.version:
                raise SnapshotValidationError("inconsistent_dependency_lock")
        elif (
            name != CLIP_PACKAGE
            or locked_version is not None
            or locked_vcs != package.vcs
        ):
            raise SnapshotValidationError("inconsistent_dependency_lock")

    return ExpectedSnapshot(packages=packages)


def _parse_direct_url(text: str | None) -> tuple[str, str, str, str]:
    if text is None:
        return "missing", "", "", ""
    try:
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("invalid direct URL metadata")
        vcs_info = payload.get("vcs_info")
        if not isinstance(vcs_info, dict):
            raise ValueError("missing VCS metadata")
        origin = _canonical_vcs_origin(payload.get("url"))
        kind = vcs_info.get("vcs")
        revision = vcs_info.get("commit_id")
        if not isinstance(kind, str) or not isinstance(revision, str):
            raise ValueError("invalid VCS metadata")
        if not _REVISION_RE.fullmatch(revision):
            raise ValueError("invalid VCS revision")
        return "present", origin, revision.lower(), kind.casefold()
    except (json.JSONDecodeError, TypeError, ValueError, UnicodeError):
        return "invalid", "", "", ""


def collect_installed_inventory(
    distributions: Iterable[Any],
) -> dict[str, InstalledPackage]:
    """Collect canonical active-environment metadata from injected distributions."""

    installed: dict[str, InstalledPackage] = {}
    try:
        for distribution in distributions:
            package_metadata = distribution.metadata
            raw_name = package_metadata.get("Name")
            name = canonicalize_package_name(raw_name)
            version = _validated_version(distribution.version)
            if name in installed:
                raise ActiveInventoryError("duplicate installed distribution")
            if name == CLIP_PACKAGE:
                direct_url_text = distribution.read_text("direct_url.json")
                state, origin, revision, kind = _parse_direct_url(direct_url_text)
                installed[name] = InstalledPackage(
                    version=version,
                    vcs_state=state,
                    vcs_origin=origin,
                    vcs_revision=revision,
                    vcs_kind=kind,
                )
            else:
                installed[name] = InstalledPackage(version=version)
    except ActiveInventoryError:
        raise
    except Exception as exc:
        raise ActiveInventoryError("invalid installed distribution metadata") from exc
    return installed


def check_dependency_snapshot(
    expected: ExpectedSnapshot,
    installed: Mapping[str, InstalledPackage],
) -> SnapshotCheckResult:
    expected_names = set(expected.packages)
    installed_names = set(installed)
    common_names = expected_names.intersection(installed_names)

    issues: dict[str, tuple[str, ...]] = {
        "missing": tuple(sorted(expected_names - installed_names)),
        "unexpected": tuple(sorted(installed_names - expected_names)),
        "version_mismatch": tuple(
            sorted(
                name
                for name in common_names
                if installed[name].version != expected.packages[name].version
            )
        ),
        "vcs_metadata_missing": (),
        "vcs_metadata_invalid": (),
        "vcs_type_mismatch": (),
        "vcs_origin_mismatch": (),
        "vcs_revision_mismatch": (),
    }

    if CLIP_PACKAGE in common_names:
        actual_clip = installed[CLIP_PACKAGE]
        expected_clip = expected.packages[CLIP_PACKAGE]
        if actual_clip.vcs_state == "missing":
            issues["vcs_metadata_missing"] = (CLIP_PACKAGE,)
        elif actual_clip.vcs_state != "present" or expected_clip.vcs is None:
            issues["vcs_metadata_invalid"] = (CLIP_PACKAGE,)
        else:
            if actual_clip.vcs_kind != "git":
                issues["vcs_type_mismatch"] = (CLIP_PACKAGE,)
            if actual_clip.vcs_origin != expected_clip.vcs.origin:
                issues["vcs_origin_mismatch"] = (CLIP_PACKAGE,)
            if actual_clip.vcs_revision != expected_clip.vcs.revision:
                issues["vcs_revision_mismatch"] = (CLIP_PACKAGE,)

    return SnapshotCheckResult(
        expected_components=len(expected_names),
        installed_components=len(installed_names),
        issues=issues,
    )


def evaluate_dependency_snapshot(
    report_text: str,
    lock_text: str,
    distributions: Iterable[Any],
) -> SnapshotCheckResult:
    expected = load_expected_snapshot(report_text, lock_text)
    installed = collect_installed_inventory(distributions)
    return check_dependency_snapshot(expected, installed)


def _read_snapshot_text(path: Path) -> str:
    try:
        if path.is_symlink() or not path.is_file():
            raise OSError("invalid snapshot file")
        if path.stat().st_size > MAX_SNAPSHOT_BYTES:
            raise OSError("snapshot file is too large")
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise SnapshotValidationError("snapshot_input_unavailable") from exc


def _write_payload(stream: TextIO, payload: Mapping[str, Any]) -> None:
    stream.write(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    )
    stream.write("\n")


def main(
    argv: list[str] | None = None,
    *,
    text_reader: Callable[[Path], str] = _read_snapshot_text,
    distributions_provider: Callable[[], Iterable[Any]] = metadata.distributions,
    output: TextIO | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Check the active environment against the exact dependency snapshot."
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    args = parser.parse_args(argv)
    stream = output or sys.stdout

    try:
        report_text = text_reader(args.report)
        lock_text = text_reader(args.lock)
        result = evaluate_dependency_snapshot(
            report_text,
            lock_text,
            distributions_provider(),
        )
    except SnapshotValidationError as exc:
        _write_payload(stream, {"ok": False, "errors": [exc.code]})
        return 2
    except ActiveInventoryError:
        _write_payload(stream, {"ok": False, "errors": ["malformed_active_inventory"]})
        return 2
    except Exception:
        _write_payload(stream, {"ok": False, "errors": ["dependency_check_failed"]})
        return 2

    _write_payload(stream, result.to_payload())
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
