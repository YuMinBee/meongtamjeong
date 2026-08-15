from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from scripts import check_dependency_snapshot as checker


REVISION = "dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1"
ORIGIN = "https://github.com/openai/CLIP.git"
BOOTSTRAP = {"pip": "25.2", "setuptools": "80.10.2", "wheel": "0.47.0"}


class FakeDistribution:
    def __init__(self, name: str, version: str, direct_url: object = None) -> None:
        self.metadata = {"Name": name}
        self.version = version
        self._direct_url = direct_url

    def read_text(self, filename: str) -> str | None:
        assert filename == "direct_url.json"
        if self._direct_url is None or isinstance(self._direct_url, str):
            return self._direct_url
        return json.dumps(self._direct_url)


def snapshot_fixture() -> tuple[str, str, list[FakeDistribution]]:
    versions = {
        **BOOTSTRAP,
        "clip": "1.0",
        **{f"package-{index:02d}": f"1.0.{index}" for index in range(71)},
    }
    dependencies = []
    for name, version in sorted(versions.items()):
        entry = {
            "canonical_name": name,
            "name": name,
            "installed_name": name,
            "installed_version": version,
            "installed": True,
            "direct": name == "clip",
        }
        if name == "clip":
            entry.update(
                {
                    "origin_url": ORIGIN,
                    "source_url": f"git+{ORIGIN}@{REVISION}",
                }
            )
        dependencies.append(entry)
    report = {
        "schema_version": 1,
        "report_type": "installed-environment-python-dependency-inventory",
        "summary": {
            "components": 75,
            "installed_environment_components": 75,
        },
        "dependencies": dependencies,
        "missing_dependencies": [],
        "unknown_license_dependencies": [],
    }
    lock_lines = []
    for name, version in sorted(versions.items()):
        if name in BOOTSTRAP:
            continue
        if name == "clip":
            lock_lines.append(f"clip @ git+{ORIGIN}@{REVISION}")
        else:
            lock_lines.append(f"{name}=={version}")
    clip_direct_url = {
        "url": ORIGIN,
        "vcs_info": {
            "vcs": "git",
            "requested_revision": REVISION,
            "commit_id": REVISION,
        },
    }
    distributions = [
        FakeDistribution(
            name,
            version,
            clip_direct_url if name == "clip" else None,
        )
        for name, version in versions.items()
    ]
    return json.dumps(report), "\n".join(lock_lines), distributions


def test_exact_75_component_snapshot_passes_with_bootstrap_and_clip_vcs() -> None:
    report, lock, distributions = snapshot_fixture()

    result = checker.evaluate_dependency_snapshot(report, lock, distributions)

    assert result.ok is True
    assert result.expected_components == 75
    assert result.installed_components == 75
    assert result.to_payload()["issues"] == {}


def test_missing_unexpected_and_version_mismatch_report_package_names_only() -> None:
    report, lock, distributions = snapshot_fixture()
    distributions = [
        distribution
        for distribution in distributions
        if distribution.metadata["Name"] != "package-00"
    ]
    for distribution in distributions:
        if distribution.metadata["Name"] == "package-01":
            distribution.version = "9.9.9"
    distributions.append(FakeDistribution("unexpected-tool", "1.0"))

    result = checker.evaluate_dependency_snapshot(report, lock, distributions)
    payload_text = json.dumps(result.to_payload(), sort_keys=True)

    assert result.ok is False
    assert result.issues["missing"] == ("package-00",)
    assert result.issues["unexpected"] == ("unexpected-tool",)
    assert result.issues["version_mismatch"] == ("package-01",)
    assert "9.9.9" not in payload_text
    assert REVISION not in payload_text
    assert "github.com" not in payload_text


def test_bootstrap_package_versions_are_part_of_the_exact_inventory() -> None:
    report, lock, distributions = snapshot_fixture()
    for distribution in distributions:
        if distribution.metadata["Name"] == "pip":
            distribution.version = "0.0"

    result = checker.evaluate_dependency_snapshot(report, lock, distributions)

    assert result.issues["version_mismatch"] == ("pip",)


@pytest.mark.parametrize(
    "mutation, expected_code",
    [
        (
            lambda document: document["dependencies"].pop(),
            "inconsistent_dependency_report",
        ),
        (
            lambda document: document["summary"].update({"components": 74}),
            "inconsistent_dependency_report",
        ),
        (
            lambda document: document["dependencies"][0].update(
                {"installed_version": ""}
            ),
            "malformed_dependency_report",
        ),
        (
            lambda document: document.update({"missing_dependencies": ["package-00"]}),
            "inconsistent_dependency_report",
        ),
    ],
)
def test_malformed_or_inconsistent_report_fails_closed(mutation, expected_code) -> None:
    report, lock, _distributions = snapshot_fixture()
    document = json.loads(report)
    mutation(document)

    with pytest.raises(checker.SnapshotValidationError) as caught:
        checker.load_expected_snapshot(json.dumps(document), lock)

    assert caught.value.code == expected_code


def test_duplicate_json_key_is_rejected() -> None:
    report, lock, _distributions = snapshot_fixture()
    malformed = report[:-1] + ',"schema_version":1}'

    with pytest.raises(checker.SnapshotValidationError) as caught:
        checker.load_expected_snapshot(malformed, lock)

    assert caught.value.code == "malformed_dependency_report"


@pytest.mark.parametrize(
    "mutate_lock, expected_code",
    [
        (lambda lines: lines[:-1], "inconsistent_dependency_lock"),
        (
            lambda lines: [*lines, "pip==25.2"],
            "inconsistent_dependency_lock",
        ),
        (
            lambda lines: [
                "package-00>=1.0.0" if line.startswith("package-00==") else line
                for line in lines
            ],
            "malformed_dependency_lock",
        ),
        (
            lambda lines: [*lines, lines[0]],
            "malformed_dependency_lock",
        ),
    ],
)
def test_lock_must_be_complete_unique_and_exactly_pinned(
    mutate_lock, expected_code
) -> None:
    report, lock, _distributions = snapshot_fixture()
    mutated = "\n".join(mutate_lock(lock.splitlines()))

    with pytest.raises(checker.SnapshotValidationError) as caught:
        checker.load_expected_snapshot(report, mutated)

    assert caught.value.code == expected_code


@pytest.mark.parametrize(
    "replacement",
    [
        f"clip @ git+https://example.org/openai/CLIP.git@{REVISION}",
        f"clip @ git+{ORIGIN}@{'a' * 40}",
    ],
)
def test_lock_clip_origin_and_revision_must_match_report(replacement: str) -> None:
    report, lock, _distributions = snapshot_fixture()
    mutated = "\n".join(
        replacement if line.startswith("clip @ ") else line
        for line in lock.splitlines()
    )

    with pytest.raises(checker.SnapshotValidationError) as caught:
        checker.load_expected_snapshot(report, mutated)

    assert caught.value.code == "inconsistent_dependency_lock"


@pytest.mark.parametrize(
    "direct_url, issue",
    [
        (None, "vcs_metadata_missing"),
        (
            {
                "url": "https://example.org/openai/CLIP.git",
                "vcs_info": {"vcs": "git", "commit_id": REVISION},
            },
            "vcs_origin_mismatch",
        ),
        (
            {
                "url": ORIGIN,
                "vcs_info": {"vcs": "git", "commit_id": "a" * 40},
            },
            "vcs_revision_mismatch",
        ),
        (
            {
                "url": ORIGIN,
                "vcs_info": {"vcs": "hg", "commit_id": REVISION},
            },
            "vcs_type_mismatch",
        ),
    ],
)
def test_active_clip_direct_url_origin_revision_and_vcs_are_verified(
    direct_url: object, issue: str
) -> None:
    report, lock, distributions = snapshot_fixture()
    clip = next(
        distribution
        for distribution in distributions
        if distribution.metadata["Name"] == "clip"
    )
    clip._direct_url = direct_url

    result = checker.evaluate_dependency_snapshot(report, lock, distributions)

    assert result.issues[issue] == ("clip",)


def test_duplicate_canonical_installed_distribution_is_rejected() -> None:
    report, lock, distributions = snapshot_fixture()
    distributions.append(FakeDistribution("Package_00", "1.0.0"))

    with pytest.raises(checker.ActiveInventoryError):
        checker.evaluate_dependency_snapshot(report, lock, distributions)


def test_cli_output_never_contains_paths_versions_or_credential_urls(
    tmp_path: Path,
) -> None:
    report, lock, distributions = snapshot_fixture()
    clip = next(
        distribution
        for distribution in distributions
        if distribution.metadata["Name"] == "clip"
    )
    clip._direct_url = {
        "url": "https://user:credential@github.com/openai/CLIP.git",
        "vcs_info": {"vcs": "git", "commit_id": REVISION},
    }
    report_path = tmp_path / "private-report.json"
    lock_path = tmp_path / "private-lock.txt"
    texts = {report_path: report, lock_path: lock}
    output = io.StringIO()

    exit_code = checker.main(
        ["--report", str(report_path), "--lock", str(lock_path)],
        text_reader=lambda path: texts[path],
        distributions_provider=lambda: distributions,
        output=output,
    )
    payload_text = output.getvalue()
    payload = json.loads(payload_text)

    assert exit_code == 1
    assert payload["issues"]["vcs_metadata_invalid"] == ["clip"]
    assert str(tmp_path) not in payload_text
    assert "credential" not in payload_text
    assert "github.com" not in payload_text
    assert REVISION not in payload_text
    assert "1.0" not in payload_text


def test_cli_malformed_snapshot_uses_stable_error_code_only() -> None:
    output = io.StringIO()

    exit_code = checker.main(
        [],
        text_reader=lambda _path: "not-json",
        distributions_provider=lambda: (),
        output=output,
    )

    assert exit_code == 2
    assert json.loads(output.getvalue()) == {
        "ok": False,
        "errors": ["malformed_dependency_report"],
    }
