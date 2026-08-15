from __future__ import annotations

import json
import re
from email.message import Message
from pathlib import Path, PurePosixPath

import pytest

from scripts import generate_dependency_report as report


class FakeDistribution:
    def __init__(
        self,
        name: str,
        version: str,
        *,
        license_expression: str | None = None,
        license_value: str | None = None,
        classifiers: tuple[str, ...] = (),
        project_urls: tuple[str, ...] = (),
        homepage: str | None = None,
        direct_url: str | None = None,
        files: tuple[str, ...] = (),
        location_root: Path | None = None,
    ) -> None:
        message = Message()
        message["Name"] = name
        if license_expression:
            message["License-Expression"] = license_expression
        if license_value:
            message["License"] = license_value
        if homepage:
            message["Home-page"] = homepage
        for classifier in classifiers:
            message["Classifier"] = classifier
        for project_url in project_urls:
            message["Project-URL"] = project_url
        self.metadata = message
        self.version = version
        self._direct_url = direct_url
        self.files = [PurePosixPath(value) for value in files]
        self._location_root = location_root

    def read_text(self, filename: str) -> str | None:
        if filename == "direct_url.json" and self._direct_url:
            return json.dumps({"url": self._direct_url})
        return None

    def locate_file(self, path: str | Path) -> Path:
        if self._location_root is None:
            raise FileNotFoundError(path)
        return self._location_root / path


def test_load_declarations_reads_includes_markers_and_environment(
    tmp_path: Path,
) -> None:
    (tmp_path / "requirements.txt").write_text(
        "# runtime dependencies\n"
        "FastAPI[all]>=0.115  # API framework\n"
        "numpy>=1.26,<3; python_version >= '3.9'\n"
        "clip @ git+https://token@example.test/org/clip.git@abc123\n",
        encoding="utf-8",
    )
    (tmp_path / "requirements-vlm.txt").write_text(
        "-r requirements.txt\ntransformers>=4.51\n", encoding="utf-8"
    )
    (tmp_path / "environment.yml").write_text(
        "name: demo\n"
        "dependencies:\n"
        "  - python=3.10\n"
        "  - pip\n"
        "  - pip:\n"
        "      - fastapi>=0.115\n"
        "      - pytest>=8\n",
        encoding="utf-8",
    )

    declarations = report.load_declarations(
        tmp_path,
        [
            Path("requirements.txt"),
            Path("requirements-vlm.txt"),
            Path("environment.yml"),
        ],
    )

    assert {item.canonical_name for item in declarations} == {
        "clip",
        "fastapi",
        "numpy",
        "pytest",
        "transformers",
    }
    clip = next(item for item in declarations if item.canonical_name == "clip")
    assert clip.direct_url == "git+https://example.test/org/clip.git@abc123"
    assert "token" not in clip.requirement
    numpy = next(item for item in declarations if item.canonical_name == "numpy")
    assert numpy.marker == "python_version >= '3.9'"
    fastapi_declarations = [
        item for item in declarations if item.canonical_name == "fastapi"
    ]
    assert {item.scope for item in fastapi_declarations} == {
        "runtime",
        "vlm",
        "environment",
    }
    assert all(
        item.extras == ("all",)
        for item in fastapi_declarations
        if item.scope in {"runtime", "vlm"}
    )
    assert (
        next(
            item for item in fastapi_declarations if item.scope == "environment"
        ).extras
        == ()
    )


def test_build_report_distinguishes_installed_missing_and_license_sources(
    tmp_path: Path,
) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text(
        "alpha>=1\nbeta>=2\ngamma @ https://user:secret@example.test/gamma.whl?token=x\n",
        encoding="utf-8",
    )
    declarations = report.load_declarations(tmp_path, [requirements])
    distributions = [
        FakeDistribution(
            "Alpha",
            "1.2.3",
            license_expression="MIT",
            project_urls=(
                "Documentation, https://docs.example.test/alpha?session=secret",
                "Source, https://github.com/example/alpha",
            ),
            files=("alpha-1.2.3.dist-info/licenses/LICENSE",),
        ),
        FakeDistribution(
            "beta",
            "2.0",
            license_value="a complete license body\nthat must not be copied",
            classifiers=("License :: OSI Approved :: Apache Software License",),
            homepage="https://example.test/beta",
        ),
    ]

    result = report.build_report(declarations, distributions=distributions)

    assert result["summary"] == {
        "direct_dependencies": 3,
        "installed": 2,
        "missing": 1,
        "unknown_license": 0,
    }
    by_name = {item["canonical_name"]: item for item in result["dependencies"]}
    assert by_name["alpha"]["license"] == "MIT"
    assert by_name["alpha"]["license_source"] == "License-Expression"
    assert by_name["alpha"]["source_url"] == "https://github.com/example/alpha"
    assert by_name["beta"]["license"] == "Apache-2.0"
    assert by_name["beta"]["license_source"] == "Classifier"
    assert by_name["gamma"]["installed"] is False
    assert by_name["gamma"]["source_url"] == "https://example.test/gamma.whl"
    serialized = report.render_json(result)
    assert "secret" not in serialized
    assert "complete license body" not in serialized


def test_report_rendering_is_deterministic_and_has_evidence(tmp_path: Path) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("zeta>=1\nalpha==2\n", encoding="utf-8")
    declarations = report.load_declarations(tmp_path, [requirements])
    distributions = [
        FakeDistribution("zeta", "1.1", license_value="MIT"),
        FakeDistribution("alpha", "2", license_expression="BSD-3-Clause"),
    ]

    first = report.build_report(declarations, distributions=distributions)
    second = report.build_report(
        list(reversed(declarations)), distributions=reversed(distributions)
    )

    assert report.render_json(first) == report.render_json(second)
    assert report.render_markdown(first) == report.render_markdown(second)
    assert [item["canonical_name"] for item in first["dependencies"]] == [
        "alpha",
        "zeta",
    ]
    markdown = report.render_markdown(first)
    assert "requirements.txt:2" in markdown
    assert "Only direct pip requirements" in markdown


def test_full_environment_inventory_includes_non_direct_distributions(
    tmp_path: Path,
) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("alpha>=1\n", encoding="utf-8")
    declarations = report.load_declarations(tmp_path, [requirements])
    distributions = [
        FakeDistribution("Alpha", "1.2.3", license_expression="MIT"),
        FakeDistribution("Transit", "4.5.6", license_expression="Apache-2.0"),
    ]

    result = report.build_report(
        declarations,
        distributions=distributions,
        include_installed_environment=True,
    )

    assert result["report_type"] == (
        "installed-environment-python-dependency-inventory"
    )
    assert result["summary"] == {
        "direct_dependencies": 1,
        "installed": 1,
        "missing": 0,
        "unknown_license": 0,
        "components": 2,
        "installed_environment_components": 2,
        "installed_environment_only_components": 1,
    }
    by_name = {item["canonical_name"]: item for item in result["dependencies"]}
    assert by_name["alpha"]["direct"] is True
    assert by_name["transit"]["direct"] is False
    assert by_name["transit"]["declarations"] == []
    assert by_name["transit"]["scopes"] == ["installed-environment"]
    markdown = report.render_markdown(result)
    assert "Total installed environment components: 2" in markdown
    assert "dependency graph edges are not resolved" in markdown


def test_license_file_header_fallback_uses_distribution_relative_file(
    tmp_path: Path,
) -> None:
    license_relative = "clip-1.0.dist-info/licenses/LICENSE"
    license_path = tmp_path / license_relative
    license_path.parent.mkdir(parents=True)
    license_path.write_text("MIT License\n\nCopyright example\n", encoding="utf-8")
    declarations = [
        report.parse_requirement(
            "clip==1.0",
            source_file="requirements.txt",
            line=1,
            manifest="requirements.txt",
            scope="runtime",
        )
    ]

    result = report.build_report(
        declarations,
        distributions=[
            FakeDistribution(
                "clip",
                "1.0",
                files=(license_relative,),
                location_root=tmp_path,
            )
        ],
    )

    dependency = result["dependencies"][0]
    assert dependency["license"] == "MIT"
    assert dependency["license_source"] == "license-file-header"
    assert result["summary"]["unknown_license"] == 0


def test_license_metadata_takes_priority_over_license_file_header(
    tmp_path: Path,
) -> None:
    license_relative = "sample-1.0.dist-info/LICENSE"
    license_path = tmp_path / license_relative
    license_path.parent.mkdir(parents=True)
    license_path.write_text("MIT License\n", encoding="utf-8")
    distribution = FakeDistribution(
        "sample",
        "1.0",
        license_expression="Apache-2.0",
        files=(license_relative,),
        location_root=tmp_path,
    )

    assert report._license_from_metadata(distribution) == (
        "Apache-2.0",
        "License-Expression",
    )


@pytest.mark.parametrize(
    ("relative_path", "content"),
    [
        ("../outside/LICENSE", b"MIT License\n"),
        ("sample-1.0.dist-info/LICENSE", b"MIT-like license\n"),
        ("sample-1.0.dist-info/LICENSE", b"\xff\xfe\x00\x00"),
    ],
)
def test_unsafe_or_ambiguous_license_files_remain_unknown(
    tmp_path: Path,
    relative_path: str,
    content: bytes,
) -> None:
    path = tmp_path / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    distribution = FakeDistribution(
        "sample",
        "1.0",
        files=(relative_path,),
        location_root=tmp_path,
    )

    assert report._license_from_metadata(distribution) == ("UNKNOWN", "missing")


def test_oversized_or_conflicting_license_files_remain_unknown(tmp_path: Path) -> None:
    mit_relative = "sample-1.0.dist-info/LICENSE-MIT"
    apache_relative = "sample-1.0.dist-info/LICENSE-APACHE"
    oversized_relative = "large-1.0.dist-info/LICENSE"
    for relative, content in (
        (mit_relative, b"MIT License\n"),
        (apache_relative, b"Apache License\n\nVersion 2.0, January 2004\n"),
        (oversized_relative, b"MIT License\n" + b"x" * report._MAX_LICENSE_FILE_BYTES),
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    conflicting = FakeDistribution(
        "sample",
        "1.0",
        files=(mit_relative, apache_relative),
        location_root=tmp_path,
    )
    oversized = FakeDistribution(
        "large",
        "1.0",
        files=(oversized_relative,),
        location_root=tmp_path,
    )

    assert report._license_from_metadata(conflicting) == ("UNKNOWN", "missing")
    assert report._license_from_metadata(oversized) == ("UNKNOWN", "missing")


def test_recursive_requirement_include_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("-r b.txt\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("-r a.txt\n", encoding="utf-8")

    with pytest.raises(ValueError, match="recursive requirement include"):
        report.load_declarations(tmp_path, [Path("a.txt")])


def test_main_fail_on_missing_and_writes_both_formats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "requirements.txt").write_text("not-installed>=1\n", encoding="utf-8")
    json_out = tmp_path / "report.json"
    markdown_out = tmp_path / "report.md"
    monkeypatch.setattr(report.metadata, "distributions", lambda: ())

    exit_code = report.main(
        [
            "--root",
            str(tmp_path),
            "--input",
            "requirements.txt",
            "--json-out",
            str(json_out),
            "--markdown-out",
            str(markdown_out),
            "--fail-on-missing",
        ]
    )

    assert exit_code == 1
    assert json.loads(json_out.read_text(encoding="utf-8"))["summary"]["missing"] == 1
    assert "NOT INSTALLED" in markdown_out.read_text(encoding="utf-8")


def test_github_actions_are_immutable_commit_pinned() -> None:
    root = Path(__file__).resolve().parents[1]
    workflows = sorted((root / ".github" / "workflows").glob("*.yml"))

    assert workflows
    for workflow in workflows:
        content = workflow.read_text(encoding="utf-8")
        action_refs = re.findall(r"uses:\s*(actions/[^@\s]+)@([^\s#]+)", content)
        assert action_refs, workflow.name
        for action, revision in action_refs:
            assert re.fullmatch(r"[0-9a-f]{40}", revision), (
                f"{workflow.name}: {action} must use an immutable commit SHA"
            )
