from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import generate_sbom as sbom


def dependency(
    name: str,
    *,
    version: str = "1.2.3",
    license_name: str = "MIT",
    direct: bool = True,
    requirement: str | None = None,
) -> dict[str, object]:
    declarations = []
    if requirement:
        declarations.append({"requirement": requirement})
    return {
        "canonical_name": name,
        "name": name,
        "installed": True,
        "installed_version": version,
        "license": license_name,
        "license_source": "License-Expression",
        "source_url": f"https://example.test/{name}",
        "direct": direct,
        "scopes": ["runtime"],
        "declarations": declarations,
    }


def report(*dependencies: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "dependencies": list(dependencies),
        "limitations": ["dependency graph edges are not resolved"],
    }


def test_spdx_asserts_only_direct_dependency_edges() -> None:
    payload = report(
        dependency("fastapi"),
        dependency("starlette", direct=False),
    )

    result = sbom.build_spdx(
        payload, created="2026-08-26T08:00:00Z", project_version="test"
    )

    dependency_edges = [
        value
        for value in result["relationships"]
        if value["relationshipType"] == "DEPENDS_ON"
    ]
    assert len(dependency_edges) == 1
    assert dependency_edges[0]["relatedSpdxElement"] == sbom._spdx_id(
        payload["dependencies"][0]
    )
    assert result["spdxVersion"] == "SPDX-2.3"
    assert result["packages"][0]["licenseDeclared"] == "Apache-2.0"


def test_cyclonedx_does_not_mislabel_vcs_dependency_as_pypi() -> None:
    clip = dependency(
        "clip",
        version="1.0",
        requirement=(
            "clip @ git+https://github.com/openai/CLIP.git@"
            "dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1"
        ),
    )

    result = sbom.build_cyclonedx(
        report(clip), created="2026-08-26T08:00:00Z", project_version="test"
    )

    component = result["components"][0]
    assert "purl" not in component
    assert component["bom-ref"].startswith("urn:meongtamjeong:dependency:clip@")
    assert result["specVersion"] == "1.6"


def test_unknown_license_is_preserved_without_invalid_spdx_expression() -> None:
    item = dependency("mystery", license_name="BSD (variant unspecified)")

    spdx = sbom.build_spdx(
        report(item), created="2026-08-26T08:00:00Z", project_version="test"
    )
    cyclonedx = sbom.build_cyclonedx(
        report(item), created="2026-08-26T08:00:00Z", project_version="test"
    )

    assert spdx["packages"][1]["licenseDeclared"] == "NOASSERTION"
    assert "BSD (variant unspecified)" in spdx["packages"][1]["comment"]
    assert cyclonedx["components"][0]["licenses"] == [
        {"license": {"name": "BSD (variant unspecified)"}}
    ]


def test_cli_writes_deterministic_documents(tmp_path: Path) -> None:
    source = tmp_path / "report.json"
    source.write_text(
        json.dumps(report(dependency("fastapi"))), encoding="utf-8"
    )
    spdx_path = tmp_path / "SBOM.spdx.json"
    cdx_path = tmp_path / "SBOM.cdx.json"
    markdown_path = tmp_path / "SBOM.md"
    argv = [
        "--report",
        str(source),
        "--spdx-out",
        str(spdx_path),
        "--cyclonedx-out",
        str(cdx_path),
        "--markdown-out",
        str(markdown_path),
        "--created",
        "2026-08-26T08:00:00Z",
        "--project-version",
        "test",
    ]

    assert sbom.main(argv) == 0
    first = (spdx_path.read_bytes(), cdx_path.read_bytes(), markdown_path.read_bytes())
    assert sbom.main(argv) == 0
    second = (spdx_path.read_bytes(), cdx_path.read_bytes(), markdown_path.read_bytes())

    assert first == second
    assert json.loads(first[0])["creationInfo"]["created"].endswith("Z")
    assert b"CycloneDX" in first[1]


def test_committed_sbom_matches_dependency_snapshot(tmp_path: Path) -> None:
    spdx_path = tmp_path / "SBOM.spdx.json"
    cdx_path = tmp_path / "SBOM.cdx.json"
    markdown_path = tmp_path / "SBOM.md"

    assert (
        sbom.main(
            [
                "--report",
                str(sbom.DEFAULT_REPORT),
                "--spdx-out",
                str(spdx_path),
                "--cyclonedx-out",
                str(cdx_path),
                "--markdown-out",
                str(markdown_path),
                "--created",
                "2026-08-26T08:05:00Z",
                "--project-version",
                "submission-2026",
            ]
        )
        == 0
    )

    assert spdx_path.read_bytes() == sbom.DEFAULT_SPDX.read_bytes()
    assert cdx_path.read_bytes() == sbom.DEFAULT_CYCLONEDX.read_bytes()
    assert markdown_path.read_bytes() == sbom.DEFAULT_MARKDOWN.read_bytes()


@pytest.mark.parametrize("created", ["2026-08-26", "now", ""])
def test_cli_rejects_non_reproducible_timestamp(
    tmp_path: Path, created: str
) -> None:
    source = tmp_path / "report.json"
    source.write_text(
        json.dumps(report(dependency("fastapi"))), encoding="utf-8"
    )

    assert sbom.main(["--report", str(source), "--created", created]) == 2
