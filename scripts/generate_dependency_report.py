"""Generate a deterministic, offline inventory of direct Python dependencies.

The report intentionally reads only requirement manifests and the active
interpreter's ``importlib.metadata`` database.  It does not resolve packages,
contact a registry, or make a legal compatibility decision.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import quote, urlsplit, urlunsplit


DEFAULT_INPUTS = (
    "requirements.txt",
    "requirements-vlm.txt",
    "requirements-dev.txt",
    "requirements-analysis.txt",
    "environment.yml",
)
SCHEMA_VERSION = 1
UNKNOWN_LICENSE = "UNKNOWN"
_MAX_LICENSE_FILE_BYTES = 128 * 1024
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")
_DIRECT_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)"
    r"(?:\[(?P<extras>[^\]]+)\])?\s*@\s*(?P<url>\S+)\s*$"
)
_STANDARD_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)"
    r"(?:\[(?P<extras>[^\]]+)\])?(?P<specifier>.*)$"
)
_CLASSIFIER_LICENSES = {
    "License :: OSI Approved :: Apache Software License": "Apache-2.0",
    "License :: OSI Approved :: BSD License": "BSD (variant unspecified)",
    "License :: OSI Approved :: MIT License": "MIT",
    "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)": "MPL-2.0",
    "License :: OSI Approved :: Python Software Foundation License": "PSF-2.0",
}
_EXACT_LICENSE_FILE_HEADERS = {
    "bsd 2-clause license": "BSD-2-Clause",
    "bsd 3-clause license": "BSD-3-Clause",
    "mit license": "MIT",
    "mozilla public license version 2.0": "MPL-2.0",
}
_SOURCE_LABEL_PRIORITY = {
    "source": 0,
    "repository": 1,
    "code": 2,
    "homepage": 3,
    "documentation": 4,
}


@dataclass(frozen=True)
class Declaration:
    """One direct dependency declaration from one top-level manifest."""

    name: str
    extras: tuple[str, ...]
    specifier: str
    marker: str
    direct_url: str | None
    source_file: str
    line: int
    manifest: str
    scope: str

    @property
    def canonical_name(self) -> str:
        return canonicalize_name(self.name)

    @property
    def requirement(self) -> str:
        extras = f"[{','.join(self.extras)}]" if self.extras else ""
        if self.direct_url:
            base = f"{self.name}{extras} @ {self.direct_url}"
        else:
            base = f"{self.name}{extras}{self.specifier}"
        return f"{base}; {self.marker}" if self.marker else base


def canonicalize_name(name: str) -> str:
    """Apply the PyPA project-name comparison rule without extra dependencies."""

    return re.sub(r"[-_.]+", "-", name).lower()


def _strip_inline_comment(line: str) -> str:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return ""
    return re.split(r"\s+#", stripped, maxsplit=1)[0].strip()


def _sanitize_url(value: str | None) -> str | None:
    """Return a report-safe remote URL with credentials/query/fragment removed."""

    if not value:
        return None
    value = value.strip()
    try:
        parts = urlsplit(value)
    except ValueError:
        return None
    if parts.scheme.lower() not in {
        "http",
        "https",
        "git+http",
        "git+https",
        "git+ssh",
        "ssh",
    }:
        return None
    if not parts.hostname:
        return None
    host = parts.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    try:
        port = parts.port
    except ValueError:
        port = None
    netloc = f"{host}:{port}" if port else host
    return urlunsplit((parts.scheme.lower(), netloc, parts.path, "", ""))


def _split_marker(value: str) -> tuple[str, str]:
    requirement, separator, marker = value.partition(";")
    return requirement.strip(), marker.strip() if separator else ""


def parse_requirement(
    value: str,
    *,
    source_file: str,
    line: int,
    manifest: str,
    scope: str,
) -> Declaration:
    """Parse the subset of PEP 508 used by this repository."""

    requirement, marker = _split_marker(_strip_inline_comment(value))
    direct_match = _DIRECT_RE.fullmatch(requirement)
    if direct_match:
        safe_url = _sanitize_url(direct_match.group("url"))
        if not safe_url:
            raise ValueError(f"{source_file}:{line}: unsupported or unsafe direct URL")
        extras = tuple(
            sorted(
                {
                    item.strip().lower()
                    for item in (direct_match.group("extras") or "").split(",")
                    if item.strip()
                }
            )
        )
        return Declaration(
            name=direct_match.group("name"),
            extras=extras,
            specifier="",
            marker=marker,
            direct_url=safe_url,
            source_file=source_file,
            line=line,
            manifest=manifest,
            scope=scope,
        )

    standard_match = _STANDARD_RE.fullmatch(requirement)
    if not standard_match or not _NAME_RE.match(requirement):
        raise ValueError(
            f"{source_file}:{line}: unsupported requirement: {requirement!r}"
        )
    specifier = standard_match.group("specifier").strip()
    if specifier and not specifier.startswith(
        ("~=", "==", "!=", "<=", ">=", "<", ">", "===")
    ):
        raise ValueError(
            f"{source_file}:{line}: unsupported requirement: {requirement!r}"
        )
    extras = tuple(
        sorted(
            {
                item.strip().lower()
                for item in (standard_match.group("extras") or "").split(",")
                if item.strip()
            }
        )
    )
    return Declaration(
        name=standard_match.group("name"),
        extras=extras,
        specifier=specifier,
        marker=marker,
        direct_url=None,
        source_file=source_file,
        line=line,
        manifest=manifest,
        scope=scope,
    )


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _scope_for(path: Path) -> str:
    name = path.name.lower()
    if name == "requirements.txt":
        return "runtime"
    if name == "requirements-vlm.txt":
        return "vlm"
    if name == "requirements-dev.txt":
        return "development"
    if name in {"environment.yml", "environment.yaml"}:
        return "environment"
    return path.stem.lower().replace("_", "-")


def _parse_requirements_file(
    path: Path,
    *,
    root: Path,
    manifest: str,
    scope: str,
    ancestry: tuple[Path, ...] = (),
) -> list[Declaration]:
    resolved = path.resolve()
    if resolved in ancestry:
        chain = " -> ".join(_display_path(item, root) for item in (*ancestry, resolved))
        raise ValueError(f"recursive requirement include: {chain}")
    source_file = _display_path(resolved, root)
    declarations: list[Declaration] = []
    for line_number, raw_line in enumerate(
        resolved.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        line = _strip_inline_comment(raw_line)
        if not line:
            continue
        include_match = re.fullmatch(r"(?:-r|--requirement)\s+(.+)", line)
        if include_match:
            include_path = (resolved.parent / include_match.group(1).strip()).resolve()
            declarations.extend(
                _parse_requirements_file(
                    include_path,
                    root=root,
                    manifest=manifest,
                    scope=scope,
                    ancestry=(*ancestry, resolved),
                )
            )
            continue
        if line.startswith("-"):
            raise ValueError(
                f"{source_file}:{line_number}: unsupported pip option: {line!r}"
            )
        declarations.append(
            parse_requirement(
                line,
                source_file=source_file,
                line=line_number,
                manifest=manifest,
                scope=scope,
            )
        )
    return declarations


def _parse_environment_file(
    path: Path, *, root: Path, manifest: str, scope: str
) -> list[Declaration]:
    """Parse only the environment's nested ``pip:`` list without PyYAML."""

    source_file = _display_path(path, root)
    declarations: list[Declaration] = []
    pip_indent: int | None = None
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        content = raw_line.lstrip()
        indent = len(raw_line) - len(content)
        if pip_indent is not None and content and indent <= pip_indent:
            pip_indent = None
        if re.fullmatch(r"-\s+pip\s*:\s*(?:#.*)?", content, flags=re.IGNORECASE):
            pip_indent = indent
            continue
        if pip_indent is None or not content or content.startswith("#"):
            continue
        item_match = re.fullmatch(r"-\s+(.+)", content)
        if not item_match:
            continue
        declarations.append(
            parse_requirement(
                item_match.group(1),
                source_file=source_file,
                line=line_number,
                manifest=manifest,
                scope=scope,
            )
        )
    return declarations


def load_declarations(root: Path, inputs: Sequence[Path]) -> list[Declaration]:
    declarations: list[Declaration] = []
    for input_path in inputs:
        path = input_path if input_path.is_absolute() else root / input_path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"dependency manifest not found: {_display_path(path, root)}"
            )
        manifest = _display_path(path, root)
        scope = _scope_for(path)
        if path.suffix.lower() in {".yml", ".yaml"}:
            parsed = _parse_environment_file(
                path, root=root, manifest=manifest, scope=scope
            )
        else:
            parsed = _parse_requirements_file(
                path, root=root, manifest=manifest, scope=scope
            )
        declarations.extend(parsed)
    return sorted(
        declarations,
        key=lambda item: (
            item.canonical_name,
            item.manifest,
            item.source_file,
            item.line,
            item.requirement,
        ),
    )


def _metadata_value(dist: Any, key: str) -> str | None:
    value = dist.metadata.get(key)
    if not value:
        return None
    return str(value).strip()


def _license_from_metadata(dist: Any) -> tuple[str, str]:
    expression = _metadata_value(dist, "License-Expression")
    if expression and expression.lower() not in {"unknown", "none"}:
        return expression[:240], "License-Expression"

    license_value = _metadata_value(dist, "License")
    if (
        license_value
        and "\n" not in license_value
        and len(license_value) <= 240
        and license_value.lower() not in {"unknown", "none"}
    ):
        return license_value, "License"

    classifiers = dist.metadata.get_all("Classifier") or []
    for classifier in sorted(str(item) for item in classifiers):
        mapped = _CLASSIFIER_LICENSES.get(classifier)
        if mapped:
            return mapped, "Classifier"
    file_license = _license_from_file_headers(dist)
    if file_license:
        return file_license, "license-file-header"
    return UNKNOWN_LICENSE, "missing"


def _safe_license_file_path(dist: Any, value: str) -> Path | None:
    """Resolve a distribution file without allowing it outside its install root."""

    normalized = value.replace("\\", "/")
    try:
        relative = PurePosixPath(normalized)
    except (TypeError, ValueError):
        return None
    if (
        not relative.parts
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or re.fullmatch(r"[A-Za-z]:", relative.parts[0])
    ):
        return None

    locate_file = getattr(dist, "locate_file", None)
    if not callable(locate_file):
        return None
    try:
        install_root = Path(locate_file("")).resolve(strict=True)
        candidate = Path(locate_file(Path(*relative.parts))).resolve(strict=True)
        candidate.relative_to(install_root)
    except (
        AttributeError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return None
    return candidate if candidate.is_file() else None


def _license_from_file_content(raw: bytes) -> str | None:
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return None
    lines = [
        " ".join(line.split()).casefold() for line in text.splitlines() if line.strip()
    ]
    if not lines:
        return None

    exact = _EXACT_LICENSE_FILE_HEADERS.get(lines[0])
    if exact:
        return exact
    if (
        lines[0] == "apache license"
        and len(lines) > 1
        and lines[1] == "version 2.0, january 2004"
    ):
        return "Apache-2.0"
    return None


def _license_from_file_headers(dist: Any) -> str | None:
    """Recognize only unambiguous standard headers in bounded license files."""

    detected: set[str] = set()
    for relative_path in _license_files(dist):
        filename = PurePosixPath(relative_path).name.casefold()
        if not any(token in filename for token in ("license", "licence", "copying")):
            continue
        path = _safe_license_file_path(dist, relative_path)
        if path is None:
            continue
        try:
            size = path.stat().st_size
            if size <= 0 or size > _MAX_LICENSE_FILE_BYTES:
                continue
            with path.open("rb") as handle:
                raw = handle.read(_MAX_LICENSE_FILE_BYTES + 1)
        except OSError:
            continue
        if len(raw) > _MAX_LICENSE_FILE_BYTES:
            continue
        value = _license_from_file_content(raw)
        if value:
            detected.add(value)

    return next(iter(detected)) if len(detected) == 1 else None


def _project_urls(dist: Any) -> list[dict[str, str]]:
    values: list[dict[str, str]] = []
    for raw in dist.metadata.get_all("Project-URL") or []:
        label, separator, url = str(raw).partition(",")
        if not separator:
            continue
        safe_url = _sanitize_url(url)
        if safe_url:
            values.append({"label": label.strip(), "url": safe_url})
    homepage = _sanitize_url(_metadata_value(dist, "Home-page"))
    if homepage and all(item["url"] != homepage for item in values):
        values.append({"label": "Home-page", "url": homepage})
    return sorted(values, key=lambda item: (item["label"].lower(), item["url"]))


def _origin_url(dist: Any) -> str | None:
    try:
        raw = dist.read_text("direct_url.json")
    except (AttributeError, FileNotFoundError, OSError):
        return None
    if not raw:
        return None
    try:
        value = json.loads(raw).get("url")
    except (AttributeError, json.JSONDecodeError):
        return None
    return _sanitize_url(value)


def _license_files(dist: Any) -> list[str]:
    files = getattr(dist, "files", None) or ()
    candidates = {
        str(path).replace("\\", "/")
        for path in files
        if any(
            token in Path(str(path)).name.lower()
            for token in ("license", "licence", "copying", "notice")
        )
    }
    return sorted(candidates)


def _installed_packages(
    distributions: Iterable[Any] | None = None,
) -> dict[str, dict[str, Any]]:
    candidates: dict[str, list[dict[str, Any]]] = {}
    source = metadata.distributions() if distributions is None else distributions
    for dist in source:
        name = _metadata_value(dist, "Name")
        if not name:
            continue
        license_value, license_source = _license_from_metadata(dist)
        project_urls = _project_urls(dist)
        record = {
            "name": name,
            "version": str(getattr(dist, "version", "") or ""),
            "license": license_value,
            "license_source": license_source,
            "project_urls": project_urls,
            "origin_url": _origin_url(dist),
            "license_files": _license_files(dist),
        }
        candidates.setdefault(canonicalize_name(name), []).append(record)

    installed: dict[str, dict[str, Any]] = {}
    for name, records in candidates.items():
        installed[name] = sorted(
            records,
            key=lambda item: (
                item["version"],
                item["name"].lower(),
                item["origin_url"] or "",
            ),
        )[-1]
    return installed


def _installed_source_url(installed: Mapping[str, Any]) -> str:
    origin_url = installed.get("origin_url")
    if origin_url:
        return str(origin_url)
    project_urls = installed.get("project_urls") or []
    if project_urls:
        ranked = sorted(
            project_urls,
            key=lambda item: (
                _SOURCE_LABEL_PRIORITY.get(item["label"].strip().lower(), 99),
                item["label"].lower(),
                item["url"],
            ),
        )
        return str(ranked[0]["url"])
    package_name = quote(str(installed["name"]), safe="")
    version = quote(str(installed["version"]), safe="")
    return f"https://pypi.org/project/{package_name}/{version}/"


def _select_source_url(
    declarations: Sequence[Declaration], installed: Mapping[str, Any] | None
) -> str:
    direct_urls = sorted({item.direct_url for item in declarations if item.direct_url})
    if direct_urls:
        return direct_urls[0]
    if installed:
        return _installed_source_url(installed)
    package_name = quote(declarations[0].name, safe="")
    return f"https://pypi.org/project/{package_name}/"


def build_report(
    declarations: Sequence[Declaration],
    *,
    distributions: Iterable[Any] | None = None,
    include_installed_environment: bool = False,
) -> dict[str, Any]:
    """Build a JSON-serializable dependency and license inventory."""

    installed_by_name = _installed_packages(distributions)
    grouped: dict[str, list[Declaration]] = {}
    for declaration in declarations:
        grouped.setdefault(declaration.canonical_name, []).append(declaration)

    dependencies: list[dict[str, Any]] = []
    for canonical_name in sorted(grouped):
        group = sorted(
            grouped[canonical_name],
            key=lambda item: (
                item.manifest,
                item.source_file,
                item.line,
                item.requirement,
            ),
        )
        installed = installed_by_name.get(canonical_name)
        declarations_json = [
            {
                "manifest": item.manifest,
                "scope": item.scope,
                "source_file": item.source_file,
                "line": item.line,
                "requirement": item.requirement,
            }
            for item in group
        ]
        dependency = {
            "name": group[0].name,
            "canonical_name": canonical_name,
            "direct": True,
            "declared_in": sorted({item.manifest for item in group}),
            "scopes": sorted({item.scope for item in group}),
            "declarations": declarations_json,
            "installed": installed is not None,
            "installed_name": installed["name"] if installed else None,
            "installed_version": installed["version"] if installed else None,
            "license": installed["license"] if installed else UNKNOWN_LICENSE,
            "license_source": installed["license_source"]
            if installed
            else "not-installed",
            "source_url": _select_source_url(group, installed),
            "project_urls": installed["project_urls"] if installed else [],
            "origin_url": installed["origin_url"] if installed else None,
            "license_files": installed["license_files"] if installed else [],
        }
        dependencies.append(dependency)

    if include_installed_environment:
        for canonical_name in sorted(set(installed_by_name) - set(grouped)):
            installed = installed_by_name[canonical_name]
            dependencies.append(
                {
                    "name": installed["name"],
                    "canonical_name": canonical_name,
                    "direct": False,
                    "declared_in": [],
                    "scopes": ["installed-environment"],
                    "declarations": [],
                    "installed": True,
                    "installed_name": installed["name"],
                    "installed_version": installed["version"],
                    "license": installed["license"],
                    "license_source": installed["license_source"],
                    "source_url": _installed_source_url(installed),
                    "project_urls": installed["project_urls"],
                    "origin_url": installed["origin_url"],
                    "license_files": installed["license_files"],
                }
            )
        dependencies.sort(key=lambda item: item["canonical_name"])

    missing = [item["canonical_name"] for item in dependencies if not item["installed"]]
    unknown_licenses = [
        item["canonical_name"]
        for item in dependencies
        if item["installed"] and item["license"] == UNKNOWN_LICENSE
    ]
    manifests = sorted({item.manifest for item in declarations})
    summary = {
        "direct_dependencies": len(grouped),
        "installed": len(grouped) - len(missing),
        "missing": len(missing),
        "unknown_license": len(unknown_licenses),
    }
    limitations = [
        "License values come from installed distribution metadata and are not legal conclusions.",
        "Binary wheels and models may contain additional components governed by separate terms.",
    ]
    if include_installed_environment:
        summary.update(
            {
                "components": len(dependencies),
                "installed_environment_components": len(installed_by_name),
                "installed_environment_only_components": len(
                    set(installed_by_name) - set(grouped)
                ),
            }
        )
        limitations.insert(
            0,
            "The complete clean validation environment is inventoried, but dependency graph edges are not resolved.",
        )
    else:
        limitations.insert(
            0,
            "Only direct pip requirements are inventoried; transitive dependencies are excluded.",
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "report_type": (
            "installed-environment-python-dependency-inventory"
            if include_installed_environment
            else "direct-python-dependency-inventory"
        ),
        "environment": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "operating_system": platform.system(),
            "machine": platform.machine(),
        },
        "inputs": manifests,
        "summary": summary,
        "missing_dependencies": missing,
        "unknown_license_dependencies": unknown_licenses,
        "dependencies": dependencies,
        "limitations": limitations,
    }


def render_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _markdown_cell(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def render_markdown(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    environment = report.get("environment") or {}
    full_environment = report.get("report_type") == (
        "installed-environment-python-dependency-inventory"
    )
    lines = [
        (
            "# Installed environment dependency and license report"
            if full_environment
            else "# Direct dependency license report"
        ),
        "",
        (
            "This environment inventory was generated offline from direct dependency "
            "manifests and every installed Python distribution in the clean validation "
            "environment."
            if full_environment
            else "This dependency inventory was generated offline from direct dependency "
            "manifests and installed Python distribution metadata."
        ),
        "",
        (
            "- Validation environment: Python "
            f"{environment.get('python_version', 'UNKNOWN')} "
            f"({environment.get('python_implementation', 'UNKNOWN')}), "
            f"{environment.get('operating_system', 'UNKNOWN')} "
            f"{environment.get('machine', 'UNKNOWN')}"
        ),
        f"- Direct dependencies: {summary['direct_dependencies']}",
        f"- Installed: {summary['installed']}",
        f"- Missing: {summary['missing']}",
        f"- Installed dependencies with unknown license metadata: "
        f"{summary['unknown_license']}",
        "",
        "| Dependency | Scope | Installed version | License metadata | Source |",
        "| --- | --- | --- | --- | --- |",
    ]
    if full_environment:
        lines[9:9] = [
            f"- Total installed environment components: {summary['components']}",
            (
                "- Components not declared directly: "
                f"{summary['installed_environment_only_components']}"
            ),
        ]
    for item in report["dependencies"]:
        installed_version = item["installed_version"] or "NOT INSTALLED"
        source_url = item["source_url"]
        source = f"[upstream]({source_url})" if source_url else "UNKNOWN"
        lines.append(
            "| "
            + " | ".join(
                (
                    _markdown_cell(item["name"]),
                    _markdown_cell(", ".join(item["scopes"])),
                    _markdown_cell(installed_version),
                    _markdown_cell(f"{item['license']} ({item['license_source']})"),
                    source,
                )
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Declaration evidence",
            "",
        ]
    )
    for item in report["dependencies"]:
        lines.append(f"### {item['name']}")
        lines.append("")
        for declaration in item["declarations"]:
            location = f"{declaration['source_file']}:{declaration['line']}"
            lines.append(
                f"- `{_markdown_cell(declaration['requirement'])}` "
                f"— `{_markdown_cell(location)}` "
                f"(manifest: `{_markdown_cell(declaration['manifest'])}`)"
            )
        if item["license_files"]:
            lines.append(
                "- Installed license/notice files: "
                + ", ".join(
                    f"`{_markdown_cell(path)}`" for path in item["license_files"]
                )
            )
        lines.append("")

    lines.extend(
        [
            "## Limitations",
            "",
            *[f"- {value}" for value in report["limitations"]],
            "",
        ]
    )
    return "\n".join(lines)


def _write_output(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root used for relative paths",
    )
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        dest="inputs",
        help="dependency manifest; repeat to override the default manifest set",
    )
    parser.add_argument("--json-out", type=Path, help="write deterministic JSON")
    parser.add_argument(
        "--markdown-out", type=Path, help="write deterministic Markdown"
    )
    parser.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="exit 1 when a declared direct dependency is not installed",
    )
    parser.add_argument(
        "--include-installed-environment",
        action="store_true",
        help=(
            "include every installed distribution from the active interpreter, "
            "not only direct manifests"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    inputs = args.inputs or [Path(value) for value in DEFAULT_INPUTS]
    try:
        declarations = load_declarations(root, inputs)
        report = build_report(
            declarations,
            include_installed_environment=bool(args.include_installed_environment),
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"dependency report error: {exc}", file=sys.stderr)
        return 2

    if args.json_out:
        _write_output(args.json_out, render_json(report))
    if args.markdown_out:
        _write_output(args.markdown_out, render_markdown(report))
    if not args.json_out and not args.markdown_out:
        sys.stdout.write(render_json(report))
    if args.fail_on_missing and report["summary"]["missing"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
