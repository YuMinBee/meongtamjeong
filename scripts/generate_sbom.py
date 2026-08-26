"""Generate deterministic SPDX 2.3 and CycloneDX 1.6 SBOM documents.

The source is the repository's validated dependency inventory rather than the
caller's active Python environment.  Dependency edges are intentionally limited
to declared direct dependencies because the inventory does not claim to be a
resolved dependency graph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import uuid
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "docs" / "dependency-report.json"
DEFAULT_SPDX = ROOT / "SBOM.spdx.json"
DEFAULT_CYCLONEDX = ROOT / "SBOM.cdx.json"
DEFAULT_MARKDOWN = ROOT / "docs" / "SBOM.md"
PROJECT_NAME = "MeongTamjeong"
PROJECT_REPOSITORY = "https://github.com/YuMinBee/meongtamjeong"
PROJECT_LICENSE = "Apache-2.0"
TOOL_NAME = "meongtamjeong-generate-sbom/1.0"

_TIMESTAMP_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
)
_VALID_LICENSE_IDS = {
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "CC0-1.0",
    "CNRI-Python",
    "ISC",
    "LLVM-exception",
    "MIT",
    "MIT-CMU",
    "MPL-2.0",
    "PSF-2.0",
}
_LICENSE_ALIASES = {
    "Apache": "Apache-2.0",
    "Apache 2.0": "Apache-2.0",
    "Apache 2.0 License": "Apache-2.0",
    "ISC License": "ISC",
    "MIT License": "MIT",
    "Mozilla Public License 2.0 (MPL 2.0)": "MPL-2.0",
    "Python Software Foundation License": "PSF-2.0",
}

_PURPOSES = {
    "accelerate": "선택 연구 모델 로딩과 장치 배치",
    "clip": "이미지-텍스트 공통 임베딩 생성",
    "faiss-cpu": "고정 벡터 인덱스의 근접 검색",
    "fastapi": "검색·문의 HTTP API 제공",
    "ftfy": "CLIP 입력 텍스트 정규화",
    "httpx": "API 회귀 테스트와 선택 네트워크 평가",
    "numpy": "벡터·수치 연산",
    "opencv-python-headless": "오프라인 이미지 품질·영역 처리",
    "openpyxl": "선택 분석 결과의 XLSX 입출력",
    "pandas": "선택 평가·분석 데이터 처리",
    "pillow": "업로드 이미지 검증과 변환",
    "pydantic": "API 요청·응답 스키마 검증",
    "pytest": "자동 회귀 테스트",
    "python-dotenv": "로컬 환경변수 파일 로딩",
    "python-multipart": "이미지 multipart 업로드 파싱",
    "regex": "CLIP 토큰화 보조",
    "requests": "공공 API와 원문 상태 조회",
    "ruff": "정적 검사와 코드 형식 검증",
    "sentencepiece": "선택 연구 모델 토크나이저",
    "torch": "CLIP 및 선택 연구 모델 추론",
    "torchvision": "이미지 전처리와 선택 객체 영역 추출",
    "tqdm": "오프라인 처리 진행률 표시",
    "transformers": "선택 DINO·VLM 연구 경로",
    "urllib3": "HTTP 연결·재시도 계층",
    "uvicorn": "FastAPI ASGI 서버 실행",
}


class SbomError(ValueError):
    """Raised when source inventory data cannot produce a reliable SBOM."""


def _canonical_name(item: dict[str, Any]) -> str:
    value = str(item.get("canonical_name") or item.get("name") or "").strip()
    if not value:
        raise SbomError("dependency has no canonical name")
    return value.casefold().replace("_", "-")


def _version(item: dict[str, Any]) -> str:
    value = str(item.get("installed_version") or "").strip()
    return value or "UNKNOWN"


def _is_vcs_requirement(item: dict[str, Any]) -> bool:
    declarations = item.get("declarations") or []
    return any(
        "git+" in str(declaration.get("requirement") or "").casefold()
        for declaration in declarations
        if isinstance(declaration, dict)
    )


def _purl(item: dict[str, Any]) -> str | None:
    version = _version(item)
    if version == "UNKNOWN" or _is_vcs_requirement(item):
        return None
    name = quote(_canonical_name(item), safe=".-")
    return f"pkg:pypi/{name}@{quote(version, safe='.+-')}"


def _spdx_id(item: dict[str, Any]) -> str:
    name = re.sub(r"[^A-Za-z0-9.-]+", "-", _canonical_name(item)).strip("-.")
    digest = hashlib.sha256(_canonical_name(item).encode("utf-8")).hexdigest()[:8]
    return f"SPDXRef-Package-{name}-{digest}"


def _bom_ref(item: dict[str, Any]) -> str:
    return _purl(item) or (
        "urn:meongtamjeong:dependency:"
        f"{quote(_canonical_name(item), safe='.-')}@"
        f"{quote(_version(item), safe='.+-')}"
    )


def _spdx_license(raw_value: object) -> str:
    raw = str(raw_value or "").strip()
    if not raw:
        return "NOASSERTION"
    raw = _LICENSE_ALIASES.get(raw, raw)
    tokens = re.findall(r"[A-Za-z0-9.+-]+|\(|\)", raw)
    identifiers = {
        token
        for token in tokens
        if token not in {"AND", "OR", "WITH", "(", ")"}
    }
    operators_removed = " ".join(tokens)
    if identifiers and identifiers <= _VALID_LICENSE_IDS:
        if re.sub(r"\s+", " ", raw).strip() == operators_removed:
            return raw
    if raw in _VALID_LICENSE_IDS:
        return raw
    return "NOASSERTION"


def _license_comment(item: dict[str, Any]) -> str:
    raw = str(item.get("license") or "UNKNOWN").strip() or "UNKNOWN"
    source = str(item.get("license_source") or "unknown source").strip()
    return f"Installed distribution metadata: {raw} ({source})."


def load_report(path: Path) -> dict[str, Any]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SbomError(f"cannot read dependency report: {path}") from exc
    dependencies = report.get("dependencies")
    if not isinstance(dependencies, list) or not dependencies:
        raise SbomError("dependency report has no dependencies")
    seen: set[str] = set()
    for item in dependencies:
        if not isinstance(item, dict):
            raise SbomError("dependency entry is not an object")
        name = _canonical_name(item)
        if name in seen:
            raise SbomError(f"duplicate dependency: {name}")
        seen.add(name)
        if not item.get("installed") or not item.get("installed_version"):
            raise SbomError(f"dependency is not installed in snapshot: {name}")
    report["dependencies"] = sorted(dependencies, key=_canonical_name)
    return report


def _source_digest(report: dict[str, Any]) -> str:
    payload = json.dumps(
        report, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_spdx(
    report: dict[str, Any], *, created: str, project_version: str
) -> dict[str, Any]:
    digest = _source_digest(report)
    packages: list[dict[str, Any]] = [
        {
            "SPDXID": "SPDXRef-RootPackage",
            "name": PROJECT_NAME,
            "versionInfo": project_version,
            "downloadLocation": PROJECT_REPOSITORY,
            "filesAnalyzed": False,
            "licenseConcluded": PROJECT_LICENSE,
            "licenseDeclared": PROJECT_LICENSE,
            "copyrightText": "Copyright YuMinBee",
            "homepage": PROJECT_REPOSITORY,
        }
    ]
    relationships: list[dict[str, str]] = [
        {
            "spdxElementId": "SPDXRef-DOCUMENT",
            "relationshipType": "DESCRIBES",
            "relatedSpdxElement": "SPDXRef-RootPackage",
        }
    ]
    for item in report["dependencies"]:
        package: dict[str, Any] = {
            "SPDXID": _spdx_id(item),
            "name": _canonical_name(item),
            "versionInfo": _version(item),
            "downloadLocation": item.get("source_url") or "NOASSERTION",
            "filesAnalyzed": False,
            "licenseConcluded": "NOASSERTION",
            "licenseDeclared": _spdx_license(item.get("license")),
            "copyrightText": "NOASSERTION",
            "comment": _license_comment(item),
        }
        purl = _purl(item)
        if purl:
            package["externalRefs"] = [
                {
                    "referenceCategory": "PACKAGE-MANAGER",
                    "referenceType": "purl",
                    "referenceLocator": purl,
                }
            ]
        packages.append(package)
        if bool(item.get("direct")):
            relationships.append(
                {
                    "spdxElementId": "SPDXRef-RootPackage",
                    "relationshipType": "DEPENDS_ON",
                    "relatedSpdxElement": _spdx_id(item),
                }
            )

    return {
        "SPDXID": "SPDXRef-DOCUMENT",
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "name": f"{PROJECT_NAME}-{project_version}-SBOM",
        "documentNamespace": f"{PROJECT_REPOSITORY}/sbom/{digest}",
        "creationInfo": {
            "created": created,
            "creators": [f"Tool: {TOOL_NAME}"],
            "comment": (
                "Generated from the validated clean-environment inventory. "
                "Only declared direct dependency edges are asserted."
            ),
        },
        "documentDescribes": ["SPDXRef-RootPackage"],
        "packages": packages,
        "relationships": relationships,
    }


def _cyclonedx_license(item: dict[str, Any]) -> list[dict[str, Any]]:
    expression = _spdx_license(item.get("license"))
    if expression != "NOASSERTION":
        return [{"expression": expression}]
    raw = str(item.get("license") or "UNKNOWN").strip() or "UNKNOWN"
    return [{"license": {"name": raw}}]


def build_cyclonedx(
    report: dict[str, Any], *, created: str, project_version: str
) -> dict[str, Any]:
    digest = _source_digest(report)
    root_ref = f"pkg:github/YuMinBee/meongtamjeong@{quote(project_version)}"
    components: list[dict[str, Any]] = []
    direct_refs: list[str] = []
    dependency_graph: list[dict[str, Any]] = []
    for item in report["dependencies"]:
        ref = _bom_ref(item)
        component: dict[str, Any] = {
            "type": "library",
            "bom-ref": ref,
            "name": _canonical_name(item),
            "version": _version(item),
            "licenses": _cyclonedx_license(item),
            "properties": [
                {"name": "meongtamjeong:direct", "value": str(bool(item.get("direct"))).lower()},
                {
                    "name": "meongtamjeong:scopes",
                    "value": ",".join(str(value) for value in item.get("scopes") or []),
                },
                {
                    "name": "meongtamjeong:license-source",
                    "value": str(item.get("license_source") or "unknown"),
                },
            ],
        }
        purl = _purl(item)
        if purl:
            component["purl"] = purl
        source_url = str(item.get("source_url") or "").strip()
        if source_url:
            component["externalReferences"] = [
                {"type": "vcs", "url": source_url}
            ]
        components.append(component)
        dependency_graph.append({"ref": ref, "dependsOn": []})
        if bool(item.get("direct")):
            direct_refs.append(ref)

    serial = uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"{PROJECT_REPOSITORY}/sbom/{digest}/{created}/{project_version}",
    )
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "serialNumber": f"urn:uuid:{serial}",
        "version": 1,
        "metadata": {
            "timestamp": created,
            "tools": {
                "components": [
                    {
                        "type": "application",
                        "name": "meongtamjeong-generate-sbom",
                        "version": "1.0",
                    }
                ]
            },
            "component": {
                "type": "application",
                "bom-ref": root_ref,
                "name": PROJECT_NAME,
                "version": project_version,
                "purl": root_ref,
                "licenses": [{"expression": PROJECT_LICENSE}],
                "externalReferences": [
                    {"type": "vcs", "url": PROJECT_REPOSITORY}
                ],
                "properties": [
                    {
                        "name": "meongtamjeong:dependency-graph-complete",
                        "value": "false",
                    }
                ],
            },
        },
        "components": components,
        "dependencies": [
            {"ref": root_ref, "dependsOn": sorted(direct_refs)},
            *dependency_graph,
        ],
    }


def render_markdown(
    report: dict[str, Any], *, created: str, project_version: str
) -> str:
    direct = [item for item in report["dependencies"] if bool(item.get("direct"))]
    lines = [
        "# 소프트웨어 자재명세서(SBOM)",
        "",
        f"- 프로젝트: `{PROJECT_NAME}` `{project_version}`",
        f"- 생성 기준 시각: `{created}`",
        f"- 전체 설치 환경 구성요소: **{len(report['dependencies'])}개**",
        f"- 선언된 직접 의존성: **{len(direct)}개**",
        "- 기계 판독 형식: `SBOM.spdx.json`(SPDX 2.3), "
        "`SBOM.cdx.json`(CycloneDX 1.6)",
        "",
        "이 목록은 `docs/dependency-report.json`에 고정된 깨끗한 검증 환경을 "
        "변환한 것입니다. 직접 선언 관계만 프로젝트의 `DEPENDS_ON`으로 기록하며 "
        "전이 의존성 간 그래프는 추정하지 않습니다. 라이선스 값은 설치 배포판 "
        "메타데이터의 감사 단서이며 법률 자문이 아닙니다.",
        "",
        "## 직접 의존성",
        "",
        "| 라이브러리 | 버전 | 라이선스 메타데이터 | 사용 목적 | 공식 저장소 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in direct:
        name = _canonical_name(item)
        source_url = str(item.get("source_url") or "").strip()
        source = f"[upstream]({source_url})" if source_url else "미상"
        purpose = _PURPOSES.get(name, "프로젝트 의존성")
        lines.append(
            f"| `{name}` | `{_version(item)}` | "
            f"{str(item.get('license') or 'UNKNOWN')} | {purpose} | {source} |"
        )
    lines.extend(
        [
            "",
            "## 포함 범위와 한계",
            "",
            "- OpenAI CLIP은 공식 Git 커밋을 고정한 VCS 의존성이므로 PyPI purl을 "
            "허위로 부여하지 않고 원본 저장소 URL을 기록합니다.",
            "- 사전학습 모델 가중치, 공공 공고 데이터, 공고 사진과 파생 벡터는 "
            "Python 패키지 SBOM과 별도의 모델·데이터 조건을 따릅니다.",
            "- 선택 DINO·VLM 연구 가중치와 진행 중인 로컬 학습 산출물은 출품 "
            "실행 프로필과 이 SBOM에 포함하지 않습니다.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--spdx-out", type=Path, default=DEFAULT_SPDX)
    parser.add_argument("--cyclonedx-out", type=Path, default=DEFAULT_CYCLONEDX)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument(
        "--created",
        required=True,
        help="reproducible UTC timestamp in YYYY-MM-DDTHH:MM:SSZ form",
    )
    parser.add_argument("--project-version", default="submission-2026")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not _TIMESTAMP_PATTERN.fullmatch(args.created):
        print("SBOM error: --created must use YYYY-MM-DDTHH:MM:SSZ", file=__import__("sys").stderr)
        return 2
    try:
        report = load_report(args.report.resolve())
        spdx = build_spdx(
            report, created=args.created, project_version=args.project_version
        )
        cyclonedx = build_cyclonedx(
            report, created=args.created, project_version=args.project_version
        )
        markdown = render_markdown(
            report, created=args.created, project_version=args.project_version
        )
        _write(args.spdx_out, _render_json(spdx))
        _write(args.cyclonedx_out, _render_json(cyclonedx))
        _write(args.markdown_out, markdown)
    except (OSError, SbomError) as exc:
        print(f"SBOM error: {exc}", file=__import__("sys").stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
