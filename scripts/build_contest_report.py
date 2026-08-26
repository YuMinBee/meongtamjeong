"""Fill the official 2026 OSS contest result-report DOCX template.

Only ``word/document.xml`` is changed.  Every other package member is copied
from the official template and verified byte-for-byte after the build.  This
keeps the organizer's page geometry, styles, relationships, and section breaks
as the design authority while making the participant-controlled fields easy to
replace from the command line.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from lxml import etree


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEPENDENCY_REPORT = ROOT / "docs" / "dependency-report.json"
DEFAULT_OUTPUT = (
    ROOT
    / "submission"
    / "2026-osscontest"
    / "2026 오픈소스 개발자대회 결과보고서_접수번호(팀명).docx"
)
OFFICIAL_TEMPLATE_SHA256 = (
    "937679bac40cbfaced3457530c232c9d190a74f6b5d67c58b4bc33014a579195"
)
DOCUMENT_XML = "word/document.xml"
CONTEST_BRANCH_URL = (
    "https://github.com/YuMinBee/meongtamjeong/tree/submission/osscontest-2026"
)

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML_NS = "http://www.w3.org/XML/1998/namespace"
NS = {"w": W_NS}


def _w(name: str) -> str:
    return f"{{{W_NS}}}{name}"


@dataclass(frozen=True)
class Paragraph:
    """One black-text paragraph inserted into a retained template cell."""

    text: str
    body: str = ""
    bold: bool = False
    align: str = "both"
    size_half_points: int = 20
    space_after_twips: int = 60
    keep_next: bool = False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _direct_dependencies(path: Path) -> list[dict[str, object]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    dependencies = report.get("dependencies")
    if not isinstance(dependencies, list):
        raise ValueError("dependency report has no dependency list")
    direct = [item for item in dependencies if isinstance(item, dict) and item.get("direct")]
    direct.sort(key=lambda item: str(item.get("canonical_name") or item.get("name") or ""))
    if len(direct) != 25:
        raise ValueError(f"expected 25 declared direct dependencies, found {len(direct)}")
    return direct


PURPOSES = {
    "accelerate": "선택 연구 모델 로딩과 장치 배치",
    "clip": "이미지·텍스트 공통 임베딩 생성",
    "faiss-cpu": "고정 벡터 인덱스 근접 검색",
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


def _package_parts(path: Path) -> list[tuple[zipfile.ZipInfo, bytes]]:
    with zipfile.ZipFile(path) as archive:
        return [(info, archive.read(info.filename)) for info in archive.infolist()]


def _body_tables(root: etree._Element) -> list[etree._Element]:
    body = root.find("w:body", NS)
    if body is None:
        raise ValueError("template has no Word document body")
    return list(body.findall("w:tbl", NS))


def _rows(table: etree._Element) -> list[etree._Element]:
    return list(table.findall("w:tr", NS))


def _cells(row: etree._Element) -> list[etree._Element]:
    return list(row.findall("w:tc", NS))


def _run(text: str, *, bold: bool, size_half_points: int) -> etree._Element:
    run = etree.Element(_w("r"))
    props = etree.SubElement(run, _w("rPr"))
    fonts = etree.SubElement(props, _w("rFonts"))
    for key in ("ascii", "hAnsi", "eastAsia", "cs"):
        fonts.set(_w(key), "맑은 고딕")
    etree.SubElement(props, _w("color")).set(_w("val"), "000000")
    etree.SubElement(props, _w("sz")).set(_w("val"), str(size_half_points))
    etree.SubElement(props, _w("szCs")).set(_w("val"), str(size_half_points))
    if bold:
        etree.SubElement(props, _w("b"))
        etree.SubElement(props, _w("bCs"))
    text_node = etree.SubElement(run, _w("t"))
    if text[:1].isspace() or text[-1:].isspace():
        text_node.set(f"{{{XML_NS}}}space", "preserve")
    text_node.text = text
    return run


def _paragraph(spec: Paragraph) -> etree._Element:
    paragraph = etree.Element(_w("p"))
    props = etree.SubElement(paragraph, _w("pPr"))
    etree.SubElement(props, _w("wordWrap"))
    etree.SubElement(props, _w("jc")).set(_w("val"), spec.align)
    spacing = etree.SubElement(props, _w("spacing"))
    spacing.set(_w("before"), "0")
    spacing.set(_w("after"), str(spec.space_after_twips))
    spacing.set(_w("line"), "240")
    spacing.set(_w("lineRule"), "auto")
    if spec.keep_next:
        etree.SubElement(props, _w("keepNext"))
    if spec.body:
        paragraph.append(
            _run(spec.text, bold=True, size_half_points=spec.size_half_points)
        )
        paragraph.append(
            _run(
                f" {spec.body}",
                bold=spec.bold,
                size_half_points=spec.size_half_points,
            )
        )
    else:
        paragraph.append(
            _run(
                spec.text,
                bold=spec.bold,
                size_half_points=spec.size_half_points,
            )
        )
    return paragraph


def _set_cell(cell: etree._Element, content: Sequence[Paragraph]) -> None:
    properties = cell.find("w:tcPr", NS)
    for child in list(cell):
        if child is not properties:
            cell.remove(child)
    for item in content:
        cell.append(_paragraph(item))
    if not content:
        cell.append(_paragraph(Paragraph("")))


def _prevent_row_split(row: etree._Element) -> None:
    properties = row.find("w:trPr", NS)
    if properties is None:
        properties = etree.Element(_w("trPr"))
        row.insert(0, properties)
    if properties.find("w:cantSplit", NS) is None:
        properties.append(etree.Element(_w("cantSplit")))


def _keep_row_with_next(row: etree._Element) -> None:
    for paragraph in row.findall(".//w:p", NS):
        properties = paragraph.find("w:pPr", NS)
        if properties is None:
            properties = etree.Element(_w("pPr"))
            paragraph.insert(0, properties)
        if properties.find("w:keepNext", NS) is None:
            properties.append(etree.Element(_w("keepNext")))


def _one(
    text: str,
    *,
    bold: bool = False,
    align: str = "center",
    size_half_points: int = 20,
) -> list[Paragraph]:
    return [
        Paragraph(
            text=text,
            bold=bold,
            align=align,
            size_half_points=size_half_points,
            space_after_twips=0,
        )
    ]


def _report_content() -> dict[int, list[Paragraph]]:
    return {
        4: [
            Paragraph(
                "자연어 또는 참고 이미지로 현재 보호 중인 구조견 공고 후보를 찾고, "
                "공고 원문 근거와 보호소 확인 질문을 함께 제시하는 설명 가능한 오픈소스 탐색 시스템입니다.",
                align="center",
                space_after_twips=0,
            )
        ],
        6: [
            Paragraph("문제 인식", "국가동물보호정보시스템의 공고는 사실 정보가 유용하지만, 사용자가 떠올린 외형을 정확한 품종·키워드로 바꾸기 어렵고 여러 공고를 반복 확인해야 합니다.", keep_next=True),
            Paragraph("목표", "참고 사진이나 일상적인 한국어 표현에서 출발해 현재 확인 가능한 후보를 좁히고, 원문 확인과 보호소 문의까지 이어지는 탐색 비용을 낮추고자 했습니다."),
            Paragraph("안전 경계", "멍탐정은 검색·추천 후보를 제시할 뿐 입양 결정을 대신하지 않습니다. 사진·품종으로 성격, 공격성, 건강, 아동·다른 동물 친화성이나 생활 적합성을 추정하지 않으며, 확인되지 않은 값은 unknown으로 유지합니다."),
            Paragraph("검증 중심 설계", "선택한 후보에 대해서는 최신 공고 상태와 원문을 다시 확인하고, 생활·성격 선호는 순위 점수가 아니라 보호소에 실제 관찰 여부를 묻는 질문으로 전환합니다. 자동 문의·예약·입양 신청은 수행하지 않습니다.", space_after_twips=0),
        ],
        7: [
            Paragraph("실행 환경", "Python 3.10, FastAPI, Uvicorn, PyTorch, OpenAI CLIP ViT-B/32, FAISS CPU, NumPy, Requests. Windows와 Linux에서 실행 가능하며 기본 서비스는 CPU로도 구동됩니다."),
            Paragraph("검색 계층", "CLIP은 이미지·텍스트 공통 벡터, FAISS는 근접 후보 검색, BM25는 공고문 어휘 검색, Pydantic은 API 계약 검증, 공고 사실 기반 규칙과 경량 Graph는 조건 결합과 근거 생성을 담당합니다."),
            Paragraph("개발·품질", "Git/GitHub, pytest 회귀 테스트, Ruff 정적 검사, GitHub Actions, 고정 입력·코드·인덱스 SHA-256, SPDX 2.3 및 CycloneDX 1.6 SBOM을 사용합니다."),
            Paragraph("배포 프로필", "공개 출품에는 public-text-only-v1을 권장합니다. 공고 텍스트 벡터만 포함하고 사진 파일·원격 사진 URL·image/crop 벡터·DINO 연구 산출물을 제외해 데이터 권리 노출을 줄입니다."),
            Paragraph("자원 조건", "GPU는 오프라인 이미지 인덱싱·선택 연구에만 선택적으로 사용합니다. 대회용 텍스트 프로필 생성과 서버 추론은 CPU 모드가 가능하며, 대규모 학습과 동시에 실행하지 않도록 런북에 분리했습니다.", space_after_twips=0),
        ],
        8: [
            Paragraph("입력·정규화", "자연어/참고 이미지 → 색상·크기·연령·지역 등 공고에서 확인 가능한 조건 정규화 → 외형으로 확인할 수 없는 생활·성격 문구 분리"),
            Paragraph("후보 생성", "정규화 질의/참고 이미지 → 동결 CLIP 임베딩 → FAISS 후보. 동시에 질의 → BM25·공고 사실·경량 Graph 후보"),
            Paragraph("결합·검증", "후보 융합 → active-only 상태 필터 → 규칙 기반 재정렬 → 점수 구성요소·공고 근거·원문 URL 제공"),
            Paragraph("후속 행동", "후보 선택 → 서버측 공고 ID 대조 → 가능한 경우 현재 상태 재조회 → 보호소 확인 질문·전화/이메일 초안 생성 → 사용자가 검토 후 수동 문의"),
            Paragraph("데이터 흐름 원칙", "비밀 API 키와 개인정보는 저장소·결과에 포함하지 않습니다. 브라우저 입력 연락처를 신뢰하지 않고 서버가 가진 공고 사실을 사용하며, 최신 조회 실패 시 status_verified=false로 명시합니다."),
            Paragraph("선택 연구 격리", "DINOv3·DINOde flow·VLM 경로는 실험 모듈로 분리되어 기본 검색의 필수 의존성이 아닙니다. 통과 기준을 충족하지 못한 행동/적합성 head는 기본 비활성화했습니다.", space_after_twips=0),
        ],
        9: [
            Paragraph("1. 활성 공고 동기화와 재현성", "국가동물보호정보시스템 구조동물 조회 API를 provider 계약으로 수집합니다. 종료·반환·입양 완료 등 비활성 공고와 권장 경로의 상태 미상 공고를 제외합니다. 임시 경로에서 수집·검증한 뒤 사람이 canonical artifact를 승격하도록 해 부분 덮어쓰기를 막습니다."),
            Paragraph("2. 멀티모달·하이브리드 검색", "텍스트와 이미지를 CLIP 공간에서 비교하고, BM25와 공고 필드 조건을 결합합니다. 최종 응답에는 단일 총점만이 아니라 벡터·어휘·조건·Graph 근거를 분리해 표시합니다."),
            Paragraph("3. 비추론 안전 설계", "appearance 경로에서 주거, 부재 시간, 활동량, 반려견 경험, 아동·다른 동물 여부와 선호 성격은 순위에 영향을 주지 않습니다. 고정 후보 11개에 대한 안전 계약 4개가 모두 통과했으며, 생활 선호는 문의 질문만 변경했습니다."),
            Paragraph("4. 문의 도우미", "선택한 공고의 원문과 상태를 재확인하고, 실제 관찰 여부를 물을 질문과 전화·이메일/문의폼 초안을 만듭니다. 자동 발송이나 공식 신청은 하지 않아 사용자가 사실을 확인하고 수정할 수 있습니다."),
            Paragraph("5. 공개·검증 가능성", "Apache-2.0 소스, 환경/lock 파일, 데이터·모델 카드, 제3자 고지, 자동 테스트, 평가 입력과 실패 사례, SPDX/CycloneDX SBOM을 함께 공개합니다. public-text-only 패키지는 사진 관련 항목을 allowlist 방식으로 제거하고 marker·index·metadata 해시를 함께 검증합니다."),
            Paragraph("구동", "저장소에서 Python 3.10 환경을 만든 뒤 requirements.lock.txt를 설치하고, 환경변수에 별도 API 키를 설정해 uvicorn app.main:app --host 127.0.0.1 --port 8000으로 실행합니다. API 문서와 app/profile_demo.html에서 텍스트/참고 이미지 검색, 후보 선택, 문의 카드 흐름을 확인합니다."),
            Paragraph("고정 검색 평가", "2026-07-26 active-only 스냅샷 1,516건(벡터 3,746행)에 대해 시스템을 보지 않은 별도 작성자가 동결한 one-shot 의미보존 24문장의 P@5 76.67%, Recall@5 24.21%, Recall@10 36.00%, nDCG@5 79.15%, MRR 82.99%, Hit@5 100%, 비활성 노출 0%였습니다."),
            Paragraph("실패 공개", "24문장 중 순위 또는 멤버십 변화가 21건, silver 지표 하락이 8건이었습니다. 부정·미지원 진단 6건은 aggregate에서 제외해 별도 6/6 PASS로 기록했습니다. 이는 공고 필드에서 만든 silver qrels 결과이며 사람의 주관적 닮음, 한국어 전반, 입양 적합성이나 성과를 의미하지 않습니다."),
            Paragraph("교차사진 감사", "두 번째 사진 120건 중 네트워크·중복 검사를 통과한 29건에서 Hit@1 82.76%, Hit@5 96.55%, MRR 89.12%였습니다. 전체 시도 기준 end-to-end Hit@1 20.00%, Hit@5 23.33%이며 감사 상태는 partial입니다. 같은 공고를 다시 찾는 제한된 대리 과제이지 서로 다른 개의 닮음 평가는 아닙니다.", space_after_twips=0),
        ],
        10: [
            Paragraph("사용자 가치", "품종명이나 정확한 검색어를 몰라도 참고 이미지·자연어에서 실제 공고 후보로 이동하고, 확인 가능한 사실과 다음 질문을 한 흐름에서 정리할 수 있습니다."),
            Paragraph("기관·개발자 활용", "보호소·동물보호단체·공공데이터 활용 개발자는 NoticeProvider 인터페이스와 provider-neutral JSON bridge로 자체 공고 소스를 연결하고, 상태 필터·근거 표시·문의 흐름을 재사용할 수 있습니다."),
            Paragraph("확장 방향", "공고 제공처 어댑터 확대, 인적 blind 관련성 평가, 긴 꼬리 표현·지역/연령 복합 질의 개선, 상태 갱신 모니터링, 접근성·다국어 UI를 단계적으로 추가할 수 있습니다."),
            Paragraph("기대 범위", "효과는 후보 탐색과 근거 확인을 돕는 범위로 한정합니다. 보호소 업무 절감, 입양 성공률, 행동·건강·안전 결과는 별도 현장 연구 없이 주장하지 않습니다.", space_after_twips=0),
        ],
        11: [
            Paragraph("차별성", "일반적인 이미지 유사도 데모에 그치지 않고 실제 active 공고, 어휘·벡터·구조화 사실 결합, 원문 근거, 최신 상태 확인, 보호소 질문까지 하나의 검증 가능한 오픈소스 흐름으로 연결했습니다."),
            Paragraph("안전의 구현", "‘성격을 추정하지 않는다’는 문구를 UI에만 두지 않고 질의 정규화, 재정렬 입력, 응답 근거, 문의 전환, 합성 회귀 테스트로 계약화했습니다. 모르는 값은 채우지 않고 다음 확인 행동을 제안합니다."),
            Paragraph("정직한 연구 선택", "선택 DINO 외부 시험에서 이미지 Recall@10은 0.877로 CLIP 0.557보다 높았지만, personality fusion nDCG@5는 0.516으로 DINO 0.533보다 낮았고 compatibility head macro AUC는 0.548에 그쳤습니다. 사전 gate를 통과하지 못한 모델은 기본 경로에 승격하지 않았습니다."),
            Paragraph("현재 한계", "고정 평가는 작은 한국어 질의 집합과 공고 필드 silver label에 의존합니다. 공고 사진의 촬영 조건, 누락 필드, 외부 서버 장애, 상태 변경이 결과에 영향을 주며, 텍스트-only 공개 프로필은 image/crop 벡터를 제외하므로 full 프로필의 시각 검색 수치를 그대로 적용할 수 없습니다."),
            Paragraph("로드맵 1", "권리 확인이 끝난 데이터만 별도 프로필로 유지하고, text-only 릴리스마다 새 index·metadata 해시와 해당 프로필 평가를 동결합니다. 만료·미발견 비율과 원문 링크 상태를 정기 점검합니다."),
            Paragraph("로드맵 2", "외부 참여자를 대상으로 사전 등록한 blind 평가와 국가동물보호정보시스템 대비 후보 확정 시간/완료율 파일럿을 수행하되, 보호소 문의·방문·입양 성과와 분리해 해석합니다."),
            Paragraph("개발 후기", "성능 수치보다 데이터 권리, 실패 분모, 최신성, 추론 금지 경계를 코드와 문서에 함께 남기는 일이 오픈소스 AI의 신뢰에 중요하다는 점을 확인했습니다. 재현 가능한 실패와 선택하지 않은 모델까지 공개하는 방향으로 프로젝트를 정리했습니다.", space_after_twips=0),
        ],
    }


def _source_url(item: dict[str, object]) -> str:
    value = str(item.get("origin_url") or item.get("source_url") or "NOASSERTION")
    if value.startswith("git+"):
        value = value[4:]
    if value.endswith(".git"):
        value = value[:-4]
    return value


def _license(item: dict[str, object]) -> str:
    name = str(item.get("canonical_name") or item.get("name") or "")
    concise = {
        "numpy": "BSD (metadata)",
        "pandas": "BSD (metadata)",
        "regex": "Apache-2.0 / CNRI-Python",
        "torch": "Apache/BSD/MIT/BSL-1.0 (상세 SBOM)",
        "torchvision": "BSD (metadata)",
        "tqdm": "MPL-2.0 / MIT",
    }
    if name in concise:
        return concise[name]
    value = str(item.get("license") or "NOASSERTION")
    aliases = {
        "Apache": "Apache-2.0",
        "Apache 2.0": "Apache-2.0",
        "Apache 2.0 License": "Apache-2.0",
        "BSD": "BSD (metadata)",
    }
    return aliases.get(value, value)


def _fill_sbom(table: etree._Element, dependencies: Sequence[dict[str, object]]) -> None:
    rows = _rows(table)
    if len(rows) < 3 or len(_cells(rows[0])) != 6:
        raise ValueError("unexpected SBOM table layout")
    prototype = copy.deepcopy(rows[2])
    for row in rows[1:]:
        table.remove(row)
    for number, item in enumerate(dependencies, start=1):
        row = copy.deepcopy(prototype)
        _prevent_row_split(row)
        cells = _cells(row)
        name = str(item.get("canonical_name") or item.get("name") or "")
        values = (
            str(number),
            name,
            str(item.get("installed_version") or "UNKNOWN"),
            _license(item),
            _source_url(item),
            PURPOSES.get(name, "직접 선언된 프로젝트 의존성"),
        )
        for index, (cell, value) in enumerate(zip(cells, values)):
            align = "center" if index < 4 else "left"
            _set_cell(
                cell,
                _one(value, align=align, size_half_points=20),
            )
        table.append(row)


def _fill_ai_spec(table: etree._Element) -> None:
    rows = _rows(table)
    if len(rows) != 12:
        raise ValueError("unexpected AI specification table layout")
    for row in rows:
        _prevent_row_split(row)
    _keep_row_with_next(rows[9])

    _set_cell(
        _cells(rows[1])[0],
        [
            Paragraph("▣ 유형 1: 외부 모델 그대로 활용 — OpenAI CLIP ViT-B/32를 추가 학습 없이 동결 추론", align="left"),
            Paragraph("□ 유형 2: 외부 모델 파인튜닝", align="left"),
            Paragraph("□ 유형 3: 자체 개발 모델", align="left"),
            Paragraph("출품 실행 범위", "DINOv3·DINOde·VLM 연구 경로는 선택 실험이며 기본 비활성화했습니다. 해당 가중치와 연구 산출물은 public-text-only-v1 출품 artifact에 포함하지 않습니다.", align="left", space_after_twips=0),
        ],
    )
    base_cells = _cells(rows[3])
    _set_cell(base_cells[1], _one("OpenAI CLIP ViT-B/32 (OpenAI)", align="left"))
    _set_cell(base_cells[3], _one("MIT License", align="left"))

    not_applicable = {
        5: "해당 없음 — 유형 1 출품 프로필이며 추가 학습을 수행하지 않음",
        6: "해당 없음 — 유형 1 출품 프로필이며 학습 데이터 정제·가공 없음",
        7: "해당 없음 — 새 가중치를 생성하거나 배포하지 않음",
        8: "해당 없음 — upstream CLIP 공개 가중치를 동결 추론에 사용",
    }
    for row_index, value in not_applicable.items():
        _set_cell(_cells(rows[row_index])[1], _one(value, align="left"))

    source_cells = _cells(rows[10])
    _set_cell(source_cells[1], _one("Apache License 2.0", align="left"))
    _set_cell(
        source_cells[3],
        _one(CONTEST_BRANCH_URL, align="left"),
    )
    _set_cell(
        _cells(rows[11])[1],
        [
            Paragraph(
                "ChatGPT/Codex를 코드 검토·문서화·디버깅 보조에 사용했습니다. 생성 제안은 작성자가 검토·수정했습니다. 적용 비율은 자동 산정하지 않아 정량 비율을 기재하지 않습니다.",
                align="left",
                space_after_twips=0,
            )
        ],
    )


def build_report(
    *,
    template: Path,
    output: Path,
    dependency_report: Path,
    team_name: str,
    team_size: str,
    division: str,
    demo_url: str,
) -> None:
    template = template.resolve()
    if _sha256(template) != OFFICIAL_TEMPLATE_SHA256:
        raise ValueError("official template SHA-256 does not match the reviewed file")

    package = _package_parts(template)
    package_map = {name.filename: data for name, data in package}
    if DOCUMENT_XML not in package_map:
        raise ValueError("template package has no word/document.xml")

    parser = etree.XMLParser(remove_blank_text=False, resolve_entities=False)
    root = etree.fromstring(package_map[DOCUMENT_XML], parser=parser)
    tables = _body_tables(root)
    if len(tables) != 9:
        raise ValueError(f"expected 9 body tables, found {len(tables)}")
    guide, _title, metadata, report, _sbom_title, sbom, _ai_title, _ai_guide, ai = tables

    body = root.find("w:body", NS)
    assert body is not None
    body.remove(guide)
    body.remove(_ai_guide)
    for child in list(body):
        if child.tag != _w("p"):
            continue
        child_text = "".join(child.itertext()).strip()
        if child_text.startswith("※ 필요 시, 행을 추가하여"):
            body.remove(child)

    metadata_rows = _rows(metadata)
    _set_cell(_cells(metadata_rows[1])[1], _one(team_name))
    _set_cell(_cells(metadata_rows[1])[3], _one(team_size))
    _set_cell(_cells(metadata_rows[2])[1], _one(division))
    _set_cell(_cells(metadata_rows[2])[3], _one("자유과제"))

    report_rows = _rows(report)
    _set_cell(_cells(report_rows[1])[1], _one("멍탐정", bold=True))
    _set_cell(
        _cells(report_rows[2])[1],
        _one(CONTEST_BRANCH_URL, align="left"),
    )
    _set_cell(_cells(report_rows[3])[1], _one(demo_url, align="left"))
    for row_index, content in _report_content().items():
        _set_cell(_cells(report_rows[row_index])[1], content)

    dependencies = _direct_dependencies(dependency_report)
    _fill_sbom(sbom, dependencies)
    _fill_ai_spec(ai)

    final_tables = _body_tables(root)
    if len(final_tables) != 7:
        raise ValueError(f"guide removal produced {len(final_tables)} body tables")
    if len(root.findall(".//w:sectPr", NS)) != 2:
        raise ValueError("section geometry changed unexpectedly")

    xml_bytes = etree.tostring(
        root,
        xml_declaration=True,
        encoding="UTF-8",
        standalone=True,
    )
    if "결과보고서 작성 안내".encode("utf-8") in xml_bytes:
        raise ValueError("guide page text remains in output")

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w") as archive:
        for info, data in package:
            archive.writestr(info, xml_bytes if info.filename == DOCUMENT_XML else data)

    output_parts = {
        info.filename: data for info, data in _package_parts(output)
    }
    for name, expected in package_map.items():
        if name == DOCUMENT_XML:
            continue
        if output_parts.get(name) != expected:
            raise ValueError(f"preserve-only package part changed: {name}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--dependency-report",
        type=Path,
        default=DEFAULT_DEPENDENCY_REPORT,
    )
    parser.add_argument(
        "--team-name",
        default="[최종 입력 필요: 참가 접수 팀명]",
    )
    parser.add_argument(
        "--team-size",
        default="[최종 입력 필요: 팀장 포함 N명]",
    )
    parser.add_argument(
        "--division",
        default="[최종 입력 필요: 학생/일반]",
    )
    parser.add_argument(
        "--demo-url",
        default="[최종 입력 필요: 3분 이내 YouTube URL]",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    build_report(
        template=args.template,
        output=args.output,
        dependency_report=args.dependency_report,
        team_name=args.team_name,
        team_size=args.team_size,
        division=args.division,
        demo_url=args.demo_url,
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
