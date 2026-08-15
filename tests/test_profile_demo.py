from __future__ import annotations

import ast
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEMO_PATH = ROOT / "app" / "profile_demo.html"
MAIN_PATH = ROOT / "app" / "main.py"


class _DemoStructureParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[tuple[str, dict[str, str | None]]] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        self.tags.append((tag, dict(attrs)))


def test_profile_demo_uses_new_profile_search_contract() -> None:
    page = DEMO_PATH.read_text(encoding="utf-8")

    assert '<html lang="ko">' in page
    assert "fetch('/search/profile'" in page
    assert "fetch('/search/appearance/image'" in page
    assert "ranking_scope:'appearance'" in page
    assert "/rag/recommend_form" not in page
    assert 'id="profileSearchForm"' in page
    assert 'id="resultsList" aria-busy="false"' in page
    assert (
        'id="resultsDescription" role="status" aria-live="polite" '
        'aria-atomic="true"' in page
    )
    assert 'id="refImage" name="ref_image" type="file"' in page
    assert 'accept="image/jpeg,image/png,image/webp"' in page

    for field in (
        "housing_type",
        "daily_absence_hours",
        "activity_level",
        "dog_experience",
        "preferred_size",
        "preferred_age",
        "preferred_region",
        "has_children",
        "has_other_pets",
    ):
        assert field in page


def test_profile_demo_surfaces_evidence_and_safety_context() -> None:
    page = DEMO_PATH.read_text(encoding="utf-8")

    for phrase in (
        "조건 일치",
        "주의해서 확인",
        "보호소에 확인",
        "입양 적합 확률이 아닙니다",
        "실제 공고 보기",
        "마지막 상태 확인",
        "마지막 확인 시 보호중",
        "보호소 문의",
        "전화 문구 복사",
        "이메일 양식 복사",
        "이메일 작성",
        "전화나 이메일을 자동으로 보내지 않습니다",
        "입양 신청이나 확정을 뜻하지 않습니다",
        "보호소 전화·이메일 정보가 없습니다",
        "종료된 공고이므로 전화·이메일 바로 연결을 비활성화했습니다",
        "신뢰할 연락처를 확인하지 못해 바로 연결을 제공하지 않습니다",
        "공고 상태가 확인되지 않아 전화·이메일 바로 연결을 제공하지 않습니다",
        "성격·생활 표현은 검색 순위에서 제외했어요",
        "입력 오타를 보수적으로 보정했어요",
        "참고사진 추가",
        "눌러서 고르거나 이곳에 사진을 놓아주세요",
        "JPG·PNG·WebP, 최대 8MB·2,500만 화소",
        "검색 버튼을 누르기 전에는 전송되지 않습니다",
        "사진은 검색 요청 처리 후 영구 보관하지 않습니다",
        "사진 제거",
    ):
        assert phrase in page

    assert "보호 중 공고" not in page

    assert "fetch('/adoption/contact_card'" in page
    assert "const preferences = inquiryPreferencesPayload()" in page
    assert '<option value="" selected>선택 안 함</option>' in page
    assert "if (housingType) preferences.housing_type = housingType" in page
    assert "if (absenceValue !== '') preferences.daily_absence_hours" in page
    assert 'id="activityLevel" name="activity_level"' in page
    assert (
        "if (activityLevel.value) preferences.activity_level = activityLevel.value"
        in page
    )
    search_profile_source = page.split("function profilePayload()", 1)[1].split(
        "function inquiryPreferencesPayload()", 1
    )[0]
    for field in (
        "housing_type",
        "daily_absence_hours",
        "activity_level",
        "dog_experience",
        "has_children",
        "has_other_pets",
    ):
        assert field not in search_profile_source
    assert "if (region) profile.preferred_region = region" in search_profile_source
    assert "payload?.latest_found" in page
    assert "new AbortController()" in page
    assert "requestSequence !== inquiryRequestSequence" in page
    assert "15000" in page
    assert "최신 확인 시간이 초과됐지만" in page
    assert "저장된 공고로 문의 문구를 먼저 준비했습니다" in page
    assert "refresh_latest:refreshLatest" in page
    assert "requestInquiryCard(desertionNo, preferences, false" in page
    assert "requestInquiryCard(desertionNo, preferences, true" in page
    assert "data-retry-inquiry" in page
    assert "safeUrl" in page
    assert "safeTel" in page
    assert "safeTel(actions.tel)" in page
    assert "actions.tel || contact.care_tel" not in page
    assert "safeMailto" in page
    assert "actions.mailto" in page
    assert "payload?.direct_contact_allowed === true" in page
    assert "payload?.direct_contact_block_reason" in page
    assert (
        "body:JSON.stringify({desertion_no:desertionNo, preferences, "
        "refresh_latest:refreshLatest})" in page
    )
    assert "body:JSON.stringify({animal, preferences:" not in page
    assert "data.nonappearance_query_terms_excluded === true" in page
    assert "data.query_policy?.typo_corrections" in page
    assert "data.query_policy?.warnings" in page
    assert "typeof payload.detail === 'object' && payload.detail.message" in page
    assert "contact.care_email" in page
    assert page.count("inquiryPreferencesPayload()") == 2
    assert "escapeHtml" in page
    assert "sessionStorage" in page
    assert "localStorage" not in page
    assert "new FormData()" in page
    assert (
        "formData.append('ref_image', selectedReferenceFile, 'reference-image')" in page
    )
    assert "referenceDropZone.addEventListener('drop'" in page
    assert "URL.createObjectURL(file)" in page
    assert "URL.revokeObjectURL(referencePreviewUrl)" in page
    assert "selectedReferenceFile" in page
    assert "ref_image_url" not in page
    assert "image_url_input" not in page
    image_selection_code = page.split("function setReferenceImage", 1)[1].split(
        "function profilePayload", 1
    )[0]
    assert "fetch(" not in image_selection_code
    assert "<!--__" not in page
    assert "공고 표기 품종" in page
    assert "meta.breed_source === 'public_notice_reported'" in page


def test_profile_demo_has_accessible_async_and_mobile_guards() -> None:
    page = DEMO_PATH.read_text(encoding="utf-8")

    assert 'aria-controls="connectionDialog" aria-label="API 연결 설정"' in page
    assert 'data-preset="fluffySmall" aria-pressed="true"' in page
    assert 'aria-expanded="false" aria-controls="${panelId}"' in page
    assert 'aria-haspopup="dialog" aria-controls="inquiryDialog"' in page
    assert "button.setAttribute('aria-pressed', String(selected))" in page
    assert "button.lastChild.textContent = willOpen ? ' 근거·점수 접기'" in page

    assert "#inquiryDialog[open] { display:flex; flex-direction:column; }" in page
    assert "max-height:min(840px, calc(100dvh - 32px))" in page
    assert "#inquiryDialog .dialog-body { min-height:0; overflow:auto; }" in page
    assert "--muted:#596a63" in page
    assert "--focus:#a54520" in page
    assert ":focus-visible { outline:3px solid var(--focus)" in page
    assert ".switch input:focus-visible + .switch-track" in page

    assert "if (searchAbortController) searchAbortController.abort()" in page
    assert "const requestSequence = ++searchRequestSequence" in page
    assert "if (requestSequence !== searchRequestSequence) return" in page
    assert "searchAbortController === controller" in page
    assert "if (inquiryAbortController) inquiryAbortController.abort()" in page
    assert "inquiryContent.setAttribute('aria-busy', 'true')" in page
    assert "inquiryContent.setAttribute('aria-busy', 'false')" in page

    assert "imagePreviewPicture.addEventListener('load'" in page
    assert "imagePreviewPicture.addEventListener('error'" in page
    assert "25_000_000" in page
    assert "resultsList.addEventListener('error'" in page
    assert "image-load-failed" in page

    assert "function legacyCopyText(value)" in page
    assert "if (!copied) copied = legacyCopyText(value)" in page
    assert "helper.setSelectionRange(0, helper.value.length)" in page
    assert "if (button.isConnected) button.focus" in page

    assert "function readSessionApiKey()" in page
    assert "function persistSessionApiKey(value)" in page
    assert "let apiKey = readSessionApiKey();" in page
    assert "fetch('/health', {headers:{'x-api-key':normalized}" in page
    assert "payload?.status !== 'ok'" in page
    assert "connectionVerifyAbortController" in page
    assert "verifySequence !== connectionVerifySequence" in page
    assert "function invalidateConnection(requestKey" in page
    assert "connectionVerified ? '연결됨'" in page
    assert "readSessionApiKey() || 'change-me'" not in page
    assert 'id="inquiryUpdateStatus" role="status" aria-live="polite"' in page
    assert 'id="inquiryContent" aria-live=' not in page
    assert "inquiryContent.setAttribute('aria-busy', refreshState === 'loading'" in page
    assert "const inquiryTimeout = window.setTimeout" in page
    first_inquiry_request = page.index(
        "requestInquiryCard(desertionNo, preferences, false"
    )
    assert page.index("const inquiryTimeout = window.setTimeout") < first_inquiry_request
    assert "apiKeyInput.addEventListener('keydown'" in page
    assert "if (event.key !== 'Enter') return" in page
    assert "localStorage" not in page


def test_profile_demo_static_dom_references_are_well_formed() -> None:
    parser = _DemoStructureParser()
    parser.feed(DEMO_PATH.read_text(encoding="utf-8"))
    ids = [attrs["id"] for _, attrs in parser.tags if attrs.get("id")]

    assert not [value for value, count in Counter(ids).items() if count > 1]

    missing_references: list[tuple[str, str, str]] = []
    for tag, attrs in parser.tags:
        for attribute in (
            "for",
            "aria-controls",
            "aria-describedby",
            "aria-labelledby",
        ):
            for reference in (attrs.get(attribute) or "").split():
                if "{" not in reference and reference not in ids:
                    missing_references.append((tag, attribute, reference))
        if tag == "button":
            assert attrs.get("type") in {"button", "submit", "reset"}
        if tag == "img":
            assert "alt" in attrs

    assert not missing_references


def test_main_exposes_profile_demo_routes_without_changing_profile_post() -> None:
    main_source = MAIN_PATH.read_text(encoding="utf-8")
    tree = ast.parse(main_source)
    routes: dict[str, set[str]] = {}
    route_sources: dict[str, str] = {}

    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        paths: set[str] = set()
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call) or not decorator.args:
                continue
            function = decorator.func
            if not isinstance(function, ast.Attribute) or function.attr not in {
                "get",
                "post",
            }:
                continue
            path = decorator.args[0]
            if isinstance(path, ast.Constant) and isinstance(path.value, str):
                paths.add(f"{function.attr.upper()} {path.value}")
        if paths:
            routes[node.name] = paths
            route_sources[node.name] = ast.get_source_segment(main_source, node) or ""

    assert routes["profile_search_demo"] == {
        "GET /demo",
        "GET /visualize/profile-search",
    }
    assert "POST /search/profile" in routes["search_profile"]
    assert routes["search_appearance_image"] == {"POST /search/appearance/image"}
    appearance_source = route_sources["search_appearance_image"]
    for required in (
        "load_uploaded_image",
        "APPEARANCE_IMAGE_ALLOWED_FORMATS",
        "analyze_appearance_query",
        "appearance_search_conditions",
        "combine_embeddings",
        "hybrid_search",
        "rerank_candidates",
        "APPEARANCE_CONDITION_KEYS",
    ):
        assert required in appearance_source
    for forbidden in (
        "gemma",
        "requests.",
        ".filename",
        "open(",
        "write(",
        "image_url",
    ):
        assert forbidden not in appearance_source
