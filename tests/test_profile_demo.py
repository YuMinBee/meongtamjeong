from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEMO_PATH = ROOT / "app" / "profile_demo.html"
MAIN_PATH = ROOT / "app" / "main.py"


def test_profile_demo_uses_new_profile_search_contract() -> None:
    page = DEMO_PATH.read_text(encoding="utf-8")

    assert '<html lang="ko">' in page
    assert "fetch('/search/profile'" in page
    assert "/rag/recommend_form" not in page
    assert 'id="profileSearchForm"' in page
    assert 'id="resultsList" aria-live="polite"' in page

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
        "이전 프로필과 비교한 순위 변화",
    ):
        assert phrase in page

    assert "safeUrl" in page
    assert "escapeHtml" in page
    assert "sessionStorage" in page
    assert "localStorage" not in page
    assert "<!--__" not in page


def test_main_exposes_profile_demo_routes_without_changing_profile_post() -> None:
    tree = ast.parse(MAIN_PATH.read_text(encoding="utf-8"))
    routes: dict[str, set[str]] = {}

    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        paths: set[str] = set()
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call) or not decorator.args:
                continue
            function = decorator.func
            if not isinstance(function, ast.Attribute) or function.attr not in {"get", "post"}:
                continue
            path = decorator.args[0]
            if isinstance(path, ast.Constant) and isinstance(path.value, str):
                paths.add(f"{function.attr.upper()} {path.value}")
        if paths:
            routes[node.name] = paths

    assert routes["profile_search_demo"] == {
        "GET /demo",
        "GET /visualize/profile-search",
    }
    assert "POST /search/profile" in routes["search_profile"]
