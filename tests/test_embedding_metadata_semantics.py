from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTED_BREED_KEYS = {
    "breed",
    "breed_name",
    "breed_code",
    "breedCd",
    "kindCd",
    "kindNm",
    "kindFullNm",
}


def _function_node(relative_path: str, function_name: str) -> ast.FunctionDef:
    tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8-sig"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return node
    raise AssertionError(f"function not found: {relative_path}:{function_name}")


def _string_literals(node: ast.AST) -> set[str]:
    return {
        child.value
        for child in ast.walk(node)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    }


def test_full_rebuild_does_not_recycle_old_breed_bearing_clip_text() -> None:
    resolver = _function_node("scripts/build_embeddings.py", "resolve_embedding_desc")

    assert "desc_full" not in _string_literals(resolver)


def test_embedding_builders_exclude_structured_breed_but_keep_color() -> None:
    for path in (
        "scripts/build_embeddings.py",
        "scripts/update_embeddings.py",
    ):
        builder = _function_node(path, "build_dog_text")
        literals = _string_literals(builder)
        assert REPORTED_BREED_KEYS.isdisjoint(literals)
        assert {"color", "colorCd"} <= literals


def test_legacy_remote_image_embedders_delegate_to_shared_downloader() -> None:
    for path, function_name in (
        ("scripts/build_embeddings.py", "get_clip_embedding_from_url"),
        ("scripts/update_embeddings.py", "embed_image_from_url"),
    ):
        source = (ROOT / path).read_text(encoding="utf-8-sig")
        function = _function_node(path, function_name)
        direct_get_calls = [
            child
            for child in ast.walk(function)
            if isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "get"
        ]

        assert "from app.public_image_download import" in source
        assert direct_get_calls == []
