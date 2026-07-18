from __future__ import annotations

import importlib
import json
import shutil
import sys
from contextlib import nullcontext
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from uuid import uuid4

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image


API_HEADERS = {"x-api-key": "test-api-key"}


class _FakeClipModel:
    def eval(self) -> "_FakeClipModel":
        return self


@pytest.fixture(scope="module")
def main_module(request: pytest.FixtureRequest):
    """Import the real app without requiring locally installed CLIP or FAISS."""

    monkeypatch = pytest.MonkeyPatch()
    runtime_root = Path(__file__).resolve().parent / ".runtime"
    workdir = runtime_root / f"api-{uuid4().hex}"
    workdir.mkdir(parents=True)

    def cleanup_runtime() -> None:
        shutil.rmtree(workdir)
        try:
            runtime_root.rmdir()
        except OSError:
            pass

    request.addfinalizer(cleanup_runtime)
    index_path = workdir / "fake.index"
    metas_path = workdir / "metas.json"
    index_path.write_bytes(b"stub")
    metas_path.write_text(
        json.dumps(
            [
                {
                    "type": "dog",
                    "desertionNo": "dog-1",
                    "breed": "mixed",
                    "noticeEdt": "20991231",
                }
            ]
        ),
        encoding="utf-8",
    )

    clip_stub = ModuleType("clip")

    def load_clip(_model_name: str, *, device: str) -> tuple[_FakeClipModel, Any]:
        del device
        return _FakeClipModel(), lambda image: image

    def tokenize(_texts: list[str], *, truncate: bool = False) -> SimpleNamespace:
        del truncate
        return SimpleNamespace(to=lambda _device: np.zeros((1, 2), dtype=np.float32))

    setattr(clip_stub, "load", load_clip)
    setattr(clip_stub, "tokenize", tokenize)

    faiss_stub = ModuleType("faiss")

    def read_index(_path: str) -> SimpleNamespace:
        return SimpleNamespace(ntotal=1)

    setattr(faiss_stub, "read_index", read_index)

    torch_stub = ModuleType("torch")

    def no_grad():
        return lambda function: function

    setattr(torch_stub, "cuda", SimpleNamespace(is_available=lambda: False))
    setattr(torch_stub, "no_grad", no_grad)
    setattr(torch_stub, "inference_mode", nullcontext)
    setattr(torch_stub, "float32", np.float32)
    setattr(torch_stub, "bfloat16", np.float32)

    monkeypatch.setitem(sys.modules, "clip", clip_stub)
    monkeypatch.setitem(sys.modules, "faiss", faiss_stub)
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setenv("API_KEY", API_HEADERS["x-api-key"])
    monkeypatch.setenv("INDEX_PATH", str(index_path))
    monkeypatch.setenv("METAS_PATH", str(metas_path))
    monkeypatch.setenv("PROFILE_COMPATIBILITY_WEIGHT", "0.25")
    monkeypatch.setenv("PROFILE_QUALITY_WEIGHT", "0.05")
    monkeypatch.setenv("PROFILE_CANDIDATE_MULTIPLIER", "5")
    monkeypatch.setenv("PROFILE_INCLUDE_UNKNOWN_NOTICES", "true")

    sys.modules.pop("app.main", None)
    try:
        module = importlib.import_module("app.main")
        yield module
    finally:
        sys.modules.pop("app.main", None)
        monkeypatch.undo()


@pytest.fixture
def client(main_module: Any):
    with TestClient(main_module.app) as test_client:
        yield test_client


@pytest.fixture
def isolated_runtime(
    monkeypatch: pytest.MonkeyPatch, main_module: Any
) -> dict[str, Any]:
    candidate = {
        "score": 0.61,
        "desertionNo": "dog-1",
        "breed": "mixed",
        "age": "2022",
        "weight": "7(Kg)",
        "sex": "F",
        "neuter": "Y",
        "region": "Seoul",
        "noticeEdt": "20991231",
        "housing_types": ["apartment"],
        "activity_level": "low",
        "max_absence_hours": 9,
        "recommended_experience": "none",
        "children_compatible": True,
        "other_pets_compatible": False,
        "vlm_attrs": {"body_size_hint": "small", "photo_quality_score": 0.8},
    }
    calls: dict[str, list[Any]] = {
        "hybrid": [],
        "gemma": [],
        "embed_text": [],
        "embed_image": [],
    }

    def fake_hybrid_search(**kwargs: Any) -> list[dict[str, Any]]:
        calls["hybrid"].append(kwargs)
        return [deepcopy(candidate)]

    def fake_gemma_recommend(
        *, profile_text: str, candidates: list[dict[str, Any]]
    ) -> str:
        calls["gemma"].append((profile_text, deepcopy(candidates)))
        return "stub recommendation"

    def fake_embed_text(text: str) -> np.ndarray:
        calls["embed_text"].append(text)
        return np.array([[1.0, 0.0]], dtype=np.float32)

    def fake_embed_image(image: Image.Image) -> np.ndarray:
        calls["embed_image"].append(image.size)
        return np.array([[0.0, 1.0]], dtype=np.float32)

    monkeypatch.setattr(main_module, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(main_module, "gemma_recommend", fake_gemma_recommend)
    monkeypatch.setattr(main_module, "embed_text", fake_embed_text)
    monkeypatch.setattr(main_module, "embed_image", fake_embed_image)
    calls["candidate"] = [candidate]
    return calls


def test_search_text_legacy_route_contract(
    client: TestClient,
    isolated_runtime: dict[str, Any],
) -> None:
    response = client.post(
        "/search/text",
        headers=API_HEADERS,
        json={"query": "small calm dog", "topk": 2},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["retrieval"] == "graph_enhanced_multimodal_rag"
    assert payload["query"] == "small calm dog"
    assert payload["results"] == isolated_runtime["candidate"]
    assert isinstance(payload["structured_query"], dict)

    assert len(isolated_runtime["hybrid"]) == 1
    search_call = isolated_runtime["hybrid"][0]
    assert search_call["topk"] == 2
    assert search_call["rerank_depth"] == 50
    assert "query_vec" not in search_call


def test_recommend_with_image_legacy_route_contract(
    client: TestClient,
    isolated_runtime: dict[str, Any],
) -> None:
    image_bytes = BytesIO()
    Image.new("RGB", (3, 2), color=(120, 80, 40)).save(image_bytes, format="PNG")
    response = client.post(
        "/recommend_with_image?topk=2",
        headers=API_HEADERS,
        data={"profile": "quiet apartment companion"},
        files={"ref_image": ("dog.png", image_bytes.getvalue(), "image/png")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "profile": "quiet apartment companion",
        "recommendation": "stub recommendation",
        "count": 1,
        "results": isolated_runtime["candidate"],
    }
    assert isolated_runtime["embed_image"] == [(3, 2)]
    assert len(isolated_runtime["embed_text"]) == 1
    assert isolated_runtime["gemma"][0][0] == "quiet apartment companion"

    search_call = isolated_runtime["hybrid"][0]
    assert search_call["topk"] == 2
    assert search_call["rerank_depth"] == 50
    assert isinstance(search_call["query_vec"], np.ndarray)
    assert search_call["query_vec"].shape == (1, 2)


def test_search_profile_route_runs_real_validation_and_reranking(
    client: TestClient,
    isolated_runtime: dict[str, Any],
    main_module: Any,
) -> None:
    profile = {
        "housing_type": "apartment",
        "daily_absence_hours": 8,
        "activity_level": "low",
        "dog_experience": "none",
        "preferred_size": "small",
        "preferred_age": "adult",
        "preferred_region": "Seoul",
        "has_children": True,
        "has_other_pets": False,
    }
    response = client.post(
        "/search/profile",
        headers=API_HEADERS,
        json={"query": "small calm dog", "profile": profile, "topk": 1},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["retrieval"] == "graph_enhanced_multimodal_rag+profile_rerank"
    assert payload["profile"] == profile
    assert payload["candidate_count"] == 1
    assert payload["count"] == 1
    assert payload["disclaimer"] == main_module.PROFILE_RESULT_DISCLAIMER
    assert payload["notice_policy"]["include_unknown"] is True

    result = payload["results"][0]
    assert result["dog_id"] == "dog-1"
    assert result["retrieval_score"] == pytest.approx(0.61)
    assert result["final_score"] >= result["retrieval_score"]
    assert result["matched_conditions"]
    assert result["applicable_count"] >= result["evaluated_count"] > 0
    assert result["evidence_coverage"] == pytest.approx(
        result["evaluated_count"] / result["applicable_count"]
    )
    assert (
        f"확인된 조건 {result['evaluated_count']}/{result['applicable_count']}"
        in result["recommendation_reason"]
    )

    search_call = isolated_runtime["hybrid"][0]
    expected_candidates = main_module.PROFILE_RERANK_SETTINGS.candidate_count(1)
    assert search_call["topk"] == expected_candidates
    assert search_call["rerank_depth"] == max(expected_candidates * 10, 80)


def test_profile_route_rejects_incomplete_profile_before_search(
    client: TestClient,
    isolated_runtime: dict[str, Any],
) -> None:
    response = client.post(
        "/search/profile",
        headers=API_HEADERS,
        json={"query": "small dog", "profile": {"housing_type": "apartment"}},
    )

    assert response.status_code == 422
    assert isolated_runtime["hybrid"] == []


def test_profile_demo_route_serves_the_profile_search_ui(client: TestClient) -> None:
    response = client.get("/demo")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "먼저 볼 친구" in response.text
    assert "fetch('/search/profile'" in response.text
    assert "/rag/recommend_form" not in response.text


def test_rag_search_skips_optional_gemma_when_disabled(
    client: TestClient,
    isolated_runtime: dict[str, Any],
    main_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main_module, "GEMMA3_ENABLED", False)

    response = client.post(
        "/rag/recommend",
        headers=API_HEADERS,
        json={
            "query": "small calm dog",
            "topk": 1,
            "include_recommendation": True,
        },
    )

    assert response.status_code == 200
    assert response.json()["recommendation"] == ""
    assert isolated_runtime["gemma"] == []
