from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import sys
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from uuid import uuid4

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.public_text_release import (
    PUBLIC_TEXT_RELEASE_PROFILE,
    public_notice_text,
    public_text_sha256,
)


API_KEY = "public-text-runtime-test-key-32-chars"
API_HEADERS = {"x-api-key": API_KEY}


class _FakeClipModel:
    def eval(self) -> "_FakeClipModel":
        return self


class _FakeIndex:
    ntotal = 1
    d = 512

    def search(self, _vector: np.ndarray, topk: int) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.zeros((1, topk), dtype=np.float32),
            np.zeros((1, topk), dtype=np.int64),
        )


def _public_meta() -> dict[str, Any]:
    row: dict[str, Any] = {
        "type": "text",
        "embedding_source": "public_notice_text",
        "desertionNo": "dog-public-1",
        "desc": "흰색 털의 소형 믹스견",
        "sex": "F",
        "age": "2024(년생)",
        "weight": "5(Kg)",
        "neuter": "Y",
        "color": "흰색",
        "breed_name": "믹스견",
        "breed_source": "public_notice_reported",
        "breed_source_label": "믹스견",
        "mixed_breed": True,
        "notice_start": "20260701",
        "notice_end": "20991231",
        "process_state": "보호중",
        "last_verified_at": "2026-07-26T00:00:00+09:00",
        "detail_url": "https://www.animal.go.kr/front/awtis/public/publicDtl.do?desertionNo=dog-public-1",
        "species": "dog",
        "upkind": "417000",
    }
    row["desc_full"] = public_notice_text(row)
    row["embedding_text_sha256"] = public_text_sha256(row["desc_full"])
    return row


@pytest.fixture(scope="module")
def public_main_module(request: pytest.FixtureRequest):
    monkeypatch = pytest.MonkeyPatch()
    runtime_root = Path(__file__).resolve().parent / ".runtime"
    workdir = runtime_root / f"public-text-{uuid4().hex}"
    workdir.mkdir(parents=True)

    def cleanup_runtime() -> None:
        shutil.rmtree(workdir)
        try:
            runtime_root.rmdir()
        except OSError:
            pass

    request.addfinalizer(cleanup_runtime)
    index_path = workdir / "dog_faiss.index"
    metas_path = workdir / "dog_metas.json"
    marker_path = workdir / "release_profile.json"
    index_path.write_bytes(b"public-text-index-fixture")
    metas_path.write_text(
        json.dumps([_public_meta()], ensure_ascii=False),
        encoding="utf-8",
    )
    marker_path.write_text(
        json.dumps(
            {
                "profile": PUBLIC_TEXT_RELEASE_PROFILE,
                "output": {
                    "dimension": 512,
                    "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
                    "metas_sha256": hashlib.sha256(metas_path.read_bytes()).hexdigest(),
                    "vector_count": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    clip_stub = ModuleType("clip")
    setattr(
        clip_stub,
        "load",
        lambda _model_name, *, device: (_FakeClipModel(), lambda image: image),
    )
    setattr(
        clip_stub,
        "tokenize",
        lambda _texts, *, truncate=False: SimpleNamespace(
            to=lambda _device: np.zeros((1, 2), dtype=np.float32)
        ),
    )
    faiss_stub = ModuleType("faiss")
    setattr(faiss_stub, "read_index", lambda _path: _FakeIndex())
    torch_stub = ModuleType("torch")
    setattr(torch_stub, "cuda", SimpleNamespace(is_available=lambda: False))
    setattr(torch_stub, "no_grad", lambda: lambda function: function)
    setattr(torch_stub, "inference_mode", nullcontext)
    setattr(torch_stub, "float32", np.float32)
    setattr(torch_stub, "bfloat16", np.float32)

    monkeypatch.setitem(sys.modules, "clip", clip_stub)
    monkeypatch.setitem(sys.modules, "faiss", faiss_stub)
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setenv("APP_ENV", "contest")
    monkeypatch.setenv("API_KEY", API_KEY)
    monkeypatch.setenv("INDEX_PATH", str(index_path))
    monkeypatch.setenv("METAS_PATH", str(metas_path))
    monkeypatch.setenv("RELEASE_PROFILE", "public-text-only")
    monkeypatch.setenv("RELEASE_PROFILE_PATH", str(marker_path))
    monkeypatch.setenv("GEMMA3_ENABLED", "false")
    monkeypatch.setenv("ANIMAL_API_KEY", "")

    sys.modules.pop("app.main", None)
    try:
        module = importlib.import_module("app.main")
        yield module
    finally:
        sys.modules.pop("app.main", None)
        monkeypatch.undo()


@pytest.fixture
def public_client(public_main_module: Any):
    with TestClient(public_main_module.app) as client:
        yield client


def test_public_runtime_health_and_visual_routes_fail_closed(
    public_main_module: Any,
    public_client: TestClient,
) -> None:
    response = public_client.get("/health", headers=API_HEADERS)
    assert response.status_code == 200
    assert response.headers["x-meongtamjeong-release-profile"] == (
        PUBLIC_TEXT_RELEASE_PROFILE
    )
    assert "content-security-policy" in response.headers
    health = response.json()
    assert health["release_profile"] == PUBLIC_TEXT_RELEASE_PROFILE
    assert health["vector_modalities"] == {"text": 1}
    assert health["public_notice_images_enabled"] is False
    assert health["public_notice_visual_asset_routes_enabled"] is False
    assert health["graph_overlay_enabled"] is False

    assert public_main_module.load_graph_overlay_metas() == []
    assert (
        public_main_module.extract_image_url(
            {"image_url": "https://photos.example/leak.jpg"}
        )
        == ""
    )
    assert (
        public_main_module.crop_file_url("data/image_crops/dog/leak.jpg", API_KEY) == ""
    )
    for route in (
        "/breeds/000114/images",
        "/visualize/breeds",
        "/visualize/image-crop/dog/leak.jpg",
        "/visualize/image-audit",
        "/live/breeds",
    ):
        blocked = public_client.get(route, headers=API_HEADERS)
        assert blocked.status_code == 404


def test_public_runtime_formats_results_without_photo_metadata(
    public_main_module: Any,
) -> None:
    formatted = public_main_module.format_hybrid_result(
        {
            "doc": {
                "meta": {
                    **_public_meta(),
                    "image_url": "https://photos.example/leak.jpg",
                    "image_urls": ["https://photos.example/leak.jpg"],
                    "image_attrs": {"photo_quality_score": 1.0},
                    "vlm_attrs": {"coat_color": "white"},
                    "photo_advice": ["leak"],
                }
            },
            "score": 0.8,
            "score_parts": {"photo_quality": None, "visual_attrs": None},
            "evidence": {},
        },
        1,
    )

    assert formatted["image_url"] == ""
    assert formatted["image_urls"] == []
    assert formatted["image_attrs"] == {}
    assert formatted["photo_quality_score"] is None
    assert formatted["photo_advice"] == []
    assert formatted["visual_attrs"] == ""
    assert formatted["crop_path"] == ""


def test_public_runtime_keeps_image_upload_search_with_text_corpus_limitation(
    public_main_module: Any,
    public_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = {
        "score": 0.72,
        "desertionNo": "dog-public-1",
        **_public_meta(),
    }
    monkeypatch.setattr(
        public_main_module,
        "embed_image",
        lambda _image: np.array([[0.0, 1.0]], dtype=np.float32),
    )
    monkeypatch.setattr(
        public_main_module,
        "embed_text",
        lambda _text: np.array([[1.0, 0.0]], dtype=np.float32),
    )
    monkeypatch.setattr(
        public_main_module,
        "hybrid_search",
        lambda **_kwargs: [candidate],
    )
    image_bytes = BytesIO()
    Image.new("RGB", (4, 4), color=(230, 220, 210)).save(image_bytes, format="PNG")

    response = public_client.post(
        "/search/appearance/image",
        headers=API_HEADERS,
        data={"query": "흰색 소형견", "topk": "1"},
        files={"ref_image": ("reference.png", image_bytes.getvalue(), "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["release_profile"] == PUBLIC_TEXT_RELEASE_PROFILE
    assert payload["candidate_vector_modalities"] == ["text"]
    assert payload["image_query_limitation"]
    assert payload["count"] == 1
    result = payload["results"][0]
    assert result["quality_score"] is None
    assert result["meta"]["image_url"] == ""
    assert result["meta"]["image_urls"] == []


def test_public_runtime_security_and_marker_contracts_fail_closed(
    public_main_module: Any,
) -> None:
    assert public_main_module._validated_app_env(" Development ") == "development"
    for invalid in ("", "staging", "prodution"):
        with pytest.raises(RuntimeError, match="APP_ENV must be one of"):
            public_main_module._validated_app_env(invalid)
    with pytest.raises(RuntimeError, match="conflicts"):
        public_main_module._resolve_release_profile(
            {"profile": "full"},
            "public-text-only",
        )
    with pytest.raises(RuntimeError, match="metadata contract failed"):
        public_main_module._validate_public_text_runtime_artifacts(
            SimpleNamespace(ntotal=1, d=512),
            [
                {
                    "type": "image",
                    "embedding_source": "full_image",
                    "desertionNo": "dog-1",
                    "image_url": "https://photos.example/leak.jpg",
                }
            ],
            {"profile": PUBLIC_TEXT_RELEASE_PROFILE},
            index_path=Path("unused.index"),
            metas_path=Path("unused.json"),
        )
