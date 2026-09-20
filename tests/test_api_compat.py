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
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.profile_rerank import normalize_dog


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
    monkeypatch.setenv("DINO_FUSION_MODE", "off")
    monkeypatch.setenv("IMAGE_UPLOAD_MAX_BYTES", "invalid-number")
    monkeypatch.setenv("IMAGE_UPLOAD_MAX_PIXELS", "invalid-number")
    monkeypatch.setenv("NOTICE_LOOKUP_TIMEOUT_SECONDS", "invalid-number")
    monkeypatch.setenv("NOTICE_LOOKUP_CACHE_TTL_SECONDS", "invalid-number")
    monkeypatch.setenv("NOTICE_LOOKUP_MAX_CONCURRENCY", "invalid-number")

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


def test_deployment_security_defaults_are_fail_closed(main_module: Any) -> None:
    assert "*" not in main_module.CORS_ALLOW_ORIGINS
    assert main_module._validated_app_env(" Development ") == "development"
    for invalid_environment in ("", "staging", "prodution"):
        with pytest.raises(RuntimeError, match="APP_ENV must be one of"):
            main_module._validated_app_env(invalid_environment)
    assert main_module._validated_api_key("local-demo", "development") == "local-demo"
    with pytest.raises(RuntimeError, match="API_KEY must be replaced"):
        main_module._validated_api_key("change-me", "contest")
    with pytest.raises(RuntimeError, match="API_KEY must be replaced"):
        main_module._validated_api_key("", "production")
    with pytest.raises(RuntimeError, match="at least 24 characters"):
        main_module._validated_api_key("too-short", "contest")
    strong_key = "contest-key-with-32-random-chars"
    assert main_module._validated_api_key(strong_key, "production") == strong_key
    assert main_module.NOTICE_LOOKUP_MAX_CONCURRENCY == 2


def test_release_profile_marker_conflicts_fail_closed(main_module: Any) -> None:
    profile = main_module.PUBLIC_TEXT_RELEASE_PROFILE
    assert main_module._resolve_release_profile({"profile": profile}, "full") == profile
    with pytest.raises(RuntimeError, match="conflicts"):
        main_module._resolve_release_profile(
            {"profile": "full"},
            "public-text-only",
        )
    with pytest.raises(RuntimeError, match="unsupported release profile marker"):
        main_module._resolve_release_profile({"profile": "visual-v2"}, "full")


def test_public_text_startup_contract_rejects_full_metadata(
    main_module: Any,
) -> None:
    with pytest.raises(RuntimeError, match="metadata contract failed"):
        main_module._validate_public_text_runtime_artifacts(
            SimpleNamespace(ntotal=1, d=512),
            [
                {
                    "type": "image",
                    "embedding_source": "full_image",
                    "desertionNo": "dog-1",
                    "image_url": "https://photos.example/dog-1.jpg",
                }
            ],
            {"profile": main_module.PUBLIC_TEXT_RELEASE_PROFILE},
            index_path=Path("unused.index"),
            metas_path=Path("unused.json"),
        )


@pytest.mark.parametrize(
    "path",
    ("/visualize/dashboard", "/visualize/adoption-flow"),
)
def test_legacy_experimental_uis_remain_available_only_for_development(
    path: str,
    client: TestClient,
    main_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main_module, "APP_ENV", "development")

    response = client.get(path, headers=API_HEADERS)

    assert response.status_code == 200
    assert "Legacy experimental UI — development only." in response.text
    assert "the appearance-first demo" in response.text
    assert "sessionStorage" in response.text
    assert "localStorage" not in response.text


@pytest.mark.parametrize(
    "path",
    ("/visualize/dashboard", "/visualize/adoption-flow"),
)
def test_development_legacy_experimental_uis_still_require_authentication(
    path: str,
    client: TestClient,
    main_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main_module, "APP_ENV", "development")

    response = client.get(path, follow_redirects=False)

    assert response.status_code == 403
    assert response.json() == {"detail": "Forbidden"}


@pytest.mark.parametrize("app_env", ("contest", "production"))
@pytest.mark.parametrize(
    "path",
    ("/visualize/dashboard", "/visualize/adoption-flow"),
)
def test_deployed_legacy_experimental_uis_redirect_without_authentication(
    app_env: str,
    path: str,
    client: TestClient,
    main_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main_module, "APP_ENV", app_env)

    response = client.get(path, follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/demo"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-meongtamjeong-legacy-ui"] == "disabled"


@pytest.mark.parametrize("app_env", ("contest", "production"))
@pytest.mark.parametrize(
    "path",
    ("/visualize/dashboard", "/visualize/adoption-flow"),
)
def test_deployed_legacy_experimental_uis_redirect_without_leaking_key(
    app_env: str,
    path: str,
    client: TestClient,
    main_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main_module, "APP_ENV", app_env)

    response = client.get(
        f"{path}?api_key={API_HEADERS['x-api-key']}",
        follow_redirects=False,
    )

    assert response.status_code == 307
    assert response.headers["location"] == "/demo"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-meongtamjeong-legacy-ui"] == "disabled"
    assert API_HEADERS["x-api-key"] not in response.headers["location"]
    assert API_HEADERS["x-api-key"] not in response.text


def test_get_query_api_key_is_development_only(
    client: TestClient,
    main_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main_module, "APP_ENV", "development")
    local_response = client.get(
        f"/health?api_key={API_HEADERS['x-api-key']}",
    )
    assert local_response.status_code == 200

    for deployed_env in ("contest", "production"):
        monkeypatch.setattr(main_module, "APP_ENV", deployed_env)
        deployed_response = client.get(
            f"/health?api_key={API_HEADERS['x-api-key']}",
        )
        assert deployed_response.status_code == 403
        assert deployed_response.json() == {"detail": "Forbidden"}

        header_response = client.get("/health", headers=API_HEADERS)
        assert header_response.status_code == 200


@pytest.mark.parametrize("app_env", ("contest", "production"))
@pytest.mark.parametrize(
    "path",
    ("/live/breeds", "/live/breeds/417000/images", "/visualize/live-breeds"),
)
def test_deployed_live_gallery_routes_cannot_trigger_bulk_public_api_fetch(
    app_env: str,
    path: str,
    client: TestClient,
    main_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main_module, "APP_ENV", app_env)

    def unexpected_fetch(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("deployed live gallery route attempted a bulk fetch")

    monkeypatch.setattr(main_module, "get_live_dog_cache", unexpected_fetch)

    response = client.get(
        f"{path}?refresh=true&years=5",
        headers=API_HEADERS,
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}


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
        "behavior_evidence_source": "shelter_reported",
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


def test_invalid_numeric_env_uses_safe_upload_and_lookup_defaults(
    main_module: Any,
) -> None:
    assert main_module.IMAGE_UPLOAD_MAX_BYTES == 8 * 1024 * 1024
    assert main_module.IMAGE_UPLOAD_MAX_PIXELS == 25_000_000
    assert main_module.NOTICE_LOOKUP_TIMEOUT_SECONDS == 12.0
    assert main_module.NOTICE_LOOKUP_CACHE_TTL_SECONDS == 300.0


def test_dino_shadow_overlap_reads_profile_reranker_dog_ids(
    main_module: Any,
) -> None:
    diagnostics = {"top_notice_ids": ["dog-1", "dog-2", "dog-3"]}
    results = [
        {"dog_id": "dog-1"},
        {"meta": {"desertionNo": "dog-2"}},
        {"notice_id": "dog-outside"},
    ]

    main_module.add_dino_served_overlap(diagnostics, results)

    assert diagnostics["served_topk_overlap"] == 2


def test_active_dino_evidence_survives_profile_result_normalization(
    main_module: Any,
) -> None:
    candidates = [
        {
            "desertionNo": "dog-1",
            "retrieval_evidence": {"vector_evidence": {"dino_raw_similarity": 0.9}},
            "hybrid_scores": {"vector": 1.0},
        }
    ]
    results = [{"dog_id": "dog-1"}]

    main_module.attach_dino_retrieval_context(results, candidates)

    assert results == [
        {
            "dog_id": "dog-1",
            "retrieval_evidence": {
                "vector_evidence": {"dino_raw_similarity": 0.9}
            },
            "hybrid_scores": {"vector": 1.0},
        }
    ]


def test_authoritative_vector_override_skips_legacy_rescoring(
    main_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def capture_rank(**kwargs: Any) -> list[dict[str, Any]]:
        captured.update(kwargs)
        return []

    def unexpected_graph(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("authoritative DINO scoring must skip graph reranking")

    monkeypatch.setattr(main_module, "rank_hybrid_documents", capture_rank)
    monkeypatch.setattr(main_module, "rerank_with_graph", unexpected_graph)
    monkeypatch.setattr(
        main_module,
        "DOG_GRAPH",
        SimpleNamespace(candidate_doc_scores=unexpected_graph),
    )

    results = main_module.hybrid_search(
        query_text="black small dog",
        structured_query={"color": ["black"], "size": "small"},
        topk=1,
        vector_scores_override={0: 1.0},
        vector_score_details_override={0: {"evidence": {"dino": True}}},
        vector_override_authoritative=True,
    )

    assert results == []
    assert captured["query_text"] == ""
    assert captured["structured"] == {}
    assert captured["extra_candidate_scores"] == {}
    assert captured["visual_metadata_enabled"] is False


@pytest.mark.parametrize(
    "path",
    [
        "/search/profile",
        "/search/appearance/image",
        "/adoption/contact_card",
    ],
)
def test_new_profile_and_contact_posts_require_api_key(
    path: str,
    client: TestClient,
) -> None:
    missing = client.post(path)
    wrong = client.post(path, headers={"x-api-key": "wrong-key"})

    assert missing.status_code == 403
    assert missing.json() == {"detail": "Forbidden"}
    assert wrong.status_code == 403
    assert wrong.json() == {"detail": "Forbidden"}


@pytest.mark.parametrize(
    ("filename", "content", "content_type", "expected_status"),
    [
        ("empty.png", b"", "image/png", 400),
        ("not-image.txt", b"plain text", "text/plain", 415),
        ("broken.png", b"\x89PNG\r\n\x1a\ntruncated", "image/png", 415),
    ],
)
def test_recommend_with_image_rejects_empty_non_image_and_damaged_uploads(
    client: TestClient,
    isolated_runtime: dict[str, Any],
    filename: str,
    content: bytes,
    content_type: str,
    expected_status: int,
) -> None:
    response = client.post(
        "/recommend_with_image",
        headers=API_HEADERS,
        data={"profile": "small white dog"},
        files={"ref_image": (filename, content, content_type)},
    )

    assert response.status_code == expected_status
    assert "traceback" not in response.text.lower()
    assert isolated_runtime["embed_image"] == []


def test_recommend_with_image_rejects_oversized_upload_before_decoding(
    client: TestClient,
    isolated_runtime: dict[str, Any],
    main_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main_module, "IMAGE_UPLOAD_MAX_BYTES", 16)
    response = client.post(
        "/recommend_with_image",
        headers=API_HEADERS,
        data={"profile": "small white dog"},
        files={"ref_image": ("large.png", b"x" * 17, "image/png")},
    )

    assert response.status_code == 413
    assert response.json() == {
        "detail": "image upload exceeds the configured size limit"
    }
    assert isolated_runtime["embed_image"] == []


def test_recommend_with_image_rejects_excessive_pixel_count_before_embedding(
    client: TestClient,
    isolated_runtime: dict[str, Any],
    main_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_bytes = BytesIO()
    Image.new("RGB", (3, 2), color=(120, 80, 40)).save(image_bytes, format="PNG")
    monkeypatch.setattr(main_module, "IMAGE_UPLOAD_MAX_PIXELS", 5)

    response = client.post(
        "/recommend_with_image",
        headers=API_HEADERS,
        data={"profile": "small white dog"},
        files={"ref_image": ("dog.png", image_bytes.getvalue(), "image/png")},
    )

    assert response.status_code == 413
    assert response.json() == {
        "detail": "image dimensions exceed the configured pixel limit"
    }
    assert isolated_runtime["embed_image"] == []


def test_recommend_with_image_turns_pillow_decompression_bomb_into_safe_4xx(
    client: TestClient,
    isolated_runtime: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_bytes = BytesIO()
    Image.new("RGB", (2, 2), color=(120, 80, 40)).save(image_bytes, format="PNG")
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1)

    response = client.post(
        "/recommend_with_image",
        headers=API_HEADERS,
        data={"profile": "small white dog"},
        files={"ref_image": ("dog.png", image_bytes.getvalue(), "image/png")},
    )

    assert response.status_code == 413
    assert "pixel limit" in response.json()["detail"]
    assert "traceback" not in response.text.lower()
    assert isolated_runtime["embed_image"] == []


def test_rag_recommend_form_uses_the_same_hardened_image_loader(
    client: TestClient,
    isolated_runtime: dict[str, Any],
) -> None:
    response = client.post(
        "/rag/recommend_form",
        headers=API_HEADERS,
        data={"query": "", "include_recommendation": "false"},
        files={"ref_image": ("broken.jpg", b"not a jpeg", "image/jpeg")},
    )

    assert response.status_code == 415
    assert "traceback" not in response.text.lower()
    assert isolated_runtime["embed_image"] == []


def test_rag_recommend_form_keeps_normal_image_response_contract(
    client: TestClient,
    isolated_runtime: dict[str, Any],
) -> None:
    image_bytes = BytesIO()
    Image.new("RGB", (3, 2), color=(120, 80, 40)).save(image_bytes, format="PNG")
    response = client.post(
        "/rag/recommend_form",
        headers=API_HEADERS,
        data={"query": "", "include_recommendation": "false", "topk": "2"},
        files={"ref_image": ("dog.png", image_bytes.getvalue(), "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["retrieval"] == "graph_enhanced_multimodal_rag"
    assert payload["count"] == 1
    assert payload["results"] == isolated_runtime["candidate"]
    assert payload["recommendation"] == ""
    assert isolated_runtime["embed_image"] == [(3, 2)]


def test_survey_form_preserves_image_validation_status_without_error_leak(
    client: TestClient,
    isolated_runtime: dict[str, Any],
) -> None:
    survey = {
        "living": "apartment",
        "family": "one person",
        "walk_time": "one hour",
        "dog_size": "small",
        "preferred_personality": "calm",
    }
    response = client.post(
        "/recommend_with_survey_form",
        headers=API_HEADERS,
        data={"survey": json.dumps(survey)},
        files={"ref_image": ("broken.png", b"not an image", "image/png")},
    )

    assert response.status_code == 415
    assert response.json() == {"detail": "a valid image file is required"}
    assert "traceback" not in response.text.lower()
    assert isolated_runtime["embed_image"] == []


def test_survey_form_keeps_normal_image_response_contract(
    client: TestClient,
    isolated_runtime: dict[str, Any],
) -> None:
    survey = {
        "living": "apartment",
        "family": "one person",
        "walk_time": "one hour",
        "dog_size": "small",
        "preferred_personality": "calm",
    }
    image_bytes = BytesIO()
    Image.new("RGB", (3, 2), color=(120, 80, 40)).save(image_bytes, format="PNG")
    response = client.post(
        "/recommend_with_survey_form?topk=2",
        headers=API_HEADERS,
        data={"survey": json.dumps(survey), "extra_text": "white coat"},
        files={"ref_image": ("dog.png", image_bytes.getvalue(), "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["extra_text"] == "white coat"
    assert payload["recommendation"] == "stub recommendation"
    assert payload["count"] == 1
    assert payload["results"] == isolated_runtime["candidate"]
    assert isolated_runtime["embed_image"] == [(3, 2)]


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
    assert payload["ranking_scope"] == "profile"
    assert payload["query_policy"] is None
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


def test_profile_route_supports_appearance_only_reranking(
    client: TestClient,
    isolated_runtime: dict[str, Any],
) -> None:
    profile = {
        "preferred_size": "small",
        "preferred_age": "adult",
        "preferred_region": "Seoul",
    }
    response = client.post(
        "/search/profile",
        headers=API_HEADERS,
        json={
            "query": "small white adult dog",
            "profile": profile,
            "topk": 1,
            "ranking_scope": "appearance",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ranking_scope"] == "appearance"
    assert payload["query_policy"] == {
        "typo_corrections": [],
        "synonym_normalizations": [],
        "unsupported_conditions": [],
        "warnings": [],
        "is_fully_supported": True,
    }
    assert payload["profile"] == profile
    assert payload["notice_policy"]["include_unknown"] is False
    assert isolated_runtime["hybrid"][0]["include_unknown_notices"] is False
    result = payload["results"][0]
    assert result["applicable_count"] == 3
    assert result["evaluated_count"] == 3
    combined = " ".join(
        result["matched_conditions"]
        + result["caution_conditions"]
        + result["unknown_conditions"]
    )
    assert "주거" not in combined
    assert "활동" not in combined
    assert "아동" not in combined


def test_appearance_scope_excludes_unknown_notice_status(
    client: TestClient,
    isolated_runtime: dict[str, Any],
    main_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    isolated_runtime["candidate"][0].pop("noticeEdt", None)
    monkeypatch.setitem(
        main_module.HYBRID_META_BY_ID,
        "dog-1",
        {"desertionNo": "dog-1"},
    )
    profile = {
        "housing_type": "apartment",
        "daily_absence_hours": 8,
        "activity_level": "low",
        "dog_experience": "none",
        "preferred_size": "small",
        "preferred_age": "adult",
        "preferred_region": "Seoul",
        "has_children": True,
        "has_other_pets": True,
    }
    response = client.post(
        "/search/profile",
        headers=API_HEADERS,
        json={
            "query": "small white adult dog",
            "profile": profile,
            "topk": 1,
            "ranking_scope": "appearance",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["notice_policy"]["include_unknown"] is False
    assert payload["count"] == 0
    assert payload["results"] == []
    assert isolated_runtime["hybrid"][0]["include_unknown_notices"] is False


def test_appearance_scope_results_are_identical_when_only_lifestyle_changes(
    client: TestClient,
    isolated_runtime: dict[str, Any],
) -> None:
    common = {
        "preferred_size": "small",
        "preferred_age": "adult",
        "preferred_region": "Seoul",
    }
    profiles = [
        {
            **common,
            "housing_type": "apartment",
            "daily_absence_hours": 10,
            "activity_level": "low",
            "dog_experience": "none",
            "has_children": True,
            "has_other_pets": True,
        },
        {
            **common,
            "housing_type": "house",
            "daily_absence_hours": 1,
            "activity_level": "high",
            "dog_experience": "experienced",
            "has_children": False,
            "has_other_pets": False,
        },
    ]

    payloads = []
    for profile in profiles:
        response = client.post(
            "/search/profile",
            headers=API_HEADERS,
            json={
                "query": "차분한 흰색 소형견",
                "profile": profile,
                "topk": 1,
                "ranking_scope": "appearance",
            },
        )
        assert response.status_code == 200
        payloads.append(response.json())

    assert payloads[0]["retrieval_query"] == "흰색 소형견"
    assert payloads[0]["results"] == payloads[1]["results"]
    assert payloads[0]["profile"] == profiles[0]
    assert payloads[1]["profile"] == profiles[1]
    assert isolated_runtime["hybrid"][0] == isolated_runtime["hybrid"][1]
    assert isolated_runtime["hybrid"][0]["excluded_graph_feature_prefixes"] == (
        "temperament:",
    )


def test_appearance_scope_excludes_closed_notice_after_retrieval(
    client: TestClient,
    isolated_runtime: dict[str, Any],
) -> None:
    isolated_runtime["candidate"][0]["processState"] = "종료(입양)"
    profile = {
        "housing_type": "other",
        "daily_absence_hours": 0,
        "activity_level": "medium",
        "dog_experience": "none",
        "preferred_size": "any",
        "preferred_age": "any",
        "preferred_region": None,
        "has_children": False,
        "has_other_pets": False,
    }

    response = client.post(
        "/search/profile",
        headers=API_HEADERS,
        json={
            "query": "흰색 강아지",
            "profile": profile,
            "topk": 1,
            "ranking_scope": "appearance",
        },
    )

    assert response.status_code == 200
    assert response.json()["results"] == []
    assert response.json()["count"] == 0


def test_appearance_scope_exposes_typo_correction_and_negation_warning(
    client: TestClient,
    isolated_runtime: dict[str, Any],
) -> None:
    response = client.post(
        "/search/profile",
        headers=API_HEADERS,
        json={
            "query": "갈색은 아닌 소형겐을 찾아줘",
            "profile": {"preferred_size": "small", "preferred_age": "any"},
            "topk": 1,
            "ranking_scope": "appearance",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["retrieval_query"] == "소형견"
    assert payload["query_policy"]["is_fully_supported"] is False
    assert payload["query_policy"]["typo_corrections"] == [
        {
            "original": "소형겐을",
            "corrected": "소형견을",
            "canonical_target": "소형견",
            "policy": "unique_one_codepoint_substitution",
        }
    ]
    unsupported = payload["query_policy"]["unsupported_conditions"]
    assert [item["term"] for item in unsupported] == ["갈색"]
    assert payload["query_policy"]["warnings"]
    assert "갈색" not in isolated_runtime["hybrid"][0]["query_text"]


def test_appearance_scope_rejects_structured_reinjection_of_negated_field(
    client: TestClient,
    isolated_runtime: dict[str, Any],
) -> None:
    response = client.post(
        "/search/profile",
        headers=API_HEADERS,
        json={
            "query": "갈색은 아닌 소형견",
            "conditions": {"coat_color": ["brown"]},
            "profile": {"preferred_size": "small", "preferred_age": "any"},
            "topk": 1,
            "ranking_scope": "appearance",
        },
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["code"] == "conflicting_negated_appearance_condition"
    assert detail["conflicting_inputs"] == ["coat_color"]
    assert isolated_runtime["hybrid"] == []


def test_appearance_image_negation_only_caption_does_not_change_image_vector(
    client: TestClient,
    isolated_runtime: dict[str, Any],
) -> None:
    image_bytes = BytesIO()
    Image.new("RGB", (3, 2), color=(120, 80, 40)).save(image_bytes, format="PNG")
    response = client.post(
        "/search/appearance/image",
        headers=API_HEADERS,
        data={"query": "검정색이 아닌 강아지", "topk": "1"},
        files={"ref_image": ("dog.png", image_bytes.getvalue(), "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["retrieval_query"] == ""
    assert payload["input_modality"] == "image"
    assert payload["query_policy"]["is_fully_supported"] is False
    assert isolated_runtime["embed_text"] == []


def test_appearance_image_rejects_negated_preference_reinjection_before_embedding(
    client: TestClient,
    isolated_runtime: dict[str, Any],
) -> None:
    image_bytes = BytesIO()
    Image.new("RGB", (3, 2), color=(120, 80, 40)).save(image_bytes, format="PNG")
    response = client.post(
        "/search/appearance/image",
        headers=API_HEADERS,
        data={"query": "소형견이 아닌 강아지", "preferred_size": "small"},
        files={"ref_image": ("dog.png", image_bytes.getvalue(), "image/png")},
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["code"] == "conflicting_negated_appearance_condition"
    assert detail["conflicting_inputs"] == ["profile.preferred_size"]
    assert isolated_runtime["embed_image"] == []
    assert isolated_runtime["hybrid"] == []


def test_appearance_image_route_combines_uploaded_image_with_safe_visual_text(
    client: TestClient,
    isolated_runtime: dict[str, Any],
    main_module: Any,
) -> None:
    image_bytes = BytesIO()
    Image.new("RGB", (4, 3), color=(245, 240, 230)).save(image_bytes, format="PNG")
    response = client.post(
        "/search/appearance/image",
        headers=API_HEADERS,
        data={
            "query": "아이와 잘 지내는 차분한 흰색 소형견",
            "preferred_size": "small",
            "preferred_age": "adult",
            "preferred_region": "Seoul",
            "topk": "1",
        },
        files={
            "ref_image": (
                "https://attacker.invalid/dog.png",
                image_bytes.getvalue(),
                "image/png",
            )
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["retrieval"] == (
        "graph_enhanced_multimodal_rag+appearance_image+profile_rerank"
    )
    assert payload["input_modality"] == "image+text"
    assert payload["ranking_scope"] == "appearance"
    assert payload["query"] == "아이와 잘 지내는 차분한 흰색 소형견"
    assert payload["retrieval_query"] == "흰색 소형견"
    assert payload["nonappearance_query_terms_excluded"] is True
    assert payload["structured_query"]["personality"] == []
    assert payload["structured_query"]["keywords"] == []
    assert payload["profile"]["preferred_size"] == "small"
    assert payload["profile"]["preferred_age"] == "adult"
    assert payload["profile"]["preferred_region"] == "Seoul"
    assert set(payload["profile"]) == {
        "preferred_size",
        "preferred_age",
        "preferred_region",
    }
    assert payload["candidate_count"] == 1
    assert payload["count"] == 1
    assert payload["disclaimer"] == main_module.PROFILE_RESULT_DISCLAIMER

    result = payload["results"][0]
    assert result["retrieval_score"] == pytest.approx(0.61)
    assert result["applicable_count"] == 3
    assert result["quality_score"] == pytest.approx(0.8)
    assert "recommendation_reason" in result
    evidence = " ".join(
        result["matched_conditions"]
        + result["caution_conditions"]
        + result["unknown_conditions"]
    )
    for forbidden in ("주거", "활동", "아동", "다른 반려동물"):
        assert forbidden not in evidence

    assert isolated_runtime["embed_image"] == [(4, 3)]
    assert len(isolated_runtime["embed_text"]) == 1
    for forbidden in ("아이", "지내", "차분", "성향", "personality"):
        assert forbidden not in isolated_runtime["embed_text"][0]
    assert isolated_runtime["gemma"] == []

    search_call = isolated_runtime["hybrid"][0]
    expected_candidates = main_module.PROFILE_RERANK_SETTINGS.candidate_count(1)
    assert search_call["topk"] == expected_candidates
    assert search_call["rerank_depth"] == max(expected_candidates * 10, 80)
    assert search_call["include_unknown_notices"] is False
    assert search_call["excluded_graph_feature_prefixes"] == ("temperament:",)
    assert search_call["query_vec"].shape == (1, 2)


def test_appearance_image_route_explicitly_closes_the_multipart_upload(
    client: TestClient,
    isolated_runtime: dict[str, Any],
    main_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_load(upload: Any, **kwargs: Any) -> Image.Image:
        captured["upload"] = upload
        return Image.new("RGB", (2, 2), color=(240, 240, 240))

    monkeypatch.setattr(main_module, "load_uploaded_image", fake_load)
    response = client.post(
        "/search/appearance/image",
        headers=API_HEADERS,
        data={
            "query": "흰색 소형견",
            "preferred_size": "small",
            "preferred_age": "adult",
            "topk": "1",
        },
        files={"ref_image": ("dog.png", b"synthetic", "image/png")},
    )

    assert response.status_code == 200
    assert isolated_runtime["embed_image"] == [(2, 2)]
    assert captured["upload"].file.closed is True


def test_appearance_image_route_supports_image_only_without_text_embedding(
    client: TestClient,
    isolated_runtime: dict[str, Any],
) -> None:
    image_bytes = BytesIO()
    Image.new("RGB", (3, 2), color=(120, 80, 40)).save(image_bytes, format="WEBP")
    response = client.post(
        "/search/appearance/image",
        headers=API_HEADERS,
        data={"preferred_size": "any", "preferred_age": "any", "topk": "2"},
        files={"ref_image": ("reference.webp", image_bytes.getvalue(), "image/webp")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["input_modality"] == "image"
    assert payload["query"] == ""
    assert payload["retrieval_query"] == ""
    assert payload["structured_query"]["personality"] == []
    assert payload["structured_query"]["keywords"] == []
    assert isolated_runtime["embed_image"] == [(3, 2)]
    assert isolated_runtime["embed_text"] == []
    assert isolated_runtime["gemma"] == []
    assert isolated_runtime["hybrid"][0]["query_vec"].shape == (1, 2)


@pytest.mark.parametrize(
    "query",
    [
        "활동적인 강아지",
        "아이들과 잘 지내는 차분한 강아지",
        "하루 8시간 혼자 있어도 괜찮은 반려견",
    ],
)
def test_appearance_image_route_ignores_nonappearance_only_text_signal(
    query: str,
    client: TestClient,
    isolated_runtime: dict[str, Any],
) -> None:
    image_bytes = BytesIO()
    Image.new("RGB", (3, 2), color=(120, 80, 40)).save(image_bytes, format="PNG")
    response = client.post(
        "/search/appearance/image",
        headers=API_HEADERS,
        data={
            "query": query,
            "preferred_size": "any",
            "preferred_age": "any",
            "topk": "2",
        },
        files={"ref_image": ("reference.png", image_bytes.getvalue(), "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["input_modality"] == "image"
    assert payload["query"] == query
    assert payload["retrieval_query"] == ""
    assert payload["nonappearance_query_terms_excluded"] is True
    assert payload["structured_query"]["personality"] == []
    assert payload["structured_query"]["keywords"] == []
    assert isolated_runtime["embed_image"] == [(3, 2)]
    assert isolated_runtime["embed_text"] == []
    assert isolated_runtime["gemma"] == []
    assert isolated_runtime["hybrid"][0]["query_text"] == ""
    np.testing.assert_array_equal(
        isolated_runtime["hybrid"][0]["query_vec"],
        np.array([[0.0, 1.0]], dtype=np.float32),
    )


def test_appearance_image_route_requires_a_real_bounded_upload(
    client: TestClient,
    isolated_runtime: dict[str, Any],
    main_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = client.post(
        "/search/appearance/image",
        headers=API_HEADERS,
        data={"query": "흰색 소형견"},
    )
    assert missing.status_code == 422

    invalid = client.post(
        "/search/appearance/image",
        headers=API_HEADERS,
        data={"query": "흰색 소형견"},
        files={"ref_image": ("not-image.txt", b"plain text", "image/png")},
    )
    assert invalid.status_code == 415

    bmp_bytes = BytesIO()
    Image.new("RGB", (2, 2), color=(120, 80, 40)).save(bmp_bytes, format="BMP")
    unsupported_format = client.post(
        "/search/appearance/image",
        headers=API_HEADERS,
        data={"query": "흰색 소형견"},
        files={"ref_image": ("disguised.png", bmp_bytes.getvalue(), "image/png")},
    )
    assert unsupported_format.status_code == 415
    assert unsupported_format.json()["detail"] == (
        "image format must be JPEG, PNG, or WebP"
    )

    monkeypatch.setattr(main_module, "IMAGE_UPLOAD_MAX_BYTES", 16)
    oversized = client.post(
        "/search/appearance/image",
        headers=API_HEADERS,
        data={"query": "흰색 소형견"},
        files={"ref_image": ("large.png", b"x" * 17, "image/png")},
    )
    assert oversized.status_code == 413
    assert oversized.json()["detail"] == (
        "image upload exceeds the configured size limit"
    )
    assert isolated_runtime["embed_image"] == []
    assert isolated_runtime["hybrid"] == []
    assert isolated_runtime["gemma"] == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("preferred_size", "giant"),
        ("preferred_age", "teen"),
        ("topk", "21"),
    ],
)
def test_appearance_image_route_validates_form_preferences_before_embedding(
    client: TestClient,
    isolated_runtime: dict[str, Any],
    field: str,
    value: str,
) -> None:
    image_bytes = BytesIO()
    Image.new("RGB", (2, 2), color=(120, 80, 40)).save(image_bytes, format="PNG")
    data = {"query": "흰색 소형견", field: value}
    response = client.post(
        "/search/appearance/image",
        headers=API_HEADERS,
        data=data,
        files={"ref_image": ("dog.png", image_bytes.getvalue(), "image/png")},
    )

    assert response.status_code == 422
    assert isolated_runtime["embed_image"] == []
    assert isolated_runtime["hybrid"] == []
    assert isolated_runtime["gemma"] == []


def test_appearance_scope_removes_behavior_and_household_terms_before_retrieval(
    client: TestClient,
    isolated_runtime: dict[str, Any],
) -> None:
    profile = {
        "housing_type": "apartment",
        "daily_absence_hours": 8,
        "activity_level": "low",
        "dog_experience": "none",
        "preferred_size": "small",
        "preferred_age": "any",
        "preferred_region": None,
        "has_children": True,
        "has_other_pets": True,
    }
    response = client.post(
        "/search/profile",
        headers=API_HEADERS,
        json={
            "query": "아이들과 잘 지내는 차분한 흰색 소형견",
            "conditions": {
                "personality": ["차분"],
                "keywords": ["아동 친화", "복슬복슬"],
                "coat_color": ["white"],
            },
            "profile": profile,
            "topk": 1,
            "ranking_scope": "appearance",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["retrieval_query"] == "흰색 소형견"
    assert payload["nonappearance_query_terms_excluded"] is True
    assert payload["structured_query"]["personality"] == []
    assert payload["structured_query"]["coat_color"] == ["white"]
    assert all(
        blocked not in payload["structured_query"]["keywords"]
        for blocked in ("아이들과", "지내는", "차분한", "아동", "친화")
    )

    search_call = isolated_runtime["hybrid"][0]
    assert search_call["query_text"]
    assert "흰색" in search_call["query_text"]
    assert "small" in search_call["query_text"]
    for blocked in ("아이", "지내", "차분", "성향", "personality"):
        assert blocked not in search_call["query_text"]
    assert search_call["structured_query"]["personality"] == []
    assert search_call["excluded_graph_feature_prefixes"] == ("temperament:",)


def test_appearance_scope_turns_nonvisual_only_query_into_neutral_search(
    client: TestClient,
    isolated_runtime: dict[str, Any],
) -> None:
    profile = {
        "housing_type": "apartment",
        "daily_absence_hours": 8,
        "activity_level": "low",
        "dog_experience": "none",
        "preferred_size": "any",
        "preferred_age": "any",
        "preferred_region": None,
        "has_children": True,
        "has_other_pets": False,
    }
    response = client.post(
        "/search/profile",
        headers=API_HEADERS,
        json={
            "query": "아이들과 잘 지내고 차분한 강아지",
            "profile": profile,
            "topk": 1,
            "ranking_scope": "appearance",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "아이들과 잘 지내고 차분한 강아지"
    assert payload["retrieval_query"] == "강아지"
    assert payload["nonappearance_query_terms_excluded"] is True
    assert payload["structured_query"]["personality"] == []
    assert isolated_runtime["hybrid"][0]["query_text"] == "강아지"


def test_profile_scope_keeps_legacy_broad_query_behavior(
    client: TestClient,
    isolated_runtime: dict[str, Any],
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
    raw_query = "아이들과 잘 지내고 차분한 강아지"
    response = client.post(
        "/search/profile",
        headers=API_HEADERS,
        json={"query": raw_query, "profile": profile, "topk": 1},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ranking_scope"] == "profile"
    assert payload["retrieval_query"] == raw_query
    assert payload["nonappearance_query_terms_excluded"] is False
    assert payload["structured_query"]["personality"] == ["차분"]
    assert "차분" in isolated_runtime["hybrid"][0]["query_text"]


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


def test_legacy_detail_url_helper_rejects_unsafe_provider_link(
    main_module: Any,
) -> None:
    resolved = main_module.resolve_detail_url(
        {
            "desertionNo": "SYNTH&A=1",
            "detail_url": "javascript:alert(1)",
        }
    )
    parsed = urlparse(resolved)

    assert parsed.scheme == "https"
    assert parsed.hostname == "www.animal.go.kr"
    assert parse_qs(parsed.query) == {
        "desertionNo": ["SYNTH&A=1"],
        "menuNo": ["1000000055"],
    }


def test_adoption_contact_card_builds_copyable_inquiry_materials(
    client: TestClient,
    main_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        main_module,
        "fetch_latest_notice_meta",
        lambda meta: (
            {
                **meta,
                "notice_no": "서울-2026-001",
                "breed": "믹스견",
                "care_name": "테스트동물보호센터",
                "care_tel": "02-1234-5678",
                "care_email": "shelter@example.org",
                "care_addr": "서울시 테스트구",
                "process_state": "보호중",
                "notice_end": "20991231",
            },
            "found",
        ),
    )
    response = client.post(
        "/adoption/contact_card",
        headers=API_HEADERS,
        json={
            "animal": {
                "desertionNo": "dog-contact-1",
                "notice_no": "서울-2026-001",
                "breed": "믹스견",
                "care_name": "테스트동물보호센터",
                "care_tel": "02-1234-5678",
                "care_email": "shelter@example.org",
                "care_addr": "서울시 테스트구",
                "process_state": "보호중",
                "notice_end": "20991231",
            },
            "preferences": {
                "desired_temperaments": ["calm", "friendly"],
                "activity_level": "high",
                "housing_type": "apartment",
                "daily_absence_hours": 8,
                "dog_experience": "none",
                "has_children": True,
                "has_other_pets": True,
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status_verified"] is True
    assert payload["connectable"] is True
    assert payload["contact_source"] == "latest_public_api"
    assert payload["direct_contact_allowed"] is True
    assert payload["direct_contact_block_reason"] == ""
    assert payload["actions"]["tel"] == "tel:0212345678"
    mailto = urlparse(payload["actions"]["mailto"])
    mailto_query = parse_qs(mailto.query)
    assert mailto.scheme == "mailto"
    assert mailto.path == "shelter@example.org"
    assert mailto_query["subject"] == [payload["email_subject"]]
    assert mailto_query["body"] == [payload["email_body"]]
    assert payload["contact_availability"] == {
        "phone": True,
        "email": True,
        "detail": True,
    }
    assert payload["actions"]["detail_url"]
    assert payload["actions"]["map_url"]
    assert payload["inquiry_questions"]
    assert payload["inquiry_script"] == payload["phone_script"]
    assert payload["email_template"] == {
        "subject": payload["email_subject"],
        "body": payload["email_body"],
    }
    assert payload["preference_usage"] == {
        "ranking": False,
        "inquiry_questions_only": True,
        "scored_fields": [],
    }
    assert payload["manual_contact_only"] is True
    assert "입양 신청이나 입양 확정" in payload["inquiry_disclaimer"]
    assert "개인정보" in payload["privacy_notice"]
    joined = " ".join(payload["inquiry_questions"])
    assert "낯선 사람" in joined
    assert "선호하는 활동량은 높은 편" in joined
    assert "실제로 관찰된 활동량" in joined
    assert "어린이" in joined
    assert "다른 개나 고양이" in joined
    assert "실제로 관찰" in payload["email_body"]


def test_adoption_contact_card_does_not_claim_unverified_active_status(
    client: TestClient,
    main_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        main_module,
        "fetch_latest_notice_meta",
        lambda _meta: (None, "api_key_missing"),
    )
    response = client.post(
        "/adoption/contact_card",
        headers=API_HEADERS,
        json={
            "animal": {
                "desertionNo": "legacy-contact-1",
                "process_state": "보호중",
                "notice_end": "20991231",
            }
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["latest_found"] is False
    assert payload["status_verified"] is False
    assert payload["connectable"] is False
    assert payload["contact_source"] == "unavailable"
    assert payload["trusted_snapshot_found"] is False
    assert payload["direct_contact_allowed"] is False
    assert payload["direct_contact_block_reason"] == "trusted_contact_unavailable"
    assert payload["actions"]["tel"] == ""
    assert payload["actions"]["mailto"] == ""
    assert payload["inquiry_script"] == payload["phone_script"]
    assert len(payload["inquiry_questions"]) == 2


def test_adoption_contact_card_can_render_snapshot_before_live_refresh(
    client: TestClient,
    main_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dog_id = "snapshot-first-contact"
    monkeypatch.setitem(
        main_module.HYBRID_META_BY_ID,
        dog_id,
        {
            "desertionNo": dog_id,
            "process_state": "보호중",
            "notice_end": "20991231",
            "care_name": "스냅샷보호소",
            "care_tel": "02-1234-5678",
        },
    )

    def must_not_fetch(_meta: dict[str, Any]) -> tuple[None, str]:
        raise AssertionError("refresh_latest=false must not call the public API")

    monkeypatch.setattr(main_module, "fetch_latest_notice_meta", must_not_fetch)
    response = client.post(
        "/adoption/contact_card",
        headers=API_HEADERS,
        json={"desertion_no": dog_id, "refresh_latest": False},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["lookup_status"] == "not_requested"
    assert payload["latest_found"] is False
    assert payload["contact_source"] == "server_snapshot"
    assert payload["direct_contact_allowed"] is True
    assert payload["actions"]["detail_url"]
    assert payload["phone_script"]


def test_adoption_contact_card_fails_fast_when_live_lookup_is_busy(
    client: TestClient,
    main_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dog_id = "busy-live-contact"
    monkeypatch.setitem(
        main_module.HYBRID_META_BY_ID,
        dog_id,
        {
            "desertionNo": dog_id,
            "process_state": "보호중",
            "notice_end": "20991231",
            "care_name": "스냅샷보호소",
        },
    )

    class BusySemaphore:
        def acquire(self, *, blocking: bool) -> bool:
            assert blocking is False
            return False

        def release(self) -> None:
            raise AssertionError("an unacquired semaphore must not be released")

    monkeypatch.setattr(main_module, "NOTICE_LOOKUP_SEMAPHORE", BusySemaphore())
    monkeypatch.setattr(
        main_module,
        "fetch_latest_notice_meta",
        lambda _meta: (_ for _ in ()).throw(
            AssertionError("busy lookups must fail before network access")
        ),
    )

    response = client.post(
        "/adoption/contact_card",
        headers=API_HEADERS,
        json={"desertion_no": dog_id, "refresh_latest": True},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["lookup_status"] == "busy"
    assert payload["contact_source"] == "server_snapshot"
    assert payload["phone_script"]


def test_adoption_contact_card_validates_inquiry_preferences(
    client: TestClient,
) -> None:
    response = client.post(
        "/adoption/contact_card",
        headers=API_HEADERS,
        json={
            "animal": {"desertionNo": "dog-contact-1"},
            "preferences": {"daily_absence_hours": 25},
        },
    )

    assert response.status_code == 422


def test_adoption_contact_card_rejects_unknown_activity_level(
    client: TestClient,
) -> None:
    response = client.post(
        "/adoption/contact_card",
        headers=API_HEADERS,
        json={
            "animal": {"desertionNo": "dog-contact-1"},
            "preferences": {"activity_level": "extreme"},
        },
    )

    assert response.status_code == 422


def test_adoption_contact_card_rejects_personal_data_in_free_text(
    client: TestClient,
) -> None:
    response = client.post(
        "/adoption/contact_card",
        headers=API_HEADERS,
        json={
            "animal": {"desertionNo": "dog-contact-1"},
            "preferences": {
                "additional_question": "제 번호 010-1234-5678로 답 주세요."
            },
        },
    )

    assert response.status_code == 422
    assert "개인 연락처나 식별번호" in response.text


def test_adoption_contact_card_omits_unsafe_or_missing_direct_actions(
    client: TestClient,
    main_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        main_module,
        "fetch_latest_notice_meta",
        lambda meta: ({**meta, "process_state": "보호중"}, "found"),
    )
    response = client.post(
        "/adoption/contact_card",
        headers=API_HEADERS,
        json={
            "animal": {
                "desertionNo": "dog-contact-missing",
                "care_tel": "12+34",
                "care_email": "not-an-email",
                "process_state": "보호중",
                "notice_end": "20991231",
            }
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["contact"]["care_email"] == ""
    assert payload["contact"]["care_tel"] == ""
    assert payload["actions"]["tel"] == ""
    assert payload["actions"]["mailto"] == ""
    assert payload["contact_availability"]["phone"] is False
    assert payload["contact_availability"]["email"] is False
    assert payload["phone_script"]
    assert payload["email_body"]


def test_adoption_contact_card_uses_server_snapshot_not_client_contact_fields(
    client: TestClient,
    main_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dog_id = "trusted-contact-1"
    monkeypatch.setitem(
        main_module.HYBRID_META_BY_ID,
        dog_id,
        {
            "desertionNo": dog_id,
            "breed": "믹스견",
            "care_name": "신뢰보호소",
            "care_tel": "02-2222-3333",
            "care_email": "trusted@example.org",
            "care_addr": "서울시 신뢰구",
            "detail_url": "https://evil.example/phishing",
            "process_state": "보호중",
            "notice_end": "20991231",
        },
    )
    monkeypatch.setattr(
        main_module,
        "fetch_latest_notice_meta",
        lambda _meta: (None, "timeout"),
    )

    response = client.post(
        "/adoption/contact_card",
        headers=API_HEADERS,
        json={
            "animal": {
                "desertionNo": dog_id,
                "care_name": "가짜보호소",
                "care_tel": "010-9999-9999",
                "care_email": "attacker@example.net",
                "care_addr": "가짜 주소",
                "detail_url": "https://evil.example/client-phishing",
            }
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["contact_source"] == "server_snapshot"
    assert payload["trusted_snapshot_found"] is True
    assert payload["direct_contact_allowed"] is True
    assert payload["contact"]["care_name"] == "신뢰보호소"
    assert payload["contact"]["care_tel"] == "02-2222-3333"
    assert payload["contact"]["care_email"] == "trusted@example.org"
    assert payload["actions"]["tel"] == "tel:0222223333"
    detail = urlparse(payload["actions"]["detail_url"])
    assert detail.hostname == "www.animal.go.kr"
    assert "evil.example" not in response.text
    assert "010-9999-9999" not in response.text


def test_adoption_contact_card_drops_unmatched_client_contact_injection(
    client: TestClient,
    main_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        main_module,
        "fetch_latest_notice_meta",
        lambda _meta: (None, "not_found"),
    )
    response = client.post(
        "/adoption/contact_card",
        headers=API_HEADERS,
        json={
            "animal": {
                "desertionNo": "unmatched-contact-1",
                "care_name": "가짜보호소",
                "care_tel": "010-9999-9999",
                "care_email": "attacker@example.net",
                "care_addr": "가짜 주소",
                "detail_url": "https://evil.example/client-phishing",
            }
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["contact_source"] == "unavailable"
    assert payload["direct_contact_allowed"] is False
    assert payload["actions"]["tel"] == ""
    assert payload["actions"]["mailto"] == ""
    assert payload["actions"]["map_url"] == ""
    assert payload["contact"]["care_name"] == ""
    assert payload["contact"]["care_tel"] == ""
    assert urlparse(payload["actions"]["detail_url"]).hostname == "www.animal.go.kr"
    assert "evil.example" not in response.text
    assert "010-9999-9999" not in response.text


@pytest.mark.parametrize(
    ("process_state", "expected_status", "expected_reason"),
    [
        ("입양완료", "closed", "notice_closed"),
        ("보호중", "expired", "notice_expired"),
    ],
)
def test_adoption_contact_card_blocks_direct_contact_for_inactive_notice(
    process_state: str,
    expected_status: str,
    expected_reason: str,
    client: TestClient,
    main_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notice_end = "20991231" if expected_status == "closed" else "20200101"
    monkeypatch.setattr(
        main_module,
        "fetch_latest_notice_meta",
        lambda meta: (
            {
                **meta,
                "process_state": process_state,
                "notice_end": notice_end,
                "care_name": "테스트보호소",
                "care_tel": "02-1234-5678",
                "care_email": "shelter@example.org",
            },
            "found",
        ),
    )
    response = client.post(
        "/adoption/contact_card",
        headers=API_HEADERS,
        json={"desertion_no": f"inactive-{expected_status}"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["notice_status"] == expected_status
    assert payload["connectable"] is False
    assert payload["direct_contact_allowed"] is False
    assert payload["direct_contact_block_reason"] == expected_reason
    assert payload["actions"]["tel"] == ""
    assert payload["actions"]["mailto"] == ""
    assert payload["contact_availability"]["phone"] is False
    assert payload["contact_availability"]["email"] is False
    assert payload["email_body"]


def test_adoption_contact_card_blocks_direct_contact_when_status_is_unknown(
    client: TestClient,
    main_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        main_module,
        "fetch_latest_notice_meta",
        lambda meta: (
            {
                **meta,
                "care_name": "테스트보호소",
                "care_tel": "02-1234-5678",
                "care_email": "shelter@example.org",
            },
            "found",
        ),
    )
    response = client.post(
        "/adoption/contact_card",
        headers=API_HEADERS,
        json={"desertion_no": "unknown-status-contact"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["notice_status"] == "unknown"
    assert payload["direct_contact_allowed"] is False
    assert payload["direct_contact_block_reason"] == "notice_status_unknown"
    assert payload["actions"]["tel"] == ""
    assert payload["actions"]["mailto"] == ""


@pytest.mark.parametrize(
    "invalid_identifier",
    ["Unknown", "x" * 101, "https://evil.example/id", {"nested": "value"}],
)
def test_adoption_contact_card_rejects_invalid_notice_identifier(
    invalid_identifier: Any,
    client: TestClient,
) -> None:
    response = client.post(
        "/adoption/contact_card",
        headers=API_HEADERS,
        json={"animal": {"desertionNo": invalid_identifier}},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "desertionNo is required"}


def test_latest_notice_lookup_has_a_bounded_request_timeout(
    main_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_settings: list[int] = []

    class TimeoutSession:
        def get(self, *_args: Any, **kwargs: Any) -> None:
            assert 0 < kwargs["timeout"] <= 0.5
            raise main_module.requests.Timeout("slow upstream")

    def build_timeout_session(*, retry_total: int = 5) -> TimeoutSession:
        session_settings.append(retry_total)
        return TimeoutSession()

    monkeypatch.setattr(main_module, "ANIMAL_API_KEY", "test-key")
    monkeypatch.setattr(main_module, "build_api_session", build_timeout_session)
    main_module.NOTICE_LOOKUP_CACHE.clear()

    latest, status = main_module.fetch_latest_notice_meta(
        {"desertionNo": "slow-notice", "species": "dog"},
        timeout_seconds=0.5,
    )

    assert latest is None
    assert status == "timeout"
    assert session_settings == [0]


def test_recommendation_falls_back_when_gemma_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
    main_module: Any,
) -> None:
    monkeypatch.setattr(main_module, "GEMMA3_ENABLED", False)

    def unexpected_generation(*_args: Any, **_kwargs: Any) -> str:
        raise AssertionError("disabled Gemma must not be loaded")

    monkeypatch.setattr(main_module, "gemma_generate_text", unexpected_generation)
    recommendation = main_module.gemma_recommend(
        profile_text="small dog",
        candidates=[{"score": 0.8}],
    )

    assert "상위 1건" in recommendation
    assert main_module.PROFILE_RESULT_DISCLAIMER in recommendation


def test_recommendation_falls_back_when_optional_gemma_runtime_fails(
    monkeypatch: pytest.MonkeyPatch,
    main_module: Any,
) -> None:
    monkeypatch.setattr(main_module, "GEMMA3_ENABLED", True)

    def failed_generation(*_args: Any, **_kwargs: Any) -> str:
        raise RuntimeError("optional model unavailable")

    monkeypatch.setattr(main_module, "gemma_generate_text", failed_generation)
    recommendation = main_module.gemma_recommend(
        profile_text="small dog",
        candidates=[{"score": 0.8}],
    )

    assert "상위 1건" in recommendation
    assert main_module.PROFILE_RESULT_DISCLAIMER in recommendation


def test_hybrid_formatter_keeps_v2_photo_and_separates_vlm_behavior(
    main_module: Any,
) -> None:
    formatted = main_module.format_hybrid_result(
        {
            "score": 0.5,
            "doc": {
                "meta": {
                    "desertionNo": "v2-dog",
                    "kindCd": "000114",
                    "kindNm": "믹스견",
                    "popfile1": "https://example.test/v2-dog.jpg",
                    "specialMark": "빨간 목줄을 착용한 상태로 발견",
                    "vlm_desc": "활동량 많음, 아이와 생활 가능으로 보임",
                }
            },
        },
        rank=1,
    )

    assert formatted["image_url"] == "https://example.test/v2-dog.jpg"
    assert formatted["notice_behavior_text"] == "빨간 목줄을 착용한 상태로 발견"
    dog = normalize_dog(formatted)
    assert dog.activity_level_hint is None
    assert dog.children_compatible is None


def test_profile_demo_route_serves_the_profile_search_ui(client: TestClient) -> None:
    response = client.get("/demo")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert 'id="heroTitle"' in response.text
    assert "마음에 그린 모습을" in response.text
    assert "보호소 문의" in response.text
    assert "fetch('/search/profile'" in response.text
    assert "ranking_scope:'appearance'" in response.text
    assert "fetch('/adoption/contact_card'" in response.text
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
