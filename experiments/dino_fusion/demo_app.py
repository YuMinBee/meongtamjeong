"""Local visual comparison for CLIP and DINO-aligned multimodal retrieval."""

from __future__ import annotations

import base64
import binascii
import io
import json
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field

from app.notice_metadata import normalize_breed_fields
from experiments.dino_fusion.alignment import (
    blend_embeddings,
    project_embeddings,
    projection_head_from_checkpoint,
)
from experiments.dino_fusion.build_index import sha256_file
from experiments.dino_fusion.core import (
    ClipEncoder,
    DinoEncoder,
    clean_text,
    collapse_visual_hits,
    safe_read_faiss_index,
)


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEMO_HTML = EXPERIMENT_DIR / "demo.html"
CLIP_INDEX = ROOT / "data" / "dog_faiss.index"
CLIP_METAS = ROOT / "data" / "dog_metas.json"
DINO_DIR = EXPERIMENT_DIR / "artifacts" / "dinov3"
FLOW_DIR = EXPERIMENT_DIR / "artifacts" / "dinode_flow"
MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000


class CompareRequest(BaseModel):
    image_data_url: str | None = Field(default=None, max_length=17_000_000)
    text: str = Field(default="", max_length=500)
    text_weight: float = Field(default=0.2, ge=0.0, le=1.0)
    topk: int = Field(default=5, ge=1, le=10)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def decode_image_data_url(value: str) -> Image.Image:
    header, separator, payload = clean_text(value).partition(",")
    if not separator or not header.startswith("data:image/") or ";base64" not in header:
        raise ValueError("지원되는 이미지 파일을 선택해 주세요.")
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("이미지 데이터를 읽을 수 없습니다.") from exc
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("이미지는 12MB 이하만 사용할 수 있습니다.")
    try:
        with Image.open(io.BytesIO(raw)) as source:
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise ValueError("이미지 해상도가 너무 큽니다.")
            return ImageOps.exif_transpose(source).convert("RGB").copy()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("이미지 파일 형식을 확인해 주세요.") from exc


class ComparisonEngine:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._state = "not_loaded"
        self._error = ""
        self._device = ""
        self._clip_encoder: Any = None
        self._dino_encoder: Any = None
        self._head: Any = None
        self._torch: Any = None
        self._clip_index: Any = None
        self._dino_index: Any = None
        self._clip_metas: list[dict[str, Any]] = []
        self._dino_metas: list[dict[str, Any]] = []
        self._clip_crop_indices: set[int] = set()
        self._crop_paths: dict[str, Path] = {}
        self._model_details: dict[str, Any] = {}

    def status(self) -> dict[str, Any]:
        return {
            "state": self._state,
            "ready": self._state == "ready",
            "device": self._device or None,
            "error": self._error or None,
        }

    def ensure_loaded(self) -> None:
        with self._lock:
            if self._state == "ready":
                return
            self._state = "loading"
            self._error = ""
            try:
                self._load()
            except Exception as exc:
                self._state = "error"
                self._error = f"{type(exc).__name__}: {exc}"
                raise
            self._state = "ready"

    def _load(self) -> None:
        torch = __import__("torch")
        faiss = __import__("faiss")
        device = "cuda" if bool(torch.cuda.is_available()) else "cpu"

        clip_metas = load_json(CLIP_METAS)
        dino_metas = load_json(DINO_DIR / "dino_metas.json")
        if not isinstance(clip_metas, list) or not isinstance(dino_metas, list):
            raise ValueError("검색 메타데이터 형식이 올바르지 않습니다.")
        clip_index = safe_read_faiss_index(CLIP_INDEX, faiss.read_index)
        dino_index = safe_read_faiss_index(DINO_DIR / "dino.index", faiss.read_index)
        if int(clip_index.ntotal) != len(clip_metas):
            raise ValueError("CLIP 인덱스와 메타데이터 행 수가 다릅니다.")
        if int(dino_index.ntotal) != len(dino_metas):
            raise ValueError("DINO 인덱스와 메타데이터 행 수가 다릅니다.")

        manifest = load_json(DINO_DIR / "dino_manifest.json")
        model = manifest.get("model") or {}
        model_id = clean_text(model.get("id"))
        revision = clean_text(model.get("resolved_revision")) or None

        training_report = load_json(FLOW_DIR / "alignment_training_report.json")
        flow_details = (training_report.get("architectures") or {}).get("flow") or {}
        checkpoint_path = FLOW_DIR / clean_text(flow_details.get("checkpoint"))
        expected_hash = clean_text(flow_details.get("checkpoint_sha256"))
        if (
            not checkpoint_path.is_file()
            or sha256_file(checkpoint_path) != expected_hash
        ):
            raise ValueError("학습된 flow 체크포인트 검증에 실패했습니다.")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        head = projection_head_from_checkpoint(checkpoint).to(device)
        head.load_state_dict(checkpoint["state_dict"])
        head.eval()

        clip_crop_indices: set[int] = set()
        crop_paths: dict[str, Path] = {}
        for row in dino_metas:
            source_index = row.get("source_meta_index")
            if not isinstance(source_index, int):
                raise ValueError("DINO 메타데이터에 원본 행 번호가 없습니다.")
            clip_crop_indices.add(source_index)
            notice_id = clean_text(row.get("notice_id") or row.get("desertionNo"))
            source_ref = clean_text(row.get("source_ref"))
            path = (ROOT / source_ref).resolve()
            if notice_id and path.is_file():
                crop_paths[notice_id] = path

        self._torch = torch
        self._device = device
        self._clip_index = clip_index
        self._dino_index = dino_index
        self._clip_metas = [dict(row) for row in clip_metas]
        self._dino_metas = [dict(row) for row in dino_metas]
        self._clip_crop_indices = clip_crop_indices
        self._crop_paths = crop_paths
        self._head = head
        self._clip_encoder = ClipEncoder("ViT-B/32", device=device)
        self._dino_encoder = DinoEncoder(
            model_id,
            device=device,
            revision=revision,
            local_files_only=True,
        )
        self._model_details = {
            "clip": "ViT-B/32",
            "dino": model_id,
            "alignment": "flow",
            "candidates": len(dino_metas),
        }

    def image_path(self, notice_id: str) -> Path | None:
        self.ensure_loaded()
        return self._crop_paths.get(clean_text(notice_id))

    def _clip_ranking(self, vector: Any) -> list[dict[str, Any]]:
        distances, indices = self._clip_index.search(
            vector, int(self._clip_index.ntotal)
        )
        pairs = [
            (score, index)
            for score, index in zip(distances[0].tolist(), indices[0].tolist())
            if int(index) in self._clip_crop_indices
        ]
        return collapse_visual_hits(
            [score for score, _ in pairs],
            [index for _, index in pairs],
            self._clip_metas,
            score_kind="distance",
        )

    def _dino_ranking(self, vector: Any) -> list[dict[str, Any]]:
        scores, indices = self._dino_index.search(vector, int(self._dino_index.ntotal))
        return collapse_visual_hits(
            scores[0].tolist(),
            indices[0].tolist(),
            self._dino_metas,
            score_kind="similarity",
        )

    def _decorate(
        self,
        ranking: list[dict[str, Any]],
        *,
        system: str,
        topk: int,
    ) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for rank, result in enumerate(ranking[:topk], start=1):
            vector_index = int(result["vector_index"])
            if system == "dino":
                source_index = self._dino_metas[vector_index]["source_meta_index"]
            else:
                source_index = vector_index
            meta = self._clip_metas[int(source_index)]
            breed = normalize_breed_fields(meta)
            notice_id = clean_text(result["notice_id"])
            output.append(
                {
                    "rank": rank,
                    "notice_id": notice_id,
                    "score": round(float(result["score"]), 6),
                    "best_modality": clean_text(result["best_modality"]),
                    "image_url": f"/api/images/{notice_id}",
                    "breed": breed["breed"],
                    "color": clean_text(meta.get("color") or meta.get("colorCd")),
                    "age": clean_text(meta.get("age") or meta.get("ageCd")),
                    "weight": clean_text(meta.get("weight")),
                    "sex": clean_text(meta.get("sex") or meta.get("sexCd")),
                    "region": clean_text(meta.get("orgNm") or meta.get("region")),
                }
            )
        return output

    def compare(
        self,
        *,
        image: Image.Image | None,
        text: str,
        text_weight: float,
        topk: int,
    ) -> dict[str, Any]:
        with self._lock:
            self.ensure_loaded()
            query_text = clean_text(text)
            if image is None and not query_text:
                raise ValueError("이미지나 검색 문장 중 하나는 입력해 주세요.")

            started = time.perf_counter()
            clip_text = None
            mapped_text = None
            if query_text:
                clip_text = self._clip_encoder.encode_text(query_text)
                with self._torch.inference_mode():
                    mapped_text = (
                        project_embeddings(
                            self._head,
                            self._torch.as_tensor(clip_text, device=self._device),
                        )
                        .detach()
                        .float()
                        .cpu()
                        .numpy()
                    )

            clip_image = self._clip_encoder.encode_image(image) if image else None
            dino_image = self._dino_encoder.encode(image) if image else None
            encoded_ms = (time.perf_counter() - started) * 1000

            if clip_image is not None and clip_text is not None:
                clip_query = blend_embeddings(
                    clip_image, clip_text, text_weight=text_weight
                )
            else:
                clip_query = clip_image if clip_image is not None else clip_text

            if dino_image is not None and mapped_text is not None:
                dino_query = blend_embeddings(
                    dino_image, mapped_text, text_weight=text_weight
                )
            else:
                dino_query = dino_image if dino_image is not None else mapped_text

            searched = time.perf_counter()
            clip_ranking = self._clip_ranking(clip_query)
            dino_ranking = self._dino_ranking(dino_query)
            search_ms = (time.perf_counter() - searched) * 1000

            mode = (
                "image_text"
                if image is not None and query_text
                else ("image" if image is not None else "text")
            )
            return {
                "query": {
                    "text": query_text,
                    "mode": mode,
                    "text_weight": text_weight if mode == "image_text" else None,
                },
                "models": self._model_details,
                "latency_ms": {
                    "encoding": round(encoded_ms, 1),
                    "search": round(search_ms, 1),
                    "total": round(encoded_ms + search_ms, 1),
                },
                "clip": self._decorate(clip_ranking, system="clip", topk=topk),
                "dino": self._decorate(dino_ranking, system="dino", topk=topk),
            }


engine = ComparisonEngine()
app = FastAPI(title="DINO + CLIP comparison", docs_url=None, redoc_url=None)


@app.get("/", response_class=HTMLResponse)
def demo_page() -> HTMLResponse:
    return HTMLResponse(DEMO_HTML.read_text(encoding="utf-8"))


@app.get("/api/status")
def status() -> dict[str, Any]:
    return engine.status()


@app.get("/api/images/{notice_id}")
def result_image(notice_id: str) -> FileResponse:
    path = engine.image_path(notice_id)
    if path is None:
        raise HTTPException(status_code=404, detail="검색 결과 이미지가 없습니다.")
    return FileResponse(path, media_type="image/jpeg")


@app.post("/api/compare")
def compare(payload: CompareRequest) -> dict[str, Any]:
    image: Image.Image | None = None
    try:
        if payload.image_data_url:
            image = decode_image_data_url(payload.image_data_url)
        return engine.compare(
            image=image,
            text=payload.text,
            text_weight=payload.text_weight,
            topk=payload.topk,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if image is not None:
            image.close()
