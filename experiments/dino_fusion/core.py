"""Core encoders and rank fusion for the training-free DINO experiment.

Heavy model libraries are imported lazily so score-fusion tests can run in the
repository's lightweight test environment.
"""

from __future__ import annotations

import importlib
import math
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np


DEFAULT_DINO_MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
VISUAL_MODALITIES = frozenset({"full_image", "crop_image"})


def clean_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def notice_id(meta: Mapping[str, Any]) -> str:
    return clean_text(meta.get("desertionNo") or meta.get("desertion_no"))


def vector_modality(meta: Mapping[str, Any]) -> str:
    row_type = clean_text(meta.get("type"))
    source = clean_text(meta.get("embedding_source"))
    if row_type == "text":
        return "text"
    if row_type == "crop_image" or source in {
        "dog_crop",
        "animal_crop",
        "crop_image",
    }:
        return "crop_image"
    if row_type == "image" or source in {"full_image", "image"}:
        return "full_image"
    return source or row_type or "unknown"


def normalize_rows(values: Any) -> np.ndarray:
    """Return finite, row-wise L2-normalized float32 vectors."""

    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] <= 0:
        raise ValueError("embeddings must be a non-empty 2-dimensional matrix")
    if not np.isfinite(array).all():
        raise ValueError("embeddings contain non-finite values")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("embeddings contain a zero-norm row")
    return np.ascontiguousarray(array / norms, dtype=np.float32)


def collapse_visual_hits(
    raw_scores: Sequence[float],
    vector_indices: Sequence[int],
    metas: Sequence[Mapping[str, Any]],
    *,
    score_kind: str,
) -> list[dict[str, Any]]:
    """Collapse vector hits to the best visual hit for each notice.

    ``score_kind`` is ``similarity`` for inner product/cosine scores and
    ``distance`` for L2 distances. Returned scores are always higher-is-better.
    """

    if len(raw_scores) != len(vector_indices):
        raise ValueError("score/index length mismatch")
    if score_kind not in {"similarity", "distance"}:
        raise ValueError("score_kind must be 'similarity' or 'distance'")

    best: dict[str, dict[str, Any]] = {}
    for raw_score, raw_index in zip(raw_scores, vector_indices):
        vector_index = int(raw_index)
        if vector_index < 0:
            continue
        if vector_index >= len(metas):
            raise ValueError(f"vector index out of range: {vector_index}")
        score_value = float(raw_score)
        if not math.isfinite(score_value):
            raise ValueError("non-finite vector score")
        if score_kind == "distance":
            if score_value < 0:
                raise ValueError("distance must be non-negative")
            score = 1.0 / (1.0 + score_value)
        else:
            score = score_value

        meta = metas[vector_index]
        dog_id = notice_id(meta)
        modality = vector_modality(meta)
        if not dog_id or modality not in VISUAL_MODALITIES:
            continue
        current = best.get(dog_id)
        if current is None or score > float(current["score"]):
            best[dog_id] = {
                "notice_id": dog_id,
                "score": score,
                "best_modality": modality,
                "vector_index": vector_index,
            }

    return sorted(
        best.values(),
        key=lambda row: (-float(row["score"]), clean_text(row["notice_id"])),
    )


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    weights: Mapping[str, float] | None = None,
    rrf_k: int = 60,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Fuse heterogeneous rankings without assuming comparable score scales."""

    if int(rrf_k) < 0:
        raise ValueError("rrf_k must be non-negative")
    if limit is not None and int(limit) <= 0:
        raise ValueError("limit must be positive")
    resolved_weights = dict(weights or {})
    fused: dict[str, dict[str, Any]] = {}

    for system, rows in rankings.items():
        weight = float(resolved_weights.get(system, 1.0))
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"invalid fusion weight for {system!r}")
        if weight == 0:
            continue
        seen: set[str] = set()
        for rank, row in enumerate(rows, start=1):
            dog_id = clean_text(row.get("notice_id") or row.get("desertionNo"))
            if not dog_id or dog_id in seen:
                continue
            seen.add(dog_id)
            contribution = weight / (int(rrf_k) + rank)
            item = fused.setdefault(
                dog_id,
                {"notice_id": dog_id, "fusion_score": 0.0, "sources": {}},
            )
            item["fusion_score"] += contribution
            item["sources"][system] = {
                "rank": rank,
                "raw_score": float(row.get("score", 0.0)),
                "best_modality": clean_text(row.get("best_modality")),
                "contribution": contribution,
            }

    ranked = sorted(
        fused.values(),
        key=lambda row: (-float(row["fusion_score"]), clean_text(row["notice_id"])),
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked[: int(limit)] if limit is not None else ranked


def safe_read_faiss_index(path: Path, read_index: Any) -> Any:
    """Read a FAISS index even when the workspace path contains Unicode."""

    try:
        return read_index(str(path))
    except RuntimeError as exc:
        if "Illegal byte sequence" not in str(exc):
            raise
        safe_path = Path(tempfile.gettempdir()) / f"dino-read-{uuid4().hex}.index"
        try:
            shutil.copyfile(path, safe_path)
            return read_index(str(safe_path))
        finally:
            safe_path.unlink(missing_ok=True)


class DinoEncoder:
    """Frozen Hugging Face DINO image encoder using the global pooler output."""

    def __init__(
        self,
        model_id: str = DEFAULT_DINO_MODEL_ID,
        *,
        device: str = "auto",
        revision: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        resolved_device = (
            "cuda"
            if device == "auto" and bool(torch.cuda.is_available())
            else ("cpu" if device == "auto" else device)
        )
        load_options: dict[str, Any] = {"local_files_only": local_files_only}
        if revision:
            load_options["revision"] = revision
        processor = transformers.AutoImageProcessor.from_pretrained(
            model_id, **load_options
        )
        model = transformers.AutoModel.from_pretrained(model_id, **load_options)
        model.to(resolved_device)
        model.eval()

        self._torch = torch
        self._processor = processor
        self._model = model
        self.device = resolved_device
        self.model_id = model_id
        self.model_family = clean_text(getattr(model.config, "model_type", "")) or "dino"
        self.requested_revision = revision or "main"
        self.resolved_revision = clean_text(
            getattr(getattr(model, "config", None), "_commit_hash", "")
        )
        self.dimension = int(getattr(model.config, "hidden_size"))
        self.pooling = "pooler_output_with_cls_fallback"

    def encode_batch(self, images: Sequence[Any]) -> np.ndarray:
        if not images:
            return np.empty((0, self.dimension), dtype=np.float32)
        rgb_images = [image.convert("RGB") for image in images]
        inputs = self._processor(images=rgb_images, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.inference_mode():
            outputs = self._model(**inputs)
            vectors = getattr(outputs, "pooler_output", None)
            if vectors is None:
                vectors = outputs.last_hidden_state[:, 0]
        return normalize_rows(vectors.detach().float().cpu().numpy())

    def encode(self, image: Any) -> np.ndarray:
        return self.encode_batch([image])


# Compatibility name retained for callers that explicitly target DINOv3.
DinoV3Encoder = DinoEncoder


class ClipEncoder:
    """Frozen OpenAI CLIP towers matching the current production index."""

    def __init__(self, model_name: str = "ViT-B/32", *, device: str = "auto") -> None:
        clip = importlib.import_module("clip")
        torch = importlib.import_module("torch")
        resolved_device = (
            "cuda"
            if device == "auto" and bool(torch.cuda.is_available())
            else ("cpu" if device == "auto" else device)
        )
        model, preprocess = clip.load(model_name, device=resolved_device)
        model.eval()
        self._clip = clip
        self._torch = torch
        self._model = model
        self._preprocess = preprocess
        self.device = resolved_device
        self.model_name = model_name
        self.text_dimension = int(model.text_projection.shape[-1])

    def encode(self, image: Any) -> np.ndarray:
        return self.encode_image(image)

    def encode_image(self, image: Any) -> np.ndarray:
        tensor = self._preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        with self._torch.inference_mode():
            vector = self._model.encode_image(tensor)
        return normalize_rows(vector.detach().float().cpu().numpy())

    def encode_text(self, text: str) -> np.ndarray:
        return self.encode_text_batch([text])

    def encode_text_batch(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.text_dimension), dtype=np.float32)
        tokens = self._clip.tokenize(list(texts), truncate=True).to(self.device)
        with self._torch.inference_mode():
            vector = self._model.encode_text(tokens)
        return normalize_rows(vector.detach().float().cpu().numpy())


ClipImageEncoder = ClipEncoder
