"""Optional DINO visual retrieval for the production search pipeline.

The existing CLIP index remains the default.  This adapter promotes the
validated pilot artifacts behind an explicit ``off``/``shadow``/``active``
rollout switch and keeps all heavyweight model imports lazy.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image


RolloutMode = Literal["off", "shadow", "active"]
TextBatchEncoder = Callable[[Sequence[str]], np.ndarray]


def _clean_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _resolve_path(root: Path, value: str, default: Path) -> Path:
    path = Path(value.strip()) if value.strip() else default
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_weight(value: float, name: str) -> float:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between zero and one")
    return float(value)


@dataclass(frozen=True)
class DinoFusionSettings:
    """Configuration for an opt-in, artifact-backed DINO rollout."""

    mode: RolloutMode
    dino_dir: Path
    flow_dir: Path
    behavior_dir: Path
    text_weight: float = 0.2
    behavior_weight: float = 0.25
    behavior_enabled: bool = False
    local_files_only: bool = True
    fail_open: bool = True
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.mode not in {"off", "shadow", "active"}:
            raise ValueError("DINO_FUSION_MODE must be off, shadow, or active")
        _validate_weight(self.text_weight, "DINO_FUSION_TEXT_WEIGHT")
        _validate_weight(self.behavior_weight, "DINO_FUSION_BEHAVIOR_WEIGHT")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("DINO_FUSION_DEVICE must be auto, cpu, or cuda")

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @classmethod
    def from_env(cls, root: Path) -> "DinoFusionSettings":
        artifact_root = root / "experiments" / "dino_fusion" / "artifacts"
        mode = _clean_text(os.getenv("DINO_FUSION_MODE", "off")).lower()
        return cls(
            mode=mode,  # type: ignore[arg-type]
            dino_dir=_resolve_path(
                root,
                os.getenv("DINO_FUSION_DINO_DIR", ""),
                artifact_root / "dinov3",
            ),
            flow_dir=_resolve_path(
                root,
                os.getenv("DINO_FUSION_FLOW_DIR", ""),
                artifact_root / "dinode_flow",
            ),
            behavior_dir=_resolve_path(
                root,
                os.getenv("DINO_FUSION_BEHAVIOR_DIR", ""),
                artifact_root / "behavior_pilot",
            ),
            text_weight=_env_float("DINO_FUSION_TEXT_WEIGHT", 0.2),
            behavior_weight=_env_float("DINO_FUSION_BEHAVIOR_WEIGHT", 0.25),
            behavior_enabled=_env_bool("DINO_FUSION_BEHAVIOR_ENABLED", False),
            local_files_only=_env_bool("DINO_FUSION_LOCAL_FILES_ONLY", True),
            fail_open=_env_bool("DINO_FUSION_FAIL_OPEN", True),
            device=_clean_text(os.getenv("DINO_FUSION_DEVICE", "auto")).lower()
            or "auto",
        )


@dataclass(frozen=True)
class BehaviorPreference:
    axis: str
    axis_index: int
    desired_label: int


def behavior_preferences_from_text(text: str) -> tuple[BehaviorPreference, ...]:
    """Extract only explicitly requested ends of the trained behavior axes."""

    if not _clean_text(text):
        return ()
    # Importing this pilot helper is cheap; torch/transformers remain lazy.
    from experiments.dino_fusion.behavior import BEHAVIOR_AXES, weak_behavior_labels

    labels = weak_behavior_labels(text)
    output: list[BehaviorPreference] = []
    for axis_index, axis in enumerate(BEHAVIOR_AXES):
        desired = int((labels.get(axis.key) or {}).get("value") or 0)
        if desired in {-1, 1}:
            output.append(
                BehaviorPreference(
                    axis=axis.key,
                    axis_index=axis_index,
                    desired_label=desired,
                )
            )
    return tuple(output)


def percentile_scores(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Convert scores to deterministic [0, 1] ranks without changing order."""

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(array):
        return array
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    ranks[order] = np.arange(len(array), dtype=np.float64)
    return ranks / max(1, len(array) - 1)


def fuse_visual_behavior_scores(
    visual_scores: Sequence[float] | np.ndarray,
    behavior_scores: Sequence[float] | np.ndarray,
    *,
    behavior_weight: float,
) -> np.ndarray:
    """Apply the mixed-image pilot's fixed 75:25 visual/behavior contract."""

    weight = _validate_weight(behavior_weight, "behavior_weight")
    visual = np.asarray(visual_scores, dtype=np.float64).reshape(-1)
    behavior = np.asarray(behavior_scores, dtype=np.float64).reshape(-1)
    if visual.shape != behavior.shape:
        raise ValueError("visual and behavior scores must have the same shape")
    return (1.0 - weight) * percentile_scores(visual) + weight * behavior


@dataclass
class DinoFusionSearchResult:
    vector_scores: dict[int, float]
    vector_score_details: dict[int, dict[str, Any]]
    ranked_notice_ids: list[str]
    query_mode: str
    behavior_preferences: tuple[BehaviorPreference, ...]
    latency_ms: dict[str, float]
    model: dict[str, Any]

    def diagnostics(self, *, served: bool, topk: int = 10) -> dict[str, Any]:
        return {
            "served": served,
            "query_mode": self.query_mode,
            "text_weight": self.model.get("text_weight"),
            "behavior_weight": (
                self.model.get("behavior_weight") if self.behavior_preferences else None
            ),
            "behavior_preferences": [
                {
                    "axis": item.axis,
                    "desired_label": item.desired_label,
                }
                for item in self.behavior_preferences
            ],
            "behavior_disclaimer": (
                "공고 원문 기반 탐색 보조 점수이며 실제 성격이나 입양 적합도 판정이 아닙니다."
                if self.behavior_preferences
                else None
            ),
            "candidate_count": len(self.vector_scores),
            "top_notice_ids": self.ranked_notice_ids[: max(1, topk)],
            "latency_ms": self.latency_ms,
            "model": self.model,
        }


class DinoFusionRuntime:
    """Lazily load, validate, and query the frozen DINO fusion artifacts."""

    def __init__(self, settings: DinoFusionSettings) -> None:
        self.settings = settings
        self._lock = threading.RLock()
        self._state = "disabled" if not settings.enabled else "not_loaded"
        self._error = ""
        self._device = ""
        self._torch: Any = None
        self._index: Any = None
        self._metas: list[dict[str, Any]] = []
        self._dino_encoder: Any = None
        self._flow_head: Any = None
        self._behavior_head: Any = None
        self._behavior_records: list[dict[str, Any]] = []
        self._behavior_probabilities: np.ndarray | None = None
        self._manifest: dict[str, Any] = {}

    def status(self) -> dict[str, Any]:
        return {
            "mode": self.settings.mode,
            "enabled": self.settings.enabled,
            "state": self._state,
            "ready": self._state == "ready",
            "behavior_enabled": self.settings.behavior_enabled,
            "behavior_ready": self._behavior_probabilities is not None,
            "device": self._device or None,
            "error": self._error or None,
            "text_weight": self.settings.text_weight,
            "behavior_weight": self.settings.behavior_weight,
            "fail_open": self.settings.fail_open,
        }

    def record_error(self, exc: Exception) -> None:
        with self._lock:
            self._state = "error"
            self._error = f"{type(exc).__name__}: {exc}"

    def _verified_file(self, path: Path, expected_hash: str, label: str) -> Path:
        if not path.is_file():
            raise FileNotFoundError(f"missing {label}: {path}")
        if expected_hash and _sha256(path) != expected_hash.lower():
            raise ValueError(f"{label} SHA-256 mismatch: {path.name}")
        return path

    def ensure_loaded(self, *, source_metadata_path: Path) -> None:
        if not self.settings.enabled:
            raise RuntimeError("DINO fusion is disabled")
        with self._lock:
            if self._state == "ready":
                return
            self._state = "loading"
            self._error = ""
            try:
                self._load(source_metadata_path=source_metadata_path)
            except Exception as exc:
                self._state = "error"
                self._error = f"{type(exc).__name__}: {exc}"
                raise
            self._state = "ready"

    def _load(self, *, source_metadata_path: Path) -> None:
        import faiss
        import torch

        from experiments.dino_fusion.alignment import (
            projection_head_from_checkpoint,
        )
        from experiments.dino_fusion.core import DinoEncoder, safe_read_faiss_index

        manifest_path = self.settings.dino_dir / "dino_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing DINO manifest: {manifest_path}")
        manifest = _load_json(manifest_path)
        if manifest.get("schema_version") != "dino-fusion-index.v1":
            raise ValueError("unsupported DINO manifest schema")

        source = manifest.get("source") or {}
        expected_source_hash = _clean_text(source.get("metadata_sha256")).lower()
        self._verified_file(
            source_metadata_path,
            expected_source_hash,
            "DINO source metadata",
        )

        output = manifest.get("output") or {}
        index_path = self._verified_file(
            self.settings.dino_dir / _clean_text(output.get("index") or "dino.index"),
            _clean_text(output.get("index_sha256")),
            "DINO index",
        )
        metas_path = self._verified_file(
            self.settings.dino_dir
            / _clean_text(output.get("metas") or "dino_metas.json"),
            _clean_text(output.get("metas_sha256")),
            "DINO metadata",
        )
        metas = _load_json(metas_path)
        if not isinstance(metas, list) or not all(
            isinstance(row, Mapping) for row in metas
        ):
            raise ValueError("DINO metadata must be a JSON list of objects")
        index = safe_read_faiss_index(index_path, faiss.read_index)
        if int(index.ntotal) != len(metas):
            raise ValueError("DINO index/metadata row mismatch")
        if int(output.get("vectors") or -1) != len(metas):
            raise ValueError("DINO manifest vector count is stale")

        model = manifest.get("model") or {}
        model_dimension = int(model.get("dimension") or 0)
        if model_dimension <= 0 or int(index.d) != model_dimension:
            raise ValueError("DINO manifest/index dimension mismatch")

        report_path = self.settings.flow_dir / "alignment_training_report.json"
        report = _load_json(report_path)
        report_encoder = (report.get("encoders") or {}).get("dino_visual") or {}
        for key in ("id", "resolved_revision"):
            if _clean_text(report_encoder.get(key)) != _clean_text(model.get(key)):
                raise ValueError(f"DINO flow report model {key} mismatch")
        flow = (report.get("architectures") or {}).get("flow") or {}
        checkpoint_path = self._verified_file(
            self.settings.flow_dir / _clean_text(flow.get("checkpoint")),
            _clean_text(flow.get("checkpoint_sha256")),
            "DINO flow checkpoint",
        )

        device = (
            "cuda"
            if self.settings.device == "auto" and bool(torch.cuda.is_available())
            else ("cpu" if self.settings.device == "auto" else self.settings.device)
        )
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if int(checkpoint.get("input_dim") or 0) != 512:
            raise ValueError("DINO flow checkpoint CLIP dimension mismatch")
        if int(checkpoint.get("output_dim") or 0) != int(index.d):
            raise ValueError("DINO flow checkpoint output dimension mismatch")
        flow_head = projection_head_from_checkpoint(checkpoint).to(device)
        flow_head.load_state_dict(checkpoint["state_dict"])
        flow_head.eval()

        model_id = _clean_text(model.get("id"))
        if not model_id:
            raise ValueError("DINO model id is missing from the manifest")
        dino_encoder = DinoEncoder(
            model_id,
            device=device,
            revision=_clean_text(model.get("resolved_revision")) or None,
            local_files_only=self.settings.local_files_only,
        )

        self._torch = torch
        self._device = device
        self._index = index
        self._metas = [dict(row) for row in metas]
        self._dino_encoder = dino_encoder
        self._flow_head = flow_head
        self._manifest = dict(manifest)

    def _prepare_behavior(
        self,
        *,
        clip_metas: Sequence[Mapping[str, Any]],
        encode_text_batch: TextBatchEncoder,
    ) -> None:
        if self._behavior_probabilities is not None:
            return
        if not self.settings.behavior_enabled:
            return

        from experiments.dino_fusion.behavior import (
            build_behavior_records,
            make_behavior_head,
        )

        report_path = self.settings.behavior_dir / "behavior_pilot_report.json"
        report = _load_json(report_path)
        if int((report.get("dataset") or {}).get("record_count") or -1) != len(
            self._metas
        ):
            raise ValueError("behavior report record count is stale")
        report_dino = (report.get("models") or {}).get("dino") or {}
        manifest_dino = self._manifest.get("model") or {}
        for key in ("id", "resolved_revision"):
            if _clean_text(report_dino.get(key)) != _clean_text(manifest_dino.get(key)):
                raise ValueError(f"behavior report DINO {key} mismatch")
        training = report.get("training") or {}
        checkpoint_path = self._verified_file(
            self.settings.behavior_dir / _clean_text(training.get("checkpoint")),
            _clean_text(training.get("checkpoint_sha256")),
            "behavior checkpoint",
        )
        checkpoint = self._torch.load(
            checkpoint_path,
            map_location=self._device,
            weights_only=True,
        )
        axis_keys = tuple(checkpoint.get("axis_keys") or ())
        expected_axis_keys = tuple(
            item.get("key") for item in (report.get("axes") or [])
        )
        if not axis_keys or axis_keys != expected_axis_keys:
            raise ValueError("behavior checkpoint/report axes do not match")

        records = build_behavior_records(clip_metas, self._metas)
        if len(records) != len(self._metas):
            raise ValueError("behavior record coverage does not match DINO metadata")
        if any(
            int(record.get("dino_row", -1)) != row for row, record in enumerate(records)
        ):
            raise ValueError("behavior records are not aligned with DINO rows")
        texts = [_clean_text(record.get("behavior_text")) for record in records]
        embeddings = np.asarray(encode_text_batch(texts), dtype=np.float32)
        input_dim = int(checkpoint.get("input_dim") or 0)
        if embeddings.shape != (len(records), input_dim):
            raise ValueError("behavior text embedding shape mismatch")

        head = make_behavior_head(
            input_dim=input_dim,
            hidden_dim=int(checkpoint.get("hidden_dim") or 128),
            output_dim=len(axis_keys),
        ).to(self._device)
        head.load_state_dict(checkpoint["state_dict"])
        head.eval()
        with self._torch.inference_mode():
            logits = head(self._torch.as_tensor(embeddings, device=self._device))
            probabilities = self._torch.sigmoid(logits).detach().float().cpu().numpy()
        self._behavior_head = head
        self._behavior_records = records
        self._behavior_probabilities = probabilities

    def search(
        self,
        *,
        image: Image.Image,
        aligned_clip_text: np.ndarray | None,
        behavior_query: str,
        doc_index_by_notice: Mapping[str, int],
        source_metadata_path: Path,
        clip_metas: Sequence[Mapping[str, Any]],
        encode_text_batch: TextBatchEncoder,
    ) -> DinoFusionSearchResult:
        """Return DINO-space scores keyed by the application's document index."""

        with self._lock:
            total_started = time.perf_counter()
            self.ensure_loaded(source_metadata_path=source_metadata_path)
            preferences = (
                behavior_preferences_from_text(behavior_query)
                if self.settings.behavior_enabled
                else ()
            )

            encode_started = time.perf_counter()
            dino_image = self._dino_encoder.encode(image)
            mapped_text: np.ndarray | None = None
            if aligned_clip_text is not None:
                clip_array = np.asarray(aligned_clip_text, dtype=np.float32)
                if clip_array.ndim == 1:
                    clip_array = clip_array.reshape(1, -1)
                with self._torch.inference_mode():
                    mapped_text = (
                        self._torch.nn.functional.normalize(
                            self._flow_head(
                                self._torch.as_tensor(
                                    clip_array,
                                    device=self._device,
                                )
                            ),
                            dim=-1,
                        )
                        .detach()
                        .float()
                        .cpu()
                        .numpy()
                    )
            query_vector = dino_image
            if mapped_text is not None:
                from experiments.dino_fusion.alignment import blend_embeddings

                query_vector = blend_embeddings(
                    dino_image,
                    mapped_text,
                    text_weight=self.settings.text_weight,
                )
            encode_ms = (time.perf_counter() - encode_started) * 1000

            search_started = time.perf_counter()
            raw_scores, raw_indices = self._index.search(
                np.asarray(query_vector, dtype=np.float32),
                int(self._index.ntotal),
            )
            rows: list[dict[str, Any]] = []
            seen: set[str] = set()
            for raw_score, vector_index in zip(
                raw_scores[0].tolist(), raw_indices[0].tolist()
            ):
                if vector_index < 0 or vector_index >= len(self._metas):
                    continue
                meta = self._metas[vector_index]
                notice_id = _clean_text(
                    meta.get("notice_id") or meta.get("desertionNo")
                )
                doc_index = doc_index_by_notice.get(notice_id)
                if not notice_id or notice_id in seen or doc_index is None:
                    continue
                seen.add(notice_id)
                rows.append(
                    {
                        "notice_id": notice_id,
                        "doc_index": int(doc_index),
                        "vector_index": int(vector_index),
                        "raw_similarity": float(raw_score),
                    }
                )
            if not rows:
                raise ValueError(
                    "DINO index produced no candidates for active metadata"
                )

            visual = np.asarray(
                [row["raw_similarity"] for row in rows], dtype=np.float64
            )
            visual_percentiles = percentile_scores(visual)
            behavior_scores: np.ndarray | None = None
            behavior_axes_by_row: list[list[dict[str, Any]]] = [[] for _ in rows]
            if preferences and self.settings.behavior_enabled:
                self._prepare_behavior(
                    clip_metas=clip_metas,
                    encode_text_batch=encode_text_batch,
                )
                if self._behavior_probabilities is not None:
                    from experiments.dino_fusion.behavior import (
                        conservative_behavior_score,
                    )

                    combined: list[float] = []
                    for output_row, row in enumerate(rows):
                        source_row = int(row["vector_index"])
                        record = self._behavior_records[source_row]
                        axis_scores: list[float] = []
                        for preference in preferences:
                            probability = float(
                                self._behavior_probabilities[
                                    source_row, preference.axis_index
                                ]
                            )
                            explicit_label = int(
                                record["labels"][preference.axis_index]
                            )
                            score, state = conservative_behavior_score(
                                probability,
                                explicit_label,
                                desired_label=preference.desired_label,
                            )
                            axis_scores.append(float(score))
                            behavior_axes_by_row[output_row].append(
                                {
                                    "axis": preference.axis,
                                    "desired_label": preference.desired_label,
                                    "score": round(float(score), 6),
                                    "state": state,
                                }
                            )
                        combined.append(float(np.mean(axis_scores)))
                    behavior_scores = np.asarray(combined, dtype=np.float64)

            if behavior_scores is not None:
                final_scores = fuse_visual_behavior_scores(
                    visual,
                    behavior_scores,
                    behavior_weight=self.settings.behavior_weight,
                )
            else:
                final_scores = visual_percentiles

            vector_scores: dict[int, float] = {}
            vector_details: dict[int, dict[str, Any]] = {}
            for row_index, row in enumerate(rows):
                doc_index = int(row["doc_index"])
                score = float(final_scores[row_index])
                vector_scores[doc_index] = score
                evidence: dict[str, Any] = {
                    "dino_raw_similarity": round(float(row["raw_similarity"]), 6),
                    "dino_visual_percentile": round(
                        float(visual_percentiles[row_index]), 6
                    ),
                }
                if behavior_scores is not None:
                    evidence["behavior_score"] = round(
                        float(behavior_scores[row_index]), 6
                    )
                    evidence["behavior_axes"] = behavior_axes_by_row[row_index]
                    evidence["behavior_source"] = (
                        "public_notice_silver_label_with_trained_head"
                    )
                modality = (
                    "dino_image+aligned_clip_text"
                    if mapped_text is not None
                    else "dino_image"
                )
                vector_details[doc_index] = {
                    "modalities": {modality: score},
                    "best_vector_score": score,
                    "best_vector_index": int(row["vector_index"]),
                    "best_modality": modality,
                    "hit_count": 1,
                    "score": score,
                    "available_weight": 1.0,
                    "missing_modalities": [],
                    "evidence": evidence,
                }

            ranked = sorted(
                rows,
                key=lambda row: vector_scores[int(row["doc_index"])],
                reverse=True,
            )
            search_ms = (time.perf_counter() - search_started) * 1000
            total_ms = (time.perf_counter() - total_started) * 1000
            model = self._manifest.get("model") or {}
            return DinoFusionSearchResult(
                vector_scores=vector_scores,
                vector_score_details=vector_details,
                ranked_notice_ids=[str(row["notice_id"]) for row in ranked],
                query_mode=(
                    "dino_image+aligned_clip_text+behavior"
                    if behavior_scores is not None
                    else (
                        "dino_image+aligned_clip_text"
                        if mapped_text is not None
                        else "dino_image"
                    )
                ),
                behavior_preferences=preferences,
                latency_ms={
                    "encoding": round(encode_ms, 1),
                    "search_and_behavior": round(search_ms, 1),
                    "total": round(total_ms, 1),
                },
                model={
                    "visual_encoder": _clean_text(model.get("id")),
                    "visual_revision": _clean_text(model.get("resolved_revision")),
                    "text_encoder": "ViT-B/32",
                    "alignment": "flow",
                    "text_weight": self.settings.text_weight,
                    "behavior_weight": self.settings.behavior_weight,
                    "foundation_encoders_frozen": True,
                },
            )
