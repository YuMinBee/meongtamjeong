"""Evaluate visual-backbone, gallery-view, and appearance-text ablations.

This is deliberately separate from ``evaluate_retrieval.py`` so the completed
compatibility experiment and its cached report remain immutable.  Model/view
and text-weight choices are made on organization-disjoint validation dogs;
the test split is evaluated only with those frozen choices.
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import os
import time
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]

from experiments.dino_fusion.alignment import (  # noqa: E402
    blend_embeddings,
    projection_head_from_checkpoint,
)
from experiments.dino_fusion.core import (  # noqa: E402
    DEFAULT_DINO_MODEL_ID,
    ClipEncoder,
    DinoEncoder,
    normalize_rows,
)
from experiments.dino_fusion_external_eval.compatibility import (  # noqa: E402
    ndcg_at_k,
)
from experiments.dino_fusion_external_eval.prepare_benchmark import (  # noqa: E402
    DEFAULT_IMAGE_ZIP,
    DEFAULT_OUTPUT_DIR,
    sha256_file,
    write_json_atomic,
)


REPORT_SCHEMA_VERSION = "petfinder-dino-performance-ablation.v1"
CACHE_SCHEMA_VERSION = "petfinder-dino-performance-embeddings.v1"
FLOW_REPORT = (
    ROOT
    / "experiments"
    / "dino_fusion"
    / "artifacts"
    / "dinode_flow"
    / "alignment_training_report.json"
)
DINO_MODELS: tuple[tuple[str, str], ...] = (
    ("dinov2_base", "facebook/dinov2-base"),
    ("dinov3_vitb16", DEFAULT_DINO_MODEL_ID),
)
GALLERY_VIEWS = ("crop", "full", "multiview")
TEXT_WEIGHT_CANDIDATES = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
MAXIMUM_RECALL10_DROP = 0.02
MAXIMUM_MRR_DROP = 0.02
MINIMUM_APPEARANCE_NDCG_GAIN = 0.02
DETECTOR_MODEL = "fasterrcnn_mobilenet_v3_large_fpn"
DETECTOR_CONFIDENCE = 0.25
CROP_MARGIN = 0.08


COLOR_KO = {
    "Apricot / Beige": "살구색 또는 베이지색",
    "Bicolor": "두 가지 색",
    "Black": "검은색",
    "Brindle": "브린들 무늬",
    "Brown / Chocolate": "갈색 또는 초콜릿색",
    "Golden": "금색",
    "Gray / Blue / Silver": "회색 또는 은색",
    "Red / Chestnut / Orange": "적갈색 또는 주황색",
    "Tricolor (Brown, Black, & White)": "갈색·검은색·흰색 삼색",
    "White / Cream": "흰색 또는 크림색",
    "Yellow / Tan / Blond / Fawn": "황갈색",
}
SIZE_KO = {
    "Small": "작은 체구",
    "Medium": "중간 체구",
    "Large": "큰 체구",
    "Extra Large": "매우 큰 체구",
}
AGE_KO = {
    "Baby": "어린 강아지",
    "Young": "어린 성견",
    "Adult": "성견",
    "Senior": "노령견",
}
COAT_KO = {
    "Short": "짧은 털",
    "Medium": "중간 길이 털",
    "Long": "긴 털",
}
APPEARANCE_FIELDS = ("color_primary", "size", "age", "coat")


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_npz_atomic(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{uuid4().hex}.npz")
    try:
        np.savez(temporary, **arrays)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def appearance_values(record: Mapping[str, Any]) -> dict[str, str]:
    appearance = record.get("appearance") or {}
    if not isinstance(appearance, Mapping):
        return {field: "" for field in APPEARANCE_FIELDS}
    return {field: _clean(appearance.get(field)) for field in APPEARANCE_FIELDS}


def is_appearance_query(record: Mapping[str, Any]) -> bool:
    """Require color so the query is more discriminative than age/size alone."""

    return bool(appearance_values(record)["color_primary"])


def appearance_query_ko(record: Mapping[str, Any]) -> str:
    values = appearance_values(record)
    clauses = [
        SIZE_KO.get(values["size"], values["size"]),
        COLOR_KO.get(values["color_primary"], values["color_primary"]),
        COAT_KO.get(values["coat"], values["coat"]),
        AGE_KO.get(values["age"], values["age"]),
    ]
    description = ", ".join(value for value in clauses if value)
    return f"{description} 강아지를 찾아줘" if description else "강아지를 찾아줘"


def appearance_relevance(
    query_record: Mapping[str, Any], candidate_records: Sequence[Mapping[str, Any]]
) -> np.ndarray:
    query = appearance_values(query_record)
    active_fields = [field for field in APPEARANCE_FIELDS if query[field]]
    if not active_fields:
        raise ValueError("appearance query has no known fields")
    grades: list[float] = []
    for candidate_record in candidate_records:
        candidate = appearance_values(candidate_record)
        grades.append(
            float(
                sum(
                    bool(candidate[field]) and candidate[field] == query[field]
                    for field in active_fields
                )
            )
        )
    return np.asarray(grades, dtype=np.float64)


def gallery_scores(
    query_vectors: np.ndarray,
    *,
    gallery_full: np.ndarray,
    gallery_crop: np.ndarray,
    crop_available: np.ndarray,
    view: str,
) -> np.ndarray:
    """Score one candidate per dog while respecting missing detector crops."""

    if view not in GALLERY_VIEWS:
        raise ValueError(f"unsupported gallery view: {view}")
    query = normalize_rows(query_vectors)
    full = query @ normalize_rows(gallery_full).T
    if view == "full":
        return full
    available = np.asarray(crop_available, dtype=bool).reshape(-1)
    if len(available) != len(gallery_crop) or len(gallery_full) != len(gallery_crop):
        raise ValueError("gallery vectors and crop availability must align")
    crop = query @ normalize_rows(gallery_crop).T
    crop[:, ~available] = -np.inf
    if view == "crop":
        return crop
    return np.maximum(full, crop)


def ranking_metrics(
    scores: np.ndarray,
    *,
    records: Sequence[Mapping[str, Any]],
    reference_positions: Sequence[int],
) -> dict[str, Any]:
    matrix = np.asarray(scores, dtype=np.float64)
    references = [int(value) for value in reference_positions]
    if matrix.ndim != 2 or matrix.shape != (len(references), len(records)):
        raise ValueError("score matrix, references, and records must align")
    if not references:
        raise ValueError("at least one reference query is required")

    identity_ranks: list[int] = []
    appearance_ndcg: list[float] = []
    for query_row, reference in enumerate(references):
        if not 0 <= reference < len(records):
            raise ValueError("reference position is out of range")
        row = matrix[query_row]
        if np.isfinite(row[reference]):
            order = np.argsort(-row, kind="mergesort")
            identity_rank = int(np.flatnonzero(order == reference)[0]) + 1
        else:
            identity_rank = len(records) + 1
        identity_ranks.append(identity_rank)

        if is_appearance_query(records[reference]):
            # The reference dog is removed for appearance relevance.  This
            # mirrors a user photo that is not itself an adoption candidate.
            keep = np.ones(len(records), dtype=bool)
            keep[reference] = False
            relevance = appearance_relevance(records[reference], records)[keep]
            appearance_ndcg.append(ndcg_at_k(relevance, row[keep], k=10))

    ranks = np.asarray(identity_ranks, dtype=np.int64)
    return {
        "identity_query_count": len(identity_ranks),
        "appearance_query_count": len(appearance_ndcg),
        "recall@1": round(float(np.mean(ranks <= 1)), 6),
        "recall@5": round(float(np.mean(ranks <= 5)), 6),
        "recall@10": round(float(np.mean(ranks <= 10)), 6),
        "mrr": round(float(np.mean(1.0 / ranks)), 6),
        "appearance_ndcg@10": round(float(np.mean(appearance_ndcg)), 6),
    }


def select_gallery_view(metrics_by_view: Mapping[str, Mapping[str, Any]]) -> str:
    """Choose on validation only, preferring the existing crop path on ties."""

    missing = set(GALLERY_VIEWS) - set(metrics_by_view)
    if missing:
        raise ValueError(f"missing gallery views: {sorted(missing)}")
    tie_priority = {"crop": 2, "multiview": 1, "full": 0}
    return max(
        GALLERY_VIEWS,
        key=lambda view: (
            float(metrics_by_view[view]["recall@10"]),
            float(metrics_by_view[view]["mrr"]),
            float(metrics_by_view[view]["appearance_ndcg@10"]),
            tie_priority[view],
        ),
    )


def select_text_weight(
    sweep: Sequence[Mapping[str, Any]],
    *,
    maximum_recall10_drop: float = MAXIMUM_RECALL10_DROP,
    maximum_mrr_drop: float = MAXIMUM_MRR_DROP,
) -> float:
    if not sweep:
        raise ValueError("text-weight sweep cannot be empty")
    eligible = [
        row
        for row in sweep
        if float(row["recall10_drop"]) <= maximum_recall10_drop
        and float(row["mrr_drop"]) <= maximum_mrr_drop
    ]
    selected = max(
        eligible or list(sweep),
        key=lambda row: (
            float(row["appearance_ndcg@10"]),
            float(row["mrr"]),
            -float(row["text_weight"]),
        ),
    )
    return float(selected["text_weight"])


def _load_archive_images(
    archive: zipfile.ZipFile,
    records: Sequence[Mapping[str, Any]],
    *,
    photo_index: int,
) -> list[Image.Image]:
    images: list[Image.Image] = []
    for record in records:
        members = record.get("image_members") or []
        if len(members) <= photo_index:
            raise ValueError(f"pet {record.get('pet_id')} lacks photo {photo_index}")
        with Image.open(io.BytesIO(archive.read(str(members[photo_index])))) as source:
            images.append(source.convert("RGB"))
    return images


def _encode_batches(
    encoder: DinoEncoder, images: Sequence[Image.Image], *, batch_size: int = 32
) -> np.ndarray:
    batches = [
        encoder.encode_batch(images[start : start + batch_size])
        for start in range(0, len(images), batch_size)
    ]
    return np.concatenate(batches, axis=0).astype(np.float32, copy=False)


def _release_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _crop_gallery_images(
    images: Sequence[Image.Image], *, device: str
) -> tuple[list[Image.Image], np.ndarray, dict[str, Any]]:
    from scripts.enrich_image_crops import expand_bbox, load_torchvision_detector

    started = time.perf_counter()
    detector = load_torchvision_detector(DETECTOR_MODEL, device)
    initialization_seconds = time.perf_counter() - started
    crops: list[Image.Image] = []
    available: list[bool] = []
    detect_started = time.perf_counter()
    for image in images:
        detections = detector.detect(
            image, species="dog", conf=DETECTOR_CONFIDENCE
        )
        if detections:
            box = expand_bbox(
                detections[0].xyxy, image.width, image.height, CROP_MARGIN
            )
            crops.append(image.crop(box))
            available.append(True)
        else:
            # A placeholder keeps matrices aligned.  Its score is masked out.
            crops.append(image.copy())
            available.append(False)
    detection_seconds = time.perf_counter() - detect_started
    detector_device = detector.device
    del detector
    _release_cuda()
    return (
        crops,
        np.asarray(available, dtype=bool),
        {
            "model": DETECTOR_MODEL,
            "device": detector_device,
            "confidence": DETECTOR_CONFIDENCE,
            "crop_margin": CROP_MARGIN,
            "initialization_seconds": round(initialization_seconds, 6),
            "detection_seconds": round(detection_seconds, 6),
            "detected": int(sum(available)),
            "missing": int(len(available) - sum(available)),
            "coverage": round(float(np.mean(available)), 6),
        },
    )


def performance_embeddings(
    records: list[dict[str, Any]],
    *,
    image_zip: Path,
    output_dir: Path,
    device: str,
    local_files_only: bool,
    force: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    cache_path = output_dir / "performance_ablation_embeddings.npz"
    metadata_path = output_dir / "performance_ablation_embeddings.json"
    records_path = output_dir / "multimodal_records.json"
    pet_ids = np.asarray([str(record["pet_id"]) for record in records])
    expected = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "records_sha256": sha256_file(records_path),
        "image_zip_sha256": sha256_file(image_zip),
        "detector_model": DETECTOR_MODEL,
        "detector_confidence": DETECTOR_CONFIDENCE,
        "crop_margin": CROP_MARGIN,
        "model_ids": {name: model_id for name, model_id in DINO_MODELS},
    }
    if not force and cache_path.is_file() and metadata_path.is_file():
        metadata = _load_json(metadata_path)
        with np.load(cache_path, allow_pickle=False) as cached:
            if all(metadata.get(key) == value for key, value in expected.items()) and np.array_equal(
                cached["pet_ids"], pet_ids
            ):
                return {key: cached[key] for key in cached.files}, {
                    **metadata,
                    "cache_hit": True,
                    "actual_this_run_seconds": 0.0,
                }

    run_started = time.perf_counter()
    with zipfile.ZipFile(image_zip) as archive:
        query_full = _load_archive_images(archive, records, photo_index=0)
        gallery_full = _load_archive_images(archive, records, photo_index=1)
    gallery_crop, crop_available, detector_timing = _crop_gallery_images(
        gallery_full, device=device
    )

    arrays: dict[str, np.ndarray] = {
        "pet_ids": pet_ids,
        "crop_available": crop_available,
    }
    model_metadata: dict[str, Any] = {}
    try:
        for name, model_id in DINO_MODELS:
            model_started = time.perf_counter()
            encoder = DinoEncoder(
                model_id,
                device=device,
                local_files_only=local_files_only,
            )
            initialization_seconds = time.perf_counter() - model_started
            encode_started = time.perf_counter()
            arrays[f"{name}_query_full"] = _encode_batches(encoder, query_full)
            arrays[f"{name}_gallery_full"] = _encode_batches(encoder, gallery_full)
            arrays[f"{name}_gallery_crop"] = _encode_batches(encoder, gallery_crop)
            if encoder.device.startswith("cuda"):
                encoder._torch.cuda.synchronize()
            encoding_seconds = time.perf_counter() - encode_started
            model_metadata[name] = {
                "model_id": encoder.model_id,
                "resolved_revision": encoder.resolved_revision,
                "dimension": encoder.dimension,
                "device": encoder.device,
                "initialization_seconds": round(initialization_seconds, 6),
                "encoding_seconds": round(encoding_seconds, 6),
                "encoded_images": len(query_full) + len(gallery_full) + len(gallery_crop),
            }
            del encoder
            _release_cuda()
    finally:
        for image in (*query_full, *gallery_full, *gallery_crop):
            image.close()

    _save_npz_atomic(cache_path, **arrays)
    metadata = {
        **expected,
        "records": len(records),
        "detector": detector_timing,
        "models": model_metadata,
        "cache_hit": False,
        "actual_this_run_seconds": round(time.perf_counter() - run_started, 6),
    }
    write_json_atomic(metadata, metadata_path, pretty=True)
    return arrays, metadata


def _load_flow(device: str, torch: Any) -> Any:
    report = _load_json(FLOW_REPORT)
    details = report["architectures"]["flow"]
    checkpoint_path = FLOW_REPORT.parent / details["checkpoint"]
    if sha256_file(checkpoint_path) != details["checkpoint_sha256"]:
        raise ValueError("DINOde-flow checkpoint SHA-256 mismatch")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    head = projection_head_from_checkpoint(checkpoint).to(device)
    head.load_state_dict(checkpoint["state_dict"])
    head.eval()
    return head


def _mapped_appearance_text(
    records: Sequence[Mapping[str, Any]], *, device: str
) -> tuple[np.ndarray, dict[str, Any]]:
    started = time.perf_counter()
    encoder = ClipEncoder(device=device)
    torch = encoder._torch
    flow = _load_flow(encoder.device, torch)
    texts = [appearance_query_ko(record) for record in records]
    clip_text = encoder.encode_text_batch(texts)
    with torch.inference_mode():
        mapped = (
            torch.nn.functional.normalize(
                flow(torch.as_tensor(clip_text, device=encoder.device)), dim=-1
            )
            .detach()
            .float()
            .cpu()
            .numpy()
        )
    timing = {
        "encoder": "OpenAI CLIP ViT-B/32",
        "alignment": "existing DINOv3 eight-step flow",
        "texts": len(texts),
        "device": encoder.device,
        "seconds": round(time.perf_counter() - started, 6),
    }
    del flow, encoder
    _release_cuda()
    return mapped, timing


def _split_positions(records: Sequence[Mapping[str, Any]], split: str) -> np.ndarray:
    return np.asarray(
        [index for index, record in enumerate(records) if record["split"] == split],
        dtype=np.int64,
    )


def _visual_metrics_for_split(
    *,
    records: list[dict[str, Any]],
    arrays: Mapping[str, np.ndarray],
    positions: np.ndarray,
) -> dict[str, dict[str, dict[str, Any]]]:
    subset_records = [records[int(index)] for index in positions]
    crop_available = arrays["crop_available"][positions]
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for model_name, _model_id in DINO_MODELS:
        query = arrays[f"{model_name}_query_full"][positions]
        gallery_full = arrays[f"{model_name}_gallery_full"][positions]
        gallery_crop = arrays[f"{model_name}_gallery_crop"][positions]
        output[model_name] = {}
        for view in GALLERY_VIEWS:
            scores = gallery_scores(
                query,
                gallery_full=gallery_full,
                gallery_crop=gallery_crop,
                crop_available=crop_available,
                view=view,
            )
            output[model_name][view] = ranking_metrics(
                scores,
                records=subset_records,
                reference_positions=range(len(subset_records)),
            )
    return output


def _text_metrics(
    *,
    records: list[dict[str, Any]],
    dino_query: np.ndarray,
    mapped_text: np.ndarray,
    gallery_full: np.ndarray,
    gallery_crop: np.ndarray,
    crop_available: np.ndarray,
    view: str,
    text_weight: float,
) -> dict[str, Any]:
    references = [
        index for index, record in enumerate(records) if is_appearance_query(record)
    ]
    query = blend_embeddings(
        dino_query[references],
        mapped_text[references],
        text_weight=text_weight,
    )
    scores = gallery_scores(
        query,
        gallery_full=gallery_full,
        gallery_crop=gallery_crop,
        crop_available=crop_available,
        view=view,
    )
    return ranking_metrics(
        scores,
        records=records,
        reference_positions=references,
    )


def evaluate(
    *,
    image_zip: Path,
    output_dir: Path,
    device: str = "auto",
    local_files_only: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    run_started = time.perf_counter()
    records_path = output_dir / "multimodal_records.json"
    all_records = _load_json(records_path)
    records = [
        record for record in all_records if record["split"] in {"validation", "test"}
    ]
    if not records:
        raise ValueError("multimodal validation/test records are missing")
    arrays, embedding_timing = performance_embeddings(
        records,
        image_zip=image_zip,
        output_dir=output_dir,
        device=device,
        local_files_only=local_files_only,
        force=force,
    )
    mapped_text, text_timing = _mapped_appearance_text(records, device=device)

    positions_by_split = {
        split: _split_positions(records, split) for split in ("validation", "test")
    }
    visual_metrics = {
        split: _visual_metrics_for_split(
            records=records,
            arrays=arrays,
            positions=positions,
        )
        for split, positions in positions_by_split.items()
    }
    selected_views = {
        model_name: select_gallery_view(visual_metrics["validation"][model_name])
        for model_name, _model_id in DINO_MODELS
    }

    selected_v3_view = selected_views["dinov3_vitb16"]
    validation_positions = positions_by_split["validation"]
    validation_records = [records[int(index)] for index in validation_positions]
    validation_kwargs = {
        "records": validation_records,
        "dino_query": arrays["dinov3_vitb16_query_full"][validation_positions],
        "mapped_text": mapped_text[validation_positions],
        "gallery_full": arrays["dinov3_vitb16_gallery_full"][validation_positions],
        "gallery_crop": arrays["dinov3_vitb16_gallery_crop"][validation_positions],
        "crop_available": arrays["crop_available"][validation_positions],
        "view": selected_v3_view,
    }
    validation_baseline = _text_metrics(**validation_kwargs, text_weight=0.0)
    validation_sweep: list[dict[str, Any]] = []
    for weight in TEXT_WEIGHT_CANDIDATES:
        metrics = _text_metrics(**validation_kwargs, text_weight=weight)
        validation_sweep.append(
            {
                "text_weight": weight,
                **metrics,
                "recall10_drop": round(
                    validation_baseline["recall@10"] - metrics["recall@10"], 6
                ),
                "mrr_drop": round(validation_baseline["mrr"] - metrics["mrr"], 6),
                "appearance_ndcg_gain": round(
                    metrics["appearance_ndcg@10"]
                    - validation_baseline["appearance_ndcg@10"],
                    6,
                ),
            }
        )
    selected_text_weight = select_text_weight(validation_sweep)

    test_positions = positions_by_split["test"]
    test_records = [records[int(index)] for index in test_positions]
    test_common = {
        "records": test_records,
        "dino_query": arrays["dinov3_vitb16_query_full"][test_positions],
        "mapped_text": mapped_text[test_positions],
        "gallery_full": arrays["dinov3_vitb16_gallery_full"][test_positions],
        "gallery_crop": arrays["dinov3_vitb16_gallery_crop"][test_positions],
        "crop_available": arrays["crop_available"][test_positions],
    }
    selected_test = _text_metrics(
        **test_common,
        view=selected_v3_view,
        text_weight=selected_text_weight,
    )
    selected_test_image_only = _text_metrics(
        **test_common,
        view=selected_v3_view,
        text_weight=0.0,
    )
    current_runtime_test = _text_metrics(
        **test_common,
        view="crop",
        text_weight=0.20,
    )
    test_gain = (
        selected_test["appearance_ndcg@10"]
        - selected_test_image_only["appearance_ndcg@10"]
    )
    test_recall_drop = (
        selected_test_image_only["recall@10"] - selected_test["recall@10"]
    )
    test_mrr_drop = selected_test_image_only["mrr"] - selected_test["mrr"]
    text_gate_checks = {
        "appearance_ndcg_gain": {
            "value": round(test_gain, 6),
            "minimum": MINIMUM_APPEARANCE_NDCG_GAIN,
            "passes": test_gain >= MINIMUM_APPEARANCE_NDCG_GAIN,
        },
        "identity_recall10_drop": {
            "value": round(test_recall_drop, 6),
            "maximum": MAXIMUM_RECALL10_DROP,
            "passes": test_recall_drop <= MAXIMUM_RECALL10_DROP,
        },
        "identity_mrr_drop": {
            "value": round(test_mrr_drop, 6),
            "maximum": MAXIMUM_MRR_DROP,
            "passes": test_mrr_drop <= MAXIMUM_MRR_DROP,
        },
    }

    selected_visual_test = {
        model_name: visual_metrics["test"][model_name][selected_views[model_name]]
        for model_name, _model_id in DINO_MODELS
    }
    v2 = selected_visual_test["dinov2_base"]
    v3 = selected_visual_test["dinov3_vitb16"]
    v2_vs_v3 = {
        "recall@10_delta_v2_minus_v3": round(v2["recall@10"] - v3["recall@10"], 6),
        "mrr_delta_v2_minus_v3": round(v2["mrr"] - v3["mrr"], 6),
        "appearance_ndcg_delta_v2_minus_v3": round(
            v2["appearance_ndcg@10"] - v3["appearance_ndcg@10"], 6
        ),
        "interpretation": (
            "DINOv2 warrants a separately trained text-alignment follow-up"
            if v2["recall@10"] >= v3["recall@10"] + 0.02
            and v2["mrr"] >= v3["mrr"]
            else "retain DINOv3 for the multimodal route"
        ),
    }

    text_gate_passes = all(
        bool(check["passes"]) for check in text_gate_checks.values()
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "evaluation_protocol": {
            "split": "organization-disjoint validation and test",
            "query_photo": "PetFinder photo 1, uncropped",
            "gallery_photo": "PetFinder photo 2 as full image and detector crop",
            "identity_target": "same dog across different photos",
            "appearance_ground_truth": (
                "PetFinder color, size, age, and coat; reference dog excluded from "
                "appearance nDCG"
            ),
            "appearance_query_language": "Korean",
            "selection_policy": (
                "gallery view and text weight selected on validation only; frozen "
                "choices applied once to test"
            ),
            "split_counts": {
                split: int(len(positions))
                for split, positions in positions_by_split.items()
            },
        },
        "predeclared_thresholds": {
            "text_weight_candidates_validation_only": list(TEXT_WEIGHT_CANDIDATES),
            "maximum_test_identity_recall10_drop": MAXIMUM_RECALL10_DROP,
            "maximum_test_identity_mrr_drop": MAXIMUM_MRR_DROP,
            "minimum_test_appearance_ndcg_gain": MINIMUM_APPEARANCE_NDCG_GAIN,
        },
        "selected_gallery_views": selected_views,
        "visual_metrics": visual_metrics,
        "selected_visual_test": selected_visual_test,
        "v2_vs_v3": v2_vs_v3,
        "text_alignment": {
            "selected_gallery_view": selected_v3_view,
            "selected_text_weight": selected_text_weight,
            "validation_image_only": validation_baseline,
            "validation_weight_sweep": validation_sweep,
            "test_selected_view_image_only": selected_test_image_only,
            "test_selected_weight": selected_test,
            "test_current_runtime_crop_weight_0_20": current_runtime_test,
            "gate": {
                "checks": text_gate_checks,
                "passes": text_gate_passes,
                "recommendation": (
                    "candidate for guarded shadow rollout"
                    if text_gate_passes
                    else "keep current setting; no validated text-weight gain"
                ),
            },
        },
        "timing": {
            "embeddings": embedding_timing,
            "text_alignment": text_timing,
            "actual_this_run_end_to_end_seconds": round(
                time.perf_counter() - run_started, 6
            ),
        },
        "limitations": [
            "PetFinder fields are shelter-entered silver appearance labels.",
            "The image archive has no stated license and remains local and ignored.",
            "Only 38 validation and 77 test dogs are available in this archive shard.",
            "DINOv2 is evaluated visually only because the existing flow head targets DINOv3.",
        ],
    }
    write_json_atomic(
        report, output_dir / "performance_ablation_report.json", pretty=True
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-zip", type=Path, default=DEFAULT_IMAGE_ZIP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow Hugging Face downloads instead of requiring the local cache.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute detector crops and image embeddings even when cached.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate(
        image_zip=args.image_zip,
        output_dir=args.output_dir,
        device=args.device,
        local_files_only=not args.allow_model_download,
        force=args.force,
    )
    print(
        json.dumps(
            {
                "selected_gallery_views": report["selected_gallery_views"],
                "selected_visual_test": report["selected_visual_test"],
                "v2_vs_v3": report["v2_vs_v3"],
                "text_alignment": report["text_alignment"],
                "timing": report["timing"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
