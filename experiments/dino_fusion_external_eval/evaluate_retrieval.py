"""Evaluate real image + Korean compatibility queries on held-out shelters."""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import zipfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.dino_fusion.alignment import (  # noqa: E402
    blend_embeddings,
    projection_head_from_checkpoint,
)
from experiments.dino_fusion.behavior import (  # noqa: E402
    BEHAVIOR_AXES,
    POSITIVE,
    UNKNOWN,
)
from experiments.dino_fusion.core import (  # noqa: E402
    ClipEncoder,
    DinoEncoder,
    normalize_rows,
)
from experiments.dino_fusion_external_eval.compatibility import (  # noqa: E402
    COMPATIBILITY_AXES,
    classification_metrics,
    fuse_visual_behavior,
    make_compatibility_head,
    ndcg_at_k,
    summarize_ranking_queries,
)
from experiments.dino_fusion_external_eval.prepare_benchmark import (  # noqa: E402
    DEFAULT_IMAGE_ZIP,
    DEFAULT_OUTPUT_DIR,
    sha256_file,
    write_json_atomic,
)


FLOW_REPORT = (
    ROOT
    / "experiments"
    / "dino_fusion"
    / "artifacts"
    / "dinode_flow"
    / "alignment_training_report.json"
)
REPORT_SCHEMA_VERSION = "petfinder-multimodal-compatibility-eval.v1"
TEXT_WEIGHT = 0.20
BEHAVIOR_WEIGHT_CANDIDATES = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
MINIMUM_NDCG_GAIN = 0.03
MAXIMUM_RECALL10_DROP = 0.02
MINIMUM_CLASSIFICATION_AUC = 0.70


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


def _clip_image_batch(encoder: ClipEncoder, images: Sequence[Image.Image]) -> np.ndarray:
    if not images:
        return np.empty((0, encoder.text_dimension), dtype=np.float32)
    tensors = [encoder._preprocess(image.convert("RGB")) for image in images]
    batch = encoder._torch.stack(tensors).to(encoder.device)
    with encoder._torch.inference_mode():
        values = encoder._model.encode_image(batch)
    return normalize_rows(values.detach().float().cpu().numpy())


def _load_images(
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
        with Image.open(io.BytesIO(archive.read(str(members[photo_index])))) as image:
            images.append(image.convert("RGB"))
    return images


def _encode_in_batches(
    images: Sequence[Image.Image],
    *,
    batch_size: int,
    encode: Callable[[Sequence[Image.Image]], np.ndarray],
) -> np.ndarray:
    batches = [
        encode(images[start : start + batch_size])
        for start in range(0, len(images), batch_size)
    ]
    return np.concatenate(batches, axis=0)


def image_embeddings(
    records: list[dict[str, Any]],
    *,
    image_zip: Path,
    output_dir: Path,
    device: str,
) -> tuple[dict[str, np.ndarray], ClipEncoder, dict[str, Any]]:
    cache_path = output_dir / "multimodal_image_embeddings.npz"
    metadata_path = output_dir / "multimodal_image_embeddings.json"
    record_ids = np.asarray([str(record["pet_id"]) for record in records])
    records_hash = sha256_file(output_dir / "multimodal_records.json")
    if cache_path.is_file() and metadata_path.is_file():
        metadata = _load_json(metadata_path)
        cached = np.load(cache_path, allow_pickle=False)
        if (
            metadata.get("records_sha256") == records_hash
            and np.array_equal(cached["pet_ids"], record_ids)
        ):
            started = time.perf_counter()
            clip_encoder = ClipEncoder(device=device)
            metadata = {
                **metadata,
                "cache_hit": True,
                "actual_this_run_seconds": round(time.perf_counter() - started, 6),
            }
            return {key: cached[key] for key in cached.files}, clip_encoder, metadata

    started = time.perf_counter()
    clip_encoder = ClipEncoder(device=device)
    dino_encoder = DinoEncoder(device=device, local_files_only=True)
    model_initialization_seconds = time.perf_counter() - started
    with zipfile.ZipFile(image_zip) as archive:
        query_images = _load_images(archive, records, photo_index=0)
        gallery_images = _load_images(archive, records, photo_index=1)

    encode_started = time.perf_counter()
    clip_query = _encode_in_batches(
        query_images,
        batch_size=128,
        encode=lambda batch: _clip_image_batch(clip_encoder, batch),
    )
    clip_gallery = _encode_in_batches(
        gallery_images,
        batch_size=128,
        encode=lambda batch: _clip_image_batch(clip_encoder, batch),
    )
    dino_query = _encode_in_batches(
        query_images, batch_size=32, encode=dino_encoder.encode_batch
    )
    dino_gallery = _encode_in_batches(
        gallery_images, batch_size=32, encode=dino_encoder.encode_batch
    )
    if dino_encoder.device == "cuda":
        dino_encoder._torch.cuda.synchronize()
    encoding_seconds = time.perf_counter() - encode_started
    arrays = {
        "pet_ids": record_ids,
        "clip_query": clip_query,
        "clip_gallery": clip_gallery,
        "dino_query": dino_query,
        "dino_gallery": dino_gallery,
    }
    _save_npz_atomic(cache_path, **arrays)
    metadata = {
        "records_sha256": records_hash,
        "image_zip_sha256": sha256_file(image_zip),
        "records": len(records),
        "clip_encoder": "OpenAI CLIP ViT-B/32",
        "dino_encoder": dino_encoder.model_id,
        "dino_revision": dino_encoder.resolved_revision,
        "model_initialization_seconds": round(model_initialization_seconds, 6),
        "actual_full_image_encoding_seconds": round(encoding_seconds, 6),
        "cache_hit": False,
        "actual_this_run_seconds": round(
            model_initialization_seconds + encoding_seconds, 6
        ),
    }
    write_json_atomic(metadata, metadata_path, pretty=True)
    return arrays, clip_encoder, metadata


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


def _queries(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    for reference_index, record in enumerate(records):
        for axis_index, axis in enumerate(COMPATIBILITY_AXES):
            desired = int(record["labels"][axis_index])
            if desired == UNKNOWN:
                continue
            queries.append(
                {
                    "query_id": f"{record['pet_id']}:{axis.key}:{desired}",
                    "reference_index": reference_index,
                    "axis_index": axis_index,
                    "axis": axis.key,
                    "desired_label": desired,
                    "text": (
                        axis.positive_query_ko
                        if desired == POSITIVE
                        else axis.negative_query_ko
                    ),
                }
            )
    return queries


def _query_result(
    *,
    query: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    scores: np.ndarray,
) -> dict[str, Any]:
    reference_index = int(query["reference_index"])
    axis_index = int(query["axis_index"])
    desired = int(query["desired_label"])
    labels = np.asarray(
        [int(record["labels"][axis_index]) for record in records], dtype=np.int64
    )
    relevance = (labels == desired).astype(np.float64)
    order = np.argsort(-scores, kind="mergesort")
    identity_rank = int(np.flatnonzero(order == reference_index)[0]) + 1
    top = order[: min(10, len(order))]
    top_labels = labels[top]
    denominator = max(1, len(top))
    return {
        "query_id": query["query_id"],
        "axis": query["axis"],
        "desired_label": desired,
        "text": query["text"],
        "reference_pet_id": records[reference_index]["pet_id"],
        "identity_rank": identity_rank,
        "recall@1": float(identity_rank <= 1),
        "recall@5": float(identity_rank <= 5),
        "recall@10": float(identity_rank <= 10),
        "behavior_ndcg@10": ndcg_at_k(relevance, scores, k=10),
        "match_rate@10": float((top_labels == desired).sum() / denominator),
        "contradiction_rate@10": float((top_labels == -desired).sum() / denominator),
        "unknown_rate@10": float((top_labels == UNKNOWN).sum() / denominator),
        "top_pet_ids": [records[int(index)]["pet_id"] for index in top[:5]],
    }


def _evaluate_scores(
    records: Sequence[Mapping[str, Any]],
    queries: Sequence[Mapping[str, Any]],
    score_for_query: Callable[[Mapping[str, Any]], np.ndarray],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = [
        _query_result(query=query, records=records, scores=score_for_query(query))
        for query in queries
    ]
    return summarize_ranking_queries(rows), rows


def _compatibility_probabilities(
    *, output_dir: Path, pet_ids: Sequence[str], device: str, torch: Any
) -> tuple[np.ndarray, dict[str, Any]] | None:
    variants = (
        (
            "evidence",
            output_dir / "compatibility_evidence_head.pt",
            output_dir / "compatibility_evidence_training_report.json",
            output_dir / "clip_evidence_embeddings.npy",
        ),
        (
            "raw",
            output_dir / "compatibility_head.pt",
            output_dir / "compatibility_training_report.json",
            output_dir / "clip_description_embeddings.npy",
        ),
    )
    records_path = output_dir / "text_records.json"
    selected = next(
        (
            variant
            for variant in variants
            if records_path.is_file() and all(path.is_file() for path in variant[1:])
        ),
        None,
    )
    if selected is None:
        return None
    text_mode, checkpoint_path, report_path, embeddings_path = selected
    report = _load_json(report_path)
    if sha256_file(checkpoint_path) != report["model"]["checkpoint_sha256"]:
        raise ValueError("compatibility checkpoint SHA-256 mismatch")
    text_records = _load_json(records_path)
    row_by_pet = {
        str(record["pet_id"]): index for index, record in enumerate(text_records)
    }
    rows = [row_by_pet[pet_id] for pet_id in pet_ids]
    embeddings = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)[rows]
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    expected_axes = [axis.key for axis in COMPATIBILITY_AXES]
    if list(checkpoint.get("axis_keys") or []) != expected_axes:
        raise ValueError("compatibility checkpoint axes do not match evaluator")
    head = make_compatibility_head(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        output_dim=int(checkpoint["output_dim"]),
    ).to(device)
    head.load_state_dict(checkpoint["state_dict"])
    head.eval()
    with torch.inference_mode():
        probabilities = (
            torch.sigmoid(head(torch.as_tensor(embeddings, device=device)))
            .detach()
            .float()
            .cpu()
            .numpy()
        )
    report = {**report, "selected_text_mode": text_mode}
    return probabilities, report


def _system_metrics(
    *,
    records: list[dict[str, Any]],
    arrays: Mapping[str, np.ndarray],
    clip_encoder: ClipEncoder,
    flow_head: Any,
    probabilities: np.ndarray | None,
    behavior_weight: float | None = None,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    torch = clip_encoder._torch
    queries = _queries(records)
    unique_texts = list(dict.fromkeys(str(query["text"]) for query in queries))
    clip_text = clip_encoder.encode_text_batch(unique_texts)
    text_by_value = {text: clip_text[index] for index, text in enumerate(unique_texts)}
    with torch.inference_mode():
        mapped = (
            torch.nn.functional.normalize(
                flow_head(torch.as_tensor(clip_text, device=clip_encoder.device)), dim=-1
            )
            .detach()
            .float()
            .cpu()
            .numpy()
        )
    mapped_by_value = {text: mapped[index] for index, text in enumerate(unique_texts)}

    dino_matrix = arrays["dino_query"] @ arrays["dino_gallery"].T
    clip_matrix = arrays["clip_query"] @ arrays["clip_gallery"].T

    def clip_text_scores(query: Mapping[str, Any]) -> np.ndarray:
        reference = arrays["clip_query"][int(query["reference_index"])]
        text = text_by_value[str(query["text"])]
        blended = normalize_rows((1.0 - TEXT_WEIGHT) * reference + TEXT_WEIGHT * text)
        return (blended @ arrays["clip_gallery"].T)[0]

    def flow_scores(query: Mapping[str, Any]) -> np.ndarray:
        reference = arrays["dino_query"][int(query["reference_index"])]
        text = mapped_by_value[str(query["text"])]
        blended = blend_embeddings(
            reference.reshape(1, -1), text.reshape(1, -1), text_weight=TEXT_WEIGHT
        )
        return (blended @ arrays["dino_gallery"].T)[0]

    score_functions: dict[str, Callable[[Mapping[str, Any]], np.ndarray]] = {
        "clip_image_only": lambda query: clip_matrix[int(query["reference_index"])],
        "clip_image_plus_korean_text": clip_text_scores,
        "dino_image_only": lambda query: dino_matrix[int(query["reference_index"])],
        "current_dinode_flow": flow_scores,
    }
    if probabilities is not None and behavior_weight is not None:

        def trained_scores(query: Mapping[str, Any]) -> np.ndarray:
            visual = dino_matrix[int(query["reference_index"])]
            axis = int(query["axis_index"])
            positive = probabilities[:, axis]
            behavior = (
                positive if int(query["desired_label"]) == POSITIVE else 1.0 - positive
            )
            return fuse_visual_behavior(
                visual, behavior, behavior_weight=behavior_weight
            )

        score_functions["dino_plus_trained_compatibility"] = trained_scores

    summaries: dict[str, Any] = {}
    details: dict[str, list[dict[str, Any]]] = {}
    for name, function in score_functions.items():
        summaries[name], details[name] = _evaluate_scores(records, queries, function)
    return summaries, details


def evaluate(
    *, image_zip: Path, output_dir: Path, device: str = "auto"
) -> dict[str, Any]:
    run_started = time.perf_counter()
    records_path = output_dir / "multimodal_records.json"
    all_records = _load_json(records_path)
    records = [
        record for record in all_records if record["split"] in {"validation", "test"}
    ]
    if not records:
        raise ValueError("multimodal validation/test records are missing")
    arrays, clip_encoder, image_timing = image_embeddings(
        records, image_zip=image_zip, output_dir=output_dir, device=device
    )
    torch = clip_encoder._torch
    resolved_device = clip_encoder.device
    flow_head = _load_flow(resolved_device, torch)
    pet_ids = [str(record["pet_id"]) for record in records]
    trained = _compatibility_probabilities(
        output_dir=output_dir,
        pet_ids=pet_ids,
        device=resolved_device,
        torch=torch,
    )
    probabilities = trained[0] if trained else None
    training_report = trained[1] if trained else None
    multimodal_classification: dict[str, Any] = {}

    split_payload: dict[str, Any] = {}
    split_details: dict[str, Any] = {}
    split_positions: dict[str, np.ndarray] = {}
    for split in ("validation", "test"):
        positions = np.asarray(
            [index for index, record in enumerate(records) if record["split"] == split]
        )
        split_positions[split] = positions
        subset_records = [records[int(index)] for index in positions]
        subset_arrays = {
            key: value[positions] if key != "pet_ids" else value[positions]
            for key, value in arrays.items()
        }
        subset_probabilities = probabilities[positions] if probabilities is not None else None
        if subset_probabilities is not None:
            multimodal_classification[split] = classification_metrics(
                np.asarray([record["labels"] for record in subset_records]),
                subset_probabilities,
            )
        summaries, details = _system_metrics(
            records=subset_records,
            arrays=subset_arrays,
            clip_encoder=clip_encoder,
            flow_head=flow_head,
            probabilities=subset_probabilities,
        )
        split_payload[split] = summaries
        split_details[split] = details

    selected_weight: float | None = None
    validation_sweep: list[dict[str, Any]] = []
    if probabilities is not None:
        positions = split_positions["validation"]
        subset_records = [records[int(index)] for index in positions]
        subset_arrays = {key: value[positions] for key, value in arrays.items()}
        baseline = split_payload["validation"]["dino_image_only"]
        for weight in BEHAVIOR_WEIGHT_CANDIDATES:
            summaries, _ = _system_metrics(
                records=subset_records,
                arrays=subset_arrays,
                clip_encoder=clip_encoder,
                flow_head=flow_head,
                probabilities=probabilities[positions],
                behavior_weight=weight,
            )
            candidate = summaries["dino_plus_trained_compatibility"]
            validation_sweep.append(
                {
                    "behavior_weight": weight,
                    **candidate,
                    "recall10_drop": round(
                        baseline["recall@10"] - candidate["recall@10"], 6
                    ),
                    "ndcg_gain": round(
                        candidate["behavior_ndcg@10"]
                        - baseline["behavior_ndcg@10"],
                        6,
                    ),
                }
            )
        eligible = [
            row
            for row in validation_sweep
            if float(row["recall10_drop"]) <= MAXIMUM_RECALL10_DROP
        ]
        selected = max(
            eligible or validation_sweep,
            key=lambda row: (
                float(row["behavior_ndcg@10"]),
                -float(row["behavior_weight"]),
            ),
        )
        selected_weight = float(selected["behavior_weight"])
        for split in ("validation", "test"):
            positions = split_positions[split]
            subset_records = [records[int(index)] for index in positions]
            subset_arrays = {key: value[positions] for key, value in arrays.items()}
            summaries, details = _system_metrics(
                records=subset_records,
                arrays=subset_arrays,
                clip_encoder=clip_encoder,
                flow_head=flow_head,
                probabilities=probabilities[positions],
                behavior_weight=selected_weight,
            )
            split_payload[split]["dino_plus_trained_compatibility"] = summaries[
                "dino_plus_trained_compatibility"
            ]
            split_details[split]["dino_plus_trained_compatibility"] = details[
                "dino_plus_trained_compatibility"
            ]

    current_axis_keys = {axis.key for axis in BEHAVIOR_AXES}
    supported_axes = [
        axis.key for axis in COMPATIBILITY_AXES if axis.key in current_axis_keys
    ]
    current_coverage = len(supported_axes) / len(COMPATIBILITY_AXES)
    pretraining_gate = {
        "predeclared_minimum_supported_axis_coverage": 0.75,
        "current_supported_axes": supported_axes,
        "current_axis_coverage": current_coverage,
        "passes": current_coverage >= 0.75,
        "action": "keep frozen system" if current_coverage >= 0.75 else "train head",
    }
    final_gate: dict[str, Any] | None = None
    if selected_weight is not None and training_report is not None:
        baseline = split_payload["test"]["dino_image_only"]
        candidate = split_payload["test"]["dino_plus_trained_compatibility"]
        ndcg_gain = candidate["behavior_ndcg@10"] - baseline["behavior_ndcg@10"]
        recall_drop = baseline["recall@10"] - candidate["recall@10"]
        classification_auc = multimodal_classification["test"]["macro_roc_auc"]
        checks = {
            "behavior_ndcg_gain": {
                "value": round(ndcg_gain, 6),
                "minimum": MINIMUM_NDCG_GAIN,
                "passes": ndcg_gain >= MINIMUM_NDCG_GAIN,
            },
            "identity_recall10_drop": {
                "value": round(recall_drop, 6),
                "maximum": MAXIMUM_RECALL10_DROP,
                "passes": recall_drop <= MAXIMUM_RECALL10_DROP,
            },
            "classification_macro_roc_auc": {
                "value": classification_auc,
                "minimum": MINIMUM_CLASSIFICATION_AUC,
                "passes": classification_auc >= MINIMUM_CLASSIFICATION_AUC,
            },
            "supported_axis_coverage": {
                "value": 1.0,
                "minimum": 0.75,
                "passes": True,
            },
        }
        final_gate = {
            "checks": checks,
            "passes": all(check["passes"] for check in checks.values()),
            "recommendation": (
                "candidate for guarded integration"
                if all(check["passes"] for check in checks.values())
                else "keep experimental and improve"
            ),
        }

    samples: list[dict[str, Any]] = []
    sample_system = (
        "dino_plus_trained_compatibility"
        if selected_weight is not None
        else "current_dinode_flow"
    )
    for row in split_details["test"][sample_system][:12]:
        samples.append(row)
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "evaluation_protocol": {
            "input": "photo 1 + Korean compatibility query",
            "gallery": "photo 2 from each held-out dog",
            "identity_target": "same PetFinder dog across different photos",
            "behavior_ground_truth": "held-out structured PetFinder field",
            "unknown_policy": "unknown is neither positive nor negative",
            "split": "organization-disjoint validation and test",
            "query_count": {
                split: split_payload[split]["dino_image_only"]["query_count"]
                for split in ("validation", "test")
            },
        },
        "systems": {
            "clip_image_only": "frozen CLIP image-to-image",
            "clip_image_plus_korean_text": "frozen CLIP 80% image + 20% Korean text",
            "dino_image_only": "frozen DINOv3 image-to-image",
            "current_dinode_flow": (
                "current 80% DINO image + 20% aligned CLIP text; compatibility parser "
                "has zero matching axes"
            ),
            "dino_plus_trained_compatibility": (
                "DINO appearance + frozen-CLIP description head "
                f"({training_report.get('selected_text_mode', 'unknown')} preprocessing)"
                if selected_weight is not None
                else "not run"
            ),
        },
        "predeclared_thresholds": {
            "behavior_weight_candidates_validation_only": list(
                BEHAVIOR_WEIGHT_CANDIDATES
            ),
            "minimum_test_behavior_ndcg_gain": MINIMUM_NDCG_GAIN,
            "maximum_test_identity_recall10_drop": MAXIMUM_RECALL10_DROP,
            "minimum_test_classification_macro_roc_auc": MINIMUM_CLASSIFICATION_AUC,
        },
        "pretraining_gate": pretraining_gate,
        "selected_behavior_weight": selected_weight,
        "validation_weight_sweep": validation_sweep,
        "multimodal_classification": multimodal_classification,
        "metrics": split_payload,
        "final_gate": final_gate,
        "timing": {
            "image_embeddings": image_timing,
            "actual_this_run_end_to_end_seconds": round(
                time.perf_counter() - run_started, 6
            ),
        },
        "samples": samples,
        "limitations": [
            "Behavior labels are shelter-entered rather than direct observations.",
            "Only 77 test dogs are available in the joined photo/text slice.",
            "Cat-positive test queries are sparse and therefore high variance.",
            "The raw photo archive has no stated license and is not redistributed.",
        ],
    }
    write_json_atomic(report, output_dir / "retrieval_evaluation_report.json", pretty=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-zip", type=Path, default=DEFAULT_IMAGE_ZIP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate(
        image_zip=args.image_zip, output_dir=args.output_dir, device=args.device
    )
    print(
        json.dumps(
            {
                "pretraining_gate": report["pretraining_gate"],
                "selected_behavior_weight": report["selected_behavior_weight"],
                "test_metrics": report["metrics"]["test"],
                "final_gate": report["final_gate"],
                "timing": report["timing"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
