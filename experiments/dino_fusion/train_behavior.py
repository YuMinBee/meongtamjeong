"""Train and evaluate an unknown-aware public-notice behavior reranker.

This is an isolated pilot.  DINOv3 and CLIP stay frozen; only a small
CLIP-note-to-behavior head is optimized.  The report compares the current
image-space personality query routes with the proposed behavior branch on the
same 1,002-crop corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.notice_metadata import normalize_breed_fields  # noqa: E402
from experiments.dino_fusion.alignment import (  # noqa: E402
    blend_embeddings,
    project_embeddings,
    projection_head_from_checkpoint,
)
from experiments.dino_fusion.behavior import (  # noqa: E402
    BEHAVIOR_AXES,
    NEGATIVE,
    POSITIVE,
    UNKNOWN,
    binary_ranking_metrics,
    build_behavior_records,
    conservative_behavior_score,
    finite_mean,
    make_behavior_head,
    percentile_scores,
    stratified_text_group_split,
)
from experiments.dino_fusion.build_index import (  # noqa: E402
    sha256_file,
    write_json_atomic,
)
from experiments.dino_fusion.core import (  # noqa: E402
    ClipEncoder,
    DinoEncoder,
    clean_text,
    normalize_rows,
    safe_read_faiss_index,
)


DEFAULT_DINO_DIR = Path(__file__).resolve().parent / "artifacts" / "dinov3"
DEFAULT_FLOW_DIR = Path(__file__).resolve().parent / "artifacts" / "dinode_flow"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "artifacts" / "behavior_pilot"
REPORT_SCHEMA_VERSION = "dino-clip-behavior-pilot.v1"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def reconstruct_index(index: Any) -> np.ndarray:
    matrix = np.empty((int(index.ntotal), int(index.d)), dtype=np.float32)
    for row in range(int(index.ntotal)):
        index.reconstruct(row, matrix[row])
    return normalize_rows(matrix)


def save_checkpoint_atomic(payload: Mapping[str, Any], path: Path, torch: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def encode_unique_notes(
    encoder: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
) -> np.ndarray:
    text_to_rows: dict[str, list[int]] = {}
    for row, record in enumerate(records):
        text = clean_text(record.get("behavior_text"))
        if text:
            text_to_rows.setdefault(text, []).append(row)
    texts = list(text_to_rows)
    vectors = np.zeros((len(records), encoder.text_dimension), dtype=np.float32)
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = encoder.encode_text_batch(batch)
        for offset, text in enumerate(batch):
            for row in text_to_rows[text]:
                vectors[row] = encoded[offset]
    return vectors


def _balanced_masked_loss(torch: Any, logits: Any, labels: Any) -> Any:
    """Balanced BCE over observed ends only; zero-valued labels abstain."""

    losses: list[Any] = []
    for axis_index in range(labels.shape[1]):
        axis_labels = labels[:, axis_index]
        positives = axis_labels.eq(POSITIVE)
        negatives = axis_labels.eq(NEGATIVE)
        if positives.any() and negatives.any():
            positive_loss = torch.nn.functional.softplus(
                -logits[positives, axis_index]
            ).mean()
            negative_loss = torch.nn.functional.softplus(
                logits[negatives, axis_index]
            ).mean()
            losses.append(0.5 * (positive_loss + negative_loss))
    if not losses:
        raise ValueError("split has no axis with both observed labels")
    return torch.stack(losses).mean()


def _label_counts(
    records: Sequence[Mapping[str, Any]], indices: Sequence[int]
) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for axis_index, axis in enumerate(BEHAVIOR_AXES):
        values = [int(records[index]["labels"][axis_index]) for index in indices]
        output[axis.key] = {
            "positive": values.count(POSITIVE),
            "negative": values.count(NEGATIVE),
            "unknown": values.count(UNKNOWN),
        }
    return output


def _classification_metrics(
    labels: np.ndarray, probabilities: np.ndarray
) -> dict[str, Any]:
    axes: dict[str, Any] = {}
    for axis_index, axis in enumerate(BEHAVIOR_AXES):
        known = labels[:, axis_index] != UNKNOWN
        truth = (labels[known, axis_index] == POSITIVE).astype(np.int64)
        scores = probabilities[known, axis_index]
        if len(np.unique(truth)) < 2:
            axes[axis.key] = {"known_count": int(len(truth)), "not_evaluable": True}
            continue
        metrics = binary_ranking_metrics(truth, scores)
        predicted = scores >= 0.5
        positive_recall = float(predicted[truth == 1].mean())
        negative_recall = float((~predicted[truth == 0]).mean())
        metrics.update(
            {
                "accuracy": round(float((predicted == truth).mean()), 6),
                "balanced_accuracy": round(
                    0.5 * (positive_recall + negative_recall), 6
                ),
            }
        )
        axes[axis.key] = metrics
    evaluable = [value for value in axes.values() if not value.get("not_evaluable")]
    return {
        "axes": axes,
        "macro_roc_auc": finite_mean([value["roc_auc"] for value in evaluable]),
        "macro_average_precision": finite_mean(
            [value["average_precision"] for value in evaluable]
        ),
        "macro_balanced_accuracy": finite_mean(
            [value["balanced_accuracy"] for value in evaluable]
        ),
    }


def _load_flow_head(
    *, report_path: Path, device: str, torch: Any
) -> tuple[Any, dict[str, Any]]:
    report = load_json(report_path)
    details = (report.get("architectures") or {}).get("flow") or {}
    checkpoint_path = report_path.parent / clean_text(details.get("checkpoint"))
    expected_hash = clean_text(details.get("checkpoint_sha256"))
    if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != expected_hash:
        raise ValueError("flow alignment checkpoint verification failed")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    head = projection_head_from_checkpoint(checkpoint).to(device)
    head.load_state_dict(checkpoint["state_dict"])
    head.eval()
    return head, report


def _axis_query_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for axis_index, axis in enumerate(BEHAVIOR_AXES):
        rows.extend(
            [
                {
                    "query_id": f"{axis.key}:positive",
                    "axis_index": axis_index,
                    "axis": axis.key,
                    "desired_label": POSITIVE,
                    "desired_label_text": axis.positive_label,
                    "text": axis.positive_query,
                },
                {
                    "query_id": f"{axis.key}:negative",
                    "axis_index": axis_index,
                    "axis": axis.key,
                    "desired_label": NEGATIVE,
                    "desired_label_text": axis.negative_label,
                    "text": axis.negative_query,
                },
            ]
        )
    return rows


def _decorate_top(
    scores: np.ndarray,
    *,
    records: Sequence[Mapping[str, Any]],
    clip_metas: Sequence[Mapping[str, Any]],
    axis_index: int,
    desired_label: int,
    probabilities: np.ndarray,
    behavior_scores: np.ndarray,
    behavior_states: Sequence[str],
    split_by_row: Mapping[int, str],
    topk: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for rank, row in enumerate(np.argsort(-scores, kind="mergesort")[:topk], start=1):
        record = records[int(row)]
        meta = clip_metas[int(record["source_meta_index"])]
        breed = normalize_breed_fields(meta)
        explicit = int(record["labels"][axis_index])
        weak = record["weak_labels"][BEHAVIOR_AXES[axis_index].key]
        evidence = (
            weak["positive_evidence"]
            if explicit == POSITIVE
            else weak["negative_evidence"]
        )
        output.append(
            {
                "rank": rank,
                "notice_id": record["notice_id"],
                "score": round(float(scores[row]), 6),
                "behavior_score": round(float(behavior_scores[row]), 6),
                "behavior_state": behavior_states[row],
                "model_probability_positive": round(
                    float(probabilities[row, axis_index]), 6
                ),
                "explicit_label": explicit,
                "matches_query": explicit == desired_label,
                "contradicts_query": explicit == -desired_label,
                "evidence": list(evidence),
                "public_note": clean_text(record.get("behavior_text"))[:320],
                "split": split_by_row.get(int(row), "unknown"),
                "image_path": clean_text(
                    record.get("image_path") or records[int(row)].get("source_ref")
                ),
                "breed": breed["breed"],
                "color": clean_text(meta.get("color") or meta.get("colorCd")),
                "age": clean_text(meta.get("age") or meta.get("ageCd")),
                "weight": clean_text(meta.get("weight")),
            }
        )
    return output


def _conservative_scores(
    probabilities: np.ndarray,
    records: Sequence[Mapping[str, Any]],
    *,
    axis_index: int,
    desired_label: int,
) -> tuple[np.ndarray, list[str]]:
    scores = np.empty(len(records), dtype=np.float64)
    states: list[str] = []
    for row, record in enumerate(records):
        score, state = conservative_behavior_score(
            float(probabilities[row, axis_index]),
            int(record["labels"][axis_index]),
            desired_label=desired_label,
        )
        scores[row] = score
        states.append(state)
    return scores, states


def _evaluate_queries(
    *,
    query_rows: Sequence[Mapping[str, Any]],
    query_clip: np.ndarray,
    query_dino: np.ndarray,
    note_vectors: np.ndarray,
    clip_crops: np.ndarray,
    dino_vectors: np.ndarray,
    probabilities: np.ndarray,
    records: Sequence[Mapping[str, Any]],
    test_indices: Sequence[int],
) -> dict[str, Any]:
    systems = (
        "clip_text_to_image",
        "dino_flow_text_to_image",
        "clip_text_to_notice",
        "trained_behavior_head",
        "dino_flow_plus_behavior",
    )
    per_query: list[dict[str, Any]] = []
    aggregate: dict[str, dict[str, list[float]]] = {
        system: {"roc_auc": [], "average_precision": [], "ndcg@10": []}
        for system in systems
    }
    for query_index, query in enumerate(query_rows):
        axis_index = int(query["axis_index"])
        desired = int(query["desired_label"])
        known_rows = np.asarray(
            [
                row
                for row in test_indices
                if int(records[row]["labels"][axis_index]) != UNKNOWN
            ],
            dtype=np.int64,
        )
        truth = np.asarray(
            [int(records[row]["labels"][axis_index]) == desired for row in known_rows],
            dtype=np.int64,
        )
        clip_scores = clip_crops @ query_clip[query_index]
        dino_scores = dino_vectors @ query_dino[query_index]
        note_scores = note_vectors @ query_clip[query_index]
        behavior_scores = (
            probabilities[:, axis_index]
            if desired == POSITIVE
            else 1.0 - probabilities[:, axis_index]
        )
        fused_scores = 0.2 * percentile_scores(dino_scores) + 0.8 * behavior_scores
        score_tables = {
            "clip_text_to_image": clip_scores,
            "dino_flow_text_to_image": dino_scores,
            "clip_text_to_notice": note_scores,
            "trained_behavior_head": behavior_scores,
            "dino_flow_plus_behavior": fused_scores,
        }
        metrics: dict[str, Any] = {}
        for system, scores in score_tables.items():
            result = binary_ranking_metrics(truth, scores[known_rows])
            metrics[system] = result
            for metric in aggregate[system]:
                aggregate[system][metric].append(float(result[metric]))
        per_query.append(
            {
                **dict(query),
                "known_test_candidates": len(known_rows),
                "systems": metrics,
            }
        )

    macro = {
        system: {metric: finite_mean(values) for metric, values in metrics.items()}
        for system, metrics in aggregate.items()
    }
    return {"queries": per_query, "macro": macro}


def _demo_rankings(
    *,
    query_rows: Sequence[Mapping[str, Any]],
    query_clip: np.ndarray,
    query_dino: np.ndarray,
    clip_crops: np.ndarray,
    dino_vectors: np.ndarray,
    probabilities: np.ndarray,
    records: Sequence[Mapping[str, Any]],
    clip_metas: Sequence[Mapping[str, Any]],
    split_by_row: Mapping[int, str],
    topk: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for query_index, query in enumerate(query_rows):
        axis_index = int(query["axis_index"])
        desired = int(query["desired_label"])
        clip_scores = clip_crops @ query_clip[query_index]
        dino_scores = dino_vectors @ query_dino[query_index]
        behavior_scores, behavior_states = _conservative_scores(
            probabilities,
            records,
            axis_index=axis_index,
            desired_label=desired,
        )
        proposed_scores = 0.2 * percentile_scores(dino_scores) + 0.8 * behavior_scores
        systems = {
            "clip": clip_scores,
            "dino_clip": dino_scores,
            "dino_clip_behavior": proposed_scores,
        }
        output.append(
            {
                **dict(query),
                "systems": {
                    system: _decorate_top(
                        scores,
                        records=records,
                        clip_metas=clip_metas,
                        axis_index=axis_index,
                        desired_label=desired,
                        probabilities=probabilities,
                        behavior_scores=behavior_scores,
                        behavior_states=behavior_states,
                        split_by_row=split_by_row,
                        topk=topk,
                    )
                    for system, scores in systems.items()
                },
            }
        )
    return output


def _mixed_demo(
    *,
    cc0_manifest_path: Path,
    clip_encoder: Any,
    dino_encoder: Any,
    flow_head: Any,
    torch: Any,
    device: str,
    query_rows: Sequence[Mapping[str, Any]],
    query_clip: np.ndarray,
    query_dino: np.ndarray,
    clip_crops: np.ndarray,
    dino_vectors: np.ndarray,
    probabilities: np.ndarray,
    records: Sequence[Mapping[str, Any]],
    clip_metas: Sequence[Mapping[str, Any]],
    split_by_row: Mapping[int, str],
    topk: int,
) -> list[dict[str, Any]]:
    manifest = load_json(cc0_manifest_path)
    by_id = {str(item.get("page_id")): item for item in manifest.get("items", [])}
    selected_ids = ("15957924", "26592914", "65257801")
    cases: list[dict[str, Any]] = []
    for page_id in selected_ids:
        item = by_id.get(page_id)
        if not item:
            continue
        image_path = (ROOT / clean_text(item.get("local_path"))).resolve()
        if not image_path.is_file():
            continue
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            clip_image = clip_encoder.encode_image(image)
            dino_image = dino_encoder.encode(image)
        base_dino = dino_vectors @ dino_image[0]

        preferences: list[dict[str, Any]] = []
        for query_index, query in enumerate(query_rows):
            # Keep one readable preference per axis in the visual comparison.
            if int(query["desired_label"]) != POSITIVE:
                if query["axis"] != "activity":
                    continue
            if query["axis"] == "activity" and int(query["desired_label"]) == POSITIVE:
                keep = True
            elif query["axis"] == "activity":
                keep = True
            else:
                keep = int(query["desired_label"]) == POSITIVE
            if not keep:
                continue

            axis_index = int(query["axis_index"])
            desired = int(query["desired_label"])
            clip_mixed = blend_embeddings(
                clip_image, query_clip[query_index : query_index + 1], text_weight=0.2
            )
            dino_mixed = blend_embeddings(
                dino_image, query_dino[query_index : query_index + 1], text_weight=0.2
            )
            behavior_scores, behavior_states = _conservative_scores(
                probabilities,
                records,
                axis_index=axis_index,
                desired_label=desired,
            )
            proposed = 0.75 * percentile_scores(base_dino) + 0.25 * behavior_scores
            systems = {
                "clip": clip_crops @ clip_mixed[0],
                "dino_clip": dino_vectors @ dino_mixed[0],
                "dino_clip_behavior": proposed,
            }
            preferences.append(
                {
                    **dict(query),
                    "systems": {
                        system: _decorate_top(
                            scores,
                            records=records,
                            clip_metas=clip_metas,
                            axis_index=axis_index,
                            desired_label=desired,
                            probabilities=probabilities,
                            behavior_scores=behavior_scores,
                            behavior_states=behavior_states,
                            split_by_row=split_by_row,
                            topk=topk,
                        )
                        for system, scores in systems.items()
                    },
                }
            )
        cases.append(
            {
                "page_id": page_id,
                "title": clean_text(item.get("title")),
                "description": clean_text(item.get("description")),
                "license": clean_text(item.get("license_short_name")),
                "source_page_url": clean_text(item.get("source_page_url")),
                "image_path": str(image_path.relative_to(ROOT)).replace("\\", "/"),
                "preferences": preferences,
            }
        )
    return cases


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-metas", type=Path, default=ROOT / "data/dog_metas.json")
    parser.add_argument(
        "--clip-index", type=Path, default=ROOT / "data/dog_faiss.index"
    )
    parser.add_argument(
        "--dino-index", type=Path, default=DEFAULT_DINO_DIR / "dino.index"
    )
    parser.add_argument(
        "--dino-metas", type=Path, default=DEFAULT_DINO_DIR / "dino_metas.json"
    )
    parser.add_argument(
        "--dino-manifest", type=Path, default=DEFAULT_DINO_DIR / "dino_manifest.json"
    )
    parser.add_argument(
        "--flow-report",
        type=Path,
        default=DEFAULT_FLOW_DIR / "alignment_training_report.json",
    )
    parser.add_argument(
        "--cc0-manifest",
        type=Path,
        default=Path(__file__).resolve().parent
        / "artifacts"
        / "cc0_dog_photos_diverse"
        / "manifest.json",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", default="behavior-pilot-v1")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size <= 0 or args.hidden_dim <= 0 or args.max_epochs <= 0:
        raise ValueError("batch-size, hidden-dim, and max-epochs must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.topk <= 0:
        raise ValueError("invalid optimization or ranking argument")

    started_total = time.perf_counter()
    torch = __import__("torch")
    faiss = __import__("faiss")
    device = (
        "cuda"
        if args.device == "auto" and bool(torch.cuda.is_available())
        else ("cpu" if args.device == "auto" else args.device)
    )
    seed_value = int(hashlib.sha256(args.seed.encode("utf-8")).hexdigest()[:8], 16)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed_value)

    clip_metas = load_json(args.clip_metas)
    dino_metas = load_json(args.dino_metas)
    dino_manifest = load_json(args.dino_manifest)
    if not isinstance(clip_metas, list) or not isinstance(dino_metas, list):
        raise ValueError("metadata files must contain lists")
    clip_index = safe_read_faiss_index(args.clip_index, faiss.read_index)
    dino_index = safe_read_faiss_index(args.dino_index, faiss.read_index)
    if int(clip_index.ntotal) != len(clip_metas) or int(dino_index.ntotal) != len(
        dino_metas
    ):
        raise ValueError("index and metadata counts differ")

    records = build_behavior_records(clip_metas, dino_metas)
    if len(records) != len(dino_metas):
        raise ValueError("behavior record coverage must match the DINO crop corpus")
    for record, dino_meta in zip(records, dino_metas):
        record["source_ref"] = clean_text(dino_meta.get("source_ref"))
        record["image_path"] = clean_text(dino_meta.get("source_ref"))
    splits = stratified_text_group_split(records, seed=args.seed)
    split_by_row = {row: split for split, rows in splits.items() for row in rows}

    clip_encoder = ClipEncoder("ViT-B/32", device=device)
    encoded_started = time.perf_counter()
    note_vectors = encode_unique_notes(
        clip_encoder, records, batch_size=args.batch_size
    )
    note_encode_seconds = time.perf_counter() - encoded_started
    labels_np = np.asarray([record["labels"] for record in records], dtype=np.int64)

    train_indices = np.asarray(
        [row for row in splits["train"] if np.any(labels_np[row] != UNKNOWN)],
        dtype=np.int64,
    )
    validation_indices = np.asarray(
        [row for row in splits["validation"] if np.any(labels_np[row] != UNKNOWN)],
        dtype=np.int64,
    )
    test_indices = np.asarray(
        [row for row in splits["test"] if np.any(labels_np[row] != UNKNOWN)],
        dtype=np.int64,
    )
    if not len(train_indices) or not len(validation_indices) or not len(test_indices):
        raise ValueError("train, validation, and test require observed behavior rows")

    head = make_behavior_head(
        input_dim=clip_encoder.text_dimension,
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    train_x = torch.as_tensor(note_vectors[train_indices], device=device)
    train_y = torch.as_tensor(labels_np[train_indices], device=device)
    validation_x = torch.as_tensor(note_vectors[validation_indices], device=device)
    validation_y = torch.as_tensor(labels_np[validation_indices], device=device)

    best_state: dict[str, Any] | None = None
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    training_started = time.perf_counter()
    for epoch in range(1, args.max_epochs + 1):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        train_logits = head(train_x)
        train_loss = _balanced_masked_loss(torch, train_logits, train_y)
        train_loss.backward()
        optimizer.step()

        head.eval()
        with torch.inference_mode():
            validation_loss = _balanced_masked_loss(
                torch, head(validation_x), validation_y
            )
        current = float(validation_loss.detach().cpu())
        if epoch == 1 or epoch % 10 == 0:
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": round(float(train_loss.detach().cpu()), 6),
                    "validation_loss": round(current, 6),
                }
            )
        if current < best_loss - 1e-5:
            best_loss = current
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in head.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    training_seconds = time.perf_counter() - training_started
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    head.load_state_dict(best_state)
    head.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "behavior_head.pt"
    save_checkpoint_atomic(
        {
            "schema_version": "behavior-head.v1",
            "input_dim": clip_encoder.text_dimension,
            "hidden_dim": args.hidden_dim,
            "axis_keys": [axis.key for axis in BEHAVIOR_AXES],
            "state_dict": best_state,
        },
        checkpoint_path,
        torch,
    )

    with torch.inference_mode():
        logits = head(torch.as_tensor(note_vectors, device=device))
        probabilities = torch.sigmoid(logits).detach().float().cpu().numpy()
    classification = {
        "validation": _classification_metrics(
            labels_np[validation_indices], probabilities[validation_indices]
        ),
        "test": _classification_metrics(
            labels_np[test_indices], probabilities[test_indices]
        ),
    }

    source_indices = np.asarray(
        [int(record["source_meta_index"]) for record in records], dtype=np.int64
    )
    clip_matrix = reconstruct_index(clip_index)
    clip_crops = normalize_rows(clip_matrix[source_indices])
    dino_vectors = reconstruct_index(dino_index)

    flow_head, flow_report = _load_flow_head(
        report_path=args.flow_report, device=device, torch=torch
    )
    query_rows = _axis_query_rows()
    query_clip = clip_encoder.encode_text_batch(
        [clean_text(row["text"]) for row in query_rows]
    )
    with torch.inference_mode():
        query_dino = (
            project_embeddings(flow_head, torch.as_tensor(query_clip, device=device))
            .detach()
            .float()
            .cpu()
            .numpy()
        )

    query_evaluation = _evaluate_queries(
        query_rows=query_rows,
        query_clip=query_clip,
        query_dino=query_dino,
        note_vectors=note_vectors,
        clip_crops=clip_crops,
        dino_vectors=dino_vectors,
        probabilities=probabilities,
        records=records,
        test_indices=test_indices,
    )
    text_demo = _demo_rankings(
        query_rows=query_rows,
        query_clip=query_clip,
        query_dino=query_dino,
        clip_crops=clip_crops,
        dino_vectors=dino_vectors,
        probabilities=probabilities,
        records=records,
        clip_metas=clip_metas,
        split_by_row=split_by_row,
        topk=args.topk,
    )

    dino_model = dino_manifest.get("model") or {}
    dino_encoder = DinoEncoder(
        clean_text(dino_model.get("id")),
        device=device,
        revision=clean_text(dino_model.get("resolved_revision")) or None,
        local_files_only=args.local_files_only,
    )
    mixed_demo = _mixed_demo(
        cc0_manifest_path=args.cc0_manifest,
        clip_encoder=clip_encoder,
        dino_encoder=dino_encoder,
        flow_head=flow_head,
        torch=torch,
        device=device,
        query_rows=query_rows,
        query_clip=query_clip,
        query_dino=query_dino,
        clip_crops=clip_crops,
        dino_vectors=dino_vectors,
        probabilities=probabilities,
        records=records,
        clip_metas=clip_metas,
        split_by_row=split_by_row,
        topk=args.topk,
    )

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "scope": {
            "candidate_corpus": "1,002 local shelter-dog crops",
            "behavior_source": "public/shelter notice text only",
            "missing_label_policy": "unknown; excluded from masked supervised loss",
            "conflict_policy": "unknown; conflicting explicit phrases abstain",
            "foundation_encoders_frozen": True,
            "production_route_changed": False,
        },
        "axes": [
            {
                "key": axis.key,
                "positive": axis.positive_label,
                "negative": axis.negative_label,
            }
            for axis in BEHAVIOR_AXES
        ],
        "dataset": {
            "record_count": len(records),
            "nonempty_behavior_text_count": sum(
                bool(record["behavior_text"]) for record in records
            ),
            "all_label_counts": _label_counts(records, range(len(records))),
            "split_row_counts": {key: len(value) for key, value in splits.items()},
            "split_observed_row_counts": {
                "train": len(train_indices),
                "validation": len(validation_indices),
                "test": len(test_indices),
            },
            "split_label_counts": {
                key: _label_counts(records, rows) for key, rows in splits.items()
            },
            "duplicate_behavior_text_cross_split": False,
        },
        "training": {
            "architecture": f"CLIP-note {clip_encoder.text_dimension}->{args.hidden_dim}->{len(BEHAVIOR_AXES)}",
            "objective": "class-balanced masked BCE on explicit positive/negative evidence only",
            "parameter_count": sum(
                parameter.numel() for parameter in head.parameters()
            ),
            "device": device,
            "best_epoch": best_epoch,
            "epochs_completed": epoch,
            "best_validation_loss": round(best_loss, 6),
            "training_seconds": round(training_seconds, 4),
            "note_encoding_seconds": round(note_encode_seconds, 4),
            "checkpoint": checkpoint_path.name,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "history": history,
        },
        "models": {
            "clip": {"id": "ViT-B/32", "frozen": True},
            "dino": dino_model,
            "flow_alignment": {
                "frozen_during_behavior_training": True,
                "checkpoint": (
                    (flow_report.get("architectures") or {}).get("flow") or {}
                ).get("checkpoint"),
            },
        },
        "classification": classification,
        "query_evaluation": query_evaluation,
        "fusion": {
            "text_only_behavior_weight": 0.8,
            "mixed_image_behavior_weight": 0.25,
            "weights_tuned_on_test": False,
            "unknown_inference_band": [0.3, 0.7],
            "ordering_contract": "explicit match > unknown/inferred > explicit contradiction",
        },
        "text_demo": text_demo,
        "mixed_demo": mixed_demo,
        "limitations": [
            "Weak labels come from the same public notice text used by the behavior branch; this is silver-label consistency, not an independent temperament validation.",
            "Public notes are incomplete and context-dependent; unknown never means absence of a trait.",
            "The three supervised axes were selected because this corpus has both explicit ends; fear/watchfulness remains evidence-only because no reliable opposite examples were present.",
            "Static DINO/CLIP image features are not interpreted as proof of personality.",
            "The small people-social and activity test subsets make their metrics high variance.",
        ],
        "elapsed_seconds": round(time.perf_counter() - started_total, 3),
    }
    report_path = args.output_dir / "behavior_pilot_report.json"
    write_json_atomic(report_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    summary = {
        "schema_version": report["schema_version"],
        "dataset": report["dataset"],
        "training": report["training"],
        "classification": report["classification"],
        "query_macro": report["query_evaluation"]["macro"],
        "elapsed_seconds": report["elapsed_seconds"],
        "output": str(args.output_dir / "behavior_pilot_report.json"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
