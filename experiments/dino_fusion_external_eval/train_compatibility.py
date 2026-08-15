"""Train a small CLIP-description compatibility head with frozen encoders."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.dino_fusion.core import ClipEncoder  # noqa: E402
from experiments.dino_fusion_external_eval.compatibility import (  # noqa: E402
    COMPATIBILITY_AXES,
    balanced_masked_loss,
    behavior_evidence_text,
    classification_metrics,
    make_compatibility_head,
)
from experiments.dino_fusion_external_eval.prepare_benchmark import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    sha256_file,
    write_json_atomic,
)


DEFAULT_RECORDS = DEFAULT_OUTPUT_DIR / "text_records.json"
REPORT_SCHEMA_VERSION = "petfinder-compatibility-head.v1"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_numpy_atomic(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{uuid4().hex}.npy")
    try:
        np.save(temporary, array, allow_pickle=False)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _save_checkpoint_atomic(payload: dict[str, Any], path: Path, torch: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _sync(torch: Any, device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def encode_descriptions(
    records: list[dict[str, Any]],
    *,
    output_dir: Path,
    batch_size: int,
    device: str,
    text_mode: str,
) -> tuple[np.ndarray, dict[str, Any], ClipEncoder]:
    if text_mode not in {"raw", "evidence"}:
        raise ValueError("text_mode must be raw or evidence")
    cache_stem = "clip_description_embeddings"
    if text_mode == "evidence":
        cache_stem = "clip_evidence_embeddings"
    cache_path = output_dir / f"{cache_stem}.npy"
    cache_meta_path = output_dir / f"{cache_stem}.json"
    records_path = output_dir / "text_records.json"
    expected_hash = sha256_file(records_path)
    if cache_path.is_file() and cache_meta_path.is_file():
        cache_meta = _load_json(cache_meta_path)
        embeddings = np.load(cache_path, allow_pickle=False)
        expected_shape = (len(records), 512)
        if (
            cache_meta.get("records_sha256") == expected_hash
            and tuple(embeddings.shape) == expected_shape
        ):
            model_started = time.perf_counter()
            encoder = ClipEncoder(device=device)
            model_seconds = time.perf_counter() - model_started
            return embeddings, {
                **cache_meta,
                "cache_hit": True,
                "model_initialization_seconds": round(model_seconds, 6),
                "actual_this_run_seconds": 0.0,
            }, encoder

    model_started = time.perf_counter()
    encoder = ClipEncoder(device=device)
    model_seconds = time.perf_counter() - model_started
    texts = [str(record["description"]) for record in records]
    if text_mode == "evidence":
        texts = [behavior_evidence_text(text) for text in texts]
    embeddings = np.empty((len(texts), encoder.text_dimension), dtype=np.float32)
    probe_count = min(len(texts), max(batch_size, 2048))

    _sync(encoder._torch, encoder.device)
    encode_started = time.perf_counter()
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        embeddings[start : start + len(batch)] = encoder.encode_text_batch(batch)
        if start == 0:
            _sync(encoder._torch, encoder.device)
            first_batch_seconds = time.perf_counter() - encode_started
    _sync(encoder._torch, encoder.device)
    encode_seconds = time.perf_counter() - encode_started

    measured_probe_seconds = max(
        first_batch_seconds,
        encode_seconds * (probe_count / max(1, len(texts))),
    )
    projected_seconds = measured_probe_seconds * len(texts) / max(1, probe_count)
    _save_numpy_atomic(embeddings, cache_path)
    cache_meta = {
        "encoder": "OpenAI CLIP ViT-B/32 text tower",
        "encoder_frozen": True,
        "text_mode": text_mode,
        "dimension": int(encoder.text_dimension),
        "records": len(records),
        "records_sha256": expected_hash,
        "batch_size": batch_size,
        "probe_records": probe_count,
        "probe_seconds": round(measured_probe_seconds, 6),
        "projected_full_encode_seconds": round(projected_seconds, 6),
        "actual_full_encode_seconds": round(encode_seconds, 6),
        "cache_path": cache_path.name,
        "cache_sha256": sha256_file(cache_path),
        "cache_hit": False,
        "model_initialization_seconds": round(model_seconds, 6),
        "actual_this_run_seconds": round(encode_seconds, 6),
    }
    write_json_atomic(cache_meta, cache_meta_path, pretty=True)
    return embeddings, cache_meta, encoder


def _indices(records: list[dict[str, Any]], split: str) -> np.ndarray:
    return np.asarray(
        [index for index, record in enumerate(records) if record["split"] == split],
        dtype=np.int64,
    )


def _predict(torch: Any, head: Any, features: Any) -> np.ndarray:
    head.eval()
    with torch.inference_mode():
        values = torch.sigmoid(head(features)).detach().float().cpu().numpy()
    return values


def train(
    *,
    records_path: Path,
    output_dir: Path,
    device: str = "auto",
    text_batch_size: int = 512,
    epochs: int = 250,
    patience: int = 30,
    learning_rate: float = 1e-3,
    seed: int = 20260815,
    text_mode: str = "raw",
) -> dict[str, Any]:
    run_started = time.perf_counter()
    records = _load_json(records_path)
    if not isinstance(records, list) or not records:
        raise ValueError("text_records.json must be a non-empty list")
    if records_path.resolve() != (output_dir / "text_records.json").resolve():
        raise ValueError("records must live in output_dir for cache provenance")

    embeddings, embedding_timing, encoder = encode_descriptions(
        records,
        output_dir=output_dir,
        batch_size=text_batch_size,
        device=device,
        text_mode=text_mode,
    )
    torch = encoder._torch
    resolved_device = encoder.device
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if resolved_device == "cuda":
        torch.cuda.manual_seed_all(seed)

    labels = np.asarray([record["labels"] for record in records], dtype=np.int64)
    split_indices = {
        split: _indices(records, split) for split in ("train", "validation", "test")
    }
    if any(not len(values) for values in split_indices.values()):
        raise ValueError("all three organization-level splits must be non-empty")

    feature_tensor = torch.as_tensor(embeddings, device=resolved_device)
    label_tensor = torch.as_tensor(labels, device=resolved_device)
    index_tensors = {
        split: torch.as_tensor(values, device=resolved_device)
        for split, values in split_indices.items()
    }
    head = make_compatibility_head(input_dim=embeddings.shape[1]).to(resolved_device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=1e-4)

    best_score = -float("inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    _sync(torch, resolved_device)
    train_started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        train_logits = head(feature_tensor[index_tensors["train"]])
        loss = balanced_masked_loss(
            torch,
            train_logits,
            label_tensor[index_tensors["train"]],
        )
        loss.backward()
        optimizer.step()

        validation_probabilities = _predict(
            torch, head, feature_tensor[index_tensors["validation"]]
        )
        validation_metrics = classification_metrics(
            labels[split_indices["validation"]], validation_probabilities
        )
        score = float(validation_metrics["macro_roc_auc"])
        if epoch == 1 or epoch % 10 == 0:
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": round(float(loss.detach().cpu()), 6),
                    "validation_macro_roc_auc": round(score, 6),
                }
            )
        if score > best_score + 1e-5:
            best_score = score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in head.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break
    _sync(torch, resolved_device)
    optimization_seconds = time.perf_counter() - train_started
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    head.load_state_dict(best_state)

    metrics: dict[str, Any] = {}
    for split, indices in split_indices.items():
        probabilities = _predict(torch, head, feature_tensor[index_tensors[split]])
        metrics[split] = classification_metrics(labels[indices], probabilities)

    checkpoint_name = (
        "compatibility_head.pt"
        if text_mode == "raw"
        else "compatibility_evidence_head.pt"
    )
    report_name = (
        "compatibility_training_report.json"
        if text_mode == "raw"
        else "compatibility_evidence_training_report.json"
    )
    checkpoint_path = output_dir / checkpoint_name
    checkpoint = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "axis_keys": [axis.key for axis in COMPATIBILITY_AXES],
        "input_dim": int(embeddings.shape[1]),
        "hidden_dim": 128,
        "output_dim": len(COMPATIBILITY_AXES),
        "encoder": "OpenAI CLIP ViT-B/32 text tower",
        "text_mode": text_mode,
        "state_dict": best_state,
    }
    _save_checkpoint_atomic(checkpoint, checkpoint_path, torch)

    gpu_name = (
        torch.cuda.get_device_name(0) if resolved_device == "cuda" else "CPU"
    )
    total_seconds = time.perf_counter() - run_started
    projected_end_to_end = (
        float(embedding_timing["model_initialization_seconds"])
        + float(embedding_timing["projected_full_encode_seconds"])
        + optimization_seconds
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "decision_scope": "experimental only; production integration is not implied",
        "dataset": {
            "records": len(records),
            "records_sha256": sha256_file(records_path),
            "organization_grouped_splits": {
                split: int(len(indices)) for split, indices in split_indices.items()
            },
        },
        "model": {
            "text_encoder": "OpenAI CLIP ViT-B/32",
            "text_encoder_frozen": True,
            "text_preprocessing": text_mode,
            "head": "Linear(512,128)-GELU-Dropout(0.1)-Linear(128,4)",
            "parameter_count": sum(parameter.numel() for parameter in head.parameters()),
            "checkpoint": checkpoint_path.name,
            "checkpoint_sha256": sha256_file(checkpoint_path),
        },
        "training": {
            "seed": seed,
            "learning_rate": learning_rate,
            "maximum_epochs": epochs,
            "epochs_completed": epoch,
            "patience": patience,
            "best_epoch": best_epoch,
            "best_validation_macro_roc_auc": round(best_score, 6),
            "history": history,
        },
        "metrics": metrics,
        "timing": {
            "hardware": gpu_name,
            "device": resolved_device,
            "model_initialization_seconds": embedding_timing[
                "model_initialization_seconds"
            ],
            "text_encoding_probe_seconds": embedding_timing["probe_seconds"],
            "projected_full_text_encoding_seconds": embedding_timing[
                "projected_full_encode_seconds"
            ],
            "actual_full_text_encoding_seconds": embedding_timing[
                "actual_full_encode_seconds"
            ],
            "text_embedding_cache_hit": embedding_timing["cache_hit"],
            "head_optimization_seconds": round(optimization_seconds, 6),
            "projected_fresh_end_to_end_seconds": round(projected_end_to_end, 6),
            "actual_this_run_end_to_end_seconds": round(total_seconds, 6),
        },
        "limitations": [
            "Labels are shelter-entered structured fields, not clinical observations.",
            "Descriptions can explicitly repeat labels; this head extracts profile evidence.",
            "Organization grouping reduces but cannot eliminate template leakage.",
            "Unknown values are excluded from supervised loss and classification metrics.",
        ],
    }
    write_json_atomic(report, output_dir / report_name, pretty=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--text-batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--text-mode", choices=("raw", "evidence"), default="raw")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = train(
        records_path=args.records,
        output_dir=args.output_dir,
        device=args.device,
        text_batch_size=args.text_batch_size,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        text_mode=args.text_mode,
    )
    print(json.dumps({"metrics": report["metrics"], "timing": report["timing"]}, indent=2))


if __name__ == "__main__":
    main()
