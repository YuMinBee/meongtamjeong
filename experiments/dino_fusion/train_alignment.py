"""Train frozen-encoder CLIP-text-to-DINO linear, MLP, or ODE-flow heads."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.dino_fusion.alignment import (  # noqa: E402
    alignment_attributes,
    collect_heldout_notice_ids,
    evaluation_prompts,
    make_projection_head,
    project_embeddings,
    semantic_signature,
    stratified_notice_split,
    training_prompts,
)
from experiments.dino_fusion.build_index import (  # noqa: E402
    sha256_file,
    write_json_atomic,
)
from experiments.dino_fusion.core import (  # noqa: E402
    ClipEncoder,
    clean_text,
    normalize_rows,
    safe_read_faiss_index,
)


DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
REPORT_SCHEMA_VERSION = "dino-text-alignment-training.v2"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def reconstruct_index(index: Any) -> np.ndarray:
    matrix = np.empty((int(index.ntotal), int(index.d)), dtype=np.float32)
    for row in range(int(index.ntotal)):
        index.reconstruct(row, matrix[row])
    return normalize_rows(matrix)


def split_hash(records: Sequence[Mapping[str, Any]], indices: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for index in sorted(indices, key=lambda value: records[value]["notice_id"]):
        digest.update(clean_text(records[index]["notice_id"]).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _year_from_report(report: Mapping[str, Any]) -> int:
    text = clean_text(report.get("reference_date"))
    return int(text[:4]) if len(text) >= 4 and text[:4].isdigit() else 2026


def build_records(
    *,
    clip_metas: Sequence[Mapping[str, Any]],
    dino_metas: Sequence[Mapping[str, Any]],
    heldout_ids: set[str],
    reference_year: int,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    records: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    seen_notices: set[str] = set()
    for dino_row, dino_meta in enumerate(dino_metas):
        dog_id = clean_text(dino_meta.get("notice_id") or dino_meta.get("desertionNo"))
        if not dog_id or dog_id in seen_notices:
            exclusions["missing_or_duplicate_notice"] += 1
            continue
        source_index = dino_meta.get("source_meta_index")
        if not isinstance(source_index, int) or not 0 <= source_index < len(clip_metas):
            exclusions["invalid_source_meta_index"] += 1
            continue
        seen_notices.add(dog_id)
        if dog_id in heldout_ids:
            exclusions["heldout_notice"] += 1
            continue
        attributes = alignment_attributes(
            clip_metas[source_index], reference_year=reference_year
        )
        if attributes is None:
            exclusions["missing_color_or_size"] += 1
            continue
        records.append(
            {
                "notice_id": dog_id,
                "dino_row": dino_row,
                "source_meta_index": source_index,
                "attributes": attributes,
                "signature": semantic_signature(attributes),
            }
        )
    return records, exclusions


def encode_prompt_table(
    encoder: Any,
    texts: Sequence[str],
    *,
    batch_size: int,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        chunks.append(encoder.encode_text_batch(texts[start : start + batch_size]))
    return normalize_rows(np.concatenate(chunks, axis=0))


def multipositive_loss(
    torch: Any,
    mapped_text: Any,
    visual: Any,
    text_labels: Any,
    visual_labels: Any,
    centroids: Any,
    *,
    temperature: float,
    centroid_weight: float,
    symmetric: bool,
) -> Any:
    logits = mapped_text @ visual.T / temperature
    positives = text_labels[:, None].eq(visual_labels[None, :])
    negative_inf = torch.finfo(logits.dtype).min
    positive_logits = logits.masked_fill(~positives, negative_inf)
    text_to_image = -(
        torch.logsumexp(positive_logits, dim=1) - torch.logsumexp(logits, dim=1)
    ).mean()
    loss = text_to_image
    if symmetric:
        positive_logits_t = logits.T.masked_fill(~positives.T, negative_inf)
        image_to_text = -(
            torch.logsumexp(positive_logits_t, dim=1) - torch.logsumexp(logits.T, dim=1)
        ).mean()
        loss = 0.5 * (text_to_image + image_to_text)
    centroid_loss = 1.0 - (mapped_text * centroids[text_labels]).sum(dim=1).mean()
    return loss + centroid_weight * centroid_loss


def save_checkpoint_atomic(payload: Mapping[str, Any], path: Path, torch: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-metas", type=Path, default=ROOT / "data/dog_metas.json")
    parser.add_argument(
        "--heldout-report",
        type=Path,
        default=ROOT / "docs/evaluation/heldout_image_retrieval.appearance_v1.json",
    )
    parser.add_argument(
        "--dino-index", type=Path, default=DEFAULT_ARTIFACT_DIR / "dino.index"
    )
    parser.add_argument(
        "--dino-metas", type=Path, default=DEFAULT_ARTIFACT_DIR / "dino_metas.json"
    )
    parser.add_argument(
        "--dino-manifest",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "dino_manifest.json",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--output-name", default="alignment_training_report.json")
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--architecture",
        action="append",
        choices=("linear", "mlp", "flow"),
        dest="architectures",
    )
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--flow-width", type=int, default=96)
    parser.add_argument("--flow-depth", type=int, default=4)
    parser.add_argument("--flow-steps", type=int, default=8)
    parser.add_argument("--flow-time-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--centroid-weight", type=float, default=0.25)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--validation-every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--prompt-batch-size", type=int, default=256)
    parser.add_argument("--seed", default="dino-alignment-pilot-v1")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.epochs <= 0 or args.validation_every <= 0 or args.patience <= 0:
        raise ValueError("epochs, validation-every, and patience must be positive")
    if (
        min(
            args.hidden_dim,
            args.flow_width,
            args.flow_depth,
            args.flow_steps,
            args.flow_time_dim,
            args.prompt_batch_size,
        )
        <= 0
    ):
        raise ValueError(
            "head dimensions, flow settings, and batch size must be positive"
        )
    if args.learning_rate <= 0 or args.temperature <= 0:
        raise ValueError("learning-rate and temperature must be positive")

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
    heldout = load_json(args.heldout_report)
    dino_manifest = load_json(args.dino_manifest)
    if not isinstance(clip_metas, list) or not isinstance(dino_metas, list):
        raise ValueError("index metadata must be JSON lists")
    dino_index = safe_read_faiss_index(args.dino_index, faiss.read_index)
    if int(dino_index.ntotal) != len(dino_metas):
        raise ValueError("DINO index/metadata row mismatch")
    dino_vectors = reconstruct_index(dino_index)
    heldout_ids = collect_heldout_notice_ids(heldout)
    reference_year = _year_from_report(heldout)
    records, exclusions = build_records(
        clip_metas=clip_metas,
        dino_metas=dino_metas,
        heldout_ids=heldout_ids,
        reference_year=reference_year,
    )
    train_indices, validation_indices = stratified_notice_split(
        records,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    if not train_indices or not validation_indices:
        raise ValueError("training and validation splits must both be non-empty")

    signatures = sorted({records[index]["signature"] for index in train_indices})
    signature_to_label = {
        signature: index for index, signature in enumerate(signatures)
    }
    signature_attributes = {
        record["signature"]: record["attributes"] for record in records
    }
    prompt_texts: list[str] = []
    train_prompt_rows: dict[str, list[int]] = {}
    eval_prompt_rows: dict[str, list[int]] = {}
    for signature in signatures:
        train_rows: list[int] = []
        grouped = training_prompts(signature_attributes[signature])
        for language in ("english", "korean"):
            for prompt in grouped[language]:
                train_rows.append(len(prompt_texts))
                prompt_texts.append(prompt)
        train_prompt_rows[signature] = train_rows
        eval_rows: list[int] = []
        heldout_prompts = evaluation_prompts(signature_attributes[signature])
        for language in ("english", "korean"):
            eval_rows.append(len(prompt_texts))
            prompt_texts.append(heldout_prompts[language])
        eval_prompt_rows[signature] = eval_rows

    clip_encoder = ClipEncoder(args.clip_model, device=device)
    prompt_vectors = encode_prompt_table(
        clip_encoder, prompt_texts, batch_size=args.prompt_batch_size
    )
    input_dim = int(prompt_vectors.shape[1])
    output_dim = int(dino_vectors.shape[1])

    train_visual_np = dino_vectors[
        [int(records[index]["dino_row"]) for index in train_indices]
    ]
    validation_visual_np = dino_vectors[
        [int(records[index]["dino_row"]) for index in validation_indices]
    ]
    train_labels_np = np.asarray(
        [signature_to_label[records[index]["signature"]] for index in train_indices],
        dtype=np.int64,
    )
    validation_labels_np = np.asarray(
        [
            signature_to_label[records[index]["signature"]]
            for index in validation_indices
        ],
        dtype=np.int64,
    )
    train_prompt_ids_np = np.asarray(
        [train_prompt_rows[records[index]["signature"]] for index in train_indices],
        dtype=np.int64,
    )
    validation_prompt_ids_np = np.asarray(
        [eval_prompt_rows[records[index]["signature"]] for index in validation_indices],
        dtype=np.int64,
    )

    train_visual = torch.as_tensor(train_visual_np, device=device)
    validation_visual = torch.as_tensor(validation_visual_np, device=device)
    train_labels = torch.as_tensor(train_labels_np, device=device)
    validation_labels = torch.as_tensor(validation_labels_np, device=device)
    train_prompt_ids = torch.as_tensor(train_prompt_ids_np, device=device)
    validation_prompt_ids = torch.as_tensor(validation_prompt_ids_np, device=device)
    all_prompt_vectors = torch.as_tensor(prompt_vectors, device=device)

    centroids = torch.zeros((len(signatures), output_dim), device=device)
    for label in range(len(signatures)):
        centroids[label] = train_visual[train_labels.eq(label)].mean(dim=0)
    centroids = torch.nn.functional.normalize(centroids, dim=-1)

    architecture_reports: dict[str, Any] = {}
    architectures = list(dict.fromkeys(args.architectures or ("linear", "mlp")))
    architecture_seed_offsets = {"linear": 0, "mlp": 1, "flow": 2}
    for architecture in architectures:
        torch.manual_seed(seed_value + architecture_seed_offsets[architecture])
        head = make_projection_head(
            architecture,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=args.hidden_dim,
            flow_width=args.flow_width,
            flow_depth=args.flow_depth,
            flow_steps=args.flow_steps,
            flow_time_dim=args.flow_time_dim,
        ).to(device)
        optimizer = torch.optim.AdamW(
            head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        generator = torch.Generator(device=device)
        generator.manual_seed(seed_value + 7)
        best_state: dict[str, Any] | None = None
        best_validation_loss = float("inf")
        best_epoch = 0
        stale_checks = 0
        last_train_loss = float("nan")
        started = time.perf_counter()

        for epoch in range(1, args.epochs + 1):
            head.train()
            variants = torch.randint(
                train_prompt_ids.shape[1],
                (train_prompt_ids.shape[0],),
                generator=generator,
                device=device,
            )
            selected_ids = train_prompt_ids[
                torch.arange(train_prompt_ids.shape[0], device=device), variants
            ]
            mapped = project_embeddings(head, all_prompt_vectors[selected_ids])
            loss = multipositive_loss(
                torch,
                mapped,
                train_visual,
                train_labels,
                train_labels,
                centroids,
                temperature=args.temperature,
                centroid_weight=args.centroid_weight,
                symmetric=True,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            last_train_loss = float(loss.detach().cpu())

            if epoch % args.validation_every != 0 and epoch != args.epochs:
                continue
            head.eval()
            with torch.inference_mode():
                expanded_ids = validation_prompt_ids.reshape(-1)
                expanded_labels = validation_labels.repeat_interleave(
                    validation_prompt_ids.shape[1]
                )
                validation_mapped = project_embeddings(
                    head, all_prompt_vectors[expanded_ids]
                )
                validation_loss = multipositive_loss(
                    torch,
                    validation_mapped,
                    validation_visual,
                    expanded_labels,
                    validation_labels,
                    centroids,
                    temperature=args.temperature,
                    centroid_weight=args.centroid_weight,
                    symmetric=False,
                )
                validation_value = float(validation_loss.cpu())
            if validation_value < best_validation_loss - 1e-5:
                best_validation_loss = validation_value
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in head.state_dict().items()
                }
                stale_checks = 0
            else:
                stale_checks += 1
            if stale_checks >= args.patience:
                break

        if best_state is None:
            raise RuntimeError(f"{architecture} training produced no checkpoint")
        elapsed = time.perf_counter() - started
        head.load_state_dict(best_state)
        checkpoint_path = args.output_dir / f"alignment_{architecture}.pt"
        save_checkpoint_atomic(
            {
                "schema_version": REPORT_SCHEMA_VERSION,
                "architecture": architecture,
                "input_dim": input_dim,
                "output_dim": output_dim,
                "hidden_dim": args.hidden_dim,
                "flow_width": args.flow_width,
                "flow_depth": args.flow_depth,
                "flow_steps": args.flow_steps,
                "flow_time_dim": args.flow_time_dim,
                "state_dict": best_state,
            },
            checkpoint_path,
            torch,
        )
        architecture_reports[architecture] = {
            "parameter_count": sum(
                parameter.numel() for parameter in head.parameters()
            ),
            "epochs_completed": epoch,
            "best_epoch": best_epoch,
            "best_validation_loss": round(best_validation_loss, 6),
            "last_train_loss": round(last_train_loss, 6),
            "training_seconds": round(elapsed, 3),
            "checkpoint": checkpoint_path.name,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "configuration": (
                dict(head.flow_config) if hasattr(head, "flow_config") else {}
            ),
        }

    signature_counts = Counter(record["signature"] for record in records)
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "training_required": True,
        "encoders": {
            "clip_text": {
                "id": args.clip_model,
                "dimension": input_dim,
                "frozen": True,
            },
            "dino_visual": {
                **(dino_manifest.get("model") or {}),
                "dimension": output_dim,
                "frozen": True,
            },
        },
        "dataset": {
            "dino_vector_count": int(dino_index.ntotal),
            "eligible_notice_count": len(records),
            "heldout_pool_notice_count": len(heldout_ids),
            "train_notice_count": len(train_indices),
            "validation_notice_count": len(validation_indices),
            "semantic_signature_count": len(signatures),
            "singleton_signature_count": sum(
                count == 1 for count in signature_counts.values()
            ),
            "exclusions": dict(sorted(exclusions.items())),
            "train_notice_sha256": split_hash(records, train_indices),
            "validation_notice_sha256": split_hash(records, validation_indices),
        },
        "prompt_protocol": {
            "languages": ["english", "korean"],
            "training_paraphrases_per_language": 3,
            "validation_paraphrases_per_language": 1,
            "attributes": ["public_color", "weight_derived_size", "age_group"],
            "validation_phrasing_seen_during_training": False,
        },
        "optimization": {
            "objective": "symmetric_multi_positive_info_nce_plus_dino_centroid_cosine",
            "seed": args.seed,
            "validation_fraction": args.validation_fraction,
            "max_epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "temperature": args.temperature,
            "centroid_weight": args.centroid_weight,
            "flow_width": args.flow_width,
            "flow_depth": args.flow_depth,
            "flow_steps": args.flow_steps,
            "flow_time_dim": args.flow_time_dim,
            "validation_every": args.validation_every,
            "early_stopping_patience_checks": args.patience,
            "device": device,
        },
        "architectures": architecture_reports,
        "limitations": [
            "Prompts are controlled metadata-derived language, not independently authored queries.",
            "Silver labels cover coarse color, size, and age rather than breed or fine markings.",
            "Validation holds out notices and phrasing, but semantic signatures overlap training by design.",
            (
                "The DINOv2 backbone is an infrastructure baseline rather than the target DINOv3 model."
                if "dinov2" in clean_text((dino_manifest.get("model") or {}).get("id"))
                else "The DINOv3 result remains a small pilot and is not a production-quality claim."
            ),
        ],
    }
    write_json_atomic(args.output_dir / args.output_name, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
