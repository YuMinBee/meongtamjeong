"""Test whether trained alignment beats dimension-only and shuffled controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.heldout_image_evaluation import (  # noqa: E402
    aggregate_rank_metrics,
    rank_record,
)
from experiments.dino_fusion.alignment import (  # noqa: E402
    collect_heldout_notice_ids,
    make_projection_head,
    project_embeddings,
    projection_head_from_checkpoint,
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
    collapse_visual_hits,
    safe_read_faiss_index,
)
from experiments.dino_fusion.evaluate_pilot import (  # noqa: E402
    attribute_relevance_summary,
    metas_by_notice,
)
from experiments.dino_fusion.train_alignment import (  # noqa: E402
    build_records,
    encode_prompt_table,
    multipositive_loss,
    reconstruct_index,
    split_hash,
)


DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts" / "dinov3"
REPORT_SCHEMA_VERSION = "dino-text-alignment-controls.v1"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _year_from_report(report: Mapping[str, Any]) -> int:
    text = clean_text(report.get("reference_date"))
    return int(text[:4]) if len(text) >= 4 and text[:4].isdigit() else 2026


def _metric_distribution(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("metric distribution cannot be empty")
    return {
        "mean": round(mean(values), 6),
        "sample_std": round(stdev(values), 6) if len(values) > 1 else 0.0,
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-metas", type=Path, default=ROOT / "data/dog_metas.json")
    parser.add_argument(
        "--heldout-report",
        type=Path,
        default=ROOT / "docs/evaluation/heldout_image_retrieval.appearance_v1.json",
    )
    parser.add_argument(
        "--alignment-evaluation-report",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "alignment_evaluation_report.json",
    )
    parser.add_argument(
        "--training-report",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "alignment_training_report.json",
    )
    parser.add_argument(
        "--dino-index", type=Path, default=DEFAULT_ARTIFACT_DIR / "dino.index"
    )
    parser.add_argument(
        "--dino-metas", type=Path, default=DEFAULT_ARTIFACT_DIR / "dino_metas.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "alignment_controls_report.json",
    )
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument("--architecture", choices=("linear", "mlp", "flow"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--prompt-batch-size", type=int, default=256)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.runs <= 0 or args.prompt_batch_size <= 0:
        raise ValueError("runs and prompt-batch-size must be positive")
    torch = __import__("torch")
    faiss = __import__("faiss")
    device = (
        "cuda"
        if args.device == "auto" and bool(torch.cuda.is_available())
        else ("cpu" if args.device == "auto" else args.device)
    )

    clip_metas = load_json(args.clip_metas)
    dino_metas = load_json(args.dino_metas)
    heldout = load_json(args.heldout_report)
    training_report = load_json(args.training_report)
    evaluation_report = load_json(args.alignment_evaluation_report)
    if not isinstance(clip_metas, list) or not isinstance(dino_metas, list):
        raise ValueError("index metadata must be JSON lists")
    dino_index = safe_read_faiss_index(args.dino_index, faiss.read_index)
    if int(dino_index.ntotal) != len(dino_metas):
        raise ValueError("DINO index/metadata row mismatch")
    dino_vectors = reconstruct_index(dino_index)

    architecture_reports = training_report.get("architectures") or {}
    validation_selected_architecture = min(
        architecture_reports,
        key=lambda name: float(
            (architecture_reports.get(name) or {}).get(
                "best_validation_loss", float("inf")
            )
        ),
    )
    selected_architecture = clean_text(args.architecture)
    if not selected_architecture:
        selected_architecture = validation_selected_architecture
    details = architecture_reports.get(selected_architecture) or {}
    checkpoint_path = args.training_report.parent / clean_text(
        details.get("checkpoint")
    )
    if sha256_file(checkpoint_path) != clean_text(details.get("checkpoint_sha256")):
        raise ValueError("trained checkpoint hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    input_dim = int(checkpoint["input_dim"])
    output_dim = int(checkpoint["output_dim"])
    hidden_dim = int(checkpoint["hidden_dim"])
    trained_head = projection_head_from_checkpoint(checkpoint).to(device)
    trained_head.load_state_dict(checkpoint["state_dict"])
    trained_head.eval()

    optimization = training_report.get("optimization") or {}
    training_seed = clean_text(optimization.get("seed"))
    seed_value = int(hashlib.sha256(training_seed.encode("utf-8")).hexdigest()[:8], 16)
    init_seed_base = (
        seed_value + {"linear": 0, "mlp": 1, "flow": 2}[selected_architecture]
    )
    best_epoch = int(details.get("best_epoch") or 0)
    if best_epoch <= 0:
        raise ValueError("trained report has no positive best epoch")

    heldout_ids = collect_heldout_notice_ids(heldout)
    records, exclusions = build_records(
        clip_metas=clip_metas,
        dino_metas=dino_metas,
        heldout_ids=heldout_ids,
        reference_year=_year_from_report(heldout),
    )
    train_indices, validation_indices = stratified_notice_split(
        records,
        validation_fraction=float(optimization["validation_fraction"]),
        seed=training_seed,
    )
    dataset_report = training_report.get("dataset") or {}
    if split_hash(records, train_indices) != clean_text(
        dataset_report.get("train_notice_sha256")
    ):
        raise ValueError("training split no longer matches the trained report")
    if split_hash(records, validation_indices) != clean_text(
        dataset_report.get("validation_notice_sha256")
    ):
        raise ValueError("validation split no longer matches the trained report")

    signatures = sorted({records[index]["signature"] for index in train_indices})
    signature_to_label = {
        signature: index for index, signature in enumerate(signatures)
    }
    signature_attributes = {
        record["signature"]: record["attributes"] for record in records
    }
    prompt_texts: list[str] = []
    prompt_rows: dict[str, list[int]] = {}
    for signature in signatures:
        rows: list[int] = []
        grouped = training_prompts(signature_attributes[signature])
        for language in ("english", "korean"):
            for prompt in grouped[language]:
                rows.append(len(prompt_texts))
                prompt_texts.append(prompt)
        prompt_rows[signature] = rows

    evaluation_queries = list(evaluation_report.get("queries") or [])
    if not evaluation_queries:
        raise ValueError("alignment evaluation report contains no queries")
    query_prompt_texts: list[list[str]] = []
    for query in evaluation_queries:
        prompts = query.get("prompts") or {}
        query_prompt_texts.append(
            [clean_text(prompts.get("english")), clean_text(prompts.get("korean"))]
        )
    if any(not value for value in prompt_texts) or any(
        not value for pair in query_prompt_texts for value in pair
    ):
        raise ValueError("empty control prompt")

    clip_encoder = ClipEncoder(args.clip_model, device=device)
    training_prompt_vectors_np = encode_prompt_table(
        clip_encoder, prompt_texts, batch_size=args.prompt_batch_size
    )
    query_prompt_vectors_np = np.concatenate(
        [clip_encoder.encode_text_batch(pair) for pair in query_prompt_texts],
        axis=0,
    )
    if (
        int(training_prompt_vectors_np.shape[1]) != input_dim
        or int(query_prompt_vectors_np.shape[1]) != input_dim
    ):
        raise ValueError("CLIP/checkpoint input dimension mismatch")
    all_prompt_vectors = torch.as_tensor(training_prompt_vectors_np, device=device)
    query_prompt_vectors = torch.as_tensor(query_prompt_vectors_np, device=device)

    train_visual = torch.as_tensor(
        dino_vectors[[int(records[index]["dino_row"]) for index in train_indices]],
        device=device,
    )
    train_labels = torch.as_tensor(
        [signature_to_label[records[index]["signature"]] for index in train_indices],
        device=device,
        dtype=torch.long,
    )
    train_prompt_ids = torch.as_tensor(
        [prompt_rows[records[index]["signature"]] for index in train_indices],
        device=device,
        dtype=torch.long,
    )

    systems: dict[str, Any] = {"trained": trained_head}
    control_training: list[dict[str, Any]] = []
    for run_index in range(args.runs):
        init_seed = init_seed_base + run_index
        torch.manual_seed(init_seed)
        random_head = make_projection_head(
            selected_architecture,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            flow_width=int(checkpoint.get("flow_width", 96)),
            flow_depth=int(checkpoint.get("flow_depth", 4)),
            flow_steps=int(checkpoint.get("flow_steps", 8)),
            flow_time_dim=int(checkpoint.get("flow_time_dim", 32)),
        ).to(device)
        random_head.eval()
        systems[f"random_{run_index:02d}"] = random_head

        torch.manual_seed(init_seed)
        shuffled_head = make_projection_head(
            selected_architecture,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            flow_width=int(checkpoint.get("flow_width", 96)),
            flow_depth=int(checkpoint.get("flow_depth", 4)),
            flow_steps=int(checkpoint.get("flow_steps", 8)),
            flow_time_dim=int(checkpoint.get("flow_time_dim", 32)),
        ).to(device)
        optimizer = torch.optim.AdamW(
            shuffled_head.parameters(),
            lr=float(optimization["learning_rate"]),
            weight_decay=float(optimization["weight_decay"]),
        )
        permutation_generator = torch.Generator(device=device)
        permutation_generator.manual_seed(seed_value + 1000 + run_index)
        permutation = torch.randperm(
            train_visual.shape[0], generator=permutation_generator, device=device
        )
        shuffled_visual = train_visual[permutation]
        shuffled_centroids = torch.zeros((len(signatures), output_dim), device=device)
        for label in range(len(signatures)):
            shuffled_centroids[label] = shuffled_visual[train_labels.eq(label)].mean(
                dim=0
            )
        shuffled_centroids = torch.nn.functional.normalize(shuffled_centroids, dim=-1)
        prompt_generator = torch.Generator(device=device)
        prompt_generator.manual_seed(seed_value + 2000 + run_index)
        started = time.perf_counter()
        final_loss = float("nan")
        for _epoch in range(best_epoch):
            shuffled_head.train()
            variants = torch.randint(
                train_prompt_ids.shape[1],
                (train_prompt_ids.shape[0],),
                generator=prompt_generator,
                device=device,
            )
            selected_ids = train_prompt_ids[
                torch.arange(train_prompt_ids.shape[0], device=device), variants
            ]
            mapped = project_embeddings(shuffled_head, all_prompt_vectors[selected_ids])
            loss = multipositive_loss(
                torch,
                mapped,
                shuffled_visual,
                train_labels,
                train_labels,
                shuffled_centroids,
                temperature=float(optimization["temperature"]),
                centroid_weight=float(optimization["centroid_weight"]),
                symmetric=True,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu())
        shuffled_head.eval()
        systems[f"shuffled_{run_index:02d}"] = shuffled_head
        control_training.append(
            {
                "run": run_index,
                "init_seed": init_seed,
                "permutation_sha256": hashlib.sha256(
                    permutation.detach().cpu().numpy().tobytes()
                ).hexdigest(),
                "epochs": best_epoch,
                "final_shuffled_loss": round(final_loss, 6),
                "training_seconds": round(time.perf_counter() - started, 3),
            }
        )

    dino_notice_ids = {
        clean_text(meta.get("notice_id") or meta.get("desertionNo"))
        for meta in dino_metas
    }
    current_by_id = metas_by_notice(clip_metas)
    query_rows = [
        {"notice_id": clean_text(query.get("notice_id")), "systems": {}}
        for query in evaluation_queries
    ]
    system_rows: dict[str, list[dict[str, Any]]] = {}
    with torch.inference_mode():
        for system, head in systems.items():
            mapped = project_embeddings(head, query_prompt_vectors)
            mapped_np = mapped.detach().float().cpu().numpy()
            for language_index, language in enumerate(("en", "ko")):
                system_name = f"{system}_{language}"
                rows: list[dict[str, Any]] = []
                for query_index, query in enumerate(query_rows):
                    vector_index = (2 * query_index) + language_index
                    scores, indices = dino_index.search(
                        mapped_np[vector_index : vector_index + 1],
                        int(dino_index.ntotal),
                    )
                    ranking = collapse_visual_hits(
                        scores[0].tolist(),
                        indices[0].tolist(),
                        dino_metas,
                        score_kind="similarity",
                    )
                    record = rank_record(
                        clean_text(query["notice_id"]), ranking, top_k=10
                    )
                    query["systems"][system_name] = record
                    rows.append(record)
                system_rows[system_name] = rows

    exact_metrics = {
        system: aggregate_rank_metrics(
            rows,
            attempted_count=len(query_rows),
            downloaded_count=len(query_rows),
            duplicate_count=0,
        )
        for system, rows in system_rows.items()
    }
    attribute_metrics = {
        system: attribute_relevance_summary(
            query_rows,
            system=system,
            metas=current_by_id,
            candidate_notice_ids=dino_notice_ids,
        )
        for system in system_rows
    }

    source_attribute = evaluation_report.get("attribute_relevance") or {}
    for language in ("en", "ko"):
        expected = (
            source_attribute.get(f"{selected_architecture}_aligned_text_{language}")
            or {}
        )
        actual = attribute_metrics[f"trained_{language}"]
        for metric in ("precision@10", "nDCG@10"):
            if not np.isclose(float(expected[metric]), float(actual[metric])):
                raise RuntimeError(
                    f"trained control audit mismatch for {language} {metric}"
                )

    summary: dict[str, Any] = {
        "trained": {},
        "random_untrained": {},
        "shuffled_association": {},
        "effect": {},
    }
    for language in ("en", "ko"):
        trained_attribute = attribute_metrics[f"trained_{language}"]
        trained_exact = exact_metrics[f"trained_{language}"]
        summary["trained"][language] = {
            "precision@10": trained_attribute["precision@10"],
            "nDCG@10": trained_attribute["nDCG@10"],
            "exact_MRR": trained_exact["MRR"],
        }
        for family, prefix in (
            ("random_untrained", "random"),
            ("shuffled_association", "shuffled"),
        ):
            names = [f"{prefix}_{index:02d}_{language}" for index in range(args.runs)]
            ndcg_values = [float(attribute_metrics[name]["nDCG@10"]) for name in names]
            precision_values = [
                float(attribute_metrics[name]["precision@10"]) for name in names
            ]
            mrr_values = [float(exact_metrics[name]["MRR"]) for name in names]
            summary[family][language] = {
                "precision@10": _metric_distribution(precision_values),
                "nDCG@10": _metric_distribution(ndcg_values),
                "exact_MRR": _metric_distribution(mrr_values),
            }
        summary["effect"][language] = {
            "trained_minus_random_mean_nDCG@10": round(
                float(trained_attribute["nDCG@10"])
                - float(summary["random_untrained"][language]["nDCG@10"]["mean"]),
                6,
            ),
            "trained_minus_shuffled_mean_nDCG@10": round(
                float(trained_attribute["nDCG@10"])
                - float(summary["shuffled_association"][language]["nDCG@10"]["mean"]),
                6,
            ),
        }

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "question": "Does learned semantic alignment beat dimension-only and shuffled controls?",
        "backbone": training_report.get("encoders") or {},
        "protocol": {
            "architecture": selected_architecture,
            "architecture_selected_on_validation": (
                selected_architecture == validation_selected_architecture
            ),
            "validation_selected_architecture": validation_selected_architecture,
            "control_runs": args.runs,
            "random_control": "same architecture with no optimizer updates",
            "shuffled_control": (
                "same architecture, optimizer, loss, and matched update count; DINO image rows randomly permuted before training"
            ),
            "matched_training_epochs": best_epoch,
            "query_count": len(query_rows),
            "query_source": "payload-verified held-out evaluation report",
            "heldout_used_for_control_training": False,
        },
        "split_audit": {
            "train_notice_count": len(train_indices),
            "validation_notice_count": len(validation_indices),
            "exclusions": dict(sorted(exclusions.items())),
            "train_notice_sha256_verified": True,
            "validation_notice_sha256_verified": True,
        },
        "summary": summary,
        "control_training": control_training,
        "exact_metrics": exact_metrics,
        "attribute_relevance": attribute_metrics,
        "limitations": [
            "Five control seeds characterize gross effects but are not a formal significance test.",
            "The 20 held-out prompts still use controlled color, size, and age vocabulary.",
            "Shuffling destroys semantic association while preserving the DINO feature distribution, but it is not the only possible negative control.",
        ],
    }
    write_json_atomic(args.output, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    print(
        json.dumps(
            {
                "schema_version": report["schema_version"],
                "protocol": report["protocol"],
                "summary": report["summary"],
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
