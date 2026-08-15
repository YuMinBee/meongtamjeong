"""Diagnostic full-text baseline for the compatibility retrieval experiment.

This is not a proposed replacement for CLIP appearance text.  It tests whether
the shelter descriptions contain usable compatibility evidence when a model can
read the complete document instead of CLIP's 77-token context.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.dino_fusion_external_eval.compatibility import (  # noqa: E402
    classification_metrics,
    fuse_visual_behavior,
)
from experiments.dino_fusion_external_eval.evaluate_retrieval import (  # noqa: E402
    BEHAVIOR_WEIGHT_CANDIDATES,
    MAXIMUM_RECALL10_DROP,
    MINIMUM_CLASSIFICATION_AUC,
    MINIMUM_NDCG_GAIN,
    _evaluate_scores,
    _queries,
)
from experiments.dino_fusion_external_eval.prepare_benchmark import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    write_json_atomic,
)


REPORT_SCHEMA_VERSION = "petfinder-sparse-compatibility-baseline.v1"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _top_pool_rerank(
    visual: np.ndarray,
    behavior: np.ndarray,
    *,
    behavior_weight: float,
    pool_size: int = 10,
) -> np.ndarray:
    """Rerank only DINO's top pool, preserving its Recall@pool membership."""

    order = np.argsort(-visual, kind="mergesort")
    cutoff = min(pool_size, len(order))
    fused = fuse_visual_behavior(
        visual[order[:cutoff]],
        behavior[order[:cutoff]],
        behavior_weight=behavior_weight,
    )
    scores = np.empty(len(order), dtype=np.float64)
    scores[order[:cutoff]] = 2.0 + fused
    if cutoff < len(order):
        scores[order[cutoff:]] = -np.arange(len(order) - cutoff, dtype=np.float64)
    return scores


def evaluate(*, output_dir: Path) -> dict[str, Any]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for this diagnostic") from exc

    started = time.perf_counter()
    text_records = _load_json(output_dir / "text_records.json")
    multimodal_all = _load_json(output_dir / "multimodal_records.json")
    multimodal = [
        record
        for record in multimodal_all
        if record["split"] in {"validation", "test"}
    ]
    arrays = np.load(output_dir / "multimodal_image_embeddings.npz", allow_pickle=False)
    if not np.array_equal(
        arrays["pet_ids"], np.asarray([str(record["pet_id"]) for record in multimodal])
    ):
        raise ValueError("image embedding cache is not aligned to multimodal records")

    train_indices = np.asarray(
        [index for index, row in enumerate(text_records) if row["split"] == "train"]
    )
    validation_indices = np.asarray(
        [
            index
            for index, row in enumerate(text_records)
            if row["split"] == "validation"
        ]
    )
    test_indices = np.asarray(
        [index for index, row in enumerate(text_records) if row["split"] == "test"]
    )
    texts = [str(record["description"]) for record in text_records]
    vectorizer = TfidfVectorizer(
        strip_accents="unicode",
        lowercase=True,
        ngram_range=(1, 2),
        min_df=3,
        max_features=100_000,
        sublinear_tf=True,
    )
    vector_started = time.perf_counter()
    train_features = vectorizer.fit_transform(
        [texts[index] for index in train_indices]
    )
    validation_features = vectorizer.transform(
        [texts[index] for index in validation_indices]
    )
    test_features = vectorizer.transform([texts[index] for index in test_indices])
    multimodal_features = vectorizer.transform(
        [str(record["description"]) for record in multimodal]
    )
    vector_seconds = time.perf_counter() - vector_started

    labels = np.asarray([record["labels"] for record in text_records], dtype=np.int64)
    full_test_probabilities = np.empty((len(test_indices), 4), dtype=np.float64)
    multimodal_probabilities = np.empty((len(multimodal), 4), dtype=np.float64)
    models: list[dict[str, Any]] = []
    optimization_started = time.perf_counter()
    for axis_index in range(4):
        best: tuple[float, float, Any] | None = None
        train_known = labels[train_indices, axis_index] != 0
        validation_known = labels[validation_indices, axis_index] != 0
        for regularization in (0.3, 1.0, 3.0):
            model = LogisticRegression(
                C=regularization,
                class_weight="balanced",
                max_iter=300,
                n_jobs=-1,
            )
            model.fit(
                train_features[train_known],
                (labels[train_indices, axis_index][train_known] == 1).astype(int),
            )
            validation_scores = model.predict_proba(
                validation_features[validation_known]
            )[:, 1]
            validation_truth = (
                labels[validation_indices, axis_index][validation_known] == 1
            ).astype(np.int64)
            positive_scores = validation_scores[validation_truth == 1]
            negative_scores = validation_scores[validation_truth == 0]
            comparisons = positive_scores[:, None] - negative_scores[None, :]
            auc = float(
                ((comparisons > 0).sum() + 0.5 * (comparisons == 0).sum())
                / comparisons.size
            )
            if best is None or auc > best[0]:
                best = (auc, regularization, model)
        if best is None:
            raise RuntimeError("sparse classifier selection failed")
        full_test_probabilities[:, axis_index] = best[2].predict_proba(test_features)[
            :, 1
        ]
        multimodal_probabilities[:, axis_index] = best[2].predict_proba(
            multimodal_features
        )[:, 1]
        models.append(
            {
                "axis_index": axis_index,
                "selected_C": best[1],
                "validation_roc_auc": round(best[0], 6),
            }
        )
    optimization_seconds = time.perf_counter() - optimization_started

    full_test_metrics = classification_metrics(
        labels[test_indices], full_test_probabilities
    )
    multimodal_classification: dict[str, Any] = {}
    split_positions: dict[str, np.ndarray] = {}
    for split in ("validation", "test"):
        positions = np.asarray(
            [index for index, record in enumerate(multimodal) if record["split"] == split]
        )
        split_positions[split] = positions
        multimodal_classification[split] = classification_metrics(
            np.asarray([multimodal[int(index)]["labels"] for index in positions]),
            multimodal_probabilities[positions],
        )

    strategy_results: dict[str, Any] = {}
    for strategy in ("global_fusion", "dino_top10_rerank"):
        sweeps: dict[str, list[dict[str, Any]]] = {}
        baselines: dict[str, Any] = {}
        for split in ("validation", "test"):
            positions = split_positions[split]
            records = [multimodal[int(index)] for index in positions]
            queries = _queries(records)
            dino_scores = (
                arrays["dino_query"][positions]
                @ arrays["dino_gallery"][positions].T
            )
            probabilities = multimodal_probabilities[positions]
            baseline, _ = _evaluate_scores(
                records,
                queries,
                lambda query: dino_scores[int(query["reference_index"])],
            )
            baselines[split] = baseline
            rows: list[dict[str, Any]] = []
            for weight in BEHAVIOR_WEIGHT_CANDIDATES:

                def score(query: dict[str, Any], weight: float = weight) -> np.ndarray:
                    axis = int(query["axis_index"])
                    positive = probabilities[:, axis]
                    behavior = (
                        positive
                        if int(query["desired_label"]) == 1
                        else 1.0 - positive
                    )
                    visual = dino_scores[int(query["reference_index"])]
                    if strategy == "global_fusion":
                        return fuse_visual_behavior(
                            visual, behavior, behavior_weight=weight
                        )
                    return _top_pool_rerank(
                        visual,
                        behavior,
                        behavior_weight=weight,
                        pool_size=10,
                    )

                metrics, _ = _evaluate_scores(records, queries, score)
                rows.append(
                    {
                        "behavior_weight": weight,
                        **metrics,
                        "ndcg_gain": round(
                            metrics["behavior_ndcg@10"]
                            - baseline["behavior_ndcg@10"],
                            6,
                        ),
                        "recall10_drop": round(
                            baseline["recall@10"] - metrics["recall@10"], 6
                        ),
                    }
                )
            sweeps[split] = rows
        eligible = [
            row
            for row in sweeps["validation"]
            if row["recall10_drop"] <= MAXIMUM_RECALL10_DROP
        ]
        selected = max(
            eligible or sweeps["validation"],
            key=lambda row: (
                row["behavior_ndcg@10"],
                -row["behavior_weight"],
            ),
        )
        selected_test = next(
            row
            for row in sweeps["test"]
            if row["behavior_weight"] == selected["behavior_weight"]
        )
        test_auc = multimodal_classification["test"]["macro_roc_auc"]
        checks = {
            "ndcg_gain": selected_test["ndcg_gain"] >= MINIMUM_NDCG_GAIN,
            "recall10_drop": (
                selected_test["recall10_drop"] <= MAXIMUM_RECALL10_DROP
            ),
            "classification_auc": test_auc >= MINIMUM_CLASSIFICATION_AUC,
        }
        strategy_results[strategy] = {
            "validation_selected_weight": selected["behavior_weight"],
            "validation_selected_metrics": selected,
            "test_baseline": baselines["test"],
            "test_selected_metrics": selected_test,
            "gate": {"checks": checks, "passes": all(checks.values())},
            "weight_sweeps": sweeps,
        }

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "purpose": "full-document information-availability diagnostic",
        "model": {
            "vectorizer": "word TF-IDF unigrams+bigrams, maximum 100,000 features",
            "classifier": "class-balanced logistic regression per axis",
            "selected_models": models,
            "production_candidate": False,
        },
        "classification": {
            "full_text_test": full_test_metrics,
            "multimodal": multimodal_classification,
        },
        "retrieval": strategy_results,
        "timing": {
            "hardware": "CPU",
            "vectorization_seconds": round(vector_seconds, 6),
            "model_selection_and_training_seconds": round(
                optimization_seconds, 6
            ),
            "actual_end_to_end_seconds": round(time.perf_counter() - started, 6),
        },
        "interpretation": (
            "If this full-text baseline beats the frozen CLIP head, the source text "
            "contains signal and CLIP context/representation is the bottleneck."
        ),
    }
    write_json_atomic(report, output_dir / "sparse_baseline_report.json", pretty=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate(output_dir=args.output_dir)
    print(
        json.dumps(
            {
                "classification": report["classification"],
                "retrieval": {
                    key: {
                        "weight": value["validation_selected_weight"],
                        "test": value["test_selected_metrics"],
                        "gate": value["gate"],
                    }
                    for key, value in report["retrieval"].items()
                },
                "timing": report["timing"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
