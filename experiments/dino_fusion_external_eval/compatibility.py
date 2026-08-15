"""Shared schema and pure metrics for the PetFinder compatibility pilot.

The values deliberately use three states.  A missing shelter field is unknown,
not negative, while an explicit ``False`` value is a supervised contradiction.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from experiments.dino_fusion.behavior import (
    NEGATIVE,
    POSITIVE,
    UNKNOWN,
    binary_ranking_metrics,
    finite_mean,
)


@dataclass(frozen=True)
class CompatibilityAxis:
    key: str
    source_column: str
    positive_query_ko: str
    negative_query_ko: str
    positive_query_en: str
    negative_query_en: str


COMPATIBILITY_AXES: tuple[CompatibilityAxis, ...] = (
    CompatibilityAxis(
        key="children_compatible",
        source_column="env_children",
        positive_query_ko="아이와 함께 살아도 잘 지내는 강아지",
        negative_query_ko="아이 없는 조용한 가정이 더 적합한 강아지",
        positive_query_en="a dog that is good with children",
        negative_query_en="a dog that needs a home without children",
    ),
    CompatibilityAxis(
        key="dogs_compatible",
        source_column="env_dogs",
        positive_query_ko="다른 강아지와 잘 지내는 강아지",
        negative_query_ko="다른 강아지가 없는 가정이 필요한 강아지",
        positive_query_en="a dog that is good with other dogs",
        negative_query_en="a dog that needs a home without other dogs",
    ),
    CompatibilityAxis(
        key="cats_compatible",
        source_column="env_cats",
        positive_query_ko="고양이와 함께 살아도 잘 지내는 강아지",
        negative_query_ko="고양이가 없는 가정이 필요한 강아지",
        positive_query_en="a dog that is good with cats",
        negative_query_en="a dog that needs a home without cats",
    ),
    CompatibilityAxis(
        key="house_trained",
        source_column="house_trained",
        positive_query_ko="배변 훈련이 되어 있는 강아지",
        negative_query_ko="배변 훈련을 처음부터 도와줄 수 있는 강아지",
        positive_query_en="a house-trained dog",
        negative_query_en="a dog that still needs house training",
    ),
)


TRUE_VALUES = frozenset({"1", "true", "t", "yes", "y"})
FALSE_VALUES = frozenset({"0", "false", "f", "no", "n"})
UNKNOWN_VALUES = frozenset({"", "na", "n/a", "nan", "none", "null", "unknown"})
_EVIDENCE_TERMS = re.compile(
    r"(?:child|children|kid|kids|toddler|baby|babies|dog|dogs|canine|"
    r"cat|cats|feline|house\s*train|housetrain|housebreak|housebroken|"
    r"potty|crate|accident)",
    re.IGNORECASE,
)
_WORD = re.compile(r"\S+")


def parse_label(value: Any) -> int:
    """Parse a source boolean without turning missing values into negatives."""

    normalized = "" if value is None else str(value).strip().lower()
    if normalized in TRUE_VALUES:
        return POSITIVE
    if normalized in FALSE_VALUES:
        return NEGATIVE
    if normalized in UNKNOWN_VALUES:
        return UNKNOWN
    raise ValueError(f"unsupported compatibility label: {value!r}")


def behavior_evidence_text(value: Any, *, window_words: int = 12) -> str:
    """Bring compatibility-bearing windows inside CLIP's 77-token context.

    This function does not infer a label.  It only selects local source-text
    windows around neutral topic words such as ``children`` or ``house-trained``.
    """

    text = "" if value is None else re.sub(r"\s+", " ", str(value)).strip()
    if not text:
        return ""
    words = list(_WORD.finditer(text))
    hit_word_indices: list[int] = []
    for match in _EVIDENCE_TERMS.finditer(text):
        for index, word in enumerate(words):
            if word.start() <= match.start() < word.end():
                hit_word_indices.append(index)
                break
    if not hit_word_indices:
        return " ".join(word.group(0) for word in words[:60])

    selected: set[int] = set()
    for index in hit_word_indices:
        selected.update(
            range(max(0, index - window_words), min(len(words), index + window_words + 1))
        )
    ordered = sorted(selected)
    # Sixty words stay safely near CLIP's 77-token limit for ordinary English.
    return " ".join(words[index].group(0) for index in ordered[:60])


def split_for_group(
    group_id: str,
    *,
    seed: str = "petfinder-external-v1",
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> str:
    """Assign a whole shelter/organization to one deterministic split."""

    if not group_id:
        raise ValueError("group_id is required")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between zero and one")
    if not 0.0 < validation_fraction < 1.0 - train_fraction:
        raise ValueError("validation_fraction leaves no test split")
    digest = hashlib.sha256(f"{seed}|{group_id}".encode()).digest()
    unit = int.from_bytes(digest[:8], "big") / float(2**64)
    if unit < train_fraction:
        return "train"
    if unit < train_fraction + validation_fraction:
        return "validation"
    return "test"


def make_compatibility_head(
    *, input_dim: int = 512, hidden_dim: int = 128, output_dim: int | None = None
) -> Any:
    torch = __import__("torch")
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.GELU(),
        torch.nn.Dropout(0.10),
        torch.nn.Linear(hidden_dim, output_dim or len(COMPATIBILITY_AXES)),
    )


def balanced_masked_loss(torch: Any, logits: Any, labels: Any) -> Any:
    """Balanced BCE over known positive and negative ends of every axis."""

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
        raise ValueError("batch has no axis with both observed labels")
    return torch.stack(losses).mean()


def classification_metrics(labels: Any, probabilities: Any) -> dict[str, Any]:
    truth_matrix = np.asarray(labels, dtype=np.int64)
    score_matrix = np.asarray(probabilities, dtype=np.float64)
    if truth_matrix.shape != score_matrix.shape:
        raise ValueError("labels and probabilities must be aligned")
    if truth_matrix.ndim != 2 or truth_matrix.shape[1] != len(COMPATIBILITY_AXES):
        raise ValueError("unexpected compatibility matrix shape")

    axes: dict[str, Any] = {}
    for axis_index, axis in enumerate(COMPATIBILITY_AXES):
        known = truth_matrix[:, axis_index] != UNKNOWN
        truth = (truth_matrix[known, axis_index] == POSITIVE).astype(np.int64)
        scores = score_matrix[known, axis_index]
        if len(np.unique(truth)) < 2:
            axes[axis.key] = {
                "known_count": int(len(truth)),
                "not_evaluable": True,
            }
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


def percentile_scores(values: Any) -> np.ndarray:
    """Convert one query's scores to stable [0, 1] ranks."""

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(array):
        return array
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    ranks[order] = np.arange(len(array), dtype=np.float64)
    return ranks / max(1, len(array) - 1)


def fuse_visual_behavior(
    visual_scores: Any, behavior_scores: Any, *, behavior_weight: float = 0.25
) -> np.ndarray:
    if not 0.0 <= behavior_weight <= 1.0:
        raise ValueError("behavior_weight must be between zero and one")
    visual = percentile_scores(visual_scores)
    behavior = np.asarray(behavior_scores, dtype=np.float64).reshape(-1)
    if visual.shape != behavior.shape or not np.isfinite(behavior).all():
        raise ValueError("visual and behavior scores must be finite and aligned")
    return (1.0 - behavior_weight) * visual + behavior_weight * behavior


def ndcg_at_k(relevance: Any, ranking_scores: Any, *, k: int = 10) -> float:
    labels = np.asarray(relevance, dtype=np.float64).reshape(-1)
    scores = np.asarray(ranking_scores, dtype=np.float64).reshape(-1)
    if labels.shape != scores.shape or not len(labels):
        raise ValueError("relevance and scores must be non-empty and aligned")
    cutoff = min(int(k), len(labels))
    discounts = 1.0 / np.log2(np.arange(2, cutoff + 2))
    ranked = labels[np.argsort(-scores, kind="mergesort")[:cutoff]]
    ideal = np.sort(labels)[::-1][:cutoff]
    dcg = float(((2.0**ranked - 1.0) * discounts).sum())
    idcg = float(((2.0**ideal - 1.0) * discounts).sum())
    return dcg / idcg if idcg else 0.0


def summarize_ranking_queries(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("at least one ranking query is required")
    recall_keys = ("recall@1", "recall@5", "recall@10")
    result: dict[str, Any] = {
        "query_count": len(rows),
        "behavior_ndcg@10": round(
            float(np.mean([float(row["behavior_ndcg@10"]) for row in rows])), 6
        ),
    }
    for key in recall_keys:
        result[key] = round(float(np.mean([float(row[key]) for row in rows])), 6)
    for key in ("match_rate@10", "contradiction_rate@10", "unknown_rate@10"):
        result[key] = round(float(np.mean([float(row[key]) for row in rows])), 6)
    return result


def finite(value: Any) -> bool:
    """Small public helper used by report validation tests."""

    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
