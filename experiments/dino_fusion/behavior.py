"""Unknown-aware weak behavior labels for the isolated DINO fusion pilot.

The public notice is treated as evidence, not as a complete personality record:
an absent phrase is always ``unknown``.  Only explicitly opposed phrases create
the two supervised ends of an axis.  This keeps the behavior experiment from
turning missing shelter notes into negative labels.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.profile_rerank import resolve_notice_behavior_text
from experiments.dino_fusion.core import clean_text


UNKNOWN = 0
POSITIVE = 1
NEGATIVE = -1


@dataclass(frozen=True)
class BehaviorAxis:
    key: str
    positive_label: str
    negative_label: str
    positive_patterns: tuple[str, ...]
    negative_patterns: tuple[str, ...]
    positive_query: str
    negative_query: str


BEHAVIOR_AXES: tuple[BehaviorAxis, ...] = (
    BehaviorAxis(
        key="gentle_handling",
        positive_label="온순·부드러움",
        negative_label="방어·입질 주의",
        positive_patterns=(
            r"매우\s*순함",
            r"순함",
            r"순한\s*편",
            r"온순",
            r"순둥",
            r"착함",
            r"착한\s*(?:아이|강아지|성격)?",
        ),
        negative_patterns=(
            r"방어적(?:인)?(?:\s*입질)?",
            r"입질",
            r"물려고",
            r"공격적",
            r"공격성",
            r"사나움",
            r"사나운",
        ),
        positive_query="온순하고 사람의 손길에 부드럽게 반응하는 강아지",
        negative_query="경계가 강하고 방어적 입질에 주의가 필요한 강아지",
    ),
    BehaviorAxis(
        key="people_social",
        positive_label="사람 친화",
        negative_label="낯선 사람 경계",
        positive_patterns=(
            r"사람(?:을|이|과|에게)?\s*(?:아주\s*)?좋아",
            r"사람(?:을|에게)?\s*잘\s*따",
            r"처음\s*보는\s*사람에게도",
            r"대인\s*친화",
            r"사람에게\s*친화",
            r"사교적",
            r"애교(?:가|도)?\s*(?:많|있)",
        ),
        negative_patterns=(
            r"낯가림",
            r"낯선\s*사람.{0,10}(?:경계|무서|두려)",
            r"사람.{0,8}(?:경계|무서|두려|피함)",
            r"사람을\s*피함",
        ),
        positive_query="낯선 사람에게도 친근하고 사람을 잘 따르는 강아지",
        negative_query="낯선 사람을 조심하고 낯가림이 있는 강아지",
    ),
    BehaviorAxis(
        key="activity",
        positive_label="활동적·고에너지",
        negative_label="차분·저자극",
        positive_patterns=(
            r"활발",
            r"발랄",
            r"에너지\s*넘",
            r"활동량\s*(?:이\s*)?많",
            r"운동량\s*(?:이\s*)?많",
        ),
        negative_patterns=(
            r"차분",
            r"얌전",
            r"조용한\s*성격",
            r"조용함",
            r"활동량\s*(?:이\s*)?적",
            r"운동량\s*(?:이\s*)?적",
        ),
        positive_query="활발하고 에너지가 많아 산책과 놀이를 좋아하는 강아지",
        negative_query="차분하고 조용한 생활에 잘 어울리는 강아지",
    ),
)


EVIDENCE_ONLY_PATTERNS: Mapping[str, tuple[str, ...]] = {
    "fear_watchful": (
        r"겁\s*(?:이\s*)?(?:많|있)",
        r"소심",
        r"두려",
        r"무서",
        r"경계",
        r"낯가림",
    ),
}


_NEGATED_SUFFIX = re.compile(r"^[^,.;/]{0,10}(?:없|아니|않|못하|안\s*함|보이지\s*않)")
_NEGATED_PREFIX = re.compile(r"(?:안|전혀|별로)\s*$")


def normalize_behavior_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", clean_text(value)).lower()
    return re.sub(r"\s+", " ", text).strip()


def public_behavior_text(meta: Mapping[str, Any]) -> str:
    """Return provenance-bearing public/shelter text only."""

    return normalize_behavior_text(resolve_notice_behavior_text(dict(meta)))


def _asserted_matches(text: str, patterns: Sequence[str]) -> list[str]:
    matches: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            before = text[max(0, match.start() - 8) : match.start()]
            after = text[match.end() : match.end() + 16]
            if _NEGATED_PREFIX.search(before) or _NEGATED_SUFFIX.search(after):
                continue
            phrase = match.group(0).strip()
            if phrase and phrase not in matches:
                matches.append(phrase)
    return matches


def weak_behavior_labels(text: Any) -> dict[str, dict[str, Any]]:
    """Extract explicit labels while keeping absence and conflicts unknown."""

    normalized = normalize_behavior_text(text)
    labels: dict[str, dict[str, Any]] = {}
    for axis in BEHAVIOR_AXES:
        positive = _asserted_matches(normalized, axis.positive_patterns)
        negative = _asserted_matches(normalized, axis.negative_patterns)
        if positive and not negative:
            value = POSITIVE
            state = "positive"
        elif negative and not positive:
            value = NEGATIVE
            state = "negative"
        elif positive and negative:
            value = UNKNOWN
            state = "conflict"
        else:
            value = UNKNOWN
            state = "unknown"
        labels[axis.key] = {
            "value": value,
            "state": state,
            "positive_evidence": positive,
            "negative_evidence": negative,
        }

    for key, patterns in EVIDENCE_ONLY_PATTERNS.items():
        evidence = _asserted_matches(normalized, patterns)
        labels[key] = {
            "value": POSITIVE if evidence else UNKNOWN,
            "state": "evidence" if evidence else "unknown",
            "positive_evidence": evidence,
            "negative_evidence": [],
        }
    return labels


def build_behavior_records(
    clip_metas: Sequence[Mapping[str, Any]],
    dino_metas: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Join the one-crop-per-notice DINO corpus to public behavior evidence."""

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for dino_row, dino_meta in enumerate(dino_metas):
        notice_id = clean_text(
            dino_meta.get("notice_id") or dino_meta.get("desertionNo")
        )
        source_index = dino_meta.get("source_meta_index")
        if (
            not notice_id
            or notice_id in seen
            or not isinstance(source_index, int)
            or not 0 <= source_index < len(clip_metas)
        ):
            continue
        seen.add(notice_id)
        meta = clip_metas[source_index]
        text = public_behavior_text(meta)
        weak = weak_behavior_labels(text)
        labels = [int(weak[axis.key]["value"]) for axis in BEHAVIOR_AXES]
        records.append(
            {
                "notice_id": notice_id,
                "dino_row": dino_row,
                "source_meta_index": source_index,
                "behavior_text": text,
                "labels": labels,
                "observed": [value != UNKNOWN for value in labels],
                "weak_labels": weak,
            }
        )
    return records


def _split_counts(group_count: int, fraction: float) -> tuple[int, int]:
    if group_count < 3:
        return 0, 0
    if group_count == 3:
        return 1, 0
    validation = max(1, round(group_count * fraction))
    test = max(1, round(group_count * fraction))
    while validation + test >= group_count:
        if validation >= test and validation > 1:
            validation -= 1
        elif test > 1:
            test -= 1
        else:
            break
    return validation, test


def stratified_text_group_split(
    records: Sequence[Mapping[str, Any]],
    *,
    fraction: float = 0.15,
    seed: str = "behavior-pilot-v1",
) -> dict[str, list[int]]:
    """Split exact duplicate notes together, stratified by weak-label signature."""

    if not 0.0 < fraction < 0.5:
        raise ValueError("fraction must be between zero and one half")
    text_groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        text = normalize_behavior_text(record.get("behavior_text"))
        group_key = text or f"__empty__:{clean_text(record.get('notice_id'))}"
        text_groups[group_key].append(index)

    signatures: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for group_key, indices in text_groups.items():
        signature = tuple(int(value) for value in records[indices[0]]["labels"])
        signatures[signature].append(group_key)

    output = {"train": [], "validation": [], "test": []}
    for signature, group_keys in sorted(signatures.items()):
        ordered = sorted(
            group_keys,
            key=lambda value: hashlib.sha256(
                f"{seed}|{signature}|{value}".encode("utf-8")
            ).hexdigest(),
        )
        validation_count, test_count = _split_counts(len(ordered), fraction)
        test_groups = set(ordered[:test_count])
        validation_groups = set(ordered[test_count : test_count + validation_count])
        for group_key in ordered:
            split = (
                "test"
                if group_key in test_groups
                else ("validation" if group_key in validation_groups else "train")
            )
            output[split].extend(text_groups[group_key])
    for values in output.values():
        values.sort()
    return output


def make_behavior_head(
    *, input_dim: int = 512, hidden_dim: int = 128, output_dim: int | None = None
) -> Any:
    torch = __import__("torch")
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.GELU(),
        torch.nn.Linear(hidden_dim, output_dim or len(BEHAVIOR_AXES)),
    )


def conservative_behavior_score(
    probability: float,
    explicit_label: int,
    *,
    desired_label: int,
    inference_band: float = 0.20,
) -> tuple[float, str]:
    """Place explicit support above unknown and contradiction below unknown."""

    if desired_label not in {POSITIVE, NEGATIVE}:
        raise ValueError("desired_label must be positive or negative")
    if explicit_label == desired_label:
        return 1.0, "explicit_match"
    if explicit_label == -desired_label:
        return 0.0, "explicit_contradiction"
    resolved = (
        float(probability) if desired_label == POSITIVE else 1.0 - float(probability)
    )
    resolved = min(1.0, max(0.0, resolved))
    score = 0.5 + inference_band * (2.0 * resolved - 1.0)
    return score, "unknown_inferred"


def percentile_scores(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(array):
        return array
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    ranks[order] = np.arange(len(array), dtype=np.float64)
    return ranks / max(1, len(array) - 1)


def binary_ranking_metrics(
    labels: Any, scores: Any, *, topk: int = 10
) -> dict[str, float]:
    truth = np.asarray(labels, dtype=np.int64).reshape(-1)
    predicted = np.asarray(scores, dtype=np.float64).reshape(-1)
    if truth.shape != predicted.shape or not len(truth):
        raise ValueError("labels and scores must be non-empty and aligned")
    positives = int(truth.sum())
    negatives = int(len(truth) - positives)
    if positives <= 0 or negatives <= 0:
        raise ValueError("binary metrics require both classes")
    order = np.argsort(-predicted, kind="mergesort")
    ranked = truth[order]

    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    average_precision = float((precision * ranked).sum() / positives)

    positive_scores = predicted[truth == 1]
    negative_scores = predicted[truth == 0]
    comparisons = positive_scores[:, None] - negative_scores[None, :]
    auc = float(
        ((comparisons > 0).sum() + 0.5 * (comparisons == 0).sum()) / comparisons.size
    )

    cutoff = min(int(topk), len(ranked))
    discounts = 1.0 / np.log2(np.arange(2, cutoff + 2))
    dcg = float((ranked[:cutoff] * discounts).sum())
    ideal_count = min(positives, cutoff)
    idcg = float(discounts[:ideal_count].sum())
    first = np.flatnonzero(ranked)
    return {
        "roc_auc": round(auc, 6),
        "average_precision": round(average_precision, 6),
        f"ndcg@{topk}": round(dcg / idcg if idcg else 0.0, 6),
        "mrr": round(1.0 / (int(first[0]) + 1) if len(first) else 0.0, 6),
        "known_count": int(len(truth)),
        "positive_count": positives,
        "negative_count": negatives,
    }


def finite_mean(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return round(float(np.mean(finite)), 6) if finite else 0.0
