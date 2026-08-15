from __future__ import annotations

import numpy as np

from experiments.dino_fusion.behavior import (
    NEGATIVE,
    POSITIVE,
    UNKNOWN,
    binary_ranking_metrics,
    conservative_behavior_score,
    public_behavior_text,
    stratified_text_group_split,
    weak_behavior_labels,
)


def test_missing_behavior_is_unknown_not_negative() -> None:
    labels = weak_behavior_labels("갈색 털에 파란 목줄을 착용함")

    assert labels["gentle_handling"]["value"] == UNKNOWN
    assert labels["people_social"]["value"] == UNKNOWN
    assert labels["activity"]["value"] == UNKNOWN


def test_explicit_opposed_behavior_phrases_create_labels() -> None:
    positive = weak_behavior_labels("매우 순함, 사람을 좋아하고 활발함")
    negative = weak_behavior_labels("방어적 입질이 있으며 낯선 사람을 경계하고 얌전함")

    assert positive["gentle_handling"]["value"] == POSITIVE
    assert positive["people_social"]["value"] == POSITIVE
    assert positive["activity"]["value"] == POSITIVE
    assert negative["gentle_handling"]["value"] == NEGATIVE
    assert negative["people_social"]["value"] == NEGATIVE
    assert negative["activity"]["value"] == NEGATIVE


def test_negated_and_conflicting_phrases_abstain() -> None:
    negated = weak_behavior_labels("입질 없음, 공격적이지 않음")
    conflict = weak_behavior_labels("온순하지만 방어적 입질이 있음")

    assert negated["gentle_handling"]["value"] == UNKNOWN
    assert conflict["gentle_handling"]["value"] == UNKNOWN
    assert conflict["gentle_handling"]["state"] == "conflict"


def test_public_behavior_text_does_not_promote_vlm_description() -> None:
    text = public_behavior_text(
        {
            "desc": "AI가 만든 사람 친화 설명",
            "vlm_desc": "사진에서 온순해 보임",
            "specialMark": "파란 목줄 착용",
        }
    )

    assert text == "파란 목줄 착용"


def test_duplicate_notes_never_cross_splits() -> None:
    records = [
        {
            "notice_id": str(index),
            "behavior_text": "같은 공고 문장" if index < 2 else f"문장 {index}",
            "labels": [POSITIVE, UNKNOWN, UNKNOWN],
        }
        for index in range(12)
    ]
    splits = stratified_text_group_split(records, seed="test")
    locations = {row: split for split, rows in splits.items() for row in rows}

    assert locations[0] == locations[1]
    assert set(locations) == set(range(len(records)))


def test_conservative_score_orders_evidence_unknown_and_contradiction() -> None:
    match, match_state = conservative_behavior_score(
        0.1, POSITIVE, desired_label=POSITIVE
    )
    unknown, unknown_state = conservative_behavior_score(
        0.9, UNKNOWN, desired_label=POSITIVE
    )
    contradiction, contradiction_state = conservative_behavior_score(
        0.9, NEGATIVE, desired_label=POSITIVE
    )

    assert match > unknown > contradiction
    assert match_state == "explicit_match"
    assert unknown_state == "unknown_inferred"
    assert contradiction_state == "explicit_contradiction"


def test_binary_ranking_metrics_reward_correct_order() -> None:
    metrics = binary_ranking_metrics(
        np.asarray([1, 0, 1, 0]), np.asarray([0.9, 0.1, 0.8, 0.2])
    )

    assert metrics["roc_auc"] == 1.0
    assert metrics["average_precision"] == 1.0
    assert metrics["ndcg@10"] == 1.0
