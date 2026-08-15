from __future__ import annotations

import numpy as np

from experiments.dino_fusion_external_eval.evaluate_performance_ablations import (
    appearance_query_ko,
    appearance_relevance,
    gallery_scores,
    is_appearance_query,
    ranking_metrics,
    select_gallery_view,
    select_text_weight,
)


def _record(
    pet_id: str,
    *,
    color: str = "Black",
    size: str = "Medium",
    age: str = "Adult",
    coat: str = "Short",
) -> dict[str, object]:
    return {
        "pet_id": pet_id,
        "appearance": {
            "color_primary": color,
            "size": size,
            "age": age,
            "coat": coat,
        },
    }


def test_appearance_query_and_relevance_use_known_structured_fields() -> None:
    query = _record("a")
    candidates = [
        _record("a"),
        _record("b", coat="Long"),
        _record("c", color="White / Cream", size="Small", coat="Long"),
    ]

    assert is_appearance_query(query) is True
    assert "검은색" in appearance_query_ko(query)
    assert "중간 체구" in appearance_query_ko(query)
    assert appearance_relevance(query, candidates).tolist() == [4.0, 3.0, 1.0]
    assert is_appearance_query(_record("x", color="")) is False


def test_gallery_scores_mask_missing_crops_and_multiview_takes_maximum() -> None:
    query = np.asarray([[1.0, 0.0]], dtype=np.float32)
    full = np.asarray([[0.8, 0.6], [0.9, 0.1]], dtype=np.float32)
    crop = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    available = np.asarray([True, False])

    crop_scores = gallery_scores(
        query,
        gallery_full=full,
        gallery_crop=crop,
        crop_available=available,
        view="crop",
    )
    multi_scores = gallery_scores(
        query,
        gallery_full=full,
        gallery_crop=crop,
        crop_available=available,
        view="multiview",
    )

    assert crop_scores[0, 0] == 1.0
    assert np.isneginf(crop_scores[0, 1])
    assert multi_scores[0, 0] == 1.0
    assert multi_scores[0, 1] > 0.9


def test_ranking_metrics_count_unavailable_identity_as_missed() -> None:
    records = [_record("a"), _record("b"), _record("c")]
    scores = np.asarray(
        [
            [-np.inf, 0.9, 0.8],
            [0.2, 1.0, 0.1],
        ]
    )

    metrics = ranking_metrics(
        scores,
        records=records,
        reference_positions=[0, 1],
    )

    assert metrics["identity_query_count"] == 2
    assert metrics["appearance_query_count"] == 2
    assert metrics["recall@1"] == 0.5


def test_validation_selectors_prefer_safe_gain_and_existing_view_on_tie() -> None:
    tied = {
        view: {"recall@10": 1.0, "mrr": 0.9, "appearance_ndcg@10": 0.8}
        for view in ("crop", "full", "multiview")
    }
    assert select_gallery_view(tied) == "crop"

    sweep = [
        {
            "text_weight": 0.0,
            "appearance_ndcg@10": 0.50,
            "mrr": 0.90,
            "recall10_drop": 0.0,
            "mrr_drop": 0.0,
        },
        {
            "text_weight": 0.1,
            "appearance_ndcg@10": 0.58,
            "mrr": 0.89,
            "recall10_drop": 0.01,
            "mrr_drop": 0.01,
        },
        {
            "text_weight": 0.2,
            "appearance_ndcg@10": 0.70,
            "mrr": 0.70,
            "recall10_drop": 0.10,
            "mrr_drop": 0.20,
        },
    ]
    assert select_text_weight(sweep) == 0.1
