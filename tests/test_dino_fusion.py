from __future__ import annotations

import numpy as np
import pytest

from experiments.dino_fusion.build_index import select_visual_rows
from experiments.dino_fusion.alignment import (
    alignment_attributes,
    blend_embeddings,
    evaluation_prompts,
    make_projection_head,
    project_embeddings,
    projection_head_from_checkpoint,
    semantic_signature,
    stratified_notice_split,
    training_prompts,
)
from experiments.dino_fusion.core import (
    collapse_visual_hits,
    normalize_rows,
    reciprocal_rank_fusion,
)
from experiments.dino_fusion.evaluate_pilot import (
    appearance_prompts,
    attribute_relevance_summary,
    paired_rank_comparison,
)
from experiments.dino_fusion.evaluate_alignment_controls import _metric_distribution


def test_normalize_rows_returns_unit_float32_and_rejects_invalid_rows() -> None:
    normalized = normalize_rows([[3.0, 4.0], [0.0, 2.0]])
    assert normalized.dtype == np.float32
    assert normalized.shape == (2, 2)
    assert np.allclose(np.linalg.norm(normalized, axis=1), 1.0)

    with pytest.raises(ValueError, match="zero-norm"):
        normalize_rows([[0.0, 0.0]])
    with pytest.raises(ValueError, match="non-finite"):
        normalize_rows([[float("nan"), 1.0]])


def test_collapse_visual_hits_excludes_text_and_deduplicates_notice() -> None:
    metas = [
        {"desertionNo": "A", "type": "image"},
        {"desertionNo": "A", "type": "crop_image"},
        {"desertionNo": "A", "type": "text"},
        {"desertionNo": "B", "type": "image"},
    ]
    ranking = collapse_visual_hits(
        [0.4, 0.1, 0.0, 0.2],
        [0, 1, 2, 3],
        metas,
        score_kind="distance",
    )
    assert [row["notice_id"] for row in ranking] == ["A", "B"]
    assert ranking[0]["best_modality"] == "crop_image"
    assert ranking[0]["score"] == pytest.approx(1 / 1.1)


def test_weighted_rrf_is_scale_free_deduplicated_and_deterministic() -> None:
    clip = [
        {"notice_id": "A", "score": 0.9},
        {"notice_id": "B", "score": 0.8},
        {"notice_id": "A", "score": 0.7},
    ]
    dino = [
        {"notice_id": "B", "score": 99.0},
        {"notice_id": "C", "score": -50.0},
    ]
    ranking = reciprocal_rank_fusion(
        {"clip": clip, "dino": dino},
        weights={"clip": 1.0, "dino": 2.0},
        rrf_k=10,
    )
    assert [row["notice_id"] for row in ranking] == ["B", "C", "A"]
    assert ranking[0]["sources"]["clip"]["rank"] == 2
    assert ranking[0]["sources"]["dino"]["rank"] == 1
    assert len(ranking[2]["sources"]) == 1


def test_select_visual_rows_keeps_requested_modalities_and_limit() -> None:
    metas = [
        {"desertionNo": "A", "type": "text"},
        {"desertionNo": "A", "type": "image"},
        {
            "desertionNo": "A",
            "type": "crop_image",
            "embedding_source": "dog_crop",
        },
        {"desertionNo": "B", "type": "crop_image"},
    ]
    selected = select_visual_rows(metas, sources={"crop_image"}, limit=1)
    assert [index for index, _meta in selected] == [2]


def test_paired_rank_comparison_counts_missing_rank_as_worst() -> None:
    queries = [
        {"systems": {"left": {"rank": 1}, "right": {"rank": 2}}},
        {"systems": {"left": {"rank": 3}, "right": {"rank": 3}}},
        {"systems": {"left": {"rank": None}, "right": {"rank": 4}}},
    ]
    assert paired_rank_comparison(queries, left="left", right="right") == {
        "wins": 1,
        "ties": 1,
        "losses": 1,
    }


def test_appearance_prompts_use_only_color_and_weight_size() -> None:
    prompts = appearance_prompts(
        {
            "color": "검정&흰색",
            "weight": "3(Kg)",
            "desc_full": "공격적이라는 비시각 문구는 사용하면 안 됨",
        }
    )
    assert prompts == {
        "english": "a photo of a very small dog with black and white fur",
        "korean": "검정과흰색 털의 아주 작은 강아지 사진",
    }


def test_attribute_relevance_uses_color_and_size_not_exact_notice() -> None:
    metas = {
        "A": {"color": "흰색", "weight": "3(Kg)"},
        "B": {"color": "흰색", "weight": "2(Kg)"},
        "C": {"color": "검정", "weight": "3(Kg)"},
    }
    queries = [
        {
            "notice_id": "A",
            "systems": {"text": {"top_ids": ["B", "C", "A"]}},
        }
    ]
    summary = attribute_relevance_summary(
        queries,
        system="text",
        metas=metas,
        candidate_notice_ids=set(metas),
    )
    assert summary["query_count"] == 1
    assert summary["precision@5"] == 0.4
    assert summary["hit@5"] == 1.0


def test_alignment_prompts_are_bilingual_and_evaluation_phrasing_is_held_out() -> None:
    attributes = alignment_attributes(
        {"color": "검정&흰색", "weight": "3(Kg)", "age": "2025(년생)"},
        reference_year=2026,
    )
    assert attributes == {
        "colors": ("black", "white"),
        "size": "tiny",
        "age": "puppy",
    }
    assert semantic_signature(attributes).startswith("colors=black,white|size=tiny")
    train = training_prompts(attributes)
    evaluation = evaluation_prompts(attributes)
    assert len(train["english"]) == len(train["korean"]) == 3
    assert evaluation["english"] not in train["english"]
    assert evaluation["korean"] not in train["korean"]


def test_stratified_split_is_deterministic_and_keeps_singletons_in_train() -> None:
    records = [
        {"notice_id": "A", "signature": "common"},
        {"notice_id": "B", "signature": "common"},
        {"notice_id": "C", "signature": "common"},
        {"notice_id": "D", "signature": "singleton"},
    ]
    first = stratified_notice_split(records, validation_fraction=0.34, seed="x")
    second = stratified_notice_split(records, validation_fraction=0.34, seed="x")
    assert first == second
    train, validation = first
    assert 3 in train
    assert 3 not in validation
    assert set(train).isdisjoint(validation)
    assert sorted(train + validation) == list(range(len(records)))


def test_blend_embeddings_stays_normalized_and_validates_weight() -> None:
    blended = blend_embeddings([[1.0, 0.0]], [[0.0, 1.0]], text_weight=0.25)
    assert np.linalg.norm(blended[0]) == pytest.approx(1.0)
    assert blended[0, 0] > blended[0, 1]
    with pytest.raises(ValueError, match="between zero and one"):
        blend_embeddings([[1.0, 0.0]], [[0.0, 1.0]], text_weight=1.1)


def test_flow_head_is_hyperspherical_tangent_and_checkpoint_portable() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(7)
    head = make_projection_head(
        "flow",
        input_dim=8,
        output_dim=12,
        flow_width=6,
        flow_depth=2,
        flow_steps=3,
        flow_time_dim=4,
    )
    values = torch.randn(5, 8)
    mapped = project_embeddings(head, values)
    assert mapped.shape == (5, 12)
    assert torch.allclose(mapped.norm(dim=-1), torch.ones(5), atol=1e-6)

    initial = head.initial_state(values)
    velocity = head.velocity(initial, torch.linspace(0.0, 1.0, 5))
    radial_component = (velocity * initial).sum(dim=-1)
    assert torch.allclose(radial_component, torch.zeros(5), atol=1e-6)

    checkpoint = {
        "architecture": "flow",
        "input_dim": 8,
        "output_dim": 12,
        "hidden_dim": 16,
        "flow_width": 6,
        "flow_depth": 2,
        "flow_steps": 3,
        "flow_time_dim": 4,
        "state_dict": head.state_dict(),
    }
    rebuilt = projection_head_from_checkpoint(checkpoint)
    rebuilt.load_state_dict(checkpoint["state_dict"])
    assert torch.allclose(project_embeddings(rebuilt, values), mapped)


def test_flow_head_matches_mlp_parameter_budget() -> None:
    pytest.importorskip("torch")
    mlp = make_projection_head("mlp", input_dim=512, output_dim=768, hidden_dim=512)
    flow = make_projection_head(
        "flow", input_dim=512, output_dim=768, flow_width=96, flow_depth=4
    )
    mlp_count = sum(parameter.numel() for parameter in mlp.parameters())
    flow_count = sum(parameter.numel() for parameter in flow.parameters())
    assert mlp_count == 656_640
    assert 0.9 * mlp_count <= flow_count <= 1.1 * mlp_count


def test_flow_head_rejects_invalid_configuration() -> None:
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="must be positive"):
        make_projection_head("flow", input_dim=8, output_dim=12, flow_steps=0)


def test_control_metric_distribution_reports_seed_variation() -> None:
    distribution = _metric_distribution([0.1, 0.2, 0.3])
    assert distribution == {
        "mean": 0.2,
        "sample_std": 0.1,
        "min": 0.1,
        "max": 0.3,
    }
