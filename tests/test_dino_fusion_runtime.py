from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.dino_fusion import (
    DinoFusionRuntime,
    DinoFusionSearchResult,
    DinoFusionSettings,
    behavior_preferences_from_text,
    fuse_visual_behavior_scores,
    percentile_scores,
)


def settings(tmp_path: Path, *, mode: str = "off") -> DinoFusionSettings:
    return DinoFusionSettings(
        mode=mode,  # type: ignore[arg-type]
        dino_dir=tmp_path / "dino",
        flow_dir=tmp_path / "flow",
        behavior_dir=tmp_path / "behavior",
    )


def test_settings_default_to_a_non_mutating_rollout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DINO_FUSION_MODE", raising=False)
    monkeypatch.delenv("DINO_FUSION_BEHAVIOR_ENABLED", raising=False)
    configured = DinoFusionSettings.from_env(tmp_path)

    assert configured.mode == "off"
    assert configured.enabled is False
    assert configured.text_weight == pytest.approx(0.2)
    assert configured.behavior_weight == pytest.approx(0.25)
    assert configured.behavior_enabled is False
    assert configured.local_files_only is True
    status = DinoFusionRuntime(configured).status()
    assert status["state"] == "disabled"
    assert status["behavior_enabled"] is False


def test_behavior_branch_requires_explicit_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DINO_FUSION_BEHAVIOR_ENABLED", "true")

    configured = DinoFusionSettings.from_env(tmp_path)

    assert configured.behavior_enabled is True
    assert DinoFusionRuntime(configured).status()["behavior_enabled"] is True


@pytest.mark.parametrize("mode", ["invalid", "production", "on"])
def test_settings_reject_ambiguous_rollout_modes(tmp_path: Path, mode: str) -> None:
    with pytest.raises(ValueError, match="off, shadow, or active"):
        settings(tmp_path, mode=mode)


def test_behavior_preferences_require_explicit_axis_language() -> None:
    preferences = behavior_preferences_from_text("온순하고 활발한 강아지")
    values = {(item.axis, item.desired_label) for item in preferences}

    assert values == {("gentle_handling", 1), ("activity", 1)}
    assert behavior_preferences_from_text("검은색 소형 강아지") == ()


def test_visual_behavior_fusion_preserves_the_pilot_weights() -> None:
    visual = np.asarray([0.1, 0.7, 0.4], dtype=np.float64)
    behavior = np.asarray([1.0, 0.0, 0.5], dtype=np.float64)

    np.testing.assert_allclose(percentile_scores(visual), [0.0, 1.0, 0.5])
    np.testing.assert_allclose(
        fuse_visual_behavior_scores(
            visual,
            behavior,
            behavior_weight=0.25,
        ),
        [0.25, 0.75, 0.5],
    )


def test_search_diagnostics_distinguish_shadow_from_served_results() -> None:
    result = DinoFusionSearchResult(
        vector_scores={3: 0.9},
        vector_score_details={3: {}},
        ranked_notice_ids=["dog-3"],
        query_mode="dino_image+aligned_clip_text",
        behavior_preferences=(),
        latency_ms={"total": 12.3},
        model={"text_weight": 0.2, "behavior_weight": 0.25},
    )

    shadow = result.diagnostics(served=False)
    active = result.diagnostics(served=True)

    assert shadow["served"] is False
    assert active["served"] is True
    assert active["top_notice_ids"] == ["dog-3"]
    assert active["behavior_weight"] is None
    assert active["behavior_disclaimer"] is None
