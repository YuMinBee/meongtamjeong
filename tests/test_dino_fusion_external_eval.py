from __future__ import annotations

import csv
import zipfile
from pathlib import Path

import numpy as np

from experiments.dino_fusion.behavior import NEGATIVE, POSITIVE, UNKNOWN
from experiments.dino_fusion_external_eval.compatibility import (
    behavior_evidence_text,
    classification_metrics,
    fuse_visual_behavior,
    ndcg_at_k,
    parse_label,
    split_for_group,
)
from experiments.dino_fusion_external_eval.prepare_benchmark import prepare


def test_parse_label_preserves_unknown() -> None:
    assert parse_label("TRUE") == POSITIVE
    assert parse_label("false") == NEGATIVE
    assert parse_label("NA") == UNKNOWN
    assert parse_label("") == UNKNOWN


def test_behavior_evidence_moves_late_compatibility_phrase_forward() -> None:
    prefix = " ".join(f"word{index}" for index in range(100))
    description = f"{prefix} Calm in a home and good with cats and children."

    evidence = behavior_evidence_text(description)

    assert "good with cats and children" in evidence
    assert len(evidence.split()) <= 60


def test_organization_split_is_deterministic_and_grouped() -> None:
    first = split_for_group("ORG-123", seed="test")
    second = split_for_group("ORG-123", seed="test")

    assert first == second
    assert first in {"train", "validation", "test"}


def test_fusion_and_ndcg_reward_behavior_match() -> None:
    visual = np.asarray([0.9, 0.8, 0.7])
    behavior = np.asarray([0.1, 0.9, 0.2])
    fused = fuse_visual_behavior(visual, behavior, behavior_weight=0.6)

    assert int(np.argmax(fused)) == 1
    assert ndcg_at_k([0, 1, 0], fused, k=3) == 1.0


def test_classification_metrics_ignore_unknown_labels() -> None:
    labels = np.asarray(
        [
            [POSITIVE, POSITIVE, POSITIVE, POSITIVE],
            [NEGATIVE, NEGATIVE, NEGATIVE, NEGATIVE],
            [UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN],
        ]
    )
    probabilities = np.asarray(
        [
            [0.9, 0.8, 0.7, 0.9],
            [0.1, 0.2, 0.3, 0.1],
            [0.0, 1.0, 0.0, 1.0],
        ]
    )

    metrics = classification_metrics(labels, probabilities)

    assert metrics["macro_roc_auc"] == 1.0
    assert all(axis["known_count"] == 2 for axis in metrics["axes"].values())


def test_prepare_joins_ids_and_keeps_photos_local(tmp_path: Path) -> None:
    csv_path = tmp_path / "dogs.csv"
    fields = [
        "id",
        "org_id",
        "species",
        "description",
        "env_children",
        "env_dogs",
        "env_cats",
        "house_trained",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "id": "123",
                "org_id": "ORG",
                "species": "Dog",
                "description": "A sufficiently long public shelter description for this dog.",
                "env_children": "TRUE",
                "env_dogs": "FALSE",
                "env_cats": "NA",
                "house_trained": "TRUE",
            }
        )
    zip_path = tmp_path / "images.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("123/123-1.jpg", b"one")
        archive.writestr("123/123-2.jpg", b"two")
    output_dir = tmp_path / "artifacts"

    manifest = prepare(
        csv_path=csv_path,
        image_zip=zip_path,
        output_dir=output_dir,
        minimum_photos=2,
    )

    assert manifest["text_dataset"]["records"] == 1
    assert manifest["multimodal_dataset"]["records"] == 1
    assert (output_dir / "multimodal_records.json").is_file()
    assert not (output_dir / "images").exists()
