import argparse
import json
from pathlib import Path

from PIL import Image

import scripts.enrich_image_crops as crops
from scripts.enrich_image_crops import (
    Detection,
    bbox_centered,
    expand_bbox,
    normalize_bbox,
    parse_torchvision_output,
    resolve_device,
)


def test_parse_torchvision_output_filters_and_ranks_target_species():
    prediction = {
        "boxes": [
            [10, 10, 20, 20],
            [0, 0, 80, 80],
            [0, 0, 90, 90],
            [5, 5, 4, 10],
            [0, 0, 10, 10],
        ],
        "labels": [2, 2, 3, 2, 2],
        "scores": [0.9, 0.6, 0.99, 0.95, 0.1],
    }

    results = parse_torchvision_output(
        prediction,
        categories=["__background__", "person", "dog", "cat"],
        image_size=(100, 100),
        species="dog",
        conf=0.25,
    )

    assert len(results) == 2
    assert results[0].xyxy == (0.0, 0.0, 80.0, 80.0)
    assert results[0].area_ratio == 0.64
    assert results[1].confidence == 0.9


def test_parse_torchvision_output_clamps_boxes_to_image():
    results = parse_torchvision_output(
        {"boxes": [[-20, -10, 120, 110]], "labels": [1], "scores": [0.8]},
        categories=["__background__", "dog"],
        image_size=(100, 100),
        species="dog",
        conf=0.25,
    )

    assert results[0].xyxy == (0.0, 0.0, 100.0, 100.0)
    assert results[0].area_ratio == 1.0


def test_parse_torchvision_output_selects_cat_without_dog_leakage():
    results = parse_torchvision_output(
        {
            "boxes": [[0, 0, 50, 50], [10, 10, 90, 90]],
            "labels": [1, 2],
            "scores": [0.95, 0.85],
        },
        categories=["__background__", "dog", "cat"],
        image_size=(100, 100),
        species="cat",
        conf=0.25,
    )

    assert len(results) == 1
    assert results[0].class_name == "cat"


def test_explicit_devices_keep_cli_compatibility_without_runtime_detection():
    assert resolve_device("0") == "cuda:0"
    assert resolve_device("cuda:1") == "cuda:1"
    assert resolve_device("cpu") == "cpu"


def test_bbox_helpers_keep_crop_in_bounds_and_report_centering():
    bbox = (10.0, 20.0, 90.0, 80.0)

    assert normalize_bbox(bbox, 100, 100) == [0.1, 0.2, 0.9, 0.8]
    assert expand_bbox(bbox, 100, 100, margin=0.25) == (0, 5, 100, 95)
    assert bbox_centered(bbox, 100, 100)


def test_process_records_preserves_crop_metadata_contract(tmp_path, monkeypatch):
    source = tmp_path / "input.json"
    output = tmp_path / "output.json"
    crop_dir = tmp_path / "crops"
    source.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "desertionNo": "dog-1",
                        "image_url": "https://example.test/dog.jpg",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    class FakeDetector:
        device = "cpu"

        @staticmethod
        def detect(image, species, conf):
            assert image.size == (100, 80)
            assert species == "dog"
            assert conf == 0.25
            return [Detection("dog", 0.9, (10.0, 8.0, 90.0, 72.0), 0.64)]

    monkeypatch.setattr(
        crops, "load_torchvision_detector", lambda model, device: FakeDetector()
    )
    monkeypatch.setattr(
        crops, "download_image", lambda url: Image.new("RGB", (100, 80), "white")
    )
    monkeypatch.setattr(
        crops,
        "compute_photo_quality",
        lambda image, prefix="": {f"{prefix}photo_quality_score": 0.75},
    )
    args = argparse.Namespace(
        input=source,
        output=output,
        crop_dir=crop_dir,
        species="dog",
        model=crops.DEFAULT_DETECTOR_MODEL,
        limit=0,
        conf=0.25,
        imgsz=640,
        device="cpu",
        crop_margin=0.08,
        checkpoint_every=0,
        retry_missing_only=False,
    )

    stats = crops.process_records(args)
    payload = json.loads(output.read_text(encoding="utf-8"))
    attrs = payload["items"][0]["image_attrs"]

    assert stats["detected"] == 1
    assert stats["quality_scored"] == 1
    assert stats["crop_quality_scored"] == 1
    assert attrs["detector_backend"] == "torchvision"
    assert attrs["target_detected"] is True
    assert attrs["dog_detected"] is True
    assert attrs["primary_bbox_norm"] == [0.1, 0.1, 0.9, 0.9]
    assert attrs["photo_quality_score"] == 0.75
    assert attrs["crop_photo_quality_score"] == 0.75
    assert Path(attrs["crop_path"]).exists()
