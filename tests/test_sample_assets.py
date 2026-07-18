from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "assets" / "samples"
MANIFEST_PATH = SAMPLE_DIR / "manifest.json"


def test_public_notice_samples_have_complete_provenance() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert manifest["source"]["dataset_url"].startswith("https://www.data.go.kr/")
    assert manifest["rights"]["project_apache_license_applies"] is False
    assert manifest["transformation"]["generative_edit"] is False
    assert len(manifest["samples"]) == 4

    declared_files = set()
    for sample in manifest["samples"]:
        file_name = sample["file"]
        declared_files.add(file_name)
        path = SAMPLE_DIR / file_name

        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sample["sha256"]
        with Image.open(path) as image:
            assert image.format == "JPEG"
            assert image.size == (sample["crop_width"], sample["crop_height"])
        assert sample["desertion_no"] in sample["notice_url"]
        assert sample["source_image_url"].startswith(("http://", "https://"))
        assert sample["notice_url"].startswith("https://www.animal.go.kr/")
        assert sample["status_at_snapshot"] == "보호중"
        assert sample["last_verified_at"]

    image_files = {path.name for path in SAMPLE_DIR.glob("*.jpg")}
    assert image_files == declared_files


def test_legacy_unattributed_samples_are_removed() -> None:
    assert not (SAMPLE_DIR / "jindo.webp").exists()
    assert not (SAMPLE_DIR / "pome.jpg").exists()
