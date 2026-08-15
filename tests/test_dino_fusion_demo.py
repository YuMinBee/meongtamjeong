from __future__ import annotations

import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from experiments.dino_fusion.demo_app import app, decode_image_data_url


def image_data_url() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 6), (20, 40, 60)).save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def test_decode_image_data_url_returns_detached_rgb_image() -> None:
    image = decode_image_data_url(image_data_url())
    try:
        assert image.mode == "RGB"
        assert image.size == (8, 6)
        assert image.getpixel((0, 0)) == (20, 40, 60)
    finally:
        image.close()


@pytest.mark.parametrize(
    "value",
    [
        "",
        "data:text/plain;base64,SGVsbG8=",
        "data:image/png;base64,not-valid-base64",
    ],
)
def test_decode_image_data_url_rejects_invalid_payload(value: str) -> None:
    with pytest.raises(ValueError, match="이미지"):
        decode_image_data_url(value)


def test_demo_page_and_status_do_not_load_models() -> None:
    client = TestClient(app)
    page = client.get("/")
    status = client.get("/api/status")

    assert page.status_code == 200
    assert "CLIP vs DINO + 정렬 CLIP" in page.text
    assert status.status_code == 200
    assert status.json()["state"] == "not_loaded"
