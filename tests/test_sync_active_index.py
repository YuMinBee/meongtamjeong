from __future__ import annotations

import argparse
import io
import json
import shutil
from collections.abc import Iterator
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from PIL import Image

from app import public_image_download as public_images
from scripts import sync_active_index as sync


REFERENCE_DATE = datetime(2026, 7, 25)


def test_sync_reexports_shared_public_image_download_contract() -> None:
    assert sync.PublicImageDownloader is public_images.PublicImageDownloader
    assert sync.parse_public_image_hostname is public_images.parse_public_image_hostname
    assert sync.DEFAULT_ALLOWED_IMAGE_HOSTS == public_images.DEFAULT_ALLOWED_IMAGE_HOSTS
    assert sync.DEFAULT_IMAGE_MAX_BYTES == public_images.DEFAULT_IMAGE_MAX_BYTES
    assert sync.DEFAULT_IMAGE_MAX_PIXELS == public_images.DEFAULT_IMAGE_MAX_PIXELS


@pytest.fixture
def workdir() -> Iterator[Path]:
    runtime_root = Path(__file__).resolve().parent / ".runtime"
    directory = runtime_root / f"sync-index-{uuid4().hex}"
    directory.mkdir(parents=True)
    try:
        yield directory
    finally:
        shutil.rmtree(directory)
        try:
            runtime_root.rmdir()
        except OSError:
            pass


class FakeEncoder:
    dimension = 2

    def encode_image(self, image: object) -> np.ndarray | None:
        assert image == "downloaded-image"
        return np.array([0.6, 0.8], dtype=np.float32)

    def encode_text(self, text: str) -> np.ndarray | None:
        assert "vlm secret" not in text
        assert "믹스견" not in text
        return np.array([1.0, 0.0], dtype=np.float32)


class FakeImageResponse:
    def __init__(
        self,
        *,
        url: str,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        chunks: list[bytes] | None = None,
    ) -> None:
        self.url = url
        self.status_code = status_code
        self.headers = headers or {}
        self.chunks = chunks or []
        self.closed = False
        self.iterated = False

    def iter_content(self, *, chunk_size: int) -> Iterator[bytes]:
        assert chunk_size == sync.IMAGE_DOWNLOAD_CHUNK_BYTES
        self.iterated = True
        yield from self.chunks

    def close(self) -> None:
        self.closed = True


class FakeImageSession:
    def __init__(self, responses: list[FakeImageResponse]) -> None:
        self.responses = iter(responses)
        self.calls: list[tuple[str, dict[str, object]]] = []

    def get(self, url: str, **kwargs: object) -> FakeImageResponse:
        self.calls.append((url, kwargs))
        return next(self.responses)


def png_bytes(size: tuple[int, int] = (2, 2)) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", size, color=(120, 80, 40)).save(output, format="PNG")
    return output.getvalue()


def test_public_image_downloader_streams_bounded_image_with_split_timeouts() -> None:
    payload = png_bytes((3, 2))
    response = FakeImageResponse(
        url="https://images.example.test/dog.png",
        headers={
            "Content-Type": "image/png; charset=binary",
            "Content-Length": str(len(payload)),
        },
        chunks=[payload[:8], payload[8:]],
    )
    session = FakeImageSession([response])
    downloader = sync.PublicImageDownloader(
        timeout=7,
        max_bytes=len(payload) + 1,
        allowed_hosts={"images.example.test"},
    )
    downloader._session = session

    image = downloader("https://images.example.test/dog.png")

    assert image is not None
    assert image.mode == "RGB"
    assert image.size == (3, 2)
    assert response.iterated is True
    assert response.closed is True
    assert len(session.calls) == 1
    _, kwargs = session.calls[0]
    assert kwargs["stream"] is True
    assert kwargs["allow_redirects"] is False
    connect_timeout, read_timeout = kwargs["timeout"]
    assert 0 < connect_timeout <= 5
    assert 0 < read_timeout <= 7


@pytest.mark.parametrize("content_type", ["application/octet-stream", ""])
def test_public_image_downloader_accepts_generic_or_missing_type_by_image_magic(
    content_type: str,
) -> None:
    payload = png_bytes()
    headers = {
        "Content-Disposition": 'attachment; filename="public-dog.png"',
    }
    if content_type:
        headers["Content-Type"] = content_type
    response = FakeImageResponse(
        url="http://openapi.animal.go.kr/public-dog.png",
        headers=headers,
        chunks=[payload],
    )
    session = FakeImageSession([response])
    downloader = sync.PublicImageDownloader()
    downloader._session = session

    image = downloader("https://openapi.animal.go.kr/public-dog.png")

    assert image is not None
    assert image.size == (2, 2)
    assert response.iterated is True
    assert response.closed is True


@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/dog.jpg",
        "data:image/png;base64,AA==",
        "ftp://images.example.test/dog.jpg",
        "//images.example.test/dog.jpg",
        "https://user:password@images.example.test/dog.jpg",
        "https://images.example.test/line\nbreak.jpg",
    ],
)
def test_public_image_downloader_rejects_non_http_or_credential_urls(url: str) -> None:
    session = FakeImageSession([])
    downloader = sync.PublicImageDownloader()
    downloader._session = session

    assert downloader(url) is None
    assert session.calls == []


def test_public_image_downloader_rejects_declared_and_streamed_oversize() -> None:
    declared = FakeImageResponse(
        url="https://images.example.test/declared.jpg",
        headers={"Content-Type": "image/jpeg", "Content-Length": "11"},
        chunks=[b"not-read"],
    )
    streamed = FakeImageResponse(
        url="https://images.example.test/streamed.jpg",
        headers={"Content-Type": "image/jpeg"},
        chunks=[b"123456", b"78901"],
    )
    downloader = sync.PublicImageDownloader(
        max_bytes=10,
        allowed_hosts={"images.example.test"},
    )

    declared_session = FakeImageSession([declared])
    downloader._session = declared_session
    assert downloader("https://images.example.test/declared.jpg") is None
    assert declared.iterated is False
    assert declared.closed is True

    streamed_session = FakeImageSession([streamed])
    downloader._session = streamed_session
    assert downloader("https://images.example.test/streamed.jpg") is None
    assert streamed.iterated is True
    assert streamed.closed is True


@pytest.mark.parametrize(
    "response",
    [
        FakeImageResponse(
            url="https://images.example.test/error.jpg",
            status_code=503,
            headers={"Content-Type": "image/jpeg"},
        ),
        FakeImageResponse(
            url="https://images.example.test/not-image.jpg",
            headers={"Content-Type": "text/html"},
            chunks=[b"<html>error</html>"],
        ),
        FakeImageResponse(
            url="file:///redirected.jpg",
            headers={"Content-Type": "image/jpeg"},
            chunks=[b"not an image"],
        ),
        FakeImageResponse(
            url="https://images.example.test/broken.jpg",
            headers={"Content-Type": "image/jpeg"},
            chunks=[b"not an image"],
        ),
    ],
)
def test_public_image_downloader_rejects_bad_responses_and_decode(
    response: FakeImageResponse,
) -> None:
    session = FakeImageSession([response])
    downloader = sync.PublicImageDownloader(allowed_hosts={"images.example.test"})
    downloader._session = session

    assert downloader("https://images.example.test/dog.jpg") is None
    assert response.closed is True


def test_public_image_downloader_prefers_https_and_keeps_http_fallback() -> None:
    payload = png_bytes()
    failed_https = FakeImageResponse(
        url="https://www.animal.go.kr/dog.png",
        status_code=503,
        headers={"Content-Type": "image/png"},
    )
    successful_http = FakeImageResponse(
        url="http://www.animal.go.kr/dog.png",
        headers={"Content-Type": "image/png"},
        chunks=[payload],
    )
    session = FakeImageSession([failed_https, successful_http])
    downloader = sync.PublicImageDownloader(
        timeout=5,
        allowed_hosts={"www.animal.go.kr"},
    )
    downloader._session = session

    image = downloader("http://www.animal.go.kr/dog.png")

    assert image is not None
    assert image.size == (2, 2)
    assert [url for url, _kwargs in session.calls] == [
        "https://www.animal.go.kr/dog.png",
        "http://www.animal.go.kr/dog.png",
    ]
    first_timeout = session.calls[0][1]["timeout"]
    second_timeout = session.calls[1][1]["timeout"]
    assert first_timeout[1] < 5
    assert second_timeout[1] > 0
    assert failed_https.closed is True
    assert successful_http.closed is True


def test_public_image_downloader_stops_after_upgraded_https_success() -> None:
    payload = png_bytes()
    successful_https = FakeImageResponse(
        url="https://openapi.animal.go.kr/dog.png",
        headers={"Content-Type": "image/png"},
        chunks=[payload],
    )
    session = FakeImageSession([successful_https])
    downloader = sync.PublicImageDownloader(timeout=15)
    downloader._session = session

    image = downloader("http://openapi.animal.go.kr/dog.png")

    assert image is not None
    assert [url for url, _kwargs in session.calls] == [
        "https://openapi.animal.go.kr/dog.png"
    ]
    assert session.calls[0][1]["timeout"][1] < 15
    assert successful_https.closed is True


def test_public_image_downloader_rejects_excessive_pixel_count() -> None:
    payload = png_bytes((3, 2))
    response = FakeImageResponse(
        url="https://images.example.test/large.png",
        headers={"Content-Type": "image/png"},
        chunks=[payload],
    )
    downloader = sync.PublicImageDownloader(
        max_pixels=5,
        allowed_hosts={"images.example.test"},
    )
    downloader._session = FakeImageSession([response])

    assert downloader("https://images.example.test/large.png") is None
    assert response.closed is True


def active_notice(dog_id: str, **extra: object) -> dict[str, object]:
    return {
        "type": "live",
        "desertionNo": dog_id,
        "process_state": "보호중",
        "notice_end": "20260810",
        "last_verified_at": "2026-07-25T12:00:00+09:00",
        "image_url": f"https://fresh.test/{dog_id}.jpg",
        "detail_url": f"https://fresh.test/notices/{dog_id}",
        "desc": f"{dog_id} 공고 설명",
        "age": "2024(년생)",
        "weight": "5(Kg)",
        "color": "흰색",
        **extra,
    }


def test_sync_retains_active_vectors_updates_metadata_and_adds_new_ids() -> None:
    existing_vectors = np.array(
        [
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
            [0.7, 0.8],
            [0.9, 0.1],
            [0.2, 0.3],
        ],
        dtype=np.float32,
    )
    existing_metas = [
        {
            "desertionNo": "A",
            "type": "image",
            "embedding_source": "full_image",
            "image_url": "https://fresh.test/A.jpg",
            "process_state": "종료",
            "vlm_attrs": {"personality": "unsupported"},
        },
        {
            "desertionNo": "A",
            "type": "text",
            "desc_full": "stale text",
            "image_url": "https://fresh.test/A.jpg",
        },
        {
            "desertionNo": "D",
            "type": "image",
            "image_url": "https://stale.test/D.jpg",
        },
        {
            "desertionNo": "D",
            "type": "crop_image",
            "image_url": "https://stale.test/D.jpg",
        },
        {
            "desertionNo": "D",
            "type": "text",
            "desc_full": "stale D text",
            "image_url": "https://stale.test/D.jpg",
        },
        {
            "desertionNo": "REMOVED",
            "type": "image",
            "image_url": "https://stale.test/removed.jpg",
        },
    ]
    fresh = [
        active_notice(
            "C",
            breed="믹스견",
            vlm_desc="vlm secret",
            personality="inferred",
        ),
        active_notice("D"),
        active_notice("A", desc="최신 공고 설명"),
    ]
    originals = deepcopy((existing_vectors, existing_metas, fresh))

    result = sync.synchronize_active_vectors(
        existing_vectors,
        existing_metas,
        fresh,
        reference_date=REFERENCE_DATE,
        encoder=FakeEncoder(),
        image_downloader=lambda url: "downloaded-image",
        quality_evaluator=lambda image: {
            "photo_quality_score": 0.75,
            "photo_quality_band": "high",
        },
    )

    assert [meta["type"] for meta in result.metas] == [
        "image",
        "text",
        "image",
        "text",
        "image",
        "text",
    ]
    assert [meta["desertionNo"] for meta in result.metas] == [
        "A",
        "A",
        "C",
        "C",
        "D",
        "D",
    ]
    np.testing.assert_array_equal(result.vectors[0], existing_vectors[0])
    np.testing.assert_array_equal(
        result.vectors[1:],
        np.array(
            [
                [1.0, 0.0],
                [0.6, 0.8],
                [1.0, 0.0],
                [0.6, 0.8],
                [1.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )

    for retained in result.metas[:2]:
        assert retained["process_state"] == "보호중"
        assert retained["image_url"] == "https://fresh.test/A.jpg"
        assert retained["detail_url"] == "https://fresh.test/notices/A"
        assert "vlm_attrs" not in retained
    assert result.metas[0]["embedding_source"] == "full_image"
    assert result.metas[0]["embedding_image_url"] == "https://fresh.test/A.jpg"
    assert result.metas[2]["image_attrs"]["photo_quality_score"] == 0.75
    assert "vlm_desc" not in result.metas[2]
    assert "personality" not in result.metas[2]
    assert result.metas[3]["embedding_source"] == "public_notice_text"
    assert result.metas[4]["embedding_image_url"] == "https://fresh.test/D.jpg"
    assert all(meta["type"] != "crop_image" for meta in result.metas)
    assert result.metas[1]["embedding_text_sha256"] == sync.public_text_sha256(
        result.metas[1]["desc_full"]
    )

    stats = result.report["stats"]
    assert stats["retained_vectors"] == 1
    assert stats["retained_image_vectors"] == 1
    assert stats["retained_text_vectors"] == 0
    assert stats["removed_existing_vectors"] == 5
    assert stats["removed_stale_vectors"] == 1
    assert stats["removed_changed_image_vectors"] == 1
    assert stats["removed_changed_crop_vectors"] == 1
    assert stats["removed_changed_text_vectors"] == 2
    assert stats["removed_existing_unique_ids"] == 1
    assert stats["refresh_required_unique_ids"] == 2
    assert stats["refresh_image_required_unique_ids"] == 1
    assert stats["refresh_text_required_unique_ids"] == 2
    assert stats["refreshed_unique_ids"] == 2
    assert stats["refreshed_image_vectors"] == 1
    assert stats["refreshed_text_vectors"] == 2
    assert stats["new_unique_ids"] == 1
    assert stats["new_image_vectors"] == 1
    assert stats["new_text_vectors"] == 1
    assert stats["output_vectors"] == 6
    assert (existing_metas, fresh) == (originals[1], originals[2])
    np.testing.assert_array_equal(existing_vectors, originals[0])


def test_image_failure_still_adds_text_only() -> None:
    current_a = active_notice("A")
    result = sync.synchronize_active_vectors(
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        [
            {
                "desertionNo": "A",
                "type": "image",
                "image_url": current_a["image_url"],
            },
            {
                "desertionNo": "A",
                "type": "text",
                "desc_full": sync.build_public_notice_text(current_a),
            },
        ],
        [current_a, active_notice("B")],
        reference_date=REFERENCE_DATE,
        encoder=FakeEncoder(),
        image_downloader=lambda url: None,
    )

    assert [meta["desertionNo"] for meta in result.metas] == ["A", "A", "B"]
    assert [meta["type"] for meta in result.metas] == ["image", "text", "text"]
    assert result.report["stats"]["new_image_download_failures"] == 1
    assert result.report["stats"]["new_text_vectors"] == 1
    assert result.report["stats"]["new_ids_without_vectors"] == 0
    assert result.report["stats"]["active_ids_without_vectors"] == 0
    assert result.report["stats"]["active_ids_without_public_text_vectors"] == 0


def test_public_text_encoder_failure_aborts_even_when_image_succeeds() -> None:
    class TextFailureEncoder(FakeEncoder):
        def encode_text(self, text: str) -> np.ndarray | None:
            raise RuntimeError("simulated persistent GPU failure")

    current_a = active_notice("A")
    with pytest.raises(
        ValueError,
        match="public text embedding failed for notice B: RuntimeError",
    ):
        sync.synchronize_active_vectors(
            np.array([[1.0, 0.0]], dtype=np.float32),
            [
                {
                    "desertionNo": "A",
                    "type": "text",
                    "desc_full": sync.build_public_notice_text(current_a),
                }
            ],
            [current_a, active_notice("B")],
            reference_date=REFERENCE_DATE,
            encoder=TextFailureEncoder(),
            image_downloader=lambda url: "downloaded-image",
        )


def test_missing_public_notice_text_aborts_instead_of_creating_image_only_id() -> None:
    no_text = {
        "desertionNo": "B",
        "process_state": "보호중",
        "notice_end": "20260810",
        "image_url": "https://fresh.test/B.jpg",
    }

    with pytest.raises(
        ValueError,
        match="public text embedding failed for notice B: public notice text is empty",
    ):
        sync.synchronize_active_vectors(
            np.array([[1.0, 0.0]], dtype=np.float32),
            [{"desertionNo": "REMOVED", "type": "text"}],
            [no_text],
            reference_date=REFERENCE_DATE,
            encoder=FakeEncoder(),
            image_downloader=lambda url: "downloaded-image",
        )


def test_changed_image_failure_removes_stale_crop_but_keeps_current_text() -> None:
    fresh = active_notice("A")
    current_text = sync.build_public_notice_text(fresh)

    result = sync.synchronize_active_vectors(
        np.array(
            [[0.1, 0.9], [0.2, 0.8], [1.0, 0.0]],
            dtype=np.float32,
        ),
        [
            {
                "desertionNo": "A",
                "type": "image",
                "image_url": "https://old.test/A.jpg",
            },
            {
                "desertionNo": "A",
                "type": "crop_image",
                "image_url": "https://old.test/A.jpg",
            },
            {
                "desertionNo": "A",
                "type": "text",
                "desc_full": current_text,
            },
        ],
        [fresh],
        reference_date=REFERENCE_DATE,
        encoder=FakeEncoder(),
        image_downloader=lambda url: None,
    )

    assert [meta["type"] for meta in result.metas] == ["text"]
    np.testing.assert_array_equal(
        result.vectors,
        np.array([[1.0, 0.0]], dtype=np.float32),
    )
    stats = result.report["stats"]
    assert stats["removed_changed_image_vectors"] == 1
    assert stats["removed_changed_crop_vectors"] == 1
    assert stats["retained_text_vectors"] == 1
    assert stats["refreshed_image_download_failures"] == 1
    assert stats["refreshes_without_replacement_vectors"] == 1


def test_unchanged_image_crop_and_text_vectors_are_retained_without_encoder() -> None:
    fresh = active_notice("A")
    current_text = sync.build_public_notice_text(fresh)
    vectors = np.array(
        [[0.1, 0.9], [0.2, 0.8], [1.0, 0.0]],
        dtype=np.float32,
    )

    result = sync.synchronize_active_vectors(
        vectors,
        [
            {
                "desertionNo": "A",
                "type": "image",
                "image_url": fresh["image_url"],
            },
            {
                "desertionNo": "A",
                "type": "crop_image",
                "image_url": fresh["image_url"],
            },
            {
                "desertionNo": "A",
                "type": "text",
                "desc_full": current_text,
            },
        ],
        [fresh],
        reference_date=REFERENCE_DATE,
        encoder=None,
        image_downloader=None,
    )

    np.testing.assert_array_equal(result.vectors, vectors)
    assert [meta["type"] for meta in result.metas] == [
        "image",
        "crop_image",
        "text",
    ]
    stats = result.report["stats"]
    assert stats["retained_vectors"] == 3
    assert stats["retained_crop_vectors"] == 1
    assert stats["refresh_required_unique_ids"] == 0
    assert stats["refreshed_unique_ids"] == 0


def test_active_selection_excludes_closed_expired_and_unknown_by_default() -> None:
    records = [
        active_notice("ACTIVE"),
        active_notice("CLOSED", process_state="입양완료"),
        active_notice("EXPIRED", process_state="", notice_end="20260720"),
        {
            "desertionNo": "UNKNOWN",
            "desc": "status missing",
        },
    ]

    selected, stats = sync.select_active_fresh_records(
        records,
        reference_date=REFERENCE_DATE,
    )
    selected_with_unknown, _ = sync.select_active_fresh_records(
        records,
        reference_date=REFERENCE_DATE,
        include_unknown=True,
    )

    assert list(selected) == ["ACTIVE"]
    assert list(selected_with_unknown) == ["ACTIVE", "UNKNOWN"]
    assert stats["fresh_status_records"] == {
        "active": 1,
        "closed": 1,
        "expired": 1,
        "unknown": 1,
    }


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (
            {
                "items": [active_notice("A")],
                "stats": {"kept_items": 2},
                "include_closed": False,
            },
            "stats.kept_items does not match items length",
        ),
        (
            {
                "items": [active_notice("A")],
                "stats": {"kept_items": 1},
                "include_closed": True,
            },
            "include_closed=true is not allowed",
        ),
    ],
)
def test_fresh_cache_rejects_inconsistent_or_closed_wrapper_metadata(
    workdir: Path,
    payload: dict[str, object],
    match: str,
) -> None:
    path = workdir / "fresh.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        sync.load_fresh_cache(path)


def test_safety_gate_argument_types_are_bounded() -> None:
    assert sync.parse_nonnegative_int("0") == 0
    assert sync.parse_fraction("0") == 0.0
    assert sync.parse_fraction("1") == 1.0
    with pytest.raises(argparse.ArgumentTypeError):
        sync.parse_nonnegative_int("-1")
    for value in ("-0.1", "1.1", "nan", "inf"):
        with pytest.raises(argparse.ArgumentTypeError):
            sync.parse_fraction(value)
    assert sync.parse_public_image_hostname("CDN.Example.org.") == "cdn.example.org"
    for value in (
        "localhost",
        "127.0.0.1",
        "10.0.0.2",
        "metadata.internal",
        "https://cdn.example.org/path",
        "user@cdn.example.org",
    ):
        with pytest.raises(argparse.ArgumentTypeError):
            sync.parse_public_image_hostname(value)


def test_public_image_downloader_rejects_non_allowlisted_host_before_request() -> None:
    session = FakeImageSession([])
    downloader = sync.PublicImageDownloader(allowed_hosts={"openapi.animal.go.kr"})
    downloader._session = session

    assert downloader("https://127.0.0.1/private.png") is None
    assert downloader("https://cdn.example.org/dog.png") is None
    assert session.calls == []


def test_duplicate_active_records_and_new_order_are_deterministic() -> None:
    sparse = active_notice("B", desc="")
    rich = active_notice("B", desc="더 완전한 최신 공고", care_name="보호소")
    first_fresh = [active_notice("C"), sparse, rich]
    second_fresh = [rich, sparse, active_notice("C")]

    first = sync.synchronize_active_vectors(
        np.array([[0.2, 0.8]], dtype=np.float32),
        [{"desertionNo": "A", "type": "text"}],
        first_fresh,
        reference_date=REFERENCE_DATE,
        encoder=FakeEncoder(),
        image_downloader=lambda url: None,
    )
    second = sync.synchronize_active_vectors(
        np.array([[0.2, 0.8]], dtype=np.float32),
        [{"desertionNo": "A", "type": "text"}],
        second_fresh,
        reference_date=REFERENCE_DATE,
        encoder=FakeEncoder(),
        image_downloader=lambda url: None,
    )

    assert first.metas == second.metas
    np.testing.assert_array_equal(first.vectors, second.vectors)
    assert first.report == second.report
    assert [meta["desertionNo"] for meta in first.metas] == ["B", "C"]
    assert first.metas[0]["desc"] == "더 완전한 최신 공고"
    assert first.report["stats"]["fresh_duplicate_active_id_records"] == 1


@pytest.mark.parametrize(
    ("vectors", "metas", "match"),
    [
        (np.array([[1.0, 0.0]], dtype=np.float32), [], "index/metas mismatch"),
        (np.array([1.0, 0.0], dtype=np.float32), [], "2-dimensional"),
        (
            np.array([[float("nan"), 0.0]], dtype=np.float32),
            [{"desertionNo": "A"}],
            "non-finite",
        ),
    ],
)
def test_existing_vector_validation_rejects_inconsistent_inputs(
    vectors: np.ndarray,
    metas: list[dict[str, object]],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        sync.validate_existing_vectors(vectors, metas)


def test_encoder_dimension_and_returned_vector_dimension_are_validated() -> None:
    class WrongDeclaredDimension(FakeEncoder):
        dimension = 3

    with pytest.raises(ValueError, match="encoder/index dimension mismatch"):
        sync.synchronize_active_vectors(
            np.array([[1.0, 0.0]], dtype=np.float32),
            [{"desertionNo": "A", "type": "text"}],
            [active_notice("A"), active_notice("B")],
            reference_date=REFERENCE_DATE,
            encoder=WrongDeclaredDimension(),
            image_downloader=lambda url: None,
        )

    class WrongReturnedDimension(FakeEncoder):
        def encode_text(self, text: str) -> np.ndarray:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)

    current_a = active_notice("A")
    with pytest.raises(ValueError, match="text:B encoder dimension mismatch"):
        sync.synchronize_active_vectors(
            np.array([[1.0, 0.0]], dtype=np.float32),
            [
                {
                    "desertionNo": "A",
                    "type": "text",
                    "desc_full": sync.build_public_notice_text(current_a),
                }
            ],
            [current_a, active_notice("B")],
            reference_date=REFERENCE_DATE,
            encoder=WrongReturnedDimension(),
            image_downloader=lambda url: None,
        )


def test_public_notice_text_does_not_use_breed_vlm_or_inferred_personality() -> None:
    text = sync.build_public_notice_text(
        {
            "desc": "공고 원문",
            "breed": "믹스견",
            "vlm_desc": "사진 추정 설명",
            "personality": "활발함",
            "sex": "F",
            "color": "갈색",
        }
    )

    assert text == "설명 공고 원문, 성별 F, 색상 갈색"
    assert "믹스견" not in text
    assert "사진 추정" not in text
    assert "활발함" not in text


class FakeIndex:
    def __init__(self, vectors: list[list[float]]):
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.ntotal = self.vectors.shape[0]
        self.d = self.vectors.shape[1]

    def reconstruct(self, position: int) -> np.ndarray:
        return self.vectors[position].copy()


class FakeOutputIndex:
    metric_type = 1

    def __init__(self, dimension: int):
        self.d = dimension
        self.vectors = np.empty((0, dimension), dtype=np.float32)
        self.ntotal = 0

    def add(self, vectors: np.ndarray) -> None:
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.ntotal = self.vectors.shape[0]


class FakeFaiss:
    METRIC_L2 = 1

    def __init__(self, existing: FakeIndex):
        existing.metric_type = self.METRIC_L2
        self.existing = existing

    def read_index(self, path: str) -> FakeIndex:
        assert Path(path).exists()
        return self.existing

    @staticmethod
    def IndexFlatL2(dimension: int) -> FakeOutputIndex:
        return FakeOutputIndex(dimension)

    @staticmethod
    def write_index(index: FakeOutputIndex, path: str) -> None:
        Path(path).write_bytes(index.vectors.tobytes(order="C"))


def test_extract_index_vectors_and_atomic_writers(
    workdir: Path,
) -> None:
    vectors = sync.extract_index_vectors(FakeIndex([[1.0, 0.0], [0.0, 1.0]]))
    np.testing.assert_array_equal(
        vectors,
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )

    index_path = workdir / "output.index"
    metas_path = workdir / "metas.json"
    sync.write_index_atomic(
        b"index payload",
        index_path,
        lambda payload, path: Path(path).write_bytes(payload),
    )
    sync.write_json_atomic(metas_path, [{"한글": "값"}])

    assert index_path.read_bytes() == b"index payload"
    raw_json = metas_path.read_bytes()
    assert not raw_json.startswith(b"\xef\xbb\xbf")
    assert raw_json.endswith(b"\n")
    assert json.loads(raw_json.decode("utf-8")) == [{"한글": "값"}]
    assert not list(workdir.glob(".*.tmp"))


def test_run_writes_consistent_index_metas_and_hash_report(
    workdir: Path,
) -> None:
    fresh_path = workdir / "fresh.json"
    input_index = workdir / "existing.index"
    input_metas = workdir / "existing-metas.json"
    index_out = workdir / "synced.index"
    metas_out = workdir / "synced-metas.json"
    report_out = workdir / "sync-report.json"
    fresh_path.write_text(
        json.dumps(
            {
                "fetched_at": "2026-07-25T12:00:00+09:00",
                "items": [active_notice("A"), active_notice("B")],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    input_index.write_bytes(b"fake existing index")
    input_metas.write_text(
        json.dumps([{"desertionNo": "A", "type": "text"}]),
        encoding="utf-8",
    )
    args = sync.parse_args(
        [
            "--fresh-cache",
            str(fresh_path),
            "--existing-index",
            str(input_index),
            "--existing-metas",
            str(input_metas),
            "--index-out",
            str(index_out),
            "--metas-out",
            str(metas_out),
            "--report-out",
            str(report_out),
            "--min-active-notices",
            "2",
            "--max-removed-id-fraction",
            "0.25",
        ]
    )

    report = sync.run(
        args,
        faiss_module=FakeFaiss(FakeIndex([[1.0, 0.0]])),
        encoder=FakeEncoder(),
        image_downloader=lambda url: None,
        quality_evaluator=lambda image: {},
    )

    output_metas = json.loads(metas_out.read_text(encoding="utf-8"))
    stored_report = json.loads(report_out.read_text(encoding="utf-8"))
    assert len(output_metas) == report["stats"]["output_vectors"] == 2
    assert index_out.stat().st_size == 2 * 2 * np.dtype(np.float32).itemsize
    assert stored_report == report
    assert report["artifacts"]["outputs"]["index"]["sha256"] == sync.sha256_file(
        index_out
    )
    assert report["artifacts"]["outputs"]["metas"]["sha256"] == sync.sha256_file(
        metas_out
    )
    assert report["safety_gates"] == {
        "policy": "fail_closed_before_encoding",
        "min_active_notices": {
            "configured_minimum": 2,
            "observed_active_unique_ids": 2,
            "passed": True,
        },
        "max_removed_id_fraction": {
            "configured_maximum": 0.25,
            "observed_removed_fraction": 0.0,
            "observed_removed_existing_unique_ids": 0,
            "observed_existing_unique_ids": 1,
            "passed": True,
        },
    }


@pytest.mark.parametrize(
    ("fresh_ids", "existing_ids", "extra_args", "match"),
    [
        (["A"], ["A"], ["--min-active-notices", "2"], "min-active-notices"),
        (
            ["A"],
            ["A", "B"],
            ["--max-removed-id-fraction", "0.4"],
            "max-removed-id-fraction",
        ),
    ],
)
def test_run_safety_gates_abort_before_any_output_is_written(
    workdir: Path,
    fresh_ids: list[str],
    existing_ids: list[str],
    extra_args: list[str],
    match: str,
) -> None:
    fresh_path = workdir / "fresh.json"
    input_index = workdir / "existing.index"
    input_metas = workdir / "existing-metas.json"
    index_out = workdir / "synced.index"
    metas_out = workdir / "synced-metas.json"
    report_out = workdir / "sync-report.json"
    fresh_path.write_text(
        json.dumps(
            {
                "fetched_at": "2026-07-25T12:00:00+09:00",
                "include_closed": False,
                "stats": {"kept_items": len(fresh_ids)},
                "items": [active_notice(dog_id) for dog_id in fresh_ids],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    input_index.write_bytes(b"must not be read")
    input_metas.write_text(
        json.dumps(
            [{"desertionNo": dog_id, "type": "text"} for dog_id in existing_ids]
        ),
        encoding="utf-8",
    )
    args = sync.parse_args(
        [
            "--fresh-cache",
            str(fresh_path),
            "--existing-index",
            str(input_index),
            "--existing-metas",
            str(input_metas),
            "--index-out",
            str(index_out),
            "--metas-out",
            str(metas_out),
            "--report-out",
            str(report_out),
            *extra_args,
        ]
    )

    with pytest.raises(ValueError, match=match):
        sync.run(args)

    assert not index_out.exists()
    assert not metas_out.exists()
    assert not report_out.exists()


def test_output_paths_cannot_overwrite_inputs_or_each_other(
    workdir: Path,
) -> None:
    fresh = workdir / "fresh.json"
    index = workdir / "input.index"
    metas = workdir / "input-metas.json"
    report = workdir / "report.json"

    with pytest.raises(ValueError, match="must not overwrite input"):
        sync.validate_output_paths(
            input_paths=[fresh, index, metas],
            output_paths=[fresh, workdir / "out.json", report],
        )
    with pytest.raises(ValueError, match="distinct paths"):
        sync.validate_output_paths(
            input_paths=[fresh, index, metas],
            output_paths=[report, report, workdir / "out.index"],
        )


def test_reference_date_requires_fixed_source_timestamp() -> None:
    assert (
        sync.reference_date_from_payload(
            {"fetched_at": "2026-07-25T23:35:45+09:00"}
        ).date()
        == REFERENCE_DATE.date()
    )
    assert sync.reference_date_from_payload({"fetched_at": "invalid"}) is None
    assert sync.reference_date_from_payload([]) is None
