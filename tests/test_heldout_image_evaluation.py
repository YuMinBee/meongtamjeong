from __future__ import annotations

import argparse
import ast
import io
import json
import socket
import threading
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Iterator

import numpy as np
import pytest
from PIL import Image

from app.heldout_image_evaluation import (
    CandidatePlan,
    DownloadedImage,
    HeldoutCandidate,
    HttpxSessionAdapter,
    ImageDownloadError,
    MARKDOWN_LIMITATION_KO_BY_EN,
    SafePublicImageDownloader,
    aggregate_rank_metrics,
    build_candidate_plan,
    build_plan_report,
    canonical_http_url,
    rank_notice_scores,
    rank_record,
    report_failures,
    report_to_markdown,
    select_stratified_sample,
    sha256_bytes,
    validate_public_http_url,
)
from app.index_provenance import (
    EXPECTED_CLIP_COMMIT,
    ProvenanceError,
    verify_installed_clip_commit,
)
from scripts import evaluate_heldout_image_retrieval as cli


ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = ROOT / "docs/evaluation/heldout_image_retrieval.appearance_v1.json"
REPORT_MD = ROOT / "docs/evaluation/heldout_image_retrieval.appearance_v1.md"


def active_row(
    dog_id: str,
    row_type: str = "image",
    primary: str | None = None,
    secondary: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    first = primary or f"https://images.example/{dog_id}-primary.jpg"
    second = secondary or f"https://images.example/{dog_id}-secondary.jpg"
    source = {
        "image": "full_image",
        "crop_image": "dog_crop",
        "text": "public_notice_text",
    }[row_type]
    return {
        "desertionNo": dog_id,
        "type": row_type,
        "embedding_source": source,
        "embedding_image_url": first if row_type != "text" else "",
        "image_url": first,
        "image_urls": [first, second],
        "process_state": "보호중",
        "notice_end": "20260810",
        "org_name": "서울특별시",
        "weight": "5(Kg)",
        **extra,
    }


def candidate(
    dog_id: str, region: str = "서울", size: str = "small"
) -> HeldoutCandidate:
    return HeldoutCandidate(
        notice_id=dog_id,
        indexed_url=f"https://images.example/{dog_id}-primary.jpg",
        secondary_url=f"https://images.example/{dog_id}-secondary.jpg",
        region=region,
        size=size,
        indexed_vector_count=2,
        modalities=("crop_image", "full_image"),
    )


def image_bytes(fmt: str = "PNG", size: tuple[int, int] = (3, 2)) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", size, color=(80, 120, 40)).save(output, format=fmt)
    return output.getvalue()


class FakeResponse:
    def __init__(
        self,
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

    def iter_content(self, *, chunk_size: int) -> Iterator[bytes]:
        assert chunk_size > 0
        yield from self.chunks

    def close(self) -> None:
        self.closed = True


class FakeSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = iter(responses)
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.headers: dict[str, str] = {}
        self.trust_env = True
        self.closed = False

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append((url, kwargs))
        return next(self.responses)

    def close(self) -> None:
        self.closed = True


class FakeHttpxResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        chunks: list[bytes] | None = None,
        stream_error: Exception | None = None,
    ) -> None:
        self.status_code = status_code
        self.headers = headers or {}
        self.chunks = chunks or []
        self.stream_error = stream_error
        self.closed = False

    def iter_bytes(self, *, chunk_size: int) -> Iterator[bytes]:
        assert chunk_size > 0
        yield from self.chunks
        if self.stream_error is not None:
            raise self.stream_error


class FakeHttpxModule:
    class HTTPError(Exception):
        pass

    class TimeoutException(HTTPError):
        pass

    class Timeout:
        def __init__(self, default: float, *, connect: float) -> None:
            self.default = default
            self.connect = connect

    class _Context:
        def __init__(self, response: FakeHttpxResponse | Exception) -> None:
            self.response = response

        def __enter__(self) -> FakeHttpxResponse:
            if isinstance(self.response, Exception):
                raise self.response
            return self.response

        def __exit__(self, *_args: Any) -> None:
            if isinstance(self.response, FakeHttpxResponse):
                self.response.closed = True

    class _Client:
        def __init__(self, owner: "FakeHttpxModule", options: dict[str, Any]) -> None:
            self.owner = owner
            self.options = options
            self.calls: list[dict[str, Any]] = []
            self.closed = False

        def stream(self, method: str, url: str, **kwargs: Any) -> Any:
            self.calls.append({"method": method, "url": url, **kwargs})
            return FakeHttpxModule._Context(next(self.owner.responses))

        def close(self) -> None:
            self.closed = True

    def __init__(self, responses: list[FakeHttpxResponse | Exception]) -> None:
        self.responses = iter(responses)
        self.clients: list[FakeHttpxModule._Client] = []

    def Client(self, **kwargs: Any) -> _Client:
        client = self._Client(self, kwargs)
        self.clients.append(client)
        return client


def resolver_for(*addresses: str):
    def resolve(host: str, port: int, *, type: int) -> list[tuple[Any, ...]]:
        assert host and port and type == socket.SOCK_STREAM
        return [
            (
                socket.AF_INET6 if ":" in address else socket.AF_INET,
                socket.SOCK_STREAM,
                6,
                "",
                (address, port),
            )
            for address in addresses
        ]

    return resolve


def test_candidate_plan_filters_status_text_only_and_url_leakage() -> None:
    shared = "https://images.example/indexed-elsewhere.jpg"
    metas = [
        active_row("A"),
        active_row("A", "crop_image"),
        active_row("A", "text"),
        active_row("B", "text"),
        active_row("C", process_state="종료"),
        active_row("D", secondary=shared),
        active_row("E", primary=shared),
    ]
    plan = build_candidate_plan(
        metas,
        reference_date=datetime(2026, 7, 26),
        max_samples=10,
        seed="fixed",
    )

    assert [item.notice_id for item in plan.eligible] == ["A", "E"]
    assert plan.eligible[0].modalities == ("crop_image", "full_image")
    assert plan.corpus["active_notices"] == 4
    assert plan.corpus["visual_indexed_notices"] == 4
    assert plan.exclusions == {
        "no_indexed_full_image": 1,
        "not_active": 1,
        "secondary_url_already_indexed": 1,
    }


def test_sha_stratified_sample_is_order_independent_and_covers_strata() -> None:
    values = [
        candidate("A"),
        candidate("B"),
        candidate("C", "경기", "medium"),
        candidate("D", "제주", "large"),
        candidate("E", "제주", "large"),
    ]
    first = select_stratified_sample(values, max_samples=4, seed="seed")
    second = select_stratified_sample(
        list(reversed(values)), max_samples=4, seed="seed"
    )

    assert first == second
    assert len(first) == 4
    assert {item.stratum for item in first} == {
        "서울|small",
        "경기|medium",
        "제주|large",
    }


def test_url_gate_blocks_credentials_localhost_private_and_mixed_dns() -> None:
    assert canonical_http_url("HTTPS://Example.COM:443/a#x") == "https://example.com/a"
    with pytest.raises(ValueError, match="credentialed_url"):
        canonical_http_url("https://user:pass@example.com/dog.jpg")
    with pytest.raises(ValueError, match="localhost_blocked"):
        validate_public_http_url("http://localhost/a", resolver=resolver_for("8.8.8.8"))
    for answers in (("127.0.0.1",), ("10.0.0.1",), ("8.8.8.8", "10.0.0.1")):
        with pytest.raises(ValueError, match="non_public_address"):
            validate_public_http_url(
                "https://images.example/a",
                resolver=resolver_for(*answers),
            )


def test_downloader_follows_safe_redirect_and_blocks_private_redirect() -> None:
    payload = image_bytes()
    safe_responses = [
        FakeResponse("https://images.example/start", 302, {"Location": "/dog.png"}),
        FakeResponse(
            "https://images.example/dog.png",
            headers={"Content-Type": "image/png"},
            chunks=[payload],
        ),
    ]
    session = FakeSession(safe_responses)
    result = SafePublicImageDownloader(
        session=session,
        resolver=resolver_for("8.8.8.8"),
        allowed_hosts=["images.example"],
        retries=0,
    ).fetch("http://images.example/start")
    assert result.payload_sha256 == sha256_bytes(payload)
    assert result.image.size == (3, 2)
    assert session.trust_env is False
    assert len(session.calls) == 2
    assert session.calls[0][0] == "https://images.example/start"
    assert all(not kwargs["allow_redirects"] for _, kwargs in session.calls)
    assert all(kwargs["proxies"] == {} for _, kwargs in session.calls)
    assert all(response.closed for response in safe_responses)
    result.image.close()

    redirect = FakeResponse(
        "https://images.example/start",
        302,
        {"Location": "https://169.254.169.254/latest/meta-data"},
    )
    blocked_session = FakeSession([redirect])
    downloader = SafePublicImageDownloader(
        session=blocked_session,
        resolver=resolver_for("8.8.8.8"),
        retries=0,
    )
    with pytest.raises(ImageDownloadError, match="non_public_address"):
        downloader.fetch("https://images.example/start")
    assert len(blocked_session.calls) == 1

    downgrade = FakeResponse(
        "https://images.example/start",
        302,
        {"Location": "http://images.example/insecure.png"},
    )
    downgrade_session = FakeSession([downgrade])
    downloader = SafePublicImageDownloader(
        session=downgrade_session,
        resolver=resolver_for("8.8.8.8"),
        allowed_hosts=["images.example"],
        retries=0,
        use_env_proxy=True,
        proxy_resolver=lambda _url: {"https": "http://proxy.example.invalid:8080"},
    )
    with pytest.raises(ImageDownloadError, match="https_downgrade_redirect"):
        downloader.fetch("https://images.example/start")
    assert len(downgrade_session.calls) == 1
    assert downgrade.closed


def test_environment_proxy_is_explicit_and_never_leaks_configuration() -> None:
    payload = image_bytes()
    disabled_session = FakeSession(
        [
            FakeResponse(
                "https://images.example/dog.png",
                headers={"Content-Type": "image/png"},
                chunks=[payload],
            )
        ]
    )

    def proxy_must_not_be_read(_url: str) -> dict[str, str]:
        raise AssertionError("proxy environment must remain opt-in")

    disabled = SafePublicImageDownloader(
        session=disabled_session,
        resolver=resolver_for("8.8.8.8"),
        allowed_hosts=["images.example"],
        proxy_resolver=proxy_must_not_be_read,
    )
    downloaded = disabled.fetch("http://images.example/dog.png")
    assert disabled_session.calls[0][0] == "https://images.example/dog.png"
    assert disabled_session.calls[0][1]["proxies"] == {}
    downloaded.image.close()
    disabled.close()
    assert disabled_session.closed

    proxy_marker = "http://proxy.example.invalid:8080"
    resolved_urls: list[str] = []

    def resolve_proxy(url: str) -> dict[str, str]:
        resolved_urls.append(url)
        return {"HTTPS": proxy_marker, "no": "ignored.example"}

    enabled_session = FakeSession(
        [
            FakeResponse(
                "https://images.example/dog.png",
                headers={"Content-Type": "image/png"},
                chunks=[payload],
            )
        ]
    )
    enabled = SafePublicImageDownloader(
        session=enabled_session,
        resolver=resolver_for("8.8.8.8"),
        allowed_hosts=["images.example"],
        use_env_proxy=True,
        proxy_resolver=resolve_proxy,
    )
    downloaded = enabled.fetch("http://images.example/dog.png")
    assert resolved_urls == ["https://images.example/dog.png"]
    assert enabled_session.calls[0][1]["proxies"] == {"https": proxy_marker}
    assert enabled_session.trust_env is False
    downloaded.image.close()
    enabled.close()
    assert enabled_session.closed

    leaked_marker = "proxy-config-marker"

    def invalid_proxy(_url: str) -> dict[str, str]:
        raise ValueError(leaked_marker)

    invalid = SafePublicImageDownloader(
        session=FakeSession([]),
        resolver=resolver_for("8.8.8.8"),
        allowed_hosts=["images.example"],
        use_env_proxy=True,
        proxy_resolver=invalid_proxy,
    )
    with pytest.raises(ImageDownloadError) as caught:
        invalid.fetch("https://images.example/dog.png")
    assert caught.value.reason == "proxy_configuration_invalid"
    assert leaked_marker not in str(caught.value)
    invalid.close()


def test_httpx_adapter_reuses_safe_streaming_contract_with_explicit_proxy() -> None:
    payload = image_bytes()
    response = FakeHttpxResponse(
        headers={"Content-Type": "image/png", "Content-Length": str(len(payload))},
        chunks=[payload[:5], payload[5:]],
    )
    httpx_module = FakeHttpxModule([response])
    session = HttpxSessionAdapter(httpx_module=httpx_module)
    proxy_marker = "http://httpx-proxy.example.invalid:8080"
    downloader = SafePublicImageDownloader(
        session=session,
        resolver=resolver_for("8.8.8.8"),
        allowed_hosts=["images.example"],
        use_env_proxy=True,
        proxy_resolver=lambda _url: {"https": proxy_marker},
        retries=0,
    )

    downloaded = downloader.fetch("http://images.example/dog.png")

    assert downloaded.payload_sha256 == sha256_bytes(payload)
    assert len(httpx_module.clients) == 1
    client = httpx_module.clients[0]
    assert client.options == {
        "trust_env": False,
        "proxy": proxy_marker,
        "follow_redirects": False,
    }
    assert client.calls[0]["method"] == "GET"
    assert client.calls[0]["url"] == "https://images.example/dog.png"
    assert client.calls[0]["follow_redirects"] is False
    assert client.calls[0]["headers"]["User-Agent"].startswith("MeongTamjeong")
    assert client.calls[0]["timeout"].connect == 5.0
    assert response.closed
    downloaded.image.close()
    downloader.close()
    assert client.closed


def test_httpx_adapter_preserves_downgrade_and_size_gates() -> None:
    redirect = FakeHttpxResponse(
        status_code=302,
        headers={"Location": "http://images.example/insecure.png"},
    )
    redirect_httpx = FakeHttpxModule([redirect])
    redirect_downloader = SafePublicImageDownloader(
        session=HttpxSessionAdapter(httpx_module=redirect_httpx),
        resolver=resolver_for("8.8.8.8"),
        allowed_hosts=["images.example"],
        retries=0,
    )
    with pytest.raises(ImageDownloadError, match="https_downgrade_redirect"):
        redirect_downloader.fetch("https://images.example/start")
    assert len(redirect_httpx.clients[0].calls) == 1
    assert redirect.closed
    redirect_downloader.close()

    oversized = FakeHttpxResponse(
        headers={"Content-Type": "image/png", "Content-Length": "101"},
    )
    oversized_httpx = FakeHttpxModule([oversized])
    oversized_downloader = SafePublicImageDownloader(
        session=HttpxSessionAdapter(httpx_module=oversized_httpx),
        resolver=resolver_for("8.8.8.8"),
        allowed_hosts=["images.example"],
        max_bytes=100,
        retries=0,
    )
    with pytest.raises(ImageDownloadError, match="declared_size_exceeded"):
        oversized_downloader.fetch("https://images.example/large.png")
    assert oversized.closed
    oversized_downloader.close()


@pytest.mark.parametrize(
    ("failure_stage", "expected_reason"),
    [("request", "request_timeout"), ("stream", "stream_read_failed")],
)
def test_httpx_errors_are_reduced_to_safe_reasons(
    failure_stage: str,
    expected_reason: str,
) -> None:
    marker = "upstream-httpx-detail-marker"
    httpx_module = FakeHttpxModule([])
    if failure_stage == "request":
        response: FakeHttpxResponse | Exception = httpx_module.TimeoutException(marker)
    else:
        response = FakeHttpxResponse(
            headers={"Content-Type": "image/png"},
            chunks=[b"partial"],
            stream_error=httpx_module.HTTPError(marker),
        )
    httpx_module.responses = iter([response])
    downloader = SafePublicImageDownloader(
        session=HttpxSessionAdapter(httpx_module=httpx_module),
        resolver=resolver_for("8.8.8.8"),
        allowed_hosts=["images.example"],
        retries=0,
    )

    with pytest.raises(ImageDownloadError) as caught:
        downloader.fetch("https://images.example/dog.png")
    assert caught.value.reason == expected_reason
    assert marker not in str(caught.value)
    downloader.close()


def test_parallel_downloads_are_bounded_ordered_and_close_thread_sessions() -> None:
    state = {
        "active": 0,
        "max_active": 0,
        "live_images": 0,
        "max_live_images": 0,
    }
    lock = threading.Lock()
    barriers = {0: threading.Barrier(4), 4: threading.Barrier(4)}
    releases = {0: threading.Event(), 4: threading.Event()}
    completion_order: list[int] = []
    created: list[Any] = []

    class TrackedImage:
        def __init__(self) -> None:
            self.closed = False
            with lock:
                state["live_images"] += 1
                state["max_live_images"] = max(
                    state["max_live_images"], state["live_images"]
                )

        def close(self) -> None:
            if self.closed:
                return
            self.closed = True
            with lock:
                state["live_images"] -= 1

    class TrackingDownloader:
        def __init__(self) -> None:
            self.closed = False
            with lock:
                created.append(self)

        def fetch(self, url: str) -> DownloadedImage:
            value = int(url)
            batch_start = 0 if value < 4 else 4
            with lock:
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
            barriers[batch_start].wait(timeout=2)
            if value == batch_start:
                releases[batch_start].wait(timeout=2)
            with lock:
                completion_order.append(value)
            if value == batch_start + 3:
                releases[batch_start].set()
            with lock:
                state["active"] -= 1
            if value == 2:
                raise ImageDownloadError("request_failed")
            payload = f"payload-{value}".encode()
            return DownloadedImage(
                image=TrackedImage(),
                payload_sha256=sha256_bytes(payload),
                byte_count=len(payload),
                image_format="PNG",
                width=2,
                height=2,
                final_url=url,
            )

        def close(self) -> None:
            self.closed = True

    outcomes: list[tuple[int, str | None]] = []
    requests = tuple((value, str(value)) for value in range(8))
    for token, downloaded, reason in cli._ordered_parallel_downloads(
        requests,
        downloader_factory=TrackingDownloader,
        workers=4,
    ):
        outcomes.append((token, reason))
        if downloaded is not None:
            downloaded.image.close()

    assert [token for token, _reason in outcomes] == list(range(8))
    assert outcomes[2] == (2, "request_failed")
    assert completion_order.index(0) > completion_order.index(3)
    assert completion_order.index(4) > completion_order.index(7)
    assert state["max_active"] == 4
    assert state["max_live_images"] <= 4
    assert state["live_images"] == 0
    assert len(created) == 4
    assert all(downloader.closed for downloader in created)

    for invalid_workers in (0, 5):
        with pytest.raises(ValueError, match="download workers"):
            list(
                cli._ordered_parallel_downloads(
                    requests[:1],
                    downloader_factory=TrackingDownloader,
                    workers=invalid_workers,
                )
            )


@pytest.mark.parametrize(
    ("response", "options", "reason"),
    [
        (
            FakeResponse(
                "https://images.example/a.png",
                headers={"Content-Type": "image/png", "Content-Length": "101"},
            ),
            {"max_bytes": 100},
            "declared_size_exceeded",
        ),
        (
            FakeResponse(
                "https://images.example/a.png",
                headers={"Content-Type": "image/png"},
                chunks=[b"a" * 60, b"b" * 41],
            ),
            {"max_bytes": 100},
            "stream_size_exceeded",
        ),
        (
            FakeResponse(
                "https://images.example/a.bmp",
                headers={"Content-Type": "image/bmp"},
                chunks=[image_bytes("BMP")],
            ),
            {},
            "unsupported_image_format",
        ),
        (
            FakeResponse(
                "https://images.example/a.png",
                headers={"Content-Type": "image/png"},
                chunks=[image_bytes(size=(3, 2))],
            ),
            {"max_pixels": 5},
            "pixel_limit_exceeded",
        ),
    ],
)
def test_downloader_enforces_size_format_and_pixel_limits(
    response: FakeResponse,
    options: dict[str, int],
    reason: str,
) -> None:
    downloader = SafePublicImageDownloader(
        session=FakeSession([response]),
        resolver=resolver_for("8.8.8.8"),
        retries=0,
        **options,
    )
    with pytest.raises(ImageDownloadError, match=reason):
        downloader.fetch("https://images.example/a.png")
    assert response.closed


def test_visual_scores_exclude_text_dedupe_notice_and_aggregate_metrics() -> None:
    metas = [
        active_row("A"),
        active_row("A", "crop_image"),
        active_row("A", "text"),
        active_row("B"),
    ]
    ranking = rank_notice_scores([0.2, 0.9, 1.0, 0.8], [0, 1, 2, 3], metas)
    assert [row["notice_id"] for row in ranking] == ["A", "B"]
    assert ranking[0]["best_modality"] == "crop_image"
    first = rank_record("B", ranking)
    assert first["rank"] == 2 and first["hit@5"] == 1
    metrics = aggregate_rank_metrics(
        [first, {**first, "rank": 6, "reciprocal_rank": 1 / 6, "hit@5": 0}],
        attempted_count=4,
        downloaded_count=3,
        duplicate_count=1,
    )
    assert metrics["download_coverage"] == 0.75
    assert metrics["evaluable_coverage"] == 0.5
    assert metrics["median_rank"] == 4.0


class FakeIndex:
    ntotal = 4
    d = 512
    metric_type = 1


class FakeModel:
    def eval(self) -> None:
        return None


class FakeClip:
    @staticmethod
    def load(_name: str, *, device: str) -> tuple[FakeModel, object]:
        assert device == "cpu"
        return FakeModel(), object()


class FakeTorch:
    __version__ = "test"

    class cuda:
        @staticmethod
        def is_available() -> bool:
            return False


class MappingDownloader:
    payloads: dict[str, bytes] = {}
    failure_reasons: dict[str, str] = {}
    calls: list[str] = []
    constructor_options: list[dict[str, Any]] = []
    closed_instances = 0

    def __init__(self, **kwargs: Any) -> None:
        self.constructor_options.append(dict(kwargs))

    def fetch(self, url: str) -> DownloadedImage:
        self.calls.append(url)
        if url in self.failure_reasons:
            raise ImageDownloadError(self.failure_reasons[url])
        payload = self.payloads[url]
        return DownloadedImage(
            image=url,
            payload_sha256=sha256_bytes(payload),
            byte_count=len(payload),
            image_format="PNG",
            width=2,
            height=2,
            final_url=url,
        )

    def close(self) -> None:
        type(self).closed_instances += 1


def test_execution_excludes_duplicates_and_zero_evaluable_fails_cli(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    all_choices = tuple(candidate(value) for value in ("A", "B", "C", "D"))
    choices = all_choices[:3]
    plan = CandidatePlan(
        eligible=all_choices,
        sample=choices,
        corpus={
            "metadata_rows": 4,
            "unique_notices": 4,
            "active_notices": 4,
            "full_image_vectors": 4,
            "crop_image_vectors": 0,
            "text_vectors": 0,
            "visual_indexed_notices": 4,
            "eligible_pairs": 4,
            "selected_pairs": 3,
            "visual_index_coverage": 1.0,
        },
        exclusions={},
        distributions={
            "eligible": {"region": {"서울": 3}, "size": {"small": 3}},
            "sample": {"region": {"서울": 3}, "size": {"small": 3}},
        },
        sample_sha256="f" * 64,
    )
    metas = [active_row(value) for value in ("A", "B", "C", "D")]
    MappingDownloader.calls = []
    MappingDownloader.constructor_options = []
    MappingDownloader.closed_instances = 0
    MappingDownloader.failure_reasons = {}
    MappingDownloader.payloads = {
        choices[0].indexed_url: b"primary-A",
        choices[1].indexed_url: b"primary-B",
        choices[2].indexed_url: b"primary-C",
        choices[0].secondary_url: b"primary-A",
        choices[1].secondary_url: b"query-shared",
        choices[2].secondary_url: b"query-shared",
    }
    index_path = tmp_path / "index.bin"
    metas_path = tmp_path / "metas.json"
    index_path.write_bytes(b"index")
    metas_path.write_text(json.dumps(metas), encoding="utf-8")
    args = argparse.Namespace(
        index=index_path,
        metas=metas_path,
        reference_date="2026-07-26",
        seed="seed",
        max_samples=3,
        clip_model="ViT-B/32",
        device="cpu",
        timeout=1.0,
        max_bytes=100,
        max_pixels=100,
        max_redirects=1,
        retries=0,
        allowed_host=("images.example",),
        download_client="httpx",
        download_workers=4,
        use_env_proxy=True,
    )
    proxy_marker = "http://report-proxy-marker.invalid:8080"
    monkeypatch.setenv("HTTPS_PROXY", proxy_marker)
    monkeypatch.setattr(
        cli,
        "load_heavy_runtime",
        lambda: {
            "clip": FakeClip(),
            "faiss": object(),
            "numpy": np,
            "torch": FakeTorch(),
        },
    )
    monkeypatch.setattr(
        cli,
        "verify_declared_provenance",
        lambda *_args: {
            "declared_index_provenance_consistent": True,
            "declared_clip_source_commit": EXPECTED_CLIP_COMMIT,
            "retained_vector_count": 0,
        },
    )
    monkeypatch.setattr(
        cli,
        "verify_installed_clip_commit",
        lambda *_args: {
            "query_encoder_commit_verified": True,
            "query_encoder_commit": EXPECTED_CLIP_COMMIT,
        },
    )
    monkeypatch.setattr(cli, "read_faiss_index", lambda *_args: FakeIndex())
    monkeypatch.setattr(
        cli,
        "extract_normalized_candidate_matrix",
        lambda *_args: np.array(
            [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.4, 0.6]],
            dtype=np.float32,
        ),
    )
    monkeypatch.setattr(cli, "SafePublicImageDownloader", MappingDownloader)
    encode_calls: list[str] = []

    def fake_encode(*args: Any) -> np.ndarray:
        encode_calls.append(args[3])
        return np.array([0.8, 0.2], dtype=np.float32)

    monkeypatch.setattr(cli, "encode_image", fake_encode)
    report = cli.evaluate_network(args, plan, metas)

    assert len(encode_calls) == 1
    assert report["metrics"]["attempted_count"] == 3
    assert report["metrics"]["evaluable_count"] == 1
    assert report["metrics"]["duplicate_count"] == 2
    assert report["metrics"]["payload_audit_source_count"] == 3
    assert all_choices[3].indexed_url not in MappingDownloader.calls
    assert report["failure_reasons"] == {
        "duplicate_secondary_query_bytes": 1,
        "query_matches_indexed_primary_bytes": 1,
    }
    assert report["execution"]["images_persisted"] is False
    assert report["runtime"]["download_client"] == "httpx"
    assert report["runtime"]["download_workers"] == 4
    assert report["runtime"]["environment_proxy_enabled"] is True
    assert report["runtime"]["environment_proxy_values_persisted"] is False
    assert report["runtime"]["declared_index_provenance_consistent"] is True
    assert report["runtime"]["query_encoder_commit_verified"] is True
    assert report["runtime"]["index_encoder_verified"] is False
    assert proxy_marker not in json.dumps(report)
    assert MappingDownloader.closed_instances > 0
    assert all(
        options["use_env_proxy"] is True
        for options in MappingDownloader.constructor_options
    )
    assert all(
        isinstance(options["session"], HttpxSessionAdapter)
        for options in MappingDownloader.constructor_options
    )
    assert report["execution"]["status"] == "completed"
    assert report["execution"]["coverage_status"] == "complete"

    MappingDownloader.calls = []
    MappingDownloader.failure_reasons = {choices[1].indexed_url: "request_failed"}
    MappingDownloader.payloads.update(
        {
            all_choices[3].indexed_url: b"primary-D",
            choices[0].secondary_url: b"query-A",
            choices[2].secondary_url: b"query-C",
        }
    )
    args.audit_all_indexed_sources = False
    partial_report = cli.evaluate_network(args, plan, metas)
    assert partial_report["metrics"]["evaluable_count"] == 2
    assert partial_report["metrics"]["payload_audit_coverage"] == 0.666667
    assert partial_report["execution"]["status"] == "completed"
    assert partial_report["execution"]["coverage_status"] == "partial"
    assert partial_report["execution"]["payload_leakage_audit_complete"] is True
    assert partial_report["execution"]["payload_leakage_audit_scope"] == (
        "successfully_downloaded_sampled_primary_sources"
    )
    assert partial_report["validation"]["passed"] is True
    assert partial_report["failure_reasons"] == {
        "indexed_source_request_failed": 1,
        "primary_unavailable": 1,
    }
    partial_markdown = report_to_markdown(partial_report)
    assert "payload audit coverage" in partial_markdown
    assert "end-to-end Hit@5" in partial_markdown
    assert "indexed_source_request_failed" in partial_markdown

    args.audit_all_indexed_sources = True
    audit_all_partial = cli.evaluate_network(args, plan, metas)
    assert audit_all_partial["metrics"]["evaluable_count"] == 2
    assert audit_all_partial["metrics"]["payload_audit_coverage"] == 0.75
    assert audit_all_partial["execution"]["status"] == "incomplete"
    assert audit_all_partial["execution"]["coverage_status"] == "partial"
    assert audit_all_partial["execution"]["payload_leakage_audit_complete"] is False
    assert audit_all_partial["validation"]["passed"] is False
    assert "network execution is incomplete" in report_failures(audit_all_partial)

    MappingDownloader.calls = []
    MappingDownloader.failure_reasons = {}
    MappingDownloader.payloads.update(
        {
            choices[0].secondary_url: b"primary-A",
            choices[1].secondary_url: b"primary-B",
            choices[2].secondary_url: b"primary-C",
        }
    )
    args.audit_all_indexed_sources = False
    encoded_before_zero = len(encode_calls)
    zero_report = cli.evaluate_network(args, plan, metas)
    assert len(encode_calls) == encoded_before_zero
    assert zero_report["metrics"]["evaluable_count"] == 0
    assert zero_report["execution"]["status"] == "incomplete"
    assert zero_report["execution"]["coverage_status"] == "complete"
    assert zero_report["execution"]["metrics_available"] is False
    assert zero_report["validation"]["passed"] is False
    assert "network execution is incomplete" in report_failures(zero_report)

    monkeypatch.setattr(cli, "build_plan", lambda _args: (plan, metas))
    monkeypatch.setattr(cli, "evaluate_network", lambda *_args: zero_report)
    assert (
        cli.main(
            [
                "--run-network",
                "--json-out",
                str(tmp_path / "zero.json"),
                "--markdown-out",
                str(tmp_path / "zero.md"),
            ]
        )
        == 1
    )


def test_plan_report_is_explicitly_non_measured() -> None:
    choices = (candidate("A"),)
    plan = CandidatePlan(
        eligible=choices,
        sample=choices,
        corpus={
            "active_notices": 2,
            "visual_indexed_notices": 1,
            "eligible_pairs": 1,
            "selected_pairs": 1,
            "visual_index_coverage": 0.5,
        },
        exclusions={"no_indexed_full_image": 1},
        distributions={
            "eligible": {"region": {"서울": 1}, "size": {"small": 1}},
            "sample": {"region": {"서울": 1}, "size": {"small": 1}},
        },
        sample_sha256="a" * 64,
    )
    report = build_plan_report(
        plan,
        reference_date="2026-07-26",
        seed="seed",
        max_samples=120,
        artifacts={
            "index": {"path": "index", "sha256": "b" * 64},
            "metas": {"path": "metas", "sha256": "c" * 64},
        },
        clip_model="ViT-B/32",
    )
    markdown = report_to_markdown(report)
    assert report_failures(report, expected_sample_sha256="a" * 64) == []
    assert report["execution"]["metrics_available"] is False
    assert "upgrade initial HTTP" in report["scope"]["transport_policy"]
    assert "아직 이미지 다운로드" in markdown
    assert "입양 적합성·성격·건강을 평가하지 않습니다" in markdown


def test_plan_cli_is_deterministic_and_check_detects_stale_hash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    index_path = tmp_path / "index.bin"
    metas_path = tmp_path / "metas.json"
    json_out = tmp_path / "report.json"
    markdown_out = tmp_path / "report.md"
    index_path.write_bytes(b"index-v1")
    metas_path.write_text(
        json.dumps([active_row("A"), active_row("A", "text")]),
        encoding="utf-8",
    )
    common = [
        "--index",
        str(index_path),
        "--metas",
        str(metas_path),
        "--json-out",
        str(json_out),
        "--markdown-out",
        str(markdown_out),
        "--max-samples",
        "1",
    ]
    monkeypatch.setattr(
        cli,
        "verify_declared_provenance",
        lambda *_args: {"declared_index_provenance_consistent": True},
    )
    assert cli.main(["--plan", *common]) == 0
    first = (json_out.read_bytes(), markdown_out.read_bytes())
    assert cli.main(["--dry-run", *common]) == 0
    assert first == (json_out.read_bytes(), markdown_out.read_bytes())
    assert cli.main(["--check", *common]) == 1
    assert cli.main(["--check", "--allow-plan-only-check", *common]) == 0
    altered = json.loads(json_out.read_text(encoding="utf-8"))
    altered["sample"][0]["notice_id"] = "ALTERED"
    json_out.write_text(json.dumps(altered, ensure_ascii=False), encoding="utf-8")
    markdown_out.write_text(report_to_markdown(altered), encoding="utf-8")
    assert cli.main(["--check", "--allow-plan-only-check", *common]) == 1
    assert cli.main(["--plan", *common]) == 0
    index_path.write_bytes(b"index-v2")
    assert cli.main(["--check", "--allow-plan-only-check", *common]) == 1


def test_cli_proxy_is_opt_in_and_workers_are_bounded() -> None:
    defaults = cli.parse_args(["--plan"])
    assert defaults.download_workers == 4
    assert defaults.download_client == "requests"
    assert defaults.use_env_proxy is False
    assert defaults.allow_plan_only_check is False

    opted_in = cli.parse_args(
        [
            "--plan",
            "--download-workers",
            "2",
            "--download-client",
            "httpx",
            "--use-env-proxy",
        ]
    )
    assert opted_in.download_workers == 2
    assert opted_in.download_client == "httpx"
    assert opted_in.use_env_proxy is True

    for invalid_workers in ("0", "5"):
        with pytest.raises(SystemExit):
            cli.parse_args(["--plan", "--download-workers", invalid_workers])
    with pytest.raises(SystemExit):
        cli.parse_args(["--plan", "--download-client", "unknown"])
    with pytest.raises(SystemExit):
        cli.parse_args(["--plan", "--allow-plan-only-check"])
    assert (
        cli.parse_args(["--check", "--allow-plan-only-check"]).allow_plan_only_check
        is True
    )

    assert (
        "httpx>=0.27,<1"
        in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    )


def test_current_snapshot_declared_provenance_is_artifact_bound() -> None:
    args = cli.parse_args(["--plan"])
    metas = cli.load_metas(args.metas)
    result = cli.verify_declared_provenance(args, len(metas))

    assert result["declared_index_provenance_consistent"] is True
    assert result["declared_dimension"] == 512
    assert result["declared_vector_count"] == len(metas) == 3746
    assert result["declared_clip_source_commit"] == EXPECTED_CLIP_COMMIT
    assert result["retained_vector_count"] == 2045


def test_declared_provenance_mismatch_fails_before_runtime_or_downloader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bad_manifest = tmp_path / "snapshot_manifest.json"
    payload = json.loads(cli.DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    payload["embedding"]["model"] = "different encoder"
    bad_manifest.write_text(json.dumps(payload), encoding="utf-8")
    args = cli.parse_args(["--run-network", "--manifest", str(bad_manifest)])
    metas = cli.load_metas(args.metas)

    def must_not_run(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("runtime and downloader must remain untouched")

    monkeypatch.setattr(cli, "load_heavy_runtime", must_not_run)
    monkeypatch.setattr(cli, "SafePublicImageDownloader", must_not_run)
    with pytest.raises(ProvenanceError, match="manifest embedding model"):
        cli.evaluate_network(args, object(), metas)


def test_installed_query_encoder_requires_official_direct_url_commit(
    tmp_path: Path,
) -> None:
    distribution_root = tmp_path / "site-packages"
    module_path = distribution_root / "clip" / "__init__.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text("", encoding="utf-8")

    class FakeDistribution:
        def __init__(self, commit: str) -> None:
            self.commit = commit

        def read_text(self, name: str) -> str:
            assert name == "direct_url.json"
            return json.dumps(
                {
                    "url": "https://github.com/openai/CLIP.git",
                    "vcs_info": {
                        "vcs": "git",
                        "commit_id": self.commit,
                    },
                }
            )

        def locate_file(self, _path: str) -> Path:
            return distribution_root

    class FakeClipModule:
        __file__ = str(module_path)

    verified = verify_installed_clip_commit(
        FakeClipModule,
        distribution_getter=lambda _name: FakeDistribution(EXPECTED_CLIP_COMMIT),
    )
    assert verified["query_encoder_commit_verified"] is True
    assert verified["query_encoder_commit"] == EXPECTED_CLIP_COMMIT

    with pytest.raises(ProvenanceError, match="installed CLIP commit"):
        verify_installed_clip_commit(
            FakeClipModule,
            distribution_getter=lambda _name: FakeDistribution("0" * 40),
        )


def test_cli_lazy_imports_heavy_and_network_packages() -> None:
    imports: set[str] = set()
    for source_path in (
        Path(cli.__file__),
        ROOT / "app" / "heldout_image_evaluation.py",
    ):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.add((node.module or "").split(".")[0])
    assert {"clip", "faiss", "torch", "numpy", "requests", "httpx"}.isdisjoint(imports)


def test_committed_completed_report_is_auditable_and_recalculable() -> None:
    assert cli.DEFAULT_JSON_OUT == REPORT_JSON
    assert cli.DEFAULT_MARKDOWN_OUT == REPORT_MD
    report = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    assert report["execution"]["status"] == "completed"
    assert report["execution"]["metrics_available"] is True
    assert report["execution"]["network_requested"] is True
    assert report["execution"]["images_persisted"] is False
    assert report["runtime"]["network_executed"] is True
    assert report["runtime"]["declared_index_provenance_consistent"] is True
    assert report["runtime"]["query_encoder_commit_verified"] is True
    assert report["runtime"]["index_encoder_verified"] is False
    assert report["corpus"]["active_notices"] == 1516
    assert report["corpus"]["visual_indexed_notices"] == 1228
    assert report["corpus"]["eligible_pairs"] == 1228
    assert report["corpus"]["selected_pairs"] == 120
    assert report["corpus"]["text_vectors"] == 1516
    assert len(report["sample"]) == 120
    metrics = report["metrics"]
    queries = report["queries"]
    attempted = metrics["attempted_count"]
    evaluable = metrics["evaluable_count"]
    downloaded = metrics["downloaded_count"]
    embedded = metrics["embedded_count"]
    primary_downloaded = metrics["primary_downloaded_count"]
    payload_audit_downloaded = metrics["payload_audit_downloaded_count"]
    payload_audit_sources = metrics["payload_audit_source_count"]
    assert attempted == report["corpus"]["selected_pairs"] == len(report["sample"])
    assert evaluable == len(queries) > 0
    assert evaluable <= embedded <= downloaded <= attempted
    assert 0 <= metrics["duplicate_count"] <= downloaded
    assert 0 <= primary_downloaded <= attempted
    assert 0 <= payload_audit_downloaded <= payload_audit_sources
    assert metrics["download_coverage"] == pytest.approx(
        round(downloaded / attempted, 6), abs=5e-7
    )
    assert metrics["evaluable_coverage"] == pytest.approx(
        round(evaluable / attempted, 6), abs=5e-7
    )
    assert metrics["primary_download_coverage"] == pytest.approx(
        round(primary_downloaded / attempted, 6), abs=5e-7
    )
    assert metrics["payload_audit_coverage"] == pytest.approx(
        round(payload_audit_downloaded / payload_audit_sources, 6), abs=5e-7
    )
    ranks = [int(row["rank"]) for row in queries]
    for cutoff in (1, 5, 10):
        hits = sum(rank <= cutoff for rank in ranks)
        assert metrics[f"hit@{cutoff}"] == pytest.approx(
            round(hits / evaluable, 6), abs=5e-7
        )
        assert metrics[f"end_to_end_hit@{cutoff}"] == pytest.approx(
            round(hits / attempted, 6), abs=5e-7
        )
    assert metrics["hit@1"] <= metrics["hit@5"] <= metrics["hit@10"]
    assert (
        metrics["end_to_end_hit@1"]
        <= metrics["end_to_end_hit@5"]
        <= metrics["end_to_end_hit@10"]
    )
    assert metrics["MRR"] == pytest.approx(
        round(sum(1.0 / rank for rank in ranks) / evaluable, 6), abs=5e-7
    )
    assert metrics["median_rank"] == pytest.approx(median(ranks))
    assert report["validation"] == {
        "passed": True,
        "failure_count": 0,
        "failures": [],
    }
    assert (
        report_failures(
            report,
            expected_sample_sha256=report["sample_policy"]["sample_sha256"],
        )
        == []
    )
    assert cli.check_report(cli.parse_args(["--check"])) == []
    assert REPORT_MD.read_text(encoding="utf-8") == report_to_markdown(report)
    assert set(report["limitations"]) == set(MARKDOWN_LIMITATION_KO_BY_EN)
    markdown = REPORT_MD.read_text(encoding="utf-8")
    for english, korean in MARKDOWN_LIMITATION_KO_BY_EN.items():
        assert english in report["limitations"]
        assert english not in markdown
        assert korean in markdown
    for name, path in (
        ("index", cli.DEFAULT_INDEX),
        ("metas", cli.DEFAULT_METAS),
        ("snapshot_manifest", cli.DEFAULT_MANIFEST),
        ("active_index_sync_report", cli.DEFAULT_SYNC_REPORT),
        ("requirements_lock", cli.DEFAULT_REQUIREMENTS_LOCK),
    ):
        assert report["artifacts"][name]["sha256"] == cli.sha256_file(path)
