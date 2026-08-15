"""Pure contracts for held-out, cross-photo CLIP retrieval evaluation.

The evaluation uses a notice's secondary public photo as a query while the
search candidates remain the already indexed primary-photo and dog-crop
vectors.  It is an identity-retrieval proxy for the visual index, not an
adoption-suitability, temperament, or user-preference evaluation.
"""

from __future__ import annotations

import hashlib
import io
import ipaddress
import math
import socket
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from statistics import mean, median
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import quote, urljoin, urlsplit, urlunsplit

from app.graph_rag import infer_region, infer_size_from_weight
from app.hybrid_rag import resolve_vector_modality
from app.notice_status import classify_notice
from app.retrieval_evaluation import merge_notice_metas, notice_id


REPORT_SCHEMA_VERSION = "heldout-image-retrieval.v1"
EVALUATION_ID = "heldout-secondary-photo.appearance.v1"
DEFAULT_REFERENCE_DATE = "2026-07-26"
DEFAULT_SEED = "20260726-heldout-secondary-photo-v1"
DEFAULT_MAX_SAMPLES = 120
VISUAL_MODALITIES = frozenset({"full_image", "crop_image"})
ALLOWED_IMAGE_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})
REDIRECT_STATUS_CODES = frozenset({301, 302, 303, 307, 308})
DOWNLOAD_CHUNK_BYTES = 64 * 1024
MARKDOWN_LIMITATION_KO_BY_EN = {
    (
        "This cross-photo same-notice task measures a visual identity "
        "retrieval proxy, not subjective similarity between different dogs."
    ): (
        "이 교차사진 동일 공고 과제는 시각적 동일성 검색의 대리 지표이며, "
        "서로 다른 개의 주관적 유사도를 측정하지 않습니다."
    ),
    (
        "It does not measure temperament, health, adoption suitability, "
        "child friendliness, or compatibility with other animals."
    ): (
        "성격, 건강, 입양 적합성, 아동 친화성, 다른 동물과의 생활 가능성을 "
        "측정하지 않습니다."
    ),
    (
        "URL inequality is checked before sampling; exact payload SHA-256 "
        "leakage and duplicate-query checks occur only during network execution."
    ): (
        "표본 선정 전에는 URL 차이만 확인하며, 실제 payload SHA-256 누출과 "
        "질의 중복은 네트워크 실행에서만 검사합니다."
    ),
    (
        "In the default sampled audit, missing primary downloads reduce payload "
        "audit coverage and are disclosed. An evaluable query is checked against "
        "its own primary SHA and all successfully downloaded sampled-primary "
        "SHAs, but duplication against an unavailable primary cannot be ruled out."
    ): (
        "기본 표본 감사에서는 primary 다운로드 실패만큼 payload 감사 커버리지가 "
        "낮아지며 이를 공개합니다. 평가 가능한 질의는 자기 primary SHA와 다운로드에 "
        "성공한 모든 표본 primary SHA에 대조하지만, 받지 못한 primary와의 중복은 "
        "배제할 수 없습니다."
    ),
    (
        "A corpus-wide audit requires --audit-all-indexed-sources, may be slow, "
        "and is complete only when every declared active visual source is "
        "successfully downloaded."
    ): (
        "전체 corpus 감사에는 --audit-all-indexed-sources가 필요하고 오래 걸릴 수 "
        "있으며, 선언된 모든 활성 시각 출처 다운로드에 성공해야만 완전합니다."
    ),
    (
        "Image transport upgrades initial HTTP metadata URLs to HTTPS and rejects "
        "redirects that downgrade HTTPS back to HTTP."
    ): (
        "이미지 전송은 초기 HTTP 메타데이터 URL을 HTTPS로 올리고 HTTPS에서 "
        "HTTP로 낮추는 리디렉트를 거부합니다."
    ),
    (
        "Generated crop bytes are not retained with the index, so exact raw-byte "
        "comparison covers their public source photo rather than the crop payload."
    ): (
        "생성한 crop 바이트는 인덱스와 함께 보관하지 않으므로 정확한 raw-byte "
        "비교는 crop payload가 아니라 공개 원본 사진을 대상으로 합니다."
    ),
    (
        "Public image URLs can change or disappear, so download and evaluable "
        "coverage must be reported with retrieval metrics."
    ): (
        "공개 이미지 URL은 바뀌거나 사라질 수 있으므로 검색 지표와 함께 다운로드 "
        "및 평가 가능 커버리지를 보고해야 합니다."
    ),
    (
        "The raw CLIP visual-index result is not the full production API, "
        "which can apply additional filtering and reranking."
    ): (
        "raw CLIP 시각 인덱스 결과는 추가 필터와 재정렬을 적용할 수 있는 전체 "
        "운영 API 결과가 아닙니다."
    ),
    (
        "The refreshed index retained image and crop vectors from earlier "
        "snapshots. It has no per-vector encoder sidecar fingerprint, so "
        "artifact-bound declarations do not prove the encoder of every vector."
    ): (
        "갱신 인덱스에는 이전 스냅샷의 이미지·crop 벡터가 남아 있습니다. 벡터별 "
        "encoder sidecar fingerprint가 없으므로 artifact 선언만으로 모든 벡터의 "
        "encoder를 증명할 수 없습니다."
    ),
}


def clean_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def parse_reference_date(value: Any) -> datetime:
    text = clean_text(value)
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y.%m.%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError(f"invalid reference_date: {text!r}")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def url_sha256(url: str) -> str:
    return sha256_bytes(canonical_http_url(url).encode("utf-8"))


def upgrade_http_to_https(value: Any) -> str:
    """Canonicalize an image URL and upgrade its initial HTTP transport."""

    url = canonical_http_url(value)
    parsed = urlsplit(url)
    if parsed.scheme == "https":
        return url
    return urlunsplit(("https", parsed.netloc, parsed.path, parsed.query, ""))


class UnsafeImageUrl(ValueError):
    """A URL cannot be used by the public-image evaluator."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class ImageDownloadError(RuntimeError):
    """A bounded public image download failed without exposing response data."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def canonical_http_url(value: Any) -> str:
    """Return a comparison-safe HTTP(S) URL without performing DNS resolution."""

    url = clean_text(value)
    if not url or any(ord(character) < 32 for character in url):
        raise UnsafeImageUrl("invalid_url")
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise UnsafeImageUrl("invalid_url") from exc
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise UnsafeImageUrl("unsupported_scheme")
    if parsed.username is not None or parsed.password is not None:
        raise UnsafeImageUrl("credentialed_url")
    hostname = clean_text(parsed.hostname).rstrip(".").lower()
    if not hostname:
        raise UnsafeImageUrl("missing_hostname")
    try:
        ascii_hostname = hostname.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise UnsafeImageUrl("invalid_hostname") from exc
    if ":" in ascii_hostname:
        host_part = f"[{ascii_hostname}]"
    else:
        host_part = ascii_hostname
    default_port = (scheme == "http" and port == 80) or (
        scheme == "https" and port == 443
    )
    netloc = host_part if port is None or default_port else f"{host_part}:{port}"
    path = parsed.path or "/"
    return urlunsplit((scheme, netloc, path, parsed.query, ""))


def _resolved_addresses(
    hostname: str,
    port: int,
    resolver: Callable[..., Any],
) -> set[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    try:
        direct = ipaddress.ip_address(hostname)
    except ValueError:
        try:
            answers = resolver(hostname, port, type=socket.SOCK_STREAM)
        except (OSError, socket.gaierror) as exc:
            raise UnsafeImageUrl("dns_resolution_failed") from exc
        addresses: set[ipaddress.IPv4Address | ipaddress.IPv6Address] = set()
        for answer in answers:
            try:
                raw_address = answer[4][0]
                addresses.add(ipaddress.ip_address(raw_address))
            except (IndexError, TypeError, ValueError):
                continue
        if not addresses:
            raise UnsafeImageUrl("dns_no_addresses")
        return addresses
    return {direct}


def validate_public_http_url(
    value: Any,
    *,
    resolver: Callable[..., Any] = socket.getaddrinfo,
    allowed_hosts: Iterable[str] | None = None,
) -> str:
    """Validate every resolved address before a request is made.

    A mixed public/private DNS answer fails closed.  This check is repeated for
    every redirect by :class:`SafePublicImageDownloader`.
    """

    url = canonical_http_url(value)
    parsed = urlsplit(url)
    hostname = clean_text(parsed.hostname).rstrip(".").lower()
    if hostname == "localhost" or hostname.endswith(".localhost"):
        raise UnsafeImageUrl("localhost_blocked")
    normalized_allowed = {
        clean_text(host).rstrip(".").lower()
        for host in (allowed_hosts or ())
        if clean_text(host)
    }
    if normalized_allowed and hostname not in normalized_allowed:
        raise UnsafeImageUrl("host_not_allowed")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    addresses = _resolved_addresses(hostname, port, resolver)
    if any(not address.is_global for address in addresses):
        raise UnsafeImageUrl("non_public_address")
    return url


@dataclass(frozen=True)
class HeldoutCandidate:
    notice_id: str
    indexed_url: str
    secondary_url: str
    region: str
    size: str
    indexed_vector_count: int
    modalities: tuple[str, ...]

    @property
    def stratum(self) -> str:
        return f"{self.region}|{self.size}"

    def public_record(self, order: int) -> dict[str, Any]:
        return {
            "sample_order": order,
            "notice_id": self.notice_id,
            "region": self.region,
            "size": self.size,
            "stratum": self.stratum,
            "indexed_vector_count": self.indexed_vector_count,
            "modalities": list(self.modalities),
            "indexed_url_sha256": url_sha256(self.indexed_url),
            "secondary_url_sha256": url_sha256(self.secondary_url),
        }


@dataclass(frozen=True)
class CandidatePlan:
    eligible: tuple[HeldoutCandidate, ...]
    sample: tuple[HeldoutCandidate, ...]
    corpus: Mapping[str, Any]
    exclusions: Mapping[str, int]
    distributions: Mapping[str, Any]
    sample_sha256: str


def _image_urls(meta: Mapping[str, Any]) -> list[str]:
    raw = meta.get("image_urls")
    values: Sequence[Any]
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        values = raw
    else:
        values = ()
    output: list[str] = []
    for value in (*values, meta.get("image_url"), meta.get("url")):
        text = clean_text(value)
        if not text:
            continue
        try:
            canonical = canonical_http_url(text)
        except UnsafeImageUrl:
            continue
        if canonical not in output:
            output.append(canonical)
    return output


def _indexed_source_url(meta: Mapping[str, Any]) -> str:
    for key in ("embedding_image_url", "image_url", "url"):
        value = clean_text(meta.get(key))
        if not value:
            continue
        try:
            return canonical_http_url(value)
        except UnsafeImageUrl:
            continue
    return ""


def _distribution(candidates: Sequence[HeldoutCandidate]) -> dict[str, Any]:
    region = Counter(candidate.region for candidate in candidates)
    size = Counter(candidate.size for candidate in candidates)
    strata = Counter(candidate.stratum for candidate in candidates)
    return {
        "region": dict(sorted(region.items())),
        "size": dict(sorted(size.items())),
        "region_size_strata": dict(sorted(strata.items())),
    }


def _selection_hash(candidate: HeldoutCandidate, seed: str) -> str:
    material = "\0".join(
        (seed, candidate.notice_id, candidate.indexed_url, candidate.secondary_url)
    )
    return sha256_bytes(material.encode("utf-8"))


def select_stratified_sample(
    candidates: Sequence[HeldoutCandidate],
    *,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    seed: str = DEFAULT_SEED,
) -> tuple[HeldoutCandidate, ...]:
    """Select at least one item per region/size stratum when capacity allows."""

    limit = max(0, int(max_samples))
    if not candidates or limit == 0:
        return ()
    unique_by_id: dict[str, HeldoutCandidate] = {}
    for candidate in candidates:
        if candidate.notice_id in unique_by_id:
            raise ValueError(f"duplicate eligible notice: {candidate.notice_id}")
        unique_by_id[candidate.notice_id] = candidate

    groups: dict[str, list[HeldoutCandidate]] = defaultdict(list)
    for candidate in unique_by_id.values():
        groups[candidate.stratum].append(candidate)
    for group in groups.values():
        group.sort(key=lambda item: (_selection_hash(item, seed), item.notice_id))

    group_order = sorted(
        groups,
        key=lambda value: (
            sha256_bytes(f"{seed}\0stratum\0{value}".encode("utf-8")),
            value,
        ),
    )
    selected: list[HeldoutCandidate] = []
    selected_ids: set[str] = set()
    for stratum in group_order[:limit]:
        candidate = groups[stratum][0]
        selected.append(candidate)
        selected_ids.add(candidate.notice_id)

    remaining = sorted(
        (
            candidate
            for candidate in unique_by_id.values()
            if candidate.notice_id not in selected_ids
        ),
        key=lambda item: (_selection_hash(item, seed), item.notice_id),
    )
    selected.extend(remaining[: max(0, limit - len(selected))])
    return tuple(selected)


def _sample_fingerprint(sample: Sequence[HeldoutCandidate], seed: str) -> str:
    lines = [
        "\t".join(
            (
                str(order),
                candidate.notice_id,
                url_sha256(candidate.indexed_url),
                url_sha256(candidate.secondary_url),
                candidate.stratum,
            )
        )
        for order, candidate in enumerate(sample, start=1)
    ]
    return sha256_bytes((seed + "\n" + "\n".join(lines)).encode("utf-8"))


def build_candidate_plan(
    metas: Sequence[Mapping[str, Any]],
    *,
    reference_date: datetime,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    seed: str = DEFAULT_SEED,
) -> CandidatePlan:
    """Discover active primary-index/secondary-query pairs deterministically."""

    notices = merge_notice_metas(metas)
    visual_rows: dict[str, list[tuple[str, str]]] = defaultdict(list)
    indexed_url_to_notices: dict[str, set[str]] = defaultdict(set)
    full_image_rows = 0
    crop_image_rows = 0
    text_rows = 0
    for row in metas:
        if not isinstance(row, Mapping):
            continue
        dog_id = notice_id(row)
        modality = resolve_vector_modality(dict(row))
        if modality == "text":
            text_rows += 1
            continue
        if modality not in VISUAL_MODALITIES or not dog_id:
            continue
        source_url = _indexed_source_url(row)
        visual_rows[dog_id].append((modality, source_url))
        if modality == "full_image":
            full_image_rows += 1
        else:
            crop_image_rows += 1
        if source_url:
            indexed_url_to_notices[source_url].add(dog_id)

    exclusions: Counter[str] = Counter()
    eligible: list[HeldoutCandidate] = []
    active_notice_count = 0
    for dog_id in sorted(notices):
        meta = notices[dog_id]
        if classify_notice(dict(meta), reference_date=reference_date) != "active":
            exclusions["not_active"] += 1
            continue
        active_notice_count += 1
        rows = visual_rows.get(dog_id, [])
        modalities = {modality for modality, _url in rows}
        if "full_image" not in modalities:
            exclusions["no_indexed_full_image"] += 1
            continue
        source_urls = {url for _modality, url in rows if url}
        if not source_urls:
            exclusions["missing_indexed_url"] += 1
            continue
        if len(source_urls) != 1:
            exclusions["ambiguous_indexed_url"] += 1
            continue
        indexed_url = next(iter(source_urls))
        notice_urls = _image_urls(meta)
        if not notice_urls or notice_urls[0] != indexed_url:
            exclusions["indexed_url_not_primary"] += 1
            continue
        distinct_urls = [url for url in notice_urls[1:] if url != indexed_url]
        if not distinct_urls:
            exclusions["no_distinct_secondary_url"] += 1
            continue
        secondary_url = next(
            (url for url in distinct_urls if not indexed_url_to_notices.get(url)),
            "",
        )
        if not secondary_url:
            exclusions["secondary_url_already_indexed"] += 1
            continue
        region = clean_text(infer_region(dict(meta))) or "unknown"
        size = clean_text(infer_size_from_weight(meta.get("weight"))) or "unknown"
        eligible.append(
            HeldoutCandidate(
                notice_id=dog_id,
                indexed_url=indexed_url,
                secondary_url=secondary_url,
                region=region,
                size=size,
                indexed_vector_count=len(rows),
                modalities=tuple(sorted(modalities)),
            )
        )

    eligible.sort(key=lambda item: item.notice_id)
    sample = select_stratified_sample(
        eligible,
        max_samples=max_samples,
        seed=seed,
    )
    corpus = {
        "metadata_rows": len(metas),
        "unique_notices": len(notices),
        "active_notices": active_notice_count,
        "full_image_vectors": full_image_rows,
        "crop_image_vectors": crop_image_rows,
        "text_vectors": text_rows,
        "visual_indexed_notices": len(visual_rows),
        "eligible_pairs": len(eligible),
        "selected_pairs": len(sample),
        "visual_index_coverage": round(len(visual_rows) / active_notice_count, 6)
        if active_notice_count
        else 0.0,
    }
    return CandidatePlan(
        eligible=tuple(eligible),
        sample=sample,
        corpus=corpus,
        exclusions=dict(sorted(exclusions.items())),
        distributions={
            "eligible": _distribution(eligible),
            "sample": _distribution(sample),
        },
        sample_sha256=_sample_fingerprint(sample, seed),
    )


@dataclass(frozen=True)
class DownloadedImage:
    image: Any
    payload_sha256: str
    byte_count: int
    image_format: str
    width: int
    height: int
    final_url: str


class DownloadTransportError(RuntimeError):
    """A client adapter failed without retaining upstream exception details."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class _HttpxResponseAdapter:
    """Expose the small streamed-response contract used by the safe downloader."""

    def __init__(self, response: Any, context: Any, httpx_module: Any) -> None:
        self.status_code = response.status_code
        self.headers = response.headers
        self._response = response
        self._context = context
        self._httpx = httpx_module
        self._closed = False

    def iter_content(self, *, chunk_size: int) -> Iterable[bytes]:
        try:
            yield from self._response.iter_bytes(chunk_size=chunk_size)
        except self._httpx.TimeoutException:
            raise DownloadTransportError("request_timeout") from None
        except (self._httpx.HTTPError, OSError):
            raise DownloadTransportError("stream_read_failed") from None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._context.__exit__(None, None, None)
        except (self._httpx.HTTPError, OSError):
            return


class HttpxSessionAdapter:
    """Lazy HTTPX adapter with explicit proxy selection and no env trust."""

    def __init__(self, *, httpx_module: Any | None = None) -> None:
        if httpx_module is None:
            import httpx

            httpx_module = httpx
        self._httpx = httpx_module
        self._clients: dict[str, Any] = {}
        self.headers: dict[str, str] = {}
        self.trust_env = False

    @staticmethod
    def _select_proxy(url: str, proxies: Mapping[str, str] | None) -> str:
        if not proxies:
            return ""
        scheme = urlsplit(url).scheme.lower()
        return clean_text(proxies.get(scheme) or proxies.get("all"))

    def _client(self, proxy_url: str) -> Any:
        client = self._clients.get(proxy_url)
        if client is not None:
            return client
        try:
            client = self._httpx.Client(
                trust_env=False,
                proxy=proxy_url or None,
                follow_redirects=False,
            )
        except (
            self._httpx.HTTPError,
            ImportError,
            OSError,
            TypeError,
            ValueError,
        ):
            reason = (
                "proxy_configuration_invalid"
                if proxy_url
                else "request_configuration_invalid"
            )
            raise DownloadTransportError(reason) from None
        self._clients[proxy_url] = client
        return client

    def get(
        self,
        url: str,
        *,
        stream: bool,
        allow_redirects: bool,
        timeout: tuple[float, float],
        proxies: Mapping[str, str] | None = None,
    ) -> _HttpxResponseAdapter:
        if not stream or allow_redirects:
            raise DownloadTransportError("request_configuration_invalid")
        proxy_url = self._select_proxy(url, proxies)
        client = self._client(proxy_url)
        try:
            connect_timeout, read_timeout = timeout
            timeout_config = self._httpx.Timeout(
                read_timeout,
                connect=connect_timeout,
            )
            context = client.stream(
                "GET",
                url,
                headers=dict(self.headers),
                follow_redirects=False,
                timeout=timeout_config,
            )
            response = context.__enter__()
        except self._httpx.TimeoutException:
            raise DownloadTransportError("request_timeout") from None
        except (self._httpx.HTTPError, OSError):
            raise DownloadTransportError("request_failed") from None
        except (TypeError, ValueError):
            raise DownloadTransportError("request_configuration_invalid") from None
        return _HttpxResponseAdapter(response, context, self._httpx)

    def close(self) -> None:
        for client in self._clients.values():
            try:
                client.close()
            except (self._httpx.HTTPError, OSError):
                continue
        self._clients.clear()


class SafePublicImageDownloader:
    """Memory-only downloader with SSRF, redirect, size, pixel, and time gates."""

    def __init__(
        self,
        *,
        timeout: float = 12.0,
        max_bytes: int = 12 * 1024 * 1024,
        max_pixels: int = 25_000_000,
        max_redirects: int = 3,
        retries: int = 1,
        allowed_hosts: Iterable[str] | None = None,
        resolver: Callable[..., Any] = socket.getaddrinfo,
        session: Any | None = None,
        clock: Callable[[], float] = time.monotonic,
        use_env_proxy: bool = False,
        proxy_resolver: Callable[[str], Mapping[str, str]] | None = None,
    ) -> None:
        import requests

        parsed_timeout = float(timeout)
        if not math.isfinite(parsed_timeout) or parsed_timeout <= 0:
            raise ValueError("timeout must be a positive finite number")
        self._requests = requests
        self._timeout = parsed_timeout
        self._max_bytes = max(1, int(max_bytes))
        self._max_pixels = max(1, int(max_pixels))
        self._max_redirects = max(0, int(max_redirects))
        self._retries = max(0, min(1, int(retries)))
        self._allowed_hosts = tuple(allowed_hosts or ())
        self._resolver = resolver
        self._clock = clock
        self._use_env_proxy = bool(use_env_proxy)
        self._proxy_resolver = proxy_resolver or requests.utils.get_environ_proxies
        self._session = session or requests.Session()
        if hasattr(self._session, "trust_env"):
            self._session.trust_env = False
        headers = getattr(self._session, "headers", None)
        if hasattr(headers, "update"):
            headers.update(
                {
                    "Accept": "image/jpeg,image/png,image/webp,image/*;q=0.8",
                    "User-Agent": "MeongTamjeong-Heldout-Eval/1.0",
                }
            )

    def close(self) -> None:
        """Close the request session without exposing its configuration."""

        close = getattr(self._session, "close", None)
        if callable(close):
            close()

    def _environment_proxies(self, url: str) -> dict[str, str]:
        if not self._use_env_proxy:
            return {}
        try:
            raw_proxies = self._proxy_resolver(url)
        except (OSError, TypeError, ValueError) as exc:
            raise ImageDownloadError("proxy_configuration_invalid") from exc
        if not isinstance(raw_proxies, Mapping):
            raise ImageDownloadError("proxy_configuration_invalid")
        proxies: dict[str, str] = {}
        for raw_key, raw_value in raw_proxies.items():
            key = clean_text(raw_key).lower()
            value = clean_text(raw_value)
            if key in {"http", "https", "all"} and value:
                proxies[key] = value
        return proxies

    def _decode(self, payload: bytes, final_url: str) -> DownloadedImage:
        from PIL import Image

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(payload)) as source:
                    image_format = clean_text(source.format).upper()
                    if image_format not in ALLOWED_IMAGE_FORMATS:
                        raise ImageDownloadError("unsupported_image_format")
                    width, height = source.size
                    if width <= 0 or height <= 0:
                        raise ImageDownloadError("invalid_image_dimensions")
                    if width * height > self._max_pixels:
                        raise ImageDownloadError("pixel_limit_exceeded")
                    source.load()
                    image = source.convert("RGB").copy()
        except ImageDownloadError:
            raise
        except (
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
            MemoryError,
            OSError,
            SyntaxError,
            ValueError,
            EOFError,
        ) as exc:
            raise ImageDownloadError("image_decode_failed") from exc
        return DownloadedImage(
            image=image,
            payload_sha256=sha256_bytes(payload),
            byte_count=len(payload),
            image_format=image_format,
            width=width,
            height=height,
            final_url=final_url,
        )

    def _read_payload(self, response: Any, deadline: float) -> bytes:
        headers = getattr(response, "headers", {}) or {}
        content_type = clean_text(headers.get("Content-Type")).split(";", 1)[0].lower()
        if content_type and not (
            content_type.startswith("image/")
            or content_type in {"application/octet-stream", "binary/octet-stream"}
        ):
            raise ImageDownloadError("disallowed_content_type")
        raw_length = clean_text(headers.get("Content-Length"))
        if raw_length:
            try:
                declared = int(raw_length)
            except ValueError as exc:
                raise ImageDownloadError("invalid_content_length") from exc
            if declared < 0 or declared > self._max_bytes:
                raise ImageDownloadError("declared_size_exceeded")
        output = io.BytesIO()
        total = 0
        try:
            chunks = response.iter_content(chunk_size=DOWNLOAD_CHUNK_BYTES)
            for chunk in chunks:
                if self._clock() > deadline:
                    raise ImageDownloadError("total_timeout")
                if not chunk:
                    continue
                total += len(chunk)
                if total > self._max_bytes:
                    raise ImageDownloadError("stream_size_exceeded")
                output.write(chunk)
        except ImageDownloadError:
            raise
        except (OSError, ValueError) as exc:
            raise ImageDownloadError("stream_read_failed") from exc
        payload = output.getvalue()
        if not payload:
            raise ImageDownloadError("empty_payload")
        return payload

    def fetch(self, url: str) -> DownloadedImage:
        deadline = self._clock() + self._timeout
        current_url = upgrade_http_to_https(url)
        redirects = 0
        attempts_left = self._retries + 1
        while attempts_left > 0:
            attempts_left -= 1
            try:
                current_url = validate_public_http_url(
                    current_url,
                    resolver=self._resolver,
                    allowed_hosts=self._allowed_hosts,
                )
            except UnsafeImageUrl as exc:
                raise ImageDownloadError(exc.reason) from exc
            response = None
            try:
                remaining = deadline - self._clock()
                if remaining <= 0:
                    raise ImageDownloadError("total_timeout")
                response = self._session.get(
                    quote(current_url, safe=":/?&=#[]%"),
                    stream=True,
                    allow_redirects=False,
                    timeout=(min(5.0, remaining), max(0.1, remaining)),
                    proxies=self._environment_proxies(current_url),
                )
                status = int(getattr(response, "status_code", 0) or 0)
                if status in REDIRECT_STATUS_CODES:
                    location = clean_text(
                        (getattr(response, "headers", {}) or {}).get("Location")
                    )
                    if not location:
                        raise ImageDownloadError("redirect_without_location")
                    if redirects >= self._max_redirects:
                        raise ImageDownloadError("redirect_limit_exceeded")
                    redirects += 1
                    redirect_url = canonical_http_url(urljoin(current_url, location))
                    if (
                        urlsplit(current_url).scheme == "https"
                        and urlsplit(redirect_url).scheme != "https"
                    ):
                        raise ImageDownloadError("https_downgrade_redirect")
                    current_url = redirect_url
                    attempts_left += 1
                    continue
                if status != 200:
                    if status in {408, 425, 429, 500, 502, 503, 504} and attempts_left:
                        continue
                    raise ImageDownloadError(f"http_status_{status or 'unknown'}")
                payload = self._read_payload(response, deadline)
                return self._decode(payload, current_url)
            except UnsafeImageUrl as exc:
                raise ImageDownloadError(exc.reason) from exc
            except ImageDownloadError:
                raise
            except DownloadTransportError as exc:
                if attempts_left:
                    continue
                raise ImageDownloadError(exc.reason) from None
            except self._requests.RequestException as exc:
                if attempts_left:
                    continue
                raise ImageDownloadError("request_failed") from exc
            finally:
                close = getattr(response, "close", None)
                if callable(close):
                    close()
        raise ImageDownloadError("request_failed")


def candidate_vector_indices(
    metas: Sequence[Mapping[str, Any]],
    *,
    reference_date: datetime,
) -> tuple[list[int], dict[str, int]]:
    """Return active visual-vector rows; text rows are always excluded."""

    indices: list[int] = []
    counts: Counter[str] = Counter()
    for index, meta in enumerate(metas):
        modality = resolve_vector_modality(dict(meta))
        if modality not in VISUAL_MODALITIES:
            counts["excluded_text_or_unknown"] += 1
            continue
        if classify_notice(dict(meta), reference_date=reference_date) != "active":
            counts["excluded_inactive"] += 1
            continue
        if not notice_id(meta):
            counts["excluded_missing_notice_id"] += 1
            continue
        indices.append(index)
        counts[modality] += 1
    counts["candidate_vectors"] = len(indices)
    return indices, dict(sorted(counts.items()))


def indexed_visual_source_urls(
    metas: Sequence[Mapping[str, Any]],
    *,
    reference_date: datetime,
) -> tuple[tuple[str, ...], dict[str, int]]:
    """Return canonical source URLs needed for an index-wide payload audit.

    Full-image and crop rows can point back to the same public primary photo,
    so URLs are deduplicated while missing provenance remains visible.
    """

    urls: set[str] = set()
    counts: Counter[str] = Counter()
    for meta in metas:
        modality = resolve_vector_modality(dict(meta))
        if modality not in VISUAL_MODALITIES:
            continue
        if classify_notice(dict(meta), reference_date=reference_date) != "active":
            continue
        if not notice_id(meta):
            continue
        counts["active_visual_rows"] += 1
        source_url = _indexed_source_url(meta)
        if not source_url:
            counts["missing_source_url_rows"] += 1
            continue
        urls.add(source_url)
    counts["unique_source_urls"] = len(urls)
    return tuple(sorted(urls)), dict(sorted(counts.items()))


def rank_notice_scores(
    vector_scores: Sequence[float],
    vector_indices: Sequence[int],
    metas: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse full-image/crop scores to one deterministic max per notice."""

    if len(vector_scores) != len(vector_indices):
        raise ValueError("vector score/index length mismatch")
    best: dict[str, dict[str, Any]] = {}
    for raw_score, vector_index in zip(vector_scores, vector_indices):
        if not 0 <= int(vector_index) < len(metas):
            raise ValueError(f"vector index out of range: {vector_index}")
        score = float(raw_score)
        if not math.isfinite(score):
            raise ValueError("non-finite vector score")
        meta = metas[int(vector_index)]
        dog_id = notice_id(meta)
        modality = resolve_vector_modality(dict(meta))
        if not dog_id or modality not in VISUAL_MODALITIES:
            continue
        current = best.get(dog_id)
        if current is None or score > float(current["score"]):
            best[dog_id] = {
                "notice_id": dog_id,
                "score": score,
                "best_modality": modality,
                "vector_index": int(vector_index),
            }
    return sorted(
        best.values(),
        key=lambda row: (-float(row["score"]), clean_text(row["notice_id"])),
    )


def rank_record(
    target_notice_id: str,
    ranking: Sequence[Mapping[str, Any]],
    *,
    top_k: int = 10,
) -> dict[str, Any]:
    target = clean_text(target_notice_id)
    rank = next(
        (
            position
            for position, row in enumerate(ranking, start=1)
            if clean_text(row.get("notice_id")) == target
        ),
        None,
    )
    target_row = next(
        (row for row in ranking if clean_text(row.get("notice_id")) == target),
        {},
    )
    return {
        "notice_id": target,
        "rank": rank,
        "reciprocal_rank": round(1.0 / rank, 8) if rank else 0.0,
        "hit@1": int(rank is not None and rank <= 1),
        "hit@5": int(rank is not None and rank <= 5),
        "hit@10": int(rank is not None and rank <= 10),
        "target_score": round(float(target_row.get("score", 0.0)), 8)
        if target_row
        else None,
        "target_best_modality": clean_text(target_row.get("best_modality")),
        "top_ids": [
            clean_text(row.get("notice_id")) for row in ranking[: max(1, int(top_k))]
        ],
    }


def aggregate_rank_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    attempted_count: int,
    downloaded_count: int,
    duplicate_count: int,
) -> dict[str, Any]:
    ranks = [
        int(row["rank"])
        for row in rows
        if isinstance(row.get("rank"), int) and int(row["rank"]) > 0
    ]
    evaluated = len(rows)
    attempted = max(0, int(attempted_count))
    downloaded = max(0, int(downloaded_count))
    return {
        "attempted_count": attempted,
        "downloaded_count": downloaded,
        "evaluable_count": evaluated,
        "duplicate_count": max(0, int(duplicate_count)),
        "download_coverage": round(downloaded / attempted, 6) if attempted else 0.0,
        "evaluable_coverage": round(evaluated / attempted, 6) if attempted else 0.0,
        "hit@1": round(mean(float(row.get("hit@1", 0)) for row in rows), 6)
        if rows
        else 0.0,
        "hit@5": round(mean(float(row.get("hit@5", 0)) for row in rows), 6)
        if rows
        else 0.0,
        "hit@10": round(mean(float(row.get("hit@10", 0)) for row in rows), 6)
        if rows
        else 0.0,
        "MRR": round(mean(float(row.get("reciprocal_rank", 0.0)) for row in rows), 6)
        if rows
        else 0.0,
        "median_rank": round(float(median(ranks)), 3) if ranks else None,
        "max_rank": max(ranks) if ranks else None,
    }


def build_plan_report(
    plan: CandidatePlan,
    *,
    reference_date: str,
    seed: str,
    max_samples: int,
    artifacts: Mapping[str, Any],
    clip_model: str,
) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "evaluation_id": EVALUATION_ID,
        "reference_date": reference_date,
        "scope": {
            "query": "one distinct secondary public-notice photo per sampled notice",
            "candidate_vectors": ["full_image", "crop_image"],
            "excluded_vectors": ["text"],
            "deduplication": "maximum CLIP cosine score per notice",
            "ground_truth": "same public notice ID",
            "default_payload_audit": (
                "each evaluable query requires its own primary SHA and is compared "
                "with the successfully downloaded sampled-primary SHA set"
            ),
            "transport_policy": (
                "upgrade initial HTTP image URLs to HTTPS and reject HTTPS-to-HTTP "
                "redirects"
            ),
        },
        "artifacts": dict(artifacts),
        "runtime": {
            "clip_model": clip_model,
            "device": None,
            "network_executed": False,
            "declared_index_provenance_consistent": None,
            "query_encoder_commit_verified": None,
            "index_encoder_verified": False,
        },
        "sample_policy": {
            "seed": seed,
            "hash": "SHA-256(seed, notice_id, indexed_url, secondary_url)",
            "max_samples": max(0, int(max_samples)),
            "stratification": (
                "one per region|size stratum when capacity allows, then global "
                "SHA-256 order"
            ),
            "sample_sha256": plan.sample_sha256,
        },
        "corpus": dict(plan.corpus),
        "exclusion_reasons": dict(plan.exclusions),
        "distributions": dict(plan.distributions),
        "sample": [
            candidate.public_record(order)
            for order, candidate in enumerate(plan.sample, start=1)
        ],
        "execution": {
            "status": "plan_only",
            "metrics_available": False,
            "network_requested": False,
            "note": (
                "No image was downloaded or encoded. Run the explicit "
                "--run-network mode to produce measured metrics."
            ),
        },
        "limitations": [
            (
                "This cross-photo same-notice task measures a visual identity "
                "retrieval proxy, not subjective similarity between different dogs."
            ),
            (
                "It does not measure temperament, health, adoption suitability, "
                "child friendliness, or compatibility with other animals."
            ),
            (
                "URL inequality is checked before sampling; exact payload SHA-256 "
                "leakage and duplicate-query checks occur only during network execution."
            ),
            (
                "In the default sampled audit, missing primary downloads reduce payload "
                "audit coverage and are disclosed. An evaluable query is checked against "
                "its own primary SHA and all successfully downloaded sampled-primary "
                "SHAs, but duplication against an unavailable primary cannot be ruled out."
            ),
            (
                "A corpus-wide audit requires --audit-all-indexed-sources, may be slow, "
                "and is complete only when every declared active visual source is "
                "successfully downloaded."
            ),
            (
                "Image transport upgrades initial HTTP metadata URLs to HTTPS and rejects "
                "redirects that downgrade HTTPS back to HTTP."
            ),
            (
                "Generated crop bytes are not retained with the index, so exact raw-byte "
                "comparison covers their public source photo rather than the crop payload."
            ),
            (
                "Public image URLs can change or disappear, so download and evaluable "
                "coverage must be reported with retrieval metrics."
            ),
            (
                "The raw CLIP visual-index result is not the full production API, "
                "which can apply additional filtering and reranking."
            ),
            (
                "The refreshed index retained image and crop vectors from earlier "
                "snapshots. It has no per-vector encoder sidecar fingerprint, so "
                "artifact-bound declarations do not prove the encoder of every vector."
            ),
        ],
    }


def report_failures(
    report: Mapping[str, Any],
    *,
    expected_sample_sha256: str | None = None,
) -> list[str]:
    failures: list[str] = []
    if clean_text(report.get("schema_version")) != REPORT_SCHEMA_VERSION:
        failures.append("unexpected report schema_version")
    if clean_text(report.get("evaluation_id")) != EVALUATION_ID:
        failures.append("unexpected evaluation_id")
    sample_policy = report.get("sample_policy")
    sample = report.get("sample")
    corpus = report.get("corpus")
    execution = report.get("execution")
    runtime = report.get("runtime")
    if not isinstance(sample_policy, Mapping):
        failures.append("sample_policy is missing")
    if not isinstance(sample, Sequence) or isinstance(sample, (str, bytes, bytearray)):
        failures.append("sample must be an array")
        sample = ()
    if not isinstance(corpus, Mapping):
        failures.append("corpus is missing")
        corpus = {}
    if not isinstance(execution, Mapping):
        failures.append("execution is missing")
        execution = {}
    if not isinstance(runtime, Mapping):
        failures.append("runtime is missing")
        runtime = {}
    ids = [
        clean_text(row.get("notice_id")) for row in sample if isinstance(row, Mapping)
    ]
    if any(not value for value in ids):
        failures.append("sample contains a missing notice ID")
    if len(ids) != len(set(ids)):
        failures.append("sample contains duplicate notice IDs")
    secondary_hashes = [
        clean_text(row.get("secondary_url_sha256"))
        for row in sample
        if isinstance(row, Mapping)
    ]
    if any(not value for value in secondary_hashes):
        failures.append("sample contains a missing secondary URL hash")
    if len(secondary_hashes) != len(set(secondary_hashes)):
        failures.append("sample contains duplicate secondary URL hashes")
    selected_pairs = int(corpus.get("selected_pairs") or 0)
    if selected_pairs != len(sample):
        failures.append("selected_pairs does not match sample length")
    if expected_sample_sha256 is not None and isinstance(sample_policy, Mapping):
        if clean_text(sample_policy.get("sample_sha256")) != expected_sample_sha256:
            failures.append("sample SHA-256 is stale")
    status = clean_text(execution.get("status"))
    if status not in {"plan_only", "completed", "incomplete"}:
        failures.append("execution status must be plan_only, completed, or incomplete")
    coverage_status = clean_text(execution.get("coverage_status"))
    if status in {"completed", "incomplete"} and coverage_status not in {
        "complete",
        "partial",
    }:
        failures.append("network execution coverage_status must be complete or partial")
    if status == "plan_only" and bool(execution.get("metrics_available")):
        failures.append("plan-only report cannot claim metrics")
    if status == "completed":
        metrics = report.get("metrics")
        if not isinstance(metrics, Mapping):
            failures.append("completed report metrics are missing")
        elif int(metrics.get("evaluable_count") or 0) <= 0:
            failures.append("completed report has no evaluable queries")
        if not bool(execution.get("payload_leakage_audit_complete")):
            failures.append("completed report payload leakage audit is incomplete")
        if (
            runtime.get("audit_all_indexed_sources") is True
            and coverage_status != "complete"
        ):
            failures.append("completed corpus-wide payload audit has partial coverage")
        if runtime.get("declared_index_provenance_consistent") is not True:
            failures.append("completed report lacks declared index provenance")
        if runtime.get("query_encoder_commit_verified") is not True:
            failures.append("completed report lacks query encoder commit verification")
        if runtime.get("index_encoder_verified") is not False:
            failures.append("completed report overclaims index encoder verification")
    if status == "incomplete":
        failures.append("network execution is incomplete")
    return failures


def _percent(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def report_to_markdown(report: Mapping[str, Any]) -> str:
    """Render a deterministic, concise companion for plan or measured output."""

    corpus = report.get("corpus") or {}
    policy = report.get("sample_policy") or {}
    execution = report.get("execution") or {}
    runtime = report.get("runtime") or {}
    artifacts = report.get("artifacts") or {}
    distributions = report.get("distributions") or {}
    lines = [
        "# Held-out 2번째 사진 검색 평가",
        "",
        "> 같은 공고의 인덱스에 사용하지 않은 2번째 사진을 질의로 삼아, "
        "1번째 사진·crop 시각 벡터에서 같은 공고를 찾는 평가입니다. "
        "입양 적합성·성격·건강을 평가하지 않습니다.",
        "",
        f"- 기준일: `{clean_text(report.get('reference_date'))}`",
        f"- 실행 상태: `{clean_text(execution.get('status'))}`",
        f"- 활성 공고: {corpus.get('active_notices', 0)}건",
        f"- 시각 인덱스 공고: {corpus.get('visual_indexed_notices', 0)}건",
        f"- 평가 가능 URL 쌍: {corpus.get('eligible_pairs', 0)}건",
        f"- 고정 표본: {corpus.get('selected_pairs', 0)}건",
        f"- 시각 인덱스 커버리지: {_percent(corpus.get('visual_index_coverage'))}",
        f"- 표본 SHA-256: `{clean_text(policy.get('sample_sha256'))}`",
        "",
        "## 입력 고정 정보",
        "",
        "| 입력 | 경로 | SHA-256 |",
        "|---|---|---|",
    ]
    for name in (
        "index",
        "metas",
        "snapshot_manifest",
        "active_index_sync_report",
        "requirements_lock",
    ):
        record = artifacts.get(name) or {}
        if not record:
            continue
        lines.append(
            f"| {name} | `{clean_text(record.get('path'))}` | "
            f"`{clean_text(record.get('sha256'))}` |"
        )
    lines.extend(
        [
            "",
            "## Encoder provenance",
            "",
            "- declared index provenance consistent: "
            f"`{runtime.get('declared_index_provenance_consistent')}`",
            "- query encoder commit verified: "
            f"`{runtime.get('query_encoder_commit_verified')}`",
            f"- full index encoder verified: `{runtime.get('index_encoder_verified')}`",
            "",
            "## 표본 정책",
            "",
            f"- seed: `{clean_text(policy.get('seed'))}`",
            f"- 최대 표본: {policy.get('max_samples', 0)}",
            f"- 선택: {clean_text(policy.get('stratification'))}",
            "- 동일 URL이거나 다른 공고의 인덱스 URL인 2번째 사진은 계획 단계에서 제외",
            "- 실제 실행에서는 primary/query payload SHA-256 동일 및 query SHA 중복도 제외",
            "",
            "### 표본 지역 분포",
            "",
            "| 지역 | 건수 |",
            "|---|---:|",
        ]
    )
    sample_distribution = distributions.get("sample") or {}
    for name, count in (sample_distribution.get("region") or {}).items():
        lines.append(f"| {name} | {count} |")
    lines.extend(["", "### 표본 크기 분포", "", "| 크기 | 건수 |", "|---|---:|"])
    for name, count in (sample_distribution.get("size") or {}).items():
        lines.append(f"| {name} | {count} |")

    if isinstance(report.get("metrics"), Mapping):
        metrics = report.get("metrics") or {}
        lines.extend(
            [
                "",
                "## 측정 결과",
                "",
                "| 지표 | 값 |",
                "|---|---:|",
                f"| audit coverage status | {clean_text(execution.get('coverage_status'))} |",
                f"| attempted | {metrics.get('attempted_count', 0)} |",
                f"| downloaded | {metrics.get('downloaded_count', 0)} |",
                f"| evaluable | {metrics.get('evaluable_count', 0)} |",
                f"| duplicate excluded | {metrics.get('duplicate_count', 0)} |",
                f"| download coverage | {_percent(metrics.get('download_coverage'))} |",
                f"| evaluable coverage | {_percent(metrics.get('evaluable_coverage'))} |",
                f"| primary download coverage | {_percent(metrics.get('primary_download_coverage'))} |",
                f"| payload audit coverage | {_percent(metrics.get('payload_audit_coverage'))} |",
                f"| Hit@1 | {_percent(metrics.get('hit@1'))} |",
                f"| Hit@5 | {_percent(metrics.get('hit@5'))} |",
                f"| Hit@10 | {_percent(metrics.get('hit@10'))} |",
                f"| end-to-end Hit@1 | {_percent(metrics.get('end_to_end_hit@1'))} |",
                f"| end-to-end Hit@5 | {_percent(metrics.get('end_to_end_hit@5'))} |",
                f"| end-to-end Hit@10 | {_percent(metrics.get('end_to_end_hit@10'))} |",
                f"| MRR | {metrics.get('MRR', 0):.4f} |",
                f"| median rank | {metrics.get('median_rank')} |",
            ]
        )
        failure_reasons = report.get("failure_reasons") or {}
        if isinstance(failure_reasons, Mapping) and failure_reasons:
            lines.extend(
                [
                    "",
                    "### Download and exclusion failures",
                    "",
                    "| reason | count |",
                    "|---|---:|",
                ]
            )
            for reason, count in sorted(failure_reasons.items()):
                lines.append(f"| {clean_text(reason)} | {int(count)} |")
    else:
        lines.extend(
            [
                "",
                "## 측정 상태",
                "",
                "이 파일은 네트워크를 사용하지 않고 만든 실행 계획입니다. "
                "아직 이미지 다운로드, CLIP 인코딩, 검색 지표 계산을 수행하지 않았습니다.",
            ]
        )
    lines.extend(["", "## 해석 제한", ""])
    for limitation in report.get("limitations") or []:
        limitation_text = clean_text(limitation)
        lines.append(
            f"- {MARKDOWN_LIMITATION_KO_BY_EN.get(limitation_text, limitation_text)}"
        )
    lines.extend(
        [
            "",
            "이미지 바이너리는 저장하거나 배포하지 않으며 실제 실행 중 메모리에서만 "
            "검사합니다. URL이 달라도 재인코딩된 동일 사진은 raw SHA-256만으로 "
            "완전히 탐지할 수 없습니다.",
            "",
        ]
    )
    return "\n".join(lines)
