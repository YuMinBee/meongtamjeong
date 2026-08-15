"""Bounded downloader for public notice images.

The caller must provide an exact allowlist of public hostnames. Redirects are
never followed, and every response must pass byte, content-type, image-format,
pixel-count, and decompression-bomb checks before an RGB image is returned.
"""

from __future__ import annotations

import argparse
import importlib
import io
import ipaddress
import math
import time
import warnings
from collections.abc import Sequence
from typing import Any
from urllib.parse import quote, urlsplit


DEFAULT_IMAGE_MAX_BYTES = 16 * 1024 * 1024
DEFAULT_IMAGE_MAX_PIXELS = 25_000_000
DEFAULT_ALLOWED_IMAGE_HOSTS = ("openapi.animal.go.kr",)
NONPUBLIC_IMAGE_HOST_SUFFIXES = (
    ".internal",
    ".local",
    ".localhost",
    ".lan",
    ".home",
    ".test",
    ".invalid",
)
IMAGE_DOWNLOAD_CHUNK_BYTES = 64 * 1024
ALLOWED_PUBLIC_IMAGE_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})


def parse_public_image_hostname(value: str) -> str:
    """Validate an explicit, exact public image hostname for a CLI option."""

    candidate = str(value or "").strip().casefold().rstrip(".")
    try:
        candidate.encode("ascii")
    except UnicodeEncodeError as exc:
        raise argparse.ArgumentTypeError(
            "image host must use an ASCII hostname (punycode is allowed)"
        ) from exc
    if (
        not candidate
        or len(candidate) > 253
        or "." not in candidate
        or candidate == "localhost"
        or candidate.endswith(NONPUBLIC_IMAGE_HOST_SUFFIXES)
        or any(character.isspace() for character in candidate)
        or any(character in candidate for character in "/\\@?#[]:")
        or any(
            not label
            or len(label) > 63
            or label.startswith("-")
            or label.endswith("-")
            or not all(character.isalnum() or character == "-" for character in label)
            for label in candidate.split(".")
        )
    ):
        raise argparse.ArgumentTypeError(
            "image host must be an exact public DNS hostname without a scheme or path"
        )
    try:
        address = ipaddress.ip_address(candidate)
    except ValueError:
        address = None
    if address is not None and not address.is_global:
        raise argparse.ArgumentTypeError("image host must not be a private IP address")
    return candidate


class PublicImageDownloader:
    """Download and decode allowlisted public images under fixed resource limits."""

    def __init__(
        self,
        timeout: float = 15.0,
        *,
        max_bytes: int = DEFAULT_IMAGE_MAX_BYTES,
        max_pixels: int = DEFAULT_IMAGE_MAX_PIXELS,
        allowed_hosts: Sequence[str] = DEFAULT_ALLOWED_IMAGE_HOSTS,
    ) -> None:
        requests = importlib.import_module("requests")
        self._requests = requests
        parsed_timeout = float(timeout)
        self._timeout = (
            parsed_timeout
            if math.isfinite(parsed_timeout) and parsed_timeout > 0
            else 15.0
        )
        self._max_bytes = max(1, int(max_bytes))
        self._max_pixels = max(1, int(max_pixels))
        self._allowed_hosts = frozenset(
            str(host).strip().casefold().rstrip(".") for host in allowed_hosts if host
        )
        if not self._allowed_hosts:
            raise ValueError("at least one allowed image host is required")
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Accept": "image/*",
                "Referer": "https://www.animal.go.kr/",
                "User-Agent": "Mozilla/5.0",
            }
        )

    def close(self) -> None:
        close = getattr(self._session, "close", None)
        if callable(close):
            close()

    def _is_http_url(self, url: Any) -> bool:
        if (
            not isinstance(url, str)
            or not url
            or any(ord(character) < 32 or character.isspace() for character in url)
            or "\\" in url
        ):
            return False
        try:
            parsed = urlsplit(url)
        except ValueError:
            return False
        return (
            parsed.scheme.lower() in {"http", "https"}
            and bool(parsed.hostname)
            and parsed.hostname.casefold().rstrip(".") in self._allowed_hosts
            and parsed.username is None
            and parsed.password is None
        )

    @staticmethod
    def _candidate_urls(url: str) -> list[str]:
        parsed = urlsplit(url)
        if parsed.scheme.lower() != "http":
            return [url]
        # The public API still returns legacy HTTP image URLs for assets that
        # are also served over HTTPS. Prefer HTTPS without following redirects.
        upgraded = parsed._replace(scheme="https").geturl()
        return [upgraded, url] if upgraded != url else [url]

    def _decode_image(self, payload: bytes) -> Any | None:
        if not payload:
            return None
        image_module = importlib.import_module("PIL.Image")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", image_module.DecompressionBombWarning)
                with image_module.open(io.BytesIO(payload)) as source:
                    image_format = str(source.format or "").upper()
                    if image_format not in ALLOWED_PUBLIC_IMAGE_FORMATS:
                        return None
                    width, height = source.size
                    if width <= 0 or height <= 0 or width * height > self._max_pixels:
                        return None
                    source.load()
                    return source.convert("RGB").copy()
        except (
            image_module.DecompressionBombError,
            image_module.DecompressionBombWarning,
            MemoryError,
            OSError,
            SyntaxError,
            ValueError,
            EOFError,
        ):
            return None

    def _read_response(self, response: Any, deadline: float) -> bytes | None:
        if getattr(response, "status_code", None) != 200:
            return None
        if not self._is_http_url(getattr(response, "url", "")):
            return None

        headers = getattr(response, "headers", {}) or {}
        content_type = str(headers.get("Content-Type", "")).split(";", 1)[0]
        normalized_content_type = content_type.strip().lower()
        # The official host sometimes sends PNGs as generic binary data. The
        # bounded payload must still pass Pillow format and pixel checks.
        if normalized_content_type and not (
            normalized_content_type.startswith("image/")
            or normalized_content_type
            in {"application/octet-stream", "binary/octet-stream"}
        ):
            return None

        content_length = str(headers.get("Content-Length", "")).strip()
        if content_length:
            try:
                declared_bytes = int(content_length)
            except ValueError:
                return None
            if declared_bytes < 0 or declared_bytes > self._max_bytes:
                return None

        payload = io.BytesIO()
        total = 0
        for chunk in response.iter_content(chunk_size=IMAGE_DOWNLOAD_CHUNK_BYTES):
            if time.monotonic() > deadline:
                return None
            if not chunk:
                continue
            total += len(chunk)
            if total > self._max_bytes:
                return None
            payload.write(chunk)
        return payload.getvalue() or None

    def __call__(self, url: str) -> Any | None:
        if not self._is_http_url(url):
            return None
        candidates = self._candidate_urls(url)
        deadline = time.monotonic() + self._timeout
        for candidate_index, candidate in enumerate(candidates):
            if not self._is_http_url(candidate):
                continue
            now = time.monotonic()
            remaining = deadline - now
            if remaining <= 0:
                break
            candidates_after_this = len(candidates) - candidate_index - 1
            if candidates_after_this:
                fallback_reserve = min(
                    remaining / 2,
                    max(0.5, remaining * 0.2),
                )
                attempt_budget = max(0.1, remaining - fallback_reserve)
            else:
                attempt_budget = remaining
            attempt_deadline = min(deadline, now + attempt_budget)
            response = None
            try:
                response = self._session.get(
                    quote(candidate, safe=":/?&=#[]%"),
                    stream=True,
                    allow_redirects=False,
                    timeout=(
                        min(5.0, attempt_budget),
                        max(0.1, attempt_budget),
                    ),
                )
                payload = self._read_response(response, attempt_deadline)
                if payload is not None:
                    image = self._decode_image(payload)
                    if image is not None:
                        return image
            except (self._requests.RequestException, OSError, ValueError):
                continue
            finally:
                close = getattr(response, "close", None)
                if callable(close):
                    close()
        return None
