"""Download dog photographs explicitly released under CC0 on Commons."""

from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import os
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
from PIL import Image, ImageOps, UnidentifiedImageError
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.enrich_image_crops import (  # noqa: E402
    compute_photo_quality,
    load_torchvision_detector,
    normalize_bbox,
)


API_URL = "https://commons.wikimedia.org/w/api.php"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent / "artifacts" / "cc0_dog_photos_diverse"
)
DEFAULT_QUERY = "dog incategory:CC-Zero filetype:bitmap"
DEFAULT_USER_AGENT = (
    "MeongTamJeongDogDataset/0.1 (https://github.com/YuMinBee/meongtamjeong) requests"
)
SCHEMA_VERSION = "wikimedia-commons-cc0-dogs.v1"
ALLOWED_LICENSES = frozenset({"cc0"})
ALLOWED_MIME_TYPES = frozenset({"image/jpeg"})
NEGATIVE_TITLE_TERMS = (
    "coat variation",
    "morphological variation",
    "anatomy",
    "diagram",
    "drawing",
    "illustration",
    "painting",
    "sculpture",
    "statue",
    "logo",
    "map",
    "poster",
)
NEGATIVE_CATEGORY_TERMS = (
    "artworks with",
    "digital representation of 2d work",
    "drawings of",
    "engravings of",
    "illustrations of",
    "oil on canvas",
    "paintings of",
    "paintings by",
    "portrait paintings",
    "sculptures of",
    "statues of",
)


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def clean_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def strip_html(value: Any) -> str:
    parser = _TextExtractor()
    parser.feed(html.unescape(clean_text(value)))
    return re.sub(r"\s+", " ", " ".join(parser.parts)).strip()


def metadata_value(metadata: Mapping[str, Any], key: str) -> str:
    field = metadata.get(key)
    if not isinstance(field, Mapping):
        return ""
    return clean_text(field.get("value"))


def candidate_artist(page: Mapping[str, Any]) -> str:
    imageinfo = (page.get("imageinfo") or [{}])[0]
    metadata = imageinfo.get("extmetadata") if isinstance(imageinfo, Mapping) else {}
    return strip_html(metadata_value(metadata or {}, "Artist"))


def title_series_key(title: str) -> str:
    value = re.sub(r"^file:", "", clean_text(title), flags=re.IGNORECASE)
    value = re.sub(r"\.(?:jpe?g|png|webp)$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\b\d+\b", " ", value)
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip().casefold()


def is_explicit_cc0(metadata: Mapping[str, Any]) -> bool:
    license_name = metadata_value(metadata, "LicenseShortName").casefold()
    license_url = metadata_value(metadata, "LicenseUrl").casefold()
    usage_terms = metadata_value(metadata, "UsageTerms").casefold()
    return (
        license_name in ALLOWED_LICENSES
        and "creativecommons.org/publicdomain/zero/1.0" in license_url
        and "creative commons zero" in usage_terms
    )


def candidate_rejection_reason(page: Mapping[str, Any]) -> str:
    imageinfo = (page.get("imageinfo") or [{}])[0]
    if not isinstance(imageinfo, Mapping):
        return "missing_imageinfo"
    if clean_text(imageinfo.get("mime")).casefold() not in ALLOWED_MIME_TYPES:
        return "not_jpeg"
    metadata = imageinfo.get("extmetadata") or {}
    if not isinstance(metadata, Mapping) or not is_explicit_cc0(metadata):
        return "not_explicit_cc0"
    title = clean_text(page.get("title")).casefold()
    if any(term in title for term in NEGATIVE_TITLE_TERMS):
        return "non_photo_title"
    categories = metadata_value(metadata, "Categories").casefold()
    if any(term in categories for term in NEGATIVE_CATEGORY_TERMS):
        return "non_photo_category"
    width = int(imageinfo.get("width") or 0)
    height = int(imageinfo.get("height") or 0)
    if min(width, height) < 320:
        return "too_small"
    if not clean_text(imageinfo.get("thumburl") or imageinfo.get("url")):
        return "missing_download_url"
    return ""


def source_page_url(title: str) -> str:
    normalized = clean_text(title).replace(" ", "_")
    return f"https://commons.wikimedia.org/wiki/{quote(normalized, safe=':_()')}"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_gallery(output_dir: Path, items: Sequence[Mapping[str, Any]]) -> Path:
    cards: list[str] = []
    for item in items:
        local_path = (ROOT / clean_text(item.get("local_path"))).resolve()
        relative_image = local_path.relative_to(output_dir).as_posix()
        title = html.escape(clean_text(item.get("title")))
        artist = html.escape(clean_text(item.get("artist")) or "creator not listed")
        source = html.escape(clean_text(item.get("source_page_url")), quote=True)
        confidence = float(
            (item.get("dog_detection") or {}).get("primary_confidence", 0)
        )
        cards.append(
            "\n".join(
                [
                    '<article class="item">',
                    f'  <img loading="lazy" src="{quote(relative_image, safe="/")}" alt="{title}">',
                    f'  <a href="{source}" target="_blank" rel="noreferrer">{title}</a>',
                    f"  <p>CC0 · {artist} · dog {confidence:.2f}</p>",
                    "</article>",
                ]
            )
        )
    gallery = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CC0 dog-photo pilot</title>
  <style>
    :root {{ color-scheme: dark; }}
    body {{ margin: 0; padding: 24px; background: #0d1117; color: #f0f3f6; font-family: system-ui, sans-serif; }}
    h1 {{ margin: 0 0 8px; }}
    .note {{ margin: 0 0 22px; color: #9da7b3; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 14px; }}
    .item {{ min-width: 0; padding: 10px; border: 1px solid #30363d; border-radius: 12px; background: #161b22; }}
    .item img {{ display: block; width: 100%; aspect-ratio: 4 / 3; object-fit: cover; border-radius: 8px; margin-bottom: 9px; }}
    .item a {{ display: block; color: #62d6a8; font-weight: 650; overflow-wrap: anywhere; }}
    .item p {{ margin: 7px 0 0; color: #9da7b3; font-size: 12px; overflow-wrap: anywhere; }}
  </style>
</head>
<body>
  <h1>CC0 dog-photo pilot · {len(items)} images</h1>
  <p class="note">각 제목을 누르면 Wikimedia Commons 원본·라이선스 페이지가 열립니다. 게시 전 수동 검토가 필요합니다.</p>
  <main class="grid">
{chr(10).join(cards)}
  </main>
</body>
</html>
"""
    gallery_path = output_dir / "gallery.html"
    gallery_path.write_text(gallery, encoding="utf-8")
    return gallery_path


def requests_session(user_agent: str) -> requests.Session:
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": user_agent,
            "Accept": "application/json,image/jpeg;q=0.9,*/*;q=0.1",
        }
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def commons_pages(
    session: requests.Session,
    *,
    query: str,
    thumb_width: int,
    max_candidates: int,
) -> Any:
    continuation: dict[str, Any] = {}
    yielded = 0
    while yielded < max_candidates:
        params: dict[str, Any] = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "generator": "search",
            "gsrsearch": query,
            "gsrnamespace": "6",
            "gsrlimit": "50",
            "prop": "imageinfo",
            "iiprop": "url|mime|size|sha1|extmetadata",
            "iiurlwidth": str(thumb_width),
            "iiextmetadatalanguage": "en",
            "iiextmetadatafilter": (
                "LicenseShortName|LicenseUrl|UsageTerms|Artist|Credit|"
                "Attribution|AttributionRequired|Copyrighted|"
                "ImageDescription|Categories|DateTimeOriginal"
            ),
            "maxlag": "5",
            **continuation,
        }
        response = session.get(API_URL, params=params, timeout=35)
        response.raise_for_status()
        payload = response.json()
        if payload.get("error"):
            raise RuntimeError(f"Commons API error: {payload['error']}")
        pages = (payload.get("query") or {}).get("pages") or []
        for page in pages:
            if yielded >= max_candidates:
                return
            yielded += 1
            yield page
        continuation = payload.get("continue") or {}
        if not continuation:
            return


def download_candidate(
    session: requests.Session,
    page: Mapping[str, Any],
    *,
    detector: Any,
    confidence: float,
    min_dog_area: float,
    images_dir: Path,
) -> tuple[dict[str, Any] | None, str]:
    rejection = candidate_rejection_reason(page)
    if rejection:
        return None, rejection

    imageinfo = page["imageinfo"][0]
    metadata = imageinfo["extmetadata"]
    download_url = clean_text(imageinfo.get("thumburl") or imageinfo.get("url"))
    response = session.get(download_url, timeout=45)
    response.raise_for_status()
    raw = response.content
    if not raw or len(raw) > 15 * 1024 * 1024:
        return None, "invalid_download_size"
    if not clean_text(response.headers.get("Content-Type")).startswith("image/"):
        return None, "invalid_content_type"

    try:
        with Image.open(io.BytesIO(raw)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB").copy()
    except (UnidentifiedImageError, OSError):
        return None, "invalid_image"
    try:
        if min(image.size) < 320:
            return None, "decoded_too_small"
        detections = detector.detect(image, species="dog", conf=confidence)
        if not detections:
            return None, "dog_not_detected"
        primary = detections[0]
        if primary.area_ratio < min_dog_area:
            return None, "dog_too_small"
        page_id = int(page["pageid"])
        filename = f"commons_{page_id}.jpg"
        output_path = images_dir / filename
        temporary = images_dir / f".{filename}.tmp-{os.getpid()}"
        images_dir.mkdir(parents=True, exist_ok=True)
        image.save(temporary, format="JPEG", quality=92, optimize=True)
        local_sha256 = hashlib.sha256(temporary.read_bytes()).hexdigest()
        temporary.replace(output_path)

        item = {
            "page_id": page_id,
            "title": clean_text(page.get("title")),
            "title_series_key": title_series_key(clean_text(page.get("title"))),
            "description": strip_html(metadata_value(metadata, "ImageDescription")),
            "categories": [
                category.strip()
                for category in metadata_value(metadata, "Categories").split("|")
                if category.strip()
            ],
            "artist": strip_html(metadata_value(metadata, "Artist")),
            "credit": strip_html(metadata_value(metadata, "Credit")),
            "license_short_name": metadata_value(metadata, "LicenseShortName"),
            "license_url": metadata_value(metadata, "LicenseUrl"),
            "usage_terms": metadata_value(metadata, "UsageTerms"),
            "attribution_required": metadata_value(
                metadata, "AttributionRequired"
            ).casefold()
            == "true",
            "commons_copyrighted_field": metadata_value(metadata, "Copyrighted"),
            "source_page_url": source_page_url(clean_text(page.get("title"))),
            "original_url": clean_text(imageinfo.get("url")),
            "download_url": download_url,
            "commons_sha1": clean_text(imageinfo.get("sha1")),
            "source_payload_sha256": hashlib.sha256(raw).hexdigest(),
            "local_path": str(output_path.relative_to(ROOT)).replace("\\", "/"),
            "local_sha256": local_sha256,
            "bytes": output_path.stat().st_size,
            "width": image.width,
            "height": image.height,
            "retrieved_at": utc_now(),
            "dog_detection": {
                "detector": detector.name,
                "device": detector.device,
                "confidence_threshold": confidence,
                "minimum_primary_area_ratio": min_dog_area,
                "count": len(detections),
                "primary_confidence": round(primary.confidence, 4),
                "primary_bbox_norm": normalize_bbox(
                    primary.xyxy, image.width, image.height
                ),
                "primary_area_ratio": round(primary.area_ratio, 4),
            },
            "photo_quality": compute_photo_quality(image),
        }
        return item, ""
    finally:
        image.close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-candidates", type=int, default=600)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--thumb-width", type=int, default=1280)
    parser.add_argument("--confidence", type=float, default=0.45)
    parser.add_argument("--min-dog-area", type=float, default=0.08)
    parser.add_argument("--max-per-artist", type=int, default=5)
    parser.add_argument("--max-per-series", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.limit <= 0 or args.max_candidates <= 0:
        raise ValueError("limit and max-candidates must be positive")
    if args.max_candidates < args.limit:
        raise ValueError("max-candidates must be at least limit")
    if not 320 <= args.thumb_width <= 2560:
        raise ValueError("thumb-width must be between 320 and 2560")
    if not 0.0 < args.confidence <= 1.0:
        raise ValueError("confidence must be in (0, 1]")
    if not 0.0 <= args.min_dog_area <= 1.0:
        raise ValueError("min-dog-area must be between zero and one")
    if args.max_per_artist <= 0 or args.max_per_series <= 0:
        raise ValueError("diversity caps must be positive")
    if "(" not in args.user_agent or ")" not in args.user_agent:
        raise ValueError("user-agent must include contact information in parentheses")

    output_dir = args.output_dir.resolve()
    images_dir = output_dir / "images"
    manifest_path = output_dir / "manifest.json"
    existing: dict[str, Any] = {}
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("existing manifest schema does not match")
    items = list(existing.get("items") or [])
    known_pages = {int(item["page_id"]) for item in items}
    known_hashes = {clean_text(item.get("local_sha256")) for item in items}
    artist_counts = Counter(
        clean_text(item.get("artist")).casefold()
        for item in items
        if clean_text(item.get("artist"))
    )
    series_counts = Counter(
        clean_text(item.get("title_series_key"))
        or title_series_key(clean_text(item.get("title")))
        for item in items
    )
    rejections = Counter(existing.get("rejections") or {})
    detector = load_torchvision_detector(
        "fasterrcnn_mobilenet_v3_large_fpn", args.device
    )

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source": "Wikimedia Commons Action API",
        "source_api": API_URL,
        "source_reuse_policy": (
            "https://commons.wikimedia.org/wiki/Commons:Reusing_content_outside_Wikimedia"
        ),
        "license_allowlist": ["CC0"],
        "license_policy": (
            "Only files whose live Commons extmetadata reports CC0, the CC0 1.0 "
            "license URL, and Creative Commons Zero usage terms are accepted."
        ),
        "query": args.query,
        "retrieval": {
            "thumb_width": args.thumb_width,
            "user_agent": args.user_agent,
            "sequential_requests": True,
            "maxlag": 5,
        },
        "dog_filter": {
            "detector": detector.name,
            "device": detector.device,
            "confidence_threshold": args.confidence,
            "minimum_primary_area_ratio": args.min_dog_area,
        },
        "diversity_filter": {
            "maximum_per_artist": args.max_per_artist,
            "maximum_per_title_series": args.max_per_series,
        },
        "limitations": [
            "Commons and this script cannot guarantee copyright status in every jurisdiction.",
            "Personality, privacy, trademark, moral-rights, and other non-copyright restrictions may still apply.",
            "Commons images may have appeared in foundation-model pretraining and are not a leakage-free benchmark.",
            "The automated dog detector can produce false positives or false negatives; manually review before publication.",
        ],
        "updated_at": utc_now(),
        "completed": len(items) >= args.limit,
        "items": items,
        "rejections": dict(rejections),
    }
    write_json_atomic(manifest_path, manifest)
    if len(items) >= args.limit:
        write_gallery(output_dir, items)
        return manifest

    session = requests_session(args.user_agent)
    attempted = 0
    try:
        for page in commons_pages(
            session,
            query=args.query,
            thumb_width=args.thumb_width,
            max_candidates=args.max_candidates,
        ):
            page_id = int(page.get("pageid") or 0)
            if not page_id or page_id in known_pages:
                continue
            attempted += 1
            artist_key = candidate_artist(page).casefold()
            series_key = title_series_key(clean_text(page.get("title")))
            if artist_key and artist_counts[artist_key] >= args.max_per_artist:
                item, rejection = None, "artist_cap"
            elif series_counts[series_key] >= args.max_per_series:
                item, rejection = None, "title_series_cap"
            else:
                try:
                    item, rejection = download_candidate(
                        session,
                        page,
                        detector=detector,
                        confidence=args.confidence,
                        min_dog_area=args.min_dog_area,
                        images_dir=images_dir,
                    )
                except requests.RequestException:
                    item, rejection = None, "download_error"
            if item is None:
                rejections[rejection or "unknown"] += 1
            elif item["local_sha256"] in known_hashes:
                Path(ROOT / item["local_path"]).unlink(missing_ok=True)
                rejections["duplicate_payload"] += 1
            else:
                items.append(item)
                known_pages.add(page_id)
                known_hashes.add(item["local_sha256"])
                if artist_key:
                    artist_counts[artist_key] += 1
                series_counts[series_key] += 1

            manifest["updated_at"] = utc_now()
            manifest["completed"] = len(items) >= args.limit
            manifest["items"] = items
            manifest["rejections"] = dict(sorted(rejections.items()))
            manifest["progress"] = {
                "accepted": len(items),
                "target": args.limit,
                "attempted_this_run": attempted,
            }
            write_json_atomic(manifest_path, manifest)
            if attempted % 5 == 0 or item is not None:
                print(
                    json.dumps(
                        {
                            "accepted": len(items),
                            "target": args.limit,
                            "attempted": attempted,
                            "last_rejection": rejection or None,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            if len(items) >= args.limit:
                break
    finally:
        session.close()

    manifest["completed"] = len(items) >= args.limit
    manifest["updated_at"] = utc_now()
    manifest["items"] = items
    manifest["rejections"] = dict(sorted(rejections.items()))
    manifest["progress"] = {
        "accepted": len(items),
        "target": args.limit,
        "attempted_this_run": attempted,
    }
    write_json_atomic(manifest_path, manifest)
    write_gallery(output_dir, items)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    report = run(parse_args(argv))
    print(
        json.dumps(
            {
                "completed": report.get("completed", len(report["items"]) > 0),
                "accepted": len(report["items"]),
                "output": str(DEFAULT_OUTPUT),
                "manifest": str(DEFAULT_OUTPUT / "manifest.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if len(report["items"]) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
