import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.species import clean_species, species_config, species_from_upkind  # noqa: E402
from app.notice_status import is_searchable_notice  # noqa: E402
from app.notice_metadata import (  # noqa: E402
    extract_image_urls,
    normalize_additional_notice_fields,
    normalize_breed_fields as normalize_reported_breed_fields,
    normalize_http_url,
)
from app.notice_provider import (  # noqa: E402
    NATIONAL_ANIMAL_API_BASE,
    LocalJsonNoticeProvider,
    NationalAnimalProtectionNoticeProvider,
    NoticeFetchRequest,
    NoticeProvider,
    validate_provider_result,
)

DATA_DIR = BASE_DIR / "data"
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)

# Backwards-compatible export for scripts/tests that imported this constant.
API_BASE = NATIONAL_ANIMAL_API_BASE
DEFAULT_API_KEY = os.getenv("ANIMAL_API_KEY", "")


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def resolve_lookback_days(years: int, days: Optional[int] = None) -> int:
    """Resolve a bounded lookup window while preserving the legacy years API."""

    if days is not None:
        if days <= 0:
            raise ValueError("days must be a positive integer")
        return days
    if years <= 0:
        raise ValueError("years must be a positive integer")
    return years * 365


def parse_api_date(value: Any) -> Optional[datetime]:
    text = clean_text(value)
    if not text:
        return None

    for fmt in ("%Y%m%d", "%Y-%m-%d", "%Y.%m.%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def is_active_notice(rec: Dict[str, Any], 기준일: datetime) -> bool:
    return is_searchable_notice(
        rec,
        reference_date=기준일,
        include_unknown=True,
    )


def normalize_breed_fields(rec: Dict[str, Any]) -> Dict[str, Any]:
    return normalize_reported_breed_fields(rec)


def extract_image_url(rec: Dict[str, Any]) -> str:
    image_urls = extract_image_urls(rec)
    return image_urls[0] if image_urls else ""


def build_live_animal_meta(rec: Dict[str, Any], species: str = "dog") -> Dict[str, Any]:
    source_fields = normalize_additional_notice_fields(rec)
    desertion_no = (
        clean_text(
            rec.get("desertionNo") or rec.get("desertion_no") or rec.get("notice_id")
        )
        or "Unknown"
    )
    notice_no = clean_text(rec.get("noticeNo") or rec.get("notice_no"))
    upkind = (
        clean_text(rec.get("upkind") or rec.get("upkindCd") or rec.get("upKindCd"))
        or species_config(species)["upkind"]
    )
    normalized_species = species_from_upkind(upkind, clean_species(species))

    return {
        "type": "live",
        "species": normalized_species,
        "upkind": upkind,
        "desertionNo": desertion_no,
        "notice_no": notice_no,
        **source_fields,
        "sex": clean_text(rec.get("sexCd")) or clean_text(rec.get("sex")) or "Unknown",
        "age": clean_text(rec.get("age")) or "Unknown",
        "weight": clean_text(rec.get("weight")) or "Unknown",
        "neuter": clean_text(rec.get("neuterYn"))
        or clean_text(rec.get("neuter"))
        or "Unknown",
        "desc": clean_text(rec.get("specialMark")) or clean_text(rec.get("desc")) or "",
        "detail_url": normalize_http_url(rec.get("detail_url") or rec.get("detailUrl"))
        or (
            "https://www.animal.go.kr/front/awtis/public/publicDtl.do?"
            + urlencode({"desertionNo": desertion_no, "menuNo": "1000000055"})
            if desertion_no != "Unknown"
            else ""
        ),
        "care_name": clean_text(rec.get("careNm") or rec.get("care_name")),
        "care_tel": clean_text(rec.get("careTel") or rec.get("care_tel")),
        "care_addr": clean_text(rec.get("careAddr") or rec.get("care_addr")),
        "org_name": clean_text(rec.get("orgNm") or rec.get("org_name")),
        "happen_place": clean_text(rec.get("happenPlace") or rec.get("happen_place")),
        "notice_start": clean_text(rec.get("noticeSdt") or rec.get("notice_start")),
        "notice_end": clean_text(rec.get("noticeEdt") or rec.get("notice_end")),
        "process_state": clean_text(
            rec.get("processState") or rec.get("process_state")
        ),
    }


def build_live_dog_meta(rec: Dict[str, Any]) -> Dict[str, Any]:
    return build_live_animal_meta(rec, species="dog")


def build_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"Accept": "application/json", "User-Agent": "Mozilla/5.0"})
    return session


def fetch_live_animals(
    api_key: str,
    species: str,
    years: int,
    rows: int,
    max_pages: int,
    include_closed: bool = False,
    days: Optional[int] = None,
    provider: Optional[NoticeProvider] = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    normalized_species = clean_species(species)
    upkind = species_config(normalized_species)["upkind"]
    lookback_days = resolve_lookback_days(years, days)
    end = now or datetime.now().astimezone()
    if end.tzinfo is None or end.utcoffset() is None:
        raise ValueError("now must include a timezone")
    start = end - timedelta(days=lookback_days)
    active_provider = provider or NationalAnimalProtectionNoticeProvider(
        api_key=api_key,
        session=build_session(),
        api_base=API_BASE,
    )
    provider_result = active_provider.fetch(
        NoticeFetchRequest(
            species=normalized_species,
            start=start,
            end=end,
            rows=rows,
            max_pages=max_pages,
        )
    )
    validate_provider_result(active_provider, provider_result)

    items: List[Dict[str, Any]] = []
    seen = set()
    stats = {
        "pages_fetched": provider_result.pages_fetched,
        "raw_items": provider_result.raw_items,
        "provider_records": len(provider_result.records),
        "kept_items": 0,
        "skipped_closed": 0,
        "skipped_no_image": 0,
        "skipped_duplicate": 0,
    }

    for source in provider_result.records:
        rec = source.normalization_input()
        if not include_closed and not is_active_notice(rec, 기준일=end):
            stats["skipped_closed"] += 1
            continue

        meta = build_live_animal_meta(rec, species=normalized_species)
        if not meta["image_url"]:
            stats["skipped_no_image"] += 1
            continue

        # This is the time at which the upstream notice was checked, not
        # an assertion that the notice will remain active afterward.
        meta["last_verified_at"] = end.isoformat(timespec="seconds")
        meta["source_status"] = source.source_status
        meta["source_provenance"] = source.provenance.as_dict()

        desertion_no = meta["desertionNo"]
        if desertion_no in seen:
            stats["skipped_duplicate"] += 1
            continue

        seen.add(desertion_no)
        items.append(meta)
        stats["kept_items"] += 1

    return {
        "fetched_at": end.isoformat(timespec="seconds"),
        "provider": active_provider.provider_id,
        "species": normalized_species,
        "upkind": upkind,
        "years": years if days is None else None,
        "days": lookback_days,
        "include_closed": include_closed,
        "items": items,
        "stats": stats,
        "provider_diagnostics": dict(provider_result.diagnostics),
    }


def fetch_live_dogs(
    api_key: str,
    years: int,
    rows: int,
    max_pages: int,
    include_closed: bool = False,
    days: Optional[int] = None,
) -> Dict[str, Any]:
    return fetch_live_animals(
        api_key,
        "dog",
        years,
        rows,
        max_pages,
        include_closed,
        days,
    )


def save_payload(payload: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="최근 N년 유기동물 공고를 공공 API에서 받아 로컬 JSON 캐시 파일로 저장합니다."
    )
    parser.add_argument(
        "--species",
        default="dog",
        choices=("dog", "cat", "other"),
        help="조회할 동물 종류",
    )
    parser.add_argument(
        "--years", type=int, default=3, help="오늘 기준 몇 년 전까지 조회할지"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="오늘 기준 조회 일수. 지정하면 --years보다 우선합니다.",
    )
    parser.add_argument("--rows", type=int, default=1000, help="페이지당 조회 건수")
    parser.add_argument("--max-pages", type=int, default=500, help="최대 페이지 수")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="출력 JSON 경로. 생략하면 species별 data/local_{species}_cache.json 사용",
    )
    parser.add_argument(
        "--api-key",
        default=DEFAULT_API_KEY,
        help="공공 유기동물 API 키",
    )
    parser.add_argument(
        "--provider",
        choices=("national-api", "local-json"),
        default="national-api",
        help="공고 원천 provider. 기본값은 국가동물보호정보시스템 API",
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        help="--provider local-json에서 읽을 canonical JSON 파일",
    )
    parser.add_argument(
        "--include-closed",
        action="store_true",
        help="공고 종료/처리 완료 데이터도 포함",
    )
    args = parser.parse_args(argv)
    if args.provider == "local-json" and args.input_json is None:
        parser.error("--provider local-json requires --input-json")
    if args.provider != "local-json" and args.input_json is not None:
        parser.error("--input-json can only be used with --provider local-json")
    return args


def provider_from_args(args: argparse.Namespace) -> NoticeProvider:
    if args.provider == "local-json":
        return LocalJsonNoticeProvider(args.input_json)
    return NationalAnimalProtectionNoticeProvider(
        api_key=args.api_key,
        session=build_session(),
        api_base=API_BASE,
    )


def main() -> None:
    args = parse_args()
    species = clean_species(args.species)
    output_path = args.out or DATA_DIR / species_config(species)["cache"]
    payload = fetch_live_animals(
        api_key=args.api_key,
        species=species,
        years=args.years,
        rows=args.rows,
        max_pages=args.max_pages,
        include_closed=args.include_closed,
        days=args.days,
        provider=provider_from_args(args),
    )
    save_payload(payload, output_path)

    stats = payload["stats"]
    print(f"[DONE] 저장 완료: {output_path}")
    print(
        f"[INFO] provider={payload['provider']} "
        f"species={payload['species']} upkind={payload['upkind']}"
    )
    print(f"[INFO] fetched_at={payload['fetched_at']}")
    print(f"[INFO] lookback_days={payload['days']}")
    print(
        f"[INFO] pages={stats['pages_fetched']} raw={stats['raw_items']} kept={stats['kept_items']}"
    )
    print(
        "[INFO] skipped "
        f"closed={stats['skipped_closed']} "
        f"no_image={stats['skipped_no_image']} "
        f"duplicate={stats['skipped_duplicate']}"
    )


if __name__ == "__main__":
    main()
