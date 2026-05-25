import argparse
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)

API_BASE = (
    "http://apis.data.go.kr/1543061/abandonmentPublicService_v2/"
    "abandonmentPublic_v2"
)
DEFAULT_API_KEY = os.getenv("ANIMAL_API_KEY", "")
DEFAULT_OUTPUT_PATH = DATA_DIR / "local_dog_cache.json"


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


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
    notice_end = parse_api_date(rec.get("noticeEdt") or rec.get("notice_end"))
    if notice_end and notice_end.date() < 기준일.date():
        return False

    process_state = clean_text(rec.get("processState") or rec.get("process_state"))
    if any(token in process_state for token in ("종료", "입양", "반환", "자연사", "안락사")):
        return False
    return True


def normalize_breed_fields(rec: Dict[str, Any]) -> Dict[str, str]:
    raw_kind = clean_text(rec.get("kindCd") or rec.get("breed") or rec.get("breed_name"))
    raw_breed_code = clean_text(rec.get("breedCd") or rec.get("breed_code"))

    breed_code = raw_breed_code or (raw_kind if raw_kind.isdigit() else "")
    breed_name = raw_kind if raw_kind and not raw_kind.isdigit() else clean_text(rec.get("breed_name"))
    breed = breed_name or breed_code or "Unknown"

    return {
        "breed": breed,
        "breed_code": breed_code,
        "breed_name": breed_name,
    }


def extract_image_url(rec: Dict[str, Any]) -> str:
    for key in ("image_url", "url", "popfile", "fileName", "thumb", "image", "img"):
        value = clean_text(rec.get(key))
        if value.startswith("http"):
            return value

    for key, value in rec.items():
        text = clean_text(value)
        if not text.startswith("http"):
            continue
        if any(token in str(key).lower() for token in ("pop", "img", "thumb")):
            return text
    return ""


def build_live_dog_meta(rec: Dict[str, Any]) -> Dict[str, Any]:
    breed_fields = normalize_breed_fields(rec)
    desertion_no = clean_text(rec.get("desertionNo")) or "Unknown"
    notice_no = clean_text(rec.get("noticeNo"))

    return {
        "type": "live",
        "desertionNo": desertion_no,
        "notice_no": notice_no,
        "breed": breed_fields["breed"],
        "breed_code": breed_fields["breed_code"],
        "breed_name": breed_fields["breed_name"],
        "sex": clean_text(rec.get("sexCd")) or clean_text(rec.get("sex")) or "Unknown",
        "age": clean_text(rec.get("age")) or "Unknown",
        "weight": clean_text(rec.get("weight")) or "Unknown",
        "neuter": clean_text(rec.get("neuterYn")) or clean_text(rec.get("neuter")) or "Unknown",
        "desc": clean_text(rec.get("specialMark")) or clean_text(rec.get("desc")) or "",
        "image_url": extract_image_url(rec),
        "detail_url": (
            "https://www.animal.go.kr/front/awtis/public/publicDtl.do"
            f"?desertionNo={desertion_no}&menuNo=1000000055"
            if desertion_no != "Unknown"
            else ""
        ),
        "care_name": clean_text(rec.get("careNm")),
        "notice_start": clean_text(rec.get("noticeSdt")),
        "notice_end": clean_text(rec.get("noticeEdt")),
        "process_state": clean_text(rec.get("processState")),
    }


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


def fetch_live_dogs(
    api_key: str,
    years: int,
    rows: int,
    max_pages: int,
    include_closed: bool = False,
) -> Dict[str, Any]:
    end = datetime.now()
    start = end - timedelta(days=years * 365)
    session = build_session()

    items: List[Dict[str, Any]] = []
    seen = set()
    stats = {
        "pages_fetched": 0,
        "raw_items": 0,
        "kept_items": 0,
        "skipped_closed": 0,
        "skipped_no_image": 0,
        "skipped_duplicate": 0,
    }

    for page in range(1, max_pages + 1):
        params = {
            "serviceKey": api_key,
            "numOfRows": rows,
            "pageNo": page,
            "_type": "json",
            "upkind": 417000,
            "bgnde": start.strftime("%Y%m%d"),
            "endde": end.strftime("%Y%m%d"),
        }

        resp = session.get(API_BASE, params=params, timeout=30)
        resp.raise_for_status()

        data = resp.json()
        body = data.get("response", {}).get("body", {}) if isinstance(data, dict) else {}
        page_items = body.get("items", {}).get("item", [])
        if isinstance(page_items, dict):
            page_items = [page_items]
        if not page_items:
            break

        stats["pages_fetched"] += 1
        stats["raw_items"] += len(page_items)

        for rec in page_items:
            if not include_closed and not is_active_notice(rec, 기준일=end):
                stats["skipped_closed"] += 1
                continue

            meta = build_live_dog_meta(rec)
            if not meta["image_url"]:
                stats["skipped_no_image"] += 1
                continue

            desertion_no = meta["desertionNo"]
            if desertion_no in seen:
                stats["skipped_duplicate"] += 1
                continue

            seen.add(desertion_no)
            items.append(meta)
            stats["kept_items"] += 1

    return {
        "fetched_at": end.isoformat(timespec="seconds"),
        "years": years,
        "days": years * 365,
        "include_closed": include_closed,
        "items": items,
        "stats": stats,
    }


def save_payload(payload: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="최근 N년 유기견 공고를 공공 API에서 받아 로컬 JSON 캐시 파일로 저장합니다."
    )
    parser.add_argument("--years", type=int, default=3, help="오늘 기준 몇 년 전까지 조회할지")
    parser.add_argument("--rows", type=int, default=1000, help="페이지당 조회 건수")
    parser.add_argument("--max-pages", type=int, default=500, help="최대 페이지 수")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="출력 JSON 경로",
    )
    parser.add_argument(
        "--api-key",
        default=DEFAULT_API_KEY,
        help="공공 유기동물 API 키",
    )
    parser.add_argument(
        "--include-closed",
        action="store_true",
        help="공고 종료/처리 완료 데이터도 포함",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = fetch_live_dogs(
        api_key=args.api_key,
        years=args.years,
        rows=args.rows,
        max_pages=args.max_pages,
        include_closed=args.include_closed,
    )
    save_payload(payload, args.out)

    stats = payload["stats"]
    print(f"[DONE] 저장 완료: {args.out}")
    print(f"[INFO] fetched_at={payload['fetched_at']}")
    print(f"[INFO] years={payload['years']} days={payload['days']}")
    print(f"[INFO] pages={stats['pages_fetched']} raw={stats['raw_items']} kept={stats['kept_items']}")
    print(
        "[INFO] skipped "
        f"closed={stats['skipped_closed']} "
        f"no_image={stats['skipped_no_image']} "
        f"duplicate={stats['skipped_duplicate']}"
    )


if __name__ == "__main__":
    main()
