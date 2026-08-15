import json
import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

import clip
import faiss
import numpy as np
import requests
import torch
from dotenv import load_dotenv
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
from app.notice_status import is_searchable_notice  # noqa: E402
from app.notice_metadata import (  # noqa: E402
    extract_image_urls,
    first_text,
    normalize_additional_notice_fields,
    normalize_breed_fields as normalize_reported_breed_fields,
)
from app.public_image_download import (  # noqa: E402
    DEFAULT_ALLOWED_IMAGE_HOSTS,
    PublicImageDownloader,
)
from app.species import species_from_upkind  # noqa: E402

DATA_DIR = BASE_DIR / "data"
load_dotenv(BASE_DIR / ".env")

# -------------------------
# 기본 설정
# -------------------------
INDEX_PATH = DATA_DIR / "dog_faiss.index"
META_PATH = DATA_DIR / "dog_metas.json"
LAST_UPDATE_PATH = DATA_DIR / "last_update.txt"
CLIP_MODEL = "ViT-B/32"


def _safe_faiss_path(path: Path) -> Path:
    safe_dir = Path(tempfile.gettempdir()) / "dog_faiss_safe"
    safe_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    return safe_dir / f"{path.stem}-{digest}{path.suffix}"


def safe_read_faiss_index(path: Path):
    try:
        return faiss.read_index(str(path))
    except RuntimeError as exc:
        if "Illegal byte sequence" not in str(exc):
            raise
        safe_path = _safe_faiss_path(path)
        if (
            not safe_path.exists()
            or safe_path.stat().st_size != path.stat().st_size
            or safe_path.stat().st_mtime < path.stat().st_mtime
        ):
            shutil.copy2(path, safe_path)
        return faiss.read_index(str(safe_path))


def safe_write_faiss_index(index_obj, path: Path) -> None:
    try:
        faiss.write_index(index_obj, str(path))
    except RuntimeError as exc:
        if "Illegal byte sequence" not in str(exc):
            raise
        safe_path = _safe_faiss_path(path)
        faiss.write_index(index_obj, str(safe_path))
        shutil.copy2(safe_path, path)


BASE = (
    "https://apis.data.go.kr/1543061/abandonmentPublicService_v2/abandonmentPublic_v2"
)
API_KEY = os.getenv("ANIMAL_API_KEY", "")
if not API_KEY:
    raise SystemExit(
        "[ERROR] ANIMAL_API_KEY is missing. Set it in .env before running vector updates."
    )

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(CLIP_MODEL, device=device)
image_downloader = PublicImageDownloader(
    timeout=10,
    allowed_hosts=DEFAULT_ALLOWED_IMAGE_HOSTS,
)

# -------------------------
# 마지막 업데이트 날짜 불러오기
# -------------------------
try:
    with LAST_UPDATE_PATH.open("r") as f:
        last_date = datetime.strptime(f.read().strip(), "%Y%m%d")
except (OSError, ValueError):
    last_date = datetime.now()  # 처음 실행이면 오늘부터 시작

today = datetime.now()
print(f"[INFO] 업데이트 범위: {last_date:%Y%m%d} ~ {today:%Y%m%d}")

# -------------------------
# 기존 index/metas 로드
# -------------------------
index = safe_read_faiss_index(INDEX_PATH)
with META_PATH.open("r", encoding="utf-8") as f:
    metas = json.load(f)

image_seen = {
    str(m.get("desertionNo"))
    for m in metas
    if m.get("desertionNo") and m.get("type") == "image"
}
text_seen = {
    str(m.get("desertionNo"))
    for m in metas
    if m.get("desertionNo") and m.get("type") == "text"
}


# -------------------------
# API fetch 함수
# -------------------------
def fetch_data(bgnde, endde, page=1, rows=500):
    params = {
        "serviceKey": API_KEY,
        "numOfRows": rows,
        "pageNo": page,
        "_type": "json",
        "upkind": 417000,
        "bgnde": bgnde,
        "endde": endde,
    }
    try:
        resp = requests.get(BASE, params=params, timeout=20)
    except requests.RequestException as exc:
        raise RuntimeError(
            f"public animal API request failed: {exc.__class__.__name__}"
        ) from None
    if resp.status_code != 200:
        raise RuntimeError(
            f"API request failed: status={resp.status_code} body={resp.text[:200]}"
        )
    try:
        data = resp.json()
        header = (
            data.get("response", {}).get("header", {}) if isinstance(data, dict) else {}
        )
        if header.get("resultCode") and header.get("resultCode") != "00":
            raise RuntimeError(
                f"API returned {header.get('resultCode')}: {header.get('resultMsg')}"
            )
    except ValueError:
        print("[WARN] JSON 파싱 실패:", resp.text[:200])
        return []
    items = data.get("response", {}).get("body", {}).get("items", {}).get("item", [])
    if isinstance(items, dict):
        items = [items]
    return items or []


def extract_image_url(rec: dict) -> str:
    image_urls = extract_image_urls(rec)
    return image_urls[0] if image_urls else ""


# -------------------------
# CLIP 임베딩 함수
# -------------------------
@torch.no_grad()
def embed_image_from_url(url: str, downloader: PublicImageDownloader | None = None):
    runtime_downloader = downloader or image_downloader
    try:
        image = runtime_downloader(url)
        if image is None:
            return None
        try:
            x = preprocess(image).unsqueeze(0).to(device)
            v = model.encode_image(x)
            v = v / v.norm(dim=-1, keepdim=True)
            return v.cpu().numpy()[0]
        finally:
            image.close()
    except Exception:
        return None


@torch.no_grad()
def embed_text(text: str):
    toks = clip.tokenize([text], truncate=True).to(device)
    v = model.encode_text(toks)
    v = v / v.norm(dim=-1, keepdim=True)
    return v.cpu().numpy()[0]


def build_dog_text(rec: dict) -> str:
    """Build observable/source-fact text without embedding reported breed."""

    desc = first_text(
        rec,
        "merged_desc",
        "vlm_desc",
        "desc",
        "base_desc",
        "specialMark",
    )
    fields = (
        ("설명", desc),
        ("성별", first_text(rec, "sexCd", "sex")),
        ("나이", first_text(rec, "age")),
        ("체중", first_text(rec, "weight")),
        ("중성화", first_text(rec, "neuterYn", "neuter")),
        ("색상", first_text(rec, "color", "colorCd")),
    )
    return ", ".join(f"{label} {value}" for label, value in fields if value)


def normalize_breed_fields(rec: dict):
    fields = normalize_reported_breed_fields(rec)
    return fields["breed"], fields["breed_code"], fields["breed_name"]


# -------------------------
# 신규 데이터 수집 + 임베딩
# -------------------------
bgnde = (last_date).strftime("%Y%m%d")
endde = today.strftime("%Y%m%d")

new_count = 0
skipped_inactive = 0
for page in range(1, 6):  # 페이지 수 조절 가능
    items = fetch_data(bgnde, endde, page=page)
    if not items:
        break

    for rec in items:
        did = str(rec.get("desertionNo") or "").strip()
        if not did:
            continue
        if not is_searchable_notice(rec, include_unknown=False):
            skipped_inactive += 1
            continue
        if did in image_seen and did in text_seen:
            continue

        upkind = str(
            rec.get("upkind") or rec.get("upkindCd") or rec.get("upKindCd") or "417000"
        ).strip()
        species = species_from_upkind(upkind, "dog")
        detail_url = (
            "https://www.animal.go.kr/front/awtis/public/publicDtl.do"
            f"?desertionNo={did}&menuNo=1000000055"
        )

        # 공통 필드 추출
        source_fields = normalize_additional_notice_fields(rec)
        breed_value, breed_code, breed_name = normalize_breed_fields(rec)
        sex = rec.get("sexCd", "")
        age = rec.get("age", "")
        weight = rec.get("weight", "")
        neuter = rec.get("neuterYn", "")
        mark = rec.get("specialMark", "")
        url = source_fields["image_url"] or extract_image_url(rec)
        added_for_dog = False

        # 이미지 임베딩
        if url and did not in image_seen:
            img_vec = embed_image_from_url(url)
            if img_vec is not None:
                index.add(np.expand_dims(img_vec, axis=0))
                metas.append(
                    {
                        **source_fields,
                        "desertionNo": did,
                        "type": "image",
                        "image_url": url,
                        "detail_url": detail_url,
                        "breed": breed_value,
                        "breed_code": breed_code,
                        "breed_name": breed_name,
                        "sex": sex,
                        "age": age,
                        "weight": weight,
                        "neuter": neuter,
                        "specialMark": mark,
                        "species": species,
                        "upkind": upkind,
                        "notice_no": rec.get("noticeNo", ""),
                        "care_name": rec.get("careNm", ""),
                        "care_tel": rec.get("careTel", ""),
                        "care_addr": rec.get("careAddr", ""),
                        "org_name": rec.get("orgNm", ""),
                        "happen_place": rec.get("happenPlace", ""),
                        "notice_start": rec.get("noticeSdt", ""),
                        "notice_end": rec.get("noticeEdt", ""),
                        "process_state": rec.get("processState", ""),
                    }
                )
                image_seen.add(did)
                added_for_dog = True

        # 텍스트 임베딩
        desc_full = build_dog_text(rec)
        if desc_full.strip() and did not in text_seen:
            txt_vec = embed_text(desc_full)
            if txt_vec is not None:
                index.add(np.expand_dims(txt_vec, axis=0))
                metas.append(
                    {
                        **source_fields,
                        "desertionNo": did,
                        "type": "text",
                        "desc_full": desc_full,
                        "detail_url": detail_url,
                        "breed": breed_value,
                        "breed_code": breed_code,
                        "breed_name": breed_name,
                        "sex": sex,
                        "age": age,
                        "weight": weight,
                        "neuter": neuter,
                        "image_url": url or "",
                        "species": species,
                        "upkind": upkind,
                        "notice_no": rec.get("noticeNo", ""),
                        "care_name": rec.get("careNm", ""),
                        "care_tel": rec.get("careTel", ""),
                        "care_addr": rec.get("careAddr", ""),
                        "org_name": rec.get("orgNm", ""),
                        "happen_place": rec.get("happenPlace", ""),
                        "notice_start": rec.get("noticeSdt", ""),
                        "notice_end": rec.get("noticeEdt", ""),
                        "process_state": rec.get("processState", ""),
                    }
                )
                text_seen.add(did)
                added_for_dog = True

        if added_for_dog:
            new_count += 1

# -------------------------
# 저장
# -------------------------
safe_write_faiss_index(index, INDEX_PATH)
with META_PATH.open("w", encoding="utf-8", newline="\n") as f:
    json.dump(metas, f, ensure_ascii=False, indent=2)
    f.write("\n")
with LAST_UPDATE_PATH.open("w") as f:
    f.write(today.strftime("%Y%m%d"))

image_downloader.close()
print(
    f"[DONE] 신규 {new_count}개 추가 완료. inactive_skip={skipped_inactive}. index={index.ntotal}, metas={len(metas)}"
)
