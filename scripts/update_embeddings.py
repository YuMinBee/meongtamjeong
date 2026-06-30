import json
import hashlib
import os
import shutil
import tempfile
import time
from pathlib import Path

import clip
import faiss
import numpy as np
import requests
import torch
from dotenv import load_dotenv
from datetime import datetime
from PIL import Image
from io import BytesIO
from urllib.parse import quote

BASE_DIR = Path(__file__).resolve().parents[1]
import sys
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
from app.notice_status import is_searchable_notice
from app.species import species_from_upkind
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

BASE = "http://apis.data.go.kr/1543061/abandonmentPublicService_v2/abandonmentPublic_v2"
API_KEY = os.getenv("ANIMAL_API_KEY", "")
if not API_KEY:
    raise SystemExit(
        "[ERROR] ANIMAL_API_KEY is missing. Set it in .env before running vector updates."
    )

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(CLIP_MODEL, device=device)

# -------------------------
# 마지막 업데이트 날짜 불러오기
# -------------------------
try:
    with LAST_UPDATE_PATH.open("r") as f:
        last_date = datetime.strptime(f.read().strip(), "%Y%m%d")
except:
    last_date = datetime.now()  # 처음 실행이면 오늘부터 시작

today = datetime.now()
print(f"[INFO] 업데이트 범위: {last_date:%Y%m%d} ~ {today:%Y%m%d}")

# -------------------------
# 기존 index/metas 로드
# -------------------------
index = safe_read_faiss_index(INDEX_PATH)
with META_PATH.open("r", encoding="utf-8") as f:
    metas = json.load(f)

image_seen = {str(m.get("desertionNo")) for m in metas if m.get("desertionNo") and m.get("type") == "image"}
text_seen = {str(m.get("desertionNo")) for m in metas if m.get("desertionNo") and m.get("type") == "text"}

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
    resp = requests.get(BASE, params=params, timeout=20)
    if resp.status_code != 200:
        raise RuntimeError(f"API request failed: status={resp.status_code} body={resp.text[:200]}")
    try:
        data = resp.json()
        header = data.get("response", {}).get("header", {}) if isinstance(data, dict) else {}
        if header.get("resultCode") and header.get("resultCode") != "00":
            raise RuntimeError(f"API returned {header.get('resultCode')}: {header.get('resultMsg')}")
    except ValueError:
        print("[WARN] JSON 파싱 실패:", resp.text[:200])
        return []
    items = data.get("response", {}).get("body", {}).get("items", {}).get("item", [])
    if isinstance(items, dict): items = [items]
    return items or []


def safe_url(url: str) -> str:
    return quote(str(url), safe=":/?&=#[]%")


def extract_image_url(rec: dict) -> str:
    for key in ("image_url", "url", "popfile", "popfile1", "popfile2", "fileName", "thumb", "image", "img"):
        value = rec.get(key)
        if isinstance(value, str) and value.startswith("http"):
            return value
    for key, value in rec.items():
        if isinstance(value, str) and value.startswith("http") and any(token in key.lower() for token in ("pop", "file", "img", "thumb")):
            return value
    return ""
# -------------------------
# CLIP 임베딩 함수
# -------------------------
@torch.no_grad()
def embed_image_from_url(url: str):
    try:
        headers = {"Accept": "image/*", "Referer": "https://www.animal.go.kr/", "User-Agent": "Mozilla/5.0"}
        r = requests.get(safe_url(url), timeout=10, headers=headers)
        if r.status_code != 200 and str(url).startswith("http://"):
            r = requests.get(safe_url("https://" + str(url)[7:]), timeout=10, headers=headers)
        if r.status_code != 200: return None
        pil = Image.open(BytesIO(r.content)).convert("RGB")
        x = preprocess(pil).unsqueeze(0).to(device)
        v = model.encode_image(x)
        v = v / v.norm(dim=-1, keepdim=True)
        return v.cpu().numpy()[0]
    except:
        return None

@torch.no_grad()
def embed_text(text: str):
    toks = clip.tokenize([text], truncate=True).to(device)
    v = model.encode_text(toks)
    v = v / v.norm(dim=-1, keepdim=True)
    return v.cpu().numpy()[0]

def build_dog_text(rec: dict) -> str:
    return f"{rec.get('kindCd','')}, 성별 {rec.get('sexCd','')}, 나이 {rec.get('age','')}, " \
           f"체중 {rec.get('weight','')}, 중성화 {rec.get('neuterYn','')}, 특징: {rec.get('specialMark','')}"


def normalize_breed_fields(rec: dict):
    raw_kind = str(rec.get("kindCd", "") or "").strip()
    raw_breed_code = str(rec.get("breedCd", "") or "").strip()

    breed_code = raw_breed_code or (raw_kind if raw_kind.isdigit() else "")
    breed_name = raw_kind if raw_kind and not raw_kind.isdigit() else ""
    breed_value = breed_name or breed_code or "Unknown"

    return breed_value, breed_code, breed_name

# -------------------------
# 신규 데이터 수집 + 임베딩
# -------------------------
bgnde = (last_date).strftime("%Y%m%d")
endde = today.strftime("%Y%m%d")

new_count = 0
skipped_inactive = 0
for page in range(1, 6):  # 페이지 수 조절 가능
    items = fetch_data(bgnde, endde, page=page)
    if not items: break

    for rec in items:
        did = str(rec.get("desertionNo") or "").strip()
        if not did:
            continue
        if not is_searchable_notice(rec, include_unknown=False):
            skipped_inactive += 1
            continue
        if did in image_seen and did in text_seen:
            continue

        upkind = str(rec.get("upkind") or rec.get("upkindCd") or "417000").strip()
        species = species_from_upkind(upkind, "dog")
        detail_url = (            "https://www.animal.go.kr/front/awtis/public/publicDtl.do"
            f"?desertionNo={did}&menuNo=1000000055"
        )

        # 공통 필드 추출
        breed_value, breed_code, breed_name = normalize_breed_fields(rec)
        sex = rec.get("sexCd", "")
        age = rec.get("age", "")
        weight = rec.get("weight", "")
        neuter = rec.get("neuterYn", "")
        mark = rec.get("specialMark", "")
        url = extract_image_url(rec)
        added_for_dog = False

        # 이미지 임베딩
        if url and did not in image_seen:
            img_vec = embed_image_from_url(url)
            if img_vec is not None:
                index.add(np.expand_dims(img_vec, axis=0))
                metas.append({
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
                })
                image_seen.add(did)
                added_for_dog = True

        # 텍스트 임베딩
        desc_full = build_dog_text(rec)
        if desc_full.strip() and did not in text_seen:
            txt_vec = embed_text(desc_full)
            if txt_vec is not None:
                index.add(np.expand_dims(txt_vec, axis=0))
                metas.append({
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
                })
                text_seen.add(did)
                added_for_dog = True

        if added_for_dog:
            new_count += 1

# -------------------------
# 저장
# -------------------------
safe_write_faiss_index(index, INDEX_PATH)
with META_PATH.open("w", encoding="utf-8") as f:
    json.dump(metas, f, ensure_ascii=False, indent=2)
with LAST_UPDATE_PATH.open("w") as f:
    f.write(today.strftime("%Y%m%d"))

print(f"[DONE] 신규 {new_count}개 추가 완료. inactive_skip={skipped_inactive}. index={index.ntotal}, metas={len(metas)}")
