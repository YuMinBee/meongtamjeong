import json
import os
import random
import time
from pathlib import Path

import clip
import faiss
import numpy as np
import requests
import torch
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import quote

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
load_dotenv(BASE_DIR / ".env")

# =========================
# 설정
# =========================
TARGET = 10000
CHECKPOINT = 500
DATE_RANGE_DAYS = 365
ROWS_PER_PAGE = 1000
MAX_PAGES = 500

BASE_SLEEP = 0.12
JITTER = 0.10
PAGE_SLEEP = 0.6

FAISS_INDEX_PATH = DATA_DIR / "dog_faiss.index"
META_PATH = DATA_DIR / "dog_metas.json"

# =========================
# 모델/세션 준비
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
print(f"[INFO] device={device}")

BASE = "http://apis.data.go.kr/1543061/abandonmentPublicService_v2/abandonmentPublic_v2"
API_KEY = os.getenv(
    "ANIMAL_API_KEY",
    "BAyDlIxqnenEk4FAAtRmypbtiEl02tPx76PpoU/x6xxA1h1CKisHTS5ezQjR8LC5GHIRlpOLEPIvdXLloh+N0g==",
)

session = requests.Session()
retries = Retry(total=5, backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504])
session.mount("http://", HTTPAdapter(max_retries=retries))
session.headers.update({"Accept": "application/json", "User-Agent": "Mozilla/5.0"})

def slowdown():
    time.sleep(max(0.01, BASE_SLEEP + random.uniform(-JITTER, JITTER)))

def safe_url(u: str) -> str:
    return quote(u, safe=":/?&=#[]%")

def fetch_data(page=1, rows=ROWS_PER_PAGE, max_wait=180):
    end = datetime.now()
    start = end - timedelta(days=DATE_RANGE_DAYS)
    params = {
        "serviceKey": API_KEY,
        "numOfRows": rows,
        "pageNo": page,
        "_type": "json",
        "upkind": 417000,   # 개(dog) 코드
        "bgnde": start.strftime("%Y%m%d"),
        "endde": end.strftime("%Y%m%d"),
    }

    start_time = time.time()
    attempt = 0

    while True:
        attempt += 1
        try:
            resp = session.get(BASE, params=params, timeout=30)
            resp.raise_for_status()
            try:
                data = resp.json()
                body = data.get("response", {}).get("body", {}) if isinstance(data, dict) else {}
                items = body.get("items", {}).get("item", [])
                if isinstance(items, dict):
                    items = [items]
                return items or []
            except ValueError:
                print(f"[WARN] page {page}: JSON 파싱 실패 (시도 {attempt})")
                print("응답 앞부분:", resp.text[:200])
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] page {page}: 요청 실패 (시도 {attempt}) → {e}")

        # 3분(180초) 넘으면 포기하고 스킵
        if time.time() - start_time > max_wait:
            print(f"[FAIL] page {page}: {max_wait}초 동안 시도했지만 실패 → 건너뜀")
            return []

        time.sleep(2)  # 재시도 간격 (2초)

def extract_image_url(rec: dict):
    if not isinstance(rec, dict): return None
    for k in ["popfile", "fileName", "thumb", "image", "img"]:
        v = rec.get(k)
        if isinstance(v, str) and v.startswith("http"):
            return v
    for k, v in rec.items():
        if isinstance(v, str) and any(s in k.lower() for s in ["pop", "img", "thumb"]):
            if v.startswith("http"): return v
    return None

# =========================
# 임베딩 함수
# =========================
def get_clip_embedding_from_url(url: str):
    try:
        slowdown()
        headers = {"Accept": "image/*", "Referer": "https://www.animal.go.kr/", "User-Agent": session.headers["User-Agent"]}
        r = session.get(safe_url(url), timeout=10, headers=headers)
        if r.status_code != 200 and url.startswith("http://"):
            r = session.get("https://" + url[7:], timeout=10, headers=headers)
        if r.status_code != 200:
            return None
        img = Image.open(BytesIO(r.content)).convert("RGB")
        inp = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(inp)
            feat /= feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy()[0]
    except:
        return None

def get_clip_text_embedding(text: str):
    try:
        tokens = clip.tokenize([text]).to(device)
        with torch.no_grad():
            feat = model.encode_text(tokens)
            feat /= feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy()[0]
    except:
        return None

def build_dog_text(rec: dict) -> str:
    """CLIP 텍스트 임베딩용 문장 생성"""
    kind = rec.get("kindCd", "")
    sex = rec.get("sexCd", "")
    age = rec.get("age", "")
    weight = rec.get("weight", "")
    neuter = rec.get("neuterYn", "")
    mark = rec.get("specialMark", "")
    return f"{kind}, 성별 {sex}, 나이 {age}, 체중 {weight}, 중성화 {neuter}, 특징: {mark}"


def normalize_breed_fields(rec: dict):
    raw_kind = str(rec.get("kindCd", "") or "").strip()
    raw_breed_code = str(rec.get("breedCd", "") or "").strip()

    breed_code = raw_breed_code or (raw_kind if raw_kind.isdigit() else "")
    breed_name = raw_kind if raw_kind and not raw_kind.isdigit() else ""
    breed_value = breed_name or breed_code or "Unknown"

    return breed_value, breed_code, breed_name

# =========================
# FAISS 준비
# =========================
d = 512  # ViT-B/32 output dimension
index = faiss.IndexFlatL2(d)
metas = []

def save_ckpt(index, metas, tag="final"):
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    with META_PATH.open("w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False, indent=2)
    print(f"[{tag}] 저장 완료 → {FAISS_INDEX_PATH}, {META_PATH}")

# =========================
# 실행
# =========================
seen = set()
ok = 0

overall = tqdm(total=TARGET, desc="총 임베딩 수집 (FAISS)")

for page in range(1, MAX_PAGES + 1):
    items = fetch_data(page=page)
    if not items:
        print(f"[INFO] page {page}: 결과 없음 → 종료")
        break

    for rec in tqdm(items, desc=f"🐶 임베딩 추출 (page {page})"):
        if ok >= TARGET:
            break
        did = str(rec.get("desertionNo") or "")
        if did in seen:
            continue
        seen.add(did)

        detail_url = (
            "https://www.animal.go.kr/front/awtis/public/publicDtl.do"
            f"?desertionNo={did}&menuNo=1000000055"
        )

        # ✅ 공통 필드 뽑기
        breed_value, breed_code, breed_name = normalize_breed_fields(rec)
        sex = rec.get("sexCd", "")
        age = rec.get("age", "")
        weight = rec.get("weight", "")
        neuter = rec.get("neuterYn", "")
        mark = rec.get("specialMark", "")

        # ✅ 1) 이미지 임베딩
        url = extract_image_url(rec)
        if url:
            vec = get_clip_embedding_from_url(url)
            if vec is not None:
                index.add(np.expand_dims(vec, axis=0))
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
                    "desc": mark  # 특징만 따로
                })

        # ✅ 2) 텍스트 임베딩 (desc_full = CLIP 입력용)
        desc_full = build_dog_text(rec)
        if desc_full.strip():
            vec_txt = get_clip_text_embedding(desc_full)
            if vec_txt is not None:
                index.add(np.expand_dims(vec_txt, axis=0))
                metas.append({
                    "desertionNo": did,
                    "type": "text",
                    "desc": desc_full,
                    "detail_url": detail_url,
                    "breed": breed_value,
                    "breed_code": breed_code,
                    "breed_name": breed_name,
                    "sex": sex,
                    "age": age,
                    "weight": weight,
                    "neuter": neuter,
                    "image_url": url or ""
                })

        ok += 1
        overall.update(1)

        if ok % CHECKPOINT == 0:
            save_ckpt(index, metas, tag=f"CKPT {ok}/{TARGET}")

    if ok >= TARGET: break
    time.sleep(PAGE_SLEEP)

overall.close()
save_ckpt(index, metas, tag="FINAL")
print(f" 완료! 벡터DB 임베딩 {ok}개 저장됨")
