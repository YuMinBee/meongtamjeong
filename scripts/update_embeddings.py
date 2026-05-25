import json
import os
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

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
load_dotenv(BASE_DIR / ".env")

# -------------------------
# 기본 설정
# -------------------------
INDEX_PATH = DATA_DIR / "dog_faiss.index"
META_PATH = DATA_DIR / "dog_metas.json"
LAST_UPDATE_PATH = DATA_DIR / "last_update.txt"
CLIP_MODEL = "ViT-B/32"

BASE = "http://apis.data.go.kr/1543061/abandonmentPublicService_v2/abandonmentPublic_v2"
API_KEY = os.getenv("ANIMAL_API_KEY", "")

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
index = faiss.read_index(str(INDEX_PATH))
with META_PATH.open("r", encoding="utf-8") as f:
    metas = json.load(f)

seen = {m["desertionNo"] for m in metas}

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
        return []
    try:
        data = resp.json()
    except Exception:
        print("[WARN] JSON 파싱 실패:", resp.text[:200])
        return []
    items = data.get("response", {}).get("body", {}).get("items", {}).get("item", [])
    if isinstance(items, dict): items = [items]
    return items or []

# -------------------------
# CLIP 임베딩 함수
# -------------------------
@torch.no_grad()
def embed_image_from_url(url: str):
    try:
        r = requests.get(url, timeout=10)
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
for page in range(1, 6):  # 페이지 수 조절 가능
    items = fetch_data(bgnde, endde, page=page)
    if not items: break

    for rec in items:
        did = str(rec.get("desertionNo"))
        if did in seen:
            continue  # 이미 있는 개체 skip

        detail_url = (
            "https://www.animal.go.kr/front/awtis/public/publicDtl.do"
            f"?desertionNo={did}&menuNo=1000000055"
        )

        # ✅ 공통 필드 추출
        breed_value, breed_code, breed_name = normalize_breed_fields(rec)
        sex = rec.get("sexCd","")
        age = rec.get("age","")
        weight = rec.get("weight","")
        neuter = rec.get("neuterYn","")
        mark = rec.get("specialMark","")
        url = rec.get("popfile")

        # ✅ 이미지 임베딩
        if url:
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
                    "specialMark": mark
                })

        # ✅ 텍스트 임베딩
        desc_full = build_dog_text(rec)
        if desc_full.strip():
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
                    "image_url": url or ""
                })

        seen.add(did)
        new_count += 1

# -------------------------
# 저장
# -------------------------
faiss.write_index(index, str(INDEX_PATH))
with META_PATH.open("w", encoding="utf-8") as f:
    json.dump(metas, f, ensure_ascii=False, indent=2)
with LAST_UPDATE_PATH.open("w") as f:
    f.write(today.strftime("%Y%m%d"))

print(f"[DONE] 신규 {new_count}개 추가 완료. index={index.ntotal}, metas={len(metas)}")
