import argparse
import hashlib
import json
import os
import shutil
import tempfile
import random
import sys
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
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.dog_attributes import build_photo_advice, format_vlm_attrs_for_embedding, summarize_vlm_attrs_ko
from app.notice_status import is_searchable_notice
from app.species import clean_species, species_config, species_from_upkind

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
DEFAULT_INPUT_PATH = DATA_DIR / "local_dog_cache_enriched.json"
FETCH_UPKIND = "417000"
ACTIVE_SPECIES = "dog"

# =========================
# 모델/세션 준비
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
print(f"[INFO] device={device}")

BASE = "https://apis.data.go.kr/1543061/abandonmentPublicService_v2/abandonmentPublic_v2"
API_KEY = os.getenv("ANIMAL_API_KEY", "")

session = requests.Session()
retries = Retry(total=5, backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504])
session.mount("http://", HTTPAdapter(max_retries=retries))
session.headers.update({"Accept": "application/json", "User-Agent": "Mozilla/5.0"})

def slowdown():
    time.sleep(max(0.01, BASE_SLEEP + random.uniform(-JITTER, JITTER)))

def safe_url(u: str) -> str:
    return quote(u, safe=":/?&=#[]%")


def clean_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def first_text(rec: dict, *keys: str) -> str:
    for key in keys:
        value = clean_text(rec.get(key))
        if value:
            return value
    return ""


def resolve_embedding_desc(rec: dict) -> str:
    attrs_text = format_vlm_attrs_for_embedding(rec.get("vlm_attrs"))
    vlm_desc = first_text(rec, "vlm_desc")
    base_desc = first_text(rec, "desc", "base_desc", "specialMark")
    merged_desc = first_text(rec, "merged_desc")
    desc_full = first_text(rec, "desc_full")

    parts = [text for text in (attrs_text, f"사진 보강 설명: {vlm_desc}" if vlm_desc else "", f"기존 설명: {base_desc}" if base_desc else "") if text]
    if parts:
        return "\n".join(parts)
    return merged_desc or desc_full

def load_input_records(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        records = payload.get("items", [])
    else:
        records = payload
    if not isinstance(records, list):
        raise ValueError(f"input records must be a list or an object with items: {path}")
    return records


def parse_args():
    parser = argparse.ArgumentParser(description="Build FAISS embeddings for animal search.")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Local JSON cache to index. Defaults to data/local_{species}_cache_enriched.json or data/local_{species}_cache.json when present.",
    )
    parser.add_argument("--species", default="dog", choices=("dog", "cat", "other"), help="Animal species namespace for default files and metadata.")
    parser.add_argument("--target", type=int, default=TARGET, help="Maximum records to process.")
    parser.add_argument("--text-only", action="store_true", help="Skip image downloads and build only text embeddings.")
    parser.add_argument("--include-closed", action="store_true", help="Include closed or expired notices in the rebuilt FAISS index.")
    parser.add_argument(
        "--exclude-unknown",
        action="store_true",
        help="Exclude notices whose active status cannot be verified.",
    )
    parser.add_argument("--index-out", type=Path, default=None, help="Output FAISS index path. Defaults to data/{species}_faiss.index.")
    parser.add_argument("--metas-out", type=Path, default=None, help="Output metadata path. Defaults to data/{species}_metas.json.")
    parser.add_argument("--upkind", default="", help="Override public API upkind code when fetching without --input.")
    return parser.parse_args()

def fetch_data(page=1, rows=ROWS_PER_PAGE, max_wait=180):
    end = datetime.now()
    start = end - timedelta(days=DATE_RANGE_DAYS)
    params = {
        "serviceKey": API_KEY,
        "numOfRows": rows,
        "pageNo": page,
        "_type": "json",
        "upkind": FETCH_UPKIND,
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
    for k in ["image_url", "url", "popfile", "popfile1", "popfile2", "fileName", "thumb", "image", "img"]:
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
def encode_clip_image(img: Image.Image):
    inp = preprocess(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(inp)
        feat /= feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy()[0]


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
        return encode_clip_image(img)
    except:
        return None


def resolve_local_path(value: str):
    text = clean_text(value)
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path if path.exists() else None


def get_clip_embedding_from_path(path: Path):
    try:
        img = Image.open(path).convert("RGB")
        return encode_clip_image(img)
    except:
        return None


def get_clip_text_embedding(text: str):
    try:
        tokens = clip.tokenize([text], truncate=True).to(device)
        with torch.no_grad():
            feat = model.encode_text(tokens)
            feat /= feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy()[0]
    except:
        return None

def build_dog_text(rec: dict) -> str:
    """Create the CLIP text input, prioritizing VLM-enriched descriptions."""
    kind = first_text(rec, "breed_name", "kindCd", "breed", "breed_code", "breedCd")
    sex = first_text(rec, "sexCd", "sex")
    age = first_text(rec, "age")
    weight = first_text(rec, "weight")
    neuter = first_text(rec, "neuterYn", "neuter")
    desc = resolve_embedding_desc(rec)

    parts = []
    if desc:
        parts.append(f"설명: {desc}")
    if kind:
        parts.append(f"품종 {kind}")
    if sex:
        parts.append(f"성별 {sex}")
    if age:
        parts.append(f"나이 {age}")
    if weight:
        parts.append(f"체중 {weight}")
    if neuter:
        parts.append(f"중성화 {neuter}")
    return ", ".join(parts)


def normalize_breed_fields(rec: dict):
    raw_kind = first_text(rec, "kindCd", "breed", "breed_name")
    raw_breed_code = first_text(rec, "breedCd", "breed_code")

    breed_code = raw_breed_code or (raw_kind if raw_kind.isdigit() else "")
    breed_name = raw_kind if raw_kind and not raw_kind.isdigit() else first_text(rec, "breed_name")
    breed_value = breed_name or breed_code or "Unknown"

    return breed_value, breed_code, breed_name

# =========================
def _safe_faiss_path(path: Path) -> Path:
    safe_dir = Path(tempfile.gettempdir()) / "dog_faiss_safe"
    safe_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    return safe_dir / f"{path.stem}-{digest}{path.suffix}"


def safe_write_faiss_index(index_obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        faiss.write_index(index_obj, str(path))
    except RuntimeError as exc:
        if "Illegal byte sequence" not in str(exc):
            raise
        safe_path = _safe_faiss_path(path)
        faiss.write_index(index_obj, str(safe_path))
        shutil.copy2(safe_path, path)

# FAISS 준비
# =========================
d = 512  # ViT-B/32 output dimension
index = faiss.IndexFlatL2(d)
metas = []

def save_ckpt(index, metas, tag="final"):
    safe_write_faiss_index(index, FAISS_INDEX_PATH)
    with META_PATH.open("w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False, indent=2)
    print(f"[{tag}] 저장 완료 → {FAISS_INDEX_PATH}, {META_PATH}")

# =========================
# 실행
# =========================
args = parse_args()
ACTIVE_SPECIES = clean_species(args.species)
species_meta = species_config(ACTIVE_SPECIES)
FETCH_UPKIND = str(args.upkind or species_meta["upkind"])
FAISS_INDEX_PATH = args.index_out or (DATA_DIR / species_meta["index"])
META_PATH = args.metas_out or (DATA_DIR / species_meta["metas"])
if args.input is None:
    for candidate_name in (species_meta["enriched_cache"], species_meta["cache"]):
        candidate = DATA_DIR / candidate_name
        if candidate.exists():
            args.input = candidate
            break
target = max(0, args.target)
seen = set()
ok = 0
skipped_inactive = 0
print(f"[INFO] species={ACTIVE_SPECIES} upkind={FETCH_UPKIND}")
print(f"[INFO] outputs index={FAISS_INDEX_PATH} metas={META_PATH}")
if args.input:
    source_records = load_input_records(args.input)
    target = min(target, len(source_records))
    page_sources = [(f"input {args.input}", source_records)]
    print(f"[INFO] local input={args.input} records={len(source_records)} target={target}")
else:
    page_sources = None
    print("[INFO] local input not provided; fetching records from public API")

overall = tqdm(total=target, desc="총 임베딩 수집 (FAISS)")

page_no = 0
while ok < target:
    page_no += 1
    if page_sources is not None:
        if page_no > len(page_sources):
            break
        page_label, items = page_sources[page_no - 1]
    else:
        if page_no > MAX_PAGES:
            break
        page_label = f"page {page_no}"
        items = fetch_data(page=page_no)

    if not items:
        print(f"[INFO] {page_label}: 결과 없음 → 종료")
        break

    for rec in tqdm(items, desc=f"임베딩 추출 ({page_label})"):
        if ok >= target:
            break

        did = first_text(rec, "desertionNo")
        if not did or did in seen:
            continue
        if not args.include_closed and not is_searchable_notice(
            rec,
            include_unknown=not args.exclude_unknown,
        ):
            skipped_inactive += 1
            continue
        seen.add(did)

        detail_url = first_text(rec, "detail_url") or (
            "https://www.animal.go.kr/front/awtis/public/publicDtl.do"
            f"?desertionNo={did}&menuNo=1000000055"
        )

        breed_value, breed_code, breed_name = normalize_breed_fields(rec)
        sex = first_text(rec, "sexCd", "sex")
        age = first_text(rec, "age")
        weight = first_text(rec, "weight")
        neuter = first_text(rec, "neuterYn", "neuter")
        desc = resolve_embedding_desc(rec)
        special_mark = first_text(rec, "specialMark") or first_text(rec, "desc")
        desc_full = build_dog_text(rec)
        url = extract_image_url(rec)

        upkind = first_text(rec, "upkind", "upkindCd") or FETCH_UPKIND
        common_meta = {
            "desertionNo": did,
            "species": species_from_upkind(upkind, ACTIVE_SPECIES),
            "upkind": upkind,
            "detail_url": detail_url,
            "notice_no": first_text(rec, "notice_no", "noticeNo"),
            "breed": breed_value,
            "breed_code": breed_code,
            "breed_name": breed_name,
            "sex": sex,
            "age": age,
            "weight": weight,
            "neuter": neuter,
            "desc": desc,
            "desc_full": desc_full,
            "specialMark": special_mark,
            "base_desc": first_text(rec, "base_desc"),
            "vlm_desc": first_text(rec, "vlm_desc"),
            "merged_desc": first_text(rec, "merged_desc"),
            "vlm_attrs": rec.get("vlm_attrs") if isinstance(rec.get("vlm_attrs"), dict) else {},
            "vlm_attr_text": first_text(rec, "vlm_attr_text") or summarize_vlm_attrs_ko(rec.get("vlm_attrs")),
            "photo_advice": rec.get("photo_advice") if isinstance(rec.get("photo_advice"), list) else build_photo_advice(rec.get("vlm_attrs")),
            "image_attrs": dict(rec.get("image_attrs")) if isinstance(rec.get("image_attrs"), dict) else {},
            "care_name": first_text(rec, "care_name", "careNm"),
            "care_tel": first_text(rec, "care_tel", "careTel"),
            "care_addr": first_text(rec, "care_addr", "careAddr"),
            "org_name": first_text(rec, "org_name", "orgNm"),
            "happen_place": first_text(rec, "happen_place", "happenPlace"),
            "notice_start": first_text(rec, "notice_start", "noticeSdt"),
            "notice_end": first_text(rec, "notice_end", "noticeEdt"),
            "process_state": first_text(rec, "process_state", "processState"),
            "image_url": url or "",
        }

        full_image_vec = None
        crop_image_vec = None
        image_attrs = common_meta.get("image_attrs") if isinstance(common_meta.get("image_attrs"), dict) else {}
        if url and not args.text_only:
            full_image_vec = get_clip_embedding_from_url(url)
        crop_path = resolve_local_path(image_attrs.get("crop_path")) if image_attrs else None
        if crop_path is not None and not args.text_only:
            crop_image_vec = get_clip_embedding_from_path(crop_path)
        if full_image_vec is not None and crop_image_vec is not None:
            similarity = float(np.dot(full_image_vec, crop_image_vec))
            image_attrs["full_crop_clip_similarity"] = round(similarity, 4)
            common_meta["image_attrs"] = image_attrs

        if full_image_vec is not None:
            image_meta = dict(common_meta)
            image_meta["type"] = "image"
            image_meta["embedding_source"] = "full_image"
            index.add(np.expand_dims(full_image_vec, axis=0))
            metas.append(image_meta)

        if crop_image_vec is not None:
            crop_meta = dict(common_meta)
            crop_meta["type"] = "crop_image"
            crop_meta["embedding_source"] = "dog_crop" if ACTIVE_SPECIES == "dog" else "animal_crop"
            index.add(np.expand_dims(crop_image_vec, axis=0))
            metas.append(crop_meta)

        if desc_full.strip():
            vec_txt = get_clip_text_embedding(desc_full)
            if vec_txt is not None:
                text_meta = dict(common_meta)
                text_meta["type"] = "text"
                index.add(np.expand_dims(vec_txt, axis=0))
                metas.append(text_meta)

        ok += 1
        overall.update(1)

        if ok % CHECKPOINT == 0:
            save_ckpt(index, metas, tag=f"CKPT {ok}/{target}")

    if page_sources is not None:
        break
    time.sleep(PAGE_SLEEP)

overall.close()
save_ckpt(index, metas, tag="FINAL")
print(f"완료! records={ok}, vectors={index.ntotal}, metas={len(metas)}, skipped_inactive={skipped_inactive}")
