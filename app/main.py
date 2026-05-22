import io
import html
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import clip
import faiss
import numpy as np
import requests
import torch
from dotenv import load_dotenv
from fastapi import Body, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
from pydantic import BaseModel
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.query_expansion import expand_query_text


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)
app = FastAPI(title="Dog Similarity + Gemma")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.getenv("API_KEY", "capstone7")
INDEX_PATH = Path(os.getenv("INDEX_PATH", str(DATA_DIR / "dog_faiss.index")))
METAS_PATH = Path(os.getenv("METAS_PATH", str(DATA_DIR / "dog_metas.json")))
CLIP_MODEL = os.getenv("CLIP_MODEL", "ViT-B/32")
ANIMAL_API_KEY = os.getenv(
    "ANIMAL_API_KEY",
    "BAyDlIxqnenEk4FAAtRmypbtiEl02tPx76PpoU/x6xxA1h1CKisHTS5ezQjR8LC5GHIRlpOLEPIvdXLloh+N0g==",
)
ANIMAL_API_BASE = (
    "http://apis.data.go.kr/1543061/abandonmentPublicService_v2/"
    "abandonmentPublic_v2"
)
LIVE_DOG_CACHE_PATH = DATA_DIR / "local_dog_cache.json"
LIVE_FETCH_YEARS = int(os.getenv("LIVE_FETCH_YEARS", "3"))
LIVE_FETCH_DAYS = int(os.getenv("LIVE_FETCH_DAYS", str(LIVE_FETCH_YEARS * 365)))
LIVE_FETCH_ROWS = int(os.getenv("LIVE_FETCH_ROWS", "1000"))
LIVE_FETCH_MAX_PAGES = int(os.getenv("LIVE_FETCH_MAX_PAGES", "500"))
LIVE_CACHE_LOCK = Lock()
GEMMA3_MODEL_PATH = Path(
    os.getenv(
        "GEMMA3_MODEL_PATH",
        "/data4tb/.hf_cache/hub/models--google--gemma-3-12b-it/snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80",
    )
)
GEMMA3_MODEL_ID = os.getenv("GEMMA3_MODEL_ID", "google/gemma-3-12b-it")
GEMMA3_GPU_MAX_MEMORY = os.getenv("GEMMA3_GPU_MAX_MEMORY", "20GiB")
GEMMA3_CPU_MAX_MEMORY = os.getenv("GEMMA3_CPU_MAX_MEMORY", "96GiB")
GEMMA3_RECOMMEND_MAX_NEW_TOKENS = int(os.getenv("GEMMA3_RECOMMEND_MAX_NEW_TOKENS", "320"))
GEMMA3_CHAT_MAX_NEW_TOKENS = int(os.getenv("GEMMA3_CHAT_MAX_NEW_TOKENS", "384"))
GEMMA3_GPU_ONLY = os.getenv("GEMMA3_GPU_ONLY", "").lower() in {"1", "true", "yes", "on"}
GEMMA_RUNTIME_LOCK = Lock()
GEMMA_PROCESSOR: Optional[Any] = None
GEMMA_MODEL: Optional[Any] = None
GEMMA_MODEL_NAME = ""


@app.middleware("http")
async def api_key_checker(request: Request, call_next):
    key = request.headers.get("x-api-key")
    if request.method == "GET" and not key:
        key = request.query_params.get("api_key")
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")
    return await call_next(request)


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(CLIP_MODEL, device=device)
model.eval()

if not INDEX_PATH.exists():
    raise FileNotFoundError(INDEX_PATH)
if not METAS_PATH.exists():
    raise FileNotFoundError(METAS_PATH)

index = faiss.read_index(str(INDEX_PATH))
with METAS_PATH.open("r", encoding="utf-8") as f:
    METAS: List[Dict[str, Any]] = json.load(f)
if index.ntotal != len(METAS):
    raise ValueError(f"Index({index.ntotal}) != Metas({len(METAS)})")


@torch.no_grad()
def embed_image(pil: Image.Image) -> np.ndarray:
    x = preprocess(pil).unsqueeze(0).to(device)
    v = model.encode_image(x)
    v = v / v.norm(dim=-1, keepdim=True)
    return v.detach().cpu().numpy().astype("float32")


@torch.no_grad()
def embed_text(text: str) -> np.ndarray:
    toks = clip.tokenize([text], truncate=True).to(device)
    v = model.encode_text(toks)
    v = v / v.norm(dim=-1, keepdim=True)
    return v.detach().cpu().numpy().astype("float32")


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def configure_hf_cache_env() -> None:
    cache_dir = BASE_DIR / ".hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    hub_cache = cache_dir / "hub"
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(hub_cache)
    os.environ["HF_HUB_DISABLE_XET"] = "1"


def resolve_gemma_model_path() -> Tuple[str, bool]:
    candidate = GEMMA3_MODEL_PATH.expanduser()
    if candidate.exists():
        return str(candidate), True
    return GEMMA3_MODEL_ID, False


def load_gemma_runtime() -> Tuple[Any, Any]:
    global GEMMA_PROCESSOR, GEMMA_MODEL, GEMMA_MODEL_NAME
    if GEMMA_PROCESSOR is not None and GEMMA_MODEL is not None:
        return GEMMA_PROCESSOR, GEMMA_MODEL

    with GEMMA_RUNTIME_LOCK:
        if GEMMA_PROCESSOR is not None and GEMMA_MODEL is not None:
            return GEMMA_PROCESSOR, GEMMA_MODEL

        configure_hf_cache_env()
        try:
            from transformers import AutoProcessor, Gemma3ForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "Gemma 추천 기능을 사용하려면 transformers, accelerate, sentencepiece가 필요합니다."
            ) from exc

        model_path, local_files_only = resolve_gemma_model_path()
        processor = AutoProcessor.from_pretrained(
            model_path,
            use_fast=False,
            local_files_only=local_files_only,
        )

        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model_kwargs: Dict[str, Any] = {
            "device_map": "auto",
            "torch_dtype": torch_dtype,
            "local_files_only": local_files_only,
        }
        if GEMMA3_GPU_ONLY:
            model_kwargs["device_map"] = {"": 0}
        elif torch.cuda.is_available():
            model_kwargs["max_memory"] = {0: GEMMA3_GPU_MAX_MEMORY, "cpu": GEMMA3_CPU_MAX_MEMORY}

        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_path,
            **model_kwargs,
        )

        GEMMA_PROCESSOR = processor
        GEMMA_MODEL = model
        GEMMA_MODEL_NAME = clean_text(getattr(model.config, "_name_or_path", "")) or model_path
        return GEMMA_PROCESSOR, GEMMA_MODEL


def gemma_generate_text(prompt: str, max_new_tokens: int) -> str:
    processor, model = load_gemma_runtime()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    generated = outputs[0][inputs["input_ids"].shape[-1] :]
    return processor.decode(generated, skip_special_tokens=True).strip()


def resolve_breed_code(meta: Dict[str, Any]) -> str:
    breed_code = clean_text(meta.get("breed_code"))
    if breed_code:
        return breed_code

    for key in ("breed", "kindCd"):
        value = clean_text(meta.get(key))
        if value.isdigit():
            return value
    return ""


def resolve_breed_name(meta: Dict[str, Any]) -> str:
    breed_name = clean_text(meta.get("breed_name"))
    if breed_name:
        return breed_name

    for key in ("breed", "kindCd"):
        value = clean_text(meta.get(key))
        if value and not value.isdigit():
            return value
    return ""


def resolve_breed_label(meta: Dict[str, Any]) -> str:
    return resolve_breed_name(meta) or resolve_breed_code(meta) or "Unknown"


def resolve_detail_url(meta: Dict[str, Any]) -> str:
    detail_url = clean_text(meta.get("detail_url"))
    if detail_url:
        detail_url = detail_url.replace("publicDetail.do", "publicDtl.do")
        if "publicDtl.do" in detail_url and "menuNo=" not in detail_url:
            sep = "&" if "?" in detail_url else "?"
            detail_url = f"{detail_url}{sep}menuNo=1000000055"
        return detail_url

    desertion_no = clean_text(meta.get("desertionNo"))
    if not desertion_no:
        return ""
    return (
        "https://www.animal.go.kr/front/awtis/public/publicDtl.do"
        f"?desertionNo={desertion_no}&menuNo=1000000055"
    )


def resolve_desc(meta: Dict[str, Any]) -> str:
    return (
        clean_text(meta.get("desc"))
        or clean_text(meta.get("desc_full"))
        or clean_text(meta.get("specialMark"))
    )


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
        lower_key = str(key).lower()
        if any(token in lower_key for token in ("pop", "img", "thumb")):
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


def is_active_notice(rec: Dict[str, Any], 기준일: Optional[datetime] = None) -> bool:
    기준일 = 기준일 or datetime.now()
    notice_end = parse_api_date(rec.get("noticeEdt") or rec.get("notice_end"))
    if notice_end and notice_end.date() < 기준일.date():
        return False

    process_state = clean_text(rec.get("processState") or rec.get("process_state"))
    if any(token in process_state for token in ("종료", "입양", "반환", "자연사", "안락사")):
        return False
    return True


def load_live_dog_cache() -> Dict[str, Any]:
    if not LIVE_DOG_CACHE_PATH.exists():
        return {"fetched_at": "", "days": 0, "items": []}

    with LIVE_DOG_CACHE_PATH.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        return {"fetched_at": "", "days": 0, "items": payload}

    items = payload.get("items")
    if not isinstance(items, list):
        items = []
    return {
        "fetched_at": clean_text(payload.get("fetched_at")),
        "days": int(payload.get("days") or 0),
        "items": items,
    }


def save_live_dog_cache(payload: Dict[str, Any]) -> None:
    with LIVE_DOG_CACHE_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_api_session() -> requests.Session:
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


def fetch_live_dog_cache(
    days: int = LIVE_FETCH_DAYS,
    rows: int = LIVE_FETCH_ROWS,
    max_pages: int = LIVE_FETCH_MAX_PAGES,
) -> Dict[str, Any]:
    end = datetime.now()
    start = end - timedelta(days=days)
    session = build_api_session()
    items: List[Dict[str, Any]] = []
    seen = set()

    for page in range(1, max_pages + 1):
        params = {
            "serviceKey": ANIMAL_API_KEY,
            "numOfRows": rows,
            "pageNo": page,
            "_type": "json",
            "upkind": 417000,
            "bgnde": start.strftime("%Y%m%d"),
            "endde": end.strftime("%Y%m%d"),
        }
        resp = session.get(ANIMAL_API_BASE, params=params, timeout=30)
        resp.raise_for_status()

        data = resp.json()
        body = data.get("response", {}).get("body", {}) if isinstance(data, dict) else {}
        page_items = body.get("items", {}).get("item", [])
        if isinstance(page_items, dict):
            page_items = [page_items]
        if not page_items:
            break

        for rec in page_items:
            if not is_active_notice(rec, 기준일=end):
                continue
            meta = build_live_dog_meta(rec)
            if not meta["image_url"]:
                continue
            desertion_no = meta["desertionNo"]
            if desertion_no in seen:
                continue
            seen.add(desertion_no)
            items.append(meta)

    payload = {
        "fetched_at": end.isoformat(timespec="seconds"),
        "days": days,
        "items": items,
    }
    save_live_dog_cache(payload)
    return payload


def get_live_dog_cache(refresh: bool = False, days: int = LIVE_FETCH_DAYS) -> Dict[str, Any]:
    with LIVE_CACHE_LOCK:
        cache = load_live_dog_cache()
        if not refresh and cache["items"] and cache.get("days") == days:
            return cache

        try:
            return fetch_live_dog_cache(days=days)
        except requests.RequestException:
            if cache["items"]:
                return cache
            raise


def build_breed_gallery_index(metas: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    buckets: Dict[str, Dict[str, Any]] = {}
    seen_items = set()

    for meta in metas:
        image_url = clean_text(meta.get("image_url") or meta.get("url"))
        if not image_url:
            continue

        desertion_no = clean_text(meta.get("desertionNo")) or image_url
        dedupe_key = (desertion_no, image_url)
        if dedupe_key in seen_items:
            continue
        seen_items.add(dedupe_key)

        breed_code = resolve_breed_code(meta) or "UNKNOWN"
        breed_name = resolve_breed_name(meta)
        bucket = buckets.setdefault(
            breed_code,
            {
                "breed_code": breed_code,
                "breed_name": breed_name,
                "count": 0,
                "images": [],
            },
        )
        if breed_name and not bucket["breed_name"]:
            bucket["breed_name"] = breed_name

        bucket["count"] += 1
        bucket["images"].append(
            {
                "desertionNo": clean_text(meta.get("desertionNo")) or "Unknown",
                "notice_no": clean_text(meta.get("notice_no") or meta.get("noticeNo")),
                "breed": resolve_breed_label(meta),
                "breed_code": breed_code,
                "breed_name": bucket["breed_name"],
                "age": clean_text(meta.get("age")) or "Unknown",
                "sex": clean_text(meta.get("sex") or meta.get("sexCd")) or "Unknown",
                "weight": clean_text(meta.get("weight")) or "Unknown",
                "neuter": clean_text(meta.get("neuter") or meta.get("neuterYn")) or "Unknown",
                "desc": resolve_desc(meta),
                "image_url": image_url,
                "detail_url": resolve_detail_url(meta),
                "care_name": clean_text(meta.get("care_name")),
                "notice_start": clean_text(meta.get("notice_start")),
                "notice_end": clean_text(meta.get("notice_end")),
                "process_state": clean_text(meta.get("process_state")),
            }
        )

    for bucket in buckets.values():
        bucket["images"].sort(
            key=lambda item: (
                item["desertionNo"] != "Unknown",
                item["desertionNo"],
            ),
            reverse=True,
        )

    return buckets


BREED_GALLERY = build_breed_gallery_index(METAS)


def list_breed_summaries(
    gallery: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    gallery = gallery or BREED_GALLERY
    summaries = []
    for bucket in gallery.values():
        sample = bucket["images"][:3]
        summaries.append(
            {
                "breed_code": bucket["breed_code"],
                "breed_name": bucket["breed_name"],
                "label": bucket["breed_name"] or bucket["breed_code"],
                "count": bucket["count"],
                "sample_images": [item["image_url"] for item in sample],
            }
        )

    summaries.sort(key=lambda item: (-item["count"], item["breed_code"]))
    return summaries


def render_breed_gallery_page(
    request: Request,
    gallery: Dict[str, Dict[str, Any]],
    page_title: str,
    page_description: str,
    summary_path: str,
    image_path_template: str,
    breed_code: Optional[str],
    limit: int,
    fetched_at: str = "",
    extra_query: str = "",
) -> HTMLResponse:
    summaries = list_breed_summaries(gallery)
    selected_code = breed_code or (summaries[0]["breed_code"] if summaries else None)
    selected_bucket = gallery.get(selected_code) if selected_code else None
    selected_images = selected_bucket["images"][:limit] if selected_bucket else []
    api_key = request.query_params.get("api_key", "")
    api_key_q = quote(api_key, safe="")

    summary_cards = []
    for summary in summaries[:120]:
        label = html.escape(summary["label"])
        href = (
            f"{summary_path}?breed_code={summary['breed_code']}"
            f"&limit={limit}&api_key={api_key_q}{extra_query}"
        )
        state_class = "breed-card selected" if summary["breed_code"] == selected_code else "breed-card"
        summary_cards.append(
            f"""
            <a class="{state_class}" href="{href}">
              <strong>{label}</strong>
              <span>code {html.escape(summary["breed_code"])}</span>
              <span>{summary["count"]} dogs</span>
            </a>
            """
        )

    image_cards = []
    for item in selected_images:
        detail_url = html.escape(item["detail_url"], quote=True)
        image_url = html.escape(item["image_url"], quote=True)
        desc = html.escape(item["desc"] or "특징 정보 없음")
        notice = ""
        if item.get("notice_start") or item.get("notice_end"):
            notice = (
                f"<span>공고 {html.escape(item.get('notice_start') or '-')}"
                f" ~ {html.escape(item.get('notice_end') or '-')}</span>"
            )
        care_name = ""
        if item.get("care_name"):
            care_name = f"<span>보호소 {html.escape(item['care_name'])}</span>"
        process_state = ""
        if item.get("process_state"):
            process_state = f"<span>상태 {html.escape(item['process_state'])}</span>"

        image_cards.append(
            f"""
            <article class="dog-card">
              <a href="{detail_url}" target="_blank" rel="noreferrer">
                <img src="{image_url}" alt="{html.escape(item['breed'])}" loading="lazy" />
              </a>
              <div class="dog-meta">
                <strong>{html.escape(item["desertionNo"])}</strong>
                <span>{html.escape(item["breed"])}</span>
                <span>{html.escape(item["sex"])} / {html.escape(item["age"])}</span>
                <span>{html.escape(item["weight"])} / 중성화 {html.escape(item["neuter"])}</span>
                {care_name}
                {process_state}
                {notice}
                <p>{desc}</p>
              </div>
            </article>
            """
        )

    selected_title = "선택된 종 코드가 없습니다."
    json_link = "#"
    if selected_bucket:
        selected_name = selected_bucket["breed_name"] or selected_bucket["breed_code"]
        selected_title = (
            f"{html.escape(selected_name)} "
            f"({html.escape(selected_bucket['breed_code'])}) "
            f"- {selected_bucket['count']}마리"
        )
        json_link = (
            f"{image_path_template.format(breed_code=selected_bucket['breed_code'])}"
            f"?limit={limit}&api_key={api_key_q}{extra_query}"
        )

    fetched_line = ""
    if fetched_at:
        fetched_line = f"<p>최근 수집 시각: {html.escape(fetched_at)}</p>"

    page = f"""
    <!doctype html>
    <html lang="ko">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>{html.escape(page_title)}</title>
        <style>
          :root {{
            --bg: #f5efe6;
            --panel: rgba(255, 252, 247, 0.92);
            --ink: #231815;
            --muted: #6c6258;
            --line: rgba(35, 24, 21, 0.12);
            --accent: #c96b37;
          }}
          * {{ box-sizing: border-box; }}
          body {{
            margin: 0;
            font-family: "Gill Sans", "Noto Sans KR", sans-serif;
            color: var(--ink);
            background:
              radial-gradient(circle at top left, rgba(255, 255, 255, 0.8), transparent 30%),
              linear-gradient(135deg, #f7f2ea 0%, #e8d7c5 100%);
          }}
          .shell {{
            width: min(1400px, calc(100vw - 32px));
            margin: 24px auto 40px;
          }}
          .hero {{
            padding: 28px;
            border: 1px solid var(--line);
            border-radius: 28px;
            background: var(--panel);
            box-shadow: 0 20px 60px rgba(35, 24, 21, 0.08);
          }}
          h1 {{
            margin: 0 0 8px;
            font-size: clamp(2rem, 3vw, 3.4rem);
            line-height: 0.95;
            letter-spacing: -0.04em;
          }}
          .hero p {{
            margin: 0;
            color: var(--muted);
          }}
          .layout {{
            display: grid;
            grid-template-columns: minmax(260px, 320px) 1fr;
            gap: 20px;
            margin-top: 20px;
          }}
          .panel {{
            border: 1px solid var(--line);
            border-radius: 24px;
            background: var(--panel);
            padding: 20px;
            box-shadow: 0 16px 40px rgba(35, 24, 21, 0.06);
          }}
          .breed-list {{
            display: grid;
            gap: 10px;
            max-height: 78vh;
            overflow: auto;
          }}
          .breed-card {{
            display: grid;
            gap: 4px;
            padding: 14px 16px;
            border-radius: 18px;
            border: 1px solid var(--line);
            text-decoration: none;
            color: inherit;
            background: rgba(255, 255, 255, 0.72);
            transition: transform 0.18s ease, border-color 0.18s ease, background 0.18s ease;
          }}
          .breed-card:hover {{
            transform: translateY(-1px);
            border-color: rgba(201, 107, 55, 0.5);
          }}
          .breed-card.selected {{
            background: linear-gradient(135deg, #fff5ef 0%, #f7ddcf 100%);
            border-color: rgba(201, 107, 55, 0.7);
          }}
          .breed-card span {{
            color: var(--muted);
            font-size: 0.92rem;
          }}
          .section-head {{
            display: flex;
            justify-content: space-between;
            align-items: end;
            gap: 12px;
            margin-bottom: 16px;
          }}
          .section-head h2 {{
            margin: 0;
            font-size: clamp(1.3rem, 2vw, 2rem);
          }}
          .section-head a {{
            color: var(--accent);
            text-decoration: none;
            font-weight: 600;
          }}
          .gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
            gap: 16px;
          }}
          .dog-card {{
            overflow: hidden;
            border-radius: 22px;
            border: 1px solid var(--line);
            background: rgba(255, 255, 255, 0.82);
          }}
          .dog-card img {{
            display: block;
            width: 100%;
            aspect-ratio: 1 / 1;
            object-fit: cover;
            background: #eadfd4;
          }}
          .dog-meta {{
            display: grid;
            gap: 6px;
            padding: 14px;
          }}
          .dog-meta span {{
            color: var(--muted);
            font-size: 0.92rem;
          }}
          .dog-meta p {{
            margin: 4px 0 0;
            color: var(--ink);
            font-size: 0.95rem;
            line-height: 1.45;
          }}
          @media (max-width: 900px) {{
            .layout {{
              grid-template-columns: 1fr;
            }}
            .breed-list {{
              max-height: none;
            }}
          }}
        </style>
      </head>
      <body>
        <main class="shell">
          <section class="hero">
            <h1>{html.escape(page_title)}</h1>
            <p>{html.escape(page_description)}</p>
            {fetched_line}
          </section>
          <section class="layout">
            <aside class="panel">
              <div class="section-head">
                <h2>종 코드 목록</h2>
                <span>{len(summaries)} codes</span>
              </div>
              <div class="breed-list">
                {''.join(summary_cards) or '<p>표시할 종 코드가 없습니다.</p>'}
              </div>
            </aside>
            <section class="panel">
              <div class="section-head">
                <div>
                  <h2>{selected_title}</h2>
                  <p>최대 {limit}장까지 표시합니다.</p>
                </div>
                <a href="{json_link}" target="_blank" rel="noreferrer">JSON 보기</a>
              </div>
              <div class="gallery">
                {''.join(image_cards) or '<p>선택한 종 코드에 표시할 이미지가 없습니다.</p>'}
              </div>
            </section>
          </section>
        </main>
      </body>
    </html>
    """
    return HTMLResponse(page)


def search(vec: np.ndarray, topk: int = 5):
    D, I = index.search(vec, topk)
    results = []

    for rank, (d, i) in enumerate(zip(D[0].tolist(), I[0].tolist())):
        if i == -1:
            continue
        m = METAS[i]

        if m.get("type") == "text":
            did = m.get("desertionNo")
            img_meta = next(
                (
                    mm
                    for mm in METAS
                    if mm.get("desertionNo") == did and mm.get("type") == "image"
                ),
                None,
            )
            if img_meta:
                m = img_meta

        desertion_no = m.get("desertionNo", "Unknown")
        image_url = m.get("image_url") or m.get("url") or ""
        detail_url = resolve_detail_url(m)

        results.append(
            {
                "rank": rank + 1,
                "score": float(1 / (1 + d)),
                "desertionNo": clean_text(desertion_no) or "Unknown",
                "breed": resolve_breed_label(m),
                "breed_code": resolve_breed_code(m),
                "breed_name": resolve_breed_name(m),
                "age": clean_text(m.get("age")) or "Unknown",
                "sex": clean_text(m.get("sex") or m.get("sexCd")) or "Unknown",
                "weight": clean_text(m.get("weight")) or "Unknown",
                "neuter": clean_text(m.get("neuter") or m.get("neuterYn")) or "Unknown",
                "desc": resolve_desc(m),
                "image_url": clean_text(image_url),
                "detail_url": detail_url,
            }
        )

    return results


def gemma_recommend(profile_text: str, candidates: List[Dict[str, Any]]) -> str:
    lines = []
    for c in candidates:
        lines.append(
            {
                "score": round(c["score"], 4),
                "breed": c.get("breed") or c.get("type") or "Unknown",
                "age": c.get("age", "Unknown"),
                "sex": c.get("sex", "Unknown"),
                "desc": c.get("desc", ""),
                "url": c.get("url") or c.get("detail_url") or "",
            }
        )

    context = json.dumps(lines, ensure_ascii=False)

    prompt = f"""
너는 반려견 입양 컨설턴트야. 아래 사용자의 프로필을 고려해서
후보 목록에 있는 실제 유기견 중 가장 적합한 6마리를 추천해줘.

- 반드시 후보 목록에 있는 정보만 사용해.
- 유사도 점수가 높은 후보를 우선하되,
  점수가 비슷하다면 성별, 나이, 체중 등이 다양한 후보들을 골라서 균형 있게 추천해.
- 다만 후보에 적힌 간단한 정보(품종, 성별, 나이, 체중, 중성화, 특징)를 바탕으로
  사용자의 프로필 조건에 맞는 부분을 해석해서 설명을 보완해줘.
- 예를 들어 "산책 주 3회 가능"이라고 하면, 활동량이 많은 품종이나 어린 강아지를 우선 추천해.
- "차분한 성격"이라고 하면, 특징에 '순함', '조용함' 같은 단어가 있으면 강조해.
- 불필요한 정보는 만들지 말고, 간단 명확하게 작성해.

[사용자 프로필]
{profile_text}

[후보 목록]
{context}
"""
    return gemma_generate_text(
        prompt=prompt,
        max_new_tokens=GEMMA3_RECOMMEND_MAX_NEW_TOKENS,
    )


class TextQuery(BaseModel):
    query: str
    topk: Optional[int] = 5


@app.get("/breeds")
def breed_summary(limit: int = Query(100, ge=1, le=500)):
    summaries = list_breed_summaries()
    return JSONResponse(
        {
            "count": len(summaries),
            "results": summaries[:limit],
        }
    )


@app.get("/breeds/{breed_code}/images")
def breed_images(
    breed_code: str,
    limit: int = Query(60, ge=1, le=500),
):
    bucket = BREED_GALLERY.get(breed_code)
    if not bucket:
        raise HTTPException(status_code=404, detail="Breed code not found")

    return JSONResponse(
        {
            "breed_code": bucket["breed_code"],
            "breed_name": bucket["breed_name"],
            "count": bucket["count"],
            "images": bucket["images"][:limit],
        }
    )


@app.get("/visualize/breeds", response_class=HTMLResponse)
def visualize_breeds(
    request: Request,
    breed_code: Optional[str] = Query(None),
    limit: int = Query(60, ge=1, le=200),
):
    return render_breed_gallery_page(
        request=request,
        gallery=BREED_GALLERY,
        page_title="Breed Code Gallery",
        page_description="임베딩 메타에서 이미지가 있는 항목만 뽑아서 종 코드별로 묶은 화면입니다.",
        summary_path="/visualize/breeds",
        image_path_template="/breeds/{breed_code}/images",
        breed_code=breed_code,
        limit=limit,
    )


@app.get("/live/breeds")
def live_breed_summary(
    refresh: bool = Query(False),
    years: int = Query(LIVE_FETCH_YEARS, ge=1, le=5),
    limit: int = Query(100, ge=1, le=500),
):
    days = years * 365
    cache = get_live_dog_cache(refresh=refresh, days=days)
    gallery = build_breed_gallery_index(cache["items"])
    summaries = list_breed_summaries(gallery)
    return JSONResponse(
        {
            "count": len(summaries),
            "fetched_at": cache.get("fetched_at", ""),
            "days": cache.get("days", days),
            "results": summaries[:limit],
        }
    )


@app.get("/live/breeds/{breed_code}/images")
def live_breed_images(
    breed_code: str,
    refresh: bool = Query(False),
    years: int = Query(LIVE_FETCH_YEARS, ge=1, le=5),
    limit: int = Query(60, ge=1, le=500),
):
    days = years * 365
    cache = get_live_dog_cache(refresh=refresh, days=days)
    gallery = build_breed_gallery_index(cache["items"])
    bucket = gallery.get(breed_code)
    if not bucket:
        raise HTTPException(status_code=404, detail="Breed code not found")

    return JSONResponse(
        {
            "breed_code": bucket["breed_code"],
            "breed_name": bucket["breed_name"],
            "count": bucket["count"],
            "fetched_at": cache.get("fetched_at", ""),
            "days": cache.get("days", days),
            "images": bucket["images"][:limit],
        }
    )


@app.get("/visualize/live-breeds", response_class=HTMLResponse)
def visualize_live_breeds(
    request: Request,
    breed_code: Optional[str] = Query(None),
    refresh: bool = Query(False),
    years: int = Query(LIVE_FETCH_YEARS, ge=1, le=5),
    limit: int = Query(60, ge=1, le=200),
):
    days = years * 365
    cache = get_live_dog_cache(refresh=refresh, days=days)
    gallery = build_breed_gallery_index(cache["items"])
    return render_breed_gallery_page(
        request=request,
        gallery=gallery,
        page_title="Live Breed Code Gallery",
        page_description=f"오늘 기준 최근 {years}년 유기견 공고 중 아직 종료되지 않은 데이터만 종 코드별로 묶은 화면입니다.",
        summary_path="/visualize/live-breeds",
        image_path_template="/live/breeds/{breed_code}/images",
        breed_code=breed_code,
        limit=limit,
        fetched_at=cache.get("fetched_at", ""),
        extra_query=f"&refresh={'true' if refresh else 'false'}&years={years}",
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "index_size": index.ntotal,
        "device": device,
        "breed_groups": len(BREED_GALLERY),
    }


@app.post("/search/text")
def search_text(body: TextQuery):
    search_query = expand_query_text(body.query)
    vec = embed_text(search_query)
    return JSONResponse({"results": search(vec, body.topk or 5)})


@app.post("/recommend")
def recommend(body: TextQuery):
    search_query = expand_query_text(body.query)
    vec = embed_text(search_query)
    results = search(vec, body.topk or 5)
    msg = gemma_recommend(profile_text=body.query, candidates=results)
    return JSONResponse({"recommendation": msg, "results": results})


@app.post("/recommend_with_image")
async def recommend_with_image(
    profile: str = Form(..., description="예: 20대 여성, 1인 가구, 평일 야근 잦음, 산책 주 3회 가능"),
    ref_image: UploadFile = File(...),
    topk: int = Query(5, ge=1, le=20),
):
    data = await ref_image.read()
    pil = Image.open(io.BytesIO(data)).convert("RGB")
    vec = embed_image(pil)
    results = search(vec, topk)
    msg = gemma_recommend(profile_text=profile, candidates=results)

    return JSONResponse(
        {
            "profile": profile,
            "recommendation": msg,
            "count": len(results),
            "results": results,
        }
    )


class SurveyInput(BaseModel):
    living: str
    family: str
    walk_time: str
    dog_size: str
    preferred_personality: str


def survey_to_text(data: SurveyInput) -> str:
    return (
        f"{data.living}에 사는 {data.family}, 하루 산책 {data.walk_time}, "
        f"{data.dog_size} 선호, {data.preferred_personality} 성격 선호"
    )


def combine_embeddings(survey_vec, text_vec=None, img_vec=None):
    weights, vecs = [], []

    if img_vec is not None:
        if survey_vec is not None:
            vecs.append(survey_vec)
            weights.append(0.3)
        if text_vec is not None:
            vecs.append(text_vec)
            weights.append(0.1)
        vecs.append(img_vec)
        weights.append(0.6)
    else:
        if survey_vec is not None:
            vecs.append(survey_vec)
            weights.append(0.7)
        if text_vec is not None:
            vecs.append(text_vec)
            weights.append(0.3)

    weights = np.array(weights) / np.sum(weights)
    return np.average(vecs, axis=0, weights=weights).reshape(1, -1)


async def _handle_recommend(
    survey_obj: SurveyInput,
    extra_text: Optional[str],
    ref_image: Optional[UploadFile],
    topk: int,
):
    survey_text = survey_to_text(survey_obj)
    survey_vec = embed_text(survey_text)
    text_vec = embed_text(extra_text) if extra_text else None

    img_vec = None
    if ref_image:
        img_bytes = await ref_image.read()
        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_vec = embed_image(pil)

    final_vec = combine_embeddings(survey_vec, text_vec, img_vec)
    results = search(final_vec, topk)
    msg = gemma_recommend(profile_text=survey_text, candidates=results)

    return JSONResponse(
        {
            "survey_profile": survey_text,
            "extra_text": extra_text,
            "recommendation": msg,
            "count": len(results),
            "results": results,
        }
    )


@app.post("/recommend_with_survey_form")
async def recommend_with_survey_form(
    survey: str = Form(...),
    extra_text: Optional[str] = Form(None),
    ref_image: Optional[UploadFile] = File(None),
    topk: int = Query(5, ge=1, le=20),
):
    try:
        data = json.loads(survey)
        survey_obj = SurveyInput(**data)
        return await _handle_recommend(survey_obj, extra_text, ref_image, topk)
    except Exception as e:
        import traceback

        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/recommend_with_survey_json")
async def recommend_with_survey_json(
    survey: SurveyInput,
    extra_text: Optional[str] = Body(None),
    topk: int = Query(5, ge=1, le=20),
):
    try:
        return await _handle_recommend(survey, extra_text, None, topk)
    except Exception as e:
        import traceback

        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/chat")
def chat(message: str = Form(...)):
    prompt = f"""
너는 유기견 입양 탐색을 돕는 한국어 상담 도우미야.
사실이 확실하지 않은 내용은 추정하지 말고, 사용자가 공고 기반으로 비교할 수 있게 짧고 명확하게 답해.

[사용자 메시지]
{message}
"""
    reply = gemma_generate_text(
        prompt=prompt,
        max_new_tokens=GEMMA3_CHAT_MAX_NEW_TOKENS,
    )
    return {"reply": reply}
