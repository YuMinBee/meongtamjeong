import io
import html
import hashlib
import json
import os
import shutil
import tempfile
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
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from PIL import Image
from pydantic import BaseModel
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.dog_attributes import build_photo_advice, extract_json_object, summarize_vlm_attrs_ko
from app.hybrid_rag import (
    BM25Index,
    build_hybrid_documents,
    build_search_query_text,
    merge_structured_query,
    parse_structured_query,
    rank_hybrid_documents,
    resolve_photo_quality_score,
    vector_hits_to_doc_modality_scores,
)
from app.graph_rag import build_dog_graph, rerank_with_graph
from app.notice_status import classify_notice, is_searchable_notice, notice_filter_reason
from app.species import clean_species, species_config, species_from_upkind
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

API_KEY = os.getenv("API_KEY", "change-me")
NOTICE_FILTER_INACTIVE = os.getenv("NOTICE_FILTER_INACTIVE", "true").lower() in {"1", "true", "yes", "on"}
NOTICE_INCLUDE_UNKNOWN = os.getenv("NOTICE_INCLUDE_UNKNOWN", "false").lower() in {"1", "true", "yes", "on"}
INDEX_PATH = Path(os.getenv("INDEX_PATH", str(DATA_DIR / "dog_faiss.index")))
METAS_PATH = Path(os.getenv("METAS_PATH", str(DATA_DIR / "dog_metas.json")))
CLIP_MODEL = os.getenv("CLIP_MODEL", "ViT-B/32")
ANIMAL_API_KEY = os.getenv("ANIMAL_API_KEY", "")
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
GEMMA3_MODEL_PATH = os.getenv("GEMMA3_MODEL_PATH", "")
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

def read_faiss_index(path: Path) -> Any:
    try:
        return faiss.read_index(str(path))
    except RuntimeError as exc:
        if "Illegal byte sequence" not in str(exc):
            raise
        safe_dir = Path(tempfile.gettempdir()) / "dog_faiss_safe"
        safe_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
        safe_path = safe_dir / f"{path.stem}-{digest}{path.suffix}"
        if (
            not safe_path.exists()
            or safe_path.stat().st_size != path.stat().st_size
            or safe_path.stat().st_mtime < path.stat().st_mtime
        ):
            shutil.copy2(path, safe_path)
        return faiss.read_index(str(safe_path))


index = read_faiss_index(INDEX_PATH)
with METAS_PATH.open("r", encoding="utf-8") as f:
    METAS: List[Dict[str, Any]] = json.load(f)
if index.ntotal != len(METAS):
    raise ValueError(f"Index({index.ntotal}) != Metas({len(METAS)})")


def load_graph_overlay_metas() -> List[Dict[str, Any]]:
    overlays: List[Dict[str, Any]] = []
    seen = set()
    for path in (DATA_DIR / "local_dog_cache_enriched.json", DATA_DIR / "local_dog_cache.json"):
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            continue
        items = payload.get("items", []) if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            dog_id = str(item.get("desertionNo") or item.get("desertion_no") or "").strip()
            if dog_id and dog_id not in seen:
                overlays.append(item)
                seen.add(dog_id)
    return overlays

HYBRID_DOCS, HYBRID_VECTOR_TO_DOC = build_hybrid_documents(METAS)
HYBRID_BM25 = BM25Index(HYBRID_DOCS)
GRAPH_OVERLAY_METAS = load_graph_overlay_metas()
DOG_GRAPH = build_dog_graph(HYBRID_DOCS, GRAPH_OVERLAY_METAS)


def notice_status_by_doc_index() -> Dict[int, str]:
    return {
        doc_index: classify_notice(doc.get("meta") or {})
        for doc_index, doc in enumerate(HYBRID_DOCS)
    }


def notice_status_counts() -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for status in notice_status_by_doc_index().values():
        counts[status] = counts.get(status, 0) + 1
    return counts


def active_doc_indices() -> List[int]:
    return [
        doc_index
        for doc_index, status in notice_status_by_doc_index().items()
        if status == "active"
    ]


def serving_doc_indices() -> List[int]:
    if not NOTICE_FILTER_INACTIVE:
        return list(range(len(HYBRID_DOCS)))
    return [
        doc_index
        for doc_index, doc in enumerate(HYBRID_DOCS)
        if is_searchable_notice(doc.get("meta") or {}, include_unknown=NOTICE_INCLUDE_UNKNOWN)
    ]


def graph_status_summaries() -> Dict[str, Any]:
    status_by_index = notice_status_by_doc_index()
    serving_indices = serving_doc_indices()
    active_indices = [
        doc_index
        for doc_index, status in status_by_index.items()
        if status == "active"
    ]
    return {
        "total": DOG_GRAPH.summary(),
        "serving": DOG_GRAPH.summary_for_doc_indices(serving_indices),
        "active_only": DOG_GRAPH.summary_for_doc_indices(active_indices),
        "by_notice_status": DOG_GRAPH.summary_by_notice_status(status_by_index),
        "notice_status": notice_status_counts(),
        "hidden_docs": len(HYBRID_DOCS) - len(serving_indices),
        "filters": {
            "filter_inactive": NOTICE_FILTER_INACTIVE,
            "include_unknown": NOTICE_INCLUDE_UNKNOWN,
        },
    }


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
    candidate = Path(GEMMA3_MODEL_PATH).expanduser() if GEMMA3_MODEL_PATH else None
    if candidate and candidate.exists():
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
        clean_text(meta.get("merged_desc"))
        or clean_text(meta.get("vlm_desc"))
        or clean_text(meta.get("desc"))
        or clean_text(meta.get("desc_full"))
        or clean_text(meta.get("specialMark"))
        or summarize_vlm_attrs_ko(meta.get("vlm_attrs"))
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
    for key in ("image_url", "url", "popfile", "popfile1", "popfile2", "fileName", "thumb", "image", "img"):
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


def infer_species(meta: Dict[str, Any]) -> str:
    explicit = clean_text(meta.get("species") or meta.get("animal_species"))
    if explicit:
        return clean_species(explicit)
    upkind = clean_text(meta.get("upkind") or meta.get("upkindCd"))
    if upkind:
        return species_from_upkind(upkind, "dog")
    return "dog"


def resolve_upkind(meta: Dict[str, Any]) -> str:
    upkind = clean_text(meta.get("upkind") or meta.get("upkindCd"))
    if upkind:
        return upkind
    return str(species_config(infer_species(meta))["upkind"])


def resolve_contact_fields(meta: Dict[str, Any]) -> Dict[str, str]:
    return {
        "care_name": clean_text(meta.get("care_name") or meta.get("careNm")),
        "care_tel": clean_text(meta.get("care_tel") or meta.get("careTel")),
        "care_addr": clean_text(meta.get("care_addr") or meta.get("careAddr")),
        "org_name": clean_text(meta.get("org_name") or meta.get("orgNm")),
        "happen_place": clean_text(meta.get("happen_place") or meta.get("happenPlace")),
        "notice_no": clean_text(meta.get("notice_no") or meta.get("noticeNo")),
        "detail_url": resolve_detail_url(meta),
    }


def build_live_animal_meta(rec: Dict[str, Any], species: str = "dog") -> Dict[str, Any]:
    breed_fields = normalize_breed_fields(rec)
    desertion_no = clean_text(rec.get("desertionNo")) or "Unknown"
    notice_no = clean_text(rec.get("noticeNo"))
    upkind = clean_text(rec.get("upkind") or rec.get("upkindCd")) or species_config(species)["upkind"]
    normalized_species = species_from_upkind(upkind, clean_species(species))
    return {
        "type": "live",
        "species": normalized_species,
        "upkind": upkind,
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
        "care_name": clean_text(rec.get("careNm") or rec.get("care_name")),
        "care_tel": clean_text(rec.get("careTel") or rec.get("care_tel")),
        "care_addr": clean_text(rec.get("careAddr") or rec.get("care_addr")),
        "org_name": clean_text(rec.get("orgNm") or rec.get("org_name")),
        "happen_place": clean_text(rec.get("happenPlace") or rec.get("happen_place")),
        "notice_start": clean_text(rec.get("noticeSdt") or rec.get("notice_start")),
        "notice_end": clean_text(rec.get("noticeEdt") or rec.get("notice_end")),
        "process_state": clean_text(rec.get("processState") or rec.get("process_state")),
    }


def build_live_dog_meta(rec: Dict[str, Any]) -> Dict[str, Any]:
    return build_live_animal_meta(rec, species="dog")


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


def _notice_search_window(meta: Dict[str, Any]) -> Tuple[datetime, datetime]:
    end = datetime.now()
    notice_start = parse_api_date(meta.get("notice_start") or meta.get("noticeSdt"))
    if notice_start:
        start = notice_start - timedelta(days=30)
        if start > end:
            start = end - timedelta(days=30)
        return start, end
    return end - timedelta(days=365), end


def fetch_latest_notice_meta(meta: Dict[str, Any], max_pages: int = 30, rows: int = 1000) -> Tuple[Optional[Dict[str, Any]], str]:
    desertion_no = clean_text(meta.get("desertionNo") or meta.get("desertion_no"))
    if not desertion_no:
        return None, "missing_desertion_no"
    if not ANIMAL_API_KEY:
        return None, "missing_api_key"

    preferred_species = infer_species(meta)
    species_order = [preferred_species] + [name for name in ("dog", "cat", "other") if name != preferred_species]
    start, end = _notice_search_window(meta)
    session = build_api_session()

    for species in species_order:
        upkind = species_config(species)["upkind"]
        for page in range(1, max_pages + 1):
            params = {
                "serviceKey": ANIMAL_API_KEY,
                "numOfRows": rows,
                "pageNo": page,
                "_type": "json",
                "upkind": upkind,
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
                if clean_text(rec.get("desertionNo")) == desertion_no:
                    return build_live_animal_meta(rec, species=species), "found"
            if len(page_items) < rows:
                break
    return None, "not_found"


def tel_href(value: str) -> str:
    normalized = "".join(ch for ch in clean_text(value) if ch.isdigit() or ch == "+")
    return f"tel:{normalized}" if normalized else ""


def build_inquiry_script(meta: Dict[str, Any]) -> str:
    contact = resolve_contact_fields(meta)
    breed = resolve_breed_label(meta)
    notice_no = contact.get("notice_no") or clean_text(meta.get("desertionNo"))
    care_name = contact.get("care_name") or "보호소"
    return (
        f"안녕하세요. {care_name} 맞으실까요? "
        f"동물보호관리시스템 공고번호 {notice_no}, {breed} 공고를 보고 문의드립니다. "
        "아직 보호중인지, 방문 가능 시간과 입양 상담 절차를 안내받을 수 있을까요?"
    )


def build_adoption_contact_card(meta: Dict[str, Any], latest_found: bool, lookup_status: str) -> Dict[str, Any]:
    contact = resolve_contact_fields(meta)
    notice_status = classify_notice(meta)
    map_query = contact.get("care_addr") or contact.get("care_name")
    map_url = f"https://map.naver.com/p/search/{quote(map_query)}" if map_query else ""
    return {
        "latest_found": latest_found,
        "lookup_status": lookup_status,
        "connectable": notice_status == "active",
        "notice_status": notice_status,
        "notice_filter_reason": notice_filter_reason(meta),
        "animal": {
            "species": infer_species(meta),
            "upkind": resolve_upkind(meta),
            "desertionNo": clean_text(meta.get("desertionNo")),
            "breed": resolve_breed_label(meta),
            "sex": clean_text(meta.get("sex") or meta.get("sexCd")),
            "age": clean_text(meta.get("age")),
            "notice_start": clean_text(meta.get("notice_start") or meta.get("noticeSdt")),
            "notice_end": clean_text(meta.get("notice_end") or meta.get("noticeEdt")),
            "process_state": clean_text(meta.get("process_state") or meta.get("processState")),
        },
        "contact": contact,
        "actions": {
            "tel": tel_href(contact.get("care_tel", "")),
            "detail_url": contact.get("detail_url", ""),
            "map_url": map_url,
        },
        "inquiry_script": build_inquiry_script(meta),
    }

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
                "visual_attrs": summarize_vlm_attrs_ko(meta.get("vlm_attrs")),
                "photo_advice": meta.get("photo_advice") if isinstance(meta.get("photo_advice"), list) else build_photo_advice(meta.get("vlm_attrs")),
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
            text_meta = m
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
                merged_meta = dict(text_meta)
                merged_meta.update(img_meta)
                for key in ("vlm_attrs", "vlm_attr_text", "photo_advice", "vlm_desc", "merged_desc", "desc_full"):
                    if not merged_meta.get(key) and text_meta.get(key):
                        merged_meta[key] = text_meta.get(key)
                m = merged_meta

        visual_attrs = summarize_vlm_attrs_ko(m.get("vlm_attrs"))
        photo_advice = m.get("photo_advice") if isinstance(m.get("photo_advice"), list) else build_photo_advice(m.get("vlm_attrs"))
        image_attrs = m.get("image_attrs") if isinstance(m.get("image_attrs"), dict) else {}
        photo_quality_score = resolve_photo_quality_score(m)
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
                "visual_attrs": visual_attrs,
                "photo_advice": photo_advice,
                "photo_quality_score": photo_quality_score,
                "image_attrs": image_attrs,
                "crop_path": clean_text(image_attrs.get("crop_path")),
                "embedding_source": clean_text(m.get("embedding_source")),
                "image_url": clean_text(image_url),
                "species": infer_species(m),
                "upkind": resolve_upkind(m),
                **resolve_contact_fields(m),
            }
        )

    return results


def format_hybrid_result(item: Dict[str, Any], rank: int) -> Dict[str, Any]:
    meta = item["doc"].get("meta") or {}
    visual_attrs = summarize_vlm_attrs_ko(meta.get("vlm_attrs"))
    photo_advice = meta.get("photo_advice") if isinstance(meta.get("photo_advice"), list) else build_photo_advice(meta.get("vlm_attrs"))
    image_attrs = meta.get("image_attrs") if isinstance(meta.get("image_attrs"), dict) else {}
    photo_quality_score = resolve_photo_quality_score(meta)
    desertion_no = meta.get("desertionNo", "Unknown")
    image_url = meta.get("image_url") or meta.get("url") or ""
    return {
        "rank": rank,
        "score": float(item["score"]),
        "hybrid_scores": item.get("score_parts", {}),
        "retrieval_evidence": item.get("evidence", {}),
        "desertionNo": clean_text(desertion_no) or "Unknown",
        "breed": resolve_breed_label(meta),
        "breed_code": resolve_breed_code(meta),
        "breed_name": resolve_breed_name(meta),
        "age": clean_text(meta.get("age")) or "Unknown",
        "sex": clean_text(meta.get("sex") or meta.get("sexCd")) or "Unknown",
        "weight": clean_text(meta.get("weight")) or "Unknown",
        "neuter": clean_text(meta.get("neuter") or meta.get("neuterYn")) or "Unknown",
        "desc": resolve_desc(meta),
        "visual_attrs": visual_attrs,
        "photo_advice": photo_advice,
        "photo_quality_score": photo_quality_score,
        "image_attrs": image_attrs,
        "crop_path": clean_text(image_attrs.get("crop_path")),
        "embedding_source": clean_text(meta.get("embedding_source")),
        "notice_status": classify_notice(meta),
        "notice_filter_reason": notice_filter_reason(meta),
        "process_state": clean_text(meta.get("process_state") or meta.get("processState")),
        "notice_start": clean_text(meta.get("notice_start") or meta.get("noticeSdt")),
        "notice_end": clean_text(meta.get("notice_end") or meta.get("noticeEdt")),
        "image_url": clean_text(image_url),
        "species": infer_species(meta),
        "upkind": resolve_upkind(meta),
        **resolve_contact_fields(meta),
    }


def hybrid_search(
    query_text: str,
    structured_query: Dict[str, Any],
    topk: int = 10,
    rerank_depth: int = 80,
    strict_filters: bool = False,
    query_vec: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    topk = max(1, min(int(topk or 10), 50))
    rerank_depth = max(topk, min(int(rerank_depth or 80), max(index.ntotal, topk)))
    search_text = build_search_query_text(query_text, structured_query)
    vec = query_vec if query_vec is not None else embed_text(search_text or query_text)
    vector_search_depth = min(index.ntotal, max(rerank_depth * 3, topk * 30))
    D, I = index.search(vec, vector_search_depth)
    vector_scores, vector_score_details = vector_hits_to_doc_modality_scores(
        D[0].tolist(),
        I[0].tolist(),
        HYBRID_VECTOR_TO_DOC,
        METAS,
    )
    graph_candidate_scores = DOG_GRAPH.candidate_doc_scores(
        structured_query,
        search_text or query_text,
        limit=0,
    )
    candidate_limit = max(topk, min(rerank_depth, topk * 4))
    ranked = rank_hybrid_documents(
        docs=HYBRID_DOCS,
        bm25=HYBRID_BM25,
        structured=structured_query,
        query_text=search_text or query_text,
        vector_scores=vector_scores,
        vector_score_details=vector_score_details,
        topk=candidate_limit,
        rerank_depth=rerank_depth,
        strict_filters=strict_filters,
        extra_candidate_indices=graph_candidate_scores.keys(),
        filter_inactive_notices=NOTICE_FILTER_INACTIVE,
        include_unknown_notices=NOTICE_INCLUDE_UNKNOWN,
    )
    ranked = rerank_with_graph(
        ranked=ranked,
        graph=DOG_GRAPH,
        structured=structured_query,
        query_text=search_text or query_text,
        topk=topk,
    )
    return [format_hybrid_result(item, rank) for rank, item in enumerate(ranked, start=1)]


def parse_query_conditions_with_gemma(raw_query: str, survey_text: str = "", extra_text: str = "") -> Dict[str, Any]:
    prompt = f"""
너는 유기견 입양 검색 조건을 구조화하는 파서야. 아래 사용자 입력을 JSON 객체 하나로 변환해.
허용 키는 coat_color, fur_length, ear_shape, body_size_hint, sex, age_hint, personality, face_visible, whole_body_visible, min_photo_quality, keywords 뿐이야.
배열 필드는 배열로, 모르는 값은 빈 배열 또는 null로 둬. 설명 문장이나 마크다운 없이 JSON만 출력해.

값 예시:
- coat_color: ["white", "brown", "black", "cream", "gray", "spotted"]
- fur_length: ["short", "medium", "long", "curly", "fluffy"]
- ear_shape: ["upright", "floppy", "semi_upright", "folded"]
- body_size_hint: ["tiny", "small", "medium", "large"]
- sex: ["M", "F"]
- age_hint: ["puppy", "adult", "senior"]

[사용자 입력]
{raw_query}

[설문 요약]
{survey_text}

[추가 입력]
{extra_text}
"""
    raw = gemma_generate_text(prompt, max_new_tokens=220)
    return json.loads(extract_json_object(raw))

def gemma_recommend(profile_text: str, candidates: List[Dict[str, Any]]) -> str:
    lines = []
    for c in candidates:
        lines.append(
            {
                "score": round(c["score"], 4),
                "breed": c.get("breed") or c.get("type") or "Unknown",
                "age": c.get("age", "Unknown"),
                "sex": c.get("sex", "Unknown"),
                "weight": c.get("weight", "Unknown"),
                "desc": c.get("desc", ""),
                "visual_attrs": c.get("visual_attrs", ""),
                "photo_advice": c.get("photo_advice", []),
                "photo_quality_score": c.get("photo_quality_score"),
                "hybrid_scores": c.get("hybrid_scores", {}),
                "retrieval_evidence": c.get("retrieval_evidence", {}),
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
- 후보에 적힌 기본 정보, 특징, visual_attrs를 바탕으로
  사용자의 프로필 조건에 맞는 부분을 해석해서 설명을 보완해줘.
- 예를 들어 "산책 주 3회 가능"이라고 하면, 활동량이 많은 품종이나 어린 강아지를 우선 추천해.
- "차분한 성격"이라고 하면, 특징에 '순함', '조용함' 같은 단어가 있으면 강조해.
- retrieval_evidence와 hybrid_scores가 있으면 벡터 검색, 키워드, 조건 매칭 중 어떤 근거가 강했는지 추천 이유에 반영해.
- visual_attrs가 있으면 "흰색 장모", "귀가 선", "얼굴 확인 가능"처럼 사진 기반 근거를 추천 이유에 반영해.
- photo_advice가 있으면 추천과 별도로 "사진 보완" 한 줄을 짧게 덧붙여도 좋아.
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
    graph_summary = graph_status_summaries()
    return {
        "status": "ok",
        "index_size": index.ntotal,
        "hybrid_docs": len(HYBRID_DOCS),
        "device": device,
        "breed_groups": len(BREED_GALLERY),
        "graph": graph_summary["total"],
        "serving_graph": graph_summary["serving"],
        "active_graph": graph_summary["active_only"],
        "graph_by_notice_status": graph_summary["by_notice_status"],
        "hidden_docs": graph_summary["hidden_docs"],
        "notice_status": graph_summary["notice_status"],
        "notice_filter_inactive": NOTICE_FILTER_INACTIVE,
        "notice_include_unknown": NOTICE_INCLUDE_UNKNOWN,
    }



@app.get("/rag/graph")
def rag_graph_summary():
    graph_summary = graph_status_summaries()
    serving_indices = serving_doc_indices()
    active_indices = active_doc_indices()
    return {
        "retrieval": "graph_enhanced_multimodal_rag",
        "summary": graph_summary["total"],
        "serving_summary": graph_summary["serving"],
        "active_summary": graph_summary["active_only"],
        "by_notice_status": graph_summary["by_notice_status"],
        "notice_status": graph_summary["notice_status"],
        "hidden_docs": graph_summary["hidden_docs"],
        "filters": graph_summary["filters"],
        "feature_examples": DOG_GRAPH.labels_for(DOG_GRAPH.feature_to_docs.keys(), limit=24),
        "serving_feature_examples": DOG_GRAPH.top_features_for_doc_indices(serving_indices, limit=16),
        "active_feature_examples": DOG_GRAPH.top_features_for_doc_indices(active_indices, limit=16),
    }


@app.post("/search/text")
def search_text(body: TextQuery):
    structured = parse_structured_query(body.query)
    search_query = build_search_query_text(body.query, structured)
    results = hybrid_search(
        query_text=search_query,
        structured_query=structured,
        topk=body.topk or 5,
        rerank_depth=max((body.topk or 5) * 10, 50),
    )
    return JSONResponse(
        {
            "retrieval": "graph_enhanced_multimodal_rag",
            "query": body.query,
            "structured_query": structured,
            "results": results,
        }
    )


@app.post("/recommend")
def recommend(body: TextQuery):
    structured = parse_structured_query(body.query)
    search_query = build_search_query_text(body.query, structured)
    results = hybrid_search(
        query_text=search_query,
        structured_query=structured,
        topk=body.topk or 5,
        rerank_depth=max((body.topk or 5) * 10, 50),
    )
    msg = gemma_recommend(profile_text=body.query, candidates=results)
    return JSONResponse(
        {
            "retrieval": "graph_enhanced_multimodal_rag",
            "structured_query": structured,
            "recommendation": msg,
            "results": results,
        }
    )


class HybridRagQuery(BaseModel):
    query: str = ""
    survey: Optional[Dict[str, Any]] = None
    extra_text: Optional[str] = None
    topk: Optional[int] = 10
    rerank_depth: Optional[int] = 80
    strict_filters: Optional[bool] = False
    use_llm_parse: Optional[bool] = False
    include_recommendation: Optional[bool] = True


@app.post("/rag/recommend")
def rag_recommend(body: HybridRagQuery):
    survey = body.survey or {}
    survey_text = " ".join(clean_text(value) for value in survey.values()) if isinstance(survey, dict) else ""
    raw_query = " ".join(
        part for part in (clean_text(body.query), survey_text, clean_text(body.extra_text)) if part
    )
    structured = parse_structured_query(raw_query, survey if isinstance(survey, dict) else None)
    parse_source = "rules"
    parse_error = ""
    if body.use_llm_parse:
        try:
            llm_structured = parse_query_conditions_with_gemma(
                raw_query=clean_text(body.query),
                survey_text=survey_text,
                extra_text=clean_text(body.extra_text),
            )
            structured = merge_structured_query(structured, llm_structured)
            parse_source = "llm+rules"
        except Exception as exc:
            parse_error = str(exc)
            parse_source = "rules_fallback"

    search_query = build_search_query_text(raw_query, structured)
    results = hybrid_search(
        query_text=search_query,
        structured_query=structured,
        topk=body.topk or 10,
        rerank_depth=body.rerank_depth or 80,
        strict_filters=bool(body.strict_filters),
    )
    recommendation = ""
    if body.include_recommendation:
        recommendation = gemma_recommend(profile_text=raw_query, candidates=results)

    return JSONResponse(
        {
            "retrieval": "graph_enhanced_multimodal_rag",
            "parse_source": parse_source,
            "parse_error": parse_error,
            "query": raw_query,
            "structured_query": structured,
            "search_query": search_query,
            "recommendation": recommendation,
            "count": len(results),
            "results": results,
        }
    )

class ShelterNoticeDraftInput(BaseModel):
    dog: Dict[str, Any]
    tone: Optional[str] = "따뜻하고 정확한"
    sentence_count: Optional[int] = 4


@app.post("/shelter/notice_draft")
def shelter_notice_draft(body: ShelterNoticeDraftInput):
    dog = body.dog
    visual_attrs = summarize_vlm_attrs_ko(dog.get("vlm_attrs"))
    photo_advice = dog.get("photo_advice") if isinstance(dog.get("photo_advice"), list) else build_photo_advice(dog.get("vlm_attrs"))
    sentence_count = min(5, max(3, int(body.sentence_count or 4)))
    context = {
        "desertionNo": clean_text(dog.get("desertionNo")),
        "breed": clean_text(dog.get("breed") or dog.get("breed_name") or dog.get("breed_code")),
        "sex": clean_text(dog.get("sex")),
        "age": clean_text(dog.get("age")),
        "weight": clean_text(dog.get("weight")),
        "neuter": clean_text(dog.get("neuter")),
        "base_desc": clean_text(dog.get("desc") or dog.get("base_desc") or dog.get("specialMark")),
        "vlm_desc": clean_text(dog.get("vlm_desc")),
        "visual_attrs": visual_attrs,
        "photo_advice": photo_advice,
    }
    prompt = f"""
너는 보호소 공고 작성 보조 도구야. 아래 확인된 정보만 사용해서 입양 공고 문구를 한국어 {sentence_count}문장으로 작성해.

- 사진 기반 속성은 visual_attrs와 vlm_desc에 있는 내용만 사용해.
- 성격, 건강, 훈련 여부, 실제 품종처럼 확인되지 않은 내용은 단정하지 마.
- 과장된 홍보 문구보다 보호자가 실제로 확인할 수 있는 외형과 공고 정보를 정확히 전달해.
- 마지막에 사진 품질 보완이 필요하면 보호소 내부 메모로 한 문장을 따로 적어.
- 번호 목록이나 마크다운 없이 작성해.

[톤]
{body.tone}

[공고 정보 JSON]
{json.dumps(context, ensure_ascii=False)}
"""
    draft = gemma_generate_text(prompt, max_new_tokens=GEMMA3_CHAT_MAX_NEW_TOKENS)
    return JSONResponse(
        {
            "notice_draft": draft,
            "visual_attrs": visual_attrs,
            "photo_advice": photo_advice,
        }
    )


class AdoptionContactInput(BaseModel):
    animal: Optional[Dict[str, Any]] = None
    desertion_no: Optional[str] = None
    species: Optional[str] = "dog"


@app.post("/adoption/contact_card")
def adoption_contact_card(body: AdoptionContactInput):
    base = dict(body.animal or {})
    if body.desertion_no:
        base["desertionNo"] = clean_text(body.desertion_no)
    if body.species and not base.get("species"):
        base["species"] = clean_species(body.species)

    if not clean_text(base.get("desertionNo")):
        raise HTTPException(status_code=400, detail="desertionNo is required")

    latest = None
    lookup_status = "not_started"
    try:
        latest, lookup_status = fetch_latest_notice_meta(base)
    except requests.RequestException as exc:
        lookup_status = f"request_error:{exc.__class__.__name__}"

    merged = dict(base)
    if latest:
        merged.update(latest)
    return JSONResponse(build_adoption_contact_card(merged, latest_found=latest is not None, lookup_status=lookup_status))

@app.post("/recommend_with_image")
async def recommend_with_image(
    profile: str = Form(..., description="예: 20대 여성, 1인 가구, 평일 야근 잦음, 산책 주 3회 가능"),
    ref_image: UploadFile = File(...),
    topk: int = Query(5, ge=1, le=20),
):
    structured = parse_structured_query(profile)
    search_query = build_search_query_text(profile, structured)
    data = await ref_image.read()
    pil = Image.open(io.BytesIO(data)).convert("RGB")
    img_vec = embed_image(pil)
    text_vec = embed_text(search_query) if search_query else None
    final_vec = combine_embeddings(text_vec, None, img_vec) if text_vec is not None else img_vec
    results = hybrid_search(
        query_text=search_query or profile,
        structured_query=structured,
        topk=topk,
        rerank_depth=max(topk * 10, 50),
        query_vec=final_vec,
    )
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
    raw_query = " ".join(part for part in (survey_text, clean_text(extra_text)) if part)
    structured = parse_structured_query(raw_query, survey_obj.model_dump())
    search_query = build_search_query_text(raw_query, structured)
    results = hybrid_search(
        query_text=search_query or raw_query,
        structured_query=structured,
        topk=topk,
        rerank_depth=max(topk * 10, 50),
        query_vec=final_vec,
    )
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


@app.post("/rag/recommend_form")
async def rag_recommend_form(
    query: str = Form(""),
    living: str = Form(""),
    family: str = Form(""),
    walk_time: str = Form(""),
    dog_size: str = Form(""),
    preferred_personality: str = Form(""),
    extra_text: str = Form(""),
    ref_image: Optional[UploadFile] = File(None),
    topk: int = Form(10),
    rerank_depth: int = Form(80),
    strict_filters: bool = Form(False),
    use_llm_parse: bool = Form(False),
    include_recommendation: bool = Form(True),
):
    survey = {
        "living": living,
        "family": family,
        "walk_time": walk_time,
        "dog_size": dog_size,
        "preferred_personality": preferred_personality,
    }
    survey_text = " ".join(clean_text(value) for value in survey.values() if clean_text(value))
    raw_query = " ".join(part for part in (clean_text(query), survey_text, clean_text(extra_text)) if part)
    structured = parse_structured_query(raw_query, survey)
    parse_source = "rules"
    parse_error = ""
    if use_llm_parse:
        try:
            llm_structured = parse_query_conditions_with_gemma(query, survey_text, extra_text)
            structured = merge_structured_query(structured, llm_structured)
            parse_source = "llm+rules"
        except Exception as exc:
            parse_error = str(exc)
            parse_source = "rules_fallback"

    search_query = build_search_query_text(raw_query, structured)
    text_vec = embed_text(search_query) if search_query else None
    img_vec = None
    if ref_image and ref_image.filename:
        img_bytes = await ref_image.read()
        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_vec = embed_image(pil)

    if text_vec is not None and img_vec is not None:
        query_vec = combine_embeddings(text_vec, None, img_vec)
    elif text_vec is not None:
        query_vec = text_vec
    elif img_vec is not None:
        query_vec = img_vec
    else:
        raise HTTPException(status_code=400, detail="query or ref_image is required")

    results = hybrid_search(
        query_text=search_query or raw_query,
        structured_query=structured,
        topk=topk,
        rerank_depth=rerank_depth,
        strict_filters=strict_filters,
        query_vec=query_vec,
    )
    recommendation = gemma_recommend(profile_text=raw_query, candidates=results) if include_recommendation else ""
    return JSONResponse(
        {
            "retrieval": "graph_enhanced_multimodal_rag",
            "parse_source": parse_source,
            "parse_error": parse_error,
            "query": raw_query,
            "structured_query": structured,
            "search_query": search_query,
            "recommendation": recommendation,
            "count": len(results),
            "results": results,
        }
    )





def crop_file_url(crop_path: str, api_key: str) -> str:
    text = clean_text(crop_path).replace("\\", "/")
    if not text:
        return ""
    parts = [part for part in text.split("/") if part]
    try:
        idx = parts.index("image_crops")
        species = parts[idx + 1]
        filename = parts[idx + 2]
    except (ValueError, IndexError):
        return ""
    return f"/visualize/image-crop/{quote(species)}/{quote(filename)}?api_key={quote(api_key)}"


@app.get("/visualize/image-crop/{species}/{filename}")
def image_crop_file(species: str, filename: str):
    species = clean_text(species).lower()
    filename = Path(clean_text(filename)).name
    if species not in {"dog", "cat", "other"} or not filename:
        raise HTTPException(status_code=404, detail="Image not found")
    root = (DATA_DIR / "image_crops").resolve()
    path = (root / species / filename).resolve()
    try:
        valid_path = path.is_relative_to(root)
    except AttributeError:
        valid_path = str(path).startswith(str(root))
    if not valid_path or not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)


def build_image_audit_payload(api_key: str) -> Dict[str, Any]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for vector_index, meta in enumerate(METAS):
        did = clean_text(meta.get("desertionNo")) or f"idx-{vector_index}"
        attrs = meta.get("image_attrs") if isinstance(meta.get("image_attrs"), dict) else {}
        priority = 0
        if attrs:
            priority += 10
        if attrs.get("crop_path"):
            priority += 6
        if meta.get("type") == "image":
            priority += 3
        elif meta.get("type") == "crop_image":
            priority += 2
        elif meta.get("type") == "text":
            priority += 1

        current = by_id.get(did)
        if current is None or priority > current["priority"]:
            merged = dict(current.get("meta") or {}) if current else {}
            merged.update(meta)
            by_id[did] = {"priority": priority, "meta": merged}
        elif current:
            merged = current["meta"]
            for key, value in meta.items():
                if value and not merged.get(key):
                    merged[key] = value
            if attrs and not isinstance(merged.get("image_attrs"), dict):
                merged["image_attrs"] = attrs

    records: List[Dict[str, Any]] = []
    for item in by_id.values():
        meta = item.get("meta") or {}
        attrs = meta.get("image_attrs") if isinstance(meta.get("image_attrs"), dict) else {}
        quality = attrs.get("photo_quality_score")
        crop_quality = attrs.get("crop_photo_quality_score")
        area = attrs.get("dog_area_ratio") if attrs.get("dog_area_ratio") is not None else attrs.get("target_area_ratio")
        bbox_norm = attrs.get("primary_bbox_norm") if isinstance(attrs.get("primary_bbox_norm"), list) else []
        detected = attrs.get("target_detected") is True or attrs.get("dog_detected") is True
        recovered = attrs.get("recovered_by_retry") is True
        primary_confidence = attrs.get("primary_confidence")
        crop_path = clean_text(attrs.get("crop_path"))
        review_flags: List[str] = []
        if not attrs:
            review_flags.append("missing_attrs")
        if recovered:
            review_flags.append("recovered")
        if attrs and not detected:
            review_flags.append("not_detected")
        if recovered and isinstance(primary_confidence, (int, float)) and primary_confidence < 0.25:
            review_flags.append("low_conf_retry")
        if isinstance(quality, (int, float)) and quality < 0.50:
            review_flags.append("low_quality")
        if isinstance(area, (int, float)) and (area < 0.08 or area > 0.75):
            review_flags.append("odd_area")
        if attrs.get("bbox_centered") is False:
            review_flags.append("off_center")
        if not crop_path and detected:
            review_flags.append("missing_crop")

        records.append(
            {
                "desertionNo": clean_text(meta.get("desertionNo")),
                "breed": resolve_breed_label(meta),
                "sex": clean_text(meta.get("sex") or meta.get("sexCd")),
                "age": clean_text(meta.get("age")),
                "weight": clean_text(meta.get("weight")),
                "desc": resolve_desc(meta),
                "care_name": clean_text(meta.get("care_name") or meta.get("careNm")),
                "notice_start": clean_text(meta.get("notice_start") or meta.get("noticeSdt")),
                "notice_end": clean_text(meta.get("notice_end") or meta.get("noticeEdt")),
                "notice_status": classify_notice(meta),
                "image_url": clean_text(meta.get("image_url") or meta.get("url")),
                "detail_url": resolve_detail_url(meta),
                "crop_url": crop_file_url(crop_path, api_key),
                "crop_path": crop_path,
                "target_detected": detected,
                "detection_count": attrs.get("detection_count"),
                "dog_count": attrs.get("dog_count"),
                "photo_quality_score": quality,
                "photo_quality_band": clean_text(attrs.get("photo_quality_band")),
                "crop_photo_quality_score": crop_quality,
                "dog_area_ratio": area,
                "bbox_centered": attrs.get("bbox_centered"),
                "primary_confidence": primary_confidence,
                "full_crop_clip_similarity": attrs.get("full_crop_clip_similarity"),
                "recovered_by_retry": recovered,
                "retry_conf": attrs.get("retry_conf"),
                "retry_imgsz": attrs.get("retry_imgsz"),
                "bbox_norm": bbox_norm,
                "review_flags": review_flags,
            }
        )

    def as_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def review_priority(record: Dict[str, Any]) -> Tuple[int, float, str]:
        flags = set(record.get("review_flags") or [])
        if "not_detected" in flags:
            bucket = 0
        elif "low_quality" in flags:
            bucket = 1
        elif "odd_area" in flags or "off_center" in flags:
            bucket = 2
        else:
            bucket = 3
        return (bucket, -as_float(record.get("photo_quality_score")), clean_text(record.get("desertionNo")))

    records.sort(key=review_priority)
    total = len(records)
    detected_count = sum(1 for record in records if record.get("target_detected"))
    crop_count = sum(1 for record in records if record.get("crop_url"))
    scored_count = sum(1 for record in records if isinstance(record.get("photo_quality_score"), (int, float)))
    low_quality_count = sum(1 for record in records if "low_quality" in (record.get("review_flags") or []))
    odd_area_count = sum(1 for record in records if "odd_area" in (record.get("review_flags") or []))
    recovered_count = sum(1 for record in records if record.get("recovered_by_retry"))
    low_conf_retry_count = sum(1 for record in records if "low_conf_retry" in (record.get("review_flags") or []))
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "records": records,
        "summary": {
            "total": total,
            "detected": detected_count,
            "not_detected": total - detected_count,
            "crops": crop_count,
            "scored": scored_count,
            "low_quality": low_quality_count,
            "odd_area": odd_area_count,
            "recovered": recovered_count,
            "low_conf_retry": low_conf_retry_count,
            "detected_rate": round(detected_count / total, 4) if total else 0,
            "crop_rate": round(crop_count / total, 4) if total else 0,
            "quality_coverage": round(scored_count / total, 4) if total else 0,
        },
    }


@app.get("/visualize/image-audit", response_class=HTMLResponse)
def image_audit_ui(request: Request):
    api_key = request.query_params.get("api_key", "")
    payload_json = json.dumps(build_image_audit_payload(api_key), ensure_ascii=False).replace("</", "<\\/")
    page = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Image Audit</title>
  <style>
    :root { --bg:#f7f8f6; --panel:#fff; --ink:#17211d; --muted:#65736d; --line:#d9e0dc; --soft:#eef3f0; --accent:#1f6b4b; --bad:#a33a31; --warn:#956100; --good:#28724d; }
    * { box-sizing:border-box; }
    body { margin:0; font-family:system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:var(--bg); color:var(--ink); }
    main { width:min(1600px, calc(100vw - 28px)); margin:0 auto; padding:18px 0 34px; }
    header { display:grid; grid-template-columns:1fr auto; gap:14px; align-items:end; border-bottom:1px solid var(--line); padding-bottom:14px; }
    h1 { margin:0; font-size:1.35rem; letter-spacing:0; }
    h2 { margin:0; font-size:.98rem; letter-spacing:0; }
    a { color:var(--accent); text-decoration:none; }
    .muted { color:var(--muted); font-size:.86rem; }
    .toolbar { display:grid; grid-template-columns:180px 170px 140px minmax(180px, 280px); gap:10px; align-items:end; margin:14px 0; }
    label { display:grid; gap:5px; color:var(--muted); font-size:.8rem; }
    select, input { border:1px solid var(--line); border-radius:6px; background:white; color:var(--ink); padding:9px 10px; font:inherit; min-height:38px; }
    .metrics { display:grid; grid-template-columns:repeat(auto-fit, minmax(128px, 1fr)); gap:10px; margin:14px 0; }
    .metric { background:white; border:1px solid var(--line); border-radius:8px; padding:10px; min-height:70px; display:grid; align-content:center; gap:4px; }
    .metric span { color:var(--muted); font-size:.76rem; }
    .metric strong { font-size:1.06rem; }
    .cards { display:grid; grid-template-columns:repeat(auto-fill, minmax(430px, 1fr)); gap:14px; }
    .card { background:var(--panel); border:1px solid var(--line); border-radius:8px; overflow:hidden; display:grid; grid-template-rows:auto 1fr; }
    .media { display:grid; grid-template-columns:1fr 1fr; gap:1px; background:var(--line); }
    figure { margin:0; background:var(--soft); min-height:230px; position:relative; display:grid; align-items:center; justify-items:center; overflow:hidden; }
    figure img { width:100%; height:auto; display:block; background:var(--soft); }
    figure.crop img { height:100%; min-height:230px; object-fit:contain; }
    figcaption { position:absolute; left:8px; top:8px; background:rgba(255,255,255,.92); border:1px solid var(--line); border-radius:999px; padding:3px 8px; font-size:.72rem; color:var(--ink); }
    .bbox { position:absolute; border:2px solid #20b26b; box-shadow:0 0 0 1px rgba(0,0,0,.18); pointer-events:none; }
    .body { padding:12px; display:grid; gap:9px; }
    .title { display:flex; justify-content:space-between; gap:8px; align-items:start; }
    .title strong { overflow-wrap:anywhere; }
    .chips { display:flex; gap:6px; flex-wrap:wrap; }
    .chip { border:1px solid var(--line); border-radius:999px; padding:3px 8px; font-size:.74rem; background:#fbfcfb; color:var(--ink); }
    .chip.good { color:var(--good); background:#f2f8f4; border-color:#b9d8c8; }
    .chip.warn { color:var(--warn); background:#fff8e8; border-color:#e1c889; }
    .chip.bad { color:var(--bad); background:#fff1ef; border-color:#e0b6b0; }
    .desc { color:#35433d; font-size:.86rem; line-height:1.42; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; }
    .empty { border:1px dashed var(--line); color:var(--muted); min-height:230px; display:grid; place-items:center; padding:20px; text-align:center; }
    @media (max-width:900px) { .toolbar { grid-template-columns:1fr 1fr; } .cards { grid-template-columns:1fr; } }
    @media (max-width:620px) { main { width:min(100vw - 18px, 1600px); } header, .toolbar, .media { grid-template-columns:1fr; } }
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>Image Audit</h1>
      <div class="muted" id="status"></div>
    </div>
    <a href="/visualize/dashboard?api_key=__API_KEY__">Dashboard</a>
  </header>
  <section class="metrics" id="metrics"></section>
  <section class="toolbar">
    <label>Filter<select id="filter">
      <option value="review">Needs review</option>
      <option value="all">All</option>
      <option value="detected">Detected</option>
      <option value="failed">Not detected</option>
      <option value="recovered">Recovered retry</option>
      <option value="low_conf_retry">Low confidence retry</option>
      <option value="low">Low quality</option>
      <option value="odd">Odd area/off center</option>
      <option value="high">High quality</option>
    </select></label>
    <label>Sort<select id="sort">
      <option value="review">Review priority</option>
      <option value="quality_desc">Quality high</option>
      <option value="quality_asc">Quality low</option>
      <option value="area_desc">Area large</option>
      <option value="area_asc">Area small</option>
      <option value="confidence_asc">Confidence low</option>
    </select></label>
    <label>Limit<select id="limit">
      <option>30</option><option selected>60</option><option>120</option><option>240</option><option value="all">All</option>
    </select></label>
    <label>Search<input id="query" placeholder="notice, shelter, breed" /></label>
  </section>
  <section class="cards" id="cards"></section>
</main>
<script>
const audit = __PAYLOAD__;
const records = audit.records || [];
const metricsEl = document.getElementById('metrics');
const cardsEl = document.getElementById('cards');
const statusEl = document.getElementById('status');
const controls = ['filter','sort','limit','query'].map(id => document.getElementById(id));
function esc(value) { return String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch])); }
function num(value, digits=3) { return Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : '-'; }
function metric(label, value) { return `<div class="metric"><span>${esc(label)}</span><strong>${esc(value)}</strong></div>`; }
function hasFlag(record, name) { return (record.review_flags || []).includes(name); }
function renderMetrics() {
  const s = audit.summary || {};
  metricsEl.innerHTML = [
    metric('Total', s.total || 0),
    metric('Detected', `${s.detected || 0} (${num((s.detected_rate || 0) * 100, 1)}%)`),
    metric('Not detected', s.not_detected || 0),
    metric('Crops', `${s.crops || 0} (${num((s.crop_rate || 0) * 100, 1)}%)`),
    metric('Quality scored', `${s.scored || 0} (${num((s.quality_coverage || 0) * 100, 1)}%)`),
    metric('Recovered', s.recovered || 0),
    metric('Low conf retry', s.low_conf_retry || 0),
    metric('Low quality', s.low_quality || 0),
    metric('Odd area', s.odd_area || 0),
    metric('Generated', audit.generated_at || ''),
  ].join('');
}
function filteredRecords() {
  const filter = document.getElementById('filter').value;
  const query = document.getElementById('query').value.trim().toLowerCase();
  let items = records.filter(record => {
    if (filter === 'review' && !(record.review_flags || []).length) return false;
    if (filter === 'detected' && !record.target_detected) return false;
    if (filter === 'failed' && record.target_detected) return false;
    if (filter === 'recovered' && !record.recovered_by_retry) return false;
    if (filter === 'low_conf_retry' && !hasFlag(record, 'low_conf_retry')) return false;
    if (filter === 'low' && !hasFlag(record, 'low_quality')) return false;
    if (filter === 'odd' && !(hasFlag(record, 'odd_area') || hasFlag(record, 'off_center'))) return false;
    if (filter === 'high' && Number(record.photo_quality_score || 0) < 0.72) return false;
    if (!query) return true;
    const haystack = [record.desertionNo, record.breed, record.care_name, record.desc, record.notice_status].join(' ').toLowerCase();
    return haystack.includes(query);
  });
  const sort = document.getElementById('sort').value;
  const value = (record, key) => Number(record[key] ?? -1);
  items = [...items].sort((a, b) => {
    if (sort === 'quality_desc') return value(b, 'photo_quality_score') - value(a, 'photo_quality_score');
    if (sort === 'quality_asc') return value(a, 'photo_quality_score') - value(b, 'photo_quality_score');
    if (sort === 'area_desc') return value(b, 'dog_area_ratio') - value(a, 'dog_area_ratio');
    if (sort === 'area_asc') return value(a, 'dog_area_ratio') - value(b, 'dog_area_ratio');
    if (sort === 'confidence_asc') return value(a, 'primary_confidence') - value(b, 'primary_confidence');
    return 0;
  });
  const limit = document.getElementById('limit').value;
  return limit === 'all' ? items : items.slice(0, Number(limit));
}
function chipClass(flag) {
  if (['not_detected','low_quality','missing_attrs','missing_crop'].includes(flag)) return 'bad';
  if (['odd_area','off_center','low_conf_retry'].includes(flag)) return 'warn';
  return 'good';
}
function bbox(record) {
  const box = record.bbox_norm || [];
  if (box.length !== 4 || !record.target_detected) return '';
  const [x1, y1, x2, y2] = box.map(Number);
  return `<span class="bbox" style="left:${x1 * 100}%;top:${y1 * 100}%;width:${(x2 - x1) * 100}%;height:${(y2 - y1) * 100}%"></span>`;
}
function flags(record) {
  const values = record.review_flags?.length ? record.review_flags : ['ok'];
  return values.map(flag => `<span class="chip ${chipClass(flag)}">${esc(flag)}</span>`).join('');
}
function card(record) {
  const qualityClass = Number(record.photo_quality_score || 0) >= 0.72 ? 'good' : Number(record.photo_quality_score || 0) < 0.5 ? 'bad' : 'warn';
  return `<article class="card">
    <div class="media">
      <figure>
        ${record.image_url ? `<img src="${esc(record.image_url)}" loading="lazy" alt="">${bbox(record)}` : '<div class="empty">No original image</div>'}
        <figcaption>original</figcaption>
      </figure>
      <figure class="crop">
        ${record.crop_url ? `<img src="${esc(record.crop_url)}" loading="lazy" alt="">` : '<div class="empty">No crop</div>'}
        <figcaption>crop</figcaption>
      </figure>
    </div>
    <div class="body">
      <div class="title"><strong>${esc(record.desertionNo || 'unknown')}</strong><a href="${esc(record.detail_url || '#')}" target="_blank" rel="noreferrer">detail</a></div>
      <div class="muted">${esc(record.breed)} / ${esc(record.sex)} / ${esc(record.age)} / ${esc(record.weight)}</div>
      <div class="chips">
        <span class="chip ${record.target_detected ? 'good' : 'bad'}">detected ${esc(record.target_detected)}</span>
        <span class="chip ${qualityClass}">q ${num(record.photo_quality_score)}</span>
        <span class="chip">crop q ${num(record.crop_photo_quality_score)}</span>
        <span class="chip">area ${num(record.dog_area_ratio)}</span>
        <span class="chip">conf ${num(record.primary_confidence)}</span>
        <span class="chip">sim ${num(record.full_crop_clip_similarity)}</span>
        <span class="chip">center ${esc(record.bbox_centered)}</span>
      </div>
      <div class="chips">${flags(record)}</div>
      <div class="desc">${esc(record.desc)}</div>
      <div class="muted">${esc(record.care_name)} · ${esc(record.notice_status)} · ${esc(record.notice_start)}-${esc(record.notice_end)}</div>
    </div>
  </article>`;
}
function render() {
  const items = filteredRecords();
  cardsEl.innerHTML = items.map(card).join('');
  statusEl.textContent = `${items.length} shown from ${records.length}`;
}
controls.forEach(control => control.addEventListener('input', render));
renderMetrics();
render();
</script>
</body>
</html>
""".replace("__PAYLOAD__", payload_json).replace("__API_KEY__", quote(api_key))
    return HTMLResponse(page)
@app.get("/visualize/dashboard", response_class=HTMLResponse)
def feature_dashboard_ui(request: Request):
    api_key_json = json.dumps(request.query_params.get("api_key", ""))
    page = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Dog RAG Dashboard</title>
  <style>
    :root {
      --bg:#f7f8f6; --panel:#ffffff; --ink:#17211d; --muted:#66736d;
      --line:#d9e0dc; --soft:#eef3f0; --accent:#1f6b4b; --accent-2:#2b5f86;
      --warn:#9a5b00; --bad:#a33a31; --good:#26724d;
    }
    * { box-sizing:border-box; }
    body { margin:0; font-family:system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:var(--bg); color:var(--ink); }
    main { width:min(1440px, calc(100vw - 32px)); margin:0 auto; padding:18px 0 34px; }
    header { display:grid; grid-template-columns:1fr minmax(240px, 360px); gap:16px; align-items:end; padding-bottom:12px; border-bottom:1px solid var(--line); }
    h1 { margin:0; font-size:1.35rem; letter-spacing:0; }
    h2 { margin:0; font-size:1rem; letter-spacing:0; }
    h3 { margin:0; font-size:.98rem; letter-spacing:0; }
    label { display:grid; gap:6px; font-size:.83rem; color:var(--muted); }
    input, textarea, select { width:100%; border:1px solid var(--line); border-radius:6px; background:white; color:var(--ink); font:inherit; padding:9px 10px; }
    textarea { min-height:96px; resize:vertical; line-height:1.45; }
    button { border:0; border-radius:6px; background:var(--accent); color:white; font:inherit; font-weight:700; padding:10px 12px; cursor:pointer; min-height:40px; }
    button.secondary { background:var(--accent-2); }
    button.ghost { background:white; color:var(--ink); border:1px solid var(--line); }
    button:disabled { opacity:.58; cursor:wait; }
    .toolbar { display:grid; grid-template-columns:1fr auto; gap:10px; align-items:end; }
    .tabs { display:flex; flex-wrap:wrap; gap:8px; margin:16px 0; }
    .tabs button { background:white; color:var(--ink); border:1px solid var(--line); }
    .tabs button.active { background:var(--ink); color:white; border-color:var(--ink); }
    .grid { display:grid; grid-template-columns:390px 1fr; gap:16px; align-items:start; }
    .panel { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; }
    .section { display:grid; gap:12px; }
    .row { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
    .checks { display:flex; gap:12px; flex-wrap:wrap; align-items:center; }
    .checks label { display:flex; flex-direction:row; gap:7px; align-items:center; color:var(--ink); margin:0; }
    .checks input { width:auto; }
    .metrics { display:grid; grid-template-columns:repeat(auto-fit, minmax(130px, 1fr)); gap:10px; margin-bottom:16px; }
    .metric { background:white; border:1px solid var(--line); border-radius:8px; padding:10px; min-height:74px; display:grid; align-content:center; gap:4px; }
    .metric span { color:var(--muted); font-size:.78rem; }
    .metric strong { font-size:1.08rem; overflow-wrap:anywhere; }
    .output { display:grid; gap:14px; }
    .summary { background:var(--soft); border:1px solid var(--line); border-radius:8px; padding:12px; white-space:pre-wrap; line-height:1.5; min-height:72px; }
    .results { display:grid; grid-template-columns:repeat(auto-fill, minmax(280px, 1fr)); gap:12px; }
    .dog { background:white; border:1px solid var(--line); border-radius:8px; overflow:hidden; display:grid; grid-template-rows:auto 1fr; min-width:0; }
    .dog.selected { border-color:var(--accent); box-shadow:0 0 0 2px rgba(31,107,75,.14); }
    .dog img { width:100%; aspect-ratio:4/3; object-fit:cover; background:var(--soft); display:block; }
    .dog-body { padding:12px; display:grid; gap:8px; }
    .meta { color:var(--muted); font-size:.84rem; line-height:1.4; }
    .desc { line-height:1.45; font-size:.9rem; }
    .chips { display:flex; flex-wrap:wrap; gap:6px; }
    .chip { border:1px solid var(--line); border-radius:999px; padding:3px 8px; font-size:.76rem; background:#fbfcfb; color:var(--ink); }
    .chip.good { color:var(--good); border-color:#bad8c8; background:#f2f8f4; }
    .chip.warn { color:var(--warn); border-color:#e1c889; background:#fff8e8; }
    .mono { font-family:ui-monospace, SFMono-Regular, Consolas, monospace; font-size:.8rem; overflow:auto; white-space:pre-wrap; max-height:360px; }
    .split { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
    .hidden { display:none; }
    @media (max-width:1100px) { .grid { grid-template-columns:1fr; } .metrics { grid-template-columns:repeat(3, 1fr); } }
    @media (max-width:720px) { main { width:min(100vw - 20px, 1440px); } header, .row, .split { grid-template-columns:1fr; } .metrics { grid-template-columns:repeat(2, 1fr); } }
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>Dog RAG Dashboard</h1>
      <div class="meta" id="status">Idle</div>
    </div>
    <div class="toolbar">
      <label>API key<input id="apiKey" autocomplete="off" /></label>
      <button id="loadBtn" class="secondary" type="button">Load</button>
    </div>
  </header>

  <nav class="tabs" aria-label="Dashboard views">
    <button class="active" type="button" data-view="searchView">Search</button>
    <button type="button" data-view="systemView">System</button>
    <button type="button" data-view="noticeView">Notice</button>
    <button type="button" data-view="jsonView">JSON</button>
  </nav>

  <section class="metrics" id="metrics"></section>

  <section id="searchView" class="grid view">
    <form id="searchForm" class="panel section">
      <h2>Retrieval</h2>
      <label>Query<textarea name="query">small calm dog, clear face photo, good for a first-time adopter</textarea></label>
      <div class="row">
        <label>Living<input name="living" placeholder="apartment" /></label>
        <label>Family<input name="family" placeholder="single person, child" /></label>
        <label>Walk time<input name="walk_time" placeholder="30 minutes daily" /></label>
        <label>Preferred size<input name="dog_size" placeholder="small" /></label>
      </div>
      <label>Preferred temperament<input name="preferred_personality" placeholder="calm, friendly" /></label>
      <label>Extra text<textarea name="extra_text" placeholder="Any additional adoption condition"></textarea></label>
      <label>Reference image<input name="ref_image" type="file" accept="image/*" /></label>
      <div class="row">
        <label>Top K<input name="topk" type="number" min="1" max="30" value="8" /></label>
        <label>Rerank depth<input name="rerank_depth" type="number" min="20" max="300" value="300" /></label>
      </div>
      <div class="checks">
        <label><input name="strict_filters" type="checkbox" /> Strict</label>
        <label><input name="use_llm_parse" type="checkbox" /> LLM parse</label>
        <label><input name="include_recommendation" type="checkbox" /> Gemma text</label>
      </div>
      <div class="row">
        <button id="searchBtn" type="submit">Run Search</button>
        <button id="recommendBtn" class="secondary" type="button">Run Recommendation</button>
      </div>
    </form>
    <section class="output">
      <div class="summary" id="summary">No results yet.</div>
      <div class="results" id="results"></div>
    </section>
  </section>

  <section id="systemView" class="view hidden">
    <div class="split">
      <div class="panel section"><h2>Health</h2><pre class="mono" id="healthJson">{}</pre></div>
      <div class="panel section"><h2>Graph</h2><pre class="mono" id="graphJson">{}</pre></div>
    </div>
  </section>

  <section id="noticeView" class="view hidden">
    <div class="grid">
      <div class="panel section">
        <h2>Notice Draft</h2>
        <label>Selected dog<input id="selectedDog" readonly /></label>
        <label>Tone<input id="noticeTone" value="warm and factual" /></label>
        <label>Sentences<input id="sentenceCount" type="number" min="3" max="5" value="4" /></label>
        <button id="noticeBtn" type="button">Generate Draft</button>
        <button id="contactBtn" class="secondary" type="button">Contact Card</button>
      </div>
      <div class="section">
        <div class="summary" id="noticeOutput">Select a result first.</div>
        <div class="summary" id="contactOutput">Select a result first.</div>
      </div>
    </div>
  </section>

  <section id="jsonView" class="view hidden">
    <div class="panel section"><h2>Last Response</h2><pre class="mono" id="lastJson">{}</pre></div>
  </section>
</main>

<script>
const initialApiKey = __API_KEY__;
const apiKeyInput = document.getElementById('apiKey');
const statusEl = document.getElementById('status');
const metricsEl = document.getElementById('metrics');
const healthJson = document.getElementById('healthJson');
const graphJson = document.getElementById('graphJson');
const summaryEl = document.getElementById('summary');
const resultsEl = document.getElementById('results');
const lastJson = document.getElementById('lastJson');
const selectedDog = document.getElementById('selectedDog');
const noticeOutput = document.getElementById('noticeOutput');
const contactOutput = document.getElementById('contactOutput');
let lastResponse = null;
let selected = null;
apiKeyInput.value = initialApiKey || localStorage.getItem('dogDashboardApiKey') || '';

function key() { return apiKeyInput.value.trim(); }
function setStatus(text) { statusEl.textContent = text; }
function escapeHtml(value) { return String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch])); }
function pretty(value) { return JSON.stringify(value || {}, null, 2); }
function metric(label, value) { return `<div class="metric"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`; }
function numberLabel(value) { return Number(value || 0).toLocaleString('ko-KR'); }
async function fetchJson(url, options = {}) {
  const headers = new Headers(options.headers || {});
  headers.set('x-api-key', key());
  const res = await fetch(url, {...options, headers});
  const payload = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(payload.detail || payload.error || res.statusText);
  return payload;
}
async function loadSystem() {
  localStorage.setItem('dogDashboardApiKey', key());
  setStatus('Loading system');
  const [health, graph] = await Promise.all([fetchJson('/health'), fetchJson('/rag/graph')]);
  healthJson.textContent = pretty(health);
  graphJson.textContent = pretty(graph);
  const totalGraph = graph.summary || {};
  const servingGraph = graph.serving_summary || health.serving_graph || {};
  const activeGraph = graph.active_summary || health.active_graph || {};
  const notice = graph.notice_status || health.notice_status || {};
  const hiddenDocs = graph.hidden_docs ?? health.hidden_docs ?? 0;
  metricsEl.innerHTML = [
    metric('Status', health.status || 'unknown'),
    metric('Vectors', numberLabel(health.index_size)),
    metric('Serving docs', numberLabel(servingGraph.dogs || activeGraph.dogs || notice.active)),
    metric('Hidden docs', numberLabel(hiddenDocs)),
    metric('Serving edges', numberLabel(servingGraph.edges)),
    metric('Total edges', numberLabel(totalGraph.edges)),
    metric('Active notices', numberLabel(notice.active)),
    metric('Unknown notices', numberLabel(notice.unknown)),
    metric('Device', health.device || 'unknown'),
  ].join('');
  setStatus('Ready');
}
function buildFormData(includeRecommendation) {
  const form = document.getElementById('searchForm');
  const data = new FormData(form);
  data.set('include_recommendation', includeRecommendation ? 'true' : (data.get('include_recommendation') ? 'true' : 'false'));
  if (!data.get('ref_image') || !data.get('ref_image').name) data.delete('ref_image');
  return data;
}
function renderResponse(payload) {
  lastResponse = payload;
  lastJson.textContent = pretty(payload);
  summaryEl.textContent = [
    payload.retrieval ? `retrieval: ${payload.retrieval}` : '',
    payload.parse_source ? `parse: ${payload.parse_source}` : '',
    payload.search_query ? `search: ${payload.search_query}` : '',
    payload.recommendation || '',
  ].filter(Boolean).join('\n\n') || 'Done.';
  const results = payload.results || [];
  resultsEl.innerHTML = results.map((item, index) => renderDog(item, index)).join('');
  resultsEl.querySelectorAll('[data-index]').forEach(button => {
    button.addEventListener('click', () => selectResult(Number(button.dataset.index)));
  });
  setStatus(`${results.length} results`);
}
function scoreChips(scores) {
  return Object.entries(scores || {}).map(([name, value]) => `<span class="chip">${escapeHtml(name)} ${Number(value || 0).toFixed(3)}</span>`).join('');
}
function evidenceChips(evidence) {
  const values = [];
  for (const key of ['matched_conditions', 'graph_matched_edges', 'graph_similar_to_edges']) {
    for (const item of (evidence?.[key] || []).slice(0, 4)) values.push(item);
  }
  return values.map(item => `<span class="chip good">${escapeHtml(item)}</span>`).join('');
}
function renderDog(item, index) {
  const evidence = item.retrieval_evidence || {};
  const advice = (item.photo_advice || []).slice(0, 2).map(text => `<span class="chip warn">${escapeHtml(text)}</span>`).join('');
  return `<article class="dog ${selected === item ? 'selected' : ''}">
    ${item.image_url ? `<img src="${escapeHtml(item.image_url)}" alt="">` : '<div></div>'}
    <div class="dog-body">
      <h3>${escapeHtml(item.rank)}. ${escapeHtml(item.breed)}</h3>
      <div class="meta">${escapeHtml(item.sex)} / ${escapeHtml(item.age)} / ${escapeHtml(item.weight)} / ${escapeHtml(item.notice_status || "unknown")}</div>
      <div class="desc">${escapeHtml(item.desc).slice(0, 220)}</div>
      <div class="chips">${scoreChips(item.hybrid_scores)}</div>
      <div class="chips">${evidenceChips(evidence)}</div>
      ${item.visual_attrs ? `<div class="meta">${escapeHtml(item.visual_attrs)}</div>` : ''}
      ${item.care_name ? `<div class="meta">${escapeHtml(item.care_name)}${item.care_tel ? ' / ' + escapeHtml(item.care_tel) : ''}</div>` : ''}
      ${advice ? `<div class="chips">${advice}</div>` : ''}
      <button class="ghost" type="button" data-index="${index}">Select</button>
    </div>
  </article>`;
}
function selectResult(index) {
  selected = lastResponse?.results?.[index] || null;
  selectedDog.value = selected ? `${selected.desertionNo || ''} ${selected.breed || ''}`.trim() : '';
  noticeOutput.textContent = selected ? 'Ready.' : 'Select a result first.';
  contactOutput.textContent = selected ? 'Ready.' : 'Select a result first.';
  document.querySelectorAll('.dog').forEach((node, i) => node.classList.toggle('selected', i === index));
}
function renderContactCard(payload) {
  const animal = payload.animal || {};
  const contact = payload.contact || {};
  const actions = payload.actions || {};
  const links = [
    actions.tel ? `<a href="${escapeHtml(actions.tel)}">Call</a>` : '',
    actions.detail_url ? `<a href="${escapeHtml(actions.detail_url)}" target="_blank" rel="noreferrer">Detail</a>` : '',
    actions.map_url ? `<a href="${escapeHtml(actions.map_url)}" target="_blank" rel="noreferrer">Map</a>` : '',
  ].filter(Boolean).join(' · ');
  return `<div class="section">
    <div><strong>${escapeHtml(animal.breed || '')}</strong> <span class="chip ${payload.connectable ? 'good' : 'warn'}">${escapeHtml(payload.notice_status || 'unknown')}</span></div>
    <div class="meta">${escapeHtml(contact.care_name || '')}${contact.care_tel ? ' / ' + escapeHtml(contact.care_tel) : ''}</div>
    <div class="meta">${escapeHtml(contact.care_addr || '')}</div>
    <div class="meta">${escapeHtml(contact.notice_no || animal.desertionNo || '')}</div>
    <div class="chips">${links}</div>
    <div class="summary">${escapeHtml(payload.inquiry_script || '')}</div>
  </div>`;
}
async function runSearch(includeRecommendation) {
  setStatus(includeRecommendation ? 'Running recommendation' : 'Running search');
  const payload = await fetchJson('/rag/recommend_form', { method:'POST', body:buildFormData(includeRecommendation) });
  renderResponse(payload);
}

document.getElementById('loadBtn').addEventListener('click', () => loadSystem().catch(err => { setStatus('Error'); healthJson.textContent = String(err); }));
document.getElementById('searchForm').addEventListener('submit', event => { event.preventDefault(); runSearch(false).catch(err => { setStatus('Error'); summaryEl.textContent = String(err); }); });
document.getElementById('recommendBtn').addEventListener('click', () => runSearch(true).catch(err => { setStatus('Error'); summaryEl.textContent = String(err); }));
document.getElementById('contactBtn').addEventListener('click', async () => {
  if (!selected) { contactOutput.textContent = 'Select a result first.'; return; }
  setStatus('Checking contact');
  contactOutput.textContent = 'Working...';
  try {
    const payload = await fetchJson('/adoption/contact_card', {
      method:'POST',
      headers:{'content-type':'application/json'},
      body:JSON.stringify({animal:selected})
    });
    contactOutput.innerHTML = renderContactCard(payload);
    setStatus('Contact ready');
  } catch (err) { contactOutput.textContent = String(err); setStatus('Error'); }
});document.getElementById('noticeBtn').addEventListener('click', async () => {
  if (!selected) { noticeOutput.textContent = 'Select a result first.'; return; }
  setStatus('Generating notice');
  noticeOutput.textContent = 'Working...';
  try {
    const payload = await fetchJson('/shelter/notice_draft', {
      method:'POST',
      headers:{'content-type':'application/json'},
      body:JSON.stringify({dog:selected, tone:document.getElementById('noticeTone').value, sentence_count:Number(document.getElementById('sentenceCount').value || 4)})
    });
    noticeOutput.textContent = payload.notice_draft || pretty(payload);
    setStatus('Notice ready');
  } catch (err) { noticeOutput.textContent = String(err); setStatus('Error'); }
});
document.querySelectorAll('.tabs button').forEach(button => {
  button.addEventListener('click', () => {
    document.querySelectorAll('.tabs button').forEach(node => node.classList.remove('active'));
    document.querySelectorAll('.view').forEach(node => node.classList.add('hidden'));
    button.classList.add('active');
    document.getElementById(button.dataset.view).classList.remove('hidden');
  });
});
loadSystem().catch(err => { setStatus('Error'); healthJson.textContent = String(err); });
</script>
</body>
</html>
""".replace("__API_KEY__", api_key_json)
    return HTMLResponse(page)
@app.get("/visualize/adoption-flow", response_class=HTMLResponse)
def adoption_flow_ui(request: Request):
    api_key_json = json.dumps(request.query_params.get("api_key", ""))
    page = """
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Adoption Flow</title>
  <style>
    :root { --ink:#17211b; --muted:#66736c; --line:#d9e2dc; --soft:#f4f7f5; --accent:#236b4b; --warn:#8a5b00; }
    * { box-sizing:border-box; }
    body { margin:0; font-family:system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color:var(--ink); background:#fbfcfb; }
    main { width:min(1180px, calc(100vw - 32px)); margin:0 auto; padding:24px 0 40px; }
    header { display:flex; justify-content:space-between; align-items:flex-end; gap:16px; border-bottom:1px solid var(--line); padding-bottom:14px; }
    h1 { margin:0; font-size:1.4rem; letter-spacing:0; }
    .status { color:var(--muted); font-size:.92rem; }
    form { display:grid; grid-template-columns:1.1fr .9fr; gap:18px; margin-top:20px; align-items:start; }
    fieldset { border:1px solid var(--line); background:white; border-radius:8px; padding:16px; margin:0; }
    legend { padding:0 6px; font-weight:700; }
    label { display:grid; gap:6px; font-size:.9rem; color:var(--muted); margin:0 0 12px; }
    input, textarea, select { width:100%; border:1px solid var(--line); border-radius:6px; padding:10px 11px; font:inherit; color:var(--ink); background:white; }
    textarea { min-height:108px; resize:vertical; }
    .grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
    .checks { display:flex; flex-wrap:wrap; gap:12px; margin-top:4px; }
    .checks label { display:flex; grid-template-columns:none; align-items:center; gap:7px; margin:0; color:var(--ink); }
    .checks input { width:auto; }
    button { border:0; background:var(--accent); color:white; border-radius:6px; padding:11px 14px; font-weight:700; cursor:pointer; }
    button:disabled { opacity:.58; cursor:wait; }
    .output { margin-top:22px; display:grid; gap:16px; }
    .summary { border:1px solid var(--line); background:var(--soft); border-radius:8px; padding:14px; white-space:pre-wrap; line-height:1.55; }
    .results { display:grid; grid-template-columns:repeat(auto-fill, minmax(260px, 1fr)); gap:14px; }
    .dog { border:1px solid var(--line); background:white; border-radius:8px; overflow:hidden; display:grid; }
    .dog img { width:100%; aspect-ratio:4/3; object-fit:cover; background:var(--soft); }
    .dog .body { padding:12px; display:grid; gap:8px; }
    .dog h2 { margin:0; font-size:1rem; }
    .meta { color:var(--muted); font-size:.86rem; line-height:1.45; }
    .desc { font-size:.92rem; line-height:1.5; }
    .evidence { color:var(--accent); font-size:.84rem; line-height:1.45; }
    .advice { color:var(--warn); font-size:.84rem; line-height:1.45; }
    @media (max-width:860px) { form { grid-template-columns:1fr; } .grid { grid-template-columns:1fr; } }
  </style>
</head>
<body>
<main>
  <header>
    <h1>입양 후보 탐색</h1>
    <div class="status" id="status">대기</div>
  </header>
  <form id="flowForm">
    <fieldset>
      <legend>조건</legend>
      <label>API key<input name="api_key" id="apiKey" autocomplete="off" /></label>
      <label>생활환경<textarea name="query" placeholder="예: 지방-서울, 크기 소형, 흰색 장모, 차분한 성향, 얼굴이 잘 보이는 사진"></textarea></label>
      <div class="grid">
        <label>거주 환경<input name="living" placeholder="아파트, 단독주택" /></label>
        <label>가족 구성<input name="family" placeholder="1인 가구, 아이 있음" /></label>
        <label>산책 가능 시간<input name="walk_time" placeholder="하루 30분, 주 3회" /></label>
        <label>선호 크기<input name="dog_size" placeholder="소형, 중형" /></label>
      </div>
      <label>선호 성향<input name="preferred_personality" placeholder="차분함, 사람 좋아함" /></label>
    </fieldset>
    <fieldset>
      <legend>검색</legend>
      <label>참고 이미지<input name="ref_image" type="file" accept="image/*" /></label>
      <label>추가 조건<textarea name="extra_text" placeholder="피해야 할 조건이나 보호소에 확인할 질문"></textarea></label>
      <div class="grid">
        <label>후보 수<input name="topk" type="number" min="1" max="30" value="10" /></label>
        <label>재랭킹 폭<input name="rerank_depth" type="number" min="20" max="300" value="80" /></label>
      </div>
      <div class="checks">
        <label><input name="strict_filters" type="checkbox" /> 조건 엄격 적용</label>
        <label><input name="use_llm_parse" type="checkbox" /> LLM 파싱</label>
        <label><input name="include_recommendation" type="checkbox" checked /> 추천 설명</label>
      </div>
      <button id="submitBtn" type="submit">검색</button>
    </fieldset>
  </form>
  <section class="output">
    <div class="summary" id="summary">결과가 여기에 표시됩니다.</div>
    <div class="results" id="results"></div>
  </section>
</main>
<script>
const initialApiKey = __API_KEY__;
const form = document.getElementById('flowForm');
const apiKey = document.getElementById('apiKey');
const statusEl = document.getElementById('status');
const summaryEl = document.getElementById('summary');
const resultsEl = document.getElementById('results');
const submitBtn = document.getElementById('submitBtn');
apiKey.value = initialApiKey || localStorage.getItem('dogApiKey') || '';
function escapeHtml(value) {
  return String(value || '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
}
function render(data) {
  summaryEl.textContent = [
    data.recommendation || '',
    data.parse_source ? `조건 파싱: ${data.parse_source}` : '',
    data.search_query ? `검색 문장: ${data.search_query}` : ''
  ].filter(Boolean).join('\n\n') || '추천 설명 없이 결과만 표시합니다.';
  resultsEl.innerHTML = (data.results || []).map(item => {
    const evidence = item.retrieval_evidence || {};
    const matched = (evidence.matched_conditions || []).join(', ');
    const advice = (item.photo_advice || []).join(' ');
    return `<article class="dog">
      ${item.image_url ? `<img src="${escapeHtml(item.image_url)}" alt="">` : ''}
      <div class="body">
        <h2>${item.rank}. ${escapeHtml(item.breed)} <span class="meta">${Number(item.score || 0).toFixed(3)}</span></h2>
        <div class="meta">${escapeHtml(item.sex)} · ${escapeHtml(item.age)} · ${escapeHtml(item.weight)}</div>
        <div class="desc">${escapeHtml(item.desc).slice(0, 180)}</div>
        ${item.visual_attrs ? `<div class="evidence">${escapeHtml(item.visual_attrs)}</div>` : ''}
        ${matched ? `<div class="evidence">조건 근거: ${escapeHtml(matched)}</div>` : ''}
        ${advice ? `<div class="advice">${escapeHtml(advice)}</div>` : ''}
      </div>
    </article>`;
  }).join('');
}
form.addEventListener('submit', async event => {
  event.preventDefault();
  const key = apiKey.value.trim();
  localStorage.setItem('dogApiKey', key);
  const data = new FormData(form);
  data.delete('api_key');
  submitBtn.disabled = true;
  statusEl.textContent = '검색 중';
  try {
    const res = await fetch('/rag/recommend_form', { method: 'POST', headers: {'x-api-key': key}, body: data });
    const payload = await res.json();
    if (!res.ok) throw new Error(payload.detail || payload.error || res.statusText);
    render(payload);
    statusEl.textContent = `${payload.count || 0}건`;
  } catch (error) {
    summaryEl.textContent = String(error);
    resultsEl.innerHTML = '';
    statusEl.textContent = '오류';
  } finally {
    submitBtn.disabled = false;
  }
});
</script>
</body>
</html>
""".replace("__API_KEY__", api_key_json)
    return HTMLResponse(page)

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
