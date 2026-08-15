"""
FAISS 인덱스 기반 로컬 유기견 검색 (2가지 옵션)
- data/dog_faiss.index + data/dog_metas.json 로드
- CLIP으로 쿼리 임베딩 생성
- (1) 로컬 이미지로만 검색
- (2) 로컬 이미지 + 텍스트 조합 검색
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import faiss
import torch
import clip
from PIL import Image

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

# =========================
# 설정
# =========================
INDEX_PATH = DATA_DIR / "dog_faiss.index"
METAS_PATH = DATA_DIR / "dog_metas.json"
CLIP_MODEL = "ViT-B/32"

TOPK_DEFAULT = 5

# =========================
# CLIP 로드
# =========================
print("[INFO] CLIP 모델 로딩 중...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(CLIP_MODEL, device=device)
model.eval()
print(f"[INFO] Device: {device}")

# =========================
# FAISS + METAS 로드
# =========================
if not INDEX_PATH.exists():
    raise FileNotFoundError(f"Index not found: {INDEX_PATH}")
if not METAS_PATH.exists():
    raise FileNotFoundError(f"Metas not found: {METAS_PATH}")

index = faiss.read_index(str(INDEX_PATH))
with METAS_PATH.open("r", encoding="utf-8") as f:
    METAS: List[Dict[str, Any]] = json.load(f)

# metas가 list of list로 저장된 경우 flatten
if len(METAS) > 0 and isinstance(METAS[0], list):
    METAS = [x for sub in METAS for x in sub]

print(f"[INFO] index.ntotal={index.ntotal}, metas={len(METAS)}")

# 정합성 체크 (불일치하면 검색 결과 매핑이 틀어짐)
if index.ntotal != len(METAS):
    raise ValueError(f"[ERROR] Index({index.ntotal}) != Metas({len(METAS)}) "
                     f"→ 인덱스와 메타를 같은 생성 로직으로 맞춰야 합니다.")

# =========================
# 임베딩 함수
# =========================
@torch.no_grad()
def embed_text(text: str) -> np.ndarray:
    toks = clip.tokenize([text], truncate=True).to(device)
    v = model.encode_text(toks)
    v = v / v.norm(dim=-1, keepdim=True)
    return v.detach().cpu().numpy().astype("float32")  # shape (1,512)

@torch.no_grad()
def embed_image_from_path(path: str) -> np.ndarray:
    pil = Image.open(path).convert("RGB")
    x = preprocess(pil).unsqueeze(0).to(device)
    v = model.encode_image(x)
    v = v / v.norm(dim=-1, keepdim=True)
    return v.detach().cpu().numpy().astype("float32")  # shape (1,512)

def combine_embeddings(img_vec: np.ndarray,
                       text_vec: Optional[np.ndarray] = None,
                       w_img: float = 0.7) -> np.ndarray:
    """
    img_vec, text_vec: shape (1,512)
    """
    if text_vec is None:
        v = img_vec
    else:
        w_img = float(np.clip(w_img, 0.0, 1.0))
        v = (w_img * img_vec) + ((1.0 - w_img) * text_vec)

    # 정규화
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    return v.astype("float32")

# =========================
# 검색 + 결과 후처리
# =========================
def _pick_image_meta_if_text(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    metas에 text/image 타입이 섞여있을 때:
    text가 걸리면 같은 desertionNo의 image meta로 치환(가능하면)
    """
    if meta.get("type") != "text":
        return meta

    did = meta.get("desertionNo")
    if not did:
        return meta

    # 같은 desertionNo의 image meta 찾기
    for m in METAS:
        if m.get("desertionNo") == did and m.get("type") == "image":
            return m
    return meta

def search(vec: np.ndarray, topk: int = 5) -> List[Dict[str, Any]]:
    distances, indices = index.search(vec, topk)

    results = []
    for rank, (dist, idx) in enumerate(
        zip(distances[0].tolist(), indices[0].tolist()),
        start=1,
    ):
        if idx < 0:
            continue

        meta = METAS[idx]
        meta = _pick_image_meta_if_text(meta)

        desertionNo = meta.get("desertionNo", "Unknown")
        image_url = meta.get("image_url") or meta.get("url") or ""

        # 사이트 상세 링크 (너가 원래 쓰던 포맷)
        detail_url = (
            "https://www.animal.go.kr/front/awtis/public/publicDtl.do"
            f"?desertionNo={desertionNo}&menuNo=1000000055"
            if desertionNo != "Unknown" else ""
        )

        # L2 dist -> 점수(0~1 비슷하게) 변환 (그냥 보기 편하게)
        score = float(1.0 / (1.0 + dist))

        results.append({
            "rank": rank,
            "score": score,
            "dist": float(dist),
            "desertionNo": desertionNo,
            "breed": meta.get("kindCd") or meta.get("breed") or "Unknown",
            "sex": meta.get("sex") or meta.get("sexCd") or "Unknown",
            "age": meta.get("age", "Unknown"),
            "weight": meta.get("weight", "Unknown"),
            "neuter": meta.get("neuter") or meta.get("neuterYn") or "Unknown",
            "desc": meta.get("desc") or meta.get("specialMark") or "",
            "image_url": image_url,
            "detail_url": detail_url,
        })

    return results

def print_results(results: List[Dict[str, Any]]):
    print("\n" + "=" * 80)
    print("🐕 유사한 유기견 검색 결과 (FAISS)")
    print("=" * 80)

    for r in results:
        print(f"\n[{r['rank']}위] score={r['score']:.4f} (dist={r['dist']:.4f})")
        print(f"  📋 공고번호: {r['desertionNo']}")
        print(f"  🐶 품종: {r['breed']}")
        print(f"  ⚥ 성별: {r['sex']} | 나이: {r['age']} | 체중: {r['weight']}")
        print(f"  💉 중성화: {r['neuter']}")
        d = r["desc"] or ""
        print(f"  📝 특징: {d[:120]}{'...' if len(d) > 120 else ''}")
        print(f"  🔗 이미지: {r['image_url']}")
        print(f"  🌐 상세: {r['detail_url']}")

    print("\n" + "=" * 80)

# =========================
# 메인
# =========================
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("검색 방법을 선택하세요:")
    print("  1. 로컬 이미지로만 검색 (FAISS)")
    print("  2. 로컬 이미지 + 텍스트 조합 검색 (FAISS)")
    print("=" * 60)

    choice = input("\n선택 (1/2): ").strip()
    img_path = input("로컬 이미지 파일 경로 입력: ").strip()

    if not img_path:
        print("[ERROR] 이미지 경로가 비어 있습니다.")
        sys.exit(1)

    try:
        img_vec = embed_image_from_path(img_path)
    except Exception as e:
        print(f"[ERROR] 이미지 임베딩 실패: {e}")
        sys.exit(1)

    if choice == "1":
        q = combine_embeddings(img_vec, None, w_img=1.0)

    elif choice == "2":
        text = input("추가 텍스트 입력 (예: '차분한', '소형견', '흰색'): ").strip()
        if not text:
            print("[ERROR] 2번은 텍스트가 필요합니다.")
            sys.exit(1)
        text_vec = embed_text(text)
        # 기본: 이미지 0.7, 텍스트 0.3
        q = combine_embeddings(img_vec, text_vec, w_img=0.7)

    else:
        print("[ERROR] 잘못된 선택입니다.")
        sys.exit(1)

    topk = int(input(f"\n몇 개의 결과를 보시겠습니까? (기본 {TOPK_DEFAULT}): ").strip() or str(TOPK_DEFAULT))
    results = search(q, topk=topk)
    print_results(results)
