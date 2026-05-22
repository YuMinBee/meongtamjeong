import json
from pathlib import Path

import faiss
import numpy as np
import torch
import clip

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

# 1. CLIP 모델 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 2. 메타데이터 로드
with (DATA_DIR / "dog_metas.json").open("r", encoding="utf-8") as f:
    metas = json.load(f)

print(f"원본 데이터 개수: {len(metas)}")

# 3. 리스트 안에 리스트(flatten) 처리
if isinstance(metas[0], list):
    metas = [item for sublist in metas for item in sublist]

print(f"flatten 이후 데이터 개수: {len(metas)}")

# 4. 텍스트 임베딩 생성
embeddings = []
id_list = []

for m in metas:
    if m.get("type") == "text":
        text = m.get("desc", "")
        tokens = clip.tokenize([text]).to(device)

        with torch.no_grad():
            text_emb = model.encode_text(tokens)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)  # 정규화
            embeddings.append(text_emb.cpu().numpy())
            id_list.append(m.get("desertionNo"))

print(f"임베딩된 텍스트 개수: {len(embeddings)}")

# 5. numpy 배열 변환
embeddings = np.vstack(embeddings).astype("float32")

# 6. FAISS 인덱스 생성 (L2 거리 기반)
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# 7. 인덱스 + ID 매핑 저장
faiss.write_index(index, str(DATA_DIR / "dog_faiss.index"))
with (DATA_DIR / "dog_ids.json").open("w", encoding="utf-8") as f:
    json.dump(id_list, f, ensure_ascii=False, indent=2)

print("✅ FAISS 인덱스(dog_faiss.index)와 ID 매핑(dog_ids.json) 저장 완료!")
