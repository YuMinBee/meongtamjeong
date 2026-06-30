import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.dog_attributes import normalize_vlm_attrs, summarize_vlm_attrs_ko
from app.notice_status import is_searchable_notice


TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]+")

COLOR_TERMS = {
    "white": ("흰", "하얀", "백색", "화이트", "white"),
    "black": ("검", "검정", "까만", "블랙", "black"),
    "brown": ("갈색", "브라운", "brown"),
    "tan": ("황색", "황갈", "탄", "tan"),
    "cream": ("크림", "아이보리", "cream", "ivory"),
    "gray": ("회색", "그레이", "gray", "grey"),
    "spotted": ("무늬", "얼룩", "반점", "spotted"),
}

SIZE_TERMS = {
    "tiny": ("초소형", "아주 작", "미니", "tiny"),
    "small": ("소형", "작은", "작고", "작은개", "small"),
    "medium": ("중형", "중간", "medium"),
    "large": ("대형", "큰", "크고", "large"),
}

FUR_TERMS = {
    "short": ("단모", "짧은 털", "털이 짧", "short"),
    "medium": ("중간 털", "중간 길이", "medium"),
    "long": ("장모", "긴 털", "털이 긴", "long"),
    "curly": ("곱슬", "푸들", "curly"),
    "fluffy": ("복슬", "복슬복슬", "풍성", "fluffy"),
}

EAR_TERMS = {
    "upright": ("귀가 선", "귀가 서", "쫑긋", "upright"),
    "floppy": ("처진 귀", "귀가 처", "floppy"),
    "semi_upright": ("반쯤 선", "반립", "semi"),
    "folded": ("접힌 귀", "folded"),
}

SEX_TERMS = {
    "M": ("수컷", "남아", "남자", "male", "수"),
    "F": ("암컷", "여아", "여자", "female", "암"),
}

PERSONALITY_TERMS = (
    "순함",
    "차분",
    "조용",
    "활발",
    "애교",
    "사람 좋아",
    "겁",
    "소심",
    "경계",
    "공격",
)

AGE_TERMS = {
    "puppy": ("강아지", "새끼", "어린", "개월", "puppy"),
    "adult": ("성견", "adult"),
    "senior": ("노견", "고령", "senior"),
}


REQUIRED_QUERY_KEYS = {
    "coat_color": list,
    "fur_length": list,
    "ear_shape": list,
    "body_size_hint": list,
    "sex": list,
    "age_hint": list,
    "personality": list,
    "face_visible": object,
    "whole_body_visible": object,
    "min_photo_quality": object,
    "keywords": list,
}


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(clean_text(text))]


def _contains_any(text: str, terms: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in terms)


def _collect_terms(text: str, mapping: Dict[str, Iterable[str]]) -> List[str]:
    return [key for key, terms in mapping.items() if _contains_any(text, terms)]


def _append_unique(values: List[str], additions: Iterable[str]) -> None:
    for value in additions:
        value = clean_text(value)
        if value and value not in values:
            values.append(value)


def empty_structured_query(raw_query: str = "") -> Dict[str, Any]:
    return {
        "raw_query": clean_text(raw_query),
        "coat_color": [],
        "fur_length": [],
        "ear_shape": [],
        "body_size_hint": [],
        "sex": [],
        "age_hint": [],
        "personality": [],
        "face_visible": None,
        "whole_body_visible": None,
        "min_photo_quality": None,
        "keywords": [],
    }


def normalize_structured_query(value: Any, raw_query: str = "") -> Dict[str, Any]:
    query = empty_structured_query(raw_query)
    if not isinstance(value, dict):
        return query

    for key in ("coat_color", "fur_length", "ear_shape", "body_size_hint", "sex", "age_hint", "personality", "keywords"):
        current = value.get(key, [])
        if isinstance(current, list):
            query[key] = [clean_text(item) for item in current if clean_text(item)]
        elif clean_text(current):
            query[key] = [clean_text(current)]

    for key in ("face_visible", "whole_body_visible"):
        current = value.get(key)
        if isinstance(current, bool) or current is None:
            query[key] = current
        else:
            lowered = clean_text(current).lower()
            if lowered in {"true", "yes", "1", "visible", "보임"}:
                query[key] = True
            elif lowered in {"false", "no", "0", "hidden", "안보임", "안 보임"}:
                query[key] = False

    try:
        quality = float(value.get("min_photo_quality"))
    except (TypeError, ValueError):
        quality = None
    query["min_photo_quality"] = quality
    if clean_text(value.get("raw_query")):
        query["raw_query"] = clean_text(value.get("raw_query"))
    return query


def merge_structured_query(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = normalize_structured_query(base, base.get("raw_query", ""))
    other = normalize_structured_query(override, merged.get("raw_query", ""))
    for key in ("coat_color", "fur_length", "ear_shape", "body_size_hint", "sex", "age_hint", "personality", "keywords"):
        _append_unique(merged[key], other.get(key, []))
    for key in ("face_visible", "whole_body_visible", "min_photo_quality"):
        if other.get(key) is not None:
            merged[key] = other[key]
    if other.get("raw_query"):
        merged["raw_query"] = other["raw_query"]
    return merged


def parse_structured_query(text: str, survey: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    query = empty_structured_query(text)
    combined = clean_text(text)
    if survey:
        combined = " ".join([combined] + [clean_text(value) for value in survey.values()])

    _append_unique(query["coat_color"], _collect_terms(combined, COLOR_TERMS))
    _append_unique(query["fur_length"], _collect_terms(combined, FUR_TERMS))
    _append_unique(query["ear_shape"], _collect_terms(combined, EAR_TERMS))
    _append_unique(query["body_size_hint"], _collect_terms(combined, SIZE_TERMS))
    _append_unique(query["sex"], _collect_terms(combined, SEX_TERMS))
    _append_unique(query["age_hint"], _collect_terms(combined, AGE_TERMS))

    for term in PERSONALITY_TERMS:
        if term in combined and term not in query["personality"]:
            query["personality"].append(term)

    if _contains_any(combined, ("얼굴 잘", "얼굴이 잘", "정면", "눈이 잘", "face visible")):
        query["face_visible"] = True
    if _contains_any(combined, ("얼굴 안", "얼굴 가려", "얼굴이 안", "face hidden")):
        query["face_visible"] = False
    if _contains_any(combined, ("전신", "몸 전체", "whole body")):
        query["whole_body_visible"] = True
    if _contains_any(combined, ("사진 선명", "사진 잘", "잘 보이는 사진", "고화질")):
        query["min_photo_quality"] = 0.65

    stopwords = {"강아지", "유기견", "추천", "찾아줘", "원해", "좋아", "가능", "사진", "조건"}
    query["keywords"] = [token for token in tokenize(combined) if len(token) > 1 and token not in stopwords][:20]
    return query


def build_search_query_text(raw_query: str, structured: Dict[str, Any]) -> str:
    parts = [clean_text(raw_query)]
    labels = {
        "coat_color": "털색",
        "fur_length": "털길이",
        "ear_shape": "귀모양",
        "body_size_hint": "몸집",
        "sex": "성별",
        "age_hint": "나이",
        "personality": "성향",
    }
    for key, label in labels.items():
        values = structured.get(key) or []
        if values:
            parts.append(f"{label} {' '.join(values)}")
    if structured.get("face_visible") is True:
        parts.append("얼굴이 잘 보이는 사진")
    if structured.get("whole_body_visible") is True:
        parts.append("전신이 보이는 사진")
    return " ".join(part for part in parts if part)


def weight_to_size_hint(weight: Any) -> str:
    text = clean_text(weight).replace(",", "")
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)
    if not match:
        return ""
    kg = float(match.group(1))
    if kg <= 3:
        return "tiny"
    if kg <= 8:
        return "small"
    if kg <= 18:
        return "medium"
    return "large"


def resolve_doc_text(meta: Dict[str, Any]) -> str:
    attrs_summary = summarize_vlm_attrs_ko(meta.get("vlm_attrs"))
    bits = [
        clean_text(meta.get("breed")),
        clean_text(meta.get("breed_name")),
        clean_text(meta.get("breed_code")),
        clean_text(meta.get("sex")),
        clean_text(meta.get("age")),
        clean_text(meta.get("weight")),
        clean_text(meta.get("neuter")),
        clean_text(meta.get("desc")),
        clean_text(meta.get("desc_full")),
        clean_text(meta.get("specialMark")),
        clean_text(meta.get("vlm_desc")),
        clean_text(meta.get("merged_desc")),
        clean_text(meta.get("vlm_attr_text")),
        attrs_summary,
        " ".join(clean_text(item) for item in meta.get("photo_advice", []) if clean_text(item)) if isinstance(meta.get("photo_advice"), list) else "",
    ]
    return " ".join(bit for bit in bits if bit)


def build_hybrid_documents(metas: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[int, int]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    vector_to_doc: Dict[int, int] = {}

    for vector_index, meta in enumerate(metas):
        doc_id = clean_text(meta.get("desertionNo")) or f"idx-{vector_index}"
        doc = by_id.setdefault(
            doc_id,
            {
                "doc_id": doc_id,
                "meta": {},
                "texts": [],
                "vector_indices": [],
            },
        )
        if not doc["meta"] or meta.get("type") == "image":
            merged = dict(doc["meta"])
            merged.update(meta)
            doc["meta"] = merged
        else:
            for key, value in meta.items():
                if not doc["meta"].get(key) and value:
                    doc["meta"][key] = value
        text = resolve_doc_text(meta)
        if text:
            doc["texts"].append(text)
        doc["vector_indices"].append(vector_index)

    docs = list(by_id.values())
    for doc_index, doc in enumerate(docs):
        full_text = " ".join(doc["texts"] + [resolve_doc_text(doc["meta"])])
        doc["text"] = full_text
        doc["tokens"] = tokenize(full_text)
        for vector_index in doc["vector_indices"]:
            vector_to_doc[vector_index] = doc_index
    return docs, vector_to_doc


class BM25Index:
    def __init__(self, docs: List[Dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.doc_freq: Dict[str, int] = defaultdict(int)
        self.term_freqs: List[Counter] = []
        self.doc_lengths: List[int] = []
        for doc in docs:
            counts = Counter(doc.get("tokens") or [])
            self.term_freqs.append(counts)
            self.doc_lengths.append(sum(counts.values()))
            for token in counts:
                self.doc_freq[token] += 1
        self.avgdl = (sum(self.doc_lengths) / len(self.doc_lengths)) if self.doc_lengths else 0.0

    def score_doc(self, query_tokens: List[str], doc_index: int) -> float:
        if not query_tokens or not self.docs:
            return 0.0
        counts = self.term_freqs[doc_index]
        doc_len = self.doc_lengths[doc_index] or 1
        score = 0.0
        for token in query_tokens:
            freq = counts.get(token, 0)
            if freq <= 0:
                continue
            df = self.doc_freq.get(token, 0)
            idf = math.log(1.0 + (len(self.docs) - df + 0.5) / (df + 0.5))
            denom = freq + self.k1 * (1.0 - self.b + self.b * doc_len / (self.avgdl or 1.0))
            score += idf * (freq * (self.k1 + 1.0)) / denom
        return score

    def top_scores(self, query_text: str, limit: int) -> Dict[int, float]:
        query_tokens = tokenize(query_text)
        if not query_tokens:
            return {}
        scored = [(index, self.score_doc(query_tokens, index)) for index in range(len(self.docs))]
        scored = [(index, score) for index, score in scored if score > 0]
        scored.sort(key=lambda item: item[1], reverse=True)
        return dict(scored[:limit])


def vector_hits_to_doc_scores(
    distances: List[float],
    indices: List[int],
    vector_to_doc: Dict[int, int],
) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    for dist, vector_index in zip(distances, indices):
        if vector_index < 0:
            continue
        doc_index = vector_to_doc.get(vector_index)
        if doc_index is None:
            continue
        score = float(1.0 / (1.0 + dist))
        scores[doc_index] = max(scores.get(doc_index, 0.0), score)
    return scores


VECTOR_MODALITY_WEIGHTS = {
    "text": 0.45,
    "full_image": 0.25,
    "crop_image": 0.20,
}


def resolve_vector_modality(meta: Dict[str, Any]) -> str:
    meta_type = clean_text(meta.get("type"))
    source = clean_text(meta.get("embedding_source"))
    if meta_type == "text":
        return "text"
    if meta_type == "crop_image" or source in {"dog_crop", "animal_crop", "crop_image"}:
        return "crop_image"
    if meta_type == "image" or source in {"full_image", "image"}:
        return "full_image"
    return source or meta_type or "unknown"


def weighted_available_score(
    modality_scores: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, float, List[str]]:
    weights = weights or VECTOR_MODALITY_WEIGHTS
    numerator = 0.0
    denominator = 0.0
    missing: List[str] = []
    for modality, weight in weights.items():
        score = modality_scores.get(modality)
        if score is None:
            missing.append(modality)
            continue
        numerator += score * weight
        denominator += weight
    if denominator <= 0:
        return 0.0, 0.0, missing
    return numerator / denominator, denominator, missing


def vector_hits_to_doc_modality_scores(
    distances: List[float],
    indices: List[int],
    vector_to_doc: Dict[int, int],
    metas: List[Dict[str, Any]],
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[int, float], Dict[int, Dict[str, Any]]]:
    per_doc: Dict[int, Dict[str, Any]] = defaultdict(
        lambda: {
            "modalities": {},
            "best_vector_score": 0.0,
            "best_vector_index": None,
            "best_modality": "",
            "hit_count": 0,
        }
    )
    for dist, vector_index in zip(distances, indices):
        if vector_index < 0:
            continue
        doc_index = vector_to_doc.get(vector_index)
        if doc_index is None or vector_index >= len(metas):
            continue
        score = float(1.0 / (1.0 + dist))
        modality = resolve_vector_modality(metas[vector_index])
        doc_scores = per_doc[doc_index]
        doc_scores["hit_count"] += 1
        modalities = doc_scores["modalities"]
        if score > modalities.get(modality, 0.0):
            modalities[modality] = score
        if score > doc_scores["best_vector_score"]:
            doc_scores["best_vector_score"] = score
            doc_scores["best_vector_index"] = vector_index
            doc_scores["best_modality"] = modality

    aggregated: Dict[int, float] = {}
    details: Dict[int, Dict[str, Any]] = {}
    for doc_index, doc_scores in per_doc.items():
        modalities = doc_scores["modalities"]
        score, available_weight, missing = weighted_available_score(modalities, weights)
        aggregated[doc_index] = score
        details[doc_index] = {
            **doc_scores,
            "score": score,
            "available_weight": available_weight,
            "missing_modalities": missing,
        }
    return aggregated, details

def _text_contains_any(doc_text: str, terms: Iterable[str]) -> bool:
    lowered = doc_text.lower()
    return any(clean_text(term).lower() in lowered for term in terms if clean_text(term))


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_image_attrs(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def resolve_photo_quality_score(meta: Dict[str, Any]) -> Optional[float]:
    image_attrs = normalize_image_attrs(meta.get("image_attrs"))
    quality = _as_float(image_attrs.get("photo_quality_score"))
    if quality is not None:
        return quality
    crop_quality = _as_float(image_attrs.get("crop_photo_quality_score"))
    if crop_quality is not None:
        return crop_quality
    attrs = normalize_vlm_attrs(meta.get("vlm_attrs"))
    return _as_float(attrs.get("photo_quality_score"))


def visual_attr_score(meta: Dict[str, Any]) -> float:
    image_attrs = normalize_image_attrs(meta.get("image_attrs"))
    if not image_attrs:
        quality = resolve_photo_quality_score(meta)
        return quality if quality is not None else 0.5

    score = 0.0
    weight = 0.0

    quality = resolve_photo_quality_score(meta)
    if quality is not None:
        score += max(0.0, min(1.0, quality)) * 0.42
        weight += 0.42

    detected = image_attrs.get("target_detected")
    if detected is None:
        detected = image_attrs.get("dog_detected")
    if detected is not None:
        score += (1.0 if detected is True else 0.0) * 0.18
        weight += 0.18

    area = _as_float(image_attrs.get("dog_area_ratio") or image_attrs.get("target_area_ratio"))
    if area is not None:
        if area <= 0.0:
            area_score = 0.0
        elif area < 0.08:
            area_score = 0.45 * (area / 0.08)
        elif area <= 0.45:
            area_score = min(1.0, area / 0.28)
        else:
            area_score = max(0.55, 1.0 - ((area - 0.45) / 0.55) * 0.35)
        score += max(0.0, min(1.0, area_score)) * 0.18
        weight += 0.18

    centered = image_attrs.get("bbox_centered")
    if centered is not None:
        score += (1.0 if centered is True else 0.35) * 0.12
        weight += 0.12

    confidence = _as_float(image_attrs.get("primary_confidence"))
    if confidence is not None:
        score += max(0.0, min(1.0, confidence)) * 0.05
        weight += 0.05

    similarity = _as_float(image_attrs.get("full_crop_clip_similarity"))
    if similarity is not None:
        score += max(0.0, min(1.0, similarity)) * 0.05
        weight += 0.05

    return score / weight if weight else 0.5

def condition_score(doc: Dict[str, Any], structured: Dict[str, Any]) -> Tuple[float, List[str], List[str], List[str]]:
    meta = doc.get("meta") or {}
    attrs = normalize_vlm_attrs(meta.get("vlm_attrs"))
    text = doc.get("text", "")
    matched: List[str] = []
    missing: List[str] = []
    mismatched: List[str] = []
    total = 0
    points = 0.0

    def check(name: str, desired: List[str], actual: Any, text_terms: Optional[List[str]] = None) -> None:
        nonlocal total, points
        if not desired:
            return
        total += 1
        actual_values = actual if isinstance(actual, list) else ([actual] if actual else [])
        actual_values = [clean_text(value).lower() for value in actual_values if clean_text(value)]
        desired_values = [clean_text(value).lower() for value in desired if clean_text(value)]
        if any(value in actual_values for value in desired_values) or (text_terms and _text_contains_any(text, text_terms)):
            points += 1.0
            matched.append(f"{name}: {', '.join(desired)}")
        elif actual_values:
            mismatched.append(f"{name}: requested {', '.join(desired)}, found {', '.join(actual_values)}")
        else:
            points += 0.2
            missing.append(f"{name}: metadata missing")

    color_terms = [term for color in structured.get("coat_color", []) for term in COLOR_TERMS.get(color, (color,))]
    check("coat_color", structured.get("coat_color", []), attrs.get("coat_color"), color_terms)
    fur_terms = [term for fur in structured.get("fur_length", []) for term in FUR_TERMS.get(fur, (fur,))]
    check("fur_length", structured.get("fur_length", []), attrs.get("fur_length"), fur_terms)
    ear_terms = [term for ear in structured.get("ear_shape", []) for term in EAR_TERMS.get(ear, (ear,))]
    check("ear_shape", structured.get("ear_shape", []), attrs.get("ear_shape"), ear_terms)

    inferred_size = attrs.get("body_size_hint") if attrs.get("body_size_hint") != "unknown" else weight_to_size_hint(meta.get("weight"))
    size_terms = [term for size in structured.get("body_size_hint", []) for term in SIZE_TERMS.get(size, (size,))]
    check("body_size", structured.get("body_size_hint", []), inferred_size, size_terms)
    check("sex", structured.get("sex", []), clean_text(meta.get("sex") or meta.get("sexCd")))
    check("age", structured.get("age_hint", []), "", [term for age in structured.get("age_hint", []) for term in AGE_TERMS.get(age, (age,))])
    check("personality", structured.get("personality", []), "", structured.get("personality", []))

    for key, label in (("face_visible", "face_visible"), ("whole_body_visible", "whole_body_visible")):
        desired_bool = structured.get(key)
        if desired_bool is None:
            continue
        total += 1
        actual_bool = attrs.get(key)
        if actual_bool is desired_bool:
            points += 1.0
            matched.append(f"{label}: {desired_bool}")
        elif actual_bool is None:
            points += 0.2
            missing.append(f"{label}: metadata missing")
        else:
            mismatched.append(f"{label}: requested {desired_bool}, found {actual_bool}")

    min_quality = structured.get("min_photo_quality")
    if min_quality is not None:
        total += 1
        quality = resolve_photo_quality_score(meta)
        if quality is None:
            points += 0.2
            missing.append("photo_quality_score: metadata missing")
        elif quality >= min_quality:
            points += 1.0
            matched.append(f"photo_quality_score: {quality:.2f}")
        else:
            mismatched.append(f"photo_quality_score: requested >= {min_quality:.2f}, found {quality:.2f}")

    if total == 0:
        return 0.0, matched, missing, mismatched
    return min(1.0, points / total), matched, missing, mismatched


def rank_hybrid_documents(
    docs: List[Dict[str, Any]],
    bm25: BM25Index,
    structured: Dict[str, Any],
    query_text: str,
    vector_scores: Dict[int, float],
    vector_score_details: Optional[Dict[int, Dict[str, Any]]] = None,
    topk: int = 10,
    rerank_depth: int = 80,
    strict_filters: bool = False,
    extra_candidate_indices: Optional[Iterable[int]] = None,
    filter_inactive_notices: bool = True,
    include_unknown_notices: bool = True,
) -> List[Dict[str, Any]]:
    bm25_scores = bm25.top_scores(query_text, max(rerank_depth, topk * 5))
    candidate_indices = set(vector_scores) | set(bm25_scores)
    if extra_candidate_indices:
        candidate_indices.update(
            doc_index
            for doc_index in extra_candidate_indices
            if 0 <= doc_index < len(docs)
        )
    if not candidate_indices:
        candidate_indices = set(range(min(len(docs), rerank_depth)))

    max_bm25 = max(bm25_scores.values(), default=0.0) or 1.0
    ranked: List[Dict[str, Any]] = []
    for doc_index in candidate_indices:
        doc = docs[doc_index]
        meta = doc.get("meta") or {}
        if filter_inactive_notices and not is_searchable_notice(meta, include_unknown=include_unknown_notices):
            continue
        cond_score, matched, missing, mismatched = condition_score(doc, structured)
        if strict_filters and mismatched:
            continue
        quality = resolve_photo_quality_score(meta)
        photo_score = quality if quality is not None else 0.5
        visual_score = visual_attr_score(meta)
        vector_score = vector_scores.get(doc_index, 0.0)
        vector_detail = (vector_score_details or {}).get(doc_index, {})
        modality_scores = vector_detail.get("modalities") if isinstance(vector_detail.get("modalities"), dict) else {}
        bm25_norm = bm25_scores.get(doc_index, 0.0) / max_bm25
        final_score = (0.50 * vector_score) + (0.20 * bm25_norm) + (0.22 * cond_score) + (0.08 * visual_score)
        ranked.append(
            {
                "doc": doc,
                "doc_index": doc_index,
                "score": final_score,
                "score_parts": {
                    "vector": round(vector_score, 4),
                    "vector_modalities": {key: round(value, 4) for key, value in modality_scores.items()},
                    "vector_available_weight": round(float(vector_detail.get("available_weight") or 0.0), 4),
                    "vector_best_modality": clean_text(vector_detail.get("best_modality")),
                    "bm25": round(bm25_norm, 4),
                    "conditions": round(cond_score, 4),
                    "photo_quality": round(photo_score, 4),
                    "visual_attrs": round(visual_score, 4),
                },
                "evidence": {
                    "matched_conditions": matched,
                    "missing_conditions": missing,
                    "mismatched_conditions": mismatched,
                    "bm25_terms": tokenize(query_text)[:12],
                    "missing_vector_modalities": vector_detail.get("missing_modalities", []),
                    "vector_hit_count": vector_detail.get("hit_count", 0),
                },
            }
        )
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:topk]
