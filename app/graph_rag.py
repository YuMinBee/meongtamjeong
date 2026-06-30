from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from app.dog_attributes import normalize_vlm_attrs


YEAR_RE = re.compile(r"(19|20)\d{2}")
WEIGHT_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)")

REGION_HINTS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("서울", ("서울",)),
    ("경기", ("경기", "수원", "성남", "고양", "용인", "부천", "안산", "안양", "평택", "의정부", "시흥", "화성", "김포", "파주", "광명", "군포", "하남", "이천", "안성", "오산", "양주", "포천", "여주", "가평", "양평")),
    ("인천", ("인천",)),
    ("강원", ("강원", "춘천", "원주", "강릉", "동해", "태백", "속초", "삼척")),
    ("충북", ("충북", "청주", "충주", "제천", "음성", "진천")),
    ("충남", ("충남", "천안", "공주", "아산", "논산", "보령", "서산", "당진", "홍성")),
    ("대전", ("대전",)),
    ("세종", ("세종",)),
    ("전북", ("전북", "전주", "군산", "익산", "정읍", "남원", "김제")),
    ("전남", ("전남", "목포", "여수", "순천", "나주", "광양")),
    ("광주", ("광주",)),
    ("경북", ("경북", "포항", "경주", "김천", "안동", "구미", "영주", "영천", "상주", "문경")),
    ("경남", ("경남", "창원", "진주", "통영", "사천", "김해", "밀양", "거제", "양산")),
    ("대구", ("대구",)),
    ("울산", ("울산",)),
    ("부산", ("부산",)),
    ("제주", ("제주",)),
)

TEXT_FEATURE_HINTS: Tuple[Tuple[str, str, float, Tuple[str, ...]], ...] = (
    ("temperament:calm", "Trait: calm temperament", 0.65, ("차분", "순함", "온순", "얌전", "조용", "착함")),
    ("temperament:friendly", "Trait: people friendly", 0.65, ("사람", "잘따", "잘 따", "애교", "친화", "뽀뽀", "좋아함")),
    ("temperament:active", "Trait: active", 0.45, ("활발", "발랄", "장난", "산책", "에너지")),
    ("temperament:watchful", "Trait: watchful", 0.35, ("경계", "낯가림", "겁", "소심")),
)

QUERY_TEXT_FEATURE_HINTS: Tuple[Tuple[str, str, float, Tuple[str, ...]], ...] = (
    ("temperament:calm", "Trait: calm temperament", 0.65, ("차분", "순한", "순함", "온순", "얌전", "조용", "초보", "아이", "어린이", "가족")),
    ("temperament:friendly", "Trait: people friendly", 0.65, ("사람", "잘 따", "잘따", "친화", "애교", "가족", "아이", "어린이")),
    ("temperament:active", "Trait: active", 0.45, ("활발", "산책", "운동", "놀이", "장난")),
    ("temperament:watchful", "Trait: watchful", 0.35, ("경계", "낯가림", "소심", "겁")),
)

FIELD_ALIASES = {
    "desertion_no": ("desertionNo", "desertion_no"),
    "shelter": ("care_name", "careNm", "shelter", "shelter_name"),
    "status": ("process_state", "processState", "status"),
    "region": ("region", "org_name", "orgNm", "care_addr", "careAddr", "happen_place", "happenPlace", "care_name", "careNm"),
    "breed_code": ("breed_code", "breedCd", "breed"),
    "breed_name": ("breed_name", "kindCd"),
    "sex": ("sex", "sexCd"),
    "age": ("age",),
    "weight": ("weight",),
    "neuter": ("neuter", "neuterYn"),
}


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return bool(clean_text(value))


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [clean_text(item) for item in value if clean_text(item)]
    text = clean_text(value)
    return [text] if text else []


def _field(meta: Dict[str, Any], field_name: str) -> str:
    for key in FIELD_ALIASES.get(field_name, (field_name,)):
        value = clean_text(meta.get(key))
        if value:
            return value
    return ""


def _feature_value(value: Any) -> str:
    return clean_text(value).lower()


def _feature_weight(feature: str) -> float:
    if feature.startswith(("region:", "shelter:", "status:", "notice:")):
        return 1.15
    if feature.startswith(("trait:", "photo:", "sex:", "age:")):
        return 1.0
    if feature.startswith("temperament:"):
        return 0.65
    if feature.startswith("breed:"):
        return 0.35
    return 0.75


def infer_size_from_weight(weight: Any) -> str:
    match = WEIGHT_RE.search(clean_text(weight).replace(",", ""))
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


def infer_age_hint(age: Any, reference_year: Optional[int] = None) -> str:
    text = clean_text(age)
    if not text:
        return ""
    if reference_year is None:
        reference_year = datetime.now().year
    match = YEAR_RE.search(text)
    if match:
        years = reference_year - int(match.group(0))
        if years <= 1:
            return "puppy"
        if years >= 8:
            return "senior"
        return "adult"
    lowered = text.lower()
    if any(token in lowered for token in ("puppy", "개월", "새끼")):
        return "puppy"
    if any(token in lowered for token in ("senior", "노령", "노견")):
        return "senior"
    return ""


def infer_region(meta: Dict[str, Any]) -> str:
    haystack = " ".join(_field(meta, name) for name in ("region", "shelter"))
    if not haystack:
        return ""
    for region, hints in REGION_HINTS:
        if any(hint and hint in haystack for hint in hints):
            return region
    return ""


def _text_contains_any(text: str, hints: Iterable[str]) -> bool:
    return any(hint and hint in text for hint in hints)


def _overlay_by_id(overlay_metas: Optional[Iterable[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    overlays: Dict[str, Dict[str, Any]] = {}
    if not overlay_metas:
        return overlays
    for meta in overlay_metas:
        if not isinstance(meta, dict):
            continue
        dog_id = clean_text(meta.get("desertionNo") or meta.get("desertion_no"))
        if dog_id and dog_id not in overlays:
            overlays[dog_id] = meta
    return overlays


@dataclass
class DogGraphIndex:
    docs: List[Dict[str, Any]]
    doc_features: List[Set[str]]
    feature_to_docs: Dict[str, Set[int]]
    feature_labels: Dict[str, str]
    feature_weights: Dict[str, float]
    region_features: Dict[str, str] = field(default_factory=dict)
    shelter_features: Dict[str, str] = field(default_factory=dict)
    status_features: Dict[str, str] = field(default_factory=dict)
    overlay_matched: int = 0

    def remember_feature(self, feature: str, label: str, weight: Optional[float] = None) -> None:
        if not feature:
            return
        self.feature_labels.setdefault(feature, label or feature)
        self.feature_weights.setdefault(feature, weight if weight is not None else _feature_weight(feature))

    def labels_for(self, features: Iterable[str], limit: int = 12) -> List[str]:
        ordered = sorted(set(features), key=lambda item: self.feature_labels.get(item, item))
        return [self.feature_labels.get(feature, feature) for feature in ordered[:limit]]

    def query_features(self, structured: Dict[str, Any], query_text: str) -> Set[str]:
        structured = structured or {}
        features: Set[str] = set()

        def add(feature: str, label: str, weight: Optional[float] = None) -> None:
            self.remember_feature(feature, label, weight)
            features.add(feature)

        for color in _as_list(structured.get("coat_color")):
            value = _feature_value(color)
            add(f"trait:coat_color:{value}", f"Trait: coat_color={value}")
        for fur in _as_list(structured.get("fur_length")):
            value = _feature_value(fur)
            add(f"trait:fur_length:{value}", f"Trait: fur_length={value}")
        for ear in _as_list(structured.get("ear_shape")):
            value = _feature_value(ear)
            add(f"trait:ear_shape:{value}", f"Trait: ear_shape={value}")
        for size in _as_list(structured.get("body_size_hint")):
            value = _feature_value(size)
            add(f"trait:body_size:{value}", f"Trait: body_size={value}")
        for sex in _as_list(structured.get("sex")):
            value = clean_text(sex).upper()
            add(f"sex:{value}", f"Sex: {value}")
        for age in _as_list(structured.get("age_hint")):
            value = _feature_value(age)
            add(f"age:{value}", f"Age: {value}")

        if structured.get("face_visible") is True:
            add("photo:face_visible", "Photo: face visible")
        elif structured.get("face_visible") is False:
            add("photo:face_hidden", "Photo: face hidden")
        if structured.get("whole_body_visible") is True:
            add("photo:whole_body_visible", "Photo: whole body visible")
        elif structured.get("whole_body_visible") is False:
            add("photo:whole_body_hidden", "Photo: whole body hidden")

        try:
            min_quality = float(structured.get("min_photo_quality"))
        except (TypeError, ValueError):
            min_quality = 0.0
        if min_quality >= 0.7:
            add("photo:quality_good", "Photo: quality >= 0.70")
        elif min_quality > 0:
            add("photo:quality_usable", "Photo: quality >= 0.55")

        text = clean_text(query_text)
        lowered = text.lower()
        for region, hints in REGION_HINTS:
            if _text_contains_any(text, hints):
                add(f"region:{region}", f"Region: {region}", 1.15)
        for label, feature in self.region_features.items():
            if label and label in lowered:
                add(feature, self.feature_labels.get(feature, feature), self.feature_weights.get(feature))
        for label, feature in self.shelter_features.items():
            if label and len(label) >= 3 and label in lowered:
                add(feature, self.feature_labels.get(feature, feature), self.feature_weights.get(feature))
        for label, feature in self.status_features.items():
            if label and label in lowered:
                add(feature, self.feature_labels.get(feature, feature), self.feature_weights.get(feature))

        if "보호중" in text or "입양 가능" in text:
            add("notice:active", "Notice: active/adoptable", 1.1)
        if "사진" in text and any(token in text for token in ("선명", "품질", "잘 보", "또렷")):
            add("photo:quality_good", "Photo: quality >= 0.70")
        if "얼굴" in text and any(token in text for token in ("잘", "보", "확인")):
            add("photo:face_visible", "Photo: face visible")
        if "전신" in text or "몸 전체" in text:
            add("photo:whole_body_visible", "Photo: whole body visible")

        for feature, label, weight, hints in QUERY_TEXT_FEATURE_HINTS:
            if _text_contains_any(text, hints):
                add(feature, label, weight)
        return features

    def candidate_doc_scores(self, structured: Dict[str, Any], query_text: str, limit: int) -> Dict[int, float]:
        features = self.query_features(structured, query_text)
        scores: Dict[int, float] = {}
        for feature in features:
            weight = self.feature_weights.get(feature, _feature_weight(feature))
            for doc_index in self.feature_to_docs.get(feature, set()):
                scores[doc_index] = scores.get(doc_index, 0.0) + weight
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        if limit > 0:
            ranked = ranked[:limit]
        return dict(ranked)

    def match_score(self, doc_index: int, query_features: Set[str]) -> Tuple[float, List[str], List[str]]:
        if not query_features or doc_index < 0 or doc_index >= len(self.doc_features):
            return 0.0, [], []
        doc_features = self.doc_features[doc_index]
        matched = query_features & doc_features
        missing = query_features - doc_features
        total_weight = sum(self.feature_weights.get(feature, _feature_weight(feature)) for feature in query_features) or 1.0
        matched_weight = sum(self.feature_weights.get(feature, _feature_weight(feature)) for feature in matched)
        return min(1.0, matched_weight / total_weight), self.labels_for(matched), self.labels_for(missing)

    def neighbor_score(self, doc_index: int, seed_indices: Sequence[int]) -> Tuple[float, List[str]]:
        if doc_index < 0 or doc_index >= len(self.doc_features):
            return 0.0, []
        base = self.doc_features[doc_index]
        if not base:
            return 0.0, []
        best_score = 0.0
        best_shared: Set[str] = set()
        for seed_index in seed_indices:
            if seed_index == doc_index or seed_index < 0 or seed_index >= len(self.doc_features):
                continue
            other = self.doc_features[seed_index]
            shared = base & other
            if not shared:
                continue
            union = base | other
            shared_weight = sum(self.feature_weights.get(feature, _feature_weight(feature)) for feature in shared)
            union_weight = sum(self.feature_weights.get(feature, _feature_weight(feature)) for feature in union) or 1.0
            score = shared_weight / union_weight
            if score > best_score:
                best_score = score
                best_shared = shared
        return min(1.0, best_score), self.labels_for(best_shared, limit=8)

    def summary(self) -> Dict[str, Any]:
        return {
            "dogs": len(self.docs),
            "features": len(self.feature_to_docs),
            "edges": sum(len(features) for features in self.doc_features),
            "regions": len(self.region_features),
            "shelters": len(self.shelter_features),
            "statuses": len(self.status_features),
            "overlay_matched": self.overlay_matched,
        }

    def _valid_doc_indices(self, doc_indices: Iterable[int]) -> List[int]:
        selected: List[int] = []
        seen: Set[int] = set()
        for doc_index in doc_indices:
            if not isinstance(doc_index, int):
                continue
            if doc_index in seen or doc_index < 0 or doc_index >= len(self.doc_features):
                continue
            selected.append(doc_index)
            seen.add(doc_index)
        return selected

    def feature_counts_for_doc_indices(self, doc_indices: Iterable[int]) -> Counter[str]:
        counts: Counter[str] = Counter()
        for doc_index in self._valid_doc_indices(doc_indices):
            counts.update(self.doc_features[doc_index])
        return counts

    def summary_for_doc_indices(self, doc_indices: Iterable[int]) -> Dict[str, Any]:
        selected = self._valid_doc_indices(doc_indices)
        feature_counts = self.feature_counts_for_doc_indices(selected)
        edges = sum(feature_counts.values())
        dogs = len(selected)
        return {
            "dogs": dogs,
            "features": len(feature_counts),
            "edges": edges,
            "avg_edges_per_dog": round(edges / dogs, 2) if dogs else 0.0,
            "regions": sum(1 for feature in feature_counts if feature.startswith("region:")),
            "shelters": sum(1 for feature in feature_counts if feature.startswith("shelter:")),
            "statuses": sum(1 for feature in feature_counts if feature.startswith("status:")),
            "notice_flags": sum(1 for feature in feature_counts if feature.startswith("notice:")),
        }

    def summary_by_notice_status(self, status_by_doc_index: Dict[int, str]) -> Dict[str, Dict[str, Any]]:
        grouped: DefaultDict[str, List[int]] = defaultdict(list)
        for doc_index, status in status_by_doc_index.items():
            grouped[status or "unknown"].append(doc_index)
        return {
            status: self.summary_for_doc_indices(indices)
            for status, indices in sorted(grouped.items())
        }

    def top_features_for_doc_indices(self, doc_indices: Iterable[int], limit: int = 12) -> List[Dict[str, Any]]:
        counts = self.feature_counts_for_doc_indices(doc_indices)
        ranked = sorted(counts.items(), key=lambda item: (-item[1], self.feature_labels.get(item[0], item[0])))
        return [
            {
                "feature": feature,
                "label": self.feature_labels.get(feature, feature),
                "docs": count,
            }
            for feature, count in ranked[:limit]
        ]


def build_dog_graph(
    docs: List[Dict[str, Any]],
    overlay_metas: Optional[Iterable[Dict[str, Any]]] = None,
) -> DogGraphIndex:
    overlays = _overlay_by_id(overlay_metas)
    doc_features: List[Set[str]] = []
    feature_to_docs: DefaultDict[str, Set[int]] = defaultdict(set)
    feature_labels: Dict[str, str] = {}
    feature_weights: Dict[str, float] = {}
    region_features: Dict[str, str] = {}
    shelter_features: Dict[str, str] = {}
    status_features: Dict[str, str] = {}
    overlay_matched = 0

    def remember(feature: str, label: str, weight: Optional[float] = None) -> None:
        if not feature:
            return
        feature_labels.setdefault(feature, label or feature)
        feature_weights.setdefault(feature, weight if weight is not None else _feature_weight(feature))

    for doc_index, doc in enumerate(docs):
        meta = doc.get("meta") or {}
        doc_id = clean_text(doc.get("doc_id") or meta.get("desertionNo") or meta.get("desertion_no"))
        overlay = overlays.get(doc_id)
        if overlay:
            overlay_matched += 1
            for key, value in overlay.items():
                if _has_value(value) and not _has_value(meta.get(key)):
                    meta[key] = value

        features: Set[str] = set()

        def add(feature: str, label: str, weight: Optional[float] = None) -> None:
            if not feature:
                return
            remember(feature, label, weight)
            features.add(feature)
            feature_to_docs[feature].add(doc_index)

        attrs = normalize_vlm_attrs(meta.get("vlm_attrs"))
        for color in attrs.get("coat_color") or []:
            value = _feature_value(color)
            add(f"trait:coat_color:{value}", f"Trait: coat_color={value}")
        if attrs.get("fur_length") and attrs.get("fur_length") != "unknown":
            value = _feature_value(attrs.get("fur_length"))
            add(f"trait:fur_length:{value}", f"Trait: fur_length={value}")
        if attrs.get("ear_shape") and attrs.get("ear_shape") != "unknown":
            value = _feature_value(attrs.get("ear_shape"))
            add(f"trait:ear_shape:{value}", f"Trait: ear_shape={value}")

        size = clean_text(attrs.get("body_size_hint"))
        if not size or size == "unknown":
            size = infer_size_from_weight(_field(meta, "weight"))
        if size:
            value = _feature_value(size)
            add(f"trait:body_size:{value}", f"Trait: body_size={value}")

        face_visible = attrs.get("face_visible")
        if face_visible is True:
            add("photo:face_visible", "Photo: face visible")
        elif face_visible is False:
            add("photo:face_hidden", "Photo: face hidden")
        whole_body_visible = attrs.get("whole_body_visible")
        if whole_body_visible is True:
            add("photo:whole_body_visible", "Photo: whole body visible")
        elif whole_body_visible is False:
            add("photo:whole_body_hidden", "Photo: whole body hidden")

        quality = attrs.get("photo_quality_score")
        if isinstance(quality, float):
            if quality >= 0.70:
                add("photo:quality_good", "Photo: quality >= 0.70")
                add("photo:quality_usable", "Photo: quality >= 0.55")
            elif quality >= 0.55:
                add("photo:quality_usable", "Photo: quality >= 0.55")
            else:
                add("photo:quality_weak", "Photo: quality < 0.55", 0.35)

        sex = clean_text(_field(meta, "sex")).upper()
        if sex:
            add(f"sex:{sex}", f"Sex: {sex}")
        age_hint = infer_age_hint(_field(meta, "age"))
        if age_hint:
            add(f"age:{age_hint}", f"Age: {age_hint}")
        neuter = clean_text(_field(meta, "neuter")).upper()
        if neuter:
            add(f"neuter:{neuter}", f"Neuter: {neuter}", 0.45)

        breed_code = clean_text(_field(meta, "breed_code"))
        if breed_code:
            add(f"breed:code:{breed_code.lower()}", f"Breed code: {breed_code}", 0.35)
        breed_name = clean_text(_field(meta, "breed_name"))
        if breed_name and not breed_name.isdigit():
            add(f"breed:name:{breed_name.lower()}", f"Breed: {breed_name}", 0.35)

        shelter = clean_text(_field(meta, "shelter"))
        if shelter:
            feature = f"shelter:{shelter.lower()}"
            add(feature, f"Shelter: {shelter}", 1.15)
            shelter_features[shelter.lower()] = feature

        region = infer_region(meta)
        if region:
            feature = f"region:{region}"
            add(feature, f"Region: {region}", 1.15)
            region_features[region.lower()] = feature

        status = clean_text(_field(meta, "status"))
        if status:
            feature = f"status:{status.lower()}"
            add(feature, f"Status: {status}", 1.15)
            status_features[status.lower()] = feature
            if "보호" in status and not any(token in status for token in ("종료", "입양", "반환", "자연사", "안락사")):
                add("notice:active", "Notice: active/adoptable", 1.1)

        doc_text = " ".join(
            clean_text(meta.get(key))
            for key in ("desc", "desc_full", "specialMark", "merged_desc", "vlm_desc", "vlm_attr_text")
        )
        for feature, label, weight, hints in TEXT_FEATURE_HINTS:
            if _text_contains_any(doc_text, hints):
                add(feature, label, weight)

        doc_features.append(features)

    return DogGraphIndex(
        docs=docs,
        doc_features=doc_features,
        feature_to_docs=dict(feature_to_docs),
        feature_labels=feature_labels,
        feature_weights=feature_weights,
        region_features=region_features,
        shelter_features=shelter_features,
        status_features=status_features,
        overlay_matched=overlay_matched,
    )


def rerank_with_graph(
    ranked: List[Dict[str, Any]],
    graph: DogGraphIndex,
    structured: Dict[str, Any],
    query_text: str,
    topk: int,
) -> List[Dict[str, Any]]:
    if not ranked:
        return []

    topk = max(1, int(topk or 10))
    query_features = graph.query_features(structured, query_text)
    graph_seed_scores = graph.candidate_doc_scores(structured, query_text, limit=0)
    max_seed = max(graph_seed_scores.values(), default=0.0) or 1.0
    seed_indices = [
        item.get("doc_index")
        for item in ranked[: max(topk, 8)]
        if isinstance(item.get("doc_index"), int)
    ]

    reranked: List[Dict[str, Any]] = []
    graph_weight = 0.18 if query_features else 0.05
    for item in ranked:
        doc_index = item.get("doc_index")
        if not isinstance(doc_index, int):
            reranked.append(item)
            continue

        condition_score, matched_edges, missing_edges = graph.match_score(doc_index, query_features)
        seed_score = graph_seed_scores.get(doc_index, 0.0) / max_seed
        neighbor_score, similar_edges = graph.neighbor_score(doc_index, seed_indices)
        if query_features:
            graph_score = (0.70 * condition_score) + (0.20 * seed_score) + (0.10 * neighbor_score)
        else:
            graph_score = neighbor_score

        updated = dict(item)
        base_score = float(item.get("score") or 0.0)
        updated["score"] = ((1.0 - graph_weight) * base_score) + (graph_weight * graph_score)

        score_parts = dict(item.get("score_parts") or {})
        score_parts.update(
            {
                "graph": round(graph_score, 4),
                "graph_conditions": round(condition_score, 4),
                "graph_expansion": round(seed_score, 4),
                "graph_similarity": round(neighbor_score, 4),
            }
        )
        updated["score_parts"] = score_parts

        evidence = dict(item.get("evidence") or {})
        evidence.update(
            {
                "graph_query_features": graph.labels_for(query_features),
                "graph_matched_edges": matched_edges,
                "graph_missing_edges": missing_edges,
                "graph_similar_to_edges": similar_edges,
            }
        )
        updated["evidence"] = evidence
        reranked.append(updated)

    reranked.sort(key=lambda item: item["score"], reverse=True)
    return reranked[:topk]