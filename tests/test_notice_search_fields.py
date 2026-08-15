from __future__ import annotations

import pytest

from app.graph_rag import build_dog_graph, infer_region, rerank_with_graph
from app.hybrid_rag import (
    BM25Index,
    build_hybrid_documents,
    condition_score,
    parse_structured_query,
    rank_hybrid_documents,
    resolve_doc_text,
    tokenize,
)


def test_public_notice_fields_are_flattened_for_lexical_search() -> None:
    meta = {
        "desertionNo": "dog-public-fields",
        "breed_source_label": "[개] 믹스견",
        "breed_source": "public_notice_reported",
        "breed_code": "000114",
        "color": "흰색 갈색",
        "health_checks": ["심장사상충", "파보"],
        "vaccinations": ["종합백신", "광견병"],
        "safety_health_note": "외관상 건강",
        "safety_social_note": "사회성 확인 필요",
        "photo_advice": ["얼굴 사진 보강", "전신 사진 보강"],
    }

    text = resolve_doc_text(meta)

    for phrase in (
        "[개] 믹스견",
        "000114",
        "흰색 갈색",
        "심장사상충 파보",
        "종합백신 광견병",
        "외관상 건강",
        "사회성 확인 필요",
        "얼굴 사진 보강 전신 사진 보강",
    ):
        assert phrase in text
    assert "['심장사상충'" not in text

    docs, _ = build_hybrid_documents([meta])
    index = BM25Index(docs)
    assert index.score_doc(tokenize("심장사상충"), 0) > 0
    assert index.score_doc(tokenize("믹스견"), 0) > 0


@pytest.mark.parametrize(
    ("official_name", "expected"),
    [
        ("서울특별시 종로구", "서울"),
        ("경기도 수원시", "경기"),
        ("인천광역시", "인천"),
        ("강원특별자치도 양구군", "강원"),
        ("충청북도 옥천군", "충북"),
        ("충청남도 부여군", "충남"),
        ("대전광역시", "대전"),
        ("세종특별자치시", "세종"),
        ("전북특별자치도 완주군", "전북"),
        ("전라남도 신안군", "전남"),
        ("광주광역시", "광주"),
        ("경상북도 청도군", "경북"),
        ("경상남도 남해군", "경남"),
        ("대구광역시", "대구"),
        ("울산광역시", "울산"),
        ("부산광역시", "부산"),
        ("제주특별자치도", "제주"),
    ],
)
def test_region_inference_accepts_official_province_names(
    official_name: str,
    expected: str,
) -> None:
    assert infer_region({"org_name": official_name}) == expected


def test_generic_dog_word_does_not_silently_add_puppy_or_sex_filters() -> None:
    generic = parse_structured_query("복슬복슬한 검은 강아지")
    shy = parse_structured_query("수줍은 검은 강아지")
    dark_brown = parse_structured_query("암갈색 강아지")

    assert generic["age_hint"] == []
    assert shy["sex"] == []
    assert dark_brown["sex"] == []
    assert parse_structured_query("초소형 강아지")["body_size_hint"] == ["tiny"]
    assert parse_structured_query("초소형 또는 소형견")["body_size_hint"] == [
        "tiny",
        "small",
    ]
    assert parse_structured_query("황갈색 강아지")["coat_color"] == ["tan"]
    assert parse_structured_query("어린 강아지")["age_hint"] == ["puppy"]
    assert parse_structured_query("서울의 어린 소형견")["age_hint"] == ["puppy"]
    assert parse_structured_query("어린이와 지낸 강아지")["age_hint"] == []
    assert parse_structured_query("생후 6개월 강아지")["age_hint"] == ["puppy"]
    assert parse_structured_query("8kg 이하의 노령 소형견")["age_hint"] == ["senior"]
    assert parse_structured_query("수컷 성견")["sex"] == ["M"]
    assert parse_structured_query("수컷 성견")["age_hint"] == ["adult"]


def test_age_condition_uses_notice_age_instead_of_generic_text() -> None:
    puppy_query = parse_structured_query("어린 강아지")
    puppy_doc = {"meta": {"age": "생후 8개월"}, "text": "강아지 보호 공고"}
    senior_doc = {"meta": {"age": "노령견"}, "text": "강아지 보호 공고"}

    puppy_score, puppy_matches, _, _ = condition_score(puppy_doc, puppy_query)
    senior_score, _, _, senior_mismatches = condition_score(senior_doc, puppy_query)

    assert puppy_score == pytest.approx(1.0)
    assert puppy_matches == ["age: puppy"]
    assert senior_score == pytest.approx(0.0)
    assert senior_mismatches == ["age: requested puppy, found senior"]


def test_size_condition_falls_back_to_public_notice_weight_without_vlm() -> None:
    query = parse_structured_query("초소형 강아지")
    tiny_doc = {"meta": {"weight": "2.8(Kg)"}, "text": "보호 공고"}
    medium_doc = {"meta": {"weight": "12(Kg)"}, "text": "보호 공고"}

    tiny_score, tiny_matches, _, _ = condition_score(tiny_doc, query)
    medium_score, _, _, medium_mismatches = condition_score(medium_doc, query)

    assert tiny_score == pytest.approx(1.0)
    assert tiny_matches == ["body_size: tiny"]
    assert medium_score == pytest.approx(0.0)
    assert medium_mismatches == ["body_size: requested tiny, found medium"]


def test_graph_accepts_public_breed_alias_without_behavior_inference() -> None:
    docs = [
        {
            "doc_id": "dog-public-breed",
            "meta": {
                "desertionNo": "dog-public-breed",
                "kindNm": "믹스견",
                "kindCd": "000114",
            },
        }
    ]

    graph = build_dog_graph(docs)
    features = graph.doc_features[0]

    assert "breed:name:믹스견" in features
    assert "breed:code:000114" in features
    assert not any(feature.startswith("temperament:") for feature in features)


def _graph_doc(dog_id: str, breed_name: str, breed_code: str) -> dict[str, object]:
    return {
        "doc_id": dog_id,
        "meta": {
            "desertionNo": dog_id,
            "kindNm": breed_name,
            "kindCd": breed_code,
        },
    }


def test_graph_breed_signal_requires_an_exact_name_or_code_query() -> None:
    docs = [
        _graph_doc("mixed", "믹스견", "000114"),
        _graph_doc("maltese", "말티즈", "000128"),
    ]
    graph = build_dog_graph(docs)

    assert graph.query_features({}, "말티즈를 찾고 있어요") == {"breed:name:말티즈"}
    assert graph.candidate_doc_scores({}, "품종 코드 000128", 10) == {
        1: pytest.approx(0.35)
    }
    assert graph.query_features({}, "말티즈믹스처럼 보이는 개") == set()
    assert graph.query_features({}, "작고 차분한 개") == {"temperament:calm"}

    ranked = [
        {"doc_index": index, "score": 0.5, "doc": doc} for index, doc in enumerate(docs)
    ]
    reranked = rerank_with_graph(ranked, graph, {}, "말티즈", topk=2)

    assert reranked[0]["doc"]["doc_id"] == "maltese"
    assert "Breed: 말티즈" in reranked[0]["evidence"]["graph_matched_edges"]


def test_reported_breed_does_not_affect_unrelated_neighbor_similarity() -> None:
    docs = [
        _graph_doc("mixed-a", "믹스견", "000114"),
        _graph_doc("mixed-b", "믹스견", "000114"),
        _graph_doc("maltese", "말티즈", "000128"),
    ]
    graph = build_dog_graph(docs)
    ranked = [
        {"doc_index": index, "score": 0.5, "doc": doc} for index, doc in enumerate(docs)
    ]

    for query in ("", "보호소 공고를 보여줘"):
        reranked = rerank_with_graph(ranked, graph, {}, query, topk=3)
        assert all(item["score"] == pytest.approx(0.475) for item in reranked)
        assert all(
            not any(
                edge.startswith(("Breed:", "Breed code:"))
                for edge in item["evidence"]["graph_similar_to_edges"]
            )
            for item in reranked
        )


def test_explicit_region_can_outrank_visual_similarity_from_another_region() -> None:
    docs = [
        {
            "doc_id": "jeju",
            "meta": {"desertionNo": "jeju", "org_name": "제주특별자치도"},
        },
        {
            "doc_id": "seoul",
            "meta": {"desertionNo": "seoul", "org_name": "서울특별시"},
        },
    ]
    graph = build_dog_graph(docs)
    ranked = [
        {"doc_index": 1, "score": 0.6, "doc": docs[1]},
        {"doc_index": 0, "score": 0.45, "doc": docs[0]},
    ]

    reranked = rerank_with_graph(ranked, graph, {}, "제주 강아지", topk=2)

    assert reranked[0]["doc"]["doc_id"] == "jeju"
    assert "Region: 제주" in reranked[0]["evidence"]["graph_matched_edges"]


def test_graph_candidate_prior_survives_first_stage_candidate_truncation() -> None:
    docs = [
        {"doc_id": "visual", "meta": {}, "text": ""},
        {"doc_id": "explicit", "meta": {}, "text": ""},
    ]
    ranked = rank_hybrid_documents(
        docs=docs,
        bm25=BM25Index(docs),
        structured={},
        query_text="",
        vector_scores={0: 0.6, 1: 0.45},
        topk=1,
        extra_candidate_scores={1: 1.0},
    )

    assert ranked[0]["doc"]["doc_id"] == "explicit"
    assert ranked[0]["score_parts"]["candidate_prior"] == pytest.approx(1.0)


def test_appearance_graph_rerank_excludes_temperament_neighbor_edges() -> None:
    docs = [
        {
            "doc_id": "calm-a",
            "meta": {
                "desertionNo": "calm-a",
                "specialMark": "차분한 성격",
            },
        },
        {
            "doc_id": "calm-b",
            "meta": {
                "desertionNo": "calm-b",
                "specialMark": "차분한 성격",
            },
        },
        {
            "doc_id": "no-temperament",
            "meta": {"desertionNo": "no-temperament"},
        },
    ]
    graph = build_dog_graph(docs)
    ranked = [
        {"doc_index": index, "score": 0.5, "doc": doc} for index, doc in enumerate(docs)
    ]

    legacy = rerank_with_graph(ranked, graph, {}, "", topk=3)
    appearance = rerank_with_graph(
        ranked,
        graph,
        {},
        "",
        topk=3,
        excluded_feature_prefixes=("temperament:",),
    )

    assert (
        graph.candidate_doc_scores(
            {},
            "차분한 강아지",
            3,
            excluded_feature_prefixes=("temperament:",),
        )
        == {}
    )
    assert legacy[0]["score"] > legacy[-1]["score"]
    assert all(item["score"] == pytest.approx(0.475) for item in appearance)
    assert all(
        not any(
            edge.startswith("Trait:")
            for edge in item["evidence"]["graph_similar_to_edges"]
        )
        for item in appearance
    )


def test_ambiguous_or_negative_words_do_not_create_temperament_edges() -> None:
    docs = [
        {
            "doc_id": "negative-context",
            "meta": {
                "desertionNo": "negative-context",
                "desc": "사람을 무서워하고 산책 경험 없음",
            },
        },
        {
            "doc_id": "explicit-context",
            "meta": {
                "desertionNo": "explicit-context",
                "desc": "사람을 좋아하고 활동량 많음",
            },
        },
        {
            "doc_id": "vlm-only-context",
            "meta": {
                "desertionNo": "vlm-only-context",
                "desc": "사람을 좋아하고 활동량 많음",
                "vlm_desc": "사진을 보고 생성한 표시용 설명",
            },
        },
    ]
    graph = build_dog_graph(docs)

    assert "temperament:friendly" not in graph.doc_features[0]
    assert "temperament:active" not in graph.doc_features[0]
    assert "temperament:friendly" in graph.doc_features[1]
    assert "temperament:active" in graph.doc_features[1]
    assert not any(
        feature.startswith("temperament:") for feature in graph.doc_features[2]
    )
    assert not any(
        feature.startswith("temperament:")
        for feature in graph.query_features({}, "아이 있는 가족")
    )
