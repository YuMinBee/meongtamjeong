from __future__ import annotations

import pytest

from app.appearance_query import (
    NEUTRAL_DOG_QUERY,
    UNSUPPORTED_NEGATED_APPEARANCE_CODE,
    UNSUPPORTED_NEGATED_APPEARANCE_MESSAGE,
    analyze_appearance_query,
    appearance_query_excludes_nonvisual_terms,
    appearance_query_has_unsupported_negation,
    appearance_search_conditions,
    normalize_appearance_query,
)


@pytest.mark.parametrize(
    ("raw", "kept", "removed"),
    [
        (
            "차분한 흰색 소형견",
            {"흰색", "소형견"},
            {"차분한"},
        ),
        (
            "아파트에서 키우기 좋은 검은 털 중형견",
            {"검정색", "털", "중형견"},
            {"아파트에서", "키우기", "좋은"},
        ),
        (
            "아이보리 털에 귀가 쫑긋한 강아지",
            {"크림색", "털에", "귀가", "쫑긋한", "강아지"},
            set(),
        ),
        (
            "calm apartment-friendly black medium dog",
            {"black", "medium", "dog"},
            {"calm"},
        ),
    ],
)
def test_normalize_appearance_query_keeps_visual_language_only(
    raw: str,
    kept: set[str],
    removed: set[str],
) -> None:
    normalized = normalize_appearance_query(raw)
    tokens = set(normalized.split())

    assert kept <= tokens
    assert not (removed & tokens)


@pytest.mark.parametrize(
    "raw",
    [
        "아이들과 잘 지내고 차분한 강아지",
        "하루 8시간 혼자 있어도 괜찮은 반려견",
        "초보 보호자에게 적합한 조용한 유기견",
        "kid friendly calm apartment dog",
        "아이 없는 집에서 키우기 좋은 온화한 강아지",
    ],
)
def test_nonappearance_only_query_becomes_neutral_dog_search(raw: str) -> None:
    assert normalize_appearance_query(raw) == NEUTRAL_DOG_QUERY


def test_appearance_conditions_drop_personality_and_free_keywords() -> None:
    filtered = appearance_search_conditions(
        {
            "coat_color": ["white"],
            "body_size_hint": ["small"],
            "personality": ["차분"],
            "keywords": ["아이 친화", "복슬복슬"],
            "face_visible": True,
        }
    )

    assert filtered == {
        "coat_color": ["white"],
        "body_size_hint": ["small"],
        "face_visible": True,
    }


def test_exclusion_flag_ignores_visual_only_punctuation_normalization() -> None:
    assert appearance_query_excludes_nonvisual_terms("흰색, 복슬복슬한 소형견") is False
    assert appearance_query_excludes_nonvisual_terms("차분한 흰색 소형견") is True


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("다른 개와 잘 지내는 흰색 강아지", "흰색 강아지"),
        ("고양이와 친하고 사교적인 갈색 소형견", "갈색 소형견"),
        ("산책을 많이 해야 하는 활동적인 검은 중형견", "검정색 중형견"),
    ],
)
def test_household_and_activity_clauses_do_not_leak_into_retrieval(
    raw: str,
    expected: str,
) -> None:
    assert normalize_appearance_query(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("활동성이 많은 검은 강아지", "검정색 강아지"),
        ("산책을 적게 해도 되는 흰색 소형견", "흰색 소형견"),
        ("초보자가 키우기 쉬운 작은 개", "소형견"),
        ("성격이 온순하고 사람을 좋아하는 흰색 장모견", "흰색 장모견"),
        ("high energy black dog", "black dog"),
        ("low activity small dog", "small dog"),
        ("can stay home alone 8 hours brown dog", "brown dog"),
        ("good with cats fluffy dog", "fluffy dog"),
    ],
)
def test_common_nonappearance_phrases_leave_only_visual_terms(
    raw: str,
    expected: str,
) -> None:
    assert normalize_appearance_query(raw) == expected
    assert appearance_query_excludes_nonvisual_terms(raw) is True


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("겁이 별로 없는 강아지", NEUTRAL_DOG_QUERY),
        ("낯선 사람을 무서워하지 않는 강아지", NEUTRAL_DOG_QUERY),
        ("소음에 민감하지 않은 강아지", NEUTRAL_DOG_QUERY),
        ("무던한 성격의 강아지", NEUTRAL_DOG_QUERY),
        ("독립심이 강한 갈색 강아지", "갈색 강아지"),
        ("분리 불안이 없는 흰색 강아지", "흰색 강아지"),
        ("다른 반려견과 문제없이 지내는 검은 강아지", "검정색 강아지"),
        ("아기와 함께 살 수 있는 소형견", "소형견"),
    ],
)
def test_common_negated_or_indirect_behavior_phrases_do_not_leak(
    raw: str,
    expected: str,
) -> None:
    assert normalize_appearance_query(raw) == expected
    assert appearance_query_excludes_nonvisual_terms(raw) is True


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("갈색의 작은 강아지를 보여줘", "갈색 소형견을 찾아줘", "갈색 소형견"),
        ("검은색 중간 크기 강아지", "검정색 중형견", "검정색 중형견"),
        (
            "경기 지역에서 보호 중인 작은 성견",
            "경기도에 있는 소형 성견",
            "경기도 소형견 성견",
        ),
        (
            "몸무게 3~8kg인 다 자란 강아지",
            "3kg에서 8kg 사이의 성견",
            "성견 3kg 이상 8kg 이하",
        ),
        (
            "몸무게 8kg 이하인 나이 든 작은 강아지",
            "8kg 이하의 노령 소형견",
            "소형견 노령견 8kg 이하",
        ),
    ],
)
def test_general_appearance_synonyms_share_canonical_retrieval_text(
    left: str,
    right: str,
    expected: str,
) -> None:
    assert normalize_appearance_query(left) == expected
    assert normalize_appearance_query(right) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("쪼꼬만 밤색 강쥐 있나여??", "갈색 소형견"),
        ("하얀 털에 아주 조그마한 개로요", "흰색 초소형견"),
        ("경기쪽 중간덩치 누런갈색 강아지", "경기도 황갈색 중형견"),
        ("3키로부터 8키로까지, 다 큰 강아지", "성견 3kg 이상 8kg 이하"),
        ("8키로 이하 나이많은 쪼그만 개", "소형견 노령견 8kg 이하"),
    ],
)
def test_v2_observed_colloquial_failures_are_development_regressions(
    raw: str,
    expected: str,
) -> None:
    """Lock fixes informed by the completed one-shot v2 failure analysis."""

    assert normalize_appearance_query(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "오늘 쪼꼬만 가방을 샀어요",
        "밤색 지갑을 찾고 있어요",
        "중간덩치 책상을 주문했어요",
        "나이많은 선배와 만났어요",
        "다 큰 문제를 해결했어요",
        "나는 3키로미터를 달렸어요",
    ],
)
def test_colloquial_dog_rules_do_not_rewrite_general_korean_prose(raw: str) -> None:
    analysis = analyze_appearance_query(raw)

    assert analysis["normalized_query"] == raw
    assert analysis["synonym_normalizations"] == []


@pytest.mark.parametrize(
    ("raw", "expected", "original", "corrected"),
    [
        ("갈색 소형겐을 찾아줘", "갈색 소형견", "소형겐을", "소형견을"),
        (
            "경기도에 있는 황갈섹 중형견",
            "경기도 황갈색 중형견",
            "황갈섹",
            "황갈색",
        ),
        (
            "경기도에 있는 소형 성갼",
            "경기도 소형견 성견",
            "성갼",
            "성견",
        ),
        ("서울의 어링 소형견", "서울 소형견 어린", "어링", "어린"),
    ],
)
def test_unique_domain_one_codepoint_typos_are_corrected_and_reported(
    raw: str,
    expected: str,
    original: str,
    corrected: str,
) -> None:
    analysis = analyze_appearance_query(raw)

    assert analysis["normalized_query"] == expected
    assert any(
        item["original"] == original and item["corrected"] == corrected
        for item in analysis["typo_corrections"]
    )


@pytest.mark.parametrize(
    "raw",
    [
        "성갼",  # two-syllable correction has no separate dog-domain anchor
        "어링",  # ordinary short words are never pulled toward the lexicon alone
        "구형견을 찾아줘",  # equally close to several size terms, so ambiguous
        "오늘 가방을 샀어요",  # general Korean prose
        "검정색 중형견을 보거 싶어",  # generic verb typo is outside the domain lexicon
        "소형견과 중형견 사이",  # already-valid lexicon terms
    ],
)
def test_typo_policy_does_not_overcorrect_short_ambiguous_or_general_text(
    raw: str,
) -> None:
    analysis = analyze_appearance_query(raw)

    assert analysis["typo_corrections"] == []


@pytest.mark.parametrize(
    ("raw", "expected", "field"),
    [
        ("갈색은 아닌 소형견", "소형견", "color"),
        ("경기도가 아닌 지역의 소형 성견", "소형견 성견", "region"),
        ("검정색 말고 흰색 중형견", "흰색 중형견", "color"),
        ("3kg 미만 또는 8kg 초과인 성견", "성견", "weight"),
        ("8kg을 초과하는 노령 소형견", "소형견 노령견", "weight"),
    ],
)
def test_negated_objective_is_removed_instead_of_becoming_positive_filter(
    raw: str,
    expected: str,
    field: str,
) -> None:
    analysis = analyze_appearance_query(raw)

    assert analysis["normalized_query"] == expected
    assert analysis["is_fully_supported"] is False
    assert analysis["warnings"]
    assert appearance_query_has_unsupported_negation(raw) is True
    assert any(
        item["code"] == UNSUPPORTED_NEGATED_APPEARANCE_CODE and item["field"] == field
        for item in analysis["unsupported_conditions"]
    )


@pytest.mark.parametrize(
    ("raw", "expected_fields"),
    [
        ("갈색 소형견은 빼고 찾아줘", {"color", "size"}),
        ("8kg 이하 노령견은 제외해줘", {"weight", "age"}),
    ],
)
def test_compound_negation_drops_the_whole_objective_clause_fail_safe(
    raw: str,
    expected_fields: set[str],
) -> None:
    analysis = analyze_appearance_query(raw)

    assert analysis["normalized_query"] == NEUTRAL_DOG_QUERY
    assert analysis["is_fully_supported"] is False
    assert analysis["warnings"] == [UNSUPPORTED_NEGATED_APPEARANCE_MESSAGE]
    assert {
        item["field"] for item in analysis["unsupported_conditions"]
    } == expected_fields


@pytest.mark.parametrize(
    "raw",
    [
        "8kg 이하의 노령 소형견",
        "차분하지 않은 갈색 소형견",
        "얼굴이 안 보이는 흰색 강아지",
    ],
)
def test_supported_positive_or_nonappearance_negation_has_no_false_warning(
    raw: str,
) -> None:
    analysis = analyze_appearance_query(raw)

    assert analysis["unsupported_conditions"] == []
    assert analysis["warnings"] == []
    assert analysis["is_fully_supported"] is True
