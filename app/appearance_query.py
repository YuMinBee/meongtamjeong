"""Policy helpers for appearance-only search.

The regular hybrid search intentionally accepts broad natural-language input.
The contest demo has a narrower contract: personality and household-fit
preferences must become shelter questions, never retrieval signals.  These
helpers enforce that narrower contract without changing the legacy/profile
search behavior.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from typing import Any


NEUTRAL_DOG_QUERY = "강아지"

APPEARANCE_SEARCH_CONDITION_KEYS = frozenset(
    {
        "coat_color",
        "fur_length",
        "ear_shape",
        "body_size_hint",
        "sex",
        "age_hint",
        "face_visible",
        "whole_body_visible",
        "min_photo_quality",
    }
)

# Public-notice temperament edges remain available to the legacy search flow,
# but must not influence the contest-facing appearance-only rank, even as an
# indirect graph-neighbor similarity signal.
APPEARANCE_GRAPH_EXCLUDED_FEATURE_PREFIXES = ("temperament:",)

_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]+")
_GENERIC_DOG_TOKENS = frozenset(
    {
        "개",
        "개의",
        "강아지",
        "강아지를",
        "유기견",
        "유기견을",
        "반려견",
        "반려견을",
        "dog",
        "dogs",
        "shelterdog",
        "shelterdogs",
        "찾아줘",
        "찾고",
        "싶어",
        "싶어요",
        "원해",
        "원해요",
        "추천",
        "추천해줘",
        "보여줘",
        "please",
        "find",
        "show",
        "want",
    }
)

_KOREAN_NON_APPEARANCE_PREFIXES = (
    # Temperament, social behaviour, and training.
    "차분",
    "조용",
    "온순",
    "온화",
    "순한",
    "순하",
    "얌전",
    "착한",
    "착하",
    "친화",
    "사람친화",
    "활발",
    "발랄",
    "명랑",
    "활동적",
    "활동량",
    "활동성",
    "에너지",
    "애교",
    "다정",
    "느긋",
    "사교",
    "친근",
    "장난",
    "호기심",
    "예민",
    "낯가림",
    "영리",
    "똑똑",
    "충성",
    "독립적",
    "독립성",
    "독립심",
    "겁",
    "소심",
    "낯선",
    "무서",
    "소음",
    "민감",
    "무던",
    "경계",
    "공격",
    "사회성",
    "성격",
    "성향",
    "기질",
    "분리불안",
    "분리",
    "불안",
    "짖음",
    "짖지",
    "훈련",
    "배변",
    # Housing, absence, and guardian experience.
    "아파트",
    "빌라",
    "원룸",
    "오피스텔",
    "주택",
    "단독주택",
    "공동주택",
    "마당",
    "엘리베이터",
    "계단",
    "직장인",
    "1인가구",
    "가구",
    "실내생활",
    "실외생활",
    "부재",
    "외출",
    "혼자",
    "초보",
    "첫반려견",
    "반려경험",
    "반려견경험",
    "입양경험",
    "경험자",
    # Children and other animals.
    "아동",
    "어린이",
    "유아",
    "아기",
    "반려동물",
    "다른동물",
    "다른반려동물",
    "고양이",
    "다견",
    "다묘",
    "합사",
    "문제없",
    "산책",
    "운동",
    "놀이",
)

_KOREAN_COMPATIBILITY_PREFIXES = (
    "잘지내",
    "지내",
    "어울리",
    "적합",
    "가능",
    "함께살",
    "키우",
    "살기",
    "괜찮",
    "좋은",
    "좋아",
    "가족",
    "가정",
    "보호자",
    "사람",
    "쉬운",
    "쉽",
)

_ENGLISH_NON_APPEARANCE = frozenset(
    {
        "active",
        "activity",
        "affectionate",
        "aggressive",
        "aggression",
        "anxiety",
        "alone",
        "apartment",
        "behavior",
        "behaviour",
        "bark",
        "barking",
        "baby",
        "calm",
        "cat",
        "cats",
        "child",
        "children",
        "family",
        "fearful",
        "friendly",
        "gentle",
        "docile",
        "easygoing",
        "energetic",
        "energy",
        "exercise",
        "exercises",
        "exercising",
        "good",
        "home",
        "house",
        "independent",
        "intelligent",
        "kid",
        "kids",
        "infant",
        "lifestyle",
        "maintenance",
        "noise",
        "owner",
        "personality",
        "playful",
        "pet",
        "pets",
        "quiet",
        "loyal",
        "obedient",
        "outgoing",
        "shy",
        "sensitive",
        "separation",
        "social",
        "sociable",
        "smart",
        "temperament",
        "toddler",
        "training",
        "wary",
        "walk",
        "walking",
        "walks",
        "with",
    }
)

_ENGLISH_NON_APPEARANCE_PREFIXES = (
    "childfriendly",
    "experiencedowner",
    "firsttimeowner",
    "goodwith",
    "kidfriendly",
    "petfriendly",
)

_CHILD_TOKEN_RE = re.compile(r"^아이(?:$|들|와|과|랑|가|는|를|에게|도|있는|없는)")
_ABSENCE_CONTEXT_RE = re.compile(
    r"(?:부재|외출|혼자|집을\s*비우|집\s*비우|home\s*alone|absence)",
    re.IGNORECASE,
)
_EXPERIENCE_CONTEXT_RE = re.compile(
    r"(?:초보|처음\s*(?:키우|입양)|반려견?\s*경험|first[- ]?time\s+owner|experienced\s+owner)",
    re.IGNORECASE,
)
_HOUSEHOLD_FIT_CONTEXT_RE = re.compile(
    r"(?:아이(?!보리)|아동|어린이|유아|아기|다른\s*(?:개|강아지|반려견|동물|반려동물)|"
    r"고양이|반려동물|합사|child|kid|cat|other\s+(?:dog|pet|animal))",
    re.IGNORECASE,
)
_BEHAVIOR_CONTEXT_RE = re.compile(
    r"(?:성격|성향|기질|겁|낯선|무서|소음|민감|무던|독립|분리\s*불안|"
    r"문제없이|경계|공격|짖|훈련|배변|사람|temperament|behavio(?:u)?r|"
    r"anxiety|bark|noise|sensitive|stranger)",
    re.IGNORECASE,
)
_ACTIVITY_CONTEXT_RE = re.compile(
    r"(?:활동|산책|운동|놀이|에너지|activity|active|energy|energetic|exercise|walk)",
    re.IGNORECASE,
)

# The aliases below are intentionally small and objective.  They normalize only
# notice/search facts (region, visible appearance, sex, and age/weight hints),
# never temperament or household suitability.  Canonical Korean terms are also
# understood by the existing structured-query parser.
_CONCEPT_ORDER = ("region", "color", "size", "age", "fur", "ear", "sex")
_APPEARANCE_CONCEPT_GROUPS: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "region": (
        ("서울", ("서울", "서울시", "서울특별시")),
        ("부산", ("부산", "부산시", "부산광역시")),
        ("대구", ("대구", "대구시", "대구광역시")),
        ("인천", ("인천", "인천시", "인천광역시")),
        ("광주", ("광주", "광주시", "광주광역시")),
        ("대전", ("대전", "대전시", "대전광역시")),
        ("울산", ("울산", "울산시", "울산광역시")),
        ("세종", ("세종", "세종시", "세종특별자치시")),
        ("경기도", ("경기", "경기도", "경기쪽")),
        ("강원도", ("강원", "강원도", "강원특별자치도")),
        ("충북", ("충북", "충청북도")),
        ("충남", ("충남", "충청남도")),
        ("전북", ("전북", "전라북도", "전북특별자치도")),
        ("전남", ("전남", "전라남도")),
        ("경북", ("경북", "경상북도")),
        ("경남", ("경남", "경상남도")),
        ("제주", ("제주", "제주도", "제주특별자치도")),
    ),
    "color": (
        ("흰색", ("흰색", "흰", "하얀", "하얀색", "백색", "화이트")),
        (
            "검정색",
            ("검정색", "검은", "검은색", "까만", "까만색", "흑색", "블랙"),
        ),
        # Preserve the already-public safety-contract literal while still
        # canonicalizing adjective/swatch synonyms above.
        ("검정", ("검정",)),
        ("갈색", ("갈색", "브라운")),
        ("황갈색", ("황갈색", "황갈빛", "탄색")),
        ("크림색", ("크림색", "크림빛", "아이보리")),
        ("회색", ("회색", "잿빛", "그레이")),
        ("얼룩무늬", ("얼룩무늬", "얼룩", "점박이", "반점무늬")),
    ),
    "size": (
        ("초소형견", ("초소형견", "초소형", "미니견")),
        ("소형견", ("소형견", "소형", "작은", "작고")),
        ("중형견", ("중형견", "중형")),
        ("대형견", ("대형견", "대형")),
    ),
    "age": (
        ("어린", ("어린", "새끼", "퍼피")),
        ("성견", ("성견", "성체견")),
        ("노령견", ("노령견", "노령", "노견", "고령견", "시니어견")),
    ),
    "fur": (
        ("단모", ("단모", "짧은털")),
        ("장모", ("장모", "긴털")),
        ("곱슬털", ("곱슬털", "곱슬")),
        ("복슬복슬", ("복슬복슬", "복슬", "풍성한털")),
    ),
    # Ear phrases are deliberately left as free visual text.  The current
    # parser distinguishes word order (``귀가 선``/``처진 귀``), so forcing a
    # compact canonical token here could silently lose a structured signal.
    "ear": (),
    "sex": (
        ("수컷", ("수컷", "남아")),
        ("암컷", ("암컷", "여아")),
    ),
}

_PHRASE_NORMALIZATION_RULES: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (re.compile(r"아주\s+작(?:은|고)"), "초소형견", "size"),
    (re.compile(r"중간\s+크기(?:의|인)?"), "중형견", "size"),
    (re.compile(r"다\s+큰\s+(?:강아지|개)"), "성견", "age"),
    (re.compile(r"큰\s+(?:강아지|개|성견)"), "대형견", "size"),
    (re.compile(r"다\s+자란(?:\s+(?:강아지|개))?"), "성견", "age"),
    (re.compile(r"나이\s+든(?:\s+(?:강아지|개))?"), "노령견", "age"),
    (re.compile(r"연한\s+크림(?:빛|색)"), "크림색", "color"),
    (re.compile(r"짧은\s+털"), "단모", "fur"),
    (re.compile(r"긴\s+털"), "장모", "fur"),
)

# These development-case colloquialisms are intentionally gated by an explicit
# dog noun.  That keeps phrases such as ``쪼꼬만 가방`` in ordinary Korean
# prose from being silently converted into a dog-size constraint.
_COLLOQUIAL_DOG_CONTEXT_RE = re.compile(
    r"(?:강아지|유기견|반려견|강쥐|댕댕이|(?:^|\s)개(?:로요)?(?:\s|$))"
)
_COLLOQUIAL_PHRASE_NORMALIZATION_RULES: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (re.compile(r"쪼꼬만"), "소형견", "size"),
    (re.compile(r"아주\s+조그마한"), "초소형견", "size"),
    (re.compile(r"쪼그만"), "소형견", "size"),
    (re.compile(r"중간\s*덩치(?:의|인)?"), "중형견", "size"),
    (re.compile(r"나이\s*많은"), "노령견", "age"),
    (re.compile(r"누런\s*갈색"), "황갈색", "color"),
    (re.compile(r"밤색"), "갈색", "color"),
    (re.compile(r"하얀\s+털(?:에|인)?"), "흰색", "color"),
)

_WEIGHT_RANGE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?P<low>\d+(?:[.]\d+)?)\s*(?:kg|킬로그램|키로)?\s*[~～-]\s*"
        r"(?P<high>\d+(?:[.]\d+)?)\s*(?:kg|킬로그램|키로)"
        r"(?=$|\s|[,.)~～-]|인|의|은|는|이|가|을|를)"
    ),
    re.compile(
        r"(?P<low>\d+(?:[.]\d+)?)\s*(?:kg|킬로그램|키로)(?:에서|부터)\s*"
        r"(?P<high>\d+(?:[.]\d+)?)\s*(?:kg|킬로그램|키로)\s*"
        r"(?:까지|사이(?:의|인)?)"
    ),
)
_WEIGHT_UNIT_PATTERN = re.compile(
    r"(?P<value>\d+(?:[.]\d+)?)\s*(?P<unit>킬로그램|키로)(?=$|\s|[,.)~～-])"
)

_KOREAN_PARTICLE_SUFFIXES = (
    "에게서",
    "으로",
    "에서",
    "까지",
    "부터",
    "처럼",
    "보다",
    "하고",
    "이며",
    "이고",
    "에게",
    "랑",
    "인",
    "의",
    "에",
    "이",
    "가",
    "은",
    "는",
    "을",
    "를",
    "와",
    "과",
    "도",
    "만",
)

_QUERY_FILLER_TOKENS = frozenset(
    {
        "찾아줘",
        "찾아주세요",
        "보여줘",
        "보여주세요",
        "보고",
        "싶어",
        "싶어요",
        "추천",
        "추천해줘",
        "원해",
        "원해요",
        "지역",
        "지역의",
        "지역에서",
        "곳에서",
        "보호",
        "중",
        "중인",
        "중에서",
        "있는",
        "인",
        "아이",
        "몸무게",
        "몸무게가",
        "체중",
        "또는",
        "그리고",
        "있나여",
        "개로요",
    }
)

_DOG_GENERIC_ROOTS = frozenset(
    {"개", "강아지", "강쥐", "댕댕이", "유기견", "반려견", "dog", "dogs"}
)
_DOG_DENOTING_CONCEPTS = frozenset(
    {"초소형견", "소형견", "중형견", "대형견", "성견", "노령견"}
)

_TYPO_TARGETS = frozenset(
    {
        "강아지",
        "초소형견",
        "소형견",
        "중형견",
        "대형견",
        "검정색",
        "황갈색",
        "크림색",
        "아이보리",
        "얼룩무늬",
        "노령견",
        "경기도",
        "전라북도",
        "경상남도",
    }
)
_SHORT_CONTEXTUAL_TYPO_TARGETS = frozenset(
    {"흰색", "갈색", "성견", "어린", "노령", "단모", "장모"}
)
_DOG_DOMAIN_ANCHOR_RE = re.compile(
    r"(?:강아지|유기견|반려견|초?소형|중형|대형|노령|성견|보호\s*중|"
    r"\d+(?:[.]\d+)?\s*(?:kg|킬로그램))",
    re.IGNORECASE,
)

_NEGATION_MARKER = r"(?:아닌|아니고|아니라|제외(?:해(?:줘|주세요)?|하고|한|할)?|빼고|말고|원하지\s*않(?:는|은|아|아요)?|싫(?:은|어|어요)?)"
_CANONICAL_NEGATABLE_TERMS = tuple(
    sorted(
        {
            canonical
            for category in _CONCEPT_ORDER
            for canonical, _aliases in _APPEARANCE_CONCEPT_GROUPS[category]
        },
        key=len,
        reverse=True,
    )
)
_NEGATABLE_TERM_PATTERN = "|".join(
    re.escape(term) for term in _CANONICAL_NEGATABLE_TERMS
)
_NEGATED_OBJECTIVE_RE = re.compile(
    rf"(?P<expression>(?P<term>{_NEGATABLE_TERM_PATTERN})"
    rf"(?:이|가|은|는|을|를|도)?\s*{_NEGATION_MARKER})"
)
_PREFIX_NEGATED_OBJECTIVE_RE = re.compile(
    rf"(?P<expression>(?:제외(?:할|한)?|빼고|말고)\s*"
    rf"(?P<term>{_NEGATABLE_TERM_PATTERN})(?:이|가|은|는|을|를)?)"
)
_NEGATED_WEIGHT_RE = re.compile(
    r"(?P<expression>\d+(?:[.]\d+)?\s*(?:kg|킬로그램)(?:이|가|은|는|을|를)?\s*"
    r"(?P<marker>미만|초과)(?:인|하는|의)?)",
    re.IGNORECASE,
)
_OBJECTIVE_SEQUENCE_TOKEN_PATTERN = (
    rf"(?:{_NEGATABLE_TERM_PATTERN}|\d+(?:[.]\d+)?(?:kg|킬로그램)|"
    r"이상|이하|미만|초과)"
)
_NEGATED_OBJECTIVE_SEQUENCE_RE = re.compile(
    rf"(?P<expression>(?P<terms>{_OBJECTIVE_SEQUENCE_TOKEN_PATTERN}"
    rf"(?:\s+{_OBJECTIVE_SEQUENCE_TOKEN_PATTERN})*)\s*{_NEGATION_MARKER})"
)

UNSUPPORTED_NEGATED_APPEARANCE_CODE = "unsupported_negated_appearance_condition"
UNSUPPORTED_NEGATED_APPEARANCE_MESSAGE = (
    "부정 외형 조건은 아직 지원하지 않아 해당 조건을 검색 신호에서 제외했습니다. "
    "원하는 조건을 긍정형으로 다시 입력해 주세요."
)


def _compact_token(token: str) -> str:
    return re.sub(r"[-_]", "", token.casefold())


def _is_non_appearance_token(
    token: str,
    *,
    absence_context: bool,
    experience_context: bool,
    household_fit_context: bool,
    activity_context: bool,
    behavior_context: bool,
) -> bool:
    lowered = token.casefold()
    compact = _compact_token(token)

    if lowered in _ENGLISH_NON_APPEARANCE:
        return True
    if any(compact.startswith(prefix) for prefix in _ENGLISH_NON_APPEARANCE_PREFIXES):
        return True
    if any(token.startswith(prefix) for prefix in _KOREAN_NON_APPEARANCE_PREFIXES):
        return True
    if _CHILD_TOKEN_RE.match(token):
        return True
    if any(token.startswith(prefix) for prefix in _KOREAN_COMPATIBILITY_PREFIXES):
        return True
    if token == "잘":
        return True
    if token == "다른" or token.startswith(
        ("개와", "개랑", "개에게", "강아지와", "강아지랑", "강아지에게")
    ):
        return True

    if household_fit_context and token.startswith(
        (
            "있는",
            "없는",
            "함께",
            "같이",
            "동거",
            "친한",
            "친하",
            "사이좋",
            "집",
            "반려견",
            "문제없",
        )
    ):
        return True
    if household_fit_context and token in {"살", "수"}:
        return True
    if behavior_context and (
        token.startswith(("별로", "없는", "않", "강한", "약한"))
        or lowered in {"not", "without", "does", "doesn"}
    ):
        return True
    if activity_context and (
        token.startswith(
            (
                "매일",
                "많",
                "적",
                "높",
                "낮",
                "보통",
                "해야",
                "해도",
                "하는",
                "되는",
                "필요",
            )
        )
        or lowered
        in {
            "high",
            "low",
            "medium",
            "moderate",
            "daily",
            "much",
            "many",
            "lot",
            "lots",
            "need",
            "needs",
            "require",
            "requires",
        }
        or re.fullmatch(r"\d+(?:시간|회)?", token)
    ):
        return True

    if absence_context and (
        token in {"집", "집을", "비워", "비우는", "비운"}
        or token.startswith(
            ("하루", "시간", "동안", "장시간", "오랜", "있어", "지낼", "견딜")
        )
        or lowered
        in {
            "can",
            "stay",
            "stays",
            "hour",
            "hours",
            "day",
            "days",
            "daily",
            "long",
            "time",
            "away",
        }
        or re.fullmatch(r"\d+(?:시간)?", token)
    ):
        return True
    if experience_context:
        if token in {
            "처음",
            "경험",
            "경험이",
            "경험은",
            "없어",
            "없어요",
            "있어",
            "있어요",
        }:
            return True
        if lowered in {
            "first",
            "time",
            "new",
            "beginner",
            "experienced",
            "inexperienced",
            "easy",
        }:
            return True
    if household_fit_context and lowered in {
        "get",
        "gets",
        "along",
        "live",
        "lives",
        "living",
        "can",
        "suitable",
    }:
        return True
    return False


def _nonappearance_context(text: str) -> dict[str, bool]:
    """Return the shared context switches used by the safety filter."""

    return {
        "absence_context": bool(_ABSENCE_CONTEXT_RE.search(text)),
        "experience_context": bool(_EXPERIENCE_CONTEXT_RE.search(text)),
        "household_fit_context": bool(_HOUSEHOLD_FIT_CONTEXT_RE.search(text)),
        "activity_context": bool(_ACTIVITY_CONTEXT_RE.search(text)),
        "behavior_context": bool(_BEHAVIOR_CONTEXT_RE.search(text)),
    }


def _contains_nonappearance_token(text: str) -> bool:
    context = _nonappearance_context(text)
    return any(
        _is_non_appearance_token(token, **context) for token in _TOKEN_RE.findall(text)
    )


_CONCEPT_ALIAS_LOOKUP: dict[str, tuple[str, str]] = {}
_CANONICAL_CATEGORY_BY_TERM: dict[str, str] = {}
for _category in _CONCEPT_ORDER:
    for _canonical, _aliases in _APPEARANCE_CONCEPT_GROUPS[_category]:
        _CANONICAL_CATEGORY_BY_TERM[_canonical] = _category
        for _alias in (*_aliases, _canonical):
            _CONCEPT_ALIAS_LOOKUP[_alias.casefold()] = (_category, _canonical)


def _concept_token_match(token: str) -> tuple[str, str, str] | None:
    """Return category, canonical term, and matched root for one token."""

    lowered = token.casefold()
    direct = _CONCEPT_ALIAS_LOOKUP.get(lowered)
    if direct is not None:
        return (*direct, token)
    for suffix in _KOREAN_PARTICLE_SUFFIXES:
        if not token.endswith(suffix) or len(token) <= len(suffix):
            continue
        root = token[: -len(suffix)]
        matched = _CONCEPT_ALIAS_LOOKUP.get(root.casefold())
        if matched is not None:
            return (*matched, root)
    return None


def _generic_dog_root(token: str) -> str | None:
    lowered = token.casefold()
    if lowered in _DOG_GENERIC_ROOTS:
        return lowered
    for suffix in _KOREAN_PARTICLE_SUFFIXES:
        if not token.endswith(suffix) or len(token) <= len(suffix):
            continue
        root = token[: -len(suffix)].casefold()
        if root in _DOG_GENERIC_ROOTS:
            return root
    return None


def _comparator_root(token: str) -> str | None:
    if token in {"이상", "이하"}:
        return token
    for root in ("이상", "이하"):
        for suffix in _KOREAN_PARTICLE_SUFFIXES:
            if token == root + suffix:
                return root
    return None


def _substitution_distance(left: str, right: str) -> int:
    if len(left) != len(right):
        return max(len(left), len(right))
    return sum(left_char != right_char for left_char, right_char in zip(left, right))


def _typo_root_candidates(token: str) -> list[tuple[str, str]]:
    candidates = [(token, "")]
    for suffix in _KOREAN_PARTICLE_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 2:
            candidates.append((token[: -len(suffix)], suffix))
    return candidates


def _correct_conservative_typos(text: str) -> tuple[str, list[dict[str, str]]]:
    """Correct unique domain-lexicon substitutions, never general prose.

    Only same-length, one-codepoint substitutions are accepted.  Two-syllable
    targets require a separate dog-domain anchor elsewhere in the query; this
    prevents words in ordinary Korean sentences from being pulled toward the
    dog-search lexicon.
    """

    has_domain_anchor = bool(_DOG_DOMAIN_ANCHOR_RE.search(text))
    corrections: list[dict[str, str]] = []

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        if not re.fullmatch(r"[가-힣]+", token):
            return token
        if _concept_token_match(token) is not None or _generic_dog_root(token):
            return token

        corrected_surfaces: set[tuple[str, str]] = set()
        for root, suffix in _typo_root_candidates(token):
            if not re.fullmatch(r"[가-힣]+", root):
                continue
            if (
                any(
                    root.startswith(prefix)
                    for prefix in _KOREAN_NON_APPEARANCE_PREFIXES
                )
                or any(
                    root.startswith(prefix) for prefix in _KOREAN_COMPATIBILITY_PREFIXES
                )
                or root.casefold() in _QUERY_FILLER_TOKENS
            ):
                continue
            allowed_targets = set(_TYPO_TARGETS)
            if has_domain_anchor and len(root) == 2:
                allowed_targets.update(_SHORT_CONTEXTUAL_TYPO_TARGETS)
            for target in allowed_targets:
                if len(root) != len(target):
                    continue
                if _substitution_distance(root, target) == 1:
                    corrected_surfaces.add((target + suffix, target))
        if len(corrected_surfaces) != 1:
            return token
        corrected, target = corrected_surfaces.pop()
        corrections.append(
            {
                "original": token,
                "corrected": corrected,
                "canonical_target": target,
                "policy": "unique_one_codepoint_substitution",
            }
        )
        return corrected

    return _TOKEN_RE.sub(replace, text), corrections


def _normalize_phrases(
    text: str,
) -> tuple[str, list[dict[str, str]]]:
    normalizations: list[dict[str, str]] = []

    def replace_weight(match: re.Match[str]) -> str:
        replacement = f"{match.group('low')}kg 이상 {match.group('high')}kg 이하"
        normalizations.append(
            {
                "original": match.group(0),
                "canonical": replacement,
                "field": "weight",
            }
        )
        return replacement

    for pattern in _WEIGHT_RANGE_PATTERNS:
        text = pattern.sub(replace_weight, text)

    def replace_weight_unit(match: re.Match[str]) -> str:
        replacement = f"{match.group('value')}kg"
        normalizations.append(
            {
                "original": match.group(0),
                "canonical": replacement,
                "field": "weight",
            }
        )
        return replacement

    text = _WEIGHT_UNIT_PATTERN.sub(replace_weight_unit, text)

    phrase_rules = list(_PHRASE_NORMALIZATION_RULES)
    if _COLLOQUIAL_DOG_CONTEXT_RE.search(text):
        phrase_rules.extend(_COLLOQUIAL_PHRASE_NORMALIZATION_RULES)

    for pattern, replacement, field in phrase_rules:

        def replace_phrase(
            match: re.Match[str],
            *,
            canonical: str = replacement,
            concept_field: str = field,
        ) -> str:
            original = match.group(0)
            if original != canonical:
                normalizations.append(
                    {
                        "original": original,
                        "canonical": canonical,
                        "field": concept_field,
                    }
                )
            return canonical

        text = pattern.sub(replace_phrase, text)
    return text, normalizations


def _canonicalize_surface_tokens(
    text: str,
    normalizations: list[dict[str, str]],
) -> str:
    tokens: list[str] = []
    for token in _TOKEN_RE.findall(text):
        match = _concept_token_match(token)
        if match is None:
            tokens.append(token)
            continue
        category, canonical, matched_root = match
        if matched_root.casefold() != canonical.casefold():
            normalizations.append(
                {
                    "original": matched_root,
                    "canonical": canonical,
                    "field": category,
                }
            )
        tokens.append(canonical)
    return " ".join(tokens)


def _remove_unsupported_negations(
    text: str,
) -> tuple[str, list[dict[str, str]]]:
    unsupported: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def record(*, expression: str, field: str, term: str) -> None:
        key = (field, expression)
        if key in seen:
            return
        seen.add(key)
        unsupported.append(
            {
                "code": UNSUPPORTED_NEGATED_APPEARANCE_CODE,
                "field": field,
                "term": term,
                "expression": expression,
                "message": UNSUPPORTED_NEGATED_APPEARANCE_MESSAGE,
            }
        )

    def remove_objective(match: re.Match[str]) -> str:
        term = match.group("term")
        record(
            expression=match.group("expression"),
            field=_CANONICAL_CATEGORY_BY_TERM.get(term, "appearance"),
            term=term,
        )
        return " "

    def remove_objective_sequence(match: re.Match[str]) -> str:
        """Remove every adjacent objective in one unsupported negated clause.

        A request such as ``갈색 소형견은 빼고`` negates the color-size
        combination, not merely the last token.  Keeping either token as a
        positive signal is less safe than dropping the complete, contiguous
        objective sequence and asking the user for a positive formulation.
        """

        expression = match.group("expression")
        terms = match.group("terms")
        weight_terms: list[str] = []
        for token in _TOKEN_RE.findall(terms):
            category = _CANONICAL_CATEGORY_BY_TERM.get(token)
            if category is not None:
                record(expression=expression, field=category, term=token)
                continue
            if re.fullmatch(
                r"\d+(?:[.]\d+)?(?:kg|킬로그램)|이상|이하|미만|초과",
                token,
                re.IGNORECASE,
            ):
                weight_terms.append(token)
        if weight_terms:
            record(
                expression=expression,
                field="weight",
                term=" ".join(weight_terms),
            )
        return " "

    text = _NEGATED_OBJECTIVE_SEQUENCE_RE.sub(remove_objective_sequence, text)
    text = _NEGATED_OBJECTIVE_RE.sub(remove_objective, text)
    text = _PREFIX_NEGATED_OBJECTIVE_RE.sub(remove_objective, text)

    def remove_weight(match: re.Match[str]) -> str:
        record(
            expression=match.group("expression"),
            field="weight",
            term=match.group(0),
        )
        return " "

    text = _NEGATED_WEIGHT_RE.sub(remove_weight, text)
    return text, unsupported


def _assemble_normalized_query(text: str) -> tuple[str, bool]:
    context = _nonappearance_context(text)
    concepts: dict[str, list[str]] = {category: [] for category in _CONCEPT_ORDER}
    unknown: list[str] = []
    dog_label: str | None = None
    nonappearance_excluded = False

    for token in _TOKEN_RE.findall(text):
        if _is_non_appearance_token(token, **context):
            nonappearance_excluded = True
            continue
        match = _concept_token_match(token)
        if match is not None:
            category, canonical, _root = match
            if canonical not in concepts[category]:
                concepts[category].append(canonical)
            continue
        comparator = _comparator_root(token)
        if comparator is not None:
            if comparator not in unknown:
                unknown.append(comparator)
            continue
        dog_root = _generic_dog_root(token)
        if dog_root is not None:
            if dog_label is None:
                dog_label = "dog" if dog_root in {"dog", "dogs"} else "강아지"
            continue
        if token.casefold() in _QUERY_FILLER_TOKENS:
            continue
        if token not in unknown:
            unknown.append(token)

    # ``초소형`` is a narrower size, not a simultaneous small+tiny request.
    if "초소형견" in concepts["size"]:
        concepts["size"] = [value for value in concepts["size"] if value != "소형견"]

    canonical_terms = [
        term for category in _CONCEPT_ORDER for term in concepts[category]
    ]
    meaningful = [*canonical_terms, *unknown]
    if not meaningful:
        return NEUTRAL_DOG_QUERY, nonappearance_excluded
    if dog_label is not None and not any(
        term in _DOG_DENOTING_CONCEPTS for term in canonical_terms
    ):
        meaningful.append(dog_label)
    return " ".join(meaningful), nonappearance_excluded


def analyze_appearance_query(value: Any) -> dict[str, Any]:
    """Analyze and safely normalize a contest-facing appearance query.

    The return value is JSON-compatible so API routes can expose warnings
    without depending on an LLM or changing the legacy string helper.  Negated
    objective conditions are removed from retrieval instead of being silently
    interpreted as positive matches.
    """

    raw_query = unicodedata.normalize("NFKC", str(value or "")).strip()
    if not raw_query:
        return {
            "raw_query": raw_query,
            "normalized_query": NEUTRAL_DOG_QUERY,
            "nonappearance_terms_excluded": False,
            "typo_corrections": [],
            "synonym_normalizations": [],
            "unsupported_conditions": [],
            "warnings": [],
            "is_fully_supported": True,
        }

    corrected, typo_corrections = _correct_conservative_typos(raw_query)
    phrase_normalized, synonym_normalizations = _normalize_phrases(corrected)
    canonical_surface = _canonicalize_surface_tokens(
        phrase_normalized,
        synonym_normalizations,
    )
    safe_surface, unsupported_conditions = _remove_unsupported_negations(
        canonical_surface
    )
    normalized_query, excluded_during_assembly = _assemble_normalized_query(
        safe_surface
    )
    nonappearance_terms_excluded = (
        _contains_nonappearance_token(raw_query) or excluded_during_assembly
    )
    warnings = list(dict.fromkeys(item["message"] for item in unsupported_conditions))
    return {
        "raw_query": raw_query,
        "normalized_query": normalized_query,
        "nonappearance_terms_excluded": nonappearance_terms_excluded,
        "typo_corrections": typo_corrections,
        "synonym_normalizations": synonym_normalizations,
        "unsupported_conditions": unsupported_conditions,
        "warnings": warnings,
        "is_fully_supported": not unsupported_conditions,
    }


def normalize_appearance_query(value: Any) -> str:
    """Return the safe, deterministic appearance retrieval query.

    This compatibility wrapper retains the original string return type.  New
    API surfaces should also expose :func:`analyze_appearance_query` so users
    can see typo corrections and unsupported negated conditions.
    """

    return str(analyze_appearance_query(value)["normalized_query"])


def appearance_query_has_unsupported_negation(value: Any) -> bool:
    """Report whether a negated objective condition was removed."""

    return bool(analyze_appearance_query(value)["unsupported_conditions"])


def appearance_query_excludes_nonvisual_terms(value: Any) -> bool:
    """Report whether the policy removes at least one non-visual token."""

    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    return bool(text and _contains_nonappearance_token(text))


def appearance_search_conditions(
    value: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Keep only objective appearance/notice fields for appearance search."""

    if not isinstance(value, Mapping):
        return {}
    return {
        key: item
        for key, item in value.items()
        if key in APPEARANCE_SEARCH_CONDITION_KEYS
    }
