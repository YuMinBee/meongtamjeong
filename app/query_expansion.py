from typing import List


QUERY_ALIAS_GROUPS = [
    {
        "triggers": ["눈매", "눈빛", "눈망울", "또렷", "선명", "똘망똘망", "초롱초롱", "맑은"],
        "expansions": ["눈", "눈매", "눈빛", "눈망울", "또렷", "선명", "똘망똘망", "초롱초롱", "맑은"],
    },
    {
        "triggers": ["쫑긋", "솟은 귀", "선 귀", "귀가 서", "귀가 선"],
        "expansions": ["귀", "쫑긋", "솟은 귀", "선 귀"],
    },
    {
        "triggers": ["복슬", "풍성", "덥수룩", "빽빽"],
        "expansions": ["복슬", "풍성", "덥수룩", "빽빽"],
    },
    {
        "triggers": ["짧은 털", "짧고", "단모", "촘촘"],
        "expansions": ["짧은 털", "짧고", "단모", "촘촘"],
    },
]


def clean_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def expand_query_text(query: str) -> str:
    query = clean_text(query)
    if not query:
        return ""

    lowered = query.lower()
    additions: List[str] = []
    for group in QUERY_ALIAS_GROUPS:
        if not any(trigger.lower() in lowered for trigger in group["triggers"]):
            continue
        for expansion in group["expansions"]:
            if expansion.lower() in lowered or expansion in additions:
                continue
            additions.append(expansion)

    if not additions:
        return query
    return f"{query} {' '.join(additions)}"
