from app.hybrid_rag import build_search_query_text, parse_structured_query


def clean_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def expand_query_text(query: str) -> str:
    """Backward-compatible wrapper for condition-aware query text.

    Older code called this function to append hand-written aliases. The hybrid RAG
    path now extracts a structured condition object first, then serializes those
    fields into searchable text for CLIP/BM25 without maintaining alias groups.
    """
    query = clean_text(query)
    if not query:
        return ""
    structured = parse_structured_query(query)
    return build_search_query_text(query, structured)