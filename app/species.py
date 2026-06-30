from __future__ import annotations

from typing import Any, Dict

SPECIES: Dict[str, Dict[str, Any]] = {
    "dog": {
        "label": "dog",
        "label_ko": "개",
        "upkind": "417000",
        "cache": "local_dog_cache.json",
        "enriched_cache": "local_dog_cache_enriched.json",
        "index": "dog_faiss.index",
        "metas": "dog_metas.json",
    },
    "cat": {
        "label": "cat",
        "label_ko": "고양이",
        "upkind": "422400",
        "cache": "local_cat_cache.json",
        "enriched_cache": "local_cat_cache_enriched.json",
        "index": "cat_faiss.index",
        "metas": "cat_metas.json",
    },
    "other": {
        "label": "other",
        "label_ko": "기타",
        "upkind": "429900",
        "cache": "local_other_cache.json",
        "enriched_cache": "local_other_cache_enriched.json",
        "index": "other_faiss.index",
        "metas": "other_metas.json",
    },
}

UPKIND_TO_SPECIES = {str(config["upkind"]): name for name, config in SPECIES.items()}


def clean_species(value: Any, default: str = "dog") -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "dogs": "dog",
        "강아지": "dog",
        "개": "dog",
        "고양이": "cat",
        "cats": "cat",
        "기타": "other",
        "etc": "other",
    }
    text = aliases.get(text, text)
    return text if text in SPECIES else default


def species_from_upkind(upkind: Any, default: str = "dog") -> str:
    return UPKIND_TO_SPECIES.get(str(upkind or "").strip(), default)


def species_config(species: Any) -> Dict[str, Any]:
    return SPECIES[clean_species(species)]