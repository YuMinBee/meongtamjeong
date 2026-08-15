import json
import re
import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

DEFAULT_FIELDS = ["specialMark", "desc", "kindCd", "breed"]
TOKEN_RE = re.compile(r"[가-힣]+|[a-zA-Z]+|\d+")


def load_metas(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        metas = json.load(f)

    # list of list flatten
    if isinstance(metas, list) and metas and isinstance(metas[0], list):
        metas = [x for sub in metas for x in sub]

    if not isinstance(metas, list):
        raise ValueError("metas must be a list")

    metas = [m for m in metas if isinstance(m, dict)]
    return metas


def normalize_text(s: str) -> str:
    s = s.replace("\u3000", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def iter_texts(metas: List[Dict[str, Any]], fields: List[str], only_type: str = "") -> Iterable[str]:
    for m in metas:
        if only_type and m.get("type") != only_type:
            continue
        for k in fields:
            v = m.get(k)
            if isinstance(v, str) and v.strip():
                yield v.strip()


def word_counter(texts: Iterable[str], min_len: int = 2, keep_numbers: bool = True) -> Counter:
    c = Counter()
    for s in texts:
        s = normalize_text(s)
        toks = TOKEN_RE.findall(s)
        for t in toks:
            if not keep_numbers and t.isdigit():
                continue
            if len(t) < min_len:
                continue
            c[t.lower()] += 1
    return c


def char_counter(texts: Iterable[str], keep_space: bool = False) -> Counter:
    c = Counter()
    for s in texts:
        s = normalize_text(s)
        for ch in s:
            if not keep_space and ch.isspace():
                continue
            c[ch] += 1
    return c


def ngram_counter(texts: Iterable[str], n: int = 2, min_count: int = 1) -> Counter:
    c = Counter()
    for s in texts:
        s = normalize_text(s).replace(" ", "")
        if len(s) < n:
            continue
        for i in range(len(s) - n + 1):
            c[s[i:i+n]] += 1

    if min_count > 1:
        c = Counter({k: v for k, v in c.items() if v >= min_count})
    return c


def counter_to_df(counter: Counter, col_name: str) -> pd.DataFrame:
    items = counter.most_common()
    df = pd.DataFrame(items, columns=[col_name, "count"])
    df["rank"] = range(1, len(df) + 1)
    df = df[["rank", col_name, "count"]]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metas", default=str(DATA_DIR / "dog_metas.json"), help="path to dog_metas.json")
    ap.add_argument("--out", default=str(DATA_DIR / "meta_text_freq.xlsx"), help="output xlsx path")

    ap.add_argument("--fields", default=",".join(DEFAULT_FIELDS),
                    help="comma-separated fields (e.g. specialMark,desc)")
    ap.add_argument("--only-type", default="", help="filter metas by type: image/text, empty=all")

    ap.add_argument("--min-word-len", type=int, default=2)
    ap.add_argument("--no-numbers", action="store_true", help="exclude numeric tokens")

    ap.add_argument("--topk", type=int, default=5000, help="max rows saved per sheet (safety)")
    ap.add_argument("--chars", action="store_true", help="export char frequency sheet")
    ap.add_argument("--keep-space", action="store_true", help="keep spaces in char counting")

    ap.add_argument("--ngram", type=int, default=0, help="enable n-gram (e.g. 2 or 3). 0=off")
    ap.add_argument("--ngram-min-count", type=int, default=2)

    args = ap.parse_args()

    fields = [x.strip() for x in args.fields.split(",") if x.strip()]
    metas = load_metas(args.metas)

    texts = list(iter_texts(metas, fields, only_type=args.only_type))
    if not texts:
        raise SystemExit("[ERROR] No texts found. Check --fields / --only-type / --metas")

    # counters
    wc = word_counter(texts, min_len=args.min_word_len, keep_numbers=not args.no_numbers)
    cc = char_counter(texts, keep_space=args.keep_space) if args.chars else None
    ng = ngram_counter(texts, n=args.ngram, min_count=args.ngram_min_count) if args.ngram and args.ngram > 0 else None

    # dataframes (topk 제한)
    df_words = counter_to_df(wc, "token").head(args.topk)

    df_chars = None
    if cc is not None:
        df_chars = counter_to_df(cc, "char").head(args.topk)

    df_ngrams = None
    if ng is not None:
        df_ngrams = counter_to_df(ng, f"{args.ngram}-gram").head(args.topk)

    # summary
    summary = {
        "metas_count": len(metas),
        "texts_collected": len(texts),
        "fields": ", ".join(fields),
        "only_type": args.only_type or "ALL",
        "min_word_len": args.min_word_len,
        "keep_numbers": (not args.no_numbers),
        "export_chars": bool(args.chars),
        "export_ngram": int(args.ngram) if args.ngram else 0,
        "ngram_min_count": int(args.ngram_min_count),
        "rows_saved_topk": int(args.topk),
        "unique_word_tokens": len(wc),
        "unique_chars": len(cc) if cc is not None else 0,
        "unique_ngrams": len(ng) if ng is not None else 0,
    }
    df_summary = pd.DataFrame(list(summary.items()), columns=["key", "value"])

    # write excel
    with pd.ExcelWriter(args.out, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="summary", index=False)
        df_words.to_excel(writer, sheet_name="words", index=False)
        if df_chars is not None:
            df_chars.to_excel(writer, sheet_name="chars", index=False)
        if df_ngrams is not None:
            df_ngrams.to_excel(writer, sheet_name="ngrams", index=False)

    print(f"[OK] Saved: {args.out}")
    print(f" - words rows: {len(df_words)}")
    if df_chars is not None:
        print(f" - chars rows: {len(df_chars)}")
    if df_ngrams is not None:
        print(f" - ngrams rows: {len(df_ngrams)}")


if __name__ == "__main__":
    main()
