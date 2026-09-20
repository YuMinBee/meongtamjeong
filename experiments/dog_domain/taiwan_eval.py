"""Fixed-weight photo-plus-translated-text retrieval of OTHER Taiwan notices."""

import argparse
import re
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.composed_retrieval.download import sha256
from experiments.composed_retrieval.metrics import cluster_interval
from experiments.dino_fusion.core import ClipEncoder, DinoEncoder, normalize_rows
from experiments.dino_fusion.evaluate_alignment import load_heads
from experiments.dog_domain.download_taiwan import OUT as DATA
from experiments.dog_domain.notice_extension import HEADS, read, write

OUT = DATA / "evaluation"
PROTOCOL = Path(__file__).with_name("TAIWAN_EVAL_PROTOCOL.md")
COLORS = {
    "黑": "black",
    "黃": "yellow",
    "白": "white",
    "棕": "brown",
    "咖啡": "brown",
    "灰": "gray",
    "米": "cream",
    "虎斑": "brindle",
    "三花": "tricolor",
    "花": "multicolored",
}
KOREAN = {
    "black": "검은색",
    "yellow": "노란색",
    "white": "흰색",
    "brown": "갈색",
    "gray": "회색",
    "cream": "크림색",
    "brindle": "호랑이 무늬",
    "tricolor": "삼색",
    "multicolored": "여러 색",
}
SIZES = {
    "SMALL": ("small", "작은"),
    "MEDIUM": ("medium-sized", "중간 크기의"),
    "BIG": ("large", "큰"),
}


def attributes(row):
    raw = str(row["color_original"]).strip().replace("色", "")
    tokens = []
    for word, label in sorted(COLORS.items(), key=lambda item: -len(item[0])):
        if word in raw:
            tokens.append(label)
            raw = raw.replace(word, "")
    if raw or not tokens or row["size_original"] not in SIZES:
        return None
    return tuple(sorted(set(tokens))), row["size_original"]


def prompt(row, language):
    colors, size = attributes(row)
    if language == "english":
        return f"find a {SIZES[size][0]} dog whose coat is {' and '.join(colors)}"
    return f"{'과 '.join(KOREAN[c] for c in colors)} 털에 {SIZES[size][1]} 체격인 강아지를 찾아줘"


def images():
    import torch

    torch.set_num_threads(4)
    rows = read(DATA / "unique_image_records.json")
    records = [r for r in rows if attributes(r) is not None]
    write(
        OUT / "selection.json",
        {
            "source_sha256": sha256(DATA / "unique_image_records.json"),
            "protocol_sha256": sha256(PROTOCOL),
            "initial_unique": len(rows),
            "known_attributes": len(records),
            "selected_ids": [r["animal_id"] for r in records],
        },
    )
    clip = ClipEncoder(device="cuda")
    dino = DinoEncoder(
        "facebook/dinov3-vitb16-pretrain-lvd1689m",
        device="cuda",
        revision="5931719e67bbdb9737e363e781fb0c67687896bc",
        local_files_only=True,
    )
    cf, df, thumbnails, hashes = [], [], [], []
    for start in range(0, len(records), 32):
        pictures = []
        for row in records[start : start + 32]:
            path = DATA / row["image_path"]
            assert sha256(path) == row["file_sha256"]
            with Image.open(path) as img:
                rgb = img.convert("RGB")
            pictures.append(rgb)
            thumbnails.append(np.asarray(rgb.resize((32, 32)), dtype=np.uint8))
            gray = np.asarray(rgb.convert("L").resize((9, 8)), dtype=np.float32)
            hashes.append(np.packbits((gray[:, 1:] > gray[:, :-1]).flatten()))
        with torch.inference_mode():
            tensor = torch.stack([clip._preprocess(im) for im in pictures]).to("cuda")
            cf.append(
                normalize_rows(clip._model.encode_image(tensor).float().cpu().numpy())
            )
        df.append(dino.encode_batch(pictures))
        if start % 320 == 0:
            print(
                "image features", start + len(pictures), "/", len(records), flush=True
            )
    thumbs, dhash = np.array(thumbnails), np.array(hashes)
    bits = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)
    parents = list(range(len(records)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    pairs = []
    for i in range(len(records)):
        close = (
            np.flatnonzero(
                bits[np.bitwise_xor(dhash[i], dhash[i + 1 :])].sum(axis=1) <= 3
            )
            + i
            + 1
        )
        for j in close:
            if (
                np.abs(thumbs[i].astype(float) - thumbs[j].astype(float)).mean() / 255
                < 0.035
            ):
                a, b = root(i), root(int(j))
                parents[max(a, b)] = min(a, b)
                pairs.append([records[i]["animal_id"], records[j]["animal_id"]])
    keep = [i for i in range(len(records)) if root(i) == i]
    write(OUT / "near_copy_pairs.json", pairs)
    final = [records[i] for i in keep]
    write(OUT / "records.json", final)
    np.savez_compressed(
        OUT / "images.npz",
        clip=np.concatenate(cf)[keep],
        dino=np.concatenate(df)[keep],
        animal_ids=np.array([r["animal_id"] for r in final]),
    )
    write(
        OUT / "images.json",
        {
            "protocol_sha256": sha256(PROTOCOL),
            "records_sha256": sha256(OUT / "records.json"),
            "features_sha256": sha256(OUT / "images.npz"),
            "near_copy_pairs": len(pairs),
            "near_copy_removed": len(records) - len(final),
            "gallery": len(final),
            "shelters": len({r["shelter_id"] for r in final}),
            "dino_revision": dino.resolved_revision,
            "clip_weights_sha256": sha256(Path.home() / ".cache/clip/ViT-B-32.pt"),
        },
    )
    print(read(OUT / "images.json"), flush=True)


def translate():
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    torch.set_num_threads(4)
    settings = read(OUT / "translator.json")
    tokenizer = AutoTokenizer.from_pretrained(
        settings["model"],
        revision=settings["revision"],
        src_lang="zho_Hant",
        local_files_only=True,
    )
    model = (
        AutoModelForSeq2SeqLM.from_pretrained(
            settings["model"],
            revision=settings["revision"],
            local_files_only=True,
            dtype=torch.float16,
        )
        .to("cuda")
        .eval()
    )
    texts = sorted(
        {
            str(r["description_original"] or "").strip()
            for r in read(DATA / "unique_image_records.json")
        }
        - {""}
    )
    text_segments, pieces = {}, set()
    for text in texts:
        segments = []
        for sentence in re.split(r"(?<=[。！？!?；;])|\r?\n", text):
            sentence = sentence.strip()
            if not sentence:
                continue
            ids = tokenizer(sentence, add_special_tokens=False)["input_ids"]
            segments.extend(
                [sentence]
                if len(ids) <= 400
                else [
                    tokenizer.decode(ids[i : i + 400], skip_special_tokens=True)
                    for i in range(0, len(ids), 400)
                ]
            )
        text_segments[text] = segments
        pieces.update(segments)
    cache_path = OUT / "translation_segments.json"
    cache = read(cache_path) if cache_path.exists() else {}
    pending = sorted(
        pieces - set(cache), key=lambda s: (len(tokenizer(s)["input_ids"]), s)
    )
    print(
        "translation unique documents",
        len(texts),
        "segments",
        len(pieces),
        "pending",
        len(pending),
        flush=True,
    )
    for start in range(0, len(pending), 16):
        batch = pending[start : start + 16]
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to("cuda")
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
                num_beams=4,
                do_sample=False,
                max_new_tokens=512,
            )
        translated = tokenizer.batch_decode(output, skip_special_tokens=True)
        for src, dst, tokens in zip(batch, translated, output):
            real = tokens[tokens != tokenizer.pad_token_id].tolist()
            cache[src] = {
                "english": dst,
                "output_capped": real[-1] != tokenizer.eos_token_id,
            }
        write(cache_path, cache)
        if start % 160 == 0:
            print(
                "translated segments", start + len(batch), "/", len(pending), flush=True
            )
    translations = {
        t: {
            "english": " ".join(cache[s]["english"] for s in segments),
            "segments": len(segments),
            "capped_segments": sum(cache[s]["output_capped"] for s in segments),
        }
        for t, segments in text_segments.items()
    }
    write(OUT / "translations_en.json", translations)
    write(
        OUT / "translation_manifest.json",
        {
            **settings,
            "protocol_sha256": sha256(PROTOCOL),
            "source_sha256": sha256(DATA / "unique_image_records.json"),
            "translated_unique_descriptions": len(texts),
            "unique_segments": len(pieces),
            "capped_segments": sum(r["output_capped"] for r in cache.values()),
            "translation_sha256": sha256(OUT / "translations_en.json"),
            "generation": {"beams": 4, "sample": False, "max_new_tokens": 512},
            "input": "Original remarks only; no images or structured attributes",
            "license": "CC-BY-NC-4.0 research-only model",
        },
    )
    print("translation completed", len(translations), flush=True)


def metrics(scores, labels, refs):
    values, top_indices = [], []
    for row, ref in zip(scores, refs):
        row = row.copy()
        row[ref] = -np.inf
        # Full stable sort gives deterministic ties; no self match is scored.
        order = np.argsort(-row, kind="stable")[:10]
        truth = labels == labels[ref]
        truth[ref] = False
        count = int(truth.sum())
        discount = 1 / np.log2(np.arange(2, 12))
        ndcg = (
            float(np.dot(truth[order], discount) / discount[: min(count, 10)].sum())
            if count
            else np.nan
        )
        values.append([ndcg, float(truth[order].mean()) if count else np.nan])
        top_indices.append(order)
    return np.array(values), np.array(top_indices)


def evaluate(include_qwen=True):
    import torch

    torch.set_num_threads(4)
    records = read(OUT / "records.json")
    manifest = read(OUT / "images.json")
    assert manifest["records_sha256"] == sha256(OUT / "records.json")
    assert manifest["features_sha256"] == sha256(OUT / "images.npz")
    assert manifest["protocol_sha256"] == sha256(PROTOCOL)
    translation_manifest = read(OUT / "translation_manifest.json")
    assert translation_manifest["translation_sha256"] == sha256(
        OUT / "translations_en.json"
    )
    assert translation_manifest["protocol_sha256"] == sha256(PROTOCOL)
    translations = read(OUT / "translations_en.json")
    qwen_manifest, qwen = {}, {}
    if include_qwen:
        qwen_manifest = read(OUT / "qwen_translation_manifest.json")
        assert qwen_manifest["translation_sha256"] == sha256(
            OUT / "qwen_translations_en.json"
        )
        assert qwen_manifest["amendment_sha256"] == sha256(
            Path(__file__).with_name("TAIWAN_TRANSLATION_AMENDMENT.md")
        )
        qwen = read(OUT / "qwen_translations_en.json")
    with np.load(OUT / "images.npz") as data:
        ci, di = data["clip"], data["dino"]
        assert list(data["animal_ids"]) == [r["animal_id"] for r in records]
    labels = np.array(
        ["|".join([",".join(attributes(r)[0]), attributes(r)[1]]) for r in records]
    )
    groups = np.array([r["shelter_id"] for r in records])
    encoder = ClipEncoder(device="cuda")
    heads = load_heads(
        report=read(HEADS / "alignment_training_report.json"),
        artifact_dir=HEADS,
        device="cuda",
        torch=torch,
    )
    result = {
        "status": "completed",
        "gallery": len(records),
        "image_manifest": manifest,
        "translation_manifest": translation_manifest,
        "qwen_translation_manifest": qwen_manifest,
        "metrics": ["nDCG@10", "P@10"],
        "checkpoint_report_sha256": sha256(HEADS / "alignment_training_report.json"),
        "conditions": {},
    }
    arrays, rankings, text_features = {}, {}, {}
    cached = None
    if include_qwen and (OUT / "base_results.json").exists():
        cached = read(OUT / "base_results.json")
        for key in [
            "image_manifest",
            "translation_manifest",
            "checkpoint_report_sha256",
        ]:
            assert cached[key] == result[key]
        for filename, target in [
            ("query_metrics", arrays),
            ("top10", rankings),
            ("text_features", text_features),
        ]:
            assert cached["cache_sha256"][filename] == sha256(
                OUT / f"base_{filename}.npz"
            )
            with np.load(OUT / f"base_{filename}.npz") as data:
                target.update({key: data[key] for key in data.files})
    for condition in [
        "template_english",
        "template_korean",
        "prose_chinese",
        "prose_english",
        "prose_english_qwen",
    ]:
        if condition == "prose_english_qwen" and not include_qwen:
            continue
        if cached and condition in cached["conditions"]:
            result["conditions"][condition] = cached["conditions"][condition]
            continue
        refs = np.array(
            [
                i
                for i, r in enumerate(records)
                if not condition.startswith("prose")
                or str(r["description_original"] or "").strip()
            ]
        )
        texts = []
        for i in refs:
            row = records[i]
            if condition.startswith("template"):
                texts.append(prompt(row, condition.split("_")[1]))
            else:
                raw = str(row["description_original"]).strip()
                texts.append(
                    raw
                    if condition == "prose_chinese"
                    else (qwen if condition == "prose_english_qwen" else translations)[
                        raw
                    ]["english"]
                )
        text = np.concatenate(
            [
                encoder.encode_text_batch(texts[i : i + 64])
                for i in range(0, len(texts), 64)
            ]
        )
        text_features[condition] = text
        mapped = {}
        for name, head in heads.items():
            with torch.inference_mode():
                mapped[name] = normalize_rows(
                    head(torch.tensor(text, device="cuda")).cpu().numpy()
                )
        truncated = 0
        for sentence in texts:
            try:
                encoder._clip.tokenize(sentence, truncate=False)
            except RuntimeError:
                truncated += 1
        known = np.array([np.count_nonzero(labels == labels[i]) > 1 for i in refs])

        def summary(v):
            return cluster_interval(v[known], groups[refs][known])

        condition_report = {
            "queries": len(refs),
            "evaluable_queries": int(known.sum()),
            "truncated_texts": truncated,
            "shelters": len(set(groups[refs])),
            "results": {},
            "paired": {},
        }
        methods = [
            "CLIP_image",
            "DINO_image",
            "CLIP_mix",
            "late_mix",
            "CLIP_shift",
            "CLIP_text_only",
            "flow_mix",
            "linear_mix",
            "mlp_mix",
            "flow_shift",
            "flow_text_only",
            "linear_text_only",
            "mlp_text_only",
        ]
        matrices = {}
        for method in methods:
            values, top = [], []
            for start in range(0, len(refs), 256):
                ix = refs[start : start + 256]
                tx = text[start : start + 256]
                shifted = np.roll(text, 1, axis=0)[start : start + 256]
                if method == "CLIP_image":
                    scores = ci[ix] @ ci.T
                elif method == "DINO_image":
                    scores = di[ix] @ di.T
                elif method == "CLIP_mix":
                    scores = normalize_rows(0.8 * ci[ix] + 0.2 * tx) @ ci.T
                elif method == "CLIP_shift":
                    scores = normalize_rows(0.8 * ci[ix] + 0.2 * shifted) @ ci.T
                elif method == "late_mix":
                    scores = 0.8 * (di[ix] @ di.T) + 0.2 * (tx @ ci.T)
                elif method == "CLIP_text_only":
                    scores = tx @ ci.T
                else:
                    name = method.split("_")[0]
                    mt = mapped[name][start : start + 256]
                    if method.endswith("shift"):
                        mt = np.roll(mapped[name], 1, axis=0)[start : start + 256]
                    scores = (
                        mt @ di.T
                        if method.endswith("text_only")
                        else normalize_rows(0.8 * di[ix] + 0.2 * mt) @ di.T
                    )
                v, t = metrics(scores, labels, ix)
                values.append(v)
                top.append(t)
            values = np.concatenate(values)
            matrices[method] = values
            arrays[f"{condition}/{method}"] = values
            rankings[f"{condition}/{method}"] = np.concatenate(top)
            condition_report["results"][method] = summary(values)
        for other in [
            "CLIP_mix",
            "DINO_image",
            "late_mix",
            "linear_mix",
            "mlp_mix",
            "flow_shift",
        ]:
            condition_report["paired"][f"flow_mix minus {other}"] = summary(
                matrices["flow_mix"] - matrices[other]
            )
        arrays[f"{condition}/refs"] = refs
        condition_report["shift_same_label_fraction"] = float(
            np.mean(labels[refs] == np.roll(labels[refs], 1))
        )
        result["conditions"][condition] = condition_report
        print(
            condition,
            len(refs),
            {
                m: round(v["mean"][0] * 100, 3)
                for m, v in condition_report["results"].items()
            },
            flush=True,
        )
    prose_refs = arrays["prose_english/refs"]
    valid = np.array([np.count_nonzero(labels == labels[i]) > 1 for i in prose_refs])
    result["english_translation_minus_chinese"] = {
        m: cluster_interval(
            (arrays["prose_english/" + m] - arrays["prose_chinese/" + m])[valid],
            groups[prose_refs][valid],
        )
        for m in ["CLIP_mix", "flow_mix", "CLIP_text_only", "flow_text_only"]
    }
    prefix = "" if include_qwen else "base_"
    np.savez_compressed(OUT / f"{prefix}query_metrics.npz", **arrays)
    result["qwen_translation_contrasts"] = {}
    for other in ["prose_chinese", "prose_english"] if include_qwen else []:
        for method in ["CLIP_mix", "flow_mix", "CLIP_text_only", "flow_text_only"]:
            result["qwen_translation_contrasts"][f"{method}: Qwen minus {other}"] = (
                cluster_interval(
                    (
                        arrays["prose_english_qwen/" + method]
                        - arrays[other + "/" + method]
                    )[valid],
                    groups[prose_refs][valid],
                )
            )
    np.savez_compressed(OUT / f"{prefix}top10.npz", **rankings)
    np.savez_compressed(OUT / f"{prefix}text_features.npz", **text_features)
    result["cache_sha256"] = {
        name: sha256(OUT / f"{prefix}{name}.npz")
        for name in ["top10", "text_features", "query_metrics"]
    }
    write(OUT / f"{prefix}results.json", result)
    if not include_qwen:
        return
    lines = [
        "# Taiwan translated-text evaluation",
        "",
        f"Gallery: {len(records)} photos. Reference photo excluded. Frozen heads; text weight .20.",
        "Relevance: exact normalized color set AND Taiwan size category. Scores below are nDCG@10 x 100, not accuracy.",
        "",
        "| Method | English template | Korean template | Chinese prose | NLLB English prose | Qwen English prose |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        cells = [
            f"{result['conditions'][c]['results'][method]['mean'][0] * 100:.2f}"
            for c in result["conditions"]
        ]
        lines.append("| " + method + " | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Paired differences",
        "",
        "| Condition / Flow comparison | nDCG difference [95% CI], points |",
        "|---|---|",
    ]
    for condition, details in result["conditions"].items():
        for pair, stats in details["paired"].items():
            lines.append(
                f"| {condition}: {pair} | {stats['mean'][0] * 100:+.2f} [{stats['ci95_low'][0] * 100:+.2f}, {stats['ci95_high'][0] * 100:+.2f}] |"
            )
    lines += ["", "## Coverage", ""]
    for condition, details in result["conditions"].items():
        lines.append(
            f"- {condition}: {details['queries']} queries; {details['evaluable_queries']} with another relevant candidate; {details['truncated_texts']} CLIP-truncated texts."
        )
    lines += [
        "",
        "Shelter-cluster bootstrap intervals (2,000 samples); no multiplicity correction for exploratory contrasts.",
        "Templates are dictionary-rendered attributes. English prose uses automatic NLLB and Qwen translations; neither is human validated.",
        "Prose is scored only against appearance proxy labels, not full-description relevance. No same-animal accuracy claim.",
        "Taiwan size labels differ from Korean weight bins. All original heads remain frozen; no Taiwan training or tuning.",
        "",
    ]
    Path(__file__).with_name("TAIWAN_EVAL_RESULTS.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "stage", choices=["images", "translate", "evaluate", "evaluate_base"]
    )
    a = p.parse_args()
    {
        "images": images,
        "translate": translate,
        "evaluate": evaluate,
        "evaluate_base": lambda: evaluate(False),
    }[a.stage]()
