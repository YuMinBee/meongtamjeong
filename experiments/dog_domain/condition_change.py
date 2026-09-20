"""CPU-only, fixed-photo attribute interventions using verified frozen caches."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from experiments.composed_retrieval.download import sha256
from experiments.composed_retrieval.metrics import cluster_interval
from experiments.dino_fusion.core import normalize_rows
from experiments.dino_fusion.evaluate_alignment import load_heads
from experiments.dog_domain.notice_extension import HEADS, OUT as KOREA, read, write
from experiments.dog_domain.taiwan_eval import OUT as TAIWAN, attributes

OUT = Path("D:/meongtamjeong_research/condition_change_20260917")
PROTOCOL = Path(__file__).with_name("CONDITION_CHANGE_PROTOCOL.md")
METHODS = [
    "CLIP_image",
    "DINO_image",
    "CLIP_mix",
    "late_mix",
    "CLIP_text_only",
    "linear_mix",
    "mlp_mix",
    "flow_mix",
    "linear_text_only",
    "mlp_text_only",
    "flow_text_only",
    "metadata_filter_DINO",
]


def load_data(dataset):
    base = KOREA if dataset == "korea" else TAIWAN
    records = read(base / "records.json")
    manifest = read(base / ("features.json" if dataset == "korea" else "images.json"))
    feature = base / ("features.npz" if dataset == "korea" else "images.npz")
    assert sha256(base / "records.json") == manifest["records_sha256"]
    assert sha256(feature) == manifest["features_sha256"]
    with np.load(feature) as f:
        if dataset == "korea":
            assert list(f["notice_ids"]) == [r["notice_id"] for r in records]
            cq, cg, dq, dg = [
                f[k]
                for k in ["clip_query", "clip_gallery", "dino_query", "dino_gallery"]
            ]
            texts = {lang: f["text_" + lang] for lang in ["korean", "english"]}
            labels = [
                (
                    tuple(sorted(r["attributes"]["colors"])),
                    r["attributes"]["size"],
                    r["attributes"].get("age", ""),
                )
                for r in records
            ]
            groups = [r["group"] for r in records]
            ids = [r["notice_id"] for r in records]
        else:
            assert list(f["animal_ids"]) == [r["animal_id"] for r in records]
            cq = cg = f["clip"]
            dq = dg = f["dino"]
            labels = [(*attributes(r), "") for r in records]
            groups = [str(r["shelter_id"]) for r in records]
            ids = [r["animal_id"] for r in records]
            result = read(base / "results.json")
            text_file = base / "text_features.npz"
            assert sha256(text_file) == result["cache_sha256"]["text_features"]
            with np.load(text_file) as t:
                texts = {lang: t["template_" + lang] for lang in ["korean", "english"]}
            assert len(result["conditions"]) > 0
    provenance = {
        "records": sha256(base / "records.json"),
        "features": sha256(feature),
        "texts": sha256(base / "text_features.npz")
        if dataset == "taiwan"
        else sha256(feature),
    }
    return records, labels, np.array(groups), ids, (cq, cg, dq, dg), texts, provenance


def relevant(labels, target, dataset):
    colors, size, _ = target
    return np.array(
        [
            s == size
            and (c == colors if dataset == "taiwan" else bool(set(c) & set(colors)))
            for c, s, _ in labels
        ],
        dtype=bool,
    )


def select_queries(labels, ids, dataset):
    representatives = {}
    for i, label in enumerate(labels):
        representatives.setdefault(label, i)
    signatures = sorted(representatives)
    masks = {label: relevant(labels, label, dataset) for label in signatures}
    queries, excluded = [], Counter()
    for i, source in enumerate(labels):
        for axis in ["size", "color"]:
            choices = []
            for label in signatures:
                colors, size, age = label
                valid = age == source[2] and (
                    (colors == source[0] and size != source[1])
                    if axis == "size"
                    else (size == source[1] and not set(colors) & set(source[0]))
                )
                if valid and masks[label].sum() >= 10:
                    choices.append(label)
            if not choices:
                excluded[axis] += 1
                continue
            target = min(
                choices,
                key=lambda x: hashlib.sha256(
                    f"condition-change-v1|{dataset}|{ids[i]}|{axis}|{x}".encode()
                ).digest(),
            )
            assert not masks[target][i]
            queries.append(
                {
                    "reference": i,
                    "target_text": representatives[target],
                    "axis": axis,
                    "source": source,
                    "target": target,
                }
            )
    return queries, dict(excluded)


def rank(scores, reference):
    scores = scores.copy()
    scores[reference] = -np.inf
    return np.argsort(-scores, kind="stable")[:10]


def measure(top, mask):
    hits = mask[top].astype(float)
    discount = 1 / np.log2(np.arange(2, 12))
    return np.array(
        [
            (hits * discount).sum() / discount[: min(10, int(mask.sum()))].sum(),
            hits.mean(),
        ]
    )


def run():
    torch.set_num_threads(4)
    assert not (OUT / "results.json").exists(), "Do not overwrite evaluated outcomes"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "executed_condition_change.py").write_bytes(Path(__file__).read_bytes())
    heads = load_heads(
        report=read(HEADS / "alignment_training_report.json"),
        artifact_dir=HEADS,
        device="cpu",
        torch=torch,
    )
    report = {
        "protocol_sha256": sha256(PROTOCOL),
        "code_sha256": sha256(Path(__file__)),
        "checkpoint_report_sha256": sha256(HEADS / "alignment_training_report.json"),
        "metrics": [
            "target_nDCG10",
            "target_P10",
            "target_P10_gain",
            "top10_set_changed",
        ],
        "datasets": {},
    }
    for dataset in ["korea", "taiwan"]:
        records, labels, groups, ids, features, texts, provenance = load_data(dataset)
        cq, cg, dq, dg = features
        queries, excluded = select_queries(labels, ids, dataset)
        write(OUT / f"{dataset}_queries.json", queries)
        unique_targets = sorted({q["target_text"] for q in queries})
        masks = {t: relevant(labels, labels[t], dataset) for t in unique_targets}
        ci, di = cq @ cg.T, dq @ dg.T
        summary = {
            "gallery": len(records),
            "queries": len(queries),
            "excluded": excluded,
            "provenance": provenance,
            "conditions": {},
        }
        for lang, text in texts.items():
            with torch.inference_mode():
                mapped = {
                    name: normalize_rows(head(torch.tensor(text)).numpy())
                    for name, head in heads.items()
                }
            tx = {"CLIP": text @ cg.T, **{k: v @ dg.T for k, v in mapped.items()}}

            def score(method, i, t):
                if method == "CLIP_image":
                    return ci[i]
                if method == "DINO_image":
                    return di[i]
                if method == "metadata_filter_DINO":
                    mask = relevant(labels, labels[t], dataset)
                    return np.where(mask, di[i], -1e9)
                if method.endswith("text_only"):
                    return tx[method.split("_")[0]][t]
                if method == "late_mix":
                    return 0.8 * di[i] + 0.2 * tx["CLIP"][t]
                name = method.split("_")[0]
                return 0.8 * (ci[i] if name == "CLIP" else di[i]) + 0.2 * tx[name][t]

            arrays, tops = {}, {}
            for method in METHODS:
                values, changed_tops, original_tops = [], [], []
                for q in queries:
                    i, t = q["reference"], q["target_text"]
                    changed = rank(score(method, i, t), i)
                    original = rank(score(method, i, i), i)
                    mask = masks[t].copy()
                    mask[i] = False
                    v = measure(changed, mask)
                    gain = v[1] - measure(original, mask)[1]
                    values.append([*v, gain, float(set(changed) != set(original))])
                    changed_tops.append(changed)
                    original_tops.append(original)
                arrays[method] = np.array(values)
                tops[method + "_changed"] = np.array(changed_tops, dtype=np.int32)
                tops[method + "_original"] = np.array(original_tops, dtype=np.int32)
                print(
                    dataset,
                    lang,
                    method,
                    np.round(arrays[method].mean(0) * 100, 2),
                    flush=True,
                )
            for method in ["CLIP_image", "DINO_image"]:
                assert np.all(arrays[method][:, 2:] == 0)
            assert np.allclose(arrays["metadata_filter_DINO"][:, :2], 1)
            np.savez_compressed(OUT / f"{dataset}_{lang}_metrics.npz", **arrays)
            np.savez_compressed(OUT / f"{dataset}_{lang}_top10.npz", **tops)
            qgroups = groups[[q["reference"] for q in queries]]
            for axis in ["size", "color"]:
                keep = np.array([q["axis"] == axis for q in queries])
                stats = {
                    m: cluster_interval(v[keep], qgroups[keep])
                    for m, v in arrays.items()
                }
                differences = {
                    m: cluster_interval(
                        (arrays["flow_mix"] - arrays[m])[keep], qgroups[keep]
                    )
                    for m in ["CLIP_mix", "DINO_image", "flow_text_only", "mlp_mix"]
                }
                summary["conditions"][lang + "_" + axis] = {
                    "methods": stats,
                    "flow_minus": differences,
                }
        report["datasets"][dataset] = summary
        write(OUT / "progress.json", report)
    write(OUT / "results.json", report)
    lines = [
        "# Fixed-photo condition-change results",
        "",
        "Scores x100. Columns: target nDCG@10 / P@10 / P@10 gain over original text.",
        "Silver attribute labels; no photographic preference evaluation.",
        "",
    ]
    for dataset, data in report["datasets"].items():
        for condition, result in data["conditions"].items():
            lines += [
                f"## {dataset} / {condition}",
                "",
                "| Method | nDCG | P10 | P10 gain |",
                "|---|---:|---:|---:|",
            ]
            for name, stats in result["methods"].items():
                a, b, c, _ = np.array(stats["mean"]) * 100
                lines.append(f"| {name} | {a:.2f} | {b:.2f} | {c:+.2f} |")
            lines += [
                "",
                "Flow minus CLIP mix nDCG 95% shelter-bootstrap interval: "
                + str(
                    np.round(
                        np.array(
                            [
                                result["flow_minus"]["CLIP_mix"][k][0]
                                for k in ["mean", "ci95_low", "ci95_high"]
                            ]
                        )
                        * 100,
                        2,
                    ).tolist()
                ),
                "",
            ]
    Path(__file__).with_name("CONDITION_CHANGE_RESULTS.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print("COMPLETE", OUT, flush=True)


if __name__ == "__main__":
    run()
