"""Recompute saved metrics and independently score sampled rankings in float64."""

import numpy as np
import torch

from experiments.composed_retrieval.download import sha256
from experiments.dino_fusion.core import normalize_rows
from experiments.dino_fusion.evaluate_alignment import load_heads
from experiments.dog_domain.condition_change import (
    OUT,
    PROTOCOL,
    METHODS,
    load_data,
    TAIWAN,
)
from experiments.dog_domain.notice_extension import HEADS, read, write


def main():
    torch.set_num_threads(4)
    report = read(OUT / "results.json")
    assert report["protocol_sha256"] == sha256(PROTOCOL)
    assert report["code_sha256"] == sha256(OUT / "executed_condition_change.py")
    tw = read(TAIWAN / "results.json")
    assert tw["cache_sha256"]["text_features"] == sha256(TAIWAN / "text_features.npz")
    heads = load_heads(
        report=read(HEADS / "alignment_training_report.json"),
        artifact_dir=HEADS,
        device="cpu",
        torch=torch,
    )
    metrics_checked = rankings_checked = near_ties = 0
    for dataset in ["korea", "taiwan"]:
        records, labels, groups, ids, features, texts, provenance = load_data(dataset)
        assert provenance == report["datasets"][dataset]["provenance"]
        qs = read(OUT / f"{dataset}_queries.json")
        cq, cg, dq, dg = [x.astype(np.float64) for x in features]
        relevance_cache = {}

        def relevance(target, ref):
            colors, size, _ = target
            key = (tuple(colors), size)
            if key not in relevance_cache:
                relevance_cache[key] = np.array(
                    [
                        s == size
                        and (
                            set(c) == set(colors)
                            if dataset == "taiwan"
                            else len(set(c).intersection(colors)) > 0
                        )
                        for c, s, _ in labels
                    ]
                )
            mask = relevance_cache[key].copy()
            mask[ref] = False
            return mask

        for q in qs:
            i, j = q["reference"], q["target_text"]
            a, b = labels[i], labels[j]
            assert a[2] == b[2]
            if q["axis"] == "size":
                assert a[0] == b[0] and a[1] != b[1]
            else:
                assert a[1] == b[1] and not set(a[0]) & set(b[0])
            assert relevance(q["target"], i).sum() >= 10
        sample = np.random.default_rng(20260917).choice(
            len(qs), min(20, len(qs)), replace=False
        )
        for lang, text in texts.items():
            with torch.inference_mode():
                mapped = {
                    name: normalize_rows(head(torch.tensor(text)).numpy()).astype(
                        np.float64
                    )
                    for name, head in heads.items()
                }
            text64 = text.astype(np.float64)

            def scores(method, i, t):
                if method == "CLIP_image":
                    return cg @ cq[i]
                if method == "DINO_image":
                    return dg @ dq[i]
                if method == "metadata_filter_DINO":
                    return np.where(relevance(labels[t], i), dg @ dq[i], -1e9)
                if method == "late_mix":
                    return 0.8 * (dg @ dq[i]) + 0.2 * (cg @ text64[t])
                name = method.split("_")[0]
                gallery, image, txt = (
                    (cg, cq[i], text64[t])
                    if name == "CLIP"
                    else (dg, dq[i], mapped[name][t])
                )
                if method.endswith("text_only"):
                    return gallery @ txt
                query = 0.8 * image + 0.2 * txt
                return gallery @ (query / np.linalg.norm(query))

            with (
                np.load(OUT / f"{dataset}_{lang}_top10.npz") as rankings,
                np.load(OUT / f"{dataset}_{lang}_metrics.npz") as values,
            ):
                rankings = {k: rankings[k] for k in rankings.files}
                values = {k: values[k] for k in values.files}
                for method in METHODS:
                    for k, q in enumerate(qs):
                        ref = q["reference"]
                        changed = rankings[method + "_changed"][k]
                        old = rankings[method + "_original"][k]
                        assert (
                            len(set(changed)) == 10
                            and ref not in changed
                            and ref not in old
                        )
                        mask = relevance(q["target"], ref)
                        hit = mask[changed]
                        discount = 1 / np.log2(np.arange(2, 12))
                        expected = [
                            (hit * discount).sum() / discount.sum(),
                            hit.mean(),
                            hit.mean() - mask[old].mean(),
                            float(set(changed) != set(old)),
                        ]
                        np.testing.assert_allclose(
                            expected, values[method][k], atol=1e-12
                        )
                        metrics_checked += 4
                    for k in sample:
                        q = qs[k]
                        for suffix, t in [
                            ("changed", q["target_text"]),
                            ("original", q["reference"]),
                        ]:
                            s = scores(method, q["reference"], t)
                            s[q["reference"]] = -np.inf
                            expected = np.argsort(-s, kind="stable")[:10]
                            actual = rankings[method + "_" + suffix][k]
                            if not np.array_equal(expected, actual):
                                # Float32 GEMM vs float64: accept ONLY score-equivalent near ties.
                                assert np.max(np.abs(s[expected] - s[actual])) < 2e-6
                                near_ties += 1
                            rankings_checked += 1
                print(dataset, lang, "verified", flush=True)
    result = {
        "status": "passed",
        "metric_values_recomputed": metrics_checked,
        "float64_rankings_checked": rankings_checked,
        "float_precision_near_ties": near_ties,
        "results_sha256": sha256(OUT / "results.json"),
    }
    write(OUT / "verification.json", result)
    print(result)


if __name__ == "__main__":
    main()
