"""Independent NumPy spot-checks of completed rankings; never selects a method."""

from __future__ import annotations

import numpy as np

from experiments.composed_retrieval.download import sha256
from experiments.composed_retrieval.evaluate import EVALUATION
from experiments.composed_retrieval.features import load_features
from experiments.composed_retrieval.prepare import ARTIFACTS, TASKS, load_json, write_json
from experiments.composed_retrieval.train import MODELS


def unit(value):
    return value / np.linalg.norm(value, axis=-1, keepdims=True)


def run():
    import torch
    from experiments.dino_fusion.alignment import make_projection_head

    torch.set_num_threads(4)
    index, features = load_features()
    images = {key: i for i, key in enumerate(index["image_keys"])}
    texts = {key: i for i, key in enumerate(index["texts"])}
    selection = load_json(EVALUATION / "frozen_selection.json")
    heads = {}
    for architecture in ("linear", "flow"):
        checkpoint = torch.load(MODELS / f"{architecture}_7.pt", map_location="cpu", weights_only=True)
        head = make_projection_head(architecture, input_dim=512, output_dim=768).eval()
        head.load_state_dict(checkpoint["state_dict"])
        heads[architecture] = head
    labels = ("clip_image", "clip_text", "dino_image", "clip_composition", "late_fusion",
              "unaligned_residual", "linear", "linear_residual", "flow_fixed_0.2")
    rng = np.random.default_rng(20260916)
    checks, maximum_difference = [], 0.
    for dataset in ("circo", *[f"genecis_{task}" for task in TASKS]):
        split = "audit" if dataset == "circo" else "external"
        manifest = load_json(ARTIFACTS / f"manifest_{dataset}.json")
        queries = [q for q in manifest["queries"] if q["split"] == split]
        sample = set(rng.choice(len(queries), 4, replace=False).tolist())
        # Include known contradictory target slots as integrity checks.
        sample.update(i for i, q in enumerate(queries) if q["id"] in ("focus_attribute:14", "change_object:19"))
        with np.load(EVALUATION / f"{dataset}_{split}_per_query.npz") as saved:
            assert saved["query_ids"].tolist() == [q["id"] for q in queries]
            for i in sorted(sample):
                query = queries[i]
                keys = sorted(a["key"] for a in manifest["assets"]) if dataset == "circo" else query["candidate_assets"]
                gallery_ids = [images[key] for key in keys]
                cg, dg = features["clip"][gallery_ids], features["dino"][gallery_ids]
                ci, di = features["clip"][images[query["reference"]]], features["dino"][images[query["reference"]]]
                text = features["text"][texts[query["text"]]]
                mapped = {}
                with torch.inference_mode():
                    for architecture, head in heads.items():
                        mapped[architecture] = unit(head(torch.from_numpy(text[None])).numpy()[0])
                for label in labels:
                    config = ({"family": "flow", "text_weight": .2} if label == "flow_fixed_0.2"
                              else selection["selected"][label]["config"])
                    family, w = config["family"], config.get("text_weight", 0.)
                    if family == "clip_image":
                        scores = cg @ ci
                    elif family == "clip_text":
                        scores = cg @ text
                    elif family == "dino_image":
                        scores = dg @ di
                    elif family == "late_fusion":
                        scores = (1 - w) * (dg @ di) + w * (cg @ text)
                    else:
                        clip_composed = cg @ unit((1 - w) * ci + w * text)
                        if family == "clip_composition":
                            scores = clip_composed
                        else:
                            architecture = family.split("_")[0]
                            dino_composed = (dg @ unit((1 - w) * di + w * mapped[architecture])
                                             if architecture in heads else dg @ di)
                            scores = (config["residual_weight"] * dino_composed +
                                      (1 - config["residual_weight"]) * clip_composed
                                      if family.endswith("residual") else dino_composed)
                    if dataset == "circo":
                        scores[keys.index(query["reference"])] = -np.inf
                        order = np.argsort(-scores, kind="stable")[:50]
                        hits = np.array([keys[j] in query["positives"] for j in order])
                        actual = np.array([sum(hits[r] * hits[:r + 1].sum() / (r + 1) for r in range(k)) /
                                           min(k, len(query["positives"])) for k in (5, 10, 25, 50)])
                    else:
                        order = np.argsort(-scores, kind="stable")[:3]
                        actual = np.array([float(any(query["candidates"][j] in query["positives"]
                                                     for j in order[:k])) for k in (1, 2, 3)])
                    expected = saved[label][0, i]
                    difference = float(np.max(np.abs(actual - expected)))
                    maximum_difference = max(maximum_difference, difference)
                    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0.,
                                               err_msg=f"{dataset}/{query['id']}/{label}")
                checks.append({"dataset": dataset, "query": query["id"], "methods": len(labels)})
    result = {"passed": True, "checks": checks, "query_method_checks": len(checks) * len(labels),
              "maximum_metric_difference": maximum_difference,
              "selection_sha256": sha256(EVALUATION / "frozen_selection.json"),
              "verification": "CPU head inference, direct NumPy query composition, full stable sorting, independent metric formulas"}
    write_json(EVALUATION / "independent_verification.json", result)
    print(f"Independent verification passed: {result['query_method_checks']} query/method checks", flush=True)


if __name__ == "__main__":
    run()
