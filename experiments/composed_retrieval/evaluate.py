"""Select on CIRCO development, freeze, then audit and external GeneCIS."""

from __future__ import annotations

from datetime import datetime, timezone
import gc
import time
from pathlib import Path

import numpy as np

from experiments.composed_retrieval.download import sha256
from experiments.composed_retrieval.features import digest, load_features
from experiments.composed_retrieval.metrics import (
    average_precision, cluster_interval, random_average_precision, recall, stable_topk,
)
from experiments.composed_retrieval.prepare import ARTIFACTS, TASKS, load_json, write_json
from experiments.composed_retrieval.train import ARCHITECTURES, MODELS, SEEDS

WEIGHTS = (0., .1, .2, .4, .6, .8, 1.)
RESIDUAL_WEIGHTS = (0., .25, .5, .75, 1.)
EVALUATION = ARTIFACTS / "evaluation"


def candidates():
    for family in ("clip_image", "clip_text", "dino_image"):
        yield {"family": family}
    for family in ("clip_composition", "late_fusion", *ARCHITECTURES,
                   "unaligned_residual", *[f"{a}_residual" for a in ARCHITECTURES]):
        for weight in WEIGHTS:
            residuals = RESIDUAL_WEIGHTS if family.endswith("residual") else (None,)
            for residual in residuals:
                row = {"family": family, "text_weight": weight}
                if residual is not None:
                    row["residual_weight"] = residual
                yield row


def architecture_for(config):
    family = config["family"]
    return next((name for name in ARCHITECTURES if family in (name, f"{name}_residual")), None)


class Scorer:
    def __init__(self, manifest, index, features, projected, *, split):
        import torch

        self.torch = torch
        self.queries = [q for q in manifest["queries"] if q["split"] == split]
        self.is_circo = manifest["dataset"] == "circo"
        self.names = [f"mAP@{k}" for k in (5, 10, 25, 50)] if self.is_circo else ["R@1", "R@2", "R@3"]
        image_index = {key: i for i, key in enumerate(index["image_keys"])}
        text_index = {key: i for i, key in enumerate(index["texts"])}
        refs = [image_index[q["reference"]] for q in self.queries]
        text_ids = [text_index[q["text"]] for q in self.queries]
        self.cref = torch.tensor(features["clip"][refs], device="cuda")
        self.dref = torch.tensor(features["dino"][refs], device="cuda")
        self.ctext = torch.tensor(features["text"][text_ids], device="cuda")
        self.mapped = {name: torch.tensor(rows[text_ids], device="cuda") for name, rows in projected.items()}
        if self.is_circo:
            keys = sorted(a["key"] for a in manifest["assets"])
            self.gallery_index = {key: i for i, key in enumerate(keys)}
            ids = [image_index[key] for key in keys]
            self.cg = torch.tensor(features["clip"][ids], device="cuda")
            self.dg = torch.tensor(features["dino"][ids], device="cuda")
            self.ref_positions = torch.tensor([self.gallery_index[q["reference"]] for q in self.queries], device="cuda")
            self.positive_positions = [{self.gallery_index[p] for p in q["positives"]} for q in self.queries]
        else:
            width = max(len(q["candidates"]) for q in self.queries)
            ids = np.zeros((len(self.queries), width), dtype=np.int64)
            valid = np.zeros_like(ids, dtype=bool)
            self.positive_positions = []
            for i, query in enumerate(self.queries):
                if query["candidates"] != sorted(query["candidates"]):
                    raise ValueError("Candidate slots must be sorted for stable ties")
                count = len(query["candidate_assets"])
                ids[i, :count] = [image_index[key] for key in query["candidate_assets"]]
                valid[i, :count] = True
                self.positive_positions.append({query["candidates"].index(p) for p in query["positives"]})
            self.valid = torch.tensor(valid, device="cuda")
            self.cg = torch.tensor(features["clip"][ids], device="cuda")
            self.dg = torch.tensor(features["dino"][ids], device="cuda")
        self.ci = self.similarity(self.cref, self.cg)
        self.ct = self.similarity(self.ctext, self.cg)
        self.di = self.similarity(self.dref, self.dg)
        self.dt = {name: self.similarity(rows, self.dg) for name, rows in self.mapped.items()}

    def similarity(self, query, gallery):
        if self.is_circo:
            return query @ gallery.T
        return self.torch.einsum("qd,qnd->qn", query, gallery)

    def composition(self, image_scores, text_scores, image, text, weight):
        norm = self.torch.linalg.vector_norm((1 - weight) * image + weight * text, dim=1, keepdim=True)
        return ((1 - weight) * image_scores + weight * text_scores) / norm.clamp_min(1e-12)

    def score(self, config, seed=7):
        family = config["family"]
        weight = config.get("text_weight", 0.)
        if family == "clip_image":
            scores = self.ci
        elif family == "clip_text":
            scores = self.ct
        elif family == "dino_image":
            scores = self.di
        elif family == "late_fusion":
            scores = (1 - weight) * self.di + weight * self.ct
        else:
            clip_composed = self.composition(self.ci, self.ct, self.cref, self.ctext, weight)
            if family == "clip_composition":
                scores = clip_composed
            else:
                architecture = architecture_for(config)
                if architecture:
                    key = f"{architecture}_{seed}"
                    dino_composed = self.composition(self.di, self.dt[key], self.dref, self.mapped[key], weight)
                else:
                    dino_composed = self.di
                if family.endswith("residual"):
                    r = config["residual_weight"]
                    scores = r * dino_composed + (1 - r) * clip_composed
                else:
                    scores = dino_composed
        scores = scores.clone()
        if not bool(self.torch.isfinite(scores).all()):
            raise ValueError("Non-finite retrieval scores")
        if self.is_circo:
            scores[self.torch.arange(len(self.queries), device="cuda"), self.ref_positions] = -self.torch.inf
            top = stable_topk(scores, 50).cpu().numpy()
            rows = [average_precision(row, positives) for row, positives in zip(top, self.positive_positions)]
        else:
            scores.masked_fill_(~self.valid, -self.torch.inf)
            top = stable_topk(scores, 3).cpu().numpy()
            rows = [recall(row, positives) for row, positives in zip(top, self.positive_positions)]
        return np.stack(rows)


def projected_texts(index, features):
    import torch
    from experiments.dino_fusion.alignment import make_projection_head, project_embeddings

    output, provenance = {}, {}
    # Only query texts are needed for inference; retain global indexing for clarity.
    query_texts = set()
    for name in ["circo", *[f"genecis_{task}" for task in TASKS]]:
        query_texts.update(q["text"] for q in load_json(ARTIFACTS / f"manifest_{name}.json")["queries"])
    text_ids = [i for i, text in enumerate(index["texts"]) if text in query_texts]
    texts = torch.tensor(features["text"][text_ids], device="cuda")
    for architecture in ARCHITECTURES:
        for seed in SEEDS:
            name = f"{architecture}_{seed}"
            path = MODELS / f"{name}.pt"
            report = load_json(MODELS / f"{name}.json")
            if not report["complete"] or report["feature_signature"] != index["signature"]:
                raise RuntimeError(f"Incomplete or stale head {name}")
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            head = make_projection_head(architecture, input_dim=512, output_dim=768).cuda().eval()
            head.load_state_dict(checkpoint["state_dict"])
            with torch.inference_mode():
                rows = np.concatenate([project_embeddings(head, batch).cpu().numpy() for batch in texts.split(256)])
            output[name] = np.zeros((len(index["texts"]), 768), dtype=np.float32)
            output[name][text_ids] = rows
            provenance[name] = {"sha256": sha256(path), "best_epoch": checkpoint["epoch"],
                                "validation_loss": checkpoint["validation_loss"], "parameters": report["parameters"]}
    return output, provenance


def evaluate_config(scorer, config):
    seeds = SEEDS if architecture_for(config) else (7,)
    return np.stack([scorer.score(config, seed) for seed in seeds])


def run():
    import torch

    torch.set_num_threads(4)
    EVALUATION.mkdir(exist_ok=True)
    started = time.monotonic()
    index, features = load_features()
    projected, heads = projected_texts(index, features)
    manifests = {name: load_json(ARTIFACTS / f"manifest_{name}.json") for name in
                 ["circo", *[f"genecis_{task}" for task in TASKS]]}
    provenance = {"features": index["signature"], "heads": heads,
                  "manifests": {name: digest(data) for name, data in manifests.items()},
                  "grid": list(candidates()), "implementation": "composition-evaluation-v1",
                  "code": {name: sha256(Path(__file__).with_name(name)) for name in ("evaluate.py", "metrics.py")}}
    signature = digest(provenance)
    selection_path = EVALUATION / "frozen_selection.json"
    if selection_path.exists():
        selection = load_json(selection_path)
        if selection["signature"] != signature:
            raise RuntimeError("Frozen selection provenance mismatch")
    else:
        scorer = Scorer(manifests["circo"], index, features, projected, split="development")
        best, grid = {}, []
        for i, config in enumerate(candidates()):
            values = evaluate_config(scorer, config)
            mean = float(values.mean())
            row = {"config": config, "objective": mean,
                   "seed_metric_means": values.mean(axis=1).tolist()}
            grid.append(row)
            family = config["family"]
            if family not in best or mean > best[family]["objective"] + 1e-12:
                best[family] = row
            if i % 10 == 0:
                print(f"development grid {i + 1}/{len(provenance['grid'])} elapsed={time.monotonic() - started:.1f}s", flush=True)
        # Insertion order is the prespecified simple-to-complex family ordering.
        winner = max(best, key=lambda name: best[name]["objective"])
        selection = {"signature": signature, "provenance": provenance,
                     "frozen_at": datetime.now(timezone.utc).isoformat(),
                     "selection_dataset": "CIRCO local development only", "queries": len(scorer.queries),
                     "metric_names": scorer.names, "selected": best, "winner": winner}
        write_json(EVALUATION / "development_grid.json", grid)
        write_json(selection_path, selection)
        del scorer
        gc.collect()
        torch.cuda.empty_cache()
        print(f"selection frozen: {winner}; SHA256={sha256(selection_path)}", flush=True)
    configs = {family: row["config"] for family, row in selection["selected"].items()}
    configs.update({f"{family}_fixed_0.2": {"family": family, "text_weight": .2}
                    for family in ("clip_composition", *ARCHITECTURES)})
    for name, split in [("circo", "audit"), *[(f"genecis_{task}", "external") for task in TASKS]]:
        report_path = EVALUATION / f"{name}_{split}.json"
        if report_path.exists():
            if load_json(report_path)["signature"] != signature:
                raise RuntimeError(f"Stale evaluation {name}")
            continue
        scorer = Scorer(manifests[name], index, features, projected, split=split)
        groups = [q["group"] for q in scorer.queries]
        per_query, results = {}, {}
        for label, config in configs.items():
            values = evaluate_config(scorer, config)
            per_query[label] = values
            results[label] = {"config": config, "seed_metric_means": values.mean(axis=1).tolist(),
                              **cluster_interval(values.mean(axis=0), groups)}
        comparisons = {}
        for label in results:
            if label in ("clip_image", "clip_text", "dino_image"):
                continue
            comparisons[label] = {baseline: cluster_interval(
                per_query[label].mean(axis=0) - per_query[baseline].mean(axis=0), groups)
                for baseline in ("clip_composition", "late_fusion", "unaligned_residual") if label != baseline}
        with (EVALUATION / f"{name}_{split}_per_query.npz").open("wb") as handle:
            np.savez_compressed(handle, **per_query,
                                query_ids=np.array([q["id"] for q in scorer.queries]), groups=np.array(groups))
        report = {"signature": signature, "selection_sha256": sha256(selection_path),
                  "dataset": name, "split": split, "metric_names": scorer.names,
                  "queries": len(scorer.queries), "groups": len(set(groups)),
                  "chance": np.mean([random_average_precision(len(scorer.gallery_index) - 1, len(p))
                                      for p in scorer.positive_positions], axis=0).tolist() if scorer.is_circo else
                  [float(np.mean([min(k / len(q['candidates']), 1.) for q in scorer.queries])) for k in (1, 2, 3)],
                  "results": results, "paired_differences": comparisons,
                  "elapsed_seconds_since_run_start": time.monotonic() - started}
        write_json(report_path, report)
        print(f"evaluation complete {name}/{split} queries={len(scorer.queries)} "
              f"winner={selection['winner']} means={results[selection['winner']]['mean']}", flush=True)
        del scorer
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    run()
