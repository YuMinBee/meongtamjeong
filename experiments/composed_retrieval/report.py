"""Render aggregate research results without distributing raw photographs."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.composed_retrieval.download import sha256
from experiments.composed_retrieval.evaluate import EVALUATION
from experiments.composed_retrieval.features import CACHE
from experiments.composed_retrieval.prepare import TASKS, load_json
from experiments.composed_retrieval.train import ARCHITECTURES, MODELS, SEEDS


def percent(values):
    return " / ".join(f"{v * 100:.2f}" for v in values)


def run():
    selection = load_json(EVALUATION / "frozen_selection.json")
    reports = {"CIRCO audit": load_json(EVALUATION / "circo_audit.json")}
    reports.update({task: load_json(EVALUATION / f"genecis_{task}_external.json") for task in TASKS})
    provenance = load_json(CACHE / "index.json")["provenance"]
    winner = selection["winner"]
    lines = ["# CLIP–DINO composed retrieval: experiment 1", "",
             "Completed frozen-protocol experiment. Scores below are percentages.", "",
             f"Development-selected family: **{winner}**. This choice was frozen before",
             "the local CIRCO audit and all four GeneCIS evaluations. Family hyperparameters",
             "were selected separately using mean development mAP@5/10/25/50 across seeds.", "",
             "## Data and interpretation", "",
             "- COCO paired supervision: 5,000 training / 1,000 validation images; all captions.",
             "- CIRCO: 116 development / 104 audit queries; full 123,403-image gallery.",
             "  Reference/positive connected components keep labeled images out of both splits.",
             "  These are **local partitions of official validation**, not official test results.",
             "- GeneCIS: all 8,032 queries across four official tasks; no task-specific tuning.",
             "  Official candidate slots and duplicate labels are preserved.",
             "- All 51,208 VG-linked COCO IDs excluded before training-image sampling.",
             "  Source-ID and exact decoded-image checks passed; pretraining overlap is unknown.",
             "- CIRR has annotations only and is not scored.", "",
             "## Encoders and methods", "",
             f"CLIP `{provenance['clip']}` weights SHA-256 `{provenance['clip_weights_sha256']}`.",
             f"DINO `{provenance['dino']}` revision `{provenance['dino_revision']}`.",
             f"GPU: {provenance['gpu']}; torch {provenance['torch']}.",
             "Both frozen encoders use their native preprocessing of the same image/crop.",
             "CLIP uses resize/center-crop; DINO uses its released 224×224 processor.",
             "Raw captions/conditions are tokenized with CLIP truncation enabled.", "",
             "For normalized features c(image), t(text), d(image), and mapped normalized text h(t):", "",
             "- CLIP composition: cosine(normalize((1−w)c(reference)+w·t), c(candidate)).",
             "- Aligned DINO: cosine(normalize((1−w)d(reference)+w·h(t)), d(candidate)).",
             "- Late fusion: (1−w)·DINO image cosine + w·CLIP text-to-image cosine.",
             "- Residual: r·aligned DINO composition + (1−r)·CLIP composition, same w.",
             "- Unaligned residual control: r·DINO image cosine + (1−r)·CLIP composition.",
             "- Linear/MLP/flow share caption pairs, loss, optimizer and 30-epoch budget;",
             "  checkpoints selected by COCO validation loss. Seeds: 7, 42, 123.",
             "  These methods use supervised paired-data training and are not training-free.", "",
             "## Frozen choices", "",
             "| Family | Text weight w | DINO residual weight r | Development mean mAP |",
             "|---|---:|---:|---:|"]
    for family, row in selection["selected"].items():
        config = row["config"]
        lines.append(f"| {family} | {config.get('text_weight', '—')} | {config.get('residual_weight', '—')} | {100 * row['objective']:.2f} |")
    lines.extend(["", "## Held-out results", "",
                  "Learned methods show the mean across three seeds. Baselines are deterministic.",
                  "CIRCO columns are mAP@5 / @10 / @25 / @50; GeneCIS columns are Recall@1.", "",
                  "| Method | CIRCO audit | Focus attribute | Change attribute | Focus object | Change object |",
                  "|---|---:|---:|---:|---:|---:|"])
    for label in reports["CIRCO audit"]["results"]:
        cells = [percent(reports["CIRCO audit"]["results"][label]["mean"])]
        cells.extend(f"{reports[task]['results'][label]['mean'][0] * 100:.2f}" for task in TASKS)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.extend(["", "## Paired uncertainty for the development-selected family", "",
                  "Differences are percentage points, with 95% paired cluster-bootstrap intervals",
                  "(2,000 resamples). Seed scores are averaged per query before resampling.",
                  "CIRCO clusters are connected components; GeneCIS clusters are reference images.",
                  "CIRCO below uses mAP@5; GeneCIS uses Recall@1. These exploratory comparisons",
                  "are not adjusted for multiple comparisons.", "",
                  "| Dataset | Baseline | Difference | 95% CI |",
                  "|---|---|---:|---:|"])
    for dataset, report in reports.items():
        for baseline, values in report["paired_differences"].get(winner, {}).items():
            lines.append(f"| {dataset} | {baseline} | {values['mean'][0] * 100:+.2f} | "
                         f"[{values['ci95_low'][0] * 100:+.2f}, {values['ci95_high'][0] * 100:+.2f}] |")
    lines.extend(["", "## Head training", "",
                  "Validation loss uses all held-out captions against the entire 1,000-image validation gallery.", "",
                  "| Head | Parameters | Best epochs (7/42/123) | Validation losses | Training seconds (sum) |",
                  "|---|---:|---|---|---:|"])
    for architecture in ARCHITECTURES:
        rows = [load_json(MODELS / f"{architecture}_{seed}.json") for seed in SEEDS]
        lines.append(f"| {architecture} | {rows[0]['parameters']:,} | "
                     + " / ".join(str(row["best_epoch"]) for row in rows) + " | "
                     + " / ".join(f"{row['best_validation_loss']:.4f}" for row in rows)
                     + f" | {sum(row['elapsed_seconds'] for row in rows):.1f} |")
    lines.extend(["", "## Full task metrics and seed variability", ""])
    for dataset, report in reports.items():
        lines.extend([f"### {dataset}", "", f"Queries: {report['queries']}; clusters: {report['groups']}.",
                      "Metrics: " + " / ".join(report["metric_names"]) + ".", "",
                      "| Method | Mean | 95% CI lower | 95% CI upper | Seed SD |",
                      "|---|---|---|---|---|"])
        for label, result in report["results"].items():
            rows = np.asarray(result["seed_metric_means"])
            sd = percent(rows.std(axis=0, ddof=1)) if len(rows) > 1 else "—"
            lines.append(f"| {label} | {percent(result['mean'])} | {percent(result['ci95_low'])} | "
                         f"{percent(result['ci95_high'])} | {sd} |")
        if report["chance"] is not None:
            lines.extend(["", "Random-ranking expectation: " + percent(report["chance"]) + "."])
        lines.append("")
    lines.extend(["## Limits and reproducibility", "",
                  "This is a pilot at a fixed 5,000-image training scale. A small local CIRCO audit",
                  "and four related GeneCIS tasks do not establish broad superiority or novelty.",
                  "The heads are different parameter counts, though data and optimization budgets match.",
                  "CLIP and DINO have different pretraining and native preprocessing; neither is controlled",
                  "by this experiment. English general-image benchmarks do not establish Korean-language",
                  "or lost-dog retrieval performance. Flow is the project's existing hyperspherical mapper, not an",
                  "official reproduction of another paper. Further changes require a new prospective",
                  "protocol and fresh held-out data, rather than tuning on these audit outcomes.", "",
                  f"Selection artifact SHA-256: `{sha256(EVALUATION / 'frozen_selection.json')}`.",
                  "Raw per-query, per-seed metrics: `artifacts/evaluation/*_per_query.npz`.",
                  "Machine-readable full results and paired differences: `artifacts/evaluation/*.json`.",
                  "Data integrity: `artifacts/data_readiness.json`; encoder/crop/source hashes:",
                  "`artifacts/features/index.json` and image chunks. All large/raw artifacts are Git-ignored.", "",
                  "Primary benchmark sources: [CIRCO](https://github.com/miccunifi/CIRCO),",
                  "[GeneCIS](https://github.com/facebookresearch/genecis).", ""])
    path = Path(__file__).with_name("RESULTS.md")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"report written: {path}", flush=True)


if __name__ == "__main__":
    run()
