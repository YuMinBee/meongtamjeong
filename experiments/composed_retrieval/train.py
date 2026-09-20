"""Matched COCO supervision for the existing linear, MLP and flow heads."""

from __future__ import annotations

import os
import random
import time
from pathlib import Path

import numpy as np

from experiments.composed_retrieval.features import digest, load_features
from experiments.composed_retrieval.download import sha256
from experiments.composed_retrieval.prepare import ARTIFACTS, load_json, write_json

SEEDS = (7, 42, 123)
ARCHITECTURES = ("linear", "mlp", "flow")
TRAINING = {"epochs": 30, "batch_size": 256, "learning_rate": .001,
            "weight_decay": .0001, "temperature": .07,
            "loss": "text_to_image_multi_positive_logsumexp",
            "checkpoint_selection": "all_validation_captions_to_1000_images_cross_entropy"}
MODELS = ARTIFACTS / "heads"


def multi_positive_loss(mapped, visual, text_labels, visual_labels, temperature=.07):
    import torch

    logits = mapped @ visual.T / temperature
    positives = text_labels[:, None] == visual_labels[None, :]
    if not bool(positives.any(dim=1).all()):
        raise ValueError("Every text must have at least one positive image")
    return (torch.logsumexp(logits, dim=1) -
            torch.logsumexp(logits.masked_fill(~positives, -torch.inf), dim=1)).mean()


def run():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from experiments.dino_fusion.alignment import make_projection_head, project_embeddings

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    index, features = load_features()
    image_index = {key: i for i, key in enumerate(index["image_keys"])}
    text_index = {key: i for i, key in enumerate(index["texts"])}
    records = load_json(ARTIFACTS / "training_manifest.json")["records"]
    images = torch.tensor(np.stack([features["dino"][image_index[f"train:{r['image_id']}"]]
                                    for r in records]), device="cuda")
    del features["clip"], features["dino"]
    text_ids, labels, train_rows, val_rows, val_images = [], [], [], [], []
    for i, record in enumerate(records):
        if record["split"] == "validation":
            val_images.append(i)
        for caption in record["captions"]:
            rows = train_rows if record["split"] == "train" else val_rows
            rows.append(len(labels))
            labels.append(i)
            text_ids.append(text_index[caption])
    texts = torch.tensor(features["text"][text_ids], device="cuda")
    labels = torch.tensor(labels, dtype=torch.long, device="cuda")
    train_rows = torch.tensor(train_rows, dtype=torch.long, device="cuda")
    val_rows = torch.tensor(val_rows, dtype=torch.long, device="cuda")
    val_images = torch.tensor(val_images, dtype=torch.long, device="cuda")
    signature = digest({"features": index["signature"], "config": TRAINING,
                        "implementation": "matched-heads-v1",
                        "train_code": sha256(Path(__file__)),
                        "head_code": sha256(Path(__file__).parents[1] / "dino_fusion/alignment.py")})
    MODELS.mkdir(exist_ok=True)
    for architecture in ARCHITECTURES:
        for seed in SEEDS:
            name = f"{architecture}_{seed}"
            checkpoint = MODELS / f"{name}.pt"
            report = MODELS / f"{name}.json"
            if report.exists():
                previous = load_json(report)
                if previous["signature"] != signature:
                    raise RuntimeError(f"Stale model {name}")
                if previous["complete"]:
                    print(f"training cached {name}", flush=True)
                    continue
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            head = make_projection_head(architecture, input_dim=512, output_dim=768).cuda()
            optimizer = torch.optim.AdamW(head.parameters(), lr=TRAINING["learning_rate"],
                                          weight_decay=TRAINING["weight_decay"])
            best, best_epoch, history = float("inf"), None, []
            started = time.monotonic()
            for epoch in range(1, TRAINING["epochs"] + 1):
                head.train()
                order = train_rows[torch.randperm(len(train_rows), device="cuda")]
                total = 0.
                for batch in order.split(TRAINING["batch_size"]):
                    optimizer.zero_grad(set_to_none=True)
                    mapped = project_embeddings(head, texts[batch])
                    loss = multi_positive_loss(mapped, images[labels[batch]], labels[batch],
                                               labels[batch], TRAINING["temperature"])
                    loss.backward()
                    optimizer.step()
                    total += loss.item() * len(batch)
                head.eval()
                validation = 0.
                with torch.inference_mode():
                    for batch in val_rows.split(TRAINING["batch_size"]):
                        mapped = project_embeddings(head, texts[batch])
                        loss = multi_positive_loss(mapped, images[val_images], labels[batch],
                                                   val_images, TRAINING["temperature"])
                        validation += loss.item() * len(batch)
                validation /= len(val_rows)
                if not np.isfinite(validation):
                    raise RuntimeError(f"Non-finite validation loss: {name}")
                if validation < best:
                    best, best_epoch = validation, epoch
                    temporary = checkpoint.with_suffix(".partial")
                    torch.save({"state_dict": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                                "architecture": architecture, "seed": seed,
                                "input_dim": 512, "output_dim": 768,
                                "signature": signature, "feature_signature": index["signature"],
                                "epoch": epoch, "validation_loss": best}, temporary)
                    temporary.replace(checkpoint)
                history.append({"epoch": epoch, "train_loss": total / len(train_rows),
                                "validation_loss": validation})
                write_json(report, {"complete": epoch == TRAINING["epochs"],
                                    "signature": signature, "feature_signature": index["signature"],
                                    "architecture": architecture, "seed": seed, "config": TRAINING,
                                    "parameters": sum(p.numel() for p in head.parameters()),
                                    "best_epoch": best_epoch, "best_validation_loss": best,
                                    "training_captions": len(train_rows), "validation_captions": len(val_rows),
                                    "elapsed_seconds": time.monotonic() - started, "history": history})
                if epoch == 1 or epoch % 5 == 0:
                    print(f"training {name} epoch={epoch}/30 train={history[-1]['train_loss']:.4f} "
                          f"val={validation:.4f} best={best:.4f} elapsed={time.monotonic() - started:.1f}s",
                          flush=True)
            del head, optimizer


if __name__ == "__main__":
    run()
