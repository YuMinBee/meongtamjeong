"""Frozen visual transfer on the publisher's MPDD split; no text proxy labels."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import io
from pathlib import Path, PurePosixPath
import time
import zipfile

import numpy as np
from PIL import Image, ImageFilter

from experiments.composed_retrieval.download import ROOT, fetch, sha256
from experiments.composed_retrieval.features import canonical_provenance, digest
from experiments.composed_retrieval.metrics import cluster_interval
from experiments.composed_retrieval.prepare import load_json, write_json

OUT = Path(__file__).parent / "artifacts"
ARCHIVE = ROOT / "tmp/dog_retrieval/MPDD.zip"
EXPECTED = "6c800c1b4aa67629544dec7444dee85bc57781abde1ae1d077ea9ef1804284cd"
URL = "https://data.mendeley.com/public-files/datasets/v5j6m8dzhv/files/05d1d583-faf6-410d-89a5-a6b1134b6e5e/file_downloaded"
METRICS = ["Recall@1", "Recall@5", "Recall@10", "mAP", "MRR"]


def inspect_archive():
    fetch(URL, ARCHIVE)
    if sha256(ARCHIVE) != EXPECTED:
        raise ValueError("Publisher archive checksum mismatch")
    records = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        if archive.testzip() is not None:
            raise ValueError("Archive CRC failure")
        for member in sorted(archive.namelist()):
            if not member.lower().endswith(".jpg"):
                continue
            path = PurePosixPath(member)
            tokens = path.stem.split("_")
            payload = archive.read(member)
            with Image.open(io.BytesIO(payload)) as opened:
                image = opened.convert("RGB")
                image.load()
            records.append({"member": member, "split": path.parent.name, "identity": tokens[0],
                            "c_code": tokens[1].split("s")[0], "size": list(image.size),
                            "sha256": hashlib.sha256(payload).hexdigest(),
                            "pixels_sha256": hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()})
    ids = {split: {r["identity"] for r in records if r["split"] == split}
           for split in ("train", "val", "query", "gallery")}
    if (ids["train"] | ids["val"]) & (ids["query"] | ids["gallery"]):
        raise ValueError("Identity leakage across development and test")
    if ids["val"] - ids["train"] or ids["query"] - ids["gallery"]:
        raise ValueError("Missing gallery identity")
    if len(records) != 1657:
        raise ValueError("Unexpected image count")
    duplicates = defaultdict(list)
    for r in records:
        duplicates[r["pixels_sha256"]].append(r["member"])
    report = {"archive_sha256": EXPECTED, "source": URL, "license": "CC BY 4.0",
              "images": len(records), "identities": len({r['identity'] for r in records}),
              "split_images": dict(Counter(r["split"] for r in records)),
              "split_identities": {key: len(value) for key, value in ids.items()},
              "decode_failures": 0, "cross_split_identity_overlap": 0,
              "decoded_duplicate_groups": [v for v in duplicates.values() if len(v) > 1],
              "records": records}
    write_json(OUT / "data_audit.json", report)
    return records


def perturb(image, condition):
    if condition == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=2))
    if condition == "lowres":
        size = image.size
        ratio = min(1., 64 / max(size))
        small = tuple(max(1, round(v * ratio)) for v in size)
        return image.resize(small, Image.Resampling.BILINEAR).resize(size, Image.Resampling.BILINEAR)
    if condition != "clean":
        raise ValueError(condition)
    return image


def encode(records):
    import torch
    from experiments.dino_fusion.core import ClipEncoder, DinoEncoder, normalize_rows

    torch.set_num_threads(4)
    clip = ClipEncoder(device="cuda")
    dino = DinoEncoder(device="cuda", local_files_only=True)
    provenance = canonical_provenance({"clip": clip.model_name, "clip_preprocess": str(clip._preprocess),
                  "clip_sha256": sha256(Path.home() / '.cache/clip/ViT-B-32.pt'),
                  "dino": dino.model_id, "revision": dino.resolved_revision,
                  "dino_processor": dino._processor.to_dict(), "pooling": dino.pooling,
                  "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                  "records": digest(records), "protocol": sha256(Path(__file__).with_name('PROTOCOL.md')),
                  "code_sha256": sha256(Path(__file__))})
    signature = digest(provenance)
    result = {}
    started = time.monotonic()
    with zipfile.ZipFile(ARCHIVE) as archive:
        for condition in ("clean", "blur", "lowres"):
            selected = list(range(len(records))) if condition == "clean" else [
                i for i, r in enumerate(records) if r["split"] == "query"]
            path = OUT / f"features_{condition}.npz"
            if path.exists():
                with np.load(path) as data:
                    if str(data["signature"]) != signature:
                        raise ValueError("Incompatible cache")
                    result[condition] = {k: data[k] for k in ("indices", "clip", "dino")}
                continue
            clip_rows, dino_rows = [], []
            for start in range(0, len(selected), 64):
                images = []
                for i in selected[start:start + 64]:
                    with Image.open(io.BytesIO(archive.read(records[i]["member"]))) as opened:
                        images.append(perturb(opened.convert("RGB"), condition))
                tensors = torch.stack([clip._preprocess(im) for im in images]).cuda()
                with torch.inference_mode():
                    clip_rows.append(normalize_rows(clip._model.encode_image(tensors).float().cpu().numpy()))
                dino_rows.append(dino.encode_batch(images))
            result[condition] = {"indices": np.array(selected), "clip": np.concatenate(clip_rows),
                                 "dino": np.concatenate(dino_rows)}
            with path.with_suffix('.partial').open('wb') as handle:
                np.savez(handle, **result[condition], signature=signature)
            path.with_suffix('.partial').replace(path)
            print(f"encoded {condition}: {len(selected)} images", flush=True)
    write_json(OUT / "encoders.json", {"signature": signature, "provenance": provenance,
                                      "elapsed_seconds_this_invocation": time.monotonic() - started})
    return result, signature


def rank_metrics(scores, queries, gallery, *, exclude_c_code=False):
    rows, valid, excluded = [], [], []
    for query, score in zip(queries, scores):
        keep = np.array([r["pixels_sha256"] != query["pixels_sha256"] and not (
            exclude_c_code and r["identity"] == query["identity"] and r["c_code"] == query["c_code"])
            for r in gallery])
        positives = np.array([r["identity"] == query["identity"] for r in gallery]) & keep
        if not positives.any():
            excluded.append(query["member"])
            continue
        eligible = np.flatnonzero(keep)
        ranking = eligible[np.argsort(-score[eligible], kind="stable")]
        hits = positives[ranking]
        ranks = np.flatnonzero(hits) + 1
        rows.append([float(hits[:k].any()) for k in (1, 5, 10)] +
                    [float(np.mean(np.arange(1, len(ranks) + 1) / ranks)), float(1 / ranks[0])])
        valid.append(query)
    if not rows:
        raise ValueError("No evaluable queries")
    return np.array(rows), valid, excluded


def identity_means(values, queries):
    ids = sorted({q["identity"] for q in queries})
    return np.stack([values[[q["identity"] == identity for q in queries]].mean(axis=0) for identity in ids]), ids


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    records = inspect_archive()
    features, signature = encode(records)
    clean = features["clean"]
    train_ids = [i for i, r in enumerate(records) if r["split"] == "train"]
    val_ids = [i for i, r in enumerate(records) if r["split"] == "val"]
    query_ids = [i for i, r in enumerate(records) if r["split"] == "query"]
    gallery_ids = [i for i, r in enumerate(records) if r["split"] == "gallery"]
    selection_path = OUT / "frozen_selection.json"
    if selection_path.exists():
        selection = load_json(selection_path)
        if selection["signature"] != signature:
            raise ValueError("Selection signature mismatch")
    else:
        ci = clean["clip"][val_ids] @ clean["clip"][train_ids].T
        di = clean["dino"][val_ids] @ clean["dino"][train_ids].T
        grid = []
        for weight in (0., 1., .25, .5, .75):
            values, valid, excluded = rank_metrics((1 - weight) * ci + weight * di,
                [records[i] for i in val_ids], [records[i] for i in train_ids])
            macro, _ = identity_means(values, valid)
            grid.append({"dino_weight": weight, "identity_macro": macro.mean(axis=0).tolist(), "excluded": excluded})
        best = max(grid, key=lambda row: row["identity_macro"][3])
        selection = {"signature": signature, "grid": grid, "selected_dino_weight": best["dino_weight"],
                     "selection": "distributed val queries vs train gallery, identity-macro mAP"}
        write_json(selection_path, selection)
    print(f"Frozen DINO weight: {selection['selected_dino_weight']}", flush=True)
    output = {"signature": signature, "selection_sha256": sha256(selection_path), "metric_names": METRICS,
              "conditions": {}}
    queries, gallery = [records[i] for i in query_ids], [records[i] for i in gallery_ids]
    for condition in features:
        f = features[condition]
        positions = {int(i): j for j, i in enumerate(f["indices"])}
        ci = f["clip"][[positions[i] for i in query_ids]] @ clean["clip"][gallery_ids].T
        di = f["dino"][[positions[i] for i in query_ids]] @ clean["dino"][gallery_ids].T
        for sensitivity in (False, True):
            key = condition + ("_different_c_code" if sensitivity else "")
            results, arrays, identities = {}, {}, None
            for method, weight in {"CLIP": 0., "DINO": 1., "equal_fusion": .5,
                                   "selected_fusion": selection["selected_dino_weight"]}.items():
                values, valid, excluded = rank_metrics((1 - weight) * ci + weight * di, queries, gallery,
                                                       exclude_c_code=sensitivity)
                macro, identities = identity_means(values, valid)
                arrays[method] = macro
                arrays[f"{method}_per_query"] = values
                results[method] = {"dino_weight": weight, "query_micro_mean": values.mean(axis=0).tolist(),
                                   "excluded_queries": excluded, "evaluated_queries": len(valid),
                                   **cluster_interval(macro, identities)}
            differences = {method: cluster_interval(arrays[method] - arrays["CLIP"], identities)
                           for method in ("DINO", "equal_fusion", "selected_fusion")}
            differences["selected_minus_DINO"] = cluster_interval(arrays["selected_fusion"] - arrays["DINO"], identities)
            np.savez_compressed(OUT / f"{key}_per_query.npz", **arrays, identities=np.array(identities),
                                query_members=np.array([q["member"] for q in valid]))
            output["conditions"][key] = {"results": results, "paired_differences": differences}
            print(key, {m: r["mean"] for m, r in results.items()}, flush=True)
    write_json(OUT / "results.json", output)
    audit = load_json(OUT / 'data_audit.json')
    lines = ["# MPDD dog retrieval results", "", "Frozen CLIP/DINO visual transfer; no head training or text input.",
             "This experiment cannot establish a benefit from CLIP-text-to-DINO Flow alignment.", "",
             f"Data: {audit['images']} photos, {audit['identities']} filename IDs (publisher describes 192).",
             f"Split images: {audit['split_images']}. Split IDs: {audit['split_identities']}.",
             f"Decoded duplicate groups: {len(audit['decoded_duplicate_groups'])}.",
             f"Selected DINO score weight: {selection['selected_dino_weight']} (clean validation only).", "",
             "Scores are percentages, macro-averaged over identities. No published-baseline comparability is implied.",
             "Query perturbations retain the clean gallery. Different-c-code is a filename-code sensitivity analysis.", "",
             "| Condition | Method | R@1 | R@5 | R@10 | mAP | MRR |", "|---|---|---:|---:|---:|---:|---:|"]
    for condition, values in output["conditions"].items():
        for method, result in values["results"].items():
            lines.append(f"| {condition} | {method} | " + ' | '.join(f"{v*100:.2f}" for v in result['mean']) + ' |')
    lines.extend(["", "## Paired differences from CLIP, percentage points", "",
                  "95% bootstrap intervals over dog IDs, 2,000 resamples; exploratory, no multiplicity correction.", "",
                  "| Condition | Contrast | R@1 difference [95% CI] | mAP difference [95% CI] |",
                  "|---|---|---|---|"])
    for condition, values in output["conditions"].items():
        for method, result in values["paired_differences"].items():
            cells = [f"{100*result['mean'][i]:+.2f} [{100*result['ci95_low'][i]:+.2f}, {100*result['ci95_high'][i]:+.2f}]" for i in (0,3)]
            lines.append(f"| {condition} | {method} | " + ' | '.join(cells) + ' |')
    lines.extend(["", "## Scope", "", "Same-ID photos may still share acquisition backgrounds; exact-hash checks do not rule out near duplicates.",
                  "No camera/time metadata was verified. Synthetic blur/downsampling is not evidence about real missing-dog deployment.",
                  "Foundation-model pretraining overlap is unknown. Results are from a small released dog dataset.",
                  "Native preprocessing differs between encoders. No application configuration was changed.", "",
                  "Source: Zhimin He (2023), [MPDD v1](https://data.mendeley.com/datasets/v5j6m8dzhv/1), CC BY 4.0.",
                  "See PROTOCOL.md and artifacts/{data_audit,encoders,frozen_selection,results}.json for provenance.", ""])
    Path(__file__).with_name('RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    run()
