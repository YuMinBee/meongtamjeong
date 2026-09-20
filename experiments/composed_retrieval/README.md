# CLIP–DINO composed retrieval data preparation

This experiment evaluates the project's core idea: mapping CLIP language features
into a DINO visual space and composing them with a reference image. See
[PROTOCOL.md](PROTOCOL.md) for the prospective method comparisons and split rules.

**GPU experiments authorized on 2026-09-16 after the data-only preparation completed.**
The preparation/validation commands below still import no Torch, CLIP or Transformers.
The existing `dog-rag` Conda environment has all required data tools.

The first GPU run is complete. See [RESULTS.md](RESULTS.md) for full quantitative
results for the historical run and its limitations.

## GPU experiments

From the repository root in the `dog-rag` environment (CUDA-enabled PyTorch):

```powershell
python -u -m experiments.composed_retrieval.run
```

This separate runner checks data readiness and executes frozen CLIP/DINO feature
extraction, matched COCO training (three heads × three seeds), development-only
selection, held-out/external evaluation and `RESULTS.md` generation. It records
its PID, stage, environment and code hashes in `artifacts/gpu_job.json`; a process
lock prevents duplicate runners. Do not run individual GPU stages concurrently.
Feature chunks and completed heads resume automatically. A changed protocol,
encoder, training implementation or evaluation implementation fails closed
against incompatible cached outputs.

The individual stages are `features`, `train`, `evaluate`, and `report` under
`experiments.composed_retrieval`. Evaluation writes `frozen_selection.json`
before accessing audit or external task scores. Per-query arrays preserve seeds
as a separate axis; bootstrap resamples reference groups after averaging seeds.

Data preparation completed with all 123,403 CIRCO gallery images, 5,000 COCO
validation images, 23,640 VG originals and 6,000 alignment images verified.
The combined feature catalog deduplicates assets across tasks: 163,486 image/crop
assets and 30,550 distinct text strings. ZIP files are read directly.

## Datasets

| Resource | Role | Local location |
|---|---|---|
| CIRCO official val annotations (220 queries) + full unlabeled2017 gallery (123,403 images) | Composed retrieval development/audit | `tmp/composed_retrieval/annotations/circo`, `unlabeled2017.zip` |
| GeneCIS v0, all four tasks (8,032 queries) | Conditional similarity external evaluation | `tmp/composed_retrieval/annotations/genecis`, `vg`, `val2017.zip` |
| COCO2017 train caption pairs, deterministic 6,000-image subset | Shared alignment training (5,000) / checkpoint validation (1,000) | `tmp/composed_retrieval/train2017` |
| CIRR rc2 train/val/test annotations and image ID lists | Future additional benchmark; raw images require official NLVR2 access | `tmp/composed_retrieval/annotations/cirr` |

The 6,000-image COCO subset excludes all 51,208 COCO IDs linked from Visual
Genome metadata before sampling, not just the current GeneCIS query references.
All captions for a photo stay together. CIRCO development/audit queries are grouped
by connected components of reference AND positive images, avoiding shared labeled
images between those subsets. CIRCO images remain in their original ZIP
to avoid a second 20 GB copy. The feature loader reads ZIP members directly.

## Reproduce/resume (data only)

From the repository root, in `dog-rag`:

```powershell
python -m experiments.composed_retrieval.download annotations
python -m experiments.composed_retrieval.download cirr-annotations
python -m experiments.composed_retrieval.download coco-annotations
python -m experiments.composed_retrieval.download coco-val
python -m experiments.composed_retrieval.download circo --workers 48
python -m experiments.composed_retrieval.prepare training --workers 24
python -m experiments.composed_retrieval.prepare vg --workers 48
python -m experiments.composed_retrieval.prepare manifests
python -m experiments.composed_retrieval.validate --deep
```

`prepare training` only downloads photos and builds a training-data manifest;
it does **not** train a model. Commands can resume completed downloads. Never
launch two downloaders writing the same target simultaneously.

`artifacts/data_readiness.json` is authoritative: readiness requires complete
archives, image integrity, positive/gallery checks, source-ID overlap checks,
and a deep validation pass. An absent/false readiness report means preparation
is not yet complete. `artifacts/` and raw data are Git-ignored.

To run/resume the entire **data-only** job, including automatic final CPU checks:

```powershell
python -u -m experiments.composed_retrieval.data_only
```

The job writes `artifacts/data_job.json` (`waiting_for_existing_downloads`, `downloading`, `validating_cpu_only`,
`complete`, or an explicit failure status). A lock prevents duplicate orchestrators.
Do not run individual download commands concurrently with that job. It exits
after data checks and has no subsequent model-training stage.

## Annotation fidelity

GeneCIS's released lists contain a small number of repeated candidate images.
Every original candidate slot is preserved; duplicates are recorded separately.
Candidate-slot IDs are stable SHA-256 IDs, with a separate `candidate_assets`
list. `positives` names the official positive SLOT, not every occurrence of the
same image. This avoids silently fixing labels or treating all duplicates as
positives. Stable slot order, rather than original positive-first order, is the
declared tie policy. The published data's duplicate/label noise remains a limit.

Attribute assets retain original floating-point `[x,y,width,height]` boxes and
the published GeneCIS v0 dilation/padding policy identifier; originals are not
cropped or overwritten during preparation. Encoding reproduces that policy
exactly. Object tasks use full COCO validation images. General-image
benchmarks must not use the application's dog-only detector.

## Provenance and access

- [CIRCO authors](https://github.com/miccunifi/CIRCO): pinned source revision and
  annotation hashes are in `tmp/composed_retrieval/sources.json`.
- [GeneCIS authors](https://github.com/facebookresearch/genecis): same manifest.
- [COCO](https://cocodataset.org/#download): official S3 image/annotation archives;
  `.source.json` records URL/size/ETag, `.sha256` records downloaded content hash.
- [Visual Genome author-maintained metadata source](https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip):
  metadata SHA-256 is preserved; photos use the Stanford URLs in each metadata
  record, saved under the annotation's image ID.
- [CIRR access instructions](https://github.com/Cuberick-Orion/CIRR#raw-images):
  NLVR2 form/terms and possibly author contact are required. No form/email has
  been submitted, and no restricted mirror is used. Annotations alone do not
  make CIRR ready for new CLIP/DINO embeddings.

Licenses for code, annotations and photographs are separate. This preparation
does not redistribute any downloaded images or claim that their licenses are
the application's Apache-2.0 license.
