# CLIP + DINO shared-space retrieval experiment

This directory keeps the experiment source and generated artifacts separate
from the tracked production `data/dog_faiss.index`. The opt-in adapter in
`app/dino_fusion.py` imports the selected components only when the rollout is
enabled. The experiment tests whether DINOv3's visual structure signal adds
value to the existing CLIP image retrieval, then trains a small text projection
without changing either frozen foundation encoder.

## Repository hygiene

- Track the Python source, tests, fixed query definitions, and Markdown result
  summaries under `experiments/dino_fusion/`.
- Keep generated checkpoints, indices, embeddings, and machine-readable reports
  under an `artifacts/` directory. Each experiment artifact directory contains
  its own deny-all `.gitignore`.
- Keep downloaded datasets, refreshed local snapshots, detector crops, and
  provider archives under the repository-root `tmp/`; the root `.gitignore`
  excludes that directory.
- Keep Codex/IDE HTML previews under `.codex/visualizations/`; they remain local
  and are not the canonical evaluation record.
- Never use `git add -f` for these generated paths. Preserve publishable metrics
  and limitations in the tracked result summaries instead.

## Stage 1: does the visual pilot train anything?

No. Both OpenAI CLIP and DINOv3 are frozen. DINO vectors are stored in a
separate cosine-similarity FAISS index, and CLIP/DINO result rankings are fused
per notice with weighted reciprocal-rank fusion (RRF). RRF is used because
CLIP's L2 distances and DINO's cosine scores do not share a calibrated scale.

Stage 2 now implements that later step: small Linear and MLP heads map frozen
CLIP text features onto the frozen DINO visual manifold. DINO and CLIP remain
frozen.

## Prerequisites

1. Use the existing full runtime environment and install
   `experiments/dino_fusion/requirements.txt` if `transformers` is absent.
2. Request access to the gated Meta DINOv3 model on Hugging Face, accept its
   license, and authenticate with `huggingface-cli login`.
3. Review the DINOv3 redistribution and attribution requirements before using
   an experiment artifact in a public release. Do not commit a Hugging Face
   token or downloaded model cache.

The default backbone is DINOv3 ViT-B/16. On first use, Hugging Face downloads
the frozen weights into its normal cache. Generated indices and manifests go
under `artifacts/`, which is ignored by Git.

If DINOv3 access has not been approved yet, the exact same pipeline can be
smoke-tested with the public DINOv2 Base checkpoint. Treat this only as an
infrastructure pilot, not as a DINOv3/DINOde result:

```powershell
python -m experiments.dino_fusion.build_index `
  --model-id facebook/dinov2-base `
  --source crop_image `
  --limit 32
```

## 1. Smoke-build a crop-only index

The repository already has Faster R-CNN dog crops locally. This command avoids
network image downloads and is the quickest end-to-end check:

```powershell
python -m experiments.dino_fusion.build_index `
  --source crop_image `
  --limit 32 `
  --batch-size 16
```

Remove `--limit` for all available crops. Build a full experimental index with
both the locally stored crops and allowlisted public primary images:

```powershell
python -m experiments.dino_fusion.build_index `
  --source full_image `
  --source crop_image `
  --batch-size 16 `
  --overwrite
```

This writes only:

```text
experiments/dino_fusion/artifacts/
├── dino.index
├── dino_metas.json
└── dino_manifest.json
```

## 2. Search and fuse CLIP+DINO rankings

```powershell
python -m experiments.dino_fusion.search assets/samples/notice_428349202600501.jpg
```

The JSON response reports the independent CLIP and DINO ranks and each RRF
contribution. `--clip-weight` and `--dino-weight` are experimental parameters;
do not tune them on the held-out test set.

## Evaluation order

Compare the following under the same candidate corpus and query set:

1. existing CLIP image retrieval;
2. frozen DINOv3 image retrieval;
3. training-free CLIP+DINO RRF;
4. only if (3) is promising, a trained CLIP-text-to-DINO alignment head.

Primary metrics are same-notice Hit@1/5/10 and MRR for distinct secondary
photos, plus latency and peak VRAM. Report full-image and Faster R-CNN crop
ablations separately. The existing held-out evaluation has limited evaluable
coverage, so a gain on that subset is a signal for further study, not a final
claim.

After building the complete crop index, the reproducible subset of the prior
held-out run can be evaluated without saving query images:

```powershell
python -m experiments.dino_fusion.evaluate_pilot --local-files-only
```

The evaluator verifies both the stored secondary-URL hash and downloaded image
payload hash before including a query. It compares production CLIP full+crop,
same-corpus CLIP crop-only, DINO crop-only, and equal-weight crop-only RRF.
It also runs controlled Korean and English color/size prompts directly against
the crop images, evaluates current CLIP image+text averaging, and evaluates
DINO-image/CLIP-text rank fusion. Exact-notice metrics and coarse color+size
silver relevance are reported separately.

## 3. Train the shared-space heads

The complete held-out sample pool is excluded before training. The loss treats
all images with the same public color, weight-derived size, and age group as
positives so a generic description is not forced to identify one arbitrary dog.
English and Korean paraphrases are both used. Validation holds out notice IDs
and uses sentence patterns absent from training.

```powershell
python -m experiments.dino_fusion.train_alignment `
  --dino-index experiments/dino_fusion/artifacts/dinov3/dino.index `
  --dino-metas experiments/dino_fusion/artifacts/dinov3/dino_metas.json `
  --dino-manifest experiments/dino_fusion/artifacts/dinov3/dino_manifest.json `
  --output-dir experiments/dino_fusion/artifacts/dinov3
```

This trains only a 393,216-parameter Linear head and a 656,640-parameter MLP.
Checkpoints and reports stay in the ignored artifact directory.

## 4. Evaluate unseen notices and natural-language phrasings

```powershell
python -m experiments.dino_fusion.evaluate_alignment `
  --dino-index experiments/dino_fusion/artifacts/dinov3/dino.index `
  --dino-metas experiments/dino_fusion/artifacts/dinov3/dino_metas.json `
  --dino-manifest experiments/dino_fusion/artifacts/dinov3/dino_manifest.json `
  --training-report experiments/dino_fusion/artifacts/dinov3/alignment_training_report.json `
  --output experiments/dino_fusion/artifacts/dinov3/alignment_evaluation_report.json `
  --local-files-only
```

The evaluation compares raw CLIP text retrieval, projected text retrieval in
the DINO index, DINO image retrieval, and a fixed 20% text / 80% image blend in
one normalized DINO space. See `ALIGNMENT_RESULTS.md` for the pilot outcome.

## 5. Verify that learning, not dimension matching, causes the gain

```powershell
python -m experiments.dino_fusion.evaluate_alignment_controls
```

This compares the validation-selected trained MLP against five untrained random
heads and five heads trained for the same 30 updates after randomly permuting
the DINO image associations. It reuses the exact payload-verified held-out text
queries without downloading images again. The generated control report remains
under the ignored `artifacts/dinov3/` directory.

## 6. Apply a DINOde-style ODE flow to this corpus

The retrieval adaptation keeps both encoders frozen and replaces the static
text projection with an eight-step time-conditioned tangent flow on the unit
sphere. It is independently implemented here and does not copy the upstream
DINOde segmentation code.

Train Linear, MLP, and capacity-matched flow heads without overwriting the
earlier artifacts:

```powershell
python -m experiments.dino_fusion.train_alignment `
  --dino-index experiments/dino_fusion/artifacts/dinov3/dino.index `
  --dino-metas experiments/dino_fusion/artifacts/dinov3/dino_metas.json `
  --dino-manifest experiments/dino_fusion/artifacts/dinov3/dino_manifest.json `
  --output-dir experiments/dino_fusion/artifacts/dinode_flow `
  --architecture linear `
  --architecture mlp `
  --architecture flow
```

Evaluate with the same held-out protocol, or query the real experimental dog
corpus directly:

```powershell
python -m experiments.dino_fusion.aligned_search `
  --text "검정색과 흰색 털이 섞인 작은 강아지를 찾아줘" `
  --topk 10 `
  --local-files-only
```

See `DINODE_FLOW_RESULTS.md` for the completed comparison. Flow achieved the
best pilot text-retrieval nDCG@10 in both English (0.5111) and Korean (0.7221),
while the default application route remains unchanged pending a fresh
human-labeled evaluation. The runtime adapter described below is opt-in.

## 7. Add an unknown-aware public-notice behavior branch

Static images cannot establish personality. This follow-up keeps DINOv3, CLIP,
and the existing flow head frozen, then trains a 66k-parameter behavior head on
explicitly opposed phrases in provenance-bearing public notice text. Missing
traits and conflicting phrases abstain from the masked loss instead of becoming
negative labels.

```powershell
C:\Users\sally\anaconda3\envs\dog-rag\python.exe `
  -m experiments.dino_fusion.train_behavior `
  --local-files-only
```

The controlled silver-label pilot compares CLIP image-space text retrieval,
DINO-flow image-space text retrieval, untrained notice-text retrieval, the
trained behavior head, and their fixed-weight fusion. See
`BEHAVIOR_PILOT_RESULTS.md` for the result and limitations. Generated behavior
checkpoints and reports remain under ignored `artifacts/behavior_pilot/`.

## 8. Compare both systems with your own image and text

Run the isolated local demo without changing the production API:

```powershell
conda run -n meong-contest-full python -m uvicorn `
  experiments.dino_fusion.demo_app:app `
  --host 127.0.0.1 `
  --port 8765
```

Open `http://127.0.0.1:8765`, upload a photo, and enter Korean or English text.
The page searches the same 1,002-crop candidate corpus in two columns: the
matched CLIP image+text baseline and the DINOv3 image+flow-aligned CLIP text
system. The pilot's fixed blend uses 80% image and 20% text by default.

For the fresh human comparison, use the fixed 30-query set in
[`human_eval_queries.v1.json`](human_eval_queries.v1.json). Every query contains
both a CC0 reference image and Korean text. Ten fixed images are repeated across
image+appearance, image+personality, and image+appearance+personality conditions
so the effect of each text branch can be compared as paired cases. The readable
design and scoring contract are in
[`HUMAN_EVAL_QUERIES.md`](HUMAN_EVAL_QUERIES.md).

## 9. Roll the pilot into the application without replacing CLIP

`app/dino_fusion.py` validates and lazily loads these ignored artifacts. The
default `DINO_FUSION_MODE=off` neither imports Transformers nor changes a
ranking. Use `shadow` first to keep serving CLIP while returning DINO candidate
IDs and latency under `retrieval_upgrade`; use `active` only after reviewing a
fresh query set. The adapter is connected to `/search/appearance/image` and
`/recommend_with_image`.

```dotenv
# Set this in .env, then restart the API.
DINO_FUSION_MODE=shadow
DINO_FUSION_TEXT_WEIGHT=0.20
DINO_FUSION_BEHAVIOR_ENABLED=false
DINO_FUSION_BEHAVIOR_WEIGHT=0.25
DINO_FUSION_LOCAL_FILES_ONLY=true
```

The appearance endpoint still discards behavior/lifestyle language.  The
behavior head is disabled by default after it failed the organization-held-out
external gate.  It may be explicitly enabled only to reproduce the isolated
legacy study; missing notice evidence stays unknown and its output is never
proof of a dog's temperament.  The `public-text-only-v1` release profile never
loads the DINO visual index, even if the rollout variable is set.  The final
project choice and all follow-up evidence are summarized in
[`FINAL_DECISION.md`](FINAL_DECISION.md).
