# CLIP + DINO training-free pilot

Date: 2026-08-14

## Outcome

The training-free visual branch is promising enough to continue. On the exact
same 1,002 Faster R-CNN crop candidates, frozen DINOv2 Base improved over the
current frozen OpenAI CLIP ViT-B/32 image tower on the reproducible held-out
subset. Equal-weight rank fusion matched DINO's Hit@K but did not improve its
MRR, so the pilot supports a DINO visual branch more strongly than naive 1:1
fusion.

This is an infrastructure and hypothesis pilot using public DINOv2 Base because
DINOv3 access was gated and the local environment was not authenticated. It is
not a DINOv3 or DINOde result.

Historical note: DINOv3 access was subsequently approved on 2026-08-14. The
frozen DINOv3 and trained shared-space follow-up is reported in
`ALIGNMENT_RESULTS.md`.

## Setup

- DINO: `facebook/dinov2-base`, frozen, revision
  `f9e44c814b77203eaa57a6bdbbd535f21ede1415`, CLS/global pooler, 768 dimensions.
- CLIP: current OpenAI CLIP ViT-B/32, frozen, 512 dimensions.
- DINO corpus: 1,002 existing local Faster R-CNN dog crops; no build failures.
- Queries: 21 distinct secondary photos whose URL and payload hashes still
  matched the previous held-out execution and whose notice had a DINO crop.
- Fusion: equal-weight reciprocal-rank fusion with `k=60`; no tuning.
- Query images were decoded in memory and were not saved by the pilot.

## Results

| System | Candidate corpus | Hit@1 | Hit@5 | Hit@10 | MRR | Maximum rank |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CLIP full+crop | 1,228 full + 1,002 crop vectors | 76.19% | 95.24% | 95.24% | 0.8497 | 94 |
| CLIP crop-only | Same 1,002 crops as DINO | 80.95% | 90.48% | 90.48% | 0.8594 | 62 |
| DINO crop-only | 1,002 crops | **90.48%** | **95.24%** | **100.00%** | **0.9345** | **8** |
| CLIP + DINO RRF | Same 1,002 crops | **90.48%** | **95.24%** | **100.00%** | 0.9202 | **8** |

Against same-corpus CLIP crop-only, DINO produced a better target rank on 4 of
21 queries, tied on 17, and was worse on 0. The sample is too small for a strong
statistical claim. One difficult query improved from CLIP rank 62 to DINO rank
8. Equal-weight RRF tied DINO on most queries but degraded one target from rank
2 to rank 5, explaining its lower MRR.

## Interpretation

The result supports the original motivation: detector crops and DINO are not
redundant. Faster R-CNN isolates the dog, while DINO appears to provide a useful
within-dog visual representation. It does not yet show that CLIP and DINO should
always be blended. A cleaner product hypothesis is now:

- use DINO as the primary image-to-image retrieval branch;
- keep CLIP for text-to-image semantics;
- fuse only for mixed image+text requests, after validating weights on a
  separate validation set.

## Mixed image + natural-language pilot

The same 21 held-out photos were paired with conservative appearance prompts
derived only from public color and weight-based size. Candidate notice text
vectors were excluded: CLIP text embeddings were compared directly with the
same 1,002 crop images. Both Korean and English prompts were tested.

Exact-notice retrieval and coarse attribute relevance answer different
questions. Exact-notice rank measures whether adding generic language preserves
the identity signal from the reference photo. Attribute relevance counts any
candidate with a matching public color and size as relevant.

| System | Exact Hit@1 | Exact MRR | Attribute P@5 | Attribute P@10 | Attribute nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| DINO image | **90.48%** | **0.9345** | **0.520** | 0.410 | **0.5026** |
| CLIP English text | 0.00% | 0.0171 | 0.340 | 0.365 | 0.3740 |
| CLIP Korean text | 0.00% | 0.0039 | 0.170 | 0.105 | 0.1600 |
| DINO image + English text RRF | 19.05% | 0.3249 | 0.490 | **0.490** | 0.4979 |
| DINO image + Korean text RRF | 0.00% | 0.1543 | 0.340 | 0.345 | 0.3406 |

Equal-weight English text fusion increased coarse attribute precision@10 from
0.410 to 0.490, but it overwhelmed the reference-photo identity signal. Korean
text was substantially weaker than English with the current OpenAI CLIP
ViT-B/32 text tower. The result does not support equal-weight fusion.

For the next no-training iteration, supported Korean attributes should be
normalized into English CLIP prompts, and text should be used as a light
reranker or constraint inside a DINO candidate pool rather than as an equal
candidate generator. Weight or reranking-depth selection needs a separate
validation set; selecting them on these 21 held-out queries would overfit.

## Limitations and next gate

- Coverage was 21/29 previously evaluable queries because 8 targets had no
  local crop in the DINO corpus.
- The query set was inherited from a prior CLIP evaluation and is small.
- Only crop descriptors and DINOv2 Base were tested; full-image DINO, DINOv3,
  alternative pooling, independently authored Korean queries, and independent
  weight selection remain untested.
- The mixed prompts were generated from public metadata rather than authored
  independently by users. Their coarse relevance labels reuse those same
  public color and size fields and are silver diagnostics, not human judgments.
- The next meaningful gate after this historical stage was DINOv3 plus an
  alignment head; that follow-up has now been run on the same small held-out
  pool. A larger independently authored evaluation set is still required.
