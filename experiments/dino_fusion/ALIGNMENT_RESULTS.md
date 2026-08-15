# CLIP-text-to-DINO shared-space pilot

Date: 2026-08-14

## Outcome

The shared-space idea is supported as a research direction, but DINOv3 is not
uniformly better than DINOv2 on this small dataset. Frozen DINOv2 remained
slightly stronger for exact same-dog image retrieval. DINOv3 produced the
stronger trained Korean text alignment, and a light same-space text blend
improved DINOv3's difficult image ranks without harming Hit@1.

This is a 20-query pilot with metadata-derived prompts and silver attribute
labels. It is evidence to expand the evaluation, not a production superiority
claim.

## Controlled setup

- Visual backbones: frozen `facebook/dinov2-base` and frozen
  `facebook/dinov3-vitb16-pretrain-lvd1689m`, both 768 dimensions.
- Text encoder: frozen OpenAI CLIP ViT-B/32, 512 dimensions.
- Trainable heads: Linear, 393,216 parameters; MLP, 656,640 parameters.
- Corpus: the same 1,002 local Faster R-CNN dog crops for every DINO run.
- Split: all 120 notices in the previous held-out sample pool excluded;
  720 training notices and 145 validation notices.
- Training language: three English and three Korean paraphrases per semantic
  signature. Validation and held-out sentence forms were absent from training.
- Loss: symmetric multi-positive InfoNCE plus cosine alignment to the DINO
  centroid for public color, weight-derived size, and age group.
- Held-out: 20 payload-verified secondary photos with a DINO crop and sufficient
  public attributes. Query images were decoded only in memory.
- Mixed query: a fixed, untuned 80% DINO image / 20% aligned text blend.

Both full training jobs, including CLIP prompt encoding, took about 9 seconds
on the RTX 4090. Head optimization itself was under one second per backbone.
The first DINOv3 download and 1,002-vector index build took about 52 seconds.

## Frozen image retrieval: V2 versus V3

| Backbone | Hit@1 | Hit@5 | Hit@10 | MRR | Max rank | Attribute nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DINOv2 Base | 95% | **100%** | 100% | **0.9750** | **2** | **0.5026** |
| DINOv3 ViT-B/16 | 95% | 95% | 100% | 0.9563 | 8 | 0.4825 |

DINOv3 therefore did not beat DINOv2 for pure instance matching here. With only
20 queries, the difference is one difficult case and should not be generalized.
Against CLIP on the broader 21-query image-only set, however, DINOv3 was clearly
stronger on the same crop corpus: Hit@1 rose from 80.95% to 90.48%, Hit@10 from
90.48% to 100%, and MRR from 0.8594 to 0.9266. DINOv3 ranked the target higher
on four queries, tied on 16, and ranked it lower on one.

## Natural-language retrieval

Exact-notice rank is not a suitable primary metric for generic text such as
"find a small white adult dog": many notices are valid. The table uses coarse
color+size relevance. The MLP was selected before the held-out run because it
had the lower validation loss; Linear results are included as an ablation.

| Visual space / text head | Language | P@10 | nDCG@10 |
| --- | --- | ---: | ---: |
| Raw CLIP image space | English | 0.255 | 0.2854 |
| DINOv2 + Linear | English | 0.430 | 0.4436 |
| DINOv2 + MLP | English | 0.410 | 0.4204 |
| DINOv3 + Linear | English | **0.475** | **0.4746** |
| DINOv3 + MLP | English | 0.390 | 0.4019 |
| Raw CLIP image space | Korean | 0.110 | 0.1225 |
| DINOv2 + Linear | Korean | 0.570 | 0.5522 |
| DINOv2 + MLP | Korean | 0.445 | 0.4671 |
| DINOv3 + Linear | Korean | **0.635** | **0.6780** |
| DINOv3 + MLP | Korean | 0.600 | 0.6038 |

The main gain is not that DINO understands language by itself. The learned head
turns frozen CLIP text into a vector that can directly query the DINO index.
The large Korean gain also needs caution: training and evaluation use different
sentences but share a small controlled vocabulary.

## Does learning matter, or did we only match 512D to 768D?

Two negative controls were repeated with five random seeds using the
validation-selected MLP:

- **Random:** the identical 512D-to-768D MLP with no optimizer update.
- **Shuffled:** the identical model, loss, optimizer, and 30-epoch update count,
  but with the DINO image rows randomly permuted before training.

| System | English nDCG@10 | Korean nDCG@10 |
| --- | ---: | ---: |
| Trained with correct associations | **0.4019** | **0.6038** |
| Random, five-seed mean ± sample SD | 0.1602 ± 0.0681 | 0.1549 ± 0.0569 |
| Shuffled, five-seed mean ± sample SD | 0.1314 ± 0.0459 | 0.1343 ± 0.0451 |

The trained result exceeded the best of all five random seeds and all five
shuffled seeds in both languages. Its absolute nDCG advantage was +0.242 over
the random mean and +0.270 over the shuffled mean in English, and +0.449 and
+0.469 respectively in Korean. This rules out simple output-dimension matching
as the explanation for the observed pilot gain: correct text-to-image
associations are necessary. It still does not establish broad free-form
language generalization beyond the controlled vocabulary.

## Same-space image + text

For the validation-selected DINOv3 MLP, adding 20% Korean text changed exact
MRR from 0.9563 to 0.9625, Hit@5 from 95% to 100%, maximum rank from 8 to 4,
and attribute nDCG@10 from 0.4825 to 0.5136. Per-query exact ranks improved once,
tied 19 times, and never worsened. English produced a similar small improvement.

This is the clearest advantage over the earlier equal-weight rank fusion: image
and text are comparable vectors, so text can gently steer the image query rather
than competing as a separate ranking that overwhelms identity.

## Decision

Continue the shared-space branch as an experiment with DINOv3, using the MLP as
the preselected primary head and Linear as an ablation. Do not replace the
production CLIP pipeline yet. The next gate is a larger set of independently
written Korean/English queries, human relevance labels, and blend-weight
selection on validation data rather than these held-out queries.

## Follow-up: DINOde-style flow application

A capacity-matched, time-conditioned hyperspherical ODE flow was subsequently
trained on the same split and connected to a runnable corpus search path. It
improved text-only attribute nDCG@10 to 0.5111 in English and 0.7221 in Korean,
above the MLP's 0.4019 and 0.6038. Five-run random and shuffled controls were
far lower. See `DINODE_FLOW_RESULTS.md` for the architecture, controls, actual
search command, and the test-reuse caveat.
