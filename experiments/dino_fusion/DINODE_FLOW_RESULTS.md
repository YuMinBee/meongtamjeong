# DINOde-style flow applied to dog retrieval

Date: 2026-08-14

## What was applied

This is not the VOC segmentation reproduction. It is a retrieval-oriented,
independent implementation of the DINOde alignment idea inside this project's
isolated dog-search experiment:

- frozen CLIP ViT-B/32 text encoder (512D);
- frozen DINOv3 ViT-B/16 visual encoder and existing 1,002-crop index (768D);
- trainable 512D-to-768D initial projection;
- an eight-step time-conditioned ODE velocity field projected onto the unit
  hypersphere's tangent space;
- Euler updates with unit-sphere retraction after every step.

The flow uses 621,984 trainable parameters, versus 656,640 for the existing
MLP. This keeps the comparison close in capacity. No CLIP or DINO parameter was
updated, and all 120 notices in the previous held-out pool remained excluded
from training.

## Training and validation

All heads used the same 720 training notices, 145 validation notices, prompts,
loss, optimizer, and seed. The complete job took 13.1 seconds on the RTX 4090;
flow-head optimization took 4.0 seconds.

| Head | Parameters | Best epoch | Validation loss |
|---|---:|---:|---:|
| Linear | 393,216 | 35 | 2.911558 |
| MLP | 656,640 | 30 | **2.729726** |
| ODE flow | 621,984 | 80 | 2.795228 |

The MLP remains the architecture selected by validation loss. Flow results
below are therefore a post-selection pilot signal, not a clean claim that flow
is universally superior.

## Held-out natural-language retrieval

Twenty payload-verified, untouched notices were evaluated with sentence forms
absent from training. Attribute nDCG@10 measures color and size agreement; it
is more appropriate than exact-notice rank for generic descriptions.

| Text system | English P@10 | English nDCG@10 | Korean P@10 | Korean nDCG@10 |
|---|---:|---:|---:|---:|
| Raw CLIP | 0.255 | 0.2854 | 0.110 | 0.1225 |
| Linear to DINOv3 | 0.475 | 0.4746 | 0.635 | 0.6780 |
| MLP to DINOv3 | 0.390 | 0.4019 | 0.600 | 0.6038 |
| **ODE flow to DINOv3** | **0.490** | **0.5111** | **0.680** | **0.7221** |

Flow improved nDCG@10 over the MLP by +0.1092 in English and +0.1183 in
Korean. It also improved over raw CLIP by +0.2257 and +0.5996 respectively.

For exact same-dog image retrieval, 80% DINO image + 20% Korean flow text gave
Hit@1/5/10 = 95%/100%/100%, MRR 0.9600, and maximum rank 5. The prior MLP was
slightly better on exact identity (MRR 0.9625, maximum rank 4), so flow's clear
strength in this pilot is semantic text retrieval rather than image identity.

### Direct multimodal comparison

The current CLIP-only multimodal baseline was rerun on the exact same 20
queries, same crop corpus, and same 80% image / 20% text weight as the aligned
DINO systems.

| Image + text system | Exact Hit@1 | Exact Hit@5 | Exact Hit@10 | Exact MRR | EN attribute nDCG@10 | KO attribute nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| CLIP image + CLIP text | 85% | 95% | 95% | 0.8932 | 0.4205 | 0.4058 |
| DINO image + MLP-aligned CLIP text | **95%** | **100%** | **100%** | **0.9625** | 0.5147 | 0.5136 |
| DINO image + flow-aligned CLIP text | **95%** | **100%** | **100%** | 0.9600 | **0.5316** | **0.5375** |

This is the comparison matching the project's real image-plus-language input.
The useful fusion is a light same-space blend after alignment. Earlier naive
equal-weight rank fusion of independent CLIP and DINO rankings damaged identity
retrieval and should not be used.

## Learning controls

Five random untrained flow heads and five identically trained heads with
shuffled text-image associations were evaluated on the same held-out queries.

| Flow condition | English nDCG@10 | Korean nDCG@10 |
|---|---:|---:|
| Correctly trained | **0.5111** | **0.7221** |
| Random, five-run mean | 0.1954 | 0.1512 |
| Shuffled, five-run mean | 0.1140 | 0.1719 |

Correct training beat the best random and shuffled runs in both languages.
The gain is therefore not explained by parameter count or converting 512D to
768D alone.

## Usable project path

The trained flow now directly searches the project's dog corpus and can also
blend a DINO image query with aligned CLIP text:

```powershell
python -m experiments.dino_fusion.aligned_search `
  --text "검정색과 흰색 털이 섞인 작은 강아지를 찾아줘" `
  --topk 10 `
  --local-files-only

python -m experiments.dino_fusion.aligned_search `
  --text "small black and white dog" `
  --image assets/samples/notice_428349202600501.jpg `
  --text-weight 0.2 `
  --topk 10 `
  --local-files-only
```

An end-to-end Korean smoke query returned black-and-white dogs at ranks 1 and
3 and attached their public breed, color, age, weight, sex, status, and image
metadata to the results.

## Decision

Keep the production CLIP route unchanged, but retain ODE flow as the leading
experimental **text-to-DINO retrieval** method. Promotion now requires a new
query set written independently by people with human relevance judgments,
because choosing flow from these 20 held-out outcomes would reuse the test set
for model selection.
