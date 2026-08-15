# DINO retrieval performance ablations

Date: 2026-08-15 (Asia/Seoul)

## Decision

Keep the current DINOv3 crop-gallery configuration and the existing 0.20 text
weight in `shadow`.  The extra performance round found useful trade-offs, but
no change passed the predeclared test gate.

- DINOv2 recovered more same-dog targets by rank 10, but DINOv3 placed the
  correct dog materially higher on average and retained the existing text-flow
  integration.
- Validation selected the crop gallery for DINOv3.  Multiview looked better on
  several test ranking metrics, but it was not the validation-selected choice
  and therefore is not promoted from this run.
- Validation selected a 0.25 Korean appearance-text weight, but its test
  appearance nDCG gain was only `+0.0087`, below the fixed `+0.02` gate.  The
  existing 0.20 setting was effectively tied.

No foundation encoder was fine-tuned.

## Protocol

- Dataset: the existing PetFinder multi-photo external benchmark.
- Split: complete organizations remain disjoint.
- Validation: 38 dogs; used to select gallery view and text weight.
- Test: 77 dogs; used with frozen validation choices.
- Query: uncropped photo 1.
- Gallery: photo 2 as a full image, Faster R-CNN dog crop, or the maximum score
  over both views.
- Identity target: the same dog across two different photos.
- Appearance relevance: exact agreement on known PetFinder color, size, age,
  and coat fields.  The reference dog is removed when computing appearance
  nDCG so the metric measures similar alternatives rather than rewarding the
  identity target twice.
- Korean appearance prompts are sent through the existing frozen CLIP text
  encoder and DINOv3 flow head.

The detector found a dog in 113/115 gallery images (`98.26%`).  Missing crops
remain unavailable in crop-only evaluation and fall back to the full view only
in the explicit multiview system.

## Frozen visual systems

Validation selected `multiview` for DINOv2 and `crop` for DINOv3.  Their frozen
test results were:

| Validation-selected system | Recall@1 | Recall@5 | Recall@10 | MRR | Appearance nDCG@10 |
|---|---:|---:|---:|---:|---:|
| DINOv2 Base + multiview | 0.6104 | 0.8442 | **0.9481** | 0.7176 | 0.4680 |
| DINOv3 ViT-B/16 + crop | **0.6883** | 0.8442 | 0.9091 | **0.7650** | **0.4701** |

DINOv2's Recall@10 advantage was `+0.0390`, while its MRR was `-0.0474`
below DINOv3 and its appearance nDCG was `-0.0022` lower.  The evidence does
not justify replacing the multimodal DINOv3 route.  It does show that DINOv2
and DINOv3 make different late-rank errors, which could motivate a future
fusion experiment only with a new selection set.

For transparency, the DINOv3 view ablation was:

| Split | Gallery | Recall@1 | Recall@10 | MRR | Appearance nDCG@10 |
|---|---|---:|---:|---:|---:|
| Validation | crop | 0.6053 | **1.0000** | 0.7295 | **0.5290** |
| Validation | full | **0.6579** | 0.9737 | 0.7376 | 0.5083 |
| Validation | multiview | **0.6579** | 0.9737 | **0.7569** | 0.5267 |
| Test | crop | 0.6883 | 0.9091 | 0.7650 | **0.4701** |
| Test | full | **0.7143** | 0.8831 | 0.7749 | **0.4701** |
| Test | multiview | 0.7013 | **0.9221** | **0.7783** | 0.4693 |

The test-only multiview improvement is not used to revise the selection rule.

## Text-weight selection

The candidate weights were fixed at `0.00`, `0.05`, `0.10`, `0.15`, `0.20`,
`0.25`, and `0.30`.  A candidate had to preserve validation Recall@10 and MRR
within `0.02`; eligible candidates were ordered by appearance nDCG, then MRR,
then lower weight.  This selected `0.25` on the DINOv3 crop gallery.

| Test system on 51 appearance queries | Recall@1 | Recall@10 | MRR | Appearance nDCG@10 |
|---|---:|---:|---:|---:|
| DINOv3 crop, image only | **0.7059** | 0.9020 | **0.7895** | 0.4701 |
| DINOv3 crop, selected weight 0.25 | 0.6863 | **0.9216** | 0.7729 | **0.4788** |
| DINOv3 crop, current weight 0.20 | 0.6863 | **0.9216** | 0.7760 | 0.4772 |

The selected weight preserved the safety metrics but improved appearance nDCG
by only `+0.0087`, so the `+0.02` minimum-gain check failed.  The difference
between 0.25 and the current 0.20 was just `+0.0016` nDCG with slightly lower
MRR.  Changing the runtime weight is not supported.

## Timing

RTX 4090, local model cache:

- detector initialization and 115 crops: 5.57 seconds;
- DINOv2 initialization plus 345 image encodes: 3.79 seconds;
- DINOv3 initialization plus 345 image encodes: 0.97 seconds;
- CLIP text plus existing flow for 115 prompts: 1.98 seconds;
- fresh end-to-end run: 13.39 seconds;
- cached reproducibility run: 4.75 seconds.

The raw image archive and generated embeddings remain ignored and local because
the archive's dataset card does not state a redistribution license.

## Reproduce

```powershell
conda run -n dog-rag python `
  -m experiments.dino_fusion_external_eval.evaluate_performance_ablations
```

Use `--force` to rebuild detector crops and all image embeddings.  The machine
readable report is written to ignored
`artifacts/performance_ablation_report.json`.
