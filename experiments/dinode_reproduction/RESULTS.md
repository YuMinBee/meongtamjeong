# DINOde VOC20 reproduction result

## Outcome

The official released checkpoint was reproduced exactly on the full PASCAL
VOC 2012 validation protocol:

| Setting | Official mIoU | Local mIoU | Absolute delta |
|---|---:|---:|---:|
| VOC20, background excluded, `--no_cache` | 91.1495% | 91.1495% | 0.0000 |

- Validation samples: 1,449 (91 batches)
- Evaluation loop: approximately 97 seconds
- Python pipeline including model initialization: approximately 103 seconds
- GPU: NVIDIA GeForce RTX 4090 24 GB
- Official checkpoint SHA256:
  `494354fe148a278a99c8186a59d9fa5ef14cf6254e312195eb08cac69cac1377`
- VOC archive MD5: `6cd6e144f989b92b3379bac3b3de84fd`
- Upstream commit: `0a2c5182c44107fdd8c786b2192fcc3c5e5bebd1`

The ignored raw log is under
`vendor/DINOde/outputs/0814_161611_eval_voc20_no_cache/eval_log.txt`.

## Environment and platform note

The official package versions were used: Python 3.10, PyTorch 2.6.0+cu118,
torchvision 0.21.0+cu118, and transformers 4.56.2. DINOv3 ViT-L/16 and
OpenCLIP ViT-L/14 (`laion2b_s32b_b82k`) were loaded as the frozen encoders.

One Windows-only data-loading adjustment was necessary. The upstream
`collate_val` function is local to `main()` and therefore cannot be pickled by
Windows spawn workers. The harness changes `data.num_workers` from 4 to 0 in a
generated runtime config. Model operations, prompts, checkpoint, sliding
window, seed, dataset, and metrics remain unchanged. The exact numerical match
confirms that this throughput-only adjustment did not alter the result.

## What this does and does not reproduce

This is a faithful **evaluation reproduction** of the authors' released
checkpoint and full VOC20 benchmark. It validates the repository, frozen
DINOv3/CLIP encoders, learned DINOde head, data processing, and reported metric
on this protocol.

It is not yet a from-scratch training reproduction. Official training uses
COCO-Stuff captions and, with feature caching, needs roughly 270 GB. Training
should be treated as a separate stage only if we need to audit convergence or
modify the method itself.

## Relevance to the dog-retrieval project

The official method is denser and more structured than our existing pilot:
it aligns CLIP text with DINOv3 spatial tokens through a learned
text-conditioned ODE head for open-vocabulary segmentation. Our current dog
pilot instead learns a small global CLIP-text-to-DINO adapter and evaluates
retrieval.

The most useful next transfer experiment is therefore not to drop the
segmentation head into production unchanged. In the isolated dog experiment,
compare three matched settings on the same held-out queries:

1. Current global MLP alignment.
2. A DINOde-style text-conditioned flow on the global/crop representation.
3. The same flow with patch-aware region scoring from DINOv3 spatial tokens.

Keep DINOv3 and CLIP frozen in all three, match trainable-parameter budgets,
and report both identity retrieval (Hit@K/MRR) and attribute agreement
(nDCG@10). This separates whether DINOde's flow contributes beyond the simple
MLP and whether its dense-token advantage transfers from segmentation to
fine-grained dog matching.
