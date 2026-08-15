# External compatibility pilot results

## Decision

Keep the compatibility branch experimental. DINOv3 is a clear visual upgrade,
but neither the current aligned-text route nor the trained CLIP compatibility
head passed the preregistered retrieval gate on organization-held-out dogs.

The result does support a better architecture: keep DINO for appearance and
route adoption-profile conditions through a full-document text branch. A
simple sparse full-text diagnostic found substantially more compatibility
signal than the 77-token CLIP representation, although its safe retrieval gain
was still below the production threshold.

## Frozen systems on 203 test queries

| System | Behavior nDCG@10 | Same-dog Recall@10 |
|---|---:|---:|
| CLIP image only | 0.498 | 0.557 |
| CLIP image + Korean text | 0.506 | 0.581 |
| DINOv3 image only | **0.533** | 0.877 |
| Current DINOde-flow route | 0.516 | **0.882** |

The current behavior parser supports none of the four common compatibility
axes, so its structured query coverage is 0%. Passing the Korean condition into
the appearance alignment alone reduced behavior nDCG.

## Trained CLIP head

Only a 512→128→4 head was optimized; CLIP and DINOv3 stayed frozen. The raw
description checkpoint reached macro ROC-AUC 0.725 on all organization-held-out
text records, but only 0.553 on the 77-dog multimodal test slice. The
evidence-window control reached 0.548 on that slice. Its validation-selected
fusion produced nDCG 0.532 and Recall@10 0.877, so it did not beat DINO-only.

The gap matters: aggregate text classification was not a reliable proxy for
the exact multimodal population used by search.

## Full-document diagnostic

A word unigram/bigram TF-IDF logistic model was used only to test whether the
source descriptions contain recoverable information. On the multimodal test
slice its per-axis ROC-AUC was:

| Axis | CLIP raw head | Full-document diagnostic |
|---|---:|---:|
| Children compatible | 0.645 | 0.731 |
| Dogs compatible | 0.516 | 0.819 |
| Cats compatible | 0.420 | 0.733 |
| House trained | 0.631 | 0.677 |
| Macro | 0.553 | **0.740** |

At the validation-selected safe weight of 0.10, full-document fusion improved
test nDCG from 0.533 to 0.552 (+0.019) and preserved Recall@10 (0.877→0.882).
This is promising but below the predeclared +0.030 nDCG promotion threshold.

## Measured time on RTX 4090

- Model initialization: 4.28 seconds.
- CLIP encoding of 49,022 raw descriptions: 33.54 seconds.
- Head optimization: 21.91 seconds.
- Predicted fresh end-to-end time from the throughput probe: 59.73 seconds.
- Actual fresh end-to-end time: 62.47 seconds (4.6% above prediction).
- Measured cached head retrain: 27.96 seconds.

An average four-window CLIP experiment can be estimated directly from measured
throughput: about 134 seconds for encoding plus 30–60 seconds for pooling/head
training, or roughly 3–4 minutes total on the same GPU. Encoder fine-tuning is
not justified until that cheaper experiment and a larger multimodal holdout
pass.

## Limits

- Structured fields are shelter-entered evidence, not direct behavioral tests.
- The joined test set has 77 dogs; only five cat-compatible positives are
  present, so that axis is high variance.
- Photos and descriptions were collected at different dates.
- The image archive has no stated license and remains local and ignored.
