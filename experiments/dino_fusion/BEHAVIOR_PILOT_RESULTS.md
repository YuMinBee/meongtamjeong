# Unknown-aware behavior reranking pilot

Date: 2026-08-14

This isolated pilot tests whether public-notice behavior evidence can supply the
personality information that static DINO/CLIP image features do not contain.
It does not change the production search route.

## Protocol

- Candidate corpus: the same 1,002 local shelter-dog crops used by the DINOv3
  pilot.
- Behavior source: provenance-bearing public/shelter notice text only. VLM
  descriptions are excluded.
- Missing phrases: `unknown`, never negative.
- Conflicting positive and negative phrases: `unknown`/abstain.
- Supervised axes with enough explicit evidence at both ends:
  - gentle/soft handling vs defensive/bite caution;
  - people-friendly vs wary of unfamiliar people;
  - active/high-energy vs calm/low-arousal.
- Frozen encoders: OpenAI CLIP ViT-B/32 and DINOv3 ViT-B/16.
- Trainable component: a 66,051-parameter `512 -> 128 -> 3` behavior head over
  frozen CLIP embeddings of public notice text.
- Objective: class-balanced masked binary cross-entropy over explicit labels
  only. Exact duplicate notice texts stay in the same split.

Run:

```powershell
C:\Users\sally\anaconda3\envs\dog-rag\python.exe `
  -m experiments.dino_fusion.train_behavior `
  --local-files-only
```

Generated checkpoints and the JSON report remain under the ignored
`artifacts/behavior_pilot/` directory.

## Corpus and runtime

| Item | Value |
|---|---:|
| Candidate notices | 1,002 |
| Non-empty public behavior texts | 997 |
| Observed-label train / validation / test rows | 293 / 51 / 53 |
| Head parameters | 66,051 |
| Best epoch | 208 |
| Optimization time on RTX 4090 | 1.40 s |
| End-to-end run including encoders and demos | 9.02 s |

Explicit label coverage was uneven. Gentle handling had 298 positive and 34
negative records; people sociality had 57 positive and 19 negative records;
activity had 38 positive and 43 negative records. Fear/watchfulness was kept as
evidence-only because the corpus contained no reliable opposite examples.

## Retrieval result

The table reports macro nDCG@10 across six controlled Korean behavior queries
on the held-out known-label candidates.

| System | nDCG@10 |
|---|---:|
| CLIP text -> crop image | 0.6652 |
| DINO-flow-aligned CLIP text -> DINO crop | 0.6168 |
| CLIP text -> public notice text | 0.6583 |
| Trained behavior head | 0.7955 |
| DINO-flow image space + behavior head | **0.8281** |

The intended runtime ordering contract is:

```text
explicit public-note match > unknown/inferred > explicit contradiction
```

This prevents missing text from being treated as a negative label. For mixed
image-and-personality demos, DINO supplies 75% of the rank-percentile score and
the evidence-gated behavior branch supplies 25%; this weight was fixed before
examining the demo rankings.

## Interpretation

The result supports the narrower claim that a separate behavior-text branch can
move rankings using information unavailable to static image encoders. It does
not show that the model has learned true dog personality:

- the labels and the behavior inputs both come from incomplete public notices;
- the test set is especially small for people-social and activity examples;
- the six queries are controlled templates rather than independent user
  language;
- public descriptions can be subjective and context-dependent.

The next valid step is a blinded manual review of 150-200 notices, followed by
evaluation on those human labels without changing the rules, split, or fusion
weight.
