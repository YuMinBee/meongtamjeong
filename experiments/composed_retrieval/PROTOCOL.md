# CLIP–DINO composed retrieval: prospective protocol v1

Frozen design date: 2026-09-16. No benchmark scores have been inspected at writing.

Research question: does aligning frozen CLIP text features with frozen DINOv3
image features improve image-plus-text retrieval, and does retaining the original
CLIP compositional score protect language information lost during alignment?

## Data and separation

- COCO 2017 training image/caption pairs: deterministic 5,000 training images
  and 1,000 validation images, seed 20260916. All captions of an image share its
  split. Exclude Visual Genome-linked COCO image IDs before sampling. Validation
  caption-to-image loss selects training checkpoints, never benchmark labels.
- CIRCO official validation: 220 annotated queries against the COMPLETE official
  123,403-image unlabeled2017 gallery. Queries linked by ANY reference or annotated
  positive image form connected components, assigned to development or held-out
  audit by SHA-256 parity of the component's smallest image ID. Development selects fusion
  weights and the candidate method; the audit subset is evaluated afterwards.
  This split is a LOCAL protocol, not the official hidden-test benchmark. Report
  every count/hash and never label local validation scores as official test.
- GeneCIS v0: all four released tasks, with provided reference/target/distractor
  sets and published crop/pad convention. External evaluation only, after CIRCO
  development choices are frozen. Report each task separately. Missing source
  images must be recovered where possible; otherwise fail closed for complete
  evaluation and explicitly label any diagnostic subset/coverage.
- CIRR: adapter may be supplied, but no benchmark claim without raw images;
  official NLVR2 access requires a form. Do not obtain gated copies by bypassing
  that process or send email/form submissions on the user's behalf.

## Candidates, fixed before scoring

Frozen encoders: OpenAI CLIP ViT-B/32 and DINOv3 ViT-B/16 (local pinned revision).
Both use their native model preprocessing of the SAME source/crop. No dog-only
detector is applied to general objects. Cache normalized embeddings with content
hashes, source identity, crop geometry, encoder revision and preprocessing.

1. CLIP image only, CLIP text only, DINO image only.
2. CLIP image/text weighted composition.
3. DINO-image + CLIP-text-image late score fusion (no learned alignment).
4. CLIP-text → DINO linear, MLP, and existing 8-step flow alignment; same COCO
   captions, frozen features, loss, epochs and data for each head.
5. Proposed diagnostic: residual score fusion of aligned DINO composition and
   original CLIP composition. This is an engineering hypothesis, not a claim
   of algorithmic novelty. Include an unaligned DINO+CLIP-composition control.

Train heads at seeds 7, 42, 123; at most 30 epochs, AdamW lr 0.001, weight decay
0.0001, batch 256, contrastive temperature 0.07. Duplicate target images in a
batch are multi-positive, never false negatives. Select checkpoint by held-out
COCO validation loss. Linear/MLP/flow retain the existing project head defaults.

Fusion text weights: 0, 0.1, 0.2, 0.4, 0.6, 0.8, 1. Residual/visual weights:
0, 0.25, 0.5, 0.75, 1. Select on CIRCO development mean mAP@5/10/25/50;
ties prefer the simpler method/lower weight. Select each family's hyperparameters
using its mean development score across seeds, then freeze for all evaluations.
Also retain the project's fixed 0.8 image / 0.2 text setting as a fixed baseline.

## Metrics and reporting

CIRCO: mAP@5/10/25/50, denominator min(K, number of annotated positives), exclude
reference image. GeneCIS: Recall@1/2/3; rank the positive among the exact supplied
gallery (target plus distractors). Deterministic stable tie handling by image ID,
not by positive-first ordering. For GeneCIS use stable SHA-256 candidate-slot IDs,
preserve duplicate official candidate slots, and report their counts/label noise.
Report chance levels and image/text-only controls.

Report per-query scores, three seed results, paired bootstrap 95% confidence
intervals grouped by reference image (2,000 resamples), and bootstrap intervals
for differences (CIRCO groups use the linked-image components). Seeds are not
additional independent query samples. Track source
download coverage, gallery completeness, exact and decoded-image hashes, overlap
audit, model/data versions, trainable parameter count, training/encoding timing,
and actual hardware. Pretraining overlap cannot be excluded for web-trained
foundation models. COCO alignment is supervised paired-data training, NOT a
training-free method. Existing dog-trained flow may be reported separately as
out-of-domain transfer, not confused with COCO-trained heads.

Do not revise hypotheses/hyperparameters based on audit/GeneCIS results. Findings
that motivate further methods belong to a subsequent protocol/new holdout.

Data-only amendment before any model scores (2026-09-16): the original
reference-only partition had 5 labeled images shared across development/audit
(4 shared positives). Component grouping replaces it to remove this overlap.
The shared full retrieval gallery is retained, as required by the benchmark.

## Primary sources

- https://github.com/miccunifi/CIRCO
- https://github.com/facebookresearch/genecis
- https://github.com/facebookresearch/genecis/blob/main/datasets/vaw_dataset.py
- https://cocodataset.org/#download
- https://github.com/Cuberick-Orion/CIRR

Raw data, downloaded annotations and model outputs are local ignored artifacts.
Code/tests/protocol and aggregate research reports may be versioned. Source
licenses apply separately; no raw image redistribution is part of this experiment.
