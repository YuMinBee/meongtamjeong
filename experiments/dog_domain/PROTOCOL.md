# Dog-domain external evaluation, prospective v1

2026-09-16. Written before computing MPDD model scores. PetFace is pending access.
This is an application-domain validation of the project's frozen visual encoders,
not a new architecture claim and not a test of text alignment/Flow.

## MPDD

Source: Zhimin He (2023), Multi-pose dog dataset, version 1,
https://data.mendeley.com/datasets/v5j6m8dzhv/1, CC BY 4.0.
Publisher SHA256: 6c800c1b4aa67629544dec7444dee85bc57781abde1ae1d077ea9ef1804284cd.
Read the original archive directly. Preserve distributed train/val/query/gallery
folders. Use val queries against train gallery ONLY for weight selection;
use query against gallery ONLY for held-out evaluation. No encoder training.
Archive inspection found 1,657 photos and 191 distinct filename identity tokens
(95 train/val, 96 query/gallery), whereas the landing page describes 192 dogs.
Report actual coverage and this discrepancy without inventing a missing ID.

Models: frozen CLIP ViT-B/32 and DINOv3 ViT-B/16, native preprocessing of the
same RGB source; no additional detector. Record revisions and hashes.
Systems: CLIP cosine, DINO cosine, 50/50 score fusion, validation-selected
score fusion (DINO weight 0, .25, .5, .75, 1). Choose highest identity-macro
mAP on clean validation, ties prefer endpoints then lower weight. Freeze
selection JSON before any test scores. A selected endpoint is not evidence
for combining both encoders.

Primary task: same-identity retrieval in the distributed query/gallery split,
excluding byte/decoded identical source images; retain different photos with
the same identity. Stable score-descending/member-name-ascending ranking.
Secondary sensitivity: additionally exclude same-ID gallery images with the
same filename c-code; do not assert c-code is a real camera without documentation.
Report queries with no eligible positive instead of treating them as failures.
No official-paper reproduction claim for either metric variant.

Robustness: clean gallery, query-only Gaussian blur radius 2 pixels, and
query-only resize to longest side 64 then back to original size (bilinear).
These synthetic perturbations are not real occlusion/pose annotations. Apply
the clean-validation choice unchanged. No post-hoc perturbation selection.

Report Recall@1/5/10, full mAP and MRR, macro-average over dog IDs first;
also retain query-micro means and per-query metrics. Paired bootstrap over dog
IDs, 2,000 resamples, 95% intervals, same resamples for method differences.
No random-training seeds: encoders frozen; bootstrap seed 20260916 is not a
training seed. Runtime and image decoding/duplicate coverage are recorded.

## Existing image-plus-text evidence

The project's 20-query dog-notice pilot and 77-dog PetFinder test have already
been inspected. Preserve their outputs and describe them as previous pilot
evidence, not new independent test sets. Record what identity and attribute
labels actually establish. Do not generate captions and call them new human
ground truth. Do not tune Flow/text weights against those already-seen tests.
PetFace or a fresh identity/organization-separated labeled corpus is needed
for a new independent multimodal claim; PetFace itself is not mandatory.
