# Frozen fusion: follow-up validation

Weight fixed from the original MPDD validation: 25% CLIP + 75% DINO. No training or new tuning.
Original MPDD: fusion-only correct = 1, DINO-only correct = 0.
Original exact paired ID sign-test p=1.0000.

DogFaceNet official-test cohort is the primary external test; all-IDs is secondary.
All use one gallery photo per identity and all remaining photos as queries. These are custom retrieval protocols.
MPDD resampling reuses the same identities and is a sensitivity analysis, not new external evidence.

| Cohort | IDs | Images | Gallery draws | CLIP R@1 | DINO R@1 | Fusion R@1 | Fusion−DINO [95% CI], pp |
|---|---:|---:|---|---:|---:|---:|---|
| dogface_official_test | 139 | 697 | hash-fixed | 49.80 | 90.33 | 89.26 | -1.06 [-2.96, +0.43] |
| dogface_official_test | 139 | 697 | 20 repeats | 49.60 | 90.66 | 89.80 | -0.86 [-1.45, -0.33] |
| dogface_all_frozen | 1393 | 8363 | hash-fixed | 34.28 | 76.30 | 75.57 | -0.73 [-1.18, -0.31] |
| dogface_all_frozen | 1393 | 8363 | 20 repeats | 33.64 | 76.60 | 75.79 | -0.81 [-1.00, -0.63] |
| mpdd_gallery_sensitivity | 96 | 625 | hash-fixed | 62.75 | 88.67 | 88.83 | +0.16 [-0.69, +1.02] |
| mpdd_gallery_sensitivity | 96 | 625 | 100 repeats | 60.49 | 89.38 | 89.47 | +0.09 [-0.35, +0.54] |

## Paired R@1 tests

| Cohort / analysis | Winning / tied / losing IDs | Sign-flip p |
|---|---|---:|
| dogface_official_test / primary | 3 / 129 / 7 | 0.28441 |
| dogface_official_test / repeated_gallery_identity_averaged | 15 / 88 / 36 | 0.00085 |
| dogface_all_frozen / primary | 45 / 1248 / 100 | 0.00078 |
| dogface_all_frozen / repeated_gallery_identity_averaged | 255 / 636 / 502 | 0.00001 |
| mpdd_gallery_sensitivity / primary | 4 / 89 / 3 | 0.76600 |
| mpdd_gallery_sensitivity / repeated_gallery_identity_averaged | 31 / 40 / 25 | 0.70354 |

Bootstrap: 2,000 identity resamples after averaging repeats within ID. Repeats are not independent sample-size increases.
Sign-flip: paired identity-level mean differences, 100,000 random sign draws, +1 correction.
Only hash-fixed official-test R@1 is the primary hypothesis; remaining comparisons are exploratory without multiplicity correction.
Exact duplicate checks do not eliminate same-session near duplicates, foundation pretraining overlap, or unknown identity overlap across corpora.
DogFaceNet faces are pre-aligned and 224px; results do not establish full-body lost-dog deployment performance or text/Flow efficacy.
Data source: [Mougeot, DogFaceNet datasets](https://zenodo.org/records/12578449). No raw images redistributed.
All counts, source hashes, per-ID/per-query metrics and repeats are in artifacts/followup/.
