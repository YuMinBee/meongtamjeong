# MPDD dog retrieval results

Frozen CLIP/DINO visual transfer; no head training or text input.
This experiment cannot establish a benefit from CLIP-text-to-DINO Flow alignment.

Data: 1657 photos, 191 filename IDs (publisher describes 192).
Split images: {'gallery': 521, 'query': 104, 'train': 921, 'val': 111}. Split IDs: {'train': 95, 'val': 95, 'query': 96, 'gallery': 96}.
Decoded duplicate groups: 0.
Selected DINO score weight: 0.75 (clean validation only).

Scores are percentages, macro-averaged over identities. No published-baseline comparability is implied.
Query perturbations retain the clean gallery. Different-c-code is a filename-code sensitivity analysis.

| Condition | Method | R@1 | R@5 | R@10 | mAP | MRR |
|---|---|---:|---:|---:|---:|---:|
| clean | CLIP | 75.52 | 89.41 | 96.53 | 55.65 | 82.20 |
| clean | DINO | 96.88 | 100.00 | 100.00 | 88.76 | 98.09 |
| clean | equal_fusion | 97.57 | 100.00 | 100.00 | 87.45 | 98.47 |
| clean | selected_fusion | 97.92 | 100.00 | 100.00 | 88.94 | 98.65 |
| clean_different_c_code | CLIP | 72.40 | 89.41 | 95.49 | 55.77 | 80.37 |
| clean_different_c_code | DINO | 95.83 | 98.96 | 100.00 | 87.99 | 97.22 |
| clean_different_c_code | equal_fusion | 96.53 | 100.00 | 100.00 | 87.11 | 97.64 |
| clean_different_c_code | selected_fusion | 96.88 | 98.96 | 100.00 | 88.24 | 97.78 |
| blur | CLIP | 34.38 | 53.12 | 66.15 | 27.90 | 44.92 |
| blur | DINO | 94.79 | 97.92 | 98.96 | 86.63 | 96.28 |
| blur | equal_fusion | 94.62 | 96.88 | 98.96 | 83.72 | 95.71 |
| blur | selected_fusion | 94.79 | 97.92 | 98.96 | 86.30 | 95.80 |
| blur_different_c_code | CLIP | 32.29 | 51.04 | 60.94 | 27.20 | 42.14 |
| blur_different_c_code | DINO | 93.75 | 97.92 | 98.96 | 86.13 | 95.50 |
| blur_different_c_code | equal_fusion | 93.58 | 96.88 | 97.92 | 83.14 | 94.84 |
| blur_different_c_code | selected_fusion | 93.75 | 97.92 | 98.96 | 85.89 | 94.97 |
| lowres | CLIP | 34.38 | 57.81 | 69.97 | 26.06 | 45.60 |
| lowres | DINO | 94.79 | 98.96 | 100.00 | 85.18 | 95.80 |
| lowres | equal_fusion | 93.40 | 97.92 | 97.92 | 83.30 | 95.18 |
| lowres | selected_fusion | 94.79 | 95.83 | 98.96 | 85.14 | 95.69 |
| lowres_different_c_code | CLIP | 31.25 | 53.65 | 63.72 | 25.19 | 41.99 |
| lowres_different_c_code | DINO | 94.79 | 97.92 | 100.00 | 84.58 | 95.72 |
| lowres_different_c_code | equal_fusion | 93.40 | 97.92 | 97.92 | 82.78 | 94.87 |
| lowres_different_c_code | selected_fusion | 94.79 | 95.83 | 98.96 | 84.62 | 95.55 |

## Paired differences from CLIP, percentage points

95% bootstrap intervals over dog IDs, 2,000 resamples; exploratory, no multiplicity correction.

| Condition | Contrast | R@1 difference [95% CI] | mAP difference [95% CI] |
|---|---|---|---|
| clean | DINO | +21.35 [+12.50, +30.73] | +33.12 [+27.72, +38.98] |
| clean | equal_fusion | +22.05 [+14.06, +31.08] | +31.80 [+26.83, +37.17] |
| clean | selected_fusion | +22.40 [+14.06, +31.25] | +33.29 [+28.16, +39.01] |
| clean | selected_minus_DINO | +1.04 [+0.00, +3.12] | +0.18 [-0.67, +1.13] |
| clean_different_c_code | DINO | +23.44 [+14.58, +32.81] | +32.22 [+26.46, +38.35] |
| clean_different_c_code | equal_fusion | +24.13 [+15.62, +33.33] | +31.33 [+26.17, +36.95] |
| clean_different_c_code | selected_fusion | +24.48 [+15.62, +33.85] | +32.46 [+27.02, +38.38] |
| clean_different_c_code | selected_minus_DINO | +1.04 [+0.00, +3.12] | +0.25 [-0.57, +1.10] |
| blur | DINO | +60.42 [+51.04, +70.31] | +58.72 [+52.75, +65.04] |
| blur | equal_fusion | +60.24 [+50.69, +70.14] | +55.81 [+49.78, +61.95] |
| blur | selected_fusion | +60.42 [+51.04, +70.31] | +58.40 [+52.29, +64.53] |
| blur | selected_minus_DINO | +0.00 [+0.00, +0.00] | -0.33 [-1.18, +0.43] |
| blur_different_c_code | DINO | +61.46 [+52.08, +70.83] | +58.93 [+52.58, +65.51] |
| blur_different_c_code | equal_fusion | +61.28 [+51.91, +70.83] | +55.94 [+49.70, +62.15] |
| blur_different_c_code | selected_fusion | +61.46 [+52.08, +70.83] | +58.69 [+52.45, +65.11] |
| blur_different_c_code | selected_minus_DINO | +0.00 [+0.00, +0.00] | -0.24 [-1.02, +0.52] |
| lowres | DINO | +60.42 [+50.52, +70.83] | +59.12 [+52.90, +65.39] |
| lowres | equal_fusion | +59.03 [+48.78, +69.44] | +57.24 [+51.66, +63.22] |
| lowres | selected_fusion | +60.42 [+50.52, +70.83] | +59.08 [+53.21, +65.11] |
| lowres | selected_minus_DINO | +0.00 [+0.00, +0.00] | -0.04 [-0.63, +0.55] |
| lowres_different_c_code | DINO | +63.54 [+54.17, +73.96] | +59.39 [+53.01, +66.02] |
| lowres_different_c_code | equal_fusion | +62.15 [+52.43, +72.40] | +57.58 [+51.68, +63.84] |
| lowres_different_c_code | selected_fusion | +63.54 [+54.17, +73.96] | +59.43 [+53.25, +65.81] |
| lowres_different_c_code | selected_minus_DINO | +0.00 [+0.00, +0.00] | +0.04 [-0.56, +0.64] |

## Scope

Same-ID photos may still share acquisition backgrounds; exact-hash checks do not rule out near duplicates.
No camera/time metadata was verified. Synthetic blur/downsampling is not evidence about real missing-dog deployment.
Foundation-model pretraining overlap is unknown. Results are from a small released dog dataset.
Native preprocessing differs between encoders. No application configuration was changed.

Source: Zhimin He (2023), [MPDD v1](https://data.mendeley.com/datasets/v5j6m8dzhv/1), CC BY 4.0.
See PROTOCOL.md and artifacts/{data_audit,encoders,frozen_selection,results}.json for provenance.
