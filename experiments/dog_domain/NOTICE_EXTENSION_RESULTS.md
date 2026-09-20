# New Korean notice results

611 new notice pairs; 136 shelters; full-photo gallery of 611.
Fixed text weight .20; frozen original heads; no test tuning. See NOTICE_EXTENSION_PROTOCOL.md.

| Method | R@1 | R@5 | R@10 | MRR | Color+size nDCG@10 |
|---|---:|---:|---:|---:|---:|
| CLIP_image | 71.03 | 83.47 | 87.23 | 76.93 | 33.01 |
| DINO_image | 92.96 | 96.89 | 98.85 | 94.91 | 40.63 |
| korean/CLIP_mix | 70.70 | 83.63 | 87.23 | 76.71 | 32.97 |
| korean/late_mix | 92.64 | 96.89 | 98.53 | 94.72 | 40.65 |
| korean/CLIP_shift | 71.03 | 83.96 | 87.23 | 76.89 | 32.87 |
| korean/CLIP_text_only | 0.16 | 0.82 | 1.64 | 1.08 | 17.29 |
| korean/flow_mix | 91.98 | 97.05 | 98.36 | 94.30 | 43.98 |
| korean/flow_text_only | 1.31 | 7.53 | 14.89 | 6.19 | 49.73 |
| korean/flow_shift | 91.00 | 96.56 | 98.20 | 93.49 | 40.79 |
| korean/linear_mix | 91.33 | 97.55 | 98.53 | 94.01 | 43.75 |
| korean/linear_text_only | 1.31 | 6.55 | 12.77 | 5.69 | 51.71 |
| korean/mlp_mix | 91.33 | 97.05 | 98.36 | 93.86 | 43.85 |
| korean/mlp_text_only | 0.82 | 6.55 | 11.78 | 5.26 | 46.84 |
| english/CLIP_mix | 71.36 | 83.96 | 87.56 | 77.39 | 35.21 |
| english/late_mix | 92.64 | 96.89 | 98.69 | 94.72 | 41.12 |
| english/CLIP_shift | 70.87 | 83.31 | 87.07 | 76.70 | 33.41 |
| english/CLIP_text_only | 1.31 | 3.93 | 7.04 | 3.90 | 33.96 |
| english/flow_mix | 91.16 | 97.05 | 98.36 | 93.92 | 43.14 |
| english/flow_text_only | 1.31 | 8.84 | 16.69 | 6.52 | 41.94 |
| english/flow_shift | 91.33 | 96.73 | 98.69 | 93.74 | 40.20 |
| english/linear_mix | 91.00 | 97.38 | 98.20 | 93.79 | 43.73 |
| english/linear_text_only | 1.47 | 8.02 | 13.91 | 6.23 | 41.02 |
| english/mlp_mix | 91.16 | 97.38 | 98.36 | 93.89 | 43.83 |
| english/mlp_text_only | 1.31 | 7.20 | 14.73 | 6.29 | 42.66 |
| description/CLIP_mix | 70.54 | 83.80 | 87.40 | 76.65 | 32.90 |
| description/late_mix | 92.47 | 96.89 | 98.85 | 94.67 | 40.61 |
| description/CLIP_shift | 71.03 | 83.63 | 87.40 | 76.97 | 32.94 |
| description/CLIP_text_only | 0.49 | 1.31 | 1.64 | 1.41 | 19.13 |
| description/flow_mix | 92.14 | 97.38 | 98.69 | 94.29 | 40.32 |
| description/flow_text_only | 0.00 | 1.31 | 2.29 | 1.16 | 13.95 |
| description/flow_shift | 92.47 | 97.22 | 98.20 | 94.45 | 40.40 |
| description/linear_mix | 91.16 | 97.05 | 98.04 | 93.82 | 40.13 |
| description/linear_text_only | 0.16 | 1.15 | 2.45 | 1.27 | 13.89 |
| description/mlp_mix | 91.33 | 96.73 | 98.20 | 93.79 | 39.85 |
| description/mlp_text_only | 0.65 | 1.31 | 2.29 | 1.63 | 14.12 |

Paired shelter-bootstrap 95% intervals, percentage points; exploratory contrasts are not multiplicity-adjusted.

| Contrast | R@1 difference [95% CI] | Semantic difference [95% CI] |
|---|---|---|
| korean/flow_mix minus korean/CLIP_mix | +21.28 [+16.73, +25.85] | +11.01 [+8.44, +13.51] |
| korean/flow_mix minus DINO_image | -0.98 [-1.97, -0.15] | +3.35 [+2.72, +3.96] |
| korean/flow_mix minus korean/late_mix | -0.65 [-1.63, +0.17] | +3.33 [+2.71, +3.98] |
| korean/flow_mix minus korean/linear_mix | +0.65 [-0.18, +1.52] | +0.23 [-0.15, +0.61] |
| korean/flow_mix minus korean/mlp_mix | +0.65 [-0.37, +1.58] | +0.13 [-0.27, +0.53] |
| korean/flow_mix minus korean/flow_shift | +0.98 [+0.00, +2.15] | +3.19 [+2.35, +3.93] |
| english/flow_mix minus english/CLIP_mix | +19.80 [+15.45, +24.43] | +7.93 [+5.59, +10.10] |
| english/flow_mix minus DINO_image | -1.80 [-3.16, -0.74] | +2.51 [+1.78, +3.19] |
| english/flow_mix minus english/late_mix | -1.47 [-2.71, -0.50] | +2.02 [+1.31, +2.69] |
| english/flow_mix minus english/linear_mix | +0.16 [-0.56, +0.85] | -0.59 [-0.93, -0.25] |
| english/flow_mix minus english/mlp_mix | +0.00 [-1.10, +1.00] | -0.69 [-1.13, -0.25] |
| english/flow_mix minus english/flow_shift | -0.16 [-1.36, +0.90] | +2.94 [+2.09, +3.80] |
| description/flow_mix minus description/CLIP_mix | +21.60 [+17.05, +26.27] | +7.41 [+5.20, +9.61] |
| description/flow_mix minus DINO_image | -0.82 [-1.97, +0.15] | -0.32 [-0.87, +0.21] |
| description/flow_mix minus description/late_mix | -0.33 [-1.49, +0.69] | -0.29 [-0.84, +0.26] |
| description/flow_mix minus description/linear_mix | +0.98 [+0.16, +1.99] | +0.18 [-0.25, +0.57] |
| description/flow_mix minus description/mlp_mix | +0.82 [-0.20, +1.86] | +0.47 [-0.10, +1.00] |
| description/flow_mix minus description/flow_shift | -0.33 [-1.28, +0.70] | -0.09 [-0.74, +0.53] |

Public attributes are silver relevance labels, not human query judgments. The query notice is excluded from semantic ranking.
Descriptions are original public text. Their semantic scores measure only color/size agreement, not full-description relevance.
Same-notice identity is a proxy, not verified same-animal identification. Near copies across different notices and repeat animals remain possible.
The original pilot used crops and a different gallery; absolute scores should not be directly compared.
