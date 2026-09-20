# Fixed-photo condition-change results

Scores x100. Columns: target nDCG@10 / P@10 / P@10 gain over original text.
Silver attribute labels; no photographic preference evaluation.

## korea / korean_size

| Method | nDCG | P10 | P10 gain |
|---|---:|---:|---:|
| CLIP_image | 10.28 | 10.09 | +0.00 |
| DINO_image | 9.92 | 9.98 | +0.00 |
| CLIP_mix | 10.23 | 10.05 | +0.05 |
| late_mix | 9.94 | 10.07 | +0.07 |
| CLIP_text_only | 13.37 | 13.92 | +0.25 |
| linear_mix | 12.55 | 12.87 | +3.10 |
| mlp_mix | 12.51 | 12.87 | +3.65 |
| flow_mix | 12.08 | 12.29 | +2.62 |
| linear_text_only | 40.58 | 38.84 | +31.82 |
| mlp_text_only | 35.50 | 34.75 | +26.99 |
| flow_text_only | 36.09 | 37.60 | +29.03 |
| metadata_filter_DINO | 100.00 | 100.00 | +100.00 |

Flow minus CLIP mix nDCG 95% shelter-bootstrap interval: [1.85, 0.44, 3.41]

## korea / korean_color

| Method | nDCG | P10 | P10 gain |
|---|---:|---:|---:|
| CLIP_image | 14.04 | 14.44 | +0.00 |
| DINO_image | 10.20 | 10.61 | +0.00 |
| CLIP_mix | 13.84 | 14.16 | -0.07 |
| late_mix | 10.04 | 10.44 | +0.03 |
| CLIP_text_only | 11.76 | 10.82 | -0.36 |
| linear_mix | 12.33 | 12.64 | +0.79 |
| mlp_mix | 12.11 | 12.59 | +1.39 |
| flow_mix | 12.34 | 12.77 | +1.30 |
| linear_text_only | 41.84 | 41.15 | +21.48 |
| mlp_text_only | 38.76 | 37.85 | +22.56 |
| flow_text_only | 41.27 | 41.85 | +25.87 |
| metadata_filter_DINO | 100.00 | 100.00 | +85.38 |

Flow minus CLIP mix nDCG 95% shelter-bootstrap interval: [-1.5, -3.15, 0.14]

## korea / english_size

| Method | nDCG | P10 | P10 gain |
|---|---:|---:|---:|
| CLIP_image | 10.28 | 10.09 | +0.00 |
| DINO_image | 9.92 | 9.98 | +0.00 |
| CLIP_mix | 10.95 | 10.67 | +0.49 |
| late_mix | 10.13 | 10.27 | +0.29 |
| CLIP_text_only | 25.59 | 23.66 | +9.13 |
| linear_mix | 12.10 | 12.42 | +2.49 |
| mlp_mix | 12.09 | 12.42 | +2.33 |
| flow_mix | 12.01 | 12.36 | +2.51 |
| linear_text_only | 31.17 | 31.39 | +18.36 |
| mlp_text_only | 29.90 | 29.96 | +16.99 |
| flow_text_only | 30.71 | 32.62 | +22.33 |
| metadata_filter_DINO | 100.00 | 100.00 | +100.00 |

Flow minus CLIP mix nDCG 95% shelter-bootstrap interval: [1.06, -0.46, 2.54]

## korea / english_color

| Method | nDCG | P10 | P10 gain |
|---|---:|---:|---:|
| CLIP_image | 14.04 | 14.44 | +0.00 |
| DINO_image | 10.20 | 10.61 | +0.00 |
| CLIP_mix | 15.52 | 15.90 | +2.07 |
| late_mix | 10.50 | 10.97 | +0.48 |
| CLIP_text_only | 23.97 | 22.05 | +15.74 |
| linear_mix | 12.38 | 12.95 | +2.61 |
| mlp_mix | 12.81 | 13.31 | +3.16 |
| flow_mix | 12.31 | 12.87 | +2.44 |
| linear_text_only | 33.71 | 34.54 | +26.39 |
| mlp_text_only | 34.72 | 34.98 | +26.30 |
| flow_text_only | 36.93 | 37.77 | +28.57 |
| metadata_filter_DINO | 100.00 | 100.00 | +85.38 |

Flow minus CLIP mix nDCG 95% shelter-bootstrap interval: [-3.22, -4.96, -1.55]

## taiwan / korean_size

| Method | nDCG | P10 | P10 gain |
|---|---:|---:|---:|
| CLIP_image | 9.24 | 9.27 | +0.00 |
| DINO_image | 12.66 | 13.00 | +0.00 |
| CLIP_mix | 9.33 | 9.34 | +0.01 |
| late_mix | 12.71 | 13.05 | +0.02 |
| CLIP_text_only | 8.10 | 8.89 | +3.51 |
| linear_mix | 13.09 | 13.41 | +0.23 |
| mlp_mix | 12.96 | 13.24 | +0.17 |
| flow_mix | 13.00 | 13.35 | +0.19 |
| linear_text_only | 16.18 | 14.28 | +0.36 |
| mlp_text_only | 15.51 | 15.93 | +4.46 |
| flow_text_only | 24.36 | 26.46 | +16.82 |
| metadata_filter_DINO | 100.00 | 100.00 | +100.00 |

Flow minus CLIP mix nDCG 95% shelter-bootstrap interval: [3.67, 2.88, 4.45]

## taiwan / korean_color

| Method | nDCG | P10 | P10 gain |
|---|---:|---:|---:|
| CLIP_image | 1.97 | 1.98 | +0.00 |
| DINO_image | 0.90 | 0.89 | +0.00 |
| CLIP_mix | 1.99 | 2.01 | +0.00 |
| late_mix | 0.90 | 0.88 | -0.01 |
| CLIP_text_only | 2.54 | 2.88 | +0.14 |
| linear_mix | 1.00 | 0.98 | +0.11 |
| mlp_mix | 1.03 | 1.02 | +0.13 |
| flow_mix | 1.08 | 1.05 | +0.16 |
| linear_text_only | 8.40 | 7.05 | +5.14 |
| mlp_text_only | 8.65 | 7.58 | +5.54 |
| flow_text_only | 10.10 | 9.98 | +8.22 |
| metadata_filter_DINO | 100.00 | 100.00 | +99.97 |

Flow minus CLIP mix nDCG 95% shelter-bootstrap interval: [-0.91, -1.19, -0.64]

## taiwan / english_size

| Method | nDCG | P10 | P10 gain |
|---|---:|---:|---:|
| CLIP_image | 9.24 | 9.27 | +0.00 |
| DINO_image | 12.66 | 13.00 | +0.00 |
| CLIP_mix | 10.47 | 10.49 | +0.18 |
| late_mix | 12.83 | 13.13 | +0.01 |
| CLIP_text_only | 30.67 | 28.20 | +12.30 |
| linear_mix | 13.75 | 14.06 | +0.61 |
| mlp_mix | 13.90 | 14.24 | +0.81 |
| flow_mix | 13.58 | 13.89 | +0.36 |
| linear_text_only | 29.81 | 32.13 | +15.65 |
| mlp_text_only | 33.42 | 33.17 | +20.00 |
| flow_text_only | 27.19 | 28.78 | +10.89 |
| metadata_filter_DINO | 100.00 | 100.00 | +100.00 |

Flow minus CLIP mix nDCG 95% shelter-bootstrap interval: [3.11, 2.24, 3.93]

## taiwan / english_color

| Method | nDCG | P10 | P10 gain |
|---|---:|---:|---:|
| CLIP_image | 1.97 | 1.98 | +0.00 |
| DINO_image | 0.90 | 0.89 | +0.00 |
| CLIP_mix | 2.95 | 2.98 | +1.32 |
| late_mix | 0.95 | 0.94 | +0.08 |
| CLIP_text_only | 12.18 | 11.26 | +10.27 |
| linear_mix | 1.27 | 1.26 | +0.46 |
| mlp_mix | 1.33 | 1.32 | +0.54 |
| flow_mix | 1.29 | 1.29 | +0.51 |
| linear_text_only | 13.86 | 14.02 | +12.67 |
| mlp_text_only | 16.04 | 14.09 | +12.96 |
| flow_text_only | 12.77 | 12.87 | +11.71 |
| metadata_filter_DINO | 100.00 | 100.00 | +99.97 |

Flow minus CLIP mix nDCG 95% shelter-bootstrap interval: [-1.66, -2.02, -1.27]
