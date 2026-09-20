# CLIP–DINO composed retrieval: experiment 1

Completed frozen-protocol experiment. Scores below are percentages.

Development-selected family: **linear_residual**. This choice was frozen before
the local CIRCO audit and all four GeneCIS evaluations. Family hyperparameters
were selected separately using mean development mAP@5/10/25/50 across seeds.

## Data and interpretation

- COCO paired supervision: 5,000 training / 1,000 validation images; all captions.
- CIRCO: 116 development / 104 audit queries; full 123,403-image gallery.
  Reference/positive connected components keep labeled images out of both splits.
  These are **local partitions of official validation**, not official test results.
- GeneCIS: all 8,032 queries across four official tasks; no task-specific tuning.
  Official candidate slots and duplicate labels are preserved.
- All 51,208 VG-linked COCO IDs excluded before training-image sampling.
  Source-ID and exact decoded-image checks passed; pretraining overlap is unknown.
- CIRR has annotations only and is not scored.

## Encoders and methods

CLIP `ViT-B/32` weights SHA-256 `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`.
DINO `facebook/dinov3-vitb16-pretrain-lvd1689m` revision `5931719e67bbdb9737e363e781fb0c67687896bc`.
GPU: NVIDIA GeForce RTX 4090; torch 2.11.0+cu128.
Both frozen encoders use their native preprocessing of the same image/crop.
CLIP uses resize/center-crop; DINO uses its released 224×224 processor.
Raw captions/conditions are tokenized with CLIP truncation enabled.

For normalized features c(image), t(text), d(image), and mapped normalized text h(t):

- CLIP composition: cosine(normalize((1−w)c(reference)+w·t), c(candidate)).
- Aligned DINO: cosine(normalize((1−w)d(reference)+w·h(t)), d(candidate)).
- Late fusion: (1−w)·DINO image cosine + w·CLIP text-to-image cosine.
- Residual: r·aligned DINO composition + (1−r)·CLIP composition, same w.
- Unaligned residual control: r·DINO image cosine + (1−r)·CLIP composition.
- Linear/MLP/flow share caption pairs, loss, optimizer and 30-epoch budget;
  checkpoints selected by COCO validation loss. Seeds: 7, 42, 123.
  These methods use supervised paired-data training and are not training-free.

## Frozen choices

| Family | Text weight w | DINO residual weight r | Development mean mAP |
|---|---:|---:|---:|
| clip_image | — | — | 2.48 |
| clip_text | — | — | 2.48 |
| dino_image | — | — | 5.16 |
| clip_composition | 0.8 | — | 8.29 |
| late_fusion | 0.8 | — | 10.40 |
| linear | 0.6 | — | 7.58 |
| mlp | 0.6 | — | 7.27 |
| flow | 0.6 | — | 7.46 |
| unaligned_residual | 1.0 | 0.25 | 9.42 |
| linear_residual | 0.6 | 0.5 | 10.76 |
| mlp_residual | 0.6 | 0.5 | 10.31 |
| flow_residual | 0.6 | 0.5 | 10.07 |

## Held-out results

Learned methods show the mean across three seeds. Baselines are deterministic.
CIRCO columns are mAP@5 / @10 / @25 / @50; GeneCIS columns are Recall@1.

| Method | CIRCO audit | Focus attribute | Change attribute | Focus object | Change object |
|---|---:|---:|---:|---:|---:|
| clip_image | 4.12 / 4.66 / 5.26 / 5.70 | 16.95 | 11.46 | 10.10 | 8.01 |
| clip_text | 3.55 / 3.91 / 4.48 / 4.56 | 9.80 | 8.48 | 7.55 | 7.04 |
| dino_image | 3.79 / 4.60 / 5.51 / 5.80 | 18.60 | 11.36 | 10.97 | 8.37 |
| clip_composition | 11.77 / 12.14 / 13.23 / 13.82 | 14.65 | 12.26 | 12.96 | 12.40 |
| late_fusion | 12.70 / 13.29 / 14.54 / 15.12 | 17.35 | 13.35 | 13.67 | 12.24 |
| linear | 5.84 / 7.15 / 8.55 / 9.11 | 17.80 | 12.37 | 12.89 | 14.47 |
| mlp | 5.93 / 7.10 / 8.50 / 9.09 | 17.03 | 12.01 | 13.01 | 14.40 |
| flow | 5.34 / 6.43 / 7.79 / 8.32 | 17.95 | 11.99 | 12.84 | 13.96 |
| unaligned_residual | 11.33 / 12.25 / 13.46 / 13.99 | 18.20 | 13.49 | 13.72 | 11.94 |
| linear_residual | 10.48 / 12.14 / 13.75 / 14.22 | 18.35 | 12.97 | 13.72 | 14.06 |
| mlp_residual | 10.47 / 12.39 / 13.96 / 14.43 | 17.98 | 12.93 | 13.66 | 13.96 |
| flow_residual | 9.73 / 11.18 / 12.86 / 13.36 | 18.60 | 12.88 | 13.59 | 13.79 |
| clip_composition_fixed_0.2 | 4.87 / 5.50 / 6.30 / 6.67 | 16.75 | 11.51 | 10.15 | 8.57 |
| linear_fixed_0.2 | 4.78 / 5.56 / 6.52 / 6.90 | 18.32 | 11.95 | 12.07 | 9.98 |
| mlp_fixed_0.2 | 4.67 / 5.41 / 6.38 / 6.78 | 18.32 | 12.06 | 12.09 | 9.80 |
| flow_fixed_0.2 | 4.98 / 5.73 / 6.65 / 7.06 | 18.58 | 11.81 | 11.97 | 9.98 |

## Paired uncertainty for the development-selected family

Differences are percentage points, with 95% paired cluster-bootstrap intervals
(2,000 resamples). Seed scores are averaged per query before resampling.
CIRCO clusters are connected components; GeneCIS clusters are reference images.
CIRCO below uses mAP@5; GeneCIS uses Recall@1. These exploratory comparisons
are not adjusted for multiple comparisons.

| Dataset | Baseline | Difference | 95% CI |
|---|---|---:|---:|
| CIRCO audit | clip_composition | -1.30 | [-5.73, +3.30] |
| CIRCO audit | late_fusion | -2.22 | [-5.61, +1.03] |
| CIRCO audit | unaligned_residual | -0.85 | [-3.91, +2.03] |
| focus_attribute | clip_composition | +3.70 | [+2.04, +5.48] |
| focus_attribute | late_fusion | +1.00 | [-0.32, +2.40] |
| focus_attribute | unaligned_residual | +0.15 | [-1.18, +1.50] |
| change_attribute | clip_composition | +0.71 | [-0.95, +2.38] |
| change_attribute | late_fusion | -0.38 | [-1.82, +0.94] |
| change_attribute | unaligned_residual | -0.52 | [-1.84, +0.75] |
| focus_object | clip_composition | +0.77 | [-1.83, +2.99] |
| focus_object | late_fusion | +0.05 | [-2.38, +2.27] |
| focus_object | unaligned_residual | -0.00 | [-2.46, +2.22] |
| change_object | clip_composition | +1.67 | [-0.35, +3.73] |
| change_object | late_fusion | +1.82 | [-0.13, +3.73] |
| change_object | unaligned_residual | +2.13 | [+0.34, +3.81] |

## Head training

Validation loss uses all held-out captions against the entire 1,000-image validation gallery.

| Head | Parameters | Best epochs (7/42/123) | Validation losses | Training seconds (sum) |
|---|---:|---|---|---:|
| linear | 393,216 | 7 / 8 / 7 | 2.8743 / 2.8733 / 2.8741 | 18.4 |
| mlp | 656,640 | 7 / 8 / 10 | 2.8873 / 2.8851 / 2.8859 | 21.8 |
| flow | 621,984 | 5 / 6 / 5 | 2.8422 / 2.8395 / 2.8428 | 239.2 |

## Full task metrics and seed variability

### CIRCO audit

Queries: 104; clusters: 101.
Metrics: mAP@5 / mAP@10 / mAP@25 / mAP@50.

| Method | Mean | 95% CI lower | 95% CI upper | Seed SD |
|---|---|---|---|---|
| clip_image | 4.12 / 4.66 / 5.26 / 5.70 | 1.39 / 1.86 / 2.38 / 2.81 | 7.78 / 8.32 / 9.07 / 9.48 | — |
| clip_text | 3.55 / 3.91 / 4.48 / 4.56 | 1.40 / 1.76 / 2.21 / 2.29 | 6.21 / 6.64 / 7.33 / 7.43 | — |
| dino_image | 3.79 / 4.60 / 5.51 / 5.80 | 1.75 / 2.41 / 3.20 / 3.48 | 6.36 / 7.29 / 8.28 / 8.54 | — |
| clip_composition | 11.77 / 12.14 / 13.23 / 13.82 | 7.28 / 7.59 / 8.67 / 9.29 | 16.64 / 16.95 / 18.18 / 18.75 | — |
| late_fusion | 12.70 / 13.29 / 14.54 / 15.12 | 8.87 / 9.37 / 10.61 / 11.22 | 17.40 / 17.92 / 19.11 / 19.69 | — |
| linear | 5.84 / 7.15 / 8.55 / 9.11 | 3.66 / 4.90 / 6.16 / 6.70 | 8.35 / 9.77 / 11.29 / 11.84 | 0.51 / 0.46 / 0.42 / 0.44 |
| mlp | 5.93 / 7.10 / 8.50 / 9.09 | 3.66 / 4.75 / 5.95 / 6.51 | 8.57 / 9.76 / 11.24 / 11.83 | 0.71 / 0.74 / 0.68 / 0.72 |
| flow | 5.34 / 6.43 / 7.79 / 8.32 | 3.36 / 4.26 / 5.43 / 5.96 | 7.65 / 8.80 / 10.26 / 10.75 | 0.66 / 0.50 / 0.61 / 0.62 |
| unaligned_residual | 11.33 / 12.25 / 13.46 / 13.99 | 7.49 / 8.44 / 9.70 / 10.23 | 16.03 / 16.77 / 17.98 / 18.47 | — |
| linear_residual | 10.48 / 12.14 / 13.75 / 14.22 | 6.88 / 8.42 / 10.07 / 10.55 | 14.84 / 16.56 / 18.00 / 18.44 | 0.38 / 0.26 / 0.22 / 0.25 |
| mlp_residual | 10.47 / 12.39 / 13.96 / 14.43 | 6.73 / 8.59 / 10.15 / 10.61 | 14.92 / 16.89 / 18.48 / 18.94 | 0.34 / 0.07 / 0.19 / 0.10 |
| flow_residual | 9.73 / 11.18 / 12.86 / 13.36 | 6.31 / 7.72 / 9.32 / 9.85 | 13.64 / 15.10 / 16.77 / 17.26 | 0.59 / 0.62 / 0.77 / 0.68 |
| clip_composition_fixed_0.2 | 4.87 / 5.50 / 6.30 / 6.67 | 1.99 / 2.58 / 3.25 / 3.61 | 8.57 / 9.18 / 10.00 / 10.38 | — |
| linear_fixed_0.2 | 4.78 / 5.56 / 6.52 / 6.90 | 2.48 / 3.20 / 4.07 / 4.39 | 7.65 / 8.42 / 9.51 / 9.86 | 0.34 / 0.34 / 0.32 / 0.33 |
| mlp_fixed_0.2 | 4.67 / 5.41 / 6.38 / 6.78 | 2.47 / 3.15 / 3.98 / 4.37 | 7.38 / 8.16 / 9.32 / 9.67 | 0.12 / 0.12 / 0.11 / 0.12 |
| flow_fixed_0.2 | 4.98 / 5.73 / 6.65 / 7.06 | 2.63 / 3.30 / 4.16 / 4.52 | 7.83 / 8.61 / 9.68 / 10.09 | 0.36 / 0.42 / 0.39 / 0.39 |

Random-ranking expectation: 0.00 / 0.00 / 0.00 / 0.00.

### focus_attribute

Queries: 2000; clusters: 1863.
Metrics: R@1 / R@2 / R@3.

| Method | Mean | 95% CI lower | 95% CI upper | Seed SD |
|---|---|---|---|---|
| clip_image | 16.95 / 29.05 / 40.90 | 15.32 / 26.99 / 38.62 | 18.63 / 31.08 / 43.00 | — |
| clip_text | 9.80 / 19.55 / 28.95 | 8.49 / 17.75 / 26.92 | 11.10 / 21.34 / 31.07 | — |
| dino_image | 18.60 / 32.65 / 43.20 | 16.88 / 30.54 / 40.94 | 20.37 / 34.73 / 45.40 | — |
| clip_composition | 14.65 / 25.60 / 35.85 | 13.12 / 23.72 / 33.74 | 16.24 / 27.53 / 37.96 | — |
| late_fusion | 17.35 / 29.95 / 40.95 | 15.74 / 27.96 / 38.79 | 19.15 / 32.00 / 43.16 | — |
| linear | 17.80 / 31.23 / 42.13 | 16.21 / 29.35 / 40.10 | 19.38 / 33.15 / 44.23 | 0.13 / 0.20 / 0.03 |
| mlp | 17.03 / 30.48 / 41.37 | 15.44 / 28.56 / 39.26 | 18.60 / 32.39 / 43.47 | 0.40 / 0.28 / 0.24 |
| flow | 17.95 / 31.42 / 42.55 | 16.48 / 29.62 / 40.55 | 19.46 / 33.30 / 44.48 | 0.13 / 0.65 / 0.98 |
| unaligned_residual | 18.20 / 30.85 / 41.15 | 16.48 / 28.81 / 39.00 | 20.01 / 32.92 / 43.34 | — |
| linear_residual | 18.35 / 31.50 / 42.28 | 16.71 / 29.51 / 40.22 | 20.02 / 33.40 / 44.27 | 0.31 / 0.28 / 0.13 |
| mlp_residual | 17.98 / 31.17 / 42.35 | 16.42 / 29.21 / 40.23 | 19.63 / 33.03 / 44.33 | 0.14 / 0.08 / 0.30 |
| flow_residual | 18.60 / 32.28 / 42.80 | 17.02 / 30.45 / 40.79 | 20.21 / 34.14 / 44.75 | 0.40 / 1.15 / 1.04 |
| clip_composition_fixed_0.2 | 16.75 / 29.10 / 40.65 | 15.10 / 27.00 / 38.40 | 18.46 / 31.14 / 42.79 | — |
| linear_fixed_0.2 | 18.32 / 33.10 / 42.87 | 16.61 / 30.96 / 40.69 | 20.03 / 35.17 / 45.03 | 0.10 / 0.28 / 0.16 |
| mlp_fixed_0.2 | 18.32 / 33.18 / 42.88 | 16.61 / 31.04 / 40.72 | 20.00 / 35.20 / 45.00 | 0.21 / 0.25 / 0.35 |
| flow_fixed_0.2 | 18.58 / 32.90 / 43.03 | 16.89 / 30.81 / 40.87 | 20.35 / 34.88 / 45.16 | 0.23 / 0.22 / 0.08 |

Random-ranking expectation: 10.00 / 20.00 / 30.00.

### change_attribute

Queries: 2112; clusters: 1797.
Metrics: R@1 / R@2 / R@3.

| Method | Mean | 95% CI lower | 95% CI upper | Seed SD |
|---|---|---|---|---|
| clip_image | 11.46 / 20.79 / 29.64 | 9.96 / 18.91 / 27.54 | 13.03 / 22.66 / 31.80 | — |
| clip_text | 8.48 / 16.67 / 24.05 | 7.15 / 15.02 / 22.01 | 9.88 / 18.53 / 26.02 | — |
| dino_image | 11.36 / 21.02 / 30.21 | 10.00 / 19.16 / 28.05 | 12.76 / 22.87 / 32.39 | — |
| clip_composition | 12.26 / 21.26 / 30.73 | 10.85 / 19.46 / 28.56 | 13.82 / 23.14 / 32.99 | — |
| late_fusion | 13.35 / 23.44 / 32.53 | 11.86 / 21.50 / 30.48 | 15.03 / 25.46 / 34.80 | — |
| linear | 12.37 / 22.13 / 31.49 | 11.00 / 20.33 / 29.53 | 13.85 / 24.00 / 33.57 | 0.07 / 0.40 / 0.24 |
| mlp | 12.01 / 21.64 / 31.17 | 10.65 / 19.85 / 29.18 | 13.43 / 23.56 / 33.24 | 0.30 / 0.68 / 0.21 |
| flow | 11.99 / 21.48 / 30.71 | 10.66 / 19.73 / 28.73 | 13.34 / 23.31 / 32.73 | 0.49 / 0.12 / 0.32 |
| unaligned_residual | 13.49 / 23.30 / 32.91 | 11.96 / 21.35 / 30.73 | 15.08 / 25.18 / 35.20 | — |
| linear_residual | 12.97 / 23.69 / 32.04 | 11.51 / 21.84 / 30.01 | 14.44 / 25.54 / 34.11 | 0.33 / 0.52 / 0.77 |
| mlp_residual | 12.93 / 23.06 / 32.23 | 11.51 / 21.24 / 30.22 | 14.43 / 24.89 / 34.29 | 0.48 / 0.41 / 0.19 |
| flow_residual | 12.88 / 23.00 / 31.77 | 11.47 / 21.19 / 29.82 | 14.24 / 24.80 / 33.79 | 0.16 / 0.18 / 0.42 |
| clip_composition_fixed_0.2 | 11.51 / 21.02 / 30.30 | 10.05 / 19.09 / 28.23 | 12.96 / 22.94 / 32.46 | — |
| linear_fixed_0.2 | 11.95 / 21.21 / 30.52 | 10.53 / 19.30 / 28.40 | 13.39 / 23.14 / 32.71 | 0.14 / 0.13 / 0.23 |
| mlp_fixed_0.2 | 12.06 / 21.31 / 30.54 | 10.67 / 19.43 / 28.44 | 13.53 / 23.22 / 32.66 | 0.05 / 0.05 / 0.09 |
| flow_fixed_0.2 | 11.81 / 21.32 / 30.63 | 10.38 / 19.49 / 28.54 | 13.22 / 23.21 / 32.79 | 0.14 / 0.23 / 0.16 |

Random-ranking expectation: 6.67 / 13.33 / 20.00.

### focus_object

Queries: 1960; clusters: 218.
Metrics: R@1 / R@2 / R@3.

| Method | Mean | 95% CI lower | 95% CI upper | Seed SD |
|---|---|---|---|---|
| clip_image | 10.10 / 19.23 / 27.81 | 8.17 / 16.68 / 25.24 | 12.16 / 21.78 / 30.48 | — |
| clip_text | 7.55 / 16.73 / 26.38 | 5.79 / 13.93 / 23.39 | 9.42 / 19.53 / 29.56 | — |
| dino_image | 10.97 / 18.16 / 28.83 | 9.40 / 16.24 / 26.01 | 12.48 / 20.16 / 31.66 | — |
| clip_composition | 12.96 / 23.88 / 33.67 | 10.65 / 21.01 / 30.47 | 15.47 / 26.81 / 36.84 | — |
| late_fusion | 13.67 / 23.27 / 33.67 | 11.41 / 20.52 / 30.55 | 16.22 / 26.39 / 36.91 | — |
| linear | 12.89 / 23.83 / 33.72 | 10.91 / 21.05 / 30.63 | 14.84 / 26.67 / 37.03 | 0.51 / 0.18 / 0.22 |
| mlp | 13.01 / 23.66 / 33.11 | 10.99 / 20.80 / 29.99 | 14.99 / 26.73 / 36.43 | 0.20 / 0.77 / 0.26 |
| flow | 12.84 / 23.55 / 33.04 | 11.01 / 20.97 / 30.09 | 14.76 / 26.19 / 36.17 | 0.16 / 0.67 / 0.66 |
| unaligned_residual | 13.72 / 23.06 / 32.86 | 11.38 / 20.35 / 29.83 | 16.18 / 25.88 / 35.89 | — |
| linear_residual | 13.72 / 24.71 / 34.42 | 11.69 / 21.87 / 31.25 | 15.91 / 27.58 / 37.54 | 0.49 / 0.61 / 0.39 |
| mlp_residual | 13.66 / 24.63 / 32.99 | 11.72 / 21.89 / 29.91 | 15.72 / 27.45 / 36.15 | 0.41 / 1.03 / 0.50 |
| flow_residual | 13.59 / 23.62 / 34.01 | 11.60 / 21.00 / 30.91 | 15.72 / 26.33 / 37.19 | 0.29 / 0.56 / 0.75 |
| clip_composition_fixed_0.2 | 10.15 / 20.56 / 28.67 | 8.18 / 18.01 / 25.86 | 12.20 / 23.24 / 31.60 | — |
| linear_fixed_0.2 | 12.07 / 20.00 / 30.31 | 10.30 / 17.98 / 27.59 | 13.85 / 22.06 / 33.18 | 0.06 / 0.13 / 0.00 |
| mlp_fixed_0.2 | 12.09 / 20.17 / 29.90 | 10.29 / 18.11 / 27.20 | 13.88 / 22.23 / 32.68 | 0.13 / 0.11 / 0.18 |
| flow_fixed_0.2 | 11.97 / 20.00 / 30.27 | 10.19 / 18.00 / 27.57 | 13.79 / 22.08 / 33.05 | 0.21 / 0.37 / 0.16 |

Random-ranking expectation: 6.67 / 13.33 / 20.00.

### change_object

Queries: 1960; clusters: 218.
Metrics: R@1 / R@2 / R@3.

| Method | Mean | 95% CI lower | 95% CI upper | Seed SD |
|---|---|---|---|---|
| clip_image | 8.01 / 17.60 / 25.92 | 6.51 / 15.26 / 23.25 | 9.54 / 19.90 / 28.49 | — |
| clip_text | 7.04 / 16.22 / 24.29 | 5.78 / 14.18 / 21.90 | 8.42 / 18.44 / 26.77 | — |
| dino_image | 8.37 / 16.73 / 26.22 | 6.91 / 14.60 / 23.48 | 9.90 / 18.83 / 29.00 | — |
| clip_composition | 12.40 / 22.60 / 32.19 | 10.49 / 20.00 / 29.49 | 14.24 / 25.10 / 34.82 | — |
| late_fusion | 12.24 / 23.27 / 33.11 | 10.33 / 20.78 / 30.21 | 14.11 / 25.82 / 36.21 | — |
| linear | 14.47 / 24.88 / 36.26 | 12.71 / 22.53 / 33.64 | 16.32 / 27.29 / 38.97 | 0.45 / 0.39 / 0.32 |
| mlp | 14.40 / 24.93 / 36.07 | 12.69 / 22.73 / 33.49 | 16.19 / 27.18 / 38.86 | 0.24 / 0.51 / 0.51 |
| flow | 13.96 / 24.69 / 35.60 | 12.36 / 22.46 / 33.14 | 15.57 / 27.00 / 38.18 | 0.11 / 0.39 / 0.66 |
| unaligned_residual | 11.94 / 23.37 / 32.86 | 10.08 / 20.85 / 29.93 | 13.84 / 25.84 / 35.89 | — |
| linear_residual | 14.06 / 25.15 / 35.99 | 12.22 / 22.71 / 33.35 | 15.90 / 27.52 / 38.68 | 0.16 / 0.41 / 0.64 |
| mlp_residual | 13.96 / 25.27 / 35.49 | 12.08 / 22.88 / 32.85 | 15.81 / 27.70 / 38.14 | 0.13 / 0.45 / 0.31 |
| flow_residual | 13.79 / 24.47 / 35.39 | 12.00 / 22.17 / 32.80 | 15.58 / 26.78 / 37.96 | 0.51 / 0.55 / 0.72 |
| clip_composition_fixed_0.2 | 8.57 / 18.21 / 27.14 | 7.02 / 15.89 / 24.41 | 10.10 / 20.43 / 29.81 | — |
| linear_fixed_0.2 | 9.98 / 18.37 / 29.05 | 8.26 / 16.22 / 26.29 | 11.77 / 20.44 / 31.89 | 0.18 / 0.15 / 0.16 |
| mlp_fixed_0.2 | 9.80 / 18.32 / 29.18 | 8.10 / 16.21 / 26.41 | 11.59 / 20.38 / 31.99 | 0.22 / 0.09 / 0.09 |
| flow_fixed_0.2 | 9.98 / 18.27 / 28.88 | 8.23 / 16.08 / 26.11 | 11.76 / 20.34 / 31.65 | 0.39 / 0.13 / 0.13 |

Random-ranking expectation: 6.67 / 13.33 / 20.00.

## Limits and reproducibility

This is a pilot at a fixed 5,000-image training scale. A small local CIRCO audit
and four related GeneCIS tasks do not establish broad superiority or novelty.
The heads are different parameter counts, though data and optimization budgets match.
CLIP and DINO have different pretraining and native preprocessing; neither is controlled
by this experiment. English general-image benchmarks do not establish Korean-language
or lost-dog retrieval performance. Flow is the project's existing hyperspherical mapper, not an
official reproduction of another paper. Further changes require a new prospective
protocol and fresh held-out data, rather than tuning on these audit outcomes.

Selection artifact SHA-256: `5f0c5ef86b0d303e985d22e433fbb9b313afabb429f6124ae534642994422804`.
Raw per-query, per-seed metrics: `artifacts/evaluation/*_per_query.npz`.
Machine-readable full results and paired differences: `artifacts/evaluation/*.json`.
Data integrity: `artifacts/data_readiness.json`; encoder/crop/source hashes:
`artifacts/features/index.json` and image chunks. All large/raw artifacts are Git-ignored.

Primary benchmark sources: [CIRCO](https://github.com/miccunifi/CIRCO),
[GeneCIS](https://github.com/facebookresearch/genecis).
