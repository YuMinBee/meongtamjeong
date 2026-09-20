# Existing dog-corpus text diagnostic

**Exploratory reuse of an already-inspected test corpus; not a new independent validation.**

51 color-known query dogs, 77 full-photo gallery dogs; text weight fixed at .20.
Original dog-trained Flow; no training or tuning. Query photo and target photo are distinct archive members.
Structured color is proxy relevance. Unknown-color candidates and the query dog are excluded from color nDCG.
Shifted texts retained the original color for 7.84% of queries.

| Method | R@1 | R@5 | R@10 | MRR | Color nDCG@10 |
|---|---:|---:|---:|---:|---:|
| CLIP_image | 47.06 | 56.86 | 62.75 | 53.34 | 21.96 |
| DINO_image | 72.55 | 86.27 | 90.20 | 79.73 | 37.04 |
| CLIP_text_0.2 | 41.18 | 58.82 | 64.71 | 50.34 | 22.55 |
| late_text_0.2 | 68.63 | 86.27 | 90.20 | 78.06 | 36.49 |
| Flow_text_0.2 | 68.63 | 90.20 | 90.20 | 77.83 | 39.67 |
| CLIP_shifted_text | 41.18 | 58.82 | 64.71 | 50.59 | 22.37 |
| Flow_shifted_text | 66.67 | 88.24 | 88.24 | 75.47 | 36.69 |

## Paired differences

Percentage points, organization-cluster bootstrap 95% intervals; no multiplicity correction.

| Contrast | R@1 delta [CI] | Color nDCG delta [CI] |
|---|---|---|
| Flow_text_0.2 minus DINO_image | -3.92 [-9.52, +0.00] | +2.63 [+1.13, +4.24] |
| Flow_text_0.2 minus CLIP_text_0.2 | +27.45 [+15.62, +39.02] | +17.12 [+11.84, +22.08] |
| Flow_text_0.2 minus late_text_0.2 | +0.00 [-5.00, +5.66] | +3.18 [+1.52, +4.93] |
| Flow_text_0.2 minus Flow_shifted_text | +1.96 [-4.26, +9.09] | +2.98 [+0.52, +5.33] |
| CLIP_text_0.2 minus CLIP_shifted_text | +0.00 [-6.38, +5.45] | +0.18 [-0.38, +1.03] |

This diagnostic cannot establish free-form Korean understanding, temperament, deployment benefit, or novel architecture superiority.
Original photo data has no stated redistribution license in the local dataset card; no raw photos are published here.
