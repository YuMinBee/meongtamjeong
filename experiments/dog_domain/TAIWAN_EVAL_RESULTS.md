# Taiwan translated-text evaluation

Gallery: 5427 photos. Reference photo excluded. Frozen heads; text weight .20.
Relevance: exact normalized color set AND Taiwan size category. Scores below are nDCG@10 x 100, not accuracy.

| Method | English template | Korean template | Chinese prose | NLLB English prose | Qwen English prose |
|---|---:|---:|---:|---:|---:|
| CLIP_image | 34.72 | 34.72 | 34.98 | 34.98 | 34.98 |
| DINO_image | 45.29 | 45.29 | 47.45 | 47.45 | 47.45 |
| CLIP_mix | 37.81 | 34.77 | 35.17 | 34.89 | 34.71 |
| late_mix | 45.54 | 45.16 | 47.36 | 47.50 | 47.39 |
| CLIP_shift | 34.79 | 34.85 | 35.17 | 34.96 | 34.79 |
| CLIP_text_only | 37.11 | 12.28 | 4.47 | 9.88 | 10.20 |
| flow_mix | 46.46 | 45.32 | 46.97 | 46.91 | 47.29 |
| linear_mix | 46.66 | 45.24 | 46.87 | 46.84 | 47.17 |
| mlp_mix | 46.88 | 45.16 | 46.72 | 46.88 | 47.23 |
| flow_shift | 44.81 | 44.67 | 46.95 | 46.88 | 47.14 |
| flow_text_only | 29.50 | 35.14 | 5.74 | 6.44 | 5.24 |
| linear_text_only | 28.85 | 24.51 | 5.92 | 7.33 | 5.73 |
| mlp_text_only | 41.14 | 28.66 | 9.82 | 9.64 | 6.65 |

## Paired differences

| Condition / Flow comparison | nDCG difference [95% CI], points |
|---|---|
| template_english: flow_mix minus CLIP_mix | +8.65 [+6.99, +10.31] |
| template_english: flow_mix minus DINO_image | +1.17 [+0.94, +1.42] |
| template_english: flow_mix minus late_mix | +0.93 [+0.62, +1.24] |
| template_english: flow_mix minus linear_mix | -0.20 [-0.40, +0.01] |
| template_english: flow_mix minus mlp_mix | -0.42 [-0.62, -0.23] |
| template_english: flow_mix minus flow_shift | +1.65 [+1.40, +1.90] |
| template_korean: flow_mix minus CLIP_mix | +10.55 [+9.08, +11.98] |
| template_korean: flow_mix minus DINO_image | +0.03 [-0.21, +0.29] |
| template_korean: flow_mix minus late_mix | +0.16 [-0.09, +0.35] |
| template_korean: flow_mix minus linear_mix | +0.08 [-0.09, +0.25] |
| template_korean: flow_mix minus mlp_mix | +0.16 [-0.05, +0.36] |
| template_korean: flow_mix minus flow_shift | +0.65 [+0.40, +0.88] |
| prose_chinese: flow_mix minus CLIP_mix | +11.80 [+10.10, +13.14] |
| prose_chinese: flow_mix minus DINO_image | -0.48 [-0.73, -0.08] |
| prose_chinese: flow_mix minus late_mix | -0.40 [-0.69, -0.09] |
| prose_chinese: flow_mix minus linear_mix | +0.10 [-0.12, +0.45] |
| prose_chinese: flow_mix minus mlp_mix | +0.25 [-0.07, +0.69] |
| prose_chinese: flow_mix minus flow_shift | +0.02 [-0.14, +0.18] |
| prose_english: flow_mix minus CLIP_mix | +12.02 [+10.35, +13.29] |
| prose_english: flow_mix minus DINO_image | -0.54 [-0.81, -0.18] |
| prose_english: flow_mix minus late_mix | -0.59 [-0.93, -0.26] |
| prose_english: flow_mix minus linear_mix | +0.08 [-0.14, +0.37] |
| prose_english: flow_mix minus mlp_mix | +0.04 [-0.38, +0.38] |
| prose_english: flow_mix minus flow_shift | +0.04 [-0.42, +0.36] |
| prose_english_qwen: flow_mix minus CLIP_mix | +12.59 [+10.88, +13.84] |
| prose_english_qwen: flow_mix minus DINO_image | -0.16 [-0.43, +0.16] |
| prose_english_qwen: flow_mix minus late_mix | -0.09 [-0.42, +0.28] |
| prose_english_qwen: flow_mix minus linear_mix | +0.12 [-0.20, +0.50] |
| prose_english_qwen: flow_mix minus mlp_mix | +0.07 [-0.40, +0.50] |
| prose_english_qwen: flow_mix minus flow_shift | +0.15 [-0.28, +0.56] |

## Coverage

- template_english: 5427 queries; 5422 with another relevant candidate; 0 CLIP-truncated texts.
- template_korean: 5427 queries; 5422 with another relevant candidate; 0 CLIP-truncated texts.
- prose_chinese: 1824 queries; 1824 with another relevant candidate; 718 CLIP-truncated texts.
- prose_english: 1824 queries; 1824 with another relevant candidate; 97 CLIP-truncated texts.
- prose_english_qwen: 1824 queries; 1824 with another relevant candidate; 61 CLIP-truncated texts.

Shelter-cluster bootstrap intervals (2,000 samples); no multiplicity correction for exploratory contrasts.
Templates are dictionary-rendered attributes. English prose uses automatic NLLB and Qwen translations; neither is human validated.
Prose is scored only against appearance proxy labels, not full-description relevance. No same-animal accuracy claim.
Taiwan size labels differ from Korean weight bins. All original heads remain frozen; no Taiwan training or tuning.
