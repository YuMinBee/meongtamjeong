# 사전 지정 비교의 다중비교 보정 (Holm)

[프로토콜](STRONG_CLIP_PROTOCOL.md#다중비교-보정). 문서별로 한 묶음. p는 보호소/등록자 단위 paired bootstrap 2,000회에서 계산한 양측 값이며 (k+1)/(B+1) 방식이라 최소값이 약 0.001이다. 차이는 × 100.

## dino.txt · Talk2DINO 기준선 (RELATED_BASELINES)

비교 39개. Holm 보정 후 5%에서 유의: 25개.

| 지표 | 비교 | 차이 | p | Holm p | 5% 유의 |
|---|---|---:|---:|---:|:---:|
| PetFinder nDCG@10 | dinoL_linear_mix − dinotxt_mix | +4.81 | 0.0010 | 0.0390 | ✓ |
| PetFinder nDCG@10 | dino_linear_mix − t2d_mix | +2.77 | 0.0010 | 0.0390 | ✓ |
| PetFinder nDCG@10 | dinotxt_mix − dinotxt_image | +0.74 | 0.0010 | 0.0390 | ✓ |
| PetFinder nDCG@10 | t2d_mix − t2d_image | +0.36 | 0.0250 | 0.3248 |  |
| PetFinder nDCG@10 | DINOL_image − dinotxt_image | +3.03 | 0.0010 | 0.0390 | ✓ |
| PetFinder nDCG@10 | dinoL_linear_mix − DINOL_image | +2.52 | 0.0010 | 0.0390 | ✓ |
| PetFinder nDCG@10 | t2darch_mix − t2d_mix | +2.81 | 0.0010 | 0.0390 | ✓ |
| PetFinder nDCG@10 | dinoL_mlp_mix − dinotxt_mix | +4.85 | 0.0010 | 0.0390 | ✓ |
| PetFinder nDCG@10 | dinotxt_mix − CLIP_mix | +5.17 | 0.0010 | 0.0390 | ✓ |
| PetFinder nDCG@10 | t2d_mix − CLIP_mix | +7.01 | 0.0010 | 0.0390 | ✓ |
| PetFinder nDCG@10 | dinoL_linear_mix − dino_linear_mix | +0.21 | 0.4928 | 1.0000 |  |
| PetFinder nDCG@10 | t2darch_mix − dino_linear_mix | +0.04 | 0.4608 | 1.0000 |  |
| Korea nDCG@10 | dinoL_linear_mix − dinotxt_mix | +3.88 | 0.0010 | 0.0390 | ✓ |
| Korea nDCG@10 | dino_linear_mix − t2d_mix | +2.80 | 0.0010 | 0.0390 | ✓ |
| Korea nDCG@10 | dinotxt_mix − dinotxt_image | +0.72 | 0.0010 | 0.0390 | ✓ |
| Korea nDCG@10 | t2d_mix − t2d_image | -0.24 | 0.4138 | 1.0000 |  |
| Korea nDCG@10 | DINOL_image − dinotxt_image | +2.76 | 0.0010 | 0.0390 | ✓ |
| Korea nDCG@10 | dinoL_linear_mix − DINOL_image | +1.84 | 0.0010 | 0.0390 | ✓ |
| Korea nDCG@10 | t2darch_mix − t2d_mix | +2.86 | 0.0010 | 0.0390 | ✓ |
| Korea nDCG@10 | dinoL_mlp_mix − dinotxt_mix | +3.91 | 0.0010 | 0.0390 | ✓ |
| Korea nDCG@10 | dinotxt_mix − CLIP_mix | +4.88 | 0.0010 | 0.0390 | ✓ |
| Korea nDCG@10 | t2d_mix − CLIP_mix | +5.14 | 0.0010 | 0.0390 | ✓ |
| Korea nDCG@10 | dinoL_linear_mix − dino_linear_mix | +0.83 | 0.0410 | 0.4918 |  |
| Korea nDCG@10 | t2darch_mix − dino_linear_mix | +0.06 | 0.3818 | 1.0000 |  |
| Korea R@1 | DINOL_image − dinotxt_image | +3.76 | 0.0020 | 0.0390 | ✓ |
| Korea R@1 | dinotxt_mix − dinotxt_image | +0.00 | 1.0000 | 1.0000 |  |
| Korea R@1 | t2d_mix − t2d_image | +0.00 | 1.0000 | 1.0000 |  |
| Taiwan nDCG@10 | dinoL_linear_mix − dinotxt_mix | +1.36 | 0.2359 | 1.0000 |  |
| Taiwan nDCG@10 | dino_linear_mix − t2d_mix | +2.56 | 0.0010 | 0.0390 | ✓ |
| Taiwan nDCG@10 | dinotxt_mix − dinotxt_image | +0.74 | 0.0010 | 0.0390 | ✓ |
| Taiwan nDCG@10 | t2d_mix − t2d_image | +0.42 | 0.0920 | 1.0000 |  |
| Taiwan nDCG@10 | DINOL_image − dinotxt_image | -0.02 | 0.9625 | 1.0000 |  |
| Taiwan nDCG@10 | dinoL_linear_mix − DINOL_image | +2.12 | 0.0010 | 0.0390 | ✓ |
| Taiwan nDCG@10 | t2darch_mix − t2d_mix | +2.66 | 0.0010 | 0.0390 | ✓ |
| Taiwan nDCG@10 | dinoL_mlp_mix − dinotxt_mix | +1.51 | 0.1889 | 1.0000 |  |
| Taiwan nDCG@10 | dinotxt_mix − CLIP_mix | +8.65 | 0.0010 | 0.0390 | ✓ |
| Taiwan nDCG@10 | t2d_mix − CLIP_mix | +7.13 | 0.0010 | 0.0390 | ✓ |
| Taiwan nDCG@10 | dinoL_linear_mix − dino_linear_mix | +0.32 | 0.4258 | 1.0000 |  |
| Taiwan nDCG@10 | t2darch_mix − dino_linear_mix | +0.10 | 0.0220 | 0.3078 |  |

## PetFinder 같은 개 찾기 (PETFINDER_IDENTITY)

비교 8개. Holm 보정 후 5%에서 유의: 4개.

| 지표 | 비교 | 차이 | p | Holm p | 5% 유의 |
|---|---|---:|---:|---:|:---:|
| PetFinder R@1 | DINO_image − CLIP_image | +26.76 | 0.0010 | 0.0080 | ✓ |
| PetFinder R@1 | dino_linear_mix − CLIP_mix | +25.86 | 0.0010 | 0.0080 | ✓ |
| PetFinder R@1 | dino_linear_mix − DINO_image | -0.09 | 0.8566 | 1.0000 |  |
| PetFinder R@1 | DINOL_image − dinotxt_image | +9.28 | 0.0010 | 0.0080 | ✓ |
| PetFinder R@1 | dinoL_linear_mix − dinotxt_mix | +8.68 | 0.0010 | 0.0080 | ✓ |
| PetFinder R@1 | dinoL_linear_mix − DINOL_image | -0.60 | 0.1769 | 0.7076 |  |
| PetFinder R@1 | dinotxt_mix − dinotxt_image | +0.00 | 1.0000 | 1.0000 |  |
| PetFinder R@1 | t2d_mix − t2d_image | -0.72 | 0.3448 | 1.0000 |  |

## 강한 CLIP 계열 기준선 (STRONG_CLIP)

비교 16개. Holm 보정 후 5%에서 유의: 14개.

| 지표 | 비교 | 차이 | p | Holm p | 5% 유의 |
|---|---|---:|---:|---:|:---:|
| PetFinder nDCG@10 | dino_linear_mix − siglip2L_mix | +5.86 | 0.0010 | 0.0160 | ✓ |
| PetFinder nDCG@10 | dinoL_linear_mix − siglip2L_mix | +6.07 | 0.0010 | 0.0160 | ✓ |
| PetFinder nDCG@10 | dino_linear_mix − siglip2B_mix | +6.32 | 0.0010 | 0.0160 | ✓ |
| PetFinder nDCG@10 | dino_linear_mix − clipL_mix | +5.06 | 0.0010 | 0.0160 | ✓ |
| Korea nDCG@10 | dino_linear_mix − siglip2L_mix | +4.68 | 0.0010 | 0.0160 | ✓ |
| Korea nDCG@10 | dinoL_linear_mix − siglip2L_mix | +5.50 | 0.0010 | 0.0160 | ✓ |
| Korea nDCG@10 | dino_linear_mix − siglip2B_mix | +6.37 | 0.0010 | 0.0160 | ✓ |
| Korea nDCG@10 | dino_linear_mix − clipL_mix | +5.61 | 0.0010 | 0.0160 | ✓ |
| Korea R@1 | DINO_image − siglip2L_image | +8.51 | 0.0010 | 0.0160 | ✓ |
| Korea R@1 | DINOL_image − siglip2L_image | +9.49 | 0.0010 | 0.0160 | ✓ |
| Taiwan nDCG@10 | dino_linear_mix − siglip2L_mix | +1.52 | 0.1569 | 0.2619 |  |
| Taiwan nDCG@10 | dinoL_linear_mix − siglip2L_mix | +1.83 | 0.1309 | 0.2619 |  |
| Taiwan nDCG@10 | dino_linear_mix − siglip2B_mix | +2.46 | 0.0040 | 0.0160 | ✓ |
| Taiwan nDCG@10 | dino_linear_mix − clipL_mix | +2.79 | 0.0020 | 0.0160 | ✓ |
| PetFinder R@1 | DINO_image − siglip2L_image | +7.84 | 0.0010 | 0.0160 | ✓ |
| PetFinder R@1 | DINOL_image − siglip2L_image | +14.59 | 0.0010 | 0.0160 | ✓ |
