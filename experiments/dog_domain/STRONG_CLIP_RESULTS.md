# 강한 CLIP 계열 기준선 결과

[사전 프로토콜](STRONG_CLIP_PROTOCOL.md). 모든 기준선은 공개 가중치 zero-shot, 사진 0.8 + 글 0.2.

## 1. 속성 조건 검색 (nDCG@10 × 100)

| 방법 | PetFinder | 한국 | 대만 |
|---|---:|---:|---:|
| CLIP B/32 사진+글 (기존 기준선) | 48.45 | 35.21 | 37.81 |
| CLIP L/14 (LAION) 사진만 | 51.24 | 35.74 | 41.77 |
| CLIP L/14 (LAION) 사진+글 | 53.16 | 37.53 | 44.72 |
| SigLIP 2 B/16 사진만 | 50.35 | 35.42 | 42.71 |
| SigLIP 2 B/16 사진+글 | 51.91 | 36.77 | 45.04 |
| SigLIP 2 So400m 사진만 | 50.01 | 35.98 | 42.63 |
| SigLIP 2 So400m 글만 | 57.65 | 45.92 | 58.20 |
| SigLIP 2 So400m 사진+글 | 52.36 | 38.47 | 45.98 |
| DINOv3-B 사진만 | 55.38 | 40.63 | 45.29 |
| DINOv3-B + Linear (논문 주 방법) | 58.22 | 43.14 | 47.50 |
| DINOv3-L + Linear | 58.43 | 43.97 | 47.82 |

| 비교 (차이 × 100, 95% 구간) | PetFinder | 한국 | 대만 |
|---|---|---|---|
| dino_linear_mix minus siglip2L_mix | +5.86 [+4.59, +7.06] | +4.68 [+2.58, +6.47] | +1.52 [-0.61, +3.57] |
| dinoL_linear_mix minus siglip2L_mix | +6.07 [+4.93, +7.16] | +5.50 [+3.59, +7.12] | +1.83 [-0.61, +3.95] |
| dino_linear_mix minus siglip2B_mix | +6.32 [+5.02, +7.52] | +6.37 [+4.97, +7.82] | +2.46 [+0.87, +4.25] |
| dino_linear_mix minus clipL_mix | +5.06 [+3.70, +6.28] | +5.61 [+3.83, +7.32] | +2.79 [+1.23, +4.32] |

## 2. 같은 개 찾기 (R@1 × 100)

| 방법 | 한국 | PetFinder |
|---|---:|---:|
| CLIP_image | – | 43.69 |
| clipL_image | 84.29 | 54.50 |
| siglip2B_image | 81.83 | 58.20 |
| siglip2L_image | 84.45 | 62.61 |
| dinotxt_image | 90.18 | 67.93 |
| DINO_image | 92.96 | 70.45 |
| DINOL_image | 93.94 | 77.21 |

| 비교 (R@1 차이 × 100) | 한국 | PetFinder |
|---|---|---|
| DINO_image minus siglip2L_image | +8.51 [+5.22, +11.46] | +7.84 [+4.59, +10.80] |
| DINOL_image minus siglip2L_image | +9.49 [+5.70, +12.73] | +14.59 [+11.63, +17.77] |

한국 CLIP B/32 사진만(CLIP_image)은 기존 한국 평가 파일에 없어 비워 둔다.

## 해석상 주의

- 기준선은 모두 zero-shot이고 우리 헤드는 PetFinder로 학습했다.
- SigLIP 2 So400m은 입력 384px, 학습 데이터·모델 크기가 달라 순수한 구조 비교가 아니다.
- 구간은 다중비교 보정 전이며, 보정 결과는 MULTIPLICITY_RESULTS.md에 있다.
