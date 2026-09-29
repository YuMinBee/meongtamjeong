# dino.txt · Talk2DINO 기준선 비교 결과

[사전 프로토콜](RELATED_BASELINES_PROTOCOL.md)에 따라 실행. 코드: `python -m experiments.dog_domain.related_baselines --stage all`.
값은 nDCG@10 × 100 (정확도 아님). 영어 템플릿 조건, 글 비중 0.20 고정, 학습 모델은 시드 17/29/43 질의별 평균.
기존 참조 3개(CLIP_mix, DINO_image, dino_linear_mix)는 저장 특징으로 재계산해 논문 수치와 일치함을 확인했다.

## 1. 속성 조건 검색

| 방법 | PetFinder | 한국 | 대만 |
|---|---:|---:|---:|
| CLIP B/32 사진+글 (기존 기준선) | 48.45 | 35.21 | 37.81 |
| DINOv3-B 사진만 | 55.38 | 40.63 | 45.29 |
| DINOv3-B + Linear 정렬 (논문 주 방법) | 58.22 | 43.14 | 47.50 |
| Talk2DINO 이미지만 (DINOv2-B reg, 어텐션 풀링) | 55.09 | 40.58 | 44.52 |
| Talk2DINO 백본 CLS만 | 56.29 | 40.52 | 44.60 |
| Talk2DINO 글만 | 51.39 | 30.64 | 35.57 |
| Talk2DINO 글만 (헤드 최대, 원 학습 유사도) | 56.40 | 33.04 | 31.30 |
| Talk2DINO 사진+글 (zero-shot) | 55.45 | 40.35 | 44.94 |
| Talk2DINO 구조 헤드를 우리 데이터로 재학습 (DINOv3-B) | 58.27 | 43.21 | 47.60 |
| ↳ 글만 | 66.07 | 35.93 | 50.62 |
| DINOv3-L 사진만 | 55.91 | 42.13 | 45.70 |
| dino.txt 사진만 | 52.88 | 39.37 | 45.72 |
| dino.txt 글만 | 55.03 | 43.84 | 16.31 |
| dino.txt 사진+글 (zero-shot) | 53.62 | 40.09 | 46.46 |
| DINOv3-L + Linear 정렬 (우리 방식) | 58.43 | 43.97 | 47.82 |
| ↳ 글만 | 68.31 | 37.62 | 56.69 |
| DINOv3-L + MLP 정렬 | 58.47 | 44.00 | 47.98 |

## 2. 사전 지정 비교 (차이 × 100, 보호소 단위 paired bootstrap 95% 구간)

| 비교 | PetFinder | 한국 | 대만 |
|---|---|---|---|
| dinoL_linear_mix minus dinotxt_mix | +4.81 [+3.94, +5.70] | +3.88 [+2.48, +5.14] | +1.36 [-0.96, +3.39] |
| dino_linear_mix minus t2d_mix | +2.77 [+1.95, +3.63] | +2.80 [+1.79, +3.84] | +2.56 [+1.77, +3.31] |
| dinotxt_mix minus dinotxt_image | +0.74 [+0.56, +0.92] | +0.72 [+0.46, +0.97] | +0.74 [+0.64, +0.86] |
| t2d_mix minus t2d_image | +0.36 [+0.03, +0.72] | -0.24 [-0.83, +0.32] | +0.42 [-0.07, +0.87] |
| DINOL_image minus dinotxt_image | +3.03 [+2.18, +3.84] | +2.76 [+1.33, +4.04] | -0.02 [-2.38, +2.00] |
| dinoL_linear_mix minus DINOL_image | +2.52 [+2.16, +2.88] | +1.84 [+1.45, +2.26] | +2.12 [+1.81, +2.40] |
| t2darch_mix minus t2d_mix | +2.81 [+2.03, +3.65] | +2.86 [+1.86, +3.90] | +2.66 [+1.87, +3.39] |
| dinoL_mlp_mix minus dinotxt_mix | +4.85 [+3.95, +5.75] | +3.91 [+2.53, +5.16] | +1.51 [-0.83, +3.54] |
| dinotxt_mix minus CLIP_mix | +5.17 [+4.09, +6.20] | +4.88 [+3.07, +6.47] | +8.65 [+7.29, +10.10] |
| t2d_mix minus CLIP_mix | +7.01 [+5.65, +8.34] | +5.14 [+3.21, +6.95] | +7.13 [+4.97, +9.12] |
| dinoL_linear_mix minus dino_linear_mix | +0.21 [-0.42, +0.84] | +0.83 [+0.05, +1.63] | +0.32 [-0.47, +1.05] |
| t2darch_mix minus dino_linear_mix | +0.04 [-0.07, +0.15] | +0.06 [-0.08, +0.20] | +0.10 [+0.01, +0.19] |

## 3. 한국: 같은 공고의 다른 사진 찾기 (R@1 / MRR × 100)

| 방법 | R@1 | MRR |
|---|---:|---:|
| DINO_image | 92.96 | 94.91 |
| DINOL_image | 93.94 | 95.89 |
| dinotxt_image | 90.18 | 92.72 |
| t2d_image | 85.76 | 89.57 |
| t2d_cls_image | 92.80 | 95.13 |
| dino_linear_mix | 92.64 | 94.78 |
| dinoL_linear_mix | 94.27 | 96.07 |
| dinotxt_mix | 90.18 | 92.74 |
| t2d_mix | 85.76 | 89.46 |

| 비교 (R@1 차이) | 한국 |
|---|---|
| DINOL_image minus dinotxt_image | +3.76 [+1.12, +6.60] |
| dinotxt_mix minus dinotxt_image | +0.00 [-0.51, +0.47] |
| t2d_mix minus t2d_image | +0.00 [-0.92, +0.91] |

## 해석상 주의

- dino.txt·Talk2DINO는 공개 가중치 그대로(zero-shot), 우리 헤드는 PetFinder로 학습했다. 방법 자체의 우열이 아니라 대규모 일반 정렬과 소규모 도메인 정렬의 비교다.
- 백본(DINOv2-B/DINOv3-B/DINOv3-L), 텍스트 인코더(CLIP B/16·B/32, dino.txt 자체), 입력 해상도(518/224)가 방법마다 다르다.
- 구간은 다중비교 보정 전 탐색적 구간이며 라벨 오류를 반영하지 않는다. 정답은 공고 기재 색·크기 기반 silver label이다.
