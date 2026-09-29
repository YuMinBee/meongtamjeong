# 강한 CLIP 계열 기준선 프로토콜

2026-09-29 작성. **결과를 보기 전에 고정한 계획이다.** 변경 시 하단 기록에 추가한다.

## 목적

기존 기준선 CLIP ViT-B/32(2021)는 작고 오래됐다. 최신·대형 이미지-텍스트 모델로 바꿔도
DINO 기반 도메인 정렬의 이점이 유지되는지 확인한다.

## 모델 (모두 공개 가중치, zero-shot, 추가 학습 없음)

| 이름 | 모델 | 비고 |
|---|---|---|
| `siglip2L` | `google/siglip2-so400m-patch14-384` | **주 기준선**(가장 강한 모델), 다국어 |
| `siglip2B` | `google/siglip2-base-patch16-224` | DINOv3-B와 비슷한 크기 |
| `clipL` | `laion/CLIP-ViT-L-14-laion2B-s32B-b82K` | 대형 CLIP |

각 모델은 자기 이미지·텍스트 공간에서 `M_image`, `M_text_only`, `M_mix`(정규화 사진 0.8 + 글 0.2)를
평가한다. 조건 문장은 기존과 같은 영어 템플릿을 그대로 쓴다(모델별 문장 조정 없음).
SigLIP 2 텍스트는 모델 카드 권장대로 `padding="max_length", max_length=64`로 토큰화한다.

## 평가

1. 속성 조건 검색: PetFinder·한국·대만, nDCG@10, 기존 정답·자기 제외 규칙 동일.
2. 같은 개 찾기: 한국(같은 공고 다른 사진, 갤러리 611) R@1, PetFinder(두 번째 사진 → 대표 사진,
   갤러리 1,618, 거의 같은 사진 1쌍 제외) R@1.

## 사전 지정 비교

- `dino_linear_mix − siglip2L_mix`, `dinoL_linear_mix − siglip2L_mix` (주)
- `dino_linear_mix − siglip2B_mix`, `dino_linear_mix − clipL_mix`
- 같은 개 찾기: `DINO_image − siglip2L_image`, `DINOL_image − siglip2L_image`
- 보호소/등록자 단위 paired cluster bootstrap 2,000회.

## 다중비교 보정

이번 문서와 기존 두 문서(RELATED_BASELINES, PETFINDER_IDENTITY)의 사전 지정 비교 전체에 대해,
문서별로 한 묶음(family)으로 Holm 보정을 적용한 결과를 함께 보고한다. p값은 paired bootstrap
분포에서 양측 `2·min(P(d≤0), P(d≥0))`로 계산한다(2,000회라 최소 약 0.001). 색 라벨 검증과
색 묶음 분석은 민감도 분석이라 보정 대상에서 제외한다.

## 해석 규칙

- 구간·보정 결과가 0을 포함하면 우열을 주장하지 않는다. 기준선이 이기면 그대로 보고한다.
- SigLIP 2는 해상도(384)와 학습 데이터 규모가 달라 순수한 구조 비교가 아니다.

## 변경 기록

(없음)
