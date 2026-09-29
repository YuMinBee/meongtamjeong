# dino.txt · Talk2DINO 기준선 비교 프로토콜

2026-09-28 작성. **결과를 보기 전에 고정한 계획이다.** 실행 후 이 문서를 수정하지 않으며,
변경이 필요하면 하단 "변경 기록"에 이유와 시점을 추가한다.

## 목적

심사에서 예상되는 두 질문에 답한다.

1. DINOv3에는 공식 텍스트 정렬 모델(dino.txt)이 있다. 도메인 데이터로 학습한 작은
   CLIP→DINO 매핑이 이것보다 나은가, 적어도 비슷한가?
2. Talk2DINO는 고정된 CLIP 텍스트를 고정된 DINOv2 공간으로 매핑한다. 구조가 비슷한
   공개 모델을 그대로 가져와도 되는가, 도메인 학습이 필요한가?

## 고정 사항 (기존 논문과 동일)

- 데이터: PetFinder test 1,618 / 한국 611(서로 다른 사진 쌍) / 대만 5,427.
- 조건 문장: 영어 템플릿. PetFinder `Find a {size} dog whose coat is {colors}.`,
  한국 `evaluation_prompts(...)['english']`, 대만 `taiwan_eval.prompt(row, 'english')`.
- 정답: 한국·PetFinder는 색 하나 이상 겹침 AND 크기 일치, 대만은 색 집합 완전 일치 AND 체형 일치.
  자기 공고는 후보에서 제외.
- 지표: nDCG@10(주), P@10. 한국은 같은 공고 다른 사진 찾기 R@1/R@5/R@10/MRR도 보고.
- 결합: 각 모델의 **자기 공간 안에서** 정규화된 이미지 0.8 + 텍스트 0.2, 다시 정규화 후 코사인.
  어떤 기준선에도 가중치 탐색을 하지 않는다.
- 구간: 보호소/등록자 단위 paired cluster bootstrap 2,000회, 다중비교 보정 없음(탐색적).
- 학습 모델은 시드 17/29/43의 질의별 점수 평균.

## 비교 방법

| 이름 | 이미지 | 텍스트 | 학습 |
|---|---|---|---|
| `dinotxt_*` | dino.txt 비전 타워(DINOv3 ViT-L/16 + 헤드 블록, 2048-D) | dino.txt 텍스트 타워 | 공개 가중치 그대로 (zero-shot) |
| `DINOL_image` | DINOv3 ViT-L/16 원본 pooler 출력 (1024-D) | – | – |
| `dinoL_linear_*`, `dinoL_mlp_*` | DINOv3 ViT-L/16 원본 | CLIP ViT-B/32 → 학습 매핑 | 기존 v2(full-listing) 학습 절차를 ViT-L 특징으로 동일 반복 |
| `t2d_*` | DINOv2 ViT-B/14-reg, 평균 어텐션 가중 패치 풀링 (768-D) | CLIP ViT-B/16 → Talk2DINO 공개 투영 | 공개 가중치 그대로 (zero-shot) |
| `t2darch_*` | DINOv3 ViT-B/16 원본 (기존과 동일) | CLIP ViT-B/32 → Talk2DINO 구조 헤드(Linear-tanh-Linear, 768 은닉) | 기존 v2 학습 절차로 재학습 |
| 기존 참조 | CLIP_mix, DINO_image, dino_linear_mix(B) | | 저장된 특징·체크포인트로 재계산, 논문 수치와 일치 확인 |

Talk2DINO는 원래 분할용이라 전역 이미지 벡터가 없다. 공개 코드의 `avg_self_attn_token`
변형(마지막 블록 CLS→패치 어텐션을 헤드 평균 후 softmax, 패치 가중 평균)을 주 이미지 벡터로
정한다. Talk2DINO 학습 시 유사도(헤드별 풀링 벡터와의 최대 코사인)를 쓰는 텍스트 단독 검색
`t2d_text_maxhead`는 보조로 보고한다.

## 주 비교 (사전 지정)

1. `dinoL_linear_mix − dinotxt_mix` (같은 ViT-L 백본 계열)
2. `dino_linear_mix − t2d_mix` (ViT-B 계열, 기존 논문 모델 vs Talk2DINO zero-shot)
3. `dinotxt_mix − dinotxt_image`, `t2d_mix − t2d_image` (공개 정렬 텍스트의 추가 효과)
4. `DINOL_image − dinotxt_image` 및 한국 같은 개체 R@1 비교 (텍스트 정렬을 위해 이미지 공간을
   바꾸는 것이 시각 검색에 미치는 영향)

보조: `dinoL_linear_mix − DINOL_image`, `t2darch_mix − t2d_mix`(도메인 학습 효과),
`*_text_only`, `dinoL_mlp_mix`, 각 방법 − `CLIP_mix`.

## 해석 규칙

- 구간이 0을 포함하면 "차이를 확인하지 못함"으로 쓰고 우열을 주장하지 않는다.
- 결과가 우리 방법에 불리해도 본문 표에 모두 싣는다.
- dino.txt·Talk2DINO는 zero-shot이고 우리 헤드는 PetFinder로 학습했다. 차이는
  "대규모 일반 정렬 vs 소규모 도메인 정렬"의 비교이며 방법 자체의 우열로 일반화하지 않는다.
- 백본 크기·텍스트 인코더·입력 해상도가 방법마다 다르며, 이를 표에 명시한다.

## 가중치 출처

- dino.txt: 공식 파일명 `dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth`,
  백본 `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`. Meta 배포 경로 대신 HuggingFace 미러에서
  받고 SHA-256 앞 8자리가 공식 파일명 해시와 일치하는지 확인한다. 코드는
  facebookresearch/dinov3 커밋 `6876159a11b4df116f30f667f8c9888617df0751`.
- Talk2DINO: `lorebianchi98/Talk2DINO-ViTB` 리비전 `d120439255ae423ad0a3f4a13896bb74ae48792f`의
  `model.safetensors`(DINOv2·CLIP·투영 가중치 포함).

## 변경 기록

- 2026-09-28, 결과 확인 전: 4090 PC의 전원 문제로 실행 장비를 RTX 3090(z490, Linux)으로 옮겼다.
  기존 캐시·체크포인트는 그대로 쓰고 새 특징·헤드만 3090에서 만든다. 학습 재현 검사는
  "검증 손실 차이 < 0.02"로 정했다(실측 차이 2.4e-7, 최적 epoch 동일).
- 2026-09-28, 결과 확인 전: 기존 참조 재계산에서 대만 CLIP_mix가 저장값과 1.2e-5(0.001점) 달랐다
  (GPU 행렬곱에 따른 근접 동점 순서 차이). 유사도 계산을 기존 코드와 같은 CPU float32로 바꾸고,
  허용 오차를 1e-4(0.01점)로 두며 실제 차이를 결과 파일에 기록한다. 새 방법 점수는 이 시점까지
  열람하지 않았다(실패한 실행의 로그에 일부 출력됐을 수 있으나 확인하지 않음).
