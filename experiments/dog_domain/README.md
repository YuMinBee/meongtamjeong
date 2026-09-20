# 강아지 도메인 검색 실험

## 현재 파이프라인

공개 공고의 사진과 자연어 속성 조건을 결합하는 검색을 평가합니다.
CLIP 텍스트 특징을 DINO 시각 공간으로 정렬하는 Linear·MLP 비교와
기존 Flow 실험은 별도 실행·기록으로 구분합니다. 연구 코드를 실행해도 서비스의 기본 모델이 자동으로 바뀌지 않습니다.

| 작업 | 진입점 |
|---|---|
| PetFinder 데이터 확인 | `python -m experiments.dog_domain.inspect_petfinder` |
| 전체 공고 정보로 학습 | `python -m experiments.dog_domain.petfinder_expanded` |
| 확장 속성·외부 공고 평가 | `python -m experiments.dog_domain.petfinder_expanded_eval` |
| 학습 산출물 검증 | `python -m experiments.dog_domain.verify_petfinder_training` |
| 사람 평가 서버 | `python -m experiments.dog_domain.visual_study_server --data PATH` |
| DINO 사진 단독 추가 쌍 준비 | `python -m experiments.dog_domain.prepare_dino_visual_supplement` |
| 추가 평가 완료 후 병합 | `python -m experiments.dog_domain.merge_dino_visual_supplement` |

실험은 저장소 루트에서 실행합니다. CUDA 학습에는 `experiments/dino_fusion/requirements.txt`의
추가 의존성 및 접근 가능한 모델 캐시가 필요합니다. Kaggle 등 데이터는 원 배포처의 이용 조건에 따라 별도로 확보해야 합니다.
여러 스크립트는 기존 연구 환경의 `D:/meongtamjeong_research/` 및 `tmp/` 캐시를 기본값으로 사용하므로,
다른 환경에서는 모듈 상단의 경로 설정을 먼저 맞추세요. 원본 데이터와 체크포인트는 배포하지 않습니다.

[평가 도구 사용법](VISUAL_EVALUATION.md) · [PetFinder 확장 프로토콜](PETFINDER_EXPANDED_PROTOCOL_KO.md)

## 초기 파일럿 기록

아래 내용은 2026-09-16에 수행한 초기 MPDD·DogFaceNet 실험 기록입니다.
이후 학습 정렬·자연어 조건 검색 및 사람 평가와는 목적이 다릅니다.

### MPDD·DogFaceNet

2026-09-16 완료. PetFace 사용 없음. 추가 사람 평가 없음. 앱 설정 변경 없음.

## 추가 검증 완료: 98%와 97%의 차이가 유지되는가?

기존 MPDD 104개 질문에서 결합만 맞힌 것은 **1개**, DINO만 맞힌 것은
0개였다. 개체 단위 정확한 양측 부호검정 p=1.0으로, 이 1개만으로
일반적인 결합 우위를 주장할 수 없다.

가중치를 CLIP 0.25 / DINO 0.75로 고정하고 새 검증을 진행했다.
아래 수치는 개체별 평균 Recall@1이며, 각 행은 후보 구성이 다르므로
원래 MPDD 표와 절대 점수를 직접 비교하면 안 된다.

| 추가 검증 | CLIP | DINO | 고정 결합 | 결합−DINO, 95% CI |
|---|---:|---:|---:|---|
| DogFaceNet 원래 test 139마리, 개체당 후보 1장 | 49.80% | 90.33% | 89.26% | −1.06%p [−2.96, +0.43] |
| 같은 139마리, 후보 사진 20회 변경 후 평균 | 49.60% | 90.66% | 89.80% | −0.86%p [−1.45, −0.33] |
| DogFaceNet 전체 1,393마리, 보조 분석 | 34.28% | 76.30% | 75.57% | −0.73%p [−1.18, −0.31] |
| MPDD 기존 test 96마리, 후보 사진 100회 변경 후 평균 | 60.49% | 89.38% | 89.47% | +0.09%p [−0.35, +0.54] |

새 데이터의 주 평가(첫 행)에서는 결합이 더 높지 않았다. 사전에 고정한
주 비교의 sign-flip p=0.28441이며, 나머지 행은 보조·민감도 분석이다.
반복 횟수를 독립적인 표본 수로 취급하지 않고 각 개체 안에서 평균을 낸
뒤 개체 단위 신뢰구간을 계산했다. DogFaceNet 전체 분석은 test 개체도
포함하므로 별개의 세 번째 데이터셋으로 세지 않는다.

**현재 결론:** DINO의 시각 검색 개선은 새 강아지 데이터에서도 확인됐지만,
고정된 CLIP 이미지 점수 결합의 추가 이점은 재현되지 않았다. 이는 이미지
점수 결합에 관한 결과이며, 사진+텍스트 Flow의 효과를 부정하거나 입증하지 않는다.

DogFaceNet은 미리 정렬된 얼굴 사진이다. 본 실행은 공개 test 개체 목록을
이용한 자체 검색 규칙이며 원 논문의 verification 실험을 그대로 재현한 것이 아니다.
원본 8,363장 모두 디코딩에 성공했고 정확한 픽셀 중복 및 MPDD와의 정확한
이미지 중복은 없었다. 유사한 연속 촬영이나 사전학습 중복까지 배제한 것은 아니다.

전체 결과: [FOLLOWUP_RESULTS.md](FOLLOWUP_RESULTS.md).
사전 계획: [FOLLOWUP_PROTOCOL.md](FOLLOWUP_PROTOCOL.md).
추가 테스트 포함 6개 통과, 별도 CPU 전체 정렬로 90개 쿼리·방법 조합을
대조해 모두 일치했다. 원본 실험 결과와 가중치는 변경하지 않았다.

```powershell
python -m experiments.dog_domain.followup
python -m experiments.dog_domain.verify_followup
```

재실행에는 `tmp/dog_retrieval/`의 공개 DogFaceNet ZIP 및 클래스 목록이 필요하다.
출처: [Guillaume Mougeot, The DogFaceNet datasets](https://zenodo.org/records/12578449).

## 1. 새 외부 이미지 평가: MPDD

공식 배포 ZIP의 분할을 유지했다. 검증 111장/95개체를 학습 폴더 921장과
검색해 결합 가중치를 선택한 뒤, 별개 96개체의 질문 104장과 후보 521장으로
평가했다. 학습 폴더는 검증용 후보로만 쓰며, 이번에는 모델을 학습하지 않았다.
다운로드 SHA-256과 전체 이미지 디코딩을 확인했다. 픽셀 단위 동일 이미지
중복은 없었다. 배포 설명의 192마리와 실제 파일명 ID 191개의 차이를 기록했다.

| 방법 | 개체별 평균 Recall@1 | mAP |
|---|---:|---:|
| CLIP ViT-B/32 | 75.52% | 55.65% |
| DINOv3 ViT-B/16 | 96.88% | 88.76% |
| CLIP 이미지 점수 0.25 + DINO 이미지 점수 0.75 | 97.92% | 88.94% |

DINO−CLIP의 Recall@1 차이는 +21.35%p, 개체 단위 paired 95% CI는
[+12.50, +30.73]%p다. 반면 선택된 결합−DINO의 mAP 차이는 +0.18%p,
CI [−0.67, +1.13]%p라서 두 인코더를 결합하는 추가 이점은 확실하지 않다.

블러·저해상도 질문 및 파일명 c-code 제외 조건도 사전에 정해 평가했다.
이는 인공 열화와 파일명 기반 민감도 분석이며, 실제 촬영 환경/카메라
일반화를 입증한 것은 아니다. 모든 조건에서 평가 가능한 질문은 104장이었다.
CLIP과 DINO는 같은 원본을 입력받지만 각자의 기본 전처리를 사용했다.

**이 실험에는 텍스트가 없으므로 Flow의 효과를 입증하지 않는다.**
전체 지표·신뢰구간: [RESULTS.md](RESULTS.md).
사전 계획: [PROTOCOL.md](PROTOCOL.md).

## 2. 기존 공개 공고 기반 사진+텍스트 추가 진단

기존 PetFinder 평가 자료의 77마리 중 색상 정보가 있는 51마리를 질문으로
사용했다. 질문은 사진 1 + 구조화된 색상으로 만든 한국어 설명, 후보는 사진 2다.
기존 강아지 공고로 학습한 Flow 체크포인트를 그대로 사용했다.
모든 텍스트 비중은 기존 프로젝트의 0.20으로 고정했고 재학습·튜닝하지 않았다.

이 자료는 이미 이전 실험에서 본 데이터다. **추가 진단이며 새로운 독립
검증 결과가 아니다.** 이번 결과를 새 holdout이라고 쓰지 않는다.

| 방법 | 같은 강아지 Recall@1 | 색상 nDCG@10 |
|---|---:|---:|
| CLIP 이미지+텍스트 | 41.18% | 22.55% |
| DINO 이미지 단독 | 72.55% | 37.04% |
| DINO 이미지+CLIP 텍스트 점수 결합 | 68.63% | 36.49% |
| DINO 이미지+기존 Flow 정렬 텍스트 | 68.63% | 39.67% |
| DINO 이미지+다른 질문에서 가져온 Flow 텍스트 | 66.67% | 36.69% |

Flow의 올바른 텍스트는 DINO 단독 대비 색상 nDCG +2.63%p
(기관 단위 paired 95% CI [+1.13, +4.24]%p), 이동시킨 텍스트 대비 +2.98%p
([+0.52, +5.33]%p)였다. 그러나 같은 개체 Recall@1은 DINO 단독 대비
−3.92%p였다. 색상 조건과 개체 식별 목표 사이의 차이를 숨기면 안 된다.
여러 비교에 대한 보정은 하지 않은 탐색적 구간이다.

색상 nDCG는 알려진 기본 색상과의 일치에만 근거한다. 자기 자신과 색상
미상 후보는 제외했다. 모든 자연어 의미를 이해했다는 지표가 아니다.
이동된 설명의 7.84%는 우연히 원래 색상과 같았으며, 이를 모두 틀린
설명이라고 취급하지 않았다.

전체 진단: [TEXT_DIAGNOSTIC.md](TEXT_DIAGNOSTIC.md).
고정한 규칙: [TEXT_DIAGNOSTIC_PROTOCOL.md](TEXT_DIAGNOSTIC_PROTOCOL.md).

## 논문 방향

현재 가장 타당한 방향은 **강아지 검색에서 DINO 시각 표현의 효과와
CLIP 텍스트 정렬의 역할을 분리해 평가하는 응용·분석 연구**다.
MPDD에서 기존 CLIP 대비 시각 개선은 확인됐다. 기존 공고 기반 진단에서는
Flow가 색상 조건 반영에 도움이 되는 신호가 있지만 독립 재검증이 필요하다.
성능 개선을 Flow 하나의 효과로 묶거나 모든 검색 목적에 우월하다고 쓰지 않는다.

PetFace 승인 후에는 아직 보지 않은 개체 분할에서 같은 사진+속성 실험을
고정한 설정으로 검증할 수 있다. 승인 전에도 이번 자료로 데이터/방법/실험
절을 정리할 수 있지만, 현재 단계에서 멀티모달 우월성이나 게재를 보장하지 않는다.

## 재현

저장소 루트에서 `dog-rag` 환경 사용:

```powershell
python -m experiments.dog_domain.mpdd
python -m experiments.dog_domain.text_diagnostic
python -m pytest tests/test_dog_domain.py -q
```

MPDD 특징 캐시는 원본·인코더·계획·실험 코드 해시가 일치할 때 재사용한다.
기존 PetFinder 이미지 특징과 원본 Flow 체크포인트는 로컬에 있어야 한다.
원본 데이터와 개별 점수/모델은 Git에서 제외하고 집계 보고서만 제공한다.
추가 테스트 3개 통과, Ruff 통과. 별도 argmax 계산으로 MPDD DINO의
query-micro Recall@1이 저장 결과(97.1154%)와 일치함을 확인했다.
표의 96.875%는 질문 수가 많은 개체에 치우치지 않는 identity-macro 값이다.

출처: Zhimin He (2023), [Multi-pose dog dataset v1](https://data.mendeley.com/datasets/v5j6m8dzhv/1),
DOI 10.17632/v5j6m8dzhv.1, CC BY 4.0.
