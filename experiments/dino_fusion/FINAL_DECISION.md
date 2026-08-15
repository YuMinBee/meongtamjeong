# CLIP + DINO 최종 프로젝트 결정

기준일: 2026-08-15 (Asia/Seoul)

## 최종 상태

현재 프로젝트의 권장 구성은 다음과 같다.

- 시각 인코더: frozen DINOv3 ViT-B/16
- 자연어 인코더: frozen OpenAI CLIP ViT-B/32
- 공유공간 정렬: 학습된 8-step flow head
- 혼합 비중: DINO 이미지 80%, 정렬 CLIP 외형 텍스트 20%
- 후보 이미지: Faster R-CNN dog crop
- 배포 모드: `shadow`
- 공고문 약라벨 행동 head: 기본 비활성화

`/search/appearance/image`에서는 사진과 관찰 가능한 외형 문장만 순위에
사용한다. 성격·생활 조건은 사진에서 추론하지 않고 후보를 선택한 뒤 보호소에
확인할 질문으로 바꾼다. 레거시 행동 head는 연구 재현을 위해 코드와 artifact
형식만 보존하며, `DINO_FUSION_BEHAVIOR_ENABLED=false`가 기본이다.

## 결정 근거

### 이미지 검색

- 내부 동일-crop 비교: CLIP Hit@1 `80.95%`, DINOv3 `90.48%`.
- PetFinder 기관 분리 test 77마리: CLIP Recall@10 `0.557`, DINOv3
  `0.877`.
- 추가 backbone/view ablation에서 DINOv2는 늦은 순위 Recall@10이 더
  높았지만, DINOv3가 MRR `0.7650` 대 `0.7176`으로 상위 순위가 더 좋았고
  외형 nDCG도 근소하게 높았다. 기존 flow도 DINOv3 공간에 맞춰져 있으므로
  backbone을 교체하지 않는다.

### 이미지 + 외형 텍스트

- 내부 20-query 비교: CLIP Hit@1 `85%`, DINOv3+flow `95%`.
- 내부 한국어 외형 nDCG@10: `0.4058`에서 `0.5375`.
- 외부 validation이 고른 텍스트 비중 0.25는 test에서 이미지 단독 대비 외형
  nDCG를 `+0.0087`만 개선해 사전 기준 `+0.02`를 통과하지 못했다.
- 현재 0.20과 0.25 차이는 nDCG `0.0016`에 불과하고 0.20의 MRR이 조금 더
  높아 런타임 비중을 바꾸지 않는다.

### 성격·생활 조건

- PetFinder 203개 test 질의에서 DINO 이미지 단독 행동 nDCG는 `0.533`,
  현재 DINOde 텍스트 경로는 `0.516`이었다.
- 학습한 CLIP 호환성 head의 multimodal macro ROC-AUC는 `0.548`이었고
  최종 retrieval gate를 통과하지 못했다.
- full-document sparse 진단은 macro AUC `0.740`으로 정보 자체는 있음을
  보였지만 nDCG 증가는 `+0.019`로 적용 기준 `+0.03`보다 낮았다.

따라서 성격 정보는 이미지 또는 품종에서 생성하지 않고, 구조화된 관찰 기록이
확보되기 전까지 검색 순위 신호로 승격하지 않는다.

## 운영 경계

| 설정 | 최종값 | 의미 |
|---|---|---|
| `DINO_FUSION_MODE` | 기본 `off`, 로컬 파일럿 `shadow` | CLIP 결과를 제공하면서 DINO 후보와 지연시간만 관찰 |
| `DINO_FUSION_TEXT_WEIGHT` | `0.20` | 검증되지 않은 0.25 변경을 적용하지 않음 |
| `DINO_FUSION_BEHAVIOR_ENABLED` | `false` | 실패한 약라벨 성격 브랜치를 서비스 순위에서 제외 |
| `DINO_FUSION_LOCAL_FILES_ONLY` | `true` | 요청 중 모델 다운로드 금지 |
| `DINO_FUSION_FAIL_OPEN` | `true` | DINO 오류 시 기존 CLIP 결과 유지 |

최신 분리 스냅샷은 활성 공고 1,212건, CLIP 벡터 3,522개, DINO crop 벡터
1,138개로 구성되며 원본 추적 artifact를 교체하지 않는다. `active` 전환은
별도 승인 전까지 하지 않는다.

## 지금 하지 않는 작업

- CLIP 또는 DINO foundation encoder 전체 fine-tuning
- 더 큰 DINO backbone으로 무조건 교체
- 현재 test 결과를 보고 추가 weight tuning
- 공고 문구 약라벨 성격 head 반복 학습
- 사진만으로 실제 성격·공격성·사회성을 추론

추가 행동 연구는 장문 공고와 실제 구조화된 관찰 라벨이 충분히 확보된 경우에만
새 split과 새 gate를 먼저 고정하고 재개한다.

## 재현 자료

- [`PILOT_RESULTS.md`](PILOT_RESULTS.md): 초기 CLIP/DINO 이미지 비교
- [`DINODE_FLOW_RESULTS.md`](DINODE_FLOW_RESULTS.md): 공유공간 정렬 결과
- [`REFRESH_20260815.md`](REFRESH_20260815.md): 활성 공고 1,212건 갱신
- [`../dino_fusion_external_eval/RESULTS.md`](../dino_fusion_external_eval/RESULTS.md): 외부 성격·호환성 평가
- [`../dino_fusion_external_eval/PERFORMANCE_ABLATION_RESULTS.md`](../dino_fusion_external_eval/PERFORMANCE_ABLATION_RESULTS.md): v2/v3·crop/full·weight 최종 ablation

브라우저 블라인드 평가의 투표 export는 현재 저장소에 포함되어 있지 않으므로,
이를 사람 평가 성능으로 주장하지 않는다. 자동 데이터셋 결과와 사용자가 확인한
소규모 탐색 경험은 구분해 기록한다.
