# Held-out 2번째 사진 검색 평가

> 같은 공고의 인덱스에 사용하지 않은 2번째 사진을 질의로 삼아, 1번째 사진·crop 시각 벡터에서 같은 공고를 찾는 평가입니다. 입양 적합성·성격·건강을 평가하지 않습니다.

- 기준일: `2026-07-26`
- 실행 상태: `completed`
- 활성 공고: 1516건
- 시각 인덱스 공고: 1228건
- 평가 가능 URL 쌍: 1228건
- 고정 표본: 120건
- 시각 인덱스 커버리지: 81.00%
- 표본 SHA-256: `62b8f34190bb543f9e146373c62511e3d78f0780580874b842bc240301b57ba8`

## 입력 고정 정보

| 입력 | 경로 | SHA-256 |
|---|---|---|
| index | `data/dog_faiss.index` | `2722aef8257b8748bd2cdd00d4224e7b4dee3d1d552a235439eb4c97c71aface` |
| metas | `data/dog_metas.json` | `723a4df01e2802ad5d42bd276eefbded097a0a6e72900784395c4c19626b6c00` |
| snapshot_manifest | `data/snapshot_manifest.json` | `f79a4b3010b52c9584525af99b1acd4007fc6376fc8244deab18058b74fa7413` |
| active_index_sync_report | `data/active_index_sync_report.json` | `924a3d657a9bb09d063a13a6c329f0b9965a4d772ed51d539dbdcd61f02d28ae` |
| requirements_lock | `requirements.lock.txt` | `6d89251f5851a0b38a66b0ed1109f0e4f556d7f6a8440e5282f2c5f937fb267e` |

## Encoder provenance

- declared index provenance consistent: `True`
- query encoder commit verified: `True`
- full index encoder verified: `False`

## 표본 정책

- seed: `20260726-heldout-secondary-photo-v1`
- 최대 표본: 120
- 선택: one per region|size stratum when capacity allows, then global SHA-256 order
- 동일 URL이거나 다른 공고의 인덱스 URL인 2번째 사진은 계획 단계에서 제외
- 실제 실행에서는 primary/query payload SHA-256 동일 및 query SHA 중복도 제외

### 표본 지역 분포

| 지역 | 건수 |
|---|---:|
| 강원 | 7 |
| 경기 | 22 |
| 경남 | 11 |
| 경북 | 10 |
| 대구 | 4 |
| 대전 | 2 |
| 부산 | 6 |
| 서울 | 5 |
| 세종 | 1 |
| 울산 | 3 |
| 인천 | 6 |
| 전남 | 15 |
| 전북 | 8 |
| 제주 | 7 |
| 충남 | 7 |
| 충북 | 6 |

### 표본 크기 분포

| 크기 | 건수 |
|---|---:|
| large | 16 |
| medium | 29 |
| small | 37 |
| tiny | 38 |

## 측정 결과

| 지표 | 값 |
|---|---:|
| audit coverage status | partial |
| attempted | 120 |
| downloaded | 55 |
| evaluable | 29 |
| duplicate excluded | 26 |
| download coverage | 45.83% |
| evaluable coverage | 24.17% |
| primary download coverage | 70.83% |
| payload audit coverage | 70.83% |
| Hit@1 | 82.76% |
| Hit@5 | 96.55% |
| Hit@10 | 96.55% |
| end-to-end Hit@1 | 20.00% |
| end-to-end Hit@5 | 23.33% |
| end-to-end Hit@10 | 23.33% |
| MRR | 0.8912 |
| median rank | 1.0 |

### Download and exclusion failures

| reason | count |
|---|---:|
| indexed_source_request_timeout | 1 |
| indexed_source_total_timeout | 34 |
| primary_unavailable | 35 |
| query_matches_indexed_primary_bytes | 26 |
| secondary_http_status_404 | 3 |
| secondary_total_timeout | 27 |

## 해석 제한

- 이 교차사진 동일 공고 과제는 시각적 동일성 검색의 대리 지표이며, 서로 다른 개의 주관적 유사도를 측정하지 않습니다.
- 성격, 건강, 입양 적합성, 아동 친화성, 다른 동물과의 생활 가능성을 측정하지 않습니다.
- 표본 선정 전에는 URL 차이만 확인하며, 실제 payload SHA-256 누출과 질의 중복은 네트워크 실행에서만 검사합니다.
- 기본 표본 감사에서는 primary 다운로드 실패만큼 payload 감사 커버리지가 낮아지며 이를 공개합니다. 평가 가능한 질의는 자기 primary SHA와 다운로드에 성공한 모든 표본 primary SHA에 대조하지만, 받지 못한 primary와의 중복은 배제할 수 없습니다.
- 전체 corpus 감사에는 --audit-all-indexed-sources가 필요하고 오래 걸릴 수 있으며, 선언된 모든 활성 시각 출처 다운로드에 성공해야만 완전합니다.
- 이미지 전송은 초기 HTTP 메타데이터 URL을 HTTPS로 올리고 HTTPS에서 HTTP로 낮추는 리디렉트를 거부합니다.
- 생성한 crop 바이트는 인덱스와 함께 보관하지 않으므로 정확한 raw-byte 비교는 crop payload가 아니라 공개 원본 사진을 대상으로 합니다.
- 공개 이미지 URL은 바뀌거나 사라질 수 있으므로 검색 지표와 함께 다운로드 및 평가 가능 커버리지를 보고해야 합니다.
- raw CLIP 시각 인덱스 결과는 추가 필터와 재정렬을 적용할 수 있는 전체 운영 API 결과가 아닙니다.
- 갱신 인덱스에는 이전 스냅샷의 이미지·crop 벡터가 남아 있습니다. 벡터별 encoder sidecar fingerprint가 없으므로 artifact 선언만으로 모든 벡터의 encoder를 증명할 수 없습니다.

이미지 바이너리는 저장하거나 배포하지 않으며 실제 실행 중 메모리에서만 검사합니다. URL이 달라도 재인코딩된 동일 사진은 raw SHA-256만으로 완전히 탐지할 수 없습니다.
