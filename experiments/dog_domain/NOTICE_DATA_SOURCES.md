# 공고 데이터 추가 조사와 확대 평가 — 2026-09-16

## 실제 확보한 데이터

| 데이터 | 이번 확인·확보 | 활용과 제약 |
|---|---|---|
| [한국 국가동물보호정보시스템](https://www.data.go.kr/data/15098931/openapi.do) | 최근 32일 개 공고 4,801건 새로 수집 | 사진 여러 장, 한국어 특이사항, 색상·체중·나이. 기존 학습 이후 공고를 별도 평가에 사용 |
| [대만 농업부 동물 입양 공고](https://data.gov.tw/dataset/85903) | 전체 8,341건, 개 5,933건. 개 중 사진 URL 5,516건, 비고 1,933건 | 중국어 비고·색상·체형·사진. 이번 응답은 개 ID당 한 행, 단일 사진 필드. 사진 두 장의 동일 개체 검색에는 부족하며, 텍스트→이미지 또는 자기 사진을 제외한 속성 검색 후보. 아직 성능 평가하지 않음 |
| [미국 PetFinder / The Pudding](https://raw.githubusercontent.com/the-pudding/data/master/dog-shelters/README.md) | 기존 로컬 설명 데이터와 이미지 연결 자료 사용 가능 | 설명은 2019년 자료. [이미지 아카이브](https://huggingface.co/datasets/drzraf/petfinder-dogs)는 2023년 자료라 ID 결합과 시점 차이 확인 필요. 기존 연결 424마리 중 평가 결과를 이미 본 표본은 새 독립 테스트로 취급하지 않음 |

추가 후보인 [PetFinder.my Adoption Prediction](https://www.kaggle.com/c/petfinder-adoption-prediction/data/)는
사진·자연어 Description·색상·MaturitySize·PetID를 함께 제공한다. 한 공고에 여러 동물이 있을 수 있으므로
`Type=1`, `Quantity=1` 및 실제 다른 사진 여부 확인이 필요하다. 현 환경에는 Kaggle 인증 파일이 없으며,
원본 다운로드·규칙 동의 상태는 확인되지 않아 이번 평가에는 포함하지 않았다.
`AdoptionSpeed`는 입양 속도 정답이므로 검색 정확도 정답으로 사용하지 않는다.

한국 공식 포털은 무료·이용허락범위 제한 없음으로 표시한다. 대만은 정부자료개방 이용허락 1판으로 표시한다.
미국 이미지 데이터 카드는 라이선스를 `unknown`으로 표시하므로 원본 사진을 배포하지 않는다.
공개 필드가 있다는 사실과 자연어 검색의 정답이 있다는 사실은 별개다.

## 이번에 끝낸 국내 확대 평가

기존 모델을 재학습하지 않고 새 공고만 평가했다. 이전 메타데이터 22개 파일과 과거 평가 목록을 확인하고,
과거 공고 ID 3,915개 및 2026-08-16 이전 발생 공고를 제외했다. 색상·크기가 있는 적격 공고 4,581개에서
점수와 무관한 고정 해시로 1,000개를 선정했다. 사진 실패 28개, 다른 사진 미확보 355개,
공고 간 완전 동일 이미지가 있는 6개를 제외해 **611개 공고·136개 보호소**를 평가했다.

사진은 두 모델 모두 원본 전체 사진을 사용했다. 기존 20개 파일럿은 크롭·다른 후보 집합이므로
점수 자체를 직접 이어 붙이지 않는다. 공고 ID 중복과 선택 표본 내 완전 동일 이미지 중복은 검사했지만,
다른 공고에 같은 개가 재등록됐는지, 사진에 여러 마리가 있는지는 수동으로 확정하지 않았다.

| 한국어 입력 / 방법 | 같은 공고 1위 적중률 | 색상+크기 검색 nDCG@10 ×100 |
|---|---:|---:|
| 사진만 CLIP | 71.03% | 33.01 |
| 사진만 DINO | 92.96% | 40.63 |
| 속성 문장 + CLIP | 70.70% | 32.97 |
| 속성 문장 + DINO·Flow | 91.98% | 43.98 |
| 실제 공고 설명 + DINO·Flow | 92.14% | 40.32 |

속성 문장 조건에서 Flow는 DINO 사진만 대비 색상+크기 nDCG가 **+3.35점**이었다
(보호소 단위 bootstrap 95% 구간 **+2.72~+3.96**). 반대로 같은 공고 1위 적중률은 **−0.98%p**였다
(**−1.97~−0.15%p**). 따라서 모든 목적에서 성능이 오른다는 결과가 아니다.

색상·크기 문장의 Flow는 Linear 43.75, MLP 43.85와 차이가 작으며 우월성이 명확하지 않다.
실제 공고 설명을 그대로 입력하면 DINO 사진만 대비 속성 점수 차이는 **−0.32점**
(**−0.87~+0.21**)이고, 설명을 다른 공고와 바꿔 넣은 대조군 대비도 이득이 확인되지 않았다.
원문 611개 중 7개는 CLIP 최대 토큰 길이로 잘렸다. 원문 평가는 설명 전체의 의미 정확도를 채점한 것이 아니다.

현재 근거에 맞는 연구 방향은 **“유기견 검색에서 CLIP 텍스트와 DINO 시각 특징의 정렬은
명시적인 외형 조건 검색을 개선하지만, 자연스러운 공고 설명과 개체 식별에서는 한계가 있다”**이다.
CLIP 대비 같은 공고 검색 향상은 대부분 DINO 시각 특징 자체의 효과이므로 Flow의 텍스트 효과와 구분한다.

자세한 모든 비교와 구간은 [NOTICE_EXTENSION_RESULTS.md](NOTICE_EXTENSION_RESULTS.md),
사전 고정한 기준은 [NOTICE_EXTENSION_PROTOCOL.md](NOTICE_EXTENSION_PROTOCOL.md)에 있다.
원본·사진·특징·쿼리별 점수는 git에서 제외되는 `tmp/notice_extension_20260916/`에 보관한다.

## 재현

원본 스냅샷이 있으면 재수집하지 않는다. 새로운 날짜의 API 응답은 별도 실험으로 저장한다.

```powershell
python -m scripts.fetch_live_dogs --days 32 --rows 1000 --max-pages 10 --include-closed --out tmp/notice_extension_20260916/notices.json
python -m experiments.dog_domain.notice_extension prepare
python -m experiments.dog_domain.notice_extension features
python -m experiments.dog_domain.notice_extension evaluate
python -m experiments.dog_domain.verify_notice_extension
```

대만 원본 응답은 `tmp/notice_extension_20260916/taiwan/notices.json`, 출처·해시는 `source.json`에 저장했다.
공식 JSON 주소:
`https://data.moa.gov.tw/Service/OpenData/TransService.aspx?IsTransData=1&UnitId=QcbUEzN6E6DL`.
