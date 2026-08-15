# Blind human relevance protocol

이 도구는 공고 필드로 자동 생성한 silver label의 한계를 보완하기 위한 소규모 사람 검수입니다. 검색 시스템 이름과 원래 순위를 숨긴 뒤, 같은 질의에서 여러 시스템의 상위 후보를 합쳐 0/1/2 관련성 라벨을 받습니다. 입양 적합성이나 성격을 평가하는 도구가 아닙니다.

## 평가 전 고정할 것

라벨을 받기 전에 다음을 정하고 바꾸지 않습니다.

- 비교할 retrieval report와 시스템 ID
- 질의 목록
- 시스템별 pool depth(권장 시작값: 5)
- random seed
- 검수자 수(작은 파일럿도 가능하지만, 대회 근거에는 서로 상의하지 않는 3명 이상 권장)

후보 pool은 각 시스템의 질의별 top-k 합집합입니다. 같은 공고가 두 시스템에 나오면 한 번만 평가합니다. 생성된 `task_key.json`은 시스템 이름과 원래 순위를 담으므로 검수가 끝날 때까지 검수자에게 주지 않습니다.

## 1. 과제 생성

기본값은 retrieval report의 `baseline_system`과 `headline_system`을 비교합니다. 명시적으로 고정하려면 `--systems`를 사용합니다.

```powershell
conda run -n meong-contest-full python scripts/generate_blind_relevance_task.py `
  --systems clip_active hybrid_natural_graph `
  --depth 5 `
  --seed 20260726 `
  --image-policy local-only `
  --output-dir data/reports/blind_relevance
```

산출물:

- `task.html`: 검수자에게 전달하는 정적 로컬 화면
- `labels_template.csv`: 수기 입력용 빈 CSV
- `task_key.json`: 집계용 비공개 key

`local-only`는 네트워크 요청을 하지 않고 저장소의 로컬 crop만 참조합니다. 생성 로그의 `missing_images`가 0이 아니면 외형 평가를 시작하지 마세요. 로컬 crop을 준비하거나, 공공 공고 이미지 접속이 허용된 내부 평가에서는 `prefer-local`을 선택해 부족한 사진만 원본 URL로 표시할 수 있습니다. HTML은 평가 서버나 분석 서비스를 사용하지 않으며 라벨을 업로드하지 않습니다.

## 2. 라벨 기준

검수자는 이름이나 이메일 대신 `reviewer-01` 같은 임의 코드를 사용합니다.

- `0 = 무관`: 질의의 관찰 가능한 외형·공개 공고 조건과 관련이 없음
- `1 = 부분 일치`: 일부 조건만 일치하거나 사진만으로 판단이 애매함
- `2 = 잘 일치`: 관찰 가능한 조건이 질의와 명확히 잘 일치함

모든 후보를 평가한 뒤 화면의 `CSV 저장` 버튼을 누릅니다. 사진으로 성격, 공격성, 활동량, 아동·다른 동물 친화성, 입양 적합성을 추정하지 않습니다. 사진이 보이지 않거나 조건을 확인할 수 없을 때 임의로 0점을 주지 말고 과제를 중단해 평가 담당자에게 알립니다.

## 3. 여러 응답 집계

각 CSV는 한 검수자가 과제 전체를 완료해야 합니다. 누락, 중복, 0/1/2 외 값, 다른 task ID, 공백이 든 검수자 코드는 오류로 종료됩니다.

```powershell
conda run -n meong-contest-full python scripts/aggregate_blind_relevance.py `
  --key data/reports/blind_relevance/task_key.json `
  --labels data/reports/blind_relevance/reviewer-01.csv data/reports/blind_relevance/reviewer-02.csv `
  --json-out data/reports/blind_relevance/result.json `
  --markdown-out data/reports/blind_relevance/result.md
```

집계는 후보별 검수자 점수의 산술평균을 사용하고, gain `2^grade - 1`의 시스템별 nDCG@k를 질의 단위로 계산한 뒤 macro average를 냅니다. 시스템 간 “선호도”는 질의별 nDCG 승/무/패와 무승부 제외 승률입니다. 이는 별도의 목록 선호 투표가 아니라 관련성 라벨에서 파생한 비교입니다.

JSON에는 검수자 수, 질의 수, 고유 query-candidate 수, 총 라벨 수, 점수 분포가 함께 기록됩니다. 검수자 코드는 가명이어야 하며 결과와 원본 CSV에 개인정보를 넣지 않습니다.

## 해석 제한

- 비교한 시스템의 top-k 합집합 밖 후보는 평가하지 않으므로 전체 corpus 품질을 증명하지 않습니다.
- 적은 질의·검수자 표본은 모집단 수준의 통계적 결론으로 표현하면 안 됩니다.
- 공개 공고 이미지와 필드의 오류·누락이 사람 판단에도 영향을 줄 수 있습니다.
- 관련성 점수는 보호소 방문 전 외형 탐색 품질만 다루며 입양 결정, 성격, 안전, 건강 적합성을 측정하지 않습니다.
- 사람이 작성한 실제 라벨은 코드가 자동 생성하지 않습니다. 제출 근거로 쓰려면 실제 독립 검수 절차와 원본 CSV를 별도 보존해야 합니다.
