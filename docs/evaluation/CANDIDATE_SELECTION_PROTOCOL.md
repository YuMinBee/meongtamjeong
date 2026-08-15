# 보호소 방문 후보 3개 선택 시간 파일럿

이 프로토콜은 고정된 유기견 공고 후보 안에서 먼저 확인할 후보 3개를 고르는 데 걸린 시간을 로컬에서 기술적으로 측정합니다. 입양 적합성, 성격, 건강, 안전, 실제 보호소 방문 또는 입양 성과를 평가하지 않습니다. 사람 결과가 생성되기 전에는 성능 개선을 주장할 수 없습니다.

## 비교 계약

- `baseline`: 검색 평가 리포트의 `baseline_system`
- `meongtamjeong`: 같은 리포트의 `headline_system`
- 각 과제의 후보 pool: 두 시스템의 고정 Top-k 합집합 중 저장소 안의 로컬 crop을 확인할 수 있는 공고
- 두 조건은 정확히 같은 후보와 같은 사진·공고 사실을 보여주며 순서만 다릅니다.
- 한 시스템의 Top-k에 없는 합집합 후보는 해당 시스템 순서의 뒤에 opaque candidate code 순으로 놓습니다.
- 점수, 시스템명, 원래 순위, 공고 ID는 참여자 화면과 CSV에서 숨깁니다.
- 공고 사진은 생성 시 data URI로 HTML에 포함합니다. 브라우저는 네트워크, 원격 이미지, 분석 SDK, 쿠키, `localStorage`, `sessionStorage`를 사용하지 않습니다.
- 생성기는 과제마다 로컬 사진 후보와 공고 사실 기준 evidence-confirmed 후보가 각각 최소 3건인지 검사합니다.

기본 과제 ID는 `brown-small`, `jeju-white-puppy`, `gyeongnam-black-small`, `senior-under-8kg`입니다. 이는 외형, 지역, 연령 조건을 나누어 보기 위한 사전 고정 기본값이며 사람 결과를 본 뒤 유리한 과제로 교체해서는 안 됩니다. 다른 과제를 사용할 경우 사람 측정 전에 네 ID, depth, seed, timeout을 고정하고 그 이유를 결과와 함께 공개합니다.

공고 사실 qrel은 빠른 무작위 선택을 감지하는 guardrail입니다. `unknown`을 부적합으로 판단하지 않으며, 단지 “공고로 조건 일치를 확인함”에 포함하지 않습니다.

## 참여자와 순서 배정

권장 인원은 4명입니다. 3~5명도 실행할 수 있지만 4명이 아니면 완전한 균형 설계가 아니므로 그대로 공개합니다. 진행자는 참여 전에 `P01`처럼 대문자 `P`와 숫자 2~4자리로만 구성된 코드를 사전 배정합니다. 참여자가 이름·별명·이메일에서 코드를 만들게 하지 않습니다.

| 순서표 | 수행 순서 |
|---|---|
| S1 | T1-baseline → T2-meongtamjeong → T3-meongtamjeong → T4-baseline |
| S2 | T2-meongtamjeong → T3-baseline → T4-baseline → T1-meongtamjeong |
| S3 | T3-baseline → T4-meongtamjeong → T1-meongtamjeong → T2-baseline |
| S4 | T4-meongtamjeong → T1-baseline → T2-baseline → T3-meongtamjeong |

실제 HTML에는 역할 대신 seed로 결정된 A/B condition code만 들어갑니다. 4명일 때 S1~S4를 한 번씩 배정하면 각 과제와 각 순서 위치에서 두 조건이 2회씩 나타납니다. 3명은 S1~S3, 5명은 S1~S4 후 배정 횟수가 가장 적은 순서표 하나를 추가하고 실제 `sequence_counts`를 보고합니다.

## 개인정보와 진행 규칙

수집하는 값은 진행자 사전 배정 참여자 코드, 배정 순서표, opaque task·condition·candidate code, 경과시간, 첫 선택 시간, 선택 토글 횟수, 탭 비활성 시간, timeout·완료 여부뿐입니다. HTML과 CSV 로더는 `^P[0-9]{2,4}$`만 허용하며 `Alice`, `YuMin`, 이메일, 공백 포함 이름을 거부합니다.

다음 값은 수집하지 않습니다.

- 이름, 이메일, 전화번호, 나이, 성별, IP, 브라우저 user-agent
- 자유서술, 개인 참고사진, API 키
- 정확한 시작·종료 시각
- 서버 로그, 원격 분석 이벤트

참여자에게 목적, 진행자가 배정한 코드, 중단 가능성, 원본 CSV 보관·삭제 방침을 먼저 설명합니다. 원본 CSV는 Git에 커밋하지 않고 ignore된 `data/reports/candidate_selection/`에만 보관합니다. 집계 리포트에는 원본 참여자 코드를 싣지 않고 study ID로 해시한 참조 코드만 기록합니다.

진행 순서는 다음과 같습니다.

1. 동일한 기기·브라우저·화면 크기를 사용합니다.
2. 진행자는 `task_key.json`을 참여자에게 보여주지 않고 S1~S4 중 하나와 사전 배정한 `P` + 숫자 2~4자리 코드를 알려줍니다.
3. 참여자는 배정받은 코드를 그대로 입력하며 이름·별명·이메일은 입력할 수 없습니다.
4. 기록되지 않는 연습 화면에서 선택과 취소를 익힙니다.
5. 각 과제에서 `과제 시작`을 누른 뒤 사진과 공고 사실을 보고 후보 3개를 선택합니다.
6. `3개 후보 확정`을 누르면 측정이 끝납니다. 기본 제한시간은 180초입니다.
7. 네 과제 후 CSV를 저장합니다. 파일명에도 이름이나 이메일을 넣지 않습니다.
8. 탭 전환 시간은 `hidden_ms`로 보존합니다. 임의로 삭제하지 말고 결과와 함께 확인합니다.

`과제 시작`을 누르면 먼저 모든 로컬 카드 이미지의 브라우저 `decode()` 완료를 확인합니다. 그 뒤 카드를 표시하고 두 번의 `requestAnimationFrame`을 지나 적어도 한 번의 화면 그리기 기회를 준 다음 `performance.now()`와 timeout을 함께 시작합니다. 이미지가 하나라도 디코딩되지 않거나 `decode()`를 지원하지 않으면 오류를 표시하고 해당 trial의 시간·선택 결과를 기록하지 않습니다. 다시 시도해도 실패하면 측정을 중단하고 진행자에게 알립니다. 모델 실행, 네트워크, 질의 작성, 공공 사이트 로딩 시간은 포함하지 않으므로 결과는 반드시 “이미지 디코딩·표시 후 후보 선택 시간”이라고 부릅니다. timeout은 설정된 cap 시간으로 집계해 실패를 빠른 완료처럼 보이지 않게 합니다.

## 실행

### clean clone의 로컬 crop 선행조건

`data/dog_metas.json`은 추적되지만 `data/image_crops*/`는 용량·재배포 경계 때문에 Git에서 제외됩니다. 따라서 clean clone에서 바로 `generate`를 실행하면 원격 사진으로 대체하지 않고 다음과 같은 local crop preflight 오류로 종료되는 것이 정상입니다.

```text
local crop preflight failed ... Network fallback is intentionally disabled
```

이미 권리와 출처를 확인한 동일 스냅샷 crop 묶음이 있다면 메타데이터의 `image_attrs.crop_path`와 같은 저장소 내부 경로에 복원합니다. 없다면 사람 평가를 시작하기 전에 기존 공개데이터 crop 도구로 별도 ignore 경로에 재생성할 수 있습니다. 아래 단계는 공공 이미지 다운로드와 TorchVision 가중치 다운로드가 발생할 수 있으므로 네트워크 정책과 사진 이용 조건을 먼저 확인해야 합니다. 생성된 사진과 메타데이터는 커밋하지 않습니다.

```powershell
conda run -n meong-contest-full python scripts/enrich_image_crops.py `
  --input data/dog_metas.json `
  --output data/reports/candidate_selection/dog_metas_with_local_crops.json `
  --crop-dir data/reports/candidate_selection/local_crops `
  --limit 0 `
  --device cpu
```

그 후 생성된 메타데이터를 명시해 고정 과제를 만듭니다.

```powershell
conda run -n meong-contest-full python scripts/candidate_selection_study.py generate `
  --metas data/reports/candidate_selection/dog_metas_with_local_crops.json
```

이미 현재 `crop_path` 파일이 모두 준비된 작업공간에서는 기본 명령을 사용할 수 있습니다.

```powershell
conda run -n meong-contest-full python scripts/candidate_selection_study.py generate
```

생성물은 기본적으로 `data/reports/candidate_selection/` 아래에 생깁니다.

- `task.html`: 참여자에게 전달하는 네트워크 없는 로컬 화면
- `task_key.json`: 시스템·공고·evidence 매핑 및 소스·HTML 해시. 참여자와 분리

사람 평가 전에 소스, 일정, 키와 HTML 무결성을 검사합니다.

```powershell
conda run -n meong-contest-full python scripts/candidate_selection_study.py check `
  --key data/reports/candidate_selection/task_key.json
```

참여자 CSV를 모두 받은 뒤 집계합니다.

```powershell
conda run -n meong-contest-full python scripts/candidate_selection_study.py aggregate `
  --key data/reports/candidate_selection/task_key.json `
  --labels data/reports/candidate_selection/P01.csv data/reports/candidate_selection/P02.csv data/reports/candidate_selection/P03.csv data/reports/candidate_selection/P04.csv `
  --json-out data/reports/candidate_selection/result.json `
  --markdown-out data/reports/candidate_selection/result.md
```

마지막으로 원본 CSV에서 결과를 다시 계산하고 JSON·Markdown이 일치하는지 검사합니다.

```powershell
conda run -n meong-contest-full python scripts/candidate_selection_study.py check `
  --key data/reports/candidate_selection/task_key.json `
  --labels data/reports/candidate_selection/P01.csv data/reports/candidate_selection/P02.csv data/reports/candidate_selection/P03.csv data/reports/candidate_selection/P04.csv `
  --result data/reports/candidate_selection/result.json `
  --markdown data/reports/candidate_selection/result.md
```

`check`는 소스·HTML SHA-256, study material, 조건 매핑, 네 개 순서표 균형, trial token, CSV 열·과제 완주·후보 집합·시간 범위·완료 상태, 결과 재계산을 검사합니다. 정적 HTML에 비밀 서명키가 없으므로 의도적인 시간 조작이나 지시 준수 자체를 증명하지는 못합니다.

`check`는 현재 소스에서 과제를 결정적으로 다시 생성하므로 crop 파일이 평가 후 삭제·교체된 경우에도 실패합니다. 최종 집계를 검증할 때까지 key, HTML, 입력 메타데이터와 로컬 crop을 같은 상태로 보존합니다.

## 출력과 해석

집계 리포트는 다음을 분리해 제공합니다.

- 조건별 시행 수, 완료율, timeout 수
- 완료 시행의 3/3 공고사실 확인율과 전체 선택의 evidence-confirmed 비율
- timeout을 cap으로 포함한 시간과 완료 시행만의 시간에 대한 최소·25%·중앙·75%·최대
- 참여자별 조건 중앙시간과 `meongtamjeong - baseline` 차이
- 멍탐정이 더 빠른 참여자 수, baseline이 더 빠른 참여자 수, 동률
- 과제·조건별 시행 수, 완료율, capped 중앙시간
- 과제별 원 Top-k 합집합, 로컬 사진 사용 가능 수, 사진 누락 제외 수, 공고사실 확인 후보 수
- 실제 순서표 배정 수와 4인 완전 균형 여부
- 선택 토글, 첫 선택, 탭 비활성 시간

시행 행을 독립 표본처럼 합쳐 유의확률을 계산하지 않습니다. 주 비교 단위는 각 참여자의 조건별 capped 중앙시간입니다. 참여자가 3~5명뿐이므로 p-value, 신뢰구간, 모집단 평균 또는 인과적 일반화는 제시하지 않습니다.

사용 가능한 표현 예시는 다음과 같습니다.

> 4명 로컬 파일럿의 고정 후보 pool에서 멍탐정 순서의 참여자별 capped 중앙 후보확정 시간이 내부 baseline보다 X ms 짧았습니다. 과제 완료율과 공고사실 3/3 확인율은 각각 함께 보고합니다.

다음 표현은 사용할 수 없습니다.

- 멍탐정이 입양 성공률을 높였다.
- 개체의 성격·건강·아동 친화성·합사 적합성을 검증했다.
- 대한민국 사용자의 평균 탐색 시간을 줄였다.
- 공공 동물보호 포털보다 빠르다.
- 작은 표본의 기술 통계를 통계적으로 유의한 성능 향상이라고 표현한다.

또한 이 파일럿은 같은 고정 후보 pool의 순서만 비교하므로 전체 corpus recall, 새 질의 생성, 모델·네트워크 지연, 실제 공고 확인, 보호소 문의와 방문 과정은 평가하지 않습니다. 시간 차이는 완료율과 공고사실 확인 guardrail이 악화되지 않았을 때도 소수 참여자에 대한 기술 통계로만 설명합니다.
