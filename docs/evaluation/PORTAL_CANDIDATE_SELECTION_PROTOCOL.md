# 공식 포털 대 멍탐정 end-to-end 후보 탐색 파일럿

이 문서는 국가동물보호정보시스템과 멍탐정에서 같은 종류의 외형 과제를 수행해 **활성 공고 후보 3개를 확정하기까지의 시간, 완료율, 최종 후보의 blind 관련성**만 기술적으로 비교하는 사전 고정 프로토콜입니다. 3~5명의 소규모 파일럿이며 보호소 문의, 방문, 입양, 성격, 건강 또는 안전 성과를 평가하지 않습니다.

저장소의 기존 `candidate_selection_study.py`는 같은 로컬 후보 pool 안에서 내부 CLIP 검색 순서 두 개를 비교하는 보조 실험입니다. 이 외부 포털 비교와 데이터·결과·표현을 합치지 않습니다. 외부 비교에는 별도 `portal_candidate_selection_study.py`와 `portal-candidate-selection-result.v1` 스키마만 사용합니다.

현재 추적된 JSON은 `template_no_human_results` 상태입니다. 실제 사람 결과나 개선 수치는 들어 있지 않습니다. 첫 참여자 전에 스냅샷·커밋·실시일을 넣어 한 번 freeze하고, 관찰 후 과제·cap·순서표를 바꾸지 않습니다.

## 고정 비교 계약

- 비교 조건: 국가동물보호정보시스템 `official_portal` 대 멍탐정 `meongtamjeong`
- 대상: 개 공고만
- 과제: 기존 외형 검색 평가에서 미리 정한 4개 과제
  - `brown-small`: 갈색 소형견
  - `jeju-white-puppy`: 제주에서 보호 중인 흰색 어린 강아지
  - `gyeongnam-black-small`: 경남의 검정색 소형견
  - `senior-under-8kg`: 8kg 이하의 노령 소형견
- 완료: 제한시간 안에 서로 다른 **활성 공고 3건**의 공식 공고 ID와 상세 링크를 확정
- 제한시간: 과제당 300초
- 권장 참여자: 4명. 최소 3명, 최대 5명
- 주요 지표: 전체 예정 시행 완료율과 참여자별 조건 중앙 capped 시간
- guardrail: 최종 후보의 출처 blind 관련성, 실패 유형별 건수
- 통계: 기술 통계만. p-value, 신뢰구간, 모집단 일반화 금지

공식 포털은 외형 자연어 검색을 제공하지 않으므로 참여자가 이용 가능한 필터·목록·상세 페이지를 자유롭게 사용합니다. 멍탐정에서는 고정된 문장을 그대로 입력합니다. 진행자가 특정 후보, 검색어 변형, 페이지 또는 필터 조합을 알려주지 않습니다.

## 데이터 동등성과 freeze

사람 측정 당일 멍탐정 공고를 최신화하고, 동일한 접수일 범위와 개 축종으로 공식 포털을 엽니다. 최신화 후 가능하면 90분 안의 한 세션으로 전 참여자를 측정합니다. 공식 포털은 계속 변할 수 있으므로 두 corpus가 완전히 동일하다고 주장하지 않습니다. 대신 아래를 protocol에 고정합니다.

- 측정일
- 멍탐정 실행 커밋
- 멍탐정 공고 스냅샷과 실제 FAISS 인덱스의 SHA-256·크기
- 사전 고정 과제·순서표·cap의 SHA-256

측정 전 과제당 정답 후보 수를 찾아보거나 결과가 유리한 과제로 교체하지 않습니다. 특정 과제에서 후보 3건을 찾지 못하는 상황 자체가 완료율 결과입니다. 공고 스냅샷을 바꾸거나 서버를 재시작해야 하면 기존 trial과 섞지 말고 새 protocol을 freeze합니다.

## 참여자 배정과 순서 효과

참여자는 이름·이메일과 무관하게 진행자가 사전 배정한 `P01` 형식 코드만 사용합니다. 각 참여자는 네 과제를 한 번씩만 보고 두 조건을 두 번씩 수행합니다. 같은 사람이 같은 과제를 양쪽에서 반복해 후보를 기억하지 않도록 했고, 4명일 때 각 과제가 두 조건에 각각 두 번 배정됩니다.

| 순서표 | 시행 순서 |
|---|---|
| S1 | brown-small/포털 → jeju-white-puppy/멍탐정 → gyeongnam-black-small/멍탐정 → senior-under-8kg/포털 |
| S2 | jeju-white-puppy/멍탐정 → gyeongnam-black-small/포털 → senior-under-8kg/포털 → brown-small/멍탐정 |
| S3 | gyeongnam-black-small/포털 → senior-under-8kg/멍탐정 → brown-small/멍탐정 → jeju-white-puppy/포털 |
| S4 | senior-under-8kg/멍탐정 → brown-small/포털 → jeju-white-puppy/포털 → gyeongnam-black-small/멍탐정 |

- 3명: 서로 다른 세 순서표를 한 번씩 사용
- 4명: S1~S4를 정확히 한 번씩 사용
- 5명: S1~S4를 한 번씩 사용하고 한 순서표를 한 번 추가. 실제 불균형 공개

CLI `prepare`의 기본 배정은 P01/S1, P02/S2, P03/S3, P04/S4, P05/S1입니다. 사람 결과를 본 뒤 순서표를 다시 배정하지 않습니다.

## 측정 시작과 종료

동일한 PC, 브라우저, 화면 크기와 네트워크를 사용합니다. 광고 차단기·확장 프로그램 상태도 고정합니다. 비측정 연습은 공통 조작 설명과 결과 상세 링크를 여는 방법까지만 허용하며, 네 고정 과제나 실제 후보를 사용하지 않습니다.

각 시행은 다음과 같습니다.

1. 배정 조건의 시작 화면이 정상 로드됐는지 확인합니다. 시작 화면 자체가 열리지 않으면 타이머를 시작하지 않고 `network_failure/system_launch` 또는 `technical_failure/system_launch`로 기록합니다.
2. 진행자가 고정 과제 문장을 읽어 주고 `시작`과 동시에 stopwatch를 누릅니다.
3. 참가자는 검색·필터·목록·상세 확인을 직접 수행합니다. 검색 중 제품 응답시간과 페이지 이동시간은 측정에 포함합니다.
4. 최종 후보마다 실제 국가동물보호정보시스템 상세 페이지를 열어 공고 ID, URL과 상태를 확인합니다.
5. 서로 다른 활성 후보 3건을 확인한 순간 타이머를 멈추고 `completed`로 기록합니다.
6. 300초에 도달하면 즉시 멈추고 `time_cap`, `duration_ms=300000`으로 기록합니다.

정확한 시작·종료 시각은 기록하지 않고 경과시간만 기록합니다. 사람이 중단 의사를 밝히면 실제 경과시간과 `participant_stopped`를 기록하되 시간 비교에서는 cap으로 처리합니다. 실패 후 다시 같은 시행을 반복하지 않습니다.

## 활성 상태와 후보 기록

후보마다 다음 세 필드를 함께 기록합니다.

- 국가동물보호정보시스템의 15자리 `desertionNo`
- `https://www.animal.go.kr/front/awtis/public/publicDtl.do?...` 형식의 실제 상세 URL
- `active | inactive | unknown`

멍탐정 카드가 제공한 실제 공고 링크도 최종적으로 같은 공식 상세 URL을 사용합니다. 이 때문에 blind label sheet에서는 후보를 어느 제품에서 골랐는지 URL만으로 알 수 없습니다.

공식 상세 페이지에서 현재 보호·공고 중임이 명확하고 종료 상태가 없을 때만 `active`입니다. 입양 완료, 반환, 기증, 종료, 자연사, 안락사 등 비활성 상태는 `inactive`입니다. 페이지가 열리지 않거나 상태 문구가 불명확하면 추정하지 않고 `unknown`입니다. `completed`는 세 후보가 모두 `active`일 때만 허용됩니다. 종료·불명 후보를 다른 후보로 교체할 시간이 없으면 해당 trial은 완료가 아닙니다.

## 실패 유형

`outcome`과 `failure_stage`는 고정 값만 사용하며 자유서술하지 않습니다.

- `completed`: 활성 후보 3건 확정
- `time_cap`: 300초 도달
- `participant_stopped`: 참여자 자발적 중단
- `network_failure`: DNS, 연결, HTTP 등 네트워크 문제로 진행 불가
- `technical_failure`: 브라우저 또는 제품 오류로 진행 불가

실패 단계는 `system_launch | task_setup | search | notice_detail | active_verification` 중 하나입니다. 완료만 `none`을 사용합니다. 네트워크·기술 실패는 완료율 전체 분모에는 남기고 별도 건수를 공개하지만 capped 시간 비교와 participant paired 시간에서는 제외합니다. time cap과 참여자 중단은 빠른 실패처럼 보이지 않도록 300초로 처리합니다.

## 개인정보 금지

수집하는 값은 사전 배정 P 코드, 순서표, trial·task·system 코드, 통제된 outcome·failure stage, 경과시간, 공개 공고 ID·URL·활성 상태뿐입니다. 다음은 수집하지 않습니다.

- 이름, 별명, 이메일, 전화번호, 나이, 성별
- IP, user-agent, 계정 ID, 쿠키, 분석 이벤트
- 정확한 시작·종료 시각
- 자유서술, 화면 녹화, 음성, 얼굴
- 개인 참고사진 또는 검색 기록

원본 CSV는 Git에 커밋하지 않고 ignore된 `data/reports/portal_candidate_selection/`에만 둡니다. 결과 JSON에는 원본 P 코드 대신 protocol별 hash 참조만 포함됩니다.

## blind 관련성 판정

전 시행 종료 후 `blind` 명령으로 최종 후보를 `(task_id, notice_id)` 기준 중복 제거하고 순서를 고정 seed로 섞습니다. 생성 CSV에는 참여자 코드, 선택 제품, 원래 순위와 시간이 없습니다. 독립 판정자에게 이 CSV만 줍니다. 판정자는 공식 공고 상세의 공개 정보와 사진을 보고 각 과제 조건 전체에 대해 다음 중 하나만 기록합니다.

- `relevant`: 모든 명시 조건을 공고 사실 또는 사진에서 확인
- `not_relevant`: 하나 이상의 조건이 명확히 불일치
- `unknown`: 페이지 변경·정보 부족·사진 모호성 때문에 확인 불가

성격, 친화성, 건강은 판정하지 않습니다. 색상은 공고 표기 또는 사진에서 분명한 경우만 확인합니다. 크기는 명시 크기 또는 체중을 사용하며 현재 평가 규칙과 같이 3kg 이하는 `tiny`, 3kg 초과 8kg 이하는 `small`, 8kg 초과 18kg 이하는 `medium`, 그 초과는 `large`로 고정합니다. 어린 개체는 측정일 기준 1세 이하, 노령은 8세 이상으로 고정합니다. 지역은 보호 지역의 광역자치단체를 사용합니다. 불명확한 값은 `unknown`이며 오답으로 바꾸지 않습니다.

같은 task·공고 쌍이 여러 번 선택되면 한 번 판정해 모든 선택 이벤트에 동일하게 적용합니다. 집계는 후보 선택 이벤트 기준 관련성 수와 중복 제거 후보 수를 모두 냅니다. 판정자는 참여자일 필요가 없으며 판정자 개인정보도 수집하지 않습니다.

## 실행

먼저 템플릿 자체를 검사합니다.

```powershell
conda run -n meong-contest-full python scripts/portal_candidate_selection_study.py validate-template
```

공고 최신화와 실행 버전 고정 후, 첫 참여자 전에 protocol을 freeze합니다. 먼저 `git status --short` 출력이 없는 clean commit인지 확인합니다.

```powershell
git status --short
$studyDate = Get-Date -Format yyyy-MM-dd
$sourceCommit = (git rev-parse HEAD).Trim()
conda run -n meong-contest-full python scripts/portal_candidate_selection_study.py freeze `
  --study-date $studyDate `
  --source-commit $sourceCommit
```

첫 trial 직전에 freeze된 템플릿·공고 스냅샷·FAISS 인덱스가 현재 파일과 같은지 확인합니다.

```powershell
conda run -n meong-contest-full python scripts/portal_candidate_selection_study.py check `
  --protocol data/reports/portal_candidate_selection/protocol.json
```

4인용 prefilled CSV를 만듭니다.

```powershell
conda run -n meong-contest-full python scripts/portal_candidate_selection_study.py prepare `
  --protocol data/reports/portal_candidate_selection/protocol.json `
  --participants 4
```

정적 헤더 템플릿은 `docs/evaluation/templates/portal_candidate_selection_trials.csv`와 `portal_candidate_selection_blind_labels.csv`에도 있습니다. 실제 측정에는 schedule이 이미 들어간 `prepare` 결과를 권장합니다.

사람 trial이 끝난 뒤 source-blind 판정지를 만듭니다.

```powershell
conda run -n meong-contest-full python scripts/portal_candidate_selection_study.py blind `
  --protocol data/reports/portal_candidate_selection/protocol.json `
  --trials data/reports/portal_candidate_selection/P01.csv data/reports/portal_candidate_selection/P02.csv data/reports/portal_candidate_selection/P03.csv data/reports/portal_candidate_selection/P04.csv
```

독립 판정자가 `relevance_label`을 모두 채운 뒤 집계합니다.

```powershell
conda run -n meong-contest-full python scripts/portal_candidate_selection_study.py aggregate `
  --protocol data/reports/portal_candidate_selection/protocol.json `
  --trials data/reports/portal_candidate_selection/P01.csv data/reports/portal_candidate_selection/P02.csv data/reports/portal_candidate_selection/P03.csv data/reports/portal_candidate_selection/P04.csv `
  --labels data/reports/portal_candidate_selection/blind_labels.csv `
  --json-out data/reports/portal_candidate_selection/result.json `
  --markdown-out data/reports/portal_candidate_selection/result.md
```

원본에서 결과를 다시 계산해 일치 여부를 확인합니다.

```powershell
conda run -n meong-contest-full python scripts/portal_candidate_selection_study.py check `
  --protocol data/reports/portal_candidate_selection/protocol.json `
  --trials data/reports/portal_candidate_selection/P01.csv data/reports/portal_candidate_selection/P02.csv data/reports/portal_candidate_selection/P03.csv data/reports/portal_candidate_selection/P04.csv `
  --labels data/reports/portal_candidate_selection/blind_labels.csv `
  --result data/reports/portal_candidate_selection/result.json `
  --markdown data/reports/portal_candidate_selection/result.md
```

## 허용되는 해석

결과가 실제로 모인 뒤에만 다음처럼 표현할 수 있습니다.

> 4명 고정 과제 파일럿에서 멍탐정 조건의 활성 후보 3건 완료율은 X/Y, 공식 포털은 A/B였습니다. 외부 장애를 제외하고 미완료를 cap으로 둔 참여자별 조건 중앙시간 차이와 source-blind 최종 후보 관련성을 함께 보고했습니다.

다음 표현은 금지합니다.

- 멍탐정이 입양 성공률이나 보호소 방문률을 높였다.
- 대한민국 사용자의 평균 탐색 시간을 줄였다.
- 성격·건강·아동 친화성·합사 적합성을 검증했다.
- 3~5명 기술 통계를 통계적으로 유의한 우월성으로 표현한다.
- 아직 사람 측정을 하지 않은 템플릿·테스트 결과를 실제 효용 개선으로 표현한다.
