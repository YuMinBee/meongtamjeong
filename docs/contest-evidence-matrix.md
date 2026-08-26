# 2026 오픈소스 개발자대회 제출 증거 매트릭스

> 기준일: **2026-07-26**
> 용도: 제출 직전 개발보고서·3분 영상·소스코드가 같은 release artifact를 설명하는지 확인하는 내부 통제 문서

## 1. 공개된 공식 요건

공개된 [NIPA 대회 공고](https://www.nipa.kr/home/bsnsAll/0/nttDetail?bbsNo=4&bsnsDtlsIemNo=58&nttNo=16815&tab=3)와 [오픈소스 포털 대회 페이지](https://www.oss.kr/pages/2)에서 확인되는 범위는 다음과 같다.

- 출품작 제출일: **2026-08-27**
- 제출물: 개발·결과보고서, **3분 이내** 시연영상, 소스코드 등 산출물
- 평가 흐름: 1차 서면평가 **30점** → 기능테스트 → 라이선스 검증 → 2차 발표평가 **70점**
- 오리엔테이션에서 참가자 숙지사항·평가기준을 공지한다고 안내됨

공개 페이지에는 세부 항목별 배점표와 제출 파일 형식이 보이지 않는다. 따라서 아래의 테스트 수, 정량평가, 보안 gate는 멍탐정이 스스로 정한 품질 기준이지 공식 세부 배점이라고 주장하지 않는다. 참가자가 받은 7월 23일 오리엔테이션 자료나 제출 양식이 있으면 그 원본을 최우선 기준으로 이 문서를 갱신한다.

## 2. 공식 단계와 현재 증거

| 공식 단계·제출물 | 현재 산출물 | 자동 검증 | 2026-07-26 상태 | 최종 완료 조건 |
| --- | --- | --- | --- | --- |
| 개발·결과보고서 | [`contest-development-report-draft.md`](contest-development-report-draft.md) | 링크된 JSON·Markdown 평가의 `--check` | 초안 | 공식 양식 반영, 최종 profile·tag·hash·파일럿 결과와 모든 기준일 일치 |
| 3분 이내 시연영상 | [`demo-video-script.md`](demo-video-script.md), [`demo-runbook.md`](demo-runbook.md) | readiness·runtime smoke | 대본 2분 55초, 영상 미제작 | 최종 export 3분 미만, 재생 확인, 권리·비밀·개인정보 프레임 검수 |
| 소스코드·산출물 | Git 저장소, [`package_release.py`](../scripts/package_release.py) | 추적 파일·비밀 패턴·annotated tag·clean HEAD·embedded manifest·ZIP SHA 검사 | 구현 완료, worktree dirty | YuMinBee 명의 clean release commit·annotated tag에서 실제 ZIP 생성·압축 해제 실행 |
| 기능테스트 | API·데모·평가·provider·release 도구 | pytest, Ruff, full/public runtime 및 release ZIP runtime smoke | 현재 PASS | fresh snapshot·clean clone·최종 tag에서 전체 gate 재실행 |
| 라이선스·SBOM 검증 | [`LICENSE`](../LICENSE), [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md), [`DATA_CARD.md`](../DATA_CARD.md), [`MODEL_CARD.md`](../MODEL_CARD.md), [`SBOM.md`](SBOM.md), SPDX·CycloneDX SBOM | dependency inventory·SBOM 생성·이력 비밀·package 검사 | 코드·75개 구성요소 SBOM과 reachable Git 이력 비밀검사 완료, 사진 파생물 권리는 미확인 | 권리 노출 최소화 `public-text-only-v1` 선택; clean release package 검사 PASS |
| 2차 발표 | 3분 동선, 예상 질문, 장애 대체 동선 | 발표 PC 수동 QA | 대본·Q&A 준비, 실기 QA 미완료 | 발표 해상도·키보드·네트워크 단절·외부 링크 실패 포함 리허설 완료 |

## 3. 심사 주장과 재현 증거

| 주장 | 대표 근거 | 허용되는 해석 | 금지되는 확대 해석 |
| --- | --- | --- | --- |
| 자연어 표현 변화에도 후보를 찾는다 | [최종 frozen v3](evaluation/query_holdout.appearance_v3.md): 24문장 P@5 76.67%, nDCG@5 79.15%, Hit@5 100%, inactive 0%; 하락 8건 공개 | 같은 snapshot의 공고 사실 silver qrels에 대한 one-shot 기술 지표 | 사람의 주관적 외형 만족도, 한국어 전반, 입양 성공률 |
| 각 검색 구성요소가 결과에 기여한다 | [단계별 ablation](evaluation/retrieval_eval.appearance_v1.md): CLIP → BM25 → 자연어 조건 → Graph | 같은 12질의·같은 qrels의 개발 회귀 비교 | 독립 미공개 성능 또는 각 단계의 인과 효과 일반화 |
| 생활·성격을 사진에서 추정하지 않는다 | [비추론 safety contract](evaluation/safety_contract.appearance_v1.md) 4/4 PASS와 테스트 | 고정 계약 입력에서 순위 격리·문의 질문 전환이 일관됨 | 모든 자유문장의 완전 차단, 실제 성격 판정 |
| 프로필에 따라 확인 순서가 바뀐다 | [프로필 재정렬 계약](evaluation/profile_rerank.appearance_v1.md) | 같은 후보군에서 크기·연령·지역의 규칙 구현이 일관됨 | 최적 가중치, 사용자 효용, 입양 적합성 |
| 이미지 검색 기능이 동작한다 | [교차사진 평가](evaluation/heldout_image_retrieval.appearance_v1.md): 최초 120건, 평가 가능 29건, 조건부 Hit@5 96.55%, end-to-end 23.33%, `partial` | 같은 공고의 다른 사진을 찾는 제한된 대리 과제 | 전체 이미지 검색 정확도 96.55%, 서로 다른 개의 닮음 평가 |
| 실제 후보 탐색 비용을 줄일 수 있다 | [공식 포털 비교 프로토콜](evaluation/PORTAL_CANDIDATE_SELECTION_PROTOCOL.md) | 현재는 검증 가능한 제품 가설과 사전 정의된 측정 방법 | 참여자 결과가 생기기 전 시간 절감·효용 입증 주장 |
| 다른 기관도 연결할 수 있다 | [`provider-adapter.md`](provider-adapter.md), 합성 fixture와 contract test | provider별 필드 매핑을 구현하면 공통 상태·정규화 흐름 재사용 가능 | 모든 기관에 무수정 연결 가능 |

대표 성과 슬라이드는 개발용 12질의의 가장 높은 수치만 떼지 않고 **최종 frozen v3의 수치와 실패 8건**을 함께 사용한다. 이미지 수치는 `29/120` 분모와 `partial` 상태를 함께 말할 수 있을 때만 보조 근거로 사용한다.

## 4. 최종 release seal

아래 값이 하나라도 비어 있으면 보고서·영상·ZIP을 최종 제출본으로 부르지 않는다.

| 항목 | 최종 값 |
| --- | --- |
| Release tag / tag object / commit | `[TAG]` / `[TAG_OBJECT_SHA]` / `[COMMIT_SHA]` |
| Release profile | `[full | public-text-only-v1]` |
| Snapshot 기준일 / 마지막 공고 종료일 | `[YYYY-MM-DD]` / `[YYYY-MM-DD]` |
| ZIP SHA-256 / embedded manifest | `[SHA256]` / `[release/manifest.json 검증 PASS]` |
| Index / metas SHA-256 | `[SHA256]` / `[SHA256]` |
| Profile 적용 평가 요약 | `[full reports | public-text-only.summary SHA256]` |
| 개발보고서 SHA-256 | `[SHA256]` |
| 최종 영상 SHA-256 / 재생시간 | `[SHA256]` / `[MM:SS]` |
| Clean-clone release gate | `[PASS, UTC timestamp]` |
| Dependency snapshot / Pytest / Ruff / runtime smoke | `[75/75 PASS]` / `[N passed]` / `[PASS]` / `[source+ZIP PASS]` |
| 측정 환경 | `[OS, CPU, GPU 또는 CPU-only, RAM, Python]` |
| 사람 파일럿 | `[N 및 report hash | 미실시라고 명시]` |
| 사진·파생물 권리 근거 | `[응답 문서 | public-text-only manifest]` |
| 과거 비밀키 조치 | `[폐기 확인 및 이력 검사 PASS]` |

## 5. 출품 직전 단일 판정

1. 사진·crop·사진 파생 벡터의 공개 재배포 근거가 있으면 `full`을 선택한다.
2. 근거를 받지 못하면 이력이 없는 공개 저장소에서 `public-text-only-v1`을 선택한다. 기존 full 평가·화면·영상 수치는 재사용하지 않는다.
3. 선택한 profile의 fresh snapshot에서 평가와 문서를 다시 만들고, 영상도 같은 tag로 촬영한다.
4. [`verify_contest_release.py`](../scripts/verify_contest_release.py)의 생략 없는 전체 gate와 clean-clone 검증을 통과한다.
5. 이 문서의 release seal을 채운 뒤 개발보고서·영상·ZIP의 hash를 대조한다.

현재 가장 큰 외부 의존 항목은 사진 파생물 권리 확인과 실제 사람 파일럿이다. 가장 큰 로컬 운영 항목은 fresh snapshot, clean commit, clean-clone 검증과 영상 제작이다.
