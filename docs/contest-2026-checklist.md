# 2026 오픈소스 개발자대회 제출 체크리스트

## 공식 일정

- 참가 접수: 2026-06-15 ~ 2026-07-17 23:59
- 오리엔테이션·세부 평가기준 공지: 2026-07-23
- 출품작 제출: 2026-08-27
- 1차 서면평가: 2026-09-03
- 멘토링: 2026-09-18 ~ 2026-10-09
- 기능·라이선스 검증: 2026-10-12 ~ 2026-10-28
- 2차 발표평가: 2026-11-04

공식 안내:

- https://www.nipa.kr/home/bsnsAll/6/nttDetail?bbsNo=4&nttNo=16815&tab=2
- https://www.oss.kr/pages/2

공식 제출 단계와 현재 증거의 1:1 대응은 [제출 증거 매트릭스](contest-evidence-matrix.md)에서 관리합니다. 공개 페이지에 없는 세부 배점은 공식 기준으로 추정하지 않으며, 7월 23일 오리엔테이션 원본 자료를 확보하면 매트릭스와 최종 보고서 형식을 먼저 갱신합니다.

## 참가 접수

- [ ] 학생부문 / 자유과제 / 인공지능 선택 확인
- [ ] 사회문제해결 프로젝트 / 생활 분야 선택 확인
- [ ] 개발 목적·프로젝트 소개·기대효과 최종 저장
- [ ] 접수 완료 화면과 접수번호 보관
- [ ] GitHub 저장소 URL과 기본 브랜치 확인

## P0: 출품 전에 반드시 해결

- [x] 최상위 프로젝트 `LICENSE` 확정 (Apache-2.0, YuMinBee)
- [x] Ultralytics 직접 의존성을 제거하고 TorchVision Faster R-CNN으로 교체
- [x] 깨끗한 환경 75개 패키지의 실제 버전·라이선스 메타데이터와 원문 파일 재검증 (누락·미상 0건)
- [x] 출처 불명 AI 샘플 제거 및 공공데이터 공고 샘플 provenance manifest 추가
- [x] 공고 메타데이터·텍스트의 공식 API `제한 없음` 표시와 출처 기록
- [ ] [확인 문안](data-rights-inquiry-template.md)으로 샘플 사진/crop과 사진·crop 파생 FAISS 벡터의 공개 재배포 범위를 제공기관에 확인하고 응답 보존(미확인 시 해당 artifact를 제외하고 공식 API 기반 로컬 재생성 방식으로 전환)
- [x] 제출 ZIP 후보의 비밀키·토큰 패턴 검사 및 `.env`·가중치 제외 자동화
- [ ] Git 전체 기록의 과거 비밀키 검사 및 발견 시 키 폐기 확인
- [x] 2026-07-26 active-only 공고 스냅샷 재생성 (활성 1,516건)
- [ ] 출품·시연 직전 공공 API 상태 재확인 및 필요 시 스냅샷 재갱신
- [x] 공고 상태·종료일·지역·보호소·마지막 확인 시각 커버리지 기록
- [x] FAISS `ntotal`과 메타 행 수·순서 일치 검증 (3,746행)
- [x] closed/expired 공고가 Top-K에 노출되지 않는 회귀 테스트
- [x] 제출 데모와 README·Model Card의 권장 경로를 `ranking_scope=appearance`로 통일
- [x] 권장 `appearance`에서 주거·부재·활동·경험·아동·다른 동물·선호 성격을 후보 순위에서 제외하고 문의 질문으로만 사용
- [x] 365일 authoritative cache 수집 및 이전 manifest 대비 공고 수·페이지 완주 확인 (61페이지, 원시 60,085건)
- [x] [active 동기화 report](../data/active_index_sync_report.json)의 유지·삭제·갱신·신규 통계와 입력·출력 SHA-256 검증
- [x] 제출 메타에서 VLM·생성 행동·성격 파생필드 0건 확인
- [x] 동일 사진 URL·동일 텍스트 hash만 유지하고 변경 URL의 기존 image/crop을 제거한 동기화 결과 기록
- [x] fail-closed gate와 별도 임시 출력의 strict readiness·artifact integrity 통과 후 FAISS·메타·manifest 교체
- [x] 모든 원격 공고 이미지 CLI를 정확한 호스트 allowlist·redirect 금지·16MiB·2,500만 화소·실제 포맷·decompression bomb 제한의 공용 downloader로 통일
- [ ] clean clone에서 한 명령 설치·실행 검증
- [x] 깨끗한 CPU 환경에서 외부 공공 API 키 없이 텍스트·합성 이미지 검색과 문의 문구 smoke 통과
- [x] 깨끗한 75개 패키지 검증 환경에서 전체 자동 회귀 569개와 Ruff PASS, full 3,746벡터와 public-text-only 1,516벡터 실제 CPU CLIP/FAISS smoke PASS
- [x] 정확한 75개 의존성·CLIP Git provenance를 보고서와 대조하는 실패 폐쇄 검사
- [x] public-text-only에서 적용 가능한 평가를 다시 실행하고 artifact SHA에 묶은 결정적 요약을 ZIP에 포함(full holdout·교차사진은 N/A)
- [x] [현재 날짜 freshness부터 평가·이력 비밀·clean package·실제 tagged ZIP 생성·manifest·압축 해제 runtime·전체 테스트까지 한 번에 검사하는 릴리스 게이트와 운영 런북](contest-release-runbook.md)
- [x] 공식 제출물·평가 단계와 저장소 증거·허용 주장·최종 release hash를 대조하는 [제출 증거 매트릭스](contest-evidence-matrix.md)
- [x] 오염된 시스템 전체 lock을 2026-07-26 깨끗한 Windows Conda 검증 환경의 `requirements.lock.txt`로 교체
- [x] `last_verified_at` 형식과 시연 기준 최대 경과일 검증 추가

## 평가 산출물

- [x] [동일 후보군을 서로 다른 외형 프로필로 재정렬하는 결정적 계약 사례](evaluation/profile_rerank.appearance_v1.md)
- [x] [동일 active corpus에서 CLIP 단독 / +BM25 / +구조조건 / +Graph 단계별 ablation](evaluation/retrieval_eval.appearance_v1.md)
- [x] [개발셋 48개 변형과 부정 표현 12개 회귀](evaluation/query_robustness.appearance_v1.md): v2 실패 후 보강한 현재 P@5 90.00%, Hit@5 100%, Jaccard@10 100%; 독립 성능으로 사용 금지
- [x] [보강 전 역사적 frozen one-shot v2 holdout](evaluation/query_holdout.appearance_v2.md): 12개 P@5 48.33%, Hit@5 66.67%, Jaccard@10 21.99%; 실패 발견 결과를 삭제·재실행하지 않고 보존
- [x] [최종 독립 frozen one-shot v3 holdout](evaluation/query_holdout.appearance_v3.md): 24개 P@5 76.67%, nDCG@5 79.15%, Hit@5 100%, Jaccard@10 57.78%; 실버 지표 하락 8/24, 부정·미지원 진단 6/6 PASS, inactive 0%
- [x] 동일 후보군에 대한 크기·연령·지역 `appearance` 재정렬 계약 산출물 (독립 성능평가 아님)
- [x] `Precision@5`, `Recall@K`, `nDCG@5`, MRR, Hit@5
- [x] inactive 노출률
- [x] 결정적 프로필 계약 후보에서 입양 적합성 확정 문구 탐지 0건
- [x] `applicable_count`, `evaluated_count`, `evidence_coverage`
- [x] 지역·보호소·크기·연령·사진 품질별 노출 차이
- [x] 사진 품질 가중치 `0.00`/`0.05`의 고정 후보 순위 민감도와 quality unknown 상대효과 기록
- [x] [인덱스 미사용 두 번째 사진 120건의 동일 공고 시각 검색 평가](evaluation/heldout_image_retrieval.appearance_v1.md): primary 85건, secondary 55건, 중복 26건 제외, 평가 가능 29건; 조건부 Hit@5 96.55%, end-to-end Hit@5 23.33%, partial coverage 공개
- [ ] 독립 검수자 blind human relevance 라벨 수집·집계
- [x] [보호소 방문 후보 3개 선택 시간의 로컬·블라인드·균형 파일럿 도구와 사전 고정 프로토콜](evaluation/CANDIDATE_SELECTION_PROTOCOL.md)
- [x] [공식 포털 대 멍탐정의 end-to-end 후보 탐색 비교 프로토콜·교차 배정·freeze·blind 판정·집계 도구](evaluation/PORTAL_CANDIDATE_SELECTION_PROTOCOL.md) (실제 사람 결과 없음)
- [ ] 3~5명의 실제 참여자 결과 수집·집계(공식 포털 대비 완료율·capped 중앙시간·source-blind 후보 관련성·실패 유형 함께 보고)
- [x] 검색평가 수치를 재생성하는 명령, 고정 쿼리, 전체 JSON/Markdown
- [x] [appearance 질의 동치, 생활정보 순위 불변·미입력 생활필드 무주입, 문의 질문 변화, 비외형 표현 누출 safety-contract](evaluation/safety_contract.appearance_v1.md)
- [x] safety-contract 4/4 통과와 입력 메타·설정 SHA-256 기록
- [x] README 사회문제 통계에 출처 링크·조사연도·표본 명시
- [x] legacy Gemma 프롬프트를 적합성 단정이 아닌 탐색 근거·확인 필요 사항 요약으로 수정

## 사용자 화면

- [x] 공고 상태와 마지막 확인 시각을 카드 최상단에 표시
- [x] 확인된 일치 / 주의 / 정보 없음 / 보호소 질문을 분리
- [x] 기본 데모 검색 요청·응답에는 사용자가 고른 외형 선호만 포함하고 생활정보는 문의 단계에서만 선택 전송
- [x] `contest`·`production`에서 레거시 실험 UI를 무인증 `/demo` 리디렉트하고 브라우저 키 영구 저장 제거
- [ ] 선택 연구 화면을 공개할 경우 VLM 외형 정보는 공고 사실이 아니라 사진 관찰값으로 표시
- [x] 최종 점수를 입양 적합도 확률처럼 표현하지 않음
- [x] 서버측 공고 ID 대조 후 active 상태에만 직접 전화·메일 링크 제공
- [x] 실제 공고·전화·지도 링크와 복사 가능한 보호소 전화/이메일·문의폼 문안 제공
- [x] 자동 전송·공식 입양 신청·입양 확정이 아님을 화면과 API에 표시
- [ ] `/demo` 전체 흐름의 브라우저별 수동 시각 QA

## 8월 제출 패키지

- [ ] 개발보고서
- [ ] 3분 이내 시연영상
- [ ] 공개 소스코드와 릴리스 태그
- [x] README 빠른 시작
- [x] Data Card / Model Card / Third-party notices 초안
- [x] 외형 검색 평가 데이터와 결과 리포트
- [ ] 오프라인 백업 데모와 시연용 고정 스냅샷
- [x] 깨끗한 검증 환경의 전체 설치 패키지 75개 인벤토리·라이선스 목록(의존성 그래프 없는 inventory이며 CycloneDX/SPDX SBOM 주장은 하지 않음)
- [x] 잠금 파일 `pip-audit` 재검사에서 자동 확인 가능한 알려진 취약점 0건(`pytest` 수정 후); VCS CLIP 1건은 자동 점검 제외로 별도 고지
- [ ] 출품·발표 태그 직전 dependency audit 재실행 및 신규 finding triage

## 현재 차단 사항

- 프로젝트 라이선스는 Apache-2.0으로 확정했으며 데이터·사진·모델의 별도 조건을 함께 고지해야 합니다.
- 2026-07-26 active-only 스냅샷은 strict readiness와 artifact integrity를 통과했지만 공고 상태는 계속 바뀌므로 출품·시연 직전에 다시 확인해야 합니다.
- 사진 다운로드·갱신 실패 288건은 공개 텍스트로 검색할 수 있으나 이미지 검색 커버리지는 낮습니다.
- 증분 임베딩은 로컬 GPU의 `openai-clip 1.0.1`로 실행했고 제출 runtime은 공식 OpenAI CLIP commit을 고정했습니다. 두 provenance를 같다고 주장하지 않습니다.
- 공고 사실 기반 자연어 평가는 silver label입니다. 독립 v3도 동일 저장소·24개 수작업 한국어 one-shot 범위이며 사람의 주관적 외형 관련성을 대신하지 않습니다. 두 번째 사진 평가는 같은 공고를 다시 찾는 시각적 동일성 대리 과제이고, blind human relevance와 공식 포털 대비 실제 후보 선택 시간 평가는 아직 없습니다.
- 공식 API는 공고 텍스트·메타데이터의 이용허락범위를 `제한 없음`으로 표시합니다. 다만 촬영자·별도 라이선스가 표시되지 않은 샘플 사진/crop과 사진·crop 파생 FAISS 벡터는 제공기관 확인 응답이 아직 없으며, 확인하지 못하면 해당 artifact를 릴리스에서 제외하고 공식 API 기반 로컬 재생성 방식으로 전환해야 합니다.
- 공식 포털 대비 실제 후보 선택 파일럿, 수동 시각 QA, 개발보고서·시연영상·릴리스 태그가 아직 남아 있습니다.
