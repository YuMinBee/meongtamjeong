# 멍탐정 대회 릴리스 런북

이 문서는 출품 ZIP과 발표 데모가 **같은 검증된 검색 artifact**를 사용하도록 만드는 운영 절차입니다. 현재 스냅샷의 가장 늦은 `notice_end`는 `2026-08-07`입니다. 따라서 이를 갱신하지 않고 `2026-08-27` 제출일에 당일 기준 strict 검사를 실행하면 active 공고가 0건이 되어 실패하는 것이 정상입니다.

## 최소 갱신 일정

| 기간 | 목적 | 완료 조건 |
| --- | --- | --- |
| 2026-08-20~26 | 8월 27일 제출 직전 스냅샷 | 새 공고 동기화, 개발 평가 6종 재생성, 새 blind holdout 동결·one-shot, clean-HEAD 전체 게이트, 제출 ZIP 해시 기록 |
| 2026-10-09~11 | 본선·발표 준비 스냅샷 | 같은 절차 반복, 실제 발표 PC에서 수동 QA |
| 2026-11-02~03 | 최종 시연 직전 스냅샷 | 같은 절차 반복, 오프라인 백업과 현장용 ZIP 고정 |

공고는 수시로 종료될 수 있으므로 위 세 번은 최소 횟수입니다. 각 발표 당일에는 공고 링크와 문의 카드의 최신 상태도 다시 확인합니다.

## 0. 사람 확인이 먼저 필요한 항목

- 배포 서버는 `APP_ENV=contest`로 실행하고 서비스 접근용 `API_KEY`를 임의의 24자 이상 값으로 설정합니다. 공공데이터 조회용 `ANIMAL_API_KEY`는 별도 비밀이며 저장소, 명령행, 로그, 화면 녹화에 넣지 않습니다.
- 공공 API의 이용허락 표시만으로 개별 사진 촬영자의 재배포 권리까지 자동 확정되지 않습니다. 제출 ZIP에 포함하는 샘플 사진과 임베딩의 재배포 근거를 확인하고, 확인할 수 없으면 사진을 제외하거나 권리가 분명한 자료로 교체합니다.
- Git 과거 이력에서 발견된 키가 실제 키였는지 소유자가 확인합니다. 실제 키라면 먼저 폐기·재발급합니다. 이력 재작성은 기존 commit·tag·ZIP 해시를 모두 바꾸고 공동 작업자의 재복제를 요구하므로 담당자가 범위와 공지를 결정한 뒤 수행해야 합니다.
- 후보 선택 시간 실험과 블라인드 관련성 평가는 실제 참여·측정 전에는 성능 향상 근거로 쓰지 않습니다. 브라우저, 발표 해상도, 키보드 탐색, 네트워크 장애, 이미지 실패 대체 UI는 사람이 직접 확인합니다.

## 1. 격리된 임시 출력으로 갱신

검증된 Conda 환경과 저장소 루트에서 시작합니다. 먼저 기존 `data/dog_faiss.index`, `data/dog_metas.json`, `data/active_index_sync_report.json`, `data/snapshot_manifest.json`을 저장소 밖의 날짜별 백업 폴더에 복사하고 SHA-256을 기록합니다. 아래 `<TEMP_DIR>`는 저장소 밖의 새 폴더, `<YYYY-MM-DD>`는 실제 수집일입니다. 비밀 키 값은 명령 인자로 넘기지 않습니다.

```text
python scripts/fetch_live_dogs.py --days 365 --out <TEMP_DIR>/fresh_dogs.json

python scripts/reuse_image_enrichment.py \
  --fresh-cache <TEMP_DIR>/fresh_dogs.json \
  --existing-metas data/dog_metas.json \
  --output <TEMP_DIR>/fresh_dogs_reused.json

python scripts/sync_active_index.py \
  --fresh-cache <TEMP_DIR>/fresh_dogs_reused.json \
  --existing-index data/dog_faiss.index \
  --existing-metas data/dog_metas.json \
  --index-out <TEMP_DIR>/dog_faiss.index \
  --metas-out <TEMP_DIR>/dog_metas.json \
  --report-out <TEMP_DIR>/active_index_sync_report.json \
  --reference-date <YYYY-MM-DD>
```

`fetch_live_dogs.py`는 `.env`의 `ANIMAL_API_KEY`를 읽습니다. 정식 갱신에서는 365일보다 짧은 조회 범위를 쓰지 않습니다. 365일보다 오래 보호 중인 공고가 누락될 가능성도 있으므로 API 페이지 완주 여부, 이전 manifest 대비 active 공고 수, 제거 비율, 사진·텍스트 임베딩 실패 수를 사람이 확인합니다. `sync_active_index.py`는 기본적으로 `openapi.animal.go.kr` 이미지만 요청하고 redirect를 따르지 않습니다. 다른 공개 provider를 쓰는 경우에만 검토한 정확한 DNS 호스트를 `--allowed-image-host`로 추가합니다. 기본 제거 상한은 0.5이고 최소 active 수 기본값은 1이므로, 릴리스 담당자는 이전 스냅샷 규모를 근거로 `--min-active-notices`를 더 엄격하게 정해 실행할 수 있습니다. 단지 통과시키기 위해 안전 한도나 이미지 호스트 제한을 낮추지 않습니다.

## 2. 임시 artifact strict 검증과 수동 승격

운영 파일을 건드리기 전에 임시 결과를 당일 기준으로 검사합니다.

```text
python scripts/check_contest_readiness.py \
  --metas <TEMP_DIR>/dog_metas.json \
  --index <TEMP_DIR>/dog_faiss.index \
  --reference-date <YYYY-MM-DD> \
  --max-age-days 0 \
  --strict
```

다음 항목을 모두 검토합니다.

1. 임시 동기화 report의 safety gate, active ID, text-vector 누락, 사진 실패, 제거·신규·갱신 수
2. FAISS 차원·행 수·L2 형식, 메타 행 정렬, non-finite/zero/non-unit 벡터, 금지 파생 필드
3. 입력·출력 SHA-256과 canonical content hash, 수집 시각, 조회 범위, 모델·런타임 provenance
4. 무작위 실제 공고 링크와 공공 API 원문 비교

현재 저장소에는 검토 완료된 임시 파일을 운영 파일로 자동 승격하는 helper가 없습니다. 따라서 이 런북은 위험한 덮어쓰기 명령을 제공하지 않습니다. 담당자가 백업과 검증 결과를 재확인하고, 가능하면 독립 검토자에게도 확인받은 뒤 파일 관리자에서 임시 index·metas·sync report를 각각의 추적 경로로 수동 교체합니다. 이어서 `data/snapshot_manifest.json`의 시각, 통계, 모델, 정책, 모든 SHA-256을 실제 새 파일에 맞게 갱신합니다. 교체 직후 동일한 strict 명령을 이번에는 `data/dog_metas.json`과 `data/dog_faiss.index`에 다시 실행합니다. 실패하면 백업으로 되돌리고 원인을 조사하며 부분 교체 상태를 커밋하지 않습니다.

## 3. 같은 artifact에서 평가 6종 재생성

검색 서비스와 평가는 반드시 방금 승격한 동일 index·metas를 사용해야 합니다. 새 기준일에 맞춰 `data/eval_queries.appearance_v1.json`, `data/eval_query_variants.appearance_v1.json`, `data/eval_profiles.appearance_v1.json`, `data/eval_safety_contract.appearance_v1.json`, held-out 평가 기본 기준일, manifest와 문서의 고정 기준일을 검토합니다. `.github/workflows/ci.yml`과 `.github/workflows/full-runtime-smoke.yml`의 fixed reference date도 함께 바꿉니다. CI 날짜만 바꾸거나 보고서만 재사용하지 않습니다.

```text
python scripts/evaluate_retrieval.py
python scripts/evaluate_query_robustness.py
python scripts/evaluate_profile_reranking.py
python scripts/evaluate_safety_contract.py
python scripts/evaluate_exposure_representation.py
python scripts/evaluate_heldout_image_retrieval.py --run-network
python scripts/portal_candidate_selection_study.py validate-template
```

위 6개 개발 평가와 별도로, 현재 추적된 보강 전 v2와 최종 독립 v3 one-shot의 무결성을 확인합니다. 새 snapshot에 일반화 수치를 귀속하려면 기존 holdout을 다시 실행하지 말고, 시스템을 보지 않은 작성자가 새 문구를 먼저 동결한 다음 새 버전 holdout을 정확히 한 번 실행합니다.

```bash
python scripts/evaluate_query_holdout.py --check
python scripts/evaluate_query_holdout_v3.py --check
```

Held-out 평가는 명시적 네트워크 실행이며 다운로드 성공률과 평가 가능 표본 수가 달라질 수 있습니다. 조건부 Hit@K와 전체 계획 표본 기준 end-to-end Hit@K를 구분해 기록하고, 실패·중복 제외를 숨기지 않습니다. 생성 후에는 개발 평가 여섯 명령을 각각 `--check`로 다시 실행하고 portal 템플릿은 `validate-template`로 확인합니다.

## 4. 릴리스 후보 확정과 전체 게이트

`verify_contest_release.py`는 현재 날짜 기준 7일 freshness, 개발 평가 6종, 역사적 v2와 최종 독립 v3의 무결성, 결과 없는 portal pilot 프로토콜, Git 전체 이력 비밀 검사, clean-HEAD 패키지 검사, 정확한 75개 의존성 스냅샷, Ruff, 전체 pytest, 실제 CLIP/FAISS runtime smoke, 마지막 Git clean 상태를 모두 확인합니다. `--required-tag`를 주면 실제 제출 ZIP을 만들고 embedded manifest의 tag·commit·profile과 모든 파일의 SHA-256을 검증한 뒤 안전한 임시 경로에 풀어 ZIP 내부 runtime smoke도 실행합니다. 각 자식 명령의 원문 출력은 저장하거나 재출력하지 않고 단계명·exit code·시간만 JSON으로 남깁니다.

```text
python scripts/verify_contest_release.py
```

`--skip-slow`는 pytest와 full runtime smoke만 생략하는 개발 사전점검입니다. 결과는 항상 `mode=preflight`, `ready=false`이고 종료 코드도 0이 아니므로 제출 PASS 근거로 사용할 수 없습니다.

승격과 평가 재생성 직후 태그 없이 위 게이트를 한 번 실행해 모든 독립 단계를 먼저 확인합니다. 이 단계는 실제 ZIP identity가 없으므로 의도적으로 `mode=preflight`, `ready=false`입니다. 패키지 검사와 `git_clean`은 clean HEAD를 요구하므로 아직 커밋하지 않은 올바른 변경 때문에 실패하는 것도 정상입니다. 그 실패를 모두 고치고 결과를 검토한 뒤 Git author·committer가 YuMinBee인지 확인해 릴리스 후보 commit을 만들고, clean commit에서 태그 없는 전체 preflight를 다시 통과시킵니다.

그 다음 **annotated** release tag를 만들고 tag가 정확히 HEAD를 가리키는지 확인한 뒤 아래 최종 게이트를 실행합니다. 이 실행이 같은 tag의 최종 ZIP까지 생성·검증하며 `mode=release`, `ready=true`, 종료 코드 0이어야 합니다. 실패하면 기존 tag를 옮겨 덮지 않고 수정 commit과 새 tag로 절차를 반복합니다. `--allow-dirty` 같은 우회 옵션은 릴리스 절차에 사용하지 않습니다.

```text
python scripts/verify_contest_release.py \
  --required-tag <RELEASE_TAG> \
  --json-out dist/contest-release-gate.json
```

게이트 JSON의 `release_identity`에 기록된 archive path·ZIP SHA-256·commit·tag·annotated tag object ID와 사용한 Conda 환경을 릴리스 기록에 남깁니다. ZIP을 다른 폴더로 복사한 뒤에도 SHA-256이 같은지 확인합니다. tag 이후 코드·데이터·문서가 하나라도 바뀌면 기존 PASS와 ZIP을 폐기하고 새 commit·tag로 전체 절차를 반복합니다.

마지막으로 [`contest-evidence-matrix.md`](contest-evidence-matrix.md)의 release seal에 tag·commit·profile·snapshot·ZIP·index·metas·보고서·영상 hash와 검증 시각을 채웁니다. 개발보고서, 영상 자막과 제출 ZIP이 이 한 release identity를 가리키지 않으면 제출하지 않습니다.

## 5. 발표 직전 수동 QA

- 실제 발표 PC의 깨끗한 clone과 `requirements.lock.txt` 환경에서 서버를 시작하고 텍스트·이미지 검색, 프로필 외형 재정렬, 문의 카드와 실제 공고 링크를 확인합니다.
- 잘못된 서비스 키, 키 없음, 24자 미만 키, 종료 공고, 외부 API timeout, 이미지 404, 네트워크 단절이 안전하게 실패하는지 확인합니다.
- 화면 녹화와 현장 시연에는 `.env`, 요청 헤더, 공공 API 키, 로컬 경로가 노출되지 않게 합니다.
- 검증된 ZIP, 모델 캐시, 발표 자료, 짧은 녹화 데모를 오프라인 매체에 준비하되, 권리가 확인되지 않은 사진 원본과 수집 cache는 포함하지 않습니다.
- 사람 실험 결과, 사진 권리, 과거 키 조치, 브라우저 수동 QA가 끝나지 않았다면 자동 게이트가 녹색이어도 해당 항목을 완료로 주장하지 않습니다.
