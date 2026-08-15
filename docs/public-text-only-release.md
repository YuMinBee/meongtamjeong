# `public-text-only-v1` 배포 가이드

## 목적

`public-text-only-v1`은 공고 사진 관련 권리 노출을 최소화하기 위한 결정론적 배포
대안입니다. 권리 안전성이나 법적 이용 가능성을 판정하지 않습니다. 현재 또는 과거
Git 원격 이력에 공개된 샘플 사진과 사진 파생 artifact의 권리 상태나 회수 조치도
해결하지 않습니다.

현재 스냅샷에서는 활성 공고 1,516건의 `public_notice_text` 벡터 1,516개만 남기고,
사진 1,228개와 crop 1,002개 벡터를 제외합니다.

## 파생 계약

- source row는 `type=text`, `embedding_source=public_notice_text`여야 합니다.
- canonical 공개 공고 텍스트와 `desc_full`, `embedding_text_sha256`가 일치해야 합니다.
- 공고마다 검증된 text row가 정확히 하나 있어야 합니다.
- metadata와 manifest source는 고정 allowlist 밖의 필드를 제거합니다.
- 사진 파일, 원격 공고 사진 URL, image/crop 벡터, VLM·detector 속성, crop 경로와
  사진 품질 점수는 출력하지 않습니다.

```powershell
python scripts/build_public_text_release.py --output-dir dist/public-text-only
```

## 패키징

의도한 변경을 commit하고 annotated tag를 만든 clean HEAD에서 실행합니다. 아래 전체 게이트가 profile 적용 평가와 결정적 요약을 만든 뒤 같은 tag의 ZIP을 생성·검증합니다.

```powershell
python scripts/verify_contest_release.py --profile public-text-only --required-tag <RELEASE_TAG>
```

public ZIP은 canonical index·metadata·manifest를 text-only 파생물로 교체하고,
sample 디렉터리, full-profile 평가·문서·workflow와 적용 불가능한 테스트를 제외합니다.
루트 README, DATA_CARD, MODEL_CARD, CONTRIBUTING은 번들의 실제 계약으로 교체합니다.
게이트가 새 artifact에서 다시 실행한 retrieval·query robustness·exposure·profile·safety
결과는 `docs/evaluation/public-text-only.summary.json`과 `.md`로 포함합니다. 요약은
index·metadata SHA-256과 입력 SHA를 고정하고, 같은 tag에서 달라질 수 있는 실행 시각,
latency, runtime과 로컬 경로를 제외합니다. raw 평가 리포트는 ZIP에 포함하지 않습니다.

다른 기관 공고를 같은 수집 경로에 연결하는 [Provider adapter](provider-adapter.md)와
결과 없는 [공식 포털 후보 탐색 파일럿 프로토콜](evaluation/PORTAL_CANDIDATE_SELECTION_PROTOCOL.md)은
profile-neutral 오픈소스 도구이므로 가이드·프로토콜·템플릿·관련 테스트까지 함께
포함합니다. `data/examples/notice-provider.synthetic.json`의 URL은 연결 불가능한
`.invalid` 합성 문자열입니다. 실제 원격 사진 참조를 제외한다는 계약은 canonical
운영 `dog_metas.json`, `snapshot_manifest.json`, FAISS index에 적용되며 이 합성
contract fixture를 실제 공고 메타로 해석하지 않습니다.

## 런타임

marker/index/metas SHA-256, vector count, dimension과 metadata allowlist가 어긋나면
시작하지 않습니다. graph overlay, 원격 공고 사진, 사진 품질 점수와 공고
gallery/crop/audit route는 비활성화합니다. 사용자 업로드 이미지 검색은 호환성을
위해 유지하지만 공고 text vector와 비교하므로 full 시각 검색 품질을 주장하지
않습니다.

```powershell
python scripts/smoke_full_runtime.py --profile public-text-only --index dist/public-text-only/dog_faiss.index --metas dist/public-text-only/dog_metas.json --release-profile-marker dist/public-text-only/release_profile.json
```

## 평가 해석

full 멀티모달 artifact에서 만든 retrieval, query robustness, exposure, profile,
safety, held-out 리포트는 text-only 결과로 재사용하지 않습니다. 특히 held-out image
retrieval은 visual vector가 없는 프로필에서 적용 불가(`N/A`)이며 PASS로 세지
않습니다. 공고 상태는 연락 전 실제 공고 링크에서 다시 확인해야 합니다.

ZIP의 결정적 평가 요약에 적힌 명령은 요약 자체의 `reference_date`를 읽어 같은
기준일로 profile 적용 평가를 다시 만듭니다. 이 결과도 공고 사실 기반 silver label과
고정 계약 평가이며 사람의 주관적 외형 선호나 입양 성과를 입증하지 않습니다.

같은 이유로 full artifact에 고정된 query holdout v2/v3의 app·runner·입력·SHA
sidecar·리포트·테스트는 어느 한쪽만 남기지 않고 public-text ZIP에서 모두
제외합니다. 포털 파일럿은 실제 사람 결과가 아닌 템플릿만 포함하며, 실행할 때는
이 ZIP의 text-only index·metadata와 실행 commit을 첫 참여자 전에 새로 freeze해야
합니다.

GitHub URL을 출품물로 쓰려면 별도 text-only clean branch/repository를 만들거나
공고 사진 및 사진 파생 artifact의 재배포 범위를 서면으로 확인해야 합니다.
