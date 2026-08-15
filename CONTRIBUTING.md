# 멍탐정 기여 가이드

## 개발 환경

Python 3.10을 기준으로 합니다. 테스트와 정적 검사만 수행할 때는 GPU·VLM 패키지를 설치할 필요가 없습니다.

```bash
conda env create -f environment.yml
conda activate dog-rag
python -m pytest -q
```

이미 환경을 만든 경우 `conda env update -n dog-rag -f environment.yml --prune`으로 맞춥니다. 선택 Gemma/VLM 기능은 기본 기여 환경에 포함하지 않으며 필요할 때만 `requirements-vlm.txt`를 별도로 설치합니다.

## 변경 원칙

- 기존 텍스트·이미지 검색 API의 호환성을 보존합니다.
- 공고에 없는 성격·건강·사회성 정보를 추정하지 않습니다.
- unknown을 임의의 긍정·부정 사실로 변환하지 않습니다.
- 데이터·모델·샘플 이미지의 출처와 라이선스를 함께 기록합니다.
- 실제 API 키, 토큰, 개인 연락처와 비공개 데이터는 커밋하지 않습니다.
- 새 점수나 가중치는 회귀 테스트와 계산 근거를 함께 추가합니다.

## 제출 전 검사

```bash
python -m pytest -q
ruff check app scripts tests
python scripts/check_contest_readiness.py --metas data/dog_metas.json --index data/dog_faiss.index --strict
python scripts/smoke_full_runtime.py
```

데이터 readiness의 strict 실패를 무시한 채 운영 또는 대회 시연용 릴리스를 만들지 마세요.

릴리스 후보는 clean commit과 태그에서 다음 단일 게이트도 통과해야 합니다. `--skip-slow` 결과는 제출 PASS로 사용할 수 없습니다.

```bash
python scripts/verify_contest_release.py --required-tag <RELEASE_TAG>
```

## Pull request

PR에는 문제, 변경 이유, 사용자 영향, 검증 명령과 데이터·라이선스 영향을 적습니다. 기능 변경과 대규모 데이터 artifact 갱신은 가능한 한 별도 커밋으로 분리합니다.

별도로 명시하지 않는 한 프로젝트에 의도적으로 제출한 기여는 Apache License 2.0 조건으로 제공하는 데 동의한 것으로 봅니다. 제출자는 해당 기여를 제공할 권한이 있어야 하며 제3자 코드·데이터·모델을 포함할 때는 출처와 별도 조건을 밝혀야 합니다.
