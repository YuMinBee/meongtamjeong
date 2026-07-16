# 멍탐정 기여 가이드

## 개발 환경

Python 3.10을 기준으로 합니다. 테스트와 정적 검사만 수행할 때는 GPU·VLM 패키지를 설치할 필요가 없습니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Windows PowerShell에서는 `.venv\Scripts\Activate.ps1`을 사용합니다. 전체 애플리케이션 실행은 `requirements.txt` 또는 `environment.yml`을 참고하세요.

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
ruff check app/profile_rerank.py app/notice_status.py app/graph_rag.py scripts/fetch_live_dogs.py scripts/merge_dog_metadata.py scripts/check_contest_readiness.py tests
python scripts/check_contest_readiness.py --metas data/dog_metas.json --index data/dog_faiss.index --strict
```

데이터 readiness의 strict 실패를 무시한 채 운영 또는 대회 시연용 릴리스를 만들지 마세요.

## Pull request

PR에는 문제, 변경 이유, 사용자 영향, 검증 명령과 데이터·라이선스 영향을 적습니다. 기능 변경과 대규모 데이터 artifact 갱신은 가능한 한 별도 커밋으로 분리합니다.
