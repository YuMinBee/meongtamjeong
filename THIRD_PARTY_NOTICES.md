# 제3자 소프트웨어·모델 고지

이 문서는 멍탐정이 직접 포함하거나 실행 시 연동하는 주요 제3자 구성요소를 정리한 감사 목록입니다. 각 배포 전에 실제 설치 버전의 원문 라이선스와 의존성 트리를 다시 확인해야 합니다.

| 구성요소 | 역할 | 알려진 라이선스/약관 | 배포 메모 |
| --- | --- | --- | --- |
| FastAPI | HTTP API | MIT | Python 패키지 |
| Uvicorn | ASGI 서버 | BSD-3-Clause | Python 패키지 |
| Pydantic | 요청·응답 스키마 | MIT | Python 패키지 |
| NumPy | 수치 연산 | BSD-3-Clause | Python 패키지 |
| Pillow | 이미지 처리 | HPND | Python 패키지 |
| Requests | HTTP 클라이언트 | Apache-2.0 | Python 패키지 |
| python-dotenv | 환경변수 파일 로딩 | BSD-3-Clause | Python 패키지 |
| python-multipart | multipart 요청 파싱 | Apache-2.0 | Python 패키지 |
| PyTorch / torchvision | 모델 실행·이미지 전처리 | BSD-style | Python 패키지 |
| OpenAI CLIP | 이미지·텍스트 임베딩 | MIT | 코드·가중치 출처와 버전을 릴리스 manifest에 기록 |
| ftfy | CLIP 텍스트 정규화 | Apache-2.0 | Python 패키지 |
| regex | CLIP 토큰화 보조 | Python 1.6/CNRI-derived + Apache-2.0 | 포함된 `LICENSE.txt` 기준 |
| tqdm | 진행률 표시 | MPL-2.0 AND MIT | Python 패키지 메타데이터 기준 |
| FAISS | 벡터 검색 | MIT | Python 패키지/인덱스 |
| Hugging Face Transformers | 선택적 Gemma 실행 | Apache-2.0 | Python 패키지 |
| Accelerate | 선택적 모델 로딩 | Apache-2.0 | Python 패키지 |
| SentencePiece | 선택적 토크나이저 | Apache-2.0 | Python 패키지 |
| HTTPX | API 테스트 클라이언트 | BSD-3-Clause | 개발 의존성 |
| pytest | 자동 테스트 | MIT | 개발 의존성 |
| Ruff | 정적 검사·포맷 | MIT | 개발 의존성 |
| Ultralytics YOLO | 선택적 객체 영역 추출 | AGPL-3.0 또는 Enterprise | 현재 프로젝트 라이선스와의 호환 결정을 출품 전 완료해야 함 |
| Google Gemma 3 | 선택적 생성·VLM 모델 | Gemma Terms of Use | 가중치는 저장소에 포함하지 않음. 사용자가 별도 약관에 동의해 준비 |

주요 원문:

- OpenAI CLIP: https://github.com/openai/CLIP
- FAISS: https://github.com/facebookresearch/faiss
- Ultralytics 라이선스 안내: https://docs.ultralytics.com/help/contributing
- Gemma Terms of Use: https://ai.google.dev/gemma/terms
- Gemma Prohibited Use Policy: https://ai.google.dev/gemma/prohibited_use_policy

## 프로젝트 라이선스 결정 필요

현재 코드 저장소에는 최상위 `LICENSE`가 없습니다. Ultralytics YOLO를 현재 형태로 유지해 배포한다면 공식 안내의 AGPL-3.0 의무를 기준으로 프로젝트 전체 라이선스 호환성을 검토해야 합니다. permissive 라이선스를 원한다면 객체 탐지 구성요소를 호환 가능한 대안으로 교체한 뒤 다시 감사해야 합니다.

## 모델 배포

Gemma 모델 파일과 파생 가중치는 이 저장소에 포함하지 않습니다. Gemma 또는 파생 모델을 배포하는 별도 릴리스는 Gemma Terms가 요구하는 계약·고지·Notice 조건을 따라야 합니다. 단순 모델 ID 표기는 가중치 재배포를 의미하지 않습니다.

## 데이터와 이미지

구조동물 메타데이터 출처와 이용범위는 `DATA_CARD.md`에 기록합니다. 저장소의 샘플 이미지 두 개는 출처·라이선스 확인 전까지 공식 출품 릴리스 자산으로 간주하지 않습니다.
