# 제3자 소프트웨어·모델 고지

이 문서는 멍탐정이 직접 포함하거나 실행 시 연동하는 주요 제3자 구성요소를 정리한 감사 목록입니다. 각 배포 전에 실제 설치 버전의 원문 라이선스와 의존성 트리를 다시 확인해야 합니다.

2026-07-26 깨끗한 제출 검증 환경의 설치 패키지 75개와 실제 버전, 배포 메타데이터의 라이선스·원문 파일은 [`docs/dependency-report.md`](docs/dependency-report.md)에 고정했습니다. 이 검증 스냅샷을 바탕으로 [SPDX 2.3](SBOM.spdx.json), [CycloneDX 1.6](SBOM.cdx.json), [사람이 읽는 SBOM](docs/SBOM.md)을 함께 제공합니다. 선언된 직접 의존성 누락과 라이선스 메타데이터 미상 항목은 각각 0건입니다. 전이 의존성 간 관계는 추정하지 않으며, 이 자동 목록은 법률 자문이나 라이선스 적합성 판정을 대신하지 않습니다.

| 구성요소 | 역할 | 알려진 라이선스/약관 | 배포 메모 |
| --- | --- | --- | --- |
| FastAPI | HTTP API | MIT | Python 패키지 |
| Uvicorn | ASGI 서버 | BSD-3-Clause | Python 패키지 |
| Pydantic | 요청·응답 스키마 | MIT | Python 패키지 |
| NumPy | 수치 연산 | BSD-3-Clause | Python 패키지 |
| Pillow | 이미지 처리 | MIT-CMU | Python 패키지 |
| Requests | HTTP 클라이언트 | Apache-2.0 | Python 패키지 |
| urllib3 | 재시도·연결 관리 | MIT | Python 패키지. 애플리케이션이 직접 import |
| python-dotenv | 환경변수 파일 로딩 | BSD-3-Clause | Python 패키지 |
| python-multipart | multipart 요청 파싱 | Apache-2.0 | Python 패키지 |
| PyTorch / torchvision | 모델 실행·이미지 전처리·Faster R-CNN 객체 탐지 | BSD-3-Clause | Python 패키지. 사전학습 가중치는 학습 데이터 조건을 별도 확인 |
| OpenCV Python wheel | 사진 선명도·노출·대비 품질 특징 계산 | 패키징 코드 MIT, OpenCV Apache-2.0, FFmpeg LGPL-2.1 및 추가 제3자 조건 | headless wheel의 `LICENSE.txt`, `LICENSE-3RD-PARTY.txt`를 함께 확인 |
| OpenAI CLIP | 이미지·텍스트 임베딩 | MIT | 코드·가중치 출처와 버전을 릴리스 manifest에 기록 |
| ftfy | CLIP 텍스트 정규화 | Apache-2.0 | Python 패키지 |
| regex | CLIP 토큰화 보조 | Python 1.6/CNRI-derived + Apache-2.0 | 포함된 `LICENSE.txt` 기준 |
| tqdm | 진행률 표시 | MPL-2.0 AND MIT | Python 패키지 메타데이터 기준 |
| FAISS | 벡터 검색 | MIT | Python 패키지/인덱스 |
| Hugging Face Transformers | 선택적 Gemma·DINOv3 실행 | Apache-2.0 | Python 패키지 |
| Accelerate | 선택적 모델 로딩 | Apache-2.0 | Python 패키지 |
| SentencePiece | 선택적 토크나이저 | Apache-2.0 | Python 패키지 |
| pandas / openpyxl | 선택적 분석·Excel 리포트 | BSD-3-Clause / MIT | `requirements-analysis.txt`에서만 설치 |
| HTTPX | API 테스트 클라이언트·선택 held-out 공공 이미지 평가 전송 | BSD-3-Clause | 기본 서비스는 Requests 사용, 네트워크 평가는 명시 옵션 |
| pytest | 자동 테스트 | MIT | 개발 의존성 |
| Ruff | 정적 검사·포맷 | MIT | 개발 의존성 |
| Google Gemma 3 | 선택적 생성·VLM 모델 | Gemma Terms of Use | 가중치는 저장소에 포함하지 않음. 사용자가 별도 약관에 동의해 준비 |
| Meta DINOv3 ViT-B/16 | 선택적 시각 검색 연구 경로 | 모델별 gated 이용조건 | 가중치는 저장소에 포함하지 않음. 사용자가 모델 페이지에서 접근 승인을 받고 조건을 검토해 준비 |

주요 원문:

- OpenAI CLIP: https://github.com/openai/CLIP
- FAISS: https://github.com/facebookresearch/faiss
- TorchVision: https://github.com/pytorch/vision
- Pillow: https://github.com/python-pillow/Pillow/blob/main/LICENSE
- OpenCV Python wheel: https://github.com/opencv/opencv-python
- SentencePiece: https://github.com/google/sentencepiece/blob/master/LICENSE
- COCO 데이터셋: https://cocodataset.org/
- Gemma Terms of Use: https://ai.google.dev/gemma/terms
- Gemma Prohibited Use Policy: https://ai.google.dev/gemma/prohibited_use_policy
- DINOv3 ViT-B/16 model page: https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m

## 프로젝트 라이선스

프로젝트 소스코드는 Apache License 2.0으로 배포하며 저작권자는 `YuMinBee`입니다. 과거 로컬 Git 작성자 설정으로 생긴 이름 차이는 `.mailmap`에서 동일한 작성자로 정규화합니다. Ultralytics 직접 의존성과 YOLO 가중치는 현재 배포 구성에서 제거하고 TorchVision Faster R-CNN으로 교체했습니다. 데이터, 공고 사진, 사전학습 모델과 제3자 패키지는 프로젝트 코드 라이선스의 적용 대상이 아니며 각 원문 조건을 따릅니다.

## 모델 배포

Gemma와 DINOv3 모델 파일, CLIP→DINO flow head와 공고문 행동 head는 이 저장소에 포함하지 않습니다. Gemma 또는 모델 파생물을 배포하는 별도 릴리스는 각 모델 약관이 요구하는 계약·고지·Notice와 재배포 조건을 다시 검토해야 합니다. 단순 모델 ID와 로컬 생성 절차 표기는 가중치 재배포를 의미하지 않습니다.

## 데이터와 이미지

구조동물 메타데이터 출처와 이용범위는 `DATA_CARD.md`에 기록합니다. `assets/samples/`의 공고 사진 crop 네 개는 농림축산식품부 국가동물보호정보시스템 공고에서 유래했으며 파일별 출처와 해시는 `assets/samples/manifest.json`에 기록합니다. 이 사진들은 프로젝트 소스의 Apache-2.0으로 재허가하지 않습니다.

추적된 `data/dog_metas.json`은 공공 API의 공고 텍스트·메타데이터를, `data/dog_faiss.index`는 그 텍스트와 공고 사진·crop에서 만든 벡터를 포함합니다. 이 데이터·사진·파생 벡터에도 프로젝트 소스의 Apache-2.0을 적용하지 않습니다. API 텍스트·메타데이터는 공식 상세 페이지의 `제한 없음` 표시와 출처를 근거로 사용합니다. 개별 촬영자·별도 라이선스가 표시되지 않은 사진·crop과 사진 파생 벡터는 릴리스 전 제공기관의 확인 응답을 보존하거나, 해당 산출물을 제외하고 사용자가 공식 API로 로컬 재생성하도록 배포합니다.
