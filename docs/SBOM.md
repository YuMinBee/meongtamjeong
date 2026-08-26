# 소프트웨어 자재명세서(SBOM)

- 프로젝트: `MeongTamjeong` `submission-2026`
- 생성 기준 시각: `2026-08-26T08:05:00Z`
- 전체 설치 환경 구성요소: **75개**
- 선언된 직접 의존성: **25개**
- 기계 판독 형식: `SBOM.spdx.json`(SPDX 2.3), `SBOM.cdx.json`(CycloneDX 1.6)

이 목록은 `docs/dependency-report.json`에 고정된 깨끗한 검증 환경을 변환한 것입니다. 직접 선언 관계만 프로젝트의 `DEPENDS_ON`으로 기록하며 전이 의존성 간 그래프는 추정하지 않습니다. 라이선스 값은 설치 배포판 메타데이터의 감사 단서이며 법률 자문이 아닙니다.

## 직접 의존성

| 라이브러리 | 버전 | 라이선스 메타데이터 | 사용 목적 | 공식 저장소 |
| --- | --- | --- | --- | --- |
| `accelerate` | `1.14.0` | Apache | 선택 연구 모델 로딩과 장치 배치 | [upstream](https://github.com/huggingface/accelerate) |
| `clip` | `1.0` | MIT | 이미지-텍스트 공통 임베딩 생성 | [upstream](git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1) |
| `faiss-cpu` | `1.14.3` | MIT | 고정 벡터 인덱스의 근접 검색 | [upstream](https://github.com/facebookresearch/faiss) |
| `fastapi` | `0.140.0` | MIT | 검색·문의 HTTP API 제공 | [upstream](https://github.com/fastapi/fastapi) |
| `ftfy` | `6.3.1` | Apache-2.0 | CLIP 입력 텍스트 정규화 | [upstream](https://github.com/rspeer/python-ftfy) |
| `httpx` | `0.28.1` | BSD-3-Clause | API 회귀 테스트와 선택 네트워크 평가 | [upstream](https://github.com/encode/httpx) |
| `numpy` | `2.2.6` | BSD (variant unspecified) | 벡터·수치 연산 | [upstream](https://github.com/numpy/numpy) |
| `opencv-python-headless` | `4.13.0.92` | Apache 2.0 | 오프라인 이미지 품질·영역 처리 | [upstream](https://github.com/opencv/opencv-python) |
| `openpyxl` | `3.1.5` | MIT | 선택 분석 결과의 XLSX 입출력 | [upstream](https://foss.heptapod.net/openpyxl/openpyxl) |
| `pandas` | `2.3.3` | BSD (variant unspecified) | 선택 평가·분석 데이터 처리 | [upstream](https://github.com/pandas-dev/pandas) |
| `pillow` | `12.3.0` | MIT-CMU | 업로드 이미지 검증과 변환 | [upstream](https://github.com/python-pillow/Pillow) |
| `pydantic` | `2.13.4` | MIT | API 요청·응답 스키마 검증 | [upstream](https://github.com/pydantic/pydantic) |
| `pytest` | `9.0.3` | MIT | 자동 회귀 테스트 | [upstream](https://github.com/pytest-dev/pytest) |
| `python-dotenv` | `1.2.2` | BSD-3-Clause | 로컬 환경변수 파일 로딩 | [upstream](https://github.com/theskumar/python-dotenv) |
| `python-multipart` | `0.0.32` | Apache-2.0 | 이미지 multipart 업로드 파싱 | [upstream](https://github.com/Kludex/python-multipart) |
| `regex` | `2026.7.19` | Apache-2.0 AND CNRI-Python | CLIP 토큰화 보조 | [upstream](https://github.com/mrabarnett/mrab-regex) |
| `requests` | `2.34.2` | Apache-2.0 | 공공 API와 원문 상태 조회 | [upstream](https://github.com/psf/requests) |
| `ruff` | `0.15.21` | MIT | 정적 검사와 코드 형식 검증 | [upstream](https://github.com/astral-sh/ruff) |
| `sentencepiece` | `0.2.2` | Apache-2.0 | 선택 연구 모델 토크나이저 | [upstream](https://github.com/google/sentencepiece) |
| `torch` | `2.13.0` | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT | CLIP 및 선택 연구 모델 추론 | [upstream](https://github.com/pytorch/pytorch) |
| `torchvision` | `0.28.0` | BSD | 이미지 전처리와 선택 객체 영역 추출 | [upstream](https://github.com/pytorch/vision) |
| `tqdm` | `4.69.1` | MPL-2.0 AND MIT | 오프라인 처리 진행률 표시 | [upstream](https://github.com/tqdm/tqdm) |
| `transformers` | `5.14.1` | Apache 2.0 License | 선택 DINO·VLM 연구 경로 | [upstream](https://github.com/huggingface/transformers) |
| `urllib3` | `2.7.0` | MIT | HTTP 연결·재시도 계층 | [upstream](https://github.com/urllib3/urllib3) |
| `uvicorn` | `0.51.0` | BSD-3-Clause | FastAPI ASGI 서버 실행 | [upstream](https://github.com/Kludex/uvicorn) |

## 포함 범위와 한계

- OpenAI CLIP은 공식 Git 커밋을 고정한 VCS 의존성이므로 PyPI purl을 허위로 부여하지 않고 원본 저장소 URL을 기록합니다.
- 사전학습 모델 가중치, 공공 공고 데이터, 공고 사진과 파생 벡터는 Python 패키지 SBOM과 별도의 모델·데이터 조건을 따릅니다.
- 선택 DINO·VLM 연구 가중치와 진행 중인 로컬 학습 산출물은 출품 실행 프로필과 이 SBOM에 포함하지 않습니다.
