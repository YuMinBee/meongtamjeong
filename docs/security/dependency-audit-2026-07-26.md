# Python 의존성 알려진 취약점 점검 — 2026-07-26

이 기록은 `requirements.lock.txt`를 `pip-audit 2.9.0`으로 조회한 시점의 알려진 취약점 점검 결과입니다. 보안 보증이나 이후에 공개될 취약점의 부재를 뜻하지 않습니다.

- 입력 SHA-256: `6d89251f5851a0b38a66b0ed1109f0e4f556d7f6a8440e5282f2c5f937fb267e`
- PyPI/OSV로 확인 가능한 잠금 의존성의 알려진 취약점: **0건**
- 자동 점검 제외: **1건** — PyPI 패키지가 아닌 공식 OpenAI CLIP Git commit 고정 항목
- 조치: 최초 점검에서 개발 의존성 `pytest 8.4.2`의 `PYSEC-2026-1845` 1건을 확인해 `pytest 9.0.3`으로 올린 뒤 재검사했습니다.

재현 명령:

```bash
python -m pip install "pip-audit==2.9.0"
pip-audit -r requirements.lock.txt -f json
```

`clip @ git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1`은 pip-audit가 “PyPI에서 찾을 수 없음”으로 제외합니다. 따라서 해당 저장소·고정 commit은 별도 소스 검토 대상으로 남기며, 이 문서의 0건 수치에 포함하지 않습니다.

취약점 데이터베이스는 변하므로 출품 ZIP과 발표용 태그를 만들기 직전에 같은 명령을 다시 실행하고, 새 finding이 있으면 영향 범위·수정 버전·수용 여부를 별도로 기록해야 합니다.
