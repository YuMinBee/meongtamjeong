# 2026 오픈소스 개발자대회 제출 문서

공식 결과보고서 DOCX를 기준으로 작성한 제출 초안입니다. 안내 첫 페이지와 회색
작성 예시는 제거했고, 결과보고서 본문 4쪽(5쪽 제한 이내), SBOM 2쪽, AI 모델 활용
명세 2쪽으로 렌더링을 확인했습니다.

## 현재 파일

- `2026 오픈소스 개발자대회 결과보고서_접수번호(팀명).docx`: 수정 가능한 공식 양식
- `2026 오픈소스 개발자대회 결과보고서_접수번호(팀명).pdf`: 위 DOCX의 PDF 렌더
- `report-qa.json`: 템플릿·산출물 해시와 페이지 검사 기록
- `meongtamjeong-public-text-only-v1.zip`: 사진 권리 노출을 줄인 검증용 공개 소스 번들
- `public-text-package.json`: 위 ZIP의 source commit·artifact hash·제외 통계
- `scripts/build_contest_report.py`: 공식 템플릿에서 문서를 다시 만드는 도구
- `scripts/render_contest_report.ps1`: 설치 없이 준비한 LibreOffice로 PDF를 다시 만드는 도구

공개 소스 번들은 commit `9f7b3ead8c818353faf339a075f4fa5b6349019c`에서 두 번
독립 생성해 같은 SHA-256이 나오는 것을 확인했습니다. 2026-07-26 기준 공개 공고 텍스트
벡터 1,516개만 포함하며, 사진 벡터 1,228개와 crop 벡터 1,002개, 샘플 사진, 원격 사진
URL, full 프로필 전용 평가를 제외합니다. 아직 접수번호가 확정된 최종 tag 산출물은
아니며, 최신 1,266건 재동기화 뒤 다시 생성해야 합니다.

## 제출 전에 반드시 넣을 정보

현재 DOCX/PDF에는 아래 네 곳이 `[최종 입력 필요: ...]`로 남아 있습니다.

1. 참가 접수와 동일한 팀명
2. 팀장 포함 팀 인원
3. 참가부문(학생 또는 일반)
4. 3분 이내 YouTube 시연영상 URL

파일명에도 실제 접수번호와 팀명을 넣어야 합니다. 프로젝트명은 `멍탐정`, 과제유형은
`자유과제`, 공개 저장소는 아래 대회용 브랜치로 작성했습니다.

`https://github.com/YuMinBee/meongtamjeong/tree/submission/osscontest-2026`

## 접수정보를 넣어 다시 생성

Word에서 네 자리표시자를 직접 바꿔도 됩니다. 같은 문서를 재생성하려면 저장소
루트의 PowerShell에서 다음과 같이 실행합니다.

```powershell
python scripts/build_contest_report.py `
  --template "tmp/osscontest-2026/official-result-template.docx" `
  --team-name "실제 팀명" `
  --team-size "실제 N명" `
  --division "학생" `
  --demo-url "https://youtu.be/실제영상ID" `
  --output "submission/2026-osscontest/2026 오픈소스 개발자대회 결과보고서_실제접수번호(실제팀명).docx"
```

그 뒤 현재 준비된 CPU 전용 휴대용 LibreOffice 렌더러로 PDF를 만듭니다.

```powershell
powershell -ExecutionPolicy Bypass -File scripts/render_contest_report.ps1 `
  -InputPath "submission/2026-osscontest/2026 오픈소스 개발자대회 결과보고서_실제접수번호(실제팀명).docx"
```

PDF가 생기면 팀명·URL·표가 잘리지 않았는지 Word와 PDF에서 한 번씩 확인합니다.

## 자원 여유가 생긴 뒤에만 할 일

2026-08-26 16:13 KST에 활성 공고 1,266건의 최신 원본을 아래 임시 경로에 받았습니다.

`%TEMP%\meong-osscontest-20260826\fresh_dogs.json`

현재 대규모 학습과 자원 충돌을 피하기 위해 CLIP 로딩·텍스트 임베딩·FAISS 재생성은
일부러 실행하지 않았습니다. 학습이 완전히 끝나고 GPU 작업이 없는 것을 확인한 뒤
`docs/contest-release-runbook.md`의 `--text-only --device cpu` 절차를 순서대로 실행해
새 public-text-only index와 해당 프로필 평가를 동결해야 합니다. 이때도 WSL이나 GPU를
사용하지 말고, 기존 canonical 파일을 자동으로 덮어쓰지 않습니다.

## 최종 제출 체크

- DOCX와 PDF 파일명에 실제 접수번호·팀명 반영
- 두 파일의 팀명·인원·참가부문·YouTube URL 일치
- YouTube 영상 공개 또는 일부 공개, 재생시간 3분 이내
- 대회용 브랜치가 GitHub에서 로그인 없이 열리는지 확인
- 저장소의 `SBOM.spdx.json`, `SBOM.cdx.json`, `docs/SBOM.md` 확인
- DOCX 원본과 같은 내용의 PDF를 대회 홈페이지에 함께 업로드

공식 제출 마감은 2026-08-27 18:00 KST입니다.
