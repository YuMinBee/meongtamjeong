# 사람 시각 평가 도구

참고 사진과 후보 사진의 외형 유사성을 0–3점으로 채점합니다. 화면에는 방법명,
검색 순위, 속성 라벨이 표시되지 않습니다. 점수를 누르면 서버 저장이 끝난 뒤
다음 항목으로 이동하며, 이전 버튼으로 돌아가 수정할 수 있습니다.
저장에 실패하면 현재 항목에 머뭅니다. 사진은 클릭하여 확대할 수 있습니다.

## 실행

```powershell
python -m experiments.dog_domain.visual_study_server --data D:/path/to/study --port 8765
```

데이터 디렉터리에는 별도로 준비한 `study.json`, `access.json`, `assets/`가 필요합니다.
`prepare_visual_study`는 준비된 학습 체크포인트와 공고 캐시를 이용해 원래 연구를 구성하며,
이미 고정된 연구가 있으면 덮어쓰지 않습니다. 다운로드나 GPU 추론 없이 샘플 테스트만
하려면 아래 단위 테스트를 실행하세요.

평점과 변경 이력은 `ratings.sqlite3`에 저장됩니다. 평가자별 비밀 토큰을 포함하는
`/?key=...` 링크를 각자 전달하며, 같은 링크로 재접속하면 저장된 진행 상황을 읽습니다.
관리자 링크로 진행 상황과 JSON 내보내기를 확인할 수 있습니다.
관리자 토큰, 평가자 토큰, 평점 파일은 저장소에 커밋하지 마세요.

## 외부 접속

`scripts/start_visual_study.ps1`과 `scripts/share_visual_study.ps1`은 기존 Windows 연구
PC를 위한 편의 스크립트입니다. Python·데이터·cloudflared 경로를 먼저 확인하세요.
공유 스크립트는 루프백 주소에 추가 서버를 열고 Cloudflare 임시 HTTPS 터널을 연결합니다.
PC가 켜져 있어야 하며 터널을 재시작하면 주소가 바뀔 수 있습니다.
관리자 기능은 `--public-raters-only` 서버에서 접근할 수 없습니다.
원래 연구에서는 R1도 외부 접속을 제한하며, 보완 연구는 `protocol.public_raters`에
명시된 평가자에게만 접근을 허용합니다.

## 집계와 추가 평가

`visual_study_metrics.py`는 공통 판단 풀의 선형 gain nDCG와 ordinal Krippendorff's α를 계산합니다.
미평가를 0점으로 대신하지 않으며, 등록된 평가자들의 점수가 모두 모여야 완결된 비교를 출력합니다.
평가자 수는 서버 접근 설정에서 전달합니다. 지표 함수만 직접 호출할 때의 기본값은 원래 R1–R3입니다.

`prepare_dino_visual_supplement`는 DINO 이미지 단독 결과 중 기존에 평가하지 않은 쌍만
별도 연구로 만듭니다. `merge_dino_visual_supplement`는 고정된 원래 평점과 추가 평점을 합쳐
확장된 공통 풀에서 모든 방법을 다시 계산합니다. 후보 풀이 바뀌면 IDCG도 달라지므로
기존 풀의 점수와 확장 풀의 점수를 섞어서 비교하지 않습니다.

## 검증

```powershell
python -m pytest tests/test_visual_study.py -q
```

`scripts/check_visual_study_ui.py`는 실제 연구의 사진·설정을 별도 QA 디렉터리로 복사하여
저장, 실패 시 이동 방지, 이전 평점 수정, 확대, 내보내기와 모바일 화면을 검사합니다.
실제 평점 DB는 수정하지 않습니다. Playwright와 Microsoft Edge가 필요합니다.
