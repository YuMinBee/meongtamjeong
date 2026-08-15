# 공고 Provider Adapter

이 경계의 목적은 멍탐정의 검색·정규화·상태 필터·동기화 로직을 유지한 채
공고 수집 원천만 교체할 수 있게 하는 것입니다. 기본값은 기존
국가동물보호정보시스템 공공 API이고, 네트워크 없이 시험할 수 있는 로컬
JSON provider도 같은 실제 수집 경로를 사용합니다.

provider 계약의 적용 범위는 `scripts/fetch_live_dogs.py`입니다. FastAPI의
최신 공고 재확인 경로는 호환성 보호를 위해 아직 기존 공공 API 구현을
사용합니다. provider 출력은 먼저 기존 `local_*_cache.json` 형식으로
정규화된 뒤 기존 sync 입력이 됩니다. `sync_active_index.py`의 검색·상태·벡터
정책은 그대로이며, 외부 이미지 요청에는 별도의 정확한 호스트 allowlist와
리다이렉트 금지 정책을 적용합니다.

## 15분 연결: JSON bridge

새 기관의 데이터를 아래처럼 합성 예제와 같은 JSON으로 내보내는 것이 가장
짧은 연결 방법입니다.

```text
python scripts/fetch_live_dogs.py \
  --provider local-json \
  --input-json data/examples/notice-provider.synthetic.json \
  --species dog \
  --out tmp/provider-smoke.json
```

Windows PowerShell에서는 줄바꿈 대신 한 줄로 실행해도 됩니다. 예제의 공고,
사진 URL, 기관명은 모두 합성이며 실제 데이터나 개인정보가 아닙니다. 이
예제를 운영 인덱스에 합치지 마십시오.

입력 루트는 JSON 배열 또는 `{"items": [...]}`입니다. 각 항목의 최소 필드는
다음과 같습니다.

| 필드 | 필수 | 의미 |
| --- | --- | --- |
| `notice_id` | 예 | 원천 내에서 안정적인 공고 ID |
| `source_status` | 예 | 기관이 제공한 원문 상태. 없으면 명시적으로 `unknown` |
| `species` | 권장 | `dog`, `cat`, `other`; 지정하면 CLI의 `--species`와 다른 행은 제외 |
| `image_url` 또는 `image_urls` | 현재 수집 정책상 필요 | 사진이 없으면 기존 정책대로 cache 결과에서 제외 |
| 나머지 공고 필드 | 선택 | 아래 alias 중 기관이 실제로 제공한 값만 기록 |

현재 공통 정규화기가 이해하는 주요 provider-neutral alias는
`notice_no`, `breed`, `breed_code`, `breed_name`, `breed_full_name`,
`sex`, `age`, `weight`, `neuter`, `color`, `desc`, `care_name`,
`care_tel`, `care_addr`, `org_name`, `happen_place`, `happen_date`,
`notice_start`, `notice_end`, `process_state`, `detail_url`입니다. 기존 공공
API의 `desertionNo`, `kindNm`, `sexCd`, `noticeSdt` 같은 필드도 계속
지원합니다.

값이 없으면 필드를 생략하거나 `unknown`으로 두십시오. 품종, 성격, 공격성,
아동·다른 동물 친화성 등을 사진이나 품종 상식으로 추정해 채우면 안 됩니다.
`source_status`도 adapter가 임의로 “입양 가능”으로 바꾸지 않고 기관 원문을
전달합니다. 검색 가능 여부는 이후 기존 `app.notice_status`가 동일한 규칙으로
판정합니다.

출력 확인:

```text
python -m json.tool tmp/provider-smoke.json
```

정상 출력에는 최상위 `provider`, `provider_diagnostics`가 있고 각 보존된
항목에는 다음 provenance가 붙습니다.

```json
{
  "source_status": "active",
  "source_provenance": {
    "schema_version": "notice-source-record.v1",
    "provider_id": "local-json",
    "source_record_id": "SYNTH-DOG-001",
    "retrieved_at": "2026-07-26T12:00:00+09:00",
    "source_uri": "local-json:sha256:<입력 JSON의 SHA-256>"
  }
}
```

`retrieved_at`은 “이 시각까지 공고가 계속 유효하다”는 뜻이 아니라 해당
원천을 확인한 시각입니다. 로컬 입력은 파일명 대신 내용 SHA-256을 남겨
개인 이름이 포함될 수 있는 경로를 노출하지 않습니다. API 키, 인증 헤더,
서명된 URL, 개인정보를 `source_uri`, 예외 메시지, JSON에 넣지 마십시오.

## Python provider 직접 구현

JSON 변환 없이 연결하려면 `app.notice_provider.NoticeProvider` 계약을
구현합니다. 구현해야 하는 메서드는 `fetch(request)` 하나입니다.

```python
from app.notice_provider import (
    NoticeProvenance,
    NoticeProviderResult,
    NoticeSourceRecord,
)


class MyShelterProvider:
    provider_id = "my-shelter-open-data"

    def fetch(self, request):
        rows = load_open_records(request.start, request.end)  # 기관별 구현
        records = []
        for row in rows:
            notice_id = str(row["id"])
            records.append(
                NoticeSourceRecord(
                    notice_id=notice_id,
                    source_status=str(row.get("status") or "unknown"),
                    source_record={
                        "notice_id": notice_id,
                        "species": "dog",
                        "desc": row.get("description") or "",
                        "image_urls": row.get("open_image_urls") or [],
                        "notice_end": row.get("notice_end") or "",
                    },
                    provenance=NoticeProvenance(
                        provider_id=self.provider_id,
                        source_record_id=notice_id,
                        retrieved_at=request.retrieved_at,
                        source_uri="https://data.example.invalid/notices",
                    ),
                )
            )
        return NoticeProviderResult(
            records=tuple(records),
            pages_fetched=1,
            raw_items=len(rows),
        )
```

기존 수집 함수를 그대로 재사용합니다.

```python
from scripts.fetch_live_dogs import fetch_live_animals, save_payload

payload = fetch_live_animals(
    api_key="",
    species="dog",
    years=1,
    rows=1000,
    max_pages=100,
    provider=MyShelterProvider(),
)
save_payload(payload, output_path)
```

공통 경로는 provider 결과에 대해 다음 계약을 실행 중에 검증합니다.

- `provider_id`, `notice_id`, provenance의 `source_record_id`는 비어 있지 않음
- `retrieved_at`은 timezone이 포함된 ISO-8601
- provenance의 `provider_id`는 실행 provider와 일치
- `raw_items >= len(records)`이며 page·diagnostic count는 음수가 아님
- source record는 mapping이고 상태가 없으면 `unknown`

provider는 **원천 수집과 필드 대응만** 담당합니다. 활성 상태 제외, 이미지
필수 정책, 중복 제거, 보수적 품종 정규화, cache 스키마는 공통 수집기가
담당합니다. 따라서 adapter가 바뀌어도 검색·sync 정책은 한 곳에 남습니다.

## 기본 공공 API와 기존 명령 호환성

아무 옵션도 추가하지 않은 기존 명령은 계속 기본 provider를 사용합니다.

```text
python scripts/fetch_live_dogs.py --days 365
```

`ANIMAL_API_KEY`는 기존처럼 `.env`에서 읽습니다. 키 값은 provenance와
오류 메시지에 기록되지 않습니다. 기본 provider가 반환하는 `source_uri`도
query parameter가 없는 endpoint만 보존합니다.

수집 cache를 검토한 뒤에는 기존 런북대로 `reuse_image_enrichment.py`와
`sync_active_index.py`를 실행합니다. 새 provider를 연결했다는 이유로 sync
안전 한도나 active 상태 검증을 낮추지 마십시오.

기본 sync는 `openapi.animal.go.kr`의 이미지만 내려받습니다. 다른 기관의
공개 이미지 호스트를 쓸 때는 스킴·경로가 없는 실제 공개 DNS 호스트를
명시적으로 추가해야 합니다.

```text
python scripts/sync_active_index.py ... --allowed-image-host images.example.org
```

옵션은 여러 번 지정할 수 있습니다. localhost, 사설 IP, 내부용 suffix,
credential이 든 URL은 거부하며 HTTP redirect도 따라가지 않습니다. 기관이
CDN redirect를 사용한다면 최종 공개 CDN URL을 원천 필드에 넣고 그 정확한
호스트를 allowlist에 추가합니다. 이 제한을 우회하기 위해 광범위한 프록시나
내부 호스트를 허용하지 마십시오.

## 검증

```text
ruff check app/notice_provider.py scripts/fetch_live_dogs.py tests/test_notice_provider.py
pytest -q tests/test_notice_provider.py tests/test_notice_metadata.py
```

contract test는 기본 공공 API adapter를 가짜 HTTP 응답으로, 로컬 JSON
adapter를 임시 합성 파일로 실행합니다. 외부 네트워크·LLM·유료 API를
호출하지 않습니다.
