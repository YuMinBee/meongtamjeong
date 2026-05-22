import argparse
import html
import json
from pathlib import Path
from typing import Any, Dict, List


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DEFAULT_INPUT_PATH = DATA_DIR / "local_dog_cache.json"
DEFAULT_OUTPUT_PATH = DATA_DIR / "live_breed_gallery.html"


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def load_items(input_path: Path) -> Dict[str, Any]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return {"fetched_at": "", "years": "", "items": payload}
    return {
        "fetched_at": clean_text(payload.get("fetched_at")),
        "years": clean_text(payload.get("years")),
        "items": payload.get("items") or [],
    }


def resolve_detail_url(item: Dict[str, Any]) -> str:
    detail_url = clean_text(item.get("detail_url"))
    desertion_no = clean_text(item.get("desertionNo") or item.get("desertion_no"))

    if detail_url:
        detail_url = detail_url.replace("publicDetail.do", "publicDtl.do")
        if "publicDtl.do" in detail_url and "menuNo=" not in detail_url:
            sep = "&" if "?" in detail_url else "?"
            detail_url = f"{detail_url}{sep}menuNo=1000000055"
        return detail_url

    if not desertion_no:
        return ""

    return (
        "https://www.animal.go.kr/front/awtis/public/publicDtl.do"
        f"?desertionNo={desertion_no}&menuNo=1000000055"
    )


def group_by_breed(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}

    for item in items:
        image_url = clean_text(item.get("image_url"))
        if not image_url:
            continue

        breed_code = clean_text(item.get("breed_code")) or "UNKNOWN"
        breed_name = clean_text(item.get("breed_name"))
        label = breed_name or breed_code

        bucket = grouped.setdefault(
            breed_code,
            {
                "breed_code": breed_code,
                "breed_name": breed_name,
                "label": label,
                "items": [],
            },
        )
        if breed_name and not bucket["breed_name"]:
            bucket["breed_name"] = breed_name
            bucket["label"] = breed_name

        bucket["items"].append(
            {
                "desertionNo": clean_text(item.get("desertionNo")) or "Unknown",
                "notice_no": clean_text(item.get("notice_no") or item.get("noticeNo")) or "",
                "breed": clean_text(item.get("breed")) or label,
                "breed_code": breed_code,
                "age": clean_text(item.get("age")) or "Unknown",
                "sex": clean_text(item.get("sex")) or "Unknown",
                "weight": clean_text(item.get("weight")) or "Unknown",
                "neuter": clean_text(item.get("neuter")) or "Unknown",
                "process_state": clean_text(item.get("process_state")) or "",
                "care_name": clean_text(item.get("care_name")) or "",
                "notice_start": clean_text(item.get("notice_start")) or "",
                "notice_end": clean_text(item.get("notice_end")) or "",
                "desc": clean_text(item.get("merged_desc")) or clean_text(item.get("desc")) or "",
                "base_desc": clean_text(item.get("desc")) or "",
                "vlm_desc": clean_text(item.get("vlm_desc")) or "",
                "image_url": image_url,
                "detail_url": resolve_detail_url(item),
            }
        )

    results = []
    for bucket in grouped.values():
        bucket["items"].sort(key=lambda item: item["desertionNo"], reverse=True)
        bucket["count"] = len(bucket["items"])
        bucket["sample_images"] = [item["image_url"] for item in bucket["items"][:3]]
        bucket["search_blob"] = " ".join(
            [
                bucket["breed_code"],
                clean_text(bucket.get("breed_name")),
                clean_text(bucket.get("label")),
                " ".join(clean_text(item.get("desertionNo")) for item in bucket["items"]),
                " ".join(clean_text(item.get("notice_no")) for item in bucket["items"]),
                " ".join(clean_text(item.get("desc")) for item in bucket["items"][:10]),
            ]
        ).lower()
        results.append(bucket)

    results.sort(key=lambda item: (-item["count"], item["breed_code"]))
    return results


def build_html(grouped: List[Dict[str, Any]], fetched_at: str, years: str) -> str:
    grouped_json = json.dumps(grouped, ensure_ascii=False)
    subtitle_bits = []
    if years:
        subtitle_bits.append(f"최근 {years}년")
    subtitle_bits.append("종 코드별 유기견 사진 갤러리")
    if fetched_at:
        subtitle_bits.append(f"수집 시각 {fetched_at}")
    subtitle = " / ".join(subtitle_bits)

    return f"""<!doctype html>
<html lang="ko">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Live Breed Gallery</title>
    <style>
      :root {{
        --bg: #f3eee6;
        --panel: rgba(255, 250, 243, 0.95);
        --ink: #241816;
        --muted: #6d6258;
        --line: rgba(36, 24, 22, 0.12);
        --accent: #c76b38;
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        color: var(--ink);
        font-family: "Gill Sans", "Noto Sans KR", sans-serif;
        background:
          radial-gradient(circle at top left, rgba(255,255,255,0.9), transparent 32%),
          linear-gradient(135deg, #f6f1ea 0%, #e6d3c2 100%);
      }}
      .shell {{
        width: min(1440px, calc(100vw - 28px));
        margin: 20px auto 36px;
      }}
      .hero {{
        border: 1px solid var(--line);
        border-radius: 28px;
        background: var(--panel);
        padding: 28px;
        box-shadow: 0 20px 60px rgba(36, 24, 22, 0.08);
      }}
      h1 {{
        margin: 0 0 8px;
        font-size: clamp(2rem, 3vw, 3.3rem);
        line-height: 0.95;
        letter-spacing: -0.04em;
      }}
      .hero p {{
        margin: 0;
        color: var(--muted);
      }}
      .layout {{
        display: grid;
        grid-template-columns: minmax(260px, 330px) 1fr;
        gap: 18px;
        margin-top: 18px;
      }}
      .panel {{
        border: 1px solid var(--line);
        border-radius: 24px;
        background: var(--panel);
        padding: 18px;
        box-shadow: 0 16px 40px rgba(36, 24, 22, 0.06);
      }}
      .toolbar {{
        display: grid;
        gap: 10px;
        margin-bottom: 14px;
      }}
      .toolbar input {{
        width: 100%;
        padding: 12px 14px;
        border-radius: 14px;
        border: 1px solid var(--line);
        background: rgba(255,255,255,0.85);
        font: inherit;
      }}
      .breed-list {{
        display: grid;
        gap: 10px;
        max-height: 76vh;
        overflow: auto;
      }}
      .breed-card {{
        display: grid;
        gap: 4px;
        padding: 13px 15px;
        border-radius: 18px;
        border: 1px solid var(--line);
        background: rgba(255,255,255,0.78);
        cursor: pointer;
        transition: transform 0.18s ease, border-color 0.18s ease, background 0.18s ease;
      }}
      .breed-card:hover {{
        transform: translateY(-1px);
        border-color: rgba(199, 107, 56, 0.45);
      }}
      .breed-card.active {{
        background: linear-gradient(135deg, #fff6ef 0%, #f8ddce 100%);
        border-color: rgba(199, 107, 56, 0.75);
      }}
      .breed-card span {{
        color: var(--muted);
        font-size: 0.92rem;
      }}
      .section-head {{
        display: flex;
        justify-content: space-between;
        align-items: end;
        gap: 12px;
        margin-bottom: 14px;
      }}
      .section-head h2 {{
        margin: 0;
        font-size: clamp(1.3rem, 2vw, 2rem);
      }}
      .gallery {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
        gap: 16px;
      }}
      .dog-card {{
        overflow: hidden;
        border-radius: 22px;
        border: 1px solid var(--line);
        background: rgba(255,255,255,0.84);
      }}
      .dog-card img {{
        display: block;
        width: 100%;
        aspect-ratio: 1 / 1;
        object-fit: cover;
        background: #eaded3;
      }}
      .dog-meta {{
        display: grid;
        gap: 6px;
        padding: 14px;
      }}
      .dog-meta span {{
        color: var(--muted);
        font-size: 0.92rem;
      }}
      .dog-meta p {{
        margin: 4px 0 0;
        font-size: 0.95rem;
        line-height: 1.45;
      }}
      .empty {{
        color: var(--muted);
        padding: 16px 4px;
      }}
      @media (max-width: 920px) {{
        .layout {{ grid-template-columns: 1fr; }}
        .breed-list {{ max-height: none; }}
      }}
    </style>
  </head>
  <body>
    <main class="shell">
      <section class="hero">
        <h1>Breed Code Gallery</h1>
        <p>{html.escape(subtitle)}</p>
      </section>
      <section class="layout">
        <aside class="panel">
          <div class="toolbar">
            <input id="searchInput" type="search" placeholder="종 코드, 품종명, 공고번호 검색" />
          </div>
          <div class="breed-list" id="breedList"></div>
        </aside>
        <section class="panel">
          <div class="section-head">
            <div>
              <h2 id="galleryTitle">불러오는 중</h2>
              <p id="galleryMeta" class="empty"></p>
            </div>
          </div>
          <div class="gallery" id="gallery"></div>
        </section>
      </section>
    </main>
    <script>
      const grouped = {grouped_json};
      const breedList = document.getElementById("breedList");
      const gallery = document.getElementById("gallery");
      const galleryTitle = document.getElementById("galleryTitle");
      const galleryMeta = document.getElementById("galleryMeta");
      const searchInput = document.getElementById("searchInput");
      let activeCode = grouped.length ? grouped[0].breed_code : null;

      function escapeHtml(value) {{
        return String(value ?? "")
          .replace(/&/g, "&amp;")
          .replace(/</g, "&lt;")
          .replace(/>/g, "&gt;")
          .replace(/"/g, "&quot;")
          .replace(/'/g, "&#39;");
      }}

      function escapeMultiline(value) {{
        return escapeHtml(value).replace(/\\n/g, "<br />");
      }}

      function filteredGroups() {{
        const q = searchInput.value.trim().toLowerCase();
        if (!q) return grouped;
        return grouped.filter((item) => {{
          return String(item.search_blob || "").includes(q);
        }});
      }}

      function renderBreedList() {{
        const groups = filteredGroups();
        if (!groups.length) {{
          breedList.innerHTML = '<div class="empty">검색 결과가 없습니다.</div>';
          galleryTitle.textContent = "표시할 종 코드가 없습니다.";
          galleryMeta.textContent = "";
          gallery.innerHTML = "";
          return;
        }}

        if (!groups.some((item) => item.breed_code === activeCode)) {{
          activeCode = groups[0].breed_code;
        }}

        breedList.innerHTML = groups.map((item) => `
          <button class="breed-card ${{item.breed_code === activeCode ? "active" : ""}}" data-code="${{escapeHtml(item.breed_code)}}">
            <strong>${{escapeHtml(item.label)}}</strong>
            <span>code ${{escapeHtml(item.breed_code)}}</span>
            <span>${{item.count}} dogs</span>
          </button>
        `).join("");

        breedList.querySelectorAll(".breed-card").forEach((button) => {{
          button.addEventListener("click", () => {{
            activeCode = button.dataset.code;
            renderBreedList();
            renderGallery();
          }});
        }});
      }}

      function renderGallery() {{
        const current = grouped.find((item) => item.breed_code === activeCode);
        if (!current) {{
          galleryTitle.textContent = "표시할 종 코드가 없습니다.";
          galleryMeta.textContent = "";
          gallery.innerHTML = "";
          return;
        }}

        galleryTitle.textContent = `${{current.label}} (${{current.breed_code}})`;
        galleryMeta.textContent = `${{current.count}}마리`;
        gallery.innerHTML = current.items.map((item) => `
          <article class="dog-card">
            <a href="${{escapeHtml(item.detail_url)}}" target="_blank" rel="noreferrer">
              <img src="${{escapeHtml(item.image_url)}}" alt="${{escapeHtml(item.breed)}}" loading="lazy" />
            </a>
              <div class="dog-meta">
                <strong>${{escapeHtml(item.desertionNo)}}</strong>
                ${{item.notice_no ? `<span>공고번호 ${{escapeHtml(item.notice_no)}}</span>` : ""}}
                <span>${{escapeHtml(item.breed)}}</span>
                <span>${{escapeHtml(item.sex)}} / ${{escapeHtml(item.age)}}</span>
              <span>${{escapeHtml(item.weight)}} / 중성화 ${{escapeHtml(item.neuter)}}</span>
              ${{item.process_state ? `<span>상태 ${{escapeHtml(item.process_state)}}</span>` : ""}}
              ${{item.care_name ? `<span>보호소 ${{escapeHtml(item.care_name)}}</span>` : ""}}
              ${{(item.notice_start || item.notice_end) ? `<span>공고 ${{escapeHtml(item.notice_start || "-")}} ~ ${{escapeHtml(item.notice_end || "-")}}</span>` : ""}}
              <p>${{escapeMultiline(item.desc || "특징 정보 없음")}}</p>
            </div>
          </article>
        `).join("");
      }}

      searchInput.addEventListener("input", () => {{
        renderBreedList();
        renderGallery();
      }});

      renderBreedList();
      renderGallery();
    </script>
  </body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="local_dog_cache.json을 종 코드별 HTML 갤러리로 렌더링합니다.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="입력 JSON 경로")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="출력 HTML 경로")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_items(args.input)
    grouped = group_by_breed(payload["items"])
    html_doc = build_html(grouped, fetched_at=payload["fetched_at"], years=payload["years"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_doc, encoding="utf-8")

    print(f"[DONE] HTML 저장 완료: {args.output}")
    print(f"[INFO] breed_codes={len(grouped)} items={sum(item['count'] for item in grouped)}")


if __name__ == "__main__":
    main()
