import argparse
import io
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import torch
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DEFAULT_INPUT_PATH = DATA_DIR / "local_dog_cache.json"
DEFAULT_OUTPUT_PATH = DATA_DIR / "local_dog_cache_enriched.json"
DEFAULT_LOCAL_MODEL_PATH = os.getenv("GEMMA3_MODEL_PATH", "")
DEFAULT_MODEL_ID = os.getenv("GEMMA3_MODEL_ID", "google/gemma-3-12b-it")

PROMPT_TEMPLATE = """당신은 유기견 공고를 보강하는 한국어 비전 언어 모델입니다.

아래 정보를 바탕으로 사진에서 직접 확인 가능한 외형, 분위기, 눈에 띄는 특징만 보강해서 짧게 설명하세요.
절대 사진만 보고 알 수 없는 사실을 단정하지 마세요.
성격, 건강상태, 품종은 기존 설명에 없으면 추정이라고만 적으세요.
출력은 한국어 2~4문장으로만 작성하세요.
불필요한 서론, 번호, 마크다운, JSON은 쓰지 마세요.

[기존 공고 설명]
{base_desc}

[기초 정보]
- 공고번호: {desertion_no}
- 종 코드: {breed_code}
- 성별: {sex}
- 나이: {age}
- 체중: {weight}
- 중성화: {neuter}
- 현재 상태: {process_state}

사진을 보고 기존 설명에 덧붙일 수 있는 관찰만 간결하게 작성하세요.
"""


def configure_hf_cache_env() -> None:
    cache_dir = BASE_DIR / ".hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    hub_cache = cache_dir / "hub"
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(hub_cache)
    os.environ["HF_HUB_DISABLE_XET"] = "1"


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def ensure_writable_hf_cache() -> None:
    configure_hf_cache_env()


def resolve_model_path(model_path_arg: Optional[str]) -> Tuple[str, bool]:
    candidate_value = model_path_arg or DEFAULT_LOCAL_MODEL_PATH
    candidate = Path(candidate_value).expanduser() if candidate_value else None
    if candidate and candidate.exists():
        return str(candidate), True
    return DEFAULT_MODEL_ID, False


def build_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": "Mozilla/5.0", "Accept": "image/*"})
    return session


def load_payload(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return {"items": payload}
    return payload


def save_payload(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def download_image(session: requests.Session, url: str) -> Image.Image:
    resp = session.get(url, timeout=20)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def build_prompt(item: Dict[str, Any]) -> str:
    return PROMPT_TEMPLATE.format(
        base_desc=clean_text(item.get("desc")) or "없음",
        desertion_no=clean_text(item.get("desertionNo")) or "Unknown",
        breed_code=clean_text(item.get("breed_code")) or "Unknown",
        sex=clean_text(item.get("sex")) or "Unknown",
        age=clean_text(item.get("age")) or "Unknown",
        weight=clean_text(item.get("weight")) or "Unknown",
        neuter=clean_text(item.get("neuter")) or "Unknown",
        process_state=clean_text(item.get("process_state")) or "Unknown",
    )


def merge_description(base_desc: str, vlm_desc: str) -> str:
    base_desc = clean_text(base_desc)
    vlm_desc = clean_text(vlm_desc)
    if base_desc and vlm_desc:
        return f"기존 설명: {base_desc}\n사진 보강 설명: {vlm_desc}"
    return base_desc or vlm_desc


def generate_vlm_description(
    processor: Any,
    model: Any,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int,
    use_cache: bool,
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    if torch.cuda.is_available():
        inputs = inputs.to(model.device, dtype=torch.bfloat16)
    else:
        inputs = inputs.to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=use_cache,
        )

    generated = outputs[0][inputs["input_ids"].shape[-1] :]
    text = processor.decode(generated, skip_special_tokens=True).strip()
    return text


def enrich_items(
    payload: Dict[str, Any],
    processor: Any,
    model: Any,
    limit: Optional[int],
    overwrite: bool,
    max_new_tokens: int,
    output_path: Path,
    checkpoint_every: int,
    image_max_size: int,
    use_cache: bool,
    retry_only_oom: bool,
    skip_non_retryable_errors: bool,
) -> Dict[str, int]:
    items = payload.get("items") or []
    session = build_session()
    attempted = 0
    updated = 0
    skipped = 0
    errors = 0

    for item in items:
        if clean_text(item.get("vlm_desc")) and not overwrite:
            skipped += 1
            continue

        error_text = clean_text(item.get("vlm_error"))
        if retry_only_oom:
            if not error_text or "CUDA out of memory" not in error_text:
                skipped += 1
                continue
        elif skip_non_retryable_errors and "404 Client Error" in error_text:
            skipped += 1
            continue

        image_url = clean_text(item.get("image_url"))
        if not image_url:
            item["vlm_error"] = "missing_image_url"
            errors += 1
            continue

        if limit is not None and attempted >= limit:
            break
        attempted += 1

        try:
            image = download_image(session, image_url)
            if image_max_size > 0:
                image.thumbnail((image_max_size, image_max_size))
            prompt = build_prompt(item)
            vlm_desc = generate_vlm_description(
                processor=processor,
                model=model,
                image=image,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                use_cache=use_cache,
            )
            item["vlm_desc"] = vlm_desc
            item["merged_desc"] = merge_description(item.get("desc", ""), vlm_desc)
            item["vlm_model"] = clean_text(getattr(model.config, "_name_or_path", "")) or DEFAULT_MODEL_ID
            item["vlm_updated_at"] = datetime.now().isoformat(timespec="seconds")
            item.pop("vlm_error", None)
            updated += 1
            print(f"[OK] {item.get('desertionNo')} enriched")
            if checkpoint_every > 0 and updated % checkpoint_every == 0:
                save_payload(payload, output_path)
                print(f"[CKPT] saved after {updated} updates -> {output_path}")
        except Exception as exc:
            item["vlm_error"] = str(exc)
            errors += 1
            print(f"[ERR] {item.get('desertionNo')} {exc}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return {"attempted": attempted, "updated": updated, "skipped": skipped, "errors": errors}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="유기견 JSON에 Gemma 3 VLM 설명을 추가합니다."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="입력 JSON 경로")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="출력 JSON 경로")
    parser.add_argument("--model-path", help="로컬 Gemma 3 snapshot 경로")
    parser.add_argument("--limit", type=int, default=None, help="처리할 최대 항목 수")
    parser.add_argument("--overwrite", action="store_true", help="이미 vlm_desc가 있어도 덮어쓰기")
    parser.add_argument("--max-new-tokens", type=int, default=120, help="생성 토큰 수")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="몇 건마다 중간 저장할지")
    parser.add_argument("--image-max-size", type=int, default=512, help="입력 이미지 최대 한 변 크기")
    parser.add_argument("--gpu-max-memory", default="18GiB", help="GPU 최대 메모리")
    parser.add_argument("--cpu-max-memory", default="96GiB", help="CPU 최대 메모리")
    parser.set_defaults(use_cache=True)
    parser.add_argument("--use-cache", action="store_true", dest="use_cache", help="generation cache 사용")
    parser.add_argument("--no-use-cache", action="store_false", dest="use_cache", help="generation cache 미사용")
    parser.add_argument("--load-in-8bit", action="store_true", help="bitsandbytes 8bit로 모델 로드")
    parser.add_argument("--gpu-only", action="store_true", help="CPU offload 없이 모델 전체를 GPU에 로드")
    parser.add_argument("--retry-only-oom", action="store_true", help="기존 OOM 실패 항목만 재시도")
    parser.add_argument("--skip-non-retryable-errors", action="store_true", help="404 같은 비재시도 오류는 건너뛰기")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_writable_hf_cache()
    from transformers import AutoProcessor, BitsAndBytesConfig, Gemma3ForConditionalGeneration

    model_path, local_files_only = resolve_model_path(args.model_path)
    payload = load_payload(args.input)

    print(f"[INFO] model={model_path}")
    print("[INFO] loading processor...")
    processor = AutoProcessor.from_pretrained(
        model_path,
        use_fast=False,
        local_files_only=local_files_only,
    )

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print("[INFO] loading model...")
    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch_dtype,
        "local_files_only": local_files_only,
    }
    if args.gpu_only:
        model_kwargs["device_map"] = {"": 0}
    elif torch.cuda.is_available() and not args.load_in_8bit:
        model_kwargs["max_memory"] = {0: args.gpu_max_memory, "cpu": args.cpu_max_memory}
    if args.load_in_8bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=False,
        )
        model_kwargs.pop("torch_dtype", None)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_path,
        **model_kwargs,
    )

    stats = enrich_items(
        payload=payload,
        processor=processor,
        model=model,
        limit=args.limit,
        overwrite=args.overwrite,
        max_new_tokens=args.max_new_tokens,
        output_path=args.output,
        checkpoint_every=args.checkpoint_every,
        image_max_size=args.image_max_size,
        use_cache=args.use_cache,
        retry_only_oom=args.retry_only_oom,
        skip_non_retryable_errors=args.skip_non_retryable_errors,
    )
    save_payload(payload, args.output)

    print(f"[DONE] saved: {args.output}")
    print(
        f"[INFO] attempted={stats['attempted']} updated={stats['updated']} skipped={stats['skipped']} errors={stats['errors']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
