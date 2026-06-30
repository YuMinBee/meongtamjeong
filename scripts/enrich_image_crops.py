from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote

import requests
from PIL import Image
from io import BytesIO

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.species import clean_species

DATA_DIR = BASE_DIR / "data"
DEFAULT_INPUT = DATA_DIR / "local_dog_cache.json"
DEFAULT_OUTPUT = DATA_DIR / "local_dog_cache_image_attrs.json"
DEFAULT_CROP_DIR = DATA_DIR / "image_crops"
TARGET_CLASS_NAMES = {
    "dog": {"dog"},
    "cat": {"cat"},
}


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def safe_url(url: str) -> str:
    return quote(str(url), safe=":/?&=#[]%")


def extract_image_url(rec: Dict[str, Any]) -> str:
    for key in ("image_url", "url", "popfile", "popfile1", "popfile2", "fileName", "thumb", "image", "img"):
        value = clean_text(rec.get(key))
        if value.startswith("http"):
            return value
    for key, value in rec.items():
        text = clean_text(value)
        if text.startswith("http") and any(token in str(key).lower() for token in ("pop", "file", "img", "thumb")):
            return text
    return ""


def load_records(path: Path) -> Tuple[Any, List[Dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        items = payload.get("items", [])
    else:
        items = payload
    if not isinstance(items, list):
        raise ValueError(f"input must be a list or an object with items: {path}")
    return payload, [item for item in items if isinstance(item, dict)]


def save_records(payload: Any, items: List[Dict[str, Any]], output: Path, stats: Dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, dict):
        updated = dict(payload)
        updated["items"] = items
        updated["image_crop_stats"] = stats
    else:
        updated = items
    output.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")


def download_image(url: str, timeout: int = 20) -> Image.Image:
    headers = {"Accept": "image/*", "Referer": "https://www.animal.go.kr/", "User-Agent": "Mozilla/5.0"}
    resp = requests.get(safe_url(url), timeout=timeout, headers=headers)
    if resp.status_code != 200 and url.startswith("http://"):
        resp = requests.get(safe_url("https://" + url[7:]), timeout=timeout, headers=headers)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")


@dataclass
class Detection:
    class_name: str
    confidence: float
    xyxy: Tuple[float, float, float, float]
    area_ratio: float


def normalize_bbox(xyxy: Tuple[float, float, float, float], width: int, height: int) -> List[float]:
    x1, y1, x2, y2 = xyxy
    return [
        round(max(0.0, min(1.0, x1 / width)), 4),
        round(max(0.0, min(1.0, y1 / height)), 4),
        round(max(0.0, min(1.0, x2 / width)), 4),
        round(max(0.0, min(1.0, y2 / height)), 4),
    ]


def expand_bbox(xyxy: Tuple[float, float, float, float], width: int, height: int, margin: float) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    pad_x = box_w * margin
    pad_y = box_h * margin
    return (
        max(0, int(round(x1 - pad_x))),
        max(0, int(round(y1 - pad_y))),
        min(width, int(round(x2 + pad_x))),
        min(height, int(round(y2 + pad_y))),
    )


def bbox_centered(xyxy: Tuple[float, float, float, float], width: int, height: int, tolerance: float = 0.24) -> bool:
    x1, y1, x2, y2 = xyxy
    cx = ((x1 + x2) / 2.0) / max(1, width)
    cy = ((y1 + y2) / 2.0) / max(1, height)
    return abs(cx - 0.5) <= tolerance and abs(cy - 0.5) <= tolerance


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def compute_photo_quality(image: Image.Image, prefix: str = "") -> Dict[str, Any]:
    """Estimate image usability with lightweight OpenCV features."""
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        return {f"{prefix}photo_quality_error": f"{exc.__class__.__name__}: {exc}"}

    arr = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean() / 255.0)
    contrast = float(gray.std() / 255.0)
    pixels = max(1, int(image.width) * int(image.height))

    sharpness_score = clamp01(blur_var / 600.0)
    contrast_score = clamp01(contrast / 0.28)
    exposure_score = clamp01(1.0 - abs(brightness - 0.52) / 0.52)
    resolution_score = clamp01(pixels / float(640 * 480))
    quality_score = (
        0.38 * sharpness_score
        + 0.22 * contrast_score
        + 0.20 * exposure_score
        + 0.20 * resolution_score
    )
    if quality_score >= 0.72:
        quality_band = "high"
    elif quality_score >= 0.50:
        quality_band = "medium"
    else:
        quality_band = "low"

    return {
        f"{prefix}photo_quality_score": round(quality_score, 4),
        f"{prefix}photo_quality_band": quality_band,
        f"{prefix}sharpness_score": round(sharpness_score, 4),
        f"{prefix}blur_laplacian_var": round(blur_var, 2),
        f"{prefix}brightness": round(brightness, 4),
        f"{prefix}contrast": round(contrast, 4),
        f"{prefix}exposure_score": round(exposure_score, 4),
        f"{prefix}resolution_score": round(resolution_score, 4),
    }


def load_yolo(model_name: str):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is required. Install with: pip install ultralytics opencv-python-headless") from exc
    return YOLO(model_name)


def detect_target(model: Any, image: Image.Image, species: str, conf: float, imgsz: int, device: str = "") -> List[Detection]:
    import numpy as np

    target_names = TARGET_CLASS_NAMES.get(species, {species})
    predict_kwargs = {"source": np.asarray(image), "conf": conf, "imgsz": imgsz, "verbose": False}
    if clean_text(device):
        predict_kwargs["device"] = clean_text(device)
    result = model.predict(**predict_kwargs)[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    names = getattr(model, "names", {}) or getattr(result, "names", {}) or {}
    width, height = image.size
    detections: List[Detection] = []
    for box in boxes:
        cls_id = int(box.cls[0].item()) if getattr(box, "cls", None) is not None else -1
        class_name = clean_text(names.get(cls_id, str(cls_id))).lower()
        if class_name not in target_names:
            continue
        xyxy_values = box.xyxy[0].detach().cpu().tolist()
        x1, y1, x2, y2 = [float(v) for v in xyxy_values]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        detections.append(
            Detection(
                class_name=class_name,
                confidence=float(box.conf[0].item()) if getattr(box, "conf", None) is not None else 0.0,
                xyxy=(x1, y1, x2, y2),
                area_ratio=area / max(1, width * height),
            )
        )
    detections.sort(key=lambda det: (det.confidence * 0.45) + (det.area_ratio * 0.55), reverse=True)
    return detections


def crop_filename(record: Dict[str, Any], index: int, species: str) -> str:
    dog_id = clean_text(record.get("desertionNo") or record.get("desertion_no") or record.get("notice_no"))
    safe_id = "".join(ch for ch in dog_id if ch.isalnum() or ch in ("-", "_")) or f"item_{index:05d}"
    return f"{safe_id}.jpg"


def process_records(args: argparse.Namespace) -> Dict[str, Any]:
    species = clean_species(args.species)
    payload, records = load_records(args.input)
    if args.limit > 0:
        work_records = records[: args.limit]
    else:
        work_records = records

    model = load_yolo(args.model)
    crop_dir = args.crop_dir / species
    crop_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "species": species,
        "model": args.model,
        "input": str(args.input),
        "output": str(args.output),
        "crop_dir": str(crop_dir),
        "records_total": len(records),
        "records_processed": 0,
        "attempted": 0,
        "skipped_existing_detected": 0,
        "downloaded": 0,
        "quality_scored": 0,
        "crop_quality_scored": 0,
        "detected": 0,
        "recovered": 0,
        "no_detection": 0,
        "errors": 0,
        "retry_missing_only": bool(args.retry_missing_only),
        "conf": args.conf,
        "imgsz": args.imgsz,
        "device": args.device,
    }

    def maybe_checkpoint() -> None:
        if args.checkpoint_every <= 0:
            return
        if stats["records_processed"] % args.checkpoint_every != 0:
            return
        save_records(payload, records, args.output, stats)
        print(json.dumps({"progress": stats}, ensure_ascii=False), flush=True)

    for idx, rec in enumerate(work_records):
        stats["records_processed"] += 1
        previous_attrs = rec.get("image_attrs") if isinstance(rec.get("image_attrs"), dict) else {}
        previous_detected = previous_attrs.get("target_detected") is True or previous_attrs.get("dog_detected") is True
        previous_error = clean_text(previous_attrs.get("error"))
        if args.retry_missing_only and previous_detected:
            stats["skipped_existing_detected"] += 1
            maybe_checkpoint()
            continue

        url = extract_image_url(rec)
        attrs: Dict[str, Any] = {
            "detector": args.model,
            "detector_type": "yolo_detection_only",
            "target_species": species,
            "target_detected": False,
            "detection_count": 0,
            "retry_missing_only": bool(args.retry_missing_only),
            "retry_conf": args.conf,
            "retry_imgsz": args.imgsz,
            "retry_device": args.device,
            "previous_target_detected": previous_detected,
            "previous_error": previous_error,
        }
        stats["attempted"] += 1
        if species == "dog":
            attrs["dog_detected"] = False
            attrs["dog_count"] = 0
        if not url:
            attrs["error"] = "missing_image_url"
            rec["image_attrs"] = attrs
            stats["errors"] += 1
            maybe_checkpoint()
            continue

        try:
            image = download_image(url)
            stats["downloaded"] += 1
            width, height = image.size
            attrs.update(compute_photo_quality(image))
            if "photo_quality_score" in attrs:
                stats["quality_scored"] += 1
            detections = detect_target(model, image, species=species, conf=args.conf, imgsz=args.imgsz, device=args.device)
            attrs.update({"image_width": width, "image_height": height, "detection_count": len(detections)})
            if species == "dog":
                attrs["dog_count"] = len(detections)
            if not detections:
                stats["no_detection"] += 1
                rec["image_attrs"] = attrs
                maybe_checkpoint()
                continue

            primary = detections[0]
            crop_box = expand_bbox(primary.xyxy, width, height, args.crop_margin)
            crop = image.crop(crop_box)
            out_crop_path = crop_dir / crop_filename(rec, idx, species)
            out_crop_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                rel_crop_path = out_crop_path.resolve().relative_to(BASE_DIR)
            except ValueError:
                rel_crop_path = out_crop_path.resolve()
            crop.save(out_crop_path, format="JPEG", quality=92)
            attrs.update(compute_photo_quality(crop, prefix="crop_"))
            if "crop_photo_quality_score" in attrs:
                stats["crop_quality_scored"] += 1

            attrs.update(
                {
                    "target_detected": True,
                    "primary_class": primary.class_name,
                    "primary_confidence": round(primary.confidence, 4),
                    "primary_bbox_xyxy": [round(v, 2) for v in primary.xyxy],
                    "primary_bbox_norm": normalize_bbox(primary.xyxy, width, height),
                    "crop_bbox_xyxy": list(crop_box),
                    "dog_area_ratio" if species == "dog" else "target_area_ratio": round(primary.area_ratio, 4),
                    "bbox_centered": bbox_centered(primary.xyxy, width, height),
                    "crop_path": str(rel_crop_path).replace("\\", "/"),
                    "crop_width": crop.size[0],
                    "crop_height": crop.size[1],
                }
            )
            if species == "dog":
                attrs["dog_detected"] = True
            if args.retry_missing_only and not previous_detected:
                attrs["recovered_by_retry"] = True
                stats["recovered"] += 1
            rec["image_attrs"] = attrs
            stats["detected"] += 1
            maybe_checkpoint()
        except Exception as exc:
            attrs["error"] = f"{exc.__class__.__name__}: {exc}"
            rec["image_attrs"] = attrs
            stats["errors"] += 1
            maybe_checkpoint()

    save_records(payload, records, args.output, stats)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create YOLO detection-only animal bbox/crop metadata for adoption notice images.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input cache/metas JSON path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output enriched JSON path.")
    parser.add_argument("--crop-dir", type=Path, default=DEFAULT_CROP_DIR, help="Directory where crop images are saved.")
    parser.add_argument("--species", default="dog", choices=("dog", "cat"), help="Target species class to detect.")
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics YOLO model name/path.")
    parser.add_argument("--limit", type=int, default=50, help="Maximum records to process. Use 0 for all records.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--device", default="", help="Ultralytics device, for example 0/cuda:0/cpu. Empty lets Ultralytics choose.")
    parser.add_argument("--crop-margin", type=float, default=0.08, help="BBox expansion ratio before saving crop.")
    parser.add_argument("--checkpoint-every", type=int, default=100, help="Save partial output and print progress every N processed records. Use 0 to disable.")
    parser.add_argument("--retry-missing-only", action="store_true", help="Preserve existing successful detections and retry only records without a detected crop.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = process_records(args)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()