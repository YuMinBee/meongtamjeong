"""Evaluate trained CLIP-text-to-DINO heads on untouched held-out notices."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.heldout_image_evaluation import (  # noqa: E402
    ImageDownloadError,
    SafePublicImageDownloader,
    aggregate_rank_metrics,
    rank_record,
)
from experiments.dino_fusion.alignment import (  # noqa: E402
    alignment_attributes,
    blend_embeddings,
    evaluation_prompts,
    project_embeddings,
    projection_head_from_checkpoint,
)
from experiments.dino_fusion.build_index import (  # noqa: E402
    sha256_file,
    write_json_atomic,
)
from experiments.dino_fusion.core import (  # noqa: E402
    ClipEncoder,
    DinoEncoder,
    clean_text,
    collapse_visual_hits,
    safe_read_faiss_index,
)
from experiments.dino_fusion.evaluate_pilot import (  # noqa: E402
    attribute_relevance_summary,
    mean_latency,
    metas_by_notice,
    paired_rank_comparison,
    secondary_url,
)


DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
DEFAULT_ALLOWED_HOSTS = ("openapi.animal.go.kr",)
REPORT_SCHEMA_VERSION = "dino-text-alignment-evaluation.v2"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _year_from_report(report: Mapping[str, Any]) -> int:
    text = clean_text(report.get("reference_date"))
    return int(text[:4]) if len(text) >= 4 and text[:4].isdigit() else 2026


def load_heads(
    *,
    report: Mapping[str, Any],
    artifact_dir: Path,
    device: str,
    torch: Any,
) -> dict[str, Any]:
    heads: dict[str, Any] = {}
    for architecture, details in (report.get("architectures") or {}).items():
        checkpoint_path = artifact_dir / clean_text(details.get("checkpoint"))
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")
        if sha256_file(checkpoint_path) != clean_text(details.get("checkpoint_sha256")):
            raise ValueError(f"checkpoint hash mismatch: {checkpoint_path.name}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        head = projection_head_from_checkpoint(checkpoint).to(device)
        head.load_state_dict(checkpoint["state_dict"])
        head.eval()
        heads[architecture] = head
    if not heads:
        raise ValueError("training report contains no architectures")
    return heads


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--heldout-report",
        type=Path,
        default=ROOT / "docs/evaluation/heldout_image_retrieval.appearance_v1.json",
    )
    parser.add_argument(
        "--clip-index", type=Path, default=ROOT / "data/dog_faiss.index"
    )
    parser.add_argument("--clip-metas", type=Path, default=ROOT / "data/dog_metas.json")
    parser.add_argument(
        "--dino-index", type=Path, default=DEFAULT_ARTIFACT_DIR / "dino.index"
    )
    parser.add_argument(
        "--dino-metas", type=Path, default=DEFAULT_ARTIFACT_DIR / "dino_metas.json"
    )
    parser.add_argument(
        "--dino-manifest",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "dino_manifest.json",
    )
    parser.add_argument(
        "--training-report",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "alignment_training_report.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "alignment_evaluation_report.json",
    )
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--text-weight", type=float, default=0.2)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=12.0)
    parser.add_argument("--allowed-host", action="append", dest="allowed_hosts")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.topk <= 0 or args.limit < 0:
        raise ValueError("topk must be positive and limit must be non-negative")
    if not 0.0 <= args.text_weight <= 1.0:
        raise ValueError("text-weight must be between zero and one")
    torch = __import__("torch")
    faiss = __import__("faiss")
    device = (
        "cuda"
        if args.device == "auto" and bool(torch.cuda.is_available())
        else ("cpu" if args.device == "auto" else args.device)
    )

    heldout = load_json(args.heldout_report)
    clip_metas = load_json(args.clip_metas)
    dino_metas = load_json(args.dino_metas)
    dino_manifest = load_json(args.dino_manifest)
    training_report = load_json(args.training_report)
    if not isinstance(clip_metas, list) or not isinstance(dino_metas, list):
        raise ValueError("index metadata must be JSON lists")
    clip_index = safe_read_faiss_index(args.clip_index, faiss.read_index)
    dino_index = safe_read_faiss_index(args.dino_index, faiss.read_index)
    if int(clip_index.ntotal) != len(clip_metas):
        raise ValueError("CLIP index/metadata row mismatch")
    if int(dino_index.ntotal) != len(dino_metas):
        raise ValueError("DINO index/metadata row mismatch")

    dino_model = dino_manifest.get("model") or {}
    dino_model_id = clean_text(dino_model.get("id"))
    dino_revision = clean_text(dino_model.get("resolved_revision")) or None
    clip_encoder = ClipEncoder(args.clip_model, device=device)
    dino_encoder = DinoEncoder(
        dino_model_id,
        device=device,
        revision=dino_revision,
        local_files_only=args.local_files_only,
    )
    heads = load_heads(
        report=training_report,
        artifact_dir=args.training_report.parent,
        device=device,
        torch=torch,
    )
    if int(dino_index.d) != int(dino_encoder.dimension):
        raise ValueError("DINO encoder/index dimension mismatch")

    source_indices = {
        int(meta["source_meta_index"])
        for meta in dino_metas
        if isinstance(meta.get("source_meta_index"), int)
    }
    if len(source_indices) != len(dino_metas):
        raise ValueError("DINO metadata source indices are missing or duplicated")
    current_by_id = metas_by_notice(clip_metas)
    sample_by_id = {
        clean_text(row.get("notice_id")): row for row in heldout.get("sample", [])
    }
    previous_queries = list(heldout.get("queries") or [])
    if args.limit:
        previous_queries = previous_queries[: args.limit]
    dino_notice_ids = {
        clean_text(meta.get("notice_id") or meta.get("desertionNo"))
        for meta in dino_metas
    }
    reference_year = _year_from_report(heldout)

    system_names = [
        "clip_image",
        "clip_text_en",
        "clip_text_ko",
        "clip_image_text_en",
        "clip_image_text_ko",
        "dino_image",
    ]
    for architecture in heads:
        for language in ("en", "ko"):
            system_names.extend(
                [
                    f"{architecture}_aligned_text_{language}",
                    f"{architecture}_image_text_{language}",
                ]
            )
    system_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in system_names}
    query_rows: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
    attempted = 0
    downloaded = 0
    downloader = SafePublicImageDownloader(
        timeout=args.timeout,
        allowed_hosts=tuple(args.allowed_hosts or DEFAULT_ALLOWED_HOSTS),
    )

    def dino_ranking(vector: Any) -> list[dict[str, Any]]:
        scores, indices = dino_index.search(vector, int(dino_index.ntotal))
        return collapse_visual_hits(
            scores[0].tolist(),
            indices[0].tolist(),
            dino_metas,
            score_kind="similarity",
        )

    def clip_crop_ranking(vector: Any) -> list[dict[str, Any]]:
        distances, indices = clip_index.search(vector, int(clip_index.ntotal))
        pairs = [
            (score, index)
            for score, index in zip(distances[0].tolist(), indices[0].tolist())
            if int(index) in source_indices
        ]
        return collapse_visual_hits(
            [pair[0] for pair in pairs],
            [pair[1] for pair in pairs],
            clip_metas,
            score_kind="distance",
        )

    try:
        for previous in previous_queries:
            dog_id = clean_text(previous.get("notice_id"))
            if dog_id not in dino_notice_ids:
                failures["target_missing_dino_crop"] += 1
                continue
            meta = current_by_id.get(dog_id) or {}
            attributes = alignment_attributes(meta, reference_year=reference_year)
            if attributes is None:
                failures["missing_alignment_attributes"] += 1
                continue
            sample = sample_by_id.get(dog_id) or {}
            url = secondary_url(meta)
            if not url:
                failures["missing_secondary_url"] += 1
                continue
            expected_url_hash = clean_text(sample.get("secondary_url_sha256"))
            actual_url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()
            if expected_url_hash and actual_url_hash != expected_url_hash:
                failures["secondary_url_changed"] += 1
                continue
            attempted += 1
            try:
                downloaded_image = downloader.fetch(url)
            except ImageDownloadError as exc:
                failures[f"download_{exc.reason}"] += 1
                continue
            try:
                if downloaded_image.payload_sha256 != clean_text(
                    previous.get("query_payload_sha256")
                ):
                    failures["secondary_payload_changed"] += 1
                    continue
                downloaded += 1
                prompts = evaluation_prompts(attributes)
                started = time.perf_counter()
                clip_text = clip_encoder.encode_text_batch(
                    [prompts["english"], prompts["korean"]]
                )
                clip_text_ms = (time.perf_counter() - started) * 1000
                started = time.perf_counter()
                clip_image = clip_encoder.encode_image(downloaded_image.image)
                clip_image_ms = (time.perf_counter() - started) * 1000
                started = time.perf_counter()
                dino_image = dino_encoder.encode(downloaded_image.image)
                dino_image_ms = (time.perf_counter() - started) * 1000
                rankings: dict[str, list[dict[str, Any]]] = {
                    "clip_image": clip_crop_ranking(clip_image),
                    "dino_image": dino_ranking(dino_image),
                    "clip_text_en": clip_crop_ranking(clip_text[0:1]),
                    "clip_text_ko": clip_crop_ranking(clip_text[1:2]),
                    "clip_image_text_en": clip_crop_ranking(
                        blend_embeddings(
                            clip_image,
                            clip_text[0:1],
                            text_weight=args.text_weight,
                        )
                    ),
                    "clip_image_text_ko": clip_crop_ranking(
                        blend_embeddings(
                            clip_image,
                            clip_text[1:2],
                            text_weight=args.text_weight,
                        )
                    ),
                }
                projection_times: dict[str, float] = {}
                for architecture, head in heads.items():
                    started = time.perf_counter()
                    with torch.inference_mode():
                        mapped_tensor = project_embeddings(
                            head, torch.as_tensor(clip_text, device=device)
                        )
                    mapped = mapped_tensor.detach().float().cpu().numpy()
                    projection_times[architecture] = (
                        time.perf_counter() - started
                    ) * 1000
                    for language_index, language in enumerate(("en", "ko")):
                        text_vector = mapped[language_index : language_index + 1]
                        rankings[f"{architecture}_aligned_text_{language}"] = (
                            dino_ranking(text_vector)
                        )
                        blended = blend_embeddings(
                            dino_image,
                            text_vector,
                            text_weight=args.text_weight,
                        )
                        rankings[f"{architecture}_image_text_{language}"] = (
                            dino_ranking(blended)
                        )

                records = {
                    system: rank_record(dog_id, ranking, top_k=args.topk)
                    for system, ranking in rankings.items()
                }
                for system, record in records.items():
                    system_rows[system].append(record)
                query_rows.append(
                    {
                        "notice_id": dog_id,
                        "sample_order": previous.get("sample_order"),
                        "payload_sha256": downloaded_image.payload_sha256,
                        "prompts": prompts,
                        "clip_text_encode_ms": round(clip_text_ms, 3),
                        "clip_image_encode_ms": round(clip_image_ms, 3),
                        "dino_image_encode_ms": round(dino_image_ms, 3),
                        "projection_ms": {
                            key: round(value, 3)
                            for key, value in projection_times.items()
                        },
                        "systems": records,
                    }
                )
            finally:
                downloaded_image.image.close()
    finally:
        downloader.close()

    systems = {
        system: aggregate_rank_metrics(
            rows,
            attempted_count=attempted,
            downloaded_count=downloaded,
            duplicate_count=0,
        )
        for system, rows in system_rows.items()
    }
    systems["dino_image"]["mean_encode_ms"] = mean_latency(
        query_rows, "dino_image_encode_ms"
    )
    systems["clip_image"]["mean_encode_ms"] = mean_latency(
        query_rows, "clip_image_encode_ms"
    )
    for language in ("en", "ko"):
        systems[f"clip_text_{language}"]["mean_encode_ms"] = mean_latency(
            query_rows, "clip_text_encode_ms"
        )
        systems[f"clip_image_text_{language}"]["mean_encode_ms"] = round(
            (mean_latency(query_rows, "clip_image_encode_ms") or 0.0)
            + (mean_latency(query_rows, "clip_text_encode_ms") or 0.0),
            3,
        )
    for architecture in heads:
        projection_values = [
            float((row.get("projection_ms") or {}).get(architecture, 0.0))
            for row in query_rows
        ]
        projection_mean = mean(projection_values) if projection_values else 0.0
        for language in ("en", "ko"):
            systems[f"{architecture}_aligned_text_{language}"]["mean_encode_ms"] = (
                round(
                    (mean_latency(query_rows, "clip_text_encode_ms") or 0.0)
                    + projection_mean,
                    3,
                )
            )
            systems[f"{architecture}_image_text_{language}"]["mean_encode_ms"] = round(
                (mean_latency(query_rows, "clip_text_encode_ms") or 0.0)
                + (mean_latency(query_rows, "dino_image_encode_ms") or 0.0)
                + projection_mean,
                3,
            )

    attribute_relevance = {
        system: attribute_relevance_summary(
            query_rows,
            system=system,
            metas=current_by_id,
            candidate_notice_ids=dino_notice_ids,
        )
        for system in system_rows
    }
    paired_comparisons: dict[str, Any] = {}
    for architecture in heads:
        for language in ("en", "ko"):
            paired_comparisons[
                f"{architecture}_image_text_{language}_vs_dino_image"
            ] = paired_rank_comparison(
                query_rows,
                left=f"{architecture}_image_text_{language}",
                right="dino_image",
            )

    validation_selected_architecture = min(
        heads,
        key=lambda name: float(
            ((training_report.get("architectures") or {}).get(name) or {}).get(
                "best_validation_loss", float("inf")
            )
        ),
    )
    aligned_text_deltas: dict[str, Any] = {}
    for architecture in heads:
        for language in ("en", "ko"):
            system = f"{architecture}_aligned_text_{language}"
            baseline = f"clip_text_{language}"
            aligned_text_deltas[system] = {
                "attribute_ndcg_delta_vs_same_language_clip": round(
                    float(attribute_relevance[system]["nDCG@10"])
                    - float(attribute_relevance[baseline]["nDCG@10"]),
                    6,
                ),
                "attribute_precision10_delta_vs_same_language_clip": round(
                    float(attribute_relevance[system]["precision@10"])
                    - float(attribute_relevance[baseline]["precision@10"]),
                    6,
                ),
            }
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "training_required": True,
        "evaluation_leakage_check": {
            "heldout_pool_excluded_from_training": True,
            "evaluation_prompt_seen_during_training": False,
            "checkpoint_hashes_verified": True,
        },
        "scope": {
            "query_source": "previously verified distinct secondary photos",
            "query_storage": "memory_only",
            "candidate_corpus": "notices with local Faster R-CNN crops",
            "text_prompt_policy": (
                "unseen bilingual templates derived only from public color, weight-based size, and age group"
            ),
            "same_space_fusion": (
                "normalized convex blend of DINO image and projected CLIP text vectors"
            ),
        },
        "models": {
            "clip": {"id": args.clip_model, "frozen": True},
            "dino": dino_model,
            "heads": training_report.get("architectures") or {},
        },
        "corpus": {
            "clip_vectors": int(clip_index.ntotal),
            "dino_crop_vectors": int(dino_index.ntotal),
            "target_covered_and_attributed_queries": attempted,
            "payload_verified_queries": downloaded,
        },
        "fusion": {"text_weight": args.text_weight, "tuned_on_heldout": False},
        "systems": systems,
        "attribute_relevance": attribute_relevance,
        "paired_comparisons": paired_comparisons,
        "comparison_summary": {
            "validation_selected_architecture": validation_selected_architecture,
            "heldout_used_for_model_selection": False,
            "aligned_text_deltas": aligned_text_deltas,
        },
        "failures": dict(sorted(failures.items())),
        "queries": query_rows,
        "limitations": [
            "The held-out set is small and was created for image identity retrieval.",
            "Text prompts are controlled metadata-derived queries, not independent user language.",
            "Color/size attribute relevance is a silver metric, not human relevance judgment.",
            "The fixed 0.2 text blend weight was chosen a priori and not tuned on held-out data.",
            "This evaluates a DINOv2 fallback and must be repeated after DINOv3 access approval.",
        ],
    }
    write_json_atomic(args.output, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    summary = {
        "schema_version": report["schema_version"],
        "corpus": report["corpus"],
        "systems": report["systems"],
        "attribute_relevance": report["attribute_relevance"],
        "paired_comparisons": report["paired_comparisons"],
        "comparison_summary": report["comparison_summary"],
        "failures": report["failures"],
        "output": str(args.output),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
