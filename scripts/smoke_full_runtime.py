"""Exercise the real CPU CLIP/FAISS application path without external APIs."""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


# Running ``python scripts/smoke_full_runtime.py`` makes ``scripts/`` the
# first import root. Add the repository root explicitly so the documented
# command and CI workflow can import ``app`` from any current directory.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


SMOKE_AUTH_VALUE = "-".join(("local", "runtime", "smoke", "contest", "key"))
FULL_RELEASE_PROFILE = "full"
PUBLIC_TEXT_PACKAGE_PROFILE = "public-text-only"
PUBLIC_TEXT_RUNTIME_PROFILE = "public-text-only-v1"
REQUIRED_RESULT_KEYS = {
    "retrieval_score",
    "compatibility_score",
    "quality_score",
    "final_score",
    "matched_conditions",
    "caution_conditions",
    "unknown_conditions",
    "recommendation_reason",
    "source_url",
}


def configure_offline_cpu_runtime(
    *,
    profile: str = FULL_RELEASE_PROFILE,
    index_path: Path | None = None,
    metas_path: Path | None = None,
    release_profile_path: Path | None = None,
) -> None:
    """Set deterministic, non-secret settings before importing ``app.main``."""

    os.environ["API_KEY"] = SMOKE_AUTH_VALUE
    os.environ["APP_ENV"] = "contest"
    os.environ["ANIMAL_API_KEY"] = ""
    os.environ["INDEX_PATH"] = str(
        (index_path or REPOSITORY_ROOT / "data" / "dog_faiss.index").resolve()
    )
    os.environ["METAS_PATH"] = str(
        (metas_path or REPOSITORY_ROOT / "data" / "dog_metas.json").resolve()
    )
    os.environ["RELEASE_PROFILE"] = profile
    if release_profile_path is not None:
        os.environ["RELEASE_PROFILE_PATH"] = str(release_profile_path.resolve())
    else:
        os.environ.pop("RELEASE_PROFILE_PATH", None)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["GEMMA3_ENABLED"] = "false"
    os.environ["NOTICE_FILTER_INACTIVE"] = "true"
    os.environ["NOTICE_INCLUDE_UNKNOWN"] = "false"
    os.environ["PROFILE_INCLUDE_UNKNOWN_NOTICES"] = "false"


def require_ok(response: Any, label: str) -> Mapping[str, Any]:
    if response.status_code != 200:
        raise RuntimeError(f"{label} returned HTTP {response.status_code}")
    payload = response.json()
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"{label} returned a non-object JSON response")
    return payload


def require_results(payload: Mapping[str, Any], label: str) -> list[Mapping[str, Any]]:
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        raise RuntimeError(f"{label} returned no candidates")
    for position, result in enumerate(results):
        if not isinstance(result, Mapping):
            raise RuntimeError(f"{label} result {position} is not an object")
        missing = REQUIRED_RESULT_KEYS - set(result)
        if missing:
            raise RuntimeError(
                f"{label} result {position} is missing keys: "
                + ", ".join(sorted(missing))
            )
        meta = result.get("meta")
        if isinstance(meta, Mapping) and meta.get("notice_status") in {
            "closed",
            "expired",
        }:
            raise RuntimeError(f"{label} exposed an inactive notice")
    return results


def generated_reference_png() -> bytes:
    """Create a tiny synthetic image in memory; no redistributable asset needed."""

    from PIL import Image

    image = Image.new("RGB", (96, 96), color=(238, 231, 216))
    for x in range(24, 72):
        for y in range(30, 78):
            image.putpixel((x, y), (125, 86, 54))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def run_smoke(
    *,
    skip_image: bool = False,
    profile: str = FULL_RELEASE_PROFILE,
    index_path: Path | None = None,
    metas_path: Path | None = None,
    release_profile_path: Path | None = None,
) -> dict[str, Any]:
    configure_offline_cpu_runtime(
        profile=profile,
        index_path=index_path,
        metas_path=metas_path,
        release_profile_path=release_profile_path,
    )

    from fastapi.testclient import TestClient

    from app import main as main_module

    headers = {"x-api-key": SMOKE_AUTH_VALUE}
    started = time.perf_counter()
    with TestClient(main_module.app) as client:
        missing_auth = client.get("/health")
        if missing_auth.status_code != 403:
            raise RuntimeError("contest runtime allowed a request without header auth")
        query_auth = client.get(f"/health?api_key={SMOKE_AUTH_VALUE}")
        if query_auth.status_code != 403:
            raise RuntimeError("contest runtime accepted a query-string API key")
        health = require_ok(client.get("/health", headers=headers), "health")
        if int(health.get("index_size") or 0) <= 0:
            raise RuntimeError("health reported an empty FAISS index")
        if health.get("notice_filter_inactive") is not True:
            raise RuntimeError("inactive-notice filtering is disabled")
        expected_runtime_profile = (
            PUBLIC_TEXT_RUNTIME_PROFILE
            if profile == PUBLIC_TEXT_PACKAGE_PROFILE
            else FULL_RELEASE_PROFILE
        )
        if health.get("release_profile") != expected_runtime_profile:
            raise RuntimeError("health reported the wrong release profile")
        if profile == PUBLIC_TEXT_PACKAGE_PROFILE:
            if health.get("vector_modalities") != {
                "text": int(health.get("index_size") or 0)
            }:
                raise RuntimeError("public-text runtime exposed a non-text vector")
            for field in (
                "public_notice_images_enabled",
                "public_notice_visual_asset_routes_enabled",
                "graph_overlay_enabled",
            ):
                if health.get(field) is not False:
                    raise RuntimeError(f"public-text runtime did not disable {field}")
            if not str(
                client.get("/health", headers=headers).headers.get(
                    "content-security-policy"
                )
                or ""
            ).strip():
                raise RuntimeError("public-text runtime response policy is missing")

        profile_response = client.post(
            "/search/profile",
            headers=headers,
            json={
                "query": "복슬복슬한 밝은 털의 작은 강아지",
                "ranking_scope": "appearance",
                "profile": {
                    "preferred_size": "small",
                    "preferred_age": "any",
                },
                "topk": 3,
            },
        )
        profile_payload = require_ok(profile_response, "appearance text search")
        if profile_payload.get("ranking_scope") != "appearance":
            raise RuntimeError("appearance text search used the wrong ranking scope")
        if set(profile_payload.get("profile") or {}) != {
            "preferred_size",
            "preferred_age",
        }:
            raise RuntimeError("appearance text search injected non-appearance fields")
        text_results = require_results(profile_payload, "appearance text search")
        if profile == PUBLIC_TEXT_PACKAGE_PROFILE:
            for result in text_results:
                if result.get("quality_score") is not None:
                    raise RuntimeError("public-text result exposed photo quality")
                meta = result.get("meta")
                if isinstance(meta, Mapping) and (
                    meta.get("image_url")
                    or meta.get("image_urls")
                    or meta.get("photo_quality_score") is not None
                ):
                    raise RuntimeError(
                        "public-text result exposed notice photo metadata"
                    )

        image_results: list[Mapping[str, Any]] = []
        if not skip_image:
            image_response = client.post(
                "/search/appearance/image",
                headers=headers,
                data={
                    "query": "밝은 털의 작은 강아지",
                    "preferred_size": "small",
                    "preferred_age": "any",
                    "topk": "2",
                },
                files={
                    "ref_image": (
                        "generated-smoke.png",
                        generated_reference_png(),
                        "image/png",
                    )
                },
            )
            image_payload = require_ok(image_response, "appearance image search")
            if image_payload.get("input_modality") != "image+text":
                raise RuntimeError("appearance image search lost its modality marker")
            image_results = require_results(
                image_payload,
                "appearance image search",
            )
            if profile == PUBLIC_TEXT_PACKAGE_PROFILE:
                if image_payload.get("candidate_vector_modalities") != ["text"]:
                    raise RuntimeError(
                        "public-text image search did not declare its text-vector corpus"
                    )
                if not str(image_payload.get("image_query_limitation") or "").strip():
                    raise RuntimeError(
                        "public-text image search omitted its quality limitation"
                    )

        if profile == PUBLIC_TEXT_PACKAGE_PROFILE:
            for route in (
                "/breeds/not-a-breed/images",
                "/visualize/image-crop/dog/not-a-file.jpg",
                "/visualize/image-audit",
                "/live/breeds",
            ):
                blocked = client.get(route, headers=headers)
                if blocked.status_code != 404:
                    raise RuntimeError(
                        "public-text runtime exposed a public-notice visual route"
                    )

        first = text_results[0]
        dog_id = str(first.get("dog_id") or "").strip()
        if not dog_id:
            raise RuntimeError("appearance result has no dog_id")
        contact_response = client.post(
            "/adoption/contact_card",
            headers=headers,
            json={
                "desertion_no": dog_id,
                "preferences": {
                    "desired_temperaments": ["calm"],
                    "housing_type": "apartment",
                },
            },
        )
        contact = require_ok(contact_response, "contact card")
        if not str(contact.get("phone_script") or "").strip():
            raise RuntimeError("contact card has no phone script")
        if not str(contact.get("email_body") or "").strip():
            raise RuntimeError("contact card has no email template")
        if contact.get("manual_contact_only") is not True:
            raise RuntimeError("contact card does not enforce manual contact")

    return {
        "ok": True,
        "release_profile": health.get("release_profile"),
        "vector_modalities": health.get("vector_modalities"),
        "device": health.get("device"),
        "index_size": int(health.get("index_size") or 0),
        "hybrid_docs": int(health.get("hybrid_docs") or 0),
        "text_result_count": len(text_results),
        "image_result_count": len(image_results),
        "contact_template": "verified",
        "external_api_used": False,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-image",
        action="store_true",
        help="skip the real image-embedding request for a faster text-only check",
    )
    parser.add_argument(
        "--profile",
        choices=(FULL_RELEASE_PROFILE, PUBLIC_TEXT_PACKAGE_PROFILE),
        default=FULL_RELEASE_PROFILE,
    )
    parser.add_argument("--index", type=Path)
    parser.add_argument("--metas", type=Path)
    parser.add_argument("--release-profile-marker", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_smoke(
            skip_image=bool(args.skip_image),
            profile=args.profile,
            index_path=args.index,
            metas_path=args.metas,
            release_profile_path=args.release_profile_marker,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
