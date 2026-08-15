from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from app import public_text_evaluation_summary as summary
from scripts import build_public_text_evaluation_summary as summary_cli


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _all_mapping_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key).casefold())
            keys.update(_all_mapping_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_all_mapping_keys(child))
    return keys


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    reports = tmp_path / "reports"
    index = tmp_path / "dog_faiss.index"
    metas = tmp_path / "dog_metas.json"
    marker = tmp_path / "release_profile.json"
    index.write_bytes(b"deterministic public text index")
    metas.write_bytes(b"[]\n")
    index_sha = hashlib.sha256(index.read_bytes()).hexdigest()
    metas_sha = hashlib.sha256(metas.read_bytes()).hexdigest()
    stable_sha = "a" * 64
    metrics = {
        "query_count": 12,
        "precision@5": 0.5,
        "recall@5": 0.4,
        "recall@10": 0.6,
        "nDCG@5": 0.55,
        "MRR": 0.7,
        "hit@5": 0.8,
        "inactive_exposure": 0.0,
    }
    _write_json(
        marker,
        {
            "profile": summary.SUMMARY_PROFILE,
            "output": {
                "dimension": 512,
                "index_sha256": index_sha,
                "metas_sha256": metas_sha,
                "unique_notice_count": 1516,
                "vector_count": 1516,
            },
        },
    )
    _write_json(
        reports / "retrieval.json",
        {
            "generated_at": "changes on every run",
            "reference_date": "2026-07-26",
            "artifacts": {
                "index": {"sha256": index_sha},
                "metas": {"sha256": metas_sha},
                "queries": {"sha256": stable_sha},
            },
            "label_policy": {
                "relevance": "explicit notice fields",
                "source_fields": ["color", "age", "weight"],
                "uses_ranked_results": False,
                "uses_result_text": False,
                "uses_vlm_attributes": False,
                "version": "silver-fixture-v1",
            },
            "systems": {
                "hybrid_natural_graph": {
                    "latency_ms": 999.0,
                    "metrics": metrics,
                    "role": "headline_natural_only",
                }
            },
            "validation": {"passed": True},
        },
    )
    _write_json(
        reports / "query-robustness.json",
        {
            "generated_at": "changes on every run",
            "reference_date": "2026-07-26",
            "artifacts": {
                "index": {"sha256": index_sha},
                "metas": {"sha256": metas_sha},
                "base_queries": {"sha256": stable_sha},
                "variants": {"sha256": stable_sha},
            },
            "robustness_summary": {
                "metrics": metrics,
                "stability": {
                    "variant_count": 48,
                    "mean_top5_jaccard": 0.7,
                    "mean_top10_jaccard": 0.8,
                    "mean_rbo@10": 0.75,
                    "min_top5_jaccard": 0.5,
                    "min_top10_jaccard": 0.6,
                    "min_rbo@10": 0.55,
                    "top10_exact_match_rate": 0.25,
                },
            },
            "validation": {"passed": True},
        },
    )
    _write_json(
        reports / "exposure.json",
        {
            "reference_date": "2026-07-26",
            "inputs": {
                "source_retrieval_artifact_sha256": {
                    "index": index_sha,
                    "metas": metas_sha,
                }
            },
            "systems": {
                "hybrid_natural_graph": {
                    "query_evidence_coverage": {
                        "overall": {
                            "applicable_count": 10,
                            "evaluated_count": 8,
                            "unknown_count": 2,
                            "evidence_coverage": 0.8,
                        }
                    }
                }
            },
            "validation": {"passed": True},
        },
    )
    _write_json(
        reports / "profile.json",
        {
            "evaluation_reference_date": "2026-07-26",
            "inputs": {
                "dog_metas_sha256": metas_sha,
                "profile_config_sha256": stable_sha,
            },
            "summary": {
                "profile_count": 2,
                "profiles_with_different_top1_from_baseline": 1,
                "distinct_profile_top1_count": 2,
                "unknown_neutrality_failures": 0,
                "evidence_consistency_failures": 0,
                "adoption_suitability_certainty_phrase_hits": 0,
                "quality_sensitivity_failures": 0,
                "passed": True,
            },
        },
    )
    _write_json(
        reports / "safety.json",
        {
            "evaluation_reference_date": "2026-07-26",
            "inputs": {
                "dog_metas_sha256": metas_sha,
                "config_sha256": stable_sha,
            },
            "summary": {
                "total_contracts": 4,
                "passed_contracts": 4,
                "failed_contracts": 0,
                "passed": True,
            },
        },
    )
    return reports, index, metas, marker


def test_builds_deterministic_profile_bound_summary(tmp_path: Path) -> None:
    reports, index, metas, marker = _fixture(tmp_path)

    first = summary.build_public_text_evaluation_summary(
        reports_dir=reports,
        index_path=index,
        metas_path=metas,
        profile_marker_path=marker,
    )
    second = summary.build_public_text_evaluation_summary(
        reports_dir=reports,
        index_path=index,
        metas_path=metas,
        profile_marker_path=marker,
    )

    assert first.json_bytes == second.json_bytes
    assert first.markdown_bytes == second.markdown_bytes
    assert (
        first.payload["artifacts"]["index_sha256"]
        == hashlib.sha256(index.read_bytes()).hexdigest()
    )
    serialized = first.json_bytes.decode("utf-8").casefold()
    assert "generated_at" not in serialized
    summary_keys = _all_mapping_keys(first.payload)
    assert not any("latency" in key for key in summary_keys)
    assert not any("runtime" in key for key in summary_keys)
    assert "held-out image retrieval" in serialized
    assert "n/a" in serialized
    assert "P@5" in first.markdown_bytes.decode("utf-8")


def test_builder_rejects_report_bound_to_another_artifact(tmp_path: Path) -> None:
    reports, index, metas, marker = _fixture(tmp_path)
    retrieval_path = reports / "retrieval.json"
    retrieval = json.loads(retrieval_path.read_text(encoding="utf-8"))
    retrieval["artifacts"]["index"]["sha256"] = "b" * 64
    _write_json(retrieval_path, retrieval)

    with pytest.raises(
        summary.PublicTextEvaluationSummaryError,
        match="not bound",
    ):
        summary.build_public_text_evaluation_summary(
            reports_dir=reports,
            index_path=index,
            metas_path=metas,
            profile_marker_path=marker,
        )


def test_builder_requires_every_evaluation_to_pass(tmp_path: Path) -> None:
    reports, index, metas, marker = _fixture(tmp_path)
    safety_path = reports / "safety.json"
    safety = json.loads(safety_path.read_text(encoding="utf-8"))
    safety["summary"]["passed"] = False
    _write_json(safety_path, safety)

    with pytest.raises(
        summary.PublicTextEvaluationSummaryError,
        match="did not pass",
    ):
        summary.build_public_text_evaluation_summary(
            reports_dir=reports,
            index_path=index,
            metas_path=metas,
            profile_marker_path=marker,
        )


def test_encoded_summary_rejects_volatile_or_tampered_content(tmp_path: Path) -> None:
    reports, index, metas, marker = _fixture(tmp_path)
    artifacts = summary.build_public_text_evaluation_summary(
        reports_dir=reports,
        index_path=index,
        metas_path=metas,
        profile_marker_path=marker,
    )
    payload = copy.deepcopy(dict(artifacts.payload))
    payload["generated_at"] = "volatile"
    with pytest.raises(
        summary.PublicTextEvaluationSummaryError,
        match="volatile",
    ):
        summary.encode_public_text_evaluation_summary(payload)

    with pytest.raises(
        summary.PublicTextEvaluationSummaryError,
        match="does not match",
    ):
        summary.validate_public_text_evaluation_summary_files(
            artifacts.json_bytes,
            artifacts.markdown_bytes + b"tampered",
            expected_index_sha256=artifacts.payload["artifacts"]["index_sha256"],
            expected_metas_sha256=artifacts.payload["artifacts"]["metas_sha256"],
        )


def test_cli_writes_canonical_summary_and_check_only_is_non_mutating(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    reports, index, metas, marker = _fixture(tmp_path)
    output = tmp_path / "output"
    base_args = [
        "--reports-dir",
        str(reports),
        "--index",
        str(index),
        "--metas",
        str(metas),
        "--profile-marker",
        str(marker),
        "--output-dir",
        str(output),
    ]

    assert summary_cli.main([*base_args, "--check-only"]) == 0
    assert not output.exists()
    assert json.loads(capsys.readouterr().out)["checked_only"] is True

    assert summary_cli.main(base_args) == 0
    assert (output / summary.SUMMARY_JSON_NAME).is_file()
    assert (output / summary.SUMMARY_MARKDOWN_NAME).is_file()
    assert json.loads(capsys.readouterr().out)["ok"] is True
