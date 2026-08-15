from __future__ import annotations

import json
import stat
import subprocess
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app import public_text_evaluation_summary as evaluation_summary
from app.public_text_release import (
    PUBLIC_TEXT_METADATA_FIELDS,
    deterministic_json_bytes,
    derive_public_text_release,
    public_notice_text,
    public_text_sha256,
)
from scripts import package_release as release


@pytest.fixture
def git_repo(tmp_path: Path) -> Iterator[Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Package Test"],
        check=True,
    )
    yield repo


def _write(path: Path, content: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")


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


def _track(repo: Path, *paths: str) -> None:
    subprocess.run(["git", "-C", str(repo), "add", "--", *paths], check=True)


def _commit(repo: Path, message: str = "fixture") -> str:
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", message], check=True)
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def _write_tagged_release_fixture(
    repo: Path, *, profile: str = release.FULL_RELEASE_PROFILE
) -> None:
    paths = sorted(release.required_release_paths(profile))
    for name in paths:
        target = repo / name
        if target.exists():
            continue
        if name == ".env.contest.example":
            content: str | bytes = (
                "API_KEY=change-me\n"
                "INDEX_PATH=./data/dog_faiss.index\n"
                "METAS_PATH=./data/dog_metas.json\n"
            )
        elif name == "data/dog_faiss.index":
            content = b"fixture-index"
        elif name.endswith(".json"):
            content = "{}\n"
        else:
            content = "fixture\n"
        _write(target, content)
    _track(repo, *paths)


class _ProfileFakeIndex:
    def __init__(
        self,
        dimension: int,
        *,
        vectors: list[list[float]] | None = None,
        metric_type: int = 1,
    ) -> None:
        self.d = dimension
        self.metric_type = metric_type
        self.vectors = [np.asarray(row, dtype=np.float32) for row in vectors or []]

    @property
    def ntotal(self) -> int:
        return len(self.vectors)

    def reconstruct(self, position: int) -> np.ndarray:
        return self.vectors[position].copy()

    def add(self, matrix: np.ndarray) -> None:
        self.vectors.extend(np.asarray(matrix, dtype=np.float32).copy())


class _ProfileFakeFaiss:
    METRIC_L2 = 1

    @staticmethod
    def IndexFlatL2(dimension: int) -> _ProfileFakeIndex:
        return _ProfileFakeIndex(dimension)

    @staticmethod
    def serialize_index(index: _ProfileFakeIndex) -> np.ndarray:
        payload = json.dumps(
            {
                "d": index.d,
                "metric_type": index.metric_type,
                "vectors": [row.tolist() for row in index.vectors],
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return np.frombuffer(payload, dtype=np.uint8).copy()

    @staticmethod
    def deserialize_index(value: np.ndarray) -> _ProfileFakeIndex:
        payload = json.loads(np.asarray(value, dtype=np.uint8).tobytes().decode())
        return _ProfileFakeIndex(
            int(payload["d"]),
            vectors=payload["vectors"],
            metric_type=int(payload["metric_type"]),
        )


def _profile_vector(position: int) -> list[float]:
    vector = [0.0] * 512
    vector[position] = 1.0
    return vector


def _profile_text_meta(notice_id: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "desertionNo": notice_id,
        "type": "text",
        "embedding_source": "public_notice_text",
        "desc": f"공개 설명 {notice_id}",
        "sex": "M",
        "age": "2024(년생)",
        "weight": "5(Kg)",
        "neuter": "Y",
        "color": "흰색",
        "detail_url": f"https://www.animal.go.kr/notice/{notice_id}",
        "image_url": f"https://photos.example/{notice_id}.jpg",
        "image_urls": [f"https://photos.example/{notice_id}.jpg"],
        "image_attrs": {"photo_quality_score": 0.9},
    }
    row["desc_full"] = public_notice_text(row)
    row["embedding_text_sha256"] = public_text_sha256(row["desc_full"])
    return row


def _write_public_text_source_fixture(repo: Path) -> None:
    text_meta = _profile_text_meta("dog-a")
    metas = [
        {**text_meta, "type": "image", "embedding_source": "full_image"},
        text_meta,
        {**text_meta, "type": "crop_image", "embedding_source": "dog_crop"},
    ]
    index = _ProfileFakeIndex(
        512,
        vectors=[_profile_vector(0), _profile_vector(1), _profile_vector(2)],
    )
    _write(
        repo / ".env.contest.example",
        "API_KEY=change-me\n"
        "INDEX_PATH=./data/dog_faiss.index\n"
        "METAS_PATH=./data/dog_metas.json\n",
    )
    _write(
        repo / "README.md",
        "full vectors: 3,746\n"
        "[visual evaluation](docs/evaluation/heldout_image_retrieval.appearance_v1.md)\n",
    )
    _write(repo / "DATA_CARD.md", "same-name artifact has 3,746 rows\n")
    _write(repo / "MODEL_CARD.md", "heldout image metrics from full profile\n")
    _write(
        repo / "THIRD_PARTY_NOTICES.md",
        "[dependency report](docs/dependency-report.md)\n",
    )
    _write(repo / "docs" / "dependency-report.md", "# Dependencies\n")
    _write(repo / "docs" / "dependency-report.json", "{}\n")
    _write(
        repo / "docs" / "contest-development-report-draft.md",
        "full-profile vectors: 3,746\n",
    )
    _write(repo / "CONTRIBUTING.md", "run the full release gate\n")
    _write(repo / ".github" / "workflows" / "ci.yml", "name: full-ci\n")
    _write(
        repo / "data" / "dog_faiss.index",
        _ProfileFakeFaiss.serialize_index(index).tobytes(),
    )
    _write(repo / "data" / "dog_metas.json", deterministic_json_bytes(metas))
    _write(
        repo / "data" / "snapshot_manifest.json",
        deterministic_json_bytes(
            {
                "schema_version": 1,
                "generated_at": "2026-07-26T00:00:00+09:00",
                "source": {"provider": "public-provider"},
                "embedding": {
                    "model": "OpenAI CLIP ViT-B/32",
                    "vector_types": {"image": 1, "crop_image": 1, "text": 1},
                },
            }
        ),
    )
    _write(repo / "data" / "active_index_sync_report.json", "{}\n")
    profile_neutral_tooling = {
        "app/notice_provider.py": "# provider contract\n",
        "app/portal_candidate_selection_study.py": "# portal study contract\n",
        "scripts/fetch_live_dogs.py": "# provider collection CLI\n",
        "scripts/portal_candidate_selection_study.py": "# portal study CLI\n",
        "docs/provider-adapter.md": "# Provider adapter\n",
        "docs/evaluation/PORTAL_CANDIDATE_SELECTION_PROTOCOL.md": (
            "# Portal candidate selection protocol\n"
        ),
        "docs/evaluation/portal_candidate_selection.protocol.v1.json": "{}\n",
        "docs/evaluation/templates/portal_candidate_selection_trials.csv": (
            "protocol_id,participant_code\n"
        ),
        "docs/evaluation/templates/portal_candidate_selection_blind_labels.csv": (
            "protocol_id,candidate_ref\n"
        ),
        "tests/test_appearance_query.py": "# appearance query tests\n",
        "tests/test_notice_metadata.py": "# notice metadata tests\n",
        "tests/test_notice_provider.py": "# provider tests\n",
        "tests/test_portal_candidate_selection_study.py": "# portal study tests\n",
    }
    for path, content in profile_neutral_tooling.items():
        _write(repo / path, content)
    _write(
        repo / "data" / "examples" / "notice-provider.synthetic.json",
        json.dumps(
            {
                "description": "synthetic provider contract fixture",
                "items": [
                    {
                        "notice_id": "SYNTH-DOG-001",
                        "image_urls": [
                            "https://example.invalid/assets/synthetic-dog.jpg"
                        ],
                        "detail_url": ("https://example.invalid/notices/SYNTH-DOG-001"),
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n",
    )
    full_profile_holdout_files = {
        "app/query_holdout_evaluation.py",
        "app/query_holdout_v3_evaluation.py",
        "scripts/evaluate_query_holdout.py",
        "scripts/evaluate_query_holdout_v3.py",
        "data/eval_query_holdout.appearance_v2.json",
        "data/eval_query_holdout.appearance_v2.json.sha256",
        "data/eval_query_holdout.appearance_v3.json",
        "data/eval_query_holdout.appearance_v3.sha256",
        "docs/evaluation/query_holdout.appearance_v2.json",
        "docs/evaluation/query_holdout.appearance_v2.md",
        "docs/evaluation/query_holdout.appearance_v3.json",
        "docs/evaluation/query_holdout.appearance_v3.md",
        "docs/evaluation/query_holdout.appearance_v3.run.json",
        "tests/test_query_holdout_evaluation.py",
        "tests/test_query_holdout_v3_evaluation.py",
    }
    for path in full_profile_holdout_files:
        _write(repo / path, "full-profile holdout artifact\n")
    _write(
        repo / "docs" / "evaluation" / "heldout_image_retrieval.appearance_v1.json",
        "{}\n",
    )
    _write(
        repo / "docs" / "evaluation" / "query_robustness.appearance_v1.json",
        "{}\n",
    )
    _write(
        repo / "docs" / "evaluation" / "query_robustness.appearance_v1.md",
        "full profile robustness\n",
    )
    _write(repo / "assets" / "samples" / "notice.jpg", b"public photo bytes")
    _write(
        repo / "assets" / "samples" / "manifest.json",
        '{"source_image_url":"https://photos.example/dog-a.jpg"}\n',
    )
    _write(
        repo / "assets" / "samples" / "README.md",
        "![sample](notice.jpg)\n",
    )
    _track(
        repo,
        ".env.contest.example",
        "README.md",
        "DATA_CARD.md",
        "MODEL_CARD.md",
        "THIRD_PARTY_NOTICES.md",
        "docs/dependency-report.md",
        "docs/dependency-report.json",
        "docs/contest-development-report-draft.md",
        "CONTRIBUTING.md",
        ".github/workflows/ci.yml",
        "data/dog_faiss.index",
        "data/dog_metas.json",
        "data/snapshot_manifest.json",
        "data/active_index_sync_report.json",
        *profile_neutral_tooling,
        "data/examples/notice-provider.synthetic.json",
        *full_profile_holdout_files,
        "docs/evaluation/heldout_image_retrieval.appearance_v1.json",
        "docs/evaluation/query_robustness.appearance_v1.json",
        "docs/evaluation/query_robustness.appearance_v1.md",
        "assets/samples/notice.jpg",
        "assets/samples/manifest.json",
        "assets/samples/README.md",
    )


def _evaluation_summary_payload(
    *,
    index_sha256: str,
    metas_sha256: str,
    vector_count: int,
    unique_notice_count: int,
    dimension: int,
) -> dict[str, Any]:
    stable_sha = "c" * 64
    retrieval_metrics = {
        "query_count": 12,
        "precision@5": 0.5,
        "recall@5": 0.4,
        "recall@10": 0.6,
        "nDCG@5": 0.55,
        "MRR": 0.7,
        "hit@5": 0.8,
        "inactive_exposure": 0.0,
    }
    return {
        "artifacts": {
            "dimension": dimension,
            "index_sha256": index_sha256,
            "metas_sha256": metas_sha256,
            "unique_notice_count": unique_notice_count,
            "vector_count": vector_count,
        },
        "evaluation_inputs": {
            "profile_config_sha256": stable_sha,
            "retrieval_queries_sha256": stable_sha,
            "robustness_base_queries_sha256": stable_sha,
            "robustness_variants_sha256": stable_sha,
            "safety_config_sha256": stable_sha,
        },
        "label_policy": {
            "relevance": "explicit public notice fields",
            "source_fields": ["color", "age", "weight"],
            "uses_ranked_results": False,
            "uses_result_text": False,
            "uses_vlm_attributes": False,
            "version": "silver-fixture-v1",
        },
        "limitations": list(evaluation_summary.LIMITATIONS),
        "not_applicable": list(evaluation_summary.NOT_APPLICABLE),
        "profile": evaluation_summary.SUMMARY_PROFILE,
        "reference_date": "2026-07-26",
        "reproduction_commands": list(evaluation_summary.REPRODUCTION_COMMANDS),
        "results": {
            "exposure_representation": {
                "overall": {
                    "applicable_count": 10,
                    "evaluated_count": 8,
                    "unknown_count": 2,
                    "evidence_coverage": 0.8,
                },
                "system": "hybrid_natural_graph",
                "validation_passed": True,
            },
            "profile_reranking": {
                "summary": {
                    "profile_count": 2,
                    "profiles_with_different_top1_from_baseline": 1,
                    "distinct_profile_top1_count": 2,
                    "unknown_neutrality_failures": 0,
                    "evidence_consistency_failures": 0,
                    "adoption_suitability_certainty_phrase_hits": 0,
                    "quality_sensitivity_failures": 0,
                },
                "validation_passed": True,
            },
            "query_robustness": {
                "metrics": dict(retrieval_metrics),
                "stability": {
                    "variant_count": 4,
                    "mean_top5_jaccard": 0.7,
                    "mean_top10_jaccard": 0.8,
                    "mean_rbo@10": 0.75,
                    "min_top5_jaccard": 0.5,
                    "min_top10_jaccard": 0.6,
                    "min_rbo@10": 0.55,
                    "top10_exact_match_rate": 0.25,
                },
                "validation_passed": True,
            },
            "retrieval": {
                "metrics": retrieval_metrics,
                "role": "headline_natural_only",
                "system": "hybrid_natural_graph",
                "validation_passed": True,
            },
            "safety_contract": {
                "summary": {
                    "total_contracts": 4,
                    "passed_contracts": 4,
                    "failed_contracts": 0,
                },
                "validation_passed": True,
            },
        },
        "schema_version": evaluation_summary.SUMMARY_SCHEMA_VERSION,
    }


def _write_public_text_evidence_fixture(repo: Path, evidence_dir: Path) -> Path:
    artifacts = derive_public_text_release(
        (repo / "data" / "dog_faiss.index").read_bytes(),
        (repo / "data" / "dog_metas.json").read_bytes(),
        source_manifest_bytes=(repo / "data" / "snapshot_manifest.json").read_bytes(),
        faiss_module=_ProfileFakeFaiss,
    )
    encoded = evaluation_summary.encode_public_text_evaluation_summary(
        _evaluation_summary_payload(
            index_sha256=artifacts.index_sha256,
            metas_sha256=artifacts.metas_sha256,
            vector_count=artifacts.vector_count,
            unique_notice_count=artifacts.unique_notice_count,
            dimension=artifacts.dimension,
        )
    )
    _write(
        evidence_dir / evaluation_summary.SUMMARY_JSON_NAME,
        encoded.json_bytes,
    )
    _write(
        evidence_dir / evaluation_summary.SUMMARY_MARKDOWN_NAME,
        encoded.markdown_bytes,
    )
    return evidence_dir


def test_packages_current_bytes_of_tracked_files_and_excludes_risky_paths(
    git_repo: Path, tmp_path: Path
) -> None:
    _write(git_repo / "README.md", "committed\n")
    _write(git_repo / ".env", "API_KEY=real-but-excluded\n")
    _write(git_repo / "weights.pt", b"model")
    _write(git_repo / "pkg" / "__pycache__" / "module.pyc", b"\0cache")
    _write(git_repo / ".env.example", "API_KEY=change-me\n")
    _track(
        git_repo,
        "README.md",
        ".env",
        "weights.pt",
        "pkg/__pycache__/module.pyc",
        ".env.example",
    )

    # Working-tree bytes, rather than the staged blob, must be submitted.
    _write(git_repo / "README.md", "dirty but intended\n")
    # Explicitly excluded local artifacts stay outside the package.
    _write(git_repo / ".env.local", "OPENAI_API_KEY=sk-" + "z" * 30 + "\n")

    output = tmp_path / "submission.zip"
    result = release.package_release(git_repo, output=output)

    assert result.checked_only is False
    assert result.file_count == 2
    assert result.excluded_count == 3
    assert result.sha256
    with zipfile.ZipFile(output) as archive:
        assert archive.namelist() == [".env.example", "README.md"]
        assert archive.read("README.md") == (git_repo / "README.md").read_bytes()
        assert archive.read(".env.example") == (git_repo / ".env.example").read_bytes()
        for info in archive.infolist():
            mode = (info.external_attr >> 16) & 0xFFFF
            assert stat.S_ISREG(mode)


@pytest.mark.parametrize("check_only", [True, False])
def test_untracked_release_source_fails_closed_without_exposing_contents(
    git_repo: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    check_only: bool,
) -> None:
    _write(git_repo / "README.md", "tracked\n")
    _track(git_repo, "README.md")
    secret_contents = "private implementation detail that must not be displayed"
    _write(git_repo / "src" / "new_feature.py", secret_contents)
    output = tmp_path / "must-not-exist.zip"
    args = ["--repo", str(git_repo), "--output", str(output)]
    if check_only:
        args.append("--check-only")

    exit_code = release.main(args)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert not output.exists()
    assert secret_contents not in captured.out
    assert secret_contents not in captured.err
    assert "new_feature.py" not in captured.out
    assert "new_feature.py" not in captured.err
    payload = json.loads(captured.err)
    assert payload["ok"] is False
    assert "1 release-eligible untracked file(s)" in payload["error"]
    assert "stage/commit first" in payload["error"]


def test_ignored_and_explicitly_excluded_untracked_files_are_allowed(
    git_repo: Path,
) -> None:
    _write(git_repo / ".gitignore", "ignored.log\n")
    _write(git_repo / "README.md", "tracked\n")
    _track(git_repo, ".gitignore", "README.md")
    _write(git_repo / "ignored.log", "ignored by Git\n")
    _write(git_repo / ".env", "API_KEY=local-secret\n")
    _write(git_repo / ".env.local", "API_KEY=another-local-secret\n")
    _write(git_repo / "weights.pt", b"model")
    _write(git_repo / "pkg" / "__pycache__" / "module.pyc", b"\0cache")

    result = release.package_release(git_repo, check_only=True)

    assert result.file_count == 2


def test_existing_output_archive_is_not_treated_as_untracked_source(
    git_repo: Path,
) -> None:
    _write(git_repo / "README.md", "tracked\n")
    _track(git_repo, "README.md")
    output = git_repo / "dist" / "submission.zip"
    _write(output, b"stale generated archive")

    checked = release.package_release(
        git_repo,
        output=output,
        check_only=True,
    )
    built = release.package_release(git_repo, output=output)

    assert checked.checked_only is True
    assert built.checked_only is False
    assert output.is_file()
    with zipfile.ZipFile(output) as archive:
        assert archive.namelist() == ["README.md"]


def test_untracked_unsafe_paths_and_symlinks_fail_without_disclosing_names(
    git_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unsafe_name = "../sensitive-untracked-name"
    with pytest.raises(release.PackagingError) as unsafe_error:
        release._parse_untracked_paths(f"{unsafe_name}\0".encode())
    assert "unsafe untracked path" in str(unsafe_error.value)
    assert unsafe_name not in str(unsafe_error.value)

    link_name = "sensitive-untracked-link"
    monkeypatch.setattr(
        release,
        "_run_git",
        lambda _repo, *_args: f"{link_name}\0".encode(),
    )
    monkeypatch.setattr(
        Path,
        "lstat",
        lambda _self: type("FakeStat", (), {"st_mode": stat.S_IFLNK})(),
    )

    with pytest.raises(release.PackagingError) as link_error:
        release._reject_release_eligible_untracked_files(
            git_repo,
            output_resolved=None,
        )
    assert "symbolic link" in str(link_error.value)
    assert link_name not in str(link_error.value)


def test_archive_is_deterministic_and_check_only_does_not_write(
    git_repo: Path, tmp_path: Path
) -> None:
    _write(git_repo / "b.txt", "B\n")
    _write(git_repo / "a.txt", "A\n")
    _track(git_repo, "a.txt", "b.txt")

    check_target = tmp_path / "check-only.zip"
    checked = release.package_release(git_repo, output=check_target, check_only=True)
    assert checked.checked_only is True
    assert checked.output is None
    assert checked.sha256 is None
    assert not check_target.exists()

    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    first_result = release.package_release(git_repo, output=first)
    second_result = release.package_release(git_repo, output=second)

    assert first.read_bytes() == second.read_bytes()
    assert first_result.sha256 == second_result.sha256
    with zipfile.ZipFile(first) as archive:
        assert archive.namelist() == ["a.txt", "b.txt"]
        assert {info.date_time for info in archive.infolist()} == {
            release.FIXED_ZIP_TIMESTAMP
        }


def test_tracked_secret_aborts_without_printing_secret_value(
    git_repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    secret = "-".join(("opaque", "public", "data", "credential", "123456789"))
    secret_name = "_".join(("ANIMAL", "API", "KEY"))
    _write(git_repo / "settings.py", f'{secret_name} = "{secret}"\n')
    _track(git_repo, "settings.py")
    output = tmp_path / "must-not-exist.zip"

    exit_code = release.main(["--repo", str(git_repo), "--output", str(output)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert not output.exists()
    assert secret not in captured.out
    assert secret not in captured.err
    payload = json.loads(captured.err)
    assert payload["ok"] is False
    assert "settings.py:1" in payload["error"]
    assert "secret values were not displayed" in payload["error"]

    binary_token = b"ghp_" + b"a" * 36
    findings = release.scan_secrets("asset.bin", b"\0prefix\n" + binary_token)
    assert [(item.path, item.kind) for item in findings] == [
        ("asset.bin", "github-token")
    ]


def test_placeholder_values_are_allowed(git_repo: Path) -> None:
    _write(
        git_repo / ".env.example",
        "API_KEY=change-me\nCLIENT_SECRET=${CLIENT_SECRET}\nPASSWORD=your-password\n",
    )
    _write(
        git_repo / "config.py",
        'API_KEY = os.getenv("API_KEY", "")\n'
        'api_key = request.query_params.get("api_key", "")\n',
    )
    _track(git_repo, ".env.example", "config.py")

    result = release.package_release(git_repo, check_only=True)

    assert result.file_count == 2


def test_runtime_secret_variables_are_not_mistaken_for_embedded_values() -> None:
    embedded_value = "-".join(("opaque", "public", "data", "credential", "123456789"))
    javascript = (
        f"let apiKey = '';\napiKey = normalized;\napiKey = '{embedded_value}';\n"
    ).encode()

    findings = release.scan_secrets("app/demo.js", javascript)

    assert {(item.line, item.kind) for item in findings} == {
        (3, "environment-secret"),
        (3, "named-secret"),
    }
    env_findings = release.scan_secrets(".env.example", b"API_KEY=normalized\n")
    assert [(item.line, item.kind) for item in env_findings] == [
        (1, "environment-secret")
    ]

    python_keywords = b'provider(\n    api_key=api_key,\n    client_secret="",\n)\n'
    assert release.scan_secrets("app/provider.py", python_keywords) == []


def test_clean_release_records_head_and_can_require_exact_tag(
    git_repo: Path, tmp_path: Path
) -> None:
    _write(git_repo / "README.md", "release\n")
    _write_tagged_release_fixture(git_repo)
    commit = _commit(git_repo)
    subprocess.run(
        ["git", "-C", str(git_repo), "tag", "-a", "v1.0.0", "-m", "release"],
        check=True,
    )

    output = tmp_path / "release.zip"
    result = release.package_release(
        git_repo,
        output=output,
        require_clean=True,
        required_tag="v1.0.0",
    )

    assert result.commit_sha == commit
    assert result.as_dict()["source"] == "clean HEAD working tree"
    assert result.release_tag == "v1.0.0"
    assert result.tag_object_id is not None
    assert result.release_manifest == release.RELEASE_MANIFEST_PATH
    with zipfile.ZipFile(output) as archive:
        manifest = json.loads(archive.read(release.RELEASE_MANIFEST_PATH))
        assert manifest["project"] == "meongtamjeong"
        assert manifest["profile"] == release.FULL_RELEASE_PROFILE
        assert manifest["release_tag"] == "v1.0.0"
        assert manifest["commit_sha"] == commit
        assert manifest["tag_object_id"] == result.tag_object_id
        payload_names = {item["path"] for item in manifest["files"]}
        assert payload_names == set(archive.namelist()) - {
            release.RELEASE_MANIFEST_PATH
        }
        for item in manifest["files"]:
            data = archive.read(item["path"])
            assert item["size"] == len(data)
            assert item["sha256"] == release.hashlib.sha256(data).hexdigest()


def test_tagged_public_text_release_embeds_profile_evidence(
    git_repo: Path,
    tmp_path: Path,
) -> None:
    _write_public_text_source_fixture(git_repo)
    _write_tagged_release_fixture(
        git_repo,
        profile=release.PUBLIC_TEXT_PACKAGE_PROFILE,
    )
    evidence_dir = _write_public_text_evidence_fixture(
        git_repo,
        tmp_path / "public-tag-evidence",
    )
    commit = _commit(git_repo, "public release")
    tag = "public-text-v1"
    subprocess.run(
        ["git", "-C", str(git_repo), "tag", "-a", tag, "-m", "release"],
        check=True,
    )

    output = tmp_path / "public-release.zip"
    result = release.package_release(
        git_repo,
        output=output,
        required_tag=tag,
        profile=release.PUBLIC_TEXT_PACKAGE_PROFILE,
        public_evidence_dir=evidence_dir,
        faiss_module=_ProfileFakeFaiss,
    )

    assert result.commit_sha == commit
    assert result.profile == release.PUBLIC_TEXT_PACKAGE_PROFILE
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        assert (
            release.required_release_paths(release.PUBLIC_TEXT_PACKAGE_PROFILE) <= names
        )
        manifest = json.loads(archive.read(release.RELEASE_MANIFEST_PATH))
        assert manifest["profile"] == release.PUBLIC_TEXT_PACKAGE_PROFILE
        assert manifest["release_tag"] == tag
        summary_payload = json.loads(
            archive.read(release.PUBLIC_TEXT_EVALUATION_SUMMARY_JSON)
        )
        profile = json.loads(archive.read(release.PUBLIC_TEXT_PROFILE_MARKER))
        assert (
            summary_payload["artifacts"]["index_sha256"]
            == profile["output"]["index_sha256"]
        )
        assert (
            summary_payload["artifacts"]["metas_sha256"]
            == profile["output"]["metas_sha256"]
        )


def test_tagged_release_requires_complete_submission_surface(
    git_repo: Path, tmp_path: Path
) -> None:
    _write(git_repo / "README.md", "incomplete release\n")
    _track(git_repo, "README.md")
    _commit(git_repo)
    subprocess.run(
        ["git", "-C", str(git_repo), "tag", "-a", "incomplete", "-m", "release"],
        check=True,
    )

    with pytest.raises(release.PackagingError, match="mandatory submission"):
        release.package_release(
            git_repo,
            output=tmp_path / "incomplete.zip",
            required_tag="incomplete",
        )


def test_tagged_release_rejects_lightweight_tag(git_repo: Path, tmp_path: Path) -> None:
    _write_tagged_release_fixture(git_repo)
    _commit(git_repo)
    subprocess.run(["git", "-C", str(git_repo), "tag", "lightweight"], check=True)

    with pytest.raises(release.PackagingError, match="must be annotated"):
        release.package_release(
            git_repo,
            output=tmp_path / "lightweight.zip",
            required_tag="lightweight",
        )


def test_clean_release_rejects_staged_and_unstaged_tracked_changes(
    git_repo: Path,
) -> None:
    _write(git_repo / "README.md", "base\n")
    _track(git_repo, "README.md")
    _commit(git_repo)

    _write(git_repo / "README.md", "unstaged\n")
    with pytest.raises(release.PackagingError, match="match HEAD"):
        release.package_release(git_repo, check_only=True, require_clean=True)

    _track(git_repo, "README.md")
    with pytest.raises(release.PackagingError, match="match HEAD"):
        release.package_release(git_repo, check_only=True, require_clean=True)


def test_contest_environment_artifacts_must_be_in_release(git_repo: Path) -> None:
    _write(
        git_repo / ".env.contest.example",
        "API_KEY=change-me\n"
        "INDEX_PATH=./data/dog_faiss.index\n"
        "METAS_PATH=./data/dog_metas.json\n",
    )
    _track(git_repo, ".env.contest.example")

    with pytest.raises(release.PackagingError, match="not included in release"):
        release.package_release(git_repo, check_only=True)

    _write(git_repo / "data" / "dog_faiss.index", b"index")
    _write(git_repo / "data" / "dog_metas.json", "[]\n")
    _track(git_repo, "data/dog_faiss.index", "data/dog_metas.json")

    result = release.package_release(git_repo, check_only=True)

    assert result.file_count == 3


def test_rejects_symlink_git_mode_and_path_traversal(
    git_repo: Path, tmp_path: Path
) -> None:
    blob = (
        subprocess.run(
            ["git", "-C", str(git_repo), "hash-object", "-w", "--stdin"],
            input=b"target.txt",
            check=True,
            stdout=subprocess.PIPE,
        )
        .stdout.decode("ascii")
        .strip()
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(git_repo),
            "update-index",
            "--add",
            "--cacheinfo",
            f"120000,{blob},unsafe-link",
        ],
        check=True,
    )

    with pytest.raises(release.PackagingError, match="not a regular file"):
        release.package_release(git_repo, check_only=True)
    with pytest.raises(release.PackagingError, match="unsafe archive path"):
        release._validate_archive_name("../escape.txt")

    malicious_zip = tmp_path / "malicious.zip"
    with zipfile.ZipFile(malicious_zip, "w") as archive:
        archive.writestr("../escape.txt", "bad")
    with pytest.raises(release.PackagingError):
        release.validate_zip(malicious_zip, [])


def test_public_text_profile_replaces_canonical_data_and_excludes_visual_artifacts(
    git_repo: Path,
    tmp_path: Path,
) -> None:
    _write_public_text_source_fixture(git_repo)
    commit = _commit(git_repo, "public text fixture")
    first = tmp_path / "public-text-first.zip"
    second = tmp_path / "public-text-second.zip"
    evidence_dir = _write_public_text_evidence_fixture(
        git_repo,
        tmp_path / "public-text-evidence",
    )

    first_result = release.package_release(
        git_repo,
        output=first,
        profile=release.PUBLIC_TEXT_PACKAGE_PROFILE,
        public_evidence_dir=evidence_dir,
        faiss_module=_ProfileFakeFaiss,
    )
    second_result = release.package_release(
        git_repo,
        output=second,
        profile=release.PUBLIC_TEXT_PACKAGE_PROFILE,
        public_evidence_dir=evidence_dir,
        faiss_module=_ProfileFakeFaiss,
    )

    assert first.read_bytes() == second.read_bytes()
    assert first_result.sha256 == second_result.sha256
    assert first_result.commit_sha == commit
    assert first_result.profile == release.PUBLIC_TEXT_PACKAGE_PROFILE
    assert first_result.profile_summary is not None
    assert first_result.profile_summary["vectors"] == 1

    with zipfile.ZipFile(first) as archive:
        names = set(archive.namelist())
        assert "assets/samples/notice.jpg" not in names
        assert not any(name.startswith("assets/samples/") for name in names)
        assert "data/active_index_sync_report.json" not in names
        assert "docs/evaluation/heldout_image_retrieval.appearance_v1.json" not in names
        assert "docs/evaluation/query_robustness.appearance_v1.json" not in names
        assert "docs/evaluation/query_robustness.appearance_v1.md" not in names
        assert "data/release_profile.json" in names
        assert "data/public_text_release_report.json" in names
        assert "docs/dependency-report.md" in names
        assert "docs/dependency-report.json" in names
        assert "docs/contest-development-report-draft.md" not in names
        assert "CONTRIBUTING.md" in names
        assert not any(name.startswith(".github/") for name in names)
        assert "docs/public-text-only-release.md" in names
        assert release.PUBLIC_TEXT_EVALUATION_SUMMARY_JSON in names
        assert release.PUBLIC_TEXT_EVALUATION_SUMMARY_MARKDOWN in names
        assert release.PUBLIC_TEXT_REQUIRED_PROFILE_NEUTRAL_TOOLING <= names
        assert "docs/provider-adapter.md" in names
        assert "docs/evaluation/PORTAL_CANDIDATE_SELECTION_PROTOCOL.md" in names
        assert "tests/test_appearance_query.py" in names
        assert "tests/test_notice_metadata.py" in names
        assert "tests/test_notice_provider.py" in names
        assert "tests/test_portal_candidate_selection_study.py" in names
        assert not any(
            release._is_public_text_full_profile_holdout(name) for name in names
        )
        assert not any(
            name.startswith("docs/") and name not in release.PUBLIC_TEXT_RETAINED_DOCS
            for name in names
        )

        public_readme = archive.read("README.md").decode("utf-8")
        assert "public-text-only-v1" in public_readme
        assert "3,746" not in public_readme
        assert "docs/provider-adapter.md" in public_readme
        assert "docs/evaluation/PORTAL_CANDIDATE_SELECTION_PROTOCOL.md" in public_readme
        assert "query_holdout.appearance_v2" not in public_readme
        assert "query_holdout.appearance_v3" not in public_readme
        assert "source_image_url" not in public_readme
        assert ".invalid" in public_readme
        assert "canonical 운영 검색 artifact" in public_readme
        assert "public-text-only evaluation summary" in public_readme
        assert "텍스트 벡터 1개" in public_readme
        assert "conda env create -f environment.release.yml" in public_readme
        assert "conda activate meongtamjeong-release" in public_readme
        assert "RandomNumberGenerator" in public_readme
        assert 'curl.exe -H "x-api-key: $env:API_KEY"' in public_readme
        assert "replace-me-with-at-least-24-random-characters" not in public_readme

        public_data_card = archive.read("DATA_CARD.md").decode("utf-8")
        public_model_card = archive.read("MODEL_CARD.md").decode("utf-8")
        assert "public-text-only-v1" in public_data_card
        assert "공개 공고 텍스트 메타 1행" in public_data_card
        assert "3,746" not in public_data_card
        assert ".invalid" in public_data_card
        assert "canonical 운영 검색 artifact" in public_data_card
        assert "public-text-only-v1" in public_model_card
        assert "heldout image metrics" not in public_model_card
        assert "public-text-only-v1" in archive.read("CONTRIBUTING.md").decode("utf-8")
        assert "N/A" in archive.read("docs/public-text-only-release.md").decode("utf-8")

        metas = json.loads(archive.read("data/dog_metas.json"))
        assert len(metas) == 1
        assert set(metas[0]) <= PUBLIC_TEXT_METADATA_FIELDS
        assert metas[0]["type"] == "text"
        assert "image_url" not in metas[0]
        assert "image_urls" not in metas[0]
        assert "image_attrs" not in metas[0]

        profile = json.loads(archive.read("data/release_profile.json"))
        summary = json.loads(archive.read(release.PUBLIC_TEXT_EVALUATION_SUMMARY_JSON))
        assert summary["artifacts"]["index_sha256"] == profile["output"]["index_sha256"]
        assert summary["artifacts"]["metas_sha256"] == profile["output"]["metas_sha256"]
        assert "generated_at" not in json.dumps(summary)
        summary_keys = _all_mapping_keys(summary)
        assert not any("latency" in key for key in summary_keys)
        assert not any("runtime" in key for key in summary_keys)
        assert profile["profile"] == "public-text-only-v1"
        assert profile["runtime_policy"] == {
            "graph_overlay_enabled": False,
            "photo_quality_enabled": False,
            "remote_notice_photo_display_enabled": False,
            "visual_asset_routes_enabled": False,
        }
        derived_index = _ProfileFakeFaiss.deserialize_index(
            np.frombuffer(archive.read("data/dog_faiss.index"), dtype=np.uint8)
        )
        assert derived_index.ntotal == 1
        assert np.array_equal(
            derived_index.reconstruct(0),
            np.asarray(_profile_vector(1), dtype=np.float32),
        )


def test_full_profile_keeps_existing_package_bytes_unchanged(
    git_repo: Path,
    tmp_path: Path,
) -> None:
    _write_public_text_source_fixture(git_repo)
    _commit(git_repo, "full profile fixture")
    output = tmp_path / "full.zip"

    result = release.package_release(
        git_repo,
        output=output,
        profile=release.FULL_RELEASE_PROFILE,
        require_clean=True,
    )

    assert result.profile == release.FULL_RELEASE_PROFILE
    assert result.profile_summary is None
    with zipfile.ZipFile(output) as archive:
        assert (
            archive.read("data/dog_faiss.index")
            == (git_repo / "data" / "dog_faiss.index").read_bytes()
        )
        assert (
            archive.read("data/dog_metas.json")
            == (git_repo / "data" / "dog_metas.json").read_bytes()
        )
        assert "assets/samples/notice.jpg" in archive.namelist()
        assert "assets/samples/manifest.json" in archive.namelist()
        assert b"3,746" in archive.read("README.md")
        assert "data/release_profile.json" not in archive.namelist()
        assert "app/query_holdout_evaluation.py" in archive.namelist()
        assert "app/query_holdout_v3_evaluation.py" in archive.namelist()
        assert "docs/evaluation/query_holdout.appearance_v3.json" in archive.namelist()


def test_public_text_profile_requires_clean_head_even_programmatically(
    git_repo: Path,
) -> None:
    _write_public_text_source_fixture(git_repo)
    _commit(git_repo, "public text clean fixture")
    _write(git_repo / "data" / "snapshot_manifest.json", "{}\n")

    with pytest.raises(release.PackagingError, match="match HEAD"):
        release.package_release(
            git_repo,
            check_only=True,
            require_clean=False,
            profile=release.PUBLIC_TEXT_PACKAGE_PROFILE,
            faiss_module=_ProfileFakeFaiss,
        )


def test_public_text_profile_requires_gate_generated_evidence(
    git_repo: Path,
) -> None:
    _write_public_text_source_fixture(git_repo)
    _commit(git_repo, "public text evidence fixture")

    with pytest.raises(release.PackagingError, match="gate-generated"):
        release.package_release(
            git_repo,
            check_only=True,
            profile=release.PUBLIC_TEXT_PACKAGE_PROFILE,
            faiss_module=_ProfileFakeFaiss,
        )


def test_public_text_cli_rejects_allow_dirty_before_packaging(
    git_repo: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = release.main(
        [
            "--repo",
            str(git_repo),
            "--check-only",
            "--profile",
            release.PUBLIC_TEXT_PACKAGE_PROFILE,
            "--allow-dirty",
        ]
    )

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().err)
    assert payload["ok"] is False
    assert "not permitted" in payload["error"]


def test_public_text_markdown_link_validation_fails_closed() -> None:
    with pytest.raises(release.PackagingError, match="broken local link"):
        release._validate_public_text_markdown_links(
            [
                release.TrackedFile(
                    "README.md",
                    "100644",
                    b"[missing](docs/removed.md)\n",
                )
            ]
        )


def test_public_text_provider_example_rejects_routable_url_safely() -> None:
    actual_url = "https://photos.example.org/dog.jpg"
    item = release.TrackedFile(
        release.PUBLIC_TEXT_PROVIDER_EXAMPLE,
        "100644",
        json.dumps({"items": [{"image_urls": [actual_url]}]}).encode(),
    )

    with pytest.raises(release.PackagingError) as captured:
        release._validate_public_text_provider_example(item)

    assert "routable or credentialed URL" in str(captured.value)
    assert actual_url not in str(captured.value)
