import os
import sys
from pathlib import Path

from scripts import smoke_full_runtime


def test_smoke_runtime_bootstrap_uses_repository_absolute_paths(
    monkeypatch,
) -> None:
    for name in ("INDEX_PATH", "METAS_PATH"):
        monkeypatch.delenv(name, raising=False)

    smoke_full_runtime.configure_offline_cpu_runtime()

    repository_root = smoke_full_runtime.REPOSITORY_ROOT
    assert str(repository_root) in sys.path
    assert Path(os.environ["INDEX_PATH"]) == (
        repository_root / "data" / "dog_faiss.index"
    )
    assert Path(os.environ["METAS_PATH"]) == (
        repository_root / "data" / "dog_metas.json"
    )
    assert Path(os.environ["INDEX_PATH"]).is_absolute()
    assert Path(os.environ["METAS_PATH"]).is_absolute()
    assert os.environ["APP_ENV"] == "contest"
    assert len(os.environ["API_KEY"]) >= 24
    assert os.environ["GEMMA3_ENABLED"] == "false"
    assert os.environ["RELEASE_PROFILE"] == "full"


def test_smoke_runtime_bootstrap_accepts_public_text_artifact_paths(
    tmp_path: Path,
) -> None:
    index_path = tmp_path / "dog_faiss.index"
    metas_path = tmp_path / "dog_metas.json"
    marker_path = tmp_path / "release_profile.json"

    smoke_full_runtime.configure_offline_cpu_runtime(
        profile=smoke_full_runtime.PUBLIC_TEXT_PACKAGE_PROFILE,
        index_path=index_path,
        metas_path=metas_path,
        release_profile_path=marker_path,
    )

    assert os.environ["RELEASE_PROFILE"] == "public-text-only"
    assert Path(os.environ["INDEX_PATH"]) == index_path.resolve()
    assert Path(os.environ["METAS_PATH"]) == metas_path.resolve()
    assert Path(os.environ["RELEASE_PROFILE_PATH"]) == marker_path.resolve()
