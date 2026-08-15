"""Run the fail-closed contest release verification suite.

The orchestrator intentionally does not relay or persist child-process output.
Each underlying command remains the source of its own detailed diagnostics;
this command emits only a small, value-free JSON summary suitable for a
release checklist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable, Sequence
from uuid import uuid4


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_STEP_TIMEOUT_SECONDS = 30 * 60
SCHEMA_VERSION = 1
FULL_RELEASE_PROFILE = "full"
PUBLIC_TEXT_RELEASE_PROFILE = "public-text-only"
RELEASE_PROFILE_CHOICES = (FULL_RELEASE_PROFILE, PUBLIC_TEXT_RELEASE_PROFILE)
PUBLIC_TEXT_STAGE = "dist/public-text-release-gate"
PUBLIC_TEXT_TESTS = (
    "tests/test_package_release.py",
    "tests/test_appearance_query.py",
    "tests/test_dependency_snapshot.py",
    "tests/test_notice_metadata.py",
    "tests/test_notice_provider.py",
    "tests/test_portal_candidate_selection_study.py",
    "tests/test_public_text_evaluation_summary.py",
    "tests/test_public_text_release.py",
    "tests/test_public_text_runtime.py",
    "tests/test_smoke_full_runtime.py",
    "tests/test_verify_contest_release.py",
    "tests/test_verify_release_archive.py",
)
RELEASE_MANIFEST_PATH = "release/manifest.json"
RELEASE_ARCHIVE_PATHS = {
    FULL_RELEASE_PROFILE: "dist/meongtamjeong-contest-full.zip",
    PUBLIC_TEXT_RELEASE_PROFILE: "dist/meongtamjeong-contest-public-text-only.zip",
}


@dataclass(frozen=True)
class Step:
    """One independently executable release check."""

    name: str
    argv: tuple[str, ...]
    inspect_stdout_for_cleanliness: bool = False
    slow: bool = False
    not_applicable_reason: str | None = None


Runner = Callable[..., subprocess.CompletedProcess[bytes]]


def current_local_date() -> date:
    """Return the operator machine's local calendar date."""

    return date.today()


def build_steps(
    *,
    reference_date: date,
    required_tag: str | None = None,
    python_executable: str | None = None,
    profile: str = FULL_RELEASE_PROFILE,
) -> tuple[Step, ...]:
    """Build the fixed release suite as argv lists, never shell strings."""

    if profile not in RELEASE_PROFILE_CHOICES:
        raise ValueError(f"unsupported release profile: {profile}")
    python = python_executable or sys.executable
    archive_path = RELEASE_ARCHIVE_PATHS[profile]
    package_argv = [python, "scripts/package_release.py"]
    if required_tag is not None:
        package_argv.extend(("--output", archive_path, "--require-tag", required_tag))
    else:
        package_argv.append("--check-only")
    archive_steps: tuple[Step, ...] = ()
    if required_tag is not None:
        archive_steps = (
            Step(
                "release_archive_runtime",
                (
                    python,
                    "scripts/verify_release_archive.py",
                    "--archive",
                    archive_path,
                    "--profile",
                    profile,
                    "--expected-tag",
                    required_tag,
                ),
                slow=True,
            ),
        )

    if profile == PUBLIC_TEXT_RELEASE_PROFILE:
        package_argv.extend(
            (
                "--profile",
                PUBLIC_TEXT_RELEASE_PROFILE,
                "--public-evidence-dir",
                PUBLIC_TEXT_STAGE,
            )
        )
        index_path = f"{PUBLIC_TEXT_STAGE}/dog_faiss.index"
        metas_path = f"{PUBLIC_TEXT_STAGE}/dog_metas.json"
        retrieval_json = f"{PUBLIC_TEXT_STAGE}/retrieval.json"
        retrieval_markdown = f"{PUBLIC_TEXT_STAGE}/retrieval.md"
        return (
            Step(
                "public_text_artifact_contract",
                (
                    python,
                    "scripts/build_public_text_release.py",
                    "--output-dir",
                    PUBLIC_TEXT_STAGE,
                ),
            ),
            Step(
                "contest_readiness",
                (
                    python,
                    "scripts/check_contest_readiness.py",
                    "--metas",
                    metas_path,
                    "--index",
                    index_path,
                    "--reference-date",
                    reference_date.isoformat(),
                    "--max-age-days",
                    "7",
                    "--strict",
                ),
            ),
            Step(
                "evaluation_retrieval",
                (
                    python,
                    "scripts/evaluate_retrieval.py",
                    "--index",
                    index_path,
                    "--metas",
                    metas_path,
                    "--json-out",
                    retrieval_json,
                    "--markdown-out",
                    retrieval_markdown,
                    "--reference-date",
                    reference_date.isoformat(),
                    "--device",
                    "cpu",
                    "--warmup",
                    "0",
                ),
                slow=True,
            ),
            Step(
                "evaluation_query_robustness",
                (
                    python,
                    "scripts/evaluate_query_robustness.py",
                    "--index",
                    index_path,
                    "--metas",
                    metas_path,
                    "--json-out",
                    f"{PUBLIC_TEXT_STAGE}/query-robustness.json",
                    "--markdown-out",
                    f"{PUBLIC_TEXT_STAGE}/query-robustness.md",
                    "--reference-date",
                    reference_date.isoformat(),
                    "--device",
                    "cpu",
                    "--no-warmup",
                ),
                slow=True,
            ),
            Step(
                "evaluation_query_holdout_v2_historical_integrity",
                (),
                not_applicable_reason=(
                    "not applicable: historical v2 is bound to the full multimodal "
                    "artifact and is not a public-text result"
                ),
            ),
            Step(
                "evaluation_query_holdout_v3_independent",
                (),
                not_applicable_reason=(
                    "not applicable: independent v3 is bound to the full multimodal "
                    "artifact and is not a public-text result"
                ),
            ),
            Step(
                "evaluation_exposure_representation",
                (
                    python,
                    "scripts/evaluate_exposure_representation.py",
                    "--metas",
                    metas_path,
                    "--retrieval-report",
                    retrieval_json,
                    "--json-out",
                    f"{PUBLIC_TEXT_STAGE}/exposure.json",
                    "--markdown-out",
                    f"{PUBLIC_TEXT_STAGE}/exposure.md",
                ),
                slow=True,
            ),
            Step(
                "evaluation_profile_reranking",
                (
                    python,
                    "scripts/evaluate_profile_reranking.py",
                    "--metas",
                    metas_path,
                    "--json-out",
                    f"{PUBLIC_TEXT_STAGE}/profile.json",
                    "--markdown-out",
                    f"{PUBLIC_TEXT_STAGE}/profile.md",
                ),
            ),
            Step(
                "evaluation_safety_contract",
                (
                    python,
                    "scripts/evaluate_safety_contract.py",
                    "--metas",
                    metas_path,
                    "--json-out",
                    f"{PUBLIC_TEXT_STAGE}/safety.json",
                    "--markdown-out",
                    f"{PUBLIC_TEXT_STAGE}/safety.md",
                ),
            ),
            Step(
                "public_text_evaluation_summary",
                (
                    python,
                    "scripts/build_public_text_evaluation_summary.py",
                    "--reports-dir",
                    PUBLIC_TEXT_STAGE,
                    "--index",
                    index_path,
                    "--metas",
                    metas_path,
                    "--profile-marker",
                    f"{PUBLIC_TEXT_STAGE}/release_profile.json",
                    "--output-dir",
                    PUBLIC_TEXT_STAGE,
                ),
            ),
            Step(
                "evaluation_heldout_image_retrieval",
                (),
                not_applicable_reason=(
                    "not applicable: public-text-only has no image/crop vectors; "
                    "this result is not counted as PASS"
                ),
            ),
            Step(
                "portal_protocol_template",
                (
                    python,
                    "scripts/portal_candidate_selection_study.py",
                    "validate-template",
                ),
            ),
            Step(
                "git_history_secret_scan",
                (
                    python,
                    "scripts/scan_git_history_secrets.py",
                    "--strict-user-refs",
                ),
            ),
            Step("release_package_check", tuple(package_argv)),
            *archive_steps,
            Step("dependency_check", (python, "-m", "pip", "check")),
            Step(
                "dependency_snapshot_match",
                (python, "scripts/check_dependency_snapshot.py"),
            ),
            Step(
                "ruff",
                (python, "-m", "ruff", "check", "app", "scripts", "tests"),
            ),
            Step(
                "pytest",
                (python, "-m", "pytest", "-q", *PUBLIC_TEXT_TESTS),
                slow=True,
            ),
            Step(
                "full_runtime_smoke",
                (
                    python,
                    "scripts/smoke_full_runtime.py",
                    "--profile",
                    PUBLIC_TEXT_RELEASE_PROFILE,
                    "--index",
                    index_path,
                    "--metas",
                    metas_path,
                    "--release-profile-marker",
                    f"{PUBLIC_TEXT_STAGE}/release_profile.json",
                ),
                slow=True,
            ),
            Step(
                "git_clean",
                ("git", "status", "--porcelain=v1", "--untracked-files=all"),
                inspect_stdout_for_cleanliness=True,
            ),
        )
    return (
        Step(
            "contest_readiness",
            (
                python,
                "scripts/check_contest_readiness.py",
                "--metas",
                "data/dog_metas.json",
                "--index",
                "data/dog_faiss.index",
                "--reference-date",
                reference_date.isoformat(),
                "--max-age-days",
                "7",
                "--strict",
            ),
        ),
        Step(
            "evaluation_retrieval",
            (python, "scripts/evaluate_retrieval.py", "--check"),
        ),
        Step(
            "evaluation_query_robustness",
            (python, "scripts/evaluate_query_robustness.py", "--check"),
        ),
        Step(
            "evaluation_query_holdout_v2_historical_integrity",
            (python, "scripts/evaluate_query_holdout.py", "--check"),
        ),
        Step(
            "evaluation_query_holdout_v3_independent",
            (python, "scripts/evaluate_query_holdout_v3.py", "--check"),
        ),
        Step(
            "evaluation_exposure_representation",
            (
                python,
                "scripts/evaluate_exposure_representation.py",
                "--check",
            ),
        ),
        Step(
            "evaluation_profile_reranking",
            (python, "scripts/evaluate_profile_reranking.py", "--check"),
        ),
        Step(
            "evaluation_safety_contract",
            (python, "scripts/evaluate_safety_contract.py", "--check"),
        ),
        Step(
            "evaluation_heldout_image_retrieval",
            (
                python,
                "scripts/evaluate_heldout_image_retrieval.py",
                "--check",
            ),
        ),
        Step(
            "portal_protocol_template",
            (
                python,
                "scripts/portal_candidate_selection_study.py",
                "validate-template",
            ),
        ),
        Step(
            "git_history_secret_scan",
            (
                python,
                "scripts/scan_git_history_secrets.py",
                "--strict-user-refs",
            ),
        ),
        Step("release_package_check", tuple(package_argv)),
        *archive_steps,
        Step("dependency_check", (python, "-m", "pip", "check")),
        Step(
            "dependency_snapshot_match",
            (python, "scripts/check_dependency_snapshot.py"),
        ),
        Step(
            "ruff",
            (python, "-m", "ruff", "check", "app", "scripts", "tests"),
        ),
        Step("pytest", (python, "-m", "pytest", "-q"), slow=True),
        Step(
            "full_runtime_smoke",
            (python, "scripts/smoke_full_runtime.py"),
            slow=True,
        ),
        Step(
            "git_clean",
            (
                "git",
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ),
            inspect_stdout_for_cleanliness=True,
        ),
    )


def _result(
    *,
    name: str,
    status: str,
    exit_code: int | None,
    duration_seconds: float,
    message: str,
) -> dict[str, object]:
    return {
        "name": name,
        "status": status,
        "exit_code": exit_code,
        "duration_seconds": round(max(0.0, duration_seconds), 3),
        "message": message,
    }


def run_step(
    step: Step,
    *,
    repo: Path,
    timeout_seconds: int,
    runner: Runner | None = None,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, object]:
    """Execute one step while discarding all diagnostic child output."""

    started = monotonic()
    command_runner = runner or subprocess.run
    stdout_target = (
        subprocess.PIPE if step.inspect_stdout_for_cleanliness else subprocess.DEVNULL
    )
    try:
        completed = command_runner(
            list(step.argv),
            cwd=str(repo),
            check=False,
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=stdout_target,
            stderr=subprocess.DEVNULL,
            timeout=timeout_seconds,
        )
        exit_code = int(completed.returncode)
        dirty = False
        if step.inspect_stdout_for_cleanliness and exit_code == 0:
            output = completed.stdout
            if isinstance(output, str):
                dirty = bool(output.strip())
            elif isinstance(output, bytes):
                dirty = bool(output.strip())
            elif output is not None:
                dirty = True
            if dirty:
                exit_code = 1

        duration = monotonic() - started
        if exit_code == 0:
            message = (
                "working tree and index are clean"
                if step.inspect_stdout_for_cleanliness
                else "completed successfully"
            )
            return _result(
                name=step.name,
                status="passed",
                exit_code=0,
                duration_seconds=duration,
                message=message,
            )
        message = (
            "working tree or index is not clean"
            if dirty
            else (
                f"command failed with exit code {exit_code}; "
                "rerun this step directly for details"
            )
        )
        return _result(
            name=step.name,
            status="failed",
            exit_code=exit_code,
            duration_seconds=duration,
            message=message,
        )
    except subprocess.TimeoutExpired:
        return _result(
            name=step.name,
            status="failed",
            exit_code=124,
            duration_seconds=monotonic() - started,
            message="timed out after the configured per-step limit",
        )
    except FileNotFoundError:
        return _result(
            name=step.name,
            status="failed",
            exit_code=127,
            duration_seconds=monotonic() - started,
            message="required executable was not found",
        )
    except OSError:
        return _result(
            name=step.name,
            status="failed",
            exit_code=126,
            duration_seconds=monotonic() - started,
            message="operating system could not start the command",
        )


def run_release_gate(
    *,
    repo: Path = BASE_DIR,
    reference_date: date | None = None,
    required_tag: str | None = None,
    skip_slow: bool = False,
    timeout_seconds: int = DEFAULT_STEP_TIMEOUT_SECONDS,
    runner: Runner | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    profile: str = FULL_RELEASE_PROFILE,
) -> dict[str, object]:
    """Run every independent check and return a secret-free summary."""

    root = repo.resolve()
    today = reference_date or current_local_date()
    results: list[dict[str, object]] = []
    for step in build_steps(
        reference_date=today,
        required_tag=required_tag,
        profile=profile,
    ):
        if step.not_applicable_reason is not None:
            results.append(
                _result(
                    name=step.name,
                    status="not_applicable",
                    exit_code=None,
                    duration_seconds=0.0,
                    message=step.not_applicable_reason,
                )
            )
            continue
        if skip_slow and step.slow:
            results.append(
                _result(
                    name=step.name,
                    status="skipped",
                    exit_code=None,
                    duration_seconds=0.0,
                    message=(
                        "omitted only for development preflight; "
                        "required for a release decision"
                    ),
                )
            )
            continue
        results.append(
            run_step(
                step,
                repo=root,
                timeout_seconds=timeout_seconds,
                runner=runner,
                monotonic=monotonic,
            )
        )

    release_identity: dict[str, str] | None = None
    if required_tag is not None:
        try:
            release_identity = _read_release_identity(
                root,
                profile=profile,
                expected_tag=required_tag,
            )
        except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile):
            results.append(
                _result(
                    name="release_archive_identity",
                    status="failed",
                    exit_code=1,
                    duration_seconds=0.0,
                    message="release archive identity could not be verified",
                )
            )
        else:
            results.append(
                _result(
                    name="release_archive_identity",
                    status="passed",
                    exit_code=0,
                    duration_seconds=0.0,
                    message="release archive tag, commit, profile, and SHA-256 recorded",
                )
            )

    passed = sum(item["status"] == "passed" for item in results)
    failed = sum(item["status"] == "failed" for item in results)
    skipped = sum(item["status"] == "skipped" for item in results)
    not_applicable = sum(item["status"] == "not_applicable" for item in results)
    preflight_passed = failed == 0
    ready = (
        preflight_passed
        and skipped == 0
        and not skip_slow
        and required_tag is not None
        and release_identity is not None
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": (
            "release" if required_tag is not None and not skip_slow else "preflight"
        ),
        "profile": profile,
        "reference_date": today.isoformat(),
        "required_tag_enforced": required_tag is not None,
        "release_identity": release_identity,
        "ready": ready,
        "preflight_passed": preflight_passed,
        "summary": {
            "total": len(results),
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "not_applicable": not_applicable,
        },
        "steps": results,
    }


def _read_release_identity(
    repo: Path,
    *,
    profile: str,
    expected_tag: str,
) -> dict[str, str]:
    """Read only non-secret identity fields from the verified release ZIP."""

    relative_archive = RELEASE_ARCHIVE_PATHS[profile]
    archive_path = (repo / relative_archive).resolve(strict=True)
    archive_path.relative_to(repo.resolve())
    archive_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read(RELEASE_MANIFEST_PATH))
    if not isinstance(manifest, dict):
        raise ValueError("release manifest must be an object")
    tag = manifest.get("release_tag")
    commit = manifest.get("commit_sha")
    tag_object_id = manifest.get("tag_object_id")
    manifest_profile = manifest.get("profile")
    if tag != expected_tag or manifest_profile != profile:
        raise ValueError("release identity does not match requested tag/profile")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("release manifest commit is invalid")
    if not isinstance(tag_object_id, str) or not re.fullmatch(
        r"[0-9a-f]{40}", tag_object_id
    ):
        raise ValueError("release manifest tag object is invalid")
    return {
        "archive_path": relative_archive,
        "archive_sha256": archive_sha256,
        "commit_sha": commit,
        "profile": profile,
        "tag": expected_tag,
        "tag_object_id": tag_object_id,
    }


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run every contest release gate and emit only a compact JSON summary."
        )
    )
    parser.add_argument(
        "--profile",
        choices=RELEASE_PROFILE_CHOICES,
        default=FULL_RELEASE_PROFILE,
        help="artifact profile to verify (default: full)",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=BASE_DIR,
        help="repository root (default: this project)",
    )
    parser.add_argument(
        "--required-tag",
        help="require this exact Git tag to resolve to clean HEAD",
    )
    parser.add_argument(
        "--skip-slow",
        action="store_true",
        help=(
            "development preflight only: omit pytest and full runtime smoke; "
            "the result can never be release-ready"
        ),
    )
    parser.add_argument(
        "--step-timeout-seconds",
        type=_positive_int,
        default=DEFAULT_STEP_TIMEOUT_SECONDS,
        help=(
            "maximum runtime for each child command "
            f"(default: {DEFAULT_STEP_TIMEOUT_SECONDS})"
        ),
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help=(
            "atomically write the same JSON summary; paths inside the repository "
            "must be under ignored dist/"
        ),
    )
    return parser


def _safe_json_target(repo: Path, target: Path) -> Path:
    resolved_repo = repo.resolve()
    resolved_target = target.resolve()
    try:
        resolved_target.relative_to(resolved_repo)
    except ValueError:
        return resolved_target
    try:
        resolved_target.relative_to(resolved_repo / "dist")
    except ValueError as exc:
        raise ValueError(
            "JSON output inside the repository is allowed only under dist/"
        ) from exc
    return resolved_target


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _with_output_result(
    report: dict[str, object],
    *,
    status: str,
) -> dict[str, object]:
    updated = dict(report)
    steps = [dict(item) for item in report["steps"]]  # type: ignore[index]
    succeeded = status == "passed"
    steps.append(
        _result(
            name="json_output",
            status=status,
            exit_code=0 if succeeded else 1,
            duration_seconds=0.0,
            message=(
                "summary written atomically"
                if succeeded
                else "requested JSON summary could not be written safely"
            ),
        )
    )
    summary = dict(report["summary"])  # type: ignore[arg-type]
    summary["total"] = int(summary["total"]) + 1
    summary[status] = int(summary[status]) + 1
    updated["steps"] = steps
    updated["summary"] = summary
    if not succeeded:
        updated["ready"] = False
        updated["preflight_passed"] = False
    return updated


def report_text(report: dict[str, object]) -> str:
    return json.dumps(report, ensure_ascii=False, sort_keys=True) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_release_gate(
        repo=args.repo,
        required_tag=args.required_tag,
        skip_slow=bool(args.skip_slow),
        timeout_seconds=args.step_timeout_seconds,
        profile=args.profile,
    )
    if args.json_out is not None:
        try:
            target = _safe_json_target(args.repo, args.json_out)
            completed_report = _with_output_result(report, status="passed")
            _write_atomic(target, report_text(completed_report))
            report = completed_report
        except (OSError, ValueError):
            report = _with_output_result(report, status="failed")

    print(report_text(report), end="")
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
