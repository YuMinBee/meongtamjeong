"""Finish all data preparation in the background; never load or execute a model."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
import traceback

from experiments.composed_retrieval.download import (
    DATA, S3, annotations, cirr_annotations, ranged_download,
)
from experiments.composed_retrieval.prepare import (
    ARTIFACTS, prepare_training, prepare_vg, write_json,
)
from experiments.composed_retrieval.validate import run


def state(status: str, **extra) -> None:
    write_json(ARTIFACTS / "data_job.json", {
        "status": status, "pid": os.getpid(), "gpu_used": False,
        "updated_at": datetime.now(timezone.utc).isoformat(), **extra,
    })


def wait_for_downloaders(pids: list[int]) -> None:
    """Adopt already running downloads without interrupting in-flight byte ranges."""
    if not pids:
        return
    if os.name != "nt":
        raise ValueError("adopting process handles is supported on Windows only")
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handles = [(pid, kernel.OpenProcess(0x00100000, False, pid)) for pid in pids]
    try:
        for pid, handle in handles:
            if not handle:
                continue  # Process already exited; downloader's resume checks still run.
            while True:
                state("waiting_for_existing_downloads", waiting_for_pid=pid, adopted_pids=pids)
                result = kernel.WaitForSingleObject(handle, 30000)
                if result == 0:
                    break
                if result != 258:
                    raise OSError("failed waiting for downloader process")
    finally:
        for _, handle in handles:
            if handle:
                kernel.CloseHandle(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-for-pids", type=int, nargs="*", default=[])
    args = parser.parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    lock = (ARTIFACTS / "data_job.lock").open("a+b")
    if os.name == "nt":
        import msvcrt
        lock.seek(0)
        if not lock.read(1):
            lock.write(b"1")
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    try:
        wait_for_downloaders(args.wait_for_pids)
        state("preparing_annotations")
        annotations()
        cirr_annotations()
        ranged_download(f"{S3}/annotations/annotations_trainval2017.zip",
                        DATA / "annotations_trainval2017.zip", workers=12)
        ranged_download(f"{S3}/zips/val2017.zip", DATA / "val2017.zip", workers=12)
        jobs = {
            "circo_images": lambda: ranged_download(f"{S3}/zips/unlabeled2017.zip",
                                                     DATA / "unlabeled2017.zip", workers=48),
            "genecis_images": lambda: prepare_vg(workers=48),
            "coco_training_images": lambda: prepare_training(workers=24),
        }
        completed, errors = [], []
        state("downloading", completed=completed)
        with ThreadPoolExecutor(3) as pool:
            futures = {pool.submit(job): name for name, job in jobs.items()}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    future.result()
                    completed.append(name)
                except Exception as error:
                    errors.append({"job": name, "error": str(error)})
                    traceback.print_exc()
                state("downloading", completed=completed, errors=errors)
        if errors:
            state("download_failed", completed=completed, errors=errors)
            raise RuntimeError("data downloads failed; see data_job.json and log")
        state("validating_cpu_only", completed=completed)
        report = run(deep=True)
        if report["errors"]:
            state("validation_failed", errors=report["errors"])
            raise RuntimeError("data integrity checks failed")
        state("complete", readiness_report="artifacts/data_readiness.json",
              cirr="annotations_only_raw_images_pending_official_access")
        print("DATA PREPARATION COMPLETE. No models, GPU encoding or training were run.", flush=True)
    except Exception as error:
        report_path = ARTIFACTS / "data_job.json"
        previous = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
        if previous.get("status") not in {"download_failed", "validation_failed"}:
            state("failed", error=str(error))
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()
