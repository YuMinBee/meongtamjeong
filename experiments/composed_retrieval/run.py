"""GPU experiment runner, explicitly separate from the data-only runner."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

from experiments.composed_retrieval.download import sha256
from experiments.composed_retrieval.prepare import ARTIFACTS, load_json, write_json


def run():
    import msvcrt

    ARTIFACTS.mkdir(exist_ok=True)
    lock = (ARTIFACTS / "gpu_job.lock").open("a+b")
    lock.seek(0)
    if not lock.read(1):
        lock.write(b"0")
        lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    status_path = ARTIFACTS / "gpu_job.json"
    started = time.monotonic()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    environment = {"python": sys.version, "executable": sys.executable,
                   "platform": platform.platform(), "pid": os.getpid(),
                   "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                                   "--format=csv"], text=True).strip()}
    code = list(Path(__file__).parent.glob("*.py"))
    code.extend([Path(__file__).parents[1] / "dino_fusion" / name for name in ("core.py", "alignment.py")])
    environment["code_sha256"] = {str(path): sha256(path) for path in code}

    def status(stage, **extra):
        write_json(status_path, {"status": stage, "updated_at": datetime.now(timezone.utc).isoformat(),
                                 "elapsed_seconds": time.monotonic() - started,
                                 "environment": environment, **extra})
        print(f"GPU job: {stage}", flush=True)

    try:
        if not load_json(ARTIFACTS / "data_readiness.json")["ready_for_gpu_experiments"]:
            raise RuntimeError("Data not ready")
        for module in ("features", "train", "evaluate", "report"):
            status(module)
            subprocess.run([sys.executable, "-u", "-m", f"experiments.composed_retrieval.{module}"], check=True)
        status("complete", report=str(Path(__file__).with_name("RESULTS.md")))
    except BaseException:
        status("failed", traceback=traceback.format_exc())
        raise
    finally:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        lock.close()


if __name__ == "__main__":
    run()
