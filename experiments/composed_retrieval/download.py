"""Resumable, length-verified downloads from the benchmark authors' sources."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import threading

import requests

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tmp" / "composed_retrieval"
S3 = "https://s3.amazonaws.com/images.cocodataset.org"
_LOCAL = threading.local()


def session() -> requests.Session:
    if not hasattr(_LOCAL, "session"):
        _LOCAL.session = requests.Session()
    return _LOCAL.session


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(value), encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url: str, path: Path, *, timeout: int = 90) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size:
        return path
    for attempt in range(4):
        try:
            with session().get(url, stream=True, timeout=(20, timeout)) as response:
                response.raise_for_status()
                part = path.with_suffix(path.suffix + ".partial")
                count = 0
                with part.open("wb") as handle:
                    for block in response.iter_content(1024 * 1024):
                        handle.write(block)
                        count += len(block)
                expected = response.headers.get("Content-Length")
                if expected and not response.headers.get("Content-Encoding"):
                    if count != int(expected):
                        raise ValueError(f"short download: {path.name}: {count}/{expected}")
                part.replace(path)
                return path
        except (requests.RequestException, ValueError):
            if attempt == 3:
                raise
            time.sleep(1 + attempt)
    raise AssertionError("unreachable")


def ranged_download(url: str, path: Path, *, workers: int = 12) -> Path:
    """Download disjoint byte ranges; completed blocks survive an interrupted run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()
        size = int(response.headers["Content-Length"])
        etag = response.headers.get("ETag", "")
    info = path.with_suffix(path.suffix + ".source.json")
    signature = {"url": url, "size": size, "etag": etag}
    if info.exists() and json.loads(info.read_text()) != signature:
        raise ValueError(f"upstream archive changed: {path}")
    info.write_text(json.dumps(signature), encoding="utf-8")
    if path.exists() and path.stat().st_size == size:
        return path
    part = path.with_suffix(path.suffix + ".partial")
    state = path.with_suffix(path.suffix + ".blocks.json")
    completed = set(json.loads(state.read_text()) if state.exists() and part.exists() else [])
    if not part.exists():
        with part.open("wb") as handle:
            handle.truncate(size)
    block_size = 32 * 1024 * 1024
    blocks = (size + block_size - 1) // block_size
    started = time.monotonic()

    def one(index: int) -> int:
        left, right = index * block_size, min(size, (index + 1) * block_size) - 1
        for attempt in range(4):
            try:
                with session().get(url, headers={"Range": f"bytes={left}-{right}",
                                               "If-Match": etag},
                                  timeout=(20, 120), stream=True) as response:
                    response.raise_for_status()
                    if response.status_code != 206:
                        raise ValueError("server did not honor byte range")
                    if response.headers.get("Content-Range") != f"bytes {left}-{right}/{size}":
                        raise ValueError("incorrect range response")
                    offset = left
                    with part.open("r+b") as handle:
                        handle.seek(left)
                        for block in response.iter_content(1024 * 1024):
                            if offset + len(block) > right + 1:
                                raise ValueError("range overflow")
                            handle.write(block)
                            offset += len(block)
                    if offset != right + 1:
                        raise ValueError("truncated byte range")
                return index
            except (requests.RequestException, ValueError):
                if attempt == 3:
                    raise
                time.sleep(1 + attempt)
        raise AssertionError("unreachable")

    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(one, index) for index in range(blocks) if index not in completed]
        for future in as_completed(futures):
            completed.add(future.result())
            atomic_json(state, sorted(completed))
            if len(completed) % 16 == 0 or len(completed) == blocks:
                print(f"{path.name}: {len(completed)}/{blocks} blocks, "
                      f"{time.monotonic() - started:.1f}s", flush=True)
    part.replace(path)
    return path


def annotations() -> dict:
    existing = DATA / "sources.json"
    if existing.exists():
        manifest = json.loads(existing.read_text())
        for entry in manifest["files"]:
            path = fetch(entry["url"], ROOT / entry["path"])
            if sha256(path) != entry["sha256"]:
                raise ValueError(f"cached annotation hash changed: {path}")
        return manifest
    sources = {}
    for repo in ("facebookresearch/genecis", "miccunifi/CIRCO"):
        response = requests.get(f"https://api.github.com/repos/{repo}/commits/main", timeout=30)
        response.raise_for_status()
        sources[repo] = response.json()["sha"]
    jobs = []
    for task in ("focus_attribute", "change_attribute", "focus_object", "change_object"):
        repo = "facebookresearch/genecis"
        jobs.append((f"https://raw.githubusercontent.com/{repo}/{sources[repo]}/genecis/{task}.json",
                     DATA / "annotations" / "genecis" / f"{task}.json"))
    for split in ("val", "test"):
        repo = "miccunifi/CIRCO"
        jobs.append((f"https://raw.githubusercontent.com/{repo}/{sources[repo]}/annotations/{split}.json",
                     DATA / "annotations" / "circo" / f"{split}.json"))
    for url, path in jobs:
        fetch(url, path)
        print(path.name, "records", len(json.loads(path.read_text())), flush=True)
    manifest = {"revisions": sources, "files": [
        {"url": url, "path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
        for url, path in jobs]}
    (DATA / "sources.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def cirr_annotations() -> None:
    """Prepare public annotations only; raw NLVR2 access remains a separate step."""
    existing = DATA / "cirr_sources.json"
    if existing.exists():
        for entry in json.loads(existing.read_text())["files"]:
            path = fetch(entry["url"], ROOT / entry["path"])
            if sha256(path) != entry["sha256"]:
                raise ValueError("CIRR annotation hash mismatch")
        return
    repo = "Cuberick-Orion/CIRR"
    response = requests.get(f"https://api.github.com/repos/{repo}/commits/cirr_dataset", timeout=30)
    response.raise_for_status()
    revision = response.json()["sha"]
    entries = []
    for split in ("train", "val", "test1"):
        for relative in (f"captions/cap.rc2.{split}.json", f"image_splits/split.rc2.{split}.json"):
            url = f"https://raw.githubusercontent.com/{repo}/{revision}/{relative}"
            path = fetch(url, DATA / "annotations" / "cirr" / relative)
            count = len(json.loads(path.read_text()))
            entries.append({"url": url, "path": str(path.relative_to(ROOT)),
                            "sha256": sha256(path), "records": count})
            print(relative, count, flush=True)
    existing.write_text(json.dumps({"revision": revision, "files": entries,
                                   "raw_images": "pending_official_NLVR2_access"}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", choices=("annotations", "cirr-annotations", "circo", "coco-val", "coco-annotations"))
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if args.target == "annotations":
        annotations()
    elif args.target == "cirr-annotations":
        cirr_annotations()
    else:
        filename = {"circo": "zips/unlabeled2017.zip", "coco-val": "zips/val2017.zip",
                    "coco-annotations": "annotations/annotations_trainval2017.zip"}[args.target]
        path = ranged_download(f"{S3}/{filename}", DATA / Path(filename).name, workers=args.workers)
        digest = sha256(path)
        path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n")
        print(path.name, path.stat().st_size, "sha256", digest, flush=True)


if __name__ == "__main__":
    main()
