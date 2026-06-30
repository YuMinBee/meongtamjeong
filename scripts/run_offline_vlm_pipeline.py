import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DEFAULT_INPUT = DATA_DIR / "local_dog_cache.json"
DEFAULT_OUTPUT = DATA_DIR / "local_dog_cache_enriched.json"


def display_arg(part: str) -> str:
    text = str(part).encode("unicode_escape").decode("ascii")
    return f'"{text}"' if " " in text else text


def run_step(command: List[str], dry_run: bool) -> None:
    printable = " ".join(display_arg(part) for part in command)
    print(f"[RUN] {printable}")
    if dry_run:
        return
    subprocess.run(command, cwd=BASE_DIR, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the offline VLM attribute extraction and FAISS rebuild flow."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Raw dog cache JSON.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Enriched cache JSON.")
    parser.add_argument("--task", choices=("attributes", "both"), default="attributes")
    parser.add_argument("--limit", type=int, default=None, help="Maximum records to enrich in this run.")
    parser.add_argument("--target", type=int, default=10000, help="Maximum records to index after enrichment.")
    parser.add_argument("--model-path", default="", help="Local VLM snapshot path.")
    parser.add_argument("--model-class", choices=("gemma3", "auto"), default="gemma3")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing VLM attributes.")
    parser.add_argument("--text-only", action="store_true", help="Build only text embeddings.")
    parser.add_argument("--skip-enrich", action="store_true", help="Skip VLM enrichment.")
    parser.add_argument("--skip-index", action="store_true", help="Skip FAISS rebuild.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    enrich_script = BASE_DIR / "scripts" / "enrich_live_descriptions.py"
    build_script = BASE_DIR / "scripts" / "build_embeddings.py"

    if not args.skip_enrich:
        enrich_cmd = [
            sys.executable,
            str(enrich_script),
            "--input",
            str(args.input),
            "--output",
            str(args.output),
            "--task",
            args.task,
            "--model-class",
            args.model_class,
        ]
        if args.model_path:
            enrich_cmd.extend(["--model-path", args.model_path])
        if args.limit is not None:
            enrich_cmd.extend(["--limit", str(args.limit)])
        if args.overwrite:
            enrich_cmd.append("--overwrite")
        run_step(enrich_cmd, args.dry_run)

    if not args.skip_index:
        build_cmd = [
            sys.executable,
            str(build_script),
            "--input",
            str(args.output),
            "--target",
            str(args.target),
        ]
        if args.text_only:
            build_cmd.append("--text-only")
        run_step(build_cmd, args.dry_run)

    print("[DONE] offline VLM attribute pipeline complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())