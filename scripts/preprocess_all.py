"""
Run the preprocessing pipeline from raw dataset files to training caches.

Steps:
  1. preprocess_audio.py   -> cache/_audio/
  2. preprocess_labels.py  -> cache/_labels/, cache/stage1/, stage detail caches
  3. build_stage234_cache.py placeholder downstream caches
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("preprocess_all")

STEPS = {
    "audio": ("scripts/preprocess_audio.py", "Phase 1: audio features -> cache/_audio/"),
    "labels": ("scripts/preprocess_labels.py", "Phase 2a: labels + stage caches"),
    "cache234": ("scripts/build_stage234_cache.py", "Phase 2b: placeholder downstream caches"),
}


def run_step(script: str, description: str, args: argparse.Namespace) -> bool:
    logger.info("=" * 60)
    logger.info(description)
    logger.info("=" * 60)
    t0 = time.time()

    cmd = [sys.executable, str(PROJECT_ROOT / script)]

    if "build_stage234" not in script:
        if args.num_workers:
            cmd.extend(["--num-workers", str(args.num_workers)])
        if args.data_root:
            cmd.extend(["--data-root", args.data_root])
        if args.limit:
            cmd.extend(["--limit", str(args.limit)])
        if args.force:
            cmd.append("--force")

    if "audio" in script:
        if args.subdiv:
            cmd.extend(["--subdiv", str(args.subdiv)])
        cmd.extend(["--cache-root", str(Path(args.cache_root) / "_audio")])
        cmd.extend(["--encodec-layers", str(args.encodec_layers)])

    if "labels" in script:
        if args.subdiv:
            cmd.extend(["--subdiv", str(args.subdiv)])
        if args.maxsubdiv:
            cmd.extend(["--maxsubdiv", str(args.maxsubdiv)])
        if args.max_tokens:
            cmd.extend(["--max-tokens", str(args.max_tokens)])
        cmd.extend(["--cache-root", str(args.cache_root)])

    if "build_stage234" in script:
        cmd.extend(["--step", "build-caches", "--placeholder", "--cache-root", str(args.cache_root)])
        cmd.extend(["--config", "configs/rotating_4090.yaml"])
        if args.num_workers:
            cmd.extend(["--num-workers", str(args.num_workers)])
        if args.limit:
            cmd.extend(["--limit", str(args.limit)])

    logger.info("Command: %s", " ".join(cmd))
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        logger.info("[%s] %s", Path(script).stem, line.rstrip())
    returncode = proc.wait()

    elapsed = time.time() - t0
    if returncode == 0:
        logger.info("Done (%.0fs)", elapsed)
        return True

    logger.error("Failed (exit=%s, %.0fs)", returncode, elapsed)
    return False


def main() -> None:
    p = argparse.ArgumentParser(description="Run preprocessing pipeline")
    p.add_argument("--steps", default="all", help="Steps to run: audio, labels, cache234, all")
    p.add_argument("--data-root", default="datasets")
    p.add_argument("--cache-root", default="/data/maiG_v2/cache")
    p.add_argument("--limit", type=int, default=None, help="Limit song count for testing")
    p.add_argument("--num-workers", type=int, default=1, help="Parallel worker count")
    p.add_argument("--subdiv", type=int, default=64)
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--maxsubdiv", type=int, default=64, help="Normalized chart subdivision")
    p.add_argument("--force", action="store_true", help="Rebuild existing cache files")
    p.add_argument("--encodec-layers", type=int, default=1, help="EnCodec layer count (1/2)")
    args = p.parse_args()

    steps_to_run = ["audio", "labels", "cache234"] if args.steps == "all" else [s.strip() for s in args.steps.split(",")]
    unknown = set(steps_to_run) - set(STEPS)
    if unknown:
        logger.error("Unknown steps: %s. Available: %s", unknown, list(STEPS))
        sys.exit(1)

    t_total = time.time()
    failed: list[str] = []
    for step_name in steps_to_run:
        script, desc = STEPS[step_name]
        if not run_step(script, desc, args):
            failed.append(step_name)
            logger.error("Step '%s' failed; stopping pipeline.", step_name)
            break

    total_elapsed = time.time() - t_total
    if failed:
        logger.error("Preprocess failed. Failed steps: %s (%.0fs)", failed, total_elapsed)
        logger.info("Hints:")
        logger.info("  1. Make sure dependencies are installed: pip install librosa soundfile torch")
        logger.info("  2. Re-run without --force to keep existing successful cache files")
        logger.info("  3. Use --num-workers 1 for the most stable logs")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("All preprocessing steps completed. (%.0fs)", total_elapsed)
    logger.info("=" * 60)
    logger.info("Next:")
    logger.info("  python train.py --config configs/rotating_4090.yaml")
    logger.info("  After Stage 1 training, rebuild downstream caches with real hidden states:")
    logger.info("  python scripts/build_stage234_cache.py --step all --checkpoint /data/maiG_v2/runs/rotating_4090/best.pt")


if __name__ == "__main__":
    main()
