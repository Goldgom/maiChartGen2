"""One-shot training pipeline with terminal logging."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
CONFIG = "configs/rotating_4090.yaml"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_all")


def setup_file_logging(log_file: str | None) -> None:
    if not log_file:
        return
    path = Path(log_file)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(handler)
    logger.info("Logging to %s", path)


def load_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
        return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.warning("Could not load config %s: %s", path, exc)
        return {}


def stage_checkpoint(checkpoint_dir: str, stage: str, kind: str = "best") -> Path:
    stage_dir_path = Path(checkpoint_dir) / stage / f"{kind}.pt"
    if stage_dir_path.exists():
        return stage_dir_path
    legacy_stage_path = Path(checkpoint_dir) / f"{stage}_{kind}.pt"
    if legacy_stage_path.exists():
        return legacy_stage_path
    return stage_dir_path


def resume_args(checkpoint_dir: str, stage: str) -> list[str]:
    ckpt = stage_checkpoint(checkpoint_dir, stage, "best")
    return ["--resume", str(ckpt)] if ckpt.exists() else []


def run(cmd: list[str], desc: str) -> bool:
    logger.info("=" * 60)
    logger.info(desc)
    logger.info("Command: %s", " ".join(cmd))
    logger.info("=" * 60)
    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        logger.info("[%s] %s", Path(cmd[0]).name, line.rstrip())
    returncode = proc.wait()
    elapsed = time.time() - t0
    if returncode == 0:
        logger.info("Done (%.0fs)", elapsed)
        return True
    logger.error("Failed (exit=%s, %.0fs)", returncode, elapsed)
    return False


def main() -> None:
    p = argparse.ArgumentParser(description="One-shot full training pipeline")
    p.add_argument("--config", default=CONFIG)
    p.add_argument("--cache-root", default="/data/maiG_v2/cache")
    p.add_argument("--checkpoint-dir", default="/data/maiG_v2/runs/rotating_4090")
    p.add_argument("--stage1-epochs", type=int, default=1)
    p.add_argument("--refine-epochs", type=int, default=1)
    p.add_argument("--skip-preprocess", action="store_true", default=True)
    p.add_argument("--do-preprocess", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-file", default="terminal.log", help="Append terminal output to this file")
    args = p.parse_args()

    setup_file_logging(args.log_file)
    logger.info("train_all started")
    logger.info("config=%s cache_root=%s checkpoint_dir=%s", args.config, args.cache_root, args.checkpoint_dir)

    t_total = time.time()

    if args.do_preprocess:
        cfg = load_config(args.config)
        maxsubdiv = int((cfg.get("data", {}) or {}).get("maxsubdiv", 64))
        if not run(
            [PYTHON, "scripts/preprocess_all.py", "--num-workers", str(args.num_workers), "--maxsubdiv", str(maxsubdiv), "--cache-root", args.cache_root],
            "Phase 1: preprocess",
        ):
            sys.exit(1)
    else:
        cache_audio = Path(args.cache_root) / "_audio"
        n_audio = len(list(cache_audio.glob("*.pt"))) if cache_audio.exists() else 0
        if n_audio == 0:
            logger.error("Audio cache is empty: %s", cache_audio)
            logger.error("Run preprocessing first, or pass --do-preprocess")
            sys.exit(1)
        logger.info("Audio cache ready: %d files", n_audio)

    if not run(
        [PYTHON, "train.py", "--config", args.config, "--train-stage", "stage1", "--max-epochs", str(args.stage1_epochs)],
        f"Phase 2: train stage1 ({args.stage1_epochs} epochs)",
    ):
        sys.exit(1)

    checkpoint = str(stage_checkpoint(args.checkpoint_dir, "stage1", "best"))
    if not run(
        [PYTHON, "scripts/build_stage234_cache.py", "--step", "all", "--checkpoint", checkpoint, "--config", args.config, "--cache-root", args.cache_root, "--num-workers", str(args.num_workers)],
        "Phase 3: export hidden + build downstream caches",
    ):
        sys.exit(1)

    for stage, label in [
        ("stage2_star", "Stage 2 star detail"),
        ("hold", "Stage 3 hold duration"),
        ("touch_hold", "Stage 4 touch hold duration"),
        ("stage5_touch", "Stage 5 touch pattern"),
        ("stage6_break_note", "Stage 6 break note"),
        ("stage7_firework_note", "Stage 7 firework note"),
    ]:
        if not run(
            [PYTHON, "train.py", "--config", args.config, "--train-stage", stage, "--max-epochs", str(args.refine_epochs)] + resume_args(args.checkpoint_dir, stage),
            f"Phase refine: {label} ({args.refine_epochs} epochs)",
        ):
            sys.exit(1)

    total = time.time() - t_total
    logger.info("=" * 60)
    logger.info("All stages complete. elapsed=%.1fh", total / 3600)
    logger.info("Checkpoint dir: %s", args.checkpoint_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
