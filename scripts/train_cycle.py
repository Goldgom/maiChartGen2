"""Cycle training driver with terminal logging."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_cycle")


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


def stage_checkpoint(checkpoint_dir: str, stage: str, kind: str = "best") -> Path:
    stage_dir_path = Path(checkpoint_dir) / stage / f"{kind}.pt"
    if stage_dir_path.exists():
        return stage_dir_path
    legacy_stage_path = Path(checkpoint_dir) / f"{stage}_{kind}.pt"
    if legacy_stage_path.exists():
        return legacy_stage_path
    return stage_dir_path


def load_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
        return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.warning("Could not load config %s: %s", path, exc)
        return {}


def stage_cycle_epochs(cfg: dict[str, Any], stage: str, override: int | None) -> int:
    if override is not None:
        return int(override)
    train_cfg = cfg.get("train", {}) or {}
    stage_cfg = (train_cfg.get("stages", {}) or {}).get(stage, {}) or {}
    return int(stage_cfg.get("epochs_per_cycle", train_cfg.get("epochs_per_cycle", 1)))


def run(cmd: list[str], desc: str) -> bool:
    logger.info("-" * 60)
    logger.info(desc)
    logger.info("Command: %s", " ".join(cmd))
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
    p = argparse.ArgumentParser(description="Cycle training driver")
    p.add_argument("--config", default="configs/rotating_4090.yaml")
    p.add_argument("--cache-root", default="/data/maiG_v2/cache")
    p.add_argument("--checkpoint-dir", default="/data/maiG_v2/runs/rotating_4090")
    p.add_argument("--cycles", type=int, default=20)
    p.add_argument("--epochs-per-cycle", type=int, default=None)
    p.add_argument("--no-val", action="store_true")
    p.add_argument("--skip-preprocess", action="store_true", default=True)
    p.add_argument("--do-preprocess", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-file", default="terminal.log", help="Append terminal output to this file")
    args = p.parse_args()

    setup_file_logging(args.log_file)
    logger.info("train_cycle started")
    logger.info(
        "config=%s cache_root=%s checkpoint_dir=%s cycles=%s epochs_per_cycle=%s",
        args.config,
        args.cache_root,
        args.checkpoint_dir,
        args.cycles,
        args.epochs_per_cycle,
    )

    t_total = time.time()
    cfg = load_config(args.config)

    if args.do_preprocess:
        maxsubdiv = int((cfg.get("data", {}) or {}).get("maxsubdiv", 64))
        if not run(
            [
                PYTHON,
                "scripts/preprocess_all.py",
                "--num-workers",
                str(args.num_workers),
                "--maxsubdiv",
                str(maxsubdiv),
                "--cache-root",
                args.cache_root,
            ],
            "Preprocess",
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

    extra_flags: list[str] = []
    if args.no_val:
        extra_flags.append("--no-val")

    stage_order = [
        ("stage2_star", "Stage 2 star detail"),
        ("hold", "Stage 3 hold duration"),
        ("touch_hold", "Stage 4 touch hold duration"),
        ("stage5_touch", "Stage 5 touch pattern"),
        ("stage6_break_note", "Stage 6 break note"),
        ("stage7_firework_note", "Stage 7 firework note"),
    ]

    for cycle in range(1, args.cycles + 1):
        logger.info("=" * 60)
        logger.info("Cycle %d/%d", cycle, args.cycles)
        logger.info("=" * 60)

        epochs = stage_cycle_epochs(cfg, "stage1", args.epochs_per_cycle)
        ckpt = stage_checkpoint(args.checkpoint_dir, "stage1", "last")
        resume_flag = ["--resume", str(ckpt)] if ckpt.exists() else []
        if not run(
            [PYTHON, "train.py", "--config", args.config, "--train-stage", "stage1", "--max-epochs", str(epochs)]
            + resume_flag
            + extra_flags,
            f"[Cycle {cycle}] Train stage1 ({epochs} epochs)",
        ):
            sys.exit(1)

        ckpt = stage_checkpoint(args.checkpoint_dir, "stage1", "best")
        if not run(
            [PYTHON, "scripts/build_stage234_cache.py", "--step", "all", "--checkpoint", str(ckpt), "--config", args.config, "--cache-root", args.cache_root],
            f"[Cycle {cycle}] Export hidden + rebuild downstream caches",
        ):
            sys.exit(1)

        for stage, label in stage_order:
            epochs = stage_cycle_epochs(cfg, stage, args.epochs_per_cycle)
            ckpt = stage_checkpoint(args.checkpoint_dir, stage, "last")
            resume_flag = ["--resume", str(ckpt)] if ckpt.exists() else []
            if not run(
                [PYTHON, "train.py", "--config", args.config, "--train-stage", stage, "--max-epochs", str(epochs)]
                + resume_flag
                + extra_flags,
                f"[Cycle {cycle}] {label} ({epochs} epochs)",
            ):
                sys.exit(1)

        logger.info("Cycle %d complete", cycle)

    total = time.time() - t_total
    total_epochs = args.cycles * (stage_cycle_epochs(cfg, "stage1", args.epochs_per_cycle) + sum(stage_cycle_epochs(cfg, s, args.epochs_per_cycle) for s, _ in stage_order))
    logger.info("=" * 60)
    logger.info("All cycles complete: cycles=%d elapsed=%.1fh total_epochs~%d", args.cycles, total / 3600, total_epochs)
    logger.info("Checkpoint dir: %s", args.checkpoint_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()