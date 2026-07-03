"""
循环训练: Stage 1 ↔ Stage 2/3/4 交替训练

每轮循环 (以 epoch 为单位，1 epoch = 全量数据过一遍):
  1. 训练 Stage 1 (N epochs)  →  每 epoch 后验证集评估，判断 best
  2. 导出 Stage 1 hidden → 重建 Touch/Break/Spike/Slide 缓存
  3. 训练 Stage 2 (Touch)   N epochs
  4. 训练 Stage 2.5 (Slide) N epochs
  5. 训练 Stage 3 (Break)   N epochs
  6. 训练 Stage 4 (Spike)   N epochs
  → 重复

用法:
  python scripts/train_cycle.py --cycles 20 --epochs-per-cycle 1
  python scripts/train_cycle.py --cycles 10 --epochs-per-cycle 2 --no-val
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_cycle")

PYTHON = sys.executable


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
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def stage_cycle_epochs(cfg: dict[str, Any], stage: str, override: int | None) -> int:
    """读取每 stage 每循环的 epoch 数。"""
    if override is not None:
        return override
    train_cfg = cfg.get("train", {}) or {}
    stage_cfg = (train_cfg.get("stages", {}) or {}).get(stage, {}) or {}
    return int(stage_cfg.get("epochs_per_cycle", train_cfg.get("epochs_per_cycle", 1)))


def run(cmd: list[str], desc: str) -> bool:
    logger.info("-" * 50)
    logger.info(desc)
    logger.info(f"  {' '.join(cmd)}")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=_PROJECT_ROOT)
    elapsed = time.time() - t0
    if r.returncode == 0:
        logger.info(f"  ✅ {elapsed:.0f}s")
        return True
    logger.error(f"  ❌ exit={r.returncode} {elapsed:.0f}s")
    return False


def main():
    p = argparse.ArgumentParser(description="循环训练 (epoch-based)")
    p.add_argument("--config", default="configs/rotating_4090.yaml")
    p.add_argument("--cache-root", default="/data/maiG_v2/cache")
    p.add_argument("--checkpoint-dir", default="/data/maiG_v2/runs/rotating_4090")
    p.add_argument("--cycles", type=int, default=20, help="循环次数")
    p.add_argument("--epochs-per-cycle", type=int, default=None,
                   help="每阶段每轮训练 epoch 数；默认读取 config (1)")
    p.add_argument("--no-val", action="store_true",
                   help="禁用验证集划分")
    p.add_argument("--skip-preprocess", action="store_true", default=True,
                   help="跳过预处理 (默认)")
    p.add_argument("--do-preprocess", action="store_true",
                   help="先运行预处理再训练")
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()

    t_total = time.time()
    cfg = load_config(args.config)

    # ── 预处理 ──
    if args.do_preprocess:
        if not run(
            [PYTHON, "scripts/preprocess_all.py",
             "--num-workers", str(args.num_workers),
             "--cache-root", args.cache_root],
            "预处理"
        ):
            sys.exit(1)
    else:
        cache_audio = Path(args.cache_root) / "_audio"
        if not cache_audio.exists() or not list(cache_audio.glob("*.pt")):
            logger.error(f"缓存为空: {cache_audio}")
            logger.error("请先运行: python scripts/preprocess_all.py")
            sys.exit(1)
        logger.info(f"缓存就绪 ({len(list(cache_audio.glob('*.pt')))} 首)")

    # ── 构建传递给 train.py 的额外参数 ──
    extra_flags: list[str] = []
    if args.no_val:
        extra_flags.append("--no-val")

    for cycle in range(1, args.cycles + 1):
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"  Cycle {cycle}/{args.cycles}")
        logger.info("=" * 60)

        # 1. 训练 Stage 1（epoch-based）
        E = stage_cycle_epochs(cfg, "stage1", args.epochs_per_cycle)
        ckpt = stage_checkpoint(args.checkpoint_dir, "stage1", "last")
        resume_flag = ["--resume", str(ckpt)] if ckpt.exists() else []
        if not run(
            [PYTHON, "train.py", "--config", args.config,
             "--train-stage", "stage1", "--max-epochs", str(E)] + resume_flag + extra_flags,
            f"[Cycle {cycle}] Stage 1 训练 ({E} epochs)"
        ):
            sys.exit(1)

        # 2. 重建缓存（使用 stage1 best checkpoint）
        ckpt = stage_checkpoint(args.checkpoint_dir, "stage1", "best")
        if not run(
            [PYTHON, "scripts/build_stage234_cache.py",
             "--step", "all", "--checkpoint", str(ckpt),
             "--config", args.config, "--cache-root", args.cache_root],
            f"[Cycle {cycle}] 导出 Hidden + 重建缓存"
        ):
            sys.exit(1)

        # 3-6. Stage 2/2.5/3/4（epoch-based）
        for stage, name in [("touch", "Touch"), ("slide", "Slide"),
                            ("break", "Break"), ("spike", "Spike")]:
            E = stage_cycle_epochs(cfg, stage, args.epochs_per_cycle)
            ckpt = stage_checkpoint(args.checkpoint_dir, stage, "last")
            resume_flag = ["--resume", str(ckpt)] if ckpt.exists() else []
            if not run(
                [PYTHON, "train.py", "--config", args.config,
                 "--train-stage", stage, "--max-epochs", str(E)] + resume_flag + extra_flags,
                f"[Cycle {cycle}] Stage - {name} ({E} epochs)"
            ):
                sys.exit(1)

        logger.info(f"Cycle {cycle} 完成")

    total = time.time() - t_total
    stages = ["stage1", "touch", "slide", "break", "spike"]
    total_epochs = args.cycles * sum(stage_cycle_epochs(cfg, s, args.epochs_per_cycle) for s in stages)
    logger.info("=" * 60)
    logger.info(f"🎉 全部 {args.cycles} 轮完成! ({total/3600:.1f}h, ~{total_epochs} total epochs)")
    logger.info(f"  Checkpoint: {args.checkpoint_dir}/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
