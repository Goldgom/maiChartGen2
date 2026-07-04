"""
一键完整训练流程: 预处理 → 训练 Stage 1 → 导出 Hidden → 训练 Stage 2/3/4

用法:
  python scripts/train_all.py                              # 全流程
  python scripts/train_all.py --skip-preprocess             # 跳过预处理
  python scripts/train_all.py --config configs/rotating_4090.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_all")

PYTHON = sys.executable
CONFIG = "configs/rotating_4090.yaml"


def load_config(path: str | Path) -> dict[str, Any]:
    from typing import Any
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
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
    logger.info(f"  命令: {' '.join(cmd)}")
    logger.info("=" * 60)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=_PROJECT_ROOT)
    elapsed = time.time() - t0
    if r.returncode == 0:
        logger.info(f"✅ 完成 ({elapsed:.0f}s)")
        return True
    else:
        logger.error(f"❌ 失败 (exit={r.returncode}, {elapsed:.0f}s)")
        return False


def main():
    p = argparse.ArgumentParser(description="一键完整训练")
    p.add_argument("--config", default=CONFIG)
    p.add_argument("--cache-root", default="/data/maiG_v2/cache")
    p.add_argument("--checkpoint-dir", default="/data/maiG_v2/runs/rotating_4090")
    p.add_argument("--stage1-turns", type=int, default=5000)
    p.add_argument("--stage2-turns", type=int, default=2000)
    p.add_argument("--stage25-turns", type=int, default=2000)
    p.add_argument("--stage3-turns", type=int, default=2000)
    p.add_argument("--stage4-turns", type=int, default=2000)
    p.add_argument("--skip-preprocess", action="store_true", default=True,
                   help="跳过预处理 (默认)")
    p.add_argument("--do-preprocess", action="store_true",
                   help="先运行预处理")
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()

    t_total = time.time()

    # ═══════════════════════════════════════════════════════════════
    # Phase 1: 预处理
    # ═══════════════════════════════════════════════════════════════
    if args.do_preprocess:
        cfg = load_config(args.config)
        maxsubdiv = int((cfg.get("data", {}) or {}).get("maxsubdiv", 64))
        if not run(
            [PYTHON, "scripts/preprocess_all.py",
             "--num-workers", str(args.num_workers),
             "--maxsubdiv", str(maxsubdiv),
             "--cache-root", args.cache_root],
            "Phase 1: 预处理"
        ):
            sys.exit(1)
    else:
        cache_audio = Path(args.cache_root) / "_audio"
        if not cache_audio.exists() or not list(cache_audio.glob("*.pt")):
            logger.error(f"缓存为空: {cache_audio}，请运行 python scripts/preprocess_all.py")
            sys.exit(1)
        logger.info(f"缓存就绪 ({len(list(cache_audio.glob('*.pt')))} 首)")

    # ═══════════════════════════════════════════════════════════════
    # Phase 2: 训练 Stage 1 (maiG)
    # ═══════════════════════════════════════════════════════════════
    if not run(
        [PYTHON, "train.py", "--config", args.config,
         "--train-stage", "stage1", "--max-epochs", "1"],
        f"Phase 2: 训练 Stage 1 - maiG"
    ):
        sys.exit(1)

    # ═══════════════════════════════════════════════════════════════
    # Phase 3: 导出 Hidden + 构建缓存
    # ═══════════════════════════════════════════════════════════════
    checkpoint = str(stage_checkpoint(args.checkpoint_dir, "stage1", "best"))
    if not run(
        [PYTHON, "scripts/build_stage234_cache.py",
         "--step", "all", "--checkpoint", checkpoint,
         "--config", args.config, "--cache-root", args.cache_root],
        f"Phase 3: 导出 Hidden + 构建缓存"
    ):
        sys.exit(1)

    # ═══════════════════════════════════════════════════════════════
    # Phase 4-10: 训练 Stage 2-7
    # ═══════════════════════════════════════════════════════════════
    for stage, phase_name in [
        ("slide",      "Stage2 Slide星星"),
        ("hold",       "Stage3 Hold时长"),
        ("touch_hold", "Stage4 TouchHold"),
        ("star",       "Stage5 星星精炼"),
        ("break",      "Stage6 Break绝赞"),
        ("spike",      "Stage7 Spike烟花"),
    ]:
        if not run(
            [PYTHON, "train.py", "--config", args.config,
             "--train-stage", stage, "--max-epochs", "1"] + resume_args(args.checkpoint_dir, stage),
            f"Phase - {phase_name}"
        ):
            sys.exit(1)

    total = time.time() - t_total
    logger.info("=" * 60)
    logger.info(f"🎉 全部完成! ({total/3600:.1f}h)")
    logger.info(f"  Checkpoint: {args.checkpoint_dir}/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
