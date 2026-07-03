"""
一键预处理: 从原始数据到完整训练缓存

依次执行:
  1. preprocess_audio.py   → cache/_audio/    (音频特征)
  2. preprocess_labels.py  → cache/_labels/   (标注)
                            → cache/stage1/   (Stage 1 训练数据)
                            → cache/slide/    (Slide 训练数据)

用法:
  python scripts/preprocess_all.py                          # 全部
  python scripts/preprocess_all.py --limit 50               # 前50首测试
  python scripts/preprocess_all.py --num-workers 4          # 4线程
  python scripts/preprocess_all.py --skip-existing          # 跳过已有
  python scripts/preprocess_all.py --steps audio            # 只跑音频

Stage 1 训练完成后，运行 Phase 2b:
  python scripts/build_stage234_cache.py --step all --checkpoint /data/maiG_v2/runs/.../best.pt
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
logger = logging.getLogger("preprocess_all")

STEPS = {
    "audio":  ("scripts/preprocess_audio.py",  "Phase 1: 音频特征 → cache/_audio/"),
    "labels": ("scripts/preprocess_labels.py", "Phase 2a: 标签 + Stage1/Slide → cache/"),
    "cache234": ("scripts/build_stage234_cache.py", "Phase 2b: Touch/Break/Spike 缓存 (占位)"),
}


def run_step(script: str, description: str, args: argparse.Namespace) -> bool:
    logger.info("=" * 60)
    logger.info(description)
    logger.info("=" * 60)
    t0 = time.time()

    cmd = [sys.executable, str(Path(_PROJECT_ROOT) / script)]

    # 通用参数 (仅 audio & labels)
    if "build_stage234" not in script:
        if args.num_workers:
            cmd.extend(["--num-workers", str(args.num_workers)])
        if args.data_root:
            cmd.extend(["--data-root", args.data_root])

    if args.limit:
        cmd.extend(["--limit", str(args.limit)])
    # 默认跳过已有缓存，--force 强制重跑
    if args.force:
        cmd.append("--force")

    # audio 专用
    if "audio" in script:
        if args.subdiv:
            cmd.extend(["--subdiv", str(args.subdiv)])
        # 音频存到独立子目录
        cmd.extend(["--cache-root", str(Path(args.cache_root) / "_audio")])
        cmd.extend(["--encodec-layers", str(args.encodec_layers)])

    # labels 专用
    if "labels" in script:
        if args.subdiv:
            cmd.extend(["--subdiv", str(args.subdiv)])
        if args.max_tokens:
            cmd.extend(["--max-tokens", str(args.max_tokens)])
        cmd.extend(["--cache-root", str(args.cache_root)])

    # build_stage234 专用 — 只传必要参数
    if "build_stage234" in script:
        cmd.extend(["--step", "build-caches", "--placeholder", "--cache-root", str(args.cache_root)])
        cmd.extend(["--config", "configs/rotating_4090.yaml"])
        if args.limit:
            cmd.extend(["--limit", str(args.limit)])
        if args.force:
            cmd.append("--force")

    logger.info(f"  命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=_PROJECT_ROOT)

    elapsed = time.time() - t0
    if result.returncode == 0:
        logger.info(f"  ✅ 完成 ({elapsed:.0f}s)")
        return True
    else:
        logger.error(f"  ❌ 失败 (exit={result.returncode}, {elapsed:.0f}s)")
        return False


def main():
    p = argparse.ArgumentParser(description="一键预处理")
    p.add_argument("--steps", default="all", help="执行的步骤: audio, labels, all")
    p.add_argument("--data-root", default="datasets")
    p.add_argument("--cache-root", default="/data/maiG_v2/cache")
    p.add_argument("--limit", type=int, default=None, help="限制歌曲数 (测试用)")
    p.add_argument("--num-workers", type=int, default=1, help="并行线程数")
    p.add_argument("--subdiv", type=int, default=64)
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--force", action="store_true", help="强制重新处理")
    p.add_argument("--encodec-layers", type=int, default=1, help="EnCodec 层数 (1/2)")
    args = p.parse_args()

    steps_to_run = ["audio", "labels", "cache234"] if args.steps == "all" else [s.strip() for s in args.steps.split(",")]
    unknown = set(steps_to_run) - set(STEPS)
    if unknown:
        logger.error(f"未知步骤: {unknown}. 可选: {list(STEPS)}")
        sys.exit(1)

    t_total = time.time()
    failed = []

    for step_name in steps_to_run:
        script, desc = STEPS[step_name]
        if not run_step(script, desc, args):
            failed.append(step_name)
            logger.error(f"步骤 '{step_name}' 失败，停止后续步骤")
            break

    total_elapsed = time.time() - t_total
    if failed:
        logger.error(f"预处理未完成! 失败步骤: {failed} ({total_elapsed:.0f}s)")
        logger.info("")
        logger.info("提示:")
        logger.info("  1. 确保已安装依赖: pip install librosa soundfile torch")
        logger.info("  2. 使用 --skip-existing 跳过已成功的文件续跑")
        logger.info("  3. 单线程更稳定: --num-workers 1")
        sys.exit(1)
    else:
        logger.info("=" * 60)
        logger.info(f"🎉 全部完成! ({total_elapsed:.0f}s)")
        logger.info("=" * 60)
        logger.info("")
        logger.info("下一步:")
        logger.info("  训练:")
        logger.info("     python train.py --config configs/rotating_4090.yaml")
        logger.info("")
        logger.info("  (Stage 1 训练完成后, 用真实 hidden 替换占位缓存:)")
        logger.info("     python scripts/build_stage234_cache.py --step all \\")
        logger.info("         --checkpoint /data/maiG_v2/runs/rotating_4090/best.pt")


if __name__ == "__main__":
    main()
