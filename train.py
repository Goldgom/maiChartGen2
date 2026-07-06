from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Any

import torch

from train.data import StageCacheDataset, SplitStageDataset, build_loader, make_train_val_split
from train.metrics import VAL_FN_MAP
from train.optim import build_optimizer, build_scheduler
from train.recipes import break_step, slide_step, spike_step, stage1_step, touch_step, hold_step, touch_hold_step, star_step, touch_pattern_step
from train.trainer import RotatingMultiStageTrainer, StageRuntime
from models import MaiGenerator, TouchRefiner, SlidePathGenerator, BreakClassifier, SpikeClassifier, HoldDurationPredictor, TouchHoldDurationPredictor, TouchPatternRefiner
from models.slide_stage import SlideStarRefiner


def _load_config(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
        return yaml.safe_load(text)
    except Exception:
        return _load_yaml_subset(text)


def _load_yaml_subset(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(0, result)]

    def parse_value(raw: str):
        raw = raw.strip()
        if raw.lower() in {"true", "false"}:
            return raw.lower() == "true"
        if raw.lower() == "null":
            return None
        if raw.startswith("[") and raw.endswith("]"):
            return ast.literal_eval(raw)
        try:
            if "." in raw:
                return float(raw)
            return int(raw)
        except ValueError:
            return raw.strip('"').strip("'")

    for line in text.splitlines():
        line = line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        while stack and indent < stack[-1][0]:
            stack.pop()
        current = stack[-1][1]

        if stripped.endswith(":"):
            key = stripped[:-1].strip()
            current[key] = {}
            stack.append((indent + 2, current[key]))
            continue

        if ":" in stripped:
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value == "":
                current[key] = {}
                stack.append((indent + 2, current[key]))
            else:
                current[key] = parse_value(value)
            continue
    return result


def _build_model(stage: str, cfg: dict[str, Any]):
    if stage == "stage2_star":
        return SlidePathGenerator(**dict(cfg["models"].get("slide", {})))
    if stage == "stage5_touch":
        return TouchPatternRefiner(**dict(cfg["models"].get("stage5_touch", cfg["models"].get("touch", {}))))
    if stage == "stage6_break_note":
        return BreakClassifier(**dict(cfg["models"].get("break", {})))
    if stage == "stage7_firework_note":
        return SpikeClassifier(**dict(cfg["models"].get("spike", {})))
    mcfg = dict(cfg["models"][stage])  # copy
    if stage == "stage1":
        ec = cfg.get("audio_encodec", {})
        ms = cfg.get("audio_multiscale", {})
        mcfg.setdefault("audio_vocab_size", ec.get("n_layers", 1) * 1024 + 3)
        mcfg["global_stride"] = ms.get("global_stride", 8)
        mcfg["local_window_s"] = ms.get("local_window_s", 5.0)
        mcfg["local_slots_per_sec"] = ms.get("local_slots_per_sec", 184)
        mcfg["local_dilation_base"] = ms.get("local_dilation_base", 4)
        mcfg["max_spectral_len"] = ms.get("max_spectral_len", 16384)
        mcfg["use_spectral_sliding_window"] = ms.get("use_spectral_sliding_window", False)
        mcfg["spectral_window_len"] = ms.get("spectral_window_len", 4096)
        mcfg["spectral_window_stride"] = ms.get("spectral_window_stride", 2048)
        return MaiGenerator(**mcfg)
    if stage == "touch":
        return TouchRefiner(**mcfg)
    if stage == "slide":
        return SlidePathGenerator(**mcfg)
    if stage == "break":
        return BreakClassifier(**mcfg)
    if stage == "spike":
        return SpikeClassifier(**mcfg)
    if stage == "hold":
        return HoldDurationPredictor(**mcfg)
    if stage == "touch_hold":
        return TouchHoldDurationPredictor(**mcfg)
    if stage == "star":
        return SlideStarRefiner(**mcfg)
    raise ValueError(stage)


def _stage_cfg(cfg: dict[str, Any], section: str, stage: str) -> dict[str, Any]:
    base = dict(cfg.get(section, {}))
    overrides = base.pop("stages", {}) or {}
    if stage in overrides:
        base.update(overrides[stage] or {})
    return base


def _stage_value(value: Any, stage: str, default: Any) -> Any:
    if isinstance(value, dict):
        return value.get(stage, default)
    return default if value is None else value


def _build_stage(
    stage: str,
    cfg: dict[str, Any],
    train_ids: set[str] | None = None,
    val_ids: set[str] | None = None,
) -> StageRuntime | None:
    data_cfg = cfg["data"]
    train_cfg = _stage_cfg(cfg, "train", stage)
    optim_cfg = _stage_cfg(cfg, "optim", stage)
    sched_cfg = _stage_cfg(cfg, "sched", stage)

    cache_dir = Path(data_cfg["cache_root"])
    # star stage 复用 slide 缓存（同数据源，精炼任务）
    cache_stage = {"star": "slide"}.get(stage, stage)
    max_tokens = data_cfg.get(f"max_{stage}_tokens", data_cfg.get("max_tokens"))
    max_onset = data_cfg.get(f"max_{stage}_onset", data_cfg.get("max_onset"))
    detail_context_window = data_cfg.get(
        f"{stage}_context_window",
        data_cfg.get("detail_context_window", 512),
    )
    detail_hidden_cache_size = data_cfg.get(
        f"{stage}_hidden_cache_size",
        data_cfg.get("detail_hidden_cache_size", 2),
    )
    default_shuffle = not (cache_stage == "stage2_star" and detail_context_window)
    shuffle_train = bool(data_cfg.get(f"{stage}_shuffle", data_cfg.get("shuffle", default_shuffle)))

    # ── 构建训练集 ──
    if train_ids is not None:
        train_dataset = SplitStageDataset(
            cache_dir, cache_stage, train_ids,
            max_tokens=int(max_tokens) if max_tokens is not None else None,
            max_onset=int(max_onset) if max_onset is not None else None,
            detail_context_window=int(detail_context_window) if detail_context_window else None,
            hidden_cache_size=int(detail_hidden_cache_size) if detail_hidden_cache_size is not None else None,
        )
    else:
        train_dataset = StageCacheDataset(
            cache_dir, cache_stage,
            max_tokens=int(max_tokens) if max_tokens is not None else None,
            max_onset=int(max_onset) if max_onset is not None else None,
            detail_context_window=int(detail_context_window) if detail_context_window else None,
            hidden_cache_size=int(detail_hidden_cache_size) if detail_hidden_cache_size is not None else None,
        )

    if len(train_dataset) == 0:
        raise FileNotFoundError(
            f"Stage '{stage}': cache/{cache_stage}/ 目录为空或划分后无训练数据。请先运行预处理:\n"
            f"  python scripts/preprocess_all.py"
        )

    train_loader = build_loader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=shuffle_train,
        num_workers=data_cfg.get("num_workers", 0),
    )

    # ── 构建验证集 ──
    val_loader = None
    if val_ids is not None and len(val_ids) > 0:
        val_dataset = SplitStageDataset(
            cache_dir, cache_stage, val_ids,
            max_tokens=int(max_tokens) if max_tokens is not None else None,
            max_onset=int(max_onset) if max_onset is not None else None,
            detail_context_window=int(detail_context_window) if detail_context_window else None,
            hidden_cache_size=int(detail_hidden_cache_size) if detail_hidden_cache_size is not None else None,
        )
        if len(val_dataset) > 0:
            val_loader = build_loader(
                val_dataset,
                batch_size=train_cfg["batch_size"],
                shuffle=False,  # 验证集不打乱
                num_workers=data_cfg.get("num_workers", 0),
            )
            import logging
            logging.info(
                "Stage '%s': train=%d samples, val=%d samples",
                stage, len(train_dataset), len(val_dataset),
            )
        else:
            import logging
            logging.warning("Stage '%s': 验证集为空（val_ids 中无对应缓存），跳过验证", stage)

    model = _build_model(stage, cfg)
    optimizer = build_optimizer(model.parameters(), optim_cfg)
    scheduler = build_scheduler(optimizer, sched_cfg, total_steps=1)
    step_map = {
        "stage1": stage1_step,
        "stage2_star": slide_step,
        "touch": touch_step,
        "stage5_touch": touch_pattern_step,
        "slide": slide_step,
        "break": break_step,
        "stage6_break_note": break_step,
        "spike": spike_step,
        "stage7_firework_note": spike_step,
        "hold": hold_step,
        "touch_hold": touch_hold_step,
        "star": star_step,
    }
    return StageRuntime(
        name=stage,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        step_fn=step_map[stage],
        val_fn=VAL_FN_MAP.get(stage),
        turn_batches=int(_stage_value(train_cfg.get("turn_batches"), stage, 1)),
        grad_accum_steps=int(train_cfg.get("grad_accum_steps", 1)),
        offload_to_cpu=bool(train_cfg.get("offload_to_cpu", False)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rotating_4090.yaml")
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="最大训练 epoch 数（每 epoch = 全量数据过一遍）")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--train-stage", default=None,
                        choices=["stage1", "stage2_star", "touch", "slide", "hold", "touch_hold", "stage5_touch", "star", "break", "stage6_break_note", "spike", "stage7_firework_note"],
                        help="只训练指定 stage (不分阶段训练请省略)")
    parser.add_argument("--no-val", action="store_true",
                        help="禁用验证集划分，全部数据用于训练")
    parser.add_argument("--split-file", default=None,
                        help="train/val 划分 JSON 文件路径（由 build_stage_split.py 生成）")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    torch.manual_seed(int(cfg.get("seed", 42)))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.get("seed", 42)))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ── Train / Val 划分 ──
    train_ids: set[str] | None = None
    val_ids: set[str] | None = None

    if not args.no_val:
        val_cfg = cfg.get("val", {}) or {}
        val_level_threshold = float(val_cfg.get("max_val_level", 14.0))
        val_ratio = float(val_cfg.get("val_ratio", 0.10))
        val_seed = int(val_cfg.get("seed", cfg.get("seed", 42)))
        split_file = args.split_file or val_cfg.get("split_file")
        cache_root = cfg.get("data", {}).get("cache_root", "cache")

        train_ids, val_ids = make_train_val_split(
            cache_root=cache_root,
            val_level_threshold=val_level_threshold,
            val_ratio=val_ratio,
            seed=val_seed,
            split_file=split_file,
        )
        import logging
        logging.info(
            "Train/Val 划分: train=%d charts, val=%d charts (max_val_level=%.0f, val_ratio=%.0f%%)",
            len(train_ids), len(val_ids), val_level_threshold, val_ratio * 100,
        )

    all_stage_names = ["stage1", "stage2_star", "hold", "touch_hold", "stage5_touch", "stage6_break_note", "stage7_firework_note"]
    if args.train_stage:
        stage_names = [args.train_stage]
        import logging
        logging.info(f"单阶段训练模式: {args.train_stage}")
    else:
        stage_names = all_stage_names

    stages = [s for s in (_build_stage(stage, cfg, train_ids, val_ids) for stage in stage_names) if s is not None]
    precision = "bf16" if cfg.get("use_bf16", False) and torch.cuda.is_available() else str(cfg.get("precision", "amp"))
    trainer = RotatingMultiStageTrainer(
        stages=stages,
        device=cfg.get("device", "cuda"),
        grad_clip_norm=float(cfg["train"].get("clip_grad_norm", 1.0)),
        precision=precision,
        log_every=int(cfg["train"].get("log_every", 20)),
        eval_every_turns=int(cfg["train"].get("eval_every_turns", 1)),
        save_every_turns=int(cfg["train"].get("save_every_turns", 1)),
        checkpoint_dir=cfg["checkpoint"]["dir"],
        keep_last=int(cfg["checkpoint"].get("keep_last", 3)),
        best_metric=str(cfg["checkpoint"].get("best_metric", "val_loss")),
        best_mode=str(cfg["checkpoint"].get("best_mode", "min")),
        resume_path=args.resume,
        cfg=cfg,
    )
    if args.resume:
        try:
            trainer.load(args.resume, restore_progress=(args.train_stage is None))
        except TypeError:
            trainer.load(args.resume)

    # ── 训练前数据集报告 ──
    from train.dataset_info import print_dataset_info, print_runtime_dataset_info
    cache_root = cfg.get("data", {}).get("cache_root", "cache")
    stage_label = args.train_stage or "all"
    print_dataset_info(cache_root, prefix=stage_label)
    print_runtime_dataset_info(stages, train_ids=train_ids, val_ids=val_ids)

    # ── 确定训练上限 ──
    if args.max_turns is not None:
        max_turns = args.max_turns
    elif args.train_stage:
        max_turns = int(_stage_cfg(cfg, "train", args.train_stage).get("max_turns", cfg["train"]["max_turns"]))
    else:
        max_turns = int(cfg["train"]["max_turns"])

    # epoch 上限（优先 CLI，其次 config）
    max_epochs = args.max_epochs
    if max_epochs is None:
        max_epochs = int(cfg["train"].get("max_epochs", 0)) or None

    # ── 训练配置摘要 ──
    import logging
    log = logging.getLogger(__name__)
    log.info("=" * 60)
    log.info("训练配置摘要")
    log.info("=" * 60)
    for s in stages:
        train_n = len(s.train_loader.dataset)
        val_n = len(s.val_loader.dataset) if s.val_loader else 0
        batch_size = max(1, int(cfg["train"].get("batch_size", 1)))
        turns_per_epoch = max(1, math.ceil(train_n / max(1, s.turn_batches * batch_size)))
        log.info(
            "  %s: train=%d  val=%d  ~%d turns/epoch",
            s.name, train_n, val_n, turns_per_epoch,
        )
    if max_epochs:
        log.info("  训练模式: epoch (%d epochs)", max_epochs)
    elif max_turns:
        log.info("  训练模式: turn (%d max turns)", max_turns)
    log.info("=" * 60)

    trainer.fit(max_turns=max_turns, max_epochs=max_epochs)


if __name__ == "__main__":
    main()
