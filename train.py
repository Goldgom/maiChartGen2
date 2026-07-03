from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import torch

from train.data import StageCacheDataset, build_loader
from train.optim import build_optimizer, build_scheduler
from train.recipes import break_step, slide_step, spike_step, stage1_step, touch_step
from train.trainer import RotatingMultiStageTrainer, StageRuntime
from models import MaiGenerator, TouchRefiner, SlidePathGenerator, BreakClassifier, SpikeClassifier


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


def _build_stage(stage: str, cfg: dict[str, Any]) -> StageRuntime | None:
    data_cfg = cfg["data"]
    train_cfg = _stage_cfg(cfg, "train", stage)
    optim_cfg = _stage_cfg(cfg, "optim", stage)
    sched_cfg = _stage_cfg(cfg, "sched", stage)

    cache_dir = Path(data_cfg["cache_root"])
    max_tokens = data_cfg.get(f"max_{stage}_tokens", data_cfg.get("max_tokens"))
    max_onset = data_cfg.get(f"max_{stage}_onset", data_cfg.get("max_onset"))
    dataset = StageCacheDataset(
        cache_dir,
        stage,
        max_tokens=int(max_tokens) if max_tokens is not None else None,
        max_onset=int(max_onset) if max_onset is not None else None,
    )
    if len(dataset) == 0:
        raise FileNotFoundError(
            f"Stage '{stage}': cache/{stage}/ 目录为空。请先运行预处理:\n"
            f"  python scripts/preprocess_all.py"
        )
    loader = build_loader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 0),
    )

    model = _build_model(stage, cfg)
    optimizer = build_optimizer(model.parameters(), optim_cfg)
    scheduler = build_scheduler(optimizer, sched_cfg, total_steps=1)
    step_map = {
        "stage1": stage1_step,
        "touch": touch_step,
        "slide": slide_step,
        "break": break_step,
        "spike": spike_step,
    }
    return StageRuntime(
        name=stage,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=loader,
        val_loader=None,
        step_fn=step_map[stage],
        turn_batches=int(_stage_value(train_cfg.get("turn_batches"), stage, 1)),
        grad_accum_steps=int(train_cfg.get("grad_accum_steps", 1)),
        offload_to_cpu=bool(train_cfg.get("offload_to_cpu", False)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rotating_4090.yaml")
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--train-stage", default=None,
                        choices=["stage1", "touch", "slide", "break", "spike"],
                        help="只训练指定 stage (不分阶段训练请省略)")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    torch.manual_seed(int(cfg.get("seed", 42)))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.get("seed", 42)))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    all_stage_names = ["stage1", "touch", "slide", "break", "spike"]
    if args.train_stage:
        stage_names = [args.train_stage]
        import logging
        logging.info(f"单阶段训练模式: {args.train_stage}")
    else:
        stage_names = all_stage_names

    stages = [s for s in (_build_stage(stage, cfg) for stage in stage_names) if s is not None]
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
        best_metric=str(cfg["checkpoint"].get("best_metric", "loss")),
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
    from train.dataset_info import print_dataset_info
    cache_root = cfg.get("data", {}).get("cache_root", "cache")
    stage_label = args.train_stage or "all"
    print_dataset_info(cache_root, prefix=stage_label)

    if args.max_turns is not None:
        max_turns = args.max_turns
    elif args.train_stage:
        max_turns = int(_stage_cfg(cfg, "train", args.train_stage).get("max_turns", cfg["train"]["max_turns"]))
    else:
        max_turns = int(cfg["train"]["max_turns"])
    trainer.fit(max_turns=max_turns)


if __name__ == "__main__":
    main()
