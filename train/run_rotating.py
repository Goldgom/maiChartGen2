from __future__ import annotations

import argparse

from torch.utils.data import DataLoader

from .data import StageCacheDataset, SplitStageDataset, build_loader, make_train_val_split
from .metrics import VAL_FN_MAP
from .optim import build_optimizer, build_scheduler
from .recipes import (
    StageRecipe,
    break_step,
    slide_step,
    spike_step,
    stage1_step,
    touch_step,
)
from .trainer import RotatingMultiStageTrainer, StageRuntime
from models import MaiGenerator, TouchRefiner, SlidePathGenerator, BreakClassifier, SpikeClassifier


def build_stage(
    name: str,
    cache_root: str,
    batch_size: int,
    turn_batches: int,
    offload: bool,
    lr: float,
    train_ids: set[str] | None = None,
    val_ids: set[str] | None = None,
):
    # ── 训练集 ──
    if train_ids is not None:
        dataset = SplitStageDataset(cache_root, name, train_ids)
    else:
        dataset = StageCacheDataset(cache_root, name)
    if len(dataset) == 0:
        return None
    loader = build_loader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # ── 验证集 ──
    val_loader = None
    if val_ids is not None and len(val_ids) > 0:
        val_dataset = SplitStageDataset(cache_root, name, val_ids)
        if len(val_dataset) > 0:
            val_loader = build_loader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    if name == "stage1":
        model = MaiGenerator(hidden_dim=64, num_layers=2, num_heads=4)
        step_fn = stage1_step
    elif name == "touch":
        model = TouchRefiner(hidden_dim=64, num_layers=2, num_heads=4)
        step_fn = touch_step
    elif name == "slide":
        model = SlidePathGenerator(hidden_dim=64, num_layers=2, num_heads=4)
        step_fn = slide_step
    elif name == "break":
        model = BreakClassifier(hidden_dim=64, num_layers=2, num_heads=4)
        step_fn = break_step
    elif name == "spike":
        model = SpikeClassifier(hidden_dim=64, num_layers=2, num_heads=4)
        step_fn = spike_step
    else:
        raise ValueError(name)

    optimizer = build_optimizer(model.parameters(), {"name": "adamw", "lr": lr, "weight_decay": 0.01})
    scheduler = build_scheduler(optimizer, {"name": "constant"}, total_steps=1)
    return StageRuntime(
        name=name,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=loader,
        val_loader=val_loader,
        step_fn=step_fn,
        val_fn=VAL_FN_MAP.get(name),
        turn_batches=turn_batches,
        offload_to_cpu=offload,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", default="/data/maiG_v2/cache")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--turn-batches", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--no-val", action="store_true")
    parser.add_argument("--split-file", default=None)
    args = parser.parse_args()

    train_ids: set[str] | None = None
    val_ids: set[str] | None = None
    if not args.no_val:
        train_ids, val_ids = make_train_val_split(
            cache_root=args.cache_root,
            val_level_threshold=14.0,
            val_ratio=0.10,
            seed=42,
            split_file=args.split_file,
        )

    stages = [
        build_stage("stage1", args.cache_root, args.batch_size, args.turn_batches, args.offload, args.lr, train_ids, val_ids),
        build_stage("touch", args.cache_root, args.batch_size, args.turn_batches, args.offload, args.lr, train_ids, val_ids),
        build_stage("slide", args.cache_root, args.batch_size, args.turn_batches, args.offload, args.lr, train_ids, val_ids),
        build_stage("break", args.cache_root, args.batch_size, args.turn_batches, args.offload, args.lr, train_ids, val_ids),
        build_stage("spike", args.cache_root, args.batch_size, args.turn_batches, args.offload, args.lr, train_ids, val_ids),
    ]
    stages = [s for s in stages if s is not None]
    trainer = RotatingMultiStageTrainer(
        stages, device="cuda", grad_clip_norm=1.0, precision="amp",
        checkpoint_dir="runs/rotating",
        best_metric="val_loss" if val_ids else "loss",
    )
    trainer.fit(max_turns=args.max_turns, max_epochs=args.max_epochs)


if __name__ == "__main__":
    main()
