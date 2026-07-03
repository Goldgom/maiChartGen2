from __future__ import annotations

import argparse

from torch.utils.data import DataLoader

from .data import StageCacheDataset, build_loader
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


def build_stage(name: str, cache_root: str, batch_size: int, turn_batches: int, offload: bool, lr: float):
    dataset = StageCacheDataset(cache_root, name)
    if len(dataset) == 0:
        return None
    loader = build_loader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

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
        val_loader=None,
        step_fn=step_fn,
        turn_batches=turn_batches,
        offload_to_cpu=offload,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", default="/data/maiG_v2/cache")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--turn-batches", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--offload", action="store_true")
    args = parser.parse_args()

    stages = [
        build_stage("stage1", args.cache_root, args.batch_size, args.turn_batches, args.offload, args.lr),
        build_stage("touch", args.cache_root, args.batch_size, args.turn_batches, args.offload, args.lr),
        build_stage("slide", args.cache_root, args.batch_size, args.turn_batches, args.offload, args.lr),
        build_stage("break", args.cache_root, args.batch_size, args.turn_batches, args.offload, args.lr),
        build_stage("spike", args.cache_root, args.batch_size, args.turn_batches, args.offload, args.lr),
    ]
    stages = [s for s in stages if s is not None]
    trainer = RotatingMultiStageTrainer(stages, device="cuda", grad_clip_norm=1.0, precision="amp", checkpoint_dir="runs/rotating")
    trainer.fit(max_turns=args.max_turns)


if __name__ == "__main__":
    main()
