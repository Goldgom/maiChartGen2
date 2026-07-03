from __future__ import annotations

import math

import torch


def build_optimizer(params, cfg: dict):
    name = cfg.get("name", "adamw").lower()
    lr = float(cfg.get("lr", 3e-4))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    betas = tuple(cfg.get("betas", (0.9, 0.95)))

    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, betas=betas)
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(optimizer, cfg: dict, total_steps: int):
    name = cfg.get("name", "cosine").lower()
    warmup_steps = int(cfg.get("warmup_steps", 0))
    min_lr = float(cfg.get("min_lr", 1e-5))

    if name == "constant":
        return None

    if name == "cosine":
        def lr_lambda(step: int):
            if total_steps <= 0:
                return 1.0
            if step < warmup_steps:
                return max(1e-8, step / max(1, warmup_steps))
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
            base_lr = 1.0
            return max(min_lr / max(base_lr, 1e-8), cosine)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        scheduler.last_epoch = 0
        return scheduler

    raise ValueError(f"Unsupported scheduler: {name}")
