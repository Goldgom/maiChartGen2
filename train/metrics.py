from __future__ import annotations

import torch


def token_accuracy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> float:
    pred = logits.argmax(dim=-1)
    mask = targets != ignore_index
    if mask.sum().item() == 0:
        return 0.0
    return float((pred[mask] == targets[mask]).float().mean().item())


def binary_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == targets).float().mean().item())


# ── Validation step functions (no_grad) ────────────────────────────────────


def _as_batch_val(x: torch.Tensor | None) -> torch.Tensor | None:
    if x is None:
        return None
    if x.dim() == 0:
        return x.view(1)
    if x.dim() == 1:
        return x.unsqueeze(0)
    return x


def stage1_val_fn(model, batch: dict, device: torch.device) -> dict[str, float]:
    """Stage 1 验证: 计算 loss + token accuracy。"""
    from train.recipes import stage1_step
    loss, stats = stage1_step(model, batch, device)
    return {"val_loss": float(loss.detach().item())}


def touch_val_fn(model, batch: dict, device: torch.device) -> dict[str, float]:
    """Touch 验证: 计算 loss + zone accuracy。"""
    from train.recipes import touch_step
    loss, stats = touch_step(model, batch, device)
    return {"val_loss": float(loss.detach().item())}


def slide_val_fn(model, batch: dict, device: torch.device) -> dict[str, float]:
    """Slide 验证: 计算 loss。"""
    from train.recipes import slide_step
    loss, stats = slide_step(model, batch, device)
    return {"val_loss": float(loss.detach().item())}


def break_val_fn(model, batch: dict, device: torch.device) -> dict[str, float]:
    """Break 验证: 计算 loss。"""
    from train.recipes import break_step
    loss, stats = break_step(model, batch, device)
    return {"val_loss": float(loss.detach().item())}


def spike_val_fn(model, batch: dict, device: torch.device) -> dict[str, float]:
    """Spike 验证: 计算 loss。"""
    from train.recipes import spike_step
    loss, stats = spike_step(model, batch, device)
    return {"val_loss": float(loss.detach().item())}


# 验证函数映射
VAL_FN_MAP: dict[str, callable] = {
    "stage1": stage1_val_fn,
    "touch": touch_val_fn,
    "slide": slide_val_fn,
    "break": break_val_fn,
    "spike": spike_val_fn,
}

