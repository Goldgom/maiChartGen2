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

