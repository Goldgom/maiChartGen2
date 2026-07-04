from __future__ import annotations

import torch


def select_by_generation_probability(
    probs: torch.Tensor,
    valid_mask: torch.Tensor,
    generation_prob: float,
    *,
    mode: str = "topk",
    temperature: float = 1.0,
) -> torch.Tensor:
    """Select positive labels from model probabilities under a target rate.

    Args:
        probs: Probability tensor, any shape.
        valid_mask: Boolean tensor with the same shape; invalid entries are never selected.
        generation_prob: Desired fraction of valid entries to select, in [0, 1].
        mode:
            - "topk": select the highest-probability entries. Deterministic.
            - "sample": sample without replacement using probabilities as weights.
            - "bernoulli": independent Bernoulli using calibrated probabilities whose
              mean approximately matches generation_prob.
        temperature: Sampling/top-k sharpness. Lower is sharper.
    """
    if probs.shape != valid_mask.shape:
        raise ValueError(f"shape mismatch: probs={tuple(probs.shape)} mask={tuple(valid_mask.shape)}")

    valid = valid_mask.bool()
    out = torch.zeros_like(valid)
    n_valid = int(valid.sum().item())
    if n_valid == 0:
        return out

    rate = max(0.0, min(1.0, float(generation_prob)))
    if rate <= 0.0:
        return out
    if rate >= 1.0:
        out[valid] = True
        return out

    flat_probs = probs[valid].float().clamp(0.0, 1.0)
    k = int(round(n_valid * rate))
    if k <= 0:
        return out

    temp = max(1e-6, float(temperature))
    scores = torch.logit(flat_probs.clamp(1e-6, 1.0 - 1e-6)) / temp

    if mode == "topk":
        idx = torch.topk(scores, k=min(k, n_valid)).indices
    elif mode == "sample":
        weights = torch.softmax(scores, dim=0)
        idx = torch.multinomial(weights, num_samples=min(k, n_valid), replacement=False)
    elif mode == "bernoulli":
        mean = flat_probs.mean().clamp_min(1e-6)
        calibrated = (flat_probs * (rate / mean)).clamp(0.0, 1.0)
        chosen = torch.rand_like(calibrated) < calibrated
        valid_indices = torch.nonzero(valid, as_tuple=False)
        out[tuple(valid_indices[chosen].T)] = True
        return out
    else:
        raise ValueError(f"unknown selection mode: {mode}")

    valid_indices = torch.nonzero(valid, as_tuple=False)
    out[tuple(valid_indices[idx].T)] = True
    return out
