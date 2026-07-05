from __future__ import annotations

import torch


def align_sequence_features(features: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    Align [B, T_src, D] features onto a token/event axis with length target_len.
    Uses nearest-neighbor index mapping instead of naive truncation.
    """
    if target_len <= 0:
        raise ValueError("target_len must be positive")
    if features.size(1) == target_len:
        return features
    if features.size(1) == 1:
        return features.expand(features.size(0), target_len, features.size(2))

    src_len = int(features.size(1))
    device = features.device
    idx = torch.linspace(0, src_len - 1, target_len, device=device)
    idx = idx.round().long().clamp_(0, src_len - 1)
    return features.index_select(1, idx)


def gather_query_features(features: torch.Tensor, query_slots: torch.Tensor) -> torch.Tensor:
    """
    Gather [B, D] query features from [B, T, D] aligned sequence features.
    """
    bsz, seq_len, _ = features.shape
    slots = query_slots.view(-1).long().clamp(0, max(seq_len - 1, 0))
    return features[torch.arange(bsz, device=features.device), slots]
