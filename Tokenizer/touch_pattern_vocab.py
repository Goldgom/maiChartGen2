"""Touch pattern tokenizer for Stage 5 refinement.

Each token is a 33-bit mask over maimai DX touch zones. This represents every
possible simultaneous touch-zone arrangement exactly without a giant softmax.
"""

from __future__ import annotations

import torch

from Tokenizer.touch_expander import NUM_ZONES, zone_index, zone_name

TOUCH_PATTERN_NUM_ZONES = NUM_ZONES
TOUCH_PATTERN_VOCAB_SIZE = 1 << TOUCH_PATTERN_NUM_ZONES


def encode_zones(zones: list[int] | set[int] | tuple[int, ...]) -> int:
    token = 0
    for z in zones:
        zi = int(z)
        if not (0 <= zi < TOUCH_PATTERN_NUM_ZONES):
            raise ValueError(f"touch zone out of range: {zi}")
        token |= 1 << zi
    return token


def decode_token(token: int) -> list[int]:
    value = int(token)
    if value < 0 or value >= TOUCH_PATTERN_VOCAB_SIZE:
        raise ValueError(f"touch pattern token out of range: {value}")
    return [i for i in range(TOUCH_PATTERN_NUM_ZONES) if value & (1 << i)]


def encode_zone_names(names: list[str] | tuple[str, ...] | set[str]) -> int:
    return encode_zones([zone_index(n) for n in names])


def decode_token_names(token: int) -> list[str]:
    return [zone_name(z) for z in decode_token(token)]


def token_to_multihot(token: int, *, dtype=torch.float32) -> torch.Tensor:
    y = torch.zeros(TOUCH_PATTERN_NUM_ZONES, dtype=dtype)
    for z in decode_token(token):
        y[z] = 1
    return y


def multihot_to_token(values: torch.Tensor, threshold: float = 0.5) -> int:
    if values.numel() != TOUCH_PATTERN_NUM_ZONES:
        raise ValueError(f"expected {TOUCH_PATTERN_NUM_ZONES} zones, got {values.numel()}")
    active = torch.nonzero(values.reshape(-1) >= threshold, as_tuple=False).reshape(-1).tolist()
    return encode_zones(active)


def zones_to_multihot(zones: list[int] | set[int] | tuple[int, ...], *, dtype=torch.float32) -> torch.Tensor:
    return token_to_multihot(encode_zones(zones), dtype=dtype)
