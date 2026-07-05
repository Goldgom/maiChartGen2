from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.audio_context import align_sequence_features
from modules.chart_blocks import BidirectionalRefinerBlock
from Tokenizer.config_vocab import (
    BTN_HOLD_START,
    BTN_PRESS,
    ID_TO_CONFIG,
    TCH_HOLD_START,
    TCH_TOUCH,
)


class TouchRefiner(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 768,
        num_layers: int = 6,
        num_heads: int = 12,
        num_zones: int = 33,
        num_states: int = 3,
        vocab_size: int = 161512,
        dropout: float = 0.1,
        onset_dim: int = 3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_zones = num_zones
        self.num_states = num_states
        self.pad_token_id = 0

        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Embedding(16384, hidden_dim)
        self.stage1_proj = nn.Linear(hidden_dim, hidden_dim)
        self.audio_proj = nn.Linear(hidden_dim, hidden_dim)
        self.onset_proj = nn.Linear(onset_dim, hidden_dim)
        self.layers = nn.ModuleList([BidirectionalRefinerBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)])
        self.zone_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_zones * num_states),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, config_tokens: torch.Tensor, stage1_hidden: torch.Tensor, audio_memory: torch.Tensor | None = None, onset: torch.Tensor | None = None) -> torch.Tensor:
        bsz, seq_len = config_tokens.shape
        stage1_hidden = self.stage1_proj(stage1_hidden)
        if audio_memory is not None:
            audio_memory = self.audio_proj(audio_memory)
            memory = torch.cat([stage1_hidden, audio_memory], dim=1)
        else:
            memory = stage1_hidden
        pos = torch.arange(seq_len, device=config_tokens.device).unsqueeze(0).expand(bsz, -1)
        x = self.token_embed(config_tokens) + self.pos_embed(pos) + align_sequence_features(stage1_hidden, seq_len)
        if audio_memory is not None:
            x = x + align_sequence_features(audio_memory, seq_len)
        if onset is not None:
            x = x + align_sequence_features(self.onset_proj(onset.float()), seq_len)
        for layer in self.layers:
            x = layer(x, memory)
        logits = self.zone_head(x)
        return logits.view(bsz, seq_len, self.num_zones, self.num_states)

    def compute_loss(self, zone_logits: torch.Tensor, zone_targets: torch.Tensor, config_tokens: torch.Tensor) -> torch.Tensor:
        valid = config_tokens != self.pad_token_id
        if valid.sum() == 0:
            return zone_logits.sum() * 0.0
        flat_logits = zone_logits[valid].reshape(-1, self.num_states)
        flat_targets = zone_targets[valid].reshape(-1)
        mask = flat_targets > 0
        if mask.sum() == 0:
            return zone_logits.sum() * 0.0
        return F.cross_entropy(flat_logits[mask], flat_targets[mask] - 1)


def build_zone_targets(config_tokens: torch.Tensor) -> torch.Tensor:
    if config_tokens.dim() != 1:
        raise ValueError("build_zone_targets expects 1D token list/tensor")
    targets = torch.zeros(config_tokens.numel(), 33, dtype=torch.long)
    for t, tid in enumerate(config_tokens.tolist()):
        sc = ID_TO_CONFIG.get(tid)
        if sc is None:
            continue
        for zone, state in sc.touches:
            if state not in (TCH_TOUCH, TCH_HOLD_START):
                continue
            if 0 <= zone < 33:
                targets[t, zone] = 2 if state == TCH_HOLD_START else 1
        for pos, state in sc.buttons:
            if state == BTN_HOLD_START:
                pass
            elif state == BTN_PRESS:
                pass
    return targets


def extract_touch_mask(config_tokens: torch.Tensor) -> torch.Tensor:
    if config_tokens.dim() != 1:
        raise ValueError("extract_touch_mask expects 1D token list/tensor")
    mask = torch.zeros(config_tokens.numel(), 33, dtype=torch.bool)
    for t, tid in enumerate(config_tokens.tolist()):
        sc = ID_TO_CONFIG.get(tid)
        if sc is None:
            continue
        for zone, _ in sc.touches:
            if 0 <= zone < 33:
                mask[t, zone] = True
    return mask


touchG = TouchRefiner
