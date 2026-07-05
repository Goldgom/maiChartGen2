from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.audio_context import align_sequence_features
from Tokenizer.touch_pattern_vocab import TOUCH_PATTERN_NUM_ZONES


class TouchPatternRefiner(nn.Module):
    """Stage 5: refine coarse touch heads into exact touch-zone patterns."""

    def __init__(
        self,
        hidden_dim: int = 384,
        num_layers: int = 4,
        num_heads: int = 6,
        vocab_size: int = 161512,
        dropout: float = 0.1,
        stage1_dim: int = 768,
        onset_dim: int = 3,
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Embedding(16384, hidden_dim)
        self.stage1_proj = nn.Linear(stage1_dim, hidden_dim)
        self.audio_proj = nn.Linear(stage1_dim, hidden_dim)
        self.onset_proj = nn.Linear(onset_dim, hidden_dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                hidden_dim,
                num_heads,
                hidden_dim * 4,
                dropout,
                batch_first=True,
                activation="gelu",
            )
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(hidden_dim, TOUCH_PATTERN_NUM_ZONES)

    def _align_hidden(self, tokens: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.size(1) == tokens.size(1):
            return hidden
        if hidden.size(1) < tokens.size(1):
            pad = hidden[:, -1:, :].expand(hidden.size(0), tokens.size(1) - hidden.size(1), -1)
            return torch.cat([hidden, pad], dim=1)
        return hidden[:, :tokens.size(1), :]

    def forward(
        self,
        tokens: torch.Tensor,
        stage1_hidden: torch.Tensor,
        audio_memory: torch.Tensor | None = None,
        onset: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seq_len = tokens.shape
        stage1_hidden = self._align_hidden(tokens, stage1_hidden)
        x = self.token_embed(tokens)
        pos = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(bsz, -1)
        x = x + self.pos_embed(pos) + self.stage1_proj(stage1_hidden)

        if audio_memory is not None:
            a = self.audio_proj(audio_memory)
            x = x + align_sequence_features(a, seq_len)

        if onset is not None:
            o = self.onset_proj(onset.float())
            x = x + align_sequence_features(o, seq_len)

        for layer in self.layers:
            x = layer(x)

        return self.head(x)

    def compute_loss(
        self,
        logits: torch.Tensor,
        pattern_targets: torch.Tensor,
        touch_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid = touch_mask.bool()
        if valid.sum() == 0:
            return logits.sum() * 0.0
        pred = logits[valid].reshape(-1, TOUCH_PATTERN_NUM_ZONES)
        tgt = pattern_targets[valid].reshape(-1, TOUCH_PATTERN_NUM_ZONES).float()
        return F.binary_cross_entropy_with_logits(pred, tgt)


touchPatternG = TouchPatternRefiner
