from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BreakClassifier(nn.Module):
    def __init__(self, hidden_dim: int = 384, num_layers: int = 4, num_heads: int = 6, num_positions: int = 8, vocab_size: int = 161512, dropout: float = 0.1, stage1_dim: int = 768):
        super().__init__()
        self.num_positions = num_positions
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Embedding(16384, hidden_dim)
        self.stage1_proj = nn.Linear(stage1_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 4, dropout, batch_first=True, activation="gelu") for _ in range(num_layers)]
        )
        self.head = nn.Linear(hidden_dim, num_positions * 2)

    def _align_hidden(self, tokens: torch.Tensor, stage1_hidden: torch.Tensor) -> torch.Tensor:
        if stage1_hidden.size(1) == tokens.size(1):
            return stage1_hidden
        if stage1_hidden.size(1) < tokens.size(1):
            pad = stage1_hidden[:, -1:, :].expand(stage1_hidden.size(0), tokens.size(1) - stage1_hidden.size(1), -1)
            return torch.cat([stage1_hidden, pad], dim=1)
        return stage1_hidden[:, :tokens.size(1), :]

    def forward(self, tokens: torch.Tensor, stage1_hidden: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = tokens.shape
        stage1_hidden = self._align_hidden(tokens, stage1_hidden)
        stage1_hidden = self.stage1_proj(stage1_hidden)
        pos = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(bsz, -1)
        x = self.token_embed(tokens) + self.pos_embed(pos)
        x = x + stage1_hidden
        for layer in self.layers:
            x = layer(x)
        return self.head(x).view(bsz, seq_len, self.num_positions, 2)

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor, press_mask: torch.Tensor) -> torch.Tensor:
        valid = press_mask.bool()
        if valid.sum() == 0:
            return logits.sum() * 0.0
        flat_logits = logits[valid].reshape(-1, 2)
        flat_targets = targets[valid].reshape(-1) - 1  # 1→0(tap), 2→1(break)
        return F.cross_entropy(flat_logits, flat_targets)


breakG = BreakClassifier
