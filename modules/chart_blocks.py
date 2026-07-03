from __future__ import annotations

import torch
import torch.nn as nn


class CausalDecoderBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.ada_ln = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))

    def forward(self, x: torch.Tensor, memory: torch.Tensor, cond: torch.Tensor, causal_mask=None) -> torch.Tensor:
        s1, b1, s2, b2, s3, b3 = self.ada_ln(cond).chunk(6, dim=-1)

        r = x
        x = self.norm1(x) * (1 + s1.unsqueeze(1)) + b1.unsqueeze(1)
        x = self.self_attn(x, x, x, attn_mask=causal_mask, need_weights=False)[0]
        x = self.dropout(x) + r

        r = x
        x = self.norm2(x) * (1 + s2.unsqueeze(1)) + b2.unsqueeze(1)
        x = self.cross_attn(x, memory, memory, need_weights=False)[0]
        x = self.dropout(x) + r

        r = x
        x = self.norm3(x) * (1 + s3.unsqueeze(1)) + b3.unsqueeze(1)
        x = self.ffn(x) + r
        return x


class CausalPathBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, causal_mask=None) -> torch.Tensor:
        r = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x, attn_mask=causal_mask, need_weights=False)[0]
        x = self.dropout(x) + r

        r = x
        x = self.norm2(x)
        x = self.cross_attn(x, memory, memory, need_weights=False)[0]
        x = self.dropout(x) + r

        r = x
        x = self.norm3(x)
        x = self.ffn(x) + r

        # ── 数值保护：NaN/inf 替换为 0，防止后续层放大 ──
        if not torch.isfinite(x).all():
            x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
            # 对极端值做软 clamp
            x = torch.clamp(x, min=-1e4, max=1e4)

        return x


class BidirectionalRefinerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        r = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x, need_weights=False)[0]
        x = self.dropout(x) + r

        r = x
        x = self.norm2(x)
        x = self.cross_attn(x, memory, memory, need_weights=False)[0]
        x = self.dropout(x) + r

        r = x
        x = self.norm3(x)
        x = self.ffn(x) + r
        return x
