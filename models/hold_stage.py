from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.audio_context import align_sequence_features, gather_query_features


DEFAULT_MAXSUBDIV = 64


def duration_to_row_count(dur: tuple[int, int] | None, maxsubdiv: int = DEFAULT_MAXSUBDIV) -> int:
    if dur is None:
        return 0
    num, den = max(1, int(dur[0])), max(1, int(dur[1]))
    beats = num / den
    return max(1, int(round(beats * max(1, int(maxsubdiv)))))


def row_count_to_duration(rows: int, maxsubdiv: int = DEFAULT_MAXSUBDIV) -> tuple[int, int]:
    rows = max(1, int(rows))
    return (rows, max(1, int(maxsubdiv)))


class HoldDurationPredictor(nn.Module):
    """
    Predict one queried hold event at a time.

    Each training sample contains a partially completed chart sequence where
    previously predicted holds have already been written back as ongoing states.
    `query_slots` identifies the current hold head whose duration should be
    completed.
    """

    def __init__(
        self,
        hidden_dim: int = 384,
        num_layers: int = 4,
        num_heads: int = 6,
        vocab_size: int = 161512,
        dropout: float = 0.1,
        stage1_dim: int = 768,
        onset_dim: int = 3,
        max_hold_rows: int = 512,
        max_seq_len: int = 16384,
    ):
        super().__init__()
        self.max_hold_rows = int(max_hold_rows)
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Embedding(max_seq_len, hidden_dim)
        self.query_embed = nn.Embedding(2, hidden_dim)
        self.stage1_proj = nn.Linear(stage1_dim, hidden_dim)
        self.audio_proj = nn.Linear(stage1_dim, hidden_dim)
        self.onset_proj = nn.Linear(onset_dim, hidden_dim)
        self.local_fuse = nn.Linear(hidden_dim * 3, hidden_dim)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    hidden_dim,
                    num_heads,
                    hidden_dim * 4,
                    dropout,
                    batch_first=True,
                    activation="gelu",
                )
                for _ in range(num_layers)
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def _align(self, tokens: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.size(1) == tokens.size(1):
            return hidden
        if hidden.size(1) < tokens.size(1):
            pad = hidden[:, -1:, :].expand(hidden.size(0), tokens.size(1) - hidden.size(1), -1)
            return torch.cat([hidden, pad], dim=1)
        return hidden[:, : tokens.size(1), :]

    def _build_query_flags(self, tokens: torch.Tensor, query_slots: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = tokens.shape
        slots = query_slots.view(-1).long().clamp(0, max(seq_len - 1, 0))
        flags = torch.zeros(bsz, seq_len, dtype=torch.long, device=tokens.device)
        flags.scatter_(1, slots.unsqueeze(1), 1)
        return flags

    def forward(
        self,
        tokens: torch.Tensor,
        stage1_hidden: torch.Tensor,
        query_slots: torch.Tensor,
        audio_memory: torch.Tensor | None = None,
        onset: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        bsz, seq_len = tokens.shape
        stage1_hidden = self._align(tokens, stage1_hidden)
        s1 = self.stage1_proj(stage1_hidden)

        pos = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(bsz, -1)
        query_flags = self._build_query_flags(tokens, query_slots)
        x = self.token_embed(tokens) + self.pos_embed(pos) + self.query_embed(query_flags) + s1

        if audio_memory is not None:
            a = self.audio_proj(audio_memory)
            a_pool = a.mean(dim=1, keepdim=True).expand(bsz, seq_len, -1)
            x = x + a_pool
        else:
            a = None

        if onset is not None:
            o = self.onset_proj(onset.float())
            o_aligned = align_sequence_features(o, seq_len)
            x = x + o_aligned
        else:
            o_aligned = None

        for layer in self.layers:
            x = layer(x)

        slots = query_slots.view(-1).long().clamp(0, max(seq_len - 1, 0))
        q_hidden = x[torch.arange(bsz, device=tokens.device), slots]
        local_parts = [q_hidden, gather_query_features(s1, query_slots)]
        if a is not None:
            local_parts.append(gather_query_features(align_sequence_features(a, seq_len), query_slots))
        elif o_aligned is not None:
            local_parts.append(gather_query_features(o_aligned, query_slots))
        else:
            local_parts.append(torch.zeros_like(q_hidden))
        q_hidden = self.local_fuse(torch.cat(local_parts, dim=-1))
        row_logits = self.head(q_hidden).squeeze(-1)
        return {"row_logits": row_logits, "hidden": x}

    def compute_loss(self, outputs: dict, row_targets: torch.Tensor) -> torch.Tensor:
        pred = outputs["row_logits"].float()
        tgt = row_targets.view(-1).float().clamp(min=0, max=self.max_hold_rows)
        if pred.numel() == 0 or tgt.numel() == 0:
            return pred.sum() * 0.0
        return F.smooth_l1_loss(pred, tgt)


class TouchHoldDurationPredictor(HoldDurationPredictor):
    pass


holdG = HoldDurationPredictor
touchHoldG = TouchHoldDurationPredictor
