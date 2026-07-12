"""
models/stage5_ex.py — Stage 5: 逐 token 预测 Ex-note

输入:
  - stage4_chart:  (B, T) int   含 break 标记的完整谱面 token
  - audio_tokens/beat/difficulty/level/tags

输出:
  - ex_logits: (B, T, 2)  每个位置 ex/not-ex logits

训练:
  - 双向 Transformer (非 causal)
  - Binary Cross-Entropy, 仅在 DX 谱面 (有 ex note) 上训练
  - 只在 note 位置计算 loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import (
    StageConfig, AudioEncoder, ConditionEmbedding,
    make_model, build_causal_mask,
)


class Stage5ExModel(nn.Module):
    """Stage 5: 双向预测 Ex-note 标记 (仅 DX 谱面)"""

    def __init__(self, cfg: StageConfig):
        super().__init__()
        self.cfg = cfg

        self.audio_encoder = AudioEncoder(cfg)
        self.cond_embed = ConditionEmbedding(cfg)
        self.chart_embed = nn.Embedding(cfg.chart_vocab_size, cfg.d_model)

        self.layers = make_model(cfg, cfg.n_layer, cross_attn=True)

        self.ln_final = nn.LayerNorm(cfg.d_model)
        self.max_object_slots = getattr(cfg, "max_object_slots", 16)
        self.slot_embed = nn.Embedding(self.max_object_slots, cfg.d_model)
        self.head = nn.Linear(cfg.d_model, 2)

    def forward(
        self,
        stage4_chart: torch.Tensor,
        audio_tokens: torch.Tensor,
        beat_signal: torch.Tensor,
        difficulty: torch.Tensor,
        level: torch.Tensor,
        tag_ids: torch.Tensor,
        ex_targets: torch.Tensor | None = None,  # (B, T) bool
        note_mask: torch.Tensor | None = None,
    ) -> dict:
        B, T, _ = audio_tokens.shape
        device = audio_tokens.device

        audio_feat = self.audio_encoder(audio_tokens)
        cond = self.cond_embed(beat_signal, difficulty, level, tag_ids)
        x = self.chart_embed(stage4_chart.long()) + cond

        causal_mask = build_causal_mask(T, device)
        for layer in self.layers:
            x = layer(x, memory=audio_feat, causal_mask=causal_mask)

        x = self.ln_final(x)
        slot_ids = torch.arange(self.max_object_slots, device=device)
        slot_x = x.unsqueeze(2) + self.slot_embed(slot_ids).view(1, 1, self.max_object_slots, -1)
        logits = self.head(slot_x)

        result = {"logits": logits, "frame_logits": logits[:, :, 0, :]}

        if ex_targets is not None and note_mask is not None:
            if ex_targets.dim() == 2:
                ex_targets = ex_targets.unsqueeze(-1)
            if note_mask.dim() == 2:
                note_mask = note_mask.unsqueeze(-1)
            if ex_targets.shape[-1] > self.max_object_slots:
                ex_targets = ex_targets[..., :self.max_object_slots]
                note_mask = note_mask[..., :self.max_object_slots]
            if ex_targets.shape[-1] < self.max_object_slots:
                pad = self.max_object_slots - ex_targets.shape[-1]
                ex_targets = F.pad(ex_targets, (0, pad))
                note_mask = F.pad(note_mask, (0, pad))
            active = note_mask
            if active.sum() > 0:
                loss = F.cross_entropy(
                    logits[active],
                    ex_targets[active].long(),
                )
            else:
                loss = torch.tensor(0.0, device=device)
            result["loss"] = loss

        return result

    @torch.no_grad()
    def predict(self, stage4_chart, audio_tokens, beat_signal,
                difficulty, level, tag_ids) -> torch.Tensor:
        self.eval()
        result = self.forward(
            stage4_chart, audio_tokens, beat_signal,
            difficulty, level, tag_ids,
        )
        return result["logits"].argmax(dim=-1).bool()
