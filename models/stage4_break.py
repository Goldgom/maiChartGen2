"""
models/stage4_break.py — Stage 4: 逐 token 预测 Break

输入:
  - stage3_chart:  (B, T) int   含 hold/slide 参数的完整谱面 token
  - audio_tokens/beat/difficulty/level/tags

输出:
  - break_logits: (B, T, 2)  每个位置 break/not-break logits

训练:
  - 双向 Transformer (非 causal)
  - Binary Cross-Entropy, 在所有 note 位置计算
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import (
    StageConfig, AudioEncoder, ConditionEmbedding,
    ChartAudioFusion, make_model, build_causal_mask,
)


class Stage4BreakModel(nn.Module):
    """Stage 4: 双向预测 Break 标记"""

    def __init__(self, cfg: StageConfig):
        super().__init__()
        self.cfg = cfg

        self.audio_encoder = AudioEncoder(cfg)
        self.cond_embed = ConditionEmbedding(cfg)
        self.chart_embed = nn.Embedding(cfg.chart_vocab_size, cfg.d_model)
        self.chart_fusion = ChartAudioFusion(cfg)

        # 双向: 不带 causal mask
        self.layers = make_model(cfg, cfg.n_layer, cross_attn=True)

        self.ln_final = nn.LayerNorm(cfg.d_model)
        self.max_object_slots = getattr(cfg, "max_object_slots", 16)
        self.slot_embed = nn.Embedding(self.max_object_slots, cfg.d_model)
        self.head = nn.Linear(cfg.d_model, 2)  # [not_break, break]

    def forward(
        self,
        stage3_chart: torch.Tensor,
        audio_tokens: torch.Tensor,
        beat_signal: torch.Tensor,
        difficulty: torch.Tensor,
        level: torch.Tensor,
        tag_ids: torch.Tensor,
        break_targets: torch.Tensor | None = None,  # (B, T) bool
        note_mask: torch.Tensor | None = None,       # (B, T) bool, 哪些位置有音符
    ) -> dict:
        """
        Returns:
            logits: (B, T, 2)
            loss:   仅在 note 位置计算 BCE
        """
        B, T, _ = audio_tokens.shape
        device = audio_tokens.device

        # 音频 + 条件
        audio_feat = self.audio_encoder(audio_tokens)
        # 谱面嵌入 + 位置编码 + 同帧音频依赖 + 条件
        chart_x = self.chart_embed(stage3_chart.long())
        cond = self.cond_embed(
            beat_signal,
            difficulty,
            level,
            tag_ids,
            frame_query=chart_x,
        )
        x = self.chart_fusion(chart_x, audio_feat, cond)

        # 双向 Transformer
        causal_mask = build_causal_mask(T, device)
        for layer in self.layers:
            x = layer(x, memory=audio_feat, causal_mask=causal_mask)

        # 二分类头
        x = self.ln_final(x)
        slot_ids = torch.arange(self.max_object_slots, device=device)
        slot_x = x.unsqueeze(2) + self.slot_embed(slot_ids).view(1, 1, self.max_object_slots, -1)
        logits = self.head(slot_x)  # (B, T, S, 2)

        result = {"logits": logits, "frame_logits": logits[:, :, 0, :]}

        if break_targets is not None and note_mask is not None:
            if break_targets.dim() == 2:
                break_targets = break_targets.unsqueeze(-1)
            if note_mask.dim() == 2:
                note_mask = note_mask.unsqueeze(-1)
            if break_targets.shape[-1] > self.max_object_slots:
                break_targets = break_targets[..., :self.max_object_slots]
                note_mask = note_mask[..., :self.max_object_slots]
            if break_targets.shape[-1] < self.max_object_slots:
                pad = self.max_object_slots - break_targets.shape[-1]
                break_targets = F.pad(break_targets, (0, pad))
                note_mask = F.pad(note_mask, (0, pad))
            active = note_mask
            if active.sum() > 0:
                loss = F.cross_entropy(
                    logits[active],
                    break_targets[active].long(),
                )
            else:
                loss = torch.tensor(0.0, device=device)
            result["loss"] = loss

        return result

    @torch.no_grad()
    def predict(self, stage3_chart, audio_tokens, beat_signal,
                difficulty, level, tag_ids) -> torch.Tensor:
        """输出 break 预测 (bool)"""
        self.eval()
        result = self.forward(
            stage3_chart, audio_tokens, beat_signal,
            difficulty, level, tag_ids,
        )
        return result["logits"].argmax(dim=-1).bool()
