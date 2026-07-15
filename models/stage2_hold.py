"""
models/stage2_hold.py — Stage 2: 自回归补全 Hold 长度

输入:
  - stage1_chart:  (B, T) int   Stage1 输出的谱面 token
  - audio_tokens:  (B, T, C)
  - beat_signal:   (B, T, 2)
  - difficulty/level/tags

输出:
  - hold_dur_logits: (B, T, D)  hold 持续时间 logits (D=离散桶数)
  - 仅在 hold token 位置产生有效输出

训练:
  - Causal mask: 第 i 个 hold 只能看到第 0..i-1 个 hold 的信息
  - 构造 hold-only 序列 (去掉非 hold token), 自回归预测每个 hold 的持续时间
  - Loss: Cross-Entropy, 只在 hold 位置计算
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import (
    StageConfig, AudioEncoder, ConditionEmbedding,
    ChartAudioFusion, make_model, build_causal_mask,
)


class Stage2HoldModel(nn.Module):
    """Stage 2: 自回归补全 Hold 持续时间"""

    def __init__(self, cfg: StageConfig):
        super().__init__()
        self.cfg = cfg

        self.audio_encoder = AudioEncoder(cfg)
        self.cond_embed = ConditionEmbedding(cfg)

        # 谱面 token 嵌入
        self.chart_embed = nn.Embedding(cfg.chart_vocab_size, cfg.d_model)
        self.chart_fusion = ChartAudioFusion(cfg)

        # Transformer (causal, cross-attn 到音频)
        self.layers = make_model(cfg, cfg.n_layer, cross_attn=True)

        # 输出: 离散化 hold 持续时间
        self.max_hold_slots = getattr(cfg, "max_hold_slots", 8)
        self.slot_embed = nn.Embedding(self.max_hold_slots, cfg.d_model)
        self.ln_final = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.hold_dur_bins)

    def forward(
        self,
        stage1_chart: torch.Tensor,     # (B, T) int
        audio_tokens: torch.Tensor,
        beat_signal: torch.Tensor,
        difficulty: torch.Tensor,
        level: torch.Tensor,
        tag_ids: torch.Tensor,
        hold_dur_targets: torch.Tensor | None = None,  # (B, T) int, 0=not hold
        hold_mask: torch.Tensor | None = None,          # (B, T) bool
    ) -> dict:
        """
        Args:
            stage1_chart: Stage1 输出的谱面 token
            hold_dur_targets: hold 持续时间目标 (离散桶 ID), 0=非 hold
            hold_mask: 标记哪些位置是 hold

        Returns:
            logits: (B, T, D)
            loss:   仅在 hold 位置计算
        """
        B, T, _ = audio_tokens.shape
        device = audio_tokens.device

        # 1. 音频 + 条件
        audio_feat = self.audio_encoder(audio_tokens)
        # 2. 谱面 token 嵌入 + 位置编码 + 同帧音频依赖 + 条件
        chart_x = self.chart_embed(stage1_chart.long())
        cond = self.cond_embed(
            beat_signal,
            difficulty,
            level,
            tag_ids,
            frame_query=chart_x,
        )
        x = self.chart_fusion(chart_x, audio_feat, cond)  # (B, T, d_model)

        # 3. Causal mask: 每帧只能看到之前
        causal_mask = build_causal_mask(T, device)

        # 4. Transformer (causal, cross-attn)
        for layer in self.layers:
            x = layer(x, memory=audio_feat, causal_mask=causal_mask)

        # 5. Hold 持续时间预测
        x = self.ln_final(x)
        slot_ids = torch.arange(self.max_hold_slots, device=device)
        slot_x = x.unsqueeze(2) + self.slot_embed(slot_ids).view(1, 1, self.max_hold_slots, -1)
        logits = self.head(slot_x)  # (B, T, S, D)

        result = {"logits": logits, "frame_logits": logits[:, :, 0, :]}

        if hold_dur_targets is not None and hold_mask is not None:
            # 只在 hold 位置计算 loss
            if hold_dur_targets.dim() == 2:
                hold_dur_targets = hold_dur_targets.unsqueeze(-1)
            if hold_mask.dim() == 2:
                hold_mask = hold_mask.unsqueeze(-1)
            if hold_dur_targets.shape[-1] > self.max_hold_slots:
                hold_dur_targets = hold_dur_targets[..., :self.max_hold_slots]
                hold_mask = hold_mask[..., :self.max_hold_slots]
            if hold_dur_targets.shape[-1] < self.max_hold_slots:
                pad = self.max_hold_slots - hold_dur_targets.shape[-1]
                hold_dur_targets = F.pad(hold_dur_targets, (0, pad))
                hold_mask = F.pad(hold_mask, (0, pad))
            active = hold_mask & (hold_dur_targets > 0)
            if active.sum() > 0:
                loss = F.cross_entropy(
                    logits[active],
                    hold_dur_targets[active].long(),
                )
            else:
                loss = torch.tensor(0.0, device=device)
            result["loss"] = loss

        return result

    @torch.no_grad()
    def generate(
        self,
        stage1_chart: torch.Tensor,
        audio_tokens: torch.Tensor,
        beat_signal: torch.Tensor,
        difficulty: torch.Tensor,
        level: torch.Tensor,
        tag_ids: torch.Tensor,
        hold_mask: torch.Tensor,
        temperature: float = 0.8,
    ) -> torch.Tensor:
        """自回归生成 hold 持续时间"""
        self.eval()
        result = self.forward(
            stage1_chart, audio_tokens, beat_signal,
            difficulty, level, tag_ids,
        )
        logits = result["logits"]
        if temperature > 0:
            logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        dur_preds = torch.multinomial(
            probs.reshape(-1, self.cfg.hold_dur_bins), 1
        ).reshape(stage1_chart.shape[0], stage1_chart.shape[1], self.max_hold_slots)
        # 只在 hold_mask 位置保留预测值
        if hold_mask.dim() == 2:
            hold_mask = hold_mask.unsqueeze(-1)
        return dur_preds * hold_mask.long()
