"""
models/stage1_chart.py — Stage 1: 音频 + 条件 → 扁平谱面 token

输入:
  - audio_tokens:  (B, T, C)  EnCodec 音频 token
  - beat_signal:   (B, T, 2)  节拍信号 [beat, downbeat]
  - difficulty:    (B,) int    难度 ID (1~6)
  - level:         (B,) float  等级 (线性!)
  - tag_ids:       (B, K) int  标签 token, -1=padding

输出:
  - chart_logits:  (B, T, V)   谱面 token logits (V=chart_vocab_size)

训练: 标准 Cross-Entropy, 每个帧位置独立预测 (非自回归)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import (
    StageConfig, AudioEncoder, ConditionEmbedding,
    make_model,
)


class Stage1ChartModel(nn.Module):
    """Stage 1: 音乐→谱面骨架"""

    def __init__(self, cfg: StageConfig):
        super().__init__()
        self.cfg = cfg

        # 音频编码器
        self.audio_encoder = AudioEncoder(cfg)

        # 条件嵌入
        self.cond_embed = ConditionEmbedding(cfg)

        # 输入投影
        self.input_proj = nn.Linear(cfg.d_model, cfg.d_model)

        # Transformer 层 (带 cross-attention 到音频)
        self.layers = make_model(cfg, cfg.n_layer, cross_attn=True)

        # 输出头
        self.ln_final = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.chart_vocab_size)

    def forward(
        self,
        audio_tokens: torch.Tensor,
        beat_signal: torch.Tensor,
        difficulty: torch.Tensor,
        level: torch.Tensor,
        tag_ids: torch.Tensor,
        chart_tokens: torch.Tensor | None = None,  # (B, T) 训练时提供
    ) -> dict:
        """
        Returns:
            logits:  (B, T, V)  谱面 token logits
            loss:    scalar     交叉熵 (仅 chart_tokens 提供时)
        """
        B, T, _ = audio_tokens.shape

        # 1. 音频编码
        audio_feat = self.audio_encoder(audio_tokens)  # (B, T, d_model)

        # 2. 条件嵌入
        cond = self.cond_embed(beat_signal, difficulty, level, tag_ids)  # (B, T, d_model)

        # 3. 输入 = 条件 (Stage1 没有历史 token 输入)
        x = self.input_proj(cond)  # (B, T, d_model)

        # 4. Transformer 编码 (当前帧可关注全局音频)
        for layer in self.layers:
            x = layer(x, memory=audio_feat)

        # 5. 输出
        x = self.ln_final(x)
        logits = self.head(x)  # (B, T, V)

        result = {"logits": logits}

        if chart_tokens is not None:
            # 0 是合法的 no-note 类，不要忽略；
            # 否则一个切片里如果恰好全是空位，CE 会因为“没有有效目标”返回 NaN。
            loss = F.cross_entropy(
                logits.reshape(-1, self.cfg.chart_vocab_size),
                chart_tokens.reshape(-1).long(),
            )
            result["loss"] = loss

        return result

    @torch.no_grad()
    def generate(
        self,
        audio_tokens: torch.Tensor,
        beat_signal: torch.Tensor,
        difficulty: torch.Tensor,
        level: torch.Tensor,
        tag_ids: torch.Tensor,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> torch.Tensor:
        """生成谱面 token 序列"""
        self.eval()
        result = self.forward(audio_tokens, beat_signal, difficulty, level, tag_ids)
        logits = result["logits"]  # (B, T, V)

        # 采样
        if temperature > 0:
            logits = logits / temperature

        if top_k > 0:
            topk_vals, _ = torch.topk(logits, top_k, dim=-1)
            min_topk = topk_vals[:, :, -1:]
            logits = torch.where(logits < min_topk,
                                 torch.full_like(logits, float("-inf")),
                                 logits)

        probs = F.softmax(logits, dim=-1)
        tokens = torch.multinomial(
            probs.reshape(-1, self.cfg.chart_vocab_size), 1
        ).reshape(audio_tokens.shape[0], -1)

        return tokens
