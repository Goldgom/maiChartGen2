"""
models/stage3_slide.py — Stage 3: 自回归补全 Slide 路径

输入:
  - stage2_chart:  (B, T) int   含 hold 长度的谱面 token
  - audio_tokens/beat/difficulty/level/tags

输出:
  - slide_path_logits: (B, T, S, S_vocab)  slide 路径 token logits

Slide 路径独立编码:
  - 每种不同的 slide 路径 (如 "-4", ">5-8", "V28", "*V28") 映射为独立的 token
  - 同一帧多个 slide/timing token 按 slot 序列建模

训练:
  - 仅 slide 位置计算 loss
  - 时间维度使用 causal mask
  - slot 维度使用 causal transformer 建模同帧多 token 关系
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import (
    StageConfig, AudioEncoder, ConditionEmbedding,
    ChartAudioFusion, make_model, build_causal_mask, _init_weights,
)


class Stage3SlideModel(nn.Module):
    """Stage 3: 自回归补全 Slide 路径

    Slide 路径 token:
      - 每个 slide segment 是一个 token (如 ">5", "-4", "V28")
      - 多段用多个 token 表示
      - 特殊 <EOS> token 表示路径结束
    """

    def __init__(self, cfg: StageConfig):
        super().__init__()
        self.cfg = cfg

        self.audio_encoder = AudioEncoder(cfg)
        self.cond_embed = ConditionEmbedding(cfg)
        self.chart_embed = nn.Embedding(cfg.chart_vocab_size, cfg.d_model)
        self.chart_fusion = ChartAudioFusion(cfg)

        self.layers = make_model(cfg, cfg.n_layer, cross_attn=True)

        self.ln_final = nn.LayerNorm(cfg.d_model)
        self.max_slide_slots = getattr(cfg, "max_slide_slots", 8)
        self.slot_embed = nn.Embedding(self.max_slide_slots, cfg.d_model)
        slot_layers = max(1, int(getattr(cfg, "slot_n_layer", 2)))
        self.slot_layers = make_model(cfg, slot_layers, cross_attn=False)
        self.slot_ln = nn.LayerNorm(cfg.d_model)
        # 每个 slide 位置输出多个路径+timing token
        self.head = nn.Linear(cfg.d_model, cfg.slide_vocab_size)
        for module in [self.chart_embed, self.ln_final, self.slot_embed, self.slot_ln, self.head]:
            module.apply(lambda m: _init_weights(m, cfg.init_std))

    def forward(
        self,
        stage2_chart: torch.Tensor,     # (B, T) int
        audio_tokens: torch.Tensor,
        beat_signal: torch.Tensor,
        difficulty: torch.Tensor,
        level: torch.Tensor,
        tag_ids: torch.Tensor,
        slide_path_targets: torch.Tensor | None = None,  # (B, T, L) int, L=最大路径长度
        slide_mask: torch.Tensor | None = None,           # (B, T) bool
    ) -> dict:
        """
        slide_path_targets: (B, T, L), 每帧最多 L 个路径 token
          - 0 = padding (该帧非 slide, 或路径结束)
          - >0 = 路径 segment token ID

        训练时对 slide 位置的每个路径+timing token 计算 loss。
        """
        B, T, _ = audio_tokens.shape
        device = audio_tokens.device

        # 1. 音频 + 条件
        audio_feat = self.audio_encoder(audio_tokens)
        # 2. 谱面嵌入 + 位置编码 + 同帧音频依赖 + 条件
        chart_x = self.chart_embed(stage2_chart.long())
        cond = self.cond_embed(
            beat_signal,
            difficulty,
            level,
            tag_ids,
            frame_query=chart_x,
        )
        x = self.chart_fusion(chart_x, audio_feat, cond)

        # 3. Causal transformer
        causal_mask = build_causal_mask(T, device)
        for layer in self.layers:
            x = layer(x, memory=audio_feat, causal_mask=causal_mask)

        # 4. Slide 路径预测: 对每一帧内部的 slot 序列做短程因果建模。
        x = self.ln_final(x)
        slot_ids = torch.arange(self.max_slide_slots, device=device)
        slot_x = x.unsqueeze(2) + self.slot_embed(slot_ids).view(1, 1, self.max_slide_slots, -1)
        slot_x = slot_x.reshape(B * T, self.max_slide_slots, -1)
        slot_mask = build_causal_mask(self.max_slide_slots, device)
        for layer in self.slot_layers:
            slot_x = layer(slot_x, causal_mask=slot_mask)
        slot_x = self.slot_ln(slot_x).reshape(B, T, self.max_slide_slots, -1)
        logits = self.head(slot_x)  # (B, T, S, V_slide)

        result = {"logits": logits, "frame_logits": logits[:, :, 0, :]}

        if slide_path_targets is not None and slide_mask is not None:
            # 只在存在 slide token 的 slot 上计算 loss。
            if slide_path_targets.dim() == 2:
                slide_path_targets = slide_path_targets.unsqueeze(-1)
            if slide_mask.dim() == 2:
                slide_mask = slide_mask.unsqueeze(-1)
            if slide_path_targets.shape[-1] > self.max_slide_slots:
                slide_path_targets = slide_path_targets[..., :self.max_slide_slots]
                slide_mask = slide_mask[..., :self.max_slide_slots]
            if slide_path_targets.shape[-1] < self.max_slide_slots:
                pad = self.max_slide_slots - slide_path_targets.shape[-1]
                slide_path_targets = F.pad(slide_path_targets, (0, pad))
                slide_mask = F.pad(slide_mask, (0, pad))
            active = slide_mask & (slide_path_targets > 0)
            if active.sum() > 0:
                loss = F.cross_entropy(
                    logits[active],
                    slide_path_targets[active].long(),
                )
            else:
                loss = torch.tensor(0.0, device=device)
            result["loss"] = loss

        return result


# ============================================================
# Slide 路径词汇表构建器
# ============================================================

def build_slide_vocab(tokens: list) -> dict[str, int]:
    """从 slide token 中提取所有唯一路径, 构建词汇表

    路径分解规则:
      "path:-4"          → ["-4"]
      "path:>5-8"        → [">5", "-8"]
      "path:>8*V28"      → [">8", "*V28"]
      "path:V35"         → ["V35"]

    Args:
        tokens: SimaiToken 列表 (含 params)

    Returns:
        {path_segment: id}, 0 保留给 <PAD>, 1 保留给 <EOS>
    """
    from SimaiToken import SimaiTokenType

    vocab = {"<PAD>": 0, "<EOS>": 1}
    next_id = 2

    for t in tokens:
        if t.token_type != SimaiTokenType.SLIDE:
            continue
        path = t.slide_path
        if not path:
            continue
        # 分解路径: ">5-8" → [">5", "-8"], ">8*V28" → [">8", "*V28"]
        segments = _split_slide_path(path)
        for seg in segments:
            if seg not in vocab:
                vocab[seg] = next_id
                next_id += 1

    return vocab


def _split_slide_path(path: str) -> list[str]:
    """分解 slide 路径字符串为 segment 列表"""
    import re
    # 匹配: connector + digits, 如 "-4", ">5", "V28", "*V28"
    # connector 可以是: -, >, <, ^, v, V, p, q, s, z, w, pp, qq
    # * 后跟 connector: *V28, *-4
    segments = re.findall(
        r'\*?(?:pp|qq|[-><^vVpqszw])\d+',
        path
    )
    return segments if segments else [path]
