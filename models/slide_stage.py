from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.chart_blocks import CausalPathBlock
from Tokenizer.slide_star_vocab import (
    SLD_STAR_BOS, SLD_STAR_EOS, SLD_STAR_PAD, SLD_STAR_VOCAB_SIZE,
    SLD_POS_BASE, SLD_POS_COUNT, SLD_CONN_BASE, SLD_CONN_COUNT,
    SLD_DUR_NUM_BASE, SLD_DUR_NUM_COUNT, SLD_DUR_DEN_BASE, SLD_DUR_DEN_COUNT,
    pos_to_id,
)


class SlidePathGenerator(nn.Module):
    """
    Stage 2: 星星（Slide）详细路径生成。

    输入:
      - target_path:  目标 token 序列 [BOS, dur_num, dur_den, CONN, POS, ..., EOS]
      - start_pos:    星星起始按钮 (1-8)
      - audio_memory: 全局音频摘要 [B, T_audio, D]
      - stage1_hidden: Stage 1 输出的 hidden states [B, T_tok, D]
      - onset:         节拍特征 [B, T_onset, F]

    输出: 自回归预测 slide star 完整路径 tokens
    """

    def __init__(self, hidden_dim=512, num_layers=6, num_heads=8, dropout=0.1,
                 stage1_dim: int = 768, onset_dim: int = 3, max_path_len: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bos_token_id = SLD_STAR_BOS
        self.pad_token_id = SLD_STAR_PAD
        self.vocab_size = SLD_STAR_VOCAB_SIZE
        self.max_path_len = int(max_path_len)

        # ── 输入投影 ──
        self.start_embed = nn.Embedding(9, hidden_dim)           # 起始按钮 1-8 + 0
        self.tok_embed = nn.Embedding(SLD_STAR_VOCAB_SIZE, hidden_dim)
        self.pos_embed = nn.Embedding(self.max_path_len, hidden_dim)             # 最大路径长度
        self.audio_proj = nn.Linear(stage1_dim, hidden_dim)       # 全局音频
        self.stage1_proj = nn.Linear(stage1_dim, hidden_dim)      # Stage 1 hidden
        self.onset_proj = nn.Linear(onset_dim, hidden_dim)        # 节拍特征

        # ── Transformer 层 ──
        self.layers = nn.ModuleList([
            CausalPathBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # ── 输出头 ──
        self.head = nn.Linear(hidden_dim, SLD_STAR_VOCAB_SIZE)

    def _normalize_duration(self, duration):
        if isinstance(duration, tuple):
            return torch.tensor([duration], dtype=torch.float32)
        if isinstance(duration, list):
            return torch.tensor(duration, dtype=torch.float32)
        if torch.is_tensor(duration):
            duration = duration.float()
            while duration.dim() > 2 and duration.size(0) == 1:
                duration = duration.squeeze(0)
            if duration.dim() == 1:
                duration = duration.unsqueeze(0)
            return duration
        raise TypeError("duration must be tuple/list/tensor")

    def _normalize_index(self, value):
        if isinstance(value, int):
            return torch.tensor([value], dtype=torch.long)
        if torch.is_tensor(value):
            value = value.long()
            while value.dim() > 1 and value.size(0) == 1:
                value = value.squeeze(0)
            if value.dim() == 0:
                value = value.unsqueeze(0)
            return value
        raise TypeError("index must be int/tensor")

    def forward(self, target_path, start_pos, audio_memory,
                stage1_hidden=None, onset=None):
        """
        target_path:  [B, T_path]  slide star token 序列
        start_pos:    [B] 或 [B, 1]  起始按钮 1-8
        audio_memory: [B, T_audio, stage1_dim]
        stage1_hidden:[B, T_tok, stage1_dim]  可选
        onset:        [B, T_onset, onset_dim] 可选
        """
        B, T_path = target_path.shape

        # ── 输入 NaN 保护 ──
        audio_memory = self.audio_proj(audio_memory)
        if torch.isnan(audio_memory).any():
            audio_memory = torch.nan_to_num(audio_memory, nan=0.0)

        # ── 构建条件上下文: [start_embed | audio | stage1 | onset] ──
        start_pos = self._normalize_index(start_pos).to(target_path.device)
        cond_parts = [self.start_embed(start_pos)]  # [B, D] or [B, 1, D]
        if cond_parts[0].dim() == 2:
            cond_parts[0] = cond_parts[0].unsqueeze(1)  # → [B, 1, D]

        # stage1_hidden 取全局摘要（池化）
        if stage1_hidden is not None:
            s1 = self.stage1_proj(stage1_hidden.to(target_path.device))
            cond_parts.append(s1.mean(dim=1, keepdim=True))  # [B, 1, D]

        # onset 全局摘要
        if onset is not None:
            o = self.onset_proj(onset.to(target_path.device).float())
            cond_parts.append(o.mean(dim=1, keepdim=True))

        ctx = torch.cat(cond_parts + [audio_memory], dim=1)

        # ── 自回归输入/输出 ──
        inp = target_path[:, :-1]   # [BOS, ..., second-to-last]
        tgt = target_path[:, 1:]    # [dur_num, ..., EOS]
        T_in = inp.size(1)

        pos = torch.arange(T_in, device=target_path.device).unsqueeze(0).expand(B, -1)
        x = self.tok_embed(inp) + self.pos_embed(pos)

        # ── causal mask ──
        model_dtype = next(self.parameters()).dtype
        mask = torch.triu(
            torch.full((T_in, T_in), float("-inf"), device=target_path.device, dtype=model_dtype),
            diagonal=1,
        )
        for layer in self.layers:
            x = layer(x, ctx, mask)

        logits = self.head(x)
        logits = torch.clamp(logits, min=-50.0, max=50.0)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt.reshape(-1),
            ignore_index=SLD_STAR_PAD,
        )
        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def generate(self, start_pos, audio_memory, stage1_hidden=None, onset=None,
                 max_steps=16, temperature=0.8, top_k=10):
        """autoregressive 生成完整星星路径。"""
        device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype

        audio_memory = self.audio_proj(audio_memory.to(device))
        start_pos = self._normalize_index(start_pos).to(device)

        cond_parts = [self.start_embed(start_pos)]
        if stage1_hidden is not None:
            s1 = self.stage1_proj(stage1_hidden.to(device))
            cond_parts.append(s1.mean(dim=1, keepdim=True))
        if onset is not None:
            o = self.onset_proj(onset.to(device).float())
            cond_parts.append(o.mean(dim=1, keepdim=True))
        ctx = torch.cat(cond_parts + [audio_memory], dim=1)

        generated = [SLD_STAR_BOS]
        for _ in range(max_steps):
            tokens = torch.tensor([generated], device=device)
            pos = torch.arange(tokens.size(1), device=device).unsqueeze(0)
            x = self.tok_embed(tokens) + self.pos_embed(pos)

            mask = torch.triu(
                torch.full((tokens.size(1), tokens.size(1)), float("-inf"),
                           device=device, dtype=model_dtype),
                diagonal=1,
            )
            for layer in self.layers:
                x = layer(x, ctx, mask)

            logits = self.head(x[:, -1, :]) / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, -1:]] = float("-inf")

            probs = torch.softmax(logits, dim=-1)
            if torch.isnan(probs).any():
                probs = torch.ones_like(probs) / probs.size(-1)
            next_tok = torch.multinomial(probs, 1).item()
            generated.append(next_tok)
            if next_tok == SLD_STAR_EOS:
                break

        return generated


# ── 兼容旧接口 ──
def build_slide_target_path(positions: list[int], slide_types: list[str]) -> list[int]:
    """旧接口兼容：将 slide 路径转为旧格式 token 序列。"""
    from Tokenizer.MaiChartTokenizer import SLD_TO_ID, SLD_CHAR_TO_TYPE, SLD_TYPE_BASE, BOS, EOS
    result = []
    if not positions:
        return result
    for i in range(1, len(positions)):
        conn_char = slide_types[i - 1] if i - 1 < len(slide_types) else "-"
        result.append(SLD_CHAR_TO_TYPE.get(conn_char, SLD_TYPE_BASE))
        result.append(SLD_TO_ID.get(positions[i], SLD_TO_ID[1]))
    return result


def decode_slide_path(tokens: list[int]) -> tuple[list[int], list[str]]:
    """旧接口兼容：解码旧格式 token 序列。"""
    from Tokenizer.MaiChartTokenizer import (
        SLD_TYPE_BASE, SLD_TYPE_END, ID_TO_SLD, SLD_TYPE_TO_CHAR,
    )
    positions = []
    connectors = []
    for tid in tokens:
        if SLD_TYPE_BASE <= tid < SLD_TYPE_END:
            connectors.append(SLD_TYPE_TO_CHAR.get(tid, "-"))
        elif tid in ID_TO_SLD:
            positions.append(ID_TO_SLD[tid])
    return positions, connectors


slideG = SlidePathGenerator


# ═══════════════════════════════════════════════════════════════════════
# Stage 5: Slide Star 精炼器 — 细化星星排列
# ═══════════════════════════════════════════════════════════════════════

class SlideStarRefiner(nn.Module):
    """
    Stage 5: 将 Stage 2 生成的粗粒度 slide star 路径精炼为更详细的排列。

    输入:
      - coarse_path:     Stage 2 生成的 token 序列 [B, T_s2]
      - stage1_hidden:   Stage 1 hidden [B, T_tok, D]
      - audio_memory:    全局音频 [B, T_a, D]
      - onset:           节拍特征 [B, T_o, F]
      - star_mask:       [B, T_tok] bool, 标记哪里有 slide_start

    输出:
      - refined_logits:  [B, T_refined, vocab_size]  精炼后的 slide star tokens
      - 自回归生成更详细的星星路径
    """

    def __init__(self, hidden_dim=384, num_layers=4, num_heads=6, dropout=0.1,
                 stage1_dim: int = 768, onset_dim: int = 3, max_path_len: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = SLD_STAR_VOCAB_SIZE
        self.bos_token_id = SLD_STAR_BOS
        self.pad_token_id = SLD_STAR_PAD
        self.max_path_len = int(max_path_len)

        # ── 输入投影 ──
        self.coarse_embed = nn.Embedding(SLD_STAR_VOCAB_SIZE, hidden_dim)
        self.pos_embed = nn.Embedding(self.max_path_len, hidden_dim)
        self.stage1_proj = nn.Linear(stage1_dim, hidden_dim)
        self.audio_proj = nn.Linear(stage1_dim, hidden_dim)
        self.onset_proj = nn.Linear(onset_dim, hidden_dim)
        self.star_pos_embed = nn.Embedding(2, hidden_dim)  # is_star / not_star

        # ── Transformer (bidirectional for refinement) ──
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                hidden_dim, num_heads, hidden_dim * 4, dropout,
                batch_first=True, activation="gelu",
            )
            for _ in range(num_layers)
        ])

        # ── 输出头 ──
        self.head = nn.Linear(hidden_dim, SLD_STAR_VOCAB_SIZE)

    def forward(self, coarse_path, stage1_hidden, audio_memory=None, onset=None,
                star_mask=None, target_path=None):
        """
        coarse_path:   [B, T_coarse]  Stage 2 的 coarse slide path tokens
        target_path:   [B, T_target]  目标精炼 tokens（训练时使用，推理时 None）
        star_mask:     [B, T_tok]     标记 slide_start 位置
        """
        B, T_c = coarse_path.shape

        # ── 输入嵌入 ──
        pos = torch.arange(T_c, device=coarse_path.device).unsqueeze(0).expand(B, -1)
        x = self.coarse_embed(coarse_path) + self.pos_embed(pos)

        # ── 上下文 ──
        # Stage 1 hidden: 池化到固定长度
        s1 = self.stage1_proj(stage1_hidden)
        s1_pool = s1.mean(dim=1, keepdim=True).expand(B, T_c, -1)
        x = x + s1_pool

        if audio_memory is not None:
            a = self.audio_proj(audio_memory)
            a_pool = a.mean(dim=1, keepdim=True).expand(B, T_c, -1)
            x = x + a_pool

        if onset is not None:
            o = self.onset_proj(onset.float())
            if o.size(1) != T_c:
                o = o.mean(dim=1, keepdim=True).expand(B, T_c, -1)
            x = x + o

        # ── 注意力 ──
        for layer in self.layers:
            x = layer(x)

        logits = self.head(x)
        logits = torch.clamp(logits, min=-50.0, max=50.0)

        if target_path is not None:
            # 训练模式：cross-entropy
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_path.reshape(-1),
                ignore_index=SLD_STAR_PAD,
            )
            return {"logits": logits, "loss": loss}
        return {"logits": logits}

    @torch.no_grad()
    def refine(self, coarse_tokens, stage1_hidden, audio_memory=None, onset=None,
               max_steps=32, temperature=0.6):
        """自回归精炼星星路径。"""
        device = next(self.parameters()).device
        coarse = torch.tensor([coarse_tokens], device=device)
        B, T_c = coarse.shape

        pos = torch.arange(T_c, device=device).unsqueeze(0)
        x = self.coarse_embed(coarse) + self.pos_embed(pos)

        s1 = self.stage1_proj(stage1_hidden.to(device))
        x = x + s1.mean(dim=1, keepdim=True).expand(B, T_c, -1)
        if audio_memory is not None:
            a = self.audio_proj(audio_memory.to(device))
            x = x + a.mean(dim=1, keepdim=True).expand(B, T_c, -1)

        for layer in self.layers:
            x = layer(x)

        # 取最后一个位置的精炼 logits
        logits = self.head(x[:, -1:, :]) / temperature
        probs = torch.softmax(logits, dim=-1).squeeze(1)
        return probs.cpu()


starG = SlideStarRefiner

