"""
models/common.py — 所有 Stage 共享的组件

- StageConfig: 模型超参配置
- FastAHPE: 累积 Householder 位置编码 (快速向量化)
- AudioEncoder: EnCodec token → 连续嵌入
- ConditionEmbedding: 节拍/难度/等级/标签 → 条件向量
- TransformerBlock: Pre-LN Transformer + GELU
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# StageConfig
# ============================================================

@dataclass
class StageConfig:
    d_model: int = 512
    n_head: int = 8
    n_layer: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 8192
    ahpe_householder_order: int = 2
    ahpe_learnable: bool = True
    audio_codebook_size: int = 1024
    audio_num_codebooks: int = 8
    beat_dim: int = 2
    difficulty_dim: int = 16
    level_dim: int = 16
    tag_dim: int = 64
    max_tags: int = 32
    tag_vocab_size: int = 256
    chart_vocab_size: int = 512
    slide_vocab_size: int = 256
    hold_dur_bins: int = 64
    max_hold_slots: int = 8
    max_slide_slots: int = 8
    max_object_slots: int = 16
    init_std: float = 0.02

    def __post_init__(self):
        assert self.d_model % self.n_head == 0
        assert self.d_model % (2 * self.ahpe_householder_order) == 0


# ============================================================
# 初始化工具
# ============================================================

def _init_weights(module: nn.Module, std: float):
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, std=std)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.Conv1d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.MultiheadAttention):
        for p in module.parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif p.dim() == 1:
                nn.init.zeros_(p)


# ============================================================
# FastAHPE — 快速累积 Householder 位置编码
# ============================================================

class FastAHPE(nn.Module):
    """累积 Householder 位置编码 (向量化快速版)

    Householder反射: H_u(v) = v - 2<u,v>u
    位置 t 的编码 = scale * H_{u_t} ∘ ... ∘ H_{u_1} (base)

    相比正弦PE的优势: 可学习、可表达更丰富的位置关系
    """

    def __init__(self, d_model: int, max_len: int = 8192,
                 order: int = 2, learnable: bool = True):
        super().__init__()
        self.d_model = d_model
        self.order = order
        self.d_sub = d_model // order

        self.base = nn.Parameter(
            torch.randn(order, self.d_sub) * 0.02,
            requires_grad=learnable,
        )
        self.W = nn.Parameter(
            torch.randn(order, self.d_sub, 2) * 0.02,
            requires_grad=learnable,
        )
        self.b = nn.Parameter(
            torch.zeros(order, self.d_sub),
            requires_grad=learnable,
        )
        self.scale = nn.Parameter(torch.ones(1), requires_grad=learnable)
        self._cache: dict = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        if self.training:
            pe = self._compute_train(T)
        else:
            device_key = x.device.index if x.device.type == "cuda" else 0
            key = (T, device_key)
            if key not in self._cache:
                with torch.no_grad():
                    self._cache[key] = self._compute_train(T).detach()
            pe = self._cache[key]
        return x + self.scale * pe.unsqueeze(0)

    def _compute_train(self, T: int) -> torch.Tensor:
        t = torch.arange(T, dtype=torch.float, device=self.base.device)
        phase = 2.0 * math.pi * t / max(T, 1.0)
        c = phase.cos().view(T, 1)
        s = phase.sin().view(T, 1)
        Wc = self.W[:, :, 0]
        Ws = self.W[:, :, 1]
        u = Wc.unsqueeze(0) * c.unsqueeze(-1) + Ws.unsqueeze(0) * s.unsqueeze(-1)
        u = u + self.b.unsqueeze(0)
        u = F.normalize(u, p=2, dim=-1)

        # 非 in-place 累积: 用列表 + stack
        v_list = [self.base.unsqueeze(0)]  # (1, order, d_sub)
        for i in range(1, T):
            ui = u[i:i+1]
            v_prev = v_list[-1]
            dot = (ui * v_prev).sum(dim=-1, keepdim=True)
            v_i = v_prev - 2.0 * dot * ui
            v_list.append(v_i)
        v = torch.cat(v_list, dim=0)  # (T, order, d_sub)
        return v.reshape(T, self.d_model)

    def clear_cache(self):
        self._cache.clear()


AccumulatingHouseholderPE = FastAHPE


# ============================================================
# AudioEncoder
# ============================================================

class AudioEncoder(nn.Module):
    def __init__(self, cfg: StageConfig):
        super().__init__()
        self.num_codebooks = cfg.audio_num_codebooks
        self.embeddings = nn.ModuleList([
            nn.Embedding(cfg.audio_codebook_size, cfg.d_model)
            for _ in range(cfg.audio_num_codebooks)
        ])
        self.conv_in = nn.Conv1d(cfg.d_model, cfg.d_model, 3, padding=1)
        self.conv_out = nn.Conv1d(cfg.d_model, cfg.d_model, 3, padding=1)
        self.pe = FastAHPE(cfg.d_model, cfg.max_seq_len,
                           order=cfg.ahpe_householder_order,
                           learnable=cfg.ahpe_learnable)
        self.ln = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.apply(lambda m: _init_weights(m, cfg.init_std))

    def forward(self, audio_tokens: torch.Tensor) -> torch.Tensor:
        B, T, C = audio_tokens.shape
        x = sum(
            self.embeddings[c](audio_tokens[:, :, c].long())
            for c in range(min(C, self.num_codebooks))
        )
        x = x.transpose(1, 2)
        x = F.gelu(self.conv_in(x))
        x = F.gelu(self.conv_out(x))
        x = x.transpose(1, 2)
        x = self.pe(x)
        x = self.ln(x)
        return self.dropout(x)


# ============================================================
# ConditionEmbedding
# ============================================================

class ConditionEmbedding(nn.Module):
    def __init__(self, cfg: StageConfig):
        super().__init__()
        self.beat_proj = nn.Sequential(
            nn.Linear(cfg.beat_dim, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.diff_linear = nn.Linear(1, cfg.difficulty_dim)
        self.lvl_linear = nn.Linear(1, cfg.level_dim)
        self.tag_embed = nn.Embedding(cfg.tag_vocab_size, cfg.tag_dim)
        merge_in = cfg.d_model + cfg.difficulty_dim + cfg.level_dim + cfg.tag_dim
        self.merge = nn.Sequential(
            nn.Linear(merge_in, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.ln = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.apply(lambda m: _init_weights(m, cfg.init_std))

    def forward(self, beat, difficulty, level, tags):
        B, T, _ = beat.shape
        b = self.beat_proj(beat)
        diff = difficulty.float().clamp(0, 7).unsqueeze(-1)
        d = self.diff_linear(diff).unsqueeze(1).expand(-1, T, -1)
        lvl = level.float().unsqueeze(-1)
        l = self.lvl_linear(lvl).unsqueeze(1).expand(-1, T, -1)
        t_emb = self.tag_embed(tags.clamp(0, 255))
        mask = (tags >= 0).float().unsqueeze(-1)
        t_pool = (t_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        t_pool = t_pool.unsqueeze(1).expand(-1, T, -1)
        x = torch.cat([b, d, l, t_pool], dim=-1)
        x = self.merge(x)
        return self.dropout(self.ln(x))


# ============================================================
# TransformerBlock (Pre-LN + GELU)
# ============================================================

class TransformerBlock(nn.Module):
    def __init__(self, cfg: StageConfig, cross_attn: bool = False):
        super().__init__()
        self.cross_attn = cross_attn
        d = cfg.d_model
        self.ln1 = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(
            d, cfg.n_head, dropout=cfg.dropout, batch_first=True)
        self.drop_attn = nn.Dropout(cfg.dropout)
        if cross_attn:
            self.ln_cross = nn.LayerNorm(d)
            self.cross_attn_layer = nn.MultiheadAttention(
                d, cfg.n_head, dropout=cfg.dropout, batch_first=True)
            self.drop_cross = nn.Dropout(cfg.dropout)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, cfg.d_ff), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, d))
        self.drop_ff = nn.Dropout(cfg.dropout)
        self.apply(lambda m: _init_weights(m, cfg.init_std))

    def forward(self, x, memory=None, causal_mask=None):
        r = x; x = self.ln1(x)
        x = self.self_attn(x, x, x, attn_mask=causal_mask, need_weights=False)[0]
        x = self.drop_attn(x) + r
        if self.cross_attn and memory is not None:
            r = x; x = self.ln_cross(x)
            x = self.cross_attn_layer(x, memory, memory, need_weights=False)[0]
            x = self.drop_cross(x) + r
        r = x; x = self.ln2(x); x = self.ff(x); x = self.drop_ff(x) + r
        return x


def build_causal_mask(T: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(T, T, device=device) * float("-inf"), diagonal=1)


def make_model(cfg: StageConfig, n_layer: int, cross_attn: bool = False):
    return nn.ModuleList([
        TransformerBlock(cfg, cross_attn=cross_attn) for _ in range(n_layer)
    ])
