import math

import torch
import torch.nn as nn

from utils.rope import Rope


class FeedForward(nn.Module):
    """Transformer 前馈网络 (FFN)。

    标准设计：Linear → GELU → Dropout → Linear → Dropout，
    中间维度 = dmodel × expansion (默认 4 倍扩展)。
    """

    def __init__(
        self,
        dmodel: int = 512,
        expansion: int = 4,
        dropout: float = 0.1,
        precision: torch.dtype = torch.float32,
    ):
        super().__init__()
        hidden_dim = dmodel * expansion
        self.linear1 = nn.Linear(dmodel, hidden_dim, dtype=precision)
        self.linear2 = nn.Linear(hidden_dim, dmodel, dtype=precision)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class MHAttention(nn.Module):
    """多头自注意力 + FFN 的完整 Transformer Block (Pre-LN 结构)。

    结构:
        x = x + MHA(LayerNorm(x))    ← 自注意力 + 残差
        x = x + FFN(LayerNorm(x))    ← 前馈网络 + 残差

    Args:
        q_dim:    Query 输入维度。
        k_dim:    Key   输入维度。
        v_dim:    Value 输入维度。
        dmodel:   模型总维度 (必须能被 heads 整除)。
        heads:    注意力头数。
        ffn_expansion: FFN 隐藏层扩展倍数 (默认 4)。
        precision: 线性层计算精度。
        atten_mask: 可选的注意力掩码生成函数。
        dropout:   Dropout 概率。
    """

    def __init__(
        self,
        q_dim: int = 512,
        k_dim: int = 512,
        v_dim: int = 512,
        dmodel: int = 512,
        precision: torch.dtype = torch.float32,
        heads: int = 8,
        ffn_expansion: int = 4,
        atten_mask=None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if dmodel % heads != 0:
            raise ValueError("dmodel must be divisible by heads.")

        # ---- 自注意力子层 ----
        self.q_proj = nn.Linear(q_dim, dmodel, dtype=precision)
        self.k_proj = nn.Linear(k_dim, dmodel, dtype=precision)
        self.v_proj = nn.Linear(v_dim, dmodel, dtype=precision)
        self.output_projection = nn.Linear(dmodel, dmodel, dtype=precision)
        self.heads = heads
        self.dmodel = dmodel
        self.atten_mask = atten_mask
        self.rope = Rope(dmodel=dmodel // heads, precision=precision)
        self.attention_dropout = nn.Dropout(dropout)

        # ---- FFN 子层 ----
        self.ffn = FeedForward(dmodel=dmodel, expansion=ffn_expansion, dropout=dropout, precision=precision)

        # ---- LayerNorm (Pre-LN) ----
        self.norm1 = nn.LayerNorm(dmodel, dtype=precision)
        self.norm2 = nn.LayerNorm(dmodel, dtype=precision)

        self.output_dropout = nn.Dropout(dropout)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """前向传播 (Pre-LN Transformer Block)。

        Args:
            Q: Query  [B, seq_len, q_dim]。
            K: Key    [B, seq_len, k_dim]。
            V: Value  [B, seq_len, v_dim]。

        Returns:
            变换后的序列 [B, seq_len, dmodel]。
        """
        # ============ 1. 自注意力子层 (Pre-LN + 残差) ============
        residual = Q
        Q_norm = self.norm1(Q)
        K_norm = self.norm1(K)
        V_norm = self.norm1(V)

        batch_size = Q_norm.size(0)
        head_dim = self.dmodel // self.heads

        Q_proj = self.q_proj(Q_norm).view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        K_proj = self.k_proj(K_norm).view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        V_proj = self.v_proj(V_norm).view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        Q_proj = self.rope(Q_proj)
        K_proj = self.rope(K_proj)

        scores = torch.matmul(Q_proj, K_proj.transpose(-2, -1)) / math.sqrt(head_dim)
        if self.atten_mask is not None:
            mask = self.atten_mask(scores.size(-1)).to(scores.device)
            scores = scores.masked_fill(~mask, float("-inf"))

        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)

        attn_out = torch.matmul(attention_weights, V_proj)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, -1, self.dmodel)
        attn_out = self.output_projection(attn_out)
        attn_out = self.output_dropout(attn_out)

        x = residual + attn_out                    # 残差连接

        # ============ 2. FFN 子层 (Pre-LN + 残差) ============
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x                           # 残差连接

        return x
