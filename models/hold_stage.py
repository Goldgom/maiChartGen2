"""
Stage 3/4: Hold / Touch Hold 持续时间预测

Stage 3 (HoldDurationPredictor):
  输入: stage1 tokens, stage1_hidden, audio_memory, onset
  输出: 每个 hold_start slot 的持续时间 (dur_num, dur_den)

Stage 4 (TouchHoldDurationPredictor):
  同上，处理 touch hold
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════
# 持续时间词汇表（复用）
# ═══════════════════════════════════════════════════════════════════════

DUR_NUM_VALUES = [1, 2, 3, 4, 6, 8, 12, 16]
DUR_DEN_VALUES = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]

NUM_CLASSES = len(DUR_NUM_VALUES)   # 8
DEN_CLASSES = len(DUR_DEN_VALUES)   # 12


def duration_to_labels(dur: tuple[int, int] | None) -> tuple[int, int]:
    """将 hold_duration 转为分类标签 (num_idx, den_idx)。"""
    if dur is None:
        return (0, 0)
    num, den = max(1, dur[0]), max(1, dur[1])

    def _nearest(values, v):
        return min(range(len(values)), key=lambda i: abs(values[i] - v))

    return (_nearest(DUR_NUM_VALUES, num), _nearest(DUR_DEN_VALUES, den))


def labels_to_duration(num_idx: int, den_idx: int) -> tuple[int, int]:
    """分类标签 → (num, den)。"""
    return (
        DUR_NUM_VALUES[max(0, min(NUM_CLASSES - 1, num_idx))],
        DUR_DEN_VALUES[max(0, min(DEN_CLASSES - 1, den_idx))],
    )


# ═══════════════════════════════════════════════════════════════════════
# Stage 3: Hold 持续时间预测
# ═══════════════════════════════════════════════════════════════════════

class HoldDurationPredictor(nn.Module):
    """
    预测 Hold 的持续时间。

    输入:
      - tokens:         [B, T]  Stage 1 生成的 config token 序列
      - stage1_hidden:  [B, T, D]  Stage 1 hidden states
      - hold_mask:      [B, T]  bool, True=该 slot 是 hold_start
      - audio_memory:   [B, T_a, D_a]  全局音频
      - onset:          [B, T_o, F]    节拍特征 (可选)

    输出:
      - num_logits: [B, T, 8]  分子分类 logits
      - den_logits: [B, T, 12] 分母分类 logits
    """

    def __init__(self, hidden_dim: int = 384, num_layers: int = 4, num_heads: int = 6,
                 vocab_size: int = 161512, dropout: float = 0.1,
                 stage1_dim: int = 768, onset_dim: int = 3):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Embedding(16384, hidden_dim)
        self.stage1_proj = nn.Linear(stage1_dim, hidden_dim)
        self.audio_proj = nn.Linear(stage1_dim, hidden_dim)
        self.onset_proj = nn.Linear(onset_dim, hidden_dim)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                hidden_dim, num_heads, hidden_dim * 4, dropout,
                batch_first=True, activation="gelu",
            )
            for _ in range(num_layers)
        ])

        # 双头输出：分子 8 类 + 分母 12 类
        self.num_head = nn.Linear(hidden_dim, NUM_CLASSES)
        self.den_head = nn.Linear(hidden_dim, DEN_CLASSES)

        # hold_start 检测头（辅助判断哪个 slot 是 hold start）
        self.start_head = nn.Linear(hidden_dim, 1)

    def _align(self, tokens: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """对齐 stage1_hidden 与 tokens 长度。"""
        if hidden.size(1) == tokens.size(1):
            return hidden
        if hidden.size(1) < tokens.size(1):
            pad = hidden[:, -1:, :].expand(hidden.size(0), tokens.size(1) - hidden.size(1), -1)
            return torch.cat([hidden, pad], dim=1)
        return hidden[:, :tokens.size(1), :]

    def forward(self, tokens: torch.Tensor, stage1_hidden: torch.Tensor,
                audio_memory: torch.Tensor | None = None,
                onset: torch.Tensor | None = None):
        B, T = tokens.shape

        # Stage 1 hidden 上下文
        stage1_hidden = self._align(tokens, stage1_hidden)
        s1 = self.stage1_proj(stage1_hidden)

        pos = torch.arange(T, device=tokens.device).unsqueeze(0).expand(B, -1)
        x = self.token_embed(tokens) + self.pos_embed(pos) + s1

        # 全局音频池化后广播
        if audio_memory is not None:
            a = self.audio_proj(audio_memory)
            a_pool = a.mean(dim=1, keepdim=True).expand(B, T, -1)
            x = x + a_pool

        # onset
        if onset is not None:
            o = self.onset_proj(onset.float())
            o_aligned = self._align(tokens, o)
            x = x + o_aligned

        for layer in self.layers:
            x = layer(x)

        num_logits = self.num_head(x)   # [B, T, 8]
        den_logits = self.den_head(x)   # [B, T, 12]

        return {
            "num_logits": num_logits,
            "den_logits": den_logits,
            "hidden": x,
        }

    def compute_loss(self, outputs: dict, num_targets: torch.Tensor,
                     den_targets: torch.Tensor, hold_mask: torch.Tensor) -> torch.Tensor:
        """
        num_targets: [B, T]  0-7 分子类别
        den_targets: [B, T]  0-11 分母类别
        hold_mask:   [B, T]  True=需要计算 loss 的 hold_start slot
        """
        valid = hold_mask.bool()
        if valid.sum() == 0:
            return outputs["num_logits"].sum() * 0.0

        num_logits = outputs["num_logits"][valid]  # [N, 8]
        den_logits = outputs["den_logits"][valid]  # [N, 12]
        num_tgt = num_targets[valid]               # [N]
        den_tgt = den_targets[valid]               # [N]

        loss_num = F.cross_entropy(num_logits, num_tgt)
        loss_den = F.cross_entropy(den_logits, den_tgt)
        return loss_num + loss_den


# ═══════════════════════════════════════════════════════════════════════
# Stage 4: Touch Hold 持续时间预测
# ═══════════════════════════════════════════════════════════════════════

class TouchHoldDurationPredictor(HoldDurationPredictor):
    """
    预测 Touch Hold 的持续时间。
    结构与 HoldDurationPredictor 相同，但输入/输出针对 touch hold。
    """
    pass


# ═══════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════

def build_hold_targets(grid_notes: list, maxsubdiv: int = 64) -> dict[str, torch.Tensor]:
    """
    从归一化网格 notes 构建 hold 训练目标。

    Returns:
      hold_mask:    [T] bool, True=hold_start slot
      num_targets:  [T] long, 分子类别 0-7
      den_targets:  [T] long, 分母类别 0-11
    """
    T = len(grid_notes)
    hold_mask = torch.zeros(T, dtype=torch.bool)
    num_targets = torch.zeros(T, dtype=torch.long)
    den_targets = torch.zeros(T, dtype=torch.long)

    i = 0
    while i < T:
        note = grid_notes[i]
        if note.is_hold and note.hold_duration and not getattr(note, '_hold_consumed', False):
            # Hold start
            hold_mask[i] = True
            num_idx, den_idx = duration_to_labels(note.hold_duration)
            num_targets[i] = num_idx
            den_targets[i] = den_idx

            # 标记后续持续 slot
            j = i + 1
            while j < T and grid_notes[j].is_hold:
                grid_notes[j]._hold_consumed = True
                j += 1
            i = j
        else:
            i += 1

    return {"hold_mask": hold_mask, "num_targets": num_targets, "den_targets": den_targets}


def build_touch_hold_targets(grid_notes: list, maxsubdiv: int = 64) -> dict[str, torch.Tensor]:
    """
    从归一化网格 notes 构建 touch hold 训练目标。
    类似 build_hold_targets，但针对 touch hold。
    """
    T = len(grid_notes)
    hold_mask = torch.zeros(T, dtype=torch.bool)
    num_targets = torch.zeros(T, dtype=torch.long)
    den_targets = torch.zeros(T, dtype=torch.long)

    i = 0
    while i < T:
        note = grid_notes[i]
        is_th = getattr(note, 'is_touch_hold', False) or (note.is_touch and note.is_hold)
        if is_th and note.hold_duration and not getattr(note, '_thold_consumed', False):
            hold_mask[i] = True
            num_idx, den_idx = duration_to_labels(note.hold_duration)
            num_targets[i] = num_idx
            den_targets[i] = den_idx

            j = i + 1
            while j < T and (getattr(grid_notes[j], 'is_touch_hold', False) or grid_notes[j].is_touch):
                grid_notes[j]._thold_consumed = True
                j += 1
            i = j
        else:
            i += 1

    return {"hold_mask": hold_mask, "num_targets": num_targets, "den_targets": den_targets}


holdG = HoldDurationPredictor
touchHoldG = TouchHoldDurationPredictor
