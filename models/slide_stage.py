from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.chart_blocks import CausalPathBlock
from Tokenizer.MaiChartTokenizer import BOS, PAD, SLD_TYPE_BASE, SLD_TYPE_END, SLD_TO_ID, SLD_CHAR_TO_TYPE, ID_TO_SLD, SLD_TYPE_TO_CHAR


class SlidePathGenerator(nn.Module):
    def __init__(self, hidden_dim=512, num_layers=6, num_heads=8, dropout=0.1, stage1_dim: int = 768):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bos_token_id = BOS
        self.pad_token_id = PAD
        self.btn_embed = nn.Embedding(9, hidden_dim)
        self.tok_embed = nn.Embedding(max(SLD_TYPE_END, SLD_TO_ID[8]) + 1, hidden_dim)
        self.pos_embed = nn.Embedding(32, hidden_dim)
        self.dur_proj = nn.Sequential(nn.Linear(2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.audio_proj = nn.Linear(stage1_dim, hidden_dim)  # 复用 Stage 1 音频
        self.layers = nn.ModuleList([CausalPathBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)])
        self.head = nn.Linear(hidden_dim, max(SLD_TYPE_END, SLD_TO_ID[8]) + 1)

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

    def forward(self, target_path, start_pos, end_pos, duration, audio_memory):
        B, T_path = target_path.shape
        audio_memory = self.audio_proj(audio_memory)  # [B, T_audio, stage1_dim] → [B, T_audio, hidden_dim]

        # ── 输入 NaN 检查（防止上游数据污染）──
        if torch.isnan(audio_memory).any():
            audio_memory = torch.nan_to_num(audio_memory, nan=0.0)

        start_pos = self._normalize_index(start_pos).to(target_path.device)
        end_pos = self._normalize_index(end_pos).to(target_path.device)
        duration = self._normalize_duration(duration).to(target_path.device)
        cond = self.btn_embed(start_pos) + self.btn_embed(end_pos) + self.dur_proj(duration)
        if cond.dim() == 2:
            cond = cond.unsqueeze(1)
        ctx = torch.cat([cond, audio_memory], dim=1)

        inp = target_path[:, :-1]
        tgt = target_path[:, 1:]
        T_in = inp.size(1)
        pos = torch.arange(T_in, device=target_path.device).unsqueeze(0).expand(B, -1)
        x = self.tok_embed(inp) + self.pos_embed(pos)

        # mask dtype 跟随模型精度，不再写死 float16
        model_dtype = next(self.parameters()).dtype
        mask = torch.triu(
            torch.full((T_in, T_in), float("-inf"), device=target_path.device, dtype=model_dtype),
            diagonal=1,
        )
        for layer in self.layers:
            x = layer(x, ctx, mask)

        logits = self.head(x)

        # ── logits clamp：防止 cross_entropy 因极端值产生 inf ──
        logits = torch.clamp(logits, min=-50.0, max=50.0)

        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=PAD)
        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def generate(self, audio_memory, start_pos, end_pos, duration, max_steps=8, temperature=0.8, top_k=10):
        device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype
        audio_memory = self.audio_proj(audio_memory.to(device))
        if isinstance(start_pos, int):
            start_pos = torch.tensor([start_pos], device=device)
        if isinstance(end_pos, int):
            end_pos = torch.tensor([end_pos], device=device)
        duration = self._normalize_duration(duration).to(device)

        start_pos = self._normalize_index(start_pos).to(device)
        end_pos = self._normalize_index(end_pos).to(device)
        cond = self.btn_embed(start_pos) + self.btn_embed(end_pos) + self.dur_proj(duration.float())
        if cond.dim() == 2:
            cond = cond.unsqueeze(1)
        ctx = torch.cat([cond, audio_memory], dim=1)

        generated = []
        for _ in range(max_steps):
            tokens = torch.tensor([generated], device=device) if generated else torch.empty(1, 0, dtype=torch.long, device=device)
            if tokens.numel() == 0:
                x = self.tok_embed.weight.new_zeros(1, 1, self.hidden_dim)
            else:
                pos = torch.arange(tokens.size(1), device=device).unsqueeze(0)
                x = self.tok_embed(tokens) + self.pos_embed(pos)
            mask = None if tokens.numel() == 0 else torch.triu(
                torch.full((tokens.size(1), tokens.size(1)), float("-inf"), device=device, dtype=model_dtype),
                diagonal=1,
            )
            for layer in self.layers:
                x = layer(x, ctx, mask)
            logits = self.head(x[:, -1, :]) / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, -1:]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            # ── NaN 保护 ──
            if torch.isnan(probs).any():
                probs = torch.ones_like(probs) / probs.size(-1)
            next_tok = torch.multinomial(probs, 1).item()
            generated.append(next_tok)
            if len(generated) % 2 == 0 and next_tok == int(end_pos.item()):
                break
        return generated


def build_slide_target_path(positions: list[int], slide_types: list[str]) -> list[int]:
    result = []
    if not positions:
        return result
    for i in range(1, len(positions)):
        conn_char = slide_types[i - 1] if i - 1 < len(slide_types) else "-"
        result.append(SLD_CHAR_TO_TYPE.get(conn_char, SLD_TYPE_BASE))
        result.append(SLD_TO_ID.get(positions[i], SLD_TO_ID[1]))
    return result


def decode_slide_path(tokens: list[int]) -> tuple[list[int], list[str]]:
    positions = []
    connectors = []
    for tid in tokens:
        if SLD_TYPE_BASE <= tid < SLD_TYPE_END:
            connectors.append(SLD_TYPE_TO_CHAR.get(tid, "-"))
        elif tid in ID_TO_SLD:
            positions.append(ID_TO_SLD[tid])
    return positions, connectors


slideG = SlidePathGenerator
