from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from modules.audio_encoder import DualStreamAudioEncoder
from modules.chart_blocks import CausalDecoderBlock
from Tokenizer.MaiChartTokenizer import EOS, SLD_BEG_BASE, SLD_BEG_END, SLD_END_POS_BASE, SLD_END_POS_END, DUR, ID_TO_DUR_NUM, ID_TO_DUR_DEN


class BeatPositionEncoding(nn.Module):
    def __init__(self, hidden_dim: int, subdiv: int = 64, beats_per_bar: int = 4, max_bars: int = 512):
        super().__init__()
        d = hidden_dim // 4
        self.subdiv = subdiv
        self.beats_per_bar = beats_per_bar
        self.bar_embed = nn.Embedding(max_bars, d)
        self.beat_embed = nn.Embedding(beats_per_bar, d)
        self.sub_embed = nn.Embedding(subdiv, d)
        self.global_pos = nn.Embedding(16384, hidden_dim - d * 3)  # 5min兼容

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        bar = positions // (self.beats_per_bar * self.subdiv)
        sib = positions % (self.beats_per_bar * self.subdiv)
        beat = sib // self.subdiv
        sub = sib % self.subdiv
        bar = bar.clamp(0, self.bar_embed.num_embeddings - 1)
        return torch.cat([
            self.bar_embed(bar),
            self.beat_embed(beat),
            self.sub_embed(sub),
            self.global_pos(positions.clamp(0, self.global_pos.num_embeddings - 1)),
        ], dim=-1)


class RelativeTimingEncoding(nn.Module):
    def __init__(self, hidden_dim: int, max_dist: int = 256):
        super().__init__()
        d = hidden_dim // 4
        self.max_dist = max_dist
        self.press_embed = nn.Embedding(max_dist + 1, d)
        self.hold_embed = nn.Embedding(max_dist + 1, d)
        self.slide_embed = nn.Embedding(max_dist + 1, d)
        self.touch_embed = nn.Embedding(max_dist + 1, d)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        d = distances.clamp(0, self.max_dist)
        return torch.cat([
            self.press_embed(d[..., 0]),
            self.hold_embed(d[..., 1]),
            self.slide_embed(d[..., 2]),
            self.touch_embed(d[..., 3]),
        ], dim=-1)


def compute_relative_distances(tokens: torch.Tensor) -> torch.Tensor:
    """矢量化计算相对节奏间距: [B, T, 4] — press, hold, slide, touch。"""
    from Tokenizer.MaiChartTokenizer import (
        TAP_BASE, TAP_END, BRK_BASE, BRK_END,
        HLD_BASE, HLD_END, SLD_BASE, SLD_END_TOKEN_BASE,
        SLD_BEG, SLD_END, SLD_BEG_BASE, SLD_BEG_END,
        SLD_TYPE_BASE, SLD_TYPE_END, SLD_END_POS_BASE, SLD_END_POS_END,
        WIFI_SLIDE, TCH_BASE, TCH_END, CONFIG_BASE,
    )
    from Tokenizer.config_vocab import ID_TO_CONFIG, BTN_PRESS, BTN_HOLD_START, BTN_SLIDE_START, TCH_TOUCH, TCH_HOLD_START

    B, T = tokens.shape
    device = tokens.device
    pos = torch.arange(T, device=device).unsqueeze(0)  # [1, T]

    # ── 构建 4 种事件的布尔掩码 (矢量化) ──
    is_press = (tokens >= TAP_BASE) & (tokens < TAP_END)
    is_press |= (tokens >= BRK_BASE) & (tokens < BRK_END)
    is_hold  = (tokens >= HLD_BASE) & (tokens < HLD_END)
    is_slide = (tokens >= SLD_BASE) & (tokens < SLD_END_TOKEN_BASE)
    is_slide |= (tokens == SLD_BEG) | (tokens == SLD_END) | (tokens == WIFI_SLIDE)
    is_slide |= (tokens >= SLD_TYPE_BASE) & (tokens < SLD_TYPE_END)
    is_slide |= (tokens >= SLD_BEG_BASE) & (tokens < SLD_BEG_END)
    is_slide |= (tokens >= SLD_END_POS_BASE) & (tokens < SLD_END_POS_END)
    is_touch = (tokens >= TCH_BASE) & (tokens < TCH_END)

    # Config tokens 需查表 — 少量 Python 循环
    is_config = tokens >= CONFIG_BASE
    if is_config.any():
        for b, t in is_config.nonzero(as_tuple=False).tolist():
            sc = ID_TO_CONFIG.get(tokens[b, t].item())
            if sc:
                for _, s in sc.buttons:
                    if s == BTN_PRESS:        is_press[b, t] = True
                    elif s == BTN_HOLD_START:  is_hold[b, t] = True
                    elif s == BTN_SLIDE_START: is_slide[b, t] = True
                for _, s in sc.touches:
                    if s in (TCH_TOUCH, TCH_HOLD_START):
                        is_touch[b, t] = True

    # ── 矢量化距离: cummax 传播最后出现位置 ──
    distances = torch.zeros(B, T, 4, dtype=torch.long, device=device)
    for i, mask in enumerate([is_press, is_hold, is_slide, is_touch]):
        shifted = torch.zeros_like(mask)
        shifted[:, 1:] = mask[:, :-1]                        # 右移, 当前步不计
        event_pos = pos.masked_fill(~shifted, -1)            # 出现位置=-1, 其余=pos
        last_pos, _ = event_pos.cummax(dim=1)                # 向右传播最大值
        distances[:, :, i] = (pos - last_pos).clamp(min=0)   # 距离

    return distances


def _track_event_pos(tid: int, pos: int, lp: int, lh: int, ls: int, lt: int):
    """增量更新事件追踪 — generate() O(1) 距离计算。"""
    from Tokenizer.MaiChartTokenizer import (
        TAP_BASE, TAP_END, BRK_BASE, BRK_END, HLD_BASE, HLD_END,
        SLD_BASE, SLD_END_TOKEN_BASE, SLD_BEG, SLD_END, SLD_BEG_BASE,
        SLD_BEG_END, SLD_TYPE_BASE, SLD_TYPE_END, SLD_END_POS_BASE,
        SLD_END_POS_END, WIFI_SLIDE, TCH_BASE, TCH_END, CONFIG_BASE,
    )
    from Tokenizer.config_vocab import ID_TO_CONFIG, BTN_PRESS, BTN_HOLD_START, BTN_SLIDE_START, TCH_TOUCH, TCH_HOLD_START
    if TAP_BASE <= tid < TAP_END or BRK_BASE <= tid < BRK_END:  lp = pos
    elif HLD_BASE <= tid < HLD_END:                              lh = pos
    elif (SLD_BASE <= tid < SLD_END_TOKEN_BASE or tid in (SLD_BEG, SLD_END, WIFI_SLIDE)
          or SLD_TYPE_BASE <= tid < SLD_TYPE_END or SLD_BEG_BASE <= tid < SLD_BEG_END
          or SLD_END_POS_BASE <= tid < SLD_END_POS_END):         ls = pos
    elif TCH_BASE <= tid < TCH_END:                              lt = pos
    elif tid >= CONFIG_BASE:
        sc = ID_TO_CONFIG.get(tid)
        if sc:
            for _, s in sc.buttons:
                if s == BTN_PRESS: lp = pos
                elif s == BTN_HOLD_START: lh = pos
                elif s == BTN_SLIDE_START: ls = pos
            for _, s in sc.touches:
                if s in (TCH_TOUCH, TCH_HOLD_START): lt = pos
    return lp, lh, ls, lt


def _chunked_ce(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int, chunk_size: int = 256) -> torch.Tensor:
    """分块计算 CrossEntropy，避免 vocab=161K 时 OOM。"""
    B, T, V = logits.shape
    total_loss = 0.0
    total_tokens = 0
    logits_flat = logits.reshape(-1, V)   # [B*T, V]
    targets_flat = targets.reshape(-1)     # [B*T]

    for start in range(0, logits_flat.size(0), chunk_size):
        end = min(start + chunk_size, logits_flat.size(0))
        chunk_logits = logits_flat[start:end]
        chunk_targets = targets_flat[start:end]
        mask = chunk_targets != ignore_index
        if mask.sum() == 0:
            continue
        chunk_loss = F.cross_entropy(chunk_logits, chunk_targets, ignore_index=ignore_index, reduction="sum")
        total_loss += chunk_loss
        total_tokens += mask.sum().item()

    if total_tokens == 0:
        return logits.sum() * 0.0
    return total_loss / total_tokens


def _chunked_lm_ce(
    lm_head: nn.Linear, x: torch.Tensor, targets: torch.Tensor,
    ignore_index: int, chunk_size: int = 256, loss_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """分块计算 lm_head + CrossEntropy，完全避免 [B*T, V] 大 tensor 驻留。

    每块: lm_head(x[chunk]) → CE → 累积 loss，用完即释放。
    """
    B, T, D = x.shape
    x_flat = x.reshape(-1, D).contiguous()
    tgt_flat = targets.reshape(-1).contiguous()
    if loss_mask is not None:
        mask_flat = loss_mask.reshape(-1).bool()
        tgt_flat = tgt_flat.masked_fill(~mask_flat, ignore_index)

    total_loss = 0.0
    total_tokens = 0
    for start in range(0, x_flat.size(0), chunk_size):
        end = min(start + chunk_size, x_flat.size(0))
        chunk_x = x_flat[start:end]
        chunk_tgt = tgt_flat[start:end]
        chunk_logits = lm_head(chunk_x)  # [chunk, V] — 用完即释放
        mask = chunk_tgt != ignore_index
        if mask.sum() == 0:
            del chunk_logits
            continue
        chunk_loss = F.cross_entropy(chunk_logits, chunk_tgt, ignore_index=ignore_index, reduction="sum")
        total_loss += chunk_loss
        total_tokens += mask.sum().item()
        del chunk_logits

    if total_tokens == 0:
        return x.sum() * 0.0
    return total_loss / total_tokens


class MaiGenerator(nn.Module):
    def __init__(
        self,
        hidden_dim=768,
        num_layers=12,
        num_heads=12,
        vocab_size=161512,
        subdiv=64,
        beats_per_bar=4,
        dropout=0.1,
        audio_stream_layers: int = 4,
        audio_stream_heads: int = 12,
        use_checkpoint: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.subdiv = subdiv
        self.vocab_size = vocab_size
        self.use_checkpoint = use_checkpoint
        self.use_sliding_window = bool(kwargs.get("use_sliding_window", False))
        self.window_tokens = int(kwargs.get("window_tokens", 4096))
        self.window_stride = int(kwargs.get("window_stride", max(1, self.window_tokens // 2)))
        self.max_summary_tokens = int(kwargs.get("max_summary_tokens", 16))
        self.summary_position = str(kwargs.get("summary_position", "prefix"))
        self.detach_summary = bool(kwargs.get("detach_summary", True))
        self.audio_window_tokens = int(kwargs.get("audio_window_tokens", self.window_tokens))
        self.audio_global_summary_tokens = int(kwargs.get("audio_global_summary_tokens", 16))
        self.bos_token_id = 1
        self.eos_token_id = EOS
        self.pad_token_id = 0

        self.pos_embed = BeatPositionEncoding(hidden_dim, subdiv, beats_per_bar)
        self.timing_embed = RelativeTimingEncoding(hidden_dim)
        self.audio = DualStreamAudioEncoder(
            hidden_dim=hidden_dim,
            audio_vocab_size=kwargs.get("audio_vocab_size", 8195),
            stream_layers=audio_stream_layers,
            stream_heads=audio_stream_heads,
            global_stride=kwargs.get("global_stride", 8),
            local_window_s=kwargs.get("local_window_s", 5.0),
            local_slots_per_sec=kwargs.get("local_slots_per_sec", 184),
            local_dilation_base=kwargs.get("local_dilation_base", 4),
            max_spectral_len=kwargs.get("max_spectral_len", 16384),
            use_spectral_sliding_window=kwargs.get("use_spectral_sliding_window", False),
            spectral_window_len=kwargs.get("spectral_window_len", 4096),
            spectral_window_stride=kwargs.get("spectral_window_stride", 2048),
        )
        self.cond_embed = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.chart_summary_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.audio_summary_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.raw_audio_summary_proj = nn.Sequential(
            nn.Linear(14, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.summary_type_embed = nn.Parameter(torch.zeros(2, hidden_dim))
        self.layers = nn.ModuleList([CausalDecoderBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)])
        self.lm_head = nn.Linear(hidden_dim, vocab_size)
        self._init_weights()
        nn.init.normal_(self.summary_type_embed, std=0.02)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def _align_audio_memory(self, audio_memory: torch.Tensor, target_len: int, batch_size: int) -> torch.Tensor:
        if audio_memory.size(1) < target_len:
            pad = audio_memory[:, -1:, :].expand(batch_size, target_len - audio_memory.size(1), -1)
            audio_memory = torch.cat([audio_memory, pad], dim=1)
        return audio_memory[:, :target_len, :]

    def _pool_summary(self, x: torch.Tensor, proj: nn.Module, max_tokens: int, type_idx: int) -> torch.Tensor:
        if x.size(1) == 0:
            return x[:, :0, :]
        chunks = min(max(1, max_tokens), x.size(1))
        pooled = F.adaptive_avg_pool1d(x.transpose(1, 2), chunks).transpose(1, 2)
        return proj(pooled) + self.summary_type_embed[type_idx].view(1, 1, -1)

    def _run_decoder(
        self,
        x: torch.Tensor,
        audio_memory: torch.Tensor,
        cond: torch.Tensor,
        token_len: int | None = None,
    ) -> torch.Tensor:
        total_len = x.size(1)
        mask = torch.triu(torch.ones((total_len, total_len), device=x.device, dtype=torch.bool), diagonal=1)
        if token_len is not None and token_len < total_len:
            if self.summary_position == "suffix":
                mask[:token_len, token_len:] = True
            else:
                summary_len = total_len - token_len
                mask[:summary_len, summary_len:] = True
        for layer in self.layers:
            if self.training and self.use_checkpoint:
                x = checkpoint(layer, x, audio_memory, cond, mask, use_reentrant=False)
            else:
                x = layer(x, audio_memory, cond, mask)
        return x

    def _window_starts(self, total_len: int) -> list[int]:
        window = max(1, self.window_tokens)
        stride = max(1, min(self.window_stride, window))
        if total_len <= window:
            return [0]
        starts = list(range(0, max(1, total_len - window + 1), stride))
        tail_start = max(0, total_len - window)
        if starts[-1] != tail_start:
            starts.append(tail_start)
        return sorted(set(starts))

    def _audio_window_memory(
        self,
        audio_memory: torch.Tensor,
        start: int,
        end: int,
        audio_summary: torch.Tensor | None,
    ) -> torch.Tensor:
        local_start = max(0, start)
        local_end = min(audio_memory.size(1), max(end, start + 1))
        local = audio_memory[:, local_start:local_end, :]
        if local.size(1) == 0:
            local = audio_memory[:, -1:, :]
        if self.audio_window_tokens > 0 and local.size(1) > self.audio_window_tokens:
            local = local[:, :self.audio_window_tokens, :]
        if audio_summary is not None and audio_summary.size(1) > 0:
            return torch.cat([audio_summary, local], dim=1)
        return local

    def _raw_audio_summary(
        self,
        onset: torch.Tensor,
        chroma: torch.Tensor,
        centroid: torch.Tensor,
    ) -> torch.Tensor:
        spectral = torch.cat([onset.unsqueeze(-1), chroma, centroid.unsqueeze(-1)], dim=-1)
        chunks = max(1, self.audio_global_summary_tokens)
        pooled = F.adaptive_avg_pool1d(spectral.transpose(1, 2), chunks).transpose(1, 2)
        return self.raw_audio_summary_proj(pooled) + self.summary_type_embed[1].view(1, 1, -1)

    def _slice_audio_features(
        self,
        onset: torch.Tensor,
        chroma: torch.Tensor,
        centroid: torch.Tensor,
        start: int,
        end: int,
        token_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        audio_len = onset.size(1)
        if audio_len == 0:
            return onset, chroma, centroid
        ratio = audio_len / max(1, token_len)
        s = min(max(0, int(start * ratio)), audio_len - 1)
        e = min(max(s + 1, int(end * ratio)), audio_len)
        return onset[:, s:e], chroma[:, s:e, :], centroid[:, s:e]

    def _slice_audio_tokens(
        self,
        audio_tokens: torch.Tensor | None,
        start: int,
        end: int,
        token_len: int,
    ) -> torch.Tensor | None:
        if audio_tokens is None or audio_tokens.numel() == 0:
            return audio_tokens
        if audio_tokens.dim() == 1:
            audio_tokens = audio_tokens.unsqueeze(0)
        if token_len <= 0:
            return audio_tokens[:, :0]
        ratio = audio_tokens.size(1) / max(1, token_len)
        s = min(max(0, int(start * ratio)), audio_tokens.size(1))
        e = min(max(s + 1, int(end * ratio)), audio_tokens.size(1))
        return audio_tokens[:, s:e]

    def _chart_prefix_summary(
        self,
        inp: torch.Tensor,
        end: int,
    ) -> torch.Tensor | None:
        if end <= 0 or self.max_summary_tokens <= 0:
            return None
        starts = list(range(0, end, max(1, self.window_stride)))
        starts = starts[-self.max_summary_tokens:]
        summaries: list[torch.Tensor] = []
        for start in starts:
            stop = min(start + max(1, self.window_stride), end)
            seg = inp[:, start:stop]
            if seg.size(1) == 0:
                continue
            pos = torch.arange(start, stop, device=inp.device).unsqueeze(0).expand(inp.size(0), -1)
            dist = compute_relative_distances(seg)
            h = self.token_embed(seg) + self.pos_embed(pos) + self.timing_embed(dist)
            summary = self._pool_summary(h, self.chart_summary_proj, 1, type_idx=0)
            if self.detach_summary:
                summary = summary.detach()
            summaries.append(summary)
        if not summaries:
            return None
        return torch.cat(summaries, dim=1)

    def _sample_window_start(self, total_len: int) -> int:
        if total_len <= self.window_tokens:
            return 0
        max_start = total_len - max(1, self.window_tokens)
        return int(torch.randint(0, max_start + 1, (1,), device=next(self.parameters()).device).item())

    def _forward_sampled_window(
        self,
        inp: torch.Tensor,
        tgt: torch.Tensor,
        onset: torch.Tensor,
        chroma: torch.Tensor,
        centroid: torch.Tensor,
        audio_tokens: torch.Tensor | None,
        cond: torch.Tensor,
        distances: torch.Tensor | None,
    ) -> dict[str, torch.Tensor | None]:
        B, T_in = inp.shape
        start = self._sample_window_start(T_in)
        end = min(start + max(1, self.window_tokens), T_in)
        win_inp = inp[:, start:end]
        win_tgt = tgt[:, start:end]
        win_len = win_inp.size(1)

        local_onset, local_chroma, local_centroid = self._slice_audio_features(onset, chroma, centroid, start, end, T_in)
        local_audio_tokens = self._slice_audio_tokens(audio_tokens, start, end, T_in)
        if self.training:
            audio_pack = checkpoint(self.audio, local_onset, local_chroma, local_centroid, local_audio_tokens, use_reentrant=False)
        else:
            audio_pack = self.audio(local_onset, local_chroma, local_centroid, audio_tokens=local_audio_tokens)
        local_memory = self._align_audio_memory(audio_pack.fused_memory, win_len, B)

        audio_summary = self._raw_audio_summary(onset, chroma, centroid) if self.audio_global_summary_tokens > 0 else None
        if audio_summary is not None:
            memory = torch.cat([audio_summary, local_memory], dim=1)
        else:
            memory = local_memory

        pos = torch.arange(start, end, device=inp.device).unsqueeze(0).expand(B, -1)
        if distances is None:
            win_dist = compute_relative_distances(win_inp)
        else:
            win_dist = distances[:, start:end, :]
        token_x = self.token_embed(win_inp) + self.pos_embed(pos) + self.timing_embed(win_dist)

        summary = self._chart_prefix_summary(inp, start)
        if summary is not None and self.summary_position == "suffix":
            x = torch.cat([token_x, summary], dim=1)
        elif summary is not None:
            x = torch.cat([summary, token_x], dim=1)
        else:
            x = token_x
        x = self._run_decoder(x, memory, cond, token_len=win_len)
        token_h = x[:, :win_len, :] if summary is None or self.summary_position == "suffix" else x[:, -win_len:, :]

        if start == 0:
            loss_mask = torch.ones_like(win_tgt, dtype=torch.bool)
        else:
            loss_offset = max(0, win_len - max(1, self.window_stride))
            loss_mask = torch.arange(win_len, device=inp.device).unsqueeze(0) >= loss_offset
            loss_mask = loss_mask.expand(B, -1)
        loss_mask = loss_mask & (win_tgt != self.pad_token_id)
        loss = _chunked_lm_ce(self.lm_head, token_h, win_tgt, self.pad_token_id, loss_mask=loss_mask)
        return {"logits": None, "loss": loss, "hidden_states": token_h.detach() if self.detach_summary else token_h}

    def _forward_sliding(
        self,
        inp: torch.Tensor,
        tgt: torch.Tensor,
        audio_memory: torch.Tensor,
        cond: torch.Tensor,
        distances: torch.Tensor | None,
    ) -> dict[str, torch.Tensor | None]:
        B, T_in = inp.shape
        starts = self._window_starts(T_in)
        summary_cache: list[torch.Tensor] = []
        hidden_chunks: list[torch.Tensor] = []
        total_loss = inp.new_tensor(0.0, dtype=torch.float32)
        total_tokens = 0
        audio_summary = self._pool_summary(
            audio_memory,
            self.audio_summary_proj,
            max(1, self.audio_global_summary_tokens),
            type_idx=1,
        ) if self.audio_global_summary_tokens > 0 else None

        for i, start in enumerate(starts):
            end = min(start + max(1, self.window_tokens), T_in)
            if end <= start:
                continue
            win_inp = inp[:, start:end]
            win_tgt = tgt[:, start:end]
            win_len = win_inp.size(1)
            pos = torch.arange(start, end, device=inp.device).unsqueeze(0).expand(B, -1)
            if distances is None:
                win_dist = compute_relative_distances(win_inp)
            else:
                win_dist = distances[:, start:end, :]
            token_x = self.token_embed(win_inp) + self.pos_embed(pos) + self.timing_embed(win_dist)

            summary = torch.cat(summary_cache[-self.max_summary_tokens:], dim=1) if summary_cache else None
            if summary is not None and self.summary_position == "suffix":
                x = torch.cat([token_x, summary], dim=1)
            elif summary is not None:
                x = torch.cat([summary, token_x], dim=1)
            else:
                x = token_x

            memory = self._audio_window_memory(audio_memory, start, end, audio_summary)
            x = self._run_decoder(x, memory, cond, token_len=win_len)
            token_h = x[:, :win_len, :] if summary is None or self.summary_position == "suffix" else x[:, -win_len:, :]

            if i == 0:
                loss_mask = torch.ones_like(win_tgt, dtype=torch.bool)
            else:
                prev_end = starts[i - 1] + max(1, self.window_tokens)
                loss_mask = torch.arange(start, end, device=inp.device).unsqueeze(0) >= prev_end
                if not loss_mask.any():
                    loss_mask = torch.arange(start, end, device=inp.device).unsqueeze(0) >= start + max(0, win_len - self.window_stride)
                loss_mask = loss_mask.expand(B, -1)
            loss_mask = loss_mask & (win_tgt != self.pad_token_id)
            if loss_mask.any():
                chunk_loss = _chunked_lm_ce(self.lm_head, token_h, win_tgt, self.pad_token_id, loss_mask=loss_mask)
                n = int(loss_mask.sum().item())
                total_loss = total_loss + chunk_loss.float() * n
                total_tokens += n

            keep = loss_mask[0].bool()
            kept_h = token_h[:, keep, :]
            if self.training:
                hidden_chunks.append(kept_h.detach() if self.detach_summary else kept_h)
            else:
                hidden_chunks.append(kept_h.detach().cpu())

            slid_end = min(end, start + max(1, self.window_stride))
            slid_len = max(0, slid_end - start)
            if slid_len > 0:
                slid_hidden = token_h[:, :slid_len, :]
                if self.detach_summary:
                    slid_hidden = slid_hidden.detach()
                summary_cache.append(self._pool_summary(slid_hidden, self.chart_summary_proj, 1, type_idx=0))

        if total_tokens == 0:
            loss = inp.sum() * 0.0
        else:
            loss = total_loss / total_tokens
        hidden_states = (
            torch.cat(hidden_chunks, dim=1)
            if hidden_chunks
            else inp.new_zeros(B, 0, self.hidden_dim, dtype=torch.float32)
        )
        return {"logits": None, "loss": loss, "hidden_states": hidden_states}

    def forward(self, onset, chroma, centroid, tokens, bpm, level, genre, distances=None, audio_tokens=None):
        # 确保输入有 batch 维度
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        if onset.dim() == 1:
            onset = onset.unsqueeze(0)
        if chroma.dim() == 2:
            chroma = chroma.unsqueeze(0)
        if centroid.dim() == 1:
            centroid = centroid.unsqueeze(0)

        inp = tokens[:, :-1]
        tgt = tokens[:, 1:]
        B, T_in = inp.shape

        bpm = bpm.float().reshape(B, -1)
        level = level.float().reshape(B, -1)
        genre = genre.float().reshape(B, -1)
        cond_in = torch.cat([bpm, level, genre], dim=-1)
        cond = self.cond_embed(cond_in)
        if self.training and self.use_sliding_window and T_in > self.window_tokens:
            if distances is not None:
                distances = distances[:, :T_in, :]
            return self._forward_sampled_window(
                inp, tgt, onset, chroma, centroid, audio_tokens, cond, distances
            )

        # 非滑窗路径保留整首音频编码，供短序列训练和推理使用。
        if self.training:
            audio_pack = checkpoint(self.audio, onset, chroma, centroid, audio_tokens, use_reentrant=False)
        else:
            audio_pack = self.audio(onset, chroma, centroid, audio_tokens=audio_tokens)
        audio_memory = self._align_audio_memory(audio_pack.fused_memory, T_in, B)

        if self.use_sliding_window and T_in > self.window_tokens:
            if distances is not None:
                distances = distances[:, :T_in, :]
            return self._forward_sliding(inp, tgt, audio_memory, cond, distances)

        pos = torch.arange(T_in, device=tokens.device).unsqueeze(0).expand(B, -1)
        if distances is None:
            distances = compute_relative_distances(inp)
        else:
            distances = distances[:, :T_in, :]

        x = self.token_embed(inp) + self.pos_embed(pos) + self.timing_embed(distances)
        x = self._run_decoder(x, audio_memory, cond)
        # 训练时: 分块 lm_head+CE，避免 [B,T,161K] 大 tensor 驻留
        # 推理/eval 时: 正常计算 logits（generate 需要）
        if self.training:
            loss = _chunked_lm_ce(self.lm_head, x, tgt, self.pad_token_id)
            return {"logits": None, "loss": loss, "hidden_states": x}
        else:
            logits = self.lm_head(x)
            valid = tgt != self.pad_token_id
            if valid.any():
                loss = _chunked_ce(logits, tgt, self.pad_token_id)
            else:
                loss = logits.sum() * 0.0
            return {"logits": logits, "loss": loss, "hidden_states": x}

    @torch.no_grad()
    def generate(self, onset, chroma, centroid, bpm=173.0, level=10.0, genre=0, max_steps=2048, temperature=1.0, top_k=50, audio_tokens=None):
        device = next(self.parameters()).device
        if onset.dim() == 1:
            onset = onset.unsqueeze(0); chroma = chroma.unsqueeze(0); centroid = centroid.unsqueeze(0)
        onset = onset.to(device); chroma = chroma.to(device); centroid = centroid.to(device)

        # 一次性计算 cond 和 audio_memory
        cond = self.cond_embed(torch.tensor([[bpm, level, genre]], device=device, dtype=torch.float32))
        audio_memory = self.audio(onset, chroma, centroid, audio_tokens=audio_tokens).fused_memory  # [1, T_audio, D]

        generated = [self.bos_token_id]
        # 增量追踪最后事件位置 (用于 O(1) 距离计算)
        lp = lh = ls = lt = -1

        for step in range(max_steps):
            T_cur = len(generated)
            tokens = torch.tensor([generated], device=device)

            # 增量计算最后 token 的距离
            dist_t = [
                T_cur - 1 - lp if lp >= 0 else 0,
                T_cur - 1 - lh if lh >= 0 else 0,
                T_cur - 1 - ls if ls >= 0 else 0,
                T_cur - 1 - lt if lt >= 0 else 0,
            ]

            if step == 0:
                distances = torch.tensor([dist_t], device=device).unsqueeze(0)
            else:
                # 拼接历史距离 + 新 token 距离
                new_dist = torch.tensor([dist_t], device=device).unsqueeze(0)  # [1, 1, 4]
                distances = torch.cat([prev_distances, new_dist], dim=1)

            pos = torch.arange(T_cur, device=device).unsqueeze(0)
            x = self.token_embed(tokens) + self.pos_embed(pos) + self.timing_embed(distances)
            memory = audio_memory[:, :T_cur, :] if audio_memory.size(1) >= T_cur \
                else torch.cat([audio_memory, audio_memory[:, -1:, :].expand(1, T_cur - audio_memory.size(1), -1)], dim=1)
            mask = torch.triu(torch.ones((T_cur, T_cur), device=device, dtype=torch.bool), diagonal=1)
            for layer in self.layers:
                x = layer(x, memory, cond, mask)

            logits = self.lm_head(x[:, -1, :]) / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, -1:]] = float("-inf")
            next_tok = torch.multinomial(torch.softmax(logits, dim=-1), 1).item()
            generated.append(next_tok)

            lp, lh, ls, lt = _track_event_pos(next_tok, T_cur - 1, lp, lh, ls, lt)
            prev_distances = distances

            if next_tok == self.eos_token_id:
                break

        if not generated or generated[-1] != self.eos_token_id:
            generated.append(self.eos_token_id)
        return generated

    @torch.no_grad()
    def generate_step(self, prefix: list[int], onset, chroma, centroid, bpm=173.0, level=10.0, genre=0, audio_tokens=None):
        device = next(self.parameters()).device
        tokens = torch.tensor([prefix], device=device)
        pos = torch.arange(tokens.size(1), device=device).unsqueeze(0)
        distances = compute_relative_distances(tokens)
        audio_memory = self.audio(onset.to(device), chroma.to(device), centroid.to(device), audio_tokens=audio_tokens).fused_memory
        cond = self.cond_embed(torch.tensor([[bpm, level, genre]], device=device, dtype=torch.float32))
        x = self.token_embed(tokens) + self.pos_embed(pos) + self.timing_embed(distances)
        memory = audio_memory[:, :tokens.size(1), :] if audio_memory.size(1) >= tokens.size(1) else audio_memory[:, -1:, :].expand(1, tokens.size(1), -1)
        mask = torch.triu(torch.ones((tokens.size(1), tokens.size(1)), device=device, dtype=torch.bool), diagonal=1)
        for layer in self.layers:
            x = layer(x, memory, cond, mask)
        return self.lm_head(x[:, -1, :])


def decode_stage1_slide_fields(tokens: list[int]) -> dict[str, int | tuple[int, int] | None]:
    slide_start = slide_end = None
    dur = None
    for i, tid in enumerate(tokens):
        if SLD_BEG_BASE <= tid < SLD_BEG_END:
            slide_start = tid - SLD_BEG_BASE + 1
        elif SLD_END_POS_BASE <= tid < SLD_END_POS_END:
            slide_end = tid - SLD_END_POS_BASE + 1
        elif tid == DUR and i + 2 < len(tokens):
            num_tok = tokens[i + 1]
            den_tok = tokens[i + 2]
            dur = (ID_TO_DUR_NUM.get(num_tok), ID_TO_DUR_DEN.get(den_tok))
    return {"slide_start": slide_start, "slide_end": slide_end, "duration": dur}


maiGenerator = MaiGenerator
