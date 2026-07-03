from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AudioMemory:
    discrete_memory: torch.Tensor | None
    spectral_memory: torch.Tensor
    fused_memory: torch.Tensor
    local_context: torch.Tensor
    global_memory: torch.Tensor | None = None  # 粗粒度全局特征


class _DilatedConvStack(nn.Module):
    """多尺度膨胀卷积: 逐层扩大感受野, 覆盖 target_receptive_s 秒。"""

    def __init__(self, in_dim: int, hidden_dim: int, target_receptive_s: float = 5.0, slots_per_sec: float = 184):
        super().__init__()
        target_slots = int(target_receptive_s * slots_per_sec / 2)  # 单侧
        layers = []
        ch = in_dim
        dilation = 1
        while dilation < target_slots:
            k = 3
            p = dilation
            layers.append(nn.Conv1d(ch, hidden_dim, k, dilation=dilation, padding=p))
            layers.append(nn.GELU())
            ch = hidden_dim
            dilation *= 4  # 膨胀因子逐层 4x
        layers.append(nn.Conv1d(hidden_dim, hidden_dim, 1))  # 1×1 混合
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _StreamEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        max_seq_len: int = 16384,
        use_sliding_window: bool = False,
        window_len: int = 4096,
        window_stride: int = 2048,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.use_sliding_window = use_sliding_window
        self.window_len = max(1, int(window_len))
        self.window_stride = max(1, int(window_stride))
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pos_embed = nn.Embedding(65536, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def _encode_short(self, x: torch.Tensor, pos_offset: int = 0) -> torch.Tensor:
        pos = torch.arange(pos_offset, pos_offset + x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
        pos = pos.clamp(0, self.pos_embed.num_embeddings - 1)
        x = self.input_proj(x) + self.pos_embed(pos)
        x = self.encoder(x)
        return self.norm(x)

    def _window_starts(self, total_len: int) -> list[int]:
        if total_len <= self.window_len:
            return [0]
        stride = min(self.window_stride, self.window_len)
        starts = list(range(0, max(1, total_len - self.window_len + 1), stride))
        tail = max(0, total_len - self.window_len)
        if starts[-1] != tail:
            starts.append(tail)
        return sorted(set(starts))

    def _forward_sliding(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        out = x.new_zeros(B, T, self.input_proj[-1].out_features)
        weight = x.new_zeros(B, T, 1)
        for start in self._window_starts(T):
            end = min(start + self.window_len, T)
            chunk = self._encode_short(x[:, start:end, :], pos_offset=start)
            out[:, start:end, :] = out[:, start:end, :] + chunk
            weight[:, start:end, :] = weight[:, start:end, :] + 1
        return out / weight.clamp_min(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        B, T, D = x.shape

        if self.use_sliding_window and T > self.window_len:
            return self._forward_sliding(x)

        # 超长序列先插值降采样，编码后再插值还原，避免 O(T²) 自注意力 OOM
        if T > self.max_seq_len:
            x_t = x.transpose(1, 2)  # [B, D, T]
            x_short = F.interpolate(x_t, size=self.max_seq_len, mode="linear", align_corners=False)
            x = x_short.transpose(1, 2)  # [B, max_seq_len, D]

        x = self._encode_short(x)

        # 还原到原始长度
        if T > self.max_seq_len:
            x_t2 = x.transpose(1, 2)
            x_full = F.interpolate(x_t2, size=T, mode="linear", align_corners=False)
            x = x_full.transpose(1, 2)

        return x


class DualStreamAudioEncoder(nn.Module):
    """Dual-stream audio encoder shared by Stage 1 / Stage 2.5.

    Stream A: discrete EnCodec tokens.
    Stream B: spectral features (onset/chroma/centroid).
    """

    def __init__(
        self,
        hidden_dim: int,
        audio_vocab_size: int = 8195,
        spectral_dim: int = 14,
        stream_layers: int = 2,
        stream_heads: int = 8,
        local_window: int = 8,
        global_stride: int = 8,
        local_window_s: float = 5.0,
        local_slots_per_sec: float = 184,
        local_dilation_base: int = 4,
        max_spectral_len: int = 16384,
        use_spectral_sliding_window: bool = False,
        spectral_window_len: int = 4096,
        spectral_window_stride: int = 2048,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.audio_vocab_size = audio_vocab_size
        self.local_window = local_window
        self.global_stride = global_stride

        # 全局粗粒度: AvgPool + 轻量 Conv
        self.global_pool = nn.AvgPool1d(global_stride, stride=global_stride)
        self.global_proj = nn.Sequential(
            nn.Conv1d(spectral_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
        )

        self.audio_token_embed = nn.Embedding(audio_vocab_size, hidden_dim)
        self.audio_token_pos = nn.Embedding(65536, hidden_dim)
        self.audio_token_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=stream_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True,
                activation="gelu",
            ),
            num_layers=stream_layers,
        )

        self.spectral_stream = _StreamEncoder(
            input_dim=spectral_dim,
            hidden_dim=hidden_dim,
            num_layers=stream_layers,
            num_heads=stream_heads,
            max_seq_len=max_spectral_len,
            use_sliding_window=use_spectral_sliding_window,
            window_len=spectral_window_len,
            window_stride=spectral_window_stride,
        )

        self.local_conv = _DilatedConvStack(
            in_dim=spectral_dim,
            hidden_dim=hidden_dim,
            target_receptive_s=5.0,
            slots_per_sec=184,
        )
        self.local_norm = nn.LayerNorm(hidden_dim)
        self.stream_embed = nn.Parameter(torch.zeros(2, hidden_dim))
        nn.init.normal_(self.stream_embed, std=0.02)
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def _align_length(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        if x.size(1) == target_len:
            return x
        x = x.transpose(1, 2)
        x = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
        return x.transpose(1, 2).contiguous()

    def _encode_discrete(self, audio_tokens: torch.Tensor | None) -> torch.Tensor | None:
        if audio_tokens is None or audio_tokens.numel() == 0:
            return None
        if audio_tokens.dim() == 1:
            audio_tokens = audio_tokens.unsqueeze(0)
        audio_tokens = audio_tokens[:, :24576]  # 5min × 75tok/s
        if audio_tokens.size(1) == 0:
            return None
        pos = torch.arange(audio_tokens.size(1), device=audio_tokens.device).unsqueeze(0).expand(audio_tokens.size(0), -1)
        pos = pos.clamp(0, self.audio_token_pos.num_embeddings - 1)
        x = self.audio_token_embed(audio_tokens.long()) + self.audio_token_pos(pos)
        x = self.audio_token_encoder(x)
        return x + self.stream_embed[0].view(1, 1, -1)

    def _encode_spectral(self, onset: torch.Tensor, chroma: torch.Tensor, centroid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if onset.dim() == 1:
            onset = onset.unsqueeze(0)
            chroma = chroma.unsqueeze(0)
            centroid = centroid.unsqueeze(0)
        spectral = torch.cat([onset.unsqueeze(-1), chroma, centroid.unsqueeze(-1)], dim=-1)
        spectral_memory = self.spectral_stream(spectral) + self.stream_embed[1].view(1, 1, -1)
        local_context = self.local_conv(spectral.transpose(1, 2)).transpose(1, 2)
        local_context = self.local_norm(local_context)

        # 全局粗粒度: AvgPool → Conv → 插值回原始长度
        s_t = spectral.transpose(1, 2)
        if s_t.size(-1) < self.global_stride:
            s_global = F.adaptive_avg_pool1d(s_t, 1)
        else:
            s_global = self.global_pool(s_t)
        global_feat = self.global_proj(s_global)
        global_feat = F.interpolate(global_feat, size=spectral_memory.size(1), mode="linear")
        global_memory = global_feat.transpose(1, 2).contiguous()

        return spectral_memory, local_context, global_memory

    def forward(
        self,
        onset: torch.Tensor,
        chroma: torch.Tensor,
        centroid: torch.Tensor,
        audio_tokens: torch.Tensor | None = None,
    ) -> AudioMemory:
        discrete_memory = self._encode_discrete(audio_tokens)
        spectral_memory, local_context, global_memory = self._encode_spectral(onset, chroma, centroid)
        if discrete_memory is None:
            fused = spectral_memory + 0.1 * local_context + 0.05 * global_memory
        else:
            discrete_memory = self._align_length(discrete_memory, spectral_memory.size(1))
            stacked = torch.cat([discrete_memory, spectral_memory], dim=-1)
            gate = self.fusion_gate(stacked)
            fused = gate * discrete_memory + (1.0 - gate) * spectral_memory
            fused = fused + 0.1 * self.fusion_proj(stacked)
            fused = fused + 0.1 * local_context + 0.05 * global_memory
        return AudioMemory(
            discrete_memory=discrete_memory,
            spectral_memory=spectral_memory,
            fused_memory=fused,
            local_context=local_context,
            global_memory=global_memory,
        )
