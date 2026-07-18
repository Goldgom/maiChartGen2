"""
models/stage1_chart.py - Stage 1: audio + conditions -> chart token sequence.

Stage1 is autoregressive: at frame t it predicts chart[t] from BOS/chart[:t],
the aligned music frame context, beat/difficulty/level/tags, and causal history.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import (
    StageConfig,
    AudioEncoder,
    ConditionEmbedding,
    ChartAudioFusion,
    make_model,
    build_causal_mask,
    _init_weights,
)


LogitsProcessor = Callable[[torch.Tensor], torch.Tensor]


class Stage1ChartModel(nn.Module):
    """Stage 1 autoregressive chart skeleton generator."""

    def __init__(self, cfg: StageConfig):
        super().__init__()
        self.cfg = cfg

        self.audio_encoder = AudioEncoder(cfg)
        self.cond_embed = ConditionEmbedding(cfg)

        self.chart_embed = nn.Embedding(cfg.chart_vocab_size, cfg.d_model)
        self.bos_embed = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.chart_fusion = ChartAudioFusion(cfg)

        self.layers = make_model(cfg, cfg.n_layer, cross_attn=True)

        self.ln_final = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.chart_vocab_size)
        for module in [self.chart_embed, self.ln_final, self.head]:
            module.apply(lambda m: _init_weights(m, cfg.init_std))
        nn.init.trunc_normal_(self.bos_embed, std=cfg.init_std)

    def _match_time(self, x: torch.Tensor, T: int) -> torch.Tensor:
        if x.shape[1] == T:
            return x
        return F.interpolate(
            x.transpose(1, 2),
            size=T,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    def _teacher_forcing_inputs(self, chart_tokens: torch.Tensor) -> torch.Tensor:
        B, T = chart_tokens.shape
        if T <= 0:
            return chart_tokens
        bos_id = torch.zeros(B, 1, dtype=torch.long, device=chart_tokens.device)
        return torch.cat([bos_id, chart_tokens[:, :-1].long()], dim=1)

    def _embed_history(self, input_chart_tokens: torch.Tensor) -> torch.Tensor:
        B, T = input_chart_tokens.shape
        x = self.chart_embed(input_chart_tokens.long().clamp(0, self.cfg.chart_vocab_size - 1))
        x[:, :1, :] = self.bos_embed.expand(B, 1, -1)
        return x

    def forward(
        self,
        audio_tokens: torch.Tensor,
        beat_signal: torch.Tensor,
        difficulty: torch.Tensor,
        level: torch.Tensor,
        tag_ids: torch.Tensor,
        chart_tokens: torch.Tensor | None = None,
        input_chart_tokens: torch.Tensor | None = None,
        aligned_cross_attention: bool = True,
    ) -> dict:
        """
        Args:
            chart_tokens: targets. If input_chart_tokens is omitted, teacher
                forcing uses BOS + chart_tokens[:, :-1] as the AR input.
            input_chart_tokens: already-shifted AR input tokens.
            aligned_cross_attention: when true, each chart frame cross-attends
                only to the corresponding audio frame.

        Returns:
            logits: (B, T, V)
            loss: scalar, when chart_tokens is provided
        """
        if input_chart_tokens is None:
            if chart_tokens is not None:
                input_chart_tokens = self._teacher_forcing_inputs(chart_tokens)
            else:
                B, T, _ = audio_tokens.shape
                input_chart_tokens = torch.zeros(B, T, dtype=torch.long, device=audio_tokens.device)

        B, T = input_chart_tokens.shape
        device = input_chart_tokens.device

        audio_feat = self._match_time(self.audio_encoder(audio_tokens), T)
        beat = self._match_time(beat_signal, T)
        chart_x = self._embed_history(input_chart_tokens)
        cond = self.cond_embed(
            beat,
            difficulty,
            level,
            tag_ids,
            frame_query=chart_x,
        )
        x = self.chart_fusion(chart_x, audio_feat, cond)

        causal_mask = build_causal_mask(T, device)
        cross_mask = None
        if aligned_cross_attention:
            cross_mask = torch.full((T, T), float("-inf"), device=device)
            cross_mask.fill_diagonal_(0.0)

        for layer in self.layers:
            x = layer(
                x,
                memory=audio_feat,
                causal_mask=causal_mask,
                memory_mask=cross_mask,
            )

        logits = self.head(self.ln_final(x))
        result = {"logits": logits}

        if chart_tokens is not None:
            targets = chart_tokens[:, :T].long()
            result["loss"] = F.cross_entropy(
                logits.reshape(-1, self.cfg.chart_vocab_size),
                targets.reshape(-1),
            )

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
        logits_processor: LogitsProcessor | None = None,
        aligned_cross_attention: bool = True,
    ) -> torch.Tensor:
        """Autoregressively generate one chart token per audio frame."""
        self.eval()
        B, T, _ = audio_tokens.shape
        device = audio_tokens.device
        generated = torch.zeros(B, T, dtype=torch.long, device=device)

        for t in range(T):
            input_tokens = torch.zeros(B, T, dtype=torch.long, device=device)
            if t > 0:
                input_tokens[:, 1:t + 1] = generated[:, :t]
            out = self.forward(
                audio_tokens,
                beat_signal,
                difficulty,
                level,
                tag_ids,
                input_chart_tokens=input_tokens,
                aligned_cross_attention=aligned_cross_attention,
            )
            logits = out["logits"][:, t:t + 1, :]
            if logits_processor is not None:
                logits = logits_processor(logits)
            if temperature > 0:
                logits = logits / temperature
            if top_k > 0 and top_k < logits.shape[-1]:
                topk_vals, _ = torch.topk(logits, top_k, dim=-1)
                min_topk = topk_vals[:, :, -1:]
                logits = torch.where(
                    logits < min_topk,
                    torch.full_like(logits, float("-inf")),
                    logits,
                )
            probs = F.softmax(logits, dim=-1)
            generated[:, t] = torch.multinomial(probs[:, 0, :], 1).squeeze(-1)

        return generated
