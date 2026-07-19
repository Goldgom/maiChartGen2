"""
models/stage1_chart.py - Stage 1: audio + conditions -> chart token sequence.

Stage1 is autoregressive: at frame t it predicts chart[t] from BOS/chart[:t],
the aligned music frame context, beat/difficulty/level/tags, and causal history.
"""

from __future__ import annotations

import math
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


LogitsProcessor = Callable[..., torch.Tensor]


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

    def _split_heads(self, x: torch.Tensor, n_head: int) -> torch.Tensor:
        B, T, D = x.shape
        return x.view(B, T, n_head, D // n_head).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, H, T, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * Dh)

    def _self_attn_step(
        self,
        layer,
        x: torch.Tensor,
        cache: dict | None,
        max_history: int,
    ) -> tuple[torch.Tensor, dict]:
        attn = layer.self_attn
        D = attn.embed_dim
        H = attn.num_heads

        q_w, k_w, v_w = attn.in_proj_weight.chunk(3, dim=0)
        if attn.in_proj_bias is None:
            q_b = k_b = v_b = None
        else:
            q_b, k_b, v_b = attn.in_proj_bias.chunk(3, dim=0)

        q = self._split_heads(F.linear(x, q_w, q_b), H)
        k_new = self._split_heads(F.linear(x, k_w, k_b), H)
        v_new = self._split_heads(F.linear(x, v_w, v_b), H)

        if cache is None:
            k = k_new
            v = v_new
        else:
            k = torch.cat([cache["k"], k_new], dim=2)
            v = torch.cat([cache["v"], v_new], dim=2)

        if max_history and max_history > 0 and k.shape[2] > max_history:
            k = k[:, :, -max_history:, :].contiguous()
            v = v[:, :, -max_history:, :].contiguous()

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D // H)
        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)
        out = self._merge_heads(out)
        out = attn.out_proj(out)
        return out, {"k": k, "v": v}

    def _chart_pe_step(self, chart_x: torch.Tensor, window_len: int) -> torch.Tensor:
        pe_module = self.chart_fusion.chart_pe
        key = (window_len, chart_x.device.index if chart_x.device.type == "cuda" else 0)
        if not self.training and key in pe_module._cache:
            pe = pe_module._cache[key]
        else:
            pe = pe_module._compute_train(window_len).detach()
            if not self.training:
                pe_module._cache[key] = pe
                while len(pe_module._cache) > 8:
                    pe_module._cache.pop(next(iter(pe_module._cache)))
        return chart_x + pe_module.scale * pe[-1:].view(1, 1, -1)

    def _fuse_step(
        self,
        chart_x: torch.Tensor,
        audio_feat: torch.Tensor,
        cond: torch.Tensor,
        window_len: int,
    ) -> torch.Tensor:
        chart_x = self._chart_pe_step(chart_x, window_len)
        gate = torch.sigmoid(self.chart_fusion.gate(torch.cat([chart_x, audio_feat], dim=-1)))
        audio_x = gate * self.chart_fusion.audio_proj(audio_feat)
        return self.chart_fusion.dropout(self.chart_fusion.ln(chart_x + audio_x + cond))

    def _forward_step(
        self,
        prev_token: torch.Tensor,
        audio_feat: torch.Tensor,
        beat: torch.Tensor,
        difficulty: torch.Tensor,
        level: torch.Tensor,
        tag_ids: torch.Tensor,
        layer_cache: list[dict | None],
        window_len: int,
        is_first: bool,
        max_history: int,
    ) -> tuple[torch.Tensor, list[dict | None]]:
        if is_first:
            chart_x = self.bos_embed.expand(prev_token.shape[0], 1, -1)
        else:
            chart_x = self.chart_embed(prev_token.long().clamp(0, self.cfg.chart_vocab_size - 1))

        cond = self.cond_embed(
            beat,
            difficulty,
            level,
            tag_ids,
            frame_query=chart_x,
        )
        x = self._fuse_step(chart_x, audio_feat, cond, window_len)

        new_cache: list[dict | None] = []
        for i, layer in enumerate(self.layers):
            r = x
            x_norm = layer.ln1(x)
            attn_out, cache_i = self._self_attn_step(
                layer,
                x_norm,
                layer_cache[i],
                max_history,
            )
            x = layer.drop_attn(attn_out) + r

            if layer.cross_attn:
                r = x
                x_norm = layer.ln_cross(x)
                x = layer.cross_attn_layer(
                    x_norm,
                    audio_feat,
                    audio_feat,
                    need_weights=False,
                )[0]
                x = layer.drop_cross(x) + r

            r = x
            x = layer.ln2(x)
            x = layer.ff(x)
            x = layer.drop_ff(x) + r
            new_cache.append(cache_i)

        return self.head(self.ln_final(x)), new_cache

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
        max_history: int = 512,
        use_kv_cache: bool = True,
    ) -> torch.Tensor:
        """Autoregressively generate one chart token per audio frame.

        max_history limits decoding to the latest N frames. A value <= 0 keeps
        the exact full prefix, but avoids the old full-song recompute per step.
        """
        self.eval()
        B, T, _ = audio_tokens.shape
        device = audio_tokens.device
        generated = torch.zeros(B, T, dtype=torch.long, device=device)

        if use_kv_cache and aligned_cross_attention:
            audio_feat = self._match_time(self.audio_encoder(audio_tokens), T)
            beat = self._match_time(beat_signal, T)
            layer_cache: list[dict | None] = [None] * len(self.layers)
            for t in range(T):
                window_len = t + 1
                if max_history and max_history > 0:
                    window_len = min(window_len, max_history)
                prev_token = (
                    torch.zeros(B, 1, dtype=torch.long, device=device)
                    if t == 0 else generated[:, t - 1:t]
                )
                logits, layer_cache = self._forward_step(
                    prev_token,
                    audio_feat[:, t:t + 1],
                    beat[:, t:t + 1],
                    difficulty,
                    level,
                    tag_ids,
                    layer_cache,
                    window_len,
                    t == 0,
                    max_history,
                )
                if logits_processor is not None:
                    try:
                        logits = logits_processor(logits, t, generated)
                    except TypeError:
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

        for t in range(T):
            if max_history and max_history > 0:
                start = max(0, t + 1 - max_history)
            else:
                start = 0
            end = t + 1
            L = end - start
            input_tokens = torch.zeros(B, L, dtype=torch.long, device=device)
            if L > 1:
                input_tokens[:, 1:] = generated[:, start:t]
            out = self.forward(
                audio_tokens[:, start:end],
                beat_signal[:, start:end],
                difficulty,
                level,
                tag_ids,
                input_chart_tokens=input_tokens,
                aligned_cross_attention=aligned_cross_attention,
            )
            logits = out["logits"][:, -1:, :]
            if logits_processor is not None:
                try:
                    logits = logits_processor(logits, t, generated)
                except TypeError:
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
