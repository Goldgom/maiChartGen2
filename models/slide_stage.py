from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.audio_context import align_sequence_features, gather_query_features
from modules.chart_blocks import CausalPathBlock
from Tokenizer.slide_star_vocab import (
    SLD_STAR_BOS,
    SLD_STAR_EOS,
    SLD_STAR_PAD,
    SLD_STAR_VOCAB_SIZE,
)


class SlidePathGenerator(nn.Module):
    def __init__(
        self,
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        stage1_dim: int = 768,
        onset_dim: int = 3,
        max_path_len: int = 128,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bos_token_id = SLD_STAR_BOS
        self.pad_token_id = SLD_STAR_PAD
        self.vocab_size = SLD_STAR_VOCAB_SIZE
        self.max_path_len = int(max_path_len)

        self.start_embed = nn.Embedding(9, hidden_dim)
        self.tok_embed = nn.Embedding(SLD_STAR_VOCAB_SIZE, hidden_dim)
        self.pos_embed = nn.Embedding(self.max_path_len, hidden_dim)
        self.audio_proj = nn.Linear(stage1_dim, hidden_dim)
        self.stage1_proj = nn.Linear(stage1_dim, hidden_dim)
        self.onset_proj = nn.Linear(onset_dim, hidden_dim)
        self.event_slot_embed = nn.Embedding(16384, hidden_dim)
        self.local_cond_proj = nn.Linear(hidden_dim * 3, hidden_dim)

        self.layers = nn.ModuleList([
            CausalPathBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(hidden_dim, SLD_STAR_VOCAB_SIZE)

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

    def _build_local_condition(
        self,
        audio_memory: torch.Tensor,
        event_slot: torch.Tensor | None,
        stage1_hidden: torch.Tensor | None,
        onset: torch.Tensor | None,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        cond_parts: list[torch.Tensor] = []
        if event_slot is None:
            return cond_parts, audio_memory

        seq_ref = None
        if stage1_hidden is not None:
            seq_ref = int(stage1_hidden.size(1))
        elif onset is not None:
            seq_ref = int(onset.size(1))
        else:
            seq_ref = int(event_slot.max().item()) + 2

        audio_local = gather_query_features(align_sequence_features(audio_memory, seq_ref), event_slot)
        zero = torch.zeros_like(audio_local)
        s1_local = gather_query_features(align_sequence_features(stage1_hidden, seq_ref), event_slot) if stage1_hidden is not None else zero
        onset_local = gather_query_features(align_sequence_features(onset, seq_ref), event_slot) if onset is not None else zero
        cond_parts.append(self.event_slot_embed(event_slot).unsqueeze(1))
        cond_parts.append(self.local_cond_proj(torch.cat([s1_local, audio_local, onset_local], dim=-1)).unsqueeze(1))
        return cond_parts, audio_memory

    def forward(self, target_path, start_pos, audio_memory, stage1_hidden=None, onset=None, event_slot=None):
        bsz, t_path = target_path.shape
        audio_memory = self.audio_proj(audio_memory)
        if torch.isnan(audio_memory).any():
            audio_memory = torch.nan_to_num(audio_memory, nan=0.0)

        start_pos = self._normalize_index(start_pos).to(target_path.device)
        cond_parts = [self.start_embed(start_pos)]
        if cond_parts[0].dim() == 2:
            cond_parts[0] = cond_parts[0].unsqueeze(1)

        s1 = None
        if stage1_hidden is not None:
            s1 = self.stage1_proj(stage1_hidden.to(target_path.device))
            cond_parts.append(s1.mean(dim=1, keepdim=True))

        o = None
        if onset is not None:
            o = self.onset_proj(onset.to(target_path.device).float())
            cond_parts.append(o.mean(dim=1, keepdim=True))

        if event_slot is not None:
            event_slot = self._normalize_index(event_slot).to(target_path.device)
        local_parts, audio_memory = self._build_local_condition(audio_memory, event_slot, s1, o)
        ctx = torch.cat(cond_parts + local_parts + [audio_memory], dim=1)

        inp = target_path[:, :-1]
        tgt = target_path[:, 1:]
        t_in = inp.size(1)
        pos = torch.arange(t_in, device=target_path.device).unsqueeze(0).expand(bsz, -1)
        x = self.tok_embed(inp) + self.pos_embed(pos)

        model_dtype = next(self.parameters()).dtype
        mask = torch.triu(
            torch.full((t_in, t_in), float("-inf"), device=target_path.device, dtype=model_dtype),
            diagonal=1,
        )
        for layer in self.layers:
            x = layer(x, ctx, mask)

        logits = torch.clamp(self.head(x), min=-50.0, max=50.0)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt.reshape(-1),
            ignore_index=SLD_STAR_PAD,
        )
        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def generate(self, start_pos, audio_memory, stage1_hidden=None, onset=None, event_slot=None, max_steps=16, temperature=0.8, top_k=10):
        device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype

        audio_memory = self.audio_proj(audio_memory.to(device))
        start_pos = self._normalize_index(start_pos).to(device)
        cond_parts = [self.start_embed(start_pos)]

        s1 = None
        if stage1_hidden is not None:
            s1 = self.stage1_proj(stage1_hidden.to(device))
            cond_parts.append(s1.mean(dim=1, keepdim=True))

        o = None
        if onset is not None:
            o = self.onset_proj(onset.to(device).float())
            cond_parts.append(o.mean(dim=1, keepdim=True))

        if event_slot is not None:
            event_slot = self._normalize_index(event_slot).to(device)
        local_parts, audio_memory = self._build_local_condition(audio_memory, event_slot, s1, o)
        ctx = torch.cat(cond_parts + local_parts + [audio_memory], dim=1)

        generated = [SLD_STAR_BOS]
        for _ in range(max_steps):
            tokens = torch.tensor([generated], device=device)
            pos = torch.arange(tokens.size(1), device=device).unsqueeze(0)
            x = self.tok_embed(tokens) + self.pos_embed(pos)
            mask = torch.triu(
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
            if torch.isnan(probs).any():
                probs = torch.ones_like(probs) / probs.size(-1)
            next_tok = torch.multinomial(probs, 1).item()
            generated.append(next_tok)
            if next_tok == SLD_STAR_EOS:
                break
        return generated


slideG = SlidePathGenerator


class SlideStarRefiner(nn.Module):
    def __init__(self, hidden_dim=384, num_layers=4, num_heads=6, dropout=0.1, stage1_dim: int = 768, onset_dim: int = 3, max_path_len: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = SLD_STAR_VOCAB_SIZE
        self.bos_token_id = SLD_STAR_BOS
        self.pad_token_id = SLD_STAR_PAD
        self.max_path_len = int(max_path_len)

        self.coarse_embed = nn.Embedding(SLD_STAR_VOCAB_SIZE, hidden_dim)
        self.pos_embed = nn.Embedding(self.max_path_len, hidden_dim)
        self.stage1_proj = nn.Linear(stage1_dim, hidden_dim)
        self.audio_proj = nn.Linear(stage1_dim, hidden_dim)
        self.onset_proj = nn.Linear(onset_dim, hidden_dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 4, dropout, batch_first=True, activation="gelu")
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(hidden_dim, SLD_STAR_VOCAB_SIZE)

    def forward(self, coarse_path, stage1_hidden, audio_memory=None, onset=None, star_mask=None, target_path=None):
        bsz, t_c = coarse_path.shape
        pos = torch.arange(t_c, device=coarse_path.device).unsqueeze(0).expand(bsz, -1)
        x = self.coarse_embed(coarse_path) + self.pos_embed(pos)

        s1 = self.stage1_proj(stage1_hidden)
        x = x + align_sequence_features(s1, t_c)
        if audio_memory is not None:
            a = self.audio_proj(audio_memory)
            x = x + align_sequence_features(a, t_c)
        if onset is not None:
            o = self.onset_proj(onset.float())
            x = x + align_sequence_features(o, t_c)

        for layer in self.layers:
            x = layer(x)

        logits = torch.clamp(self.head(x), min=-50.0, max=50.0)
        if target_path is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target_path.reshape(-1), ignore_index=SLD_STAR_PAD)
            return {"logits": logits, "loss": loss}
        return {"logits": logits}

    @torch.no_grad()
    def refine(self, coarse_tokens, stage1_hidden, audio_memory=None, onset=None, max_steps=32, temperature=0.6):
        device = next(self.parameters()).device
        coarse = torch.tensor([coarse_tokens], device=device)
        bsz, t_c = coarse.shape
        pos = torch.arange(t_c, device=device).unsqueeze(0)
        x = self.coarse_embed(coarse) + self.pos_embed(pos)

        s1 = self.stage1_proj(stage1_hidden.to(device))
        x = x + align_sequence_features(s1, t_c)
        if audio_memory is not None:
            a = self.audio_proj(audio_memory.to(device))
            x = x + align_sequence_features(a, t_c)
        if onset is not None:
            o = self.onset_proj(onset.to(device).float())
            x = x + align_sequence_features(o, t_c)

        for layer in self.layers:
            x = layer(x)

        logits = self.head(x[:, -1:, :]) / temperature
        probs = torch.softmax(logits, dim=-1).squeeze(1)
        return probs.cpu()


starG = SlideStarRefiner
