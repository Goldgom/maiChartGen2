from __future__ import annotations

import torch

from infer.prob_select import select_by_generation_probability
from models.spike_stage import SpikeClassifier
from Tokenizer.config_vocab import ID_TO_CONFIG


def load_model(**kwargs):
    return SpikeClassifier(**kwargs)


def infer_touch_mask(config_tokens: torch.Tensor) -> torch.Tensor:
    """Infer [B, T, 33] firework candidates from stage tokens."""
    if config_tokens.dim() == 1:
        config_tokens = config_tokens.unsqueeze(0)
    mask = torch.zeros(config_tokens.size(0), config_tokens.size(1), 33, dtype=torch.bool, device=config_tokens.device)
    for b in range(config_tokens.size(0)):
        for t, tid in enumerate(config_tokens[b].tolist()):
            sc = ID_TO_CONFIG.get(int(tid))
            if sc is None:
                continue
            for zone, _state in sc.touches:
                if 0 <= zone < 33:
                    mask[b, t, zone] = True
    return mask


@torch.no_grad()
def spike_probabilities(
    model: SpikeClassifier,
    tokens: torch.Tensor,
    stage1_hidden: torch.Tensor,
) -> torch.Tensor:
    """Return P(firework/spike) for each [batch, token, touch-zone]."""
    if hasattr(model, "predict_probabilities"):
        return model.predict_probabilities(tokens, stage1_hidden)
    logits = model(tokens, stage1_hidden)
    return torch.softmax(logits.float(), dim=-1)[..., 1]


@torch.no_grad()
def generate_spike_mask(
    model: SpikeClassifier,
    tokens: torch.Tensor,
    stage1_hidden: torch.Tensor,
    *,
    spike_generation_prob: float = 0.05,
    touch_mask: torch.Tensor | None = None,
    mode: str = "topk",
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict spike probabilities and select final firework touch zones.

    `spike_generation_prob` is the target fraction of valid touch candidates to
    mark as firework. The model probabilities decide which candidates are most
    likely; `mode="topk"` is deterministic and `mode="sample"` adds randomness.
    """
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)
    if stage1_hidden.dim() == 2:
        stage1_hidden = stage1_hidden.unsqueeze(0)
    probs = spike_probabilities(model, tokens, stage1_hidden)
    if touch_mask is None:
        touch_mask = infer_touch_mask(tokens)
    else:
        touch_mask = touch_mask.to(probs.device).bool()
    selected = select_by_generation_probability(
        probs,
        touch_mask,
        spike_generation_prob,
        mode=mode,
        temperature=temperature,
    )
    return selected, probs

