from __future__ import annotations

import torch

from infer.prob_select import select_by_generation_probability
from models.break_stage import BreakClassifier
from Tokenizer.config_vocab import ID_TO_CONFIG, BTN_PRESS, BTN_HOLD_START, BTN_SLIDE_START


def load_model(**kwargs):
    return BreakClassifier(**kwargs)


def infer_press_mask(config_tokens: torch.Tensor) -> torch.Tensor:
    """Infer [B, T, 8] break candidates from stage tokens."""
    if config_tokens.dim() == 1:
        config_tokens = config_tokens.unsqueeze(0)
    mask = torch.zeros(config_tokens.size(0), config_tokens.size(1), 8, dtype=torch.bool, device=config_tokens.device)
    for b in range(config_tokens.size(0)):
        for t, tid in enumerate(config_tokens[b].tolist()):
            sc = ID_TO_CONFIG.get(int(tid))
            if sc is None:
                continue
            for pos, state in sc.buttons:
                if state in (BTN_PRESS, BTN_HOLD_START, BTN_SLIDE_START) and 1 <= pos <= 8:
                    mask[b, t, pos - 1] = True
    return mask


@torch.no_grad()
def break_probabilities(
    model: BreakClassifier,
    tokens: torch.Tensor,
    stage1_hidden: torch.Tensor,
) -> torch.Tensor:
    """Return P(break) for each [batch, token, button-position]."""
    if hasattr(model, "predict_probabilities"):
        return model.predict_probabilities(tokens, stage1_hidden)
    logits = model(tokens, stage1_hidden)
    return torch.softmax(logits.float(), dim=-1)[..., 1]


@torch.no_grad()
def generate_break_mask(
    model: BreakClassifier,
    tokens: torch.Tensor,
    stage1_hidden: torch.Tensor,
    *,
    break_generation_prob: float = 0.08,
    press_mask: torch.Tensor | None = None,
    mode: str = "topk",
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict break probabilities and select final break positions.

    `break_generation_prob` is the target fraction of valid press/hold/slide-head
    candidates to turn into break notes. The model probabilities rank the
    candidates; `mode="topk"` is deterministic, while `mode="sample"` samples
    candidates weighted by model confidence.
    """
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)
    if stage1_hidden.dim() == 2:
        stage1_hidden = stage1_hidden.unsqueeze(0)
    probs = break_probabilities(model, tokens, stage1_hidden)
    if press_mask is None:
        press_mask = infer_press_mask(tokens)
    else:
        press_mask = press_mask.to(probs.device).bool()
    selected = select_by_generation_probability(
        probs,
        press_mask,
        break_generation_prob,
        mode=mode,
        temperature=temperature,
    )
    return selected, probs

