from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any

import torch

from models.stage1 import MaiGenerator
from models.touch_stage import TouchRefiner
from models.slide_stage import SlidePathGenerator, SlideStarRefiner
from models.hold_stage import HoldDurationPredictor, TouchHoldDurationPredictor
from models.break_stage import BreakClassifier
from models.spike_stage import SpikeClassifier
from models.touch_pattern_stage import TouchPatternRefiner


def _ensure_batch_dim(x: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(x) and x.dim() in (1, 2) and x.size(0) != 1:
        return x.unsqueeze(0)
    return x


def _as_batch(x: torch.Tensor | None) -> torch.Tensor | None:
    if x is None:
        return None
    if x.dim() == 0:
        return x.view(1)
    if x.dim() == 1:
        return x.unsqueeze(0)  # [T] → [1, T]
    return x


@dataclass
class StageRecipe:
    name: str
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: Any
    train_loader: Any
    val_loader: Any | None
    step_fn: Callable[[torch.nn.Module, dict[str, Any], torch.device], tuple[torch.Tensor, dict[str, float]]]
    val_fn: Callable[[torch.nn.Module, dict[str, Any], torch.device], dict[str, float]] | None = None
    turn_batches: int = 1
    offload_to_cpu: bool = False


def stage1_step(model: MaiGenerator, batch: dict[str, Any], device: torch.device):
    onset = _as_batch(batch["onset"]).to(device)
    chroma = _as_batch(batch["chroma"]).to(device)
    centroid = _as_batch(batch["centroid"]).to(device)
    tokens = _as_batch(batch["tokens"]).to(device)
    bpm = _as_batch(batch["bpm"]).to(device)
    level = _as_batch(batch["level"]).to(device)
    genre = _as_batch(batch["genre"]).to(device)
    out = model(
        onset,
        chroma,
        centroid,
        tokens,
        bpm,
        level,
        genre,
        distances=batch.get("distances", None).to(device) if batch.get("distances") is not None else None,
        audio_tokens=batch.get("audio_tokens", None).to(device) if batch.get("audio_tokens") is not None else None,
    )
    return out["loss"], {"loss": float(out["loss"].detach().item())}


def touch_step(model: TouchRefiner, batch: dict[str, Any], device: torch.device):
    config_tokens = _as_batch(batch["config_tokens"]).to(device)
    stage1_hidden = _as_batch(batch["stage1_hidden"]).to(device)
    audio_memory = batch.get("audio_memory", None)
    if audio_memory is not None:
        audio_memory = _as_batch(audio_memory).to(device)
    zone_targets = _as_batch(batch["zone_targets"]).to(device)
    logits = model(config_tokens, stage1_hidden, audio_memory=audio_memory)
    loss = model.compute_loss(logits, zone_targets, config_tokens)
    return loss, {"loss": float(loss.detach().item())}


def slide_step(model: SlidePathGenerator, batch: dict[str, Any], device: torch.device):
    target_path = _as_batch(batch["target_path"]).to(device)
    audio_memory = batch.get("audio_memory")
    if audio_memory is None:
        in_dim = getattr(getattr(model, "audio_proj", None), "in_features", 768)
        audio_memory = torch.zeros(target_path.size(0), 1, in_dim)
    audio_memory = _as_batch(audio_memory).to(device)
    start_pos = batch["start_pos"]
    if not torch.is_tensor(start_pos):
        start_pos = torch.as_tensor(start_pos, dtype=torch.long)
    stage1_hidden = batch.get("stage1_hidden")
    if stage1_hidden is not None:
        stage1_hidden = _as_batch(stage1_hidden).to(device)
    onset = batch.get("onset")
    if onset is not None:
        onset = _as_batch(onset).to(device)

    out = model(
        target_path, start_pos, audio_memory,
        stage1_hidden=stage1_hidden,
        onset=onset,
    )
    return out["loss"], {"loss": float(out["loss"].detach().item())}


def break_step(model: BreakClassifier, batch: dict[str, Any], device: torch.device):
    tokens = _as_batch(batch["tokens"]).to(device)
    stage1_hidden = _as_batch(batch["stage1_hidden"]).to(device)
    targets = _as_batch(batch["targets"]).to(device)
    press_mask = _as_batch(batch["press_mask"]).to(device)
    logits = model(tokens, stage1_hidden)
    loss = model.compute_loss(logits, targets, press_mask)
    return loss, {"loss": float(loss.detach().item())}


def spike_step(model: SpikeClassifier, batch: dict[str, Any], device: torch.device):
    tokens = _as_batch(batch["tokens"]).to(device)
    stage1_hidden = _as_batch(batch["stage1_hidden"]).to(device)
    targets = _as_batch(batch["targets"]).to(device)
    touch_mask = _as_batch(batch["touch_mask"]).to(device)
    logits = model(tokens, stage1_hidden)
    loss = model.compute_loss(logits, targets, touch_mask)
    return loss, {"loss": float(loss.detach().item())}


# ── Stage 3: Hold 持续时间预测 ──

def hold_step(model: HoldDurationPredictor, batch: dict[str, Any], device: torch.device):
    tokens = _as_batch(batch["tokens"]).to(device)
    stage1_hidden = _as_batch(batch["stage1_hidden"]).to(device)
    num_targets = _as_batch(batch["dur_num_targets"]).to(device)
    den_targets = _as_batch(batch["dur_den_targets"]).to(device)
    hold_mask = _as_batch(batch["hold_mask"]).bool().to(device)

    audio_memory = batch.get("audio_memory")
    if audio_memory is not None:
        audio_memory = _as_batch(audio_memory).to(device)
    onset = batch.get("onset")
    if onset is not None:
        onset = _as_batch(onset).to(device)

    out = model(tokens, stage1_hidden, audio_memory=audio_memory, onset=onset)
    loss = model.compute_loss(out, num_targets, den_targets, hold_mask)
    return loss, {"loss": float(loss.detach().item())}


# ── Stage 4: Touch Hold 持续时间预测 ──

def touch_hold_step(model: TouchHoldDurationPredictor, batch: dict[str, Any], device: torch.device):
    tokens = _as_batch(batch["tokens"]).to(device)
    stage1_hidden = _as_batch(batch["stage1_hidden"]).to(device)
    num_targets = _as_batch(batch["dur_num_targets"]).to(device)
    den_targets = _as_batch(batch["dur_den_targets"]).to(device)
    hold_mask = _as_batch(batch["touch_hold_mask"]).bool().to(device)

    audio_memory = batch.get("audio_memory")
    if audio_memory is not None:
        audio_memory = _as_batch(audio_memory).to(device)
    onset = batch.get("onset")
    if onset is not None:
        onset = _as_batch(onset).to(device)

    out = model(tokens, stage1_hidden, audio_memory=audio_memory, onset=onset)
    loss = model.compute_loss(out, num_targets, den_targets, hold_mask)
    return loss, {"loss": float(loss.detach().item())}


def touch_pattern_step(model: TouchPatternRefiner, batch: dict[str, Any], device: torch.device):
    tokens = _as_batch(batch["tokens"]).to(device)
    stage1_hidden = _as_batch(batch["stage1_hidden"]).to(device)
    pattern_targets = _as_batch(batch["touch_pattern_targets"]).to(device)
    touch_mask = _as_batch(batch["touch_pattern_mask"]).bool().to(device)

    audio_memory = batch.get("audio_memory")
    if audio_memory is not None:
        audio_memory = _as_batch(audio_memory).to(device)

    logits = model(tokens, stage1_hidden, audio_memory=audio_memory)
    loss = model.compute_loss(logits, pattern_targets, touch_mask)
    return loss, {"loss": float(loss.detach().item())}


# ── Stage 5: Slide Star 精炼 ──

def star_step(model: SlideStarRefiner, batch: dict[str, Any], device: torch.device):
    coarse_path = _as_batch(batch["coarse_path"]).to(device)
    target_path = _as_batch(batch["target_path"]).to(device)
    stage1_hidden = _as_batch(batch["stage1_hidden"]).to(device)

    audio_memory = batch.get("audio_memory")
    if audio_memory is not None:
        audio_memory = _as_batch(audio_memory).to(device)
    onset = batch.get("onset")
    if onset is not None:
        onset = _as_batch(onset).to(device)

    out = model(coarse_path, stage1_hidden,
                audio_memory=audio_memory, onset=onset, target_path=target_path)
    return out["loss"], {"loss": float(out["loss"].detach().item())}
