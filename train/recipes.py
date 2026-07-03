from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any

import torch

from models.stage1 import MaiGenerator
from models.touch_stage import TouchRefiner
from models.slide_stage import SlidePathGenerator
from models.break_stage import BreakClassifier
from models.spike_stage import SpikeClassifier


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
    audio_memory = _as_batch(batch["audio_memory"]).to(device)
    target_path = _as_batch(batch["target_path"]).to(device)
    out = model(
        target_path,
        batch["start_pos"].to(device),
        batch["end_pos"].to(device),
        batch["duration"].to(device),
        audio_memory,
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
