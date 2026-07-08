from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any

import logging
import torch

from models.stage1 import MaiGenerator
from models.touch_stage import TouchRefiner
from models.slide_stage import SlidePathGenerator, SlideStarRefiner
from models.hold_stage import HoldDurationPredictor, TouchHoldDurationPredictor
from models.break_stage import BreakClassifier
from models.spike_stage import SpikeClassifier
from models.touch_pattern_stage import TouchPatternRefiner
from Tokenizer.slide_star_vocab import SLD_STAR_VOCAB_SIZE


class SkipBatchError(RuntimeError):
    pass


_WARNED_DIM_MISMATCHES: set[tuple[str, str, int, int]] = set()


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
        return x.unsqueeze(0)
    return x


def _fit_last_dim(
    tensor: torch.Tensor | None,
    expected_dim: int | None,
    *,
    stage_name: str,
    field_name: str,
    batch: dict[str, Any],
) -> torch.Tensor | None:
    if tensor is None or expected_dim is None or tensor.dim() == 0:
        return tensor
    actual_dim = int(tensor.size(-1))
    expected_dim = int(expected_dim)
    if actual_dim == expected_dim:
        return tensor

    files = _stage2_star_files(batch)
    warn_key = (stage_name, field_name, actual_dim, expected_dim)
    if warn_key not in _WARNED_DIM_MISMATCHES:
        logging.warning(
            "%s %s dim mismatch: got %d, expected %d; auto-adjusting. files=%s",
            stage_name,
            field_name,
            actual_dim,
            expected_dim,
            files,
        )
        _WARNED_DIM_MISMATCHES.add(warn_key)

    if actual_dim > expected_dim:
        return tensor[..., :expected_dim]

    pad_shape = list(tensor.shape)
    pad_shape[-1] = expected_dim - actual_dim
    pad = torch.zeros(*pad_shape, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, pad], dim=-1)

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
    onset = batch.get("onset")
    if onset is not None:
        onset = _as_batch(onset).to(device)
    zone_targets = _as_batch(batch["zone_targets"]).to(device)
    logits = model(config_tokens, stage1_hidden, audio_memory=audio_memory, onset=onset)
    loss = model.compute_loss(logits, zone_targets, config_tokens)
    return loss, {"loss": float(loss.detach().item())}


def slide_step(model: SlidePathGenerator, batch: dict[str, Any], device: torch.device):
    target_path = _as_batch(batch["target_path"]).to(device)
    # 变长 target_path 已由 collate pad 为 0，替换为正确的 SLD_STAR_PAD
    tp_lengths = batch.get("target_path_lengths")
    if tp_lengths is not None:
        tp_lengths = torch.as_tensor(tp_lengths, dtype=torch.long, device=device)
        max_len = target_path.size(1)
        mask = torch.arange(max_len, device=device).unsqueeze(0) < tp_lengths.unsqueeze(1)
        from Tokenizer.slide_star_vocab import SLD_STAR_PAD
        target_path = torch.where(mask, target_path, torch.full_like(target_path, SLD_STAR_PAD))
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
        stage1_hidden = _fit_last_dim(
            stage1_hidden,
            getattr(getattr(model, "stage1_proj", None), "in_features", None),
            stage_name="stage2_star",
            field_name="stage1_hidden",
            batch=batch,
        )
    onset = batch.get("onset")
    if onset is not None:
        onset = _as_batch(onset).to(device)
        onset = _fit_last_dim(
            onset,
            getattr(getattr(model, "onset_proj", None), "in_features", None),
            stage_name="stage2_star",
            field_name="onset",
            batch=batch,
        )
    audio_memory = _fit_last_dim(
        audio_memory,
        getattr(getattr(model, "audio_proj", None), "in_features", None),
        stage_name="stage2_star",
        field_name="audio_memory",
        batch=batch,
    )
    event_slot = batch.get("slot")
    if event_slot is not None:
        event_slot = torch.as_tensor(event_slot).long()
        if event_slot.dim() == 0:
            event_slot = event_slot.unsqueeze(0)
        event_slot = event_slot.to(device)
    _validate_stage2_star_batch(model, batch, target_path, start_pos, audio_memory, stage1_hidden, onset)

    out = model(
        target_path, start_pos, audio_memory,
        stage1_hidden=stage1_hidden,
        onset=onset,
        event_slot=event_slot,
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


def _stage2_star_files(batch: dict[str, Any]) -> str:
    files = batch.get("_file")
    if isinstance(files, (list, tuple)):
        return ", ".join(str(x) for x in files[:4])
    if files is None:
        return "<unknown>"
    return str(files)


def _validate_stage2_star_batch(
    model: SlidePathGenerator,
    batch: dict[str, Any],
    target_path: torch.Tensor,
    start_pos: torch.Tensor,
    audio_memory: torch.Tensor,
    stage1_hidden: torch.Tensor | None,
    onset: torch.Tensor | None,
) -> None:
    files = _stage2_star_files(batch)
    if target_path.dtype != torch.long:
        raise ValueError(f"stage2_star target_path dtype must be long, got {target_path.dtype} from {files}")
    if start_pos.dtype != torch.long:
        raise ValueError(f"stage2_star start_pos dtype must be long, got {start_pos.dtype} from {files}")
    if target_path.dim() != 2:
        raise ValueError(f"stage2_star target_path must be [B, T], got {tuple(target_path.shape)} from {files}")

    target_min = int(target_path.min().item())
    target_max = int(target_path.max().item())
    if target_min < 0 or target_max >= SLD_STAR_VOCAB_SIZE:
        raise ValueError(
            f"stage2_star target_path out of range [{target_min}, {target_max}] "
            f"for vocab_size={SLD_STAR_VOCAB_SIZE} from {files}"
        )

    start_flat = start_pos.reshape(-1)
    start_min = int(start_flat.min().item())
    start_max = int(start_flat.max().item())
    if start_min < 1 or start_max > 8:
        raise ValueError(f"stage2_star start_pos out of range [{start_min}, {start_max}] from {files}")

    slot_value = batch.get("slot")
    if slot_value is not None:
        slot_tensor = torch.as_tensor(slot_value).reshape(-1)
        slot_min = int(slot_tensor.min().item())
        slot_max = int(slot_tensor.max().item())
        if slot_min < 0:
            raise ValueError(f"stage2_star slot out of range [{slot_min}, {slot_max}] from {files}")
        slot_capacity = getattr(getattr(model, "event_slot_embed", None), "num_embeddings", None)
        if slot_capacity is not None and slot_max >= slot_capacity:
            raise SkipBatchError(
                f"stage2_star slot too large: max_slot={slot_max} exceeds event_slot capacity={slot_capacity} "
                f"from {files}"
            )
        if stage1_hidden is not None and slot_max >= int(stage1_hidden.size(1)):
            raise SkipBatchError(
                f"stage2_star slot misaligned with stage1_hidden: max_slot={slot_max} >= stage1_hidden_len={stage1_hidden.size(1)} "
                f"from {files}"
            )
        if stage1_hidden is None and onset is not None and onset.dim() >= 2 and slot_max >= int(onset.size(1)):
            raise SkipBatchError(
                f"stage2_star slot misaligned with onset: max_slot={slot_max} >= onset_len={onset.size(1)} "
                f"from {files}"
            )

    pos_capacity = getattr(getattr(model, "pos_embed", None), "num_embeddings", None)
    if pos_capacity is not None and target_path.size(1) - 1 >= pos_capacity:
        raise SkipBatchError(
            f"stage2_star target_path too long: len={target_path.size(1)} exceeds pos_embed capacity={pos_capacity} "
            f"from {files}"
        )

    for name, tensor in (
        ("audio_memory", audio_memory),
        ("stage1_hidden", stage1_hidden),
        ("onset", onset),
    ):
        if tensor is not None and not torch.isfinite(tensor).all():
            raise ValueError(f"stage2_star {name} contains NaN/Inf from {files}")


# 鈹€鈹€ Stage 3: Hold 鎸佺画鏃堕棿棰勬祴 鈹€鈹€

def hold_step(model: HoldDurationPredictor, batch: dict[str, Any], device: torch.device):
    tokens = _as_batch(batch["tokens"]).to(device)
    stage1_hidden = _as_batch(batch["stage1_hidden"]).to(device)
    stage1_hidden = _fit_last_dim(
        stage1_hidden,
        getattr(getattr(model, "stage1_proj", None), "in_features", None),
        stage_name="hold",
        field_name="stage1_hidden",
        batch=batch,
    )
    row_targets = torch.as_tensor(batch.get("dur_rows_target", batch.get("dur_num_target")))
    if row_targets.dim() == 0:
        row_targets = row_targets.unsqueeze(0)
    row_targets = row_targets.to(device)
    query_slots = torch.as_tensor(batch["query_slot"]).long()
    if query_slots.dim() == 0:
        query_slots = query_slots.unsqueeze(0)
    query_slots = query_slots.to(device)

    audio_memory = batch.get("audio_memory")
    if audio_memory is not None:
        audio_memory = _as_batch(audio_memory).to(device)
        audio_memory = _fit_last_dim(
            audio_memory,
            getattr(getattr(model, "audio_proj", None), "in_features", None),
            stage_name="hold",
            field_name="audio_memory",
            batch=batch,
        )
    onset = batch.get("onset")
    if onset is not None:
        onset = _as_batch(onset).to(device)
        onset = _fit_last_dim(
            onset,
            getattr(getattr(model, "onset_proj", None), "in_features", None),
            stage_name="hold",
            field_name="onset",
            batch=batch,
        )

    out = model(tokens, stage1_hidden, query_slots, audio_memory=audio_memory, onset=onset)
    loss = model.compute_loss(out, row_targets)
    return loss, {"loss": float(loss.detach().item())}


# 鈹€鈹€ Stage 4: Touch Hold 鎸佺画鏃堕棿棰勬祴 鈹€鈹€

def touch_hold_step(model: TouchHoldDurationPredictor, batch: dict[str, Any], device: torch.device):
    tokens = _as_batch(batch["tokens"]).to(device)
    stage1_hidden = _as_batch(batch["stage1_hidden"]).to(device)
    stage1_hidden = _fit_last_dim(
        stage1_hidden,
        getattr(getattr(model, "stage1_proj", None), "in_features", None),
        stage_name="touch_hold",
        field_name="stage1_hidden",
        batch=batch,
    )
    row_targets = torch.as_tensor(batch.get("dur_rows_target", batch.get("dur_num_target")))
    if row_targets.dim() == 0:
        row_targets = row_targets.unsqueeze(0)
    row_targets = row_targets.to(device)
    query_slots = torch.as_tensor(batch["query_slot"]).long()
    if query_slots.dim() == 0:
        query_slots = query_slots.unsqueeze(0)
    query_slots = query_slots.to(device)

    audio_memory = batch.get("audio_memory")
    if audio_memory is not None:
        audio_memory = _as_batch(audio_memory).to(device)
        audio_memory = _fit_last_dim(
            audio_memory,
            getattr(getattr(model, "audio_proj", None), "in_features", None),
            stage_name="touch_hold",
            field_name="audio_memory",
            batch=batch,
        )
    onset = batch.get("onset")
    if onset is not None:
        onset = _as_batch(onset).to(device)
        onset = _fit_last_dim(
            onset,
            getattr(getattr(model, "onset_proj", None), "in_features", None),
            stage_name="touch_hold",
            field_name="onset",
            batch=batch,
        )

    out = model(tokens, stage1_hidden, query_slots, audio_memory=audio_memory, onset=onset)
    loss = model.compute_loss(out, row_targets)
    return loss, {"loss": float(loss.detach().item())}


def touch_pattern_step(model: TouchPatternRefiner, batch: dict[str, Any], device: torch.device):
    tokens = _as_batch(batch["tokens"]).to(device)
    stage1_hidden = _as_batch(batch["stage1_hidden"]).to(device)
    pattern_targets = _as_batch(batch["touch_pattern_targets"]).to(device)
    touch_mask = _as_batch(batch["touch_pattern_mask"]).bool().to(device)

    audio_memory = batch.get("audio_memory")
    if audio_memory is not None:
        audio_memory = _as_batch(audio_memory).to(device)
    onset = batch.get("onset")
    if onset is not None:
        onset = _as_batch(onset).to(device)

    logits = model(tokens, stage1_hidden, audio_memory=audio_memory, onset=onset)
    loss = model.compute_loss(logits, pattern_targets, touch_mask)
    return loss, {"loss": float(loss.detach().item())}


# 鈹€鈹€ Stage 5: Slide Star 绮剧偧 鈹€鈹€

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

