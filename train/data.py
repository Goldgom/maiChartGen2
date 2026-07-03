from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import logging

from Tokenizer.MaiChartTokenizer import BOS, EOS
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


@dataclass
class CacheSample:
    fields: dict[str, Any]


class StageCacheDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        stage: str,
        max_tokens: int | None = None,
        max_onset: int | None = None,
    ):
        self.root = Path(root)
        self.stage = stage
        stage_root = self.root / stage
        self.items = sorted(stage_root.glob("*.pt")) if stage_root.exists() else []
        self._slide_audio_cache: dict[str, torch.Tensor] = {}
        if max_tokens is not None or max_onset is not None:
            self.items = self._filter_by_length(self.items, max_tokens, max_onset)

    def _filter_by_length(
        self,
        items: list[Path],
        max_tokens: int | None,
        max_onset: int | None,
    ) -> list[Path]:
        kept: list[Path] = []
        skipped = 0
        for fp in items:
            try:
                data = torch.load(fp, map_location="cpu", weights_only=True)
            except Exception:
                kept.append(fp)
                continue
            seq = data.get("tokens", data.get("config_tokens", data.get("target_path")))
            tok_len = int(seq.size(-1)) if torch.is_tensor(seq) and seq.dim() >= 1 else 0
            onset = data.get("onset")
            onset_len = int(onset.size(-1)) if torch.is_tensor(onset) and onset.dim() >= 1 else 0
            too_long_tokens = max_tokens is not None and tok_len > max_tokens
            too_long_onset = max_onset is not None and onset_len > max_onset
            if too_long_tokens or too_long_onset:
                skipped += 1
                continue
            kept.append(fp)
        if skipped:
            logger.warning(
                "Stage '%s': skipped %d/%d over-length cache files (max_tokens=%s, max_onset=%s)",
                self.stage,
                skipped,
                len(items),
                max_tokens,
                max_onset,
            )
        return kept

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        data = torch.load(self.items[idx], map_location="cpu", weights_only=True)
        if self.stage == "slide":
            song_id = self.items[idx].stem.rsplit("_", 1)[0]
            audio_memory = self._slide_audio_cache.get(song_id)
            if audio_memory is None:
                hidden_path = self.root / "_hidden" / f"{song_id}.pt"
                if hidden_path.exists():
                    hidden = torch.load(hidden_path, map_location="cpu", weights_only=True)
                    audio_memory = hidden.get("audio_memory")
                    if torch.is_tensor(audio_memory):
                        self._slide_audio_cache[song_id] = audio_memory
            if torch.is_tensor(audio_memory):
                data["audio_memory"] = audio_memory
        data["_file"] = str(self.items[idx])  # OOM 调试用
        return data


def ensure_stage_cache(root: str | Path, stage: str) -> Path:
    root = Path(root)
    stage_root = root / stage
    stage_root.mkdir(parents=True, exist_ok=True)
    return stage_root


def build_toy_cache_sample(stage: str) -> dict[str, Any]:
    if stage == "stage1":
        return {
            "onset": torch.zeros(16),
            "chroma": torch.zeros(16, 12),
            "centroid": torch.zeros(16),
            "tokens": torch.tensor([BOS, 120, 121, EOS], dtype=torch.long),
            "bpm": torch.tensor([180.0]),
            "level": torch.tensor([12.0]),
            "genre": torch.tensor([0.0]),
        }
    if stage == "touch":
        return {
            "config_tokens": torch.tensor([1, 200, 201, 2], dtype=torch.long),
            "stage1_hidden": torch.zeros(4, 256),
            "audio_memory": torch.zeros(16, 256),  # 音频特征
            "zone_targets": torch.tensor(
                [
                    [0] * 33,
                    [1] + [0] * 32,
                    [0] * 33,
                    [0] * 33,
                ],
                dtype=torch.long,
            ),
        }
    if stage == "slide":
        return {
            "onset": torch.zeros(16),
            "chroma": torch.zeros(16, 12),
            "centroid": torch.zeros(16),
            "target_path": torch.tensor([1, 42, 43, 44, 2], dtype=torch.long),
            "start_pos": torch.tensor([1]),
            "end_pos": torch.tensor([5]),
            "duration": torch.tensor([[4.0, 1.0]]),
        }
    if stage == "break":
        return {
            "tokens": torch.zeros(8, dtype=torch.long),
            "stage1_hidden": torch.zeros(8, 384),
            "targets": torch.zeros(8, 8, dtype=torch.long),
            "press_mask": torch.zeros(8, 8, dtype=torch.bool),
        }
    if stage == "spike":
        return {
            "tokens": torch.zeros(8, dtype=torch.long),
            "stage1_hidden": torch.zeros(8, 384),
            "targets": torch.zeros(8, 33, dtype=torch.long),
            "touch_mask": torch.zeros(8, 33, dtype=torch.bool),
        }
    raise ValueError(stage)


def write_toy_cache(root: str | Path, stage: str) -> Path:
    stage_root = ensure_stage_cache(root, stage)
    sample_path = stage_root / "toy_000.pt"
    torch.save(build_toy_cache_sample(stage), sample_path)
    return stage_root


def default_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    keys = batch[0].keys()
    out: dict[str, Any] = {}
    for key in keys:
        values = [item[key] for item in batch]
        if torch.is_tensor(values[0]):
            out[key] = torch.stack(values, dim=0)
        else:
            out[key] = values
    return out


def build_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int = 0, collate_fn=default_collate):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn, pin_memory=torch.cuda.is_available())
