from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

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
                        # ── NaN 检查：发现则替换为 0 ──
                        if torch.isnan(audio_memory).any():
                            logger.warning(
                                "Slide audio_memory 含 NaN: _hidden/%s.pt，已替换为 0",
                                song_id,
                            )
                            audio_memory = torch.nan_to_num(audio_memory, nan=0.0)
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


# ── Train / Val Split ─────────────────────────────────────────────────────

def _extract_song_id(filepath: Path, stage: str) -> str:
    """从缓存文件路径中提取 song_id。"""
    stem = filepath.stem
    if stage == "stage1":
        return stem
    # touch/break/spike: {song_id}_{idx}, slide: {song_id}_{idx}
    return stem.rsplit("_", 1)[0]


def _get_level_from_cache(fp: Path) -> float:
    """从 stage1 缓存文件中读取 level。"""
    try:
        data = torch.load(fp, map_location="cpu", weights_only=True)
        level = data.get("level")
        if torch.is_tensor(level):
            return float(level.item())
        return float(level) if level is not None else 0.0
    except Exception:
        return 0.0


def build_song_level_map(cache_root: str | Path) -> dict[str, float]:
    """扫描 stage1 缓存，建立 song_id → level 映射。"""
    cache_root = Path(cache_root)
    s1_dir = cache_root / "stage1"
    if not s1_dir.exists():
        return {}

    level_map: dict[str, float] = {}
    for fp in sorted(s1_dir.glob("*.pt")):
        song_id = _extract_song_id(fp, "stage1")
        level_map[song_id] = _get_level_from_cache(fp)
    return level_map


def make_train_val_split(
    cache_root: str | Path,
    val_level_threshold: float = 14.0,
    val_ratio: float = 0.10,
    seed: int = 42,
    split_file: str | Path | None = None,
) -> tuple[set[str], set[str]]:
    """
    根据难度等级和比例划分训练集 / 验证集。

    规则:
      1. 从 level < val_level_threshold 的歌中随机抽取 val_ratio 比例作为验证集。
      2. 如果提供了 split_file (JSON)，则直接从文件读取划分。

    Returns:
        (train_ids, val_ids)  两个 song_id 集合。
    """
    cache_root = Path(cache_root)

    # 优先从 JSON 文件读取
    if split_file is not None:
        split_path = Path(split_file)
        if split_path.exists():
            data = json.loads(split_path.read_text(encoding="utf-8"))
            train_ids = {s["song_id"] for s in data.get("train_songs", [])}
            val_ids = {s["song_id"] for s in data.get("val_songs", [])}
            logger.info(
                "从 %s 加载划分: train=%d, val=%d",
                split_path, len(train_ids), len(val_ids),
            )
            return train_ids, val_ids
        logger.warning("split_file 不存在: %s，将自动划分", split_path)

    # 自动划分
    level_map = build_song_level_map(cache_root)
    if not level_map:
        logger.warning("stage1 缓存为空，无法划分 train/val")
        return set(), set()

    # 低难度候选
    low_level = [sid for sid, lv in level_map.items() if lv < val_level_threshold]
    high_level = [sid for sid, lv in level_map.items() if lv >= val_level_threshold]

    rng = random.Random(seed)
    rng.shuffle(low_level)

    val_count = max(1, int(round(len(low_level) * val_ratio)))
    val_ids = set(low_level[:val_count])
    train_ids = set(low_level[val_count:]) | set(high_level)

    logger.info(
        "自动划分: train=%d (≥Lv%.0f=%d), val=%d (全 < Lv%.0f) (val_ratio=%.0f%%)",
        len(train_ids), val_level_threshold, len(high_level),
        len(val_ids), val_level_threshold, val_ratio * 100,
    )
    return train_ids, val_ids


class SplitStageDataset(Dataset):
    """
    按 song_id 划分的训练集/验证集 Dataset，封装 StageCacheDataset。
    """

    def __init__(
        self,
        root: str | Path,
        stage: str,
        song_ids: set[str],
        max_tokens: int | None = None,
        max_onset: int | None = None,
    ):
        self.root = Path(root)
        self.stage = stage
        stage_root = self.root / stage
        all_items = sorted(stage_root.glob("*.pt")) if stage_root.exists() else []

        # 按 song_id 过滤
        self.items = [
            fp for fp in all_items
            if _extract_song_id(fp, stage) in song_ids
        ]

        self._slide_audio_cache: dict[str, torch.Tensor] = {}

        if max_tokens is not None or max_onset is not None:
            self.items = self._filter_by_length(self.items, max_tokens, max_onset)

        logger.info(
            "SplitStageDataset[%s]: %d samples (from %d songs)",
            stage, len(self.items), len(song_ids),
        )

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
                "SplitStageDataset[%s]: skipped %d/%d over-length files",
                self.stage, skipped, len(items),
            )
        return kept

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        data = torch.load(self.items[idx], map_location="cpu", weights_only=True)
        if self.stage == "slide":
            song_id = _extract_song_id(self.items[idx], self.stage)
            audio_memory = self._slide_audio_cache.get(song_id)
            if audio_memory is None:
                hidden_path = self.root / "_hidden" / f"{song_id}.pt"
                if hidden_path.exists():
                    hidden = torch.load(hidden_path, map_location="cpu", weights_only=True)
                    audio_memory = hidden.get("audio_memory")
                    if torch.is_tensor(audio_memory):
                        # ── NaN 检查 ──
                        if torch.isnan(audio_memory).any():
                            logger.warning(
                                "SplitStageDataset[slide] audio_memory 含 NaN: _hidden/%s.pt，已替换",
                                song_id,
                            )
                            audio_memory = torch.nan_to_num(audio_memory, nan=0.0)
                        self._slide_audio_cache[song_id] = audio_memory
            if torch.is_tensor(audio_memory):
                data["audio_memory"] = audio_memory
        data["_file"] = str(self.items[idx])
        return data

    def num_songs(self) -> int:
        """返回不重复的 song_id 数量。"""
        return len({_extract_song_id(fp, self.stage) for fp in self.items})
