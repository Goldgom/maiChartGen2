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


def _build_onset_features(data: dict[str, Any]) -> torch.Tensor | None:
    onset = data.get("onset")
    if not torch.is_tensor(onset):
        return None
    onset = onset.float().view(-1)

    centroid = data.get("centroid")
    if torch.is_tensor(centroid):
        centroid_feat = centroid.float().view(-1)
    else:
        centroid_feat = torch.zeros_like(onset)

    chroma = data.get("chroma")
    if torch.is_tensor(chroma) and chroma.dim() >= 2:
        chroma_feat = chroma.float().view(chroma.size(0), -1).mean(dim=-1)
    else:
        chroma_feat = torch.zeros_like(onset)

    target_len = min(onset.numel(), centroid_feat.numel(), chroma_feat.numel())
    if target_len <= 0:
        return None
    return torch.stack(
        [onset[:target_len], centroid_feat[:target_len], chroma_feat[:target_len]],
        dim=-1,
    )


def _needs_onset_upgrade(data: dict[str, Any]) -> bool:
    onset = data.get("onset")
    return torch.is_tensor(onset) and onset.dim() == 1


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
        self.items = self._filter_supervised(self.items)
        if max_tokens is not None or max_onset is not None:
            self.items = self._filter_by_length(self.items, max_tokens, max_onset)

    def _load_hidden_features(self, fp: Path) -> dict[str, torch.Tensor]:
        chart_id = _extract_chart_id(fp, self.stage)
        hidden_path = self.root / "_hidden" / f"{chart_id}.pt"
        if hidden_path.exists():
            hidden = torch.load(hidden_path, map_location="cpu", weights_only=True)
            out: dict[str, torch.Tensor] = {}
            if torch.is_tensor(hidden.get("stage1_hidden")):
                out["stage1_hidden"] = hidden["stage1_hidden"]
            if torch.is_tensor(hidden.get("audio_memory")):
                out["audio_memory"] = hidden["audio_memory"]
            return out
        return {}

    def _load_stage1_fields(self, fp: Path) -> dict[str, torch.Tensor]:
        chart_id = _extract_chart_id(fp, self.stage)
        stage1_path = self.root / "stage1" / f"{chart_id}.pt"
        if not stage1_path.exists():
            return {}
        data = torch.load(stage1_path, map_location="cpu", weights_only=True)
        out: dict[str, torch.Tensor] = {}
        onset_features = _build_onset_features(data)
        if torch.is_tensor(onset_features):
            out["onset"] = onset_features
        return out

    def _has_supervision(self, data: dict[str, Any]) -> bool:
        if self.stage == "hold":
            return "query_slot" in data and int(torch.as_tensor(data.get("dur_rows_target", 0)).item()) > 0
        if self.stage == "touch_hold":
            return "query_slot" in data and int(torch.as_tensor(data.get("dur_rows_target", 0)).item()) > 0
        if self.stage == "stage5_touch":
            return bool(torch.as_tensor(data.get("touch_pattern_mask", False)).bool().any().item())
        if self.stage == "touch":
            return bool(torch.as_tensor(data.get("zone_targets", 0)).gt(0).any().item())
        if self.stage in {"break", "stage6_break_note"}:
            return bool(torch.as_tensor(data.get("press_mask", False)).bool().any().item())
        if self.stage in {"spike", "stage7_firework_note"}:
            return bool(torch.as_tensor(data.get("touch_mask", False)).bool().any().item())
        return True

    def _filter_supervised(self, items: list[Path]) -> list[Path]:
        kept: list[Path] = []
        skipped = 0
        for fp in items:
            try:
                data = torch.load(fp, map_location="cpu", weights_only=True)
            except Exception:
                kept.append(fp)
                continue
            if self._has_supervision(data):
                kept.append(fp)
            else:
                skipped += 1
        if skipped:
            logger.info("Stage '%s': skipped %d files without supervision", self.stage, skipped)
        return kept

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
        if self.stage in {"slide", "stage2_star"}:
            chart_id = _extract_chart_id(self.items[idx], self.stage)
            song_id = _strip_lv_suffix(chart_id)
            audio_memory = self._slide_audio_cache.get(song_id)
            if audio_memory is None:
                # 查找该歌曲任意图表的 hidden（audio_memory 全曲共享）
                found = list((self.root / "_hidden").glob(f"{song_id}_lv*.pt"))
                if found:
                    hidden = torch.load(found[0], map_location="cpu", weights_only=True)
                    audio_memory = hidden.get("audio_memory")
                # fallback: 旧格式 _hidden/{song_id}.pt
                elif (self.root / "_hidden" / f"{song_id}.pt").exists():
                    hidden = torch.load(self.root / "_hidden" / f"{song_id}.pt", map_location="cpu", weights_only=True)
                    audio_memory = hidden.get("audio_memory")
                if torch.is_tensor(audio_memory):
                    if torch.isnan(audio_memory).any():
                        logger.warning(
                            "Slide audio_memory 含 NaN: _hidden/%s，已替换为 0", song_id,
                        )
                        audio_memory = torch.nan_to_num(audio_memory, nan=0.0)
                    self._slide_audio_cache[song_id] = audio_memory
            if torch.is_tensor(audio_memory):
                data["audio_memory"] = audio_memory
            if "onset" not in data or _needs_onset_upgrade(data):
                data.update(self._load_stage1_fields(self.items[idx]))
            if "stage1_hidden" not in data:
                data.update(self._load_hidden_features(self.items[idx]))
        elif self.stage in {"touch", "stage5_touch"}:
            if "onset" not in data or _needs_onset_upgrade(data):
                data.update(self._load_stage1_fields(self.items[idx]))
        elif self.stage in {"hold", "touch_hold"} and (
            "stage1_hidden" not in data or "audio_memory" not in data
        ):
            data.update(self._load_hidden_features(self.items[idx]))
            if "onset" not in data or _needs_onset_upgrade(data):
                data.update(self._load_stage1_fields(self.items[idx]))
        data["_file"] = str(self.items[idx])
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
    """从缓存文件路径中提取原始 song_id（去掉 _lv{N} 后缀）。"""
    stem = filepath.stem
    # stage1: {song_id}_lv{N}
    # touch/break/spike/hold/touch_hold: {song_id}_lv{N}_{idx}
    # slide/stage2_star: {song_id}_lv{N}_{idx:03d}
    # 先去掉最后的 _{idx} / _{idx:03d}
    if stage in {"hold", "touch_hold", "slide", "stage2_star"}:
        stem = stem.rsplit("_", 1)[0]
    # 再去掉 _lv{N}
    return _strip_lv_suffix(stem)


def _extract_chart_id(filepath: Path, stage: str) -> str:
    """从缓存文件路径中提取完整 chart_id。slide/stage2_star 返回 {song_id}_lv{N}，其余返回 {song_id}_lv{N}_{idx}。"""
    stem = filepath.stem
    # slide/stage2_star: {song_id}_lv{N}_{idx:03d} → 去掉最后一个 _{idx}
    # touch/break/spike/hold/touch_hold: {song_id}_lv{N}_{idx} → chart_id 包含 _idx
    if stage in ("hold", "touch_hold", "slide", "stage2_star"):
        stem = stem.rsplit("_", 1)[0]
    return stem


def _strip_lv_suffix(stem: str) -> str:
    """去掉 _lv{N} 后缀，返回原始 song_id。"""
    import re
    m = re.search(r'^(.*)_lv(\d+)$', stem)
    return m.group(1) if m else stem


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
    """扫描 stage1 缓存，建立 chart_id → level 映射。"""
    cache_root = Path(cache_root)
    s1_dir = cache_root / "stage1"
    if not s1_dir.exists():
        return {}

    level_map: dict[str, float] = {}
    for fp in sorted(s1_dir.glob("*_lv*.pt")):
        chart_id = _extract_chart_id(fp, "stage1")
        level_map[chart_id] = _get_level_from_cache(fp)
    return level_map


def make_train_val_split(
    cache_root: str | Path,
    val_level_threshold: float = 14.0,
    val_ratio: float = 0.10,
    seed: int = 42,
    split_file: str | Path | None = None,
) -> tuple[set[str], set[str]]:
    """
    根据难度等级和比例划分训练集 / 验证集（chart 级别）。

    Returns:
        (train_ids, val_ids)  两个 chart_id 集合 ({song_id}_lv{N})。
    """
    cache_root = Path(cache_root)

    # 优先从 JSON 文件读取
    if split_file is not None:
        split_path = Path(split_file)
        if split_path.exists():
            data = json.loads(split_path.read_text(encoding="utf-8"))
            train_ids = {s.get("chart_id", s.get("song_id")) for s in data.get("train_songs", [])}
            val_ids = {s.get("chart_id", s.get("song_id")) for s in data.get("val_songs", [])}
            logger.info(
                "从 %s 加载划分: train=%d, val=%d",
                split_path, len(train_ids), len(val_ids),
            )
            return train_ids, val_ids
        logger.warning("split_file 不存在: %s，将自动划分", split_path)

    # 自动划分（chart 级别）
    level_map = build_song_level_map(cache_root)
    if not level_map:
        logger.warning("stage1 缓存为空，无法划分 train/val")
        return set(), set()

    low_level = [cid for cid, lv in level_map.items() if lv < val_level_threshold]
    high_level = [cid for cid, lv in level_map.items() if lv >= val_level_threshold]

    rng = random.Random(seed)
    rng.shuffle(low_level)

    val_count = max(1, int(round(len(low_level) * val_ratio)))
    val_ids = set(low_level[:val_count])
    train_ids = set(low_level[val_count:]) | set(high_level)

    logger.info(
        "自动划分: train=%d charts (≥Lv%.0f=%d), val=%d charts (全 < Lv%.0f) (val_ratio=%.0f%%)",
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

        # 按 chart_id 过滤
        self.items = [
            fp for fp in all_items
            if _extract_chart_id(fp, stage) in song_ids
        ]

        self._slide_audio_cache: dict[str, torch.Tensor] = {}
        self.items = self._filter_supervised(self.items)

        if max_tokens is not None or max_onset is not None:
            self.items = self._filter_by_length(self.items, max_tokens, max_onset)

        logger.info(
            "SplitStageDataset[%s]: %d samples (from %d songs)",
            stage, len(self.items), len(song_ids),
        )

    def _has_supervision(self, data: dict[str, Any]) -> bool:
        return StageCacheDataset._has_supervision(self, data)

    def _filter_supervised(self, items: list[Path]) -> list[Path]:
        return StageCacheDataset._filter_supervised(self, items)

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
        if self.stage in {"slide", "stage2_star"}:
            chart_id = _extract_chart_id(self.items[idx], self.stage)
            song_id = _strip_lv_suffix(chart_id)
            audio_memory = self._slide_audio_cache.get(song_id)
            if audio_memory is None:
                found = list((self.root / "_hidden").glob(f"{song_id}_lv*.pt"))
                if found:
                    hidden = torch.load(found[0], map_location="cpu", weights_only=True)
                    audio_memory = hidden.get("audio_memory")
                elif (self.root / "_hidden" / f"{song_id}.pt").exists():
                    hidden = torch.load(self.root / "_hidden" / f"{song_id}.pt", map_location="cpu", weights_only=True)
                    audio_memory = hidden.get("audio_memory")
                if torch.is_tensor(audio_memory):
                    if torch.isnan(audio_memory).any():
                        logger.warning(
                            "SplitStageDataset[slide] audio_memory 含 NaN: _hidden/%s，已替换", song_id,
                        )
                        audio_memory = torch.nan_to_num(audio_memory, nan=0.0)
                    self._slide_audio_cache[song_id] = audio_memory
            if torch.is_tensor(audio_memory):
                data["audio_memory"] = audio_memory
            if "onset" not in data or _needs_onset_upgrade(data):
                stage1_path = self.root / "stage1" / f"{_extract_chart_id(self.items[idx], self.stage)}.pt"
                if stage1_path.exists():
                    s1 = torch.load(stage1_path, map_location="cpu", weights_only=True)
                    onset_features = _build_onset_features(s1)
                    if torch.is_tensor(onset_features):
                        data["onset"] = onset_features
            if "stage1_hidden" not in data:
                hidden_path = self.root / "_hidden" / f"{_extract_chart_id(self.items[idx], self.stage)}.pt"
                if hidden_path.exists():
                    hidden = torch.load(hidden_path, map_location="cpu", weights_only=True)
                    if torch.is_tensor(hidden.get("stage1_hidden")):
                        data["stage1_hidden"] = hidden["stage1_hidden"]
        elif self.stage in {"touch", "stage5_touch"}:
            stage1_path = self.root / "stage1" / f"{_extract_chart_id(self.items[idx], self.stage)}.pt"
            if ("onset" not in data or _needs_onset_upgrade(data)) and stage1_path.exists():
                s1 = torch.load(stage1_path, map_location="cpu", weights_only=True)
                onset_features = _build_onset_features(s1)
                if torch.is_tensor(onset_features):
                    data["onset"] = onset_features
        elif self.stage in {"hold", "touch_hold"} and (
            "stage1_hidden" not in data or "audio_memory" not in data
        ):
            hidden_path = self.root / "_hidden" / f"{_extract_chart_id(self.items[idx], self.stage)}.pt"
            if hidden_path.exists():
                hidden = torch.load(hidden_path, map_location="cpu", weights_only=True)
                if torch.is_tensor(hidden.get("stage1_hidden")):
                    data["stage1_hidden"] = hidden["stage1_hidden"]
                if torch.is_tensor(hidden.get("audio_memory")):
                    data["audio_memory"] = hidden["audio_memory"]
            stage1_path = self.root / "stage1" / f"{_extract_chart_id(self.items[idx], self.stage)}.pt"
            if ("onset" not in data or _needs_onset_upgrade(data)) and stage1_path.exists():
                s1 = torch.load(stage1_path, map_location="cpu", weights_only=True)
                onset_features = _build_onset_features(s1)
                if torch.is_tensor(onset_features):
                    data["onset"] = onset_features
        data["_file"] = str(self.items[idx])
        return data

    def num_songs(self) -> int:
        """返回不重复的 song_id 数量。"""
        return len({_extract_song_id(fp, self.stage) for fp in self.items})
