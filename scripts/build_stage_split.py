from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from mai_parser import Difficulty, SongDataset


def _chart_level(chart) -> float:
    return float(getattr(chart, "level_value", 0.0) or 0.0)


def _song_payload(song) -> dict[str, Any]:
    charts = []
    for idx, chart in sorted(song.charts.items()):
        charts.append({
            "chart_id": f"{song.song_id}_lv{idx}",
            "song_id": song.song_id,
            "difficulty_index": idx,
            "level": _chart_level(chart),
            "note_count": int(getattr(chart, "note_count", 0) or 0),
            "has_touch": int(getattr(chart, "touch_count", 0) or 0) > 0,
            "has_slide": int(getattr(chart, "slide_count", 0) or 0) > 0,
            "has_break": int(getattr(chart, "break_count", 0) or 0) > 0,
        })
    return {
        "song_id": song.song_id,
        "title": song.title_clean or song.title,
        "artist": song.artist,
        "genre": song.genre,
        "bpm": float(getattr(song, "bpm", 0.0) or 0.0),
        "cabinet": getattr(song.cabinet, "value", str(song.cabinet)),
        "charts": charts,
    }


def _chart_ok_for_val(chart, max_val_level: float) -> bool:
    level = _chart_level(chart)
    return level < max_val_level


def _eligible_chart_count(song) -> int:
    return sum(1 for chart in song.charts.values() if _chart_level(chart) > 0)


def build_split(dataset_root: str | Path, val_ratio: float, seed: int, max_val_level: float) -> dict[str, Any]:
    ds = SongDataset(dataset_root).load(progress=True)
    songs = list(ds.songs)
    rng = random.Random(seed)

    high_level = []
    normal = []
    for song in songs:
        has_hi = any(_chart_level(chart) >= max_val_level for chart in song.charts.values())
        if has_hi:
            high_level.append(song)
        else:
            normal.append(song)

    rng.shuffle(normal)
    rng.shuffle(high_level)

    val_target = max(1, int(round(len(songs) * val_ratio))) if songs else 0
    val_songs = []
    train_songs = []

    for song in normal:
        if len(val_songs) < val_target:
            val_songs.append(song)
        else:
            train_songs.append(song)

    # High-level charts are pinned to train when possible.
    train_songs.extend(high_level)

    train_payload = [_song_payload(s) for s in train_songs]
    val_payload = [_song_payload(s) for s in val_songs]

    return {
        "dataset_root": str(Path(dataset_root).resolve()),
        "seed": seed,
        "val_ratio": val_ratio,
        "max_val_level": max_val_level,
        "train_songs": train_payload,
        "val_songs": val_payload,
        "summary": {
            "train_songs": len(train_payload),
            "val_songs": len(val_payload),
            "train_charts": sum(len(s["charts"]) for s in train_payload),
            "val_charts": sum(len(s["charts"]) for s in val_payload),
            "train_charts_14p": sum(
                1 for s in train_payload for c in s["charts"] if c["level"] >= max_val_level
            ),
            "val_charts_14p": sum(
                1 for s in val_payload for c in s["charts"] if c["level"] >= max_val_level
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="datasets")
    parser.add_argument("--out", default="splits/train_val_split.json")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-val-level", type=float, default=14.0)
    args = parser.parse_args()

    payload = build_split(args.dataset_root, args.val_ratio, args.seed, args.max_val_level)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
