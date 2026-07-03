"""
Dataset loader for maimai training data.

Provides:
- SongDataset: scans datasets/ folder, parses all songs
- ChartDataset: PyTorch Dataset for chart note sequences
- Utility functions for note-to-tensor conversion
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Iterator, Optional, Union

from .models import Cabinet, Chart, Difficulty, Song, TouchNote
from .parser import parse_maidata_file

# Try importing PyTorch (optional)
try:
    import torch
    from torch.utils.data import Dataset as TorchDataset
    HAS_TORCH = True
except (ImportError, OSError):
    HAS_TORCH = False


class SongDataset:
    """
    A collection of parsed Song objects from the datasets/ directory.

    Usage:
        ds = SongDataset("datasets")
        ds.scan()              # discover all songs
        ds.load(use_cache=True)  # parse all songs (with caching)
        print(len(ds))          # total songs
        song = ds[0]            # first song
    """

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.songs: list[Song] = []
        self._by_id: dict[str, Song] = {}
        self._cache_path = self.root_dir / ".mai_parser_cache.pkl"

    def scan(self) -> list[Path]:
        """
        Scan the dataset directory for all song folders (containing maidata.txt).

        Returns list of found song folder paths.
        """
        folders = []
        for item in sorted(self.root_dir.iterdir()):
            if item.is_dir() and (item / "maidata.txt").exists():
                folders.append(item)
        return folders

    def load(self, use_cache: bool = True, max_workers: int = 8,
             progress: bool = True) -> "SongDataset":
        """
        Parse all songs from discovered folders.

        Args:
            use_cache: If True, try loading from cache first; save after parsing.
            max_workers: Number of threads for parallel parsing.
            progress: Print progress information.
        """
        if use_cache and self._cache_path.exists():
            try:
                with open(self._cache_path, "rb") as f:
                    self.songs = pickle.load(f)
                self._build_index()
                if progress:
                    print(f"Loaded {len(self.songs)} songs from cache.")
                return self
            except (pickle.UnpicklingError, EOFError, KeyError) as e:
                if progress:
                    print(f"Cache invalid ({e}), re-parsing...")

        folders = self.scan()
        if progress:
            print(f"Found {len(folders)} song folders. Parsing...")

        from concurrent.futures import ThreadPoolExecutor, as_completed

        songs: list[Song] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(parse_maidata_file, f / "maidata.txt"): f
                for f in folders
            }
            for i, future in enumerate(as_completed(futures)):
                folder = futures[future]
                try:
                    song = future.result()
                    song.maidata_path = str(folder / "maidata.txt")
                    song.audio_path = str(folder / "track.mp3")
                    songs.append(song)
                except Exception as e:
                    if progress:
                        print(f"  [WARN] Failed to parse {folder.name}: {e}")

                if progress and (i + 1) % 200 == 0:
                    print(f"  ... parsed {i + 1}/{len(folders)}")

        # Sort by song_id numerically if possible
        songs.sort(key=lambda s: _sort_key(s.song_id))

        self.songs = songs
        self._build_index()

        if use_cache:
            try:
                with open(self._cache_path, "wb") as f:
                    pickle.dump(self.songs, f, protocol=pickle.HIGHEST_PROTOCOL)
                if progress:
                    print(f"Saved cache to {self._cache_path}")
            except OSError as e:
                if progress:
                    print(f"  [WARN] Could not save cache: {e}")

        if progress:
            print(f"Loaded {len(self.songs)} songs.")
        return self

    def _build_index(self) -> None:
        self._by_id = {s.song_id: s for s in self.songs}

    def __len__(self) -> int:
        return len(self.songs)

    def __getitem__(self, idx: int) -> Song:
        return self.songs[idx]

    def __iter__(self) -> Iterator[Song]:
        return iter(self.songs)

    def get_by_id(self, song_id: str) -> Song | None:
        """Look up a song by its folder name / ID."""
        return self._by_id.get(song_id)

    def filter(
        self,
        cabinet: Cabinet | None = None,
        min_level: float | None = None,
        max_level: float | None = None,
        difficulty: Difficulty | None = None,
        genre: str | None = None,
        has_audio: bool = False,
    ) -> list[Song]:
        """
        Filter songs by various criteria.

        Args:
            cabinet: Filter by cabinet type (SD/DX).
            min_level: Minimum chart level.
            max_level: Maximum chart level.
            difficulty: Only songs with this difficulty.
            genre: Filter by genre name (partial match).
            has_audio: Only songs where track.mp3 exists.
        """
        result = []
        for song in self.songs:
            if cabinet is not None and song.cabinet != cabinet:
                continue
            if genre is not None and genre.lower() not in song.genre.lower():
                continue
            if has_audio:
                audio_path = self.root_dir / song.song_id / "track.mp3"
                if not audio_path.exists():
                    continue
            if difficulty is not None:
                chart = song.get_chart(difficulty)
                if chart is None:
                    continue
                if min_level is not None and chart.level_value < min_level:
                    continue
                if max_level is not None and chart.level_value > max_level:
                    continue
            result.append(song)
        return result

    def stats(self) -> dict:
        """Return summary statistics of the dataset."""
        total = len(self.songs)
        sd_count = sum(1 for s in self.songs if s.cabinet == Cabinet.SD)
        dx_count = sum(1 for s in self.songs if s.cabinet == Cabinet.DX)
        unknown_cab = total - sd_count - dx_count
        total_charts = sum(len(s.charts) for s in self.songs)
        genres = list(set(s.genre for s in self.songs if s.genre))

        # Level distribution per difficulty
        level_dist: dict[str, list[float]] = {}
        for song in self.songs:
            for idx, chart in song.charts.items():
                key = f"lv_{idx}"
                if key not in level_dist:
                    level_dist[key] = []
                level_dist[key].append(chart.level_value)

        return {
            "total_songs": total,
            "sd_songs": sd_count,
            "dx_songs": dx_count,
            "unknown_cabinet": unknown_cab,
            "total_charts": total_charts,
            "genres": sorted(genres),
            "level_distribution": {
                k: {
                    "min": min(v) if v else 0,
                    "max": max(v) if v else 0,
                    "avg": sum(v) / len(v) if v else 0,
                }
                for k, v in level_dist.items()
            },
        }


def _sort_key(song_id: str) -> tuple:
    """Sort: numeric IDs first, then strings."""
    try:
        return (0, int(song_id))
    except ValueError:
        return (1, song_id)


# ─── Note to Tensor Conversion ──────────────────────────────────────────────

# Note type encoding for training
NOTE_TYPE = {
    "rest": 0,
    "tap": 1,
    "hold": 2,
    "slide": 3,
    "break": 4,
    "touch": 5,
    "end": 6,
}

# Reverse mapping
NOTE_TYPE_INV = {v: k for k, v in NOTE_TYPE.items()}

# Number of button positions (1-8)
NUM_POSITIONS = 8


def note_to_vector(note: TouchNote, num_positions: int = NUM_POSITIONS) -> list[float]:
    """
    Convert a single TouchNote to a feature vector.

    Format (fixed-length):
      [type_rest, type_tap, type_hold, type_slide, type_break, type_touch, type_end,
       pos_1, pos_2, ..., pos_N,
       is_simultaneous, has_slide_path_len_2, has_slide_path_len_3, has_slide_path_len_4,
       hold_beat, hold_subdiv,
       beat_div_normalized]

    Returns a flat list of floats.
    """
    vec = [0.0] * (7 + num_positions + 4 + 2 + 1)

    # Note type (one-hot)
    if note.is_end:
        vec[6] = 1.0
    elif note.is_rest:
        vec[0] = 1.0
    elif note.is_touch:
        vec[5] = 1.0
    elif note.is_break:
        vec[4] = 1.0
    elif note.is_hold:
        vec[2] = 1.0
    elif note.is_slide:
        vec[3] = 1.0
    else:
        vec[1] = 1.0  # tap

    # Button positions (multi-hot)
    offset = 7
    for pos in note.positions:
        if 1 <= pos <= num_positions:
            vec[offset + pos - 1] = 1.0

    # Slide path length flag
    offset += num_positions
    sl = len(note.slide_path)
    if sl >= 2:
        vec[offset] = 1.0
    if sl >= 3:
        vec[offset + 1] = 1.0
    if sl >= 4:
        vec[offset + 2] = 1.0
    vec[offset + 3] = 1.0 if note.is_simultaneous else 0.0

    # Hold duration
    offset += 4
    if note.hold_duration:
        vec[offset] = float(note.hold_duration[0])
        vec[offset + 1] = float(note.hold_duration[1])

    # Beat division (normalized)
    vec[-1] = note.beat_div

    return vec


def chart_to_tensor(chart: Chart) -> "torch.Tensor":
    """
    Convert an entire chart's notes to a PyTorch tensor.

    Returns tensor of shape (num_notes, feature_dim).
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for tensor conversion. "
                          "Install with: pip install torch")
    vectors = [note_to_vector(n) for n in chart.notes]
    return torch.tensor(vectors, dtype=torch.float32)


def chart_to_sequence(chart: Chart) -> list[list[float]]:
    """Convert a chart to a list of feature vectors (no PyTorch dependency)."""
    return [note_to_vector(n) for n in chart.notes]


# ─── PyTorch Dataset (if torch available) ───────────────────────────────────

if HAS_TORCH:

    class ChartDataset(TorchDataset):
        """
        PyTorch Dataset for maimai chart training.

        Yields (note_sequence, metadata) pairs where:
        - note_sequence is a tensor of shape (seq_len, feature_dim)
        - metadata is a dict with song info

        Usage:
            from torch.utils.data import DataLoader
            ds = ChartDataset(song_dataset, difficulty=Difficulty.MASTER)
            loader = DataLoader(ds, batch_size=32, collate_fn=ds.collate_fn)
        """

        def __init__(
            self,
            song_dataset: SongDataset,
            difficulty: Difficulty | None = None,
            min_level: float | None = None,
            max_level: float | None = None,
            pad_to: int | None = None,
            include_metadata: bool = True,
        ):
            self.song_dataset = song_dataset
            self.difficulty = difficulty
            self.min_level = min_level
            self.max_level = max_level
            self.pad_to = pad_to
            self.include_metadata = include_metadata

            # Build filtered list of (song_index, chart_index)
            self._items: list[tuple[int, int]] = []
            for si, song in enumerate(song_dataset.songs):
                for idx, chart in song.charts.items():
                    if difficulty is not None and idx != difficulty.value:
                        continue
                    if min_level is not None and chart.level_value < min_level:
                        continue
                    if max_level is not None and chart.level_value > max_level:
                        continue
                    self._items.append((si, idx))

        def __len__(self) -> int:
            return len(self._items)

        def __getitem__(self, idx: int):
            si, ci = self._items[idx]
            song = self.song_dataset[si]
            chart = song.charts[ci]

            tensor = chart_to_tensor(chart)

            if self.pad_to is not None:
                if tensor.shape[0] < self.pad_to:
                    pad = torch.zeros(
                        self.pad_to - tensor.shape[0], tensor.shape[1]
                    )
                    tensor = torch.cat([tensor, pad], dim=0)

            if self.include_metadata:
                return tensor, {
                    "song_id": song.song_id,
                    "title": song.title_clean,
                    "artist": song.artist,
                    "bpm": song.bpm,
                    "level": chart.level_value,
                    "difficulty": chart.difficulty_index,
                    "cabinet": song.cabinet.value,
                    "note_count": chart.note_count,
                }
            return tensor

        @staticmethod
        def collate_fn(batch):
            """Custom collate for variable-length sequences."""
            if isinstance(batch[0], tuple):
                sequences = [item[0] for item in batch]
                metadata = [item[1] for item in batch]
            else:
                sequences = batch
                metadata = None

            # Pad sequences to max length in batch
            lengths = torch.tensor([s.shape[0] for s in sequences])
            max_len = lengths.max().item()
            feat_dim = sequences[0].shape[1]

            padded = torch.zeros(len(sequences), max_len, feat_dim)
            for i, seq in enumerate(sequences):
                padded[i, :seq.shape[0]] = seq

            if metadata is not None:
                return padded, lengths, metadata
            return padded, lengths

        def stats(self) -> dict:
            """Return dataset statistics."""
            levels = []
            note_counts = []
            for _, ci in self._items:
                song_idx, chart_idx = self._items[0]
            for si, ci in self._items:
                chart = self.song_dataset[si].charts[ci]
                levels.append(chart.level_value)
                note_counts.append(chart.note_count)

            return {
                "total_charts": len(self._items),
                "level_min": min(levels) if levels else 0,
                "level_max": max(levels) if levels else 0,
                "level_avg": sum(levels) / len(levels) if levels else 0,
                "note_count_min": min(note_counts) if note_counts else 0,
                "note_count_max": max(note_counts) if note_counts else 0,
                "note_count_avg": sum(note_counts) / len(note_counts)
                if note_counts else 0,
            }

else:

    class ChartDataset:
        """Stub when PyTorch is not installed."""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required for ChartDataset. "
                "Install with: pip install torch"
            )
