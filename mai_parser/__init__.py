"""
mai_parser - Maimai (舞萌) chart data parser for ML training.

Usage:
    from mai_parser import SongDataset, parse_maidata_file

    # Load entire dataset
    ds = SongDataset("datasets").load()
    print(ds.stats())

    # Parse single file
    song = parse_maidata_file("datasets/10/maidata.txt")
    print(song.title_clean, song.artist, song.bpm)

    # Convert chart to tensor for training
    chart = song.get_chart(Difficulty.MASTER)
    tensor = chart_to_tensor(chart)
"""

from .models import Cabinet, Chart, Difficulty, Song, TouchNote
from .parser import parse_maidata, parse_maidata_file
from .dataset import (
    ChartDataset,
    SongDataset,
    chart_to_sequence,
    chart_to_tensor,
    note_to_vector,
    NOTE_TYPE,
    NOTE_TYPE_INV,
    NUM_POSITIONS,
    HAS_TORCH,
)

__all__ = [
    # Models
    "Cabinet",
    "Chart",
    "Difficulty",
    "Song",
    "TouchNote",
    # Parser
    "parse_maidata",
    "parse_maidata_file",
    # Dataset
    "ChartDataset",
    "SongDataset",
    "chart_to_sequence",
    "chart_to_tensor",
    "note_to_vector",
    # Constants
    "NOTE_TYPE",
    "NOTE_TYPE_INV",
    "NUM_POSITIONS",
    "HAS_TORCH",
]
