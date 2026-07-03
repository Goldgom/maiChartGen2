"""
Data models for maimai chart parsing.

Represents the full structure of a maimai song entry:
- Song metadata (title, artist, BPM, version, etc.)
- Difficulty configs (level, charter)
- Chart note sequences
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Cabinet(Enum):
    """Cabinet / hardware type."""
    SD = "SD"
    DX = "DX"
    UNKNOWN = "UNKNOWN"


class Difficulty(Enum):
    """Difficulty tiers (1-indexed matching &inote_N)."""
    BASIC = 1       # lv_1 / inote_1
    ADVANCED = 2    # lv_2 / inote_2
    EXPERT = 3      # lv_3 / inote_3
    MASTER = 4      # lv_4 / inote_4
    ReMASTER = 5    # lv_5 / inote_5 → SD Re:MASTER is inote_6
    # DX uses lv_6 for Re:MASTER (inote_6)
    # Some charts use lv_7 / inote_7 for Ura/裏 or extra

    @classmethod
    def from_index(cls, idx: int) -> "Difficulty":
        for d in cls:
            if d.value == idx:
                return d
        return cls.ReMASTER  # fallback for 5+


@dataclass
class TouchNote:
    """
    A single parsed note/event.

    Raw format examples:
      {4}1       → beat_div=4, pos=1
      {4}1/8     → beat_div=4, pos=1/8 (simultaneous press)
      {4}1h[4:1] → beat_div=4, pos=1, modifier=hold, duration=4:1
      {4}1b      → beat_div=4, pos=1, is_break=True
      {4}3>6[4:1]→ beat_div=4, pos=3>6 (slide), modifier=slide, duration=4:1
      {1}        → beat_div=1, rest
      E          → end marker
    """
    beat_div: int = 4          # {4}, {8}, {16}, {1}, etc.
    raw: str = ""              # original text
    is_rest: bool = False
    is_end: bool = False
    is_break: bool = False     # 'b' suffix on note
    is_star: bool = False      # '*' suffix (EX note)
    is_slide: bool = False     # '>' or 'V' or other slide patterns
    is_hold: bool = False      # 'h[...]' modifier
    is_touch: bool = False     # touch panel notes (C, Cf, Ch, etc.)
    is_simultaneous: bool = False  # contains '/' for multi-tap
    positions: list[int] = field(default_factory=list)  # 1-8 button positions
    touch_regions: list[str] = field(default_factory=list)  # e.g. ["C", "B7", "E8"]
    hold_duration: tuple[int, int] | None = None  # (beat, subdiv) from h[beat:subdiv]
    slide_path: list[int] = field(default_factory=list)  # slide positions
    firework: bool = False     # 'x' suffix
    ex_notes: list[str] = field(default_factory=list)  # EX modifiers


@dataclass
class Chart:
    """
    A single difficulty chart (one &inote_N block).

    Contains the parsed note sequence and metadata about the chart.
    """
    difficulty_index: int         # 1-7, maps to &inote_N
    difficulty: Difficulty        # enum value
    level: str = ""               # e.g. "12.4", "13+", "14.6?"
    level_value: float = 0.0      # parsed numeric level
    is_plus: bool = False         # has '+' suffix
    is_ura: bool = False          # has '?' suffix (裏譜面)
    charter: str = ""             # &des_N, charter/description
    notes: list[TouchNote] = field(default_factory=list)
    # Computed stats
    note_count: int = 0           # total non-rest notes
    tap_count: int = 0
    hold_count: int = 0
    slide_count: int = 0
    break_count: int = 0
    touch_count: int = 0
    total_beats: float = 0.0      # total chart duration in beats

    def compute_stats(self) -> None:
        """Recalculate computed statistics from notes."""
        self.note_count = 0
        self.tap_count = 0
        self.hold_count = 0
        self.slide_count = 0
        self.break_count = 0
        self.touch_count = 0

        for n in self.notes:
            if n.is_rest or n.is_end:
                continue
            self.note_count += 1
            if n.is_touch:
                self.touch_count += 1
            elif n.is_break:
                self.break_count += 1
            elif n.is_hold:
                self.hold_count += 1
            elif n.is_slide:
                self.slide_count += 1
            else:
                self.tap_count += 1


@dataclass
class Song:
    """
    Full parsed song entry — maps to one folder in datasets/.
    """
    # Identifiers
    song_id: str = ""             # folder name (e.g. "10", "10021")
    title: str = ""               # &title (raw, may include [SD] etc.)
    title_clean: str = ""         # cleaned title without tags
    artist: str = ""              # &artist
    artist_id: int = 0            # &artistid
    genre: str = ""               # &genre
    genre_id: int = 0             # &genreid

    # Music info
    bpm: float = 0.0              # &wholebpm
    first: float = 0.0            # &first (offset in seconds)

    # Cabinet & version
    cabinet: Cabinet = Cabinet.UNKNOWN
    version: str = ""             # &version
    short_id: int = 0             # &shortid

    # Chart converter metadata
    converter: str = ""            # &ChartConverter
    converter_tool: str = ""       # &ChartConvertTool
    converter_version: str = ""    # &ChartConvertToolVersion

    # Raw header
    description: str = ""          # &des

    # Difficulty info (keyed by difficulty index 1-7)
    levels: dict[int, str] = field(default_factory=dict)    # &lv_N
    charters: dict[int, str] = field(default_factory=dict)  # &des_N

    # Parsed charts (keyed by difficulty index 1-7)
    charts: dict[int, Chart] = field(default_factory=dict)

    # Flags
    is_fulltouch: bool = False
    is_full: bool = False          # [FULL] in title
    is_utage: bool = False         # 宴会場 genre
    tags: list[str] = field(default_factory=list)  # [SD], [DX], [宴], etc.

    # File paths (relative to dataset root)
    maidata_path: str = ""
    audio_path: str = ""

    def has_chart(self, diff: Difficulty) -> bool:
        """Check if a difficulty level exists."""
        return diff.value in self.charts

    def get_chart(self, diff: Difficulty) -> Chart | None:
        return self.charts.get(diff.value)
