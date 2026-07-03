"""
Parser for maidata.txt files (MaichartConverter format).

Parses:
1. Header key-value pairs (&key=value)
2. Note sequence data (&inote_N= blocks)

The note format is measure-based:
  {beat_div}pos1,pos2,pos3,...

Where each position can be:
  - A number 1-8 (button position)
  - Empty (rest)
  - Number/Number (simultaneous tap)
  - Number + modifiers (h=hold, b=break, x=firework, >=/</V/slide patterns)
  - Touch region (C, B1-B8, E1-E8, A1-A8, D1-D8)
  - End marker: "E"
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from .models import Cabinet, Chart, Difficulty, Song, TouchNote

# ─── Regex patterns for note parsing ────────────────────────────────────────

# Matches {beat_div} at the start of a measure
_RE_BEAT_DIV = re.compile(r"\{(\d+)\}")

# Matches inline BPM changes: (200)
_RE_BPM_CHANGE = re.compile(r"\((\d+(?:\.\d+)?)\)")

# Matches hold duration: h[beat:subdiv]
_RE_HOLD = re.compile(r"h\[(\d+):(\d+)\]")

# Matches slide pattern with duration: >, <, -, V, v, ^
# Examples: 3>6[4:1], 6<3[8:11], 3-6[4:1], 3v6[16:3], 5>2[8:11], 3>8-6v3[8:9]
_RE_SLIDE_DUR = re.compile(r"([><Vv\-\^])(\d+)\[(\d+):(\d+)\]")

# Matches simple slide (no duration): 3>6, 4-1
_RE_SIMPLE_SLIDE = re.compile(r"(\d+)[><Vv\-\^](\d+)")

# Matches break suffix: b or $b at end
_RE_BREAK = re.compile(r"b$|\$b$")

# Matches firework/EX: x suffix
_RE_FIREWORK = re.compile(r"x$")

# Matches touch note identifiers: C, Cf, Ch, B1-B8, E1-E8, A1-A8, D1-D8
_RE_TOUCH = re.compile(r"^(C[fh]?|[ABDE][1-8])$")

# Matches simultaneous tap: 1/8, 2/5/7
_RE_SIMUL = re.compile(r"^(\d+)/(\d+)(?:/(\d+))?(?:/(\d+))?$")

# Matches a number position (1-8)
_RE_POS = re.compile(r"^\d+$")


def parse_header_line(line: str) -> tuple[str, str]:
    """Parse a single &key=value header line. Returns (key, value)."""
    line = line.strip()
    if line.startswith("&"):
        line = line[1:]  # strip &
    if "=" in line:
        key, value = line.split("=", 1)
        return key.strip(), value.strip()
    return line.strip(), ""


def _extract_bpm_changes(text: str) -> list[tuple[int, float]]:
    """Extract inline BPM changes from a measure line.
    Returns list of (position_in_measure, bpm_value).
    """
    changes = []
    for m in _RE_BPM_CHANGE.finditer(text):
        changes.append((m.start(), float(m.group(1))))
    return changes


def _parse_single_note(token: str, current_beat_div: int) -> TouchNote:
    """
    Parse a single note token (one comma-separated segment).

    Examples:
      ""        → rest
      "E"       → end marker
      "1"       → tap at position 1
      "1/8"     → simultaneous tap at 1 and 8
      "1b"      → break at position 1
      "1x"      → firework at position 1
      "1h[4:1]" → hold at position 1, duration 4:1
      "3>6[4:1]"→ slide from 3 to 6, duration 4:1
      "3-6"     → simple slide 3→6
      "C"       → touch center
      "B7/B6"   → touch simultaneous
      "Ch[2:1]" → touch hold
    """
    note = TouchNote(beat_div=current_beat_div, raw=token)

    # Empty = rest
    if not token:
        note.is_rest = True
        return note

    # End marker
    if token.strip() == "E":
        note.is_end = True
        return note

    # Work on the cleaned token (remove beat division, BPM)
    t = token.strip()

    # Remove inline BPM changes for parsing: (200), (150.5)
    t = _RE_BPM_CHANGE.sub("", t).strip()
    if not t:
        note.is_rest = True
        return note

    # --- Check for touch notes ---
    # Touch notes can have modifiers like Ch[2:1], C, B7/B6
    # They use letters A-E + number, or just C
    touch_parts = re.split(r"[/]", t)
    all_touch = all(
        _RE_TOUCH.match(re.sub(r"\[.*?\]", "", p).strip("hbfxqpb$Vv*^-><"))
        for p in touch_parts
    )

    if all_touch and any(
        re.match(r"^[A-E]", p.strip("hbfxqpb$Vv*^-><"))
        for p in touch_parts
    ):
        note.is_touch = True
        # Extract touch regions (strip modifiers)
        for p in touch_parts:
            clean = re.sub(r"\[.*?\]", "", p).strip("hbfxqpb$Vv*^-><")
            if clean:
                note.touch_regions.append(clean)
        # Parse hold/touch-hold
        hold_m = _RE_HOLD.search(t)
        if hold_m:
            note.is_hold = True
            note.hold_duration = (int(hold_m.group(1)), int(hold_m.group(2)))
        return note

    # --- Parse button note ---

    # Break suffix
    if _RE_BREAK.search(t):
        note.is_break = True
        t = _RE_BREAK.sub("", t).rstrip("$")

    # Firework suffix
    if t.endswith("x"):
        note.is_firework = True
        t = t[:-1]

    # Hold
    hold_m = _RE_HOLD.search(t)
    if hold_m:
        note.is_hold = True
        note.hold_duration = (int(hold_m.group(1)), int(hold_m.group(2)))
        t = _RE_HOLD.sub("", t)

    # Star/EX note (* suffix)
    if t.endswith("*"):
        note.is_star = True
        t = t[:-1]

    # Slide patterns
    # Pattern with duration: 3>6[4:1], 3-6[8:11], 3v6[16:3], 3>8-6v3[8:9]
    slide_m = _RE_SLIDE_DUR.search(t)
    if slide_m:
        note.is_slide = True
        note.slide_path = _extract_slide_path(t)
        hold_m2 = _RE_HOLD.search(t)
        if not hold_m2:
            # Duration is on slide: [beat:subdiv]
            note.hold_duration = (int(slide_m.group(3)), int(slide_m.group(4)))
        # Extract positions from slide
        pos_match = re.findall(r"(\d+)", t)
        note.positions = [int(p) for p in pos_match[:4]]
        return note

    # Simple slide: 3>6, 4-1, 8<5
    simple_slide = _RE_SIMPLE_SLIDE.search(t)
    if simple_slide:
        note.is_slide = True
        pos_match = re.findall(r"(\d+)", t)
        note.positions = [int(p) for p in pos_match[:4]]
        return note

    # Single V (slide continuation / endpoint)
    if re.match(r"^V\d*$", t):
        note.is_slide = True
        return note

    # Simultaneous tap: 1/8, 2/6/7
    simul_m = _RE_SIMUL.match(t)
    if simul_m:
        note.is_simultaneous = True
        note.positions = [int(g) for g in simul_m.groups() if g is not None]
        return note

    # Simple tap: 1-8
    if _RE_POS.match(t):
        note.positions = [int(t)]
        return note

    # Special modifiers: 'q', 'p', 'w', 'z', '$' etc. — treat as tap if has number
    num_match = re.findall(r"(\d+)", t)
    if num_match:
        note.positions = [int(p) for p in num_match[:4]]
        return note

    # Unknown / rest
    note.is_rest = True
    return note


def _extract_slide_path(token: str) -> list[int]:
    """Extract all position numbers from a slide pattern like '3>8-6v3[8:9]' → [3,8,6,3]."""
    nums = re.findall(r"(?<![a-zA-Z\[:])(\d+)(?![a-zA-Z\]])", token)
    return [int(n) for n in nums]


def _parse_note_sequence(raw_text: str) -> list[TouchNote]:
    """
    Parse the full note sequence from an &inote_N block.

    Handles:
    - Beat division changes: {4}, {8}, {16}, etc.
    - Inline BPM changes: (200)
    - Measure-by-measure comma-separated notes
    - End marker: E
    """
    notes: list[TouchNote] = []
    current_div = 4  # default beat division

    # Normalize: join continuation lines, split by newline
    text = raw_text.strip()

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Check for beat division change at start of line
        bd_match = _RE_BEAT_DIV.match(line)
        if bd_match:
            current_div = int(bd_match.group(1))
            line = line[bd_match.end():]  # remove {beat_div} prefix

        # If the line is empty after removing beat div, it's just a beat division
        # announcement; subsequent lines use this division
        if not line:
            continue

        # The line may start with an inline BPM change like (173){1},
        bpm_match = _RE_BPM_CHANGE.match(line)
        if bpm_match:
            line = line[bpm_match.end():]
            # Check for beat div inside the BPM line
            bd2 = _RE_BEAT_DIV.match(line) if line else None
            if bd2:
                current_div = int(bd2.group(1))
                line = line[bd2.end():]

        if not line:
            continue

        # Split by commas for individual notes
        tokens = line.split(",")

        for token in tokens:
            # Check for inline beat division change
            bd_inline = _RE_BEAT_DIV.search(token) if token else None
            local_div = current_div
            if bd_inline:
                local_div = int(bd_inline.group(1))
                token = _RE_BEAT_DIV.sub("", token).strip()

            note = _parse_single_note(token, local_div)
            notes.append(note)

            if note.is_end:
                return notes  # stop at E marker

    return notes


def parse_level_value(level_str: str) -> tuple[float, bool, bool]:
    """
    Parse a level string.
    Returns (numeric_value, is_plus, is_ura).

    Examples:
      "12.4"   → (12.4, False, False)
      "13+"    → (13.0, True, False)
      "14.6?"  → (14.6, False, True)
      "7+"     → (7.0, True, False)
      "耐"     → (-1.0, False, False)  # non-numeric
      ""       → (0.0, False, False)
    """
    is_plus = "+" in level_str
    is_ura = "?" in level_str
    # Strip non-numeric except . + ?
    clean = level_str.replace("+", "").replace("?", "").strip()
    try:
        val = float(clean) if clean else 0.0
    except ValueError:
        val = -1.0
    return val, is_plus, is_ura


def parse_maidata(content: str, song_id: str = "",
                  maidata_path: str = "", audio_path: str = "") -> Song:
    """
    Parse a complete maidata.txt content string into a Song object.

    Args:
        content: The full text content of maidata.txt
        song_id: Folder name / song identifier
        maidata_path: Relative path to maidata.txt
        audio_path: Relative path to track.mp3

    Returns:
        Song object with all parsed data
    """
    song = Song(song_id=song_id, maidata_path=maidata_path, audio_path=audio_path)

    # Strip BOM if present
    if content.startswith("\ufeff"):
        content = content[1:]

    lines = content.split("\n")
    raw_notes: dict[int, str] = {}  # difficulty_index → raw note text
    current_note_idx: Optional[int] = None
    current_note_lines: list[str] = []

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue

        if line_stripped.startswith("&"):
            key, value = parse_header_line(line_stripped)

            # If we were collecting note data, flush it
            if current_note_idx is not None:
                raw_notes[current_note_idx] = "\n".join(current_note_lines)
                current_note_idx = None
                current_note_lines = []

            if key.startswith("inote_"):
                # Note data block
                idx_str = key.replace("inote_", "")
                try:
                    current_note_idx = int(idx_str)
                except ValueError:
                    current_note_idx = None
                if value:
                    current_note_lines.append(value)
            elif key.startswith("lv_"):
                idx_str = key.replace("lv_", "")
                if not idx_str.isdigit():
                    continue
                idx = int(idx_str)
                if not value:
                    continue  # skip empty levels (no chart)
                song.levels[idx] = value
                level_val, is_plus, is_ura = parse_level_value(value)
                if level_val > 0:
                    # Create chart placeholder (notes parsed later)
                    diff = Difficulty.from_index(idx)
                    chart = Chart(
                        difficulty_index=idx,
                        difficulty=diff,
                        level=value,
                        level_value=level_val,
                        is_plus=is_plus,
                        is_ura=is_ura,
                    )
                    song.charts[idx] = chart
            elif key.startswith("des_"):
                idx_str = key.replace("des_", "")
                try:
                    idx = int(idx_str)
                except ValueError:
                    continue
                song.charters[idx] = value
                if idx in song.charts:
                    song.charts[idx].charter = value
            elif key == "title":
                song.title = value
                song.title_clean = _clean_title(value)
                song.tags = _extract_tags(value)
                song.is_full = "[FULL]" in value or "_FULLTOUCH" in value
                song.is_fulltouch = "_FULLTOUCH" in value
            elif key == "artist":
                song.artist = value
            elif key == "artistid":
                try:
                    song.artist_id = int(value)
                except ValueError:
                    pass
            elif key == "wholebpm":
                try:
                    song.bpm = float(value)
                except ValueError:
                    pass
            elif key == "first":
                try:
                    song.first = float(value)
                except ValueError:
                    pass
            elif key == "genre":
                song.genre = value
                song.is_utage = "宴会場" in value
            elif key == "genreid":
                try:
                    song.genre_id = int(value)
                except ValueError:
                    pass
            elif key == "cabinet":
                try:
                    song.cabinet = Cabinet(value.upper())
                except ValueError:
                    song.cabinet = Cabinet.UNKNOWN
            elif key == "version":
                song.version = value
            elif key == "shortid":
                try:
                    song.short_id = int(value)
                except ValueError:
                    pass
            elif key == "des":
                song.description = value
            elif key == "ChartConverter":
                song.converter = value
            elif key == "ChartConvertTool":
                song.converter_tool = value
            elif key == "ChartConvertToolVersion":
                song.converter_version = value
        elif current_note_idx is not None:
            # Collect note data lines
            current_note_lines.append(line_stripped)

    # Flush last note block
    if current_note_idx is not None:
        raw_notes[current_note_idx] = "\n".join(current_note_lines)

    # ── Parse all collected note sequences ──
    for idx, raw in raw_notes.items():
        notes = _parse_note_sequence(raw)

        if idx in song.charts:
            song.charts[idx].notes = notes
            song.charts[idx].compute_stats()
        elif idx in song.levels and song.levels[idx]:
            # Chart has notes + level but was skipped due to level_val being non-numeric (e.g. "耐")
            level_val, is_plus, is_ura = parse_level_value(song.levels[idx])
            diff = Difficulty.from_index(idx)
            chart = Chart(
                difficulty_index=idx,
                difficulty=diff,
                level=song.levels[idx],
                level_value=level_val,
                is_plus=is_plus,
                is_ura=is_ura,
            )
            chart.notes = notes
            chart.compute_stats()
            song.charts[idx] = chart

    return song


def _clean_title(title: str) -> str:
    """Remove bracket tags like [SD], [DX], [宴] from title."""
    return re.sub(r"\[.*?\]", "", title).strip()


def _extract_tags(title: str) -> list[str]:
    """Extract bracket tags from title, e.g. [SD], [DX], [宴]."""
    return re.findall(r"\[(.*?)\]", title)


def parse_maidata_file(filepath: str | Path) -> Song:
    """
    Parse a maidata.txt file from disk.

    Args:
        filepath: Path to the maidata.txt file

    Returns:
        Parsed Song object
    """
    filepath = Path(filepath)
    song_id = filepath.parent.name
    content = filepath.read_text(encoding="utf-8")

    # Build relative paths
    maidata_rel = filepath.name
    audio_rel = "track.mp3"

    return parse_maidata(
        content,
        song_id=song_id,
        maidata_path=maidata_rel,
        audio_path=audio_rel,
    )
