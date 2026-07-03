"""
Chart preprocessing pipeline — generate token sequences for all 4 stages.

Pipeline:
  1. Parse simai chart → _Note list
  2. Strip break/firework → Stage 1 clean chart
  3. Tokenize → config tokens (Stage 1) + base tokens (slides)
  4. Detect touch groups → Stage 2 compression/expansion labels
  5. Record break positions → Stage 3 labels
  6. Record firework touches → Stage 4 labels

Usage:
    python scripts/preprocess_chart.py datasets/10/maidata.txt

Output per difficulty level:
    Stage 1 tokens:  token sequence for coarse structure generation
    Stage 2 labels:  touch group → center mapping
    Stage 3 labels:  per-slot break flags
    Stage 4 labels:  per-touch firework flags
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from Tokenizer.chart_preprocess import strip_all
from Tokenizer.MaiChartTokenizer import (
    MaiChartTokenizer, _parse_simai_text, _parse_note_token,
    encode_duration_tokens, read_duration_tokens,
    PAD, BOS, EOS, RST, DUR, DIV_TO_ID, CONFIG_BASE,
    TAP_TO_ID, BRK_TO_ID, HLD_TO_ID, SLD_TO_ID,
    SLD_BEG, SLD_END, SIM_BEG, SIM_END, SIM_COUNT_2,
    TCH_TO_ID, ID_TO_TCH,
    FIREWORK, EX_NOTE, FAKE_EACH,
    token_name,
    USE_CONFIG_VOCAB, VOCAB_SIZE,
)
from Tokenizer.config_vocab import (
    ID_TO_CONFIG, CONFIG_TO_ID, SlotConfig,
    BTN_PRESS, BTN_HOLD_START, BTN_HOLD_ONGOING, BTN_SLIDE_START, BTN_SLIDE_END,
    TCH_TOUCH, TCH_HOLD_START, TCH_HOLD_ONGOING,
    config_to_base_tokens, config_name,
)
from Tokenizer.touch_expander import (
    TouchExpander, find_connected_groups, compress_group,
    zone_name as tch_zone_name, zone_index as tch_zone_index,
)


# ═══════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SlotLabels:
    """Per-time-slot labels for stages 2-4."""
    slot_index: int = 0
    raw_note: str = ""                        # original simai token

    # Stage 1: config token ID (or list of base tokens for slides)
    stage1_tokens: list[int] = field(default_factory=list)

    # Stage 2: touch compression
    # original touch zones (from raw chart, after strip)
    touch_zones: list[int] = field(default_factory=list)
    touch_states: list[int] = field(default_factory=list)
    # compressed center zone(s) from Stage 1
    touch_centers: list[int] = field(default_factory=list)

    # Stage 3: break labels
    # (position, is_break) for each button in this slot
    break_labels: list[tuple[int, bool]] = field(default_factory=list)

    # Stage 4: firework labels
    # (zone_index, is_firework) for each touch in this slot
    firework_labels: list[tuple[int, bool]] = field(default_factory=list)


@dataclass
class ChartTokens:
    """Complete token/label data for one difficulty chart."""
    difficulty: str = ""
    bpm: float = 0.0

    # Stage 1: full token sequence (config tokens + base tokens for slides)
    stage1_tokens: list[int] = field(default_factory=list)

    # Per-slot labels
    slots: list[SlotLabels] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════

def preprocess_chart(
    maidata_path: str,
    difficulty: int = 3,
) -> ChartTokens:
    """Preprocess one difficulty level of a maidata.txt.

    Args:
        maidata_path: Path to maidata.txt
        difficulty: 1=BASIC, 2=ADV, 3=EXP, 4=MAS, 5=ReMAS

    Returns:
        ChartTokens with token sequences and labels for all stages.
    """
    # 1. Read chart
    raw_chart = _extract_inote_block(maidata_path, difficulty)
    bpm = _extract_bpm(maidata_path)
    if not raw_chart:
        raise ValueError(f"No &inote_{difficulty}= block found")

    # 2. Strip break/firework → Stage 1 clean chart
    clean_chart = strip_all(raw_chart)

    # 3. Parse both versions
    notes_raw = _parse_simai_text(raw_chart)
    notes_clean = _parse_simai_text(clean_chart)

    # 4. Tokenize clean chart → Stage 1 tokens
    tokenizer = MaiChartTokenizer()
    stage1_tokens = tokenizer.encode(clean_chart)

    # 5. Build per-slot labels by aligning raw and clean notes
    chart = ChartTokens(
        difficulty=f"lv_{difficulty}",
        bpm=bpm,
        stage1_tokens=stage1_tokens,
    )

    # Align notes (they should be 1:1 since strip only removes modifiers)
    if len(notes_raw) != len(notes_clean):
        print(f"Warning: raw={len(notes_raw)} notes, clean={len(notes_clean)} notes")

    for idx, (raw_note, clean_note) in enumerate(zip(notes_raw, notes_clean)):
        slot = SlotLabels(slot_index=idx, raw_note=raw_note.raw)

        # Stage 1 tokens for this slot (extracted below)
        slot.stage1_tokens = _encode_slot(clean_note)

        # Stage 2: touch compression
        _extract_touch_labels(slot, raw_note, clean_note)

        # Stage 3: break labels
        _extract_break_labels(slot, raw_note)

        # Stage 4: firework labels
        _extract_firework_labels(slot, raw_note)

        chart.slots.append(slot)

    return chart


# ═══════════════════════════════════════════════════════════════════════
# Stage label extraction
# ═══════════════════════════════════════════════════════════════════════

def _encode_slot(note) -> list[int]:
    """Encode a single clean _Note -> token list using config tokens."""
    tokenizer = MaiChartTokenizer()
    if note.is_rest:
        return [RST]
    if note.is_end:
        return []
    raw = note.raw
    if not raw:
        return [RST]
    tokens = tokenizer.encode(raw + ",E", add_bos=False, add_eos=False)
    return tokens


def _extract_touch_labels(slot: SlotLabels, raw_note, clean_note) -> None:
    """Record touch zone info for Stage 2."""
    if not raw_note.is_touch:
        return

    for region in raw_note.touch_regions:
        zi = tch_zone_index(region)
        # Determine state
        if raw_note.is_touch_hold:
            state = TCH_HOLD_START
        elif raw_note.is_hold:
            state = TCH_HOLD_ONGOING
        else:
            state = TCH_TOUCH
        slot.touch_zones.append(zi)
        slot.touch_states.append(state)

    # Stage 1 already compresses connected groups → center zones
    if slot.touch_zones:
        groups = find_connected_groups(set(slot.touch_zones))
        slot.touch_centers = [compress_group(g) for g in groups]


def _extract_break_labels(slot: SlotLabels, raw_note) -> None:
    """Record break labels for Stage 3."""
    if raw_note.is_touch or raw_note.is_rest or raw_note.is_end:
        return
    if raw_note.is_slide:
        return  # slide breaks handled separately

    # Check per-position break
    sub_breaks = getattr(raw_note, '_sub_breaks', [])
    for idx, pos in enumerate(raw_note.positions):
        if 1 <= pos <= 8:
            is_brk = sub_breaks[idx] if idx < len(sub_breaks) else raw_note.is_break
            slot.break_labels.append((pos, is_brk))


def _extract_firework_labels(slot: SlotLabels, raw_note) -> None:
    """Record firework labels for Stage 4."""
    if not raw_note.is_touch or not raw_note.is_firework:
        return
    for region in raw_note.touch_regions:
        zi = tch_zone_index(region)
        slot.firework_labels.append((zi, True))


# ═══════════════════════════════════════════════════════════════════════
# File I/O helpers
# ═══════════════════════════════════════════════════════════════════════

def _extract_inote_block(path: str, difficulty: int) -> str:
    """Extract the &inote_N= block from a maidata.txt."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    marker = f"&inote_{difficulty}="
    idx = text.find(marker)
    if idx < 0:
        return ""

    start = idx + len(marker)
    # Find next &header or end of file
    end = len(text)
    for m in re.finditer(r"^&", text[start:], re.MULTILINE):
        if m.start() > 0:
            end = start + m.start()
            break

    return text[start:end].strip()


def _extract_bpm(path: str) -> float:
    """Extract BPM from maidata.txt header."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("&wholebpm="):
                return float(line.split("=")[1].strip())
    return 120.0


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/preprocess_chart.py <maidata.txt> [difficulty]")
        sys.exit(1)

    path = sys.argv[1]
    diff = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    print(f"Processing: {path} (difficulty={diff})")
    chart = preprocess_chart(path, diff)

    print(f"\n  BPM: {chart.bpm}")
    print(f"  Stage 1 tokens: {len(chart.stage1_tokens)}")
    print(f"  Time slots: {len(chart.slots)}")
    print()

    # Show first 15 slots
    print("  Slot  | Stage1 tokens              | Touch centers | Break      | Firework")
    print("  " + "-" * 70)
    for s in chart.slots[:15]:
        st1 = " ".join(token_name(t) for t in s.stage1_tokens[:4])
        tch = ",".join(tch_zone_name(c) for c in s.touch_centers) if s.touch_centers else "-"
        brk = ",".join(f"{p}{'b' if b else ''}" for p, b in s.break_labels) if s.break_labels else "-"
        fw = ",".join(tch_zone_name(z) for z, _ in s.firework_labels) if s.firework_labels else "-"
        print(f"  {s.slot_index:5d} | {st1:<28s} | {tch:<12s} | {brk:<10s} | {fw}")

    if len(chart.slots) > 15:
        print(f"  ... ({len(chart.slots) - 15} more slots)")

    # Stats
    total_breaks = sum(1 for s in chart.slots for _ in s.break_labels if _[1])
    total_fw = sum(len(s.firework_labels) for s in chart.slots)
    total_touch = sum(1 for s in chart.slots if s.touch_zones)
    config_count = sum(1 for t in chart.stage1_tokens if t >= CONFIG_BASE)
    total_notes = len([t for t in chart.stage1_tokens if t > EOS])

    print(f"\n  Stats:")
    print(f"    Total tokens:       {len(chart.stage1_tokens)}")
    print(f"    Config tokens:      {config_count}")
    print(f"    Slots with break:   {total_breaks}")
    print(f"    Slots with firework:{total_fw}")
    print(f"    Slots with touch:   {total_touch}")
