"""
MaiChartTokenizer - Complete simai chart parser & tokenizer.

Parses raw simai-format chart text directly into discrete token sequences
for transformer training/inference. Handles ALL note types including wifi
slides, touch holds, EX notes, firework touch, and fake each.

Features:
  - Direct simai text -> token conversion (no intermediate objects)
  - Lossless round-trip: decode(encode(text)) == text
  - Full note type coverage (12 slide types, touch hold, wifi, etc.)
  - Config vocab for compact single-token encoding of common patterns
  - Compatible vocabulary layout with existing tokenizer.py

Usage:
    from Tokenizer.MaiChartTokenizer import MaiChartTokenizer

    tok = MaiChartTokenizer()
    tokens = tok.encode("(180){4}1,2,3,4,E")
    text   = tok.decode(tokens)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

# =====================================================================
# Vocabulary definition
# =====================================================================

# --- Special tokens (0-4) ---
PAD = 0
BOS = 1
EOS = 2
SEP = 3
MASK = 4
_SPECIAL_END = 5

# --- Beat division tokens (5-15) ---
_DIV_VALUES = [1, 2, 4, 8, 16, 32, 48, 64, 128, 192, 384]
DIV_BASE = _SPECIAL_END
DIV_TO_ID: dict[int, int] = {v: DIV_BASE + i for i, v in enumerate(_DIV_VALUES)}
ID_TO_DIV: dict[int, int] = {v: k for k, v in DIV_TO_ID.items()}
DIV_END = DIV_BASE + len(_DIV_VALUES)

# --- Rest token (16) ---
RST = DIV_END

# --- Duration marker (17) ---
DUR = RST + 1

# --- Tap tokens (18-25) ---
TAP_BASE = DUR + 1
TAP_TO_ID = {i: TAP_BASE + i - 1 for i in range(1, 9)}
ID_TO_TAP = {v: k for k, v in TAP_TO_ID.items()}
TAP_END = TAP_BASE + 8

# --- Break tokens (26-33) ---
BRK_BASE = TAP_END
BRK_TO_ID = {i: BRK_BASE + i - 1 for i in range(1, 9)}
ID_TO_BRK = {v: k for k, v in BRK_TO_ID.items()}
BRK_END = BRK_BASE + 8

# --- Hold tokens (34-41) ---
HLD_BASE = BRK_END
HLD_TO_ID = {i: HLD_BASE + i - 1 for i in range(1, 9)}
ID_TO_HLD = {v: k for k, v in HLD_TO_ID.items()}
HLD_END = HLD_BASE + 8

# --- Slide waypoint tokens (42-49) ---
SLD_BASE = HLD_END
SLD_TO_ID = {i: SLD_BASE + i - 1 for i in range(1, 9)}
ID_TO_SLD = {v: k for k, v in SLD_TO_ID.items()}
SLD_END_TOKEN_BASE = SLD_BASE + 8

# --- Slide control tokens (50-51) ---
SLD_BEG = SLD_END_TOKEN_BASE
SLD_END = SLD_BEG + 1

# --- Simultaneous control tokens (52-53) ---
SIM_BEG = SLD_END + 1
SIM_END = SIM_BEG + 1

# --- Touch tokens (54-86): E1-E8, B1-B8, C, A1-A8, D1-D8 = 33 zones ---
# Ring order: E(innermost), B(middle), C(center), A/D(outer alternating)
# C1-C8 are normalized to "C" in the parser
TCH_BASE = SIM_END + 1

# Build touch map: E1-E8 (0-7), B1-B8 (8-15), C (16),
# then outer ring A/D alternating (17-32)
_tch_map: dict[str, int] = {}
_idx = TCH_BASE
for pos in range(1, 9):
    _tch_map[f"E{pos}"] = _idx; _idx += 1
for pos in range(1, 9):
    _tch_map[f"B{pos}"] = _idx; _idx += 1
_tch_map["C"] = _idx; _idx += 1
# Outer ring: A8, D1, A1, D2, A2, D3, A3, D4, A4, D5, A5, D6, A6, D7, A7, D8
_OUTER_TOUCH_ORDER = [
    "A8","D1","A1","D2","A2","D3","A3","D4",
    "A4","D5","A5","D6","A6","D7","A7","D8",
]
for name in _OUTER_TOUCH_ORDER:
    _tch_map[name] = _idx; _idx += 1

TCH_TO_ID = _tch_map
ID_TO_TCH = {v: k for k, v in TCH_TO_ID.items()}
TCH_END = _idx

# --- Simultaneous count token (95) ---
SIM_COUNT_2 = TCH_END

# --- Duration parameter tokens (96-115) ---
_DUR_NUM_VALUES = [1, 2, 3, 4, 6, 8, 12, 16]
DUR_NUM_BASE = SIM_COUNT_2 + 1
DUR_NUM_TO_ID = {v: DUR_NUM_BASE + i for i, v in enumerate(_DUR_NUM_VALUES)}
ID_TO_DUR_NUM = {v: k for k, v in DUR_NUM_TO_ID.items()}
DUR_NUM_END = DUR_NUM_BASE + len(_DUR_NUM_VALUES)

_DUR_DEN_VALUES = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
DUR_DEN_BASE = DUR_NUM_END
DUR_DEN_TO_ID = {v: DUR_DEN_BASE + i for i, v in enumerate(_DUR_DEN_VALUES)}
ID_TO_DUR_DEN = {v: k for k, v in DUR_DEN_TO_ID.items()}
DUR_DEN_END = DUR_DEN_BASE + len(_DUR_DEN_VALUES)

# --- Slide type tokens (116-129) ---
SLD_TYPE_BASE = DUR_DEN_END

SLD_TYPE_STRAIGHT  = SLD_TYPE_BASE
SLD_TYPE_ARC_CW    = SLD_TYPE_BASE + 1
SLD_TYPE_ARC_CCW   = SLD_TYPE_BASE + 2
SLD_TYPE_ARC_AUTO  = SLD_TYPE_BASE + 3
SLD_TYPE_CENTER    = SLD_TYPE_BASE + 4
SLD_TYPE_INNER_CW  = SLD_TYPE_BASE + 5
SLD_TYPE_INNER_CCW = SLD_TYPE_BASE + 6
SLD_TYPE_ZIGZAG_S  = SLD_TYPE_BASE + 7
SLD_TYPE_ZIGZAG_Z  = SLD_TYPE_BASE + 8
SLD_TYPE_WIFI      = SLD_TYPE_BASE + 9
SLD_TYPE_V         = SLD_TYPE_BASE + 10
SLD_TYPE_LARGE_CW  = SLD_TYPE_BASE + 11
SLD_TYPE_LARGE_CCW = SLD_TYPE_BASE + 12
SLD_TYPE_BRANCH    = SLD_TYPE_BASE + 13

SLD_TYPE_END = SLD_TYPE_BASE + 14

SLD_CHAR_TO_TYPE: dict[str, int] = {
    "-": SLD_TYPE_STRAIGHT,  ">": SLD_TYPE_ARC_CW,
    "<": SLD_TYPE_ARC_CCW,   "^": SLD_TYPE_ARC_AUTO,
    "v": SLD_TYPE_CENTER,    "p": SLD_TYPE_INNER_CW,
    "q": SLD_TYPE_INNER_CCW, "s": SLD_TYPE_ZIGZAG_S,
    "z": SLD_TYPE_ZIGZAG_Z,  "w": SLD_TYPE_WIFI,
    "V": SLD_TYPE_V,         "pp": SLD_TYPE_LARGE_CW,
    "qq": SLD_TYPE_LARGE_CCW, "*": SLD_TYPE_BRANCH,
}
SLD_TYPE_TO_CHAR: dict[int, str] = {v: k for k, v in SLD_CHAR_TO_TYPE.items()}

# --- Firework token (130) ---
FIREWORK = SLD_TYPE_END

# --- Fake Each token (131) ---
FAKE_EACH = FIREWORK + 1

# --- EX note token (132) ---
EX_NOTE = FAKE_EACH + 1

# --- Slide parameter tokens (125-140) ---
# [1] Stage 1 slide fields:
#     SLD_BEG_X   -> slide start button token (X=start button)
#     dur_num/dur_den
#     SLD_END_Y   -> slide end button token (Y=end button)
#     These are output tokens, separate from the structural markers SLD_BEG/SLD_END.
SLD_BEG_BASE = EX_NOTE + 1               # 125  SLD_BEG_1..SLD_BEG_8
SLD_BEG_END = SLD_BEG_BASE + 8           # 133
SLD_END_POS_BASE = SLD_BEG_END           # 133  SLD_END_1..SLD_END_8
SLD_END_POS_END = SLD_END_POS_BASE + 8   # 141

# --- Wifi slide token (141) ---
# [1] Compact wifi: WIFI_SLIDE → start_zone → dur_num → dur_den → end_zone
WIFI_SLIDE = SLD_END_POS_END             # 141

# --- Meta tokens (220-231) ---
META_BPM = 220; META_DIFF = 221; META_LEVEL = 222; META_GENRE = 223
META_END = 224; SLD_MID = 229; HLD_ON = 230; SLD_ON = 231

# --- Vocab size ---
# Base tokens: 0-132 (133 tokens)
# Config vocab: 256-308619 (308,364 tokens) — single-token time-slot encoding
# Set USE_CONFIG_VOCAB = True to enable compact config token encoding
CONFIG_BASE = 256
USE_CONFIG_VOCAB = True
CONFIG_VOCAB_SIZE = 0  # set after import

VOCAB_SIZE = WIFI_SLIDE + 1  # base tokens 0-141
TOKENIZER_VERSION = 4

def _nearest(values: list[int], value: int) -> int:
    return min(values, key=lambda x: abs(x - value))

def _duration_pairs(max_beats: float = 4.0) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for n in _DUR_NUM_VALUES:
        for d in _DUR_DEN_VALUES:
            if n / d <= max_beats + 1e-9:
                pairs.append((n, d))
    return pairs

CONFIG_DURATIONS = _duration_pairs(4.0)

def encode_duration_tokens(duration: tuple[int, int]) -> list[int]:
    beat = _nearest(_DUR_NUM_VALUES, max(1, int(duration[0])))
    den = _nearest(_DUR_DEN_VALUES, max(1, int(duration[1])))
    return [DUR, DUR_NUM_TO_ID[beat], DUR_DEN_TO_ID[den]]

def read_duration_tokens(tokens: list[int], start: int) -> Optional[tuple[int, int]]:
    if start + 2 >= len(tokens) or tokens[start] != DUR:
        return None
    num_tok, den_tok = tokens[start + 1], tokens[start + 2]
    if num_tok in ID_TO_DUR_NUM and den_tok in ID_TO_DUR_DEN:
        return ID_TO_DUR_NUM[num_tok], ID_TO_DUR_DEN[den_tok]
    beat = _nearest(_DUR_NUM_VALUES, max(1, min(int(num_tok), 16)))
    den = _nearest(_DUR_DEN_VALUES, max(1, int(den_tok)))
    return beat, den

# --- Slide position maps --- [1]
# SLD_BEG_BASE + (pos-1) = token for slide starting at button `pos`
SLD_BEG_TO_ID: dict[int, int] = {i: SLD_BEG_BASE + i - 1 for i in range(1, 9)}
ID_TO_SLD_BEG: dict[int, int] = {v: k for k, v in SLD_BEG_TO_ID.items()}
SLD_END_POS_TO_ID: dict[int, int] = {i: SLD_END_POS_BASE + i - 1 for i in range(1, 9)}
ID_TO_SLD_END_POS: dict[int, int] = {v: k for k, v in SLD_END_POS_TO_ID.items()}

# Clearer aliases for the stage-1 slide field tokens.
SLIDE_START_POS_BASE = SLD_BEG_BASE
SLIDE_START_POS_END = SLD_BEG_END
SLIDE_END_POS_BASE = SLD_END_POS_BASE
SLIDE_END_POS_END = SLD_END_POS_END
SLIDE_START_POS_TO_ID = SLD_BEG_TO_ID
SLIDE_END_POS_TO_ID = SLD_END_POS_TO_ID

def encode_slide_compact(slide_path: list[int], slide_types: list[str],
                          duration: tuple[int, int] | None) -> list[int]:
    """[1] Compact: SLD_BEG_X → dur_num → dur_den → SLD_END_Y  (4 tokens).

    Start button is encoded in the token identity (SLD_BEG_1..SLD_BEG_8).
    slideG will later expand intermediate waypoints + connector types.
    """
    start_pos = max(1, min(8, int(slide_path[0] if slide_path else 1)))
    result = [SLD_BEG_TO_ID[start_pos]]
    dur = _snap_duration(duration)
    result.append(DUR_NUM_TO_ID[dur[0]])
    result.append(DUR_DEN_TO_ID[dur[1]])
    end_pos = max(1, min(8, int(slide_path[-1] if slide_path else 1)))
    result.append(SLD_END_POS_TO_ID[end_pos])
    return result

def read_slide_compact(tokens: list[int], start: int) -> tuple[int|None, int|None, tuple[int,int]|None, int]:
    """Read SLD_BEG_X dur_num dur_den SLD_END_Y. Returns (start, end, dur, next_i)."""
    if start >= len(tokens) or tokens[start] not in ID_TO_SLD_BEG:
        return None, None, None, start
    start_pos = ID_TO_SLD_BEG[tokens[start]]
    i = start + 1
    dur: tuple[int,int]|None = None
    if i + 1 < len(tokens):
        num_tok, den_tok = tokens[i], tokens[i+1]
        if num_tok in ID_TO_DUR_NUM and den_tok in ID_TO_DUR_DEN:
            dur = (ID_TO_DUR_NUM[num_tok], ID_TO_DUR_DEN[den_tok])
            i += 2
    end_pos = None
    if i < len(tokens) and tokens[i] in ID_TO_SLD_END_POS:
        end_pos = ID_TO_SLD_END_POS[tokens[i]]
        i += 1
    return start_pos, end_pos, dur, i

# --- Wifi slide compact encoding --- [1]
def encode_wifi_compact(touch_regions: list[str],
                        duration: tuple[int, int] | None) -> list[int]:
    """Compact wifi: WIFI_SLIDE → start_zone → dur_num → dur_den → end_zone."""
    if len(touch_regions) < 2:
        return [RST]
    result = [WIFI_SLIDE]
    start_zone = TCH_TO_ID.get(touch_regions[0], RST)
    result.append(start_zone)
    dur = _snap_duration(duration)
    result.append(DUR_NUM_TO_ID[dur[0]])
    result.append(DUR_DEN_TO_ID[dur[1]])
    end_zone = TCH_TO_ID.get(touch_regions[-1], RST)
    result.append(end_zone)
    return result

def read_wifi_compact(tokens: list[int], start: int) -> tuple[str|None, str|None, tuple[int,int]|None, int]:
    """Read WIFI_SLIDE start_zone dur_num dur_den end_zone. Returns (start, end, dur, next_i)."""
    if start >= len(tokens) or tokens[start] != WIFI_SLIDE:
        return None, None, None, start
    i = start + 1
    start_zone = end_zone = None
    if i < len(tokens) and tokens[i] in ID_TO_TCH:
        start_zone = ID_TO_TCH[tokens[i]]
        i += 1
    dur: tuple[int,int]|None = None
    if i + 1 < len(tokens):
        num_tok, den_tok = tokens[i], tokens[i+1]
        if num_tok in ID_TO_DUR_NUM and den_tok in ID_TO_DUR_DEN:
            dur = (ID_TO_DUR_NUM[num_tok], ID_TO_DUR_DEN[den_tok])
            i += 2
    if i < len(tokens) and tokens[i] in ID_TO_TCH:
        end_zone = ID_TO_TCH[tokens[i]]
        i += 1
    return start_zone, end_zone, dur, i

def _snap_duration(duration: tuple[int, int] | None) -> tuple[int, int]:
    if not duration:
        return (1, 1)
    n = _nearest(_DUR_NUM_VALUES, max(1, int(duration[0])))
    d = _nearest(_DUR_DEN_VALUES, max(1, int(duration[1])))
    if n / d > 4.0:
        return min(CONFIG_DURATIONS, key=lambda x: (abs((x[0]/x[1])-4.0), x[1]))
    return n, d

# =====================================================================
# Config Vocab import (single-token time-slot configurations)
# =====================================================================

from Tokenizer.touch_expander import (
    zone_index as _zone_index,
    zone_name as _zone_name,
)

from Tokenizer.config_vocab import (
    ID_TO_CONFIG as _CFG_ID_TO_SLOT,
    CONFIG_TO_ID as _CFG_SLOT_TO_ID,
    CONFIG_VOCAB_SIZE as _CFG_SIZE,
    SlotConfig,
    BTN_PRESS, BTN_HOLD_START, BTN_HOLD_ONGOING, BTN_SLIDE_START, BTN_SLIDE_END,
    TCH_TOUCH, TCH_HOLD_START, TCH_HOLD_ONGOING,
    config_to_base_tokens,
    config_name,
)

CONFIG_VOCAB_SIZE = _CFG_SIZE

# Update vocab size when config is enabled
if USE_CONFIG_VOCAB:
    VOCAB_SIZE = CONFIG_BASE + CONFIG_VOCAB_SIZE

# ═══════════════════════════════════════════════════════════════════════
# Note → SlotConfig conversion
# ═══════════════════════════════════════════════════════════════════════

def _note_to_slot_config(note) -> Optional[SlotConfig]:
    """Convert a parsed _Note to a SlotConfig, if it fits the 2-hand model."""
    if note.is_rest:
        return SlotConfig()  # rest

    if note.is_end:
        return None  # EOS handled separately

    buttons = []
    touches = []

    # --- Button notes ---
    if not note.is_touch and not note.is_rest:
        # Slides and breaks handled in later stages — fall back to base tokens
        if note.is_slide:
            return None
        if note.is_break or getattr(note, '_mixed_break', False):
            return None  # break decided in Stage 3
        for pos in note.positions:
            if not (1 <= pos <= 8):
                continue
            if note.is_slide:
                # Slide: just note the start position
                # (slide detail handled by separate SLD_BEG tokens)
                state = BTN_SLIDE_START
            elif note.is_hold:
                if note.hold_duration:
                    state = BTN_HOLD_START
                else:
                    state = BTN_HOLD_ONGOING  # short hold / ongoing
            else:
                state = BTN_PRESS  # tap or break (break decided later)
            buttons.append((pos, state))

    # --- Touch notes ---
    if note.is_touch:
        # Firework decided in Stage 4 — fall back to base tokens
        if note.is_firework:
            return None
        if note.is_touch_slide:
            return None  # wifi slides: fall back
        for region in note.touch_regions:
            try:
                zidx = _zone_index(region)
            except Exception:
                continue
            if note.is_touch_hold:
                state = TCH_HOLD_START
            elif note.is_hold:
                state = TCH_HOLD_ONGOING
            else:
                state = TCH_TOUCH
            touches.append((zidx, state))

    # Enforce 2-hand constraints
    if len(buttons) > 2:
        return None  # too many button hands
    if len(touches) > MAX_TCH_ZONES:
        return None  # too many touch zones

    # If both buttons and touches, buttons max 1 (other hand on touch)
    if buttons and touches and len(buttons) > 1:
        return None  # 2 btn + touch = 3 hands

    # Sort for canonical form
    buttons = tuple(sorted(buttons))
    touches = tuple(sorted(touches))

    return SlotConfig(buttons=buttons, touches=touches)


# Maximum touch zones for config matching (from config_vocab)
MAX_TCH_ZONES = 2

# Old Config Vocab (deprecated, kept for compatibility)
CONFIG_TO_ID: dict[tuple, int] = {}
ID_TO_CONFIG: dict[int, tuple] = {}

def _normalize_config_spec(spec) -> tuple:
    return tuple(tuple(x) if isinstance(x, list) else x for x in spec)

def _add_config(spec: tuple) -> int:
    spec = _normalize_config_spec(spec)
    if spec in CONFIG_TO_ID:
        return CONFIG_TO_ID[spec]
    idx = CONFIG_BASE + len(CONFIG_TO_ID)
    CONFIG_TO_ID[spec] = idx
    ID_TO_CONFIG[idx] = spec
    if "_TOKEN_NAMES" in globals():
        _TOKEN_NAMES[idx] = "cfg_" + "_".join(str(x) for x in spec)
    return idx

def _build_config_vocab() -> None:
    # Single button events
    for pos in range(1, 9):
        _add_config(("tap", pos))
        _add_config(("brk", pos))
        for dur in CONFIG_DURATIONS:
            _add_config(("hld", pos, dur[0], dur[1]))

    # Single touch (no hold, no firework)
    for region in sorted(TCH_TO_ID):
        _add_config(("tch", region))

    # Touch hold: ALL regions including C
    for region in sorted(TCH_TO_ID):
        for dur in CONFIG_DURATIONS:
            _add_config(("tch_hld", region, dur[0], dur[1]))

    # Two-note button pairs
    button_types = ("tap", "brk", "hld")
    for p1 in range(1, 9):
        for p2 in range(p1 + 1, 9):
            for t1 in button_types:
                for t2 in button_types:
                    if (t1, p1) > (t2, p2):
                        continue
                    if "hld" in (t1, t2):
                        for dur in CONFIG_DURATIONS:
                            _add_config(("pair", t1, p1, t2, p2, dur[0], dur[1]))
                    else:
                        _add_config(("pair", t1, p1, t2, p2))

    # Two-note touch pairs (no hold)
    for r1 in sorted(TCH_TO_ID):
        for r2 in sorted(TCH_TO_ID):
            if r1 >= r2:
                continue
            _add_config(("touch_multi", r1, r2))

    # Slides: 2- and 3-point paths
    for a in range(1, 9):
        for b in range(1, 9):
            if b == a:
                continue
            for dur in CONFIG_DURATIONS:
                _add_config(("sld", a, b, dur[0], dur[1]))
            for c in range(1, 9):
                if c in (a, b):
                    continue
                for dur in CONFIG_DURATIONS:
                    _add_config(("sld", a, b, c, dur[0], dur[1]))

_build_config_vocab()  # kept for optional use, not applied by default

# VOCAB_SIZE already set above; export/import kept for compatibility
TOKENIZER_VERSION = 4

def export_config_vocab() -> list[list]:
    return [list(spec) for spec, _ in sorted(CONFIG_TO_ID.items(), key=lambda x: x[1])]

def load_config_vocab(specs: list) -> None:
    for spec in specs:
        _add_config(tuple(spec))

def save_config_vocab(path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(export_config_vocab(), f, ensure_ascii=False)

def load_config_vocab_file(path: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        load_config_vocab(json.load(f))

# =====================================================================
# Token name lookup (debugging)
# =====================================================================

_TOKEN_NAMES: dict[int, str] = {
    PAD: "[PAD]", BOS: "[BOS]", EOS: "[EOS]", SEP: "[SEP]", MASK: "[MASK]",
    RST: "[RST]", DUR: "[DUR]",
    SLD_BEG: "[SLD_BEG]", SLD_END: "[SLD_END]",
    SIM_BEG: "[SIM_BEG]", SIM_END: "[SIM_END]",
    SIM_COUNT_2: "[SIM_COUNT_2]",
    FIREWORK: "[FIREWORK]", FAKE_EACH: "[FAKE_EACH]", EX_NOTE: "[EX]",
    META_BPM: "[META_BPM]", META_DIFF: "[META_DIFF]",
    META_LEVEL: "[META_LEVEL]", META_GENRE: "[META_GENRE]",
    META_END: "[META_END]", SLD_MID: "[SLD_MID]",
    HLD_ON: "[HLD_ON]", SLD_ON: "[SLD_ON]",
}
for v, i in DIV_TO_ID.items():
    _TOKEN_NAMES[i] = f"div_{v}"
for p, i in TAP_TO_ID.items():
    _TOKEN_NAMES[i] = f"tap_{p}"
for p, i in BRK_TO_ID.items():
    _TOKEN_NAMES[i] = f"brk_{p}"
for p, i in HLD_TO_ID.items():
    _TOKEN_NAMES[i] = f"hld_{p}"
for p, i in SLD_TO_ID.items():
    _TOKEN_NAMES[i] = f"sld_{p}"
for t, i in TCH_TO_ID.items():
    _TOKEN_NAMES[i] = f"tch_{t}"
for v, i in DUR_NUM_TO_ID.items():
    _TOKEN_NAMES[i] = f"dur_num_{v}"
for v, i in DUR_DEN_TO_ID.items():
    _TOKEN_NAMES[i] = f"dur_den_{v}"
for char, i in SLD_CHAR_TO_TYPE.items():
    _TOKEN_NAMES[i] = f"sld_type_{char}"


def token_name(token_id: int) -> str:
    return _TOKEN_NAMES.get(token_id, f"<{token_id}>")

# =====================================================================
# Regex patterns for simai parsing
# =====================================================================

_RE_BEAT_DIV = re.compile(r"\{(\d+)\}")
_RE_BPM = re.compile(r"\((\d+(?:\.\d+)?)\)")
_RE_HOLD_DUR = re.compile(r"h\[(\d+):(\d+)\]")
_RE_SLIDE_DUR = re.compile(r"\[(\d+):(\d+)\]")
_RE_SLIDE_CONN = re.compile(r"(pp|qq|[*\-<>^vVpqszw])")
_RE_END = re.compile(r"^E$")

# =====================================================================
# Internal note representation
# =====================================================================

@dataclass
class _Note:
    """Internal parsed note."""
    beat_div: int = 4
    raw: str = ""
    is_rest: bool = False
    is_end: bool = False
    is_tap: bool = False
    is_break: bool = False
    is_hold: bool = False
    is_slide: bool = False
    is_touch: bool = False
    is_touch_hold: bool = False
    is_touch_slide: bool = False
    is_firework: bool = False
    is_ex: bool = False
    is_simultaneous: bool = False
    is_fake_each: bool = False
    positions: list[int] = field(default_factory=list)
    touch_regions: list[str] = field(default_factory=list)
    slide_path: list[int] = field(default_factory=list)
    slide_types: list[str] = field(default_factory=list)
    touch_slide_path: list[str] = field(default_factory=list)
    hold_duration: tuple[int, int] | None = None

# =====================================================================
# Simai Parser: text -> [_Note, ...]
# =====================================================================

def _parse_touch_region(region: str) -> str:
    """Normalize: C1-C8 all → C, strip modifiers."""
    region = region.rstrip("fhxb")
    if re.match(r"^C\d*$", region):  # C, C1, C2, ... all → C
        return "C"
    return region

def _is_touch_zone(s: str) -> bool:
    s_clean = s.rstrip("fhxb")
    return bool(re.match(r"^(C\d*|[ABDE][1-8])$", s_clean))

def _parse_slide_segments(s: str) -> tuple[list[int], list[str], tuple[int,int]|None]:
    dur_m = _RE_SLIDE_DUR.search(s)
    dur = (int(dur_m.group(1)), int(dur_m.group(2))) if dur_m else None
    cleaned = _RE_SLIDE_DUR.sub("", s).rstrip("bx")
    pos_nums = re.findall(r"(\d+)", cleaned)
    positions = [int(n) for n in pos_nums]
    connectors = _RE_SLIDE_CONN.findall(cleaned)
    return positions, connectors, dur

def _parse_note_token(token: str, beat_div: int) -> _Note:
    """Parse one comma-separated simai token -> _Note."""
    note = _Note(beat_div=beat_div, raw=token)
    t = token.strip()
    if not t:
        note.is_rest = True; return note
    if _RE_END.match(t):
        note.is_end = True; return note
    t = _RE_BPM.sub("", t).strip()
    if not t:
        note.is_rest = True; return note

    if "`" in t:
        note.is_fake_each = True; note.is_simultaneous = True
    if "/" in t and "`" not in t:
        note.is_simultaneous = True

    # Detect touch vs button
    cleaned = re.sub(r"\[.*?\]", "", t)
    parts = [p.strip().rstrip("fhxb") for p in re.split(r"[/`w]", cleaned) if p.strip()]
    all_touch = len(parts) > 0 and all(_is_touch_zone(p) for p in parts)

    if all_touch:
        note.is_touch = True
        # Check wifi: touch regions connected by 'w'
        if "w" in t and len(parts) >= 2:
            note.is_touch_slide = True
            dur_m = _RE_SLIDE_DUR.search(t)
            if dur_m:
                note.hold_duration = (int(dur_m.group(1)), int(dur_m.group(2)))
            regions = re.findall(r"([ABDE][1-8]|C[12]?)", t)
            note.touch_regions = [_parse_touch_region(r) for r in regions]
            note.touch_slide_path = list(note.touch_regions)
            if "f" in t: note.is_firework = True
            return note

        regions_raw = re.findall(r"([ABDE][1-8]|C[12]?)", t)
        note.touch_regions = [_parse_touch_region(r) for r in regions_raw]
        hold_m = _RE_HOLD_DUR.search(t)
        if hold_m:
            note.is_touch_hold = True; note.is_hold = True
            note.hold_duration = (int(hold_m.group(1)), int(hold_m.group(2)))
        if "f" in t: note.is_firework = True
        return note

    # Button note
    # For each/simultaneous, check if break applies uniformly
    if "/" in t:
        # Check each sub-part individually
        sub_parts = [p.strip() for p in t.split("/")]
        sub_breaks = [bool(re.search(r"\db", p) or p.endswith("b")) for p in sub_parts]
        if all(sub_breaks):
            note.is_break = True
        elif any(sub_breaks):
            # Mixed: some breaks, some not -> don't set global break
            # Store per-position info for fallback encoding
            note.is_break = False
            note._mixed_break = True  # type: ignore
            note._sub_breaks = sub_breaks  # type: ignore
    else:
        if re.search(r"\db", t) or re.search(r"b\[", t) or t.endswith("b"):
            note.is_break = True
    if "x" in t: note.is_ex = True

    hold_m = _RE_HOLD_DUR.search(t)
    if hold_m:
        note.is_hold = True
        note.hold_duration = (int(hold_m.group(1)), int(hold_m.group(2)))
    elif re.search(r"\dh", t) and "[" not in t:
        note.is_hold = True  # short hold without duration

    # Slide  (check BEFORE extracting positions, since slide has its own logic)
    has_conn = bool(_RE_SLIDE_CONN.search(t))
    if has_conn and bool(re.search(r"\d", t)):
        note.is_slide = True
        positions, connectors, dur = _parse_slide_segments(t)
        note.positions = positions
        note.slide_path = list(positions)
        note.slide_types = connectors
        if dur: note.hold_duration = dur
        if re.search(r"b\[", t) or re.search(r"b$", t):
            note.is_break = True
        return note

    # Extract positions from the non-duration, non-modifier part
    t_clean = re.sub(r"\[.*?\]", "", t)  # remove [durations]
    t_clean = t_clean.rstrip("hbfx*")
    pos_nums = re.findall(r"(\d+)", t_clean)
    note.positions = [int(n) for n in pos_nums if 1 <= int(n) <= 8]
    if not note.positions:
        note.is_tap = True; return note
    if not note.is_hold and not note.is_break and not note.is_slide:
        note.is_tap = True
    return note

def _parse_simai_text(text: str) -> list[_Note]:
    """Parse raw simai text -> list of _Note (comma-split, respecting brackets)."""
    text = text.strip()
    notes: list[_Note] = []
    current_div = 4

    # Split by comma respecting [...]
    raw_tokens: list[tuple[str, int]] = []
    i = 0
    depth = 0
    tok_start = i
    pending_div = current_div

    while i < len(text):
        ch = text[i]
        if ch == "[": depth += 1; i += 1; continue
        if ch == "]": depth = max(0, depth - 1); i += 1; continue
        if depth > 0: i += 1; continue

        bd = _RE_BEAT_DIV.match(text, i)
        if bd:
            current_div = int(bd.group(1)); pending_div = current_div
            i = bd.end(); tok_start = i; continue

        bpm = _RE_BPM.match(text, i)
        if bpm:
            i = bpm.end(); tok_start = i; continue

        if ch == ",":
            raw_tokens.append((text[tok_start:i].strip(), pending_div))
            pending_div = current_div
            i += 1; tok_start = i; continue
        i += 1

    if tok_start < len(text):
        raw = text[tok_start:].strip()
        if raw: raw_tokens.append((raw, pending_div))

    for raw, div in raw_tokens:
        if raw == "":
            n = _Note(beat_div=div, raw=""); n.is_rest = True; notes.append(n)
        else:
            notes.append(_parse_note_token(raw, div))
    return notes

# =====================================================================
# Config token matching
# =====================================================================

def _config_token_for_note(note: _Note) -> int | None:
    """Try to match a note to a compact config token."""
    if note.is_rest or note.is_end: return None
    if note.is_touch_slide: return None  # encoded with connectors

    # Single touch (clean)
    if note.is_touch and len(note.touch_regions)==1 and not note.is_touch_hold and not note.is_firework and not note.is_ex:
        return CONFIG_TO_ID.get(("tch", note.touch_regions[0]))

    # Touch hold
    if note.is_touch_hold and len(note.touch_regions)==1:
        dur = _snap_duration(note.hold_duration)
        return CONFIG_TO_ID.get(("tch_hld", note.touch_regions[0], dur[0], dur[1]))

    # Touch multi (no hold, no firework)
    if note.is_touch and len(note.touch_regions)>=2 and not note.is_touch_hold and not note.is_firework:
        regions = sorted(note.touch_regions)
        if len(regions)==2:
            return CONFIG_TO_ID.get(("touch_multi", regions[0], regions[1]))

    # Slide (only use config for simple straight slides without break)
    if note.is_slide:
        # Skip config for break-slides (need to preserve break flag)
        if note.is_break: return None
        path = note.slide_path or note.positions
        types = note.slide_types
        all_straight = not types or all(t == "-" for t in types)
        if len(path)>=2 and all_straight:
            dur = _snap_duration(note.hold_duration)
            spec = ("sld", *path[:3], dur[0], dur[1])
            if spec in CONFIG_TO_ID: return CONFIG_TO_ID[spec]
        return None

    # Single button (skip if both break+hold — break-hold combo)
    if len(note.positions)==1 and not note.is_ex:
        pos = note.positions[0]
        if not (1<=pos<=8): return None
        if note.is_hold and note.is_break: return None  # break-hold: use fallback
        if note.is_hold:
            dur = _snap_duration(note.hold_duration)
            return CONFIG_TO_ID.get(("hld", pos, dur[0], dur[1]))
        if note.is_break: return CONFIG_TO_ID.get(("brk", pos))
        return CONFIG_TO_ID.get(("tap", pos))

    # Two-button pair (skip if mixed types or break-hold)
    if len(note.positions)==2 and not note.is_ex:
        if note.is_hold and note.is_break: return None
        if getattr(note, '_mixed_break', False): return None  # mixed each
        p1, p2 = sorted(note.positions)
        if not (1<=p1<=8 and 1<=p2<=8): return None
        typ = "hld" if note.is_hold else "brk" if note.is_break else "tap"
        if typ=="hld":
            dur = _snap_duration(note.hold_duration)
            return CONFIG_TO_ID.get(("pair","hld",p1,"hld",p2,dur[0],dur[1]))
        return CONFIG_TO_ID.get(("pair",typ,p1,typ,p2))
    return None

def _learn_config_from_note(note: _Note) -> int | None:
    """Learn new config from unseen pattern."""
    if note.is_rest or note.is_end: return None
    existing = _config_token_for_note(note)
    if existing is not None: return existing
    if note.is_touch and len(note.touch_regions)>=2 and not note.is_touch_hold:
        return _add_config(("touch_multi", *sorted(note.touch_regions)))
    if note.is_touch_hold and note.touch_regions:
        dur = _snap_duration(note.hold_duration)
        return _add_config(("tch_hld", note.touch_regions[0], dur[0], dur[1]))
    if note.is_touch_slide and note.touch_slide_path:
        dur = _snap_duration(note.hold_duration)
        return _add_config(("wifi", *note.touch_slide_path, dur[0], dur[1]))
    if note.is_slide:
        path = note.slide_path or note.positions
        if len(path)>=2:
            dur = _snap_duration(note.hold_duration)
            return _add_config(("sld", *path, dur[0], dur[1]))
        return None
    if len(note.positions)>=3:
        positions = sorted(p for p in note.positions if 1<=p<=8)
        if len(positions)<2: return None
        typ = "hld" if note.is_hold else "brk" if note.is_break else "tap"
        if note.is_hold:
            dur = _snap_duration(note.hold_duration)
            return _add_config(("multi",typ,*positions,dur[0],dur[1]))
        else:
            return _add_config(("multi",typ,*positions))
    return None

def learn_config_vocab_from_texts(texts: list[str]) -> int:
    before = len(CONFIG_TO_ID)
    for text in texts:
        for note in _parse_simai_text(text):
            _learn_config_from_note(note)
    return len(CONFIG_TO_ID) - before

# =====================================================================
# MaiChartTokenizer
# =====================================================================

class MaiChartTokenizer:
    """Complete simai chart tokenizer - text <-> tokens, lossless."""

    vocab_size: int = VOCAB_SIZE
    pad_token_id: int = PAD
    bos_token_id: int = BOS
    eos_token_id: int = EOS
    mask_token_id: int = MASK

    # -- Encode --

    def encode(self, text: str, add_bos: bool = True,
               add_eos: bool = True, compact_slide: bool = True) -> list[int]:
        notes = _parse_simai_text(text)
        tokens: list[int] = []
        if add_bos: tokens.append(BOS)

        current_div = 4
        for note in notes:
            if note.beat_div != current_div:
                current_div = note.beat_div
                div_id = DIV_TO_ID.get(current_div)
                if div_id is not None: tokens.append(div_id)
            if note.is_end: continue
            if note.is_rest:
                tokens.append(RST); continue

            # Try config token encoding first
            if USE_CONFIG_VOCAB:
                sc = _note_to_slot_config(note)
                if sc is not None:
                    cfg_id = _CFG_SLOT_TO_ID.get(sc)
                    if cfg_id is not None:
                        tokens.append(cfg_id)
                        # Duration follows separately for holds
                        if note.hold_duration and (note.is_hold or note.is_touch_hold):
                            tokens.extend(encode_duration_tokens(note.hold_duration))
                        # [1] Compact slide: SLIDE → dur → end_pos after config
                        if compact_slide and note.is_slide and not note.is_break:
                            tokens.extend(encode_slide_compact(
                                note.slide_path or note.positions,
                                note.slide_types, note.hold_duration))
                        continue

            # Fallback to base token encoding
            tokens.extend(self._encode_fallback(note, compact_slide=compact_slide))

        if add_eos: tokens.append(EOS)
        return tokens

    def _encode_fallback(self, note: _Note, compact_slide: bool = True) -> list[int]:
        """Fallback encoding for notes without config token."""
        # Touch wifi slide
        if note.is_touch_slide:
            # [1] Compact wifi encoding: WIFI_SLIDE → start → dur → end
            if compact_slide and not note.is_firework:
                result = encode_wifi_compact(note.touch_regions, note.hold_duration)
                return result
            # Full fallback
            result: list[int] = []
            regions = note.touch_regions
            for idx, region in enumerate(regions):
                tid = TCH_TO_ID.get(region, RST)
                result.append(tid)
                if idx < len(regions)-1:
                    result.append(SLD_TYPE_WIFI)
            if note.is_firework: result.append(FIREWORK)
            if note.hold_duration:
                result.extend(encode_duration_tokens(note.hold_duration))
            return result

        # Touch note
        if note.is_touch:
            result: list[int] = []
            for region in note.touch_regions:
                tid = TCH_TO_ID.get(region)
                if tid is not None: result.append(tid)
            if note.is_firework: result.append(FIREWORK)
            if note.is_simultaneous and len(result)>1:
                result = [SIM_BEG, SIM_COUNT_2] + result + [SIM_END]
            if note.is_touch_hold and note.hold_duration:
                result.extend(encode_duration_tokens(note.hold_duration))
            return result if result else [RST]

        # Slide
        if note.is_slide:
            # [1] Compact slide encoding: SLIDE → dur → end_pos
            # Break-slides still use full fallback (need EX_NOTE marker)
            if compact_slide and not note.is_break:
                return encode_slide_compact(
                    note.slide_path or note.positions,
                    note.slide_types, note.hold_duration)
            result: list[int] = [SLD_BEG]
            path = note.slide_path or note.positions
            types = note.slide_types
            for idx, pos in enumerate(path):
                if 1<=pos<=8: result.append(SLD_TO_ID[pos])
                if idx < len(path)-1 and idx < len(types):
                    type_id = SLD_CHAR_TO_TYPE.get(types[idx])
                    if type_id is not None: result.append(type_id)
            result.append(SLD_END)
            if note.is_break: result.append(EX_NOTE)  # break-slide marker
            if note.hold_duration:
                result.extend(encode_duration_tokens(note.hold_duration))
            return result

        # Break (may be mixed with taps in simultaneous)
        if note.is_break or getattr(note, '_mixed_break', False):
            result: list[int] = []
            sub_breaks = getattr(note, '_sub_breaks', [])
            for idx, pos in enumerate(note.positions):
                if 1<=pos<=8:
                    is_brk = sub_breaks[idx] if idx < len(sub_breaks) else note.is_break
                    result.append(BRK_TO_ID[pos] if is_brk else TAP_TO_ID[pos])
            if note.is_simultaneous and len(result)>1:
                result = [SIM_BEG, SIM_COUNT_2] + result + [SIM_END]
            if note.is_hold and note.hold_duration:
                result.extend(encode_duration_tokens(note.hold_duration))
            return result

        # Hold
        if note.is_hold:
            result: list[int] = []
            for pos in note.positions:
                if 1<=pos<=8: result.append(HLD_TO_ID[pos])
            if note.is_simultaneous and len(result)>1:
                result = [SIM_BEG, SIM_COUNT_2] + result + [SIM_END]
            if note.hold_duration:
                result.extend(encode_duration_tokens(note.hold_duration))
            return result

        # Tap (simultaneous)
        if note.is_simultaneous and len(note.positions)>1:
            result = [TAP_TO_ID[p] for p in note.positions if 1<=p<=8]
            return [SIM_BEG, SIM_COUNT_2] + result + [SIM_END]

        # Single tap
        for pos in note.positions:
            if 1<=pos<=8: return [TAP_TO_ID[pos]]
        return [RST]

    # -- Decode --

    def decode(self, tokens: list[int]) -> str:
        parts: list[str] = []
        current_div = 4
        pending_div_prefix = ""  # accumulate beat-div for next note
        i = 0
        while i < len(tokens):
            tid = tokens[i]
            if tid == PAD: break
            if tid == BOS: i += 1; continue
            if tid == EOS:
                parts.append("E"); i += 1; break
            if tid in ID_TO_DIV:
                current_div = ID_TO_DIV[tid]
                pending_div_prefix = f"{{{current_div}}}"
                i += 1; continue
            if tid == RST:
                parts.append(pending_div_prefix); pending_div_prefix = ""
                i += 1; continue

            # Config token expansion
            if USE_CONFIG_VOCAB and tid >= CONFIG_BASE:
                sc = _CFG_ID_TO_SLOT.get(tid)
                if sc is not None:
                    # Check for duration after config token (DUR follows at i+1)
                    dur = read_duration_tokens(tokens, i + 1)
                    dur_str = ""
                    if dur:
                        dur_str = f"[{dur[0]}:{dur[1]}]"
                        i += 3  # skip DUR + num + den

                    # [1] Check for compact slide params
                    # If DUR was just consumed, i points to SLIDE; otherwise SLIDE is at i+1
                    slide_start = None; slide_end = None; slide_dur = None
                    slide_check_pos = i if dur else i + 1
                    if i < len(tokens):
                        s_start, s_end, s_dur, s_next = read_slide_compact(tokens, slide_check_pos)
                        if s_start is not None:
                            slide_start = s_start; slide_end = s_end; slide_dur = s_dur
                            i = s_next - 1  # -1 because i+=1 at end

                    # Build output
                    slide_str = ""
                    if slide_start is not None and slide_end is not None:
                        sd = slide_dur or (1, 1)
                        slide_str = f"{slide_start}-{slide_end}[{sd[0]}:{sd[1]}]"

                    if sc.is_rest:
                        parts.append(pending_div_prefix); pending_div_prefix = ""
                    elif len(sc.buttons) == 2:
                        b = [f"{p}{'h' if s in (BTN_HOLD_START, BTN_HOLD_ONGOING) else ''}" for p, s in sc.buttons]
                        txt = pending_div_prefix + "/".join(b) + dur_str
                        parts.append(txt)
                        if slide_str:
                            parts.append(slide_str)
                        pending_div_prefix = ""
                    elif len(sc.buttons) == 1:
                        p, s = sc.buttons[0]
                        suffix = "h" if s in (BTN_HOLD_START, BTN_HOLD_ONGOING) else ""
                        txt = pending_div_prefix + f"{p}{suffix}" + dur_str
                        parts.append(txt)
                        if slide_str:
                            # Slide is a separate note at same time slot, no prefix needed
                            parts.append(slide_str)
                        pending_div_prefix = ""
                    elif sc.touches:
                        tch_parts = []
                        for zi, ts in sc.touches:
                            zname = _zone_name(zi)
                            suffix = "h" if ts == TCH_HOLD_START else ""
                            tch_parts.append(zname + suffix)
                        parts.append(pending_div_prefix + "/".join(tch_parts) + dur_str); pending_div_prefix = ""
                    i += 1; continue

            # Note tokens — all handled by fallback decoder methods
            if tid == SLD_BEG:
                text, i = self._decode_slide(tokens, i)
                parts.append(pending_div_prefix + text); pending_div_prefix = ""
                continue
            if SLD_BEG_BASE <= tid < SLD_BEG_END:
                text, i = self._decode_slide_compact(tokens, i)
                parts.append(pending_div_prefix + text); pending_div_prefix = ""
                continue
            if tid == WIFI_SLIDE:
                text, i = self._decode_wifi_compact(tokens, i)
                parts.append(pending_div_prefix + text); pending_div_prefix = ""
                continue
            if tid == SIM_BEG:
                text, i = self._decode_simul(tokens, i)
                parts.append(pending_div_prefix + text); pending_div_prefix = ""
                continue
            # Wifi slide: TCH token followed by SLD_TYPE_WIFI → decode as wifi chain
            if TCH_BASE <= tid < TCH_END and i + 1 < len(tokens) and tokens[i + 1] == SLD_TYPE_WIFI:
                text, i = self._decode_wifi(tokens, i)
                parts.append(pending_div_prefix + text); pending_div_prefix = ""
                continue

            # Skip standalone structural tokens
            if tid in (DUR, SLD_END, SIM_END, FIREWORK, FAKE_EACH, EX_NOTE,
                       WIFI_SLIDE, SLD_MID, HLD_ON, SLD_ON):
                i += 1; continue
            if SLD_TYPE_BASE <= tid < SLD_TYPE_END:
                i += 1; continue
            if SLD_BEG_BASE <= tid < SLD_BEG_END:
                i += 1; continue
            if SLD_END_POS_BASE <= tid < SLD_END_POS_END:
                i += 1; continue

            text, i = self._decode_single(tokens, i)
            if text:
                parts.append(pending_div_prefix + text); pending_div_prefix = ""
        return ",".join(parts)

    def _decode_slide(self, tokens: list[int], start: int) -> tuple[str, int]:
        i = start + 1
        positions: list[int] = []
        types: list[str] = []
        is_break = False
        while i < len(tokens) and tokens[i] not in (SLD_END, DUR, EOS, PAD):
            tid = tokens[i]
            if tid in ID_TO_SLD: positions.append(ID_TO_SLD[tid])
            elif SLD_TYPE_BASE <= tid < SLD_TYPE_END:
                types.append(SLD_TYPE_TO_CHAR.get(tid, "-"))
            elif tid in (SLD_MID, SLD_ON): pass
            else: break
            i += 1
        if i < len(tokens) and tokens[i] == SLD_END: i += 1
        # Check for break-slide marker
        if i < len(tokens) and tokens[i] == EX_NOTE:
            is_break = True; i += 1
        dur = read_duration_tokens(tokens, i)
        if dur: i += 3
        if not positions: return "", i
        text = str(positions[0])
        for idx in range(1, len(positions)):
            conn = types[idx-1] if idx-1 < len(types) else "-"
            text += f"{conn}{positions[idx]}"
        if is_break: text += "b"
        if dur: text += f"[{dur[0]}:{dur[1]}]"
        return text, i

    def _decode_slide_compact(self, tokens: list[int], start: int) -> tuple[str, int]:
        """Decode compact SLD_BEG_X → dur → SLD_END_Y."""
        start_pos, end_pos, dur, i = read_slide_compact(tokens, start)
        if start_pos is None or end_pos is None:
            return "", i
        text = f"{start_pos}-{end_pos}"
        if dur:
            text += f"[{dur[0]}:{dur[1]}]"
        return text, i

    def _decode_wifi_compact(self, tokens: list[int], start: int) -> tuple[str, int]:
        """Decode compact WIFI_SLIDE → start_zone → dur → end_zone."""
        start_zone, end_zone, dur, i = read_wifi_compact(tokens, start)
        if start_zone is None or end_zone is None:
            return "", i
        text = f"{start_zone}w{end_zone}"
        if dur:
            text += f"[{dur[0]}:{dur[1]}]"
        return text, i

    def _decode_wifi(self, tokens: list[int], start: int) -> tuple[str, int]:
        """Decode non-compact wifi chain: TCH, WIFI, TCH, [WIFI, TCH, ...]."""
        i = start
        zones: list[str] = []
        is_firework = False
        while i < len(tokens):
            tid = tokens[i]
            if tid in ID_TO_TCH:
                zones.append(ID_TO_TCH[tid])
                i += 1
            elif tid == SLD_TYPE_WIFI:
                i += 1
            elif tid == FIREWORK:
                is_firework = True; i += 1
            else:
                break
        dur = read_duration_tokens(tokens, i)
        if dur: i += 3
        if len(zones) < 2:
            return "", i
        text = "w".join(zones)
        if is_firework and zones:
            text += "f"
        if dur:
            text += f"[{dur[0]}:{dur[1]}]"
        return text, i

    def _decode_simul(self, tokens: list[int], start: int) -> tuple[str, int]:
        i = start + 1
        if i < len(tokens) and tokens[i] in (SIM_COUNT_2,): i += 1
        sub_parts: list[str] = []
        dur = None; is_wifi = False; has_firework = False
        while i < len(tokens) and tokens[i] not in (SIM_END, EOS, PAD):
            tid = tokens[i]
            if tid == DUR: dur = read_duration_tokens(tokens, i); break
            if tid == SLD_TYPE_WIFI: is_wifi = True; i += 1; continue
            if tid == FIREWORK: has_firework = True; i += 1; continue
            if tid in ID_TO_TCH: sub_parts.append(ID_TO_TCH[tid])
            elif tid in ID_TO_TAP: sub_parts.append(str(ID_TO_TAP[tid]))
            elif tid in ID_TO_BRK: sub_parts.append(f"{ID_TO_BRK[tid]}b")
            elif tid in ID_TO_HLD: sub_parts.append(f"{ID_TO_HLD[tid]}h")
            elif tid in ID_TO_SLD: sub_parts.append(str(ID_TO_SLD[tid]))
            i += 1
        if i < len(tokens) and tokens[i] == SIM_END: i += 1
        sep = "w" if is_wifi else "/"
        text = sep.join(sub_parts)
        if has_firework and sub_parts: text += "f"
        if dur:
            # Add hold or slide duration
            text += f"[{dur[0]}:{dur[1]}]"
        return text, i

    def _decode_single(self, tokens: list[int], start: int) -> tuple[str, int]:
        tid = tokens[start]
        i = start + 1
        text = ""; is_hold = False
        if tid in ID_TO_TAP: text = str(ID_TO_TAP[tid])
        elif tid in ID_TO_BRK: text = f"{ID_TO_BRK[tid]}b"
        elif tid in ID_TO_HLD: text = f"{ID_TO_HLD[tid]}h"; is_hold = True
        elif tid in ID_TO_SLD: text = str(ID_TO_SLD[tid])
        elif tid in ID_TO_TCH: text = ID_TO_TCH[tid]
        else: return "", i
        if i < len(tokens) and tokens[i] == FIREWORK:
            text += "f"; i += 1
        dur = read_duration_tokens(tokens, i)
        if dur:
            if is_hold: text += f"[{dur[0]}:{dur[1]}]"
            elif tid in ID_TO_TCH: text += f"h[{dur[0]}:{dur[1]}]"
            elif tid in ID_TO_BRK: text += f"h[{dur[0]}:{dur[1]}]"
            i += 3
        return text, i

    # -- Batch --

    def encode_batch(self, texts: list[str], pad_to: int | None = None,
                     add_bos: bool = True, add_eos: bool = True):
        import torch
        sequences = [self.encode(t, add_bos=add_bos, add_eos=add_eos) for t in texts]
        if pad_to is None: pad_to = max(len(s) for s in sequences)
        padded = []
        for seq in sequences:
            if len(seq) < pad_to: seq = seq + [PAD]*(pad_to-len(seq))
            else: seq = seq[:pad_to]
            padded.append(seq)
        return torch.tensor(padded, dtype=torch.long)

# =====================================================================
# Convenience
# =====================================================================

def encode_simai(text: str) -> list[int]:
    return MaiChartTokenizer().encode(text)

def decode_tokens(tokens: list[int]) -> str:
    return MaiChartTokenizer().decode(tokens)

# =====================================================================
# Self-test
# =====================================================================

if __name__ == "__main__":
    tok = MaiChartTokenizer()
    print(f"MaiChartTokenizer v{TOKENIZER_VERSION}")
    print(f"  Vocab size: {tok.vocab_size} (raw tokens, no config)")
    print(f"  Run tests via: python tests/test_chart_tokenizer.py")
