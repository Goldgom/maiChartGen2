"""
Config Vocab - single-token time-slot configuration encoding.

Encodes button + touch state at one beat position into a single token ID
(256+), respecting the 2-hand human constraint.

Touch panel layout (maimai DX):
  Center: C -> E ring -> B ring -> A/D outer ring (alternating)
  Zone indices 0-32 (see touch_expander.py for full layout)

Stage 1 limit: max 2 touch zones, non-adjacent only.

Token layout:
  ID 256         : rest (0 btn, 0 tch)
  ID 257..296    : 1 btn (40)
  ID 297..996    : 2 btn (700)
  ID 997..4911   : touch only (3915)
  ID 4912..161511: 1 btn + touch (156600)
  Total: 161,256 configs (18 bits)
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from Tokenizer.touch_expander import (
    NUM_ZONES, _ADJ, zones_adjacent, zone_name, zone_index,
)

# =====================================================================
# Constants
# =====================================================================

CONFIG_BASE = 256
NUM_BTN_POS = 8
MAX_TCH_ZONES = 2

BTN_PRESS = 0
BTN_HOLD_START = 1
BTN_HOLD_ONGOING = 2
BTN_SLIDE_START = 3
BTN_SLIDE_END = 4
BTN_NUM_STATES = 5
BTN_STATE_NAMES = ["press", "hold_start", "hold_ongoing", "slide_start", "slide_end"]

TCH_TOUCH = 0
TCH_HOLD_START = 1
TCH_HOLD_ONGOING = 2
TCH_NUM_STATES = 3
TCH_STATE_NAMES = ["touch", "touch_hold_start", "touch_hold_ongoing"]


# =====================================================================
# SlotConfig
# =====================================================================

@dataclass(frozen=True)
class SlotConfig:
    buttons: tuple = ()   # ((pos, state), ...) sorted
    touches: tuple = ()   # ((zone_idx, state), ...) sorted

    @property
    def is_rest(self) -> bool:
        return len(self.buttons) == 0 and len(self.touches) == 0

    @property
    def num_button_hands(self) -> int:
        return len(self.buttons)

    @property
    def num_touch_hands(self) -> int:
        return 1 if self.touches else 0


# =====================================================================
# Vocab builder
# =====================================================================

def build_config_vocab() -> tuple[dict[int, SlotConfig], dict[SlotConfig, int]]:
    id_to_config: dict[int, SlotConfig] = {}
    config_to_id: dict[SlotConfig, int] = {}
    next_id = CONFIG_BASE

    def add(sc: SlotConfig) -> int:
        nonlocal next_id
        if sc in config_to_id:
            return config_to_id[sc]
        # Filter: 2-zone touch must not be adjacent
        if len(sc.touches) == 2:
            if zones_adjacent(sc.touches[0][0], sc.touches[1][0]):
                return -1
        tid = next_id
        next_id += 1
        id_to_config[tid] = sc
        config_to_id[sc] = tid
        return tid

    # rest
    add(SlotConfig())

    # 1 btn
    for p in range(1, NUM_BTN_POS + 1):
        for s in range(BTN_NUM_STATES):
            add(SlotConfig(buttons=((p, s),)))

    # 2 btn
    for p1 in range(1, NUM_BTN_POS + 1):
        for p2 in range(p1 + 1, NUM_BTN_POS + 1):
            for s1 in range(BTN_NUM_STATES):
                for s2 in range(BTN_NUM_STATES):
                    add(SlotConfig(buttons=((p1, s1), (p2, s2))))

    # touch only
    for z in range(NUM_ZONES):
        for s in range(TCH_NUM_STATES):
            add(SlotConfig(touches=((z, s),)))
    for z1 in range(NUM_ZONES):
        for z2 in range(z1 + 1, NUM_ZONES):
            for s1 in range(TCH_NUM_STATES):
                for s2 in range(TCH_NUM_STATES):
                    add(SlotConfig(touches=((z1, s1), (z2, s2))))

    # 1 btn + touch
    for p in range(1, NUM_BTN_POS + 1):
        for bs in range(BTN_NUM_STATES):
            for z in range(NUM_ZONES):
                for ts in range(TCH_NUM_STATES):
                    add(SlotConfig(buttons=((p, bs),), touches=((z, ts),)))
            for z1 in range(NUM_ZONES):
                for z2 in range(z1 + 1, NUM_ZONES):
                    for ts1 in range(TCH_NUM_STATES):
                        for ts2 in range(TCH_NUM_STATES):
                            add(SlotConfig(buttons=((p, bs),), touches=((z1, ts1), (z2, ts2))))

    return id_to_config, config_to_id


ID_TO_CONFIG, CONFIG_TO_ID = build_config_vocab()
CONFIG_VOCAB_SIZE = len(ID_TO_CONFIG)


def config_to_base_tokens(sc: SlotConfig) -> list[int]:
    """Convert SlotConfig to base token IDs."""
    from Tokenizer.MaiChartTokenizer import (
        TAP_TO_ID, HLD_TO_ID, SLD_TO_ID, SIM_BEG, SIM_END, SIM_COUNT_2,
        TCH_TO_ID, RST,
    )
    tokens = []
    btn_tokens = []
    for pos, state in sc.buttons:
        if state in (BTN_PRESS, BTN_HOLD_ONGOING):
            tid = HLD_TO_ID[pos] if state == BTN_HOLD_ONGOING else TAP_TO_ID[pos]
        elif state == BTN_HOLD_START:
            tid = HLD_TO_ID[pos]
        else:
            tid = SLD_TO_ID[pos]
        btn_tokens.append(tid)
    if len(btn_tokens) == 2:
        tokens.extend([SIM_BEG, SIM_COUNT_2, btn_tokens[0], btn_tokens[1], SIM_END])
    elif len(btn_tokens) == 1:
        tokens.append(btn_tokens[0])

    tch_tokens = []
    for zi, state in sc.touches:
        zn = zone_name(zi)
        tid = TCH_TO_ID.get(zn)
        if tid is not None:
            tch_tokens.append(tid)
    if len(tch_tokens) == 2:
        tokens.extend([SIM_BEG, SIM_COUNT_2, tch_tokens[0], tch_tokens[1], SIM_END])
    elif len(tch_tokens) == 1:
        tokens.append(tch_tokens[0])

    if not tokens:
        tokens.append(RST)
    return tokens


def config_name(sc: SlotConfig) -> str:
    parts = []
    for pos, state in sc.buttons:
        parts.append(f"btn{pos}_{BTN_STATE_NAMES[state]}")
    for zi, state in sc.touches:
        parts.append(f"tch{zone_name(zi)}_{TCH_STATE_NAMES[state]}")
    return "_".join(parts) if parts else "rest"


# =====================================================================
# Self-test
# =====================================================================

if __name__ == "__main__":
    print(f"Config Vocab")
    print(f"  Zones: {NUM_ZONES}")
    print(f"  Configs: {CONFIG_VOCAB_SIZE}")
    print(f"  Vocab range: {CONFIG_BASE}..{CONFIG_BASE + CONFIG_VOCAB_SIZE - 1}")
    print(f"  Bits: {(CONFIG_BASE + CONFIG_VOCAB_SIZE - 1).bit_length()}")
    print()

    rest = sum(1 for sc in ID_TO_CONFIG.values() if sc.is_rest)
    b1 = sum(1 for sc in ID_TO_CONFIG.values() if sc.num_button_hands == 1 and sc.num_touch_hands == 0)
    b2 = sum(1 for sc in ID_TO_CONFIG.values() if sc.num_button_hands == 2)
    t1 = sum(1 for sc in ID_TO_CONFIG.values() if sc.num_button_hands == 0 and sc.num_touch_hands == 1)
    mx = sum(1 for sc in ID_TO_CONFIG.values() if sc.num_button_hands == 1 and sc.num_touch_hands == 1)
    print(f"  rest: {rest}")
    print(f"  1 btn: {b1}")
    print(f"  2 btn: {b2}")
    print(f"  touch only: {t1}")
    print(f"  mixed: {mx}")
    print(f"  total: {CONFIG_VOCAB_SIZE}")

    print("\nFirst 5 configs:")
    for tid in range(CONFIG_BASE, CONFIG_BASE + 5):
        sc = ID_TO_CONFIG[tid]
        print(f"  {tid}: {config_name(sc)} -> {config_to_base_tokens(sc)}")
