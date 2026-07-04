"""
Phase 2a: 标签提取 + 多 Stage Token 生成

为 5 个 Stage 生成各自专用的 token 格式:

  Stage 1 (maiG):     简化 Token — Config + DUR + SLD compact
                      剥离 break/firework, touch 压缩为 Config
  Stage 2 (touchG):   每个时间槽的 Touch Zone 标签 [T, 33]
                      0=无touch, 1=touch, 2=hold_start
  Stage 2.5 (slideG): 完整 Slide Path Token — [BOS, waypoint, connector, ..., EOS]
                      输入: start_pos, end_pos, duration; 输出: 自回归路径序列
  Stage 3 (breakG):   每个时间槽的 Break 标签 [T, 8]
                      0=无press, 1=tap, 2=break, press_mask
  Stage 4 (spikeG):   每个时间槽的 Firework 标签 [T, 33]
                      0/1 per zone, touch_mask

生成缓存:
  cache/_labels/{song_id}.pt     全量原始标注
  cache/stage1/{song_id}.pt      Stage 1 训练数据
  cache/slide/{song_id}_*.pt     Stage 2.5 训练数据

用法:
  python scripts/preprocess_labels.py
  python scripts/preprocess_labels.py --limit 10 --num-workers 4
  python scripts/preprocess_labels.py --skip-existing
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("preprocess_labels")

_RE_LEVEL = re.compile(r"lv\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
_RE_GENRE = re.compile(r"genre\s*[:=]\s*(\d+)", re.IGNORECASE)
_RE_BPM  = re.compile(r"\((\d+(?:\.\d+)?)\)")
_RE_BEAT_DIV = re.compile(r"\{(\d+)\}")
_RE_DURATION = re.compile(r"\[(\d+):(\d+)\]")


def parse_meta(text: str) -> dict:
    bpm, level, genre = 150.0, 10.0, 0
    for line in text.split("\n")[:20]:
        if m := _RE_BPM.search(line):   bpm   = float(m.group(1))
        if m := _RE_LEVEL.search(line): level = float(m.group(1))
        if m := _RE_GENRE.search(line): genre = int(m.group(1))
    return {"bpm": bpm, "level": level, "genre": genre}


def _split_simai_slots(text: str) -> list[str]:
    slots: list[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(text):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            slots.append(text[start:i].strip())
            start = i + 1
    if start < len(text):
        slots.append(text[start:].strip())
    return slots


def _split_prefix_directives(slot: str) -> tuple[str, int | None, str]:
    """Return leading BPM text, last beat-div in this slot, and note body."""
    prefix: list[str] = []
    beat_div: int | None = None
    i = 0
    while i < len(slot):
        m_bpm = _RE_BPM.match(slot, i)
        if m_bpm:
            prefix.append(m_bpm.group(0))
            i = m_bpm.end()
            continue
        m_div = _RE_BEAT_DIV.match(slot, i)
        if m_div:
            beat_div = int(m_div.group(1))
            i = m_div.end()
            continue
        break
    return "".join(prefix), beat_div, slot[i:].strip()


def _merge_slot(existing: str, incoming: str) -> str:
    if not incoming:
        return existing
    if not existing:
        return incoming
    if existing == "E":
        return existing
    if incoming == "E":
        return incoming
    return f"{existing}/{incoming}"


def _scale_duration_text(note_text: str, ratio: float) -> str:
    """Scale h/slide durations when changing subdivision."""
    def repl(m: re.Match) -> str:
        num = max(1, round(int(m.group(1)) * ratio))
        den = int(m.group(2))
        return f"[{num}:{den}]"

    return _RE_DURATION.sub(repl, note_text)


def normalize_simai_text_to_maxsubdiv(text: str, maxsubdiv: int = 64) -> str:
    """
    Normalize all {n} regions into a single {maxsubdiv} grid.

    The output is intentionally text-level and inspectable: source slots are
    snapped to the nearest target slot, empty target slots are emitted as empty
    comma fields, and note durations are scaled by maxsubdiv / n.
    """
    slots = _split_simai_slots(text.strip())
    if not slots:
        return f"{{{maxsubdiv}}}E"

    current_div = 4
    pos = 0.0
    out_slots: list[str] = []
    bpm_prefix = ""
    saw_end = False

    for slot in slots:
        prefix, new_div, body = _split_prefix_directives(slot)
        if prefix and not bpm_prefix:
            bpm_prefix = prefix
        if new_div is not None:
            current_div = max(1, new_div)

        step = maxsubdiv / max(1, current_div)
        target_idx = max(0, round(pos))
        while len(out_slots) <= target_idx:
            out_slots.append("")

        if body == "E":
            out_slots[target_idx] = "E"
            saw_end = True
            break

        if body:
            scaled = _scale_duration_text(body, step)
            out_slots[target_idx] = _merge_slot(out_slots[target_idx], scaled)

        pos += step

    if not saw_end:
        end_idx = max(0, round(pos))
        while len(out_slots) <= end_idx:
            out_slots.append("")
        out_slots[end_idx] = _merge_slot(out_slots[end_idx], "E")

    return f"{bpm_prefix}{{{maxsubdiv}}}" + ",".join(out_slots)


def _simplify_stage1_slot(slot: str) -> str:
    """Human-readable stage1 text: no break/firework/durations; slide keeps head."""
    return _stage_slot_text(
        slot,
        include_slide_detail=False,
        include_button_hold_duration=False,
        include_touch_hold_duration=False,
        include_full_touch_pattern=False,
        include_break=False,
        include_firework=False,
    )


def _is_touch_piece(piece: str) -> bool:
    return bool(re.match(r"^([ABDE][1-8]|C\d*)", piece))


def _is_slide_piece(piece: str) -> bool:
    return bool(re.search(r"(pp|qq|[*\-<>^vVpqszw])", piece) and re.search(r"\d", piece))


def _strip_break_text(piece: str) -> str:
    return re.sub(r"b(?=h|/|,|\[|$)", "", piece)


def _strip_firework_text(piece: str) -> str:
    return re.sub(r"f(?=,|/|$)", "", piece)


def _slide_head(piece: str) -> str:
    m = re.search(r"([1-8])", piece)
    return m.group(1) if m else ""


def _touch_head(piece: str) -> str:
    m = re.match(r"([ABDE][1-8]|C\d*)", piece)
    if not m:
        return ""
    base = m.group(1)
    if base.startswith("C"):
        base = "C"
    if "h" in piece:
        base += "h"
    return base


def _button_head(piece: str) -> str:
    m = re.search(r"([1-8])", piece)
    if not m:
        return ""
    base = m.group(1)
    if "h" in piece:
        base += "h"
    return base


def _stage_piece_text(
    piece: str,
    *,
    include_slide_detail: bool,
    include_button_hold_duration: bool,
    include_touch_hold_duration: bool,
    include_break: bool,
    include_firework: bool,
) -> str:
    if not piece:
        return ""

    out = piece
    if not include_break:
        out = _strip_break_text(out)
    if not include_firework:
        out = _strip_firework_text(out)

    is_touch = _is_touch_piece(out)
    is_slide = (not is_touch) and _is_slide_piece(out)
    is_hold = "h" in out and "[" in out

    if is_slide and not include_slide_detail:
        return _slide_head(out)

    if is_touch:
        if not include_touch_hold_duration:
            out = _RE_DURATION.sub("", out)
        return out

    if is_hold and not include_button_hold_duration:
        out = _RE_DURATION.sub("", out)
    return out


def _stage_slot_text(
    slot: str,
    *,
    include_slide_detail: bool,
    include_button_hold_duration: bool,
    include_touch_hold_duration: bool,
    include_full_touch_pattern: bool,
    include_break: bool,
    include_firework: bool,
) -> str:
    if not slot or slot == "E":
        return slot

    parts = re.split(r"([/`])", slot)
    output: list[str] = []
    pending_sep = ""
    touch_seen = False

    for part in parts:
        if part in {"/", "`"}:
            pending_sep = part
            continue
        if not part:
            continue

        simplified = _stage_piece_text(
            part,
            include_slide_detail=include_slide_detail,
            include_button_hold_duration=include_button_hold_duration,
            include_touch_hold_duration=include_touch_hold_duration,
            include_break=include_break,
            include_firework=include_firework,
        )
        if not simplified:
            continue

        if _is_touch_piece(simplified):
            if touch_seen and not include_full_touch_pattern:
                continue
            touch_seen = True

        if output and pending_sep:
            output.append(pending_sep)
        output.append(simplified)
        pending_sep = "/"

    return "".join(output)


def _make_stage_text(prefix: str, slots: list[str], **kwargs) -> str:
    return prefix + ",".join(_stage_slot_text(s, **kwargs) for s in slots)


def make_stage_simai_texts(normalized_text: str) -> dict[str, str]:
    """Build inspectable simai text variants for staged training caches."""
    prefix = ""
    m = re.match(r"^((?:\(\d+(?:\.\d+)?\))?)\{(\d+)\}", normalized_text)
    body = normalized_text
    if m:
        prefix = m.group(0)
        body = normalized_text[m.end():]
    slots = _split_simai_slots(body)
    stage1_text = prefix + ",".join(_simplify_stage1_slot(s) for s in slots)
    return {
        "normalized": normalized_text,
        "stage1": stage1_text,
        "stage2_star": _make_stage_text(
            prefix, slots,
            include_slide_detail=True,
            include_button_hold_duration=False,
            include_touch_hold_duration=False,
            include_full_touch_pattern=False,
            include_break=False,
            include_firework=False,
        ),
        "stage3_hold": _make_stage_text(
            prefix, slots,
            include_slide_detail=True,
            include_button_hold_duration=True,
            include_touch_hold_duration=False,
            include_full_touch_pattern=False,
            include_break=False,
            include_firework=False,
        ),
        "stage4_touch_hold": _make_stage_text(
            prefix, slots,
            include_slide_detail=True,
            include_button_hold_duration=True,
            include_touch_hold_duration=True,
            include_full_touch_pattern=False,
            include_break=False,
            include_firework=False,
        ),
        "stage5_touch": _make_stage_text(
            prefix, slots,
            include_slide_detail=True,
            include_button_hold_duration=True,
            include_touch_hold_duration=True,
            include_full_touch_pattern=True,
            include_break=False,
            include_firework=False,
        ),
        "stage6_break_note": _make_stage_text(
            prefix, slots,
            include_slide_detail=True,
            include_button_hold_duration=True,
            include_touch_hold_duration=True,
            include_full_touch_pattern=True,
            include_break=True,
            include_firework=False,
        ),
        "stage7_firework_note": normalized_text,
    }


# ═══════════════════════════════════════════════════════════════════════
# Token 常量
# ═══════════════════════════════════════════════════════════════════════

from Tokenizer.MaiChartTokenizer import (
    PAD, BOS, EOS, RST, DUR, FIREWORK,
    TAP_TO_ID, HLD_TO_ID, TCH_TO_ID,
    SLD_BASE, SLD_TO_ID,
    SLD_BEG_BASE, SLD_BEG_END, ID_TO_SLD_BEG,
    SLD_END_POS_BASE, SLD_END_POS_END, ID_TO_SLD_END_POS,
    SLD_TYPE_BASE, SLD_CHAR_TO_TYPE,
    SIM_BEG, SIM_END, SIM_COUNT_2,
    ID_TO_DUR_NUM, ID_TO_DUR_DEN,
    encode_duration_tokens, encode_slide_compact, encode_wifi_compact,
    _note_to_slot_config,
)
from Tokenizer.config_vocab import CONFIG_TO_ID as CFG_SLOT_TO_ID
from Tokenizer.touch_expander import zone_index as _zone_index


# ═══════════════════════════════════════════════════════════════════════
# Subdivision Normalization — 将所有 {n} 归一化到 maxsubdiv
# ═══════════════════════════════════════════════════════════════════════

def normalize_notes_to_grid(notes: list, maxsubdiv: int = 64) -> list:
    """
    将谱面统一到 maxsubdiv 网格。每个 slot = 1/maxsubdiv 拍。

    规则:
      - {n} 替换为 {maxsubdiv}，每个原始 slot 扩展为 round(maxsubdiv/n) 个新 slot
      - Hold/Slide 持续时间乘以 ratio
      - 非整数倍时取最近整数值

    Returns:
        expanded_notes: 每个元素对应一个 maxsubdiv 格点
    """
    if not notes:
        return []

    expanded: list = []
    for note in notes:
        if note.is_end:
            expanded.append(note)
            continue

        beat_div = max(1, note.beat_div)
        ratio = maxsubdiv / beat_div
        slots = max(1, round(ratio))

        if note.is_rest:
            # rest 直接复制 slots 次
            for _ in range(slots):
                r = _make_grid_note(maxsubdiv, is_rest=True)
                expanded.append(r)
            continue

        # ── 计算持续时间（slot 数）──
        duration_slots = slots  # 默认 = 1 个原始格 → slots 个新格
        if note.is_hold and note.hold_duration:
            # hold_duration = (num, den)，单位是"拍"
            beats = note.hold_duration[0] / max(1, note.hold_duration[1])
            duration_slots = max(1, round(beats * maxsubdiv))
        elif note.is_slide:
            if note.hold_duration:
                beats = note.hold_duration[0] / max(1, note.hold_duration[1])
                duration_slots = max(1, round(beats * maxsubdiv))
            else:
                duration_slots = slots
        elif note.is_touch_hold and note.hold_duration:
            beats = note.hold_duration[0] / max(1, note.hold_duration[1])
            duration_slots = max(1, round(beats * maxsubdiv))

        # ── 第一个 slot: 原始 note ──
        first = _copy_note(note, maxsubdiv)
        expanded.append(first)

        # ── 后续 slot: hold/slide 持续部分 ──
        for s in range(1, duration_slots):
            if note.is_hold or note.is_slide or note.is_touch_hold:
                ongoing = _make_grid_note(maxsubdiv, is_rest=False)
                ongoing.positions = list(note.positions) if note.positions else []
                ongoing.touch_regions = list(note.touch_regions) if note.touch_regions else []
                if note.is_hold:
                    ongoing.is_hold = True
                elif note.is_slide:
                    ongoing.is_slide = True
                elif note.is_touch_hold:
                    ongoing.is_touch = True
                expanded.append(ongoing)
            else:
                r = _make_grid_note(maxsubdiv, is_rest=True)
                expanded.append(r)

        # ── 如果 duration_slots < slots，补 rest ──
        for _ in range(max(0, slots - duration_slots)):
            r = _make_grid_note(maxsubdiv, is_rest=True)
            expanded.append(r)

    return expanded


def normalize_notes_to_stage_grid(notes: list, maxsubdiv: int = 64) -> list:
    """Map source slots to maxsubdiv; hold/slide/touch-hold remain head-only."""
    if not notes:
        return []

    expanded: list = []
    for note in notes:
        if note.is_end:
            expanded.append(note)
            continue

        beat_div = max(1, note.beat_div)
        slots = max(1, round(maxsubdiv / beat_div))
        if note.is_rest:
            for _ in range(slots):
                expanded.append(_make_grid_note(maxsubdiv, is_rest=True))
            continue

        expanded.append(_copy_note(note, maxsubdiv))
        for _ in range(1, slots):
            expanded.append(_make_grid_note(maxsubdiv, is_rest=True))

    return expanded


def _make_grid_note(beat_div: int, is_rest: bool = False):
    """创建一个简化的格点 note 对象。"""
    from Tokenizer.MaiChartTokenizer import _Note
    n = _Note(beat_div=beat_div, raw="")
    n.is_rest = is_rest
    n.is_end = False
    n.is_break = False
    n.is_slide = False
    n.is_hold = False
    n.is_touch = False
    n.is_firework = False
    n.is_simultaneous = False
    n.positions = []
    n.touch_regions = []
    n.hold_duration = None
    return n


def _copy_note(src, beat_div: int):
    """深拷贝一个 note，替换 beat_div。"""
    n = _make_grid_note(beat_div, is_rest=src.is_rest)
    n.is_end = src.is_end
    n.is_break = src.is_break
    n.is_slide = src.is_slide
    n.is_hold = src.is_hold
    n.is_touch = src.is_touch
    n.is_firework = src.is_firework
    n.is_simultaneous = src.is_simultaneous
    n.is_touch_hold = getattr(src, 'is_touch_hold', False)
    n.is_touch_slide = getattr(src, 'is_touch_slide', False)
    n.positions = list(src.positions) if src.positions else []
    n.touch_regions = list(src.touch_regions) if src.touch_regions else []
    n.hold_duration = src.hold_duration  # 保持原始 duration 引用（stage2-4 需要）
    n.slide_path = list(getattr(src, 'slide_path', []) or [])
    n.slide_types = list(getattr(src, 'slide_types', []) or [])
    n.raw = src.raw
    return n


# ═══════════════════════════════════════════════════════════════════════
# Stage 1 Token — 每个 slot 一个 token（模糊配置，无时长/路径）
# ═══════════════════════════════════════════════════════════════════════

def _note_to_stage1_token(note) -> int:
    """
    Stage 1 专用：将一个格点 note 转为单个 config token ID。

    规则:
      - rest        → 0 (rest config)
      - tap         → config(tap, pos)
      - hold 起始   → config(hld_start, pos)  — 只保留开头
      - hold 持续   → config(hld_ongoing, pos)
      - slide 起始  → config(sld_start, pos)  — 只保留开头
      - slide 持续  → config(sld_ongoing, pos)
      - touch       → config(tch, zone)
      - touch hold  → config(tch_hld, zone)   — 只保留开头
      - break       → config(tap, pos)        — 剥离绝赞
      - firework    → config(tch, zone)       — 剥离烟花
    """
    from Tokenizer.config_vocab import (
        SlotConfig, BTN_PRESS, BTN_HOLD_START, BTN_HOLD_ONGOING,
        BTN_SLIDE_START, BTN_SLIDE_END,
        TCH_TOUCH, TCH_HOLD_START,
        CONFIG_TO_ID as CFG_ID, CONFIG_BASE,
    )
    from Tokenizer.MaiChartTokenizer import _zone_index

    if note.is_rest:
        return CFG_ID.get(SlotConfig(), CONFIG_BASE)

    buttons = []
    touches = []

    # ── 按键处理 ──
    if not note.is_touch and note.positions:
        for pos in note.positions:
            if not (1 <= pos <= 8):
                continue
            if note.is_slide:
                state = BTN_SLIDE_START  # slide 只保留开头位置
            elif note.is_hold:
                state = BTN_HOLD_START  # hold 只保留开头
            else:
                state = BTN_PRESS  # tap（含被剥离的 break）
            buttons.append((pos, state))

    # ── 触控处理 ──
    if note.is_touch and note.touch_regions:
        for region in note.touch_regions:
            try:
                zi = _zone_index(region)
            except Exception:
                continue
            if getattr(note, 'is_touch_hold', False):
                state = TCH_HOLD_START  # touch hold 只保留开头
            else:
                state = TCH_TOUCH  # touch（含被剥离的 firework）
            touches.append((zi, state))

    if not buttons and not touches:
        return CFG_ID.get(SlotConfig(), CONFIG_BASE)

    # 最多 2 按键 + 2 触控
    buttons = tuple(sorted(buttons)[:2])
    touches = tuple(sorted(touches)[:2])

    sc = SlotConfig(buttons=buttons, touches=touches)
    return CFG_ID.get(sc, CONFIG_BASE)


# ═══════════════════════════════════════════════════════════════════════
# Stage 2.5 Token — Slide 完整路径
# ═══════════════════════════════════════════════════════════════════════

def make_slide_path_tokens(note) -> list[int] | None:
    """
    完整 slide path: [BOS, waypoint, connector, waypoint, ..., EOS]

    waypoint  token: SLD_TO_ID[pos]     (42-49)
    connector token: SLD_CHAR_TO_TYPE[c] (116-129)

    示例: "1-3>5[4:1]" → [BOS, 42, 116, 44, 117, 46, EOS]
    """
    if not note.is_slide:
        return None
    path = note.slide_path or note.positions
    types = note.slide_types
    if len(path) < 2:
        return None

    tokens: list[int] = [BOS]
    for idx, pos in enumerate(path):
        if not (1 <= pos <= 8):
            continue
        tokens.append(SLD_TO_ID.get(pos, SLD_BASE))
        if idx < len(path) - 1:
            conn = types[idx] if idx < len(types) else "-"
            tokens.append(SLD_CHAR_TO_TYPE.get(conn, SLD_TYPE_BASE))
    tokens.append(EOS)
    return tokens


# ═══════════════════════════════════════════════════════════════════════
# 标签提取 — Stage 2/3/4
# ═══════════════════════════════════════════════════════════════════════

def extract_labels_for_note(note) -> dict:
    touch_row = [0] * 33
    break_row = [0] * 8
    press_row = [False] * 8
    spike_row = [0] * 33
    tmask_row = [False] * 33

    # Stage 2: Touch zone + state
    if note.is_touch:
        for region in note.touch_regions:
            try:
                zi = _zone_index(region)
                if 0 <= zi < 33:
                    tmask_row[zi] = True
                    touch_row[zi] = 2 if note.is_touch_hold else 1
            except Exception:
                pass

    # Stage 3: Break per position
    for pos in note.positions:
        if 1 <= pos <= 8:
            press_row[pos - 1] = True
            break_row[pos - 1] = 2 if note.is_break else 1

    # Stage 4: Firework per zone
    if note.is_firework and note.is_touch:
        for region in note.touch_regions:
            try:
                zi = _zone_index(region)
                if 0 <= zi < 33:
                    spike_row[zi] = 1
            except Exception:
                pass

    return {"touch": touch_row, "break": break_row, "press": press_row,
            "spike": spike_row, "tmask": tmask_row}


def _duration_labels(dur: tuple[int, int] | None) -> tuple[int, int]:
    from models.hold_stage import duration_to_labels
    return duration_to_labels(dur)


def _grid_index_for_note(note, cursor_slots: int, maxsubdiv: int) -> int:
    return cursor_slots


def build_stage_detail_targets(notes: list, maxsubdiv: int = 64) -> dict[str, Any]:
    """Build targets for stage2-7 from original parsed notes, aligned by head slot."""
    from Tokenizer.slide_star_vocab import encode_slide_star, from_mai_note
    from Tokenizer.MaiChartTokenizer import _zone_index

    hold_events: list[dict[str, Any]] = []
    touch_hold_events: list[dict[str, Any]] = []
    star_events: list[dict[str, Any]] = []
    touch_events: list[dict[str, Any]] = []
    note_break_events: list[dict[str, Any]] = []
    note_firework_events: list[dict[str, Any]] = []

    cursor = 0
    note_index = 0
    for note in notes:
        if note.is_end:
            break

        beat_div = max(1, note.beat_div)
        n_slots = max(1, round(maxsubdiv / beat_div))
        slot = _grid_index_for_note(note, cursor, maxsubdiv)

        if not note.is_rest:
            if note.is_slide:
                path = from_mai_note(note)
                if path is not None:
                    encoded = encode_slide_star(path)
                    star_events.append({
                        "slot": slot,
                        "note_index": note_index,
                        "start_pos": int(path.start_pos),
                        "target_path": torch.tensor(encoded, dtype=torch.long),
                    })

            if note.is_hold and not note.is_touch_hold and not note.is_slide:
                num_idx, den_idx = _duration_labels(note.hold_duration)
                hold_events.append({
                    "slot": slot,
                    "note_index": note_index,
                    "positions": torch.tensor(note.positions or [], dtype=torch.long),
                    "dur_num_target": int(num_idx),
                    "dur_den_target": int(den_idx),
                })

            if getattr(note, "is_touch_hold", False):
                num_idx, den_idx = _duration_labels(note.hold_duration)
                zones = []
                for region in note.touch_regions:
                    try:
                        zones.append(_zone_index(region))
                    except Exception:
                        pass
                touch_hold_events.append({
                    "slot": slot,
                    "note_index": note_index,
                    "zones": torch.tensor(zones, dtype=torch.long),
                    "dur_num_target": int(num_idx),
                    "dur_den_target": int(den_idx),
                })

            if note.is_touch:
                zones = []
                for region in note.touch_regions:
                    try:
                        zones.append(_zone_index(region))
                    except Exception:
                        pass
                touch_events.append({
                    "slot": slot,
                    "note_index": note_index,
                    "zones": torch.tensor(zones, dtype=torch.long),
                    "num_touches": int(len(zones)),
                    "is_touch_hold": bool(getattr(note, "is_touch_hold", False)),
                })
                note_firework_events.append({
                    "slot": slot,
                    "note_index": note_index,
                    "zones": torch.tensor(zones, dtype=torch.long),
                    "target": int(bool(note.is_firework)),
                })

            if note.positions:
                for pos in note.positions:
                    if 1 <= pos <= 8:
                        note_break_events.append({
                            "slot": slot,
                            "note_index": note_index,
                            "position": int(pos),
                            "target": int(bool(note.is_break)),
                        })

            note_index += 1

        cursor += n_slots

    return {
        "stage2_star_events": star_events,
        "stage3_hold_events": hold_events,
        "stage4_touch_hold_events": touch_hold_events,
        "stage5_touch_events": touch_events,
        "stage6_break_note_events": note_break_events,
        "stage7_firework_note_events": note_firework_events,
    }


# ═══════════════════════════════════════════════════════════════════════
# 单曲处理
# ═══════════════════════════════════════════════════════════════════════

def process_one(folder: Path, audio_dir: Path, max_tokens: int, maxsubdiv: int = 64) -> dict[str, Any]:
    """
    处理一首歌的所有难度 Chart。

    Stage 1: 归一化到 maxsubdiv 网格，每格一个 config token（简化版）
    Stage 2-4: 从原始 notes 生成详细标签
    """
    name = folder.name
    maidata_path = folder / "maidata.txt"
    audio_path = audio_dir / f"{name}.pt"

    if not maidata_path.exists():
        return {"folder": str(folder), "error": "missing_maidata"}
    if not audio_path.exists():
        return {"folder": str(folder), "error": "missing_audio_cache"}

    text = maidata_path.read_text(encoding="utf-8")
    meta = parse_meta(text)
    audio = torch.load(audio_path, map_location="cpu", weights_only=True)

    from Tokenizer.MaiChartTokenizer import _parse_simai_text

    # ── 解析所有难度 ──
    from mai_parser.parser import parse_maidata as parse_full
    try:
        song = parse_full(text, song_id=name)
    except Exception:
        return {"folder": str(folder), "error": "parse_failed"}

    if not song.charts:
        return {"folder": str(folder), "error": "no_charts"}

    all_charts: list[dict] = []
    for idx, chart in sorted(song.charts.items()):
        raw_notes = _extract_inote_block(text, idx)
        if not raw_notes:
            continue
        try:
            normalized_simai = normalize_simai_text_to_maxsubdiv(raw_notes, maxsubdiv)
            stage_simai = make_stage_simai_texts(normalized_simai)
            notes = _parse_simai_text(normalized_simai)
        except Exception:
            continue

        chart_id = f"{name}_lv{idx}"

        # ═════════════════════════════════════════════════
        # Stage 1: 归一化到 maxsubdiv 网格 → 每格一个 token
        # ═════════════════════════════════════════════════
        grid_notes = normalize_notes_to_stage_grid(notes, maxsubdiv)
        s1_tokens: list[int] = [BOS]
        for gn in grid_notes:
            if gn.is_end:
                break
            s1_tokens.append(_note_to_stage1_token(gn))
        s1_tokens.append(EOS)
        s1_tokens = s1_tokens[:max_tokens]
        T = len(s1_tokens)

        # ═════════════════════════════════════════════════
        # Stage 2-4: 从原始 notes 生成标签 + slide
        # ═════════════════════════════════════════════════
        touch_targets: list[list[int]] = []
        break_targets: list[list[int]] = []
        press_masks:   list[list[bool]] = []
        spike_targets: list[list[int]] = []
        touch_masks:   list[list[bool]] = []
        slide_samples: list[dict] = []
        detail_targets = build_stage_detail_targets(notes, maxsubdiv)

        for note in notes:
            if note.is_end:
                continue
            labels = extract_labels_for_note(note)
            # 每个原始 note 在网格中占用的 slot 数
            beat_div = max(1, note.beat_div)
            ratio = maxsubdiv / beat_div
            n_slots = max(1, round(ratio))

            for slot_offset in range(n_slots):
                if slot_offset == 0 and not note.is_rest:
                    touch_targets.append(labels["touch"])
                    break_targets.append(labels["break"])
                    press_masks.append(labels["press"])
                    spike_targets.append(labels["spike"])
                    touch_masks.append(labels["tmask"])
                else:
                    touch_targets.append([0]*33)
                    break_targets.append([0]*8)
                    press_masks.append([False]*8)
                    spike_targets.append([0]*33)
                    touch_masks.append([False]*33)

            # Slide path (Stage 2.5) — 从原始 note 生成
            path_tok = make_slide_path_tokens(note)
            if path_tok and len(path_tok) > 2:
                slide_samples.append({"path_tokens": torch.tensor(path_tok, dtype=torch.long)})

        # 补齐到与 stage1 tokens 对齐
        while len(touch_targets) < T:
            touch_targets.append([0]*33)
            break_targets.append([0]*8)
            press_masks.append([False]*8)
            spike_targets.append([0]*33)
            touch_masks.append([False]*33)

        # 截断
        touch_targets = touch_targets[:T]
        break_targets = break_targets[:T]
        press_masks = press_masks[:T]
        spike_targets = spike_targets[:T]
        touch_masks = touch_masks[:T]

        # Slide compact 从网格中提取
        slide_compact = _extract_compact_slides_from_grid(grid_notes, maxsubdiv)

        all_charts.append({
            "chart_id": chart_id,
            "song_id": name,
            "level": chart.level_value,
            "difficulty": idx,
            "simai": stage_simai,
            "labels": {
                "stage1_tokens": torch.tensor(s1_tokens, dtype=torch.long),
                "touch_targets": torch.tensor(touch_targets, dtype=torch.long),
                "break_targets": torch.tensor(break_targets, dtype=torch.long),
                "press_mask":    torch.tensor(press_masks, dtype=torch.bool),
                "spike_targets": torch.tensor(spike_targets, dtype=torch.long),
                "touch_mask":    torch.tensor(touch_masks, dtype=torch.bool),
                "slide_samples": slide_samples,
                "slide_compact": slide_compact,
                **detail_targets,
            },
        })

    return {
        "folder": str(folder),
        "audio": audio,
        "meta": meta,
        "charts": all_charts,
    }


def _extract_compact_slides_from_grid(grid_notes: list, maxsubdiv: int) -> list[dict]:
    """从归一化网格中提取 slide compact 样本。"""
    samples = []
    i = 0
    while i < len(grid_notes):
        note = grid_notes[i]
        if note.is_slide and note.positions and not getattr(note, '_slide_consumed', False):
            start_pos = note.positions[0]
            if not (1 <= start_pos <= 8):
                i += 1
                continue

            # 向后查找 slide 结束
            end_pos = start_pos
            j = i + 1
            while j < len(grid_notes):
                nj = grid_notes[j]
                if nj.is_rest or not nj.is_slide:
                    break
                if nj.positions:
                    end_pos = nj.positions[0] if nj.positions else end_pos
                nj._slide_consumed = True
                j += 1

            # 计算 slide 持续的 slot 数
            dur_slots = j - i
            dur_beats = dur_slots / maxsubdiv
            num = max(1, round(dur_beats * maxsubdiv))

            mid = SLD_BASE + start_pos - 1
            end_sld = SLD_BASE + end_pos - 1
            samples.append({
                "target_path": torch.tensor([BOS, mid, end_sld, EOS], dtype=torch.long),
                "start_pos": torch.tensor([start_pos], dtype=torch.long),
                "end_pos":   torch.tensor([end_pos], dtype=torch.long),
                "duration":  torch.tensor([[float(num), float(maxsubdiv)]], dtype=torch.float32),
            })
            i = j
        else:
            i += 1
    return samples


def _extract_inote_block(text: str, idx: int) -> str:
    """从 maidata.txt 中提取 &inote_{idx}= 对应的原始 note 文本。"""
    import re
    pattern = rf"&inote_{idx}="
    m = re.search(pattern, text)
    if not m:
        return ""
    start = m.end()
    # 找到下一个 &inote_ 或文件末尾
    next_m = re.search(r'&inote_\d+=', text[start:])
    end = start + next_m.start() if next_m else len(text)
    # 跳过第一行（=后面的内容，可能是空）
    return text[start:end].strip()


# ═══════════════════════════════════════════════════════════════════════
# 保存
# ═══════════════════════════════════════════════════════════════════════

def save_all(result: dict, cache_root: Path) -> None:
    name   = Path(result["folder"]).name
    audio  = result["audio"]
    meta   = result["meta"]
    charts = result.get("charts", [])

    if not charts:
        return

    (cache_root / "_labels").mkdir(parents=True, exist_ok=True)
    (cache_root / "stage1").mkdir(parents=True, exist_ok=True)
    (cache_root / "slide").mkdir(parents=True, exist_ok=True)
    (cache_root / "stage_text").mkdir(parents=True, exist_ok=True)
    (cache_root / "stage2_star").mkdir(parents=True, exist_ok=True)

    for chart_data in charts:
        chart_id = chart_data["chart_id"]
        labels = chart_data["labels"]
        simai = chart_data.get("simai", {})

        # ── _labels 全量标注 ──
        _atomic_save({
            "simai": simai,
            "stage1_tokens": labels["stage1_tokens"],
            "touch_targets": labels["touch_targets"],
            "break_targets": labels["break_targets"],
            "press_mask":    labels["press_mask"],
            "spike_targets": labels["spike_targets"],
            "touch_mask":    labels["touch_mask"],
            "slide_samples": labels["slide_samples"],
            "slide_compact": labels["slide_compact"],
            "stage2_star_events": labels["stage2_star_events"],
            "stage3_hold_events": labels["stage3_hold_events"],
            "stage4_touch_hold_events": labels["stage4_touch_hold_events"],
            "stage5_touch_events": labels["stage5_touch_events"],
            "stage6_break_note_events": labels["stage6_break_note_events"],
            "stage7_firework_note_events": labels["stage7_firework_note_events"],
        }, cache_root / "_labels" / f"{chart_id}.pt")

        text_path = cache_root / "stage_text" / f"{chart_id}.json"
        text_path.write_text(json.dumps(simai, ensure_ascii=False, indent=2), encoding="utf-8")

        # ── stage1 训练数据 ──
        try:
            from models.stage1 import compute_relative_distances
            dist = compute_relative_distances(labels["stage1_tokens"].unsqueeze(0)).squeeze(0)
        except Exception:
            dist = torch.zeros(len(labels["stage1_tokens"]), 4, dtype=torch.long)
        _atomic_save({
            "onset":    audio["onset"],
            "chroma":   audio["chroma"],
            "centroid": audio["centroid"],
            "tokens":   labels["stage1_tokens"],
            "distances": dist,
            "bpm":      torch.tensor([meta["bpm"]], dtype=torch.float32),
            "level":    torch.tensor([chart_data["level"]], dtype=torch.float32),
            "genre":    torch.tensor([float(meta["genre"])], dtype=torch.float32),
            "audio_tokens": audio.get("audio_tokens", torch.zeros(0, dtype=torch.long)),
            "normalized_simai": simai.get("normalized", ""),
            "stage1_simai": simai.get("stage1", ""),
        }, cache_root / "stage1" / f"{chart_id}.pt")

        # ── slide 训练数据 ──
        for idx, sample in enumerate(labels["slide_compact"]):
            _atomic_save(sample, cache_root / "slide" / f"{chart_id}_{idx:03d}.pt")
        for idx, sample in enumerate(labels["stage2_star_events"]):
            _atomic_save(sample, cache_root / "stage2_star" / f"{chart_id}_{idx:03d}.pt")


def _atomic_save(data: Any, path: Path) -> None:
    """原子保存: 先写 .tmp 再 rename，避免中断导致文件损坏。"""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, tmp)
    try:
        os.replace(tmp, path)
    except PermissionError:
        torch.save(data, path)
        try:
            tmp.unlink()
        except OSError:
            pass


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Phase 2a: 多 Stage Token 生成")
    p.add_argument("--data-root", default="datasets")
    p.add_argument("--cache-root", default="/data/maiG_v2/cache")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--subdiv", type=int, default=64)
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--maxsubdiv", type=int, default=64, help="归一化网格精度（每拍分割数）")
    p.add_argument("--force", action="store_true", help="强制重新处理（默认跳过已有）")
    args = p.parse_args()

    data_root  = Path(args.data_root)
    cache_root = Path(args.cache_root)
    audio_dir  = cache_root / "_audio"

    folders = sorted(
        [d for d in data_root.iterdir() if d.is_dir() and (d / "maidata.txt").exists()],
        key=lambda x: x.name)
    logger.info(f"找到 {len(folders)} 首歌曲")
    if args.limit:
        folders = folders[:args.limit]

    if not args.force:
        s1_dir = cache_root / "stage1"
        before = len(folders)
        folders = [f for f in folders if not list(s1_dir.glob(f"{f.name}_lv*.pt"))]
        logger.info(f"跳过 {before - len(folders)} 首已有，需处理 {len(folders)} 首")

    if not folders:
        logger.info("无需处理"); return

    ok = fail = 0
    failed_items: list[dict[str, str]] = []
    if args.num_workers > 1:
        with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
            fut = {ex.submit(process_one, f, audio_dir, args.max_tokens, args.maxsubdiv): f for f in folders}
            for fu in as_completed(fut):
                r = fu.result()
                if "error" in r:
                    failed_items.append({"folder": fut[fu].name, "error": str(r["error"])})
                    logger.warning(f"  ✗ {fut[fu].name}: {r['error']}"); fail += 1
                else:
                    save_all(r, cache_root); ok += 1
                if (ok + fail) % 50 == 0:
                    logger.info(f"进度: {ok}✓ / {fail}✗ / {len(folders)}")
    else:
        for i, f in enumerate(folders):
            logger.info(f"[{i+1}/{len(folders)}] {f.name}")
            r = process_one(f, audio_dir, args.max_tokens, args.maxsubdiv)
            if "error" in r:
                failed_items.append({"folder": f.name, "error": str(r["error"])})
                logger.warning(f"  ✗ {r['error']}"); fail += 1
            else:
                save_all(r, cache_root); ok += 1

    logger.info(f"完成! 成功: {ok}, 失败: {fail}")

    (cache_root / "preprocess_label_failures.json").write_text(
        json.dumps(failed_items, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
