"""
Chart pre-processing utilities for staged training.

Stage 1 training: strip break and firework modifiers from charts.
These are restored in Stage 3 (break) and Stage 4 (firework).

Usage:
    from Tokenizer.chart_preprocess import strip_break, strip_firework, strip_all

    clean = strip_break("(180){4}1b,2b,3,E")   # → "(180){4}1,2,3,E"
    clean = strip_firework("(180){4}B1f,C,E")   # → "(180){4}B1,C,E"
    clean = strip_all("(180){4}1b,B1f,E")       # → "(180){4}1,B1,E"
"""

from __future__ import annotations

import re

# ═══════════════════════════════════════════════════════════════════════
# Regex patterns
# ═══════════════════════════════════════════════════════════════════════

# Break modifier: 'b' that is NOT inside [...] and NOT part of a number
# Patterns: 1b, 2b, 1bh[4:1], 1-4b[4:1], 1b/2, 1/2b
# We strip 'b' when it's a note modifier (before h, before /, before comma, at end, before [)
_RE_BREAK = re.compile(r"b(?=h|/|,|\[|$)")

# Firework modifier: 'f' suffix on touch notes
# Patterns: B1f, Cf, B7f (f before comma, /, or end)
_RE_FIREWORK = re.compile(r"f(?=,|/|$)")

# ═══════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════


def strip_break(text: str) -> str:
    """Remove all break ('b') modifiers from a simai chart text.

    Examples:
        strip_break("1b,2,3b/4,1-4b[2:1]") → "1,2,3/4,1-4[2:1]"
        strip_break("1bh[4:1]")              → "1h[4:1]"
    """
    # Split into comma-separated tokens, respecting [...]
    return _process_tokens(text, _RE_BREAK, "")


def strip_firework(text: str) -> str:
    """Remove all firework ('f') modifiers from a simai chart text.

    Examples:
        strip_firework("B1f,Cf,B2") → "B1,C,B2"
    """
    return _process_tokens(text, _RE_FIREWORK, "")


def strip_all(text: str) -> str:
    """Remove both break and firework modifiers."""
    text = _process_tokens(text, _RE_BREAK, "")
    text = _process_tokens(text, _RE_FIREWORK, "")
    return text


# ═══════════════════════════════════════════════════════════════════════
# Internal
# ═══════════════════════════════════════════════════════════════════════

def _process_tokens(text: str, pattern: re.Pattern, replacement: str) -> str:
    """Apply regex substitution to each comma-separated token,
    respecting [...] brackets (don't modify inside brackets).
    """
    result = []
    i = 0
    depth = 0
    token_start = i

    while i < len(text):
        ch = text[i]
        if ch == "[":
            depth += 1
            i += 1
            continue
        if ch == "]":
            depth = max(0, depth - 1)
            i += 1
            continue
        if depth > 0:
            i += 1
            continue

        # Beat division {N} or BPM (NNN) — pass through untouched
        if ch == "{" or ch == "(":
            # Skip to closing brace/paren
            close = "}" if ch == "{" else ")"
            j = text.find(close, i)
            if j > i:
                result.append(text[i:j + 1])
                i = j + 1
                token_start = i
                continue

        if ch == ",":
            token = text[token_start:i]
            cleaned = pattern.sub(replacement, token)
            result.append(cleaned)
            result.append(",")
            i += 1
            token_start = i
            continue

        i += 1

    # Last token
    if token_start < len(text):
        token = text[token_start:]
        cleaned = pattern.sub(replacement, token)
        result.append(cleaned)

    return "".join(result)


# ═══════════════════════════════════════════════════════════════════════
# Batch processing
# ═══════════════════════════════════════════════════════════════════════

def strip_break_from_file(input_path: str, output_path: str) -> None:
    """Read a maidata.txt, strip breaks, write output."""
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    # Only process &inote_N= blocks
    result = _process_inote_blocks(text, strip_break)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result)


def strip_firework_from_file(input_path: str, output_path: str) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    result = _process_inote_blocks(text, strip_firework)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result)


def strip_all_from_file(input_path: str, output_path: str) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    result = _process_inote_blocks(text, strip_all)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result)


def _process_inote_blocks(text: str, fn) -> str:
    """Apply fn to each &inote_N= block, leave headers untouched."""
    lines = text.split("\n")
    result = []
    in_note_block = False

    for line in lines:
        if line.startswith("&inote_"):
            in_note_block = True
            result.append(line)
            continue
        if in_note_block:
            if line.startswith("&"):
                in_note_block = False
                result.append(line)
            else:
                result.append(fn(line))
        else:
            result.append(line)

    return "\n".join(result)


# ═══════════════════════════════════════════════════════════════════════
# Subdivision normalization
# ═══════════════════════════════════════════════════════════════════════

DEFAULT_SUBDIV = 64  # sub-beats per beat


def normalize_subdivision(text: str, subdiv: int = DEFAULT_SUBDIV,
                          snap: bool = True) -> str:
    """Normalize all beat divisions to a fixed subdivision per beat.

    Args:
        text: simai chart text
        subdiv: target subdivisions per beat (default 64)
        snap: if True, snap misaligned notes to nearest sub-beat

    Example:
        normalize_subdivision(\"{4}1,2,3,4,{8}1,2,3,4,5,6,7,8,E\")
        → \"{64}1,,,...2,,,...\"
    """
    notes = _parse_to_absolute_ticks(text)
    if snap:
        notes = _snap_to_grid(notes, subdiv)
    return _ticks_to_text(notes, subdiv)


def simplify_subdivision(text: str) -> str:
    """Reduce subdivision by finding GCD of note intervals.

    After generation, notes on a {64} grid may all land on even positions.
    This finds the GCD and reduces: {64}→{32}, {32}→{16}, etc.

    Example:
        simplify_subdivision(\"{64}1,,,,2,,,,3,,,,E\")
        → \"{16}1,2,3,E\"   (notes spaced by 4 → GCD=4 → 64/4=16)
    """
    # Extract subdivision and note positions
    bd_match = re.search(r"\{(\d+)\}", text)
    if not bd_match:
        return text
    current_subdiv = int(bd_match.group(1))

    # Remove beat div and BPM prefix for analysis
    clean = re.sub(r"\{(\d+)\}", "", text)
    clean = re.sub(r"\(\d+(?:\.\d+)?\)", "", clean)

    # Find positions of non-empty slots
    slots = _split_chart(clean)
    note_positions = []
    for i, s in enumerate(slots):
        s = s.strip()
        if s and s != "E":
            note_positions.append(i)

    if len(note_positions) < 2:
        return text  # too few notes to optimize

    # Find GCD of all intervals between consecutive notes
    import math
    intervals = []
    for i in range(1, len(note_positions)):
        intervals.append(note_positions[i] - note_positions[i - 1])

    gcd_val = intervals[0]
    for iv in intervals[1:]:
        gcd_val = math.gcd(gcd_val, iv)

    if gcd_val <= 1:
        return text  # no simplification possible

    # New subdivision = old / gcd
    new_subdiv = current_subdiv // gcd_val
    if new_subdiv < 1:
        return text

    # Rebuild chart with new subdivision
    result_parts = []
    note_idx = 0
    for i in range(0, len(slots), gcd_val):
        # Merge gcd_val consecutive slots
        merged = []
        for j in range(gcd_val):
            if i + j < len(slots) and slots[i + j].strip():
                s = slots[i + j].strip()
                if s and s != "E":
                    merged.append(s)
        if merged:
            result_parts.append("/".join(merged))
        else:
            result_parts.append("")

    # Reconstruct
    result = f"{{{new_subdiv}}}"
    has_end = slots and slots[-1].strip() == "E"
    for i, p in enumerate(result_parts):
        result += p + ","
        if has_end and i == len(result_parts) - 1:
            result += "E"
    if not has_end and result.endswith(","):
        result = result.rstrip(",")

    return result


def _snap_to_grid(notes: list[dict], subdiv: int) -> list[dict]:
    """Snap misaligned notes to the nearest sub-beat grid position.

    Some charts may have notes that don't align perfectly with the
    subdivision grid (e.g., due to BPM changes or rounding).
    This snaps each note to the closest sub-beat boundary.
    """
    LCM = 384
    tick_step = LCM // subdiv

    for note in notes:
        if "tick" not in note:
            continue
        tick = note["tick"]
        # Find nearest sub-beat boundary
        slot = round(tick / tick_step)
        note["tick"] = slot * tick_step

    # Merge notes that land on the same slot
    merged = []
    seen_ticks = set()
    for note in notes:
        tick = note.get("tick", -1)
        if tick < 0:
            merged.append(note)
            continue
        if tick in seen_ticks:
            # Merge with existing note at this tick
            for m in merged:
                if m.get("tick") == tick:
                    _merge_notes(m, note)
                    break
        else:
            seen_ticks.add(tick)
            merged.append(note)

    return merged


def _merge_notes(target: dict, source: dict) -> None:
    """Merge source note into target note (same time slot)."""
    # Merge positions
    target.setdefault("positions", [])
    for p in source.get("positions", []):
        if p not in target["positions"]:
            target["positions"].append(p)
    # Merge touch regions
    target.setdefault("touch_regions", [])
    for r in source.get("touch_regions", []):
        if r not in target["touch_regions"]:
            target["touch_regions"].append(r)
    # Flags
    if source.get("is_hold"):
        target["is_hold"] = True
        if source.get("duration"):
            target["duration"] = source["duration"]
    if source.get("is_slide"):
        target["is_slide"] = True
        target["slide_types"] = source.get("slide_types", [])
    if source.get("is_break"):
        target["is_break"] = True
    if source.get("is_firework"):
        target["is_firework"] = True


def _parse_to_absolute_ticks(text: str) -> list[dict]:
    """Parse chart text into absolute tick-positioned notes.

    Returns list of {tick, type, positions, duration, ...} dicts.
    tick = absolute position in a fixed 384-division grid (LCM of common divs).
    """
    LCM = 384  # divisible by 1,2,4,8,16,32,48,64,128,192,384
    notes = []
    current_div = 4
    abs_tick = 0  # absolute position in LCM ticks

    # Split by comma, respecting [...]
    tokens = _split_chart(text)
    for raw in tokens:
        # Check for beat division (may be standalone or inline with BPM)
        bd_match = re.search(r"\{(\d+)\}", raw)
        if bd_match:
            current_div = int(bd_match.group(1))

        # Strip BPM and beat division, keep the note part
        clean = re.sub(r"\(\d+(?:\.\d+)?\)", "", raw)
        clean = re.sub(r"\{\d+\}", "", clean).strip()
        if not clean:
            abs_tick += LCM // current_div
            continue

        tick_step = LCM // current_div

        if clean == "E":
            notes.append({"tick": abs_tick, "type": "end"})
            break

        if clean == "":
            abs_tick += tick_step
            continue

        note = _parse_note(clean)
        note["tick"] = abs_tick
        note["div"] = current_div
        notes.append(note)
        abs_tick += tick_step

    return notes


def _split_chart(text: str) -> list[str]:
    """Split chart by comma, respecting [...] brackets."""
    tokens = []
    i = 0
    depth = 0
    start = 0
    while i < len(text):
        ch = text[i]
        if ch == "[": depth += 1
        elif ch == "]": depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            tokens.append(text[start:i].strip())
            start = i + 1
        i += 1
    if start < len(text):
        tokens.append(text[start:].strip())
    return tokens


def _parse_note(raw: str) -> dict:
    """Parse a single note token into a dict."""
    note = {"type": "tap", "positions": [], "duration": None,
            "is_hold": False, "is_slide": False, "is_touch": False,
            "is_break": False, "is_firework": False,
            "touch_regions": [], "slide_types": [], "raw": raw}

    t = raw.strip()
    if not t or t == "E":
        return note

    # Touch detection
    if re.match(r"^[A-E]", t) or re.match(r"^C", t):
        note["is_touch"] = True
        regions = re.findall(r"([ABDE][1-8]|C[12]?)", t)
        note["touch_regions"] = [r.rstrip("fhxb0123456789") if r.startswith("C") else r.rstrip("fhxb") for r in regions]
        if "f" in t:
            note["is_firework"] = True
        hold_m = re.search(r"h\[(\d+):(\d+)\]", t)
        if hold_m:
            note["is_hold"] = True
            note["duration"] = (int(hold_m.group(1)), int(hold_m.group(2)))
        return note

    # Button note
    if "b" in t and ("b[" in t or re.search(r"\db", t) or t.endswith("b")):
        note["is_break"] = True

    hold_m = re.search(r"h\[(\d+):(\d+)\]", t)
    if hold_m:
        note["is_hold"] = True
        note["duration"] = (int(hold_m.group(1)), int(hold_m.group(2)))
    elif "h" in t and "[" not in t:
        note["is_hold"] = True

    # Slide
    slide_conn = re.findall(r"[>\-<^vVpqszw]|pp|qq", t)
    if slide_conn and re.search(r"\d", t):
        note["is_slide"] = True
        note["slide_types"] = slide_conn

    # Positions
    cleaned = re.sub(r"\[.*?\]", "", t).rstrip("hbfx*")
    pos_nums = re.findall(r"(\d+)", cleaned)
    note["positions"] = [int(n) for n in pos_nums if 1 <= int(n) <= 8]

    return note


def _ticks_to_text(notes: list[dict], subdiv: int) -> str:
    """Convert absolute-tick notes to normalized {subdiv} chart text."""
    if not notes:
        return "E"

    LCM = 384
    tick_step = LCM // subdiv  # ticks per sub-beat

    # Find max tick
    max_tick = max(n["tick"] for n in notes if "tick" in n)
    num_slots = max_tick // tick_step + 1

    # Assign notes to sub-beat slots
    slots = ["" for _ in range(num_slots)]
    for note in notes:
        if note.get("type") == "end":
            slots.append("E")
            continue
        tick = note["tick"]
        slot_idx = tick // tick_step

        # Reconstruct simai text for this note
        if note.get("is_touch"):
            regions = "/".join(note.get("touch_regions", []))
            s = regions
            if note.get("is_firework"):
                s += "f"
            if note.get("is_hold") and note.get("duration"):
                d = note["duration"]
                s += f"h[{d[0]}:{d[1]}]"
        elif note.get("is_slide"):
            positions = note.get("positions", [])
            types = note.get("slide_types", [])
            s = str(positions[0])
            for j in range(1, len(positions)):
                conn = types[j-1] if j-1 < len(types) else "-"
                s += f"{conn}{positions[j]}"
            if note.get("is_break"):
                s += "b"
            if note.get("duration"):
                d = note["duration"]
                s += f"[{d[0]}:{d[1]}]"
        elif note.get("is_hold"):
            positions = note.get("positions", [])
            s = "/".join(f"{p}h" for p in positions) if len(positions) > 1 else f"{positions[0]}h"
            if note.get("is_break"):
                s = s.replace("h", "bh")
            if note.get("duration"):
                d = note["duration"]
                s += f"[{d[0]}:{d[1]}]"
        else:
            positions = note.get("positions", [])
            if len(positions) > 1:
                s = "/".join(str(p) for p in positions)
            elif positions:
                s = str(positions[0])
                if note.get("is_break"):
                    s += "b"
            else:
                s = ""

        if slots[slot_idx]:
            slots[slot_idx] += "/" + s
        else:
            slots[slot_idx] = s

    # Build output
    result = f"{{{subdiv}}}"
    for s in slots:
        result += s + ","
    if result.endswith(",E,"):
        result = result[:-1]  # remove trailing comma before E

    return result


# ═══════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [
        # (input, expected_after_break_strip)
        ("(180){4}1,2,3,E", "(180){4}1,2,3,E"),
        ("(180){4}1b,2b,3b,E", "(180){4}1,2,3,E"),
        ("(180){4}1bh[4:1],E", "(180){4}1h[4:1],E"),
        ("(180){4}1-4b[4:1],E", "(180){4}1-4[4:1],E"),
        ("(180){4}1b/2,3/4b,E", "(180){4}1/2,3/4,E"),
        ("(180){4}2b/5,E", "(180){4}2/5,E"),
        ("(180){4}1h[2:1],E", "(180){4}1h[2:1],E"),  # h[2:1] preserved
        ("(180){4}C,B1f,B2,E", "(180){4}C,B1f,B2,E"), # break doesn't touch this
    ]

    print("=== strip_break ===")
    for inp, exp in tests:
        out = strip_break(inp)
        ok = "OK" if out == exp else "FAIL"
        if out != exp:
            print(f"  {ok}: {inp} -> {out} (expected {exp})")
    print("  (only failures shown)")

    tests_fw = [
        ("(180){4}B1f,Cf,E", "(180){4}B1,C,E"),
        ("(180){4}B1f,B2f,B3,E", "(180){4}B1,B2,B3,E"),
        ("(180){4}C,E", "(180){4}C,E"),
        ("(180){4}Ch[4:1],E", "(180){4}Ch[4:1],E"),  # h[4:1] not firework
    ]

    print("\n=== strip_firework ===")
    for inp, exp in tests_fw:
        out = strip_firework(inp)
        ok = "OK" if out == exp else "FAIL"
        if out != exp:
            print(f"  {ok}: {inp} -> {out} (expected {exp})")
    print("  (only failures shown)")

    # strip_all
    print("\n=== strip_all ===")
    test = "(180){4}1b,B1f,2b/5,Ch[4:1],E"
    expected = "(180){4}1,B1,2/5,Ch[4:1],E"
    out = strip_all(test)
    print(f"  {test}")
    print(f"  -> {out}")
    print(f"  {'OK' if out == expected else 'FAIL'}")
