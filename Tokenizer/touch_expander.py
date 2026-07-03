"""
Touch Expander — compress/decompress connected touch zone groups.

Stage 2 of the cascaded generation pipeline:
  Training:  find connected touch groups, compress each to a single center zone
  Inference: expand a center zone token back to a connected touch pattern

Touch panel layout (maimai DX):
  Center: C  →  E ring  →  B ring  →  A/D outer ring (alternating)

Usage:
    from Tokenizer.touch_expander import TouchExpander

    expander = TouchExpander(max_group_size=5)
    center = expander.compress({0, 1, 8})  # E1, E2, B1 → center zone
    patterns = expander.expand(center)      # all possible patterns containing center
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# ═══════════════════════════════════════════════════════════════════════
# Touch panel layout
# ═══════════════════════════════════════════════════════════════════════

NUM_ZONES = 33

_OUTER_SEQ = [
    ("A", 8), ("D", 1), ("A", 1), ("D", 2), ("A", 2), ("D", 3), ("A", 3), ("D", 4),
    ("A", 4), ("D", 5), ("A", 5), ("D", 6), ("A", 6), ("D", 7), ("A", 7), ("D", 8),
]

_ZONE_KEY_TO_IDX: dict[tuple[str, int], int] = {}
for i in range(8):
    _ZONE_KEY_TO_IDX[("E", i + 1)] = i
    _ZONE_KEY_TO_IDX[("B", i + 1)] = i + 8
_ZONE_KEY_TO_IDX[("C", 1)] = 16
for i, (ch, pos) in enumerate(_OUTER_SEQ):
    _ZONE_KEY_TO_IDX[(ch, pos)] = i + 17

_IDX_TO_NAME: dict[int, str] = {}
for (ch, pos), idx in _ZONE_KEY_TO_IDX.items():
    if ch == "C":
        _IDX_TO_NAME[idx] = "C"
    else:
        _IDX_TO_NAME[idx] = f"{ch}{pos}"


def zone_name(idx: int) -> str:
    return _IDX_TO_NAME.get(idx, "?")


def zone_index(name: str) -> int:
    if name.startswith("C"):
        return 16
    import re
    m = re.match(r"^([ABDE])(\d)$", name)
    if m:
        return _ZONE_KEY_TO_IDX.get((m.group(1), int(m.group(2))), 16)
    return 16


# ═══════════════════════════════════════════════════════════════════════
# Adjacency graph
# ═══════════════════════════════════════════════════════════════════════

def _ring_pos(idx: int) -> tuple[int, int]:
    if idx < 8:
        return (0, idx + 1)
    elif idx < 16:
        return (1, idx - 7)
    elif idx == 16:
        return (-1, 1)
    else:
        ch, pos = _OUTER_SEQ[idx - 17]
        return (2, pos)


def zones_adjacent(z1: int, z2: int) -> bool:
    if z1 == z2:
        return True
    if z1 == 16:
        return z2 < 8
    if z2 == 16:
        return z1 < 8
    r1, p1 = _ring_pos(z1)
    r2, p2 = _ring_pos(z2)

    if r1 == r2:
        if r1 in (0, 1):
            diff = (p1 - p2) % 8
            return diff in (1, 7)
        elif r1 == 2:
            o1, o2 = z1 - 17, z2 - 17
            diff = (o1 - o2) % 16
            return diff in (1, 2, 14, 15)

    if {r1, r2} == {0, 1}:
        e_pos = p1 if r1 == 0 else p2
        b_pos = p2 if r1 == 0 else p1
        return b_pos == e_pos or b_pos == ((e_pos - 2) % 8) + 1

    if {r1, r2} == {1, 2}:
        b_pos = p1 if r1 == 1 else p2
        outer_idx = z2 if r1 == 1 else z1
        ch, o_pos = _OUTER_SEQ[outer_idx - 17]
        return o_pos == b_pos or o_pos == (b_pos % 8) + 1

    return False


# Build adjacency list once
_ADJ = {i: {j for j in range(NUM_ZONES) if i != j and zones_adjacent(i, j)}
        for i in range(NUM_ZONES)}


def get_adjacent(zones: set[int]) -> set[int]:
    """Get all zones adjacent to any zone in the given set."""
    result = set()
    for z in zones:
        result.update(_ADJ[z])
    return result - zones


# ═══════════════════════════════════════════════════════════════════════
# Connected components
# ═══════════════════════════════════════════════════════════════════════

def find_connected_groups(zones: set[int]) -> list[set[int]]:
    """Partition a set of touch zones into connected groups."""
    remaining = set(zones)
    groups = []
    while remaining:
        z = remaining.pop()
        group = {z}
        stack = [z]
        while stack:
            cur = stack.pop()
            for nb in _ADJ[cur]:
                if nb in remaining:
                    remaining.discard(nb)
                    group.add(nb)
                    stack.append(nb)
        groups.append(group)
    return groups


def compress_group(group: set[int]) -> int:
    """Pick a canonical 'center' zone for a connected group.

    Strategy: zone with minimum index (deterministic).
    """
    return min(group)


def decompress_to_patterns(center: int, max_size: int = 5) -> list[set[int]]:
    """Get all connected subgraphs containing 'center', up to max_size.

    These are the possible expansions for a center zone during inference.
    """
    patterns = [{center}]
    frontier = {center}
    seen = {frozenset({center})}

    for _ in range(max_size - 1):
        new_patterns = []
        for pat in patterns:
            # Find neighbors of current pattern
            neighbors = set()
            for z in pat:
                neighbors.update(_ADJ[z])
            neighbors -= pat
            for nb in sorted(neighbors):
                new_pat = pat | {nb}
                key = frozenset(new_pat)
                if key not in seen:
                    seen.add(key)
                    new_patterns.append(new_pat)
        patterns.extend(new_patterns)
    return patterns


# ═══════════════════════════════════════════════════════════════════════
# Expansion mapping
# ═══════════════════════════════════════════════════════════════════════

def build_expansion_map(max_size: int = 5) -> dict[int, list[list[int]]]:
    """Build center_zone → list of all possible expansion patterns.

    Each pattern is a sorted list of zone indices.
    """
    exp_map: dict[int, list[list[int]]] = {i: [] for i in range(NUM_ZONES)}

    for center in range(NUM_ZONES):
        patterns = decompress_to_patterns(center, max_size)
        exp_map[center] = [sorted(p) for p in patterns]

    return exp_map


# ═══════════════════════════════════════════════════════════════════════
# TouchExpander class
# ═══════════════════════════════════════════════════════════════════════

class TouchExpander:
    """Compress/decompress touch zone configurations.

    Training:  group → compress to center zone
    Inference: center zone → expand to possible patterns
    """

    def __init__(self, max_group_size: int = 5):
        self.max_size = max_group_size
        self.expansion_map = build_expansion_map(max_group_size)

    def compress(self, zones: set[int]) -> list[tuple[int, set[int]]]:
        """Compress touch zones into center zones.

        Finds connected groups, replaces each with its center.

        Returns: list of (center_zone, original_group)
        """
        groups = find_connected_groups(zones)
        return [(compress_group(g), g) for g in groups]

    def expand(self, center: int) -> list[list[int]]:
        """Get all possible expansions for a center zone."""
        return self.expansion_map.get(center, [[center]])

    def expand_single(self, center: int) -> list[int]:
        """Expand to just the center zone (size-1 pattern)."""
        return [center]

    @property
    def total_patterns(self) -> int:
        return sum(len(v) for v in self.expansion_map.values())

    def save(self, path: str) -> None:
        """Save expansion map to JSON."""
        data = {str(k): v for k, v in self.expansion_map.items()}
        with open(path, "w") as f:
            json.dump(data, f, ensure_ascii=False)

    def load(self, path: str) -> None:
        """Load expansion map from JSON."""
        with open(path, "r") as f:
            data = json.load(f)
        self.expansion_map = {int(k): v for k, v in data.items()}


# ═══════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    expander = TouchExpander(max_group_size=5)

    print("TouchExpander")
    print(f"  Zones: {NUM_ZONES}")
    print(f"  Max group size: {expander.max_size}")
    print(f"  Total expansion patterns: {expander.total_patterns}")
    print()

    # Test compression
    tests = [
        {0, 1, 8},           # E1, E2, B1 (all adjacent via E-B edges)
        {8, 9, 17},          # B1, B2, A8 (adjacent chain)
        {0, 16},             # E1, C (adjacent)
        {0, 1},              # E1, E2 (same ring adjacent)
        {8, 19, 18},         # B1, A1, D1 (B to outer)
        {0, 10},             # E1, B3 (not adjacent)
    ]

    print("Compression tests:")
    for zones in tests:
        groups = find_connected_groups(zones)
        compressed = expander.compress(zones)
        names_in = [zone_name(z) for z in sorted(zones)]
        names_out = [(zone_name(c), [zone_name(z) for z in sorted(g)]) for c, g in compressed]
        print(f"  {{{','.join(names_in)}}} → {names_out}")

    print()
    print(f"Expansion map entries per zone (first 10):")
    for z in range(min(10, NUM_ZONES)):
        patterns = expander.expand(z)
        print(f"  {zone_name(z)}: {len(patterns)} patterns")

    print()
    print(f"  ... total patterns across all zones: {expander.total_patterns}")
