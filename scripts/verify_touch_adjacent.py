"""Verify touch panel adjacency model against user examples."""
import re

_OUTER_SEQ = [
    ("A", 8), ("D", 1), ("A", 1), ("D", 2), ("A", 2), ("D", 3), ("A", 3), ("D", 4),
    ("A", 4), ("D", 5), ("A", 5), ("D", 6), ("A", 6), ("D", 7), ("A", 7), ("D", 8),
]
_ZONE_KEY_TO_IDX = {}
for i in range(8):
    _ZONE_KEY_TO_IDX[("E", i + 1)] = i
    _ZONE_KEY_TO_IDX[("B", i + 1)] = i + 8
_ZONE_KEY_TO_IDX[("C", 1)] = 16
for i, (ch, pos) in enumerate(_OUTER_SEQ):
    _ZONE_KEY_TO_IDX[(ch, pos)] = i + 17

def ring_pos(idx):
    if idx < 8: return (0, idx + 1)
    elif idx < 16: return (1, idx - 7)
    elif idx == 16: return (-1, 1)
    else:
        ch, pos = _OUTER_SEQ[idx - 17]
        return (2, pos)

def zones_adjacent(z1, z2):
    if z1 == z2: return True
    if z1 == 16: return z2 < 8
    if z2 == 16: return z1 < 8
    r1, p1 = ring_pos(z1)
    r2, p2 = ring_pos(z2)

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

def idx(ch, pos): return _ZONE_KEY_TO_IDX[(ch, pos)]

checks = [
    ("C-E1", 16, idx("E", 1)), ("E1-B8", idx("E", 1), idx("B", 8)),
    ("E1-B1", idx("E", 1), idx("B", 1)), ("B1-E2", idx("B", 1), idx("E", 2)),
    ("B1-B2", idx("B", 1), idx("B", 2)), ("A1-D1", idx("A", 1), idx("D", 1)),
    ("A1-A2", idx("A", 1), idx("A", 2)), ("D1-D2", idx("D", 1), idx("D", 2)),
    ("D1-A8", idx("D", 1), idx("A", 8)), ("B1-A1", idx("B", 1), idx("A", 1)),
    ("B1-D2", idx("B", 1), idx("D", 2)), ("C-B1", 16, idx("B", 1)),
    ("E1-A1", idx("E", 1), idx("A", 1)),
]

all_ok = True
for name, z1, z2 in checks:
    result = zones_adjacent(z1, z2)
    # All these should be True except C-B1 and E1-A1
    expected = name not in ("C-B1", "E1-A1")
    ok = result == expected
    if not ok: all_ok = False
    print(f"  {'OK' if ok else 'FAIL'}: {name}={result} (expect {expected})")

total_pairs = sum(1 for z1 in range(33) for z2 in range(z1+1,33))
adj_pairs = sum(1 for z1 in range(33) for z2 in range(z1+1,33) if zones_adjacent(z1,z2))
non_adj = total_pairs - adj_pairs
t1 = 33 * 3
t2 = non_adj * 9
touch_total = t1 + t2
grand = 1 + 40 + 700 + touch_total + 40 * touch_total

print(f"\nPairs: {total_pairs}, Adj: {adj_pairs}, Non-adj: {non_adj}")
print(f"Touch: 1z={t1}, 2z={t2}, total={touch_total}")
print(f"Grand total: {grand}")
print(f"All: {all_ok}")
