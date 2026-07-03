"""
Final count: 2-hand model with clarified rules.

Rules:
  Q1: Multi-zone touch = 1 hand (any number of simultaneous touch zones)
  Q2: One hand cannot do button + touch simultaneously
  Q3: Slide waypoints do NOT occupy a hand (only start/end)
  Q4: Touch hold ongoing occupies a hand
  Q5: Both hands can be on touch

Model:
  Hand state = free | button(pos, type) | touch(zones_set)
  Total hands used = btn_hands + tch_hands <= 2
  btn_hands = 0 or 1 or 2 (2 only if different positions)
  tch_hands = 0 or 1 (1 if any touch zones active, regardless of count)
  Cannot have 2 btn + touch (would need 3 hands)
"""
import math

dur = 89
NZ = 8
TZ = 41

# Per-position button states
BTN = 1 + 1 + dur + 1 + 1 + 1 + dur  # tap/brk/hld_start/hld_on/sld_start/sld_end/brk_hld_start = 183

# Per-zone touch states
TCH = 1 + dur + 1 + 1 + 1  # touch/tch_hld_start/tch_hld_on/firework/tch_sld_start = 93

# ============================================================
# Touch configs (any number of zones, counts as 1 hand)
# ============================================================
MAX_TCH = 6  # practical limit

tch_k = {}
for k in range(0, MAX_TCH + 1):
    if k == 0:
        tch_k[0] = 1
    else:
        tch_k[k] = math.comb(TZ, k) * (TCH ** k)

tch_1hand = sum(tch_k[k] for k in range(1, MAX_TCH + 1))  # at least 1 zone
tch_any = tch_k[0] + tch_1hand  # 0 or more zones

print("=== Touch configurations ===")
for k in range(0, min(MAX_TCH + 1, 7)):
    label = f"{k} zones"
    print(f"  {label:<12} C({TZ},{k})={math.comb(TZ,k):>10,} x {TCH}^{k} = {tch_k[k]:>25,}")
print(f"  1+ zones (1 hand) : {tch_1hand:,}")
print(f"  0+ zones (any)    : {tch_any:,}")
print()

# ============================================================
# Button configs
# ============================================================
btn_0 = 1
btn_1 = NZ * BTN  # 1 hand on button
btn_2 = math.comb(NZ, 2) * (BTN ** 2)  # 2 hands on buttons (different pos)

print("=== Button configurations ===")
print(f"  0 btn (0 hands): {btn_0:>20,}")
print(f"  1 btn (1 hand) : {btn_1:>20,}")
print(f"  2 btn (2 hands): {btn_2:>20,}")
print()

# ============================================================
# Combinations respecting 2-hand total
# ============================================================
# 0 hands total: rest
c_rest = 1

# 1 hand total:
#   - 1 btn, 0 tch
#   - 0 btn, 1 tch (any zones)
c_1h_btn = btn_1 * tch_k[0]  # 1464 * 1
c_1h_tch = btn_0 * tch_1hand  # 1 * tch_1hand

# 2 hands total:
#   - 2 btn, 0 tch
#   - 1 btn, 1 tch
#   - 0 btn, 2 tch? No - touch always counts as 1 hand regardless of zones.
#     Q5 says both hands CAN be on touch. Each hand has its own set of zones.
#     But from chart perspective, 2-hand touch = just more touch zones active.
#     Since touch zones are already counted combinatorially per-hand,
#     "2 hands on touch" is just the same as 1 hand with touch (the zones
#     from both hands merge into one observable set).
#     HOWEVER, from a counting perspective, if Hand 1 has {B1,B2} and
#     Hand 2 has {C,E7}, this is the SAME chart as one "hand" with {B1,B2,C,E7}.
#     Since the per-hand touch already counts all zone combinations up to MAX_TCH,
#     having 2 hands on touch doesn't add new unique chart configurations.
#     
#     Wait - but if each hand can have up to 6 zones, then 2 hands could have
#     up to 12 zones! Currently MAX_TCH=6 limits one hand to 6 zones.
#     
#     For the chart, the question is just: how many touch zones are active?
#     Whether from 1 hand or 2, the observable result is the same set of zones.
#     So we DON'T double-count. Touch configs are just "set of active zones".
c_2h_btn = btn_2 * tch_k[0]     # 2 btn, 0 tch
c_2h_mix = btn_1 * tch_1hand    # 1 btn, 1 tch
# 0 btn, tch: already covered by c_1h_tch (touch is touch, regardless of 1 or 2 hands)

total = c_rest + c_1h_btn + c_1h_tch + c_2h_btn + c_2h_mix

print("=== Final count (max {} touch zones) ===".format(MAX_TCH))
print(f"  0 hands (rest)              : {c_rest:>25,}")
print(f"  1 hand: btn only            : {c_1h_btn:>25,}")
print(f"  1 hand: touch only          : {c_1h_tch:>25,}")
print(f"  2 hands: btn + btn          : {c_2h_btn:>25,}")
print(f"  2 hands: btn + touch        : {c_2h_mix:>25,}")
print(f"  ─────────────────────────────────────")
print(f"  TOTAL                       : {total:>25,}")
print()

# ============================================================
# By max touch zones
# ============================================================
print("=" * 60)
print("Sensitivity to max touch zones")
print("=" * 60)
for mt in [2, 3, 4, 5, 6, 8, 10]:
    t1 = sum(math.comb(TZ, k) * (TCH ** k) for k in range(1, mt + 1))
    t = 1 + btn_1 + t1 + btn_2 + btn_1 * t1
    print(f"  max {mt:2d} zones: {t:>30,}")

print()
print("For reference:")
print(f"  Without touch (btn only): {1 + btn_1 + btn_2:,}")
print(f"  Old config vocab size:    47,686")
print(f"  Base token vocab:         133")
