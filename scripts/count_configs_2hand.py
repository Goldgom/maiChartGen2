"""Count valid per-slot configs with 2-hand human constraint."""
import math

dur = 89  # duration combos
NZ = 8    # button positions
TZ = 41   # touch zones (A1-E8 + C)

# ============================================================
# Per-hand action states
# ============================================================
# Button actions (per position):
#   tap, break, hold_start(dur), hold_ongoing,
#   slide_start, slide_ongoing,
#   break_hold_start(dur)
btn_per_pos = 1 + 1 + dur + 1 + 1 + 1 + dur  # = 183

# Touch actions (per zone):
#   touch, touch_hold_start(dur), touch_hold_ongoing,
#   firework, touch_slide_start
tch_per_zone = 1 + dur + 1 + 1 + 1  # = 93

# Total per hand
states_per_hand = 1 + NZ * btn_per_pos + TZ * tch_per_zone
print(f"States per hand: free(1) + button({NZ}x{btn_per_pos}) + touch({TZ}x{tch_per_zone}) = {states_per_hand}")

# ============================================================
# 0-hand (rest)
# ============================================================
zero = 1

# ============================================================
# 1-hand used
# ============================================================
one_btn = NZ * btn_per_pos      # 8 x 183 = 1464
one_tch = TZ * tch_per_zone     # 41 x 93 = 3813
one_hand = one_btn + one_tch
print(f"\n1 hand: btn={one_btn} + tch={one_tch} = {one_hand}")

# ============================================================
# 2-hand used
# ============================================================
# Both on buttons (different positions)
two_btn = math.comb(NZ, 2) * (btn_per_pos ** 2)
print(f"\n2 hands, both buttons: C({NZ},2)={math.comb(NZ,2)} x {btn_per_pos}^2 = {two_btn:,}")

# Both on touch (different zones)
two_tch = math.comb(TZ, 2) * (tch_per_zone ** 2)
print(f"2 hands, both touch:   C({TZ},2)={math.comb(TZ,2)} x {tch_per_zone}^2 = {two_tch:,}")

# One button + one touch
one_each = NZ * TZ * btn_per_pos * tch_per_zone
print(f"2 hands, btn+touch:    {NZ}x{TZ} x {btn_per_pos}x{tch_per_zone} = {one_each:,}")

two_hand = two_btn + two_tch + one_each
print(f"Total 2-hand: {two_hand:,}")

# ============================================================
# Grand total
# ============================================================
grand = zero + one_hand + two_hand
print(f"\n{'='*55}")
print(f"GRAND TOTAL (2-hand, single slot): {grand:,}")
print(f"{'='*55}")

# ============================================================
# Breakdown
# ============================================================
print(f"\n  rest                          : {zero:>12,}")
print(f"  1 hand on button              : {one_btn:>12,}")
print(f"  1 hand on touch               : {one_tch:>12,}")
print(f"  2 hands, both buttons         : {two_btn:>12,}")
print(f"  2 hands, both touch           : {two_tch:>12,}")
print(f"  2 hands, btn + touch          : {one_each:>12,}")
print(f"  ─────────────────────────────")
print(f"  TOTAL                         : {grand:>12,}")

# ============================================================
# Without touch (buttons only)
# ============================================================
btn_only = 1 + NZ * btn_per_pos + math.comb(NZ, 2) * (btn_per_pos ** 2)
print(f"\nButton-only (no touch): {btn_only:,}")

# ============================================================
# Without slide complexity (slide treated as 1 action)
# ============================================================
btn_no_slide_per_pos = 1 + 1 + dur + 1 + dur  # tap, break, hld_start, hld_ongoing, brk_hld_start = 182
btn_no_slide = 1 + NZ * btn_no_slide_per_pos + math.comb(NZ, 2) * (btn_no_slide_per_pos ** 2)
print(f"Button-only (no slide states): {btn_no_slide:,}")
