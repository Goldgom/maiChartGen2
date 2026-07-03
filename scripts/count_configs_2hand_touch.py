"""Count valid per-slot configs: 2-hand button + multi-finger touch."""
import math

dur = 89
NZ = 8    # button positions
TZ = 41   # touch zones

# Per-position states
btn_states = 1 + 1 + dur + 1 + 1 + 1 + dur  # tap/brk/hld_start/hld_on/sld_start/sld_on/brk_hld_start = 183
tch_states = 1 + dur + 1 + 1 + 1            # touch/tch_hld_start/tch_hld_on/firework/tch_sld_start = 93

# ============================================================
# Touch: unlimited simultaneous (practical cap = 10)
# ============================================================
MAX_TOUCH = 10

touch_configs = {}  # k -> total configs for exactly k touch zones
for k in range(0, min(MAX_TOUCH + 1, TZ + 1)):
    if k == 0:
        touch_configs[0] = 1
    else:
        touch_configs[k] = math.comb(TZ, k) * (tch_states ** k)

touch_total = sum(touch_configs.values())

print("=== Touch-only (0-10 simultaneous zones) ===")
for k in range(0, min(MAX_TOUCH + 1, 7)):
    print(f"  {k} touch zones: C({TZ},{k})={math.comb(TZ,k):>8,} x {tch_states}^{k} = {touch_configs[k]:>20,}")
print(f"  ... (up to {MAX_TOUCH} zones)")
print(f"  Touch-only total: {touch_total:,}")
print()

# ============================================================
# 0 button + k touch
# ============================================================
zero_btn = touch_total
print(f"0 button + any touch: {zero_btn:,}")

# ============================================================
# 1 button + k touch
# ============================================================
one_btn_configs = NZ * btn_states  # 8 x 183 = 1464 (button part)
one_btn_total = one_btn_configs * touch_total
print(f"1 button + any touch: {one_btn_configs:,} x {touch_total:,} = {one_btn_total:,}")

# ============================================================
# 2 buttons + k touch
# ============================================================
two_btn_configs = math.comb(NZ, 2) * (btn_states ** 2)  # 28 x 183^2 = 937,692
two_btn_total = two_btn_configs * touch_total
print(f"2 buttons + any touch: {two_btn_configs:,} x {touch_total:,} = {two_btn_total:,}")

# ============================================================
# Grand total
# ============================================================
grand = zero_btn + one_btn_total + two_btn_total
print(f"\n{'='*60}")
print(f"GRAND TOTAL: {grand:,.0f}")
print(f"{'='*60}")

# ============================================================
# Breakdown by button count
# ============================================================
print(f"\n  Breakdown:")
print(f"    0 btn + 0-{MAX_TOUCH} tch : {zero_btn:>20,}")
print(f"    1 btn + 0-{MAX_TOUCH} tch : {one_btn_total:>20,}")
print(f"    2 btn + 0-{MAX_TOUCH} tch : {two_btn_total:>20,}")
print(f"    ─────────────────────────────")
print(f"    TOTAL                     : {grand:>20,.0f}")

# ============================================================
# More practical: limit to 5 touch zones (covers >99% of real charts)
# ============================================================
print(f"\n{'='*60}")
print(f"PRACTICAL (max 5 touch, covers >99% real charts)")
print(f"{'='*60}")
t5_total = sum(touch_configs[k] for k in range(0, 6))
grand5 = t5_total * (1 + one_btn_configs + two_btn_configs)
print(f"  Touch-only (0-5): {t5_total:,}")
print(f"  Grand total:      {grand5:,.0f}")

# ============================================================
# Compare
# ============================================================
print(f"\n{'='*60}")
print(f"COMPARISON")
print(f"{'='*60}")
print(f"  2 hands for ALL actions (prev):    13,617,382")
print(f"  2 hands btn + multi touch (max 5): {grand5:>15,.0f}")
print(f"  2 hands btn + multi touch (max 10):{grand:>15,.0f}")
print(f"  Ratio (new/old):                    {grand/13617382:.1f}x")
