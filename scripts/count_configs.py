"""Count valid per-time-slot configurations for maimai charts."""
import math

dur = 89  # duration (num,den) pairs with num/den <= 4.0

print("=" * 65)
print("Per-position states (mutually exclusive)")
print("=" * 65)
print("  empty              : 1")
print("  tap                : 1")
print("  break              : 1")
print("  hold_start         : 89")
print("  hold_ongoing       : 1")
print("  break_hold_start   : 89  (1bh[2:1] etc.)")
per_pos = 1 + 1 + 1 + dur + 1 + dur
print(f"  Total per position : {per_pos}")
print()

# ============================================================
# Only active notes (no hold ongoing)
# ============================================================
ACTIVE = 1 + 1 + dur + dur  # tap + break + hld_start(89) + break_hld_start(89) = 180

total_no_ongoing = 0
print("=== Active notes only (tap/brk/hld_start/brk_hld_start, no ongoing) ===")
for k in range(0, 9):
    combos = math.comb(8, k)
    configs = combos * (ACTIVE ** k)
    total_no_ongoing += configs
    print(f"  {k} notes: C(8,{k})={combos:3d} x 180^{k} = {configs:>20,}")

print(f"  TOTAL (no ongoing): {total_no_ongoing:>20,}")
print()

# ============================================================
# With hold ongoing mixed in
# ============================================================
total = 1  # all-empty rest
print("=== With hold ongoing (0-4 active + 0-4 ongoing) ===")
for na in range(1, 5):  # 1-4 active notes
    for no in range(0, 9 - na + 1):  # ongoing holds
        if na + no > 8:
            continue
        ways = math.comb(8, na) * math.comb(8 - na, no)
        configs = ways * (ACTIVE ** na)
        total += configs
        pct = configs / (total or 1) * 100
        print(f"  {na} active + {no} ongoing: C(8,{na})*C({8-na},{no}) * 180^{na} = {configs:>18,}")

print(f"  TOTAL (0-4 active + ongoing): {total:>20,}")
print()

# ============================================================
# Full theoretical
# ============================================================
full = 1
for na in range(1, 9):
    for no in range(0, 9 - na + 1):
        if na + no > 8:
            continue
        ways = math.comb(8, na) * math.comb(8 - na, no)
        configs = ways * (ACTIVE ** na)
        full += configs

print("=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"  Full theoretical (no slide/touch): {full:>20,}")
print(f"  Practical (<=4 active notes):      {total:>20,}  ({total/full*100:.1f}%)")
print(f"  Even 'practical' is far too large for single-token enum")
print(f"  -> 133 base tokens, compositional encoding = correct design")
