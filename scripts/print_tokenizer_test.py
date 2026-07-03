"""
Print tokenizer test results for a real chart.
Usage: python scripts/print_tokenizer_test.py
"""
import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Tokenizer.MaiChartTokenizer import (
    MaiChartTokenizer, token_name,
    TOKENIZER_VERSION, VOCAB_SIZE, USE_CONFIG_VOCAB,
)

# LOVE & JOY EXPERT (inote_3) - varied note types
chart = (
    "(173){1},"
    "{2}3,2,"
    "{2}1,8,"
    "{2}7,6,"
    "{2}5,5,"
    "{2}6,7,"
    "{2}8,1,"
    "{2}2,3,"
    "{2}4,4,"
    "{2}7,6,"
    "{2}5,4,"
    "{2}3,2,"
    "{2}1,1,"
    "{2}2,1,"
    "{2}8,7,"
    "{1}6<2[2:1],"
    "{4}1,8,2/7,,"
    "{4}7,8,2,1,"
    "{4}7,8,7,1/8,"
    "{4}5,7,4,2,"
    "{4}1,2,3,1/8,"
    "{4}2,3,4,5,"
    "{4}6,6,7,8,"
    "{4}7h[2:1],,,3,"
    "{4}1,8,2/7,,"
    "{4}7,8,2,1,"
    "{4}7,8,7,1/8,"
    "{4}5,7,4,2,"
    "{4}1,2,3,1/8,"
    "{4}3,2,1,8,"
    "{4}7,7,6,5,"
    "{1}5-8[2:1],"
    "{4}7/8,1/8,1/2,,"
    "{4}2/7,8,1,8,"
    "{4}2/6,7,8,7,"
    "{4}3/7,2,1,2,"
    "{4}3/6,5,4,5,"
    "{4}2/7,8,1,8,"
    "{4}1/8,7,2,7,"
    "{4}7h[4:3],2,2,2,"
    "{4}3,6,2,2/7,"
    "{4}1h[4:1],2,8h[4:1],7,"
    "{4}3,7h[4:1],6,,"
    "{1}4h[4:3],"
    "{1}2>5[2:1],"
    "{4}6,5,6,5,"
    "{4}3,4,3,4,"
    "{4}2h[2:1],,3,3,"
    "{4}6,7,8,,"
    "{1}8-5[2:1],"
    "E"
)

tok = MaiChartTokenizer()
tokens = tok.encode(chart)

print("=" * 70)
print("  MaiChartTokenizer - Real Chart Test")
print("  LOVE & JOY [SD]  EXPERT (inote_3)")
print("=" * 70)
print(f"  Tokenizer version : v{TOKENIZER_VERSION}")
print(f"  Vocab size        : {VOCAB_SIZE} (raw tokens, config={USE_CONFIG_VOCAB})")
print()

# Decode and check round-trip (ignore beat-div placement differences)
decoded = tok.decode(tokens)
# Strip BPM prefix and all beat-div markers for semantic comparison
expected_clean = re.sub(r"\(\d+(?:\.\d+)?\)", "", chart)
expected_clean = re.sub(r"\{\d+\}", "", expected_clean)
decoded_clean = re.sub(r"\{\d+\}", "", decoded)
roundtrip_ok = decoded_clean == expected_clean
print(f"  Round-trip (notes) : {'OK' if roundtrip_ok else 'FAIL'}")
if not roundtrip_ok:
    # Find first difference
    exp_parts = expected_clean.split(",")
    dec_parts = decoded_clean.split(",")
    for j, (e, d) in enumerate(zip(exp_parts, dec_parts)):
        if e != d:
            print(f"    First diff at note {j}: expected={e!r} decoded={d!r}")
            break
    print(f"    Expected parts: {len(exp_parts)}, Decoded parts: {len(dec_parts)}")
print()

# Split into note-level display
print("-" * 70)
print(f"  {'Token ID':<10} {'Token Name':<28} {'Note'}")
print("-" * 70)

notes_raw = [n for n in expected_clean.split(",")]
note_idx = 0
i = 0
while i < len(tokens):
    tid = tokens[i]
    name = token_name(tid)

    if tid == 1:  # BOS
        label = "(BOS)"
    elif tid == 2:  # EOS
        label = "(EOS)"
    elif tid == 16:  # RST
        label = notes_raw[note_idx] if note_idx < len(notes_raw) else "?"
        note_idx += 1
    elif tid >= 256:  # Config token
        label = notes_raw[note_idx] if note_idx < len(notes_raw) else "?"
        note_idx += 1
    elif tid in (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15):
        label = "(div)"
    elif tid == 50:  # SLD_BEG
        label = notes_raw[note_idx] if note_idx < len(notes_raw) else "?"
    elif tid in (17, 51, 52, 53, 95, 130, 131, 132):
        label = ""  # structural tokens
    elif 42 <= tid <= 49:  # SLD waypoints
        label = ""
    elif 116 <= tid <= 129:  # slide types
        label = ""
    else:
        label = ""

    print(f"  {tid:<10} {name:<28} {label}")
    i += 1

# Statistics
print()
print("-" * 70)
print("  Statistics")
print("-" * 70)

total = len(tokens)
base_count = sum(1 for t in tokens if 18 <= t <= 132)
special_count = sum(1 for t in tokens if t < 18 or t in (50, 51, 52, 53, 95))
note_tokens = base_count

print(f"  Total tokens        : {total}")
print(f"  Note tokens (18-132): {base_count}")
print(f"  Special/control     : {special_count}")

# Count by note type in the chart
tap_count = sum(1 for n in notes_raw if re.match(r"^\d+$", n))
brk_count = sum(1 for n in notes_raw if "b" in n and "[" not in n)
hld_count = sum(1 for n in notes_raw if "h[" in n and ">" not in n and "<" not in n and "-" not in n)
sld_count = sum(1 for n in notes_raw if any(c in n for c in [">", "<", "-", "v", "p", "q"]) and "[" in n)
rst_count = sum(1 for n in notes_raw if n == "")
each_count = sum(1 for n in notes_raw if "/" in n)
end_count = sum(1 for n in notes_raw if n == "E")

print()
print(f"  Chart notes         : {len(notes_raw)}")
print(f"    Tap               : {tap_count}")
print(f"    Break             : {brk_count}")
print(f"    Hold              : {hld_count}")
print(f"    Slide             : {sld_count}")
print(f"    Each (simul)      : {each_count}")
print(f"    Rest              : {rst_count}")
print(f"    End               : {end_count}")
print()

# Show first 30 tokens as compact list
print("-" * 70)
print("  Token sequence (first 40 tokens):")
print("-" * 70)
compact = []
for t in tokens[:40]:
    name = token_name(t)
    compact.append(name if name.startswith("[") else str(t))
print(f"  [{', '.join(compact)}")
if len(tokens) > 40:
    print(f"   ... ({len(tokens) - 40} more tokens) ...")
print(f"  ]")
