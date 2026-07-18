"""
直观展示 SimaiTokenizer 解析结构的 Demo
"""
from SimaiPaser import load_simai


def hr(title="", char="="):
    print(f"\n{char * 70}")
    if title:
        print(title)
        print(char * 70)


def show_chart_dataset(sid: int, diff: int, label: str):
    """展示一个谱面的解析结构"""
    data = load_simai(f"datasets/{sid}/maidata.txt")
    chart = data.charts[diff]

    hr(f"示例: datasets/{sid} ({data.title}) — {chart.difficulty_name} Lv.{chart.level}", "=")

    # 原始文本
    lines = chart.raw_notes.strip().split("\n")
    print(f"原始谱面 (前{min(6, len(lines))}行):")
    for line in lines[:6]:
        print(f"  {line}")
    print()

    # Token 列表
    print(f"Token 化结果 (前25个):")
    print(f"  {'#':<4} {'Token':<42} {'M':<4} {'Beat':<8} {'Subdiv':<7} Raw")
    print(f"  {'-'*3:<4} {'-'*41:<42} {'-'*3:<4} {'-'*7:<8} {'-'*6:<7} {'-'*15}")
    for i, t in enumerate(chart.tokens[:25]):
        print(f"  {i:<4} {t.to_string():<42} {t.measure:<4} {t.beat:<8.3f} {t.subdiv:<7} {t.raw_text}")
    print(f"  ... (共 {len(chart.tokens)} tokens, 其中 {chart.total_notes} 个音符)")

    # 分类统计
    tap_tokens = [t for t in chart.tokens if t.token_type.value == "tap"]
    each_tokens = [t for t in tap_tokens if len(t.position) > 1]
    hold_tokens = [t for t in chart.tokens if t.token_type.value == "hold"]
    slide_tokens = [t for t in chart.tokens if t.token_type.value == "slide"]
    touch_tokens = [t for t in chart.tokens if t.token_type.value == "touch"]
    break_tokens = [t for t in chart.tokens if t.has_break]
    ex_tokens = [t for t in chart.tokens if t.has_ex]
    fw_tokens = [t for t in chart.tokens if t.has_firework]

    print(f"\n  分类: TAP={len(tap_tokens)}(Each={len(each_tokens)}) "
          f"HOLD={len(hold_tokens)} SLIDE={len(slide_tokens)} "
          f"TOUCH={len(touch_tokens)} BRK={len(break_tokens)} "
          f"EX={len(ex_tokens)} FW={len(fw_tokens)}")


def show_unique_samples(chart, max_per_type=8):
    """每种类型展示不重复的示例"""
    for ttype, label in [("tap", "TAP"), ("hold", "HOLD"),
                          ("slide", "SLIDE"), ("touch", "TOUCH")]:
        tokens = [t for t in chart.tokens if t.token_type.value == ttype]
        seen = set()
        unique = []
        for t in tokens:
            s = t.to_string()
            if s not in seen:
                seen.add(s)
                unique.append(s)
            if len(unique) >= max_per_type:
                break
        if unique:
            print(f"  {label} ({len(tokens)}个):")
            for s in unique:
                print(f"    {s}")

    # Break samples
    brk = [t.to_string() for t in chart.tokens if t.has_break]
    seen = set()
    unique_brk = []
    for s in brk:
        if s not in seen:
            seen.add(s)
            unique_brk.append(s)
        if len(unique_brk) >= max_per_type:
            break
    if unique_brk:
        print(f"  BREAK ({len(brk)}个):")
        for s in unique_brk:
            print(f"    {s}")


# ============================================================
hr("", "=")
print("SimaiTokenizer 解析结构直观展示")
print("=" * 70)

# 示例1: 简单谱面
show_chart_dataset(10, 2, "Advanced")

# 示例2: 中等难度，含多种slide
data = load_simai("datasets/100/maidata.txt")
chart = data.charts[3]  # Expert
hr(f"示例: datasets/100 ({data.title}) — {chart.difficulty_name} Lv.{chart.level}", "=")
print(f"原始谱面中特殊的行 (含slide/hold):")
for line in chart.raw_notes.split("\n"):
    if any(c in line for c in ["-4", ">", "<", "V", "p", "q", "w"]):
        print(f"  {line}")
        if sum(1 for _ in [1]) > 5:
            break
print()
show_unique_samples(chart)

# 示例3: UTAGE 高难，最多样的note类型
data = load_simai("datasets/22/maidata.txt")
chart = data.charts[6]  # UTAGE
hr(f"示例: datasets/22 ({data.title}) — {chart.difficulty_name} Lv.{chart.level}", "=")
print(f"原始谱面开头:")
for line in chart.raw_notes.split("\n")[:4]:
    print(f"  {line}")
print()
show_unique_samples(chart, max_per_type=6)

hr("完成!", "=")
