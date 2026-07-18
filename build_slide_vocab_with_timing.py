"""
build_slide_vocab_with_timing.py — 重建 slide_vocab.json，每个 token 包含路径+时序

生成格式: path[timing]  例如: -5[8:1], >3[4:1], -5*V26[2:1]
相同路径 + 不同时序 = 不同 token
"""

import json
import re
from pathlib import Path
from collections import Counter

DATASETS_DIR = Path("datasets")
OUTPUT_PATH = Path("preprocessed/slide_vocab_with_timing.json")

# 旧词表备份
OLD_SLIDE_VOCAB_PATH = Path("preprocessed/slide_vocab.json")

# 正则匹配 simai slide 格式: start_pos + path + [timing]
# 例: 1-5[8:1], 1>5-8[4:1], 4-5b[2:1], 8>2*V28[4:3]
SLIDE_RE = re.compile(
    r'(?P<start>\d+)(?P<flags>[bx]*)'
    r'(?P<path>(?:pp|qq|[><^vVpqszw\-])\d*(?:'
    r'(?:pp|qq|[><^vVpqszw\-*])\d*)*)'
    r'\[(?P<timing>[^\]]+)\]'
)


def extract_slide_tokens(chart_text: str) -> list[str]:
    """从 simai 谱面中提取所有 slide 路径+时序组合"""
    tokens = []
    # 逐音符匹配
    for m in SLIDE_RE.finditer(chart_text):
        path = m.group("path")
        timing = m.group("timing")
        if not path:
            continue

        # 分解多段路径, 每段带时序
        segments = _split_path(path)
        for seg in segments:
            token = f"{seg}[{timing}]"
            tokens.append(token)

    return tokens


def _split_path(path: str) -> list[str]:
    """分解 slide 路径为 segment 列表"""
    segments = re.findall(
        r'\*?(?:pp|qq|[-><^vVpqszw])\d+',
        path
    )
    return segments if segments else [path]


def build_vocab():
    """扫描所有训练数据, 构建 slide vocab"""
    all_tokens = Counter()
    chart_count = 0
    slide_chart_count = 0

    for maidata_path in sorted(DATASETS_DIR.glob("*/maidata.txt")):
        try:
            content = maidata_path.read_text(encoding="utf-8")
        except Exception:
            continue

        tokens = extract_slide_tokens(content)
        if tokens:
            slide_chart_count += 1
            all_tokens.update(tokens)
        chart_count += 1

        if chart_count % 500 == 0:
            print(f"  Scanned {chart_count} charts, {slide_chart_count} with slides, "
                  f"{len(all_tokens)} unique tokens...")

    print(f"\nTotal charts: {chart_count}")
    print(f"Charts with slides: {slide_chart_count}")
    print(f"Unique slide tokens (path+timing): {len(all_tokens)}")

    # 构建词表: 0=<PAD>, 1=<EOS>
    vocab = {"<PAD>": 0, "<EOS>": 1}
    # 按频率排序 (高频在前)
    for token, count in all_tokens.most_common():
        vocab[token] = len(vocab)

    print(f"Vocab size (incl PAD/EOS): {len(vocab)}")

    # 统计示例
    print("\nTop 30 most common tokens:")
    for token, count in all_tokens.most_common(30):
        print(f"  {count:5d}x  {token}")

    # 统计不同 timing 的种类
    timings = set()
    for token in all_tokens:
        m = re.search(r'\[([^\]]+)\]', token)
        if m:
            timings.add(m.group(1))
    print(f"\nUnique timings: {len(timings)}")
    for t in sorted(timings, key=lambda x: (len(x), x))[:30]:
        print(f"  [{t}]")

    # 保存
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")

    # 对比旧词表
    if OLD_SLIDE_VOCAB_PATH.exists():
        with open(OLD_SLIDE_VOCAB_PATH, "r", encoding="utf-8") as f:
            old_vocab = json.load(f)
        print(f"Old vocab size: {len(old_vocab)} (path only, no timing)")
        print(f"New vocab size: {len(vocab)} (path+timing)")

    # ── 构建 path → best_timing 映射 (用于推理时注入时序) ──
    path_timing_freq: dict[str, Counter] = {}
    for token, count in all_tokens.items():
        m = re.search(r'^(.+)\[([^\]]+)\]$', token)
        if m:
            path = m.group(1)
            timing = m.group(2)
            if path not in path_timing_freq:
                path_timing_freq[path] = Counter()
            path_timing_freq[path][timing] += count

    path_best_timing = {}
    for path, tc in path_timing_freq.items():
        best_timing, _ = tc.most_common(1)[0]
        path_best_timing[path] = best_timing

    timing_map_path = Path("preprocessed/slide_path_timing_map.json")
    with open(timing_map_path, "w", encoding="utf-8") as f:
        json.dump(path_best_timing, f, ensure_ascii=False, indent=2)
    print(f"\nPath→timing map saved to {timing_map_path}")
    print(f"  {len(path_best_timing)} paths mapped")
    print(f"  Sample mappings:")
    for path, timing in list(path_best_timing.items())[:15]:
        print(f"    {path} → [{timing}]  ({path_timing_freq[path].most_common(3)})")

    return vocab


if __name__ == "__main__":
    build_vocab()
