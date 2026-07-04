"""
训练前数据集基础信息报告。

打印各 stage 缓存数量、BPM/Level/Genre 分布、token 长度统计等。
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger("dataset_info")

# ── 辅助工具 ──

def _percentile(data: list[float], p: float) -> float:
    """线性插值分位数。"""
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = k - f
    if f + 1 < len(s):
        return s[f] + c * (s[f + 1] - s[f])
    return s[f]


def _hist_str(data: list[float], bins: int = 10, width: int = 40) -> str:
    """生成 ASCII 直方图字符串。"""
    if not data:
        return "(empty)"
    lo, hi = min(data), max(data)
    if lo == hi:
        return f"all={lo:.1f}"
    step = (hi - lo) / bins
    counts = [0] * bins
    for v in data:
        idx = min(int((v - lo) / step), bins - 1)
        counts[idx] += 1
    max_c = max(counts) if counts else 1
    lines = []
    for i, c in enumerate(counts):
        bar = "█" * max(1, int(c / max_c * width))
        lines.append(f"  [{lo + i * step:6.1f}-{lo + (i + 1) * step:6.1f}] {bar} {c}")
    return "\n".join(lines)


# ── Stage 1 分析 ──

def _analyze_stage1(cache_root: Path) -> dict[str, Any]:
    s1_dir = cache_root / "stage1"
    if not s1_dir.exists():
        return {"count": 0, "error": "stage1 cache not found"}

    files = sorted(s1_dir.glob("*_lv*.pt"))
    # fallback: 旧格式无 _lv 后缀
    if not files:
        files = sorted(s1_dir.glob("*.pt"))
    if not files:
        return {"count": 0, "error": "no stage1 cache files"}

    bpms: list[float] = []
    levels: list[float] = []
    genres: list[int] = []
    tok_lens: list[int] = []
    audio_tok_lens: list[int] = []
    onset_lens: list[int] = []
    truncated: int = 0
    max_tok = 0
    errors: list[str] = []
    songs: set[str] = set()

    for fp in files:
        try:
            d = torch.load(fp, map_location="cpu", weights_only=True)
        except Exception as e:
            errors.append(f"{fp.name}: {e}")
            continue

        # 提取 song_id 用于统计
        stem = fp.stem
        import re
        m = re.search(r'^(.*)_lv\d+$', stem)
        songs.add(m.group(1) if m else stem)

        bpms.append(float(d["bpm"].item()))
        levels.append(float(d["level"].item()))
        genres.append(int(d["genre"].item()))

        tl = int(d["tokens"].size(0))
        tok_lens.append(tl)
        max_tok = max(max_tok, tl)

        at = d.get("audio_tokens")
        if at is not None and at.numel() > 0 and at.size(0) > 0:
            audio_tok_lens.append(int(at.size(0)))

        ol = int(d["onset"].size(0))
        onset_lens.append(ol)

    # 截断检测
    if tok_lens:
        len_counter = Counter(tok_lens)
        for length, cnt in len_counter.most_common(3):
            if cnt >= max(3, len(tok_lens) * 0.03):
                truncated += cnt

    return {
        "count": len(files),
        "songs": len(songs),
        "errors": len(errors),
        "bpm": {
            "min": min(bpms) if bpms else 0,
            "max": max(bpms) if bpms else 0,
            "mean": sum(bpms) / len(bpms) if bpms else 0,
            "median": _percentile(bpms, 50),
            "p5": _percentile(bpms, 5),
            "p95": _percentile(bpms, 95),
        },
        "level": {
            "min": min(levels) if levels else 0,
            "max": max(levels) if levels else 0,
            "mean": sum(levels) / len(levels) if levels else 0,
            "distribution": dict(Counter(levels).most_common(10)),
        },
        "genre": {
            "distribution": dict(Counter(genres).most_common(15)),
        },
        "tokens": {
            "min": min(tok_lens) if tok_lens else 0,
            "max": max(tok_lens) if tok_lens else 0,
            "mean": sum(tok_lens) / len(tok_lens) if tok_lens else 0,
            "median": _percentile(tok_lens, 50),
            "p95": _percentile(tok_lens, 95),
            "p99": _percentile(tok_lens, 99),
        },
        "audio_tokens": {
            "count": len(audio_tok_lens),
            "missing": len(files) - len(audio_tok_lens),
            "min": min(audio_tok_lens) if audio_tok_lens else 0,
            "max": max(audio_tok_lens) if audio_tok_lens else 0,
            "mean": sum(audio_tok_lens) / len(audio_tok_lens) if audio_tok_lens else 0,
        } if audio_tok_lens else {"count": 0, "warning": "no audio tokens found"},
        "onset_length": {
            "min": min(onset_lens) if onset_lens else 0,
            "max": max(onset_lens) if onset_lens else 0,
            "mean": sum(onset_lens) / len(onset_lens) if onset_lens else 0,
        },
        "truncated_samples": truncated,
        "max_token_limit": max_tok,
        "data": {"bpms": bpms, "levels": levels, "tok_lens": tok_lens},
    }


# ── 其他 Stage 分析 ──

def _count_stage(cache_root: Path, stage: str) -> dict[str, Any]:
    d = cache_root / stage
    if not d.exists():
        return {"count": 0, "error": f"cache/{stage} not found"}
    files = sorted(d.glob("*.pt"))
    return {"count": len(files)}


# ── 主入口 ──

def print_dataset_info(cache_root: str | Path, prefix: str = "") -> dict[str, Any]:
    """打印数据集基础信息并返回结构化结果。"""
    cache_root = Path(cache_root)
    pfx = f"[{prefix}] " if prefix else ""

    info: dict[str, Any] = {}

    # Stage 1
    s1 = _analyze_stage1(cache_root)
    info["stage1"] = s1

    # Stage 2-4
    for stage in ["touch", "slide", "break", "spike"]:
        info[stage] = _count_stage(cache_root, stage)

    # ── 格式化输出 ──
    sep = "=" * 60
    logger.info(sep)
    logger.info(f"{pfx}数据集基础信息")
    logger.info(sep)

    # Stage 1
    logger.info(f"  Stage 1: {s1['count']} charts (来自 {s1.get('songs', 0)} 首歌曲)")
    if s1.get("errors"):
        logger.warning(f"    加载错误: {s1['errors']} 首")

    bpm = s1["bpm"]
    logger.info(f"  BPM:  min={bpm['min']:.1f}  max={bpm['max']:.1f}  "
                f"mean={bpm['mean']:.1f}  median={bpm['median']:.1f}  "
                f"p5={bpm['p5']:.1f}  p95={bpm['p95']:.1f}")

    lvl = s1["level"]
    logger.info(f"  Level: min={lvl['min']:.0f}  max={lvl['max']:.0f}  "
                f"mean={lvl['mean']:.1f}")
    if lvl.get("distribution"):
        dist_str = ", ".join(f"L{int(k)}:{v}" for k, v in sorted(lvl["distribution"].items()))
        logger.info(f"    分布: {dist_str}")

    gen = s1["genre"]
    if gen.get("distribution"):
        dist_str = ", ".join(f"G{k}:{v}" for k, v in sorted(gen["distribution"].items()))
        logger.info(f"  Genre: {dist_str}")

    tok = s1["tokens"]
    logger.info(f"  Token 长度: min={tok['min']}  max={tok['max']}  "
                f"mean={tok['mean']:.0f}  median={tok['median']:.0f}  "
                f"p95={tok['p95']:.0f}  p99={tok['p99']:.0f}")
    logger.info(f"    截断样本 (估算): {s1['truncated_samples']}/{s1['count']}")

    atok = s1["audio_tokens"]
    if atok.get("count", 0) > 0:
        logger.info(f"  Audio Tokens (EnCodec): {atok['count']} 首有数据, "
                    f"{atok.get('missing', 0)} 首缺失")
        logger.info(f"    min={atok['min']}  max={atok['max']}  mean={atok['mean']:.0f}")
    elif atok.get("warning"):
        logger.warning(f"  Audio Tokens: ⚠ {atok['warning']}")

    onset = s1["onset_length"]
    logger.info(f"  Audio 帧数 (onset): min={onset['min']}  max={onset['max']}  "
                f"mean={onset['mean']:.0f}")

    # Stage 2-5
    logger.info("  ---")
    for stage in ["touch", "break", "spike", "slide"]:
        c = info[stage]["count"]
        tag = "✅" if c > 0 else "⚠ (需先运行 build_stage234_cache)"
        logger.info(f"  Stage {stage.capitalize()}: {c} charts  {tag}")

    # ── 直方图 ──
    data = s1.get("data", {})
    if data.get("tok_lens"):
        logger.info("  Token 长度分布:")
        logger.info(_hist_str(data["tok_lens"], bins=10))

    if data.get("bpms"):
        logger.info("  BPM 分布:")
        logger.info(_hist_str(data["bpms"], bins=10))

    logger.info(sep)

    return info
