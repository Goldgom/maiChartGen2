"""
Phase 2a: 标签提取 + 多 Stage Token 生成

为 5 个 Stage 生成各自专用的 token 格式:

  Stage 1 (maiG):     简化 Token — Config + DUR + SLD compact
                      剥离 break/firework, touch 压缩为 Config
  Stage 2 (touchG):   每个时间槽的 Touch Zone 标签 [T, 33]
                      0=无touch, 1=touch, 2=hold_start
  Stage 2.5 (slideG): 完整 Slide Path Token — [BOS, waypoint, connector, ..., EOS]
                      输入: start_pos, end_pos, duration; 输出: 自回归路径序列
  Stage 3 (breakG):   每个时间槽的 Break 标签 [T, 8]
                      0=无press, 1=tap, 2=break, press_mask
  Stage 4 (spikeG):   每个时间槽的 Firework 标签 [T, 33]
                      0/1 per zone, touch_mask

生成缓存:
  cache/_labels/{song_id}.pt     全量原始标注
  cache/stage1/{song_id}.pt      Stage 1 训练数据
  cache/slide/{song_id}_*.pt     Stage 2.5 训练数据

用法:
  python scripts/preprocess_labels.py
  python scripts/preprocess_labels.py --limit 10 --num-workers 4
  python scripts/preprocess_labels.py --skip-existing
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("preprocess_labels")

_RE_LEVEL = re.compile(r"lv\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
_RE_GENRE = re.compile(r"genre\s*[:=]\s*(\d+)", re.IGNORECASE)
_RE_BPM  = re.compile(r"\((\d+(?:\.\d+)?)\)")


def parse_meta(text: str) -> dict:
    bpm, level, genre = 150.0, 10.0, 0
    for line in text.split("\n")[:20]:
        if m := _RE_BPM.search(line):   bpm   = float(m.group(1))
        if m := _RE_LEVEL.search(line): level = float(m.group(1))
        if m := _RE_GENRE.search(line): genre = int(m.group(1))
    return {"bpm": bpm, "level": level, "genre": genre}


# ═══════════════════════════════════════════════════════════════════════
# Token 常量
# ═══════════════════════════════════════════════════════════════════════

from Tokenizer.MaiChartTokenizer import (
    PAD, BOS, EOS, RST, DUR, FIREWORK,
    TAP_TO_ID, HLD_TO_ID, TCH_TO_ID,
    SLD_BASE, SLD_TO_ID,
    SLD_BEG_BASE, SLD_BEG_END, ID_TO_SLD_BEG,
    SLD_END_POS_BASE, SLD_END_POS_END, ID_TO_SLD_END_POS,
    SLD_TYPE_BASE, SLD_CHAR_TO_TYPE,
    SIM_BEG, SIM_END, SIM_COUNT_2,
    ID_TO_DUR_NUM, ID_TO_DUR_DEN,
    encode_duration_tokens, encode_slide_compact, encode_wifi_compact,
    _note_to_slot_config,
)
from Tokenizer.config_vocab import CONFIG_TO_ID as CFG_SLOT_TO_ID
from Tokenizer.touch_expander import zone_index as _zone_index


# ═══════════════════════════════════════════════════════════════════════
# Stage 1 Token — 简化版 (剥离 break/firework)
# ═══════════════════════════════════════════════════════════════════════

def make_stage1_tokens(note) -> list[int]:
    """
    Stage 1 专用: Config Token + DUR + SLD Compact

    - rest   → [RST]
    - break  → 转 tap (剥离 break)
    - firework → 转普通 touch (剥离 firework)
    - hold   → Config (hld, pos, dur) + DUR
    - slide  → Config + SLD compact (SLD_BEG_X→dur→SLD_END_Y)
    - touch  → Config 或 base touch token
    """
    if note.is_rest:
        return [RST]
    if note.is_end:
        return []

    # 尝试 Config Token (自动剥离 break/firework)
    sc = _note_to_slot_config(note)
    if sc is not None:
        cfg_id = CFG_SLOT_TO_ID.get(sc)
        if cfg_id is not None:
            result = [cfg_id]
            if note.hold_duration and not note.is_slide:
                result.extend(encode_duration_tokens(note.hold_duration))
            if note.is_slide:
                result.extend(encode_slide_compact(
                    note.slide_path or note.positions,
                    note.slide_types, note.hold_duration))
            if note.is_touch_slide:
                result.extend(encode_wifi_compact(
                    note.touch_regions, note.hold_duration))
            return result

    # Fallback: 手动剥离
    return _stage1_fallback(note)


def _stage1_fallback(note) -> list[int]:
    if note.is_touch:
        if note.is_touch_slide:
            return encode_wifi_compact(note.touch_regions, note.hold_duration)
        result = [TCH_TO_ID.get(r, RST) for r in note.touch_regions]
        result = [x for x in result if x is not None]
        if len(result) > 1:
            result = [SIM_BEG, SIM_COUNT_2] + result + [SIM_END]
        if note.hold_duration:
            result.extend(encode_duration_tokens(note.hold_duration))
        return result if result else [RST]

    if note.is_slide:
        return encode_slide_compact(
            note.slide_path or note.positions,
            note.slide_types, note.hold_duration)

    if note.is_hold:
        result = [HLD_TO_ID[p] for p in note.positions if 1 <= p <= 8]
        if len(result) > 1:
            result = [SIM_BEG, SIM_COUNT_2] + result + [SIM_END]
        if note.hold_duration:
            result.extend(encode_duration_tokens(note.hold_duration))
        return result

    # Tap (原 break → tap)
    result = [TAP_TO_ID[p] for p in note.positions if 1 <= p <= 8]
    if len(result) > 1:
        result = [SIM_BEG, SIM_COUNT_2] + result + [SIM_END]
    return result if result else [RST]


# ═══════════════════════════════════════════════════════════════════════
# Stage 2.5 Token — Slide 完整路径
# ═══════════════════════════════════════════════════════════════════════

def make_slide_path_tokens(note) -> list[int] | None:
    """
    完整 slide path: [BOS, waypoint, connector, waypoint, ..., EOS]

    waypoint  token: SLD_TO_ID[pos]     (42-49)
    connector token: SLD_CHAR_TO_TYPE[c] (116-129)

    示例: "1-3>5[4:1]" → [BOS, 42, 116, 44, 117, 46, EOS]
    """
    if not note.is_slide:
        return None
    path = note.slide_path or note.positions
    types = note.slide_types
    if len(path) < 2:
        return None

    tokens: list[int] = [BOS]
    for idx, pos in enumerate(path):
        if not (1 <= pos <= 8):
            continue
        tokens.append(SLD_TO_ID.get(pos, SLD_BASE))
        if idx < len(path) - 1:
            conn = types[idx] if idx < len(types) else "-"
            tokens.append(SLD_CHAR_TO_TYPE.get(conn, SLD_TYPE_BASE))
    tokens.append(EOS)
    return tokens


# ═══════════════════════════════════════════════════════════════════════
# 标签提取 — Stage 2/3/4
# ═══════════════════════════════════════════════════════════════════════

def extract_labels_for_note(note) -> dict:
    touch_row = [0] * 33
    break_row = [0] * 8
    press_row = [False] * 8
    spike_row = [0] * 33
    tmask_row = [False] * 33

    # Stage 2: Touch zone + state
    if note.is_touch:
        for region in note.touch_regions:
            try:
                zi = _zone_index(region)
                if 0 <= zi < 33:
                    tmask_row[zi] = True
                    touch_row[zi] = 2 if note.is_touch_hold else 1
            except Exception:
                pass

    # Stage 3: Break per position
    for pos in note.positions:
        if 1 <= pos <= 8:
            press_row[pos - 1] = True
            break_row[pos - 1] = 2 if note.is_break else 1

    # Stage 4: Firework per zone
    if note.is_firework and note.is_touch:
        for region in note.touch_regions:
            try:
                zi = _zone_index(region)
                if 0 <= zi < 33:
                    spike_row[zi] = 1
            except Exception:
                pass

    return {"touch": touch_row, "break": break_row, "press": press_row,
            "spike": spike_row, "tmask": tmask_row}


# ═══════════════════════════════════════════════════════════════════════
# 单曲处理
# ═══════════════════════════════════════════════════════════════════════

def process_one(folder: Path, audio_dir: Path, max_tokens: int) -> dict[str, Any]:
    name = folder.name
    maidata_path = folder / "maidata.txt"
    audio_path = audio_dir / f"{name}.pt"

    if not maidata_path.exists():
        return {"folder": str(folder), "error": "missing_maidata"}
    if not audio_path.exists():
        return {"folder": str(folder), "error": "missing_audio_cache"}

    text = maidata_path.read_text(encoding="utf-8")
    meta = parse_meta(text)
    audio = torch.load(audio_path, map_location="cpu", weights_only=True)

    from Tokenizer.MaiChartTokenizer import _parse_simai_text, MaiChartTokenizer
    tokenizer = MaiChartTokenizer()
    notes = _parse_simai_text(text)

    # 生成各 Stage token
    s1_tokens: list[int] = [BOS]
    full_tokens: list[int] = [BOS]
    touch_targets: list[list[int]] = [[0]*33]
    break_targets: list[list[int]] = [[0]*8]
    press_masks:   list[list[bool]] = [[False]*8]
    spike_targets: list[list[int]] = [[0]*33]
    touch_masks:   list[list[bool]] = [[False]*33]
    slide_samples: list[dict] = []

    for note in notes:
        if note.is_end:
            continue
        s1_tok = [RST] if note.is_rest else make_stage1_tokens(note)
        s1_tokens.extend(s1_tok)
        full_tok = [RST] if note.is_rest else tokenizer._encode_fallback(note, compact_slide=False)
        full_tokens.extend(full_tok)

        labels = extract_labels_for_note(note)
        n = len(s1_tok)
        touch_targets.extend([labels["touch"]] * n)
        break_targets.extend([labels["break"]] * n)
        press_masks.extend([labels["press"]] * n)
        spike_targets.extend([labels["spike"]] * n)
        touch_masks.extend([labels["tmask"]] * n)

        # 完整 slide path (Stage 2.5)
        path_tok = make_slide_path_tokens(note)
        if path_tok and len(path_tok) > 2:
            slide_samples.append({"path_tokens": torch.tensor(path_tok, dtype=torch.long)})

    s1_tokens.append(EOS)
    full_tokens.append(EOS)
    touch_targets.append([0]*33)
    break_targets.append([0]*8)
    press_masks.append([False]*8)
    spike_targets.append([0]*33)
    touch_masks.append([False]*33)

    s1_tokens = s1_tokens[:max_tokens]
    T = len(s1_tokens)

    # 提取 compact slide (简化版, 供训练)
    slide_compact = _extract_compact_slides(s1_tokens)

    return {
        "folder": str(folder), "audio": audio, "meta": meta,
        "labels": {
            "stage1_tokens": torch.tensor(s1_tokens, dtype=torch.long),
            "full_tokens":   torch.tensor(full_tokens[:max_tokens], dtype=torch.long),
            "touch_targets": torch.tensor(touch_targets[:T], dtype=torch.long),
            "break_targets": torch.tensor(break_targets[:T], dtype=torch.long),
            "press_mask":    torch.tensor(press_masks[:T], dtype=torch.bool),
            "spike_targets": torch.tensor(spike_targets[:T], dtype=torch.long),
            "touch_mask":    torch.tensor(touch_masks[:T], dtype=torch.bool),
            "slide_samples": slide_samples,
            "slide_compact": slide_compact,
        },
    }


def _extract_compact_slides(tokens: list[int]) -> list[dict]:
    samples = []
    i = 0
    while i < len(tokens):
        tid = tokens[i]
        if SLD_BEG_BASE <= tid < SLD_BEG_END:
            start_pos = ID_TO_SLD_BEG.get(tid)
            dur = None
            j = i + 1
            if j + 1 < len(tokens) and tokens[j] in ID_TO_DUR_NUM and tokens[j+1] in ID_TO_DUR_DEN:
                dur = (ID_TO_DUR_NUM[tokens[j]], ID_TO_DUR_DEN[tokens[j+1]])
                j += 2
            end_pos = None
            if j < len(tokens) and SLD_END_POS_BASE <= tokens[j] < SLD_END_POS_END:
                end_pos = ID_TO_SLD_END_POS.get(tokens[j])
                j += 1
            if start_pos and end_pos and dur:
                mid = SLD_BASE + start_pos - 1
                end_sld = SLD_BASE + end_pos - 1
                samples.append({
                    "target_path": torch.tensor([BOS, mid, end_sld, EOS], dtype=torch.long),
                    "start_pos": torch.tensor([start_pos], dtype=torch.long),
                    "end_pos":   torch.tensor([end_pos], dtype=torch.long),
                    "duration":  torch.tensor([[float(dur[0]), float(dur[1])]], dtype=torch.float32),
                })
            i = j
        else:
            i += 1
    return samples


# ═══════════════════════════════════════════════════════════════════════
# 保存
# ═══════════════════════════════════════════════════════════════════════

def save_all(result: dict, cache_root: Path) -> None:
    name   = Path(result["folder"]).name
    labels = result["labels"]
    audio  = result["audio"]
    meta   = result["meta"]

    (cache_root / "_labels").mkdir(parents=True, exist_ok=True)
    _atomic_save({
        "stage1_tokens": labels["stage1_tokens"],
        "touch_targets": labels["touch_targets"],
        "break_targets": labels["break_targets"],
        "press_mask":    labels["press_mask"],
        "spike_targets": labels["spike_targets"],
        "touch_mask":    labels["touch_mask"],
        "slide_samples": labels["slide_samples"],
        "slide_compact": labels["slide_compact"],
    }, cache_root / "_labels" / f"{name}.pt")

    (cache_root / "stage1").mkdir(parents=True, exist_ok=True)
    # 预计算 distances 避免训练时 GPU sync
    try:
        from models.stage1 import compute_relative_distances
        dist = compute_relative_distances(labels["stage1_tokens"].unsqueeze(0)).squeeze(0)
    except Exception:
        dist = torch.zeros(len(labels["stage1_tokens"]), 4, dtype=torch.long)
    _atomic_save({
        "onset":    audio["onset"],
        "chroma":   audio["chroma"],
        "centroid": audio["centroid"],
        "tokens":   labels["stage1_tokens"],
        "distances": dist,
        "bpm":      torch.tensor([meta["bpm"]], dtype=torch.float32),
        "level":    torch.tensor([meta["level"]], dtype=torch.float32),
        "genre":    torch.tensor([float(meta["genre"])], dtype=torch.float32),
        "audio_tokens": audio.get("audio_tokens", torch.zeros(0, dtype=torch.long)),
    }, cache_root / "stage1" / f"{name}.pt")

    (cache_root / "slide").mkdir(parents=True, exist_ok=True)
    for idx, sample in enumerate(labels["slide_compact"]):
        # 不再重复存储音频 — Stage 1 训练后由 build_stage234_cache 注入 audio_memory
        _atomic_save(sample, cache_root / "slide" / f"{name}_{idx:03d}.pt")


def _atomic_save(data: Any, path: Path) -> None:
    """原子保存: 先写 .tmp 再 rename，避免中断导致文件损坏。"""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, tmp)
    tmp.replace(path)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Phase 2a: 多 Stage Token 生成")
    p.add_argument("--data-root", default="datasets")
    p.add_argument("--cache-root", default="/data/maiG_v2/cache")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--subdiv", type=int, default=64)
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--force", action="store_true", help="强制重新处理（默认跳过已有）")
    args = p.parse_args()

    data_root  = Path(args.data_root)
    cache_root = Path(args.cache_root)
    audio_dir  = cache_root / "_audio"

    folders = sorted(
        [d for d in data_root.iterdir() if d.is_dir() and (d / "maidata.txt").exists()],
        key=lambda x: x.name)
    logger.info(f"找到 {len(folders)} 首歌曲")
    if args.limit:
        folders = folders[:args.limit]

    if not args.force:
        s1_dir = cache_root / "stage1"
        before = len(folders)
        folders = [f for f in folders if not (s1_dir / f"{f.name}.pt").exists()]
        logger.info(f"跳过 {before - len(folders)} 首已有，需处理 {len(folders)} 首")

    if not folders:
        logger.info("无需处理"); return

    ok = fail = 0
    if args.num_workers > 1:
        with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
            fut = {ex.submit(process_one, f, audio_dir, args.max_tokens): f for f in folders}
            for fu in as_completed(fut):
                r = fu.result()
                if "error" in r:
                    logger.warning(f"  ✗ {fut[fu].name}: {r['error']}"); fail += 1
                else:
                    save_all(r, cache_root); ok += 1
                if (ok + fail) % 50 == 0:
                    logger.info(f"进度: {ok}✓ / {fail}✗ / {len(folders)}")
    else:
        for i, f in enumerate(folders):
            logger.info(f"[{i+1}/{len(folders)}] {f.name}")
            r = process_one(f, audio_dir, args.max_tokens)
            if "error" in r:
                logger.warning(f"  ✗ {r['error']}"); fail += 1
            else:
                save_all(r, cache_root); ok += 1

    logger.info(f"完成! 成功: {ok}, 失败: {fail}")


if __name__ == "__main__":
    main()
