"""
Phase 2b: Stage 1 Hidden 导出 + Touch/Break/Spike 缓存构建

在 Stage 1 训练完成后运行，分两步:

Step 1 — 导出 Stage 1 Hidden States:
  python scripts/build_stage234_cache.py --step export-hidden --checkpoint /data/maiG_v2/runs/rotating_4090/stage1/best.pt

Step 2 — 构建 Touch/Break/Spike 缓存:
  python scripts/build_stage234_cache.py --step build-caches

也可以合并执行:
  python scripts/build_stage234_cache.py --step all --checkpoint /data/maiG_v2/runs/rotating_4090/stage1/best.pt

生成的缓存:
  cache/_hidden/{song_id}.pt    Stage 1 hidden states
  cache/touch/{song_id}.pt     Stage 2 (Touch) 训练数据
  cache/break/{song_id}.pt     Stage 3 (Break) 训练数据
  cache/spike/{song_id}.pt     Stage 4 (Spike) 训练数据
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_stage234")


# ═══════════════════════════════════════════════════════════════
# Step 1: 导出 Stage 1 Hidden States
# ═══════════════════════════════════════════════════════════════

def load_stage1_model(checkpoint_path: str, cfg: dict):
    from models.stage1 import MaiGenerator

    # 从完整 config 读取参数（cfg 为顶层 yaml 字典）
    stage1_cfg = cfg.get("models", {}).get("stage1", {})
    multiscale = cfg.get("audio_multiscale", {})
    encodec = cfg.get("audio_encodec", {})

    n_layers = encodec.get("n_layers", 1)
    audio_vocab_size = stage1_cfg.get("audio_vocab_size", n_layers * 1024 + 3)

    model = MaiGenerator(
        hidden_dim=stage1_cfg.get("hidden_dim", 768),
        num_layers=stage1_cfg.get("num_layers", 12),
        num_heads=stage1_cfg.get("num_heads", 12),
        audio_stream_layers=stage1_cfg.get("audio_stream_layers", 4),
        audio_stream_heads=stage1_cfg.get("audio_stream_heads", 12),
        use_checkpoint=False,
        audio_vocab_size=audio_vocab_size,
        global_stride=multiscale.get("global_stride", 8),
        local_window_s=multiscale.get("local_window_s", 5.0),
        local_slots_per_sec=multiscale.get("local_slots_per_sec", 184),
        local_dilation_base=multiscale.get("local_dilation_base", 4),
        max_spectral_len=multiscale.get("max_spectral_len", 16384),
        use_spectral_sliding_window=multiscale.get("use_spectral_sliding_window", False),
        spectral_window_len=multiscale.get("spectral_window_len", 4096),
        spectral_window_stride=multiscale.get("spectral_window_stride", 2048),
        use_sliding_window=stage1_cfg.get("use_sliding_window", False),
        window_tokens=stage1_cfg.get("window_tokens", 4096),
        window_stride=stage1_cfg.get("window_stride", 2048),
        max_summary_tokens=stage1_cfg.get("max_summary_tokens", 16),
        summary_position=stage1_cfg.get("summary_position", "prefix"),
        detach_summary=stage1_cfg.get("detach_summary", True),
        audio_window_tokens=stage1_cfg.get("audio_window_tokens", 4096),
        audio_global_summary_tokens=stage1_cfg.get("audio_global_summary_tokens", 16),
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # Find stage1 model state
    for stage_info in ckpt.get("stages", []):
        if stage_info.get("name") == "stage1":
            model.load_state_dict(stage_info["model"], strict=False)
            break
    else:
        # Fallback: try direct
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    model.eval()
    return model


@torch.no_grad()
def export_stage1_hidden(model, stage1_cache: dict, device: str = "cpu") -> dict[str, torch.Tensor]:
    """对一首歌的 stage1 cache 运行 forward，导出 hidden states 和 audio memory。"""
    onset = stage1_cache["onset"].unsqueeze(0).to(device)
    chroma = stage1_cache["chroma"].unsqueeze(0).to(device)
    centroid = stage1_cache["centroid"].unsqueeze(0).to(device)
    tokens = stage1_cache["tokens"].unsqueeze(0).to(device)
    bpm = stage1_cache["bpm"].unsqueeze(0).to(device)
    level = stage1_cache["level"].unsqueeze(0).to(device)
    genre = stage1_cache["genre"].unsqueeze(0).to(device)

    out = model(onset, chroma, centroid, tokens, bpm, level, genre)
    hidden = out["hidden_states"].squeeze(0).contiguous().cpu()  # [T_tok, D]

    # 导出阶段避免再次整首跑 audio Transformer；缓存轻量全局音频摘要即可。
    audio_memory = model._raw_audio_summary(onset, chroma, centroid).squeeze(0).contiguous().cpu()

    return {
        "stage1_hidden": hidden,
        "audio_memory": audio_memory,
    }


def run_export_hidden(args):
    # Load config
    if args.config:
        import yaml
        with open(args.config, encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    model_cfg = cfg.get("models", {}).get("stage1", {})
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    logger.info(f"加载 Stage 1 模型: {args.checkpoint} (device={device})")
    model = load_stage1_model(args.checkpoint, cfg).to(device)

    cache_root = Path(args.cache_root)
    s1_dir = cache_root / "stage1"
    hidden_dir = cache_root / "_hidden"
    hidden_dir.mkdir(parents=True, exist_ok=True)

    s1_files = sorted(s1_dir.glob("*.pt"))
    if args.limit:
        s1_files = s1_files[:args.limit]

    logger.info(f"待导出 {len(s1_files)} 首")

    for i, fpath in enumerate(s1_files):
        name = fpath.stem
        out_path = hidden_dir / f"{name}.pt"
        if args.skip_existing and out_path.exists():
            continue

        cache = torch.load(fpath, map_location="cpu", weights_only=True)
        try:
            hidden = export_stage1_hidden(model, cache, device)
            torch.save(hidden, out_path)
        except Exception as e:
            logger.warning(f"  ✗ {name}: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        finally:
            del cache

        if (i + 1) % 50 == 0:
            logger.info(f"  进度: {i+1}/{len(s1_files)}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    logger.info(f"Hidden states 已导出到 {hidden_dir}")


# ═══════════════════════════════════════════════════════════════
# Step 2: 构建 Touch/Break/Spike 缓存
# ═══════════════════════════════════════════════════════════════

def build_stage_cache(
    labels_path: Path,
    hidden_path: Path,
    cache_root: Path,
    name: str,
) -> int:
    """构建 touch/break/spike 缓存。返回成功数量。"""
    if not labels_path.exists():
        logger.warning(f"  ⚠ 缺少 labels: {labels_path}")
        return 0
    if not hidden_path.exists():
        logger.warning(f"  ⚠ 缺少 hidden: {hidden_path}")
        return 0

    labels = torch.load(labels_path, map_location="cpu", weights_only=True)
    hidden = torch.load(hidden_path, map_location="cpu", weights_only=True)

    stage1_hidden = hidden["stage1_hidden"]
    audio_memory = hidden["audio_memory"]

    # Align lengths: stage1_hidden [T_tok, D], labels [T_tok, ...]
    T_hidden = stage1_hidden.size(0)
    T_labels = labels["stage1_tokens"].size(0)
    T = min(T_hidden, T_labels)

    ok = 0

    # ── Touch cache ──
    touch_dir = cache_root / "touch"
    touch_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config_tokens": labels["stage1_tokens"][:T],
        "stage1_hidden": stage1_hidden[:T],
        "audio_memory": audio_memory,
        "zone_targets": labels["touch_targets"][:T],
    }, touch_dir / f"{name}.pt")
    ok += 1

    # ── Break cache ──
    break_dir = cache_root / "break"
    break_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "tokens": labels["stage1_tokens"][:T],
        "stage1_hidden": stage1_hidden[:T],
        "targets": labels["break_targets"][:T],
        "press_mask": labels["press_mask"][:T],
    }, break_dir / f"{name}.pt")
    ok += 1

    # ── Spike cache ──
    spike_dir = cache_root / "spike"
    spike_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "tokens": labels["stage1_tokens"][:T],
        "stage1_hidden": stage1_hidden[:T],
        "targets": labels["spike_targets"][:T],
        "touch_mask": labels["touch_mask"][:T],
    }, spike_dir / f"{name}.pt")
    ok += 1

    # ── Hold cache (Stage 3) ──
    hold_dir = cache_root / "hold"
    hold_dir.mkdir(parents=True, exist_ok=True)
    hold_mask = torch.zeros(T, dtype=torch.bool)
    dur_num_targets = torch.zeros(T, dtype=torch.long)
    dur_den_targets = torch.zeros(T, dtype=torch.long)
    _compute_hold_targets_from_labels(labels, T, hold_mask, dur_num_targets, dur_den_targets)
    torch.save({
        "tokens": labels["stage1_tokens"][:T],
        "stage1_hidden": stage1_hidden[:T],
        "hold_mask": hold_mask,
        "dur_num_targets": dur_num_targets,
        "dur_den_targets": dur_den_targets,
    }, hold_dir / f"{name}.pt")
    ok += 1

    # ── Touch Hold cache (Stage 4) ──
    thold_dir = cache_root / "touch_hold"
    thold_dir.mkdir(parents=True, exist_ok=True)
    thold_mask = torch.zeros(T, dtype=torch.bool)
    thold_num = torch.zeros(T, dtype=torch.long)
    thold_den = torch.zeros(T, dtype=torch.long)
    _compute_touch_hold_targets_from_labels(labels, T, thold_mask, thold_num, thold_den)
    torch.save({
        "tokens": labels["stage1_tokens"][:T],
        "stage1_hidden": stage1_hidden[:T],
        "touch_hold_mask": thold_mask,
        "dur_num_targets": thold_num,
        "dur_den_targets": thold_den,
    }, thold_dir / f"{name}.pt")
    ok += 1

    return ok


def _compute_hold_targets_from_labels(labels, T, hold_mask, num_t, den_t):
    """从 labels 计算 hold duration 训练目标。"""
    from Tokenizer.config_vocab import ID_TO_CONFIG, BTN_HOLD_START
    tokens = labels["stage1_tokens"][:T].tolist()
    i = 0
    while i < T:
        tid = tokens[i]
        sc = ID_TO_CONFIG.get(tid)
        if sc is not None:
            has_hold_start = any(s == BTN_HOLD_START for _, s in sc.buttons)
            if has_hold_start:
                hold_mask[i] = True
                # 查找 hold 持续 slot 数
                j = i + 1
                while j < T:
                    sc2 = ID_TO_CONFIG.get(tokens[j])
                    if sc2 is None:
                        break
                    has_hold = any(s in (BTN_HOLD_START, 2) for _, s in sc2.buttons)  # 2 = HOLD_ONGOING
                    if not has_hold:
                        break
                    j += 1
                dur_slots = j - i
                dur_beats = dur_slots / 64  # maxsubdiv
                num_idx = min(range(8), key=lambda k: abs([1,2,3,4,6,8,12,16][k] - max(1, round(dur_beats * 64))))
                den_idx = 0  # den=64
                num_t[i] = num_idx
                den_t[i] = den_idx
                i = j
            else:
                i += 1
        else:
            i += 1


def _compute_touch_hold_targets_from_labels(labels, T, hold_mask, num_t, den_t):
    """从 labels 计算 touch hold duration 训练目标。"""
    from Tokenizer.config_vocab import ID_TO_CONFIG, TCH_HOLD_START
    tokens = labels["stage1_tokens"][:T].tolist()
    i = 0
    while i < T:
        tid = tokens[i]
        sc = ID_TO_CONFIG.get(tid)
        if sc is not None:
            has_th_start = any(s == TCH_HOLD_START for _, s in sc.touches)
            if has_th_start:
                hold_mask[i] = True
                j = i + 1
                while j < T:
                    sc2 = ID_TO_CONFIG.get(tokens[j])
                    if sc2 is None or not sc2.touches:
                        break
                    j += 1
                dur_slots = j - i
                num_idx = min(range(8), key=lambda k: abs([1,2,3,4,6,8,12,16][k] - max(1, round(dur_slots / 64 * 64))))
                num_t[i] = num_idx
                den_t[i] = 0
                i = j
            else:
                i += 1
        else:
            i += 1


def _safe_save(data: Any, path: Path) -> None:
    """原子保存: 先写临时文件再重命名，避免中断导致文件损坏。"""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, tmp)
    tmp.replace(path)


def _inject_slide_audio(label_files, hidden_files, cache_root, args):
    """将 Stage 1 的 audio_memory 注入到 slide 缓存文件中。"""
    slide_files = sorted((cache_root / "slide").glob("*.pt"))
    slide_files += sorted((cache_root / "stage2_star").glob("*.pt"))
    if not slide_files:
        return
    logger.info(f"注入 audio_memory 到 {len(slide_files)} 个 slide 文件...")
    updated = 0
    nan_count = 0
    for sf in slide_files:
        song_id = sf.stem.rsplit("_", 1)[0]
        if song_id not in hidden_files:
            continue
        try:
            slide_data = torch.load(sf, map_location="cpu", weights_only=True)
            hidden = torch.load(hidden_files[song_id], map_location="cpu", weights_only=True)
            audio_memory = hidden["audio_memory"]
            # ── NaN 检测 ──
            if torch.is_tensor(audio_memory) and torch.isnan(audio_memory).any():
                logger.warning(f"  ⚠ slide {sf.name}: audio_memory 含 NaN，用零替代")
                audio_memory = torch.nan_to_num(audio_memory, nan=0.0)
                nan_count += 1
            slide_data["audio_memory"] = audio_memory
            _safe_save(slide_data, sf)
            updated += 1
        except Exception as e:
            logger.warning(f"  ✗ slide {sf.name}: {e}")
    logger.info(f"已更新 {updated} 个 slide 文件" + (f"（其中 {nan_count} 个含 NaN）" if nan_count else ""))


def _strip_slide_audio(cache_root: Path) -> None:
    """一次性清理旧流程写入 slide 样本的重复 audio_memory。"""
    slide_files = sorted((cache_root / "slide").glob("*.pt"))
    slide_files += sorted((cache_root / "stage2_star").glob("*.pt"))
    if not slide_files:
        return
    logger.info(f"清理 slide 文件中的重复 audio_memory: {len(slide_files)} 个文件...")
    updated = 0
    for i, sf in enumerate(slide_files, 1):
        try:
            slide_data = torch.load(sf, map_location="cpu", weights_only=True)
            if "audio_memory" in slide_data:
                del slide_data["audio_memory"]
                _safe_save(slide_data, sf)
                updated += 1
        except Exception as e:
            logger.warning(f"  ✗ strip slide {sf.name}: {e}")
        if i % 50000 == 0:
            logger.info(f"  strip 进度: {i}/{len(slide_files)}")
    logger.info(f"已清理 {updated} 个 slide 文件")


def _build_event_stage_caches(cache_root: Path, cfg: dict | None = None) -> int:
    """Build new-stage placeholder caches from _labels event annotations."""
    labels_dir = cache_root / "_labels"
    label_files = sorted(labels_dir.glob("*.pt"))
    if not label_files:
        return 0

    stage1_dim = int((cfg or {}).get("models", {}).get("stage1", {}).get("hidden_dim", 768))
    audio_memory = torch.zeros(64, stage1_dim)
    ok = 0

    for lp in label_files:
        name = lp.stem
        labels = torch.load(lp, map_location="cpu", weights_only=True)
        tokens = labels["stage1_tokens"]
        T = int(tokens.size(0))
        stage1_hidden = torch.zeros(T, stage1_dim)

        hold_num = torch.zeros(T, dtype=torch.long)
        hold_den = torch.zeros(T, dtype=torch.long)
        hold_mask = torch.zeros(T, dtype=torch.bool)
        for ev in labels.get("stage3_hold_events", []):
            slot = int(ev.get("slot", -1)) + 1
            if 0 <= slot < T:
                hold_mask[slot] = True
                hold_num[slot] = int(ev.get("dur_num_target", 0))
                hold_den[slot] = int(ev.get("dur_den_target", 0))
        (cache_root / "hold").mkdir(parents=True, exist_ok=True)
        torch.save({
            "tokens": tokens,
            "stage1_hidden": stage1_hidden,
            "audio_memory": audio_memory,
            "dur_num_targets": hold_num,
            "dur_den_targets": hold_den,
            "hold_mask": hold_mask,
        }, cache_root / "hold" / f"{name}.pt")
        ok += 1

        th_num = torch.zeros(T, dtype=torch.long)
        th_den = torch.zeros(T, dtype=torch.long)
        th_mask = torch.zeros(T, dtype=torch.bool)
        for ev in labels.get("stage4_touch_hold_events", []):
            slot = int(ev.get("slot", -1)) + 1
            if 0 <= slot < T:
                th_mask[slot] = True
                th_num[slot] = int(ev.get("dur_num_target", 0))
                th_den[slot] = int(ev.get("dur_den_target", 0))
        (cache_root / "touch_hold").mkdir(parents=True, exist_ok=True)
        torch.save({
            "tokens": tokens,
            "stage1_hidden": stage1_hidden,
            "audio_memory": audio_memory,
            "dur_num_targets": th_num,
            "dur_den_targets": th_den,
            "touch_hold_mask": th_mask,
        }, cache_root / "touch_hold" / f"{name}.pt")
        ok += 1

        from Tokenizer.touch_pattern_vocab import TOUCH_PATTERN_NUM_ZONES, encode_zones
        pattern_targets = torch.zeros(T, TOUCH_PATTERN_NUM_ZONES, dtype=torch.float32)
        pattern_tokens = torch.zeros(T, dtype=torch.long)
        pattern_mask = torch.zeros(T, dtype=torch.bool)
        for ev in labels.get("stage5_touch_events", []):
            slot = int(ev.get("slot", -1)) + 1
            zones = ev.get("zones", [])
            if torch.is_tensor(zones):
                zones = zones.reshape(-1).tolist()
            zones = [int(z) for z in zones]
            if 0 <= slot < T and zones:
                pattern_mask[slot] = True
                pattern_tokens[slot] = int(encode_zones(zones))
                for z in zones:
                    if 0 <= z < TOUCH_PATTERN_NUM_ZONES:
                        pattern_targets[slot, z] = 1.0

        (cache_root / "stage5_touch").mkdir(parents=True, exist_ok=True)
        torch.save({
            "tokens": tokens,
            "stage1_hidden": stage1_hidden,
            "audio_memory": audio_memory,
            "touch_pattern_targets": pattern_targets,
            "touch_pattern_tokens": pattern_tokens,
            "touch_pattern_mask": pattern_mask,
            "touch_events": labels.get("stage5_touch_events", []),
        }, cache_root / "stage5_touch" / f"{name}.pt")
        ok += 1

        (cache_root / "stage6_break_note").mkdir(parents=True, exist_ok=True)
        torch.save({
            "tokens": tokens,
            "stage1_hidden": stage1_hidden,
            "targets": labels["break_targets"],
            "press_mask": labels["press_mask"],
            "note_events": labels.get("stage6_break_note_events", []),
        }, cache_root / "stage6_break_note" / f"{name}.pt")
        ok += 1

        (cache_root / "stage7_firework_note").mkdir(parents=True, exist_ok=True)
        torch.save({
            "tokens": tokens,
            "stage1_hidden": stage1_hidden,
            "targets": labels["spike_targets"],
            "touch_mask": labels["touch_mask"],
            "note_events": labels.get("stage7_firework_note_events", []),
        }, cache_root / "stage7_firework_note" / f"{name}.pt")
        ok += 1

    return ok


def run_build_caches(args):
    cache_root = Path(args.cache_root)
    labels_dir = cache_root / "_labels"
    hidden_dir = cache_root / "_hidden"

    label_files = {f.stem: f for f in labels_dir.glob("*.pt")}

    if args.placeholder:
        # 无 Stage 1 模型时，用零值 hidden 生成占位缓存
        # 从配置读取各 stage 的 hidden_dim
        touch_dim = 768  # 默认
        stage1_dim = 768
        break_dim = 384
        spike_dim = 384
        try:
            import yaml
            with open(args.config, encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
            touch_dim = cfg.get("models", {}).get("touch", {}).get("hidden_dim", 768)
            stage1_dim = cfg.get("models", {}).get("stage1", {}).get("hidden_dim", 768)
            break_dim = cfg.get("models", {}).get("break", {}).get("hidden_dim", 384)
            spike_dim = cfg.get("models", {}).get("spike", {}).get("hidden_dim", 384)
        except Exception:
            pass

        logger.info(f"占位模式: hidden_dim touch={touch_dim} break={break_dim} spike={spike_dim}")
        ok = 0
        for name, lp in label_files.items():
            try:
                labels = torch.load(lp, map_location="cpu", weights_only=True)
                T = labels["stage1_tokens"].size(0)

                # Touch (768)
                fake_hidden_t = torch.zeros(T, stage1_dim)
                fake_audio_t = torch.zeros(64, touch_dim)
                (cache_root / "touch").mkdir(parents=True, exist_ok=True)
                torch.save({
                    "config_tokens": labels["stage1_tokens"],
                    "stage1_hidden": fake_hidden_t,
                    "audio_memory": fake_audio_t,
                    "zone_targets": labels["touch_targets"],
                }, cache_root / "touch" / f"{name}.pt")

                # Break (384)
                fake_hidden_b = torch.zeros(T, stage1_dim)
                (cache_root / "break").mkdir(parents=True, exist_ok=True)
                torch.save({
                    "tokens": labels["stage1_tokens"],
                    "stage1_hidden": fake_hidden_b,
                    "targets": labels["break_targets"],
                    "press_mask": labels["press_mask"],
                }, cache_root / "break" / f"{name}.pt")

                # Spike (384)
                fake_hidden_s = torch.zeros(T, stage1_dim)
                (cache_root / "spike").mkdir(parents=True, exist_ok=True)
                torch.save({
                    "tokens": labels["stage1_tokens"],
                    "stage1_hidden": fake_hidden_s,
                    "targets": labels["spike_targets"],
                    "touch_mask": labels["touch_mask"],
                }, cache_root / "spike" / f"{name}.pt")
                ok += 3
            except Exception as e:
                logger.warning(f"  ✗ {name}: {e}")
        logger.info(f"完成! 生成 {ok} 个占位缓存文件")
        return

    hidden_files = {f.stem: f for f in hidden_dir.glob("*.pt")}

    if args.strip_slide_audio:
        _strip_slide_audio(cache_root)

    # Slide 样本数量很大；默认由 StageCacheDataset 按 song id 从 _hidden 懒加载 audio_memory。
    # 只有显式要求兼容旧流程时，才把 audio_memory 注入到每个 slide 文件。
    if args.inject_slide_audio:
        _inject_slide_audio(label_files, hidden_files, cache_root, args)

    common = sorted(set(label_files) & set(hidden_files))
    if args.limit:
        common = common[:args.limit]

    logger.info(f"可构建缓存: {len(common)} 首 (labels∩hidden)")

    ok = fail = 0
    for name in common:
        try:
            n = build_stage_cache(label_files[name], hidden_files[name], cache_root, name)
            ok += n
        except Exception as e:
            logger.warning(f"  ✗ {name}: {e}")
            fail += 1

    logger.info(f"完成! 生成 {ok} 个缓存文件, 失败 {fail}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Phase 2b: Hidden导出 + Touch/Break/Spike缓存")
    p.add_argument("--step", default="all", choices=["export-hidden", "build-caches", "all"])
    p.add_argument("--checkpoint", default="/data/maiG_v2/runs/rotating_4090/stage1/best.pt", help="Stage 1 checkpoint")
    p.add_argument("--config", default="configs/rotating_4090.yaml")
    p.add_argument("--cache-root", default="/data/maiG_v2/cache")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--placeholder", action="store_true", help="无 Stage 1 模型时，用零值 hidden 占位生成缓存")
    p.add_argument("--inject-slide-audio", action="store_true", help="兼容旧流程: 将 audio_memory 写入每个 slide 样本")
    p.add_argument("--strip-slide-audio", action="store_true", help="一次性删除 slide 样本里的重复 audio_memory")
    args = p.parse_args()

    if args.step in ("export-hidden", "all"):
        logger.info("=" * 60)
        logger.info("Step 1: 导出 Stage 1 Hidden States")
        logger.info("=" * 60)
        run_export_hidden(args)

    if args.step in ("build-caches", "all"):
        logger.info("=" * 60)
        logger.info("Step 2: 构建 Touch/Break/Spike 缓存")
        logger.info("=" * 60)
        run_build_caches(args)


if __name__ == "__main__":
    main()
