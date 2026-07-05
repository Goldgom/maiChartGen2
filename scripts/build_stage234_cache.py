# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
from Tokenizer.config_vocab import BTN_HOLD_ONGOING, CONFIG_TO_ID, ID_TO_CONFIG, SlotConfig, TCH_HOLD_ONGOING

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_stage234")


def _safe_save(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, tmp)
    try:
        os.replace(tmp, path)
    except PermissionError:
        torch.save(data, path)
        try:
            tmp.unlink()
        except OSError:
            pass


def _event_sort_key(ev: dict[str, Any]) -> tuple[int, int]:
    return (int(ev.get("slot", -1)), int(ev.get("note_index", -1)))


def _update_slot_token(token: torch.Tensor | int, *, add_buttons=None, add_touches=None) -> tuple[int, bool]:
    tid = int(token.item()) if torch.is_tensor(token) else int(token)
    sc = ID_TO_CONFIG.get(tid)
    if sc is None:
        return tid, False
    buttons = {int(pos): int(state) for pos, state in sc.buttons}
    touches = {int(zone): int(state) for zone, state in sc.touches}
    for pos, state in add_buttons or []:
        buttons[int(pos)] = int(state)
    for zone, state in add_touches or []:
        touches[int(zone)] = int(state)
    new_sc = SlotConfig(buttons=tuple(sorted(buttons.items())), touches=tuple(sorted(touches.items())))
    new_tid = CONFIG_TO_ID.get(new_sc)
    if new_tid is None:
        return tid, False
    return int(new_tid), True


def _backfill_hold_tokens(tokens: torch.Tensor, *, slot: int, rows: int, positions=None, zones=None) -> tuple[torch.Tensor, int]:
    out = tokens.clone()
    if rows <= 1:
        return out, 0
    max_idx = int(out.size(0)) - 2
    if max_idx <= slot:
        return out, 0
    add_buttons = [(int(pos), BTN_HOLD_ONGOING) for pos in (positions or [])]
    add_touches = [(int(zone), TCH_HOLD_ONGOING) for zone in (zones or [])]
    changed = 0
    for idx in range(slot + 1, min(max_idx, slot + rows - 1) + 1):
        new_tid, ok = _update_slot_token(out[idx], add_buttons=add_buttons, add_touches=add_touches)
        if ok:
            out[idx] = new_tid
            changed += 1
    return out, changed


def load_stage1_model(checkpoint_path: str, cfg: dict):
    from models.stage1 import MaiGenerator

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
    for stage_info in ckpt.get("stages", []):
        if stage_info.get("name") == "stage1":
            model.load_state_dict(stage_info["model"], strict=False)
            break
    else:
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    model.eval()
    return model


@torch.no_grad()
def export_stage1_hidden(model, stage1_cache: dict, device: str = "cpu") -> dict[str, torch.Tensor]:
    onset = stage1_cache["onset"].unsqueeze(0).to(device)
    chroma = stage1_cache["chroma"].unsqueeze(0).to(device)
    centroid = stage1_cache["centroid"].unsqueeze(0).to(device)
    tokens = stage1_cache["tokens"].unsqueeze(0).to(device)
    bpm = stage1_cache["bpm"].unsqueeze(0).to(device)
    level = stage1_cache["level"].unsqueeze(0).to(device)
    genre = stage1_cache["genre"].unsqueeze(0).to(device)
    out = model(onset, chroma, centroid, tokens, bpm, level, genre)
    hidden = out["hidden_states"].squeeze(0).contiguous().cpu()
    audio_memory = model._raw_audio_summary(onset, chroma, centroid).squeeze(0).contiguous().cpu()
    return {"stage1_hidden": hidden, "audio_memory": audio_memory}


def run_export_hidden(args):
    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    logger.info("加载 Stage 1 模型: %s (device=%s)", args.checkpoint, device)
    model = load_stage1_model(args.checkpoint, cfg).to(device)

    cache_root = Path(args.cache_root)
    s1_dir = cache_root / "stage1"
    hidden_dir = cache_root / "_hidden"
    hidden_dir.mkdir(parents=True, exist_ok=True)
    s1_files = sorted(s1_dir.glob("*.pt"))
    if args.limit:
        s1_files = s1_files[: args.limit]
    logger.info("待导出 %d 首", len(s1_files))

    for i, fpath in enumerate(s1_files, 1):
        out_path = hidden_dir / f"{fpath.stem}.pt"
        cache = torch.load(fpath, map_location="cpu", weights_only=True)
        hidden = export_stage1_hidden(model, cache, device)
        _safe_save(hidden, out_path)
        if i % 50 == 0 or i == len(s1_files):
            logger.info("  进度: %d/%d", i, len(s1_files))
    logger.info("Hidden states 已导出到 %s", hidden_dir)


def _load_stage1_onset(cache_root: Path, chart_id: str):
    stage1_path = cache_root / "stage1" / f"{chart_id}.pt"
    if not stage1_path.exists():
        return None
    d = torch.load(stage1_path, map_location="cpu", weights_only=True)
    onset = d.get("onset")
    return onset if torch.is_tensor(onset) else None


def _prepare_stage_cache_outputs(labels_path: Path, hidden_path: Path, cache_root: Path, chart_id: str) -> list[tuple[Path, dict[str, Any]]]:
    labels = torch.load(labels_path, map_location="cpu", weights_only=True)
    hidden = torch.load(hidden_path, map_location="cpu", weights_only=True)
    stage1_hidden = hidden["stage1_hidden"]
    audio_memory = hidden["audio_memory"]
    onset = _load_stage1_onset(cache_root, chart_id)
    t = min(stage1_hidden.size(0), labels["stage1_tokens"].size(0))
    token_slice = labels["stage1_tokens"][:t]
    hidden_slice = stage1_hidden[:t]
    outputs: list[tuple[Path, dict[str, Any]]] = []

    outputs.append((cache_root / "touch" / f"{chart_id}.pt", {
        "config_tokens": token_slice,
        "stage1_hidden": hidden_slice,
        "audio_memory": audio_memory,
        "onset": onset,
        "zone_targets": labels["touch_targets"][:t],
    }))
    outputs.append((cache_root / "break" / f"{chart_id}.pt", {
        "tokens": token_slice,
        "stage1_hidden": hidden_slice,
        "targets": labels["break_targets"][:t],
        "press_mask": labels["press_mask"][:t],
    }))
    outputs.append((cache_root / "spike" / f"{chart_id}.pt", {
        "tokens": token_slice,
        "stage1_hidden": hidden_slice,
        "targets": labels["spike_targets"][:t],
        "touch_mask": labels["touch_mask"][:t],
    }))

    hold_tokens = token_slice.clone()
    for idx, ev in enumerate(sorted(labels.get("stage3_hold_events", []), key=_event_sort_key)):
        slot = int(ev.get("slot", -1)) + 1
        rows = int(ev.get("dur_rows_target", 0))
        positions = torch.as_tensor(ev.get("positions", []), dtype=torch.long).view(-1).tolist()
        if not (0 <= slot < t) or rows <= 0:
            continue
        outputs.append((cache_root / "hold" / f"{chart_id}_{idx:03d}.pt", {
            "tokens": hold_tokens.clone(),
            "query_slot": torch.tensor(slot, dtype=torch.long),
            "dur_rows_target": torch.tensor(rows, dtype=torch.long),
            "positions": torch.tensor(positions, dtype=torch.long),
            "chart_id": chart_id,
            "event_index": idx,
        }))
        hold_tokens, _ = _backfill_hold_tokens(hold_tokens, slot=slot, rows=rows, positions=positions)

    touch_hold_tokens = token_slice.clone()
    for idx, ev in enumerate(sorted(labels.get("stage4_touch_hold_events", []), key=_event_sort_key)):
        slot = int(ev.get("slot", -1)) + 1
        rows = int(ev.get("dur_rows_target", 0))
        zones = torch.as_tensor(ev.get("zones", []), dtype=torch.long).view(-1).tolist()
        if not (0 <= slot < t) or rows <= 0:
            continue
        outputs.append((cache_root / "touch_hold" / f"{chart_id}_{idx:03d}.pt", {
            "tokens": touch_hold_tokens.clone(),
            "query_slot": torch.tensor(slot, dtype=torch.long),
            "dur_rows_target": torch.tensor(rows, dtype=torch.long),
            "zones": torch.tensor(zones, dtype=torch.long),
            "chart_id": chart_id,
            "event_index": idx,
        }))
        touch_hold_tokens, _ = _backfill_hold_tokens(touch_hold_tokens, slot=slot, rows=rows, zones=zones)

    from Tokenizer.touch_pattern_vocab import TOUCH_PATTERN_NUM_ZONES, encode_zones
    pattern_targets = torch.zeros(t, TOUCH_PATTERN_NUM_ZONES, dtype=torch.float32)
    pattern_tokens = torch.zeros(t, dtype=torch.long)
    pattern_mask = torch.zeros(t, dtype=torch.bool)
    for ev in labels.get("stage5_touch_events", []):
        slot = int(ev.get("slot", -1)) + 1
        zones = ev.get("zones", [])
        if torch.is_tensor(zones):
            zones = zones.reshape(-1).tolist()
        zones = [int(z) for z in zones]
        if 0 <= slot < t and zones:
            pattern_mask[slot] = True
            pattern_tokens[slot] = int(encode_zones(zones))
            for z in zones:
                if 0 <= z < TOUCH_PATTERN_NUM_ZONES:
                    pattern_targets[slot, z] = 1.0
    outputs.append((cache_root / "stage5_touch" / f"{chart_id}.pt", {
        "tokens": token_slice,
        "stage1_hidden": hidden_slice,
        "audio_memory": audio_memory,
        "onset": onset,
        "touch_pattern_targets": pattern_targets,
        "touch_pattern_tokens": pattern_tokens,
        "touch_pattern_mask": pattern_mask,
        "touch_events": labels.get("stage5_touch_events", []),
    }))
    outputs.append((cache_root / "stage6_break_note" / f"{chart_id}.pt", {
        "tokens": token_slice,
        "stage1_hidden": hidden_slice,
        "targets": labels["break_targets"][:t],
        "press_mask": labels["press_mask"][:t],
        "note_events": labels.get("stage6_break_note_events", []),
    }))
    outputs.append((cache_root / "stage7_firework_note" / f"{chart_id}.pt", {
        "tokens": token_slice,
        "stage1_hidden": hidden_slice,
        "targets": labels["spike_targets"][:t],
        "touch_mask": labels["touch_mask"][:t],
        "note_events": labels.get("stage7_firework_note_events", []),
    }))
    return outputs


def build_stage_cache(labels_path: Path, hidden_path: Path, cache_root: Path, chart_id: str) -> int:
    outputs = _prepare_stage_cache_outputs(labels_path, hidden_path, cache_root, chart_id)
    for path, data in outputs:
        _safe_save(data, path)
    return len(outputs)


def _build_event_stage_caches(cache_root: Path, cfg: dict | None = None) -> int:
    # already produced during build_stage_cache; keep compatibility with train_from_zero.sh
    count = 0
    for stage in ("hold", "touch_hold", "stage5_touch", "stage6_break_note", "stage7_firework_note"):
        d = cache_root / stage
        if d.exists():
            count += len(list(d.glob("*.pt")))
    return count


def run_build_caches(args):
    cache_root = Path(args.cache_root)
    labels_dir = cache_root / "_labels"
    hidden_dir = cache_root / "_hidden"
    label_files = {f.stem: f for f in labels_dir.glob("*.pt")}

    if args.placeholder:
        logger.info("占位模式启用")
        import yaml
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
        _build_event_stage_caches(cache_root, cfg)
        stage1_dim = int(cfg.get("models", {}).get("stage1", {}).get("hidden_dim", 768))
        touch_dim = int(cfg.get("models", {}).get("touch", {}).get("hidden_dim", stage1_dim))
        fake_audio = torch.zeros(64, touch_dim)
        ok = 0
        for chart_id, lp in label_files.items():
            labels = torch.load(lp, map_location="cpu", weights_only=True)
            t = labels["stage1_tokens"].size(0)
            fake_hidden = torch.zeros(t, stage1_dim)
            onset = _load_stage1_onset(cache_root, chart_id)
            _safe_save({"config_tokens": labels["stage1_tokens"], "stage1_hidden": fake_hidden, "audio_memory": fake_audio, "onset": onset, "zone_targets": labels["touch_targets"]}, cache_root / "touch" / f"{chart_id}.pt")
            _safe_save({"tokens": labels["stage1_tokens"], "stage1_hidden": fake_hidden, "targets": labels["break_targets"], "press_mask": labels["press_mask"]}, cache_root / "break" / f"{chart_id}.pt")
            _safe_save({"tokens": labels["stage1_tokens"], "stage1_hidden": fake_hidden, "targets": labels["spike_targets"], "touch_mask": labels["touch_mask"]}, cache_root / "spike" / f"{chart_id}.pt")
            ok += 3
        logger.info("完成! 生成 %d 个占位缓存文件", ok)
        return

    hidden_files = {f.stem: f for f in hidden_dir.glob("*.pt")}
    common = sorted(set(label_files) & set(hidden_files))
    if args.limit:
        common = common[: args.limit]
    logger.info("可构建缓存: %d 首 (labels∩hidden)", len(common))

    for stage_dir in ("touch", "break", "spike", "hold", "touch_hold", "stage5_touch", "stage6_break_note", "stage7_firework_note"):
        (cache_root / stage_dir).mkdir(parents=True, exist_ok=True)

    ok = fail = 0
    t0 = time.time()
    workers = max(1, int(args.num_workers or 1))
    if workers == 1:
        for i, chart_id in enumerate(common, 1):
            try:
                ok += build_stage_cache(label_files[chart_id], hidden_files[chart_id], cache_root, chart_id)
            except Exception as exc:
                logger.warning("  error %s: %s", chart_id, exc)
                fail += 1
            if i % 50 == 0 or i == len(common):
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0.0
                logger.info("  progress: %d/%d charts | cache_files=%d ok, fail=%d | %.2f charts/s", i, len(common), ok, fail, rate)
    else:
        futures = {}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for chart_id in common:
                futures[ex.submit(build_stage_cache, label_files[chart_id], hidden_files[chart_id], cache_root, chart_id)] = chart_id
            for i, fut in enumerate(as_completed(futures), 1):
                chart_id = futures[fut]
                try:
                    ok += int(fut.result())
                except Exception as exc:
                    logger.warning("  error %s: %s", chart_id, exc)
                    fail += 1
                if i % 50 == 0 or i == len(common):
                    elapsed = time.time() - t0
                    rate = i / elapsed if elapsed > 0 else 0.0
                    logger.info("  progress: %d/%d charts | cache_files=%d ok, fail=%d | %.2f charts/s", i, len(common), ok, fail, rate)
    logger.info("完成! 生成 %d 个缓存文件, 失败 %d", ok, fail)


def main():
    p = argparse.ArgumentParser(description="Stage1 hidden export and downstream cache build")
    p.add_argument("--step", default="all", choices=["export-hidden", "build-caches", "all"])
    p.add_argument("--checkpoint", default="runs/rotating_4090/stage1/best.pt")
    p.add_argument("--config", default="configs/rotating_4090.yaml")
    p.add_argument("--cache-root", default="cache")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--placeholder", action="store_true")
    p.add_argument("--inject-slide-audio", action="store_true")
    p.add_argument("--strip-slide-audio", action="store_true")
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
