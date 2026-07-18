"""
train_loop.py — 全流程预处理 + 5-Stage 无限循环训练

流程:
  1. 预处理全量数据 (跳过已存在)
  2. 按难度分层抽取验证集
  3. 循环: Stage1→2→3→4→5, 每轮每 Stage 训练 N epochs
  4. 定期验证 + 保存 best checkpoint
  5. 无限循环直到手动停止

配置: Config/default.yaml
"""

import json, math, time, signal, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════
# 系统监控
# ═══════════════════════════════════════════════════════════

class SystemMonitor:
    def __init__(self):
        self.psutil = None
        self.pynvml = None
        try:
            import psutil; self.psutil = psutil
        except ImportError:
            pass
        try:
            import pynvml
            pynvml.nvmlInit()
            self.pynvml = pynvml
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            pass

    def snapshot(self) -> dict:
        info = {}
        if self.psutil:
            mem = self.psutil.virtual_memory()
            info["cpu"] = self.psutil.cpu_percent(interval=None)
            info["ram"] = mem.percent
        if self.pynvml:
            try:
                util = self.pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                mem_info = self.pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                info["gpu"] = util.gpu
                info["vram"] = mem_info.used / mem_info.total * 100
            except Exception:
                pass
        return info

    def fmt(self) -> str:
        s = self.snapshot()
        parts = []
        if "cpu" in s: parts.append(f"CPU:{s['cpu']:.0f}%")
        if "ram" in s: parts.append(f"RAM:{s['ram']:.0f}%")
        if "gpu" in s: parts.append(f"GPU:{s['gpu']:.0f}%")
        if "vram" in s: parts.append(f"VRAM:{s['vram']:.0f}%")
        return " | ".join(parts) if parts else ""


_monitor = SystemMonitor()

_LOG_FILE = None
_should_stop = False


def log(msg: str = ""):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _LOG_FILE:
        _LOG_FILE.write(line + "\n")
        _LOG_FILE.flush()


def _on_signal(sig, frame):
    global _should_stop
    log(f"\n收到信号 {sig}, 完成当前 step 后停止...")
    _should_stop = True


signal.signal(signal.SIGINT, _on_signal)
signal.signal(signal.SIGTERM, _on_signal)


# ═══════════════════════════════════════════════════════════
# 数据
# ═══════════════════════════════════════════════════════════

def load_and_split_data(cfg):
    data_dir = Path(cfg.preprocess.output_dir)
    preprocess_all(cfg)

    all_files = sorted(data_dir.glob("*.npz"))
    log(f"预处理文件: {len(all_files)} 个")

    by_diff = defaultdict(list)
    for f in all_files:
        meta_path = f.with_suffix(".json")
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text("utf-8"))
        diff = meta["metadata"].get("difficulty_name", "Unknown")
        by_diff[diff].append(f)

    val_files = []
    n_per = cfg.train_loop.val_samples_per_difficulty
    rng = np.random.RandomState(42)
    for diff in ["Easy", "Basic", "Advanced", "Expert", "Master", "Re:Master"]:
        files = by_diff.get(diff, [])
        picked = rng.choice(files, min(n_per, len(files)), replace=False).tolist()
        val_files.extend(picked)
        log(f"  {diff}: {len(picked)} val / {len(files)} total")

    rng.shuffle(val_files)
    val_set = {str(f) for f in val_files}
    train_files = [f for f in all_files if str(f) not in val_set]
    log(f"Train: {len(train_files)}, Val: {len(val_files)}")
    return train_files, val_files


def preprocess_all(cfg):
    data_dir = Path(cfg.preprocess.output_dir)
    existing = len(list(data_dir.glob("*.npz")))
    if existing > 0:
        log(f"预处理已有 {existing} 个文件, 跳过")
        return
    log("开始全量预处理...")
    from PreProcess import Preprocessor
    # 预处理期间恢复默认 SIGINT (Ctrl+C 直接退出)
    old_handler = signal.signal(signal.SIGINT, signal.SIG_DFL)
    try:
        pp = Preprocessor(cfg)
        pp.process_all()
    except KeyboardInterrupt:
        log("\n预处理被用户中断 (Ctrl+C)")
        sys.exit(0)
    finally:
        signal.signal(signal.SIGINT, old_handler)


class ChartDataset(torch.utils.data.Dataset):
    def __init__(self, files, max_frames=0, slide_vocab=None, max_slide_slots=8,
                 random_crop=True):
        self.files = files
        self.max_frames = max_frames
        self.use_timing_slide = slide_vocab is not None
        self.slide_vocab = slide_vocab or {"<PAD>": 0}
        self.max_slide_slots = max_slide_slots
        self.random_crop = random_crop
        self.diff_map = {
            "Easy": 0,
            "Basic": 1,
            "Advanced": 2,
            "Expert": 3,
            "Master": 4,
            "Re:Master": 5,
            "UTAGE": 6,
        }

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self._load_item(self.files[idx])

    def _load_item(self, path, forced_start: int | None = None, window_frames: int | None = None):
        data = np.load(path)
        meta_path = path.with_suffix(".json")
        meta = json.loads(meta_path.read_text("utf-8")) if meta_path.exists() else {"metadata": {}}
        metadata = meta.get("metadata", {})
        T = data["audio_tokens"].shape[0]
        max_frames = self.max_frames if window_frames is None else int(window_frames)
        frame_labels = None
        if max_frames and max_frames > 0 and T > max_frames:
            if forced_start is not None:
                start = int(max(0, min(T - max_frames, forced_start)))
            elif self.random_crop:
                start = np.random.randint(0, T - max_frames + 1)
            elif self.use_timing_slide:
                frame_labels = self._extract_slide_frame_labels(meta_path)
                if frame_labels:
                    first_slide = min(frame_labels)
                    start = max(0, min(T - max_frames, first_slide - max_frames // 2))
                else:
                    start = 0
            else:
                start = 0
            end = start + max_frames
        else:
            start = 0
            end = T
        tag_ids = data.get("tag_ids", np.array([], dtype=np.int64)).astype(np.int64)
        padded_tags = np.full(32, -1, dtype=np.int64)
        n_tags = min(len(tag_ids), len(padded_tags))
        padded_tags[:n_tags] = tag_ids[:n_tags]
        diff_name = metadata.get("difficulty_name", "Expert")
        slide_path_targets = self._load_slide_targets(
            meta_path=meta_path,
            fallback=data.get("slide_path_targets", np.zeros(T, dtype=np.int32)),
            start=start,
            end=end,
            frame_labels=frame_labels,
        )
        return {
            "audio": data["audio_tokens"][start:end].astype(np.int64),
            "beat": data["beat_signal"][start:end].astype(np.float32),
            "chart": data["chart_tokens"][start:end].astype(np.int64),
            "break_mask": data.get("break_mask", np.zeros(T, dtype=bool))[start:end],
            "ex_mask": data.get("ex_mask", np.zeros(T, dtype=bool))[start:end],
            "object_mask": data.get("object_mask", np.zeros(T, dtype=bool))[start:end],
            "hold_dur_targets": data.get("hold_dur_targets", np.zeros(T, dtype=np.int32))[start:end],
            "slide_path_targets": slide_path_targets,
            "difficulty": np.array(self.diff_map.get(diff_name, 3), dtype=np.int64),
            "level": np.array(float(metadata.get("level", 10.0)), dtype=np.float32),
            "tags": padded_tags,
            "path": str(path),
            "song_id": path.stem,
            "difficulty_name": diff_name,
            "start": int(start),
            "end": int(end),
            "total_frames": int(T),
        }

    def reload_with_window(self, item, start: int, window_frames: int):
        return self._load_item(Path(item["path"]), forced_start=start, window_frames=window_frames)

    def _extract_slide_frame_labels(self, meta_path):
        try:
            from server_pipeline_stage3 import extract_slide_frame_labels_from_preprocessed
            frame_labels, _ = extract_slide_frame_labels_from_preprocessed(meta_path, self.slide_vocab)
        except Exception:
            frame_labels = {}
        return frame_labels

    def _load_slide_targets(self, meta_path, fallback, start, end, frame_labels=None):
        """Build path+timing slide targets from frame_objects for Stage3."""
        if not self.use_timing_slide:
            fallback = fallback[start:end].astype(np.int32)
            if fallback.ndim == 1:
                fallback = fallback[:, None]
            return fallback

        if frame_labels is None:
            frame_labels = self._extract_slide_frame_labels(meta_path)

        length = end - start
        if frame_labels:
            targets = np.zeros((length, self.max_slide_slots), dtype=np.int32)
            for abs_frame, token_ids in frame_labels.items():
                if abs_frame < start or abs_frame >= end:
                    continue
                rel_frame = abs_frame - start
                for slot, tid in enumerate(token_ids[:self.max_slide_slots]):
                    targets[rel_frame, slot] = int(tid)
            return targets

        fallback = fallback[start:end].astype(np.int32)
        if fallback.ndim == 1:
            fallback = fallback[:, None]
        return fallback


def collate(batch):
    max_t = max(item["audio"].shape[0] for item in batch)
    C = max(item["audio"].shape[1] for item in batch)
    B = len(batch)
    max_hold_slots = max(
        item["hold_dur_targets"].shape[1] if item["hold_dur_targets"].ndim == 2 else 1
        for item in batch
    )
    max_slide_slots = max(
        item["slide_path_targets"].shape[1] if item["slide_path_targets"].ndim == 2 else 1
        for item in batch
    )
    max_object_slots = max(
        item["object_mask"].shape[1] if item["object_mask"].ndim == 2 else 1
        for item in batch
    )
    audio_b = torch.zeros(B, max_t, C, dtype=torch.long)
    beat_b = torch.zeros(B, max_t, 2)
    chart_b = torch.zeros(B, max_t, dtype=torch.long)
    brk_b = torch.zeros(B, max_t, max_object_slots, dtype=torch.long)
    ex_b = torch.zeros(B, max_t, max_object_slots, dtype=torch.long)
    obj_b = torch.zeros(B, max_t, max_object_slots, dtype=torch.bool)
    hold_b = torch.zeros(B, max_t, max_hold_slots, dtype=torch.long)
    slide_b = torch.zeros(B, max_t, max_slide_slots, dtype=torch.long)
    valid_b = torch.zeros(B, max_t, dtype=torch.bool)
    diff_b = torch.zeros(B, dtype=torch.long)
    lvl_b = torch.zeros(B)
    tags_b = torch.full((B, 32), -1, dtype=torch.long)
    metas = []
    for i, item in enumerate(batch):
        t = item["audio"].shape[0]
        c = item["audio"].shape[1]
        audio_b[i, :t, :c] = torch.from_numpy(item["audio"])
        beat_b[i,:t] = torch.from_numpy(item["beat"])
        chart_b[i,:t] = torch.from_numpy(item["chart"])
        valid_b[i, :t] = True
        brk_targets = item["break_mask"].astype(np.int64)
        ex_targets = item["ex_mask"].astype(np.int64)
        obj_targets = item["object_mask"].astype(bool)
        if brk_targets.ndim == 1:
            brk_targets = brk_targets[:, None]
        if ex_targets.ndim == 1:
            ex_targets = ex_targets[:, None]
        if obj_targets.ndim == 1:
            obj_targets = obj_targets[:, None]
        brk_b[i, :t, :brk_targets.shape[1]] = torch.from_numpy(brk_targets)
        ex_b[i, :t, :ex_targets.shape[1]] = torch.from_numpy(ex_targets)
        obj_b[i, :t, :obj_targets.shape[1]] = torch.from_numpy(obj_targets)
        hold_targets = item["hold_dur_targets"].astype(np.int64)
        if hold_targets.ndim == 1:
            hold_targets = hold_targets[:, None]
        hold_b[i, :t, :hold_targets.shape[1]] = torch.from_numpy(hold_targets)
        slide_targets = item["slide_path_targets"].astype(np.int64)
        if slide_targets.ndim == 1:
            slide_targets = slide_targets[:, None]
        slide_b[i, :t, :slide_targets.shape[1]] = torch.from_numpy(slide_targets)
        diff_b[i] = int(item["difficulty"])
        lvl_b[i] = float(item["level"])
        tags_b[i] = torch.from_numpy(item["tags"])
        metas.append({
            "path": item.get("path", ""),
            "song_id": item.get("song_id", ""),
            "difficulty_name": item.get("difficulty_name", ""),
            "start": int(item.get("start", 0)),
            "end": int(item.get("end", t)),
            "total_frames": int(item.get("total_frames", t)),
        })
    return audio_b, beat_b, chart_b, brk_b, ex_b, obj_b, hold_b, slide_b, diff_b, lvl_b, tags_b, valid_b, metas


# ═══════════════════════════════════════════════════════════
# Trainer
# ═══════════════════════════════════════════════════════════

class Trainer:
    def __init__(self, cfg, device="cpu"):
        self.cfg = cfg
        self.device = device
        self.data_dir = Path(cfg.preprocess.output_dir)
        self.vocab_dir = Path(getattr(cfg.paths, "vocab_dir", "vocab"))
        self.ckpt_dir = Path(cfg.paths.model_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.vocab = {}
        self.global_step = 0
        self.best_val_loss = {f"stage{i}": float("inf") for i in range(1, 6)}

    def _init_data(self):
        """加载数据 + vocab (在 run 中调用, 确保预处理已完成)"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with open(self.vocab_dir / "vocab.json", "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        slide_vocab_path = self.vocab_dir / "slide_vocab_with_timing.json"
        if not slide_vocab_path.exists():
            slide_vocab_path = self.vocab_dir / "slide_vocab.json"
        if slide_vocab_path.exists():
            with open(slide_vocab_path, "r", encoding="utf-8") as f:
                self.slide_vocab = json.load(f)
        else:
            self.slide_vocab = {"<PAD>": 0}
        tag_vocab_path = self.vocab_dir / "tag_vocab.json"
        if tag_vocab_path.exists():
            with open(tag_vocab_path, "r", encoding="utf-8") as f:
                self.tag_vocab = json.load(f)
        else:
            self.tag_vocab = {}
        # 实际 vocab 大小 (至少 +1 给 0=padding/no-note)
        self.actual_vocab_size = max(self.vocab.values()) + 1
        self.actual_slide_vocab_size = max(self.slide_vocab.values()) + 1
        self.actual_tag_vocab_size = max(self.tag_vocab.values()) + 1 if self.tag_vocab else 1
        log(f"实际 chart vocab 大小: {self.actual_vocab_size} "
            f"(配置: {self.cfg.model.chart_vocab_size})")

    def _build_model(self, stage: int):
        from models.common import StageConfig
        sc = getattr(self.cfg, f"stage{stage}_model", self.cfg.stage_model)
        if getattr(sc, "model_type", "transformer") != "transformer":
            raise ValueError(f"Unsupported model_type for stage{stage}: {sc.model_type}")
        # 使用实际 vocab 大小 (来自预处理), 确保 ≥ 配置值
        chart_vs = max(getattr(self, 'actual_vocab_size', 256),
                       self.cfg.model.chart_vocab_size)
        log(f"  Stage {stage} model: type={getattr(sc, 'model_type', 'transformer')} "
            f"d_model={sc.d_model} n_layer={sc.n_layer} n_head={sc.n_head} "
            f"d_ff={sc.d_ff} slot_n_layer={getattr(sc, 'slot_n_layer', 2)} "
            f"tag_scale={getattr(sc, 'global_tag_scale', 1.0)}/"
            f"{getattr(sc, 'dynamic_tag_scale', 1.0)}")
        mcfg = StageConfig(
            d_model=sc.d_model, n_head=sc.n_head, n_layer=sc.n_layer,
            d_ff=sc.d_ff, dropout=sc.dropout,
            max_seq_len=self.cfg.model.max_seq_len, audio_num_codebooks=self.cfg.audio.num_codebooks,
            tag_vocab_size=max(getattr(self, "actual_tag_vocab_size", 1), 1),
            chart_vocab_size=chart_vs, hold_dur_bins=sc.hold_dur_bins,
            max_hold_slots=getattr(sc, "max_hold_slots", 8),
            max_slide_slots=getattr(sc, "max_slide_slots", 8),
            max_object_slots=getattr(sc, "max_object_slots", 16),
            slot_n_layer=getattr(sc, "slot_n_layer", 2),
            global_tag_scale=getattr(sc, "global_tag_scale", 1.0),
            dynamic_tag_scale=getattr(sc, "dynamic_tag_scale", 1.0),
            slide_vocab_size=max(sc.slide_vocab_size, getattr(self, "actual_slide_vocab_size", 1)),
        )
        if stage == 1:
            from models.stage1_chart import Stage1ChartModel
            return Stage1ChartModel(mcfg).to(self.device)
        elif stage == 2:
            from models.stage2_hold import Stage2HoldModel
            return Stage2HoldModel(mcfg).to(self.device)
        elif stage == 3:
            from models.stage3_slide import Stage3SlideModel
            return Stage3SlideModel(mcfg).to(self.device)
        elif stage == 4:
            from models.stage4_break import Stage4BreakModel
            return Stage4BreakModel(mcfg).to(self.device)
        elif stage == 5:
            from models.stage5_ex import Stage5ExModel
            return Stage5ExModel(mcfg).to(self.device)

    def _load_or_build(self, stage: int):
        last_path = self.ckpt_dir / f"stage{stage}_last.pt"
        best_path = self.ckpt_dir / f"stage{stage}_best.pt"
        path = last_path if last_path.exists() else best_path
        model = self._build_model(stage)
        if path.exists():
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            if "val_loss" in ckpt and path == best_path:
                self.best_val_loss[f"stage{stage}"] = min(
                    self.best_val_loss[f"stage{stage}"],
                    float(ckpt["val_loss"]),
                )
            if "global_step" in ckpt:
                self.global_step = max(self.global_step, int(ckpt["global_step"]))
            state = ckpt.get("model_state_dict", ckpt.get("model", {}))
            loaded, skipped = self._load_compatible_state(model, state)
            msg = f"  Stage {stage}: 从 {path.name} 接续 ({loaded} tensors)"
            if skipped:
                msg += f", 跳过 {len(skipped)} 个 shape 不匹配参数: {', '.join(skipped[:6])}"
                if len(skipped) > 6:
                    msg += ", ..."
            log(msg)
        return model

    def _load_compatible_state(self, model, state: dict) -> tuple[int, list[str]]:
        """Load matching tensors and keep resized heads/embeddings initialized."""
        current = model.state_dict()
        compatible = {}
        skipped = []
        for name, tensor in state.items():
            if name not in current:
                skipped.append(name)
                continue
            if current[name].shape != tensor.shape:
                skipped.append(name)
                continue
            compatible[name] = tensor
        current.update(compatible)
        model.load_state_dict(current)
        return len(compatible), skipped

    def _load_best_losses(self) -> None:
        for stage in range(1, 6):
            path = self.ckpt_dir / f"stage{stage}_best.pt"
            if not path.exists():
                continue
            try:
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
            except Exception:
                continue
            if "val_loss" in ckpt:
                self.best_val_loss[f"stage{stage}"] = float(ckpt["val_loss"])

    def _training_cfg(self, stage: int):
        return getattr(self.cfg, f"stage{stage}_training", self.cfg.training)

    def _build_optimizer(self, model, tcfg=None):
        tcfg = tcfg or self.cfg.training
        opt_name = str(getattr(tcfg, "optimizer", "adamw")).lower()
        lr = float(getattr(tcfg, "learning_rate", 1e-4))
        weight_decay = float(getattr(tcfg, "weight_decay", 0.0))
        betas = tuple(getattr(tcfg, "betas", [0.9, 0.999]))

        if opt_name == "adamw":
            return torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=betas,
            )
        if opt_name == "adam":
            return torch.optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=betas,
            )
        if opt_name == "sgd":
            return torch.optim.SGD(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.9,
            )
        raise ValueError(f"Unsupported optimizer: {opt_name}")

    def _build_scheduler(self, optimizer, total_update_steps: int, tcfg=None):
        tcfg = tcfg or self.cfg.training
        name = str(getattr(tcfg, "scheduler", "cosine")).lower()
        warmup_steps = max(0, int(getattr(tcfg, "warmup_steps", 0)))
        base_lr = float(getattr(tcfg, "learning_rate", 1e-4))
        min_lr = float(getattr(tcfg, "min_learning_rate", 0.0))
        min_ratio = min(max(min_lr / base_lr, 0.0), 1.0) if base_lr > 0 else 0.0
        total_update_steps = max(1, int(total_update_steps))

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return max(min_ratio, float(step + 1) / float(warmup_steps))
            if name == "constant":
                return 1.0
            denom = max(1, total_update_steps - warmup_steps)
            progress = min(1.0, max(0.0, (step - warmup_steps) / denom))
            if name == "linear":
                return min_ratio + (1.0 - min_ratio) * (1.0 - progress)
            if name == "cosine":
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_ratio + (1.0 - min_ratio) * cosine
            raise ValueError(f"Unsupported scheduler: {name}")

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _current_lr(self, optimizer) -> float:
        return float(optimizer.param_groups[0].get("lr", 0.0))

    def compute_loss(self, model, batch, stage: int):
        audio, beat, chart, brk, ex, obj_mask, hold_dur, slide_path, diff, lvl, tags, valid = [
            x.to(self.device) for x in batch[:12]
        ]

        if stage == 1:
            logits = model(audio, beat, diff, lvl, tags, chart_tokens=chart)["logits"]
            return F.cross_entropy(
                logits[valid],
                chart[valid].long(),
            )

        elif stage == 2:
            # Stage2 predicts only hold duration targets, but its input remains
            # the frame-level chart sequence. Token 0 empty frames are kept as
            # context, so the model can learn spacing/density around each hold.
            hold_mask = hold_dur > 0
            if hold_mask.sum() == 0:
                return None
            return model(chart, audio, beat, diff, lvl, tags,
                         hold_dur_targets=hold_dur,
                         hold_mask=hold_mask)["loss"]

        elif stage == 3:
            slide_mask = slide_path > 0
            if slide_mask.sum() == 0:
                return None
            return model(chart, audio, beat, diff, lvl, tags,
                         slide_path_targets=slide_path,
                         slide_mask=slide_mask)["loss"]

        elif stage == 4:
            if obj_mask.sum() == 0:
                return None
            return model(chart, audio, beat, diff, lvl, tags,
                         break_targets=brk,
                         note_mask=obj_mask)["loss"]

        elif stage == 5:
            if ex.sum() == 0:
                return None
            return model(chart, audio, beat, diff, lvl, tags,
                         ex_targets=ex,
                         note_mask=obj_mask)["loss"]

    def _checkpoint_payload(self, model, stage: int, val_loss: float | None = None) -> dict:
        ckpt = {
            "model_state_dict": model.state_dict(),
            "model": model.state_dict(),  # backward-compatible alias
            "config": model.cfg,
            "cfg": model.cfg,             # backward-compatible alias
            "stage": stage,
            "global_step": self.global_step,
        }
        if val_loss is not None:
            ckpt["val_loss"] = val_loss
        if stage == 3:
            ckpt["slide_vocab"] = getattr(self, "slide_vocab", {"<PAD>": 0})
        return ckpt

    def save_best(self, model, stage: int, val_loss: float):
        path = self.ckpt_dir / f"stage{stage}_best.pt"
        ckpt = self._checkpoint_payload(model, stage, val_loss)
        torch.save(ckpt, path)
        self.best_val_loss[f"stage{stage}"] = val_loss
        log(f"  -> saved stage{stage}_best.pt (val_loss={val_loss:.4f})")

    def save_last(self, model, stage: int, val_loss: float | None = None):
        path = self.ckpt_dir / f"stage{stage}_last.pt"
        ckpt = self._checkpoint_payload(model, stage, val_loss)
        torch.save(ckpt, path)
        if val_loss is None:
            log(f"  -> saved stage{stage}_last.pt")
        else:
            log(f"  -> saved stage{stage}_last.pt (val_loss={val_loss:.4f})")

    def _is_cuda_oom(self, exc: BaseException) -> bool:
        if isinstance(exc, getattr(torch.cuda, "OutOfMemoryError", RuntimeError)):
            return True
        msg = str(exc).lower()
        return "cuda" in msg and "out of memory" in msg

    def _batch_meta_summary(self, batch) -> str:
        metas = batch[12] if len(batch) > 12 else []
        if not metas:
            return "unknown batch"
        parts = []
        for meta in metas:
            song_id = meta.get("song_id") or Path(meta.get("path", "")).stem or "unknown"
            diff = meta.get("difficulty_name") or "Unknown"
            parts.append(
                f"{song_id}/{diff} frames={meta.get('start', 0)}:"
                f"{meta.get('end', 0)}/{meta.get('total_frames', 0)}"
            )
        return "; ".join(parts)

    def _oom_retry_batch(self, batch, dataset: ChartDataset, attempt: int):
        retry_max = int(getattr(self.cfg.train_loop, "oom_retry_max_frames", 2048))
        current_frames = int(batch[0].shape[1])
        if retry_max <= 0:
            retry_max = max(1, current_frames // 2)
        target_frames = min(current_frames, retry_max)
        if attempt > 1:
            target_frames = int(target_frames * (0.75 ** (attempt - 1)))
        target_frames = min(current_frames, max(128, target_frames))

        metas = batch[12] if len(batch) > 12 else []
        if not metas:
            return batch

        items = []
        for meta in metas:
            total = int(meta.get("total_frames", current_frames))
            max_start = max(0, total - target_frames)
            start = int(np.random.randint(0, max_start + 1)) if max_start > 0 else 0
            items.append(dataset.reload_with_window(meta, start, target_frames))
        retry_batch = collate(items)
        log(f"  OOM retry {attempt}: new_window={target_frames} | {self._batch_meta_summary(retry_batch)}")
        return retry_batch

    def run(self):
        global _should_stop
        train_files, val_files = load_and_split_data(self.cfg)
        self._init_data()
        self._load_best_losses()
        stage1_max_frames = getattr(self.cfg.train_loop, "max_frames", 0)
        refine_max_frames = getattr(self.cfg.train_loop, "refine_max_frames", 2048)
        start_stage = max(1, min(5, int(getattr(self.cfg.train_loop, "start_stage", 1))))
        train_mode = getattr(self.cfg.train_loop, "mode", "stage_epochs")
        round_robin = train_mode == "round_robin"
        round_num = 0
        max_rounds = self.cfg.train_loop.max_rounds

        while not _should_stop:
            round_num += 1
            log(f"\n{'='*60}")
            log(f"Round {round_num}" + (f"/{max_rounds}" if max_rounds > 0 else ""))
            log(f"Mode: {train_mode}")
            log(f"Stages: {start_stage}-5")
            log(f"{'='*60}")

            for stage in range(start_stage, 6):
                if _should_stop:
                    break
                log(f"\n--- Stage {stage} ---")
                model = self._load_or_build(stage)
                tcfg = self._training_cfg(stage)
                optimizer = self._build_optimizer(model, tcfg)
                stage_max_frames = stage1_max_frames if stage == 1 else refine_max_frames
                log(f"  max_frames={stage_max_frames} ({'full chart' if stage_max_frames == 0 else 'windowed'})")
                max_slide_slots = getattr(getattr(self.cfg, "stage3_model", self.cfg.stage_model), "max_slide_slots", 8)
                batch_size = max(1, int(getattr(tcfg, "batch_size", 2)))
                grad_accum = max(1, int(getattr(tcfg, "gradient_accumulation_steps", 1)))
                grad_clip = float(getattr(tcfg, "grad_clip", 0.0))
                num_workers = max(0, int(getattr(self.cfg.data, "num_workers", 0)))
                val_ds = ChartDataset(
                    val_files,
                    max_frames=stage_max_frames,
                    slide_vocab=self.slide_vocab if stage == 3 else None,
                    max_slide_slots=max_slide_slots,
                    random_crop=False,
                )
                val_loader = torch.utils.data.DataLoader(
                    val_ds,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=collate,
                    num_workers=num_workers,
                    pin_memory=self.device == "cuda",
                )
                self._val_loader = val_loader
                train_ds = ChartDataset(
                    train_files,
                    max_frames=stage_max_frames,
                    slide_vocab=self.slide_vocab if stage == 3 else None,
                    max_slide_slots=max_slide_slots,
                    random_crop=True,
                )
                train_loader = torch.utils.data.DataLoader(
                    train_ds,
                    batch_size=batch_size,
                    shuffle=True,
                    collate_fn=collate,
                    num_workers=num_workers,
                    pin_memory=self.device == "cuda",
                )
                epochs = 1 if round_robin else self.cfg.train_loop.epochs_per_stage
                max_epochs = max(1, int(getattr(tcfg, "max_epochs", epochs)))
                epochs = max(1, min(int(epochs), max_epochs))
                val_interval = max(1, int(self.cfg.train_loop.val_check_interval))
                total_updates = epochs * max(1, math.ceil(len(train_loader) / grad_accum))
                scheduler = self._build_scheduler(optimizer, total_updates, tcfg)
                log_every = max(1, int(getattr(self.cfg.logging, "log_every_steps", 50)))
                log(
                    f"  train: batch={batch_size} accum={grad_accum} "
                    f"effective_batch={batch_size * grad_accum} optimizer={tcfg.optimizer} "
                    f"lr={tcfg.learning_rate:.2e} scheduler={tcfg.scheduler} "
                    f"warmup={tcfg.warmup_steps} min_lr={tcfg.min_learning_rate:.2e} "
                    f"weight_decay={tcfg.weight_decay} grad_clip={grad_clip}"
                )
                t0 = time.time(); steps = 0

                for epoch in range(epochs):
                    if _should_stop: break
                    model.train(); epoch_loss = 0.0; n_steps = 0; t_epoch = time.time()
                    accum_steps = 0
                    optimizer.zero_grad(set_to_none=True)
                    for batch in train_loader:
                        if _should_stop: break
                        loss = None
                        retry_batch = batch
                        max_oom_retries = max(0, int(getattr(self.cfg.train_loop, "oom_retry_attempts", 3)))
                        oom_attempt = 0
                        while True:
                            try:
                                loss = self.compute_loss(model, retry_batch, stage)
                                if loss is not None:
                                    (loss / grad_accum).backward()
                                batch = retry_batch
                                break
                            except RuntimeError as e:
                                if not self._is_cuda_oom(e):
                                    raise
                                oom_attempt += 1
                                log(f"  CUDA OOM at stage={stage}: {self._batch_meta_summary(retry_batch)}")
                                optimizer.zero_grad(set_to_none=True)
                                model.zero_grad(set_to_none=True)
                                accum_steps = 0
                                if self.device == "cuda":
                                    torch.cuda.empty_cache()
                                if oom_attempt > max_oom_retries:
                                    log(f"  OOM retries exhausted ({max_oom_retries}); aborting batch")
                                    raise
                                retry_batch = self._oom_retry_batch(batch, train_ds, oom_attempt)
                        if loss is None: continue
                        epoch_loss += loss.item(); n_steps += 1
                        self.global_step += 1; steps += 1
                        accum_steps += 1
                        if accum_steps % grad_accum == 0:
                            if grad_clip > 0:
                                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad(set_to_none=True)
                        if steps % log_every == 0:
                            avg_l = epoch_loss / max(n_steps, 1)
                            sys_i = _monitor.fmt()
                            log(f"  s{steps:5d} loss={avg_l:.4f} "
                                f"lr={self._current_lr(optimizer):.2e} "
                                f"{n_steps/(time.time()-t_epoch):.1f}stp/s {sys_i}")
                        if steps > 0 and steps % val_interval == 0:
                            model.eval()
                            vloss = self.validate(model, self._val_loader, stage)
                            if vloss < self.best_val_loss[f"stage{stage}"]:
                                self.save_best(model, stage, vloss)
                            model.train()
                    if accum_steps % grad_accum != 0:
                        if grad_clip > 0:
                            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                    avg = epoch_loss / max(n_steps, 1)
                    log(
                        f"  E{epoch+1}/{epochs} loss={avg:.4f} steps={n_steps} "
                        f"lr={self._current_lr(optimizer):.2e} {time.time()-t0:.0f}s"
                    )
                    self.save_last(model, stage)
                    if _should_stop:
                        break

                model.eval()
                vloss = self.validate(model, self._val_loader, stage)
                if vloss < self.best_val_loss[f"stage{stage}"]:
                    self.save_best(model, stage, vloss)
                self.save_last(model, stage, vloss)
                log(f"  Stage {stage} done, val_loss={vloss:.4f}")

            if max_rounds > 0 and round_num >= max_rounds:
                log(f"\n达到 max_rounds={max_rounds}, 停止")
                break

        log(f"\n训练结束, 共 {round_num} 轮, {self.global_step} steps")

    @torch.no_grad()
    def validate(self, model, loader, stage: int):
        model.eval()
        total, n = 0.0, 0
        for batch in loader:
            loss = self.compute_loss(model, batch, stage)
            if loss is not None:
                total += loss.item(); n += 1
        return total / max(n, 1)


# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    sys.path.insert(0, str(Path(__file__).parent))
    from Config import load_config, create_default_config

    parser = argparse.ArgumentParser(description="maiChartGen3 training loop")
    parser.add_argument("--config", type=str, default=None,
                        help="配置文件 (如 server_4090), 默认使用 default")
    parser.add_argument("--start-stage", type=int, default=None,
                        help="Start training from this stage (1-5), overriding config")
    parser.add_argument("--mode", choices=("stage_epochs", "round_robin"), default=None,
                        help="Training loop mode. round_robin trains each stage for one epoch per round.")
    parser.add_argument("--round-robin", action="store_true",
                        help="Shortcut for --mode round_robin")
    args = parser.parse_args()

    if not Path("Config/default.yaml").exists():
        create_default_config()
    cfg = load_config(args.config) if args.config else load_config()
    if args.start_stage is not None:
        if not (1 <= args.start_stage <= 5):
            parser.error("--start-stage must be in [1, 5]")
        cfg.train_loop.start_stage = args.start_stage
    if args.mode is not None:
        cfg.train_loop.mode = args.mode
    if args.round_robin:
        cfg.train_loop.mode = "round_robin"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log_dir = Path(cfg.logging.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    _LOG_FILE = open(log_dir / f"train_{time.strftime('%Y%m%d_%H%M%S')}.log", "w", encoding="utf-8")

    log(f"Device: {device}")
    stage_summaries = []
    for i in range(1, 6):
        sc = getattr(cfg, f"stage{i}_model")
        stage_summaries.append(f"s{i}:d{sc.d_model}/l{sc.n_layer}/h{sc.n_head}")
    log("stage_models=" + " ".join(stage_summaries))
    log(f"mode={cfg.train_loop.mode} start_stage={cfg.train_loop.start_stage} "
        f"epochs/stage={cfg.train_loop.epochs_per_stage} val_interval={cfg.train_loop.val_check_interval}")
    log(f"ckpt_dir={cfg.paths.model_dir}")

    try:
        Trainer(cfg, device).run()
    except Exception as e:
        log(f"FATAL: {e}")
        import traceback; traceback.print_exc()
    finally:
        if _LOG_FILE:
            _LOG_FILE.close()
