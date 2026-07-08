from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import csv
import ctypes
import gc
import logging
import platform
import time
from dataclasses import dataclass, field
from datetime import datetime
from itertools import cycle
from pathlib import Path
from typing import Any, Callable

import torch

from .checkpoint import load_checkpoint, pack_config, save_checkpoint, _move_optimizer_state
from .recipes import SkipBatchError

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

try:
    import psutil
    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False


def get_cpu_memory_gb() -> float:
    """返回当前进程 RSS（GB）。"""
    if _HAS_PSUTIL:
        return psutil.Process(os.getpid()).memory_info().rss / (1024**3)
    return -1.0


def _malloc_trim() -> None:
    """Linux: 归还 glibc malloc arena 给 OS；非 Linux 无操作。"""
    if platform.system() != "Linux":
        return
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _print_oom_batch(batch: dict[str, Any], stage_name: str, prev_batch: dict[str, Any] | None = None) -> None:
    """打印 OOM 时的 batch 信息以辅助定位"""

    def _describe(b: dict[str, Any] | None, label: str) -> str:
        if b is None:
            return f"  {label}: None"
        f = b.get("_file", "?")
        tok = b.get("tokens", b.get("config_tokens", b.get("target_path")))
        tl = tok.size(-1) if tok is not None and tok.dim() >= 1 else 0  # 最后一个维度是序列长度
        onset = b.get("onset")
        ol = onset.size(-1) if onset is not None and onset.dim() >= 1 else 0
        atok = b.get("audio_tokens")
        al = atok.size(-1) if atok is not None and atok.dim() >= 1 else 0
        return f"  {label}: {f}  tok={tl}  onset={ol}  enc_tok={al}"

    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        mem_line = f"  GPU: {alloc:.2f}/{total:.2f} GiB allocated, {reserved:.2f} GiB reserved"
        summary = torch.cuda.memory_summary()
    else:
        mem_line = "  GPU: N/A"
        summary = ""

    lines = [
        "=" * 60,
        f"  💥 OOM in stage '{stage_name}'",
        _describe(batch, "当前 batch"),
        _describe(prev_batch, "前一 batch (疑凶)"),
        mem_line,
        "=" * 60,
    ]
    if summary:
        lines.append(summary)
    print("\n".join(lines), flush=True)


@dataclass
class StageRuntime:
    name: str
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: Any
    train_loader: Any
    val_loader: Any | None
    step_fn: Callable[[torch.nn.Module, dict[str, Any], torch.device], tuple[torch.Tensor, dict[str, float]]]
    val_fn: Callable[[torch.nn.Module, dict[str, Any], torch.device], dict[str, float]] | None = None
    turn_batches: int = 1
    grad_accum_steps: int = 1
    offload_to_cpu: bool = False
    iterator: Any = field(default=None, init=False)
    steps_done: int = field(default=0, init=False)
    turns_done: int = field(default=0, init=False)


class RotatingMultiStageTrainer:
    def __init__(
        self,
        stages: list[StageRuntime],
        device: str | torch.device = "cuda",
        grad_clip_norm: float = 1.0,
        precision: str = "amp",
        log_every: int = 20,
        eval_every_turns: int = 1,
        save_every_turns: int = 1,
        checkpoint_dir: str | Path = "runs/rotating",
        keep_last: int = 3,
        best_metric: str = "loss",
        best_mode: str = "min",
        resume_path: str | Path | None = None,
        cfg: Any | None = None,
    ):
        self.stages = stages
        self.device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
        self.grad_clip_norm = grad_clip_norm
        self.precision = precision
        self.log_every = log_every
        self.eval_every_turns = eval_every_turns
        self.save_every_turns = int(save_every_turns)  # 0 = 仅 epoch 时保存
        self.checkpoint_dir = Path(checkpoint_dir)
        self.keep_last = keep_last
        self.best_metric = best_metric
        self.best_mode = best_mode
        self.best_score = float("inf") if best_mode == "min" else float("-inf")
        self.last_checkpoint: Path | None = None
        self.best_checkpoint: Path | None = None
        self.global_step = 0
        self.global_turn = 0
        self.cfg = cfg
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.precision == "amp" and self.device.type == "cuda"))
        self.resume_path = Path(resume_path) if resume_path else None
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 异步保存：允许新保存覆盖旧保存

        # 训练日志
        self._log_path = self.checkpoint_dir / "training_log.csv"
        self._log_file = open(self._log_path, "a", newline="")
        self._log_writer = csv.writer(self._log_file)
        if self._log_path.stat().st_size == 0:
            self._log_writer.writerow(["timestamp", "turn", "step", "stage", "loss", "ppl", "metric", "save"])

        # ── 性能 timing 日志 ──
        self._time_path = self.checkpoint_dir / "timing_log.csv"
        self._time_file = open(self._time_path, "a", newline="")
        self._time_writer = csv.writer(self._time_file)
        if self._time_path.stat().st_size == 0:
            self._time_writer.writerow([
                "timestamp", "stage", "turn", "batch",
                "data_ms", "forward_ms", "backward_ms", "optim_ms",
                "total_ms", "tokens", "seq_len", "loss",
            ])

        for stage in self.stages:
            stage.iterator = cycle(stage.train_loader)
            if not stage.offload_to_cpu:
                stage.model.to(self.device)
                _move_optimizer_state(stage.optimizer, self.device)

        self._compile_models = cfg.get("compile", False) if cfg else False

        # ── CPU 内存上限（触发强制清理 + 保存检查点）──
        train_cfg = cfg.get("train", {}) if cfg else {}
        self._max_cpu_memory_gb = float(train_cfg.get("max_cpu_memory_gb", 0) or 0)
        self._mem_check_every = int(train_cfg.get("mem_check_every_turns", 50))
        self._mem_check_counter = 0

        # 打印混合精度配置
        self._log_precision()

    def _log_precision(self) -> None:
        """打印混合精度配置信息"""
        amp_enabled = self.device.type == "cuda" and self.precision in {"amp", "bf16"}
        scaler_on = self.scaler.is_enabled()
        dtype_name = "bfloat16" if self.precision == "bf16" else ("float16" if self.precision == "amp" else "float32")
        model_params = sum(p.numel() for stage in self.stages for p in stage.model.parameters())
        print(f"  精度模式: {dtype_name}  |  autocast={'ON' if amp_enabled else 'OFF'}  |  "
              f"GradScaler={'ON' if scaler_on else 'OFF'}  |  "
              f"总参数 {model_params/1e6:.1f}M  |  "
              f"grad_accum: {self.stages[0].grad_accum_steps if self.stages else 1}")

    def _maybe_offload(self, stage: StageRuntime, to_device: bool) -> None:
        if not stage.offload_to_cpu:
            return
        target = self.device if to_device else torch.device("cpu")
        stage.model.to(target)
        _move_optimizer_state(stage.optimizer, target)

    def _force_memory_cleanup(self, reason: str = "") -> float:
        """强制清理 CPU/GPU 内存碎片，返回当前 RSS（GB）。"""
        gc.collect()
        gc.collect()  # 两遍确保循环引用被回收
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        _malloc_trim()
        mem_gb = get_cpu_memory_gb()
        if reason:
            logger = logging.getLogger(__name__)
            logger.warning(
                "内存清理 [%s] RSS=%.2f GB (turn=%d step=%d)",
                reason, mem_gb, self.global_turn, self.global_step,
            )
        return mem_gb

    def _train_turn(self, stage: StageRuntime) -> dict[str, float]:
        self._maybe_offload(stage, True)
        stage.model.train()
        stats: dict[str, float] = {}
        accum_steps = max(1, stage.grad_accum_steps)
        stage.optimizer.zero_grad(set_to_none=True)
        did_backward = False
        did_scaled_backward = False
        did_step = False
        inner_pbar = None
        if tqdm is not None and stage.turn_batches > 1:
            inner_pbar = tqdm(total=stage.turn_batches, desc=f"{stage.name}", leave=False, dynamic_ncols=True)

        t_turn_start = time.perf_counter()

        for batch_idx in range(stage.turn_batches):
            # ── 数据加载计时 ──
            t_data_start = time.perf_counter()
            batch = next(stage.iterator)
            t_data = (time.perf_counter() - t_data_start) * 1000

            # ── Forward 计时 ──
            use_amp = self.device.type == "cuda" and self.precision in {"amp", "bf16"}
            amp_dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16

            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t_fwd_start = time.perf_counter()
            try:
                with torch.autocast(device_type=self.device.type, enabled=use_amp, dtype=amp_dtype):
                    loss, step_stats = stage.step_fn(stage.model, batch, self.device)
            except SkipBatchError as e:
                logging.warning("skip batch in stage '%s': %s", stage.name, e)
                if inner_pbar is not None:
                    inner_pbar.update(1)
                    inner_pbar.set_postfix(skip="1")
                self._last_batch = batch
                continue
            except torch.OutOfMemoryError:
                _print_oom_batch(batch, stage.name, getattr(self, "_last_batch", None))
                raise
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t_forward = (time.perf_counter() - t_fwd_start) * 1000

            if not torch.isfinite(loss):
                stats["nan_loss"] = float(loss.detach().item()) if torch.is_tensor(loss) else float("nan")
                continue

            loss = loss / max(1, stage.turn_batches * accum_steps)

            # ── Backward 计时 ──
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t_bwd_start = time.perf_counter()
            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                did_scaled_backward = True
            else:
                loss.backward()
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t_backward = (time.perf_counter() - t_bwd_start) * 1000

            did_backward = True
            stats.update(step_stats)
            self.global_step += 1

            # ── Optimizer step 计时 ──
            t_optim = 0.0
            if self.global_step % accum_steps == 0:
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                t_opt_start = time.perf_counter()
                did_step = self._step_if_ready(stage, did_scaled_backward)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                t_optim = (time.perf_counter() - t_opt_start) * 1000
                did_backward = False
                did_scaled_backward = False
                # 每次 optimizer step 后释放 CUDA 缓存，避免 turn 内碎片化 OOM
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

            # ── 记录 timing ──
            tokens = batch.get("tokens")
            seq_len = tokens.size(-1) if tokens is not None and tokens.dim() >= 1 else 0
            self._time_writer.writerow([
                datetime.now().isoformat(), stage.name, self.global_turn, batch_idx,
                f"{t_data:.1f}", f"{t_forward:.1f}", f"{t_backward:.1f}", f"{t_optim:.1f}",
                f"{t_data + t_forward + t_backward + t_optim:.1f}",
                tokens.numel() if tokens is not None else 0, seq_len,
                f"{float(loss.detach().item()):.6f}",
            ])

            if inner_pbar is not None:
                inner_pbar.update(1)
                inner_pbar.set_postfix(loss=f"{float(loss.detach().item()):.4f}")

            # 记录当前 batch 作为"上一批"，OOM 时定位疑凶
            self._last_batch = batch

        # 末尾未 step 的补刀
        if did_backward:
            did_step = self._step_if_ready(stage, did_scaled_backward)
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        if inner_pbar is not None:
            inner_pbar.close()

        if stage.scheduler is not None and did_step:
            stage.scheduler.step()
        stage.steps_done += stage.turn_batches
        stage.turns_done += 1
        self.global_turn += 1
        self._maybe_offload(stage, False)

        # 每 turn flush + 定期清理内存碎片
        self._time_file.flush()
        if self.device.type == "cuda" and self.global_turn % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()  # CPU 侧：回收 torch.load / 切片产生的碎片化内存

        return stats

    def _step_if_ready(self, stage: StageRuntime, did_scaled_backward: bool) -> bool:
        has_grad = any(p.grad is not None for p in stage.model.parameters())
        if not has_grad:
            stage.optimizer.zero_grad(set_to_none=True)
            return False
        if self.grad_clip_norm is not None and self.scaler.is_enabled() and did_scaled_backward:
            self.scaler.unscale_(stage.optimizer)
        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(stage.model.parameters(), self.grad_clip_norm)
        if self.scaler.is_enabled():
            self.scaler.step(stage.optimizer)
            self.scaler.update()
        else:
            stage.optimizer.step()
        stage.optimizer.zero_grad(set_to_none=True)
        return True

    def _should_improve(self, value: float) -> bool:
        if self.best_mode == "min":
            return value < self.best_score
        return value > self.best_score

    def _update_best(self, value: float) -> bool:
        if self._should_improve(value):
            self.best_score = value
            self._pending_best = True  # 延迟到 save 周期
            return True
        return False

    def _checkpoint_name(self, stage: StageRuntime, kind: str) -> str:
        if len(self.stages) == 1:
            return f"{stage.name}/{kind}.pt"
        return f"{kind}.pt"

    @torch.no_grad()
    def _validate(self, stage: StageRuntime) -> dict[str, float]:
        if stage.val_loader is None or stage.val_fn is None:
            return {}
        self._maybe_offload(stage, True)
        stage.model.eval()
        metrics: dict[str, float] = {}
        count = 0
        for batch in stage.val_loader:
            try:
                batch_metrics = stage.val_fn(stage.model, batch, self.device)
            except SkipBatchError as e:
                logging.warning("skip val batch in stage '%s': %s", stage.name, e)
                continue
            for k, v in batch_metrics.items():
                metrics[k] = metrics.get(k, 0.0) + float(v)
            count += 1
        if count:
            metrics = {k: v / count for k, v in metrics.items()}
        self._maybe_offload(stage, False)
        return metrics

    def save(self, name: str = "last.pt") -> Path:
        """保存 checkpoint，返回保存路径"""
        path = self.checkpoint_dir / name

        # 构建 payload：全部搬到 CPU，避免保存时持有 GPU tensor。
        payload: dict[str, Any] = {
            "global_step": self.global_step,
            "global_turn": self.global_turn,
            "cfg": pack_config(self.cfg),
            "stages": [],
        }
        for stage in self.stages:
            sd = stage.model.state_dict()
            sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
            sd_cpu = {k: v.detach().cpu() for k, v in sd.items()}

            raw_opt = stage.optimizer.state_dict()
            opt_cpu: dict[str, Any] = {}
            for ok, ov in raw_opt.items():
                if ok == "state":
                    opt_cpu[ok] = {
                        idx: {
                            sk: sv.detach().cpu() if torch.is_tensor(sv) else sv
                            for sk, sv in s.items()
                        }
                        for idx, s in ov.items()
                    }
                elif ok == "param_groups":
                    opt_cpu[ok] = [
                        {
                            pk: pv.detach().cpu() if torch.is_tensor(pv) else pv
                            for pk, pv in pg.items()
                        }
                        for pg in ov
                    ]
                else:
                    opt_cpu[ok] = ov

            payload["stages"].append({
                "name": stage.name,
                "model": sd_cpu,
                "optimizer": opt_cpu,
                "scheduler": stage.scheduler.state_dict() if stage.scheduler is not None else None,
                "steps_done": stage.steps_done,
                "turns_done": stage.turns_done,
            })

        save_checkpoint(path, payload)
        return path

    def load(self, path: str | Path, restore_progress: bool = True) -> None:
        payload = load_checkpoint(path, map_location="cpu")
        if restore_progress:
            self.global_step = int(payload.get("global_step", 0))
            self.global_turn = int(payload.get("global_turn", 0))
        stage_payloads = {s["name"]: s for s in payload.get("stages", [])}
        for stage in self.stages:
            if stage.name not in stage_payloads:
                continue
            sp = stage_payloads[stage.name]
            stage.model.load_state_dict(sp["model"], strict=False)
            stage.optimizer.load_state_dict(sp["optimizer"])
            if stage.scheduler is not None and sp.get("scheduler") is not None:
                stage.scheduler.load_state_dict(sp["scheduler"])
            if restore_progress:
                stage.steps_done = int(sp.get("steps_done", 0))
                stage.turns_done = int(sp.get("turns_done", 0))

    def fit(self, max_turns: int | None = None, max_steps: int | None = None, max_epochs: int | None = None) -> None:
        # torch.compile 必须在 load 之后调用
        if getattr(self, "_compile_models", False):
            import logging
            logging.info("启用 torch.compile ...")
            for stage in self.stages:
                try:
                    stage.model = torch.compile(stage.model, dynamic=True)
                except Exception:
                    pass

        # ── Epoch 跟踪 ──
        stage_epoch: dict[str, int] = {s.name: 0 for s in self.stages}
        stage_samples_seen: dict[str, int] = {s.name: 0 for s in self.stages}
        stage_dataset_size: dict[str, int] = {
            s.name: len(s.train_loader.dataset) for s in self.stages
        }

        stage_cycle = cycle(self.stages)
        pbar = None
        # pbar: epoch 模式下按样本数显示，turn 模式下按 turn 数显示
        if tqdm is not None:
            if max_epochs is not None:
                total_samples = sum(stage_dataset_size.values()) * max(1, int(max_epochs))
                pbar = tqdm(total=total_samples, initial=0, desc="epoch", unit="samp", dynamic_ncols=True)
            elif max_turns is not None:
                pbar = tqdm(total=max_turns, initial=self.global_turn, desc="turns", dynamic_ncols=True)

        while True:
            if max_turns is not None and self.global_turn >= max_turns:
                break
            if max_steps is not None and self.global_step >= max_steps:
                break
            if max_epochs is not None:
                if all(e >= max_epochs for e in stage_epoch.values()):
                    break

            stage = next(stage_cycle)
            stats = self._train_turn(stage)

            # ── CPU 内存上限检查：超阈值时强制清理 + 保存检查点 ──
            if self._max_cpu_memory_gb > 0:
                self._mem_check_counter += 1
                if self._mem_check_counter >= self._mem_check_every:
                    self._mem_check_counter = 0
                    mem_now = get_cpu_memory_gb()
                    if mem_now > self._max_cpu_memory_gb:
                        # 1. 获取 dataset 引用并清除内部缓存
                        ds = stage.train_loader.dataset
                        for attr in ("_hidden_cache", "_slide_audio_cache", "_onset_cache"):
                            c = getattr(ds, attr, None)
                            if isinstance(c, dict):
                                c.clear()

                        # 2. 先建新的 DataLoader，再替换旧的（避免中间态）
                        from .data import build_loader as _rebuild_loader
                        data_cfg = self.cfg.get("data", {}) if self.cfg else {}
                        new_loader = _rebuild_loader(
                            ds,
                            batch_size=int(self.cfg.get("train", {}).get("batch_size", 1)) if self.cfg else 1,
                            shuffle=True,
                            num_workers=data_cfg.get("num_workers", 0),
                            prefetch_factor=data_cfg.get("prefetch_factor", 2),
                        )
                        old_loader = stage.train_loader
                        stage.train_loader = new_loader
                        stage.iterator = cycle(new_loader)
                        del old_loader  # 释放旧 loader → 终止旧 worker 进程

                        # 3. 强制 CPU 内存回收
                        self._force_memory_cleanup(f"超过上限 {mem_now:.1f}GB > {self._max_cpu_memory_gb:.0f}GB, 重建 DataLoader")
                        self._last_batch = None

                        # 4. 保存检查点
                        self.last_checkpoint = self.save(self._checkpoint_name(stage, "memcap"))
                        mem_after = get_cpu_memory_gb()
                        print(f"[memcap] RSS {mem_now:.1f} → {mem_after:.1f}GB (已重建 DataLoader), 保存至 {self.last_checkpoint}")

            # ── Epoch 跟踪 ──
            epoch_completed = False
            samples_this_turn = stage.turn_batches * max(1, int(
                self.cfg.get("train", {}).get("batch_size", 1)
            )) if self.cfg else stage.turn_batches
            if stage.name in stage_samples_seen:
                stage_samples_seen[stage.name] += samples_this_turn
                ds_size = stage_dataset_size.get(stage.name, 1)
                if ds_size > 0 and stage_samples_seen[stage.name] >= ds_size:
                    stage_epoch[stage.name] += 1
                    stage_samples_seen[stage.name] %= ds_size
                    stage.iterator = cycle(stage.train_loader)
                    epoch_completed = True

            train_metric_value = stats.get(self.best_metric)

            if pbar is not None:
                if max_epochs is not None:
                    # epoch 模式：按当前 epoch 内的样本进度更新
                    ep_done = sum(
                        stage_dataset_size.get(s.name, 0) * stage_epoch.get(s.name, 0)
                        for s in self.stages
                    )
                    ep_current = sum(stage_samples_seen.values())
                    pbar.n = min(ep_done + ep_current, pbar.total)
                else:
                    pbar.update(1)
                loss_value = float(stats.get("loss", float("nan")))
                postfix = {
                    "stage": stage.name,
                    "loss": f"{loss_value:.4f}",
                    "ppl": f"{torch.exp(torch.tensor(loss_value)).item():.2f}" if torch.isfinite(torch.tensor(loss_value)) else "nan",
                }
                if stage.name in stage_epoch:
                    postfix["ep"] = f"{stage_epoch[stage.name]}/{max_epochs}" if max_epochs else str(stage_epoch[stage.name])
                pbar.set_postfix(postfix)

            # ── 日志 ──
            if self.global_turn % self.log_every == 0:
                msg = ", ".join(f"{k}={v:.4f}" for k, v in stats.items())
                ep_str = f"ep={stage_epoch.get(stage.name, 0)}" if stage.name in stage_epoch else ""
                print(f"[turn {self.global_turn} | step {self.global_step} | {ep_str}] {stage.name}: {msg}")
                loss_val = float(stats.get("loss", float("nan")))
                ppl_val = torch.exp(torch.tensor(loss_val)).item() if torch.isfinite(torch.tensor(loss_val)) else float("nan")
                self._log_writer.writerow([
                    datetime.now().isoformat(), self.global_turn, self.global_step,
                    stage.name, f"{loss_val:.6f}", f"{ppl_val:.4f}",
                    f"{train_metric_value:.6f}" if train_metric_value is not None else "",
                    "",
                ])
                self._log_file.flush()

            # ── 验证 + Best Model 判断 ──
            # 触发条件: epoch 完成时必然验证，或在 eval_every_turns 周期
            best_updated = False
            periodic_eval = (
                self.eval_every_turns > 0
                and self.global_turn % self.eval_every_turns == 0
            )
            should_validate = epoch_completed or periodic_eval
            if should_validate:
                val = self._validate(stage)
                if val:
                    msg = ", ".join(f"{k}={v:.4f}" for k, v in val.items())
                    tag = "epoch" if epoch_completed else "val"
                    print(f"[{tag} {stage_epoch.get(stage.name, 0)} | turn {self.global_turn} | step {self.global_step}] {stage.name}: {msg}")
                    # 用验证集指标判断 best
                    val_metric = val.get("val_loss", val.get(self.best_metric))
                    if val_metric is not None and torch.isfinite(torch.tensor(val_metric)):
                        best_updated = self._update_best(float(val_metric))
                elif epoch_completed:
                    # epoch 完成但无验证集：回退用训练 loss
                    if train_metric_value is not None and torch.isfinite(torch.tensor(train_metric_value)):
                        best_updated = self._update_best(float(train_metric_value))

            # ── 保存 ──
            periodic_save = (
                self.save_every_turns > 0
                and self.global_turn % self.save_every_turns == 0
            )
            save_this_turn = (
                epoch_completed
                or periodic_save
                or self.global_turn == max_turns
            )
            if save_this_turn:
                if getattr(self, "_pending_best", False):
                    self.best_checkpoint = self.save(self._checkpoint_name(stage, "best"))
                    self._pending_best = False
                    if epoch_completed:
                        ep = stage_epoch.get(stage.name, 0)
                        print(f"[epoch {ep}] best saved: {self.best_checkpoint}")
                self.last_checkpoint = self.save(self._checkpoint_name(stage, "last"))

        if pbar is not None:
            pbar.close()
        self._wait_saves()
        self._log_file.flush()
        self._time_file.flush()
        self._log_file.close()
        self._time_file.close()

    def _wait_saves(self) -> None:
        """等待异步保存完成"""
        return
