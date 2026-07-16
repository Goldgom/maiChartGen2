#!/usr/bin/env python3
"""
batch_infer.py — 批量推理脚本

对指定文件夹下所有 mp3/mp4 文件批量生成 maimai 谱面。
每首歌在输出目录下创建独立子文件夹，包含:
  - track.mp3     (音频)
  - maidata.txt   (谱面, 可含多个难度)
  - pv.mp4        (视频, 可选)
  - bg.png        (封面, 可选)

配置位于主配置文件的 batch_infer 段 (Config/default.yaml / Config/server_4090.yaml):
  - 输入/输出路径
  - 难度列表和等级
  - 生成参数 (温度、偏置等)
  - 输出选项 (音频转换、视频处理、封面提取)

用法:
  python batch_infer.py                          # 使用 Config/default.yaml
  python batch_infer.py --config server_4090     # 使用 Config/server_4090.yaml
  python batch_infer.py --input_dir /path/to/mp3  # 命令行覆盖输入目录
  python batch_infer.py --dry_run                 # 仅预览文件列表
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# 配置加载 — 从主配置系统读取
# ============================================================

def load_batch_config(config_name: str | None = None,
                      cli_overrides: dict | None = None) -> dict:
    """从主配置系统加载配置，转为批量推理所需的字典格式。

    优先级: CLI覆盖 > 用户YAML (batch_infer段) > 主配置默认值

    返回的字典结构与旧版 standalone batch_infer.yaml 兼容。
    """
    from Config import load_config as load_main_config

    # 不指定配置时使用 default；显式指定的配置如果有问题，直接暴露错误，
    # 避免把 server_4090 拼错后静默回退到 default。
    main_cfg = load_main_config(config_name, use_default=True)

    bi = main_cfg.batch_infer
    gen = main_cfg.generation

    # ── 标准化难度列表 (兼容旧格式) ──
    raw_diffs = bi.difficulties
    normalized_diffs: list[dict] = []
    for item in raw_diffs:
        if isinstance(item, str):
            # 旧格式: 纯字符串列表, 从 default_levels 取等级 (已废弃但兼容)
            normalized_diffs.append({"name": item, "level": 13.0})
        elif isinstance(item, dict):
            if "name" not in item:
                raise ValueError(f"batch_infer.difficulties 项缺少 name: {item}")
            normalized_diffs.append(dict(item))  # 复制一份
        else:
            raise ValueError(f"batch_infer.difficulties 不支持的项: {item!r}")

    diff_names = [d["name"] for d in normalized_diffs]
    diff_levels = {d["name"]: d.get("level", 13.0) for d in normalized_diffs}
    # 提取各难度的独立参数覆盖
    _OVERRIDABLE_KEYS = [
        "temperature", "top_k", "bpm_override",
        "density", "tap_bias", "hold_bias", "slide_bias", "wifi_bias",
        "touch_bias", "touchhold_bias", "break_bias",
        "filter_multi_tap", "allow_touch", "beat_method", "skip_stages",
    ]
    diff_params: dict[str, dict] = {}
    for d in normalized_diffs:
        overrides = {}
        for k in _OVERRIDABLE_KEYS:
            if k in d:
                overrides[k] = _normalize_skip_stages(d[k]) if k == "skip_stages" else d[k]
        diff_params[d["name"]] = overrides

    # 构建兼容字典
    cfg = {
        "batch": {
            "input_dir": bi.input_dir,
            "output_dir": bi.output_dir,
            "audio_extensions": bi.audio_extensions,
            "video_extensions": bi.video_extensions,
            "output_subdir_template": bi.output_subdir_template,
        },
        "model": {
            "data_dir": main_cfg.preprocess.output_dir,
            "ckpt_dir": main_cfg.paths.model_dir,
            "device": main_cfg.audio.device,
            "audio_codebooks": main_cfg.audio.num_codebooks,
            "premodel_path": main_cfg.audio.premodel_path,
            "target_subdiv": main_cfg.chart.target_subdiv,
            "max_tags": getattr(main_cfg.stage_model, "max_tags", 32),
        },
        "beat": {
            "method": main_cfg.beat.method,
            "bpm_min": main_cfg.beat.bpm_min,
            "bpm_max": main_cfg.beat.bpm_max,
            "tightness": main_cfg.beat.tightness,
            "time_signature": main_cfg.beat.time_signature,
            "quantize_beats": main_cfg.beat.quantize_beats,
        },
        "difficulties": {
            "enabled": diff_names,
            "default_levels": diff_levels,
            "params": diff_params,        # 各难度的独立参数覆盖
        },
        "generation": {
            "designer": bi.designer,
            "collections": bi.collections,
            "temperature": gen.temperature,
            "top_k": getattr(bi, "top_k", gen.top_k),
            "bpm_override": bi.bpm_override,
            "density": bi.density,
            "tap_bias": bi.tap_bias,
            "hold_bias": bi.hold_bias,
            "slide_bias": bi.slide_bias,
            "wifi_bias": bi.wifi_bias,
            "touch_bias": bi.touch_bias,
            "touchhold_bias": bi.touchhold_bias,
            "break_bias": bi.break_bias,
            "filter_multi_tap": bi.filter_multi_tap,
            "allow_touch": bi.allow_touch,
            "beat_method": bi.beat_method,
            "skip_stages": _normalize_skip_stages(bi.skip_stages),
        },
        "output": {
            "copy_audio": bi.copy_audio,
            "audio_format": bi.audio_format,
            "audio_bitrate": bi.audio_bitrate,
            "copy_video": bi.copy_video,
            "extract_bg": bi.extract_bg,
            "bg_max_size": bi.bg_max_size,
            "skip_existing": bi.skip_existing,
        },
        "logging": {
            "log_file": main_cfg.logging.log_dir + "/batch_infer.log",
            "verbose": True,
        },
    }

    # CLI 覆盖
    if cli_overrides:
        cfg = _deep_merge(cfg, cli_overrides)
        cfg["generation"]["skip_stages"] = _normalize_skip_stages(
            cfg["generation"].get("skip_stages")
        )
        for params in cfg["difficulties"].get("params", {}).values():
            if "skip_stages" in params:
                params["skip_stages"] = _normalize_skip_stages(params["skip_stages"])

    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并字典, override 覆盖 base"""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            base[k] = _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _normalize_skip_stages(skip_stages: list | tuple | set | str | None) -> list[str]:
    """Normalize stage skip config to canonical labels such as 'Stage 5'."""
    if skip_stages is None:
        return []
    if isinstance(skip_stages, str):
        raw_items = re.split(r"[,，]", skip_stages)
    else:
        raw_items = list(skip_stages)

    normalized = []
    for item in raw_items:
        text = str(item).strip()
        if not text:
            continue
        match = re.search(r"([1-5])", text)
        if match:
            label = f"Stage {match.group(1)}"
            if label not in normalized:
                normalized.append(label)
    return normalized


# ============================================================
# 日志
# ============================================================

class Logger:
    def __init__(self, log_file: str | None = None, verbose: bool = True):
        self.verbose = verbose
        self.log_file = None
        if log_file:
            try:
                Path(log_file).parent.mkdir(parents=True, exist_ok=True)
                self.log_file = open(log_file, "a", encoding="utf-8")
            except OSError as e:
                if self.verbose:
                    print(f"[WARN] 日志文件不可写，改为仅输出到控制台: {log_file} ({e})",
                          flush=True)

    def _write(self, msg: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {msg}"
        if self.verbose:
            print(line, flush=True)
        if self.log_file:
            self.log_file.write(line + "\n")
            self.log_file.flush()

    def info(self, msg: str):
        self._write(f"INFO  | {msg}")

    def warn(self, msg: str):
        self._write(f"WARN  | {msg}")

    def error(self, msg: str):
        self._write(f"ERROR | {msg}")

    def success(self, msg: str):
        self._write(f"OK    | {msg}")

    def close(self):
        if self.log_file:
            self.log_file.close()


# ============================================================
# 文件工具
# ============================================================

def find_ffmpeg() -> str:
    """查找 ffmpeg 可执行文件"""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    # 常见路径
    for p in ["/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"]:
        if Path(p).exists():
            return p
    return "ffmpeg"


def sanitize_filename(name: str) -> str:
    """清理文件名, 移除非法字符"""
    return re.sub(r'[<>:"/\\|?*]', '_', name).strip()


def extract_audio_from_video(video_path: Path, output_path: Path,
                              fmt: str = "mp3", bitrate: str = "192k",
                              logger: Logger | None = None) -> bool:
    """使用 ffmpeg 从视频提取音频"""
    ffmpeg = find_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y", "-i", str(video_path),
        "-vn", "-acodec", "libmp3lame" if fmt == "mp3" else "aac",
        "-b:a", bitrate,
        str(output_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            if logger:
                logger.warn(f"ffmpeg 音频提取失败: {result.stderr[:200]}")
            return False
        return output_path.exists()
    except Exception as e:
        if logger:
            logger.warn(f"ffmpeg 异常: {e}")
        return False


def extract_first_frame(video_path: Path, output_path: Path,
                         max_size: int = 512,
                         logger: Logger | None = None) -> bool:
    """使用 ffmpeg 提取视频第一帧为 PNG"""
    ffmpeg = find_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 先提取原始帧
    temp_path = output_path.with_suffix(".temp.png")
    cmd = [
        ffmpeg, "-y", "-i", str(video_path),
        "-vframes", "1", "-q:v", "2",
        str(temp_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            if logger:
                logger.warn(f"ffmpeg 提取帧失败: {result.stderr[:200]}")
            return False

        if max_size > 0:
            # 缩放到 max_size
            cmd2 = [
                ffmpeg, "-y", "-i", str(temp_path),
                "-vf", f"scale='min({max_size},iw)':'min({max_size},ih)':force_original_aspect_ratio=decrease",
                str(output_path),
            ]
            subprocess.run(cmd2, capture_output=True, text=True, timeout=30)
            temp_path.unlink(missing_ok=True)
        else:
            shutil.move(str(temp_path), str(output_path))

        return output_path.exists()
    except Exception as e:
        if logger:
            logger.warn(f"ffmpeg 提取帧异常: {e}")
        return False


def copy_or_convert_audio(src_path: Path, dst_path: Path,
                           fmt: str = "mp3", bitrate: str = "192k",
                           logger: Logger | None = None) -> bool:
    """复制或转换音频文件到目标路径"""
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    src_ext = src_path.suffix.lower()
    # 同格式直接复制
    if src_ext == f".{fmt}":
        try:
            shutil.copy2(src_path, dst_path)
            return True
        except Exception as e:
            if logger:
                logger.warn(f"复制音频失败: {e}")
            return False

    # 不同格式用 ffmpeg 转换
    return extract_audio_from_video(src_path, dst_path, fmt, bitrate, logger)


# ============================================================
# 模型加载 (从 webui.py 提取)
# ============================================================

# 这些全局变量在 init_models() 中初始化
_vocab: dict = {}
_id_to_token: dict = {}
_tag_vocab: dict = {}
_slide_vocab: dict = {}
_slide_vocab_inv: dict = {}
_path_best_timing: dict = {}
_models_cache: dict = {}
_audio_tokenizer = None
_device: str = "cpu"
_data_dir: str = "preprocessed"
_ckpt_dir: str = "checkpoints"
_audio_codebooks: int = 8
_premodel_path: str = ""
_target_subdiv: int = 4
_max_tags: int = 32
_time_signature: int = 4
_beat_cfg: dict = {}

# Token 类型 ID 集合 (延迟初始化)
_TAP_IDS: set = set()
_HOLD_IDS: set = set()
_SLIDE_IDS: set = set()
_TOUCH_IDS: set = set()
_TOUCHHOLD_IDS: set = set()
_NOTE_IDS: set = set()
_MULTI_TAP_IDS: set = set()
_VOCAB_SIZE: int = 0
_EMPTY_ID: int = 0

# 偏置掩码
BIAS_EMPTY_MASK: torch.Tensor = None
BIAS_NOTE_MASK: torch.Tensor = None
BIAS_TAP_MASK: torch.Tensor = None
BIAS_HOLD_MASK: torch.Tensor = None
BIAS_SLIDE_MASK: torch.Tensor = None
BIAS_TOUCH_MASK: torch.Tensor = None
BIAS_TOUCHHOLD_MASK: torch.Tensor = None
BIAS_MULTI_TAP_MASK: torch.Tensor = None
BIAS_WIFI_SLIDE_MASK: torch.Tensor = None

DIFFICULTIES = ["Easy", "Basic", "Advanced", "Expert", "Master", "Re:Master", "UTAGE"]
DIFF_MAP = {d: i + 1 for i, d in enumerate(DIFFICULTIES)}
DIFF_ID = {d: i for i, d in enumerate(DIFFICULTIES)}


def _count_simultaneous_taps(token_str: str) -> int:
    tap_count = 0
    for part in token_str.split("+"):
        from SimaiToken import SimaiToken, SimaiTokenType
        st = SimaiToken.from_string(part)
        if st is not None and st.token_type == SimaiTokenType.TAP:
            tap_count += len(st.position)
    return tap_count


def _build_mask(id_set: set, vocab_size: int) -> torch.Tensor:
    mask = torch.zeros(vocab_size, dtype=torch.float32)
    for i in id_set:
        if 0 <= i < vocab_size:
            mask[i] = 1.0
    return mask


def _is_wifi_slide_vocab_token(token: str) -> bool:
    path, _ = _slide_vocab_token_to_params(token)
    return re.search(r"(^|\*)w[1-8]", path) is not None


def init_models(cfg: dict, logger: Logger) -> bool:
    """初始化词表、偏置掩码、模型"""
    global _vocab, _id_to_token, _tag_vocab, _slide_vocab, _slide_vocab_inv
    global _path_best_timing, _models_cache, _audio_tokenizer, _device, _data_dir, _ckpt_dir
    global _TAP_IDS, _HOLD_IDS, _SLIDE_IDS, _TOUCH_IDS, _TOUCHHOLD_IDS
    global _NOTE_IDS, _MULTI_TAP_IDS, _VOCAB_SIZE, _EMPTY_ID
    global BIAS_EMPTY_MASK, BIAS_NOTE_MASK, BIAS_TAP_MASK, BIAS_HOLD_MASK
    global BIAS_SLIDE_MASK, BIAS_TOUCH_MASK, BIAS_TOUCHHOLD_MASK, BIAS_MULTI_TAP_MASK
    global BIAS_WIFI_SLIDE_MASK
    global _audio_codebooks, _premodel_path, _target_subdiv, _max_tags, _time_signature, _beat_cfg

    _data_dir = cfg["model"]["data_dir"]
    _ckpt_dir = cfg["model"]["ckpt_dir"]
    _device = cfg["model"]["device"]
    _audio_codebooks = cfg["model"]["audio_codebooks"]
    _premodel_path = cfg["model"].get("premodel_path", "")
    _target_subdiv = cfg["model"].get("target_subdiv", 4)
    _max_tags = cfg["model"].get("max_tags", 32)
    _time_signature = cfg.get("beat", {}).get("time_signature", 4)
    _beat_cfg = cfg.get("beat", {})

    if _device == "cuda" and not torch.cuda.is_available():
        logger.warn("CUDA 不可用, 回退到 CPU")
        _device = "cpu"

    # 加载词表
    data_path = Path(_data_dir)
    vocab_path = data_path / "vocab.json"
    if not vocab_path.exists():
        logger.error(f"词表文件不存在: {vocab_path}")
        return False

    with open(vocab_path, "r", encoding="utf-8") as f:
        _vocab = json.load(f)
    _id_to_token = {v: k for k, v in _vocab.items()}

    tag_path = data_path / "tag_vocab.json"
    if tag_path.exists():
        with open(tag_path, "r", encoding="utf-8") as f:
            _tag_vocab = json.load(f)
    else:
        _tag_vocab = {}

    slide_path = data_path / "slide_vocab.json"
    if slide_path.exists():
        _slide_vocab = json.loads(slide_path.read_text("utf-8"))
    else:
        _slide_vocab = {"<PAD>": 0}
    _slide_vocab_inv = {v: k for k, v in _slide_vocab.items()}

    timing_map_path = data_path / "slide_path_timing_map.json"
    if timing_map_path.exists():
        _path_best_timing = json.loads(timing_map_path.read_text("utf-8"))
    else:
        _path_best_timing = {}
    wifi_slide_ids = {
        int(v) for k, v in _slide_vocab.items()
        if k not in ("<PAD>", "<EOS>") and _is_wifi_slide_vocab_token(k)
    }

    # 预计算 token 类型集合
    _VOCAB_SIZE = (max(_vocab.values()) + 1) if _vocab else 1
    _EMPTY_ID = 0
    _TAP_IDS = set(v for k, v in _vocab.items() if k.startswith("tap"))
    _HOLD_IDS = set(v for k, v in _vocab.items() if k.startswith("hold"))
    _SLIDE_IDS = set(v for k, v in _vocab.items() if k.startswith("slide"))
    _TOUCH_IDS = set(v for k, v in _vocab.items() if k.startswith("touch"))
    _TOUCHHOLD_IDS = set(
        v for k, v in _vocab.items() if re.match(r"^hold[A-E]\d*$", k)
    )
    _NOTE_IDS = _TAP_IDS | _HOLD_IDS | _SLIDE_IDS | _TOUCH_IDS
    _MULTI_TAP_IDS = {
        v for k, v in _vocab.items() if _count_simultaneous_taps(k) >= 3
    }

    # 构建偏置掩码
    BIAS_EMPTY_MASK = torch.zeros(_VOCAB_SIZE, dtype=torch.float32)
    BIAS_EMPTY_MASK[_EMPTY_ID] = 1.0
    BIAS_NOTE_MASK = _build_mask(_NOTE_IDS, _VOCAB_SIZE)
    BIAS_TAP_MASK = _build_mask(_TAP_IDS, _VOCAB_SIZE)
    BIAS_HOLD_MASK = _build_mask(_HOLD_IDS, _VOCAB_SIZE)
    BIAS_SLIDE_MASK = _build_mask(_SLIDE_IDS, _VOCAB_SIZE)
    BIAS_TOUCH_MASK = _build_mask(_TOUCH_IDS, _VOCAB_SIZE)
    BIAS_TOUCHHOLD_MASK = _build_mask(_TOUCHHOLD_IDS, _VOCAB_SIZE)
    BIAS_MULTI_TAP_MASK = _build_mask(_MULTI_TAP_IDS, _VOCAB_SIZE)
    BIAS_WIFI_SLIDE_MASK = _build_mask(wifi_slide_ids, max(_slide_vocab_inv.keys(), default=0) + 1)

    # 仅缓存 CPU 模型。GPU 侧在每个 stage 推理时懒加载, 用完立刻释放。
    _models_cache = {}
    _audio_tokenizer = None

    logger.info(f"词表加载完成: chart={len(_vocab)}, tag={len(_tag_vocab)}, "
                f"slide={len(_slide_vocab)}, wifi_slide={len(wifi_slide_ids)}, device={_device}")
    return True


def _load_compatible_state(model, state: dict) -> None:
    """兼容加载模型权重"""
    current = model.state_dict()
    compatible = {}
    for name, tensor in state.items():
        if name in current and current[name].shape == tensor.shape:
            compatible[name] = tensor
    current.update(compatible)
    model.load_state_dict(current)


def _load_model(stage: int, logger: Logger):
    """延迟加载指定 Stage 的模型到 CPU 缓存。"""
    if stage in _models_cache:
        return _models_cache[stage]

    from models.stage1_chart import Stage1ChartModel
    from models.stage2_hold import Stage2HoldModel
    from models.stage3_slide import Stage3SlideModel
    from models.stage4_break import Stage4BreakModel
    from models.stage5_ex import Stage5ExModel

    candidates = [
        Path(_ckpt_dir) / f"stage{stage}_last.pt",
        Path(_ckpt_dir) / f"stage{stage}_best.pt",
        Path(_data_dir) / f"stage{stage}_last.pt",
        Path(_data_dir) / f"stage{stage}_best.pt",
        Path(_data_dir) / f"stage{stage}.pt",
    ]
    ckpt_path = next((p for p in candidates if p.exists()), None)
    if ckpt_path is None:
        raise FileNotFoundError(f"Stage {stage} checkpoint 未找到")

    logger.info(f"      加载 checkpoint 到 CPU: {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_cfg = ckpt.get("config", ckpt.get("cfg"))
    state = ckpt.get("model_state_dict", ckpt.get("model"))
    if model_cfg is None or state is None:
        raise KeyError(f"Checkpoint {ckpt_path} 缺少 config/cfg 或 model_state_dict/model")

    model_classes = {
        1: Stage1ChartModel, 2: Stage2HoldModel, 3: Stage3SlideModel,
        4: Stage4BreakModel, 5: Stage5ExModel,
    }
    model = model_classes[stage](model_cfg).to("cpu").eval()
    _load_compatible_state(model, state)
    del ckpt, state
    gc.collect()
    _models_cache[stage] = model
    return model


def _clear_device_cache() -> None:
    gc.collect()
    if _device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _clear_model_runtime_caches(model) -> None:
    for module in model.modules():
        clear_cache = getattr(module, "clear_cache", None)
        if callable(clear_cache):
            clear_cache()


def _release_runtime_caches(logger: Logger | None = None) -> None:
    """Release per-song heavy objects; vocab/config stay loaded."""
    global _models_cache, _audio_tokenizer

    released_models = len(_models_cache)
    for model in _models_cache.values():
        try:
            _clear_model_runtime_caches(model)
            model.to("cpu")
        except Exception:
            pass
    _models_cache.clear()

    released_audio = _audio_tokenizer is not None
    if _audio_tokenizer is not None:
        try:
            if getattr(_audio_tokenizer, "_model", None) is not None:
                _audio_tokenizer._model.to("cpu")
                _audio_tokenizer._model = None
            _audio_tokenizer._loaded = False
        except Exception:
            pass
        _audio_tokenizer = None

    _clear_device_cache()
    if logger and (released_models > 0 or released_audio):
        logger.info(
            f"  已释放本曲运行缓存: stage_models={released_models}, "
            f"audio_tokenizer={1 if released_audio else 0}"
        )


@contextmanager
def _stage_model(stage: int, logger: Logger):
    """Temporarily move one CPU-cached stage model to the inference device."""
    model = _load_model(stage, logger)
    try:
        if _device != "cpu":
            model.to(_device)
        model.eval()
        yield model
    finally:
        _clear_model_runtime_caches(model)
        if _device != "cpu":
            model.to("cpu")
        _clear_device_cache()


def _get_audio_tokenizer():
    """Keep EnCodec on CPU so audio preprocessing never occupies VRAM."""
    global _audio_tokenizer
    if _audio_tokenizer is None:
        from AudioTokenizer import AudioTokenizer
        _audio_tokenizer = AudioTokenizer(
            num_codebooks=_audio_codebooks,
            device="cpu",
            local_path=_premodel_path or None,
        )
    return _audio_tokenizer


# ============================================================
# 推理核心
# ============================================================

def _output_grid_index(frame_idx: int, frame_rate: float,
                       measure_dur: float, subdiv: int) -> tuple[int, int]:
    t_sec = frame_idx / frame_rate
    measure = int(t_sec / measure_dur)
    beat_in_measure = (t_sec % measure_dur) / measure_dur
    beat_idx = min(round(beat_in_measure * subdiv), subdiv - 1)
    return measure, beat_idx


def _duration_bin_to_str(dur_bin: int) -> str:
    secs = 2.0 ** (int(dur_bin) - 5)
    return f"{max(1, round(secs * 4))}:1"


def _mask_like_logits(mask: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Pad or trim a vocab mask to match a checkpoint's chart vocab dimension."""
    target = logits.shape[-1]
    mask = mask.to(logits.device)
    if mask.shape[0] == target:
        return mask
    if mask.shape[0] > target:
        return mask[:target]
    return F.pad(mask, (0, target - mask.shape[0]))


def _biased_sample(logits: torch.Tensor, temperature: float, top_k: int,
                   density: float, tap_bias: float, hold_bias: float,
                   slide_bias: float, touch_bias: float,
                   touchhold_bias: float, filter_multi_tap: bool,
                   allow_touch: bool) -> torch.Tensor:
    bias = (_mask_like_logits(BIAS_NOTE_MASK, logits) -
            _mask_like_logits(BIAS_EMPTY_MASK, logits)) * density
    bias += _mask_like_logits(BIAS_TAP_MASK, logits) * tap_bias
    bias += _mask_like_logits(BIAS_HOLD_MASK, logits) * hold_bias
    bias += _mask_like_logits(BIAS_SLIDE_MASK, logits) * slide_bias
    bias += _mask_like_logits(BIAS_TOUCH_MASK, logits) * touch_bias
    bias += _mask_like_logits(BIAS_TOUCHHOLD_MASK, logits) * touchhold_bias

    logits = logits + bias.view(1, 1, -1)
    if filter_multi_tap:
        logits = logits.masked_fill(
            _mask_like_logits(BIAS_MULTI_TAP_MASK, logits).view(1, 1, -1).bool(),
            float("-inf"),
        )
    if not allow_touch:
        touch_mask = (
            _mask_like_logits(BIAS_TOUCH_MASK, logits) +
            _mask_like_logits(BIAS_TOUCHHOLD_MASK, logits)
        ).bool()
        logits = logits.masked_fill(touch_mask.view(1, 1, -1), float("-inf"))

    if temperature > 0:
        logits = logits / temperature
    if top_k > 0 and top_k < logits.shape[-1]:
        topk_vals, _ = torch.topk(logits, top_k, dim=-1)
        min_topk = topk_vals[:, :, -1:]
        logits = torch.where(logits < min_topk,
                             torch.full_like(logits, float("-inf")), logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs.reshape(-1, logits.shape[-1]), 1).reshape(
        logits.shape[0], -1)


def _validate_slide_path(start_pos: str, path_str: str) -> bool:
    try:
        start = int(start_pos)
    except ValueError:
        return True
    m = re.match(r'^(V)([1-8])([1-8])', path_str)
    if m:
        target = int(m.group(3))
    else:
        m = re.match(r'^(pp|qq|PP|QQ|[><^vVpqszw-])([1-8])', path_str)
        if m is None:
            return True
        target = int(m.group(2))
    if m.group(1) == "-" and target == start:
        return False
    diff = abs(start - target)
    if diff == 1 or diff == 7:
        return False
    return True


def _is_slide_path_syntax(path_str: str) -> bool:
    """Return True for path strings that can legally follow a slide start button."""
    if not path_str:
        return False
    if re.match(r"^[A-E]\d*$", path_str):
        return False
    return re.match(r"^(?:\*?V[1-8][1-8]|pp[1-8]|qq[1-8]|PP[1-8]|QQ[1-8]|[><^vpqszw-][1-8])", path_str) is not None


def _slide_vocab_token_to_params(token: str) -> tuple[str, str]:
    m = re.match(r"^(.+)\[([^\]]+)\]$", token)
    if m:
        return m.group(1), m.group(2)
    return token, ""


def _default_slide_path(start_pos: str) -> str:
    try:
        start = int(start_pos)
    except ValueError:
        return "-4"
    return f"-{((start + 3) % 8) + 1}"


def _as_slot_array(arr: np.ndarray, length: int, slots: int = 1) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape[0] < length:
        pad = np.zeros((length - arr.shape[0], arr.shape[1]), dtype=arr.dtype)
        arr = np.concatenate([arr, pad], axis=0)
    if arr.shape[1] < slots:
        pad = np.zeros((arr.shape[0], slots - arr.shape[1]), dtype=arr.dtype)
        arr = np.concatenate([arr, pad], axis=1)
    return arr[:length, :slots]


def _first_slot_array(arr: np.ndarray, length: int) -> np.ndarray:
    """Match infer_full's legacy renderer: use one predicted value per frame."""
    slot_arr = _as_slot_array(arr, length, slots=1)
    return slot_arr[:, 0]


@dataclass
class AudioInferenceContext:
    tokens: np.ndarray
    beat_signal: np.ndarray
    frame_rate: float
    num_frames: int
    duration: float
    bpm: float
    measure_dur: float


def prepare_audio_context(
    audio_path: Path,
    gen_cfg: dict,
    logger: Logger,
) -> AudioInferenceContext:
    """Encode audio and beat features on CPU for reuse across difficulties."""
    from BeatTokenizer import BeatTokenizer

    logger.info(f"    编码音频 (CPU)...")
    ad = _get_audio_tokenizer().encode_file(str(audio_path))
    logger.info(f"    音频: {ad.duration:.1f}s, {ad.num_frames} frames")

    bpm_override = gen_cfg.get("bpm_override", 0)
    bt = BeatTokenizer(
        method=gen_cfg.get("beat_method", "librosa"),
        target_bpm=None if bpm_override <= 0 else bpm_override,
        bpm_min=_beat_cfg.get("bpm_min", 60),
        bpm_max=_beat_cfg.get("bpm_max", 240),
        tightness=_beat_cfg.get("tightness", 50),
        time_signature=_time_signature,
        quantize_beats=_beat_cfg.get("quantize_beats", True),
    )
    bl = bt.analyse(str(audio_path))
    logger.info(f"    节拍: BPM={bl.bpm:.1f}, {bl.num_beats} beats")

    fr = ad.frame_rate
    nf = ad.num_frames
    bpm = bl.bpm if bpm_override <= 0 else bpm_override
    measure_dur = _time_signature * 60.0 / bpm

    beat_s = np.zeros((nf, 2), dtype=np.float32)
    for b in bl.beats:
        fi = round(b.time * fr)
        if 0 <= fi < nf:
            beat_s[fi, 0] = max(beat_s[fi, 0], 0.5)
            if b.is_downbeat:
                beat_s[fi, 1] = 1.0

    return AudioInferenceContext(
        tokens=ad.tokens,
        beat_signal=(beat_s > 0.3).astype(np.float32),
        frame_rate=fr,
        num_frames=nf,
        duration=ad.duration,
        bpm=bpm,
        measure_dur=measure_dur,
    )


def _predict_stage3_slide_paths_sparse(
    m3,
    chart: torch.Tensor,
    audio: torch.Tensor,
    beat: torch.Tensor,
    diff_t: torch.Tensor,
    lvl_t: torch.Tensor,
    tags_t: torch.Tensor,
    temperature: float,
    top_k: int,
    wifi_bias: float,
    logger: Logger,
) -> np.ndarray:
    """Predict only frames that actually contain slide tokens.

    Stage3's dense head creates (T, max_slide_slots, slide_vocab) logits. Long
    songs with a large slide vocab can allocate multiple GiB there, while only
    slide frames need those logits for rendering.
    """
    from SimaiToken import SimaiToken, SimaiTokenType
    from models.common import build_causal_mask

    chart_np = chart[0].detach().cpu().numpy()
    slide_frames: list[int] = []
    slide_starts: list[str] = []
    for f, tid in enumerate(chart_np):
        tok_str = _id_to_token.get(int(tid))
        if not tok_str:
            continue
        for part in tok_str.split("+"):
            st = SimaiToken.from_string(part)
            if st is not None and st.token_type == SimaiTokenType.SLIDE:
                slide_frames.append(f)
                slide_starts.append(st.position)
                break

    T = chart.shape[1]
    slide_paths = np.zeros(T, dtype=np.int64)
    if not slide_frames:
        logger.info("      Stage3 sparse: 无 slide 帧, 跳过路径 head")
        return slide_paths

    max_slots = int(getattr(m3, "max_slide_slots", 1))
    slide_vocab_size = int(getattr(m3.cfg, "slide_vocab_size", len(_slide_vocab)))
    dense_gib = T * max_slots * slide_vocab_size * 4 / (1024 ** 3)
    sparse_mib = len(slide_frames) * slide_vocab_size * 4 / (1024 ** 2)
    logger.info(
        f"      Stage3 sparse: slide_frames={len(slide_frames)}/{T}, "
        f"dense_logits≈{dense_gib:.2f}GiB -> sparse≈{sparse_mib:.1f}MiB"
    )

    B, _, _ = audio.shape
    device = audio.device

    audio_feat = m3.audio_encoder(audio)
    chart_x = m3.chart_embed(chart.long())
    cond = m3.cond_embed(beat, diff_t, lvl_t, tags_t, frame_query=chart_x)
    x = m3.chart_fusion(chart_x, audio_feat, cond)

    causal_mask = build_causal_mask(T, device)
    for layer in m3.layers:
        x = layer(x, memory=audio_feat, causal_mask=causal_mask)
    x = m3.ln_final(x)
    del causal_mask, chart_x, cond

    frame_idx = torch.tensor(slide_frames, dtype=torch.long, device=device)
    x = x.index_select(1, frame_idx)

    # The renderer currently consumes only slot 0. Slot 0 is causal and does not
    # depend on later slots, so running a length-1 slot transformer is equivalent
    # for the value we actually use and avoids allocating all slot logits.
    slot_ids = torch.zeros(1, dtype=torch.long, device=device)
    slot_x = x.unsqueeze(2) + m3.slot_embed(slot_ids).view(1, 1, 1, -1)
    slot_x = slot_x.reshape(B * len(slide_frames), 1, -1)
    slot_mask = build_causal_mask(1, device)
    for layer in m3.slot_layers:
        slot_x = layer(slot_x, causal_mask=slot_mask)
    logits = m3.head(m3.slot_ln(slot_x).reshape(B, len(slide_frames), -1))[0]
    del x, slot_x, slot_mask, frame_idx

    slide_temp = temperature * 0.7
    sl = logits / max(slide_temp, 0.01)
    if wifi_bias:
        sl = sl + _mask_like_logits(BIAS_WIFI_SLIDE_MASK, sl).view(1, -1) * wifi_bias

    invalid_slide_ids_by_start: dict[str, list[int]] = {}
    for row, start_pos in enumerate(slide_starts):
        if start_pos not in invalid_slide_ids_by_start:
            invalid_slide_ids_by_start[start_pos] = []
            for pid, token in _slide_vocab_inv.items():
                if token in ("<PAD>", "<EOS>"):
                    continue
                path, _ = _slide_vocab_token_to_params(token)
                if not _validate_slide_path(start_pos, path) and int(pid) < sl.shape[-1]:
                    invalid_slide_ids_by_start[start_pos].append(int(pid))
        invalid_ids = invalid_slide_ids_by_start[start_pos]
        if invalid_ids:
            sl[row, invalid_ids] = float("-inf")

    slide_topk = max(10, top_k // 2)
    if slide_topk > 0 and slide_topk < sl.shape[-1]:
        topk_vals, _ = torch.topk(sl, slide_topk, dim=-1)
        min_topk = topk_vals[:, -1:]
        sl = torch.where(sl < min_topk, torch.full_like(sl, float("-inf")), sl)

    finite_rows = torch.isfinite(sl).any(dim=-1)
    if not finite_rows.all():
        sl = sl.clone()
        sl[~finite_rows] = 0.0
    probs = F.softmax(sl, dim=-1)
    pred = torch.multinomial(probs, 1).squeeze(-1).detach().cpu().numpy()
    slide_paths[np.asarray(slide_frames, dtype=np.int64)] = pred
    del logits, sl, probs
    return slide_paths


@torch.no_grad()
def infer_single(
    audio_path: Path,
    difficulty: str,
    level: float,
    designer: str,
    gen_cfg: dict,
    logger: Logger,
    audio_ctx: AudioInferenceContext | None = None,
) -> tuple[str, float, int] | None:
    """对单个音频生成单个难度的 simai 谱面

    Returns:
        (simai_text, bpm, note_count) 或 None (失败时)
    """
    from SimaiToken import SimaiToken, SimaiTokenType, _token_to_simai_note as note_to_simai

    diff_num = DIFF_MAP.get(difficulty, 5)
    diff_id = DIFF_ID.get(difficulty, 4)

    # ── 音频 + 节拍 ──
    if audio_ctx is None:
        audio_ctx = prepare_audio_context(audio_path, gen_cfg, logger)
    fr = audio_ctx.frame_rate
    nf = audio_ctx.num_frames
    bpm = audio_ctx.bpm
    subdiv = _target_subdiv
    measure_dur = audio_ctx.measure_dur

    audio = torch.from_numpy(audio_ctx.tokens).unsqueeze(0).long().to(_device)
    beat = torch.from_numpy(audio_ctx.beat_signal).unsqueeze(0).to(_device)

    diff_t = torch.tensor([diff_id], device=_device)
    lvl_t = torch.tensor([level], device=_device)

    # 标签
    tag_ids = [-1] * _max_tags
    tag_idx = 0
    diff_tag = f"difficulty:{difficulty}"
    if diff_tag in _tag_vocab and tag_idx < _max_tags:
        tag_ids[tag_idx] = _tag_vocab[diff_tag]
        tag_idx += 1
    # collection 标签 (可多选)
    collections = gen_cfg.get("collections", [])
    if collections:
        for col in collections:
            if not col or col == "无" or tag_idx >= _max_tags:
                continue
            col_tag = f"collection:{col}"
            if col_tag in _tag_vocab:
                tag_ids[tag_idx] = _tag_vocab[col_tag]
                tag_idx += 1
    if designer and designer != "AI" and tag_idx < _max_tags:
        des_tag = f"designer:{designer}"
        if des_tag in _tag_vocab:
            tag_ids[tag_idx] = _tag_vocab[des_tag]
            tag_idx += 1
    tags_t = torch.tensor([tag_ids], dtype=torch.long, device=_device)

    temperature = gen_cfg.get("temperature", 0.8)
    top_k = gen_cfg.get("top_k", 50)
    density = gen_cfg.get("density", 0.0)
    tap_bias = gen_cfg.get("tap_bias", 0.0)
    hold_bias = gen_cfg.get("hold_bias", 0.0)
    slide_bias = gen_cfg.get("slide_bias", 0.0)
    wifi_bias = gen_cfg.get("wifi_bias", 0.0)
    touch_bias = gen_cfg.get("touch_bias", 0.0)
    touchhold_bias = gen_cfg.get("touchhold_bias", 0.0)
    break_bias = gen_cfg.get("break_bias", 0.0)
    filter_multi_tap = gen_cfg.get("filter_multi_tap", True)
    allow_touch = gen_cfg.get("allow_touch", False)
    skip_stages = set(gen_cfg.get("skip_stages", []) or [])
    if skip_stages:
        logger.info(f"    跳过 Stage: {', '.join(sorted(skip_stages))}")
    if not allow_touch:
        logger.info("    Touch 音符已屏蔽")

    # ── Stage 1 ──
    logger.info(f"    Stage 1: 谱面骨架...")
    with _stage_model(1, logger) as m1:
        result1 = m1.forward(audio, beat, diff_t, lvl_t, tags_t)
        chart = _biased_sample(result1["logits"], temperature, top_k,
                               density, tap_bias, hold_bias, slide_bias,
                               touch_bias, touchhold_bias, filter_multi_tap,
                               allow_touch)
        del result1
    T = chart.shape[1]

    # ── 三押过滤 ──
    if filter_multi_tap:
        chart_np_tmp = chart[0].detach().cpu().numpy()
        tap_counts_by_grid: dict[tuple, int] = defaultdict(int)
        for fi in range(T):
            tid = int(chart_np_tmp[fi])
            token_str = _id_to_token.get(tid)
            if not token_str:
                continue
            tc = _count_simultaneous_taps(token_str)
            if tc <= 0:
                continue
            gi = _output_grid_index(fi, fr, measure_dur, subdiv)
            if tap_counts_by_grid[gi] + tc >= 3:
                chart[0, fi] = _EMPTY_ID
            else:
                tap_counts_by_grid[gi] += tc

    hold_ids = {tid for tok, tid in _vocab.items() if tok.startswith("hold")}

    # ── Stage 2: Hold ──
    if "Stage 2" in skip_stages:
        logger.info(f"    Stage 2: 跳过 Hold 持续...")
        hold_durs = np.zeros(T, dtype=np.int64)
    else:
        logger.info(f"    Stage 2: Hold 持续...")
        with _stage_model(2, logger) as m2:
            hold_mask = torch.zeros(1, T, dtype=torch.bool, device=_device)
            for hid in hold_ids:
                hold_mask = hold_mask | (chart == hid)
            dur_pred = m2.generate(chart, audio, beat, diff_t, lvl_t, tags_t,
                                   hold_mask, temperature=temperature)
            hold_durs = _first_slot_array(dur_pred[0].cpu().numpy(), T)
            del dur_pred, hold_mask

    # ── Stage 3: Slide ──
    if "Stage 3" in skip_stages:
        logger.info(f"    Stage 3: 跳过 Slide 路径...")
        slide_paths = np.zeros(T, dtype=np.int64)
    else:
        logger.info(f"    Stage 3: Slide 路径...")
        with _stage_model(3, logger) as m3:
            slide_paths = _predict_stage3_slide_paths_sparse(
                m3, chart, audio, beat, diff_t, lvl_t, tags_t,
                temperature, top_k, wifi_bias, logger,
            )

    # ── Stage 4: Break ──
    if "Stage 4" in skip_stages:
        logger.info(f"    Stage 4: 跳过 Break 标记...")
        break_pred = np.zeros(T, dtype=bool)
    else:
        logger.info(f"    Stage 4: Break 标记...")
        with _stage_model(4, logger) as m4:
            break_logits = m4.forward(chart, audio, beat, diff_t, lvl_t, tags_t)["logits"]
            if break_bias:
                break_logits = break_logits.clone()
                break_logits[..., 1] += break_bias
            break_pred = _first_slot_array(
                break_logits.argmax(dim=-1)[0].cpu().numpy(), T,
            ).astype(bool)
            del break_logits

    # ── Stage 5: Ex ──
    if "Stage 5" in skip_stages:
        logger.info(f"    Stage 5: 跳过 Ex 标记...")
        ex_pred = np.zeros_like(break_pred, dtype=bool)
    else:
        logger.info(f"    Stage 5: Ex 标记...")
        with _stage_model(5, logger) as m5:
            ex_raw = m5.predict(chart, audio, beat, diff_t, lvl_t, tags_t)
            ex_pred = _first_slot_array(
                ex_raw[0].cpu().numpy(), T,
            ).astype(bool)
            del ex_raw

    # ── 构建 simai ──
    logger.info(f"    构建 simai...")
    chart_np = chart[0].cpu().numpy()
    measures: dict[int, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
    note_count = 0

    for f in range(T):
        tid = int(chart_np[f])
        if tid <= 0:
            continue
        tok_str = _id_to_token.get(tid)
        if tok_str is None:
            continue

        st = SimaiToken.from_string(tok_str)
        if st is None:
            continue
        if not allow_touch and st.token_type == SimaiTokenType.TOUCH:
            continue
        if not allow_touch and st.token_type == SimaiTokenType.HOLD and re.match(r"^[A-E]", st.position):
            continue

        # Keep this renderer aligned with infer_full.py: one parsed token per frame.
        if st.token_type == SimaiTokenType.HOLD:
            if hold_durs[f] > 0:
                st.params["dur"] = _duration_bin_to_str(int(hold_durs[f]))
            elif "dur" not in st.params or not st.params["dur"]:
                st.params["dur"] = "4:1"

        if st.token_type == SimaiTokenType.SLIDE:
            pid = int(slide_paths[f])
            if pid > 1:
                seg = _slide_vocab_inv.get(pid, "")
                if seg and seg not in ("<PAD>", "<EOS>"):
                    path, timing = _slide_vocab_token_to_params(seg)
                    if _is_slide_path_syntax(path) and _validate_slide_path(st.position, path):
                        st.params["path"] = path
                        if timing:
                            st.params["dur"] = timing
            if "path" not in st.params or not st.params["path"]:
                st.params["path"] = _default_slide_path(st.position)
            if "dur" not in st.params or not st.params["dur"]:
                path_key = st.params.get("path", "")
                st.params["dur"] = _path_best_timing.get(path_key, "4:1")

        if break_pred[f]:
            st.params["break"] = ""
        if ex_pred[f]:
            st.params["ex"] = ""

        simai_note = note_to_simai(st)
        if not simai_note:
            continue

        m, bi = _output_grid_index(f, fr, measure_dur, subdiv)
        measures[m][bi].append(simai_note)
        note_count += len(st.position) if st.token_type == SimaiTokenType.TAP else 1

    # ── 写入 simai 文本 (仅谱面主体, 不含头部) ──
    max_m = max(measures.keys()) if measures else 0
    simai_lines = []
    for m_idx in range(max_m + 1):
        beats = measures.get(m_idx, {})
        parts = []
        for bi_idx in range(subdiv):
            if bi_idx in beats:
                parts.append("/".join(beats[bi_idx]))
            else:
                parts.append("")
        if m_idx == 0:
            simai_lines.append(f"({bpm:.1f}){{{subdiv}}}{','.join(parts)}")
        else:
            simai_lines.append(f"{{{subdiv}}}{','.join(parts)}")

    simai_body = "\n".join(simai_lines)
    del audio, beat, diff_t, lvl_t, tags_t, chart
    _clear_device_cache()
    return simai_body, bpm, note_count


# ============================================================
# 多难度合并
# ============================================================

def merge_multi_difficulty(
    title: str,
    artist: str,
    bpm: float,
    diff_results: list[tuple[str, str, float, int]],
) -> str:
    """将多个难度的推理结果合并为一个 maidata.txt

    Args:
        title: 歌曲标题
        artist: 谱面作者
        bpm: 整体 BPM
        diff_results: [(difficulty_name, simai_body, level, note_count), ...]
    """
    lines = [
        f"&title={title}",
        f"&artist={artist}",
        f"&wholebpm={bpm:.1f}",
    ]

    for diff_name, simai_body, level, note_count in diff_results:
        diff_num = DIFF_MAP.get(diff_name, 5)
        lines.append(f"&lv_{diff_num}={level:.1f}")
        lines.append(f"&des_{diff_num}={artist}")
        lines.append(f"&inote_{diff_num}=")
        lines.append(simai_body)

    return "\n".join(lines) + "\n"


# ============================================================
# 主流程
# ============================================================

def scan_input_files(input_dir: Path, cfg: dict, logger: Logger) -> list[Path]:
    """扫描输入文件夹, 返回待处理的文件列表 (去重: mp4 优先)"""
    audio_exts = set(cfg["batch"].get("audio_extensions",
                      [".mp3", ".wav", ".ogg", ".flac"]))
    video_exts = set(cfg["batch"].get("video_extensions",
                      [".mp4", ".webm", ".mkv"]))

    if not input_dir.exists():
        logger.error(f"输入文件夹不存在: {input_dir}")
        return []

    # 收集所有媒体文件；用 suffix.lower() 处理 .Mp3 这类混合大小写扩展名。
    allowed_exts = audio_exts | video_exts
    all_files = [
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in allowed_exts
    ]

    all_files = sorted(set(all_files), key=lambda p: p.stem)

    # 去重: mp4 和 mp3 同名时优先 mp4
    seen_names: dict[str, Path] = {}
    for f in all_files:
        name = f.stem
        is_video = f.suffix.lower() in video_exts
        if name not in seen_names:
            seen_names[name] = f
        elif is_video:
            # 视频优先
            if seen_names[name].suffix.lower() not in video_exts:
                seen_names[name] = f

    result = sorted(seen_names.values(), key=lambda p: p.stem)
    logger.info(f"扫描到 {len(all_files)} 个文件, 去重后 {len(result)} 个待处理")
    for f in result:
        logger.info(f"  - {f.name}")
    return result


def process_one_file(
    src_path: Path,
    cfg: dict,
    logger: Logger,
) -> bool:
    """处理单个输入文件: 生成输出文件夹及所有内容"""
    gen_cfg = cfg["generation"]
    out_cfg = cfg["output"]
    batch_cfg = cfg["batch"]

    # 输出子文件夹名
    subdir_name = batch_cfg.get("output_subdir_template", "{input_name}").format(
        input_name=sanitize_filename(src_path.stem),
    )
    output_dir = Path(batch_cfg["output_dir"]) / subdir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    maidata_path = output_dir / "maidata.txt"
    track_path = output_dir / f"track.{out_cfg['audio_format']}"
    pv_path = output_dir / "pv.mp4"
    bg_path = output_dir / "bg.png"

    # 检查是否跳过
    if out_cfg.get("skip_existing", False) and maidata_path.exists():
        logger.info(f"跳过 (已存在): {output_dir}")
        return True

    is_video = src_path.suffix.lower() in set(
        batch_cfg.get("video_extensions", [".mp4", ".webm", ".mkv"]))

    title = src_path.stem
    designer = gen_cfg.get("designer", "AI")
    logger.info(f"处理: {title} ({'视频' if is_video else '音频'})")

    # ── Step 1: 准备音频 ──
    actual_audio_path: Path = src_path
    if is_video:
        if out_cfg.get("copy_audio", True):
            logger.info("  提取音频...")
            if not extract_audio_from_video(
                src_path, track_path,
                fmt=out_cfg["audio_format"],
                bitrate=out_cfg.get("audio_bitrate", "192k"),
                logger=logger,
            ):
                logger.error(f"  音频提取失败, 跳过: {title}")
                return False
            actual_audio_path = track_path
        else:
            # 不保存音频, 直接对视频做推理 (AudioTokenizer 支持)
            actual_audio_path = src_path
    else:
        if out_cfg.get("copy_audio", True):
            logger.info("  复制音频...")
            if not copy_or_convert_audio(
                src_path, track_path,
                fmt=out_cfg["audio_format"],
                bitrate=out_cfg.get("audio_bitrate", "192k"),
                logger=logger,
            ):
                logger.error(f"  音频复制失败, 跳过: {title}")
                return False
            actual_audio_path = track_path

    # ── Step 2: 复制视频 / 提取封面 ──
    if is_video:
        if out_cfg.get("copy_video", True):
            logger.info("  复制视频...")
            try:
                shutil.copy2(src_path, pv_path)
            except Exception as e:
                logger.warn(f"  复制视频失败: {e}")

        if out_cfg.get("extract_bg", True):
            logger.info("  提取封面...")
            video_for_bg = pv_path if pv_path.exists() else src_path
            if not extract_first_frame(video_for_bg, bg_path,
                                       max_size=out_cfg.get("bg_max_size", 512),
                                       logger=logger):
                logger.warn("  封面提取失败 (非致命)")
    else:
        # 音频文件: 没有视频和封面
        pass

    # ── Step 3: 推理各难度 ──
    enabled_diffs = cfg["difficulties"].get("enabled", ["Master"])
    default_levels = cfg["difficulties"].get("default_levels", {})
    diff_params = cfg["difficulties"].get("params", {})

    if not enabled_diffs:
        logger.warn(f"  未启用任何难度, 跳过推理: {title}")
        return True

    diff_results: list[tuple[str, str, float, int]] = []
    overall_bpm = 120.0
    audio_context_cache: dict[tuple, AudioInferenceContext] = {}

    for i, diff_name in enumerate(enabled_diffs):
        if diff_name not in DIFF_MAP:
            logger.warn(f"  未知难度 '{diff_name}', 跳过")
            continue

        level = default_levels.get(diff_name, 13.0)
        # 合并全局参数 + 难度独立覆盖
        diff_gen_cfg = dict(gen_cfg)
        overrides = diff_params.get(diff_name, {})
        diff_gen_cfg.update(overrides)

        if overrides:
            override_str = ", ".join(f"{k}={v}" for k, v in overrides.items())
            logger.info(f"  [{i+1}/{len(enabled_diffs)}] Stage 1-5: {diff_name} Lv.{level} ({override_str})")
        else:
            logger.info(f"  [{i+1}/{len(enabled_diffs)}] Stage 1-5: {diff_name} Lv.{level}")

        try:
            audio_ctx_key = (
                float(diff_gen_cfg.get("bpm_override", 0) or 0),
                diff_gen_cfg.get("beat_method", "librosa"),
            )
            audio_ctx = audio_context_cache.get(audio_ctx_key)
            if audio_ctx is None:
                audio_ctx = prepare_audio_context(actual_audio_path, diff_gen_cfg, logger)
                audio_context_cache[audio_ctx_key] = audio_ctx
            else:
                logger.info("    复用音频/节拍缓存 (CPU)")

            result = infer_single(
                actual_audio_path, diff_name, level, designer,
                diff_gen_cfg, logger, audio_ctx,
            )
            if result is None:
                logger.error(f"  {diff_name} 推理失败")
                continue

            simai_body, bpm, note_count = result
            overall_bpm = bpm
            diff_results.append((diff_name, simai_body, level, note_count))
            logger.success(f"  {diff_name}: {note_count} notes, BPM={bpm:.1f}")

        except Exception as e:
            logger.error(f"  {diff_name} 推理异常: {e}")
            traceback.print_exc()
            _clear_device_cache()
            continue

    if not diff_results:
        logger.error(f"  所有难度推理均失败: {title}")
        audio_context_cache.clear()
        _clear_device_cache()
        return False

    # ── Step 4: 合并写入 maidata.txt ──
    maidata_text = merge_multi_difficulty(
        title=title,
        artist=designer,
        bpm=overall_bpm,
        diff_results=diff_results,
    )
    maidata_path.write_text(maidata_text, encoding="utf-8")
    logger.success(f"  谱面已保存: {maidata_path}")
    logger.info(f"  输出目录: {output_dir}")

    audio_context_cache.clear()
    _clear_device_cache()
    return True


def main():
    parser = argparse.ArgumentParser(description="maiChartGen3 批量推理")
    parser.add_argument("--config", type=str, default=None,
                        help="主配置文件名 (不含 .yaml, 默认使用 default, 可选 server_4090 等)")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="输入文件夹 (覆盖配置文件)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出根目录 (覆盖配置文件)")
    parser.add_argument("--device", type=str, default=None,
                        choices=["cuda", "cpu"], help="推理设备")
    parser.add_argument("--designer", type=str, default=None,
                        help="谱面作者")
    parser.add_argument("--skip_existing", action="store_true", default=None,
                        help="跳过已存在的输出")
    parser.add_argument("--dry_run", action="store_true",
                        help="仅列出待处理文件, 不实际推理")
    args = parser.parse_args()

    # 构建 CLI 覆盖
    cli_overrides = {}
    if args.input_dir:
        cli_overrides.setdefault("batch", {})["input_dir"] = args.input_dir
    if args.output_dir:
        cli_overrides.setdefault("batch", {})["output_dir"] = args.output_dir
    if args.device:
        cli_overrides.setdefault("model", {})["device"] = args.device
    if args.designer:
        cli_overrides.setdefault("generation", {})["designer"] = args.designer
    if args.skip_existing is not None:
        cli_overrides.setdefault("output", {})["skip_existing"] = args.skip_existing

    # 加载配置
    cfg = load_batch_config(args.config, cli_overrides if cli_overrides else None)

    # 初始化日志
    log_file = cfg["logging"].get("log_file", "logs/batch_infer.log")
    verbose = cfg["logging"].get("verbose", True)
    logger = Logger(log_file, verbose)

    logger.info("=" * 60)
    logger.info("maiChartGen3 批量推理启动")
    if args.config:
        logger.info(f"配置文件: Config/{args.config}.yaml")
    else:
        logger.info("配置文件: Config/default.yaml")
    logger.info(f"设备: {cfg['model']['device']}")
    logger.info("=" * 60)

    # 扫描输入文件
    input_dir = Path(cfg["batch"]["input_dir"])
    files = scan_input_files(input_dir, cfg, logger)

    if not files:
        logger.error("没有找到可处理的文件")
        logger.close()
        return

    if args.dry_run:
        logger.info("Dry-run 模式, 不执行推理")
        logger.close()
        return

    # 初始化模型 (全局词表等)
    logger.info("初始化模型和词表...")
    if not init_models(cfg, logger):
        logger.error("模型初始化失败, 终止")
        logger.close()
        return

    # 批量处理
    total = len(files)
    success_count = 0
    fail_count = 0
    start_time = time.time()

    for i, src_path in enumerate(files):
        logger.info(f"\n{'─' * 50}")
        logger.info(f"[{i+1}/{total}] {src_path.name}")
        logger.info(f"{'─' * 50}")

        try:
            if process_one_file(src_path, cfg, logger):
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            logger.error(f"未捕获的异常: {e}")
            traceback.print_exc()
            fail_count += 1
        finally:
            _release_runtime_caches(logger)

        # 进度
        elapsed = time.time() - start_time
        done = i + 1
        eta = elapsed / done * (total - done) if done > 0 else 0
        logger.info(f"进度: {done}/{total} | 成功: {success_count} | 失败: {fail_count} | "
                    f"耗时: {elapsed:.0f}s | ETA: {eta:.0f}s")

    # 总结
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("批量推理完成")
    logger.info(f"  总数: {total}")
    logger.info(f"  成功: {success_count}")
    logger.info(f"  失败: {fail_count}")
    logger.info(f"  总耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    logger.info(f"  输出目录: {cfg['batch']['output_dir']}")
    logger.info("=" * 60)

    logger.close()


if __name__ == "__main__":
    main()
