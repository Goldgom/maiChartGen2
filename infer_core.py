"""
infer_core.py — maiChartGen3 共享推理引擎

为 webui.py 和 batch_infer.py 提供统一的推理链路。
所有谱面生成逻辑集中于此，确保两个入口的行为完全一致。

用法:
  from infer_core import create_engine
  engine = create_engine(cfg)          # cfg 来自 Config.load_config()
  simai_text, info = engine.generate_chart(mp3_path, ...)

  # 批量推理优化 (预计算音频上下文):
  audio_ctx = engine.prepare_audio(mp3_path, bpm_override)
  simai_text, info = engine.generate_chart_from_ctx(audio_ctx, ...)
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from models.stage1_chart import Stage1ChartModel
from models.stage2_hold import Stage2HoldModel
from models.stage3_slide import Stage3SlideModel
from models.stage4_break import Stage4BreakModel
from models.stage5_ex import Stage5ExModel
from SimaiToken import SimaiToken, SimaiTokenType, _token_to_simai_note as note_to_simai


# ============================================================
# 音频上下文 (预计算, 可跨难度复用)
# ============================================================

@dataclass
class AudioContext:
    tokens: np.ndarray          # (num_codebooks, num_frames)
    beat_signal: np.ndarray     # (num_frames, 2) binary
    frame_rate: float
    num_frames: int
    duration: float
    bpm: float
    measure_dur: float


# ============================================================
# InferenceEngine
# ============================================================

class InferenceEngine:
    """统一的谱面推理引擎。

    封装词表、模型、偏置掩码等所有共享状态。
    webui 和 batch_infer 共用一个实例。
    """

    def __init__(self, cfg):
        self.cfg = cfg

        # ── 路径 ──
        self.data_dir = cfg.preprocess.output_dir
        self.vocab_dir = Path(getattr(cfg.paths, "vocab_dir", "vocab"))
        self.ckpt_dir = cfg.paths.model_dir
        self.output_dir = Path(cfg.paths.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ── 设备 ──
        _d = cfg.audio.device
        if _d == "cuda" and not torch.cuda.is_available():
            print("[infer_core] 警告: CUDA 不可用, 回退到 CPU")
            self.device = "cpu"
        elif _d == "cpu" and torch.cuda.is_available():
            print("[infer_core] 提示: CUDA 可用但配置为 CPU")
            self.device = "cpu"
        else:
            self.device = _d if _d in ("cuda", "cpu") else ("cuda" if torch.cuda.is_available() else "cpu")

        # ── 常量 ──
        self.DIFFICULTIES = ["Easy", "Basic", "Advanced", "Expert", "Master", "Re:Master", "UTAGE"]
        self.DIFF_MAP = {d: i + 1 for i, d in enumerate(self.DIFFICULTIES)}
        self.DIFF_ID = {d: i for i, d in enumerate(self.DIFFICULTIES)}

        # ── 加载词表 ──
        self._load_vocabs()

        # ── 构建偏置掩码 ──
        self._build_bias_masks()

        # ── 模型缓存 ──
        self._models_cache: dict = {}
        self._audio_tokenizer = None

    # ═══════════════════════════════════════════════════════════
    # 初始化
    # ═══════════════════════════════════════════════════════════

    def _load_vocabs(self):
        dp = self.vocab_dir

        with open(dp / "vocab.json", "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        self.id_to_token = {v: k for k, v in self.vocab.items()}

        tag_path = dp / "tag_vocab.json"
        self.tag_vocab = json.loads(tag_path.read_text("utf-8")) if tag_path.exists() else {}

        slide_path = dp / "slide_vocab.json"
        if slide_path.exists():
            self.slide_vocab = json.loads(slide_path.read_text("utf-8"))
        else:
            self.slide_vocab = {"<PAD>": 0}
        self.slide_vocab_inv = {v: k for k, v in self.slide_vocab.items()}

        timing_path = dp / "slide_path_timing_map.json"
        if timing_path.exists():
            self.path_best_timing = json.loads(timing_path.read_text("utf-8"))
        else:
            self.path_best_timing = {}
        print(f"Loaded path→timing map: {len(self.path_best_timing)} paths")

        # 提取 collection / designer 标签
        self.collection_tags = sorted([
            k.replace("collection:", "") for k in self.tag_vocab
            if k.startswith("collection:")
        ], key=lambda x: (0 if x == "Original" else 1, x))
        self.designer_tags = sorted([
            k.replace("designer:", "") for k in self.tag_vocab
            if k.startswith("designer:")
        ])

    def _build_bias_masks(self):
        vocab_size = max(self.vocab.values()) + 1 if self.vocab else 1
        self.vocab_size = vocab_size
        self.empty_id = 0

        def _build(id_set):
            m = torch.zeros(vocab_size, dtype=torch.float32)
            for i in id_set:
                if 0 <= i < vocab_size:
                    m[i] = 1.0
            return m

        self._tap_ids = set(v for k, v in self.vocab.items() if k.startswith("tap"))
        self._hold_ids = set(v for k, v in self.vocab.items() if k.startswith("hold"))
        self._slide_ids = set(v for k, v in self.vocab.items() if k.startswith("slide"))
        self._touch_ids = set(v for k, v in self.vocab.items() if k.startswith("touch"))
        self._touchhold_ids = set(
            v for k, v in self.vocab.items() if re.match(r"^hold[A-E]\d*$", k)
        )
        self._note_ids = self._tap_ids | self._hold_ids | self._slide_ids | self._touch_ids
        self._multi_tap_ids = {
            v for k, v in self.vocab.items() if _count_simultaneous_taps(k) >= 3
        }
        self._tap_counts = torch.zeros(vocab_size, dtype=torch.long)
        self._has_touch = torch.zeros(vocab_size, dtype=torch.bool)
        for token, tid in self.vocab.items():
            if 0 <= tid < vocab_size:
                self._tap_counts[tid] = _count_simultaneous_taps(token)
                self._has_touch[tid] = _has_touch_note(token)

        # WiFi slide IDs
        wifi_ids = {
            int(v) for k, v in self.slide_vocab.items()
            if k not in ("<PAD>", "<EOS>") and _is_wifi_slide_vocab_token(k)
        }

        self.bias_empty = torch.zeros(vocab_size, dtype=torch.float32)
        self.bias_empty[self.empty_id] = 1.0
        self.bias_note = _build(self._note_ids)
        self.bias_tap = _build(self._tap_ids)
        self.bias_hold = _build(self._hold_ids)
        self.bias_slide = _build(self._slide_ids)
        self.bias_touch = _build(self._touch_ids)
        self.bias_touchhold = _build(self._touchhold_ids)
        self.bias_multi_tap = _build(self._multi_tap_ids)
        self.bias_wifi_slide = torch.zeros(
            max(self.slide_vocab_inv.keys(), default=0) + 1, dtype=torch.float32
        )
        for i in wifi_ids:
            self.bias_wifi_slide[i] = 1.0
        print(f"Loaded wifi slide paths: {len(wifi_ids)}")

    # ═══════════════════════════════════════════════════════════
    # 模型管理
    # ═══════════════════════════════════════════════════════════

    def _load_compatible_state(self, model, state: dict) -> None:
        current = model.state_dict()
        compatible = {}
        skipped = []
        for name, tensor in state.items():
            if name in current and current[name].shape == tensor.shape:
                compatible[name] = tensor
            else:
                skipped.append(name)
        current.update(compatible)
        model.load_state_dict(current)
        if skipped:
            print(f"Skipped {len(skipped)} incompatible tensors: {skipped[:6]}")

    def load_model(self, stage: int):
        """加载指定 stage 的模型到 self.device。"""
        if stage in self._models_cache:
            return self._models_cache[stage]

        candidates = [
            Path(self.ckpt_dir) / f"stage{stage}_last.pt",
            Path(self.ckpt_dir) / f"stage{stage}_best.pt",
            Path(self.data_dir) / f"stage{stage}_last.pt",
            Path(self.data_dir) / f"stage{stage}_best.pt",
            Path(self.data_dir) / f"stage{stage}.pt",
        ]
        ckpt_path = next((p for p in candidates if p.exists()), None)
        if ckpt_path is None:
            raise FileNotFoundError(f"Stage {stage} checkpoint not found")

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        model_cfg = ckpt.get("config", ckpt.get("cfg"))
        state = ckpt.get("model_state_dict", ckpt.get("model"))

        model_classes = {
            1: Stage1ChartModel, 2: Stage2HoldModel, 3: Stage3SlideModel,
            4: Stage4BreakModel, 5: Stage5ExModel,
        }
        model = model_classes[stage](model_cfg).to(self.device).eval()
        self._load_compatible_state(model, state)
        self._models_cache[stage] = model
        return model

    def _get_audio_tokenizer(self):
        """延迟加载 EnCodec (保持 CPU 上)。"""
        if self._audio_tokenizer is None:
            from AudioTokenizer import AudioTokenizer
            self._audio_tokenizer = AudioTokenizer(
                num_codebooks=self.cfg.audio.num_codebooks,
                device="cpu",
                local_path=self.cfg.audio.premodel_path or None,
            )
        return self._audio_tokenizer

    # ═══════════════════════════════════════════════════════════
    # 音频预处理 (批量推理复用)
    # ═══════════════════════════════════════════════════════════

    def prepare_audio(self, mp3_path: str, bpm_override: float = 0.0) -> AudioContext:
        """编码音频+节拍, 返回可跨难度复用的 AudioContext。"""
        from AudioTokenizer import AudioTokenizer
        from BeatTokenizer import BeatTokenizer

        at = AudioTokenizer(
            num_codebooks=self.cfg.audio.num_codebooks,
            device="cpu",
            local_path=self.cfg.audio.premodel_path or None,
        )
        ad = at.encode_file(mp3_path)

        bt = BeatTokenizer(
            method=self.cfg.beat.method,
            target_bpm=None if bpm_override <= 0 else bpm_override,
            quantize_beats=self.cfg.beat.quantize_beats,
            bpm_min=self.cfg.beat.bpm_min,
            bpm_max=self.cfg.beat.bpm_max,
        )
        bl = bt.analyse(mp3_path)

        fr = ad.frame_rate
        nf = ad.num_frames
        bpm = bl.bpm if bpm_override <= 0 else bpm_override
        measure_dur = 4 * 60.0 / bpm

        beat_s = np.zeros((nf, 2), dtype=np.float32)
        for b in bl.beats:
            fi = round(b.time * fr)
            if 0 <= fi < nf:
                beat_s[fi, 0] = max(beat_s[fi, 0], 0.5)
                if b.is_downbeat:
                    beat_s[fi, 1] = 1.0

        return AudioContext(
            tokens=ad.tokens,
            beat_signal=(beat_s > 0.3).astype(np.float32),
            frame_rate=fr,
            num_frames=nf,
            duration=ad.duration,
            bpm=bpm,
            measure_dur=measure_dur,
        )

    # ═══════════════════════════════════════════════════════════
    # 偏置采样
    # ═══════════════════════════════════════════════════════════

    def _mask_like_logits(self, mask: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        target = logits.shape[-1]
        mask = mask.to(logits.device)
        if mask.shape[0] == target:
            return mask
        if mask.shape[0] > target:
            return mask[:target]
        return F.pad(mask, (0, target - mask.shape[0]))

    def _biased_sample(
        self, logits: torch.Tensor, temperature: float, top_k: int,
        density: float, tap_bias: float, hold_bias: float, slide_bias: float,
        touch_bias: float, touchhold_bias: float, filter_multi_tap: bool,
        allow_touch: bool = True,
    ) -> torch.Tensor:
        """对 logits 施加类型偏置后采样。"""
        device = logits.device

        bias = (self._mask_like_logits(self.bias_note, logits) -
                self._mask_like_logits(self.bias_empty, logits)) * density
        bias += self._mask_like_logits(self.bias_tap, logits) * tap_bias
        bias += self._mask_like_logits(self.bias_hold, logits) * hold_bias
        bias += self._mask_like_logits(self.bias_slide, logits) * slide_bias
        bias += self._mask_like_logits(self.bias_touch, logits) * touch_bias
        bias += self._mask_like_logits(self.bias_touchhold, logits) * touchhold_bias

        logits = logits + bias.view(1, 1, -1)
        if filter_multi_tap:
            logits = logits.masked_fill(
                self._mask_like_logits(self.bias_multi_tap, logits).view(1, 1, -1).bool(),
                float("-inf"),
            )
        if not allow_touch:
            touch_mask = (
                self._mask_like_logits(self.bias_touch, logits) +
                self._mask_like_logits(self.bias_touchhold, logits)
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
        tokens = torch.multinomial(probs.reshape(-1, logits.shape[-1]), 1).reshape(
            logits.shape[0], -1)
        return tokens

    def _apply_stage1_bias(
        self, logits: torch.Tensor, density: float, tap_bias: float,
        hold_bias: float, slide_bias: float, touch_bias: float,
        touchhold_bias: float, filter_multi_tap: bool,
        allow_touch: bool = True,
    ) -> torch.Tensor:
        """Apply Stage1 sampling biases without sampling, for AR generation."""
        bias = (self._mask_like_logits(self.bias_note, logits) -
                self._mask_like_logits(self.bias_empty, logits)) * density
        bias += self._mask_like_logits(self.bias_tap, logits) * tap_bias
        bias += self._mask_like_logits(self.bias_hold, logits) * hold_bias
        bias += self._mask_like_logits(self.bias_slide, logits) * slide_bias
        bias += self._mask_like_logits(self.bias_touch, logits) * touch_bias
        bias += self._mask_like_logits(self.bias_touchhold, logits) * touchhold_bias

        logits = logits + bias.view(1, 1, -1)
        if filter_multi_tap:
            logits = logits.masked_fill(
                self._mask_like_logits(self.bias_multi_tap, logits).view(1, 1, -1).bool(),
                float("-inf"),
            )
        if not allow_touch:
            touch_mask = (
                self._mask_like_logits(self.bias_touch, logits) +
                self._mask_like_logits(self.bias_touchhold, logits)
            ).bool()
            logits = logits.masked_fill(touch_mask.view(1, 1, -1), float("-inf"))
        return logits

    def _apply_stage1_constraints(
        self,
        logits: torch.Tensor,
        frame_idx: int,
        generated: torch.Tensor,
        frame_rate: float,
        measure_dur: float,
        subdiv: int,
        filter_multi_tap: bool,
    ) -> torch.Tensor:
        if not filter_multi_tap:
            return logits

        current_grid = _output_grid_index(frame_idx, frame_rate, measure_dur, subdiv)
        tap_counts = self._mask_like_logits(self._tap_counts, logits).to(logits.device)
        has_touch = self._mask_like_logits(self._has_touch.float(), logits).to(logits.device).bool()
        note_candidate = (tap_counts > 0) | has_touch
        masked = logits.clone()

        for batch_idx in range(generated.shape[0]):
            existing_taps = 0
            existing_touch = False
            for prev_frame in range(frame_idx):
                if _output_grid_index(prev_frame, frame_rate, measure_dur, subdiv) != current_grid:
                    continue
                tid = int(generated[batch_idx, prev_frame].item())
                if 0 <= tid < self._tap_counts.shape[0]:
                    existing_taps += int(self._tap_counts[tid].item())
                    existing_touch = existing_touch or bool(self._has_touch[tid].item())

            final_taps = existing_taps + tap_counts
            invalid = note_candidate & (
                (final_taps >= 3) |
                ((final_taps >= 2) & (existing_touch | has_touch))
            )
            if invalid.any():
                masked[batch_idx, :, invalid] = float("-inf")

        return masked

    # ═══════════════════════════════════════════════════════════
    # 三押过滤
    # ═══════════════════════════════════════════════════════════

    def _filter_multi_tap_chart(
        self, chart: torch.Tensor, frame_rate: float,
        measure_dur: float, subdiv: int,
    ) -> tuple[torch.Tensor, int, int]:
        """把最终输出格子形成三押及以上的 Stage1 token 置空。"""
        filtered_chart = chart.clone()
        chart_np = filtered_chart[0].detach().cpu().numpy()
        tap_counts_by_grid: dict[tuple, int] = defaultdict(int)
        touch_by_grid: dict[tuple, bool] = defaultdict(bool)
        filtered_tokens = 0
        filtered_taps = 0

        for frame_idx, tid in enumerate(chart_np):
            token_str = self.id_to_token.get(int(tid))
            if not token_str:
                continue
            tap_count = _count_simultaneous_taps(token_str)
            has_touch = _has_touch_note(token_str)
            if tap_count <= 0 and not has_touch:
                continue
            grid = _output_grid_index(frame_idx, frame_rate, measure_dur, subdiv)
            final_taps = tap_counts_by_grid[grid] + tap_count
            final_touch = touch_by_grid[grid] or has_touch
            if final_taps >= 3 or (final_taps >= 2 and final_touch):
                filtered_chart[0, frame_idx] = self.empty_id
                filtered_tokens += 1
                filtered_taps += tap_count
                continue
            tap_counts_by_grid[grid] += tap_count
            touch_by_grid[grid] = final_touch

        return filtered_chart, filtered_tokens, filtered_taps

    # ═══════════════════════════════════════════════════════════
    # Slide 校验
    # ═══════════════════════════════════════════════════════════

    def _validate_slide_path(self, start_pos: str, path_str: str) -> bool:
        try:
            start = int(start_pos)
        except ValueError:
            return True
        first = _slide_first_target(path_str)
        if first is None:
            return True
        connector, target = first
        if connector == "-" and target == start:
            return False
        diff = abs(start - target)
        if diff == 1 or diff == 7:
            return False
        return True

    def _invalid_slide_vocab_ids_for_start(self, start_pos: str) -> list[int]:
        invalid = []
        for pid, token in self.slide_vocab_inv.items():
            if token in ("<PAD>", "<EOS>"):
                continue
            path, _ = _slide_vocab_token_to_params(token)
            if not self._validate_slide_path(start_pos, path):
                invalid.append(int(pid))
        return invalid

    # ═══════════════════════════════════════════════════════════
    # 推理核心
    # ═══════════════════════════════════════════════════════════

    @torch.no_grad()
    def generate_chart(
        self,
        mp3_path: str,
        difficulty: str,
        level: float,
        designer: str,
        collections: list[str],
        temperature: float,
        top_k: int,
        bpm_override: float,
        density: float,
        tap_bias: float,
        hold_bias: float,
        slide_bias: float,
        wifi_bias: float,
        touch_bias: float,
        touchhold_bias: float,
        break_bias: float,
        filter_multi_tap: bool,
        skip_stages: list[str] | None = None,
        audio_ctx: AudioContext | None = None,
        allow_touch: bool = True,
        verbose: bool = True,
    ) -> tuple[str, str]:
        """
        核心推理函数，返回 (simai文本, 状态信息)。

        Args:
            mp3_path: 音频文件路径
            difficulty: 难度名称
            level: 等级
            designer: 谱面作者
            collections: 曲库标签列表
            temperature: 采样温度
            top_k: Top-K 采样
            bpm_override: BPM 覆盖 (<=0 表示自动检测)
            density: 整体密度偏置
            tap_bias: Tap 偏置
            hold_bias: Hold 偏置
            slide_bias: Slide 偏置
            wifi_bias: WiFi Slide 偏置
            touch_bias: Touch 偏置
            touchhold_bias: TouchHold 偏置
            break_bias: Break 偏置
            filter_multi_tap: 是否过滤三押及以上
            skip_stages: 跳过的 stage 列表
            audio_ctx: 预计算的音频上下文 (批量推理优化, None=自动计算)
            allow_touch: 是否允许 Touch 音符 (默认 True)
            verbose: 是否打印详细日志
        """
        skip_stages = set(skip_stages or [])

        diff_num = self.DIFF_MAP.get(difficulty, 5)
        diff_id = self.DIFF_ID.get(difficulty, 4)

        cfg = self.cfg
        device = self.device

        # ── 1. 音频编码 ──
        if audio_ctx is not None:
            fr = audio_ctx.frame_rate
            nf = audio_ctx.num_frames
            bpm = audio_ctx.bpm
            measure_dur = audio_ctx.measure_dur
            ad_tokens = audio_ctx.tokens
            beat_signal = audio_ctx.beat_signal
        else:
            from AudioTokenizer import AudioTokenizer
            from BeatTokenizer import BeatTokenizer

            if verbose:
                print("[infer] 编码音频...")
            at = AudioTokenizer(
                num_codebooks=cfg.audio.num_codebooks,
                device="cpu",
                local_path=cfg.audio.premodel_path or None,
            )
            ad = at.encode_file(mp3_path)
            bt = BeatTokenizer(
                method=cfg.beat.method,
                target_bpm=None if bpm_override <= 0 else bpm_override,
                quantize_beats=cfg.beat.quantize_beats,
                bpm_min=cfg.beat.bpm_min,
                bpm_max=cfg.beat.bpm_max,
            )
            bl = bt.analyse(mp3_path)

            fr = ad.frame_rate
            nf = ad.num_frames
            bpm = bl.bpm if bpm_override <= 0 else bpm_override
            subdiv = cfg.chart.target_subdiv
            measure_dur = 4 * 60.0 / bpm
            ad_tokens = ad.tokens

            beat_s = np.zeros((nf, 2), dtype=np.float32)
            for b in bl.beats:
                fi = round(b.time * fr)
                if 0 <= fi < nf:
                    beat_s[fi, 0] = max(beat_s[fi, 0], 0.5)
                    if b.is_downbeat:
                        beat_s[fi, 1] = 1.0
            beat_signal = (beat_s > 0.3).astype(np.float32)

        subdiv = cfg.chart.target_subdiv

        audio = torch.from_numpy(ad_tokens).unsqueeze(0).long().to(device)
        beat = torch.from_numpy(beat_signal).unsqueeze(0).to(device)

        diff_t = torch.tensor([diff_id], device=device)
        lvl_t = torch.tensor([level], device=device)

        # 构建 tag tensor
        max_tags = 32
        tag_ids = [-1] * max_tags
        tag_idx = 0
        diff_tag = f"difficulty:{difficulty}"
        if diff_tag in self.tag_vocab and tag_idx < max_tags:
            tag_ids[tag_idx] = self.tag_vocab[diff_tag]
            tag_idx += 1
        if collections:
            for col in collections:
                if not col or col == "无" or tag_idx >= max_tags:
                    continue
                col_tag = f"collection:{col}"
                if col_tag in self.tag_vocab:
                    tag_ids[tag_idx] = self.tag_vocab[col_tag]
                    tag_idx += 1
        if designer and designer != "AI" and tag_idx < max_tags:
            des_tag = f"designer:{designer}"
            if des_tag in self.tag_vocab:
                tag_ids[tag_idx] = self.tag_vocab[des_tag]
                tag_idx += 1
        tags_t = torch.tensor([tag_ids], dtype=torch.long, device=device)

        # ── Stage 1: 谱面骨架 ──
        if verbose:
            print("[infer] Stage 1: 谱面骨架...")
        m1 = self.load_model(1)
        chart = m1.generate(
            audio, beat, diff_t, lvl_t, tags_t,
            temperature=temperature,
            top_k=top_k,
            logits_processor=lambda logits, t, generated: self._apply_stage1_constraints(
                self._apply_stage1_bias(
                    logits,
                    density,
                    tap_bias,
                    hold_bias,
                    slide_bias,
                    touch_bias,
                    touchhold_bias,
                    filter_multi_tap,
                    allow_touch=allow_touch,
                ),
                t,
                generated,
                fr,
                measure_dur,
                subdiv,
                filter_multi_tap,
            ),
        )
        T = chart.shape[1]
        filtered_multi_tap_tokens = 0
        filtered_multi_tap_count = 0
        if filter_multi_tap:
            chart, filtered_multi_tap_tokens, filtered_multi_tap_count = \
                self._filter_multi_tap_chart(chart, fr, measure_dur, subdiv)

        hold_ids = {tid for tok, tid in self.vocab.items() if tok.startswith("hold")}

        # ── Stage 2: Hold 持续时间 ──
        if "Stage 2" in skip_stages:
            hold_durs = np.zeros((T, 1), dtype=np.int64)
        else:
            if verbose:
                print("[infer] Stage 2: Hold 持续时间...")
            m2 = self.load_model(2)
            hold_mask = torch.zeros(1, T, dtype=torch.bool, device=device)
            for hid in hold_ids:
                hold_mask = hold_mask | (chart == hid)
            dur_pred = m2.generate(chart, audio, beat, diff_t, lvl_t, tags_t, hold_mask,
                                   temperature=temperature)
            hold_durs = _as_slot_array(dur_pred[0].cpu().numpy(), T)

        # ── Stage 3: Slide 路径 ──
        if "Stage 3" in skip_stages:
            slide_paths = np.zeros((T, 1), dtype=np.int64)
        else:
            if verbose:
                print("[infer] Stage 3: Slide 路径...")
            m3 = self.load_model(3)
            out3 = m3(chart, audio, beat, diff_t, lvl_t, tags_t)
            slide_logits = out3["logits"][0]  # (T, S, V)
            S = slide_logits.shape[1]

            slide_temp = temperature * 0.7
            slide_topk = max(10, top_k // 2)

            if slide_temp > 0:
                sl = slide_logits / slide_temp
            else:
                sl = slide_logits.clone()
            if wifi_bias:
                sl = sl + self._mask_like_logits(self.bias_wifi_slide, sl).view(1, 1, -1) * wifi_bias

            # 屏蔽非法 slide 路径
            chart_np_for_slide = chart[0].detach().cpu().numpy()
            invalid_slide_ids_by_start: dict[str, list[int]] = {}
            for f in range(T):
                tok_str = self.id_to_token.get(int(chart_np_for_slide[f]))
                if not tok_str:
                    continue
                slide_slot = 0
                for part in tok_str.split("+"):
                    st = SimaiToken.from_string(part)
                    if st is None or st.token_type != SimaiTokenType.SLIDE:
                        continue
                    if slide_slot >= sl.shape[1]:
                        break
                    if st.position not in invalid_slide_ids_by_start:
                        invalid_slide_ids_by_start[st.position] = [
                            pid for pid in self._invalid_slide_vocab_ids_for_start(st.position)
                            if pid < sl.shape[-1]
                        ]
                    invalid_ids = invalid_slide_ids_by_start[st.position]
                    if invalid_ids:
                        sl[f, slide_slot, invalid_ids] = float("-inf")
                    slide_slot += 1

            if slide_topk > 0 and slide_topk < sl.shape[-1]:
                topk_vals, _ = torch.topk(sl, slide_topk, dim=-1)
                min_topk = topk_vals[:, :, -1:]
                sl = torch.where(sl < min_topk, torch.full_like(sl, float("-inf")), sl)

            probs = F.softmax(sl, dim=-1)
            flat_probs = probs.reshape(-1, sl.shape[-1])
            slide_paths = torch.multinomial(flat_probs, 1).reshape(T, S).cpu().numpy()

        # ── Stage 4/5: Break / Ex ──
        note_mask = (chart > 0).bool()
        if "Stage 4" in skip_stages:
            break_pred = np.zeros((T, 1), dtype=bool)
        else:
            if verbose:
                print("[infer] Stage 4: Break...")
            m4 = self.load_model(4)
            break_logits = m4.forward(chart, audio, beat, diff_t, lvl_t, tags_t)["logits"]
            if break_bias:
                break_logits = break_logits.clone()
                break_logits[..., 1] += break_bias
            break_pred = _as_slot_array(
                break_logits.argmax(dim=-1)[0].cpu().numpy(), T,
            ).astype(bool)

        if "Stage 5" in skip_stages:
            ex_pred = np.zeros_like(break_pred, dtype=bool)
        else:
            if verbose:
                print("[infer] Stage 5: Ex...")
            m5 = self.load_model(5)
            ex_pred = _as_slot_array(
                m5.predict(chart, audio, beat, diff_t, lvl_t, tags_t)[0].cpu().numpy(), T,
            ).astype(bool)

        # ── 构建 simai ──
        if verbose:
            print("[infer] 构建 simai 谱面...")
        chart_np = chart[0].cpu().numpy()
        measures: dict[int, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))

        note_count = 0
        hold_count = 0
        slide_count = 0
        tap_count = 0
        break_count = 0
        ex_count = 0

        for f in range(T):
            tid = int(chart_np[f])
            if tid <= 0:
                continue
            tok_str = self.id_to_token.get(tid)
            if tok_str is None:
                continue

            m, bi = _output_grid_index(f, fr, measure_dur, subdiv)

            hold_slot = 0
            slide_slot = 0
            frame_objects: list[tuple[SimaiToken, int]] = []

            for obj_slot, part in enumerate(tok_str.split("+")):
                st = SimaiToken.from_string(part)
                if st is None:
                    continue

                # 注入 hold 持续时间
                if st.token_type == SimaiTokenType.HOLD:
                    if hold_slot < hold_durs.shape[1] and hold_durs[f, hold_slot] > 0:
                        st.params["dur"] = _duration_bin_to_str(int(hold_durs[f, hold_slot]))
                    elif "dur" not in st.params or not st.params["dur"]:
                        st.params["dur"] = "4:1"
                    hold_slot += 1
                    hold_count += 1

                # 注入 slide 路径 + 持续时间
                if st.token_type == SimaiTokenType.SLIDE:
                    pid = int(slide_paths[f, slide_slot]) if slide_slot < slide_paths.shape[1] else 0
                    if pid > 1:
                        seg = self.slide_vocab_inv.get(pid, "")
                        if seg and seg not in ("<PAD>", "<EOS>"):
                            path, timing = _slide_vocab_token_to_params(seg)
                            if self._validate_slide_path(st.position, path):
                                st.params["path"] = path
                                if timing:
                                    st.params["dur"] = timing
                    if "path" not in st.params or not st.params["path"]:
                        st.params["path"] = _default_slide_path(st.position)
                    if "dur" not in st.params or not st.params["dur"]:
                        path_key = st.params.get("path", "")
                        if path_key and path_key in self.path_best_timing:
                            st.params["dur"] = self.path_best_timing[path_key]
                        elif hold_slot < hold_durs.shape[1] and hold_durs[f, hold_slot] > 0:
                            st.params["dur"] = _duration_bin_to_str(int(hold_durs[f, hold_slot]))
                        else:
                            st.params["dur"] = "4:1"
                    slide_slot += 1
                    slide_count += 1

                # TouchHold (hold A-E buttons) — 也注入持续时间
                if st.token_type == SimaiTokenType.HOLD and re.match(r"^[A-E]", st.position):
                    if hold_slot < hold_durs.shape[1] and hold_durs[f, hold_slot] > 0:
                        st.params["dur"] = _duration_bin_to_str(int(hold_durs[f, hold_slot]))
                    elif "dur" not in st.params or not st.params["dur"]:
                        st.params["dur"] = "4:1"
                    hold_slot += 1

                # 注入 break/ex
                if obj_slot < break_pred.shape[1] and break_pred[f, obj_slot]:
                    st.params["break"] = ""
                if obj_slot < ex_pred.shape[1] and ex_pred[f, obj_slot]:
                    st.params["ex"] = ""

                frame_objects.append((st, obj_slot))

            frame_notes = []
            for st, _ in frame_objects:
                if st.token_type == SimaiTokenType.TAP:
                    tap_count += len(st.position)
                if st.has_break:
                    break_count += 1
                if st.has_ex:
                    ex_count += 1
                frame_notes.append(note_to_simai(st))
                note_count += len(st.position) if st.token_type == SimaiTokenType.TAP else 1

            if frame_notes:
                measures[m][bi].append("/".join(frame_notes))

        # ── 写 simai 文件 ──
        title = Path(mp3_path).parent.name
        lines = [
            f"&title={title}",
            f"&artist={designer}",
            f"&wholebpm={bpm:.1f}",
            f"&lv_{diff_num}={level:.1f}",
            f"&des_{diff_num}={designer}",
            f"&inote_{diff_num}=",
        ]
        max_m = max(measures.keys()) if measures else 0
        for m_idx in range(max_m + 1):
            beats = measures.get(m_idx, {})
            parts = []
            for bi_idx in range(subdiv):
                if bi_idx in beats:
                    parts.append("/".join(beats[bi_idx]))
                else:
                    parts.append("")
            body = ",".join(parts) + ","
            if m_idx == 0:
                lines.append(f"({bpm:.1f}){{{subdiv}}}{body}")
            else:
                lines.append(f"{{{subdiv}}}{body}")
        lines.append("E")

        simai_text = "\n".join(lines)

        # 统计信息
        info = (
            f"✅ 生成完成！\n\n"
            f"📊 统计信息:\n"
            f"  - 总音符数: {note_count}\n"
            f"  - Tap: {tap_count} | Hold: {hold_count} | Slide: {slide_count}\n"
            f"  - Break: {break_count} | Ex: {ex_count}\n"
            f"  - 三押过滤: {'开启' if filter_multi_tap else '关闭'}"
            f" (vocab {len(self._multi_tap_ids)} 个, 过滤 {filtered_multi_tap_tokens} token/"
            f"{filtered_multi_tap_count} tap)\n"
            f"  - WiFi bias: {wifi_bias:.2f} (vocab {len([1 for _ in self.bias_wifi_slide if _ > 0])} 个)\n"
            f"  - 跳过: {', '.join(sorted(skip_stages)) if skip_stages else '无'}\n"
            f"  - 小节数: {max_m + 1} | 帧数: {T}\n"
            f"  - BPM: {bpm:.1f} | 难度: {difficulty} {level:.1f}\n"
            f"  - 设备: {self.device}\n"
        )

        return simai_text, info

    # ═══════════════════════════════════════════════════════════
    # 批量推理便捷方法: 返回 (body, bpm, note_count)
    # ═══════════════════════════════════════════════════════════

    @torch.no_grad()
    def generate_chart_body(
        self,
        mp3_path: str,
        difficulty: str,
        level: float,
        designer: str,
        collections: list[str],
        temperature: float,
        top_k: int,
        bpm_override: float,
        density: float,
        tap_bias: float,
        hold_bias: float,
        slide_bias: float,
        wifi_bias: float,
        touch_bias: float,
        touchhold_bias: float,
        break_bias: float,
        filter_multi_tap: bool,
        skip_stages: list[str] | None = None,
        audio_ctx: AudioContext | None = None,
        allow_touch: bool = True,
    ) -> tuple[str, float, int] | None:
        """
        批量推理版本: 返回 (simai_body, bpm, note_count)，不含头部元数据。

        头部由 batch_infer 的 merge_multi_difficulty() 统一构建。
        """
        skip_stages = set(skip_stages or [])
        diff_id = self.DIFF_ID.get(difficulty, 4)
        cfg = self.cfg
        device = self.device

        # ── 音频 ──
        if audio_ctx is not None:
            fr = audio_ctx.frame_rate
            nf = audio_ctx.num_frames
            bpm = audio_ctx.bpm
            measure_dur = audio_ctx.measure_dur
            ad_tokens = audio_ctx.tokens
            beat_signal = audio_ctx.beat_signal
        else:
            ctx = self.prepare_audio(mp3_path, bpm_override)
            fr = ctx.frame_rate
            nf = ctx.num_frames
            bpm = ctx.bpm
            measure_dur = ctx.measure_dur
            ad_tokens = ctx.tokens
            beat_signal = ctx.beat_signal

        subdiv = cfg.chart.target_subdiv
        audio = torch.from_numpy(ad_tokens).unsqueeze(0).long().to(device)
        beat = torch.from_numpy(beat_signal).unsqueeze(0).to(device)
        diff_t = torch.tensor([diff_id], device=device)
        lvl_t = torch.tensor([level], device=device)

        # 标签
        max_tags = 32
        tag_ids = [-1] * max_tags
        tag_idx = 0
        diff_tag = f"difficulty:{difficulty}"
        if diff_tag in self.tag_vocab and tag_idx < max_tags:
            tag_ids[tag_idx] = self.tag_vocab[diff_tag]
            tag_idx += 1
        if collections:
            for col in collections:
                if not col or col == "无" or tag_idx >= max_tags:
                    continue
                col_tag = f"collection:{col}"
                if col_tag in self.tag_vocab:
                    tag_ids[tag_idx] = self.tag_vocab[col_tag]
                    tag_idx += 1
        if designer and designer != "AI" and tag_idx < max_tags:
            des_tag = f"designer:{designer}"
            if des_tag in self.tag_vocab:
                tag_ids[tag_idx] = self.tag_vocab[des_tag]
                tag_idx += 1
        tags_t = torch.tensor([tag_ids], dtype=torch.long, device=device)

        # ── Stage 1 ──
        m1 = self.load_model(1)
        chart = m1.generate(
            audio, beat, diff_t, lvl_t, tags_t,
            temperature=temperature,
            top_k=top_k,
            logits_processor=lambda logits, t, generated: self._apply_stage1_constraints(
                self._apply_stage1_bias(
                    logits,
                    density,
                    tap_bias,
                    hold_bias,
                    slide_bias,
                    touch_bias,
                    touchhold_bias,
                    filter_multi_tap,
                    allow_touch=allow_touch,
                ),
                t,
                generated,
                fr,
                measure_dur,
                subdiv,
                filter_multi_tap,
            ),
        )
        T = chart.shape[1]
        if filter_multi_tap:
            chart, _, _ = self._filter_multi_tap_chart(chart, fr, measure_dur, subdiv)

        hold_ids = {tid for tok, tid in self.vocab.items() if tok.startswith("hold")}

        # ── Stage 2: Hold ──
        if "Stage 2" in skip_stages:
            hold_durs = np.zeros((T, 1), dtype=np.int64)
        else:
            m2 = self.load_model(2)
            hold_mask = torch.zeros(1, T, dtype=torch.bool, device=device)
            for hid in hold_ids:
                hold_mask = hold_mask | (chart == hid)
            dur_pred = m2.generate(chart, audio, beat, diff_t, lvl_t, tags_t, hold_mask,
                                   temperature=temperature)
            hold_durs = _as_slot_array(dur_pred[0].cpu().numpy(), T)

        # ── Stage 3: Slide ──
        if "Stage 3" in skip_stages:
            slide_paths = np.zeros((T, 1), dtype=np.int64)
        else:
            m3 = self.load_model(3)
            out3 = m3(chart, audio, beat, diff_t, lvl_t, tags_t)
            slide_logits = out3["logits"][0]
            S = slide_logits.shape[1]
            slide_temp = temperature * 0.7
            slide_topk = max(10, top_k // 2)
            sl = slide_logits / max(slide_temp, 0.01) if slide_temp > 0 else slide_logits.clone()
            if wifi_bias:
                sl = sl + self._mask_like_logits(self.bias_wifi_slide, sl).view(1, 1, -1) * wifi_bias
            # 非法路径屏蔽
            chart_np_for_slide = chart[0].detach().cpu().numpy()
            invalid_cache: dict[str, list[int]] = {}
            for f in range(T):
                tok_str = self.id_to_token.get(int(chart_np_for_slide[f]))
                if not tok_str:
                    continue
                ss = 0
                for part in tok_str.split("+"):
                    st = SimaiToken.from_string(part)
                    if st is None or st.token_type != SimaiTokenType.SLIDE:
                        continue
                    if ss >= sl.shape[1]:
                        break
                    if st.position not in invalid_cache:
                        invalid_cache[st.position] = [
                            p for p in self._invalid_slide_vocab_ids_for_start(st.position)
                            if p < sl.shape[-1]
                        ]
                    inv = invalid_cache[st.position]
                    if inv:
                        sl[f, ss, inv] = float("-inf")
                    ss += 1
            if slide_topk > 0 and slide_topk < sl.shape[-1]:
                topk_vals, _ = torch.topk(sl, slide_topk, dim=-1)
                sl = torch.where(sl < topk_vals[:, :, -1:],
                                 torch.full_like(sl, float("-inf")), sl)
            probs = F.softmax(sl, dim=-1)
            slide_paths = torch.multinomial(probs.reshape(-1, sl.shape[-1]), 1).reshape(T, S).cpu().numpy()

        # ── Stage 4: Break ──
        if "Stage 4" in skip_stages:
            break_pred = np.zeros((T, 1), dtype=bool)
        else:
            m4 = self.load_model(4)
            bl = m4.forward(chart, audio, beat, diff_t, lvl_t, tags_t)["logits"]
            if break_bias:
                bl = bl.clone()
                bl[..., 1] += break_bias
            break_pred = _as_slot_array(bl.argmax(dim=-1)[0].cpu().numpy(), T).astype(bool)

        # ── Stage 5: Ex ──
        if "Stage 5" in skip_stages:
            ex_pred = np.zeros_like(break_pred, dtype=bool)
        else:
            m5 = self.load_model(5)
            ex_pred = _as_slot_array(
                m5.predict(chart, audio, beat, diff_t, lvl_t, tags_t)[0].cpu().numpy(), T,
            ).astype(bool)

        # ── 构建 simai body ──
        chart_np = chart[0].cpu().numpy()
        measures: dict[int, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
        note_count = 0

        for f in range(T):
            tid = int(chart_np[f])
            if tid <= 0:
                continue
            tok_str = self.id_to_token.get(tid)
            if tok_str is None:
                continue

            m, bi = _output_grid_index(f, fr, measure_dur, subdiv)
            hold_slot = 0
            slide_slot = 0
            frame_objects: list[tuple[SimaiToken, int]] = []

            for obj_slot, part in enumerate(tok_str.split("+")):
                st = SimaiToken.from_string(part)
                if st is None:
                    continue

                if st.token_type == SimaiTokenType.HOLD:
                    if hold_slot < hold_durs.shape[1] and hold_durs[f, hold_slot] > 0:
                        st.params["dur"] = _duration_bin_to_str(int(hold_durs[f, hold_slot]))
                    elif "dur" not in st.params or not st.params["dur"]:
                        st.params["dur"] = "4:1"
                    hold_slot += 1

                if st.token_type == SimaiTokenType.SLIDE:
                    pid = int(slide_paths[f, slide_slot]) if slide_slot < slide_paths.shape[1] else 0
                    if pid > 1:
                        seg = self.slide_vocab_inv.get(pid, "")
                        if seg and seg not in ("<PAD>", "<EOS>"):
                            path, timing = _slide_vocab_token_to_params(seg)
                            if self._validate_slide_path(st.position, path):
                                st.params["path"] = path
                                if timing:
                                    st.params["dur"] = timing
                    if "path" not in st.params or not st.params["path"]:
                        st.params["path"] = _default_slide_path(st.position)
                    if "dur" not in st.params or not st.params["dur"]:
                        path_key = st.params.get("path", "")
                        st.params["dur"] = self.path_best_timing.get(path_key, "4:1")
                    slide_slot += 1

                if st.token_type == SimaiTokenType.HOLD and re.match(r"^[A-E]", st.position):
                    if hold_slot < hold_durs.shape[1] and hold_durs[f, hold_slot] > 0:
                        st.params["dur"] = _duration_bin_to_str(int(hold_durs[f, hold_slot]))
                    elif "dur" not in st.params or not st.params["dur"]:
                        st.params["dur"] = "4:1"
                    hold_slot += 1

                if obj_slot < break_pred.shape[1] and break_pred[f, obj_slot]:
                    st.params["break"] = ""
                if obj_slot < ex_pred.shape[1] and ex_pred[f, obj_slot]:
                    st.params["ex"] = ""

                frame_objects.append((st, obj_slot))

            frame_notes = []
            for st, _ in frame_objects:
                sn = note_to_simai(st)
                if sn:
                    frame_notes.append(sn)
                    note_count += len(st.position) if st.token_type == SimaiTokenType.TAP else 1

            if frame_notes:
                measures[m][bi].append("/".join(frame_notes))

        max_m = max(measures.keys()) if measures else 0
        body_lines = []
        for m_idx in range(max_m + 1):
            beats = measures.get(m_idx, {})
            parts = []
            for bi_idx in range(subdiv):
                if bi_idx in beats:
                    parts.append("/".join(beats[bi_idx]))
                else:
                    parts.append("")
            if m_idx == 0:
                body_lines.append(f"({bpm:.1f}){{{subdiv}}}{','.join(parts)}")
            else:
                body_lines.append(f"{{{subdiv}}}{','.join(parts)}")

        simai_body = "\n".join(body_lines) + "\nE"
        return simai_body, bpm, note_count


# ============================================================
# 工厂函数
# ============================================================

def create_engine(cfg) -> InferenceEngine:
    """从 Config 对象创建推理引擎。"""
    return InferenceEngine(cfg)


# ============================================================
# 辅助函数
# ============================================================

def _count_simultaneous_taps(token_str: str) -> int:
    tap_count = 0
    for part in token_str.split("+"):
        st = SimaiToken.from_string(part)
        if st is None:
            continue
        if st.token_type == SimaiTokenType.TAP:
            tap_count += len(st.position)
        elif st.token_type == SimaiTokenType.SLIDE:
            tap_count += 1
        elif st.token_type == SimaiTokenType.HOLD and re.fullmatch(r"\d+", st.position or ""):
            tap_count += 1
    return tap_count


def _has_touch_note(token_str: str) -> bool:
    for part in token_str.split("+"):
        st = SimaiToken.from_string(part)
        if st is None:
            continue
        if st.token_type == SimaiTokenType.TOUCH:
            return True
        if st.token_type == SimaiTokenType.HOLD and re.fullmatch(r"[A-E]\d*", st.position or ""):
            return True
    return False


def _is_wifi_slide_vocab_token(token: str) -> bool:
    path = re.sub(r"\[[^\]]+\]$", "", token)
    return re.search(r"(^|\*)w[1-8]", path) is not None


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


def _slide_first_target(path_str: str) -> tuple[str, int] | None:
    m = re.match(r'^(V)([1-8])([1-8])', path_str)
    if m:
        return m.group(1), int(m.group(3))
    m = re.match(r'^(pp|qq|PP|QQ|[><^vVpqszw-])([1-8])', path_str)
    if m:
        return m.group(1), int(m.group(2))
    return None


def _default_slide_path(start_pos: str) -> str:
    try:
        start = int(start_pos)
    except ValueError:
        return "-4"
    return f"-{((start + 3) % 8) + 1}"


def _slide_vocab_token_to_params(token: str) -> tuple[str, str]:
    m = re.match(r"^(.+)\[([^\]]+)\]$", token)
    if m:
        return m.group(1), m.group(2)
    return token, ""


def _as_slot_array(arr: np.ndarray, length: int, slots: int = 1) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape[0] < length:
        pad = np.zeros((length - arr.shape[0], arr.shape[1]), dtype=arr.dtype)
        arr = np.concatenate([arr, pad], axis=0)
    return arr[:length, :max(slots, arr.shape[1])]
