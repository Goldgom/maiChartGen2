"""
BeatTokenizer - 音频节拍检测与节奏列表生成

使用 librosa (默认) 或 beat_this 检测音频节拍，生成结构化的节拍列表。
节拍列表可用于与谱面 token 对齐，辅助谱面生成模型理解音乐节奏。

输出: 每拍的时间戳、拍内位置、是否为重拍 (downbeat)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np


# ============================================================
# BeatEvent — 单个节拍事件
# ============================================================

@dataclass
class BeatEvent:
    """一个节拍事件"""

    time: float          # 时间戳 (秒)
    beat_index: int      # 全局拍序号 (从 0 开始)
    measure: int         # 小节序号 (从 0 开始)
    beat_in_measure: int # 小节内拍位 (0-based, 0=重拍)
    is_downbeat: bool    # 是否为重拍 (每小节第一拍)
    bpm: float           # 当前 BPM

    def __repr__(self) -> str:
        db = "▼" if self.is_downbeat else "·"
        return (f"Beat(m{self.measure:03d}.{self.beat_in_measure} "
                f"t={self.time:.3f}s bpm={self.bpm:.1f} {db})")


@dataclass
class BeatList:
    """完整的节拍列表"""

    beats: list[BeatEvent] = field(default_factory=list)
    bpm: float = 120.0              # 整体 BPM (或平均)
    time_signature: int = 4         # 拍号分子
    duration: float = 0.0           # 音频时长 (秒)
    method: str = "librosa"         # 检测方法
    sample_rate: int = 22050        # 分析采样率

    @property
    def num_beats(self) -> int:
        return len(self.beats)

    @property
    def num_measures(self) -> int:
        if not self.beats:
            return 0
        return self.beats[-1].measure + 1

    @property
    def beat_interval(self) -> float:
        """平均拍间隔 (秒)"""
        return 60.0 / self.bpm if self.bpm > 0 else 0.5

    def summary(self) -> str:
        return (
            f"BeatList(beats={self.num_beats}, measures={self.num_measures}, "
            f"bpm={self.bpm:.1f}, ts={self.time_signature}/4, "
            f"duration={self.duration:.1f}s, method={self.method})"
        )

    def get_beats_in_range(self, start_time: float, end_time: float) -> list[BeatEvent]:
        """获取指定时间范围内的节拍"""
        return [b for b in self.beats if start_time <= b.time < end_time]

    def get_measure(self, measure_idx: int) -> list[BeatEvent]:
        """获取指定小节的节拍"""
        return [b for b in self.beats if b.measure == measure_idx]

    def to_array(self) -> np.ndarray:
        """转为 (N, 2) 数组: [time, bpm]"""
        return np.array([[b.time, b.bpm] for b in self.beats])

    def __repr__(self) -> str:
        return self.summary()


# ============================================================
# BeatTokenizer
# ============================================================

class BeatTokenizer:
    """音频节拍检测器

    支持两种后端:
      - librosa: 经典动态规划节拍跟踪 (默认, 稳定)
      - beat_this: 深度学习节拍跟踪 (需额外安装, 更准)
    """

    def __init__(
        self,
        method: str = "librosa",
        bpm_min: float = 60.0,
        bpm_max: float = 240.0,
        tightness: float = 50.0,
        time_signature: int = 4,
        downbeat_weight: float = 1.5,
        quantize_beats: bool = True,
        sample_rate: int = 22050,
        target_bpm: Optional[float] = None,
        beat_this_ckpt: Optional[str] = None,
        beat_this_config: Optional[str] = None,
    ):
        """
        Args:
            method: 'librosa' 或 'beat_this'
            bpm_min: 检测 BPM 下限
            bpm_max: 检测 BPM 上限
            tightness: librosa 紧密度 (0-100)
            time_signature: 每小节拍数
            downbeat_weight: 重拍权重倍数
            quantize_beats: 是否量化到均匀网格
            sample_rate: 分析用采样率
            target_bpm: 已知的参考 BPM (如谱面标注), 用于纠正半速/双倍速误检
            beat_this_ckpt: beat_this 权重文件路径 (.safetensors 或 .pt)
            beat_this_config: beat_this 模型配置文件 (.json)
        """
        self.method = method
        self.bpm_min = bpm_min
        self.bpm_max = bpm_max
        self.tightness = tightness
        self.time_signature = time_signature
        self.downbeat_weight = downbeat_weight
        self.quantize_beats = quantize_beats
        self.sample_rate = sample_rate
        self.target_bpm = target_bpm
        self.beat_this_ckpt = beat_this_ckpt
        self.beat_this_config = beat_this_config
        self._bt_model = None  # 缓存的 beat_this 模型

    def analyse(self, audio_path: Union[str, Path]) -> BeatList:
        """分析音频文件，返回节拍列表"""
        waveform, sr = self._load_audio(audio_path)
        return self.analyse_waveform(waveform, sr, Path(audio_path).name)

    def analyse_waveform(
        self, waveform: np.ndarray, sr: int, name: str = ""
    ) -> BeatList:
        """分析音频波形，返回节拍列表"""
        if self.method == "beat_this":
            return self._analyse_beatthis(waveform, sr, name)
        else:
            try:
                return self._analyse_librosa(waveform, sr, name)
            except ImportError:
                # librosa 不可用，回退到基于 BPM 的等间距网格
                return self._analyse_fallback(waveform, sr, name)

    # ---- librosa 后端 ----

    def _analyse_librosa(
        self, waveform: np.ndarray, sr: int, name: str
    ) -> BeatList:
        try:
            import librosa
        except ImportError as e:
            raise ImportError(f"librosa 导入失败: {e}\n请运行: pip install librosa")

        # 转为 mono 并重采样
        if waveform.ndim > 1:
            y = waveform.mean(axis=0)
        else:
            y = waveform
        if sr != self.sample_rate:
            try:
                import librosa as _lr
            except ImportError as e:
                raise ImportError(f"librosa 导入失败: {e}\n请运行: pip install librosa")
            y = _lr.resample(y, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate

        duration = len(y) / sr

        # 节拍跟踪
        tempo, beat_frames = librosa.beat.beat_track(
            y=y,
            sr=sr,
            onset_envelope=None,
            hop_length=512,
            start_bpm=(self.bpm_min + self.bpm_max) / 2,
            tightness=self.tightness,
            trim=True,
            bpm=None,
            units="frames",
        )

        if isinstance(tempo, np.ndarray):
            bpm = float(tempo[0]) if len(tempo) > 0 else 120.0
        else:
            bpm = float(tempo)

        # 若有已知 BPM (如谱面标注), 直接使用, 仅用检测结果对齐拍位
        if self.target_bpm is not None and self.target_bpm > 0:
            bpm = self.target_bpm

        # 帧 → 时间
        hop_length = 512
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)

        # 量化到均匀网格
        if self.quantize_beats and len(beat_times) > 1:
            beat_times = self._quantize_beat_times(beat_times, bpm)

        # 检测重拍 (downbeat): 每小节第一拍
        beats = self._build_beat_events(beat_times, bpm)

        return BeatList(
            beats=beats,
            bpm=bpm,
            time_signature=self.time_signature,
            duration=duration,
            method="librosa",
            sample_rate=sr,
        )

    # ---- 回退: BPM 等间距网格 ----

    def _analyse_fallback(
        self, waveform: np.ndarray, sr: int, name: str
    ) -> BeatList:
        """librosa 不可用时，用 BPM 生成等间距节拍网格"""
        if waveform.ndim > 1:
            duration = waveform.shape[1] / sr
        else:
            duration = len(waveform) / sr

        bpm = self.target_bpm if self.target_bpm else 120.0
        beat_interval = 60.0 / bpm
        n_beats = int(duration / beat_interval) + 1

        beats = []
        for i in range(n_beats):
            t = i * beat_interval
            measure = i // self.time_signature
            beat_in_measure = i % self.time_signature
            beats.append(BeatEvent(
                time=t,
                beat_index=i,
                measure=measure,
                beat_in_measure=beat_in_measure,
                is_downbeat=(beat_in_measure == 0),
                bpm=bpm,
            ))

        return BeatList(
            beats=beats,
            bpm=bpm,
            time_signature=self.time_signature,
            duration=duration,
            method="fallback_bpm",
            sample_rate=sr,
        )

    # ---- beat_this 后端 ----

    def _analyse_beatthis(
        self, waveform: np.ndarray, sr: int, name: str
    ) -> BeatList:
        """使用本地 beat_this 模型进行节拍检测"""
        import json, warnings

        try:
            from beat_this.model.beat_tracker import BeatThis
            from beat_this.model.postprocessor import Postprocessor
        except ImportError:
            raise ImportError("beat_this 未安装。pip install beat-this")

        # 加载模型 (缓存)
        if self._bt_model is None:
            self._load_beatthis_model()

        # 转为 mono + 22050Hz
        if waveform.ndim > 1:
            y = waveform.mean(axis=0)
        else:
            y = waveform
        if sr != 22050:
            try:
                import librosa as _lr
            except ImportError as e:
                raise ImportError(f"librosa 导入失败: {e}\n请运行: pip install librosa")
            y = _lr.resample(y, orig_sr=sr, target_sr=22050)
            sr = 22050

        y = y.astype("float32")
        duration = len(y) / sr
        sr_int = int(sr)

        # fps 从模型配置读取, 默认 50 → hop_length = sr / fps
        fps = getattr(self, "_bt_fps", 50)
        hop_length = int(sr_int / fps)

        # 计算 mel spectrogram
        import torch
        import torchaudio

        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr_int,
            n_fft=2048,
            hop_length=hop_length,
            n_mels=128,
            f_min=30,
            f_max=17000,
            power=1,
        )
        y_t = torch.from_numpy(y).float()
        spec = mel_transform(y_t).numpy()  # (n_mels, frames)

        # 模型推理
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._bt_model.to(device)
        spec_t = torch.from_numpy(spec).T.unsqueeze(0).to(device)  # (1, time, freq)

        with torch.no_grad():
            output = self._bt_model(spec_t)
            if isinstance(output, dict):
                beats_logits = output["beat"].detach()
                downs_logits = output["downbeat"].detach()
            else:
                beats_logits, downs_logits = output

        # 后处理 → 时间戳 (返回 tuple of 1 array for single input)
        postproc = Postprocessor(fps=fps)
        beat_raw, down_raw = postproc(beats_logits, downs_logits)
        beat_times_list = beat_raw[0].tolist() if isinstance(beat_raw, tuple) else beat_raw.tolist()
        downbeat_set = set(down_raw[0].tolist()) if isinstance(down_raw, tuple) else set(down_raw.tolist())

        bpm = self.target_bpm or 120.0
        if self.target_bpm is None and len(beat_times_list) > 1:
            intervals = np.diff(beat_times_list)
            median_interval = float(np.median(intervals))
            if median_interval > 0:
                bpm = 60.0 / median_interval

        beats = self._build_beat_events(beat_times_list, bpm, downbeat_set)

        return BeatList(
            beats=beats, bpm=bpm, time_signature=self.time_signature,
            duration=duration, method="beat_this", sample_rate=sr,
        )

    def _load_beatthis_model(self):
        """从 .ckpt 文件加载 beat_this 模型"""
        import json
        import torch
        from beat_this.model.beat_tracker import BeatThis

        ckpt_path = self.beat_this_ckpt or "premodels/beatthis.ckpt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # 优先从 hyper_parameters 获取配置
        hp = ckpt.get("hyper_parameters", {})
        config = {
            "spect_dim": hp.get("spect_dim", 128),
            "transformer_dim": hp.get("transformer_dim", 512),
            "ff_mult": hp.get("ff_mult", 4),
            "n_layers": hp.get("n_layers", 6),
            "head_dim": hp.get("head_dim", 32),
            "stem_dim": hp.get("stem_dim", 32),
            "dropout": hp.get("dropout", {"frontend": 0.1, "transformer": 0.2}),
            "sum_head": hp.get("sum_head", True),
            "partial_transformers": hp.get("partial_transformers", True),
        }
        # 保存 fps 用于 spectrogram 计算
        self._bt_fps = hp.get("fps", 50)

        model = BeatThis(**config)
        state_dict = ckpt.get("state_dict", ckpt)

        # 去掉 "model." 前缀
        state_dict = {
            k[6:] if k.startswith("model.") else k: v
            for k, v in state_dict.items()
        }

        model.load_state_dict(state_dict, strict=False)
        model.eval()
        self._bt_model = model

    def _load_audio(self, path: Union[str, Path]) -> tuple[np.ndarray, int]:
        """加载音频文件"""
        path = str(path)
        try:
            import soundfile as sf
            data, sr = sf.read(path, dtype="float32")
            if data.ndim == 1:
                data = data.reshape(1, -1)
            else:
                data = data.T
            return data, sr
        except Exception:
            pass
        try:
            import librosa
        except ImportError as e:
            raise ImportError(f"librosa 导入失败: {e}\n请运行: pip install librosa")
        try:
            data, sr = librosa.load(path, sr=None, mono=False)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            return data, sr
        except Exception:
            pass
        raise RuntimeError(f"无法加载音频: {path}")

    def _quantize_beat_times(
        self, beat_times: np.ndarray, bpm: float
    ) -> np.ndarray:
        """将检测到的节拍量化到均匀 BPM 网格"""
        beat_interval = 60.0 / bpm
        n_beats = len(beat_times)

        # 用第一个拍和 BPM 生成理想网格
        start = beat_times[0]
        ideal = start + np.arange(n_beats) * beat_interval

        # 混合: 实际检测 + 理想网格 (防止漂移)
        alpha = 0.3  # 量化强度 (0=纯检测, 1=纯网格)
        quantized = (1 - alpha) * beat_times + alpha * ideal
        return quantized

    def _build_beat_events(
        self,
        beat_times: np.ndarray,
        bpm: float,
        downbeats: Optional[np.ndarray] = None,
    ) -> list[BeatEvent]:
        """构建 BeatEvent 列表"""
        ts = self.time_signature
        downbeat_set = set(downbeats) if downbeats is not None else set()

        beats = []
        for i, t in enumerate(beat_times):
            measure = i // ts
            beat_in_measure = i % ts

            # 判断重拍
            if downbeats is not None:
                is_downbeat = t in downbeat_set or beat_in_measure == 0
            else:
                is_downbeat = (beat_in_measure == 0)

            beats.append(BeatEvent(
                time=float(t),
                beat_index=i,
                measure=measure,
                beat_in_measure=beat_in_measure,
                is_downbeat=is_downbeat,
                bpm=bpm,
            ))

        return beats

    def __repr__(self) -> str:
        return (
            f"BeatTokenizer(method={self.method}, bpm_range=[{self.bpm_min},{self.bpm_max}], "
            f"tightness={self.tightness}, ts={self.time_signature}/4)"
        )


# ============================================================
# 便捷函数
# ============================================================

def analyse_beats(
    audio_path: Union[str, Path],
    method: str = "librosa",
    **kwargs,
) -> BeatList:
    """快速分析音频节拍"""
    tokenizer = BeatTokenizer(method=method, **kwargs)
    return tokenizer.analyse(audio_path)


def beatlist_to_tokens(beatlist: BeatList) -> list[str]:
    """将 BeatList 转为 token 字符串列表

    格式:
      bpm<value>   — BPM 标记
      beat          — 普通拍
      downbeat      — 重拍 (每小节第一拍)
    """
    tokens = [f"bpm{beatlist.bpm:.1f}"]
    for b in beatlist.beats:
        tokens.append("downbeat" if b.is_downbeat else "beat")
    return tokens
