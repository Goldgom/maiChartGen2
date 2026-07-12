"""
Audio Tokenizer - 使用 EnCodec 神经网络编解码器将音频编码为离散 token

EnCodec (Meta, 2022): 将音频波形压缩为多层离散 token，每层代表不同粒度的音频特征。
  - 帧率: 75Hz @24kHz (每帧对应约 13.3ms)
  - 每帧 N 个 token (N = num_codebooks, 默认 8, 可降为 1-4 降低带宽)
  - 每个 token 值域: 0~1023

用于 maimai 谱面生成时，音频 token 作为模型的输入条件，与谱面 token 对齐。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torchaudio


# ============================================================
# AudioTokenData — 编码结果容器
# ============================================================

@dataclass
class AudioTokenData:
    """音频编码结果"""

    tokens: np.ndarray  # (num_frames, num_codebooks) int array, 值域 0~1023
    sample_rate: int     # 原始音频采样率
    frame_rate: float    # token 帧率 (Hz)
    duration: float      # 音频时长 (秒)
    num_codebooks: int   # 实际 codebook 数量
    original_samples: int  # 原始采样点数
    _group_sizes: list[int] = None  # 内部: 每组的 codebook 数, 供 decode 用

    @property
    def num_frames(self) -> int:
        return self.tokens.shape[0]

    @property
    def frames_per_second(self) -> float:
        return self.frame_rate

    @property
    def ms_per_frame(self) -> float:
        return 1000.0 / self.frame_rate

    def summary(self) -> str:
        return (
            f"AudioTokenData(frames={self.num_frames}, codebooks={self.num_codebooks}, "
            f"frame_rate={self.frame_rate:.1f}Hz, duration={self.duration:.1f}s, "
            f"sr={self.sample_rate}Hz)"
        )

    def __repr__(self) -> str:
        return self.summary()


# ============================================================
# AudioTokenizer
# ============================================================

class AudioTokenizer:
    """使用 EnCodec 进行音频的编码(→token)和解码(→波形)"""

    # HuggingFace 模型 ID (仅当本地不存在时使用)
    MODEL_ID = "facebook/encodec_24khz"

    # 本地模型路径
    LOCAL_MODEL_PATH: Optional[str] = None  # 可设为 premodel 路径

    # 模型原生参数
    NATIVE_SR = 24000          # EnCodec 24kHz 模型原生采样率
    NATIVE_FRAME_RATE = 75.0   # 24kHz 下每秒 75 帧 (320 samples/frame)

    # 带宽 (kbps) → 实际 codebook 数量映射 (24kHz 模型)
    # 模型支持的 bandwidths: [1.5, 3.0, 6.0, 12.0, 24.0] kbps
    BANDWIDTH_TO_CODEBOOKS = {
        1.5: 2,    # 2 codebooks
        3.0: 4,    # 4 codebooks
        6.0: 8,    # 8 codebooks
        12.0: 16,  # 16 codebooks
        24.0: 32,  # 32 codebooks
    }
    # 反向: 允许的 codebook 数 → bandwidth
    CODEBOOKS_TO_BANDWIDTH = {v: k for k, v in BANDWIDTH_TO_CODEBOOKS.items()}

    def __init__(self, num_codebooks: int = 8, device: str = "cpu",
                 local_path: Optional[str] = None):
        """
        Args:
            num_codebooks: codebook 数量 (2/4/8/16/32).
                           越少带宽越低，token 越少，但音质越差。
                           推荐: 8(高保真), 4(平衡), 2(低保真).
            device: 'cpu' 或 'cuda'
            local_path: 本地模型文件夹路径，如 'premodels/encodec_24khz'。
                        设为 None 则从 HuggingFace 下载。
        """
        if num_codebooks not in self.CODEBOOKS_TO_BANDWIDTH:
            raise ValueError(
                f"num_codebooks 必须是 {list(self.CODEBOOKS_TO_BANDWIDTH.keys())} 之一, "
                f"收到 {num_codebooks}"
            )
        self.num_codebooks = num_codebooks
        self.bandwidth = self.CODEBOOKS_TO_BANDWIDTH[num_codebooks]
        self.device = device
        self.local_path = local_path

        self._model = None
        self._loaded = False

    # ---- 模型加载 ----

    @property
    def model(self):
        if not self._loaded:
            self._load_model()
        return self._model

    def _load_model(self):
        """延迟加载 EnCodec 模型（优先本地）"""
        from transformers import EncodecModel

        model_path = self.local_path or self.MODEL_ID
        if self.local_path and Path(self.local_path).is_dir():
            self._model = EncodecModel.from_pretrained(
                self.local_path, local_files_only=True
            ).to(self.device)
        else:
            self._model = EncodecModel.from_pretrained(self.MODEL_ID).to(self.device)
        self._model.eval()
        self._loaded = True

    # ---- 编码 ----

    @torch.no_grad()
    def encode(
        self,
        waveform: Union[np.ndarray, torch.Tensor],
        sr: int,
    ) -> AudioTokenData:
        """将音频波形编码为离散 token

        Args:
            waveform: 音频波形, shape (channels, samples) 或 (samples,)
            sr: 原始采样率 (任意, 会自动重采样到 24kHz)

        Returns:
            AudioTokenData: 编码结果
        """
        # 标准化为 torch tensor (channels, samples)
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform).float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # (samples,) → (1, samples)
        elif waveform.dim() == 2 and waveform.shape[0] > waveform.shape[1]:
            waveform = waveform.T  # (samples, channels) → (channels, samples)

        waveform = waveform.to(self.device)
        original_samples = waveform.shape[-1]

        # 重采样到 24kHz
        if sr != self.NATIVE_SR:
            resampler = torchaudio.transforms.Resample(sr, self.NATIVE_SR).to(self.device)
            waveform = resampler(waveform)

        # 转为 mono (EnCodec 支持 mono 和 stereo, 但 mono 更通用)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # (C, T) → (1, T)

        # EnCodec 编码
        # 输入: (batch, channels, samples), batch=1
        inputs = waveform.unsqueeze(0)  # (1, C, T)

        encoded = self.model.encode(
            inputs,
            bandwidth=self.bandwidth,
            return_dict=True,
        )

        # encoded.audio_codes: list of (batch, group_size, frames) tensors
        # 合并所有 group: (total_codebooks, frames)
        group_sizes = []
        all_codes = []
        for code in encoded.audio_codes:
            # code shape: (1, group_size, frames)
            gs = code.shape[1]
            group_sizes.append(gs)
            for c in range(gs):
                all_codes.append(code[0, c, :])  # (frames,)
        tokens = torch.stack(all_codes, dim=0).T.cpu().numpy()  # (frames, total_codebooks)
        actual_codebooks = tokens.shape[1]

        duration = original_samples / sr

        return AudioTokenData(
            tokens=tokens,
            sample_rate=sr,
            frame_rate=self.NATIVE_FRAME_RATE,
            duration=duration,
            num_codebooks=actual_codebooks,
            original_samples=original_samples,
            _group_sizes=group_sizes,
        )

    def encode_file(self, path: Union[str, Path]) -> AudioTokenData:
        """从音频文件加载并编码

        支持格式: mp3, wav, flac, ogg 等 (通过 soundfile/librosa)
        """
        waveform, sr = self._load_audio(path)
        return self.encode(waveform, sr)

    @staticmethod
    def _load_audio(path: Union[str, Path]) -> tuple[np.ndarray, int]:
        """加载音频文件 → (waveform, sample_rate)

        尝试顺序: soundfile → librosa → torchaudio(仅wav)
        """
        path = str(path)

        # 1. soundfile (支持 mp3 需要系统有 libsndfile + mpg123)
        try:
            import soundfile as sf
            data, sr = sf.read(path, dtype="float32")
            if data.ndim == 1:
                data = data.reshape(1, -1)
            else:
                data = data.T  # (samples, channels) → (channels, samples)
            return data, sr
        except Exception:
            pass

        # 2. librosa
        try:
            import librosa
            data, sr = librosa.load(path, sr=None, mono=False)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            return data, sr
        except Exception:
            pass

        # 3. torchaudio (仅 wav 等原生格式)
        try:
            import torchaudio
            waveform, sr = torchaudio.load(str(path))
            return waveform.numpy(), sr
        except Exception:
            pass

        raise RuntimeError(f"无法加载音频文件: {path}")

    # ---- 解码 ----

    @torch.no_grad()
    def decode(self, data: AudioTokenData) -> tuple[np.ndarray, int]:
        """将 token 解码回音频波形"""
        if not self._loaded:
            self._load_model()

        # tokens shape: (frames, num_codebooks) → 按 group_sizes 拆分
        tokens_t = torch.from_numpy(data.tokens.T).long().to(self.device)  # (codebooks, frames)

        group_sizes = data._group_sizes
        if group_sizes is None:
            # 回退: 假设每组 2 个 codebook
            nc = data.num_codebooks
            if nc <= 2:
                group_sizes = [nc]
            else:
                group_sizes = [2] * (nc // 2)

        audio_codes = []
        idx = 0
        for gs in group_sizes:
            group = tokens_t[idx:idx + gs]  # (gs, frames)
            audio_codes.append(group.unsqueeze(0))  # (1, gs, frames)
            idx += gs

        decoded = self.model.decode(audio_codes, [None], return_dict=True)
        waveform = decoded.audio_values.squeeze(0).cpu().numpy()  # (channels, samples)
        return waveform, self.NATIVE_SR

    # ---- 辅助 ----

    def frame_to_time(self, frame_idx: int) -> float:
        """帧索引 → 时间 (秒)"""
        return frame_idx / self.NATIVE_FRAME_RATE

    def time_to_frame(self, seconds: float) -> int:
        """时间 (秒) → 最近的帧索引"""
        return round(seconds * self.NATIVE_FRAME_RATE)

    def __repr__(self) -> str:
        return f"AudioTokenizer(codebooks={self.num_codebooks}, device={self.device})"


# ============================================================
# 便捷函数
# ============================================================

def encode_audio(path: str | Path, num_codebooks: int = 8) -> AudioTokenData:
    """便捷函数: 加载音频文件并编码为 token"""
    tokenizer = AudioTokenizer(num_codebooks=num_codebooks)
    return tokenizer.encode_file(path)


def decode_audio(data: AudioTokenData) -> tuple[np.ndarray, int]:
    """便捷函数: 将 token 解码为波形"""
    tokenizer = AudioTokenizer(num_codebooks=data.num_codebooks)
    return tokenizer.decode(data)
