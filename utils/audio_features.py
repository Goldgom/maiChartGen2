"""
Audio feature extraction for chart generation.

Extracts onset strength and chroma features aligned to chart time slots.
These provide the model with musical context beyond raw EnCodec tokens.

Features per time slot:
  onset_strength : float   — how strong the attack is at this moment
  chroma         : [12]    — 12-dimensional pitch class profile
  spectral_centroid : float — brightness/timbre indicator

Usage:
    from utils.audio_features import extract_features

    features = extract_features("track.mp3", bpm=173, subdiv=64)
    # features.onset   : [T] float32
    # features.chroma  : [T, 12] float32
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np


@dataclass
class AudioFeatures:
    """Per-time-slot audio features aligned to chart grid."""
    onset: np.ndarray          # [T] onset strength per slot
    chroma: np.ndarray         # [T, 12] chroma features per slot
    centroid: np.ndarray       # [T] spectral centroid per slot
    num_slots: int
    bpm: float
    subdiv: int


def extract_features(
    audio_path: Union[str, Path],
    bpm: float,
    subdiv: int = 64,
    sr: int = 22050,
    beat_window_ms: float = 50.0,
) -> AudioFeatures:
    """Extract onset and chroma features aligned to chart grid.

    Args:
        audio_path: Path to audio file (mp3, wav, etc.)
        bpm: Beats per minute
        subdiv: Subdivisions per beat (matches chart preprocessing)
        sr: Analysis sample rate
        beat_window_ms: Window around each sub-beat to sample features (ms)

    Returns:
        AudioFeatures with per-slot onset, chroma, and centroid.
    """
    import librosa
    import soundfile as sf

    # --- Load audio ---
    y, orig_sr = sf.read(str(audio_path), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if orig_sr != sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
    y = np.asarray(y, dtype=np.float32)
    peak = np.abs(y).max()
    if peak > 1e-6:
        y = y / peak

    # --- Time grid ---
    beat_duration = 60.0 / bpm                    # seconds per beat
    slot_duration = beat_duration / subdiv         # seconds per sub-beat
    hop_length = 512
    frames_per_slot = int(sr * slot_duration / hop_length)

    # --- Onset strength ---
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    # --- Chroma ---
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    # chroma: [12, n_frames]

    # --- Spectral centroid ---
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
    centroid = centroid[0]  # [n_frames]

    # --- Align to chart slots ---
    n_frames = len(onset_env)
    # Estimate number of time slots
    total_beats = n_frames * hop_length / sr / beat_duration
    num_slots = int(total_beats * subdiv) + 1

    slot_onset = np.zeros(num_slots, dtype=np.float32)
    slot_chroma = np.zeros((num_slots, 12), dtype=np.float32)
    slot_centroid = np.zeros(num_slots, dtype=np.float32)

    for i in range(num_slots):
        center_frame = int(i * frames_per_slot)
        window_frames = int(beat_window_ms / 1000.0 * sr / hop_length)
        start = max(0, center_frame - window_frames // 2)
        end = min(n_frames, center_frame + window_frames // 2 + 1)

        if start < end:
            slot_onset[i] = float(np.max(onset_env[start:end]))
            slot_chroma[i] = chroma[:, start:end].mean(axis=1)
            slot_centroid[i] = float(np.mean(centroid[start:end]))

    return AudioFeatures(
        onset=slot_onset,
        chroma=slot_chroma,
        centroid=slot_centroid,
        num_slots=num_slots,
        bpm=bpm,
        subdiv=subdiv,
    )


def features_to_tensor(features: AudioFeatures) -> dict[str, "torch.Tensor"]:
    """Convert AudioFeatures to PyTorch tensors for model input."""
    import torch
    return {
        "onset": torch.from_numpy(features.onset).float().unsqueeze(0),   # [1, T]
        "chroma": torch.from_numpy(features.chroma).float().unsqueeze(0), # [1, T, 12]
        "centroid": torch.from_numpy(features.centroid).float().unsqueeze(0), # [1, T]
    }


# ═══════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python utils/audio_features.py <track.mp3> [bpm]")
        sys.exit(1)

    path = sys.argv[1]
    bpm = float(sys.argv[2]) if len(sys.argv) > 2 else 173.0

    print(f"Extracting features: {path} (BPM={bpm})")
    feats = extract_features(path, bpm=bpm, subdiv=64)

    print(f"  Slots: {feats.num_slots}")
    print(f"  Onset range: [{feats.onset.min():.3f}, {feats.onset.max():.3f}]")
    print(f"  Chroma shape: {feats.chroma.shape}")
    print(f"  Centroid range: [{feats.centroid.min():.1f}, {feats.centroid.max():.1f}]")

    # Show first 20 slots
    print(f"\n  First 20 slots:")
    print(f"  {'Slot':<6} {'Onset':<8} {'Chroma(0-3)':<30} {'Centroid':<10}")
    for i in range(min(20, feats.num_slots)):
        c = ", ".join(f"{feats.chroma[i, j]:.2f}" for j in range(4))
        print(f"  {i:<6} {feats.onset[i]:<8.3f} [{c}] {feats.centroid[i]:<10.0f}")
