"""
BPMDetector — Automatic BPM detection for maimai audio tracks.

Uses onset-strength autocorrelation (librosa) as primary method
with double/half BPM disambiguation via perceptual heuristics.

Typical maimai BPM range: 50–300, most common: 120–200.

Usage:
    from utils.bpm_detector import BPMDetector

    detector = BPMDetector()
    bpm = detector.detect("track.mp3")
    # → BPMResult(bpm=173.0, confidence=0.92, method='autocorrelation')

    # Batch detection
    results = detector.detect_batch(["track1.mp3", "track2.mp3"])

    # CLI: python utils/bpm_detector.py track.mp3
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

# Maimai typical BPM range
MIN_BPM = 30
MAX_BPM = 400

# Common target BPMs that appear in maimai charts
# When ambiguous, prefer values in this list
MAIMAI_COMMON_BPM = {
    50, 60, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125,
    128, 130, 132, 134, 135, 136, 138, 140, 142, 144, 145, 146, 148,
    150, 152, 154, 155, 156, 158, 160, 162, 164, 165, 168, 170, 171,
    172, 173, 174, 175, 176, 178, 180, 182, 184, 185, 186, 188, 190,
    192, 194, 195, 196, 198, 200, 202, 204, 205, 208, 210, 212, 214,
    215, 216, 218, 220, 222, 224, 225, 228, 230, 232, 234, 235, 236,
    238, 240, 244, 245, 246, 248, 250, 252, 255, 256, 260, 264, 268,
    270, 272, 274, 276, 278, 280, 284, 288, 290, 292, 294, 296, 300,
}

# SR for analysis (lower = faster, librosa default is 22050)
ANALYSIS_SR = 22050

# ═══════════════════════════════════════════════════════════════════════
# Result type
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class BPMResult:
    """Single BPM detection result."""
    bpm: float
    confidence: float          # 0.0 – 1.0
    method: str                # detection method used
    candidates: list[float] = field(default_factory=list)  # other BPM candidates
    raw_peaks: list[float] = field(default_factory=list)    # raw peak BPMs found


# ═══════════════════════════════════════════════════════════════════════
# Audio loading
# ═══════════════════════════════════════════════════════════════════════

def _load_audio(path: Union[str, Path], sr: int = ANALYSIS_SR) -> np.ndarray:
    """Load audio file, convert to mono, resample. Returns float32 [samples]."""
    import soundfile as sf

    data, orig_sr = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)  # stereo → mono

    # Resample if needed
    if orig_sr != sr and sr > 0:
        try:
            import librosa
            data = librosa.resample(data, orig_sr=orig_sr, target_sr=sr)
        except ImportError:
            # Fallback: simple decimation
            from scipy import signal
            data = signal.resample(data, int(len(data) * sr / orig_sr))

    # Ensure float32
    data = np.asarray(data, dtype=np.float32)
    # Normalize
    peak = np.abs(data).max()
    if peak > 1e-6:
        data = data / peak

    return data


# ═══════════════════════════════════════════════════════════════════════
# BPM detection methods
# ═══════════════════════════════════════════════════════════════════════

def _detect_autocorrelation(
    y: np.ndarray,
    sr: int,
    min_bpm: int = MIN_BPM,
    max_bpm: int = MAX_BPM,
) -> BPMResult:
    """Detect BPM using onset-strength autocorrelation.

    This is the primary method — fast, reliable for EDM/pop.
    """
    import librosa

    # Compute onset strength envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)

    # Autocorrelation of onset envelope
    ac = np.correlate(onset_env, onset_env, mode="full")
    ac = ac[len(ac) // 2:]  # take positive lags only

    # Convert lag indices to BPM values
    # lag = sr * 60 / (bpm * hop_length)
    # → bpm = sr * 60 / (lag * hop_length)
    hop_length = 512
    bpm_vals = sr * 60.0 / (np.arange(1, len(ac)) * hop_length)

    # Restrict to BPM range
    valid = (bpm_vals >= min_bpm) & (bpm_vals <= max_bpm)
    bpm_vals = bpm_vals[valid]
    ac_valid = ac[1:][valid]  # skip lag=0

    if len(ac_valid) == 0:
        return BPMResult(bpm=120.0, confidence=0.0, method="autocorrelation")

    # Weight by perceptual preference: slightly prefer 120-200 range
    weights = np.ones_like(bpm_vals)
    preferred = (bpm_vals >= 100) & (bpm_vals <= 220)
    weights[preferred] = 1.3  # mild boost for common range

    # Find peaks
    from scipy.signal import find_peaks

    peaks, props = find_peaks(ac_valid * weights, height=0, distance=3)
    if len(peaks) == 0:
        # Fallback: max value
        best_idx = np.argmax(ac_valid * weights)
        bpm = float(bpm_vals[best_idx])
        return BPMResult(bpm=bpm, confidence=0.3, method="autocorrelation",
                         candidates=[bpm])

    # Sort peaks by height (strongest first)
    peak_order = np.argsort(props["peak_heights"])[::-1]
    candidates = [float(bpm_vals[peaks[i]]) for i in peak_order[:10]]
    raw_peaks = list(candidates)

    # Disambiguate double/half BPM
    best_bpm, confidence = _disambiguate_bpm(candidates, onset_env, sr, hop_length)

    return BPMResult(
        bpm=best_bpm,
        confidence=confidence,
        method="autocorrelation",
        candidates=candidates[:5],
        raw_peaks=raw_peaks[:10],
    )


def _detect_tempogram(
    y: np.ndarray,
    sr: int,
    min_bpm: int = MIN_BPM,
    max_bpm: int = MAX_BPM,
) -> BPMResult:
    """Detect BPM using librosa tempogram (Fourier-based)."""
    import librosa

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    tempogram = librosa.feature.tempogram(
        onset_envelope=onset_env, sr=sr, hop_length=512
    )
    # Aggregate over time (mean)
    tempo_curve = tempogram.mean(axis=1)

    # Get BPM axis
    bpm_bins = librosa.tempo_frequencies(len(tempo_curve), hop_length=512, sr=sr)
    valid = (bpm_bins >= min_bpm) & (bpm_bins <= max_bpm)

    if not valid.any():
        return BPMResult(bpm=120.0, confidence=0.0, method="tempogram")

    bpm_vals = bpm_bins[valid]
    strengths = tempo_curve[valid]

    # Weighted average of top peaks
    from scipy.signal import find_peaks
    peaks, props = find_peaks(strengths, height=0, distance=2)
    if len(peaks) == 0:
        best_idx = np.argmax(strengths)
        return BPMResult(bpm=float(bpm_vals[best_idx]), confidence=0.3,
                         method="tempogram")

    peak_order = np.argsort(props["peak_heights"])[::-1]
    candidates = [float(bpm_vals[peaks[i]]) for i in peak_order[:5]]

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    best_bpm, confidence = _disambiguate_bpm(candidates, onset_env, sr, 512)

    return BPMResult(
        bpm=best_bpm,
        confidence=confidence,
        method="tempogram",
        candidates=candidates,
    )


def _detect_beat_track(
    y: np.ndarray,
    sr: int,
    min_bpm: int = MIN_BPM,
    max_bpm: int = MAX_BPM,
) -> BPMResult:
    """Detect BPM using librosa's beat tracker directly."""
    import librosa

    tempo, beats = librosa.beat.beat_track(
        y=y, sr=sr, start_bpm=140.0, tightness=100
    )

    if isinstance(tempo, np.ndarray):
        tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
    else:
        tempo = float(tempo)

    if tempo < min_bpm:
        tempo *= 2
    elif tempo > max_bpm:
        tempo /= 2

    confidence = min(1.0, len(beats) / 100.0) if beats is not None else 0.5

    return BPMResult(
        bpm=tempo,
        confidence=confidence,
        method="beat_track",
        candidates=[tempo],
    )


# ═══════════════════════════════════════════════════════════════════════
# Disambiguation
# ═══════════════════════════════════════════════════════════════════════

def _disambiguate_bpm(
    candidates: list[float],
    onset_env: np.ndarray,
    sr: int,
    hop_length: int,
) -> tuple[float, float]:
    """Choose the most likely BPM among candidates.

    Strategy:
      1. Score each candidate by beat-grid onset strength
      2. Apply harmonic penalties (double/half, 3/2, 4/3)
      3. Favor BPMs near common maimai values
      4. Favor BPMs in typical range (100-200)
    """
    if not candidates:
        return 120.0, 0.0

    # Compute beat-grid onset strength for each candidate
    strengths = []
    for bpm in candidates:
        beat_period = int(sr * 60.0 / (bpm * hop_length))
        if beat_period <= 0:
            strengths.append(0.0)
            continue
        n_beats = min(64, len(onset_env) // max(1, beat_period))
        if n_beats < 4:
            strengths.append(0.0)
            continue
        sampled = onset_env[np.arange(n_beats) * beat_period]
        strengths.append(float(np.mean(sampled)))

    if not strengths or max(strengths) <= 0:
        return candidates[0], 0.3

    max_s = max(strengths)
    norm_s = [s / max_s for s in strengths]

    # The strongest peak
    top_bpm = candidates[0]

    # Score each candidate
    scores = []
    for i, (bpm, ns) in enumerate(zip(candidates, norm_s)):
        score = ns

        # Range preference: typical maimai BPM is 100-200
        if 100 <= bpm <= 200:
            score *= 1.2
        elif 60 <= bpm <= 250:
            score *= 1.0
        else:
            score *= 0.7

        # Harmonic penalty relative to strongest peak
        if i > 0 and top_bpm > 0:
            ratio = bpm / top_bpm
            # Double/half
            if 0.48 < ratio < 0.52 or 1.95 < ratio < 2.05:
                score *= 0.55
            # 3/2 or 2/3
            elif 0.65 < ratio < 0.68 or 1.48 < ratio < 1.52:
                score *= 0.7
            # 4/3 or 3/4
            elif 0.73 < ratio < 0.77 or 1.30 < ratio < 1.36:
                score *= 0.75

        # Proximity to common maimai BPM (exact or near match)
        bpm_rounded = round(bpm)
        for common in MAIMAI_COMMON_BPM:
            if abs(bpm - common) < 1.0:
                score *= 1.08
                break
            elif abs(bpm - common) < 3.0:
                score *= 1.03
                break

        scores.append(score)

    best_idx = int(np.argmax(scores))
    confidence = min(1.0, scores[best_idx] * 0.85)

    return candidates[best_idx], confidence


# ═══════════════════════════════════════════════════════════════════════
# Ensemble
# ═══════════════════════════════════════════════════════════════════════

def _ensemble_results(results: list[BPMResult]) -> BPMResult:
    """Combine multiple detection results into a final estimate.

    Uses weighted average of primary BPMs, weighted by confidence.
    """
    if not results:
        return BPMResult(bpm=120.0, confidence=0.0, method="ensemble")

    if len(results) == 1:
        return results[0]

    # Collect primary BPMs and their confidences
    bpms = []
    weights = []
    all_candidates = []

    for r in results:
        if r.confidence > 0.05:  # ignore very low confidence
            bpms.append(r.bpm)
            weights.append(r.confidence)
        all_candidates.extend(r.candidates[:3])

    if not bpms:
        return results[0]

    bpms_arr = np.array(bpms)
    weights_arr = np.array(weights)

    # If all methods agree closely (< 5% spread), use weighted average
    bpm_range = bpms_arr.max() - bpms_arr.min()
    bpm_mean = bpms_arr.mean()
    if bpm_mean > 0 and bpm_range / bpm_mean < 0.08:
        final_bpm = float(np.average(bpms_arr, weights=weights_arr))
        confidence = float(np.mean(weights_arr))
    else:
        # Disagreement: pick the highest-confidence result
        best_idx = int(np.argmax(weights_arr))
        final_bpm = float(bpms_arr[best_idx])
        confidence = float(weights_arr[best_idx])

    # Deduplicate candidates
    unique_candidates = []
    for c in all_candidates:
        if not any(abs(c - u) < 2.0 for u in unique_candidates):
            unique_candidates.append(c)

    return BPMResult(
        bpm=final_bpm,
        confidence=min(1.0, confidence),
        method="ensemble",
        candidates=sorted(unique_candidates, key=lambda x: abs(x - final_bpm))[:5],
    )


# ═══════════════════════════════════════════════════════════════════════
# BPMDetector class
# ═══════════════════════════════════════════════════════════════════════

class BPMDetector:
    """Automatic BPM detector for maimai audio tracks.

    Uses multiple detection methods and ensembles results.
    """

    def __init__(
        self,
        min_bpm: int = MIN_BPM,
        max_bpm: int = MAX_BPM,
        sr: int = ANALYSIS_SR,
        methods: Optional[list[str]] = None,
    ):
        """
        Args:
            min_bpm: Minimum detectable BPM.
            max_bpm: Maximum detectable BPM.
            sr: Analysis sample rate.
            methods: Detection methods to use. Default: ['autocorrelation', 'tempogram'].
                     Available: 'autocorrelation', 'tempogram', 'beat_track'.
        """
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self.sr = sr
        self.methods = methods or ["autocorrelation", "tempogram"]

    def detect(self, audio_path: Union[str, Path]) -> BPMResult:
        """Detect BPM from an audio file.

        Args:
            audio_path: Path to audio file (mp3, wav, flac, etc.)

        Returns:
            BPMResult with BPM estimate and confidence.
        """
        y = _load_audio(audio_path, sr=self.sr)

        results: list[BPMResult] = []
        for method_name in self.methods:
            try:
                if method_name == "autocorrelation":
                    r = _detect_autocorrelation(y, self.sr, self.min_bpm, self.max_bpm)
                elif method_name == "tempogram":
                    r = _detect_tempogram(y, self.sr, self.min_bpm, self.max_bpm)
                elif method_name == "beat_track":
                    r = _detect_beat_track(y, self.sr, self.min_bpm, self.max_bpm)
                else:
                    logger.warning(f"Unknown BPM method: {method_name}")
                    continue
                results.append(r)
                logger.debug(f"  {method_name}: {r.bpm:.1f} BPM (conf={r.confidence:.2f})")
            except Exception as e:
                logger.warning(f"BPM method '{method_name}' failed: {e}")

        return _ensemble_results(results)

    def detect_batch(
        self,
        audio_paths: list[Union[str, Path]],
        show_progress: bool = True,
    ) -> list[BPMResult]:
        """Detect BPM for multiple audio files.

        Args:
            audio_paths: List of audio file paths.
            show_progress: Print progress to stdout.

        Returns:
            List of BPMResult, same order as input.
        """
        results = []
        for i, path in enumerate(audio_paths):
            if show_progress:
                print(f"[{i+1}/{len(audio_paths)}] {Path(path).name} ...", end=" ", flush=True)
            try:
                r = self.detect(path)
                results.append(r)
                if show_progress:
                    print(f"{r.bpm:.1f} BPM (conf={r.confidence:.2f})")
            except Exception as e:
                logger.error(f"Failed to detect BPM for {path}: {e}")
                results.append(BPMResult(bpm=0.0, confidence=0.0, method="error"))
                if show_progress:
                    print(f"ERROR: {e}")
        return results


# ═══════════════════════════════════════════════════════════════════════
# Convenience
# ═══════════════════════════════════════════════════════════════════════

def detect_bpm(audio_path: Union[str, Path]) -> float:
    """Quick BPM detection. Returns BPM value only."""
    detector = BPMDetector()
    return detector.detect(audio_path).bpm


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python utils/bpm_detector.py <audio_file> [audio_file ...]")
        print("       python utils/bpm_detector.py datasets/10/track.mp3")
        sys.exit(1)

    detector = BPMDetector()
    paths = sys.argv[1:]

    for path in paths:
        p = Path(path)
        if not p.exists():
            print(f"File not found: {path}")
            continue

        print(f"\n{'='*60}")
        print(f"  {p.name}")
        print(f"{'='*60}")

        result = detector.detect(path)

        print(f"  BPM        : {result.bpm:.1f}")
        print(f"  Confidence : {result.confidence:.2f}")
        print(f"  Method     : {result.method}")
        if result.candidates:
            print(f"  Candidates : {', '.join(f'{c:.1f}' for c in result.candidates)}")
        if result.raw_peaks:
            print(f"  Raw peaks  : {', '.join(f'{c:.1f}' for c in result.raw_peaks[:5])}")
