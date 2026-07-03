"""
MaiTrackTokenizer — Audio tokenizer using pre-trained EnCodec (Meta).

Wraps EnCodec 24kHz model to convert maimai track.mp3 audio into
discrete token sequences for transformer training.

Key design decisions:
  - Uses EnCodec 24kHz (pre-trained on speech+music, 8-layer RVQ, 1024 bins)
  - Default: 2 codebook layers (~15816 tokens/2min song)
  - Stride: 320 samples @ 24kHz = 75Hz = 13.3ms per token
  - BPM is NOT encoded (computed separately by external program)

Usage:
    from Tokenizer.MaiTrackTokenizer import MaiTrackTokenizer

    tok = MaiTrackTokenizer()
    tokens = tok.encode("datasets/10/track.mp3")         # → list[int]
    tokens_2l = tok.encode("datasets/10/track.mp3", n_layers=2)
    audio = tok.decode(tokens)                            # → torch.Tensor
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import soundfile as sf
import torch

from encodec import EncodecModel
from encodec.utils import convert_audio

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

# EnCodec 24kHz: 8 codebooks, 1024 bins each, stride=320 samples
# Frame rate = 24000 / 320 = 75 Hz (= 13.33 ms per token)
ENC_SAMPLE_RATE = 24000
ENC_STRIDE = 320          # samples per frame
ENC_FRAME_RATE = ENC_SAMPLE_RATE / ENC_STRIDE  # 75 Hz
ENC_CODEBOOK_SIZE = 1024
ENC_NUM_CODEBOOKS = 8
ENC_BANDWIDTH = 6.0       # kbps target

# Default: use only first 2 codebook layers for efficiency
DEFAULT_N_LAYERS = 2

# Special tokens (aligned with chart tokenizer)
PAD = 0
BOS = 1
EOS = 2

# Token value offset per layer (to avoid overlap)
# Layer 0: codes 0..1023 → tokens 3..1026
# Layer 1: codes 0..1023 → tokens 1027..2050
# etc.
TOKEN_OFFSET_BASE = 3  # after PAD, BOS, EOS


# ═══════════════════════════════════════════════════════════════════════
# Tokenizer class
# ═══════════════════════════════════════════════════════════════════════

class MaiTrackTokenizer:
    """
    Pre-trained audio tokenizer using Meta EnCodec.

    Converts audio waveforms to/from discrete token sequences.
    Multi-layer tokens are interleaved: [L0_t0, L1_t0, L0_t1, L1_t1, ...]

    Attributes:
        sample_rate: 24000 Hz
        frame_rate: 75 Hz (13.3ms per token)
        n_layers: Number of codebook layers used (default 2)
        vocab_size: Total vocabulary size (layers × 1024 + special tokens)
    """

    def __init__(self, n_layers: int = DEFAULT_N_LAYERS, device: str = "cpu"):
        """
        Args:
            n_layers: Number of EnCodec codebook layers to use (1-8).
                      More layers = better audio quality, more tokens.
                      1-2 layers typically sufficient for rhythm game features.
            device: Device to run the model on ("cpu" or "cuda").
        """
        self.n_layers = n_layers
        self.device = device

        # Load pre-trained model
        self._model = EncodecModel.encodec_model_24khz()
        self._model.set_target_bandwidth(ENC_BANDWIDTH)
        self._model.eval()
        self._model.to(device)

        # Vocabulary: specials + n_layers * 1024 codes
        self.vocab_size = TOKEN_OFFSET_BASE + n_layers * ENC_CODEBOOK_SIZE
        self.pad_token_id = PAD
        self.bos_token_id = BOS
        self.eos_token_id = EOS

    @property
    def sample_rate(self) -> int:
        return ENC_SAMPLE_RATE

    @property
    def frame_rate(self) -> float:
        return ENC_FRAME_RATE

    # ── Load audio ──────────────────────────────────────────────────

    def load_audio(self, path: Union[str, Path]) -> torch.Tensor:
        """
        Load an audio file and convert to 24kHz mono tensor.

        Args:
            path: Path to audio file (mp3, wav, flac, etc.).

        Returns:
            Tensor [1, samples] at 24kHz mono.
        """
        data, sr = sf.read(str(path), dtype="float32")

        # Convert to mono
        if data.ndim > 1:
            data = data.mean(axis=1)

        wav = torch.from_numpy(data.copy()).unsqueeze(0)  # [1, T]

        # Resample to 24kHz if needed
        if sr != ENC_SAMPLE_RATE:
            wav = convert_audio(wav, sr, ENC_SAMPLE_RATE, 1)

        return wav

    def load_audio_batch(self, paths: list[str],
                         max_duration: Optional[float] = None) -> tuple[torch.Tensor, list[int]]:
        """
        Load a batch of audio files, padding to same length.

        Args:
            paths: List of audio file paths.
            max_duration: Truncate to max_duration seconds (None = no truncation).

        Returns:
            (wavs [B, 1, max_samples], lengths [B])
        """
        wavs = []
        lengths = []
        for p in paths:
            wav = self.load_audio(p)
            if max_duration is not None:
                max_samples = int(max_duration * ENC_SAMPLE_RATE)
                wav = wav[:, :max_samples]
            lengths.append(wav.shape[1])
            wavs.append(wav)

        # Pad to max length
        max_len = max(lengths)
        padded = torch.zeros(len(wavs), 1, max_len)
        for i, w in enumerate(wavs):
            padded[i, :, :w.shape[1]] = w

        return padded, lengths

    # ── Encode ──────────────────────────────────────────────────────

    @torch.no_grad()
    def encode(self, audio: Union[str, Path, torch.Tensor, np.ndarray],
               n_layers: Optional[int] = None,
               add_bos: bool = True,
               add_eos: bool = True,
               interleave: bool = True) -> list[int]:
        """
        Encode audio into a discrete token sequence.

        Args:
            audio: Path to audio file, or waveform tensor [1, T] / numpy [T].
            n_layers: Override number of codebook layers (default: self.n_layers).
            add_bos: Prepend BOS token.
            add_eos: Append EOS token.
            interleave: If True, interleave layers: [L0, L1, L0, L1, ...].
                        If False, concatenate: [L0_all..., L1_all...].

        Returns:
            List of integer token IDs.
        """
        n_layers = n_layers or self.n_layers

        # Load if path
        if isinstance(audio, (str, Path)):
            wav = self.load_audio(audio).to(self.device)
        elif isinstance(audio, np.ndarray):
            wav = torch.from_numpy(audio.astype("float32")).unsqueeze(0).to(self.device)
        else:
            wav = audio.to(self.device)

        # Ensure correct shape
        if wav.dim() == 1:
            wav = wav.unsqueeze(0).unsqueeze(0)
        elif wav.dim() == 2:
            wav = wav.unsqueeze(1)

        # Resample if needed
        if wav.shape[1] != 1:
            wav = wav.mean(dim=1, keepdim=True)

        # Encode with EnCodec. Long files are returned as multiple chunks;
        # concatenate them so token length reflects the full track duration.
        encoded = self._model.encode(wav)
        codes = torch.cat([frame_codes for frame_codes, _ in encoded], dim=-1)  # [B, n_q, T]
        codes = codes[0, :n_layers, :]  # [n_layers, T]

        # Convert to flat token list
        tokens: list[int] = []
        if add_bos:
            tokens.append(BOS)

        T = codes.shape[1]
        if interleave:
            # Interleave: L0_t0, L1_t0, L0_t1, L1_t1, ...
            for t in range(T):
                for layer in range(n_layers):
                    code = codes[layer, t].item()
                    token = self._code_to_token(code, layer)
                    tokens.append(token)
        else:
            # Concatenate per layer
            for layer in range(n_layers):
                for t in range(T):
                    code = codes[layer, t].item()
                    token = self._code_to_token(code, layer)
                    tokens.append(token)

        if add_eos:
            tokens.append(EOS)

        return tokens

    @torch.no_grad()
    def encode_batch(self, audios: list[Union[str, Path, torch.Tensor]],
                     n_layers: Optional[int] = None,
                     max_duration: Optional[float] = None,
                     pad_to: Optional[int] = None,
                     return_tensors: bool = False):
        """
        Encode a batch of audio files.

        Args:
            audios: List of paths or tensors.
            n_layers: Number of codebook layers.
            max_duration: Truncate audio to this many seconds.
            pad_to: Pad token sequences to this length.
            return_tensors: Return torch tensors instead of lists.

        Returns:
            If return_tensors=False: (list[list[int]], list[int])
            If return_tensors=True: (Tensor[B, max_len], Tensor[B])
        """
        n_layers = n_layers or self.n_layers

        token_seqs = []
        for audio in audios:
            tokens = self.encode(audio, n_layers=n_layers, interleave=True)
            if pad_to is not None and len(tokens) > pad_to:
                tokens = tokens[:pad_to]
            token_seqs.append(tokens)

        lengths = [len(s) for s in token_seqs]
        max_len = max(lengths) if pad_to is None else max(pad_to, max(lengths))

        padded = []
        for seq in token_seqs:
            if len(seq) < max_len:
                seq = seq + [PAD] * (max_len - len(seq))
            padded.append(seq[:max_len])

        if return_tensors:
            return (torch.tensor(padded, dtype=torch.long),
                    torch.tensor(lengths, dtype=torch.long))

        return padded, lengths

    # ── Decode ──────────────────────────────────────────────────────

    @torch.no_grad()
    def decode(self, tokens: list[int],
               n_layers: Optional[int] = None,
               interleave: bool = True) -> torch.Tensor:
        """
        Decode a token sequence back to audio waveform.

        Args:
            tokens: Token ID list.
            n_layers: Number of codebook layers used (must match encoding).
            interleave: Whether tokens are interleaved (must match encoding).

        Returns:
            Waveform tensor [1, samples] at 24kHz.
        """
        n_layers = n_layers or self.n_layers

        # Strip BOS/EOS
        if tokens and tokens[0] == BOS:
            tokens = tokens[1:]
        if tokens and tokens[-1] == EOS:
            tokens = tokens[:-1]

        total_tokens = len(tokens)
        if interleave:
            T = total_tokens // n_layers
        else:
            T = total_tokens // n_layers

        if T == 0:
            logger.warning("Token sequence too short for decoding")
            return torch.zeros(1, ENC_STRIDE)

        # Convert tokens back to EnCodec codes
        # Full 8 layers: fill unused layers with zeros
        codes = torch.zeros(ENC_NUM_CODEBOOKS, T, dtype=torch.long, device=self.device)

        if interleave:
            for t in range(T):
                for layer in range(n_layers):
                    idx = t * n_layers + layer
                    if idx < total_tokens:
                        codes[layer, t] = self._token_to_code(tokens[idx], layer)
        else:
            for layer in range(n_layers):
                for t in range(T):
                    idx = layer * T + t
                    if idx < total_tokens:
                        codes[layer, t] = self._token_to_code(tokens[idx], layer)

        # Decode with EnCodec
        codes = codes.unsqueeze(0)  # [1, 8, T]
        decoded = self._model.decode([(codes, None)])
        return decoded.squeeze(0)  # [1, samples]

    # ── Token ↔ Code conversion ─────────────────────────────────────

    def _code_to_token(self, code: int, layer: int) -> int:
        """Convert EnCodec code (0..1023) to global token ID."""
        return TOKEN_OFFSET_BASE + layer * ENC_CODEBOOK_SIZE + code

    def _token_to_code(self, token: int, layer: int) -> int:
        """Convert global token ID back to EnCodec code (0..1023)."""
        offset = TOKEN_OFFSET_BASE + layer * ENC_CODEBOOK_SIZE
        code = token - offset
        return max(0, min(code, ENC_CODEBOOK_SIZE - 1))

    # ── Debug ───────────────────────────────────────────────────────

    def tokens_to_str(self, tokens: list[int], max_show: int = 30) -> str:
        """Pretty-print token sequence (truncated)."""
        parts = []
        for t in tokens[:max_show]:
            if t == PAD:
                parts.append("[PAD]")
            elif t == BOS:
                parts.append("[BOS]")
            elif t == EOS:
                parts.append("[EOS]")
            else:
                # Determine layer and code
                code = t - TOKEN_OFFSET_BASE
                layer = code // ENC_CODEBOOK_SIZE
                c = code % ENC_CODEBOOK_SIZE
                parts.append(f"L{layer}:{c}")
        if len(tokens) > max_show:
            parts.append(f"... ({len(tokens) - max_show} more)")
        return " ".join(parts)

    # ── Info ────────────────────────────────────────────────────────

    def token_count_estimate(self, duration_seconds: float,
                             n_layers: Optional[int] = None) -> int:
        """
        Estimate the number of tokens for a given audio duration.

        Args:
            duration_seconds: Audio duration in seconds.
            n_layers: Number of codebook layers.

        Returns:
            Estimated token count (including BOS/EOS).
        """
        n_layers = n_layers or self.n_layers
        frames = int(duration_seconds * ENC_FRAME_RATE)
        return 2 + frames * n_layers  # BOS + frames*layers + EOS

    def __repr__(self) -> str:
        return (f"MaiTrackTokenizer(n_layers={self.n_layers}, "
                f"sr={ENC_SAMPLE_RATE}Hz, "
                f"frame_rate={ENC_FRAME_RATE:.0f}Hz, "
                f"vocab_size={self.vocab_size}, "
                f"device={self.device})")


# ═══════════════════════════════════════════════════════════════════════
# Quick test
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "datasets/10/track.mp3"

    tok = MaiTrackTokenizer(n_layers=2)
    print(tok)
    print(f"Vocab size: {tok.vocab_size}")

    tokens = tok.encode(path)
    print(f"Tokens: {len(tokens)} ({tok.tokens_to_str(tokens, 30)})")

    audio = tok.decode(tokens)
    print(f"Decoded audio: {audio.shape}, {audio.shape[1]/ENC_SAMPLE_RATE:.1f}s")
