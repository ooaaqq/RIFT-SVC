"""Audio input and segment assembly for inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def load_audio(file_path: str | Path, target_sr: int) -> np.ndarray:
    audio_np, source_sr = sf.read(
        str(file_path), dtype="float32", always_2d=True
    )
    import torch
    import torchaudio

    audio = torch.from_numpy(np.ascontiguousarray(audio_np.T))
    if source_sr != target_sr:
        audio = torchaudio.functional.resample(audio, source_sr, target_sr)
    if audio.ndim > 1:
        audio = audio.mean(dim=0, keepdim=True)
    return audio.numpy().squeeze(0)


def write_audio(
    file_path: str | Path,
    audio: np.ndarray,
    sample_rate: int,
    subtype: str | None = "PCM_24",
) -> None:
    """Write inference output without silently quantizing it to PCM16."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError(f"expected mono audio, got shape {audio.shape}")
    sf.write(
        file_path,
        np.nan_to_num(audio),
        sample_rate,
        subtype=subtype or "PCM_24",
    )


def apply_fade(audio: np.ndarray, fade_samples: int, fade_in: bool) -> np.ndarray:
    if fade_samples <= 0:
        return audio
    fade_samples = min(fade_samples, len(audio))
    fade_window = np.hanning(fade_samples * 2)
    fade_curve = (
        fade_window[:fade_samples]
        if fade_in
        else fade_window[fade_samples:]
    )
    audio[:fade_samples] *= fade_curve
    return audio


def add_segment(
    result: np.ndarray,
    *,
    index: int,
    total_segments: int,
    start_sample: int,
    audio_out: np.ndarray,
    expected_length: int,
    fade_samples: int,
) -> None:
    """Add one converted segment to an existing overlap-add buffer."""
    audio_out = np.array(audio_out, dtype=np.float32, copy=True)
    if len(audio_out) > expected_length:
        audio_out = audio_out[:expected_length]
    elif len(audio_out) < expected_length:
        audio_out = np.pad(
            audio_out, (0, expected_length - len(audio_out)), mode="constant"
        )

    if fade_samples > 0 and len(audio_out) > 0:
        fade = min(fade_samples, len(audio_out))
        if index > 0:
            apply_fade(audio_out, fade, fade_in=True)
            overlap_start = max(0, start_sample)
            overlap_end = min(len(result), start_sample + fade)
            overlap_len = overlap_end - overlap_start
            if overlap_len > 0:
                result[overlap_start:overlap_end] *= np.linspace(
                    1.0, 0.0, overlap_len, dtype=np.float32
                )
        if index < total_segments - 1:
            audio_out[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)

    end_sample = min(len(result), start_sample + len(audio_out))
    if end_sample > start_sample:
        result[start_sample:end_sample] += audio_out[: end_sample - start_sample]
