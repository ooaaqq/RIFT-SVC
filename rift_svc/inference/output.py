"""Explicit, lossless audio output formats for inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def write_audio(
    path: str | Path,
    audio: np.ndarray,
    sample_rate: int,
    *,
    subtype: str | None = None,
) -> None:
    """Write audio without relying on SoundFile's format defaults.

    FLAC defaults to 24-bit PCM and WAV defaults to 24-bit PCM.  FLOAT WAV is
    available for intermediate files that must retain the model float output.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix not in {".wav", ".flac"}:
        raise ValueError("inference output must use .wav or .flac")
    if subtype is None:
        subtype = "PCM_24"
    if suffix == ".flac" and subtype == "FLOAT":
        raise ValueError("FLAC output requires an integer PCM subtype")
    sf.write(
        str(output_path),
        np.asarray(audio, dtype=np.float32),
        sample_rate,
        subtype=subtype,
    )
