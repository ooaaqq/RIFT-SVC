from __future__ import annotations

import numpy as np
import soundfile as sf

from rift_svc.inference.audio import add_segment, assemble_segments, write_audio


def test_assemble_segments_is_float32_and_exact_length() -> None:
    segments = [
        (0, np.ones(4, dtype=np.float64), 4),
        (4, np.full(2, 2.0, dtype=np.float64), 2),
    ]
    result = assemble_segments(segments, total_samples=6, fade_samples=0)

    assert result.dtype == np.float32
    assert result.shape == (6,)
    np.testing.assert_allclose(result, [1, 1, 1, 1, 2, 2])


def test_write_audio_uses_explicit_lossless_subtypes(tmp_path) -> None:
    audio = np.linspace(-0.5, 0.5, 32, dtype=np.float32)

    flac_path = tmp_path / "audio.flac"
    write_audio(flac_path, audio, 44100)
    assert sf.info(flac_path).subtype == "PCM_24"

    wav_path = tmp_path / "audio.wav"
    write_audio(wav_path, audio, 44100, subtype=None)
    assert sf.info(wav_path).subtype == "PCM_24"

    float_wav_path = tmp_path / "audio-float.wav"
    write_audio(float_wav_path, audio, 44100, subtype="FLOAT")
    assert sf.info(float_wav_path).subtype == "FLOAT"


def test_streaming_segment_add_matches_batch_assembly() -> None:
    segments = [
        (0, np.ones(6, dtype=np.float32), 6),
        (6, np.full(4, 2.0, dtype=np.float32), 4),
    ]
    expected = assemble_segments(segments, total_samples=10, fade_samples=2)
    streamed = np.zeros(12, dtype=np.float32)
    for index, (start, audio, length) in enumerate(segments):
        add_segment(
            streamed,
            index=index,
            total_segments=len(segments),
            start_sample=start,
            audio_out=audio,
            expected_length=length,
            fade_samples=2,
        )

    np.testing.assert_allclose(streamed[:10], expected)
