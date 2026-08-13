from __future__ import annotations

import numpy as np
import pytest

from rift_svc.inference.slicer import Slicer


def make_slicer() -> Slicer:
    return Slicer(
        sr=1000,
        threshold=-30,
        min_length=3000,
        min_interval=100,
        hop_size=10,
        max_sil_kept=200,
    )


def test_short_audio_is_returned_without_copying() -> None:
    audio = np.ones(2000, dtype=np.float32)

    chunks = make_slicer().slice(audio)

    assert len(chunks) == 1
    assert chunks[0][0] == 0
    assert chunks[0][1] is audio


def test_middle_silence_preserves_existing_offsets_and_lengths() -> None:
    audio = np.concatenate([np.ones(3500), np.zeros(500), np.ones(3500)]).astype(
        np.float32
    )

    chunks = make_slicer().slice(audio)

    assert [(start, chunk.shape) for start, chunk in chunks] == [
        (0, (3520,)),
        (3790, (3710,)),
    ]


def test_channel_first_audio_keeps_channel_axis() -> None:
    mono = np.concatenate([np.ones(3500), np.zeros(500), np.ones(3500)]).astype(
        np.float32
    )
    stereo = np.stack([mono, mono])

    chunks = make_slicer().slice(stereo)

    assert [(start, chunk.shape) for start, chunk in chunks] == [
        (0, (2, 3520)),
        (3790, (2, 3710)),
    ]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_length": 50}, "min_length"),
        ({"max_sil_kept": 5}, "max_sil_kept"),
    ],
)
def test_invalid_configuration_is_rejected(kwargs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Slicer(sr=1000, **kwargs)
