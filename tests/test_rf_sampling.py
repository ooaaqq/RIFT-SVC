from __future__ import annotations

import pytest
import torch
from torch import nn

from rift_svc.rf import RF


class ZeroTransformer(nn.Module):
    dim = 2

    def forward(self, *, x, **kwargs):
        return torch.zeros_like(x)


def _inputs() -> tuple[torch.Tensor, ...]:
    src_mel = torch.zeros(1, 4, 2)
    spk_id = torch.zeros(1, dtype=torch.long)
    f0 = torch.zeros(1, 4)
    rms = torch.zeros(1, 4)
    cvec = torch.zeros(1, 4, 2)
    return src_mel, spk_id, f0, rms, cvec


def test_sample_seed_is_reproducible() -> None:
    model = RF(ZeroTransformer())
    args = _inputs()

    first = model.sample(
        *args,
        steps=4,
        cfg_rescale=0.0,
        seed=7,
    )
    second = model.sample(
        *args,
        steps=4,
        cfg_rescale=0.0,
        seed=7,
    )

    torch.testing.assert_close(first, second)


def test_sample_requires_negative_content_for_content_cfg() -> None:
    model = RF(ZeroTransformer())
    with pytest.raises(ValueError, match="bad_cvec"):
        model.sample(*_inputs(), steps=2, ds_cfg_strength=0.2)


def test_sample_applies_content_and_speaker_cfg() -> None:
    model = RF(ZeroTransformer())
    args = _inputs()
    output = model.sample(
        *args,
        steps=2,
        bad_cvec=args[-1],
        ds_cfg_strength=0.2,
        spk_cfg_strength=0.8,
        seed=7,
    )
    assert output.shape == args[0].shape
