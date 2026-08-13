from __future__ import annotations

import pytest
import torch
from torch import nn

from rift_svc.rf import RF, _combine_cfg_predictions


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


def test_sample_requires_two_time_points() -> None:
    model = RF(ZeroTransformer())

    with pytest.raises(ValueError, match="at least 2"):
        model.sample(*_inputs(), steps=1)


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


def test_cfg_guidance_terms_share_the_standard_prediction() -> None:
    predictions = torch.tensor([[[[2.0]]], [[[2.0]]]])
    predictions = torch.cat(
        (
            predictions,
            torch.tensor([[[[1.0]]], [[[0.0]]]]),
            torch.tensor([[[[0.5]]], [[[1.0]]]]),
        ),
        dim=1,
    )

    guided = _combine_cfg_predictions(
        predictions,
        use_ds_cfg=True,
        use_spk_cfg=True,
        ds_cfg_strength=0.2,
        spk_cfg_strength=0.8,
        cfg_rescale=0.0,
    )

    expected = predictions[:, 0]
    expected = expected + 0.2 * (predictions[:, 0] - predictions[:, 1])
    expected = expected + 0.8 * (predictions[:, 0] - predictions[:, 2])
    torch.testing.assert_close(guided, expected)


def test_cfg_rescale_is_independent_per_sample() -> None:
    standard = torch.tensor(
        [
            [[[1.0], [2.0], [3.0]]],
            [[[100.0], [200.0], [300.0]]],
        ]
    ).squeeze(1)
    null_speaker = torch.zeros_like(standard)
    predictions = torch.stack((standard, null_speaker), dim=1)

    guided = _combine_cfg_predictions(
        predictions,
        use_ds_cfg=False,
        use_spk_cfg=True,
        ds_cfg_strength=0.0,
        spk_cfg_strength=0.8,
        cfg_rescale=1.0,
    )

    reduce_dims = tuple(range(1, guided.ndim))
    torch.testing.assert_close(
        guided.std(dim=reduce_dims),
        standard.std(dim=reduce_dims),
    )
