from __future__ import annotations

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


def test_sample_seed_is_reproducible_without_trajectory() -> None:
    model = RF(ZeroTransformer())
    args = _inputs()

    first, trajectory = model.sample(
        *args,
        steps=4,
        cfg_rescale=0.0,
        generator=torch.Generator().manual_seed(7),
    )
    second, _ = model.sample(
        *args,
        steps=4,
        cfg_rescale=0.0,
        generator=torch.Generator().manual_seed(7),
    )

    assert trajectory is None
    torch.testing.assert_close(first, second)


def test_sample_can_return_trajectory_on_request() -> None:
    model = RF(ZeroTransformer())
    output, trajectory = model.sample(
        *_inputs(),
        steps=4,
        cfg_rescale=0.0,
        noise=torch.zeros(1, 4, 2),
        return_trajectory=True,
    )

    assert trajectory is not None
    assert trajectory.shape == (4, 1, 4, 2)
    torch.testing.assert_close(output, torch.full_like(output, -5.0))


def test_sample_accepts_a_seed_directly() -> None:
    model = RF(ZeroTransformer())
    args = _inputs()

    first, _ = model.sample(*args, steps=2, cfg_rescale=0.0, seed=11)
    second, _ = model.sample(*args, steps=2, cfg_rescale=0.0, seed=11)

    torch.testing.assert_close(first, second)
