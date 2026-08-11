"""Small tensor helpers shared by the inference model."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def lens_to_mask(
    lengths: torch.Tensor,
    length: int | None = None,
) -> torch.Tensor:
    if length is None:
        length = int(lengths.amax().item())
    sequence = torch.arange(length, device=lengths.device)
    return sequence < lengths[..., None]


def linear_interpolate_tensor(tensor: torch.Tensor, new_size: int) -> torch.Tensor:
    values = tensor.transpose(0, 1).unsqueeze(0)
    values = F.interpolate(values, size=new_size, mode="linear", align_corners=True)
    return values.squeeze(0).transpose(0, 1)
