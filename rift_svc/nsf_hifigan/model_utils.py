"""Lightweight HiFi-GAN model helpers."""

from __future__ import annotations


def init_weights(module, mean: float = 0.0, std: float = 0.01) -> None:
    classname = module.__class__.__name__
    if "Conv" in classname:
        module.weight.data.normal_(mean, std)


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)
