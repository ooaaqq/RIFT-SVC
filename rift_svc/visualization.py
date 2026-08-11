"""Training visualizations and their optional plotting dependencies."""

from __future__ import annotations

import io

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def draw_mel_specs(
    ground_truth: np.ndarray,
    generated: np.ndarray,
    difference: np.ndarray,
    cache_path: str,
) -> None:
    value_min = min(ground_truth.min(), generated.min())
    value_max = max(ground_truth.max(), generated.max())

    figure, (axis_gt, axis_generated, axis_difference) = plt.subplots(
        3,
        1,
        figsize=(20, 15),
        sharex=True,
        gridspec_kw={"hspace": 0},
    )
    image_gt = axis_gt.imshow(
        ground_truth,
        origin="lower",
        aspect="auto",
        vmin=value_min,
        vmax=value_max,
    )
    axis_gt.set_ylabel("GT", fontsize=14)
    axis_gt.set_xticks([])

    axis_generated.imshow(
        generated,
        origin="lower",
        aspect="auto",
        vmin=value_min,
        vmax=value_max,
    )
    axis_generated.set_ylabel("Gen", fontsize=14)
    axis_generated.set_xticks([])

    difference_limit = max(abs(difference.min()), abs(difference.max()))
    image_difference = axis_difference.imshow(
        difference,
        origin="lower",
        aspect="auto",
        cmap="RdBu_r",
        vmin=-difference_limit,
        vmax=difference_limit,
    )
    axis_difference.set_ylabel("Diff", fontsize=14)

    figure.colorbar(
        image_gt,
        ax=[axis_gt, axis_generated],
        location="right",
        label="Magnitude",
    )
    figure.colorbar(
        image_difference,
        ax=[axis_difference],
        location="right",
        label="Difference",
    )

    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(figure)
    buffer.seek(0)
    image = Image.open(buffer).convert("RGB")
    image.save(cache_path, "JPEG", quality=85, optimize=True)
    image.close()
    buffer.close()
