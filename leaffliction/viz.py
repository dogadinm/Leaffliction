"""Matplotlib helpers for augmentation previews."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def show_augmentation_grid(
    images: Sequence[np.ndarray],
    titles: Sequence[str],
    *,
    suptitle: str | None = None,
) -> None:
    """Display a row of images with titles."""
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    plt.show()
