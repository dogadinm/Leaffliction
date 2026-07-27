"""Shared matplotlib helpers so Distribution.py and predict.py render figures
the same way instead of each hand-rolling subplot boilerplate."""
from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def show_image_grid(
    images: Sequence[np.ndarray],
    titles: Sequence[str],
    suptitle: str = "",
) -> None:
    """Display images side by side in a single row, each with its own title."""
    if len(images) != len(titles):
        raise ValueError("images and titles must be the same length")

    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.5))
    axes = [axes] if n == 1 else list(axes)
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    plt.show()


def plot_class_distribution(counts: dict[str, int], title: str) -> None:
    """Pie chart (share) + bar chart (count) of a class_counts() dict."""
    if not counts:
        raise ValueError("counts is empty, nothing to plot")

    labels, values = zip(*sorted(counts.items()))
    fig, (ax_pie, ax_bar) = plt.subplots(1, 2, figsize=(10, 5))
    ax_pie.pie(values, labels=labels, autopct="%1.1f%%")
    ax_pie.set_title(f"{title} — class share")
    ax_bar.bar(labels, values)
    ax_bar.set_title(f"{title} — image count")
    ax_bar.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    plt.show()
