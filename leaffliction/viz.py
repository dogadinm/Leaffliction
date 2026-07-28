"""Matplotlib helpers shared by the augmentation and transformation tools."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def finish_figure(
    fig: Figure,
    save_path: Path | str | None = None,
    *,
    dpi: int = 100,
) -> None:
    """Display the figure, or write it to save_path and close it.

    Batch mode must never open a window, so passing save_path switches
    the figure from interactive display to disk output.
    """
    fig.tight_layout()
    if save_path is None:
        plt.show()
        return
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _draw_image(ax: Axes, img: np.ndarray, title: str, show_axes: bool) -> None:
    """Render one image on an axis, handling grayscale and binary masks."""
    if img.ndim == 2:
        ax.imshow(img, cmap="gray")
    else:
        ax.imshow(img)
    if title:
        ax.set_title(title)
    if not show_axes:
        ax.axis("off")


def show_grid(
    images: Sequence[np.ndarray],
    titles: Sequence[str],
    *,
    ncols: int | None = None,
    suptitle: str | None = None,
    save_path: Path | str | None = None,
    show_axes: bool = False,
) -> None:
    """Display a grid of images, or save it when save_path is given.

    ncols defaults to a single row. Unused cells of the last row are
    hidden so a 5-image 2-column grid does not show an empty frame.
    """
    if not images:
        raise ValueError("show_grid needs at least one image")
    if len(images) != len(titles):
        raise ValueError(
            f"got {len(images)} images but {len(titles)} titles"
        )

    n = len(images)
    ncols = n if ncols is None else max(1, min(ncols, n))
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3 * ncols, 3 * nrows),
        squeeze=False,
    )
    cells = axes.ravel()
    for ax, img, title in zip(cells, images, titles):
        _draw_image(ax, img, title, show_axes)
    for ax in cells[n:]:
        ax.axis("off")

    if suptitle:
        fig.suptitle(suptitle)
    finish_figure(fig, save_path)


def show_color_histogram(
    histograms: Mapping[str, np.ndarray],
    colors: Mapping[str, str],
    *,
    suptitle: str | None = None,
    save_path: Path | str | None = None,
) -> None:
    """Plot the nine channel histograms of Figure IV.7.

    This is a line plot rather than an image, so it cannot share a cell
    with the transformation grid and gets its own figure.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    for name in sorted(histograms):
        ax.plot(histograms[name], label=name, color=colors.get(name),
                linewidth=1.2)
    ax.set_xlabel("Pixel intensity")
    ax.set_ylabel("Proportion of pixels (%)")
    ax.set_xlim(0, 255)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(title="color Channel", loc="upper right", fontsize="small")
    if suptitle:
        ax.set_title(suptitle)
    finish_figure(fig, save_path)


def show_augmentation_grid(
    images: Sequence[np.ndarray],
    titles: Sequence[str],
    *,
    suptitle: str | None = None,
) -> None:
    """Display a single row of images (augmentation preview)."""
    show_grid(images, titles, ncols=len(images), suptitle=suptitle)
