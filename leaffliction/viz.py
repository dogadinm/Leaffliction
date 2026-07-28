"""Matplotlib helpers shared by every CLI tool in this repo."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
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


def build_color_histogram(
    histograms: Mapping[str, np.ndarray],
    colors: Mapping[str, str],
    *,
    suptitle: str | None = None,
) -> Figure:
    """Build the nine channel line plot of Figure IV.7.

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
    return fig


def show_color_histogram(
    histograms: Mapping[str, np.ndarray],
    colors: Mapping[str, str],
    *,
    suptitle: str | None = None,
    save_path: Path | str | None = None,
) -> None:
    """Display or save the color histogram."""
    fig = build_color_histogram(histograms, colors, suptitle=suptitle)
    finish_figure(fig, save_path)


def figure_to_array(fig: Figure, *, dpi: int = 100) -> np.ndarray:
    """Render a figure into an RGB uint8 array and close it.

    A dedicated Agg canvas is attached rather than using fig.canvas,
    because the active backend may be an interactive one whose canvas
    has no pixel buffer to read.

    The dpi is set explicitly: matplotlib inherits it from the display,
    so the same figure rasterises to 900x500 on one machine and
    1800x1000 on a retina screen.
    """
    fig.set_dpi(dpi)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    plt.close(fig)
    return rgba[:, :, :3].copy()


def show_augmentation_grid(
    images: Sequence[np.ndarray],
    titles: Sequence[str],
    *,
    suptitle: str | None = None,
) -> None:
    """Display a single row of images (augmentation preview)."""
    show_grid(images, titles, ncols=len(images), suptitle=suptitle)


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
