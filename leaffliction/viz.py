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
    """Show the figure, or write it to save_path and close it.

    Batch mode must never open a window — that is what save_path is for.
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
    """Lay images out in a grid, then show or save it.

    ncols defaults to one row. Leftover cells in the last row are
    hidden, so 5 images in 2 columns leave no empty frame.
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

    A plot, not an image, so it gets its own figure rather than a cell
    in the transformation grid.
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

    Attaches its own Agg canvas: an interactive backend's canvas has no
    pixel buffer to read from. dpi is set explicitly because matplotlib
    otherwise inherits it from the display, and the same plot comes out
    twice as large on a retina screen.
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
    save_path: Path | str | None = None,
) -> None:
    """Display images side by side in a single row.

    Wraps show_grid so predict.py inherits its grayscale handling: a
    single-channel result like GaussianBlur used to come out purple and
    yellow through matplotlib's default colormap.
    """
    show_grid(images, titles, ncols=len(images), suptitle=suptitle or None,
              save_path=save_path)


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
