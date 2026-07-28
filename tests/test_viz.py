from pathlib import Path

import numpy as np
import pytest

from leaffliction.viz import build_color_histogram, figure_to_array, show_grid


def test_show_grid_writes_a_file_when_save_path_is_given(tmp_path: Path) -> None:
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    out = tmp_path / "grid.png"

    show_grid([img, img], ["a", "b"], save_path=out)

    assert out.exists()
    assert out.stat().st_size > 0


def test_show_grid_handles_grayscale_images(tmp_path: Path) -> None:
    gray = np.zeros((8, 8), dtype=np.uint8)
    out = tmp_path / "grid.png"

    show_grid([gray], ["mask"], save_path=out)

    assert out.exists()


def test_show_grid_rejects_mismatched_images_and_titles() -> None:
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        show_grid([img, img], ["only-one-title"])


def test_show_grid_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        show_grid([], [])


def test_build_color_histogram_plots_one_line_per_channel() -> None:
    histograms = {"red": np.zeros(256), "blue": np.ones(256)}
    colors = {"red": "#ff0000", "blue": "#0000ff"}

    fig = build_color_histogram(histograms, colors)

    assert len(fig.axes[0].lines) == 2


def test_figure_to_array_returns_an_rgb_uint8_image() -> None:
    histograms = {"red": np.zeros(256)}
    colors = {"red": "#ff0000"}
    fig = build_color_histogram(histograms, colors)

    array = figure_to_array(fig)

    assert array.ndim == 3
    assert array.shape[2] == 3
    assert array.dtype == np.uint8
