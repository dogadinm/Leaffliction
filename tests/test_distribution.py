from pathlib import Path

import numpy as np
import pytest

import Distribution
from leaffliction.io import save_image


def test_main_errors_on_non_directory(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    missing = tmp_path / "missing"

    code = Distribution.main([str(missing)])

    assert code == 1
    assert "not a directory" in capsys.readouterr().err


def test_main_errors_when_directory_has_no_images(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    empty = tmp_path / "Apple"
    empty.mkdir()

    code = Distribution.main([str(empty)])

    assert code == 1
    assert "no images found" in capsys.readouterr().err


def test_main_plots_the_class_distribution_for_a_valid_directory(
    toy_dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shown = []
    monkeypatch.setattr("leaffliction.viz.plt.show", lambda: shown.append(True))

    code = Distribution.main([str(toy_dataset)])

    assert code == 0
    assert shown == [True]


def test_main_works_on_any_plant_directory_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The subject requires this to work on every plant directory, not just
    Apple — exercise it with an unrelated directory/class name."""
    root = tmp_path / "Grape"
    save_image(np.zeros((4, 4, 3), dtype=np.uint8), root / "Grape_healthy" / "img (0).JPG")
    save_image(np.zeros((4, 4, 3), dtype=np.uint8), root / "Grape_black_rot" / "img (0).JPG")

    monkeypatch.setattr("leaffliction.viz.plt.show", lambda: None)

    code = Distribution.main([str(root)])

    assert code == 0
