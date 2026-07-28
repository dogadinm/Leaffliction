from pathlib import Path

import numpy as np

from leaffliction.io import (
    class_counts,
    iter_images,
    label_of,
    load_image,
    save_image,
    try_load_image,
)


def test_save_and_load_roundtrip_preserves_rgb(tmp_path: Path) -> None:
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    img[:, :, 0] = 255  # pure red in RGB
    path = tmp_path / "red.JPG"
    save_image(img, path)

    loaded = load_image(path)
    assert loaded.shape == (4, 4, 3)
    # If this were BGR, channel 2 (not 0) would be saturated instead.
    assert loaded[0, 0, 0] > 200
    assert loaded[0, 0, 2] < 50


def test_label_of_is_parent_directory_name() -> None:
    assert label_of(Path("/data/Apple/Apple_healthy/image (1).JPG")) == "Apple_healthy"


def test_iter_images_finds_nested_files_sorted(toy_dataset: Path) -> None:
    found = list(iter_images(toy_dataset))
    assert found == sorted(found)
    assert len(found) == 11


def test_class_counts_matches_fixture(toy_dataset: Path) -> None:
    counts = class_counts(toy_dataset)
    assert counts == {"Apple_healthy": 8, "Apple_scab": 3}


def test_try_load_image_returns_none_for_corrupt_file(tmp_path: Path) -> None:
    bad = tmp_path / "corrupt.jpg"
    bad.write_bytes(b"not an image")
    assert try_load_image(bad) is None


def test_iter_images_ignores_non_image_files(tmp_path: Path) -> None:
    (tmp_path / "readme.txt").write_text("not an image")
    (tmp_path / "notes.md").write_text("not an image")
    assert list(iter_images(tmp_path)) == []
