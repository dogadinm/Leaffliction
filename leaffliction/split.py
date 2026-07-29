"""Stratified train/validation split.

Only ever call this on the raw, un-augmented data set. Augmenting first
leaks a rotated copy of a training image into validation, and the
accuracy that follows is worthless. augmented_directory is built from
train_files.txt afterwards, never the other way round.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from leaffliction.io import iter_images, label_of


def stratified_split(
    root: Path, val_split: float, seed: int
) -> tuple[list[Path], list[Path]]:
    """Split each class independently so class balance is preserved.

    Returns (train_files, val_files). At least one image per class is kept
    in each half whenever the class has 2+ images.
    """
    if not 0.0 < val_split < 1.0:
        raise ValueError(f"val_split must be in (0, 1), got {val_split}")

    by_class: dict[str, list[Path]] = {}
    for path in iter_images(root):
        by_class.setdefault(label_of(path), []).append(path)

    if not by_class:
        raise ValueError(f"no images found under {root}")

    rng = np.random.default_rng(seed)
    train: list[Path] = []
    val: list[Path] = []
    for label in sorted(by_class):
        paths = sorted(by_class[label])
        rng.shuffle(paths)
        n_val = max(1, round(len(paths) * val_split)) if len(paths) > 1 else 0
        val.extend(paths[:n_val])
        train.extend(paths[n_val:])
    return train, val


def write_split(train: list[Path], val: list[Path], out_dir: Path) -> None:
    """Persist the split as plain file lists — the provable answer to
    "show me the split, did you augment before or after" at defense."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_files.txt").write_text(
        "\n".join(str(p) for p in train) + ("\n" if train else "")
    )
    (out_dir / "val_files.txt").write_text(
        "\n".join(str(p) for p in val) + ("\n" if val else "")
    )


def read_split(out_dir: Path) -> tuple[list[Path], list[Path]]:
    out_dir = Path(out_dir)
    train = [
        Path(line)
        for line in (out_dir / "train_files.txt").read_text().splitlines()
        if line
    ]
    val = [
        Path(line)
        for line in (out_dir / "val_files.txt").read_text().splitlines()
        if line
    ]
    return train, val
