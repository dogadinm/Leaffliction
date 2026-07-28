from pathlib import Path

import numpy as np

from leaffliction.io import class_counts, label_of
from leaffliction.split import stratified_split
from train import build_augmented_train_dir


def test_build_augmented_train_dir_balances_and_never_touches_val_split(
    toy_dataset: Path, tmp_path: Path
) -> None:
    """Proves trap 2 in code: the balanced training set is built from the
    train half only, and no validation image ever lands in it."""
    train_files, val_files = stratified_split(toy_dataset, val_split=0.25, seed=42)
    out_dir = tmp_path / "augmented_directory"
    rng = np.random.default_rng(42)

    written = build_augmented_train_dir(train_files, out_dir, rng)

    counts = class_counts(out_dir)
    assert set(counts) == {"Apple_healthy", "Apple_scab"}
    # every class is oversampled up to the size of the largest train-split class
    assert len(set(counts.values())) == 1
    assert sum(counts.values()) == len(written)

    val_names_by_label: dict[str, set[str]] = {}
    for path in val_files:
        val_names_by_label.setdefault(label_of(path), set()).add(path.name)

    for label, val_names in val_names_by_label.items():
        out_names = {path.name for path in (out_dir / label).iterdir()}
        assert not (val_names & out_names)


def test_build_augmented_train_dir_keeps_originals_verbatim(
    toy_dataset: Path, tmp_path: Path
) -> None:
    train_files, _ = stratified_split(toy_dataset, val_split=0.25, seed=42)
    out_dir = tmp_path / "augmented_directory"
    rng = np.random.default_rng(42)

    build_augmented_train_dir(train_files, out_dir, rng)

    train_names_by_label: dict[str, set[str]] = {}
    for path in train_files:
        train_names_by_label.setdefault(label_of(path), set()).add(path.name)

    for label, names in train_names_by_label.items():
        out_names = {path.name for path in (out_dir / label).iterdir()}
        assert names <= out_names
