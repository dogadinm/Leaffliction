from pathlib import Path

import pytest

from leaffliction.io import label_of
from leaffliction.split import read_split, stratified_split, write_split


def test_split_is_stratified_per_class(toy_dataset: Path) -> None:
    train, val = stratified_split(toy_dataset, val_split=0.25, seed=42)

    assert set(train).isdisjoint(val)
    assert len(train) + len(val) == 11

    val_labels = [label_of(p) for p in val]
    # Every class must appear in val even though 0.25 * 3 rounds under 1.
    assert "Apple_healthy" in val_labels
    assert "Apple_scab" in val_labels


def test_split_is_reproducible_with_same_seed(toy_dataset: Path) -> None:
    train1, val1 = stratified_split(toy_dataset, val_split=0.2, seed=42)
    train2, val2 = stratified_split(toy_dataset, val_split=0.2, seed=42)
    assert train1 == train2
    assert val1 == val2


def test_different_seeds_can_change_the_split(toy_dataset: Path) -> None:
    _, val_a = stratified_split(toy_dataset, val_split=0.2, seed=1)
    _, val_b = stratified_split(toy_dataset, val_split=0.2, seed=2)
    assert val_a != val_b


def test_rejects_out_of_range_val_split(toy_dataset: Path) -> None:
    with pytest.raises(ValueError):
        stratified_split(toy_dataset, val_split=1.5, seed=42)


def test_write_then_read_split_round_trips(toy_dataset: Path, tmp_path: Path) -> None:
    train, val = stratified_split(toy_dataset, val_split=0.2, seed=42)
    out_dir = tmp_path / "model"
    write_split(train, val, out_dir)

    assert (out_dir / "train_files.txt").exists()
    assert (out_dir / "val_files.txt").exists()

    read_train, read_val = read_split(out_dir)
    assert read_train == train
    assert read_val == val
