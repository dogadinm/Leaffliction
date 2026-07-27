"""TensorFlow input pipeline for training (Person B)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from leaffliction import IMG_SIZE
from leaffliction import io


def build_label_map(paths: list[Path]) -> tuple[dict[str, int], dict[int, str]]:
    """Build class index maps from image paths."""
    labels = sorted({io.label_of(path) for path in paths})
    class_to_idx = {label: idx for idx, label in enumerate(labels)}
    idx_to_class = {idx: label for label, idx in class_to_idx.items()}
    return class_to_idx, idx_to_class


def write_labels_json(model_dir: Path, idx_to_class: dict[int, str]) -> Path:
    """Write labels.json in the model contract format."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    out = model_dir / "labels.json"
    payload = {str(k): v for k, v in sorted(idx_to_class.items())}
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out


def _load_and_preprocess(
    path: tf.Tensor,
    label: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32) / 255.0
    return img, label


def make_datasets(
    train_paths: list[Path],
    val_paths: list[Path],
    *,
    batch_size: int = 32,
    model_dir: Path | None = None,
) -> tuple[tf.data.Dataset, tf.data.Dataset, dict[int, str]]:
    """Build batched train/val datasets and optionally write labels.json."""
    all_paths = train_paths + val_paths
    class_to_idx, idx_to_class = build_label_map(all_paths)

    def paths_to_ds(paths: list[Path], training: bool) -> tf.data.Dataset:
        str_paths = [str(p) for p in paths]
        labels = [class_to_idx[io.label_of(p)] for p in paths]
        ds = tf.data.Dataset.from_tensor_slices((str_paths, labels))
        ds = ds.map(
            _load_and_preprocess,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        if training:
            ds = ds.shuffle(min(len(paths), 1024), reshuffle_each_iteration=True)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = paths_to_ds(train_paths, training=True)
    val_ds = paths_to_ds(val_paths, training=False)

    if model_dir is not None:
        write_labels_json(model_dir, idx_to_class)

    return train_ds, val_ds, idx_to_class


def load_split_lists(
    train_list: Path,
    val_list: Path,
) -> tuple[list[Path], list[Path]]:
    """Load train/val path lists written by split.py."""
    def read_list(path: Path) -> list[Path]:
        return [
            Path(line.strip())
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    return read_list(train_list), read_list(val_list)


def dataset_from_augmented_root(
    root: Path,
    *,
    batch_size: int = 32,
    val_fraction: float = 0.2,
    seed: int = 42,
    model_dir: Path | None = None,
) -> tuple[tf.data.Dataset, tf.data.Dataset, dict[int, str], list[Path], list[Path]]:
    """
    Build datasets from augmented_directory layout.

    Uses stratified per-class holdout when no val list exists yet.
    """
    root = Path(root)
    all_paths = list(io.iter_images(root))
    if not all_paths:
        raise ValueError(f"No images found under {root}")

    rng = np.random.default_rng(seed)
    by_class: dict[str, list[Path]] = {}
    for path in all_paths:
        by_class.setdefault(io.label_of(path), []).append(path)

    train_paths: list[Path] = []
    val_paths: list[Path] = []
    for paths in by_class.values():
        paths = sorted(paths)
        rng.shuffle(paths)
        n_val = max(1, int(round(len(paths) * val_fraction)))
        val_paths.extend(paths[:n_val])
        train_paths.extend(paths[n_val:])

    train_ds, val_ds, idx_to_class = make_datasets(
        train_paths,
        val_paths,
        batch_size=batch_size,
        model_dir=model_dir,
    )
    return train_ds, val_ds, idx_to_class, train_paths, val_paths
