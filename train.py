#!/usr/bin/env python3
"""Part 4 (model half) — train the leaf-disease classifier.

Transfer learning on MobileNetV2: bring a frozen-base head to
convergence, then unfreeze the top block and fine-tune at a 10x lower
learning rate. A from-scratch conv-BN-ReLU-pool stack would work, but
needs far more epochs to clear 90% on a data set this small.

Split first, augment second. The stratified split is computed from the
raw directory and written to model/{train,val}_files.txt before a
single augmented pixel exists — that file is the proof. Only the train
half is then balanced by oversampling with the six offline
augmentations; validation always reads untouched originals.

--train-dir reads training pixels from a directory built the same way
instead of rebuilding it; never point it at one that predates the
split. --no-augment skips balancing, for fast toy-dataset runs.

Usage:
    ./train.py ./Apple/
    ./train.py ./Apple/ --no-augment
    ./train.py ./Apple/ --train-dir ./Apple_augmented_directory
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from leaffliction import IMG_SIZE, SEED, VAL_SPLIT
from leaffliction.augment import AUGMENTATIONS
from leaffliction.augment import apply as apply_augmentation
from leaffliction.io import iter_images, label_of, load_image, save_image, try_load_image
from leaffliction.metrics import (
    confusion_matrix,
    print_confusion_report,
    prove_validation_size,
)
from leaffliction.package import build_release
from leaffliction.split import stratified_split, write_split

MIN_VAL_IMAGES = 100


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="plant directory, e.g. ./Apple")
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=None,
        help="override: read TRAIN images from here instead of building "
        "the balanced augmented_directory internally (e.g. to reuse a "
        "directory already built from train_files.txt). Validation images "
        "always come from the raw split.",
    )
    parser.add_argument(
        "--augment-dir",
        type=Path,
        default=Path("augmented_directory"),
        help="where to build the balanced training set before training "
        "(default: augmented_directory). Ignored if --train-dir is given.",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="train directly on the raw stratified split, skipping the "
        "offline-augmentation balancing step (fast toy-dataset iteration). "
        "Ignored if --train-dir is given.",
    )
    parser.add_argument("--model-dir", type=Path, default=Path("model"))
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--val-split", type=float, default=VAL_SPLIT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--head-epochs", type=int, default=8)
    parser.add_argument("--fine-tune-epochs", type=int, default=6)
    parser.add_argument(
        "--fine-tune-layers", type=int, default=30,
        help="number of trailing base-model layers to unfreeze",
    )
    parser.add_argument(
        "--no-package", action="store_true",
        help="skip building the release zip + signature.txt",
    )
    parser.add_argument("--zip-path", type=Path, default=Path("leaffliction.zip"))
    parser.add_argument("--signature-path", type=Path, default=Path("signature.txt"))
    return parser.parse_args(argv)


def filter_readable(files: list[Path]) -> list[Path]:
    """Drop unreadable files up front, so a bad one fails here rather
    than mid-epoch."""
    kept = []
    for path in files:
        if try_load_image(path) is not None:
            kept.append(path)
    dropped = len(files) - len(kept)
    if dropped:
        print(f"warning: skipped {dropped} unreadable file(s)", file=sys.stderr)
    return kept


def build_augmented_train_dir(
    train_files: list[Path],
    output_root: Path,
    rng: np.random.Generator,
) -> list[Path]:
    """Oversample minority classes into output_root, train files only.

    train_files must be the train half of stratified_split — never a
    directory that existed before the split ran. Uses the same
    leaffliction.augment functions as Augmentation.py, so the two agree
    by construction rather than by convention.
    """
    by_class: dict[str, list[Path]] = defaultdict(list)
    for path in train_files:
        by_class[label_of(path)].append(path)

    target = max(len(paths) for paths in by_class.values())
    print(
        f"balancing {len(by_class)} classes to {target} images each "
        f"under {output_root}",
        file=sys.stderr,
    )

    written: list[Path] = []
    for label, paths in sorted(by_class.items()):
        paths = sorted(paths)
        out_dir = output_root / label
        out_dir.mkdir(parents=True, exist_ok=True)

        # Copy every original train-split image over unmodified first.
        for src in paths:
            dst = out_dir / src.name
            if not dst.exists():
                save_image(load_image(src), dst)
            written.append(dst)

        # Then oversample with the six offline augmentations up to target.
        count = len(paths)
        idx = 0
        while count < target:
            src = paths[idx % len(paths)]
            kind = AUGMENTATIONS[int(rng.integers(0, len(AUGMENTATIONS)))]
            aug_img = apply_augmentation(load_image(src), kind, rng)
            dst = out_dir / f"{src.stem}_{kind}_{count}{src.suffix}"
            save_image(aug_img, dst)
            written.append(dst)
            count += 1
            idx += 1

    print(f"wrote {len(written)} training images under {output_root}", file=sys.stderr)
    return written


def build_dataset(
    files: list[Path],
    label_index: dict[str, int],
    img_size: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    """Build the tf.data input pipeline.

    Decoding goes through tf.image rather than io.load_image because
    this runs inside the tf.data graph over the whole set every epoch,
    where a cv2 round-trip per image is the slow path. io.py still
    decides *which* files count as images — see filter_readable.
    """
    paths = [str(p) for p in files]
    labels = [label_index[label_of(p)] for p in files]

    def _decode(path, label):
        raw = tf.io.read_file(path)
        img = tf.image.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, (img_size, img_size))
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(max(len(paths), 1), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(_decode, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(num_classes: int, img_size: int) -> tuple[keras.Model, keras.Model]:
    base = keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False

    inputs = keras.Input(shape=(img_size, img_size, 3))
    # inputs are already [0, 1] float from build_dataset; MobileNetV2's
    # preprocess_input expects [0, 255] then rescales to [-1, 1] itself.
    x = keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    return model, base


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    tf.random.set_seed(args.seed)

    if not args.directory.is_dir():
        print(f"error: {args.directory} is not a directory", file=sys.stderr)
        return 1

    train_files, val_files = stratified_split(args.directory, args.val_split, args.seed)
    val_files = filter_readable(val_files)
    proof = prove_validation_size(val_files, minimum=MIN_VAL_IMAGES)
    if not proof["meets_requirement"]:
        print(
            f"warning: validation set has {proof['validation_count']} images, "
            f"the subject requires >= {proof['minimum_required']}",
            file=sys.stderr,
        )

    classes = sorted({label_of(p) for p in train_files} | {label_of(p) for p in val_files})
    label_index = {label: i for i, label in enumerate(classes)}

    args.model_dir.mkdir(parents=True, exist_ok=True)
    write_split(train_files, val_files, args.model_dir)

    if args.train_dir is not None:
        train_files = filter_readable(list(iter_images(args.train_dir)))
    elif args.no_augment:
        train_files = filter_readable(train_files)
    else:
        train_files = filter_readable(train_files)
        rng = np.random.default_rng(args.seed)
        train_files = build_augmented_train_dir(train_files, args.augment_dir, rng)

    train_ds = build_dataset(
        train_files, label_index, args.img_size, args.batch_size,
        shuffle=True, seed=args.seed,
    )
    val_ds = build_dataset(
        val_files, label_index, args.img_size, args.batch_size,
        shuffle=False, seed=args.seed,
    )

    model, base = build_model(len(classes), args.img_size)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model_path = args.model_dir / "model.keras"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(model_path), save_best_only=True, monitor="val_accuracy"
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=4, restore_best_weights=True
        ),
    ]

    history_head = model.fit(
        train_ds, validation_data=val_ds, epochs=args.head_epochs, callbacks=callbacks
    )

    base.trainable = True
    for layer in base.layers[: -args.fine_tune_layers]:
        layer.trainable = False
    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history_ft = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.head_epochs + args.fine_tune_epochs,
        initial_epoch=history_head.epoch[-1] + 1,
        callbacks=callbacks,
    )

    val_accuracy = float(
        max(history_head.history["val_accuracy"] + history_ft.history["val_accuracy"])
    )
    print(f"best val_accuracy: {val_accuracy:.4f} over {len(val_files)} images")

    idx_to_class = {idx: label for label, idx in label_index.items()}
    y_true = np.array([label_index[label_of(p)] for p in val_files])
    y_pred = np.argmax(model.predict(val_ds, verbose=0), axis=1)
    print_confusion_report(confusion_matrix(y_true, y_pred, len(classes)), idx_to_class)

    model.save(model_path)
    (args.model_dir / "labels.json").write_text(
        json.dumps({str(i): label for label, i in label_index.items()}, indent=2)
    )
    (args.model_dir / "config.json").write_text(
        json.dumps(
            {
                "img_size": args.img_size,
                "seed": args.seed,
                "val_split": args.val_split,
                "base_model": "MobileNetV2",
                "val_accuracy": val_accuracy,
                "val_count": len(val_files),
            },
            indent=2,
        )
    )

    if not args.no_package:
        digest = build_release(Path("."), args.zip_path, args.signature_path)
        print(f"packaged {args.zip_path} sha1={digest}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
