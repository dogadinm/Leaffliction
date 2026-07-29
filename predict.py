#!/usr/bin/env python3
"""Part 4 (inference) — name the disease on a single leaf image.

Shows the original, a transformed version for context, and the label.

Loads only model.keras, labels.json and config.json, and never imports
train.py: that is what lets training change shape without breaking
inference.

Usage:
    ./predict.py "<image>"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

from leaffliction.io import try_load_image
from leaffliction.transform import TRANSFORMS
from leaffliction.transform import apply as transform_apply
from leaffliction.viz import show_image_grid


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--model-dir", type=Path, default=Path("model"))
    parser.add_argument(
        "--transform",
        default="GaussianBlur",
        choices=TRANSFORMS,
        help="which leaffliction.transform kind to display next to the prediction",
    )
    parser.add_argument(
        "--no-display", action="store_true", help="skip the matplotlib window"
    )
    return parser.parse_args(argv)


def load_model_bundle(model_dir: Path):
    model_path = model_dir / "model.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"no trained model at {model_path}, run train.py first")
    model = keras.models.load_model(model_path)
    labels = json.loads((model_dir / "labels.json").read_text())
    config = json.loads((model_dir / "config.json").read_text())
    return model, labels, config


def predict(model, labels: dict, config: dict, img: np.ndarray) -> tuple[str, float]:
    size = config["img_size"]
    resized = tf.image.resize(img, (size, size)).numpy() / 255.0
    batch = np.expand_dims(resized, axis=0)
    probs = model.predict(batch, verbose=0)[0]
    idx = int(np.argmax(probs))
    return labels[str(idx)], float(probs[idx])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    img = try_load_image(args.image)
    if img is None:
        print(f"error: cannot read image {args.image}", file=sys.stderr)
        return 1

    try:
        model, labels, config = load_model_bundle(args.model_dir)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    label, confidence = predict(model, labels, config, img)

    transformed = img
    try:
        transformed = transform_apply(img, args.transform)
    except (ValueError, cv2.error) as exc:
        # argparse already validated the name, so this is a transform
        # failing on this particular image — a leaf PlantCV cannot
        # segment. Show the original twice rather than lose the label.
        print(f"warning: {args.transform} failed on this image: {exc}", file=sys.stderr)

    print(f"{args.image.name}: {label} ({confidence:.1%})")

    if not args.no_display:
        show_image_grid(
            [img, transformed],
            ["original", args.transform],
            suptitle=f"{label} ({confidence:.1%})",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
