#!/usr/bin/env python3
"""Generate a small synthetic image dataset for local development.

Not part of the graded deliverable — the real PlantVillage dataset is never
committed (see .gitignore). This exists so io.py / split.py / Distribution.py
/ train.py can be exercised end to end on ~40 images instead of 20,000
(leaffliction_team_plan.md section 4: "Test on 40 images, not 20,000").

Usage:
    python3 scripts/make_toy_dataset.py --out toy_dataset --per-class 10
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from leaffliction.io import save_image

PLANT = "Apple"
CLASSES = ["Apple_healthy", "Apple_scab", "Apple_rust"]
# Distinct base colors per class so a real model could learn to separate
# them; deliberately unbalanced to exercise the stratified-split logic.
CLASS_COLORS = {
    "Apple_healthy": (60, 160, 60),
    "Apple_scab": (140, 110, 60),
    "Apple_rust": (170, 90, 40),
}
CLASS_COUNTS = {"Apple_healthy": 16, "Apple_scab": 10, "Apple_rust": 6}


def make_image(color: tuple[int, int, int], seed: int, size: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.array(color, dtype=np.int16)
    noise = rng.integers(-20, 20, size=(size, size, 3))
    img = np.clip(base + noise, 0, 255).astype(np.uint8)
    return img


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("toy_dataset"))
    parser.add_argument(
        "--per-class",
        type=int,
        default=None,
        help="override: same image count for every class (ignores the "
        "built-in imbalance used to exercise stratified_split)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    counts = (
        {c: args.per_class for c in CLASSES} if args.per_class else CLASS_COUNTS
    )

    seed = 0
    for cls, n in counts.items():
        for i in range(n):
            img = make_image(CLASS_COLORS[cls], seed=seed)
            seed += 1
            save_image(img, args.out / PLANT / cls / f"image ({i}).JPG")

    total = sum(counts.values())
    print(f"wrote {total} images under {args.out / PLANT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
