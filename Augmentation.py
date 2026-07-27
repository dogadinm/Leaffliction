#!/usr/bin/env python3
"""Augment leaf images offline and balance training classes."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from leaffliction import SEED
from leaffliction import augment as aug
from leaffliction import io
from leaffliction.viz import show_augmentation_grid


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply six offline augmentations to leaf images.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="Image file or dataset directory to augment",
    )
    parser.add_argument(
        "--train-list",
        type=Path,
        help="train_files.txt from split.py (required for directory balancing)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("augmented_directory"),
        help="Output root for balanced augmentation (default: augmented_directory)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed (default: {SEED})",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Skip matplotlib preview (directory mode only)",
    )
    return parser.parse_args(argv)


def augment_single_image(path: Path, rng: np.random.Generator) -> int:
    """Display and write six augmentations for one image."""
    path = Path(path)
    if not path.is_file():
        print(f"Error: not a file: {path}", file=sys.stderr)
        return 1
    try:
        original = io.load_image(path)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    images = [original]
    titles = ["Original"]
    written = 0
    for kind in aug.AUGMENTATIONS:
        out = aug.apply(original, kind, rng)
        images.append(out)
        titles.append(kind)
        out_path = aug.augmented_name(path, kind)
        try:
            io.save_image(out, out_path)
            written += 1
        except OSError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    show_augmentation_grid(images, titles, suptitle=str(path))
    print(f"Wrote {written} augmented files next to {path}")
    return 0


def read_train_list(train_list: Path) -> list[Path]:
    paths: list[Path] = []
    for line in train_list.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            paths.append(Path(line))
    return paths


def balance_directory(
    train_paths: list[Path],
    output_root: Path,
    rng: np.random.Generator,
) -> int:
    """Oversample minority classes with offline augmentations (train split only)."""
    by_class: dict[str, list[Path]] = defaultdict(list)
    for path in train_paths:
        if not path.is_file():
            print(f"Warning: missing train file, skipping: {path}", file=sys.stderr)
            continue
        by_class[io.label_of(path)].append(path)

    if not by_class:
        print("Error: no valid training images found", file=sys.stderr)
        return 1

    target = max(len(paths) for paths in by_class.values())
    print(f"Balancing {len(by_class)} classes to {target} images each")

    total_written = 0
    for label, paths in sorted(by_class.items()):
        out_dir = output_root / label
        out_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        idx = 0
        while count < target:
            src = paths[idx % len(paths)]
            try:
                img = io.load_image(src)
            except ValueError as exc:
                print(f"Warning: {exc}", file=sys.stderr)
                idx += 1
                if idx >= len(paths) * 2:
                    break
                continue

            if count < len(paths) and idx < len(paths):
                dst = out_dir / src.name
                if not dst.exists():
                    io.save_image(img, dst)
                    total_written += 1
            else:
                kind = aug.AUGMENTATIONS[int(rng.integers(0, len(aug.AUGMENTATIONS)))]
                aug_img = aug.apply(img, kind, rng)
                stem = src.stem
                suffix = src.suffix
                name = f"{stem}_{kind}_{count}{suffix}"
                io.save_image(aug_img, out_dir / name)
                total_written += 1
            count += 1
            idx += 1

    print(f"Wrote {total_written} images under {output_root}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)

    if args.path is None:
        print("Error: path argument required", file=sys.stderr)
        return 1

    path = Path(args.path)
    if path.is_file():
        return augment_single_image(path, rng)

    if not path.is_dir():
        print(f"Error: path not found: {path}", file=sys.stderr)
        return 1

    if args.train_list is None:
        print(
            "Error: directory mode requires --train-list train_files.txt",
            file=sys.stderr,
        )
        return 1
    if not args.train_list.is_file():
        print(f"Error: train list not found: {args.train_list}", file=sys.stderr)
        return 1

    train_paths = read_train_list(args.train_list)
    return balance_directory(train_paths, args.output, rng)


if __name__ == "__main__":
    raise SystemExit(main())
