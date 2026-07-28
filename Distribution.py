#!/usr/bin/env python3
"""Part 1 — plot the class distribution of a plant-type directory.

Renders a pie chart (class share) and a bar chart (image count), labelled
from the subdirectory names so it works on any plant directory, not just the
one it was developed against.

Usage:
    ./Distribution.py ./Apple
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from leaffliction.io import class_counts
from leaffliction.viz import plot_class_distribution


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        type=Path,
        help="plant directory containing one subdirectory per class",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.directory.is_dir():
        print(f"error: {args.directory} is not a directory", file=sys.stderr)
        return 1

    counts = class_counts(args.directory)
    if not counts:
        print(f"error: no images found under {args.directory}", file=sys.stderr)
        return 1

    plot_class_distribution(counts, args.directory.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
