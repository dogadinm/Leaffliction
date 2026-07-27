#!/usr/bin/env python3
"""Part 2 — apply the 6 augmentations to a single image, or balance a whole
directory in batch mode.

Owned by Person B (leaffliction_team_plan.md section 1). This is a day-1
interface stub: the CLI contract is frozen so this can be invoked and its
--help works, but the implementation raises NotImplementedError until B's
branch lands.

Usage:
    ./Augmentation.py "./Apple/Apple_healthy/image (1).JPG"
"""
from __future__ import annotations

import argparse
from pathlib import Path

from leaffliction.augment import AUGMENTATIONS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path", type=Path, help="image file, or a directory to balance"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    parse_args(argv)
    raise NotImplementedError(
        f"Augmentation.py is Person B's file — implements {AUGMENTATIONS}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
