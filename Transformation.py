#!/usr/bin/env python3
"""Part 3 — PlantCV transforms: display mode for a single image, batch mode
via -src/-dst.

Owned by Person C (leaffliction_team_plan.md section 1). This is a day-1
interface stub: the CLI contract is frozen so -h works, but the
implementation raises NotImplementedError until C's branch lands.

Usage:
    ./Transformation.py "<image>"
    ./Transformation.py -src X -dst Y -mask
"""
from __future__ import annotations

import argparse
from pathlib import Path

from leaffliction.transform import TRANSFORMS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, nargs="?", help="single image to display")
    parser.add_argument("-src", type=Path, help="source directory, batch mode")
    parser.add_argument("-dst", type=Path, help="destination directory, batch mode")
    for kind in TRANSFORMS:
        parser.add_argument(f"-{kind.lower()}", action="store_true", dest=kind)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    parse_args(argv)
    raise NotImplementedError(
        f"Transformation.py is Person C's file — implements {TRANSFORMS}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
