#!/usr/bin/env python3
"""Part 3 — extract leaf features from a photo or a whole directory.

    ./Transformation.py "Apple_scab/image (1).JPG"   show them on screen
    ./Transformation.py -src Apple_scab -dst out     write them to disk
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from plantcv import plantcv as pcv

from leaffliction import io, transform, viz

# Command line flag -> name in transform.TRANSFORMS.
FLAGS = {
    "blur": "GaussianBlur",
    "mask": "Mask",
    "roi": "RoiObjects",
    "analyze": "AnalyzeObject",
    "pseudolandmarks": "Pseudolandmarks",
    "histogram": "ColorHistogram",
}


def build_parser() -> argparse.ArgumentParser:
    """Describe the command line."""
    parser = argparse.ArgumentParser(
        prog="Transformation.py",
        description="Extract leaf features from a photo or a directory.",
        epilog=(
            "examples:\n"
            "  ./Transformation.py \"Apple_scab/image (1).JPG\"\n"
            "  ./Transformation.py -src Apple_scab -dst out\n"
            "  ./Transformation.py -src Apple_scab -dst out -mask -roi\n"
            "\nWith no transformation flag every transformation runs."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "image", nargs="?", type=Path,
        help="a single image to display; omit it when using -src",
    )
    parser.add_argument(
        "-src", type=Path, metavar="DIR",
        help="directory of images to process",
    )
    parser.add_argument(
        "-dst", type=Path, metavar="DIR",
        help="directory to write the results into, required with -src",
    )
    for flag, kind in FLAGS.items():
        parser.add_argument(f"-{flag}", action="store_true",
                            help=f"apply {kind}")
    return parser


def chosen_transforms(args: argparse.Namespace) -> list[str]:
    """Return the transformations asked for, or all of them."""
    picked = [kind for flag, kind in FLAGS.items() if getattr(args, flag)]
    return picked or list(transform.TRANSFORMS)


def as_color(image: np.ndarray) -> np.ndarray:
    """Give a grayscale result three channels so it can be written."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image


def read_image(path: Path) -> np.ndarray:
    """Load an image, telling a missing path from an unreadable one.

    io.load_image calls both "could not read", which sends you hunting
    for a corrupt file when you only mistyped a path.
    """
    if not path.exists():
        raise FileNotFoundError(f"no such file: {path}")
    if path.is_dir():
        raise IsADirectoryError(f"{path} is a directory, use -src for that")
    return io.load_image(path)


def display(path: Path, kinds: list[str]) -> int:
    """Show the transformations of a single image."""
    try:
        rgb = read_image(path)
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    results = transform.apply_all(rgb, kinds)

    # The histogram is a plot with axes and a legend; squeezed into a
    # grid cell it is unreadable, so it gets its own window.
    panels = [kind for kind in kinds if kind != "ColorHistogram"]
    if panels:
        viz.show_grid(
            [rgb] + [results[kind] for kind in panels],
            ["Original"] + panels,
            ncols=3, suptitle=str(path), show_axes=True,
        )
    if "ColorHistogram" in results:
        viz.show_color_histogram(
            transform.color_histogram(rgb),
            transform.HISTOGRAM_COLORS,
            suptitle=f"color histogram - {path.name}",
        )
    return 0


def transform_file(path: Path, src: Path, dst: Path,
                   kinds: list[str]) -> int:
    """Write every chosen transformation of one image. Returns files written."""
    rgb = io.load_image(path)
    results = transform.apply_all(rgb, kinds)
    # Mirror the layout of src so class directories survive the trip.
    target_dir = dst / path.parent.relative_to(src)
    written = 0
    for kind, image in results.items():
        target = target_dir / f"{path.stem}_{kind}{path.suffix}"
        io.save_image(as_color(image), target)
        written += 1
    return written


def run_batch(src: Path, dst: Path, kinds: list[str]) -> int:
    """Process every image under src, skipping the ones that fail."""
    if not src.is_dir():
        print(f"Error: {src} is not a directory", file=sys.stderr)
        return 1

    images = list(io.iter_images(src))
    if not images:
        print(f"Error: no images found under {src}", file=sys.stderr)
        return 1

    written = 0
    skipped = []
    for path in images:
        try:
            written += transform_file(path, src, dst, kinds)
        except (ValueError, OSError, cv2.error) as exc:
            # One unreadable file must not abort a run of twenty thousand.
            skipped.append(path)
            print(f"Skipping {path}: {exc}", file=sys.stderr)

    print(f"{len(images) - len(skipped)}/{len(images)} images processed, "
          f"{written} files written to {dst}")
    if skipped:
        print(f"{len(skipped)} skipped", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    pcv.params.debug = None
    # imread prints its own C++ warning before returning None; we report
    # the failure ourselves, and the raw warning reads like a crash.
    cv2.setLogLevel(0)
    kinds = chosen_transforms(args)

    if args.src is not None:
        if args.image is not None:
            parser.error("give either an image or -src, not both")
        if args.dst is None:
            parser.error("-dst is required with -src")
        return run_batch(args.src, args.dst, kinds)

    if args.image is None:
        parser.error("give an image to display, or -src and -dst")
    return display(args.image, kinds)


if __name__ == "__main__":
    sys.exit(main())
