"""Image I/O helpers (interface owned by A; minimal implementation for B)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def iter_images(root: Path) -> Iterator[Path]:
    """Yield image paths under root (class subdirectories)."""
    root = Path(root)
    if not root.is_dir():
        return
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def load_image(path: Path) -> np.ndarray:
    """Load an image as RGB uint8 HxWx3."""
    path = Path(path)
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def label_of(path: Path) -> str:
    """Return the class label from the parent directory name."""
    return Path(path).parent.name


def class_counts(root: Path) -> dict[str, int]:
    """Count images per class under root."""
    counts: Counter[str] = Counter()
    for path in iter_images(root):
        counts[label_of(path)] += 1
    return dict(sorted(counts.items()))


def save_image(img: np.ndarray, path: Path) -> None:
    """Save an RGB uint8 image to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise OSError(f"Could not write image: {path}")
