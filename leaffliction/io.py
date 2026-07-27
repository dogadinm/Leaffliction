"""Shared image I/O primitives.

Every module in this repo loads/saves images through here — it is the only
file allowed to call cv2.imread/cv2.imwrite directly. That is what kills the
classic bug where a mask works in one file (RGB) and silently breaks in
another (BGR): there is exactly one place the byte order gets decided.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def iter_images(root: Path) -> Iterator[Path]:
    """Yield every image file under root, sorted for reproducibility."""
    root = Path(root)
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def load_image(path: Path) -> np.ndarray:
    """Load an image as RGB uint8, HxWx3. Raises ValueError if unreadable."""
    path = Path(path)
    data = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if data is None:
        raise ValueError(f"unreadable image: {path}")
    return cv2.cvtColor(data, cv2.COLOR_BGR2RGB)


def try_load_image(path: Path) -> Optional[np.ndarray]:
    """Like load_image but never raises.

    Logs a warning and returns None on failure. Use this at any boundary that
    must survive a corrupt or non-image file without a traceback
    (Distribution.py, train.py's file filtering, predict.py).
    """
    try:
        return load_image(path)
    except (ValueError, cv2.error) as exc:
        logger.warning("skipping unreadable file %s: %s", path, exc)
        return None


def label_of(path: Path) -> str:
    """Class label is the immediate parent directory name."""
    return Path(path).parent.name


def class_counts(root: Path) -> dict[str, int]:
    """Number of images per class under root, keyed by subdirectory name."""
    counts: dict[str, int] = {}
    for path in iter_images(root):
        label = label_of(path)
        counts[label] = counts.get(label, 0) + 1
    return counts


def save_image(img: np.ndarray, path: Path) -> None:
    """Save an RGB uint8 HxWx3 array to path, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise OSError(f"failed to write image: {path}")
