"""Offline image augmentation for leaf disease classification."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

AUGMENTATIONS = ["Flip", "Rotate", "Skew", "Shear", "Crop", "Distortion"]


def augmented_name(src: Path, kind: str) -> Path:
    """Return output path: {stem}_{Kind}{suffix}, e.g. image (1)_Flip.JPG."""
    src = Path(src)
    return src.with_name(f"{src.stem}_{kind}{src.suffix}")


def apply(img: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    """Apply one augmentation and return a new RGB uint8 image."""
    if kind not in AUGMENTATIONS:
        raise ValueError(f"Unknown augmentation: {kind}")
    fn = {
        "Flip": _flip,
        "Rotate": _rotate,
        "Skew": _skew,
        "Shear": _shear,
        "Crop": _crop,
        "Distortion": _distortion,
    }[kind]
    return fn(img, rng)


def _flip(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    del rng
    return cv2.flip(img, 1)


def _rotate(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    angle = float(rng.uniform(-35.0, 35.0))
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        img,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def _shear(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Affine shear: parallel lines stay parallel."""
    h, w = img.shape[:2]
    shx = float(rng.uniform(-0.25, 0.25))
    shy = float(rng.uniform(-0.15, 0.15))
    matrix = np.array(
        [[1.0, shx, -shx * h / 2], [shy, 1.0, -shy * w / 2]],
        dtype=np.float32,
    )
    return cv2.warpAffine(
        img,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def _skew(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Projective skew: parallel lines converge (perspective warp)."""
    h, w = img.shape[:2]
    dx = float(rng.uniform(0.08, 0.18) * w)
    dy = float(rng.uniform(0.04, 0.12) * h)
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = np.float32([[dx, dy], [w - 1 - dx, 0], [w - 1, h - 1], [0, h - 1 - dy]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        img,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def _crop(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    scale = float(rng.uniform(0.75, 0.92))
    crop_w = max(1, int(w * scale))
    crop_h = max(1, int(h * scale))
    x0 = int(rng.integers(0, max(1, w - crop_w + 1)))
    y0 = int(rng.integers(0, max(1, h - crop_h + 1)))
    cropped = img[y0 : y0 + crop_h, x0 : x0 + crop_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def _distortion(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Barrel or pincushion radial distortion."""
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    k = float(rng.uniform(-0.35, 0.35))
    if abs(k) < 0.05:
        k = 0.15 if rng.random() < 0.5 else -0.15

    y, x = np.indices((h, w), dtype=np.float32)
    x = (x - cx) / cx
    y = (y - cy) / cy
    r2 = x * x + y * y
    factor = 1.0 + k * r2
    map_x = ((x * factor * cx) + cx).astype(np.float32)
    map_y = ((y * factor * cy) + cy).astype(np.float32)
    return cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
