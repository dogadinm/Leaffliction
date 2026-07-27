"""Person B's augmentation functions — day-1 interface stub only.

This wires up the CLI contract all three of us froze on day 1
(leaffliction_team_plan.md section 2) so branches import cleanly before B's
implementation lands. Do not implement the bodies here — that's B's file.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

AUGMENTATIONS = ["Flip", "Rotate", "Skew", "Shear", "Crop", "Distortion"]


def apply(img: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    if kind not in AUGMENTATIONS:
        raise ValueError(f"unknown augmentation kind: {kind!r}")
    raise NotImplementedError("Person B implements this — see AUGMENTATIONS")


def augmented_name(src: Path, kind: str) -> Path:
    """e.g. src="image (1).JPG", kind="Flip" -> "image (1)_Flip.JPG" """
    if kind not in AUGMENTATIONS:
        raise ValueError(f"unknown augmentation kind: {kind!r}")
    raise NotImplementedError("Person B implements this")
