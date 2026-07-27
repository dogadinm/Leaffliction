"""Person C's PlantCV transforms — day-1 interface stub only.

This wires up the CLI contract all three of us froze on day 1
(leaffliction_team_plan.md section 2) so branches import cleanly before C's
implementation lands. Do not implement the bodies here — that's C's file.
"""
from __future__ import annotations

from typing import Any

import numpy as np

TRANSFORMS = [
    "GaussianBlur",
    "Mask",
    "RoiObjects",
    "AnalyzeObject",
    "Pseudolandmarks",
    "ColorHistogram",
]


def apply(img: np.ndarray, kind: str) -> Any:
    if kind not in TRANSFORMS:
        raise ValueError(f"unknown transform kind: {kind!r}")
    raise NotImplementedError("Person C implements this — see TRANSFORMS")
