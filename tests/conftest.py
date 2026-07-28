from pathlib import Path

import matplotlib
import numpy as np
import pytest

# Must run before any test imports pyplot (directly or via leaffliction.viz):
# the default GUI backend has no display in CI/headless runs and either
# blocks on plt.show() or fails outright.
matplotlib.use("Agg")

from leaffliction.io import save_image  # noqa: E402


@pytest.fixture
def toy_dataset(tmp_path: Path) -> Path:
    """A tiny two-class, deliberately-imbalanced image tree for testing
    io.py / split.py without touching the real (gitignored) dataset."""
    root = tmp_path / "Apple"
    rng = np.random.default_rng(0)
    counts = {"Apple_healthy": 8, "Apple_scab": 3}
    for cls, n in counts.items():
        for i in range(n):
            img = rng.integers(0, 255, size=(16, 16, 3), dtype=np.uint8)
            save_image(img, root / cls / f"image ({i}).JPG")
    return root
