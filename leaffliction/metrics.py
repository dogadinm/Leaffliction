"""Evaluation metrics, and the proof that validation is big enough."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def validation_image_count(val_paths: list[Path]) -> int:
    """Return the number of validation images (must be >= 100 for grading)."""
    return len(val_paths)


def prove_validation_size(
    val_paths: list[Path],
    minimum: int = 100,
) -> dict[str, object]:
    """Report the validation size against the 100-image requirement."""
    count = validation_image_count(val_paths)
    return {
        "validation_count": count,
        "minimum_required": minimum,
        "meets_requirement": count >= minimum,
        "paths_sample": [str(p) for p in val_paths[:5]],
    }


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Compute a confusion matrix without sklearn."""
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred):
        matrix[int(truth), int(pred)] += 1
    return matrix


def per_class_accuracy(matrix: np.ndarray) -> dict[int, float]:
    """Return recall per class from a confusion matrix."""
    accuracies: dict[int, float] = {}
    for idx in range(matrix.shape[0]):
        total = matrix[idx].sum()
        if total == 0:
            accuracies[idx] = 0.0
        else:
            accuracies[idx] = float(matrix[idx, idx] / total)
    return accuracies


def print_confusion_report(
    matrix: np.ndarray,
    idx_to_class: dict[int, str],
) -> None:
    """Print confusion matrix and per-class accuracy as a table."""
    print("Confusion matrix (rows=true, cols=predicted):")
    header = " " * 16 + " ".join(f"{i:>8}" for i in range(matrix.shape[1]))
    print(header)
    for i in range(matrix.shape[0]):
        label = idx_to_class.get(i, str(i))[:14]
        row = " ".join(f"{v:>8}" for v in matrix[i])
        print(f"{label:>16} {row}")

    print("\nPer-class accuracy (recall):")
    for idx, acc in sorted(per_class_accuracy(matrix).items()):
        name = idx_to_class.get(idx, str(idx))
        print(f"  {name}: {acc:.1%}")
