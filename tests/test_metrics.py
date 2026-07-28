from pathlib import Path

import numpy as np

from leaffliction.metrics import (
    confusion_matrix,
    per_class_accuracy,
    prove_validation_size,
)


def test_confusion_matrix_counts_true_vs_predicted() -> None:
    y_true = np.array([0, 0, 1, 1, 2])
    y_pred = np.array([0, 1, 1, 1, 2])

    matrix = confusion_matrix(y_true, y_pred, num_classes=3)

    expected = np.array(
        [
            [1, 1, 0],
            [0, 2, 0],
            [0, 0, 1],
        ]
    )
    assert np.array_equal(matrix, expected)


def test_per_class_accuracy_is_recall_per_row() -> None:
    matrix = np.array(
        [
            [3, 1],
            [2, 4],
        ]
    )

    accuracies = per_class_accuracy(matrix)

    assert accuracies[0] == 3 / 4
    assert accuracies[1] == 4 / 6


def test_per_class_accuracy_handles_a_class_with_no_examples() -> None:
    matrix = np.array(
        [
            [0, 0],
            [1, 5],
        ]
    )

    accuracies = per_class_accuracy(matrix)

    assert accuracies[0] == 0.0


def test_prove_validation_size_reports_whether_minimum_is_met() -> None:
    paths = [Path(f"img_{i}.jpg") for i in range(120)]

    proof = prove_validation_size(paths, minimum=100)
    assert proof["validation_count"] == 120
    assert proof["minimum_required"] == 100
    assert proof["meets_requirement"] is True

    short_proof = prove_validation_size(paths[:50], minimum=100)
    assert short_proof["validation_count"] == 50
    assert short_proof["meets_requirement"] is False
