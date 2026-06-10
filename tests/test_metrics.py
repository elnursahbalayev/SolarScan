"""Tests for the metrics harness. Skipped when scikit-learn is absent."""

import pytest

pytest.importorskip("sklearn", reason="requires the `ml` extra")

from solarscan.evaluation.metrics import compute_metrics  # noqa: E402
from solarscan.taxonomy import ALL_CLASSES  # noqa: E402


def test_perfect_predictions():
    y = [0, 1, 2, 3, 0, 1]
    m = compute_metrics(y, y)
    assert m.accuracy == 1.0
    assert m.macro_f1 == 1.0


def test_majority_class_collapse_tanks_macro_f1():
    # Predicting all class-0 on a balanced 3-class set: high-ish acc, terrible macro-F1.
    y_true = [0, 1, 2] * 10
    y_pred = [0] * 30
    m = compute_metrics(y_true, y_pred)
    assert m.macro_f1 < 0.3  # the imbalance trap this project is built to avoid
    assert len(m.per_class) == len(ALL_CLASSES)


def test_confusion_matrix_shape():
    n = len(ALL_CLASSES)
    m = compute_metrics([0, 1], [0, 1])
    assert len(m.confusion) == n
    assert all(len(row) == n for row in m.confusion)
