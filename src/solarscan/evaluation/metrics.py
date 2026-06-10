"""Classification metrics — the seniority signal for this project.

Reports accuracy, macro-F1 (the headline given the 57:1 imbalance), weighted-F1,
and per-class precision/recall/F1/support, plus a confusion matrix. Macro-F1 and
per-class recall on rare faults matter far more than raw accuracy here: a model
that predicts "No-Anomaly" for everything scores 50% accuracy but is useless.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from solarscan.taxonomy import ALL_CLASSES

CLASS_NAMES = [c.value for c in ALL_CLASSES]


@dataclass
class ClassificationMetrics:
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class: dict[str, dict[str, float]]
    confusion: list[list[int]] = field(default_factory=list)

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))
        return path

    def summary_line(self) -> str:
        return (
            f"acc={self.accuracy:.4f}  macro-F1={self.macro_f1:.4f}  "
            f"weighted-F1={self.weighted_f1:.4f}"
        )


def compute_metrics(y_true: list[int], y_pred: list[int]) -> ClassificationMetrics:
    labels = list(range(len(ALL_CLASSES)))
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        CLASS_NAMES[i]: {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in labels
    }
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        per_class=per_class,
        confusion=cm.tolist(),
    )


def save_confusion_png(cm: list[list[int]], path: str | Path) -> Path | None:
    """Render the confusion matrix to PNG if matplotlib is available."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.array(cm, dtype=float)
    row_sums = arr.sum(axis=1, keepdims=True)
    norm = np.divide(arr, row_sums, out=np.zeros_like(arr), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(norm, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=90, fontsize=7)
    ax.set_yticklabels(CLASS_NAMES, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (row-normalised)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
