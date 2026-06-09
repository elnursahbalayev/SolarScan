"""Deterministic, dependency-free stub models for the Phase-0 slice.

These let the full pipeline run end-to-end (and tests pass) with no torch and no
GPU. They derive a pseudo-prediction from simple image statistics so the output
is deterministic and varies with the input, but they make NO real claim of
accuracy. Replace with :mod:`solarscan.models.timm_classifier` and the
Ultralytics detector in Phase 1.
"""

from __future__ import annotations

import hashlib

from PIL.Image import Image

from solarscan.schemas import BBox, Detection
from solarscan.taxonomy import ALL_CLASSES, FaultClass


def _seed_from_image(image: Image) -> int:
    digest = hashlib.sha256(image.tobytes()).digest()
    return int.from_bytes(digest[:4], "big")


class StubDetector:
    """Tiles the frame into a grid, treating each tile as a candidate module.

    A real detector (Phase 1) replaces this with learned panel boxes; the grid is
    enough to make the end-to-end slice surface localised hot regions.
    """

    def __init__(self, grid: tuple[int, int] = (5, 4)) -> None:
        self.cols, self.rows = grid

    def detect(self, image: Image) -> list[Detection]:
        w, h = image.size
        tw, th = w / self.cols, h / self.rows
        return [
            Detection(bbox=BBox(x=c * tw, y=r * th, w=tw, h=th), score=1.0)
            for r in range(self.rows)
            for c in range(self.cols)
        ]


class StubClassifier:
    """Maps image brightness to a plausible fault class. Heuristic only."""

    def classify(self, crop: Image) -> tuple[FaultClass, dict[FaultClass, float]]:
        gray = crop.convert("L")
        hist = gray.histogram()
        total = sum(hist)
        mean = sum(i * n for i, n in enumerate(hist)) / total if total else 0.0

        # Brighter regions in thermal IR ~ hotter ~ more likely a hot-spot/diode fault.
        seed = _seed_from_image(crop)
        if mean > 200:
            predicted = FaultClass.HOT_SPOT
        elif mean > 170:
            predicted = FaultClass.DIODE
        elif mean > 140:
            predicted = (FaultClass.CRACKING, FaultClass.SOILING)[seed % 2]
        else:
            predicted = FaultClass.NO_ANOMALY

        # A peaked-but-soft probability vector around the prediction.
        probs = {c: 0.02 for c in ALL_CLASSES}
        probs[predicted] = 0.78
        total = sum(probs.values())
        probs = {c: p / total for c, p in probs.items()}
        return predicted, probs
