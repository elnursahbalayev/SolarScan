"""Model interfaces and implementations.

The pipeline depends only on the ``FaultClassifier`` / ``Detector`` protocols,
so a lightweight heuristic stub (Phase 0, no GPU) and the real timm/Ultralytics
models (Phase 1) are interchangeable.
"""

from solarscan.models.base import Detector, FaultClassifier
from solarscan.models.stub import StubClassifier, StubDetector, WholeFrameDetector

__all__ = [
    "Detector",
    "FaultClassifier",
    "StubClassifier",
    "StubDetector",
    "WholeFrameDetector",
]
