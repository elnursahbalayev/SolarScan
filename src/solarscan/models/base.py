"""Model protocols. Downstream code depends on these, not on concrete models."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from PIL.Image import Image

from solarscan.schemas import Detection
from solarscan.taxonomy import FaultClass


@runtime_checkable
class Detector(Protocol):
    """Stage 1: propose candidate panel/module regions in a frame."""

    def detect(self, image: Image) -> list[Detection]: ...


@runtime_checkable
class FaultClassifier(Protocol):
    """Stage 2: classify a cropped module region into the fault taxonomy.

    Returns the predicted class and a full probability vector over ALL_CLASSES.
    """

    def classify(self, crop: Image) -> tuple[FaultClass, dict[FaultClass, float]]: ...
