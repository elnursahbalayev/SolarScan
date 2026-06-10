"""Ultralytics YOLO module detector (Stage 1).

Finds every PV module in a wide aerial thermal frame and returns boxes in the
``Detection`` schema, so it drops into the pipeline in place of the stub tiler.
Trained on the Zenodo UAV thermal dataset (single class: module).
"""

from __future__ import annotations

from pathlib import Path

from PIL.Image import Image

from solarscan.schemas import BBox, Detection


class YoloDetector:
    def __init__(self, weights: str | Path, conf: float = 0.25, device: str | None = None) -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.conf = conf
        self.device = device

    def detect(self, image: Image) -> list[Detection]:
        result = self.model.predict(image, conf=self.conf, verbose=False, device=self.device)[0]
        detections: list[Detection] = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                Detection(
                    bbox=BBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1),
                    score=float(box.conf[0]),
                )
            )
        return detections
