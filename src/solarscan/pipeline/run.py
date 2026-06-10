"""End-to-end orchestration of the SolarScan stages.

Phase 0 wires the stub detector + classifier through scoring into an
``InspectionReport``. The georeferencing stage is a no-op placeholder until
Phase 3. Swapping in the real models requires no change here — only the
``Detector`` / ``FaultClassifier`` passed in.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from PIL import Image

from solarscan.geo.farm import FarmLayout
from solarscan.geo.georef import georeference_detections, georeference_faults
from solarscan.models import Detector, FaultClassifier, StubClassifier, StubDetector
from solarscan.report.yield_loss import estimate_module_loss_kwh
from solarscan.schemas import Fault, InspectionReport, ReportSummary
from solarscan.taxonomy import default_severity, is_anomaly, yield_loss_fraction


def run_pipeline(
    image_path: str | Path,
    detector: Detector | None = None,
    classifier: FaultClassifier | None = None,
    model_version: str = "stub-0.1.0",
    farm: FarmLayout | None = None,
    detect_only: bool = False,
) -> InspectionReport:
    """Run all stages on a single image and return the inspection report.

    If ``farm`` is given, results are georeferenced (module_id + GPS) against it.
    ``detect_only`` skips fault classification entirely — used for aerial frames
    where the classifier (trained on a different IR dataset) is not calibrated for
    the source camera. The report then lists detected/located modules, no faults.
    """
    detector = detector or StubDetector()
    classifier = classifier or StubClassifier()
    stub_detector = isinstance(detector, StubDetector)
    stub_classifier = isinstance(classifier, StubClassifier)

    image_path = Path(image_path)
    image = Image.open(image_path)
    image_w, image_h = image.size

    detections = detector.detect(image)

    faults: list[Fault] = []
    total_loss_kwh = 0.0
    for det in [] if detect_only else detections:
        crop = image.crop(
            (
                int(det.bbox.x),
                int(det.bbox.y),
                int(det.bbox.x + det.bbox.w),
                int(det.bbox.y + det.bbox.h),
            )
        )
        fault_class, probs = classifier.classify(crop)
        if not is_anomaly(fault_class):
            continue
        confidence = probs[fault_class]
        faults.append(
            Fault(
                fault_class=fault_class,
                confidence=confidence,
                severity=default_severity(fault_class),
                bbox=det.bbox,
                estimated_yield_loss_fraction=yield_loss_fraction(fault_class),
            )
        )
        total_loss_kwh += estimate_module_loss_kwh(fault_class)

    if farm is not None:
        if detect_only:
            georeference_detections(detections, farm, image_w, image_h)
        else:
            georeference_faults(faults, farm, image_w, image_h)

    summary = ReportSummary(
        n_modules_inspected=len(detections),
        n_faults=len(faults),
        faults_by_class=dict(Counter(f.fault_class.value for f in faults)),
        faults_by_severity=dict(Counter(f.severity.value for f in faults)),
        estimated_total_yield_loss_kwh=None if detect_only else round(total_loss_kwh, 3),
    )

    notes = []
    if detect_only:
        notes.append(
            "Detection-only run: modules detected & georeferenced, but fault-type "
            "classification was NOT performed — the classifier is trained on a different "
            "IR dataset and is not calibrated for this source camera (domain gap). "
            "See the model card for classifier performance on labelled IR data."
        )
        if stub_detector:
            notes.insert(0, "Detector is the heuristic stub (frame tiling) — not a trained model.")
    else:
        notes.append(
            "Yield-loss figures are estimates (module rated power x fault loss-fraction "
            "x peak-sun-hours), provided to prioritise O&M, not as metered values."
        )
        if stub_detector or stub_classifier:
            notes.insert(0, "Output uses a heuristic stub model — not a validated detector.")
    if farm is not None:
        notes.append(
            f"Locations georeferenced against farm layout '{farm.name}'"
            + (" (SYNTHETIC layout — for demonstration)." if farm.synthetic else ".")
        )
    else:
        notes.append("Georeferencing not applied (no farm layout supplied).")

    return InspectionReport(
        source=str(image_path),
        model_version=model_version,
        faults=faults,
        summary=summary,
        notes=notes,
        detections=detections,
    )
