"""Pydantic data contracts that flow through the pipeline.

These types are the stable interface between stages: a model returns
``Detection``s, classification + scoring turns them into ``Fault``s, and the
report stage consumes an ``InspectionReport``. Keeping these concrete from
Phase 0 means we can swap the stub model for the real one without touching
downstream code.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from solarscan.taxonomy import FaultClass, Severity


class BBox(BaseModel):
    """Pixel-space bounding box (top-left origin)."""

    x: float
    y: float
    w: float
    h: float


class GeoPoint(BaseModel):
    lat: float
    lon: float


class Detection(BaseModel):
    """A region proposed by the detector, before fault classification."""

    bbox: BBox
    score: float = Field(ge=0.0, le=1.0)


class Fault(BaseModel):
    """A classified, scored fault tied to an image region (and optionally GPS)."""

    fault_class: FaultClass
    confidence: float = Field(ge=0.0, le=1.0)
    severity: Severity
    bbox: BBox | None = None
    location: GeoPoint | None = None
    estimated_yield_loss_fraction: float = Field(ge=0.0, le=1.0)
    module_id: str | None = None


class ReportSummary(BaseModel):
    n_modules_inspected: int
    n_faults: int
    faults_by_class: dict[str, int]
    faults_by_severity: dict[str, int]
    estimated_total_yield_loss_kwh: float | None = None


class InspectionReport(BaseModel):
    source: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model_version: str = "stub-0.1.0"
    faults: list[Fault]
    summary: ReportSummary
    notes: list[str] = Field(default_factory=list)
    # All module regions the detector proposed (faulty + healthy), for rendering.
    detections: list[Detection] = Field(default_factory=list)
