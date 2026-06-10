"""Georeference faults against a FarmLayout and export GeoJSON."""

from __future__ import annotations

import json
from pathlib import Path

from solarscan.geo.farm import FarmLayout
from solarscan.schemas import Detection, Fault


def georeference_detections(
    detections: list[Detection], farm: FarmLayout, image_w: float, image_h: float
) -> list[Detection]:
    """Attach module_id + GPS to each detected module (detect-only mode)."""
    for d in detections:
        module_id, point = farm.locate(d.bbox, image_w, image_h)
        d.module_id = module_id
        d.location = point
    return detections


def detections_to_geojson(detections: list[Detection], farm: FarmLayout) -> dict:
    """GeoJSON FeatureCollection of detected modules (no fault assessment)."""
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [d.location.lon, d.location.lat]},
            "properties": {
                "module_id": d.module_id,
                "detection_score": round(d.score, 4),
                "status": "detected",
            },
        }
        for d in detections
        if d.location is not None
    ]
    return {
        "type": "FeatureCollection",
        "properties": {"farm": farm.name, "synthetic": farm.synthetic, "mode": "detection-only"},
        "features": features,
    }


def georeference_faults(
    faults: list[Fault], farm: FarmLayout, image_w: float, image_h: float
) -> list[Fault]:
    """Attach module_id + GPS to each fault (in place) using the farm layout."""
    for f in faults:
        if f.bbox is None:
            continue
        module_id, point = farm.locate(f.bbox, image_w, image_h)
        f.module_id = module_id
        f.location = point
    return faults


def faults_to_geojson(faults: list[Fault], farm: FarmLayout) -> dict:
    """Build a GeoJSON FeatureCollection of located faults."""
    features = []
    for f in faults:
        if f.location is None:
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [f.location.lon, f.location.lat],
                },
                "properties": {
                    "fault_class": f.fault_class.value,
                    "severity": f.severity.value,
                    "confidence": round(f.confidence, 4),
                    "module_id": f.module_id,
                    "yield_loss_fraction": f.estimated_yield_loss_fraction,
                },
            }
        )
    return {
        "type": "FeatureCollection",
        "properties": {"farm": farm.name, "synthetic": farm.synthetic},
        "features": features,
    }


def write_geojson(faults: list[Fault], farm: FarmLayout, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(faults_to_geojson(faults, farm), indent=2))
    return out_path


def write_detections_geojson(
    detections: list[Detection], farm: FarmLayout, out_path: str | Path
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(detections_to_geojson(detections, farm), indent=2))
    return out_path
