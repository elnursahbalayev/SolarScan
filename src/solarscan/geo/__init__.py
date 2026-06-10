"""Georeferencing: map faults onto a farm layout, export GeoJSON, render maps.

Without real flight telemetry we demonstrate on a documented synthetic farm
(``FarmLayout``); the same interface accepts measured GCP corners in the field.
"""

from solarscan.geo.farm import FarmLayout
from solarscan.geo.georef import (
    detections_to_geojson,
    faults_to_geojson,
    georeference_detections,
    georeference_faults,
    write_detections_geojson,
    write_geojson,
)
from solarscan.geo.visualize import (
    render_detection_map_png,
    render_fault_map_html,
    render_fault_map_png,
)

__all__ = [
    "FarmLayout",
    "faults_to_geojson",
    "detections_to_geojson",
    "georeference_faults",
    "georeference_detections",
    "write_geojson",
    "write_detections_geojson",
    "render_fault_map_html",
    "render_fault_map_png",
    "render_detection_map_png",
]
