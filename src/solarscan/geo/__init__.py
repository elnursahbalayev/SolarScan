"""Georeferencing: map faults onto a farm layout, export GeoJSON, render maps.

Without real flight telemetry we demonstrate on a documented synthetic farm
(``FarmLayout``); the same interface accepts measured GCP corners in the field.
"""

from solarscan.geo.farm import FarmLayout
from solarscan.geo.georef import faults_to_geojson, georeference_faults, write_geojson
from solarscan.geo.visualize import render_fault_map_html, render_fault_map_png

__all__ = [
    "FarmLayout",
    "faults_to_geojson",
    "georeference_faults",
    "write_geojson",
    "render_fault_map_html",
    "render_fault_map_png",
]
