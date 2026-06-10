"""Farm layout — maps image positions to real-world module IDs and GPS.

Without a real drone flight (telemetry + overlapping frames) we cannot stitch a
true orthomosaic, so georeferencing is demonstrated on an explicit, documented
**synthetic farm**: a GPS bounding box plus a regular module grid. A fault's
pixel position in the (ortho)frame maps to a grid cell -> module ID + GPS
centroid. The same interface accepts real ground-control-point corners later, so
the field path is a drop-in (provide measured corners instead of synthetic ones).

This is clearly labelled as synthetic in every generated report.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from solarscan.schemas import BBox, GeoPoint


@dataclass(frozen=True)
class FarmLayout:
    name: str
    # Axis-aligned GPS bounds of the imaged area.
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    # Module grid covering the area.
    rows: int
    cols: int
    synthetic: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> FarmLayout:
        data = yaml.safe_load(Path(path).read_text())
        return cls(**data)

    @classmethod
    def sample(cls) -> FarmLayout:
        """A small synthetic 5x4 farm (matches the demo image's tiling)."""
        return cls(
            name="Synthetic Demo Farm",
            lat_min=40.4400,
            lat_max=40.4410,
            lon_min=49.8670,
            lon_max=49.8690,
            rows=4,
            cols=5,
        )

    def locate(self, bbox: BBox, image_w: float, image_h: float) -> tuple[str, GeoPoint]:
        """Map a bbox centre in image space to (module_id, GPS centroid).

        Image origin is top-left; north is up (v=0 -> lat_max).
        """
        cx = bbox.x + bbox.w / 2
        cy = bbox.y + bbox.h / 2
        u = min(max(cx / image_w, 0.0), 0.999999)
        v = min(max(cy / image_h, 0.0), 0.999999)

        col = int(u * self.cols)
        row = int(v * self.rows)
        module_id = f"R{row + 1}C{col + 1}"

        lon = self.lon_min + u * (self.lon_max - self.lon_min)
        lat = self.lat_max - v * (self.lat_max - self.lat_min)
        return module_id, GeoPoint(lat=lat, lon=lon)

    def center(self) -> GeoPoint:
        return GeoPoint(
            lat=(self.lat_min + self.lat_max) / 2,
            lon=(self.lon_min + self.lon_max) / 2,
        )
