"""Tests for georeferencing — pure-Python, run on core deps (no ml extra needed)."""

from solarscan.geo.farm import FarmLayout
from solarscan.geo.georef import faults_to_geojson, georeference_faults
from solarscan.schemas import BBox, Fault
from solarscan.taxonomy import FaultClass, Severity


def _farm() -> FarmLayout:
    return FarmLayout.sample()  # 5 cols x 4 rows


def test_top_left_maps_to_r1c1_and_north_west():
    farm = _farm()
    module_id, pt = farm.locate(BBox(x=0, y=0, w=10, h=10), image_w=1000, image_h=1000)
    assert module_id == "R1C1"
    # top -> max latitude (north), left -> min longitude (west)
    assert pt.lat > farm.center().lat
    assert pt.lon < farm.center().lon


def test_bottom_right_maps_to_last_cell():
    farm = _farm()
    module_id, pt = farm.locate(BBox(x=990, y=990, w=8, h=8), image_w=1000, image_h=1000)
    assert module_id == f"R{farm.rows}C{farm.cols}"
    assert pt.lat < farm.center().lat
    assert pt.lon > farm.center().lon


def test_gps_within_bounds():
    farm = _farm()
    _, pt = farm.locate(BBox(x=500, y=500, w=10, h=10), 1000, 1000)
    assert farm.lat_min <= pt.lat <= farm.lat_max
    assert farm.lon_min <= pt.lon <= farm.lon_max


def test_georeference_and_geojson():
    farm = _farm()
    faults = [
        Fault(
            fault_class=FaultClass.HOT_SPOT,
            confidence=0.9,
            severity=Severity.HIGH,
            bbox=BBox(x=100, y=100, w=20, h=20),
            estimated_yield_loss_fraction=0.1,
        )
    ]
    georeference_faults(faults, farm, 1000, 1000)
    assert faults[0].location is not None
    assert faults[0].module_id is not None

    fc = faults_to_geojson(faults, farm)
    assert fc["type"] == "FeatureCollection"
    assert fc["properties"]["synthetic"] is True
    assert len(fc["features"]) == 1
    feat = fc["features"][0]
    assert feat["geometry"]["type"] == "Point"
    # GeoJSON is [lon, lat] order
    assert feat["geometry"]["coordinates"][0] == faults[0].location.lon
    assert feat["properties"]["fault_class"] == "Hot-Spot"
