"""[Phase 3] Orthomosaic stitching + georeferencing.

Maps per-frame pixel detections onto a farm layout in real-world coordinates so
faults carry GPS and can be plotted on a map. Requires the ``geo`` extra
(opencv, rasterio, pyproj, folium).

Planned API:
    stitch(frames) -> orthomosaic
    georeference(orthomosaic, gcps | flight_log) -> transform
    locate(detection, transform) -> GeoPoint
    to_geojson(faults) -> FeatureCollection
"""
