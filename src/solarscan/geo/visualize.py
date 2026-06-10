"""Render the georeferenced fault map.

Two outputs, both optional by dependency:
- PNG (matplotlib) — embedded in the PDF report; works headless.
- Interactive HTML (folium) — the click-around map for the web demo.
"""

from __future__ import annotations

from pathlib import Path

from solarscan.geo.farm import FarmLayout
from solarscan.schemas import Detection, Fault
from solarscan.taxonomy import Severity

DETECTED_COLOR = "#4ade80"  # green

SEVERITY_COLOR = {
    Severity.LOW: "#fde047",  # yellow
    Severity.MEDIUM: "#fb923c",  # orange
    Severity.HIGH: "#ef4444",  # red
    Severity.CRITICAL: "#991b1b",  # dark red
}


def render_fault_map_png(
    faults: list[Fault], farm: FarmLayout, out_path: str | Path
) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    # Draw the module grid in GPS space.
    dlon = (farm.lon_max - farm.lon_min) / farm.cols
    dlat = (farm.lat_max - farm.lat_min) / farm.rows
    for r in range(farm.rows):
        for c in range(farm.cols):
            x = farm.lon_min + c * dlon
            y = farm.lat_max - (r + 1) * dlat  # row 0 at top (north)
            ax.add_patch(
                mpatches.Rectangle((x, y), dlon, dlat, fill=False, edgecolor="#cbd5e1", lw=0.8)
            )

    for f in faults:
        if f.location is None:
            continue
        ax.scatter(
            f.location.lon,
            f.location.lat,
            c=SEVERITY_COLOR.get(f.severity, "#999999"),
            s=90,
            edgecolors="black",
            linewidths=0.5,
            zorder=3,
        )

    ax.set_xlim(farm.lon_min, farm.lon_max)
    ax.set_ylim(farm.lat_min, farm.lat_max)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    title = f"Fault map — {farm.name}"
    if farm.synthetic:
        title += "  (synthetic layout)"
    ax.set_title(title)
    handles = [
        mpatches.Patch(color=color, label=sev.value)
        for sev, color in SEVERITY_COLOR.items()
    ]
    ax.legend(handles=handles, title="Severity", loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def render_detection_map_png(
    detections: list[Detection], farm: FarmLayout, out_path: str | Path
) -> Path | None:
    """Plot georeferenced detected modules (detect-only mode) as green points."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lons = [d.location.lon for d in detections if d.location]
    lats = [d.location.lat for d in detections if d.location]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(lons, lats, c=DETECTED_COLOR, s=45, edgecolors="black", linewidths=0.4)
    ax.set_xlim(farm.lon_min, farm.lon_max)
    ax.set_ylim(farm.lat_min, farm.lat_max)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    title = f"Detected modules ({len(lons)}) — {farm.name}"
    if farm.synthetic:
        title += "  (synthetic GPS)"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def render_fault_map_html(
    faults: list[Fault], farm: FarmLayout, out_path: str | Path
) -> Path | None:
    try:
        import folium
    except ModuleNotFoundError:
        return None

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    center = farm.center()
    fmap = folium.Map(location=[center.lat, center.lon], zoom_start=18, tiles="OpenStreetMap")
    for f in faults:
        if f.location is None:
            continue
        folium.CircleMarker(
            location=[f.location.lat, f.location.lon],
            radius=7,
            color="black",
            weight=1,
            fill=True,
            fill_color=SEVERITY_COLOR.get(f.severity, "#999999"),
            fill_opacity=0.9,
            popup=f"{f.module_id}: {f.fault_class.value} ({f.severity.value})",
        ).add_to(fmap)
    fmap.save(str(out_path))
    return out_path
