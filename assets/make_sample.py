"""Refresh the demo sample crop.

Prefers a real Hot-Spot crop from the InfraredSolarModules dataset (MIT, Raptor
Maps) when it's downloaded, so the demo shows genuine model output. Falls back to
a synthetic thermal-like placeholder only if the dataset isn't present.
Run: python assets/make_sample.py
"""

import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).parent / "sample_thermal.png"
DATA_ROOT = Path(__file__).parents[1] / "data/raw/infrared-solar-modules"


def from_dataset() -> bool:
    try:
        from solarscan.data.infrared_modules import find_metadata

        meta_path = find_metadata(DATA_ROOT)
    except (ModuleNotFoundError, FileNotFoundError):
        return False
    base = meta_path.parent
    meta = json.loads(meta_path.read_text())
    for entry in meta.values():
        if entry["anomaly_class"] == "Hot-Spot":
            shutil.copy(base / entry["image_filepath"], OUT)
            print(f"wrote real Hot-Spot crop -> {OUT}")
            return True
    return False


def synthetic() -> None:
    w, h = 320, 200
    img = Image.new("L", (w, h))
    px = img.load()
    for y in range(h):
        for x in range(w):
            px[x, y] = int(120 + 30 * (x / w))
    ImageDraw.Draw(img).ellipse((230, 60, 290, 120), fill=255)
    img.save(OUT)
    print(f"wrote SYNTHETIC placeholder -> {OUT} (download the dataset for a real sample)")


if __name__ == "__main__":
    if not from_dataset():
        synthetic()
