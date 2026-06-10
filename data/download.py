"""Fetch public datasets into data/raw/.

InfraredSolarModules is a public GitHub repo (images + module_metadata.json), so
we clone it directly. The Zenodo UAV set is gated, so we print instructions.

Usage:
    python data/download.py                      # all
    python data/download.py infrared-solar-modules
"""

from __future__ import annotations

import argparse
import subprocess
import urllib.request
import zipfile
from pathlib import Path

RAW = Path(__file__).parent / "raw"

INFRARED_REPO = "https://github.com/RaptorMaps/InfraredSolarModules.git"

# Open-access (CC-BY-4.0) — direct download works despite the title implying gating.
ZENODO_ZIP_URL = (
    "https://zenodo.org/api/records/16420123/files/"
    "Thermal%20PV%20Panel%20Detection%20Dataset%20for%20UAV%20Inspection.zip/content"
)


def clone_infrared() -> None:
    dest = RAW / "infrared-solar-modules"
    if dest.exists():
        print(f"[infrared-solar-modules] already present at {dest}")
        return
    print(f"[infrared-solar-modules] cloning {INFRARED_REPO} -> {dest}")
    subprocess.run(
        ["git", "clone", "--depth", "1", INFRARED_REPO, str(dest)],
        check=True,
    )
    print("    done.")


def fetch_zenodo() -> None:
    dest = RAW / "zenodo-uav-thermal"
    if (dest / "uav_pv").exists():
        print(f"[zenodo-uav-thermal] already present at {dest / 'uav_pv'}")
        return
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "dataset.zip"
    print(f"[zenodo-uav-thermal] downloading -> {zip_path}")
    urllib.request.urlretrieve(ZENODO_ZIP_URL, zip_path)
    print("    extracting…")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    # Normalise the spaced folder name to a path-friendly one.
    spaced = dest / "Thermal PV Panel Detection Dataset for UAV Inspection"
    if spaced.exists():
        spaced.rename(dest / "uav_pv")
    zip_path.unlink(missing_ok=True)
    print(f"    done -> {dest / 'uav_pv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        nargs="?",
        choices=["infrared-solar-modules", "zenodo-uav-thermal"],
        help="Dataset to fetch (default: all).",
    )
    args = parser.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)

    if args.dataset in (None, "infrared-solar-modules"):
        clone_infrared()
    if args.dataset in (None, "zenodo-uav-thermal"):
        fetch_zenodo()


if __name__ == "__main__":
    main()
