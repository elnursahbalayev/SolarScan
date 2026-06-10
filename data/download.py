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
from pathlib import Path

RAW = Path(__file__).parent / "raw"

INFRARED_REPO = "https://github.com/RaptorMaps/InfraredSolarModules.git"

ZENODO = {
    "url": "https://zenodo.org/records/16420123",
    "note": "Gated download — fetch the archive from the record and extract into "
    "data/raw/zenodo-uav-thermal/.",
}


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


def show_zenodo() -> None:
    print("[zenodo-uav-thermal] manual download required")
    print(f"    URL : {ZENODO['url']}")
    print(f"    Note: {ZENODO['note']}")


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
        show_zenodo()


if __name__ == "__main__":
    main()
