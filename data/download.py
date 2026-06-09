"""Fetch public datasets into data/raw/.

Phase 0 ships URLs + manual instructions (the InfraredSolarModules repo and the
Zenodo record both gate downloads). Phase 1 will automate via the Kaggle API and
Zenodo's REST API. Run:  python data/download.py --help
"""

from __future__ import annotations

import argparse
from pathlib import Path

RAW = Path(__file__).parent / "raw"

DATASETS = {
    "infrared-solar-modules": {
        "url": "https://github.com/RaptorMaps/InfraredSolarModules",
        "note": "Clone the repo; images + module_metadata.json live under the dataset folder.",
    },
    "zenodo-uav-thermal": {
        "url": "https://zenodo.org/records/16420123",
        "note": "Download the archive from the Zenodo record and extract here.",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", choices=list(DATASETS), help="Dataset to fetch.")
    args = parser.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    targets = {args.dataset: DATASETS[args.dataset]} if args.dataset else DATASETS
    for name, info in targets.items():
        dest = RAW / name
        print(f"[{name}] -> {dest}")
        print(f"    URL : {info['url']}")
        print(f"    Note: {info['note']}")
    print("\nManual download for now; automation lands in Phase 1.")


if __name__ == "__main__":
    main()
