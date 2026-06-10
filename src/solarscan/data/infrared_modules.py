"""InfraredSolarModules dataset (Raptor Maps, ICLR 2020).

20,000 grayscale IR crops, 24x40 px, 12 classes. Heavily imbalanced
(No-Anomaly ~50%, Diode-Multi ~0.9%) — handled downstream via class-balanced
sampling + augmentation.

The dataset root may be nested (the published archive extracts to
``InfraredSolarModules/{module_metadata.json,images/}``), so we locate the
metadata file by search and resolve image paths relative to it.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from solarscan.taxonomy import ALL_CLASSES, FaultClass

CLASS_TO_IDX: dict[FaultClass, int] = {c: i for i, c in enumerate(ALL_CLASSES)}
IDX_TO_CLASS: dict[int, FaultClass] = {i: c for c, i in CLASS_TO_IDX.items()}


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int  # index into ALL_CLASSES


def find_metadata(root: str | Path) -> Path:
    root = Path(root)
    direct = root / "module_metadata.json"
    if direct.exists():
        return direct
    matches = [p for p in root.rglob("module_metadata.json") if "__MACOSX" not in str(p)]
    if not matches:
        raise FileNotFoundError(
            f"module_metadata.json not found under {root}. Run `python data/download.py`."
        )
    return matches[0]


def load_samples(root: str | Path) -> list[Sample]:
    meta_path = find_metadata(root)
    base = meta_path.parent
    meta = json.loads(meta_path.read_text())
    samples: list[Sample] = []
    for entry in meta.values():
        cls = FaultClass(entry["anomaly_class"])
        samples.append(Sample(path=base / entry["image_filepath"], label=CLASS_TO_IDX[cls]))
    return samples


def stratified_split(
    samples: list[Sample],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    """Per-class split so rare classes appear in every split."""
    by_class: dict[int, list[Sample]] = defaultdict(list)
    for s in samples:
        by_class[s.label].append(s)

    rng = random.Random(seed)
    train, val, test = [], [], []
    for items in by_class.values():
        items = items[:]
        rng.shuffle(items)
        n = len(items)
        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)
        test.extend(items[:n_test])
        val.extend(items[n_test : n_test + n_val])
        train.extend(items[n_test + n_val :])
    rng.shuffle(train)
    return train, val, test


def class_counts(samples: list[Sample]) -> dict[int, int]:
    counts: dict[int, int] = defaultdict(int)
    for s in samples:
        counts[s.label] += 1
    return dict(counts)


class InfraredSolarModules(Dataset):
    def __init__(self, samples: list[Sample], transform=None) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s.path).convert("RGB")  # replicate gray->3ch for pretrained backbones
        if self.transform is not None:
            img = self.transform(img)
        return img, s.label
