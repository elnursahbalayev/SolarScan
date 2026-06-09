# Datasets

Raw data is **never committed** (see `.gitignore`). Use `python data/download.py`
to fetch into `data/raw/`. All training/eval uses public datasets; the
georeferencing layer is demonstrated on a documented synthetic/sample farm.

| Dataset | Use | License | Link |
|---|---|---|---|
| **InfraredSolarModules** (Raptor Maps, ICLR 2020) — 20,000 IR images, 24×40 px, 12 classes (11 anomalies + No-Anomaly) | Stage 2 fault **classification** | See repo | https://github.com/RaptorMaps/InfraredSolarModules · [Kaggle mirror](https://www.kaggle.com/datasets/marcosgabriel/infrared-solar-modules) |
| **Thermal PV Panel Detection & Fault Dataset for UAV Inspection** (Zenodo, 2024) — aerial thermal frames, DJI Mavic 3T, Sindos/Thessaloniki | Stage 1 **detection** + Stage 3 georef demo | See record | https://zenodo.org/records/16420123 |
| **Hotspot / Snail-Trail thermal set** (arXiv 2507.20680) — 277 radiometric IR images, segmentation | Augmentation / 2nd device | See paper | https://arxiv.org/pdf/2507.20680 |

## Class taxonomy (InfraredSolarModules)

`No-Anomaly, Cell, Cell-Multi, Cracking, Hot-Spot, Hot-Spot-Multi, Shadowing,
Diode, Diode-Multi, Vegetation, Soiling, Offline-Module`

Mirrored in [`src/solarscan/taxonomy.py`](../src/solarscan/taxonomy.py).

## Known challenges (address in Phase 1)

- **Class imbalance** — No-Anomaly dominates; Diode-Multi / Cracking are rare.
  Synthetic augmentation + class-balanced sampling; report per-class recall.
- **Tiny inputs** — 24×40 px crops; choose backbone/input size accordingly.
- **Domain gap** — crops vs full aerial frames; the detector and classifier are
  trained on different sources and joined at inference.
