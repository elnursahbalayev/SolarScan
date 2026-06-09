# Model Card — SolarScan

> Template to be completed in Phase 1 once real models are trained. Empty fields
> are intentional placeholders, not omissions.

## Overview
- **Task:** PV fault detection (Stage 1) + fault classification (Stage 2), IEC-aligned taxonomy.
- **Intended use:** Prioritising solar-farm O&M from aerial thermal imagery.
- **Out of scope:** Safety-critical decisions without human review; metered yield accounting.
- **Current status:** Phase 0 ships a heuristic **stub** model only — not validated.

## Data
- Training: InfraredSolarModules (classification), Zenodo UAV thermal set (detection).
- Splits / leakage controls: _TBD Phase 1._
- Class imbalance handling: synthetic augmentation + class-balanced sampling.

## Metrics (Phase 1 — to fill)
| Metric | Value |
|---|---|
| Classifier macro-F1 | _TBD_ |
| Per-class recall (rare: Diode-Multi, Cracking) | _TBD_ |
| Detector mAP@0.5 | _TBD_ |
| Confusion matrix | _link_ |

## Edge performance (Phase 2 — to fill)
See ROADMAP.md §6 — Orin Nano vs Orin NX × FP16/INT8 (FPS, latency, accuracy delta, power).

## Limitations & ethics
- Public-data only; no proprietary/site data. Geo layer demonstrated on a synthetic farm.
- Yield-loss is an estimate with stated assumptions, not a measurement.
- Domain gap between training crops and field imagery may degrade real-world accuracy.
