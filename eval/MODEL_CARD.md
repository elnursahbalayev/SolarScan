# Model Card — SolarScan Fault Classifier

## Overview
- **Task:** IR PV-module fault classification (Stage 2), 12-class IEC-aligned taxonomy.
- **Model:** `convnext_tiny` (timm, ImageNet-pretrained), 3-ch input (grayscale replicated), 64×64.
- **Intended use:** Prioritising solar-farm O&M from aerial thermal imagery.
- **Out of scope:** Safety-critical decisions without human review; metered yield accounting.
- **Status:** Phase 1 trained model (classifier). Detector (Stage 1) and edge export (Phase 2) pending.

## Data
- **InfraredSolarModules** (Raptor Maps, ICLR 2020): 20,000 IR crops, 24×40 px, 12 classes.
- **Split:** stratified 70/15/15 train/val/test (seed 42); every class present in every split.
- **Imbalance:** ~57:1 (No-Anomaly 10,000 vs Diode-Multi 175). Handled with a
  `WeightedRandomSampler` (balanced batches) + label smoothing (0.1) + mild augmentation
  (flips, ±10° rotation, brightness/contrast jitter).
- **Selection metric:** validation **macro-F1** (not accuracy) — accuracy is misleading at this imbalance.

## Training
- AdamW (lr 3e-4, wd 0.05), cosine schedule, 50 epochs, batch 128, AMP, on an NVIDIA RTX A4000.

## Results (held-out test split)

| Metric | Value |
|---|---|
| Accuracy | **0.827** |
| Macro-F1 | **0.704** |
| Weighted-F1 | **0.824** |

### Per-class (recall / F1 / support)

| Class | Recall | F1 | Support |
|---|---|---|---|
| No-Anomaly | 0.950 | 0.941 | 1500 |
| Diode | 0.969 | 0.969 | 224 |
| Diode-Multi | 0.962 | 0.962 | 26 |
| Vegetation | 0.739 | 0.707 | 245 |
| Hot-Spot | 0.730 | 0.730 | 37 |
| Offline-Module | 0.702 | 0.704 | 124 |
| Cell | 0.694 | 0.664 | 281 |
| Shadowing | 0.671 | 0.724 | 158 |
| Cracking | 0.638 | 0.667 | 141 |
| Cell-Multi | 0.508 | 0.536 | 193 |
| Hot-Spot-Multi | 0.500 | 0.537 | 36 |
| Soiling | 0.233 | 0.304 | 30 |

Confusion matrix: `runs/convnext_tiny/confusion_matrix.png`.

### Reading the results
- Balanced sampling pays off on rare-but-separable faults: **Diode-Multi reaches 0.96 recall on 26 test samples** despite being <1% of the data.
- The weak classes (Soiling, Hot-Spot-Multi, Cell-Multi) are the visually ambiguous ones with the
  smallest support — the realistic next target for synthetic augmentation and a detector upgrade.

## Stage-1 module detector (YOLO)

Separate model that locates PV modules in **wide aerial thermal frames** so the pipeline
works on real drone footage, not just pre-cropped modules.

- **Model:** YOLO11n (Ultralytics), fine-tuned from COCO. Input 640×512.
- **Data:** Zenodo UAV thermal dataset (record 16420123) — DJI Mavic 3T, single class `module`,
  235/83/35 train/val/test, ~60 modules/frame.
- **Test metrics:** **mAP@50 = 0.995, mAP@50-95 = 0.882, precision = 0.988, recall = 0.988.**

### ⚠️ Domain gap — detector and classifier are trained on different sources
The detector (Mavic 3T RGB-colourised thermal) and the classifier (InfraredSolarModules,
grayscale IR crops) come from **different cameras/datasets**. The Zenodo set has **no
fault-type labels**, so fault classification on those aerial frames **cannot be validated**
and is therefore **not performed** — the pipeline runs in **detect-only** mode on aerial input
(modules detected + georeferenced; no fault claims). Fault classification is reported only on
the labelled InfraredSolarModules data above. Closing the gap properly requires fault-labelled
imagery from the target camera (fine-tune/recalibrate, then validate).

## Edge performance

ONNX export verified against PyTorch (max abs diff 3.3e-6). Dev-baseline latency/throughput
(RTX A4000, batch 32, measured via `solarscan benchmark`):

| Backend | Device | Precision | Mean ms | p95 ms | Throughput (img/s) |
|---|---|---|---|---|---|
| pytorch | RTX A4000 | FP16 | 1.54 | 1.76 | 20,730 |
| pytorch | RTX A4000 | FP32 | 2.41 | 2.71 | 13,269 |
| onnxruntime | CPU | FP32 | 13.91 | 14.09 | 2,301 |
| pytorch | CPU | FP32 | 27.05 | 28.12 | 1,183 |

_Pending:_ TensorRT FP16/INT8 on Jetson Orin Nano + Orin NX, via the same harness (ROADMAP §6).

## Limitations & ethics
- Public-data only; no proprietary/site data. Geo layer demonstrated on a synthetic farm.
- Yield-loss is a stated estimate, not a measurement.
- Classifier trained on 24×40 crops; real-world performance depends on the upstream detector
  (Phase 1, Ultralytics on the Zenodo UAV set) supplying comparable module crops.

## Reproduce
```bash
python data/download.py infrared-solar-modules
uv run solarscan train --config configs/default.yaml --out runs/convnext_tiny
uv run solarscan evaluate -c runs/convnext_tiny/best.pt
```
