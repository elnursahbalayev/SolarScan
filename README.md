# SolarScan — Aerial PV Fault Detection

**Inspect a whole solar farm in one drone flight and recover lost yield.**
SolarScan turns drone thermal imagery into a severity-ranked, GPS-located,
yield-loss-quantified inspection report — and runs in real time on NVIDIA Jetson
edge hardware.

<!-- TODO (Phase 4): replace with demo GIF -->
> 🎥 _Demo video coming in Phase 4 — see [ROADMAP.md](ROADMAP.md)._

---

Built by an engineer with a **BSc Petroleum Engineering** and **MSc Renewable
Engineering** — this project speaks the O&M language (yield loss, IEC fault
classes, inspection economics), not just mAP.

## Why it matters

Manual PV inspection (IV-curve tests, thermal-gun spot checks) is slow, partial,
and misses underperformers. SolarScan inspects an entire farm in one flight,
classifies every fault against the **IEC-aligned taxonomy** (hot-spot, bypass
diode, cracking, soiling, shadowing, offline module…), maps it to GPS, and
estimates the energy being lost — so O&M teams fix what actually costs money.

## Pipeline

```
thermal/RGB frames → detect modules → classify faults (12-class IEC taxonomy)
   → georeference → estimate yield loss + rank severity → PDF report
   (parallel: TensorRT INT8 edge path benchmarked on Jetson Orin Nano + NX)
```

## Quickstart

```bash
uv sync                 # core deps only — no GPU needed for the Phase-0 slice
make demo               # thermal image → JSON + PDF inspection report
make test               # run the test suite

# single-module classification (faults → module ID + GPS, GeoJSON, fault map):
uv run solarscan demo -i assets/sample_thermal.png --farm sample \
    -c runs/convnext_tiny/best.pt    # drop -c to use the stub model

# wide aerial frame → detect + georeference every module (detect-only, honest):
uv run solarscan demo -i assets/sample_aerial.jpg \
    -d runs/detector/weights/best.pt --detect-only
```

Or launch the **web try-it demo** — it auto-routes: a single module crop is classified
(fault type + severity + yield loss); a wide aerial frame is detected & georeferenced
(detect-only, no unvalidated fault claims):

```bash
uv sync --extra serve --extra geo
make serve              # stub model  → open http://localhost:8000
make serve-model        # trained classifier + detector (needs the ml extra)
```

The same service exposes `POST /inspect` (multipart image) for programmatic use.

## Project status

This repo is built in public, in phases (see [ROADMAP.md](ROADMAP.md)):

| Phase | What | Status |
|---|---|---|
| 0 | End-to-end slice (stub model → report) | ✅ runnable |
| 1 | Classifier **82.7% acc / 0.704 macro-F1** + module detector **mAP@50 0.995** ([model card](eval/MODEL_CARD.md)) | ✅ both trained |
| 2 | Edge path: ONNX export (parity-checked) + benchmark harness — **A4000 FP16 ~20.7k img/s** | ✅ dev done; TensorRT/Jetson rows pending |
| 3 | Georeferencing + report: faults → module ID + GPS, GeoJSON, fault map, PDF | ✅ on synthetic farm |
| 4 | Web try-it demo (upload → overlay + map + PDF) | ✅ built; deploy + video pending |

> Train it yourself: `python data/download.py infrared-solar-modules && uv run solarscan train`

> **Honest scope:** trained/evaluated on public datasets
> ([details](data/README.md)); the geospatial layer is demonstrated on a
> documented synthetic farm. Yield-loss figures are stated estimates, not metered
> values.

## Tech

PyTorch · Ultralytics/RT-DETR · timm · ONNX → TensorRT (INT8) · FastAPI ·
OpenCV/rasterio · ReportLab. Packaged with `uv`, tested in CI.

## Layout

`src/solarscan/` — `models/ pipeline/ geo/ report/ edge/ serve/ data/` ·
`configs/` · `data/` (cards + download) · `eval/` (metrics + model card) ·
`docker/` · `tests/`. See [ROADMAP.md](ROADMAP.md) §5.

## License

[MIT](LICENSE).
