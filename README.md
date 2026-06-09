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
```

Or hit the API:

```bash
uv sync --extra serve
make serve              # http://localhost:8000  (POST /inspect with a thermal image)
```

## Project status

This repo is built in public, in phases (see [ROADMAP.md](ROADMAP.md)):

| Phase | What | Status |
|---|---|---|
| 0 | End-to-end slice (stub model → report) | ✅ runnable now |
| 1 | Detection + classification on public datasets | ⬜ next |
| 2 | TensorRT edge path + Jetson benchmark | ⬜ the headline |
| 3 | Stitching, georeferencing, yield-loss report | ⬜ |
| 4 | Deployed try-it demo + video | ⬜ |

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
