# SolarScan — Technical Design & Roadmap

> Aerial PV fault detection: drone thermal/RGB imagery → detect, classify & georeference solar-panel faults → severity-ranked, yield-loss-quantified inspection report, with a real-time edge-inference path on NVIDIA Jetson.

---

## 1. Positioning (why this wins work)

**Audience:** energy-sector clients on Upwork (solar IPPs, O&M providers, asset owners) and the freelance market for AI/CV in renewables & oil-and-gas.

**The moat is domain + AI.** Built by an engineer with a **BSc Petroleum Engineering** and **MSc Renewable Engineering** — so the project speaks the client's language (yield loss, O&M economics, IEC fault taxonomy) *and* ships working CV. Most CV freelancers can't credibly bid an energy gig; most energy engineers can't ship a model. This sits in the overlap.

**What a client must believe after 3 minutes:** "This person built a working solar-inspection system that runs on real edge hardware and produces the report my O&M team would actually act on — they can build mine."

**The three artifacts that close the deal** (in priority order for Upwork):
1. A **2–3 min demo video**: thermal flyover → live fault overlay → georeferenced fault map → generated PDF report.
2. A **deployed try-it demo** (upload a thermal image → get faults + report back).
3. A **clean public repo** + a one-page "the problem we solve" writeup.

---

## 2. What we're building (pipeline)

```
        ┌─────────────────────────────────────────────────────────────┐
        │  INPUT: drone thermal (IR) frames  [+ RGB where available]    │
        └───────────────────────────┬─────────────────────────────────┘
                                     ▼
                    ┌────────────────────────────────┐
        STAGE 1     │  Panel/module detection (boxes) │   YOLO/RT-DETR
                    └────────────────┬───────────────┘
                                     ▼
                    ┌────────────────────────────────┐
        STAGE 2     │  Fault classification per region│   IEC taxonomy
                    │  (hotspot, diode, crack, soil…) │   (timm classifier)
                    └────────────────┬───────────────┘
                                     ▼
                    ┌────────────────────────────────┐
        STAGE 3     │  Stitch + georeference          │   ortho/GeoJSON
                    │  → faults land on farm layout    │
                    └────────────────┬───────────────┘
                                     ▼
                    ┌────────────────────────────────┐
        STAGE 4     │  Yield-loss estimate + severity  │   per-fault scoring
                    │  ranking → PDF inspection report │
                    └────────────────┬───────────────┘
                                     ▼
        ┌─────────────────────────────────────────────────────────────┐
        │  EDGE PATH (parallel): Stage 1+2 → ONNX → TensorRT (INT8)     │
        │  benchmarked on Jetson Orin Nano AND Orin NX                  │
        └─────────────────────────────────────────────────────────────┘
```

The point that separates this from a "panel classifier": it ends at a **georeferenced, yield-quantified report**, and the model **runs on real edge hardware with measured numbers** — not a notebook.

---

## 3. Differentiators (the un-passable parts)

1. **Real Jetson edge deployment, benchmarked across two devices.** PyTorch → ONNX → TensorRT with INT8 quantization, with a published table: **Orin Nano vs Orin NX × FP16 vs INT8 → FPS, latency (p50/p95), mAP/accuracy delta, power.** Almost no portfolio shows real cross-device edge numbers. This is the headline.
2. **End-to-end to a business artifact.** Output is the PDF an O&M manager acts on (fault, GPS, severity, estimated kWh/$ lost), not a label.
3. **IEC-aligned multi-class taxonomy**, not binary faulty/healthy.
4. **Honest, reproducible engineering**: pinned configs, eval harness, model card, Docker, CI. Clearly labels what is public-data vs synthetic.

---

## 4. Data strategy (public-data only; no own drone/farm)

We are honest about scope: **trained & evaluated on public datasets; the geospatial layer demonstrated on a documented synthetic/sample farm.** Reviewers respect this; faking real flights is disqualifying.

| Dataset | What it gives us | Stage | Notes |
|---|---|---|---|
| **InfraredSolarModules** (Raptor Maps, ICLR 2020) | 20,000 IR images, 24×40 px, **12 classes** (11 anomalies + No-Anomaly): Cell, Cell-Multi, Cracking, Hot-Spot, Hot-Spot-Multi, Shadowing, Diode, Diode-Multi, Vegetation, Soiling, Offline-Module | **Stage 2** (classifier) | Canonical PV-IR set. Crops, not full frames — perfect for fault-type classification. [GitHub](https://github.com/RaptorMaps/InfraredSolarModules) / [Kaggle](https://www.kaggle.com/datasets/marcosgabriel/infrared-solar-modules) |
| **Thermal PV Panel Detection & Fault Dataset for UAV Inspection** (Zenodo, 2024) | Raw aerial thermal frames (DJI Mavic 3T, Sindos/Thessaloniki PV farm) with **panel detection + fault** labels | **Stage 1 + 3** | Real aerial frames → enables detection + a credible georef demo. [zenodo.org/records/16420123](https://zenodo.org/records/16420123) |
| **Hotspot/Snail-Trail thermal set** (arXiv 2507.20680) | 277 radiometric IR images (Zenmuse XT / DJI M100), segmentation labels | Stage 1/2 aug | Adds segmentation + a second device's imagery. |
| **Synthetic augmentation** | Generated/augmented faults to balance rare classes (Diode-Multi, Cracking) | Stages 1–2 | Documented as an engineering technique (class imbalance), with before/after metrics — not a research claim. |

**Class imbalance is the real ML challenge here** (No-Anomaly dominates; some faults are rare). The augmentation + sampling strategy and its measured effect on per-class recall is a concrete competence signal.

---

## 5. System architecture & repo layout

```
solarscan/
├── README.md                  # opens with demo GIF + "problem we solve"
├── ROADMAP.md                 # this file
├── pyproject.toml             # pinned deps (uv/poetry)
├── docker/                    # train, serve, jetson images
├── configs/                   # hydra/yaml — datasets, models, training
├── data/                      # download scripts + dataset cards (not raw data)
├── src/solarscan/
│   ├── data/                  # loaders, augmentation, synthetic gen
│   ├── models/                # detection + classification, export
│   ├── pipeline/              # stage orchestration (img → report)
│   ├── geo/                   # stitching, georeferencing, GeoJSON
│   ├── report/                # yield-loss, severity, PDF generation
│   ├── edge/                  # ONNX export, TensorRT build, benchmark
│   └── serve/                 # FastAPI inference API
├── webui/                     # try-it demo (upload → overlay → report)
├── eval/                      # metrics harness, model card
├── notebooks/                 # EDA only (kept out of the critical path)
└── tests/ + .github/workflows # tests + CI
```

**Tech stack**
- **Models:** Ultralytics YOLO (v11) or RT-DETR (Stage 1); `timm` ConvNeXt/EfficientNet (Stage 2). PyTorch.
- **Edge:** ONNX → TensorRT, INT8 calibration; `trtexec` + custom Python harness on Jetson.
- **Geo:** OpenCV stitching, `rasterio`/`pyproj`, Leaflet/folium map, GeoJSON.
- **Report:** WeasyPrint or ReportLab → PDF.
- **Serve/demo:** FastAPI + lightweight web UI (or Gradio for v1 speed). Dockerized; deploy to a small GPU/CPU host or HF Spaces.
- **Quality:** pytest, ruff, GitHub Actions CI, model card, pinned configs.

---

## 6. Edge deployment plan (the headline)

1. Train Stage 1+2 in PyTorch.
2. Export → ONNX (fixed + dynamic shapes).
3. Build TensorRT engines: FP16 and INT8 (with a calibration set).
4. Benchmark harness across **every backend and device**, apples-to-apples.

   **Dev baseline (RTX A4000, measured — batch 32, classifier `convnext_tiny`):**

   | Backend | Device | Precision | Mean ms | p95 ms | Throughput (img/s) |
   |---|---|---|---|---|---|
   | pytorch | RTX A4000 | FP16 | 1.54 | 1.76 | **20,730** |
   | pytorch | RTX A4000 | FP32 | 2.41 | 2.71 | 13,269 |
   | onnxruntime | CPU | FP32 | 13.91 | 14.09 | 2,301 |
   | pytorch | CPU | FP32 | 27.05 | 28.12 | 1,183 |

   FP16 already gives a ~1.6× throughput lift over FP32 on the A4000, and ONNX parity is
   verified (max abs diff 3.3e-6). The Jetson rows below use the **same harness** on the same ONNX.

   > ⚠️ **PROJECTED — NOT YET MEASURED.** The table below is an *estimate* for planning only,
   > pending on-device benchmarking on the actual Orin Nano + NX. These figures must not be
   > quoted as measured results. They will be replaced with real `solarscan benchmark` output
   > (and the ⚠️ removed) once the engines are built on the devices.
   >
   > **Method:** projected from the measured A4000 batch-1 latency scaled by published Orin
   > dense-INT8/FP16 throughput (Orin NX 16GB ≈ 50 dense TOPS; Orin Nano 8GB ≈ 20 dense TOPS)
   > and typical TensorRT INT8 behaviour. At 64×64 the model is launch/memory-bound, not
   > compute-bound, so real gaps are usually *smaller* than raw TOPS ratios — treat these as
   > conservative order-of-magnitude estimates, single-stream (batch 1).

   | Backend | Device | Precision | Latency ms (b1) | Throughput (img/s) | macro-F1 Δ | Power (W) |
   |---|---|---|---|---|---|---|
   | TensorRT | Orin NX 16GB | FP16 | ~1.5 *(proj.)* | ~650 *(proj.)* | ≈0 | ~15–25 |
   | TensorRT | Orin NX 16GB | INT8 | ~1.0 *(proj.)* | ~1,000 *(proj.)* | −0.5 to −1.5% *(proj.)* | ~15–25 |
   | TensorRT | Orin Nano 8GB | FP16 | ~3.0 *(proj.)* | ~330 *(proj.)* | ≈0 | ~7–15 |
   | TensorRT | Orin Nano 8GB | INT8 | ~2.0 *(proj.)* | ~500 *(proj.)* | −0.5 to −1.5% *(proj.)* | ~7–15 |

   _(Note: A4000 rows above are batch-32 throughput; Jetson projections are batch-1 single-stream —
   the relevant metric for in-flight real-time inference — so the two blocks are not directly comparable.)_
   On-device steps to produce the real numbers: [docs/JETSON.md](docs/JETSON.md).

5. Show the **accuracy/throughput tradeoff** narrative (INT8 speedup vs accuracy drop) — this is the senior-engineer signal. Even the projected INT8 latencies (~1–2 ms) put the model **far inside real-time** for a drone frame budget, which is the point the table needs to make.

---

## 7. Roadmap / milestones (~3–6 months, part-time)

**Phase 0 — Thin vertical slice (week 1–2).** `thermal crop → classifier → JSON faults → minimal PDF`. Crude but runs end-to-end. De-risks integration; gives an early demo.

**Phase 1 — Detection + classification core (week 3–6).** Train Stage 1 on the Zenodo UAV set, Stage 2 on InfraredSolarModules. Eval harness, per-class metrics, confusion matrix, model card. Tackle class imbalance + synthetic augmentation here.
> **Status:** Stage 2 classifier ✅ done — `convnext_tiny`, **82.7% acc / 0.704 macro-F1** on held-out test, class-balanced sampling lifts rare faults (Diode-Multi 0.96 recall). See [model card](eval/MODEL_CARD.md). _Next: Stage 1 detector (Ultralytics on the Zenodo UAV set) + synthetic augmentation for the weak classes (Soiling, Cell-Multi, Hot-Spot-Multi)._

**Phase 2 — Edge path (week 6–9).** ONNX → TensorRT → INT8 → cross-device benchmark table. **This is the differentiator — give it real time.**

**Phase 3 — Geo + report (week 9–12).** Stitching/georeferencing on the aerial set + a synthetic farm layout; yield-loss model; severity ranking; polished PDF.
> **Status:** ✅ done on a synthetic farm. Faults → module ID + GPS ([geo/](src/solarscan/geo/)), GeoJSON export, severity-coloured fault map (matplotlib PNG + interactive folium HTML), and the PDF now embeds the map + per-fault module/GPS + total estimated yield loss. Run: `solarscan demo -i <img> --farm sample`. _Field path is a drop-in: supply measured GCP corners instead of the synthetic bounds. True orthomosaic stitching awaits a real overlapping flight (Stage-1/Zenodo)._

**Phase 4 — Demo & ship (week 12–16).** Deployed try-it demo, 2–3 min video, README polish, "problem we solve" page. Package as an Upwork portfolio piece + reusable proposal blurb.

---

## 8. Definition of done (per the portfolio bar)

- [ ] Repo runs from a clean clone via `docker compose up` / documented `make demo`.
- [ ] Eval harness reproduces reported metrics; model card published.
- [ ] Cross-device Jetson benchmark table filled with real numbers.
- [ ] End-to-end: upload thermal image(s) → georeferenced fault map + downloadable PDF report.
- [ ] 2–3 min demo video recorded.
- [ ] Deployed try-it link live.
- [ ] README opens with a demo GIF and the energy-domain "problem we solve" framing.

---

## 9. Honest scope notes

- No own drone/farm → real aerial frames come from public UAV datasets; the farm layout for georeferencing is a documented synthetic/sample, clearly labeled.
- "Yield-loss estimate" is a transparent model (fault class → expected performance impact from literature/IEC guidance), presented as an estimate with assumptions stated — not a fabricated precise number.
- Regulatory/standard references kept current and cited, not asserted from memory.
```
