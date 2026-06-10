"""SolarScan inference service + web demo.

Drag a thermal image into the browser and the service auto-routes:

* a **wide aerial frame** (a detector is configured + image is large) -> Stage-1
  module detection, georeferenced, **detect-only** (no fault claims, because the
  classifier is not calibrated for the source camera — see model card);
* a **single module crop** -> fault-type classification + severity + yield-loss.

Either way the client gets an annotated overlay, a map, a report and a PDF.

Model selection (env, both optional):
  SOLARSCAN_CHECKPOINT=/path/to/classifier_best.pt   (else heuristic stub)
  SOLARSCAN_DETECTOR=/path/to/detector_best.pt        (enables aerial mode)
Import is guarded so the package imports without the ``serve`` extra.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

try:
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.responses import FileResponse, HTMLResponse
    from fastapi.staticfiles import StaticFiles
except ModuleNotFoundError as exc:  # pragma: no cover - import-time guard
    raise ModuleNotFoundError(
        "FastAPI is not installed. Install the serve extra: `uv sync --extra serve`."
    ) from exc

from PIL import Image

from solarscan import __version__
from solarscan.geo import (
    FarmLayout,
    render_detection_map_png,
    render_fault_map_png,
    write_detections_geojson,
    write_geojson,
)
from solarscan.pipeline import run_pipeline
from solarscan.report.pdf import write_pdf_report
from solarscan.serve.render import draw_overlay, to_base64_png

STATIC_DIR = Path(__file__).parent / "static"
SESSION_ROOT = Path(tempfile.gettempdir()) / "solarscan_sessions"
SESSION_ROOT.mkdir(parents=True, exist_ok=True)

# Images at least this size (min dimension) are treated as wide aerial frames.
AERIAL_MIN_SIZE = 200

app = FastAPI(title="SolarScan", version=__version__)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_FARM_CROP = FarmLayout.sample()
_FARM_AERIAL = FarmLayout.sample_aerial()


def _load_classifier():
    ckpt = os.environ.get("SOLARSCAN_CHECKPOINT")
    if not ckpt or not Path(ckpt).exists():
        return None, "stub-0.1.0"
    try:
        from solarscan.models.timm_classifier import TimmClassifier

        model = TimmClassifier.load(ckpt)
        return model, f"{model.backbone_name}:{Path(ckpt).name}"
    except Exception as exc:  # pragma: no cover - fall back to stub on any failure
        print(f"[solarscan] could not load classifier ({exc}); using stub.")
        return None, "stub-0.1.0"


def _load_detector():
    ckpt = os.environ.get("SOLARSCAN_DETECTOR")
    if not ckpt or not Path(ckpt).exists():
        return None, None
    try:
        from solarscan.models.yolo_detector import YoloDetector

        return YoloDetector(ckpt), f"yolo:{Path(ckpt).name}"
    except Exception as exc:  # pragma: no cover
        print(f"[solarscan] could not load detector ({exc}); aerial mode disabled.")
        return None, None


_CLASSIFIER, _CLS_VERSION = _load_classifier()
_DETECTOR, _DET_VERSION = _load_detector()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC_DIR / "index.html").read_text()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "version": __version__,
        "classifier": _CLS_VERSION,
        "detector": _DET_VERSION or "none",
    }


@app.post("/inspect")
async def inspect(file: UploadFile = File(...)) -> dict:
    """Run the pipeline on an uploaded thermal image; return everything the UI needs."""
    sid = uuid.uuid4().hex
    session = SESSION_ROOT / sid
    session.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename or "upload.png").suffix or ".png"
    img_path = session / f"input{suffix}"
    img_path.write_bytes(await file.read())
    image = Image.open(img_path)

    aerial = _DETECTOR is not None and min(image.size) >= AERIAL_MIN_SIZE
    if aerial:
        farm = _FARM_AERIAL
        report = run_pipeline(
            img_path, detector=_DETECTOR, model_version=_DET_VERSION, farm=farm, detect_only=True
        )
    else:
        farm = _FARM_CROP
        report = run_pipeline(
            img_path, classifier=_CLASSIFIER, model_version=_CLS_VERSION, farm=farm
        )

    overlay = to_base64_png(draw_overlay(image, report.faults, report.detections))

    map_b64 = None
    map_path = session / "map.png"
    if aerial:
        if render_detection_map_png(report.detections, farm, map_path):
            map_b64 = to_base64_png(Image.open(map_path))
        write_detections_geojson(report.detections, farm, session / "result.geojson")
    else:
        if render_fault_map_png(report.faults, farm, map_path):
            map_b64 = to_base64_png(Image.open(map_path))
        write_geojson(report.faults, farm, session / "result.geojson")

    write_pdf_report(report, session / "report.pdf", map_image=map_path if map_b64 else None)

    return {
        "session": sid,
        "mode": "detect-only" if aerial else "classification",
        "model": _DET_VERSION if aerial else _CLS_VERSION,
        "report": report.model_dump(mode="json"),
        "overlay_png": overlay,
        "map_png": map_b64,
        "pdf_url": f"/download/{sid}/report.pdf",
        "geojson_url": f"/download/{sid}/result.geojson",
    }


@app.get("/download/{sid}/{name}")
def download(sid: str, name: str) -> FileResponse:
    # Guard against path traversal: only plain names inside a hex session dir.
    if not sid.isalnum() or "/" in name or ".." in name:
        raise HTTPException(status_code=400, detail="bad request")
    path = SESSION_ROOT / sid / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(path)
