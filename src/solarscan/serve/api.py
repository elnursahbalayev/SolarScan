"""SolarScan inference service + web demo.

Drag a thermal image into the browser -> the trained model classifies faults,
the result is georeferenced against a (synthetic) farm, and the client gets back
an annotated thermal overlay, a fault map, a severity-ranked table and a
downloadable PDF. The backbone of the "runs on YOUR data" Upwork demo.

Model selection: set SOLARSCAN_CHECKPOINT=/path/to/best.pt to serve the trained
model; otherwise the heuristic stub is used (so the service runs with no GPU and
no torch). Import is guarded so the package imports without the ``serve`` extra.
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
from solarscan.geo import FarmLayout, render_fault_map_png, write_geojson
from solarscan.pipeline import run_pipeline
from solarscan.report.pdf import write_pdf_report
from solarscan.serve.render import draw_overlay, to_base64_png

STATIC_DIR = Path(__file__).parent / "static"
SESSION_ROOT = Path(tempfile.gettempdir()) / "solarscan_sessions"
SESSION_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="SolarScan", version=__version__)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_FARM = FarmLayout.sample()


def _load_classifier():
    """Load the trained model if SOLARSCAN_CHECKPOINT is set and importable."""
    ckpt = os.environ.get("SOLARSCAN_CHECKPOINT")
    if not ckpt or not Path(ckpt).exists():
        return None, "stub-0.1.0"
    try:
        from solarscan.models.timm_classifier import TimmClassifier

        model = TimmClassifier.load(ckpt)
        return model, f"{model.backbone_name}:{Path(ckpt).name}"
    except Exception as exc:  # pragma: no cover - fall back to stub on any failure
        print(f"[solarscan] could not load checkpoint ({exc}); using stub.")
        return None, "stub-0.1.0"


_CLASSIFIER, _MODEL_VERSION = _load_classifier()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC_DIR / "index.html").read_text()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__, "model": _MODEL_VERSION}


@app.post("/inspect")
async def inspect(file: UploadFile = File(...), georeference: bool = True) -> dict:
    """Run the pipeline on an uploaded thermal image; return everything the UI needs."""
    sid = uuid.uuid4().hex
    session = SESSION_ROOT / sid
    session.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename or "upload.png").suffix or ".png"
    img_path = session / f"input{suffix}"
    img_path.write_bytes(await file.read())

    farm = _FARM if georeference else None
    report = run_pipeline(
        img_path, classifier=_CLASSIFIER, model_version=_MODEL_VERSION, farm=farm
    )

    image = Image.open(img_path)
    overlay = to_base64_png(draw_overlay(image, report.faults))

    map_b64 = None
    pdf_name = "report.pdf"
    if farm is not None:
        map_png = render_fault_map_png(report.faults, farm, session / "map.png")
        if map_png:
            map_b64 = to_base64_png(Image.open(map_png))
        write_geojson(report.faults, farm, session / "faults.geojson")
    write_pdf_report(report, session / pdf_name, map_image=session / "map.png" if map_b64 else None)

    return {
        "session": sid,
        "model": _MODEL_VERSION,
        "report": report.model_dump(mode="json"),
        "overlay_png": overlay,
        "map_png": map_b64,
        "pdf_url": f"/download/{sid}/{pdf_name}",
        "geojson_url": f"/download/{sid}/faults.geojson" if farm else None,
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
