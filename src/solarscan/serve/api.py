"""FastAPI app: upload a thermal image -> get the inspection report back.

This is the backbone of the "runs on YOUR data" Upwork demo. Import is guarded
so the package still imports without the ``serve`` extra installed.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

try:
    from fastapi import FastAPI, File, UploadFile
except ModuleNotFoundError as exc:  # pragma: no cover - import-time guard
    raise ModuleNotFoundError(
        "FastAPI is not installed. Install the serve extra: `uv sync --extra serve`."
    ) from exc

from solarscan import __version__
from solarscan.pipeline import run_pipeline

app = FastAPI(title="SolarScan", version=__version__)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.post("/inspect")
async def inspect(file: UploadFile = File(...)) -> dict:
    """Run the pipeline on an uploaded thermal image and return the report JSON."""
    suffix = Path(file.filename or "upload.png").suffix or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(await file.read())
        tmp.flush()
        report = run_pipeline(tmp.name)
    return report.model_dump(mode="json")
