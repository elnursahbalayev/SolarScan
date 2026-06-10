"""API tests via FastAPI TestClient. Skipped without the serve extra / httpx.

Runs against the stub model (no SOLARSCAN_CHECKPOINT), so no torch/GPU needed.
"""

from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="requires the `serve` extra")
pytest.importorskip("httpx", reason="requires httpx (dev extra)")

from fastapi.testclient import TestClient  # noqa: E402

from solarscan.serve.api import app  # noqa: E402

SAMPLE = Path(__file__).resolve().parents[1] / "assets" / "sample_thermal.png"

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_index_serves_html():
    r = client.get("/")
    assert r.status_code == 200
    assert "SolarScan" in r.text


def test_inspect_returns_overlay_and_report():
    with SAMPLE.open("rb") as fh:
        r = client.post("/inspect", files={"file": ("sample_thermal.png", fh, "image/png")})
    assert r.status_code == 200
    data = r.json()
    assert data["overlay_png"].startswith("data:image/png;base64,")
    assert "summary" in data["report"]
    assert data["pdf_url"].startswith("/download/")
    # PDF should then be downloadable.
    pdf = client.get(data["pdf_url"])
    assert pdf.status_code == 200
    assert pdf.headers["content-type"] == "application/pdf"


def test_download_rejects_traversal():
    r = client.get("/download/notalnum!/x")
    assert r.status_code == 400
