# Try-it Web UI (Phase 4)

The client-facing demo: drag in a thermal image → see the fault overlay → download
the PDF report. This is the single most important Upwork artifact ("does it work
on MY data?").

**Plan:** a minimal single-page frontend (or Gradio for v1 speed) calling the
FastAPI `/inspect` endpoint in [`src/solarscan/serve/api.py`](../src/solarscan/serve/api.py).
Deploy alongside the API (Docker) or to Hugging Face Spaces.

For now, exercise the backend directly:

```bash
make serve
curl -F "file=@assets/sample_thermal.png" http://localhost:8000/inspect
```
