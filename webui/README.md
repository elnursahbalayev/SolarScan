# Try-it Web UI ✅

The client-facing demo: drop in a thermal image → annotated thermal overlay +
georeferenced fault map + severity-ranked table + downloadable PDF report. This
is the single most important Upwork artifact ("does it work on MY data?").

It's a single static page ([../src/solarscan/serve/static/index.html](../src/solarscan/serve/static/index.html),
no build step) served by the FastAPI app in
[../src/solarscan/serve/api.py](../src/solarscan/serve/api.py), which calls the
same pipeline as the CLI.

## Run it

```bash
uv sync --extra serve --extra geo
make serve          # stub model      → http://localhost:8000
make serve-model    # trained model   (needs runs/convnext_tiny/best.pt + --extra ml)
```

Or via Docker:

```bash
docker compose -f docker/docker-compose.yml up --build
```

## Endpoints
- `GET /` — the web UI
- `GET /health` — status + active model
- `POST /inspect` — multipart thermal image → report JSON + overlay + map + PDF/GeoJSON URLs
- `GET /download/{session}/{file}` — fetch the generated PDF / GeoJSON

## Deploy
The stub image is CPU-only and small — suitable for Hugging Face Spaces or any
container host. For the trained-model demo, bake the checkpoint into the image (or
mount it) and set `SOLARSCAN_CHECKPOINT`.
