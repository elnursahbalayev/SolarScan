.PHONY: install install-ml install-all demo test lint format typecheck serve clean

# --- setup ---
install:           ## Core deps only (Phase-0 slice runs on this, no GPU needed)
	uv sync

install-ml:        ## Core + ML training/inference stack
	uv sync --extra ml

install-all:       ## Everything (ml + serve + geo + dev)
	uv sync --extra ml --extra serve --extra geo --extra dev

# --- run ---
demo:              ## Run the end-to-end Phase-0 slice on a sample thermal image
	uv run solarscan demo --input assets/sample_thermal.png --out outputs/

serve:             ## Launch the web demo + API (stub model)
	uv run uvicorn solarscan.serve.api:app --reload --port 8000

serve-model:       ## Launch the web demo + API with the trained classifier + detector
	SOLARSCAN_CHECKPOINT=runs/convnext_tiny/best.pt \
	SOLARSCAN_DETECTOR=runs/detector/weights/best.pt \
		uv run uvicorn solarscan.serve.api:app --port 8000

# --- quality ---
test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy src

clean:
	rm -rf outputs/ runs/ .pytest_cache .ruff_cache .mypy_cache
