"""SolarScan command-line interface.

Phase 0 implements ``demo`` end-to-end. ``train`` / ``eval`` / ``export`` /
``benchmark`` are scaffolded with clear next-step guidance and require the
optional extras (``ml``, ``edge``).
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from solarscan.pipeline import run_pipeline
from solarscan.report.pdf import write_pdf_report

app = typer.Typer(add_completion=False, help="SolarScan — aerial PV fault detection.")
console = Console()


@app.command()
def demo(
    input: Path = typer.Option(..., "--input", "-i", help="Path to a thermal image."),
    out: Path = typer.Option(Path("outputs"), "--out", "-o", help="Output directory."),
) -> None:
    """Run the end-to-end pipeline on one image: -> JSON + PDF report."""
    if not input.exists():
        console.print(f"[red]Input not found:[/red] {input}")
        raise typer.Exit(1)

    out.mkdir(parents=True, exist_ok=True)
    report = run_pipeline(input)

    json_path = out / f"{input.stem}_report.json"
    json_path.write_text(report.model_dump_json(indent=2))

    pdf_path = write_pdf_report(report, out / f"{input.stem}_report.pdf")

    console.print(f"[green]Done.[/green] {report.summary.n_faults} fault(s) found.")
    console.print(f"  JSON: {json_path}")
    console.print(f"  PDF:  {pdf_path}")


@app.command()
def train() -> None:
    """[Phase 1] Train detection/classification models. Requires `--extra ml`."""
    console.print(
        "[yellow]Not implemented yet (Phase 1).[/yellow]\n"
        "Plan: train timm classifier on InfraredSolarModules + Ultralytics detector "
        "on the Zenodo UAV set. See ROADMAP.md §7."
    )
    raise typer.Exit(0)


@app.command()
def evaluate() -> None:
    """[Phase 1] Run the eval harness (per-class metrics, confusion matrix)."""
    console.print("[yellow]Not implemented yet (Phase 1).[/yellow] See eval/ and ROADMAP.md §7.")
    raise typer.Exit(0)


@app.command()
def export() -> None:
    """[Phase 2] Export trained model -> ONNX -> TensorRT engine."""
    console.print(
        "[yellow]Not implemented yet (Phase 2).[/yellow]\n"
        "Plan: PyTorch -> ONNX -> TensorRT (FP16/INT8). See solarscan/edge/ and ROADMAP.md §6."
    )
    raise typer.Exit(0)


@app.command()
def benchmark() -> None:
    """[Phase 2] Benchmark the TensorRT engine on Jetson (run on-device)."""
    console.print(
        "[yellow]Not implemented yet (Phase 2).[/yellow]\n"
        "Plan: measure FPS / latency / power on Orin Nano + Orin NX. See ROADMAP.md §6."
    )
    raise typer.Exit(0)


@app.command()
def version() -> None:
    """Print version info as JSON."""
    from solarscan import __version__

    console.print(json.dumps({"solarscan": __version__}))


if __name__ == "__main__":
    app()
