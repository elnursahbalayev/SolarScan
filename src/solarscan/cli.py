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
    checkpoint: Path | None = typer.Option(
        None, "--checkpoint", "-c", help="Trained classifier checkpoint (.pt); omit to use stub."
    ),
    farm: str | None = typer.Option(
        None, "--farm", help="Farm layout YAML for georeferencing, or 'sample' for the demo farm."
    ),
) -> None:
    """Run the end-to-end pipeline on one image: -> JSON + PDF (+ GeoJSON & map if --farm)."""
    if not input.exists():
        console.print(f"[red]Input not found:[/red] {input}")
        raise typer.Exit(1)

    out.mkdir(parents=True, exist_ok=True)

    classifier = None
    model_version = "stub-0.1.0"
    if checkpoint is not None:
        from solarscan.models.timm_classifier import TimmClassifier  # lazy: needs torch

        classifier = TimmClassifier.load(checkpoint)
        model_version = f"{classifier.backbone_name}:{checkpoint.name}"

    layout = None
    if farm is not None:
        from solarscan.geo import FarmLayout

        layout = FarmLayout.sample() if farm == "sample" else FarmLayout.from_yaml(farm)

    report = run_pipeline(input, classifier=classifier, model_version=model_version, farm=layout)

    stem = input.stem
    json_path = out / f"{stem}_report.json"
    json_path.write_text(report.model_dump_json(indent=2))

    map_png = None
    extra_lines = []
    if layout is not None:
        from solarscan.geo import render_fault_map_html, render_fault_map_png, write_geojson

        geojson_path = write_geojson(report.faults, layout, out / f"{stem}_faults.geojson")
        extra_lines.append(f"  GeoJSON: {geojson_path}")
        map_png = render_fault_map_png(report.faults, layout, out / f"{stem}_map.png")
        if map_png:
            extra_lines.append(f"  Map PNG: {map_png}")
        map_html = render_fault_map_html(report.faults, layout, out / f"{stem}_map.html")
        if map_html:
            extra_lines.append(f"  Map HTML: {map_html}")

    pdf_path = write_pdf_report(report, out / f"{stem}_report.pdf", map_image=map_png)

    console.print(f"[green]Done.[/green] {report.summary.n_faults} fault(s) found.")
    console.print(f"  JSON: {json_path}")
    console.print(f"  PDF:  {pdf_path}")
    for line in extra_lines:
        console.print(line)


@app.command()
def train(
    config: Path | None = typer.Option(None, "--config", help="Config YAML (defaults if omitted)."),
    out: Path | None = typer.Option(None, "--out", "-o", help="Run/output directory."),
    epochs: int | None = typer.Option(None, "--epochs", help="Override config epochs."),
    limit: int | None = typer.Option(None, "--limit", help="Cap samples/split (smoke test)."),
) -> None:
    """Train the IR fault classifier on InfraredSolarModules. Requires `--extra ml`."""
    from solarscan.config import load_config
    from solarscan.training import train_classifier

    cfg = load_config(config)
    if epochs is not None:
        cfg.train.epochs = epochs
    train_classifier(cfg, out_dir=out, limit=limit)


@app.command()
def evaluate(
    checkpoint: Path = typer.Option(..., "--checkpoint", "-c", help="Model checkpoint (.pt)."),
    config: Path | None = typer.Option(None, "--config", help="Config YAML (defaults if omitted)."),
    out: Path | None = typer.Option(None, "--out", "-o", help="Where to write metrics."),
) -> None:
    """Evaluate a checkpoint on the held-out test split (macro-F1, per-class, confusion)."""
    import torch
    from torch.utils.data import DataLoader

    from solarscan.config import load_config
    from solarscan.data.infrared_modules import InfraredSolarModules, load_samples, stratified_split
    from solarscan.data.transforms import build_transforms
    from solarscan.evaluation.metrics import compute_metrics, save_confusion_png
    from solarscan.models.timm_classifier import TimmClassifier
    from solarscan.training.train_classifier import evaluate as run_eval

    cfg = load_config(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimmClassifier.load(checkpoint, device=device)

    samples = load_samples(cfg.data.root)
    _, _, test_s = stratified_split(
        samples, cfg.train.val_ratio, cfg.train.test_ratio, cfg.train.seed
    )
    test_ds = InfraredSolarModules(test_s, build_transforms(model.input_size, train=False))
    loader = DataLoader(test_ds, batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers)

    y_true, y_pred = run_eval(model, loader, device)
    metrics = compute_metrics(y_true, y_pred)
    console.print(f"[green]TEST[/green] {metrics.summary_line()}")

    out_dir = out or checkpoint.parent
    metrics.to_json(out_dir / "test_metrics.json")
    save_confusion_png(metrics.confusion, out_dir / "confusion_matrix.png")
    console.print(f"metrics -> {out_dir}")


@app.command()
def export(
    checkpoint: Path = typer.Option(..., "--checkpoint", "-c", help="Model checkpoint (.pt)."),
    out: Path | None = typer.Option(None, "--out", "-o", help="Output .onnx path."),
    opset: int = typer.Option(17, "--opset", help="ONNX opset version."),
    static_batch: bool = typer.Option(False, "--static-batch", help="Fix batch dim to 1."),
) -> None:
    """Export a checkpoint to ONNX (with PyTorch/ONNX parity check). Requires `--extra ml`."""
    from solarscan.edge.onnx_export import export_to_onnx

    path = export_to_onnx(checkpoint, out, opset=opset, dynamic_batch=not static_batch)
    console.print(f"[green]exported[/green] -> {path}")


@app.command()
def benchmark(
    checkpoint: Path = typer.Option(..., "--checkpoint", "-c", help="Model checkpoint (.pt)."),
    onnx: Path | None = typer.Option(None, "--onnx", help="ONNX model to include in the suite."),
    batch: int = typer.Option(32, "--batch", help="Batch size."),
    iters: int = typer.Option(100, "--iters", help="Timed iterations."),
    out: Path | None = typer.Option(None, "--out", "-o", help="JSON output path."),
) -> None:
    """Benchmark latency/throughput across available backends. Requires `--extra ml`."""
    from solarscan.edge.benchmark import run_suite

    run_suite(checkpoint, onnx_path=onnx, batch=batch, iters=iters, out_json=out)


@app.command()
def trt(
    onnx: Path = typer.Option(..., "--onnx", help="ONNX model path."),
    out: Path | None = typer.Option(None, "--out", "-o", help="Output .engine path."),
    precision: str = typer.Option("fp16", "--precision", help="fp32 | fp16 | int8."),
    calib: Path | None = typer.Option(None, "--calib", help="INT8 calibration cache."),
) -> None:
    """[Jetson] Build a TensorRT engine from ONNX via trtexec (run on-device)."""
    from solarscan.edge.trt_build import build_engine

    build_engine(onnx, out, precision=precision, calib_cache=calib)


@app.command()
def version() -> None:
    """Print version info as JSON."""
    from solarscan import __version__

    console.print(json.dumps({"solarscan": __version__}))


if __name__ == "__main__":
    app()
