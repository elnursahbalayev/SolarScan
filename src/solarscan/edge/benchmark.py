"""Latency / throughput benchmarking for the classifier.

Produces the cross-backend table that is this project's headline. On the dev
A4000 we measure PyTorch (FP32/FP16) and ONNX Runtime now; the same harness runs
on the Jetson Orin Nano / NX to add the TensorRT (FP16/INT8) rows later, giving a
single comparable table across all backends and devices (ROADMAP §6).

Timing is done with warmup + CUDA synchronisation so the numbers are real.
"""

from __future__ import annotations

import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from solarscan.models.timm_classifier import TimmClassifier


@dataclass
class BenchmarkResult:
    backend: str
    device: str
    dtype: str
    batch_size: int
    iters: int
    latency_ms_mean: float
    latency_ms_p50: float
    latency_ms_p95: float
    throughput_imgs_s: float

    def row(self) -> str:
        return (
            f"| {self.backend} | {self.device} | {self.dtype} | {self.batch_size} | "
            f"{self.latency_ms_mean:.2f} | {self.latency_ms_p50:.2f} | "
            f"{self.latency_ms_p95:.2f} | {self.throughput_imgs_s:.0f} |"
        )


def _summarise(times_s: list[float], backend, device, dtype, batch) -> BenchmarkResult:
    ms = [t * 1000.0 for t in times_s]
    ms_sorted = sorted(ms)
    p = lambda q: ms_sorted[min(int(q * len(ms_sorted)), len(ms_sorted) - 1)]  # noqa: E731
    mean_ms = statistics.fmean(ms)
    return BenchmarkResult(
        backend=backend,
        device=device,
        dtype=dtype,
        batch_size=batch,
        iters=len(ms),
        latency_ms_mean=mean_ms,
        latency_ms_p50=p(0.50),
        latency_ms_p95=p(0.95),
        throughput_imgs_s=batch / (mean_ms / 1000.0),
    )


def benchmark_torch(
    checkpoint: str | Path,
    device: str = "cuda",
    dtype: str = "fp32",
    batch: int = 32,
    iters: int = 100,
    warmup: int = 20,
) -> BenchmarkResult:
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model = TimmClassifier.load(checkpoint, device=dev)
    model.eval()
    torch_dtype = torch.float16 if dtype == "fp16" else torch.float32
    if dtype == "fp16":
        model = model.half()
    h, w = model.input_size
    x = torch.randn(batch, 3, h, w, device=dev, dtype=torch_dtype)

    is_cuda = dev.type == "cuda"
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        if is_cuda:
            torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(x)
            if is_cuda:
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    dev_name = torch.cuda.get_device_name(0) if is_cuda else platform.processor() or "cpu"
    return _summarise(times, "pytorch", dev_name, dtype, batch)


def benchmark_onnx(
    onnx_path: str | Path,
    providers: list[str] | None = None,
    batch: int = 32,
    iters: int = 100,
    warmup: int = 20,
) -> BenchmarkResult:
    import onnxruntime as ort

    if providers is None:
        # Prefer real compute providers; the Azure provider is a remote no-op stub.
        available = [p for p in ort.get_available_providers() if p != "AzureExecutionProvider"]
        preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        providers = [p for p in preferred if p in available] or available
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    inp = sess.get_inputs()[0]
    # Static-shape exports report a fixed batch; honour it.
    b = batch if isinstance(inp.shape[0], str) else int(inp.shape[0])
    _, c, h, w = (b, *[int(d) for d in inp.shape[1:]])
    x = np.random.randn(b, c, h, w).astype(np.float32)

    for _ in range(warmup):
        sess.run(None, {inp.name: x})
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(None, {inp.name: x})
        times.append(time.perf_counter() - t0)

    active = sess.get_providers()[0].replace("ExecutionProvider", "")
    return _summarise(times, "onnxruntime", active, "fp32", b)


def run_suite(
    checkpoint: str | Path,
    onnx_path: str | Path | None = None,
    batch: int = 32,
    iters: int = 100,
    out_json: str | Path | None = None,
) -> list[BenchmarkResult]:
    """Run all backends available on this machine and write a JSON + markdown table."""
    results: list[BenchmarkResult] = []
    results.append(benchmark_torch(checkpoint, "cpu", "fp32", batch, iters))
    if torch.cuda.is_available():
        results.append(benchmark_torch(checkpoint, "cuda", "fp32", batch, iters))
        results.append(benchmark_torch(checkpoint, "cuda", "fp16", batch, iters))
    if onnx_path and Path(onnx_path).exists():
        results.append(benchmark_onnx(onnx_path, batch=batch, iters=iters))

    table = format_table(results)
    print(table)
    if out_json:
        out = Path(out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps([asdict(r) for r in results], indent=2))
        out.with_suffix(".md").write_text(table + "\n")
        print(f"\nresults -> {out} (+ .md)")
    return results


def format_table(results: list[BenchmarkResult]) -> str:
    header = (
        "| Backend | Device | Precision | Batch | Mean ms | p50 ms | p95 ms | "
        "Throughput (img/s) |\n|---|---|---|---|---|---|---|---|"
    )
    return "\n".join([header, *(r.row() for r in results)])
