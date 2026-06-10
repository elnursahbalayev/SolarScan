"""Tests for the edge tooling. Skipped when the ml extra (torch) is absent."""

import pytest

pytest.importorskip("torch", reason="requires the `ml` extra")

from solarscan.edge.benchmark import BenchmarkResult, format_table  # noqa: E402
from solarscan.edge.trt_build import trtexec_command  # noqa: E402


def test_trtexec_fp16_flag():
    cmd = trtexec_command("m.onnx", "m.engine", precision="fp16")
    assert "--fp16" in cmd
    assert "--onnx=m.onnx" in cmd


def test_trtexec_int8_includes_calib():
    cmd = trtexec_command("m.onnx", "m.engine", precision="int8", calib_cache="calib.cache")
    assert "--int8" in cmd
    assert "--calib=calib.cache" in cmd


def test_trtexec_rejects_unknown_precision():
    with pytest.raises(ValueError):
        trtexec_command("m.onnx", "m.engine", precision="bogus")


def test_benchmark_table_formatting():
    r = BenchmarkResult(
        backend="pytorch", device="A4000", dtype="fp16", batch_size=32,
        iters=100, latency_ms_mean=1.5, latency_ms_p50=1.4,
        latency_ms_p95=1.8, throughput_imgs_s=21000.0,
    )
    table = format_table([r])
    assert "Throughput" in table
    assert "pytorch" in table and "fp16" in table and "21000" in table
