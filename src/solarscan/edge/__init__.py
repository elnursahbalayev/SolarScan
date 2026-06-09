"""[Phase 2] Edge deployment: ONNX export, TensorRT build, on-device benchmark.

This is the project's headline differentiator. Run the benchmark on BOTH the
Jetson Orin Nano and Orin NX to produce the cross-device FP16/INT8 table in
ROADMAP.md §6.

Planned modules:
    onnx_export.py  — torch model -> ONNX (static + dynamic shapes)
    trt_build.py    — ONNX -> TensorRT engine (FP16 + INT8 calibration)
    benchmark.py    — FPS / latency p50,p95 / accuracy delta / power, as JSON

`tensorrt` and `pycuda` ship with JetPack on the devices — do not install on
dev/CI machines.
"""
