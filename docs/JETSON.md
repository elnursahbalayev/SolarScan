# Jetson Edge Benchmark — on-device steps

Run this on the **Orin Nano** and the **Orin NX** to replace the *projected*
figures in [ROADMAP §6](../ROADMAP.md) with real measurements. JetPack ships
TensorRT, `trtexec`, and `tegrastats`.

## 0. Copy the artifacts to the device
You need the exported ONNX and a handful of representative crops for INT8 calibration:
```bash
scp runs/convnext_tiny/model.onnx   jetson:~/solarscan/
scp -r data/calib                   jetson:~/solarscan/   # ~200 varied IR crops
```

## 1. Build the TensorRT engines
```bash
solarscan trt --onnx model.onnx --precision fp16 -o model.fp16.engine
solarscan trt --onnx model.onnx --precision int8 --calib data/calib -o model.int8.engine
# (solarscan trt shells out to trtexec, which JetPack provides)
```

## 2. Benchmark — same harness, real numbers
Add a TensorRT row source to `solarscan benchmark` (the engine runner) and run at
batch 1 (single-stream is the in-flight metric):
```bash
solarscan benchmark -c runs/convnext_tiny/best.pt --engine model.int8.engine \
    --batch 1 --iters 500 -o results/orin_nx_int8.json
```
Set the device power mode first and record it (`sudo nvpmodel -m <mode>`, `sudo jetson_clocks`).

## 3. Capture power
In another shell while the benchmark runs:
```bash
tegrastats --interval 100 | tee results/orin_nx_int8.tegrastats.log
```
Read the `VDD_IN` / module power column for the average draw.

## 4. Verify accuracy after INT8
Re-run `solarscan evaluate` through the INT8 engine on the held-out test split and
record the macro-F1 delta vs FP16 — this is the accuracy/throughput tradeoff line.

## 5. Update the docs
Drop the measured numbers into ROADMAP §6, **remove the ⚠️ PROJECTED banner and the
*(proj.)* tags**, and note the JetPack/TensorRT versions and power mode used.
```
