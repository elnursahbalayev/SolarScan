"""Build a TensorRT engine from the ONNX model — runs ON THE JETSON.

TensorRT and `trtexec` ship with JetPack on the Orin Nano / NX; they are not
installed on the dev (A4000) box, so this module shells out to `trtexec` when it
is present and otherwise prints the exact command. INT8 needs a calibration set
of representative thermal crops.

On the Jetson:
    solarscan-trt build model.onnx --precision int8 --calib data/calib/
then benchmark the resulting .engine with the same harness used for every other
backend, so the final ROADMAP §6 table is apples-to-apples.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def trtexec_command(
    onnx_path: str | Path,
    engine_path: str | Path,
    precision: str = "fp16",
    calib_cache: str | Path | None = None,
) -> list[str]:
    """Construct the trtexec command for the requested precision."""
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--shapes=input:1x3x64x64",
    ]
    if precision == "fp16":
        cmd.append("--fp16")
    elif precision == "int8":
        cmd.append("--int8")
        if calib_cache:
            cmd.append(f"--calib={calib_cache}")
    elif precision != "fp32":
        raise ValueError(f"unknown precision: {precision}")
    return cmd


def build_engine(
    onnx_path: str | Path,
    engine_path: str | Path | None = None,
    precision: str = "fp16",
    calib_cache: str | Path | None = None,
) -> Path | None:
    """Build the engine if trtexec is available; otherwise print the command and skip."""
    onnx_path = Path(onnx_path)
    engine_path = Path(engine_path or onnx_path.with_suffix(f".{precision}.engine"))
    cmd = trtexec_command(onnx_path, engine_path, precision, calib_cache)

    if shutil.which("trtexec") is None:
        print("trtexec not found — run this on the Jetson (JetPack provides it):")
        print("  " + " ".join(str(c) for c in cmd))
        return None

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)
    print(f"engine -> {engine_path}")
    return engine_path
