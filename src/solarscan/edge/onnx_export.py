"""Export a trained TimmClassifier to ONNX and verify numerical parity.

ONNX is the portable hand-off point: from here the model runs under onnxruntime
anywhere, and on the Jetson it is built into a TensorRT engine (see trt_build.py).
We always verify PyTorch vs ONNX Runtime outputs agree before trusting the export.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from solarscan.models.timm_classifier import TimmClassifier


def export_to_onnx(
    checkpoint: str | Path,
    out_path: str | Path | None = None,
    opset: int = 17,
    dynamic_batch: bool = True,
    verify: bool = True,
) -> Path:
    """Export ``checkpoint`` to ONNX. Returns the .onnx path. Raises if parity fails."""
    checkpoint = Path(checkpoint)
    out_path = Path(out_path) if out_path else checkpoint.with_suffix(".onnx")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = TimmClassifier.load(checkpoint, device="cpu")
    model.eval()
    h, w = model.input_size
    dummy = torch.randn(1, 3, h, w)

    dynamic_axes = (
        {"input": {0: "batch"}, "logits": {0: "batch"}} if dynamic_batch else None
    )
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=opset,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        dynamo=False,  # stable TorchScript exporter; avoids the onnxscript dependency
    )

    if verify:
        _verify_parity(model, out_path, dummy)
    return out_path


def _verify_parity(model: TimmClassifier, onnx_path: Path, dummy: torch.Tensor) -> None:
    import onnxruntime as ort

    with torch.inference_mode():
        torch_out = model(dummy).numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(["logits"], {"input": dummy.numpy()})[0]

    max_abs_diff = float(np.max(np.abs(torch_out - onnx_out)))
    if not np.allclose(torch_out, onnx_out, atol=1e-3):
        raise RuntimeError(
            f"ONNX parity check failed: max abs diff {max_abs_diff:.2e} > 1e-3"
        )
    print(f"ONNX parity OK (max abs diff {max_abs_diff:.2e}) -> {onnx_path}")
