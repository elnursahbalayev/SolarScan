"""Visual rendering for the demo: thermal colormap + fault overlay.

Turns the raw grayscale IR frame into a thermal-looking RGB image and draws
severity-coloured boxes + labels over detected faults — the visual a client
recognises as "my drone footage, annotated". Matplotlib gives the thermal
colormap when present; otherwise we fall back to plain grayscale->RGB so the
endpoint still works on core deps.
"""

from __future__ import annotations

import base64
import io

from PIL import Image, ImageDraw, ImageFont

from solarscan.schemas import Detection, Fault
from solarscan.taxonomy import Severity

SEVERITY_RGB = {
    Severity.LOW: (253, 224, 71),
    Severity.MEDIUM: (251, 146, 60),
    Severity.HIGH: (239, 68, 68),
    Severity.CRITICAL: (153, 27, 27),
}


def thermal_colormap(image: Image.Image) -> Image.Image:
    """Map a grayscale IR frame to a thermal-style RGB image (inferno)."""
    gray = image.convert("L")
    try:
        import numpy as np
        from matplotlib import cm

        arr = np.asarray(gray, dtype=np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        norm = (arr - lo) / (hi - lo) if hi > lo else arr * 0
        rgb = (cm.inferno(norm)[:, :, :3] * 255).astype("uint8")
        return Image.fromarray(rgb, mode="RGB")
    except ModuleNotFoundError:
        return gray.convert("RGB")


HEALTHY_RGB = (74, 222, 128)  # green for detected, non-faulty modules


def draw_overlay(
    image: Image.Image,
    faults: list[Fault],
    detections: list[Detection] | None = None,
    upscale: int = 3,
    label_faults: bool = True,
) -> Image.Image:
    """Render the thermal frame with all detected modules + severity-coloured faults.

    Healthy detections (if provided) are drawn as thin green boxes; faults are drawn
    thicker in their severity colour on top. ``label_faults`` adds the class text
    (turn off for dense wide frames where labels would overlap).
    """
    base = thermal_colormap(image)
    if upscale > 1:
        base = base.resize((base.width * upscale, base.height * upscale), Image.NEAREST)
    draw = ImageDraw.Draw(base)
    try:
        font = ImageFont.load_default()
    except Exception:  # pragma: no cover
        font = None

    fault_boxes = {(round(f.bbox.x), round(f.bbox.y)) for f in faults if f.bbox}
    for d in detections or []:
        if (round(d.bbox.x), round(d.bbox.y)) in fault_boxes:
            continue  # drawn below as a fault
        x0, y0 = d.bbox.x * upscale, d.bbox.y * upscale
        x1, y1 = (d.bbox.x + d.bbox.w) * upscale, (d.bbox.y + d.bbox.h) * upscale
        draw.rectangle([x0, y0, x1, y1], outline=HEALTHY_RGB, width=1)

    for f in faults:
        if f.bbox is None:
            continue
        x0 = f.bbox.x * upscale
        y0 = f.bbox.y * upscale
        x1 = (f.bbox.x + f.bbox.w) * upscale
        y1 = (f.bbox.y + f.bbox.h) * upscale
        color = SEVERITY_RGB.get(f.severity, (148, 163, 184))
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        if label_faults:
            draw.text((x0 + 2, y0 + 2), f.fault_class.value, fill=color, font=font)
    return base


def to_base64_png(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
