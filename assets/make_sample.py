"""Generate a synthetic thermal-like sample image for the Phase-0 demo.

Not real data — just a deterministic placeholder so `make demo` works out of the
box. Produces a warm gradient with a bright hot-spot so the stub model flags a
fault. Run: python assets/make_sample.py
"""

from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).parent / "sample_thermal.png"


def main() -> None:
    w, h = 320, 200
    img = Image.new("L", (w, h))
    px = img.load()
    for y in range(h):
        for x in range(w):
            px[x, y] = int(120 + 30 * (x / w))  # mild warm gradient
    draw = ImageDraw.Draw(img)
    draw.ellipse((230, 60, 290, 120), fill=255)  # hot-spot
    img.save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
