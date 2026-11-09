"""Draw the ShotSieve app icon (Pillow) into a PNG: `make_icon.py OUT.png` (1024 px)."""

from __future__ import annotations

import sys

from PIL import Image, ImageDraw


def draw(size: int = 1024) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    s = size
    d.rounded_rectangle((s * 0.06, s * 0.06, s * 0.94, s * 0.94), radius=s * 0.22, fill="#1d2330")
    # a sieve: ring, grid of holes; the "kept" holes in green
    cx, cy, r = s / 2, s / 2, s * 0.30
    d.ellipse((cx - r, cy - r, cx + r, cy + r), outline="#e8ecf4", width=int(s * 0.03))
    step = r * 0.5
    for i in range(-2, 3):
        for j in range(-2, 3):
            x, y = cx + i * step, cy + j * step
            if (x - cx) ** 2 + (y - cy) ** 2 > (r * 0.86) ** 2:
                continue
            keep = (i + j) % 3 == 0
            rad = s * (0.036 if keep else 0.022)
            d.ellipse((x - rad, y - rad, x + rad, y + rad), fill="#4cc38a" if keep else "#566079")
    return img


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: make_icon.py OUT.png", file=sys.stderr)
        return 2
    draw().save(argv[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
