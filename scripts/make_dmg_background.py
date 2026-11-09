# ruff: noqa: E501
"""Draw the .dmg window background (660x400): `make_dmg_background.py OUT.png`."""

from __future__ import annotations

import sys

from PIL import Image, ImageDraw, ImageFont

W, H = 660, 400


def font(size: int) -> ImageFont.ImageFont:
    for p in ("/System/Library/Fonts/SFNS.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main(out: str) -> None:
    img = Image.new("RGB", (W, H), "#f4f5f8")
    d = ImageDraw.Draw(img)
    d.text((W / 2, 52), "Drag ShotSieve to Applications", fill="#1d2330", font=font(24), anchor="mm")
    d.text((W / 2, 84), "Private, offline AI photo culling", fill="#6b7385", font=font(14), anchor="mm")
    y = 190  # arrow between the two icon slots (170 and 490)
    d.line((255, y, 395, y), fill="#8b93a7", width=6)
    d.polygon([(395, y - 18), (425, y), (395, y + 18)], fill="#8b93a7")
    d.text((W / 2, 340), "First open: right-click ShotSieve in Applications, then Open.",
           fill="#6b7385", font=font(13), anchor="mm")
    img.save(out)


if __name__ == "__main__":
    main(sys.argv[1])
