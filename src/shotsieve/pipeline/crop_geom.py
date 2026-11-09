"""Crop geometry (docs/11_CROP.md section 4): display-normalised rectangles, the 8 EXIF orientation
transforms, `crs:Crop*` conversion (stored/sensor frame), exact full-resolution ratio snapping and
ratio parsing. Pure Python, no models, no I/O."""

from __future__ import annotations

import math
from dataclasses import dataclass

# crs:CropLeft/Top/Right/Bottom are fractions of the image in STORED (sensor) orientation, i.e.
# before the EXIF orientation is applied (11 section 4.2). If Lightroom turns out to use the
# displayed orientation instead, change only this constant (the user's manual check, acceptance 9).
CRS_FRAME = "stored"

PRESET_RATIOS = (
    "original",
    "1:1",
    "4:5",
    "5:4",
    "2:3",
    "3:2",
    "5:7",
    "7:5",
    "16:9",
    "9:16",
    "free",
)
MAX_RATIO_SPREAD = 4.0
EPS = 1e-9


@dataclass(frozen=True)
class Rect:
    """Display-normalised crop rectangle: x0 < x1, y0 < y1, all in [0, 1]."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def w(self) -> float:
        return self.x1 - self.x0

    @property
    def h(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return self.w * self.h

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)

    def iou(self, o: Rect) -> float:
        iw = min(self.x1, o.x1) - max(self.x0, o.x0)
        ih = min(self.y1, o.y1) - max(self.y0, o.y0)
        if iw <= 0 or ih <= 0:
            return 0.0
        inter = iw * ih
        return inter / (self.area + o.area - inter)

    @staticmethod
    def of(v: tuple[float, float, float, float] | list[float]) -> Rect:
        return Rect(float(v[0]), float(v[1]), float(v[2]), float(v[3]))

    def valid(self) -> bool:
        return -EPS <= self.x0 < self.x1 <= 1 + EPS and -EPS <= self.y0 < self.y1 <= 1 + EPS


# ---- the 8 EXIF orientations -----------------------------------------------------------------
# (u, v) = point in the STORED image (0..1), (x, y) = the same point in the DISPLAYED image.


def orient_point(o: int, u: float, v: float) -> tuple[float, float]:
    """Stored -> display for EXIF orientation `o` (1 id, 2 mirror-x, 3 rot180, 4 mirror-y,
    5 transpose, 6 rot90 CW, 7 transverse, 8 rot90 CCW)."""
    if o == 1:
        return u, v
    if o == 2:
        return 1 - u, v
    if o == 3:
        return 1 - u, 1 - v
    if o == 4:
        return u, 1 - v
    if o == 5:
        return v, u
    if o == 6:
        return 1 - v, u
    if o == 7:
        return 1 - v, 1 - u
    if o == 8:
        return v, 1 - u
    raise ValueError(f"EXIF orientation must be 1..8, got {o}")


_INVERSE = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 8, 7: 7, 8: 6}


def unorient_point(o: int, x: float, y: float) -> tuple[float, float]:
    """Display -> stored."""
    return orient_point(_INVERSE[_check(o)], x, y)


def _check(o: int) -> int:
    if o not in _INVERSE:
        raise ValueError(f"EXIF orientation must be 1..8, got {o}")
    return o


def _map_rect(f, r: Rect) -> Rect:
    a = f(r.x0, r.y0)
    b = f(r.x1, r.y1)
    return Rect(min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))


def display_dims(stored_w: int, stored_h: int, orientation: int) -> tuple[int, int]:
    """(w, h) of the displayed image: swapped for orientations 5-8."""
    return (stored_h, stored_w) if _check(orientation) >= 5 else (stored_w, stored_h)


def to_crs(rect: Rect, orientation: int) -> Rect:
    """Display-normalised rect -> `crs:` rect (left, top, right, bottom) in the stored frame."""
    _check(orientation)
    if CRS_FRAME == "display":
        return rect
    return _map_rect(lambda x, y: unorient_point(orientation, x, y), rect)


def from_crs(rect: Rect, orientation: int) -> Rect:
    """`crs:` rect (stored frame) -> display-normalised rect."""
    _check(orientation)
    if CRS_FRAME == "display":
        return rect
    return _map_rect(lambda u, v: orient_point(orientation, u, v), rect)


# ---- ratios ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Ratio:
    kind: str  # "fixed" | "original" | "free"
    a: float = 0.0
    b: float = 0.0

    def text(self) -> str:
        if self.kind != "fixed":
            return self.kind
        return f"{_num(self.a)}:{_num(self.b)}"

    def flipped(self) -> Ratio:
        return Ratio("fixed", self.b, self.a) if self.kind == "fixed" else self

    def resolve(self, fw: int, fh: int) -> tuple[float, float] | None:
        """(a, b) in full-resolution pixels; None for `free`."""
        if self.kind == "free":
            return None
        if self.kind == "original":
            return float(fw), float(fh)
        return self.a, self.b

    def oriented(self, fw: int, fh: int) -> Ratio:
        """A portrait ratio on a landscape image stays as given; `flip` is explicit (the UI picks
        the image's orientation by default: landscape image -> 3:2 not 2:3)."""
        if self.kind != "fixed" or fw == fh:
            return self
        if (fw > fh) != (self.a > self.b) and self.a != self.b:
            return self.flipped()
        return self


def _num(v: float) -> str:
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return s or "0"


def parse_ratio(text: str | Ratio) -> Ratio:
    """`"4:5"`, `"1.91:1"`, `"original"`, `"free"`. Numbers > 0 with at most 2 decimals and
    max/min <= 4 (1:4 .. 4:1); anything else raises ValueError."""
    if isinstance(text, Ratio):
        return text
    if not isinstance(text, str):
        raise ValueError("ratio must be a string such as '4:5'")
    t = text.strip().lower()
    if t in ("original", "orig"):
        return Ratio("original")
    if t == "free":
        return Ratio("free")
    parts = t.split(":")
    if len(parts) != 2:
        raise ValueError(f"bad ratio {text!r}: expected W:H, 'original' or 'free'")
    vals: list[float] = []
    for p in parts:
        p = p.strip()
        digits = p.replace(".", "", 1)
        if not p or not digits.isdigit() or p.startswith("+"):
            raise ValueError(f"bad ratio {text!r}: not a number")
        if "." in p and len(p.split(".")[1]) > 2:
            raise ValueError(f"bad ratio {text!r}: at most 2 decimals")
        v = float(p)
        if not (v > 0) or not math.isfinite(v):
            raise ValueError(f"bad ratio {text!r}: must be > 0")
        vals.append(v)
    a, b = vals
    if max(a, b) / min(a, b) > MAX_RATIO_SPREAD + 1e-9:
        raise ValueError(f"bad ratio {text!r}: must be between 1:4 and 4:1")
    return Ratio("fixed", a, b)


# ---- exact snapping in full-resolution pixels ------------------------------------------------


def snap_px(rect: Rect, ratio: Ratio, fw: int, fh: int) -> tuple[int, int, int, int]:
    """The pixel rectangle (px0, py0, px1, py1) of `rect` snapped to `ratio` (11 section 4.3):
    H = round(h*fh), W = round(H*a/b); if W > fw reduce H until it fits; keep the centre, clamp
    inside the frame. `|W*b - H*a| <= max(a, b)` (width within 1 px of exact for the integer H).
    `free` rounds the rectangle to whole pixels; `original` is the whole frame."""
    if ratio.kind == "original":
        return 0, 0, fw, fh
    cx = (rect.x0 + rect.x1) / 2 * fw
    cy = (rect.y0 + rect.y1) / 2 * fh
    if ratio.kind == "free":
        w = min(fw, max(1, round(rect.w * fw)))
        h = min(fh, max(1, round(rect.h * fh)))
    else:
        a, b = ratio.a, ratio.b
        h = min(fh, max(1, round(rect.h * fh)))
        w = round(h * a / b)
        if w > fw:
            h = max(1, math.floor(fw * b / a + 1e-9))
            w = round(h * a / b)
            while w > fw and h > 1:
                h -= 1
                w = round(h * a / b)
        w = max(1, w)
    px0 = int(round(cx - w / 2))
    py0 = int(round(cy - h / 2))
    px0 = min(max(px0, 0), fw - w)
    py0 = min(max(py0, 0), fh - h)
    return px0, py0, px0 + w, py0 + h


def snap_rect(rect: Rect, ratio: Ratio, fw: int, fh: int) -> tuple[Rect, tuple[int, int, int, int]]:
    px = snap_px(rect, ratio, fw, fh)
    return Rect(px[0] / fw, px[1] / fh, px[2] / fw, px[3] / fh), px


def ratio_error_px(px: tuple[int, int, int, int], ratio: Ratio, fw: int, fh: int) -> float:
    """|W - H*a/b| in pixels for a pixel rect (0 for free)."""
    ab = ratio.resolve(fw, fh)
    if ab is None:
        return 0.0
    w, h = px[2] - px[0], px[3] - px[1]
    return abs(w - h * ab[0] / ab[1])


def rect_to_px(rect: Rect, fw: int, fh: int) -> tuple[int, int, int, int]:
    return (round(rect.x0 * fw), round(rect.y0 * fh), round(rect.x1 * fw), round(rect.y1 * fh))


def matches_ratio(rect: Rect, ratio: Ratio, fw: int, fh: int, tol_px: float = 1.0) -> bool:
    """Is a (client-supplied) normalised rect within `tol_px` pixels of its exact snapped
    version (both sides rounded to whole pixels first)?"""
    if ratio.kind == "free":
        return True
    px = rect_to_px(rect, fw, fh)
    w, h = px[2] - px[0], px[3] - px[1]
    if w < 1 or h < 1:
        return False
    sp = snap_px(rect, ratio, fw, fh)
    return abs(w - (sp[2] - sp[0])) <= tol_px and abs(h - (sp[3] - sp[1])) <= tol_px
