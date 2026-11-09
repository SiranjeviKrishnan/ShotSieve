"""Autofocus-point extraction from maker notes (Canon AFInfo2, Sony FocusLocation).

`read_af_points(path, meta)` returns the focus boxes in *sensor* orientation, normalised to 0-1
(`AFPoint.x/y` = box centre, top-left origin); an empty list means unknown. The pure parser reads
only the first 512 KB of the file (no full-file reads) and never raises. An optional exiftool
fallback (one `-stay_open` process) handles files the parser could not read.
"""

from __future__ import annotations

import atexit
import json
import shutil
import struct
import subprocess
import threading
from pathlib import Path
from typing import NamedTuple

from shotsieve.log import get_logger

log = get_logger("makernotes")

HEAD_BYTES = 512 * 1024
SONY_BOX_FRAC = 0.06  # Sony reports a point only: box = 6 % of the short edge
_SONY_MAGIC = b"SONY DSC \x00\x00\x00"
_TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8}
_TYPE_FMT = {1: "B", 2: "B", 3: "H", 4: "I", 6: "b", 7: "B", 8: "h", 9: "i"}


class AFPoint(NamedTuple):
    x: float  # centre, 0-1, left to right (sensor orientation)
    y: float  # centre, 0-1, top to bottom
    w: float
    h: float


class _Tiff:
    """Minimal TIFF/IFD reader over an in-memory buffer (offsets relative to `base`)."""

    def __init__(self, buf: bytes, base: int = 0) -> None:
        self.buf, self.base = buf, base
        bom = buf[base : base + 2]
        if bom not in (b"II", b"MM"):
            raise ValueError("not a TIFF")
        self.e = "<" if bom == b"II" else ">"
        if struct.unpack_from(self.e + "H", buf, base + 2)[0] != 42:
            raise ValueError("bad TIFF magic")

    def u32(self, pos: int) -> int:
        return struct.unpack_from(self.e + "I", self.buf, pos)[0]

    def first_ifd(self) -> int:
        return self.base + self.u32(self.base + 4)

    def entries(self, ifd: int, rel_base: int | None = None) -> dict[int, tuple[int, int, int]]:
        """tag -> (type, count, absolute value position) for the IFD at absolute `ifd`."""
        rb = self.base if rel_base is None else rel_base
        n = struct.unpack_from(self.e + "H", self.buf, ifd)[0]
        if n > 1000:
            raise ValueError("implausible IFD")
        out = {}
        for i in range(n):
            p = ifd + 2 + 12 * i
            tag, typ, cnt = struct.unpack_from(self.e + "HHI", self.buf, p)
            size = _TYPE_SIZE.get(typ, 1) * cnt
            pos = p + 8 if size <= 4 else rb + self.u32(p + 8)
            out[tag] = (typ, cnt, pos)
        return out

    def values(self, entry: tuple[int, int, int]) -> list[int]:
        typ, cnt, pos = entry
        fmt = _TYPE_FMT.get(typ)
        if fmt is None or cnt > 4096:
            raise ValueError("unsupported tag type")
        return list(struct.unpack_from(f"{self.e}{cnt}{fmt}", self.buf, pos))


def _canon_points(t: _Tiff, mn: dict[int, tuple[int, int, int]]) -> list[AFPoint]:
    """Canon AFInfo2 (tag 0x0026): int16 array
    [size, areaMode, N, valid, imgW, imgH, afW, afH, widths[N], heights[N], x[N], y[N],
    inFocus[ceil(N/16)], selected[ceil(N/16)]]; positions are signed, centre-origin, Y up."""
    if 0x0026 not in mn:
        return []
    typ, cnt, pos = mn[0x0026]
    if typ not in (3, 8):
        return []
    v = list(struct.unpack_from(f"{t.e}{cnt}h", t.buf, pos))
    n = v[2]
    words = (n + 15) // 16
    if not 0 < n <= 200 or len(v) < 8 + 4 * n + words:
        return []
    afw, afh = v[6], v[7]
    if afw <= 0 or afh <= 0:
        return []
    widths, heights = v[8 : 8 + n], v[8 + n : 8 + 2 * n]
    xs, ys = v[8 + 2 * n : 8 + 3 * n], v[8 + 3 * n : 8 + 4 * n]
    in_focus = v[8 + 4 * n : 8 + 4 * n + words]
    selected = v[8 + 4 * n + words : 8 + 4 * n + 2 * words]

    def pick(mask: list[int]) -> list[int]:
        return [i for i in range(n) if (mask[i // 16] & 0xFFFF) >> (i % 16) & 1] if mask else []

    idx = pick(in_focus) or pick(selected)
    pts = []
    for i in idx:
        w, h = abs(widths[i]) / afw, abs(heights[i]) / afh
        cx, cy = 0.5 + xs[i] / afw, 0.5 - ys[i] / afh
        if w > 0 and h > 0 and 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0:
            pts.append(AFPoint(cx, cy, min(w, 1.0), min(h, 1.0)))
    return pts


def _sony_points(t: _Tiff, mn: dict[int, tuple[int, int, int]]) -> list[AFPoint]:
    """Sony FocusLocation (0x2027): int16 x4 = [W, H, X, Y], top-left origin."""
    if 0x2027 not in mn or mn[0x2027][1] != 4:
        return []
    w, h, x, y = t.values(mn[0x2027])
    if w <= 0 or h <= 0 or not (0 <= x <= w and 0 <= y <= h) or (x == 0 and y == 0):
        return []
    side = SONY_BOX_FRAC * min(w, h)
    return [AFPoint(x / w, y / h, side / w, side / h)]


def _from_tiff(buf: bytes, base: int) -> list[AFPoint]:
    t = _Tiff(buf, base)
    ifd0 = t.entries(t.first_ifd())
    make = b""
    if 0x010F in ifd0:
        _, cnt, pos = ifd0[0x010F]
        make = buf[pos : pos + cnt].split(b"\x00")[0].strip().lower()
    if 0x8769 not in ifd0:
        return []
    exif = t.entries(base + t.u32(ifd0[0x8769][2]))
    if 0x927C not in exif:
        return []
    _, cnt, pos = exif[0x927C]
    if make.startswith(b"canon"):
        return _canon_points(t, t.entries(pos))
    if make.startswith(b"sony"):
        # "SONY DSC \0\0\0" header + IFD (JPEG, older ARW); newer ARW store a bare IFD
        start = pos + 12 if buf[pos : pos + 12] == _SONY_MAGIC else pos
        return _sony_points(t, t.entries(start, rel_base=base))
    return []


def parse_af_points(buf: bytes) -> list[AFPoint]:
    """Parse AF points from the head of a JPEG / TIFF-based RAW / CR3 file. [] when unknown."""
    try:
        if buf[:2] == b"\xff\xd8":  # JPEG: Exif APP1
            p = 2
            while p + 4 < len(buf) and buf[p] == 0xFF:
                marker, size = buf[p + 1], struct.unpack_from(">H", buf, p + 2)[0]
                if marker == 0xE1 and buf[p + 4 : p + 10] == b"Exif\x00\x00":
                    return _from_tiff(buf, p + 10)
                if marker == 0xDA:
                    break
                p += 2 + size
            return []
        if buf[4:8] == b"ftyp":  # CR3: TIFF inside the CMT3 box
            i = buf.find(b"CMT3")
            return _from_tiff(buf, i + 4) if i >= 0 else []
        return _from_tiff(buf, 0)
    except (ValueError, struct.error, IndexError):
        return []


class _ExifTool:
    """One persistent `exiftool -stay_open` process, used only as a fallback."""

    def __init__(self, exe: str) -> None:
        self.proc = subprocess.Popen(  # noqa: S603
            [exe, "-stay_open", "True", "-@", "-"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )  # fmt: skip
        self.lock = threading.Lock()
        atexit.register(self.close)

    def run(self, args: list[str]) -> str:
        with self.lock:
            assert self.proc.stdin is not None and self.proc.stdout is not None
            self.proc.stdin.write(("\n".join([*args, "-execute", ""])).encode())
            self.proc.stdin.flush()
            out = []
            while (line := self.proc.stdout.readline()):
                if line.strip() == b"{ready}":
                    break
                out.append(line.decode("utf-8", "replace"))
            return "".join(out)

    def close(self) -> None:
        try:
            if self.proc.poll() is None and self.proc.stdin:
                self.proc.stdin.write(b"-stay_open\nFalse\n")
                self.proc.stdin.close()
                self.proc.wait(timeout=3)
        except Exception:  # noqa: BLE001
            self.proc.kill()


_et: _ExifTool | None = None
_ET_TAGS = (
    "-AFImageWidth", "-AFImageHeight", "-AFAreaWidths", "-AFAreaHeights", "-AFAreaXPositions",
    "-AFAreaYPositions", "-AFPointsInFocus", "-FocusLocation",
)  # fmt: skip


def _exiftool_points(path: Path) -> list[AFPoint]:
    global _et
    exe = shutil.which("exiftool")
    if not exe:
        return []
    try:
        if _et is None or _et.proc.poll() is not None:
            _et = _ExifTool(exe)
        d = json.loads(_et.run(["-j", "-n", *_ET_TAGS, str(path)]))[0]
        if "FocusLocation" in d:
            w, h, x, y = (float(v) for v in str(d["FocusLocation"]).split()[:4])
            side = SONY_BOX_FRAC * min(w, h)
            return [AFPoint(x / w, y / h, side / w, side / h)] if w > 0 and h > 0 else []
        if "AFAreaXPositions" in d and "AFImageWidth" in d:
            w, h = float(d["AFImageWidth"]), float(d["AFImageHeight"])
            ws = [float(v) for v in str(d["AFAreaWidths"]).split()]
            hs = [float(v) for v in str(d["AFAreaHeights"]).split()]
            xs = [float(v) for v in str(d["AFAreaXPositions"]).split()]
            ys = [float(v) for v in str(d["AFAreaYPositions"]).split()]
            mask = [int(v) for v in str(d.get("AFPointsInFocus", "")).replace(",", " ").split()]
            idx = mask or list(range(len(xs)))
            return [
                AFPoint(0.5 + xs[i] / w, 0.5 - ys[i] / h, ws[i] / w, hs[i] / h)
                for i in idx if i < len(xs)
            ]  # fmt: skip
    except Exception as exc:  # noqa: BLE001 - fallback is best effort
        log.debug("exiftool AF fallback failed for %s: %s", path, exc)
    return []


def read_af_points(
    path: str | Path, meta: object = None, use_exiftool: bool = False
) -> list[AFPoint]:
    """AF boxes (normalised, sensor orientation) of `path`; [] = unknown. Never raises."""
    path = Path(path)
    make = str(getattr(meta, "make", "") or "").lower()
    if make and not make.startswith(("canon", "sony")):
        return []
    try:
        with open(path, "rb") as fh:
            head = fh.read(HEAD_BYTES)
    except OSError:
        return []
    pts = parse_af_points(head)
    if not pts and use_exiftool and (not make or make.startswith(("canon", "sony"))):
        pts = _exiftool_points(path)
    return pts


def to_oriented(pt: AFPoint, orientation: int) -> AFPoint:
    """Map a sensor-orientation box into the upright (EXIF-oriented) preview frame."""
    x, y, w, h = pt
    if orientation == 2:
        return AFPoint(1 - x, y, w, h)
    if orientation == 3:
        return AFPoint(1 - x, 1 - y, w, h)
    if orientation == 4:
        return AFPoint(x, 1 - y, w, h)
    if orientation == 5:
        return AFPoint(y, x, h, w)
    if orientation == 6:
        return AFPoint(1 - y, x, h, w)
    if orientation == 7:
        return AFPoint(1 - y, 1 - x, h, w)
    if orientation == 8:
        return AFPoint(y, 1 - x, h, w)
    return pt
