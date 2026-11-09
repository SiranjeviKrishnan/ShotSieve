"""Global image statistics that drive the editing profile (T15). NaN-tolerant, ~1-3 ms."""

from __future__ import annotations

import math
import zlib
from datetime import datetime

import cv2
import numpy as np

from shotsieve.features import exposure

EDIT_STATS_VERSION = 2  # 2: camera_idx = crc32 % 16777213 (Review #7 n4; was % 997)
STATS_EDGE = 256
_N_HIST = 16
_LIN = (np.arange(256, dtype=np.float32) / 255.0) ** 2.2
_LIN6 = _LIN.astype(np.float64) ** 6  # shades-of-grey p = 6 on linear light
_LOGLIN = np.log2(np.maximum(_LIN, 2.0**-8))
_EDGES = np.linspace(-8.0, 0.0, _N_HIST + 1)  # log2 of linear luma

EDIT_STAT_NAMES: tuple[str, ...] = (
    *(f"lhist{i}" for i in range(_N_HIST)),
    "p1", "p5", "p50", "p95", "p99", "clip_hi", "clip_lo", "ev", "rms_contrast",
    "gw_rg", "gw_bg", "sop_rg", "sop_bg", "mid_a", "mid_b", "chroma_mean", "chroma_p90",
    "log2_iso", "log2_t", "fnumber", "ev100", "flash", "focal", "hour", "camera_idx",
    "wb_mired", "wb_duv",
    "n_faces", "face_luma",
)  # fmt: skip
N_STATS = len(EDIT_STAT_NAMES)


def _shrink(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    s = STATS_EDGE / max(h, w)
    if s >= 1.0:
        return rgb
    return cv2.resize(
        rgb, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA
    )


def _safe_log2(x: float | None) -> float:
    return math.log2(x) if x and x > 0 else float("nan")


def edit_stats(
    rgb: np.ndarray,
    exif: object | None = None,
    raw_wb: tuple[float, float] | None = None,
    faces: list[tuple[float, float, float, float]] | None = None,
) -> np.ndarray:
    """Statistics vector (EDIT_STAT_NAMES, float32) of an RGB uint8 preview.

    `exif` is an `ImageMeta` (or anything with the same attributes), `raw_wb` the as-shot
    (mired, duv) of a RAW (None for JPEG: NaN), `faces` normalised boxes (x, y, w, h) if known.
    """
    img = _shrink(np.ascontiguousarray(rgb[:, :, :3]))
    out = np.full(N_STATS, np.nan, dtype=np.float32)
    idx = {n: i for i, n in enumerate(EDIT_STAT_NAMES)}
    # luma
    luma = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    counts = np.bincount(luma.ravel(), minlength=256).astype(np.float64)
    total = max(counts.sum(), 1.0)
    bins = np.clip(np.digitize(_LOGLIN, _EDGES[1:-1]), 0, _N_HIST - 1)
    out[:_N_HIST] = np.bincount(bins, weights=counts, minlength=_N_HIST) / total
    cdf = np.cumsum(counts) / total
    for name, q in (("p1", 0.01), ("p5", 0.05), ("p50", 0.5), ("p95", 0.95), ("p99", 0.99)):
        out[idx[name]] = min(int(np.searchsorted(cdf, q)), 255) / 255.0
    e = exposure.analyse(luma)
    out[idx["clip_hi"]], out[idx["clip_lo"]], out[idx["ev"]] = e.clip_hi, e.clip_lo, e.ev
    mean_l = float((counts * np.arange(256)).sum() / total)
    var_l = float((counts * (np.arange(256) - mean_l) ** 2).sum() / total)
    out[idx["rms_contrast"]] = math.sqrt(var_l) / 255.0
    # colour: grey-world and shades-of-grey (p = 6) ln R/G, ln B/G on linear light
    m = np.array(
        [_LIN[img[:, :, c]].mean() for c in range(3)], dtype=np.float64
    ) + 1e-6
    out[idx["gw_rg"]], out[idx["gw_bg"]] = math.log(m[0] / m[1]), math.log(m[2] / m[1])
    p6 = np.array([_LIN6[img[:, :, c]].mean() for c in range(3)]) ** (1 / 6) + 1e-6
    out[idx["sop_rg"]], out[idx["sop_bg"]] = math.log(p6[0] / p6[1]), math.log(p6[2] / p6[1])
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    a, b = lab[:, :, 1] - 128.0, lab[:, :, 2] - 128.0
    mid = (luma > 64) & (luma < 192)
    if mid.sum() >= 16:
        out[idx["mid_a"]], out[idx["mid_b"]] = float(a[mid].mean()), float(b[mid].mean())
    chroma = np.hypot(a, b)
    ch = np.bincount(np.minimum((chroma * 2).astype(np.int32).ravel(), 400), minlength=401)
    out[idx["chroma_mean"]] = float(chroma.mean())
    out[idx["chroma_p90"]] = float(np.searchsorted(np.cumsum(ch) / ch.sum(), 0.9)) / 2.0
    # EXIF
    if exif is not None:
        iso = getattr(exif, "iso", None)
        t = getattr(exif, "exposure_time", None)
        n = getattr(exif, "fnumber", None)
        out[idx["log2_iso"]] = _safe_log2(iso)
        out[idx["log2_t"]] = _safe_log2(t)
        out[idx["fnumber"]] = float(n) if n else np.nan
        if t and n and t > 0 and n > 0:
            iso_stops = _safe_log2(iso) - math.log2(100.0) if iso else 0.0
            out[idx["ev100"]] = math.log2(n * n / t) - iso_stops
        fl = getattr(exif, "flash_fired", None)
        out[idx["flash"]] = np.nan if fl is None else float(bool(fl))
        fo = getattr(exif, "focal_length", None)
        out[idx["focal"]] = float(fo) if fo else np.nan
        ts = getattr(exif, "timestamp", None)
        if ts and getattr(exif, "ts_source", "") == "exif":
            try:
                d = datetime.fromtimestamp(float(ts))
                out[idx["hour"]] = d.hour + d.minute / 60.0
            except (OSError, ValueError, OverflowError):
                pass
        cam = f"{getattr(exif, 'make', '')}|{getattr(exif, 'model', '')}".strip("|")
        if cam:
            # < 2**24: exact in float32, ~1e-7 collision odds (Review #7 n4; was % 997)
            out[idx["camera_idx"]] = float(zlib.crc32(cam.encode()) % 16777213)
    if raw_wb is not None:
        out[idx["wb_mired"]], out[idx["wb_duv"]] = raw_wb
    if faces is not None:
        out[idx["n_faces"]] = float(len(faces))
        if faces:
            h, w = luma.shape
            vals = []
            for x, y, fw, fh in faces:
                x0, y0 = int(max(x, 0) * w), int(max(y, 0) * h)
                x1, y1 = int(min(x + fw, 1) * w), int(min(y + fh, 1) * h)
                if x1 > x0 and y1 > y0:
                    vals.append(float(luma[y0:y1, x0:x1].mean()) / 255.0)
            if vals:
                out[idx["face_luma"]] = float(np.mean(vals))
    return out
