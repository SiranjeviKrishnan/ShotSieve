"""Exposure statistics from the luma histogram."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

MID_GREY_LIN = (118.0 / 255.0) ** 2.2  # ~18% grey in the linear-ish space used here
GAMMA = 2.2
_LIN_LUT = ((np.arange(256, dtype=np.float64) / 255.0) ** GAMMA).astype(np.float64)


@dataclass(frozen=True)
class Exposure:
    mean: float  # mean luma 0..255
    lin_mean: float  # mean of (L/255)^2.2
    clip_hi: float  # fraction of pixels >= 253
    clip_lo: float  # fraction of pixels <= 2
    ev: float  # EV offset of the mean vs mid-grey (negative = dark)
    p98: float  # 98th percentile luma


def ev_from_lin(lin_mean: float) -> float:
    return math.log2(max(lin_mean, 1e-6) / MID_GREY_LIN)


def analyse(gray: np.ndarray) -> Exposure:
    """Exposure stats for a uint8 luma image (use a small thumbnail for speed)."""
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    n = max(hist.sum(), 1.0)
    p = hist / n
    mean = float((p * np.arange(256)).sum())
    lin = float((p * _LIN_LUT).sum())
    cdf = np.cumsum(p)
    p98 = float(np.searchsorted(cdf, 0.98))
    return Exposure(
        mean=mean,
        lin_mean=lin,
        clip_hi=float(p[253:].sum()),
        clip_lo=float(p[:3].sum()),
        ev=ev_from_lin(lin),
        p98=p98,
    )
