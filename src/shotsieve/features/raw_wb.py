"""As-shot white balance of a RAW file, from metadata only (no demosaic).

neutral = 1 / camera_whitebalance (DNG: AsShotNeutral when present) -> XYZ through the inverse
of `rgb_xyz_matrix` -> xy -> CCT (McCamy) + Delta-uv. The absolute value differs from Adobe's
Kelvin; it is only ever used as a *difference* (target = edited mired - as-shot mired), the same
way in training and prediction.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from shotsieve.io import dng_xmp


def xyz_to_cct_duv(xyz: np.ndarray) -> tuple[float, float] | None:
    """(CCT in K by McCamy, signed Delta-uv to the Planckian locus), or None when degenerate."""
    X, Y, Z = (float(v) for v in xyz)
    s = X + Y + Z
    if not math.isfinite(s) or s <= 0:
        return None
    x, y = X / s, Y / s
    if y <= 0 or abs(0.1858 - y) < 1e-9:
        return None
    n = (x - 0.3320) / (0.1858 - y)
    cct = 449.0 * n**3 + 3525.0 * n**2 + 6823.3 * n + 5520.33
    if not math.isfinite(cct) or not 1500.0 < cct < 40000.0:
        return None
    d = X + 15.0 * Y + 3.0 * Z
    u, v = 4.0 * X / d, 6.0 * Y / d
    t = cct
    # Krystek (1985) polynomial for the Planckian locus in CIE 1960 uv
    ul = (0.860117757 + 1.54118254e-4 * t + 1.28641212e-7 * t * t) / (
        1.0 + 8.42420235e-4 * t + 7.08145163e-7 * t * t
    )
    vl = (0.317398726 + 4.22806245e-5 * t + 4.20481691e-8 * t * t) / (
        1.0 - 2.89741816e-5 * t + 1.61456053e-7 * t * t
    )
    duv = math.hypot(u - ul, v - vl) * (1.0 if v >= vl else -1.0)
    return float(cct), float(duv)


def neutral_to_mired_duv(
    neutral: np.ndarray, rgb_xyz: np.ndarray
) -> tuple[float, float] | None:
    """`neutral`: camera RGB of the as-shot grey. `rgb_xyz`: the (3,3) XYZ -> camera matrix."""
    try:
        xyz = np.linalg.inv(rgb_xyz[:3, :3].astype(np.float64)) @ np.asarray(neutral, np.float64)
    except np.linalg.LinAlgError:
        return None
    r = xyz_to_cct_duv(xyz)
    if r is None:
        return None
    return 1e6 / r[0], r[1]


def as_shot(path: str | Path) -> tuple[float, float] | None:
    """(as-shot mired, Delta-uv) of a RAW/DNG, or None when the file has no usable metadata."""
    p = Path(path)
    try:
        import rawpy

        with rawpy.imread(str(p)) as raw:
            m = np.asarray(raw.rgb_xyz_matrix, dtype=np.float64)
            if not np.any(m[:3, :3]):
                return None
            neutral = None
            if p.suffix.lower() == ".dng":
                neutral = dng_xmp.as_shot_neutral(p)
            if neutral is None:
                cwb = np.asarray(raw.camera_whitebalance, dtype=np.float64)[:3]
                if not np.all(cwb > 0):
                    return None
                neutral = 1.0 / cwb
            return neutral_to_mired_duv(np.asarray(neutral, np.float64), m)
    except Exception:  # noqa: BLE001 - unreadable / unsupported file: no WB feature
        return None
