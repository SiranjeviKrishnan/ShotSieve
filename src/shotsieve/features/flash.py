"""Flash-misfire detection: EXIF flash flag versus luminance of time-window neighbours."""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

import numpy as np

NAN = float("nan")
MIN_NEIGHBOURS = 2


@dataclass(frozen=True)
class FlashInput:
    camera: str
    timestamp: float
    flash_fired: bool | None
    lin_mean: float  # mean of (L/255)^2.2
    usable: bool = True  # False when the timestamp is not trustworthy (file mtime)


@dataclass(frozen=True)
class FlashResult:
    delta_ev: float  # EV vs the reference neighbours (NaN when undecidable)
    misfire: bool | None  # None = no decision (too few neighbours / no flash info)


def _ev(lin: float, ref: float) -> float:
    return math.log2(max(lin, 1e-6) / max(ref, 1e-6))


def detect_flash(
    items: list[FlashInput],
    window_s: float = 10.0,
    dark_ev_nofire: float = 1.3,
    dark_ev_fired: float = 1.5,
) -> list[FlashResult]:
    """Per-image flash result, aligned with `items`. O(n * neighbours) per camera."""
    out = [FlashResult(NAN, None)] * len(items)
    by_cam: dict[str, list[int]] = {}
    for i, it in enumerate(items):
        if it.usable:
            by_cam.setdefault(it.camera, []).append(i)
    for idxs in by_cam.values():
        idxs.sort(key=lambda i: items[i].timestamp)
        ts = [items[i].timestamp for i in idxs]
        for pos, i in enumerate(idxs):
            lo = bisect.bisect_left(ts, ts[pos] - window_s)
            hi = bisect.bisect_right(ts, ts[pos] + window_s)
            nb = [idxs[k] for k in range(lo, hi) if k != pos]
            if len(nb) < MIN_NEIGHBOURS:
                continue
            me = items[i]
            fired_nb = [j for j in nb if items[j].flash_fired is True]
            all_med = float(np.median([items[j].lin_mean for j in nb]))
            if me.flash_fired is True:
                if len(fired_nb) < MIN_NEIGHBOURS:
                    out[i] = FlashResult(_ev(me.lin_mean, all_med), None)
                    continue
                d = _ev(me.lin_mean, float(np.median([items[j].lin_mean for j in fired_nb])))
                out[i] = FlashResult(d, d <= -dark_ev_fired)
            elif me.flash_fired is False:
                if len(fired_nb) / len(nb) >= 0.5:
                    d = _ev(me.lin_mean, float(np.median([items[j].lin_mean for j in fired_nb])))
                    out[i] = FlashResult(d, d <= -dark_ev_nofire)
                else:
                    out[i] = FlashResult(_ev(me.lin_mean, all_med), False)
            else:
                out[i] = FlashResult(_ev(me.lin_mean, all_med), None)
    return out
