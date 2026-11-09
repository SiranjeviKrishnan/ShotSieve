"""Electronic-shutter / light-flicker banding: horizontal brightness bands that are coherent
across the frame width and periodic along the sensor's readout direction.

`measure(gray, orientation, exposure_time)` takes the upright preview luma. Steps (1-2 ms):
  1. skip (NaN) when the shutter is >= 1/50 s (flicker integrates away) or unknown;
  2. rotate back to sensor orientation so readout bands are rows, split into 8 vertical strips,
     take each strip's row-mean of log2(luma + 1) and detrend with a gaussian (sigma = H / 25);
     rows where a strip is > 30 % clipped are masked;
  3. the banding frequency is the in-band FFT bin at which the 8 strips move most in phase;
     coherence = mean pairwise Pearson r of the strips' narrowband profiles there; amp_ev =
     robust std (1.4826 MAD) of the strip-median narrowband profile (EV, gain-corrected for the
     detrend);
  4. periodicity: share of the in-band power of the mean profile (periods H/60 .. H/3) at that
     peak.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

MAX_SHUTTER_S = 1 / 50
N_STRIPS = 8
COLS = 64  # working width (the strip means are an area average anyway)
SIGMA_DIV = 25.0
CLIP_FRAC = 0.30
MIN_VALID_STRIPS = 5  # a row needs this many unclipped strips
# Flicker (100/120 Hz) read out by a rolling shutter makes a handful of broad bands per frame
# (readout time x flicker rate: <= ~12). Scene stripes (blinds, fences, textiles, shirts) are
# finer; they stay a soft review flag and are never rejected on a banding claim.
FLICKER_K_MAX = 12
HARMONIC_MAX = 0.08  # harmonic/fundamental power above this: scene stripes, not flicker
MAX_GAIN_CORRECTION = 3.0  # the detrend attenuates long periods; amplitude is corrected, capped
# flag thresholds. Deviation from the spec (0.6 / 0.04): amp 0.05 separates the a = 0.03 sine
# (which 0.04 passes); coherence 0.5 because it is measured narrowband at the banding frequency
COHERENCE_MIN = 0.5
PEAK_FRAC_MIN = 0.25
AMP_EV_MIN = 0.05
AMP_EV_SEVERE = 0.12
NAN = float("nan")


@dataclass(frozen=True)
class Banding:
    coherence: float = NAN
    amp_ev: float = NAN
    peak_frac: float = NAN
    k: float = NAN  # bands per frame height at the banding frequency


def _table() -> np.ndarray:
    return np.log2(np.arange(256, dtype=np.float32) + 1.0)


_LOG2 = _table()


def measure(gray: np.ndarray, orientation: int = 1, exposure_time: float | None = None) -> Banding:
    if exposure_time is None or not exposure_time > 0 or exposure_time >= MAX_SHUTTER_S:
        return Banding()
    g = gray.T if orientation in (5, 6, 7, 8) else gray  # sensor rows = readout direction
    h = g.shape[0]
    if h < 120:
        return Banding()
    small = cv2.resize(g, (COLS, h), interpolation=cv2.INTER_AREA)
    per = COLS // N_STRIPS
    strips = _LOG2[small].reshape(h, N_STRIPS, per).mean(axis=2)  # (H, 8) row-mean of log2
    clipped = ((small >= 250) | (small <= 5)).reshape(h, N_STRIPS, per).mean(axis=2)
    ok = clipped <= CLIP_FRAC  # per-strip mask: a clipped strip drops out, the row stays
    valid = ok.sum(axis=1) >= MIN_VALID_STRIPS
    if valid.sum() < 0.5 * h:
        return Banding()
    ok &= valid[:, None]
    w = ok.astype(np.float32)
    sigma = h / SIGMA_DIV

    def blur(a: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(a, (0, 0), sigmaX=0.01, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)

    # normalised convolution: masked samples neither contribute to nor bias the trend
    trend = blur(strips * w) / np.maximum(blur(w), 1e-3)
    d = np.where(ok, strips - trend, 0.0).astype(np.float32)  # masked samples carry no signal
    spec = np.fft.rfft(d, axis=0)  # (K, 8): one complex amplitude per strip and frequency
    power = np.abs(spec) ** 2
    k = np.arange(spec.shape[0])
    band = (k >= 3) & (k <= 60)  # k = H / period: periods H/60 .. H/3
    total = power.sum(axis=1)
    if not band.any() or total[band].max() <= 0:
        return Banding(0.0, 0.0, 0.0)
    # banding is the one frequency at which all 8 strips move in phase (scene detail is not)
    phase = np.abs(spec.sum(axis=1)) ** 2 / (N_STRIPS * total + 1e-12)
    # ...weighted by its power, so a noise bin that happens to be in phase cannot win
    cand = np.where(band & (total >= np.median(total[band])), phase * np.sqrt(total), -1.0)
    kp = int(np.argmax(cand))
    sel = slice(max(kp - 1, 0), kp + 2)
    nb = np.zeros_like(spec)
    nb[sel] = spec[sel]
    prof = np.fft.irfft(nb, n=h, axis=0)  # narrowband strip profiles
    if valid.sum() < 32:
        return Banding(0.0, 0.0, 0.0)
    # coherence: mean pairwise Pearson r over the rows where both strips are usable
    rs = []
    for i in range(N_STRIPS):
        for j in range(i + 1, N_STRIPS):
            m = ok[:, i] & ok[:, j]
            if m.sum() < 32:
                continue
            x, y = prof[m, i], prof[m, j]
            if x.std() < 1e-9 or y.std() < 1e-9:
                rs.append(0.0)
            else:
                rs.append(float(np.corrcoef(x, y)[0, 1]))
    if len(rs) < 6:
        return Banding(0.0, 0.0, 0.0)
    coherence = float(np.mean(rs))
    pm = np.where(ok, prof, np.nan)[valid]
    med = np.nanmedian(pm, axis=1)
    amp = float(1.4826 * np.median(np.abs(med - np.median(med))))
    period = h / max(kp, 1)  # the detrend attenuates long periods: undo it (capped)
    gain = 1.0 - math.exp(-2.0 * math.pi**2 * sigma**2 / period**2)
    amp /= max(gain, 1.0 / MAX_GAIN_CORRECTION)
    mean_power = np.abs(spec.sum(axis=1)) ** 2
    peak_frac = float(mean_power[sel].sum() / max(mean_power[band].sum(), 1e-12))
    # finer coherent stripes than any flicker can draw (k > 60: blinds, fences, textiles) are
    # scene texture: the band stays a soft flag, but its frequency is not flicker-like
    fine = k > 60
    texture = bool(fine.any() and mean_power[fine].max() >= mean_power[sel].max())
    # ...and so are non-sinusoidal stripes (stairs, bleachers, louvres: sawtooth / square waves):
    # sharp edges put real power on the 2nd/3rd harmonic, a flicker band is nearly a pure sine
    if not texture and kp >= 3:
        fund = mean_power[sel].sum()
        harm = 0.0
        for m in (2, 3):
            lo, hi = max(m * kp - 1, 0), m * kp + 2
            if lo < len(mean_power):
                harm += mean_power[lo:hi].sum()
        texture = bool(harm >= HARMONIC_MAX * fund)
    return Banding(coherence, amp, min(1.0, peak_frac), NAN if texture else float(kp))


def is_banded(b: Banding) -> bool:
    return (
        b.coherence >= COHERENCE_MIN and b.peak_frac >= PEAK_FRAC_MIN and b.amp_ev >= AMP_EV_MIN
    )


def flags(peak_frac: float, k: float) -> list[str]:
    out = []
    if peak_frac == peak_frac and peak_frac >= PEAK_FRAC_MIN:
        out.append("banding_periodic")
        if is_flicker_like(k):
            out.append("banding_flicker")
    return out


def is_flicker_like(k: float) -> bool:
    """Band frequency in the range a 100/120 Hz flicker read out by a rolling shutter produces."""
    return k == k and 0 < k <= FLICKER_K_MAX
