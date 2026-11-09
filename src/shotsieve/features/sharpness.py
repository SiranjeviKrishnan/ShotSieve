"""Sharpness: variance of Laplacian and Tenengrad on normalised luma, plus face-ROI sharpness."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

GLOBAL_EDGE = 1024
# Scale-invariant face focus: every face is measured at one canonical pixel scale (area-downscaled
# to FACE_EDGE, never upscaled), and faces smaller than MIN_FACE_PX are not judged at all
# (NaN) because their Laplacian variance mostly reflects resolution, not focus.
FACE_EDGE = 96
MIN_FACE_PX = 96
# Absolute floors for the per-shoot reference (prevents blow-ups on featureless shoots).
LAP_FLOOR = 30.0
# ... and a cap on the global reference. A shoot-wide p90 is set by its most textured frames; a
# frame of smooth content (studio backdrop, sky, bokeh) is then "soft" relative to it although it
# is perfectly sharp. Measured: genuinely blurred frames have a 1024-px Laplacian variance below
# ~35, clean smooth-background frames above ~70, so a reference beyond LAP_REF_CAP only produces
# false blur verdicts.
LAP_REF_CAP = 300.0
TENEN_FLOOR = 150.0
FACE_FLOOR = 30.0


def to_luma(img_rgb: np.ndarray) -> np.ndarray:
    """uint8 luma (BT.601) of an RGB image."""
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)


def _resize_long_edge(gray: np.ndarray, edge: int) -> np.ndarray:
    h, w = gray.shape[:2]
    s = edge / max(h, w)
    if abs(s - 1.0) < 1e-3:
        return gray
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    return cv2.resize(gray, (max(1, round(w * s)), max(1, round(h * s))), interpolation=interp)


def lap_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_32F, ksize=3).var())


def tenengrad(gray: np.ndarray) -> float:
    """Mean squared Sobel gradient magnitude."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(gx * gx + gy * gy)) / 16.0


@dataclass(frozen=True)
class GlobalSharpness:
    lap_var: float
    tenengrad: float


def global_sharpness(gray: np.ndarray) -> GlobalSharpness:
    """Scale-invariant global sharpness (luma resized to a 1024 px long edge)."""
    g = _resize_long_edge(gray, GLOBAL_EDGE)
    return GlobalSharpness(lap_var(g), tenengrad(g))


def face_sharpness(gray: np.ndarray, bbox: tuple[float, float, float, float]) -> float:
    """Scale-invariant Laplacian variance of the face crop.

    The crop is area-downscaled to FACE_EDGE px (long edge) so a 400 px face and a 120 px face
    are compared at the same scale. Faces whose long edge is below MIN_FACE_PX are too small to
    judge and give NaN (callers then send the frame to review instead of rejecting it). The crop
    is never upscaled: interpolation would remove the very detail the measure looks for.
    """
    h, w = gray.shape[:2]
    x, y, bw, bh = bbox
    x0, y0 = max(0, int(round(x))), max(0, int(round(y)))
    x1, y1 = min(w, int(round(x + bw))), min(h, int(round(y + bh)))
    if max(x1 - x0, y1 - y0) < MIN_FACE_PX or min(x1 - x0, y1 - y0) < 8:
        return float("nan")
    crop = np.ascontiguousarray(gray[y0:y1, x0:x1])
    if max(crop.shape[:2]) > FACE_EDGE:
        crop = _resize_long_edge(crop, FACE_EDGE)
    return lap_var(crop)


def region_sharpness(gray: np.ndarray, box: tuple[float, float, float, float]) -> float:
    """The face measure applied to any region (T23h subject mode): the same canonical scale
    (long edge area-downscaled to FACE_EDGE, never upscaled) and Laplacian variance, but small
    regions are measured too. `box` = normalised x0, y0, x1, y1. NaN for an empty region."""
    h, w = gray.shape[:2]
    x0, y0 = max(0, int(box[0] * w)), max(0, int(box[1] * h))
    x1, y1 = min(w, int(round(box[2] * w))), min(h, int(round(box[3] * h)))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return float("nan")
    crop = np.ascontiguousarray(gray[y0:y1, x0:x1])
    if max(crop.shape[:2]) > FACE_EDGE:
        crop = _resize_long_edge(crop, FACE_EDGE)
    return lap_var(crop)


def face_luma(gray: np.ndarray, bbox: tuple[float, float, float, float]) -> float:
    """Mean luma (0-255) of the face crop, NaN for an empty crop."""
    h, w = gray.shape[:2]
    x, y, bw, bh = bbox
    x0, y0 = max(0, int(round(x))), max(0, int(round(y)))
    x1, y1 = min(w, int(round(x + bw))), min(h, int(round(y + bh)))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return float("nan")
    return float(gray[y0:y1, x0:x1].mean())


def shoot_reference(
    values: list[float], floor: float, pct: float = 90.0, cap: float | None = None
) -> float:
    """Robust per-shoot reference: p90 of the finite values, never below `floor`."""
    arr = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return floor
    ref = float(max(np.percentile(arr, pct), floor))
    return min(ref, cap) if cap is not None else ref


def normalise(value: float, ref: float) -> float:
    """clip(value / ref, 0, 1); NaN stays NaN."""
    if value is None or not np.isfinite(value):
        return float("nan")
    return float(np.clip(value / ref, 0.0, 1.0))
