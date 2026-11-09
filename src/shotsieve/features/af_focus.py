"""AF-point focus check: is the frame sharp where the camera says it focused?

`measure(gray, boxes)` works on the upright preview luma; `boxes` are normalised
(cx, cy, w, h) boxes already mapped to the preview orientation. It returns the raw Laplacian
variance in the union of the (padded) AF boxes, the ratio of the sharpest 8x8 tile to it, and
whether that peak tile touches the AF region.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MIN_BOX_PX = 64  # each box is padded to at least this size at preview scale
GRID = 8
NAN = float("nan")


@dataclass(frozen=True)
class AFMeasure:
    lap: float = NAN  # Laplacian variance inside the AF ROI
    peak_ratio: float = NAN  # max 8x8-tile variance / lap
    peak_adjacent: bool = False  # the peak tile is the AF tile or a neighbour of it


def roi_mask(shape: tuple[int, int], boxes: list[tuple[float, float, float, float]]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), bool)
    for cx, cy, bw, bh in boxes:
        pw, ph = max(bw * w, MIN_BOX_PX), max(bh * h, MIN_BOX_PX)
        x0, x1 = int(round(cx * w - pw / 2)), int(round(cx * w + pw / 2))
        y0, y1 = int(round(cy * h - ph / 2)), int(round(cy * h + ph / 2))
        mask[max(0, y0) : min(h, y1), max(0, x0) : min(w, x1)] = True
    return mask


def measure(gray: np.ndarray, boxes: list[tuple[float, float, float, float]]) -> AFMeasure:
    if not boxes:
        return AFMeasure()
    h, w = gray.shape[:2]
    mask = roi_mask((h, w), boxes)
    if mask.sum() < 16:
        return AFMeasure()
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    roi = float(lap[mask].var())
    th, tw = h // GRID, w // GRID
    if th < 4 or tw < 4:
        return AFMeasure(roi)
    tiles = lap[: th * GRID, : tw * GRID].reshape(GRID, th, GRID, tw)
    tvar = tiles.var(axis=(1, 3))
    covered = mask[: th * GRID, : tw * GRID].reshape(GRID, th, GRID, tw).mean(axis=(1, 3)) > 0.05
    py, px = np.unravel_index(int(np.argmax(tvar)), tvar.shape)
    ys, xs = np.nonzero(covered)
    adjacent = bool(len(ys) and np.max([abs(ys - py), abs(xs - px)], axis=0).min() <= 1)
    return AFMeasure(roi, float(tvar.max()) / max(roi, 1.0), adjacent)


def on_face(
    boxes: list[tuple[float, float, float, float]],
    faces: list[tuple[float, float, float, float]],
    width: int,
    height: int,
) -> bool:
    """The AF ROI overlaps an important face (IoU >= 0.1, or the AF centre is inside the face).
    `faces` are (x, y, w, h) in preview pixels."""
    for cx, cy, bw, bh in boxes:
        ax0, ay0 = (cx - bw / 2) * width, (cy - bh / 2) * height
        ax1, ay1 = ax0 + bw * width, ay0 + bh * height
        for fx, fy, fw, fh in faces:
            if fx <= cx * width <= fx + fw and fy <= cy * height <= fy + fh:
                return True
            iw = min(ax1, fx + fw) - max(ax0, fx)
            ih = min(ay1, fy + fh) - max(ay0, fy)
            if iw > 0 and ih > 0:
                inter = iw * ih
                union = (ax1 - ax0) * (ay1 - ay0) + fw * fh - inter
                if union > 0 and inter / union >= 0.1:
                    return True
    return False
