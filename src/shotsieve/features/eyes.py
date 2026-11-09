"""Eyes-closed logic over MediaPipe blink blendshapes."""

from __future__ import annotations

import math


def closed_score(blink_left: float, blink_right: float) -> float:
    """max(blinkL, blinkR); NaN if either is NaN (landmarks not found)."""
    if math.isnan(blink_left) or math.isnan(blink_right):
        return float("nan")
    return max(blink_left, blink_right)


def is_closed(
    blink_left: float, blink_right: float, thresh: float = 0.55, both_thresh: float = 0.45
) -> bool:
    """Closed if max(blink) > thresh, or both eyes > both_thresh. NaN is never 'closed'."""
    if math.isnan(blink_left) or math.isnan(blink_right):
        return False
    return max(blink_left, blink_right) > thresh or min(blink_left, blink_right) > both_thresh
