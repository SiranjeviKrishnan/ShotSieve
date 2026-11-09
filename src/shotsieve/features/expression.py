"""Smile / neutral expression scores."""

from __future__ import annotations

import math


def smile_score(smile: float) -> float:
    """Clamp a smile blendshape average into [0, 1] (NaN stays NaN)."""
    if math.isnan(smile):
        return float("nan")
    return min(1.0, max(0.0, smile))


def neutral_score(smile: float, jaw_open: float = 0.0) -> float:
    """1 - max(smile, jaw_open); a rough 'relaxed/neutral' measure."""
    if math.isnan(smile):
        return float("nan")
    jo = 0.0 if math.isnan(jaw_open) else jaw_open
    return 1.0 - min(1.0, max(smile, jo))
