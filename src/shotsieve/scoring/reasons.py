"""Reason codes and their plain-language explanations."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from shotsieve.features.vector import FeatureVector

# code -> short label (also used as a stable key in XMP/DB/UI)
REASON_LABELS: dict[str, str] = {
    "blur": "Blurry",
    "face_out_of_focus": "Missed focus on face",
    "af_missed": "Missed AF point",
    "banding": "Light-flicker banding",
    "eyes_closed": "Eyes closed",
    "exposure_bad": "Poor exposure",
    "flash_misfire": "Flash misfire",
    "accidental": "Accidental shot",
    "duplicate": "Duplicate of a better shot",
    "no_subject": "No clear subject",
    "low_quality": "Low overall quality",
    "unreadable": "Could not be read",
    # review-only explanations
    "duplicate_tie": "Near-identical to a better frame",
    "uncertain": "Borderline score",
    "best_available": "Best available frame of this moment",
    "audit": "Random spot-check of rejects",
}
REJECT_CODES = (
    "blur",
    "face_out_of_focus",
    "af_missed",
    "banding",
    "eyes_closed",
    "exposure_bad",
    "flash_misfire",
    "accidental",
    "duplicate",
    "duplicate_tie",
    "no_subject",
    "low_quality",
)


def _f(x: float | None) -> float:
    return float("nan") if x is None else float(x)


def describe(code: str, fv: FeatureVector | None = None, ctx: dict[str, Any] | None = None) -> str:
    """Plain-language sentence for a reason code. `ctx` may hold 'best' (filename of the
    better duplicate) and 'flash_delta_ev'."""
    ctx = ctx or {}
    if code == "blur":
        return (
            "The photo is blurry: the whole frame lacks fine detail, "
            "so it is soft or has motion blur."
        )
    if code == "face_out_of_focus":
        return "Focus missed the face: the background is sharp but the main subject's face is soft."
    if code == "af_missed":
        return "Focus missed the AF point (sharpest area is elsewhere)"
    if code == "banding":
        return "Light-flicker banding (electronic shutter)"
    if code == "eyes_closed":
        n = _f(fv.n_eyes_closed_important) if fv else float("nan")
        if not math.isnan(n) and n >= 2:
            return f"Eyes are closed on {int(n)} people in the photo."
        return "The main subject's eyes are closed (blink)."
    if code == "exposure_bad":
        if fv is not None:
            if fv.exp_ev < -1.0 or fv.exp_clip_lo > 0.4:
                return (
                    f"Too dark: the photo is about {abs(fv.exp_ev):.1f} EV under mid-grey exposure."
                )
            if fv.exp_ev > 1.0 or fv.exp_clip_hi > 0.25:
                return "Too bright: large areas are blown out to pure white."
        return "The exposure is poor (badly under- or over-exposed)."
    if code == "flash_misfire":
        d = ctx.get("flash_delta_ev")
        if d is None and fv is not None:
            d = fv.flash_lum_delta_ev
        extra = f" (about {abs(d):.1f} EV darker than the neighbouring flash shots)" if d else ""
        return f"The flash likely misfired or did not fully fire{extra}."
    if code == "accidental":
        return (
            "Looks like an accidental frame (nearly black, no subject), "
            "e.g. a pocket or lens-cap shot."
        )
    if code == "duplicate":
        best = ctx.get("best")
        if best:
            return f"A better version of this moment exists: {best}."
        return "A better version of this moment exists in the same group."
    if code == "no_subject":
        return "No clear subject was found and the overall quality is low."
    if code == "low_quality":
        weak = ctx.get("weakest")
        tail = f" Weakest area: {weak}." if weak else ""
        return f"Overall quality score is low.{tail}"
    if code == "unreadable":
        return "The file could not be decoded, so it was not analysed."
    if code == "duplicate_tie":
        best = ctx.get("best") or "the group's top pick"
        if ctx.get("best_uncertain"):
            return (
                f"Near-identical to {best}, which is itself borderline; "
                "compare the two and pick one."
            )
        return f"Near-identical to the best frame {best}; kept the sharper/eyes-open one."
    if code == "best_available":
        issue = ctx.get("issue") or "low_quality"
        what = {
            "blur": "it is blurry",
            "face_out_of_focus": "focus missed the face",
            "af_missed": "focus missed the AF point",
            "banding": "it has light-flicker banding",
            "eyes_closed": "eyes are closed",
            "exposure_bad": "the exposure is poor",
            "no_subject": "no clear subject was found",
            "low_quality": "its overall quality score is low",
        }.get(issue, "it has a flaw")
        weak = ctx.get("weakest")
        tail = f" (weakest area: {weak})" if weak and issue in ("low_quality", "no_subject") else ""
        return (
            f"Best available frame of this moment, but {what}{tail}. "
            "Kept for review so the moment is not lost."
        )
    if code == "uncertain":
        return "The quality score is in the borderline range; please decide manually."
    if code == "audit":
        return "Randomly selected from the rejects so your decision can calibrate the model."
    raise KeyError(f"unknown reason code {code!r}")


def describe_all(
    codes: Iterable[str], fv: FeatureVector | None = None, ctx: dict[str, Any] | None = None
) -> list[str]:
    return [describe(c, fv, ctx) for c in codes]
