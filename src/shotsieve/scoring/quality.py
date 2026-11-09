"""Rule-based quality prior: sub-scores in [0, 1], a weighted score and hard-fail rules."""

from __future__ import annotations

import math
from collections.abc import Collection
from dataclasses import dataclass, field

from shotsieve.config import Settings
from shotsieve.features import banding
from shotsieve.features.subject import subject_score
from shotsieve.features.vector import FeatureVector

UNKNOWN_EYES = 0.7
WEIGHTS = {"sharp": 0.35, "eyes": 0.30, "exposure": 0.15, "subject": 0.10, "expression": 0.10}


def _nz(x: float, default: float) -> float:
    return default if x is None or math.isnan(x) else x


def _clip(x: float) -> float:
    return min(1.0, max(0.0, x))


def sub_scores(fv: FeatureVector) -> dict[str, float]:
    has_face = fv.n_faces > 0 and not math.isnan(fv.face_sharp_norm_main)
    if has_face:
        mn = _nz(fv.face_sharp_min_important, fv.face_sharp_norm_main)
        sharp = 0.7 * fv.face_sharp_norm_main + 0.3 * mn
    else:
        sharp = _nz(fv.sharp_global_norm, 0.0)
    if fv.n_faces > 0 and not math.isnan(fv.eyes_closed_max_important):
        eyes = 1.0 - _clip((fv.eyes_closed_max_important - 0.25) / 0.5)
    elif fv.n_faces > 0:
        eyes = UNKNOWN_EYES  # landmarks missing: unknown is not "open"
    else:
        eyes = 1.0
    ev_pen = max(0.0, abs(_nz(fv.exp_ev, 0.0)) - 0.5) / 2.0
    clip_pen = (
        max(0.0, _nz(fv.exp_clip_hi, 0.0) - 0.05) / 0.2
        + max(0.0, _nz(fv.exp_clip_lo, 0.0) - 0.05) / 0.3
    )
    exposure = _clip(1.0 - ev_pen - clip_pen)
    expression = 0.5 + 0.5 * _clip(_nz(fv.smile_main, 0.0))
    subject = subject_score(
        int(fv.n_faces), _nz(fv.main_face_area, 0.0), _nz(fv.main_face_centrality, 0.0)
    )
    return {
        "sharp": _clip(sharp),
        "eyes": eyes,
        "exposure": exposure,
        "subject": subject,
        "expression": expression,
    }


CERTAIN, FIRM, SOFT = "certain", "firm", "soft"
# Severity of each hard-fail tier:
#   certain - never overridden (black frame, extreme blur, flash misfire, deep under-exposure)
#   firm    - rejected unless a trained personal model (or a calibration margin) says keep
#   soft    - a single closed eye, high-key exposure, small-face focus: goes to review as soon as
#             the keep-probability is respectable; the personal model and calibration may override
EXTREME_BLUR = 0.08
SMALL_FACE_AREA = 0.03  # main face below this share of the frame: focus verdict is only "soft"
WELL_EXPOSED_LUMA = (40.0, 220.0)
FOCUS_FAILS = ("blur", "face_out_of_focus", "af_missed")
AF_SHARP_MAX = 0.35  # af_missed: AF-ROI sharpness below this (shoot-normalised) ...
AF_PEAK_RATIO_MIN = 2.5  # ... while some other tile is this much sharper


@dataclass
class QualityResult:
    score: float
    subs: dict[str, float] = field(default_factory=dict)
    hard_fails: list[str] = field(default_factory=list)
    tiers: dict[str, str] = field(default_factory=dict)  # hard-fail code -> tier
    flicker: bool = False  # banding is flicker-like (broad, periodic); only then may it reject

    @property
    def certain(self) -> bool:
        return any(t == CERTAIN for t in self.tiers.values())

    @property
    def tier(self) -> str:
        """The strictest tier among the hard fails ('' if none)."""
        ts = set(self.tiers.values())
        for t in (CERTAIN, FIRM, SOFT):
            if t in ts:
                return t
        return ""


def classify_fails(
    fv: FeatureVector, settings: Settings, flags: Collection[str] = ()
) -> dict[str, str]:
    """Reject rules with their tier. `flags` carries context-derived conditions:
    'flash_misfire', 'eyes_single' (main face has exactly one closed eye) and
    'subject_well_exposed' (main-face luma inside [40, 220])."""
    has_face = fv.n_faces > 0
    if not has_face and _nz(fv.exp_mean, 255.0) < 8.0:
        # night sky / astro / fireworks: a long exposure is deliberate, so only ask
        long_exposure = not math.isnan(fv.shutter_log2) and fv.shutter_log2 >= 0.0
        return {"accidental": SOFT if long_exposure else CERTAIN}
    fails: dict[str, str] = {}
    gn, fn = fv.sharp_global_norm, fv.face_sharp_norm_main
    face_known = has_face and not math.isnan(fn)
    small_face = _nz(fv.main_face_area, 0.0) < SMALL_FACE_AREA
    if not math.isnan(gn) and gn < 0.15 and (not face_known or fn < 0.25):
        if gn < EXTREME_BLUR and not (has_face and not face_known):
            fails["blur"] = CERTAIN
        else:  # a face too small to judge may sit on a legitimately smooth backdrop
            fails["blur"] = SOFT if has_face and not face_known else FIRM
    if face_known and fn < 0.20 and not math.isnan(gn) and gn >= 0.35:
        fails["face_out_of_focus"] = SOFT if small_face else FIRM
    if (
        fv.af_known >= 1.0
        and fv.af_on_face < 1.0
        and _nz(fv.af_sharp_norm, 1.0) < AF_SHARP_MAX
        and _nz(fv.af_peak_ratio, 0.0) >= AF_PEAK_RATIO_MIN
        and "af_peak_adjacent" not in flags
    ):
        fails["af_missed"] = SOFT  # never CERTAIN; face sharpness covers the on-face case
    if (
        _nz(fv.banding_coherence, 0.0) >= banding.COHERENCE_MIN
        and _nz(fv.banding_amp_ev, 0.0) >= banding.AMP_EV_MIN
        and "banding_periodic" in flags
    ):
        fails["banding"] = SOFT  # review; severe + clean alternative is decided in selection
    if has_face:
        main_closed = _nz(fv.eyes_closed_main, 0.0) > settings.eyes_closed_thresh
        majority = fv.n_eyes_closed_important >= 0.5 * fv.n_faces and fv.n_eyes_closed_important
        if main_closed or majority:
            single = "eyes_single" in flags and not majority
            fails["eyes_closed"] = SOFT if single else FIRM
    ev, hi, lo = _nz(fv.exp_ev, 0.0), _nz(fv.exp_clip_hi, 0.0), _nz(fv.exp_clip_lo, 0.0)
    dark = ev < -2.5 or lo > 0.40
    if ev > 2.5 or hi > 0.25:
        # high-key / white backdrop / snow: fine when the subject itself is well exposed
        if not ("subject_well_exposed" in flags and ev <= 3.5):
            fails["exposure_bad"] = CERTAIN if ev > 3.5 else SOFT
    elif dark and hi < 0.01:
        fails["exposure_bad"] = CERTAIN
    if "flash_misfire" in flags:
        fails["flash_misfire"] = CERTAIN
    if "exposure_bad" in fails and (ev < -2.5 or "flash_misfire" in flags):
        # sharpness is unreliable on an under-exposed frame: report the cause, not the symptom
        fails = {k: t for k, t in fails.items() if k not in FOCUS_FAILS}
    elif "flash_misfire" in flags:
        fails = {k: t for k, t in fails.items() if k not in FOCUS_FAILS}
    return fails


def hard_fails(fv: FeatureVector, settings: Settings, flags: Collection[str] = ()) -> list[str]:
    """Reject rule codes that fire (see `classify_fails` for their tiers)."""
    return list(classify_fails(fv, settings, flags))


def assess(fv: FeatureVector, settings: Settings, flags: Collection[str] = ()) -> QualityResult:
    """Prior quality score in [0,1] with sub-scores and hard-fail reasons."""
    subs = sub_scores(fv)
    score = sum(WEIGHTS[k] * subs[k] for k in WEIGHTS)
    tiers = classify_fails(fv, settings, flags)
    return QualityResult(
        _clip(score), subs, list(tiers), tiers,
        flicker="banding" in tiers and "banding_flicker" in flags,
    )


def weakest_area(subs: dict[str, float]) -> str:
    names = {
        "sharp": "sharpness",
        "eyes": "eyes",
        "exposure": "exposure",
        "subject": "subject",
        "expression": "expression",
    }
    key = min(WEIGHTS, key=lambda k: (subs[k] - 1.0) * WEIGHTS[k])
    return names[key]
