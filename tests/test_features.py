import math

import cv2
import numpy as np
import pytest

from shotsieve.features import exposure, expression, eyes, sharpness, subject
from shotsieve.features.flash import FlashInput, detect_flash
from shotsieve.features.record import (
    RawFeatures,
    ShootNorm,
    build_vector,
    compute_norm,
    group_norms,
)
from shotsieve.features.subject import FaceObs
from shotsieve.features.vector import FEATURE_NAMES, SCHEMA_VERSION, FeatureVector, stack

# ---- vector -----------------------------------------------------------------------------


def test_vector_order_and_roundtrip():
    assert FEATURE_NAMES[:3] == ("sharp_global", "sharp_global_norm", "tenengrad_norm")
    assert FEATURE_NAMES[22:26] == ("rank_in_group_norm", "iso_log2", "shutter_log2", "af_known")
    assert FEATURE_NAMES[-5:] == (
        "af_sharp_norm", "af_peak_ratio", "af_on_face", "banding_amp_ev", "banding_coherence"
    )
    assert len(FEATURE_NAMES) == 31 and len(set(FEATURE_NAMES)) == 31
    assert SCHEMA_VERSION == 2
    fv = FeatureVector(sharp_global=3.0, n_faces=2, shutter_log2=-7.0)
    arr = fv.to_array()
    assert arr.shape == (31,) and arr.dtype == np.float32
    assert arr[0] == 3.0 and arr[FEATURE_NAMES.index("n_faces")] == 2
    back = FeatureVector.from_array(arr)
    assert back.shutter_log2 == -7.0 and math.isnan(back.exp_mean)
    assert FeatureVector.from_dict(fv.to_dict()).n_faces == 2
    with pytest.raises(ValueError):
        FeatureVector.from_array(np.zeros(3))
    assert stack([fv, fv]).shape == (2, 31) and stack([]).shape == (0, 31)


# ---- sharpness / exposure --------------------------------------------------------------


def test_sharpness_blur_lowers_both_metrics(rgb):
    img = rgb("lena.jpg")
    blurred = cv2.GaussianBlur(img, (0, 0), 6)
    a = sharpness.global_sharpness(sharpness.to_luma(img))
    b = sharpness.global_sharpness(sharpness.to_luma(blurred))
    assert b.lap_var < a.lap_var * 0.1 and b.tenengrad < a.tenengrad * 0.3


def test_sharpness_is_roughly_scale_invariant(rgb):
    img = rgb("building.jpg")
    big = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    a = sharpness.global_sharpness(sharpness.to_luma(img)).lap_var
    b = sharpness.global_sharpness(sharpness.to_luma(big)).lap_var
    assert 0.2 < b / a < 5


def test_face_sharpness_and_normalise(rgb):
    gray = sharpness.to_luma(rgb("lena.jpg"))
    sharp = sharpness.face_sharpness(gray, (150, 150, 200, 200))
    soft = sharpness.face_sharpness(cv2.GaussianBlur(gray, (0, 0), 5), (150, 150, 200, 200))
    assert soft < sharp
    assert math.isnan(sharpness.face_sharpness(gray, (0, 0, 2, 2)))
    assert sharpness.normalise(50, 100) == 0.5 and sharpness.normalise(500, 100) == 1.0
    assert math.isnan(sharpness.normalise(float("nan"), 100))
    assert sharpness.shoot_reference([], 30.0) == 30.0
    assert sharpness.shoot_reference([1, 2, 3], 30.0) == 30.0
    assert sharpness.shoot_reference(list(range(100, 200)), 30.0) > 180


def test_exposure_dark_and_clipped():
    mid = np.full((64, 64), 118, np.uint8)
    assert abs(exposure.analyse(mid).ev) < 0.1
    dark = (mid * 0.1).astype(np.uint8)
    ex = exposure.analyse(dark)
    assert ex.ev < -2 and ex.mean < 15
    blown = np.full((64, 64), 255, np.uint8)
    ex = exposure.analyse(blown)
    assert ex.clip_hi == 1.0 and ex.clip_lo == 0.0 and ex.ev > 2
    black = np.zeros((32, 32), np.uint8)
    assert exposure.analyse(black).clip_lo == 1.0


def test_exposure_real_fixture(rgb):
    from shotsieve.features.sharpness import to_luma

    img = rgb("lena.jpg")
    dark = np.clip((img.astype(np.float32) / 255) ** 2.2 * 2**-3, 0, 1) ** (1 / 2.2) * 255
    ev_ok = exposure.analyse(to_luma(img)).ev
    ev_dark = exposure.analyse(to_luma(dark.astype(np.uint8))).ev
    assert ev_dark < ev_ok - 2.5 and ev_dark < -2


# ---- eyes / expression / subject --------------------------------------------------------


def test_eyes_logic():
    assert eyes.closed_score(0.2, 0.7) == 0.7
    assert math.isnan(eyes.closed_score(float("nan"), 0.7))
    assert eyes.is_closed(0.6, 0.1)
    assert eyes.is_closed(0.5, 0.5)  # both > 0.45
    assert not eyes.is_closed(0.5, 0.2)
    assert not eyes.is_closed(float("nan"), float("nan"))  # unknown is never 'closed'


def test_expression():
    assert expression.smile_score(1.4) == 1.0 and expression.smile_score(-1) == 0.0
    assert math.isnan(expression.smile_score(float("nan")))
    assert expression.neutral_score(0.0) == 1.0 and expression.neutral_score(0.8) == pytest.approx(
        0.2
    )


def test_subject_importance_rules():
    faces = [
        FaceObs((0, 0, 200, 200), 0.9),
        FaceObs((300, 0, 60, 60), 0.9),  # 9% of the largest -> unimportant
        FaceObs((400, 0, 150, 150), 0.9),
        FaceObs((600, 0, 8, 8), 0.9),  # too small for the image
    ]
    out = subject.mark_important(faces, 1280, 853)
    assert [f.important for f in out] == [True, True, False, False]
    assert out[0].area >= out[1].area
    many = [FaceObs((i * 50, 0, 40, 40), 0.9) for i in range(10)]
    assert len(subject.mark_important(many, 1280, 853)) == 6
    assert subject.mark_important([], 10, 10) == []
    assert subject.centrality((590, 400, 100, 53), 1280, 853) > 0.95
    assert subject.centrality((0, 0, 10, 10), 1280, 853) < 0.1
    assert subject.subject_score(0, 0, 0) < subject.subject_score(1, 0.1, 0.9)
    d = out[0].to_dict()
    assert FaceObs.from_dict(d).bbox == out[0].bbox


# ---- record / vector assembly -----------------------------------------------------------


def _raw(lap=100.0, faces=None):
    return RawFeatures(
        width=1000, height=800, lap_var=lap, tenengrad=500, mean_luma=120, lin_mean=0.18,
        clip_hi=0.0, clip_lo=0.0, ev=0.0, p98=200, faces=faces or [],
    )  # fmt: skip


def test_raw_json_roundtrip_with_nan():
    f = FaceObs((1, 2, 3, 4), 0.9, sharp=12.0, important=True)
    raw = _raw(faces=[f])
    back = RawFeatures.from_json(raw.to_json())
    assert back.faces[0].bbox == (1, 2, 3, 4) and math.isnan(back.faces[0].blink_left)
    assert back.lap_var == 100.0


def test_build_vector_faces_and_nan_eyes():
    big = FaceObs((400, 300, 200, 200), 0.95, sharp=80, blink_left=0.8, blink_right=0.7,
                  smile=0.4, found=True, important=True)  # fmt: skip
    nolm = FaceObs((100, 300, 180, 180), 0.9, sharp=60, important=True)  # landmarks not found
    raw = _raw(faces=[big, nolm])
    norm = ShootNorm(100.0, 1000.0, 100.0)
    fv = build_vector(raw, norm, flash_fired=True, iso=400, exposure_time=1 / 125)
    assert fv.n_faces == 2 and fv.main_face_area == pytest.approx(0.05)
    assert fv.eyes_closed_main == 0.8 and fv.n_eyes_closed_important == 1
    assert fv.eyes_closed_max_important == 0.8  # NaN face ignored, not treated as open/closed
    assert fv.face_sharp_norm_main == 0.8 and fv.face_sharp_min_important == 0.6
    assert fv.flash_fired == 1.0 and fv.iso_log2 == pytest.approx(math.log2(400))
    assert fv.shutter_log2 == pytest.approx(-math.log2(125))
    empty = build_vector(_raw(), norm)
    assert empty.n_faces == 0 and math.isnan(empty.eyes_closed_main) and math.isnan(empty.iso_log2)


def test_group_norms_use_sharpest_sibling_with_floor():
    recs = [_raw(10.0), _raw(40.0), _raw(400.0), _raw(5.0)]
    norm = compute_norm(recs)
    out = group_norms(recs, [0, 0, 1, 2], norm)
    # group 0: sibling max 40 is below 0.25*p90 -> floored; singletons keep the shoot norm
    assert out[0].lap_ref == out[1].lap_ref >= 0.25 * norm.lap_ref
    assert out[2] == norm and out[3] == norm


# ---- flash ------------------------------------------------------------------------------

LIN = (118 / 255) ** 2.2


def _burst(fired, lums, t0=0.0, cam="c1"):
    return [FlashInput(cam, t0 + 0.3 * i, fired, lum) for i, lum in enumerate(lums)]


def test_flash_misfire_flagged_and_normal_not():
    items = _burst(True, [LIN, LIN, LIN * 0.25, LIN, LIN])  # frame 2 is 2 EV darker
    res = detect_flash(items)
    assert [r.misfire for r in res] == [False, False, True, False, False]
    assert res[2].delta_ev == pytest.approx(-2.0)


def test_flash_not_fired_in_flash_burst_dark():
    items = _burst(True, [LIN] * 4) + [FlashInput("c1", 1.0, False, LIN * 0.3)]
    res = detect_flash(items)
    assert res[-1].misfire is True
    ok = _burst(True, [LIN] * 4) + [FlashInput("c1", 1.0, False, LIN * 0.9)]
    assert detect_flash(ok)[-1].misfire is False


def test_flash_needs_two_neighbours_and_info():
    two = detect_flash(_burst(True, [LIN, LIN * 0.1]))
    assert [r.misfire for r in two] == [None, None] and math.isnan(two[0].delta_ev)
    far = _burst(True, [LIN, LIN, LIN * 0.1], t0=0) + [FlashInput("c1", 500.0, True, LIN)]
    assert detect_flash(far)[-1].misfire is None
    unknown = detect_flash(_burst(None, [LIN, LIN, LIN * 0.1, LIN]))
    assert all(r.misfire is None for r in unknown)
    assert not math.isnan(unknown[2].delta_ev)


def test_flash_no_flash_scene_never_flagged_and_cameras_separate():
    dark = _burst(False, [LIN, LIN * 0.1, LIN, LIN])  # nobody fired: a dark frame is not a misfire
    assert [r.misfire for r in detect_flash(dark)] == [False] * 4
    mixed = _burst(True, [LIN] * 3, cam="a") + _burst(True, [LIN * 0.1], cam="b")
    assert detect_flash(mixed)[-1].misfire is None  # other camera's frames are not neighbours
    unusable = [FlashInput("a", 0, True, LIN, usable=False)] * 4
    assert all(r.misfire is None for r in detect_flash(unusable))
