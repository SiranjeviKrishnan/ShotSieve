"""E-shutter banding: synthetic sine bands, incoherent stripes, blinds texture, rotation, rule."""

import time

import cv2
import numpy as np
import pytest

from shotsieve.config import Settings
from shotsieve.features import banding, sharpness
from shotsieve.features.record import build_vector, compute_norm
from shotsieve.pipeline import stages
from shotsieve.scoring import quality, selection

FAST = 1 / 250


def load(gen_dir, name, size=None):
    im = cv2.cvtColor(cv2.imread(str(gen_dir / "src" / name)), cv2.COLOR_BGR2RGB)
    if size:
        im = cv2.resize(im, size, interpolation=cv2.INTER_CUBIC)
    elif max(im.shape[:2]) > 1280:
        s = 1280 / max(im.shape[:2])
        im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    # mid-tone compression: no clipped highlights/shadows that would swallow the modulation
    return (im.astype(np.float32) * 0.6 + 50).astype(np.uint8)


@pytest.fixture(scope="module")
def clean(gen_dir):
    return load(gen_dir, "lena.jpg", (900, 900))  # preview-sized clean fixture


def banded(img, a, period):
    y = np.arange(img.shape[0], dtype=np.float32)[:, None, None]
    gain = 1 + a * np.sin(2 * np.pi * y / period)
    return np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def meas(img, orient=1, t=FAST):
    return banding.measure(sharpness.to_luma(img), orient, t)


@pytest.mark.parametrize("period", [20, 60, 150])
@pytest.mark.parametrize("a", [0.03, 0.05, 0.1])
def test_sine_bands_flag_from_a_005(clean, a, period):
    b = meas(banded(clean, a, period))
    assert banding.is_banded(b) == (a >= 0.05), b


def test_clean_fixtures_are_not_flagged(gen_dir):
    for name in ("lena.jpg", "building.jpg", "messi5.jpg", "group_rgb.jpg", "portrait2.jpg"):
        assert not banding.is_banded(meas(load(gen_dir, name))), name
    assert not banding.is_banded(meas(load(gen_dir, "lena.jpg", (900, 900))))


def test_stripes_on_left_half_only_are_incoherent(clean):
    img = banded(clean, 0.1, 30)
    mixed = clean.copy()
    mixed[:, : clean.shape[1] // 2] = img[:, : clean.shape[1] // 2]
    assert banding.is_banded(meas(img)) and not banding.is_banded(meas(mixed))


def test_horizontal_blinds_texture_is_not_banding(clean):
    img = clean.copy()
    h, w = img.shape[:2]
    blinds = np.where((np.arange(h) // 7 % 2 == 0)[:, None, None], 1.35, 0.7)
    sl = (slice(h // 6, 5 * h // 6), slice(w // 5, 3 * w // 5))  # a window, not the full width
    img[sl] = np.clip(img[sl].astype(np.float32) * blinds[sl[0]], 0, 255).astype(np.uint8)
    assert not banding.is_banded(meas(img))


def test_rotated_orientation_6_still_flags(clean):
    # sensor stored sideways: bands run along sensor rows == upright preview columns
    sensor = banded(clean, 0.08, 30)
    upright = np.ascontiguousarray(np.rot90(sensor, -1))  # what EXIF orientation 6 displays
    assert banding.is_banded(meas(upright, orient=6))
    assert not banding.is_banded(meas(upright, orient=1))  # same pixels, wrong readout axis


def test_slow_or_unknown_shutter_is_skipped(clean):
    img = banded(clean, 0.1, 30)
    for t in (1 / 50, 1 / 30, 1.0, None, 0.0):
        b = meas(img, t=t)
        assert np.isnan(b.amp_ev) and not banding.is_banded(b)


def test_measure_cost(clean):
    g = sharpness.to_luma(banded(clean, 0.1, 30))
    g = g[:853, :1280] if g.shape[1] >= 1280 else cv2.resize(g, (1280, 853))
    banding.measure(g, 1, FAST)
    t0 = time.perf_counter()
    for _ in range(20):
        banding.measure(g, 1, FAST)
    assert (time.perf_counter() - t0) / 20 < 0.006  # spec 2 ms; loose for CI noise


def vec(img, t=FAST, orient=1):
    raw = stages.stage_a(img, None, t, orient)
    fv = build_vector(raw, compute_norm([raw]))
    flags = banding.flags(raw.band_peak_frac, raw.band_k)
    return fv, quality.assess(fv, Settings(), flags)


def test_pipeline_stage_sets_vector_fields_and_banding_reason(clean):
    fv, q = vec(banded(clean, 0.1, 30))
    assert fv.banding_amp_ev >= 0.12 and fv.banding_coherence >= 0.5
    assert q.hard_fails == ["banding"] and q.tiers["banding"] == quality.SOFT
    fv, q = vec(clean)
    assert "banding" not in q.hard_fails
    fv, q = vec(banded(clean, 0.1, 30), t=1 / 10)
    assert np.isnan(fv.banding_amp_ev) and "banding" not in q.hard_fails


def _cand(name, g, fv, q, p):
    return selection.Candidate(name, g, fv, q, p)


@pytest.mark.parametrize("a,alt,expect", [(0.1, True, "reject"), (0.1, False, "review"),
                                          (0.06, True, "review"), (0.06, False, "review")])
def test_severe_banding_rejects_only_with_clean_alternative(clean, a, alt, expect):
    s = Settings()
    fv_b, q_b = vec(banded(clean, a, 100))  # k = 9: broad, flicker-like
    assert q_b.hard_fails == ["banding"]
    cands = [_cand("a_banded.jpg", 0, fv_b, q_b, 0.7)]
    if alt:
        fv_c, q_c = vec(clean)
        cands.insert(0, _cand("a_clean.jpg", 0, fv_c, q_c, 0.75))
    out = selection.decide(cands, s, 0.6, 0.3)
    d = out[-1]
    assert d.decision == expect and "banding" in d.reasons


# ---- Review #4 M1 ----------------------------------------------------------------------------
def _fixture_photos(gen_dir):
    real = gen_dir.parent / "real"
    files = [
        *(gen_dir / "src").glob("*.jpg"), *(gen_dir / "shoot").glob("*.jpg"), *real.glob("*.jpg")
    ]
    return sorted(p for p in files if p.name != "black.jpg")


def _load_path(p):
    im = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
    s = 1280 / max(im.shape[:2])  # the pipeline's preview size
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    im = cv2.resize(im, None, fx=s, fy=s, interpolation=interp)
    return im  # NOT contrast-compressed: real clipped sky / black regions stay


@pytest.mark.parametrize("period", [30, 60])
def test_recall_on_real_fixture_photos_with_clipped_regions(gen_dir, period):
    photos = _fixture_photos(gen_dir)
    assert len(photos) >= 19
    hits = sum(
        banding.is_banded(meas(banded(_load_path(p), 0.15, period))) for p in photos
    )
    assert hits >= 17 * len(photos) / 19, (hits, len(photos))


def test_clipped_strip_does_not_mask_whole_row_or_give_nan(clean):
    img = banded(clean, 0.1, 30)
    img[:, : img.shape[1] // 4] = 0  # a black stage edge: 2 of 8 strips clipped on every row
    b = meas(img)
    assert not np.isnan(b.amp_ev) and banding.is_banded(b), b
    img2 = banded(clean, 0.1, 30)
    img2[: img2.shape[0] * 3 // 5, : img2.shape[1] // 3] = 255  # >50 % of rows partly clipped
    b2 = meas(img2)
    assert not np.isnan(b2.amp_ev) and banding.is_banded(b2), b2


def _blinds(clean, width_frac, period_px, gain=(1.35, 0.7)):
    img = clean.copy()
    h, w = img.shape[:2]
    stripes = np.where((np.arange(h) // period_px % 2 == 0)[:, None, None], gain[0], gain[1])
    x1 = int(w * width_frac)
    img[:, :x1] = np.clip(img[:, :x1].astype(np.float32) * stripes, 0, 255).astype(np.uint8)
    return img


@pytest.mark.parametrize("period_px", [4, 7, 12])
def test_full_width_blinds_flagged_at_most_soft_never_flicker(clean, period_px):
    img = _blinds(clean, 1.0, period_px)
    fv, q = vec(img)
    assert not q.flicker, (period_px, fv.banding_amp_ev)  # fast stripes: not flicker-like


def test_full_width_blinds_never_rejected_next_to_unmeasured_or_clean_sibling(clean):
    s = Settings()
    img = _blinds(clean, 1.0, 7)
    fv_b, q_b = vec(img)
    for sib_t in (FAST, 1 / 10):  # a clean sibling, and one whose shutter was too slow to measure
        fv_c, q_c = vec(clean, t=sib_t)
        cands = [_cand("a_clean.jpg", 0, fv_c, q_c, 0.75), _cand("b_blinds.jpg", 0, fv_b, q_b, 0.7)]
        d = selection.decide(cands, s, 0.6, 0.3)[-1]
        assert not (d.decision == "reject" and "banding" in d.reasons), sib_t


def test_banding_sibling_with_unmeasured_shutter_is_not_a_clean_alternative(clean):
    s = Settings()
    fv_b, q_b = vec(banded(clean, 0.1, 150))  # low-frequency bands: flicker-like
    assert q_b.flicker
    fv_c, q_c = vec(clean, t=1 / 10)  # NaN banding on the sibling
    cands = [_cand("a_clean.jpg", 0, fv_c, q_c, 0.75), _cand("b_banded.jpg", 0, fv_b, q_b, 0.7)]
    assert selection.decide(cands, s, 0.6, 0.3)[-1].decision == "review"


def test_real_photo_stripes_texture_negatives(gen_dir):
    for p in _fixture_photos(gen_dir)[:6]:
        im = _load_path(p)
        for period_px in (5, 9):
            fv, q = vec(_blinds(im, 1.0, period_px))
            assert not q.flicker, (p.name, period_px)


# ---- Review #5 r1: coarse non-sinusoidal scene stripes (stairs, bleachers) --------------------
def _wave(img, kind, period, a=0.15, shift=0.0):
    y = (np.arange(img.shape[0], dtype=np.float32) + shift * period) / period
    frac = y - np.floor(y)
    w = {"saw": 2 * frac - 1, "square": np.where(frac < 0.5, 1.0, -1.0)}[kind]
    gain = (1 + a * w)[:, None, None]
    return np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("kind", ["saw", "square"])
@pytest.mark.parametrize("period", [60, 100, 150])
def test_coarse_sawtooth_and_square_stripes_never_flicker(clean, kind, period):
    for a in (0.1, 0.2):
        fv, q = vec(_wave(clean, kind, period, a))
        assert not q.flicker, (kind, period, a, fv.banding_amp_ev)


@pytest.mark.parametrize("kind", ["saw", "square"])
def test_coarse_stripes_next_to_clean_sibling_are_not_rejected(clean, kind):
    s = Settings()
    fv_b, q_b = vec(_wave(clean, kind, 100, 0.2))
    fv_c, q_c = vec(clean)
    cands = [_cand("a_clean.jpg", 0, fv_c, q_c, 0.75), _cand("b_stairs.jpg", 0, fv_b, q_b, 0.7)]
    d = selection.decide(cands, s, 0.6, 0.3)[-1]
    assert d.decision != "reject", d


def test_sine_flicker_still_rejects_control(clean):
    fv, q = vec(banded(clean, 0.2, 100))
    assert q.flicker
