"""Review #12 B1 (no silent subject cuts): (a) subject extension, (b) `subject_may_be_cut`,
(c) the over-segmentation fallback of the point prompt. Synthetic tokens, no models."""

from __future__ import annotations

import numpy as np

from shotsieve.models import patch_features as pf
from shotsieve.pipeline import crop_geom as g
from shotsieve.pipeline import crop_score as cs
from shotsieve.pipeline.crop_score import Scene, Subject

GH, GW = 20, 28


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _tokens(blocks, seed=0):
    """Background tokens plus `blocks` = [(vector, (x0, y0, x1, y1) in patches)]."""
    rng = np.random.default_rng(seed)
    bg = rng.normal(size=384).astype(np.float32)
    t = bg + 0.05 * rng.normal(size=(GH, GW, 384)).astype(np.float32)
    for vec, (x0, y0, x1, y1) in blocks:
        t[y0:y1, x0:x1] = vec + 0.05 * rng.normal(size=(y1 - y0, x1 - x0, 384)).astype(np.float32)
    return _unit(t).astype(np.float32)


def _vec(seed):
    return np.random.default_rng(100 + seed).normal(size=384).astype(np.float32)


def _area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


def test_extension_grows_a_truncated_box_over_same_subject_patches_and_respects_the_cap():
    # a "dog" occupying patches x 6..22, y 6..16; the detector box only covers x 6..14
    t = _tokens([(_vec(1), (6, 6, 22, 16))])
    sal = np.zeros((GH, GW), np.float32)
    box = (6 / GW, 6 / GH, 14 / GW, 16 / GH)
    ext, _halo = pf.subject_extent(t, sal, box, [], grow=True)
    assert ext[0] == box[0] and ext[1] == box[1]
    assert ext[2] > box[2] + 0.1  # grew over the tail
    assert _area(ext) <= pf.GROW_CAP * _area(box) + 1e-9  # capped at 2x the detected box
    assert ext[2] <= 22 / GW + 1e-9  # never past the subject
    # an upper cap below the needed growth stops the growth early
    old = pf.GROW_CAP
    try:
        pf.GROW_CAP = 1.1
        small, _ = pf.subject_extent(t, sal, box, [], grow=True)
    finally:
        pf.GROW_CAP = old
    assert _area(small) <= 1.1 * _area(box) + 1e-9 and small[2] < ext[2]


def test_extension_does_not_absorb_another_subject_or_a_different_background():
    t = _tokens([(_vec(1), (6, 6, 22, 16))])
    sal = np.zeros((GH, GW), np.float32)
    box = (6 / GW, 6 / GH, 14 / GW, 16 / GH)
    blocker = (16 / GW, 4 / GH, 22 / GW, 18 / GH)  # another subject on the right part
    ext, _ = pf.subject_extent(t, sal, box, [blocker], grow=True)
    assert ext[2] <= 16 / GW + 1e-9
    plain, _ = pf.subject_extent(_tokens([]), sal, box, [], grow=True)  # nothing else alike
    assert plain == box or _area(plain) <= pf.GROW_CAP * _area(box)
    assert pf.subject_extent(t, sal, box, [], grow=False)[0] == box  # persons: halo only


def _scene_with_halo(halo_box):
    """One selected object whose halo (same-subject patches) is `halo_box` (patch rectangle)."""
    m = np.zeros((GH, GW), bool)
    x0, y0, x1, y1 = halo_box
    m[y0:y1, x0:x1] = True
    s = Subject("o1", "object", (0.30, 0.30, 0.50, 0.60), halo=pf.pack_mask(m))
    return Scene([s], [s], 6000, 4000), s


def test_edge_crossing_a_same_subject_run_adds_subject_may_be_cut():
    # the halo spans x 0.30..0.80: a 9:16 crop (0.375 of the width) cannot hold it, so an edge
    # always passes through it
    sc, _ = _scene_with_halo((8, 6, 22, 12))
    res = cs.suggest(sc, g.parse_ratio("9:16"), 3)
    assert res.candidates and all("subject_may_be_cut" in c.warnings for c in res.candidates)
    # not infeasible: the warning is its own flag (counted as flagged, amber in crop.js)
    assert not res.infeasible
    assert 'subject_may_be_cut' in open("src/shotsieve/server/static/crop.js").read()


def test_no_warning_when_the_crop_holds_the_whole_subject_region():
    sc, _ = _scene_with_halo((8, 6, 14, 12))  # the halo is just the box
    res = cs.suggest(sc, g.parse_ratio("1:1"), 3)
    assert res.candidates
    assert all("subject_may_be_cut" not in c.warnings for c in res.candidates)
    # a crop that misses the halo entirely is not "cut" either
    assert not sc.halo_cut(0.0, 0.0, 0.2, 0.2)


def test_over_segmented_point_region_falls_back_to_the_15_percent_box():
    a = _unit(_vec(2))
    ortho = _unit(_vec(3) - (_vec(3) @ a) * a)
    b = 20 * (0.45 * a + np.sqrt(1 - 0.45**2) * ortho)  # cos(a, b) = 0.45: "looks a bit alike"
    a = 20 * a
    t = _tokens([(b, (0, 8, GW, 11)), (a, (10, 8, 12, 10))])  # a thin full-width band + a seed
    sal = np.zeros((GH, GW), np.float32)
    sal[8:11] = 1.0  # the band is salient: it joins the region through the loose threshold
    x, y = 11 / GW, 9 / GH
    box, how = pf.point_region(t, sal, x, y)
    assert how == "fallback"
    assert box[2] - box[0] < 0.6  # not the full-width band
    assert 0.10 <= _area(box) <= 0.20
    assert box[0] <= x <= box[2] and box[1] <= y <= box[3]
    # without the band the region is just the object
    t2 = _tokens([(a, (10, 8, 12, 10))])
    box2, how2 = pf.point_region(t2, sal * 0, x, y)
    assert how2 == "region" and _area(box2) < 0.05


def test_a_big_object_up_to_the_fitted_area_still_gives_a_region():
    a = _vec(4)
    t = _tokens([(a, (4, 3, 22, 17))])  # 18 x 14 patches = 45 %: too big on purpose
    box, how = pf.point_region(t, np.zeros((GH, GW), np.float32), 13 / GW, 10 / GH)
    assert how == "fallback"  # above PT_MAX_AREA
    t = _tokens([(a, (6, 5, 20, 15))])  # 14 x 10 = 25 %: below the fitted 35 % limit
    box, how = pf.point_region(t, np.zeros((GH, GW), np.float32), 13 / GW, 10 / GH)
    assert how == "region" and _area(box) < pf.PT_MAX_AREA + 0.02
