"""T22 step 2: crop scoring on synthetic scenes (no models): constraints, terms, alternatives."""

from __future__ import annotations

import time

import numpy as np
import pytest

from shotsieve.pipeline import crop_geom as g
from shotsieve.pipeline import crop_score as cs
from shotsieve.pipeline.crop_score import Scene, Subject

FW, FH = 6000, 4000


def face(sid, cx, cy, w=0.06, ar=1.0, **kw):
    """A face box centred (cx, cy) of width w (normalised); height keeps a square in pixels."""
    h = w * FW / FH * ar
    return Subject(sid, "face", (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), **kw)


def person(sid, x0, y0, x1, y1, **kw):
    return Subject(sid, "person", (x0, y0, x1, y1), **kw)


def run(selected, everything=None, ratio="4:5", n=3, **kw):
    sc = Scene(selected, everything if everything is not None else list(selected), FW, FH, **kw)
    return sc, cs.suggest(sc, g.parse_ratio(ratio), n)


def inside(rect, box, frac=1.0):
    iw = min(rect.x1, box[2]) - max(rect.x0, box[0])
    ih = min(rect.y1, box[3]) - max(rect.y0, box[1])
    a = max(iw, 0) * max(ih, 0) / ((box[2] - box[0]) * (box[3] - box[1]))
    return a >= frac - 1e-9


def test_face_on_left_third_stays_on_a_third():
    f = face(
        "f1",
        0.22,
        0.40,
        landmarks=((0.20, 0.39), (0.24, 0.39), (0.22, 0.42), (0.2, 0.44), (0.24, 0.44)),
    )
    _, res = run([f], ratio="3:2")
    c = res.candidates[0]
    ex = (0.22 - c.rect.x0) / c.rect.w
    assert min(abs(ex - 1 / 3), abs(ex - 0.5), abs(ex - 2 / 3)) < 0.06, ex
    assert not res.infeasible


def test_must_keep_is_never_cut_across_ratios_and_alternatives():
    f1, f2 = face("a", 0.35, 0.45), face("b", 0.55, 0.40, w=0.07)
    for r in ("1:1", "4:5", "5:4", "2:3", "3:2", "16:9", "7:5"):
        sc, res = run([f1, f2], ratio=r)
        assert res.candidates
        for c in res.candidates:
            if res.infeasible:
                continue
            for box in sc.face_boxes:
                assert inside(c.rect, tuple(box)), (r, c.rect)
            assert c.rect.valid()


def test_infeasible_is_flagged_and_maximal():
    f1, f2 = face("a", 0.08, 0.5), face("b", 0.92, 0.5)
    sc, res = run([f1, f2], ratio="9:16", n=3)
    assert res.infeasible
    for c in res.candidates:
        assert "subjects_do_not_fit" in c.warnings
    top = res.candidates[0]
    # the largest crop at 9:16 in a 3:2 frame: full height
    assert top.rect.h == pytest.approx(1.0, abs=2e-3)
    # feasible at 3:2 for the same pair: no flag
    _, ok = run([f1, f2], ratio="3:2")
    assert not ok.infeasible


def test_infeasible_prefers_faces_over_other_subjects():
    f = face("f", 0.10, 0.5, w=0.05)
    obj = Subject("o", "object", (0.80, 0.30, 0.95, 0.70))
    sc, res = run([f, obj], ratio="9:16")
    assert res.infeasible
    assert inside(res.candidates[0].rect, tuple(sc.face_boxes[0]))


def test_stranger_at_the_edge_is_avoided_when_possible():
    main = face("m", 0.40, 0.45, w=0.08)
    stranger = face("s", 0.64, 0.45, w=0.08)
    _, res = run([main], [main, stranger], ratio="1:1")
    top = res.candidates[0]
    f = 1.0 if inside(top.rect, stranger.box) else 0.0
    part = inside(top.rect, stranger.box, 0.05) and not inside(top.rect, stranger.box, 0.95)
    assert not part, (top.rect, f)
    assert top.terms["strangers"] == 1.0


def test_stranger_cut_is_reported_when_unavoidable():
    main = face("m", 0.50, 0.45, w=0.10)
    stranger = face("s", 0.56, 0.45, w=0.10)  # overlaps the main face region
    sc, res = run([main], [main, stranger], ratio="1:1")
    top = res.candidates[0]
    cut = inside(top.rect, stranger.box, 0.05) and not inside(top.rect, stranger.box, 0.95)
    assert ("unselected_person_cut" in top.warnings) == cut


def test_neck_knee_and_ankle_bands_are_penalised():
    p = person("p", 0.40, 0.05, 0.56, 0.95)
    sc = Scene([p], [p], FW, FH)
    # crop bottoms at 0.15 / 0.75 / 0.97 of the person height vs a clean 0.50 / 0.60
    ph = 0.90

    def term(rel_bottom):
        by = 0.05 + rel_bottom * ph
        x0 = np.array([0.30])
        y0 = np.array([by - 0.7])
        return float(sc.terms(x0, y0, x0 + 0.5, np.array([by]))["cut"][0])

    # use a person tall enough that the crop covers >= 40 % of it (h_C = 0.7)
    for bad in (0.16, 0.75, 0.95):
        assert term(bad) == 0.0, bad
    for good in (0.50, 0.60, 0.88):
        assert term(good) == 1.0, good


def test_headroom_window():
    f = face("f", 0.5, 0.5, w=0.1)
    sc = Scene([f], [f], FW, FH)
    top = sc.head_top

    def head(y0, h=0.6):
        return float(
            sc.terms(np.array([0.3]), np.array([y0]), np.array([0.7]), np.array([y0 + h]))["head"][
                0
            ]
        )

    assert head(top - 0.10 * 0.6) == pytest.approx(1.0)  # hr = 0.10
    assert head(top) == pytest.approx(0.0)  # hr = 0
    assert head(top - 0.40 * 0.6) == pytest.approx(0.0, abs=1e-6)
    assert 0 < head(top - 0.30 * 0.6) < 1
    _, res = run([f], ratio="4:5")
    c = res.candidates[0]
    hr = (top - c.rect.y0) / c.rect.h
    assert 0.03 <= hr <= 0.20, hr


def test_horizon_is_not_centred():
    f = face("f", 0.5, 0.35, w=0.05)
    sc, res = run([f], ratio="3:2", horizon=(0.5, 0.2))
    for c in res.candidates:
        yh = (0.5 - c.rect.y0) / c.rect.h
        assert abs(yh - 0.5) > 0.04 or c.terms["horizon"] == 0.3


def test_alternatives_are_diverse_and_deterministic():
    f = face("f", 0.45, 0.40, w=0.05)
    _, r1 = run([f], ratio="1:1")
    _, r2 = run([f], ratio="1:1")
    assert len(r1.candidates) == 3
    for i, a in enumerate(r1.candidates):
        for b in r1.candidates[i + 1 :]:
            assert a.rect.iou(b.rect) < 0.70 + 0.03  # refinement moves a pick by half a step
    assert [c.rect.as_tuple() for c in r1.candidates] == [c.rect.as_tuple() for c in r2.candidates]
    assert [c.score for c in r1.candidates] == sorted(
        (c.score for c in r1.candidates), reverse=True
    )


def test_terms_sum_with_crop_weights():
    f = face("f", 0.4, 0.4)
    sal = np.random.default_rng(1).random((20, 30))
    sc, res = run([f], ratio="4:5", saliency=sal, horizon=(0.7, 0.0))
    for c in res.candidates:
        want = sum(cs.CROP_WEIGHTS[k] * c.terms[k] for k in cs.CROP_WEIGHTS)
        assert c.score == pytest.approx(want, abs=1e-9)
    assert sum(cs.CROP_WEIGHTS.values()) == pytest.approx(1.0)


def test_ratio_exact_in_full_res_pixels_and_inside_frame():
    f = face("f", 0.3, 0.5)
    for r in ("1:1", "4:5", "16:9", "9:16", "1.91:1", "original"):
        sc, res = run([f], ratio=r)
        for c in res.candidates:
            px = c.rect_px
            assert px is not None
            ratio = g.parse_ratio(r)
            assert g.ratio_error_px(px, ratio, FW, FH) <= 1.0
            assert 0 <= px[0] < px[2] <= FW and 0 <= px[1] < px[3] <= FH
    _, res = run([f], ratio="original")
    assert res.candidates[0].rect_px == (0, 0, FW, FH)


def test_person_select_mode_keeps_face_and_body_and_face_mode_only_the_face():
    fc = face("f", 0.45, 0.25, w=0.05, linked="p")
    p = person("p", 0.38, 0.20, 0.54, 0.95, linked="f", select_mode="person")
    sc, res = run([p], [fc, p], ratio="2:3")
    top = res.candidates[0]
    assert inside(top.rect, sc.face_boxes[0]) and inside(top.rect, p.box, 0.98)
    p2 = person("p", 0.38, 0.20, 0.54, 0.95, linked="f", select_mode="face")
    sc2, res2 = run([p2], [fc, p2], ratio="1:1")
    assert len(sc2.face_boxes) == 1 and len(sc2.other_boxes) == 0
    assert res2.candidates[0].rect.area < top.rect.area + 1  # a head shot is allowed to be tight


def test_point_subject_and_empty_selection_fall_back_gracefully():
    pt = Subject("pt", "point", (0.60, 0.30, 0.72, 0.52), source="dino")
    sc, res = run([pt], ratio="3:2")
    assert inside(res.candidates[0].rect, pt.box, 0.98)
    _, none = run([], [], ratio="4:5", saliency=np.ones((8, 12)))
    assert none.candidates and not none.infeasible


def test_free_ratio_returns_candidates_from_presets():
    f = face("f", 0.4, 0.4)
    sc, res = run([f], ratio="free")
    assert res.candidates and res.ratio_px is None
    for c in res.candidates:
        assert inside(c.rect, tuple(sc.face_boxes[0]))


def test_burst_same_term_keeps_the_subject_relative_framing():
    f = face("f", 0.45, 0.40, w=0.05)
    sc, res = run([f], ratio="4:5")
    a = res.candidates[0]
    u = (0.45 - a.rect.x0) / a.rect.w
    v = (0.40 - a.rect.y0) / a.rect.h
    # the subject moved to the right of the frame in the next shot
    f2 = face("f", 0.60, 0.42, w=0.05)
    sc2 = Scene([f2], [f2], FW, FH, same_ref=(u, v))
    r2 = cs.suggest(sc2, g.parse_ratio("4:5"), 1).candidates[0]
    u2 = (0.60 - r2.rect.x0) / r2.rect.w
    assert abs(u2 - u) < 0.06
    assert "same" in r2.terms


def test_suggest_warm_speed():
    f1, f2 = face("a", 0.35, 0.45), face("b", 0.55, 0.40)
    sal = np.random.default_rng(2).random((24, 32))
    times = []
    for i in range(30):
        sc = Scene([f1, f2], [f1, f2], FW, FH, saliency=sal)
        t0 = time.perf_counter()
        cs.suggest(sc, g.parse_ratio(("4:5", "3:2", "1:1")[i % 3]), 3)
        times.append(time.perf_counter() - t0)
    assert (
        sorted(times)[int(0.95 * len(times)) - 1] <= 0.25
    )  # loose CI bound; the bench holds 60 ms
