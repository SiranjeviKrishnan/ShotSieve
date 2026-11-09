"""T22 step 1: crop geometry (11 section 4, acceptance 2 and the geometry half of 6)."""

from __future__ import annotations

import random

import pytest

from shotsieve.pipeline import crop_geom as g
from shotsieve.pipeline.crop_geom import Rect

R = Rect(0.10, 0.20, 0.55, 0.70)


@pytest.mark.parametrize("o", range(1, 9))
def test_forward_inverse_is_identity(o):
    for rect in (R, Rect(0, 0, 1, 1), Rect(0.3, 0.1, 0.4, 0.95)):
        back = g.from_crs(g.to_crs(rect, o), o)
        assert (
            max(abs(a - b) for a, b in zip(back.as_tuple(), rect.as_tuple(), strict=True)) < 1e-12
        )
        again = g.to_crs(g.from_crs(rect, o), o)
        assert (
            max(abs(a - b) for a, b in zip(again.as_tuple(), rect.as_tuple(), strict=True)) < 1e-12
        )


def test_to_crs_golden_values():
    r = Rect(0.10, 0.20, 0.55, 0.70)
    c = lambda o: tuple(round(v, 6) for v in g.to_crs(r, o).as_tuple())  # noqa: E731
    assert c(1) == (0.10, 0.20, 0.55, 0.70)
    assert c(3) == (0.45, 0.30, 0.90, 0.80)  # rot180: (1-x1, 1-y1, 1-x0, 1-y0)
    # orientation 6 (rot90 CW to display): stored point (u, v) shows at (1 - v, u), so a display
    # point (x, y) was at stored (u, v) = (y, 1 - x)
    assert c(6) == (0.20, 0.45, 0.70, 0.90)
    # orientation 8 (rot90 CCW): display (x, y) was stored at (1 - y, x)
    assert c(8) == (0.30, 0.10, 0.80, 0.55)
    assert c(2) == (0.45, 0.20, 0.90, 0.70) and c(4) == (0.10, 0.30, 0.55, 0.80)


def test_corner_mapping_of_orientation_6_and_8():
    assert g.orient_point(6, 0, 0) == (1, 0)  # stored top-left shows top-right
    assert g.orient_point(6, 1, 0) == (1, 1)  # stored top-right shows bottom-right
    assert g.orient_point(8, 0, 0) == (0, 1)  # stored top-left shows bottom-left
    assert g.orient_point(5, 0.2, 0.7) == (0.7, 0.2)
    assert g.orient_point(7, 0, 0) == (1, 1)
    with pytest.raises(ValueError):
        g.to_crs(R, 9)


def test_display_dims_swap_for_5_to_8():
    assert g.display_dims(6000, 4000, 1) == (6000, 4000)
    assert g.display_dims(6000, 4000, 6) == (4000, 6000)
    assert g.display_dims(6000, 4000, 8) == (4000, 6000)
    assert g.display_dims(6000, 4000, 3) == (6000, 4000)


@pytest.mark.parametrize(
    ("text", "kind", "a", "b"),
    [("4:5", "fixed", 4, 5), ("1.91:1", "fixed", 1.91, 1), (" 16 : 9 ", "fixed", 16, 9),
     ("Original", "original", 0, 0), ("FREE", "free", 0, 0), ("1:4", "fixed", 1, 4),
     ("4:1", "fixed", 4, 1)],
)  # fmt: skip
def test_parse_ratio_valid(text, kind, a, b):
    r = g.parse_ratio(text)
    assert (r.kind, r.a, r.b) == (kind, a, b)


@pytest.mark.parametrize(
    "bad", ["", "4", "4:5:6", "a:b", "0:1", "-1:2", "1:5", "5:1", "1.234:1", "1:", ":1", "nan:1",
            "inf:1", "+4:5", "1e1:2", None, 7],
)  # fmt: skip
def test_parse_ratio_invalid(bad):
    with pytest.raises(ValueError):
        g.parse_ratio(bad)


def test_ratio_text_flip_and_original():
    assert g.parse_ratio("4:5").flipped().text() == "5:4"
    assert g.parse_ratio("1.91:1").text() == "1.91:1"
    assert g.parse_ratio("free").flipped().kind == "free"
    assert g.parse_ratio("original").resolve(6000, 4000) == (6000.0, 4000.0)
    assert g.parse_ratio("free").resolve(10, 10) is None
    assert g.parse_ratio("2:3").oriented(6000, 4000).text() == "3:2"  # landscape image
    assert g.parse_ratio("2:3").oriented(4000, 6000).text() == "2:3"
    assert g.parse_ratio("1:1").oriented(100, 50).text() == "1:1"


def test_snap_property_1000_random_dims_ratios_and_orientations():
    rnd = random.Random(7)
    presets = [
        "1:1",
        "4:5",
        "5:4",
        "2:3",
        "3:2",
        "5:7",
        "7:5",
        "16:9",
        "9:16",
        "1.91:1",
        "1:3.5",
        "2.35:1",
    ]
    for _ in range(1000):
        sw, sh = rnd.randint(300, 9000), rnd.randint(300, 9000)
        o = rnd.randint(1, 8)
        fw, fh = g.display_dims(sw, sh, o)
        ratio = g.parse_ratio(rnd.choice(presets))
        w = rnd.uniform(0.05, 1.0)
        h = rnd.uniform(0.05, 1.0)
        x0, y0 = rnd.uniform(0, 1 - w), rnd.uniform(0, 1 - h)
        rect, px = g.snap_rect(Rect(x0, y0, x0 + w, y0 + h), ratio, fw, fh)
        W, H = px[2] - px[0], px[3] - px[1]
        assert 0 <= px[0] < px[2] <= fw and 0 <= px[1] < px[3] <= fh
        assert abs(W * ratio.b - H * ratio.a) <= max(ratio.a, ratio.b) + 1e-9
        assert abs(W - H * ratio.a / ratio.b) <= 1.0
        assert rect.valid() and g.ratio_error_px(px, ratio, fw, fh) <= 1.0
        # the stored frame round trip keeps the pixel rect exactly (px -> crs -> px)
        c = g.to_crs(rect, o)
        sw2, sh2 = (fh, fw) if o >= 5 else (fw, fh)
        back = g.from_crs(Rect(*(round(v, 6) for v in c.as_tuple())), o)
        for v, p, dim in zip(back.as_tuple(), px, (fw, fh, fw, fh), strict=True):
            assert abs(v * dim - p) <= 0.5 + 1e-6 * max(sw2, sh2)


def test_snap_centre_original_and_free():
    fw, fh = 6000, 4000
    rect, px = g.snap_rect(Rect(0.3, 0.3, 0.6, 0.6), g.parse_ratio("1:1"), fw, fh)
    assert px[2] - px[0] == px[3] - px[1] == 1200
    assert abs((px[0] + px[2]) / 2 - 0.45 * fw) <= 1
    assert g.snap_px(Rect(0.1, 0.1, 0.2, 0.2), g.parse_ratio("original"), fw, fh) == (0, 0, fw, fh)
    px = g.snap_px(Rect(0.1, 0.1, 0.2, 0.3), g.parse_ratio("free"), fw, fh)
    assert px == (600, 400, 1200, 1200)
    # too wide for the frame: shrinks H until W fits
    px = g.snap_px(Rect(0, 0.1, 1, 0.9), g.parse_ratio("4:1"), fw, fh)
    assert px[2] - px[0] <= fw and abs((px[2] - px[0]) - 4 * (px[3] - px[1])) <= 1


def test_matches_ratio_accepts_snapped_rects_and_rejects_off_ratio():
    fw, fh = 6000, 4000
    ratio = g.parse_ratio("4:5")
    rect, _ = g.snap_rect(Rect(0.2, 0.1, 0.6, 0.9), ratio, fw, fh)
    assert g.matches_ratio(rect, ratio, fw, fh)
    assert g.matches_ratio(Rect(*(round(v, 6) for v in rect.as_tuple())), ratio, fw, fh)
    assert not g.matches_ratio(Rect(0.2, 0.1, 0.6, 0.9), ratio, fw, fh)
    assert g.matches_ratio(Rect(0.2, 0.1, 0.6, 0.9), g.parse_ratio("free"), fw, fh)


def test_rect_helpers():
    a, b = Rect(0, 0, 0.5, 0.5), Rect(0.25, 0.25, 0.75, 0.75)
    assert round(a.iou(b), 6) == round(0.0625 / (0.25 + 0.25 - 0.0625), 6)
    assert a.iou(Rect(0.6, 0.6, 1, 1)) == 0.0
    assert Rect.of([0.1, 0.2, 0.3, 0.4]).w == pytest.approx(0.2)
    assert not Rect(0.5, 0.5, 0.4, 0.9).valid()
