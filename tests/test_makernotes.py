"""AF-point extraction: synthetic Canon/Sony maker-note containers + real sample heads."""

import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from shotsieve.io.makernotes import AFPoint, parse_af_points, read_af_points, to_oriented

REAL = Path(__file__).parent / "fixtures" / "real"
IMG = np.full((64, 96, 3), 128, np.uint8)


def canon_jpeg(mf, points, in_focus, selected=None, **kw):
    af = mf.canon_afinfo2(points, (6000, 4000), in_focus, selected)
    t = mf.tiff_with_makernote(b"Canon", [(0x0026, 3, af, len(af) // 2)], **kw)
    return mf.jpeg_with_tiff(t, IMG)


def test_canon_in_focus_points_centre_origin_y_up(make_fixtures):
    pts = [(300, 400, -1500, 1000), (300, 400, 600, -800), (300, 400, 0, 0)]
    pts_out = parse_af_points(canon_jpeg(make_fixtures, pts, in_focus=[1]))
    assert len(pts_out) == 1
    p = pts_out[0]
    assert (p.x, p.y) == pytest.approx((0.5 + 600 / 6000, 0.5 + 800 / 4000))  # Y up -> y grows down
    assert (p.w, p.h) == pytest.approx((300 / 6000, 400 / 4000))


def test_canon_selected_points_when_no_focus_mask(make_fixtures):
    pts = [(300, 400, -1500, 1000), (300, 400, 600, -800)]
    out = parse_af_points(canon_jpeg(make_fixtures, pts, in_focus=[], selected=[0]))
    assert len(out) == 1 and out[0].x == pytest.approx(0.25) and out[0].y == pytest.approx(0.25)


def test_canon_big_endian_and_multiple_in_focus(make_fixtures):
    mf = make_fixtures
    pts = [(300, 400, -1500, 1000), (300, 400, 600, -800), (300, 400, 0, 0)]
    af = mf.canon_afinfo2(pts, (6000, 4000), [0, 2], e=">")
    t = mf.tiff_with_makernote(b"Canon", [(0x0026, 3, af, len(af) // 2)], e=">")
    assert len(parse_af_points(mf.jpeg_with_tiff(t, IMG))) == 2


@pytest.mark.parametrize("header", [True, False])
def test_sony_focus_location(make_fixtures, header):
    mf = make_fixtures
    fl = mf.sony_focus_location(6000, 4000, 3000, 1000)
    t = mf.tiff_with_makernote(b"SONY", [(0x2027, 3, fl, 4)], sony_header=header)
    (p,) = parse_af_points(mf.jpeg_with_tiff(t, IMG))
    assert (p.x, p.y) == pytest.approx((0.5, 0.25))
    assert (p.w, p.h) == pytest.approx((0.06 * 4000 / 6000, 0.06))  # 6 % of the short edge


def test_cr3_cmt3_box_and_raw_tiff(make_fixtures):
    mf = make_fixtures
    af = mf.canon_afinfo2([(300, 400, 600, -800)], (6000, 4000), [0])
    t = mf.tiff_with_makernote(b"Canon", [(0x0026, 3, af, len(af) // 2)])
    assert len(parse_af_points(mf.cr3_like(t))) == 1  # CR3
    assert len(parse_af_points(t)) == 1  # CR2 / ARW style: TIFF at offset 0


def test_unknown_make_garbage_and_truncation_give_empty(make_fixtures, tmp_path):
    mf = make_fixtures
    t = mf.tiff_with_makernote(b"Nikon", [(0x0026, 3, b"\0\0", 1)])
    assert parse_af_points(mf.jpeg_with_tiff(t, IMG)) == []
    assert parse_af_points(b"") == [] and parse_af_points(b"\xff\xd8\xff\xe1\x00") == []
    assert parse_af_points(np.random.default_rng(0).bytes(4096)) == []
    af = mf.canon_afinfo2([(300, 400, 0, 0)], (6000, 4000), [0])
    good = mf.tiff_with_makernote(b"Canon", [(0x0026, 3, af, len(af) // 2)])
    assert parse_af_points(good[:60]) == []  # truncated inside the IFDs
    p = tmp_path / "x.jpg"
    p.write_bytes(mf.jpeg_with_tiff(good, IMG))
    assert read_af_points(p, SimpleNamespace(make="Nikon")) == []
    assert len(read_af_points(p, SimpleNamespace(make="Canon"))) == 1
    assert read_af_points(tmp_path / "missing.jpg") == []


def test_parse_cost_is_small(make_fixtures, tmp_path):
    mf = make_fixtures
    af = mf.canon_afinfo2([(300, 400, 0, 0)] * 61, (6000, 4000), [3])
    t = mf.tiff_with_makernote(b"Canon", [(0x0026, 3, af, len(af) // 2)])
    p = tmp_path / "x.jpg"
    p.write_bytes(mf.jpeg_with_tiff(t, IMG))
    t0 = time.perf_counter()
    for _ in range(200):
        assert read_af_points(p)
    assert (time.perf_counter() - t0) / 200 < 0.002  # spec: <= 0.5 ms; loose for CI noise


def test_orientation_mapping():
    p = AFPoint(0.2, 0.3, 0.1, 0.05)
    assert to_oriented(p, 1) == p
    o6 = to_oriented(p, 6)  # rotate 90 deg clockwise
    assert (o6.x, o6.y, o6.w, o6.h) == pytest.approx((0.7, 0.2, 0.05, 0.1))
    o8 = to_oriented(p, 8)  # rotate 90 deg counter-clockwise
    assert (o8.x, o8.y, o8.w, o8.h) == pytest.approx((0.3, 0.8, 0.05, 0.1))
    assert to_oriented(to_oriented(p, 3), 3) == pytest.approx(p)
    # round trip through the actual array transform used for previews
    from shotsieve.io.preview import apply_orientation

    for orient in range(1, 9):
        a = np.zeros((40, 60), np.uint8)
        a[10:14, 12:24] = 255  # box x 12..24, y 10..14
        b = apply_orientation(a, orient)
        q = to_oriented(AFPoint(18 / 60, 12 / 40, 12 / 60, 4 / 40), orient)
        ys, xs = np.nonzero(b)
        assert (q.x * b.shape[1], q.y * b.shape[0]) == pytest.approx(
            ((xs.min() + xs.max() + 1) / 2, (ys.min() + ys.max() + 1) / 2)
        )
        assert (q.w * b.shape[1], q.h * b.shape[0]) == pytest.approx(
            (xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
        )


@pytest.mark.parametrize(
    "name,make", [("canon_5dm3.head.cr2", "Canon"), ("sony_a7m3.head.arw", "Sony")]
)
def test_real_sample_heads_parse_to_point_inside_frame(name, make):
    f = REAL / name
    if not f.exists():
        pytest.skip("real sample head not present")
    pts = read_af_points(f, SimpleNamespace(make=make))
    assert pts, f"{name}: no AF point parsed"
    for p in pts:
        assert 0 <= p.x <= 1 and 0 <= p.y <= 1 and 0 < p.w <= 1 and 0 < p.h <= 1


# ---- exiftool fallback, tested with a fake `exiftool` on PATH (B2.3; nothing is installed) ----
FAKE_BIN = Path(__file__).parent / "fixtures" / "bin"


@pytest.fixture()
def fake_et(tmp_path, monkeypatch):
    import os

    from shotsieve.io import makernotes as mn

    (FAKE_BIN / "exiftool").chmod(0o755)
    d = tmp_path / "canned"
    d.mkdir()
    monkeypatch.setenv("FAKE_EXIFTOOL_DIR", str(d))
    monkeypatch.setenv("PATH", f"{FAKE_BIN}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(mn, "_et", None)
    img = tmp_path / "shot.cr3x"
    img.write_bytes(b"\x00" * 64)  # not parseable: only the fallback can answer

    class Ctl:
        dir, image = d, img

        def can(self, obj, name="shot.cr3x"):
            import json

            (d / f"{name}.json").write_text(obj if isinstance(obj, str) else json.dumps([obj]))

        def spawns(self):
            f = d / "spawns.log"
            return len(f.read_text().split()) if f.exists() else 0

    yield Ctl()
    if mn._et is not None:
        mn._et.close()
        monkeypatch.setattr(mn, "_et", None)


def test_exiftool_canon_values_centre_origin_y_up(fake_et):
    fake_et.can({
        "AFImageWidth": 6000, "AFImageHeight": 4000,
        "AFAreaWidths": "300 300 300 300", "AFAreaHeights": "400 400 400 400",
        "AFAreaXPositions": "-1500 600 0 -300", "AFAreaYPositions": "1000 -800 0 200",
        "AFPointsInFocus": "1,3",
    })  # fmt: skip
    p1, p3 = read_af_points(fake_et.image, SimpleNamespace(make="Canon"), use_exiftool=True)
    assert (p1.x, p1.y) == pytest.approx((0.5 + 600 / 6000, 0.5 + 800 / 4000))  # Y up -> down
    assert (p1.w, p1.h) == pytest.approx((300 / 6000, 400 / 4000))
    assert (p3.x, p3.y) == pytest.approx((0.5 - 300 / 6000, 0.5 - 200 / 4000))


def test_exiftool_sony_focus_location(fake_et):
    fake_et.can({"FocusLocation": "6000 4000 4500 1000"})
    (p,) = read_af_points(fake_et.image, SimpleNamespace(make="SONY"), use_exiftool=True)
    assert (p.x, p.y) == pytest.approx((0.75, 0.25))


def test_exiftool_malformed_or_crash_returns_empty_and_restarts(fake_et):
    mk = SimpleNamespace(make="Sony")
    fake_et.can("this is not json")
    assert read_af_points(fake_et.image, mk, use_exiftool=True) == []
    assert fake_et.spawns() == 1
    fake_et.can("CRASH")  # process dies without answering
    assert read_af_points(fake_et.image, mk, use_exiftool=True) == []
    fake_et.can({"FocusLocation": "6000 4000 4500 1000"})
    (p,) = read_af_points(fake_et.image, mk, use_exiftool=True)  # next call restarted it
    assert (p.x, p.y) == pytest.approx((0.75, 0.25))
    assert fake_et.spawns() == 2


def test_exiftool_disabled_never_spawns(fake_et):
    fake_et.can({"FocusLocation": "6000 4000 4500 1000"})
    assert read_af_points(fake_et.image, SimpleNamespace(make="Sony"), use_exiftool=False) == []
    assert fake_et.spawns() == 0


def test_exif_from_exiftool_reads_raw_meta(fake_et, tmp_path):
    from shotsieve.io.exif import ImageMeta, _from_exiftool

    raw = tmp_path / "frame.arw"
    raw.write_bytes(b"\x00" * 16)
    fake_et.can({
        "DateTimeOriginal": "2024:05:06 07:08:09", "SubSecTimeOriginal": "25",
        "Make": "SONY", "Model": "ILCE-7M3", "SerialNumber": "1234567",
        "Orientation": 6, "ImageWidth": 6000, "ImageHeight": 4000, "ISO": 400,
    }, "frame.arw")  # fmt: skip
    meta = ImageMeta(timestamp=0.0)
    assert _from_exiftool(raw, meta) is True
    assert meta.camera_serial == "1234567" and meta.orientation == 6
    assert meta.make == "SONY" and meta.ts_source == "exif" and meta.timestamp > 1.7e9
    assert fake_et.spawns() == 1  # the one-shot `exiftool -j -n <file>` form


# ---- m1: one real off-centre AF sample (Sony RX100 V, raw.pixls.us CC0, 512 KB head) -----------
OFF = Path(__file__).parent / "fixtures" / "sony_rx100m5a_cc0" / "rx100m5a.head.arw"
# hand-decoded once from the maker-note bytes (tag 0x2027 FocusLocation: W, H, X, Y), no tool used
OFF_RAW = (5472, 3648, 1872, 2424)


def _thumb_gray(buf: bytes) -> np.ndarray:
    """The 160x120 embedded thumbnail that sits whole inside the 512 KB head."""
    import re

    import cv2

    for m in re.finditer(b"\xff\xd8\xff", buf):
        im = cv2.imdecode(np.frombuffer(buf[m.start() :], np.uint8), cv2.IMREAD_GRAYSCALE)
        if im is not None and im.shape == (120, 160):
            return im
    raise AssertionError("no 160x120 thumbnail in the head")


def _tile_rank(gray: np.ndarray, x: float, y: float) -> int:
    """Rank (0 = sharpest) of the 8x8 Laplacian-variance tile that contains (x, y)."""
    import cv2

    lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
    h, w = gray.shape
    v = np.array([[lap[r * h // 8 : (r + 1) * h // 8, c * w // 8 : (c + 1) * w // 8].var()
                   for c in range(8)] for r in range(8)])  # fmt: skip
    rank = (-v.ravel()).argsort().argsort().reshape(8, 8)
    return int(rank[min(7, int(y * 8)), min(7, int(x * 8))])


def test_real_off_centre_sample_is_the_pinned_file():
    import hashlib

    assert hashlib.sha256(OFF.read_bytes()).hexdigest() == (
        "e41986b3ecda1b01b1c57f517f6c6956a2fbabbeaab1b9adfbef1a8b75c4963b"
    )  # recorded in rx100m5a.LICENSE.txt


def test_real_off_centre_sample_matches_hand_decoded_values():
    pts = read_af_points(OFF, SimpleNamespace(make="SONY"))
    assert len(pts) == 1
    p = pts[0]
    w, h, x, y = OFF_RAW
    assert (p.x, p.y) == pytest.approx((x / w, y / h))
    assert max(abs(p.x - 0.5), abs(p.y - 0.5)) >= 0.15  # clearly off-centre, in both axes
    assert min(abs(p.x - 0.5), abs(p.y - 0.5)) >= 0.15
    assert 0 < p.w < 0.1 and 0 < p.h < 0.1


def test_real_off_centre_point_lies_on_the_sharpest_tile_of_the_preview():
    buf = OFF.read_bytes()
    assert len(buf) == 512 * 1024
    (p,) = parse_af_points(buf)
    thumb = _thumb_gray(buf)
    assert _tile_rank(thumb, p.x, p.y) < 7  # top 10 % of the 64 tiles (independent check)
    # a wrong X/Y sign or axis swap would land elsewhere:
    assert _tile_rank(thumb, 1 - p.x, p.y) >= 7
    assert _tile_rank(thumb, p.x, 1 - p.y) >= 7
    assert _tile_rank(thumb, p.y, p.x) >= 7


@pytest.mark.parametrize("orientation", [1, 3, 6, 8])
def test_real_off_centre_point_follows_the_exif_orientation(orientation):
    """Store the thumbnail rotated as a camera would for this orientation tag; the oriented
    point must stay on the same (sharp) detail, i.e. the real data confirms the mapping."""
    from shotsieve.io.preview import apply_orientation

    buf = OFF.read_bytes()
    (p,) = parse_af_points(buf)
    thumb = _thumb_gray(buf)
    upright = apply_orientation(thumb, orientation)  # the preview as the viewer sees it
    q = to_oriented(p, orientation)
    sensor_rank = _tile_rank(thumb, p.x, p.y)
    assert _tile_rank(upright, q.x, q.y) <= sensor_rank + 2  # same neighbourhood, grid-shifted
