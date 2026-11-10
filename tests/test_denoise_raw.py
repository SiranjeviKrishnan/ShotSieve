# ruff: noqa: E501
"""T26 RawNIND (Bayer RAW denoiser): pack / unpack for every CFA order, tile seams, region == full crop,
orientation mapping, the NaN fallback, the colour match against the normal view, scene / cache plumbing.
The network is replaced by a deterministic stand-in (a nearest-neighbour demosaic of the pack) except in
the tests marked `models`, so this runs without the model file."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from shotsieve.io.preview import apply_orientation, load_full
from shotsieve.models import rawnind as rn

RAW = Path(__file__).parent / "fixtures" / "generated" / "raw"
PENTAX = RAW / "pentax_kx.dng"  # BGGR, odd width (4309), black level 0/9/1/9
PANA = RAW / "panasonic_lx1.dng"  # RGGB, odd width (3857)

needs_model = pytest.mark.skipif(not rn.available(), reason="RawNIND model not present")


def fake_tile(t: np.ndarray) -> np.ndarray:
    """A translation-equivariant stand-in for the network: nearest 2x demosaic (R, mean G, B)."""
    rgb = np.stack([t[0], (t[1] + t[2]) / 2, t[3]])
    return np.repeat(np.repeat(rgb, 2, axis=1), 2, axis=2).astype(np.float32)


def blur_tile(t: np.ndarray) -> np.ndarray:
    """Equivariant except within 2 px of the tile edge: nearest demosaic then a 5x5 box blur."""
    y = fake_tile(t)
    return np.stack([cv2.blur(c, (5, 5)) for c in y])


def whole(pack: np.ndarray, fn=fake_tile) -> np.ndarray:
    return fn(pack)


def collect(pack, fn, rect=None, **kw):
    """Concatenate the bands of iter_bands into (3, rows, cols) plus its first row."""
    bands = list(rn.iter_bands(pack, fn, rect, **kw))
    rows = [b for _, b in bands]
    ys = [y for y, _ in bands]
    assert all(ys[i] + rows[i].shape[1] == ys[i + 1] for i in range(len(ys) - 1)), (
        "bands must tile the rows"
    )
    return ys[0], np.concatenate(rows, axis=1)


# ---- CFA / pack ---------------------------------------------------------------------------------

PATTERNS = {  # libraw colour indices of the 2x2 cell: 0 R, 1 G, 2 B, 3 G2
    "RGGB": [[0, 1], [3, 2]],
    "BGGR": [[2, 3], [1, 0]],
    "GRBG": [[1, 0], [2, 3]],
    "GBRG": [[3, 2], [0, 1]],
}


def mosaic(pattern: str, H: int, W: int, seed=0):
    rng = np.random.default_rng(seed)
    colors = np.tile(np.array(PATTERNS[pattern]), (H // 2 + 1, W // 2 + 1))[:H, :W]
    img = rng.integers(100, 4000, (H, W)).astype(np.uint16)
    return img, colors


@pytest.mark.parametrize("pattern", list(PATTERNS))
@pytest.mark.parametrize("shape", [(40, 60), (41, 63)])
def test_pack_round_trip_for_every_cfa_order(pattern, shape):
    img, colors = mosaic(pattern, *shape)
    black, white = [64, 70, 80, 70], 4095.0
    pack, (oy, ox) = rn.pack_bayer(img, colors, black, white)
    h, w = pack.shape[1:]
    # the pack starts on an R site and a B site closes the cell
    assert colors[oy, ox] == 0 and colors[oy + 1, ox + 1] == 2
    assert (oy, ox) == {"RGGB": (0, 0), "BGGR": (1, 1), "GRBG": (0, 1), "GBRG": (1, 0)}[pattern]
    assert (2 * h, 2 * w) == (
        shape[0] - oy - (shape[0] - oy) % 2,
        shape[1] - ox - (shape[1] - ox) % 2,
    )
    # plane k holds exactly the sites of colour R, G, G2, B (greens either way round), normalised per colour
    for k, want in enumerate(((0,), (1, 3), (1, 3), (2,))):
        sites = colors[oy + k // 2 : oy + k // 2 + 2 * h : 2, ox + k % 2 : ox + k % 2 + 2 * w : 2]
        assert np.isin(sites, want).all()
    bl = np.array(black, np.float32)
    expect = np.clip((img.astype(np.float32) - bl[colors]) / (white - bl[colors]), 0, 1)
    back = rn.unpack_bayer(pack)
    assert np.allclose(back, expect[oy : oy + 2 * h, ox : ox + 2 * w], atol=1e-6)


def test_non_bayer_layouts_are_refused():
    xtrans = np.array(
        [
            [1, 1, 0, 1, 1, 2],
            [1, 1, 2, 1, 1, 0],
            [2, 0, 1, 0, 2, 1],
            [1, 1, 2, 1, 1, 0],
            [1, 1, 0, 1, 1, 2],
            [0, 2, 1, 2, 0, 1],
        ]
    )
    big = np.tile(xtrans, (4, 4))
    assert rn.bayer_offset(big) is None
    cmyg = np.tile(
        np.array([[0, 1], [2, 3]]), (8, 8)
    )  # no B on the diagonal of R? (R,G / B,G2): not a Bayer cell
    assert rn.bayer_offset(cmyg) is None
    assert rn.bayer_offset(np.zeros((8, 8), int)) is None
    with pytest.raises(ValueError):
        rn.pack_bayer(np.zeros((16, 16), np.uint16), big[:16, :16], [0] * 4, 255.0)


def test_fixtures_are_bayer_and_have_the_expected_layouts():
    import rawpy

    with rawpy.imread(str(PENTAX)) as r:
        assert rn.bayer_offset(r.raw_colors_visible) == (1, 1)
    with rawpy.imread(str(PANA)) as r:
        assert rn.bayer_offset(r.raw_colors_visible) == (0, 0)
    assert rn.supports(PENTAX) and rn.supports(PANA)
    assert not rn.supports(RAW.parent / "src" / "nothing.arw")  # missing file
    assert not rn.supports(RAW.parent / "src")  # not a RAW extension


# ---- tiling -------------------------------------------------------------------------------------


@pytest.mark.parametrize("hw", [(150, 170), (64, 64), (40, 100), (200, 61), (129, 130)])
def test_tile_seams_are_invisible_for_an_equivariant_model(hw):
    rng = np.random.default_rng(1)
    pack = rng.random((4, *hw), dtype=np.float32)
    y0, got = collect(pack, fake_tile, tile=64, margin=8)
    want = whole(pack)
    assert y0 == 0 and got.shape == want.shape
    assert np.abs(got - want).max() < 1e-5


def test_tiles_run_with_workers_and_report_progress():
    pack = np.random.default_rng(2).random((4, 140, 150), dtype=np.float32)
    seen = []
    _, got = collect(pack, fake_tile, tile=64, margin=8, workers=3, progress=seen.append)
    assert np.abs(got - whole(pack)).max() < 1e-5
    assert seen and seen[-1] == pytest.approx(1.0) and seen == sorted(seen)


def test_cancel_and_pause_hooks():
    pack = np.zeros((4, 140, 150), np.float32)
    calls = []
    with pytest.raises(rn.Cancelled):
        list(rn.iter_bands(pack, fake_tile, tile=64, margin=8, cancel=lambda: True))
    list(rn.iter_bands(pack, fake_tile, tile=64, margin=8, pause=lambda: calls.append(1)))
    assert calls


def test_blend_weights_sum_to_one_with_a_blurring_model():
    """With a model that is wrong near tile edges the blend stays close to the true result."""
    rng = np.random.default_rng(3)
    base = cv2.GaussianBlur(rng.random((300, 340), dtype=np.float32), (0, 0), 3)
    pack = np.stack([base] * 4).astype(np.float32)
    _, got = collect(pack, blur_tile, tile=64, margin=8)
    want = whole(pack, blur_tile)
    inner = (slice(None), slice(8, -8), slice(8, -8))
    assert np.abs(got[inner] - want[inner]).max() < 0.02


@pytest.mark.parametrize(
    "rect",
    [
        (100, 140, 120, 200),
        (0, 30, 0, 30),
        (150, 200, 210, 250),
        (60, 62, 70, 72),
        (0, 200, 0, 250),
    ],
)
def test_region_equals_the_crop_of_the_full_image(rect):
    rng = np.random.default_rng(4)
    pack = rng.random((4, 200, 250), dtype=np.float32)
    py0, py1, px0, px1 = rect
    y0, got = collect(pack, fake_tile, rect, tile=64, margin=8)
    want = whole(pack)[:, 2 * py0 : 2 * py1, 2 * px0 : 2 * px1]
    assert y0 == 2 * py0 and got.shape == want.shape
    assert np.abs(got - want).max() < 1e-5
    # and with a model that is not exactly equivariant: the margin keeps the difference small
    base = cv2.GaussianBlur(rng.random((200, 250), dtype=np.float32), (0, 0), 3)
    pack2 = np.stack([base] * 4).astype(np.float32)
    _, got2 = collect(pack2, blur_tile, rect, tile=64, margin=8)
    full2 = whole(pack2, blur_tile)[:, 2 * py0 : 2 * py1, 2 * px0 : 2 * px1]
    assert np.abs(got2 - full2).max() < 0.03


# ---- display frame -------------------------------------------------------------------------------


@pytest.mark.parametrize("orient", range(1, 9))
def test_sensor_box_inverts_the_orientation(orient):
    H, W = 30, 50
    idx = np.arange(H * W).reshape(H, W)
    oriented = apply_orientation(idx, orient)
    Ho, Wo = oriented.shape
    for u0, v0, u1, v1 in [
        (0, 0, Wo, Ho),
        (3, 5, 20, 11),
        (Wo - 7, Ho - 4, Wo, Ho),
        (10, 2, 11, 3),
    ]:
        ys, xs = np.divmod(oriented[v0:v1, u0:u1].ravel(), W)
        assert rn.sensor_box(orient, (u0, v0, u1, v1), H, W) == (
            ys.min(),
            ys.max() + 1,
            xs.min(),
            xs.max() + 1,
        )


def test_frame_registration_rules():
    f = rn.make_frame(1, (4024, 6024), (4000, 6000))  # a camera crops 12 px per side
    assert (f.cx, f.cy) == (12, 12)
    f = rn.make_frame(6, (4024, 6024), (6000, 4000))  # rotated: displayed 4000 wide x 6000 high
    assert (f.cx, f.cy) == (12, 12)
    f = rn.make_frame(
        1, (4024, 6024), (3376, 6000)
    )  # a 16:9 JPEG of a 3:2 sensor: aspect crop first
    assert f.cy == (4024 - 3376) // 2
    with pytest.raises(rn.FrameMismatch):
        rn.make_frame(1, (4024, 6024), (2000, 3000))  # a half-size picture does not register
    with pytest.raises(rn.FrameMismatch):
        rn.make_frame(1, (400, 600), (500, 700))  # bigger than the sensor


# ---- model session: NaN / range fallback ----------------------------------------------------------


class FakeSession:
    def __init__(self, fn, provider="coreml"):
        self.fn, self.provider, self.calls = fn, provider, 0

    def get_inputs(self):
        class I:  # noqa: E742
            name = "input"

        return [I()]

    def run(self, _out, feed):
        self.calls += 1
        t = feed["input"][0]
        return [self.fn(t)[None]]


def test_nan_tile_falls_back_to_the_cpu_provider_and_is_remembered(monkeypatch):
    monkeypatch.setattr(rn, "_cpu_only", False)
    bad = FakeSession(
        lambda t: np.full((3, 2 * t.shape[1], 2 * t.shape[2]), np.nan, np.float32), "coreml"
    )
    good = FakeSession(fake_tile, "cpu")
    net = rn._Net(sessions=[bad, good])
    net._path, net._cpu_opts = "x", None
    assert net.provider == "coreml"
    y = net.tile(np.random.default_rng(0).random((4, 16, 16), dtype=np.float32))
    assert np.isfinite(y).all() and y.shape == (3, 32, 32)
    assert net.provider == "cpu" and net.sessions == [good] and rn._cpu_only is True
    assert bad.calls == 1 and good.calls == 1
    net.tile(np.zeros((4, 16, 16), np.float32))
    assert bad.calls == 1  # not asked again


def test_out_of_range_tile_counts_as_a_failure_and_a_cpu_failure_raises(monkeypatch):
    monkeypatch.setattr(rn, "_cpu_only", False)
    huge = FakeSession(
        lambda t: np.full((3, 2 * t.shape[1], 2 * t.shape[2]), 5e9, np.float32), "coreml"
    )
    ok = FakeSession(fake_tile, "cpu")
    net = rn._Net(sessions=[huge, ok])
    net._path, net._cpu_opts = "x", None
    assert np.abs(net.tile(np.full((4, 8, 8), 0.5, np.float32))).max() <= 1.0
    clipped = rn._Net(
        sessions=[FakeSession(lambda t: np.full((3, 16, 16), 65504.0, np.float32), "coreml"), ok]
    )
    clipped._path, clipped._cpu_opts = "x", None
    assert clipped.tile(np.full((4, 8, 8), 0.5, np.float32)).max() <= 1.0  # fp16 flat-line: refused
    monkeypatch.setattr(rn, "_cpu_only", False)
    allbad = rn._Net(
        sessions=[FakeSession(lambda t: np.full((3, 16, 16), np.inf, np.float32), "cpu")]
    )
    with pytest.raises(RuntimeError):
        allbad.tile(np.zeros((4, 8, 8), np.float32))
    monkeypatch.setattr(rn, "_cpu_only", False)


# ---- the scene, gain and colour against the normal view -------------------------------------------


@pytest.fixture(autouse=True)
def _fresh():
    rn.release_scene()
    rn._PARAMS.clear()
    yield
    rn.release_scene()
    rn._PARAMS.clear()


@pytest.mark.parametrize("fixture", [PENTAX, PANA], ids=["bggr_pentax", "rggb_panasonic"])
def test_scene_pack_matches_the_file(fixture):
    sc = rn.scene_for(fixture)
    H, W = sc.sensor_hw
    h, w = sc.pack.shape[1:]
    assert 2 * h <= H and 2 * w <= W and H - 2 * h <= 2 and W - 2 * w <= 2
    assert (
        sc.pack.dtype == np.float32 and 0.0 <= float(sc.pack.min()) and float(sc.pack.max()) <= 1.0
    )
    assert sc.M.shape == (3, 3) and np.isfinite(sc.M).all()
    assert rn.scene_for(fixture) is sc  # kept for the next region request


@pytest.mark.parametrize("fixture", [PENTAX, PANA], ids=["bggr_pentax", "rggb_panasonic"])
def test_denoised_render_matches_the_normal_view_in_colour_and_brightness(fixture):
    off = load_full(fixture)
    den = rn.render(fixture, off, run_tile=fake_tile)
    assert den.shape == off.shape and den.dtype == np.uint8
    mo, md = off.reshape(-1, 3).mean(0), den.reshape(-1, 3).mean(0)
    assert np.abs(mo - md).max() < 4.0, (mo, md)  # mean colour within 4 / 255 per channel
    lo = off.astype(np.float32) @ [0.299, 0.587, 0.114]
    ld = den.astype(np.float32) @ [0.299, 0.587, 0.114]
    assert abs(lo.mean() - ld.mean()) / max(lo.mean(), 1) < 0.03  # luminance within 3 %
    # the picture itself, not only its mean: structure follows the normal view (blurred correlation)
    a = cv2.GaussianBlur(lo, (0, 0), 6)
    b = cv2.GaussianBlur(ld, (0, 0), 6)
    assert np.corrcoef(a.ravel(), b.ravel())[0, 1] > 0.97


def test_region_render_is_the_crop_of_the_full_render():
    off = load_full(PANA)
    full = rn.render(PANA, off, run_tile=fake_tile)
    h, w = off.shape[:2]
    for box in [
        (0, 0, 300, 200),
        (w // 3, h // 4, w // 3 + 701, h // 4 + 333),
        (w - 257, h - 129, w, h),
        (11, 13, 12, 14),
    ]:
        part = rn.render(PANA, off, box, run_tile=fake_tile)
        x0, y0, x1, y1 = box
        want = full[y0:y1, x0:x1]
        assert part.shape == want.shape, box
        assert np.abs(part.astype(int) - want.astype(int)).max() <= 1, box


def test_region_render_on_a_bggr_file_with_the_one_pixel_offset():
    off = load_full(PENTAX)
    full = rn.render(PENTAX, off, run_tile=fake_tile)
    h, w = off.shape[:2]
    for box in [(0, 0, 100, 90), (w - 80, h - 70, w, h), (500, 400, 1100, 900)]:
        part = rn.render(PENTAX, off, box, run_tile=fake_tile)
        x0, y0, x1, y1 = box
        assert np.abs(part.astype(int) - full[y0:y1, x0:x1].astype(int)).max() <= 1, box


def test_one_gain_per_image_is_shared_by_every_request(monkeypatch):
    off = load_full(PANA)
    calls = []

    def counting(t):
        calls.append(1)
        return fake_tile(t)

    rn.render(PANA, off, (0, 0, 64, 64), run_tile=counting)
    n_first = len(calls)
    rn.render(PANA, off, (100, 100, 200, 200), run_tile=counting)
    # the second request found gain + tone curve cached: only its own tiles ran, no probe tiles
    assert len(calls) - n_first <= 2 < n_first
    g = rn._PARAMS[(str(PANA), PANA.stat().st_mtime_ns, off.shape[:2])].gain
    assert g == pytest.approx(
        1.0, rel=1e-3
    )  # the nearest-demosaic stand-in already has the input's mean


def test_a_pixel_offset_changes_nothing_but_the_frame():
    """A box at the very edge of a BGGR sensor (row / column 0 has no output pixel) is filled, not garbage."""
    off = load_full(PENTAX)
    part = rn.render(PENTAX, off, (0, 0, 8, 8), run_tile=fake_tile)
    assert part.shape == (8, 8, 3) and part.max() > 0
    ref = rn.render(PENTAX, off, (0, 0, off.shape[1], off.shape[0]), run_tile=fake_tile)[:8, :8]
    assert np.abs(part.astype(int) - ref.astype(int)).max() <= 1


def test_supports_checks_the_displayed_frame():
    off = load_full(PENTAX)
    assert rn.supports(PENTAX, off.shape[:2])
    assert not rn.supports(PENTAX, (100, 100))


def test_available_respects_the_backend_switch(monkeypatch):
    monkeypatch.setenv("SHOTSIEVE_DENOISE_BACKEND", "nafnet")
    assert not rn.available()
    monkeypatch.setenv("SHOTSIEVE_DENOISE_BACKEND", "basic")
    assert not rn.available()


# ---- with the real network -------------------------------------------------------------------------


@needs_model
@pytest.mark.models
def test_real_network_denoises_a_noisy_raw_and_keeps_colour(tmp_path):
    off = load_full(PENTAX)
    den = rn.render(PENTAX, off)
    assert den.shape == off.shape and np.isfinite(den.astype(np.float32)).all()
    mo, md = off.reshape(-1, 3).mean(0), den.reshape(-1, 3).mean(0)
    assert np.abs(mo - md).max() < 6.0
    rn.release()


@needs_model
@pytest.mark.models
def test_real_network_region_matches_the_full_render_closely():
    off = load_full(PANA)
    full = rn.render(PANA, off)
    h, w = off.shape[:2]
    box = (w // 4, h // 4, w // 4 + 500, h // 4 + 400)
    part = rn.render(PANA, off, box)
    want = full[box[1] : box[3], box[0] : box[2]].astype(int)
    assert np.abs(part.astype(int) - want).mean() < 1.0
    rn.release()
