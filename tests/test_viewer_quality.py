# ruff: noqa: E501
"""T23f viewer quality: the picture the loupe shows is rendered at the size the screen needs, at
q >= 90 / 4:4:4, colour-managed to sRGB with a profile, correctly oriented and as sharp as a
high-quality downscale of the original. The cull's analysis path (`load_preview`) is not involved."""

from __future__ import annotations

import io
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageCms, JpegImagePlugin

from shotsieve.config import Settings
from shotsieve.io import viewer
from shotsieve.io.preview import load_preview
from shotsieve.pipeline.cache import Cache
from shotsieve.server.app import create_app
from tests.test_api import _row, shoot_of

P3_ICC = Path("/System/Library/ColorSync/Profiles/Display P3.icc")


def _chart(w=3000, h=2000) -> np.ndarray:
    """Sharp test chart: checkerboards of several periods and a siemens-like star."""
    img = np.full((h, w, 3), 128, np.uint8)
    b = min(w // 5, h // 3)
    for k, period in enumerate((4, 8, 16, 32)):
        x0 = b // 5 + k * (b + b // 5)
        yy, xx = np.mgrid[0:b, 0:b]
        board = (((yy // period) + (xx // period)) % 2) * 255
        img[b // 5 : b // 5 + b, x0 : x0 + b] = board[..., None].astype(np.uint8)
    cx, cy, r = w // 2, h - h // 3, min(w, h) // 4
    yy, xx = np.mgrid[0:h, 0:w]
    ang = np.arctan2(yy - cy, xx - cx)
    star = ((np.floor(ang / (np.pi / 36)) % 2) * 255).astype(np.uint8)
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r**2
    img[mask] = star[mask][:, None]
    return img


@pytest.fixture()
def shoot(tmp_path, gen_dir):
    folder = tmp_path / "shoot"
    folder.mkdir()
    Image.fromarray(_chart()).save(folder / "chart.jpg", quality=98, subsampling=0)
    shutil.copy(gen_dir / "misc" / "orient6.jpg", folder / "orient6.jpg")
    shutil.copy(gen_dir / "raw" / "pentax_kx.dng", folder / "pentax.dng")
    shutil.copy(gen_dir / "raw" / "panasonic_lx1.dng", folder / "pana.dng")
    small = np.full((200, 300, 3), (30, 90, 200), np.uint8)
    Image.fromarray(small).save(folder / "small.jpg", quality=95)
    names = ["chart.jpg", "orient6.jpg", "pentax.dng", "pana.dng", "small.jpg"]
    if P3_ICC.exists():  # a Display P3 JPEG with one very saturated, known colour
        p3 = np.full((600, 900, 3), (230, 60, 40), np.uint8)
        Image.fromarray(p3).save(folder / "p3.jpg", quality=95, icc_profile=P3_ICC.read_bytes())
        names.append("p3.jpg")
    rows = [_row(folder / n, "keep", i, True) for i, n in enumerate(names)]
    with Cache.for_folder(folder) as c:
        c.save_decisions(rows)
    return folder


@pytest.fixture()
def client(shoot):
    with TestClient(create_app(Settings(), default_folder=shoot)) as c:
        yield c


@pytest.fixture()
def ids(client, shoot):
    return {im["name"]: im["id"] for im in shoot_of(client, shoot)[0]["images"]}


def _pil(r) -> Image.Image:
    return Image.open(io.BytesIO(r.content))


def _luma(rgb) -> np.ndarray:
    return cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2GRAY)


def _lapvar(rgb) -> float:
    return float(cv2.Laplacian(_luma(rgb), cv2.CV_64F).var())


@pytest.mark.parametrize(
    ("w", "edge"), [(900, 1280), (1280, 1280), (1500, 2048), (2500, 2880), (3500, 3840)]
)
def test_served_size_matches_the_requested_bucket(client, ids, w, edge):
    r = client.get(f"/api/preview/{ids['pentax.dng']}", params={"w": w})
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    im = _pil(r)
    assert max(im.size) == edge, im.size
    assert abs(im.size[0] / im.size[1] - 4288 / 2848) < 0.003  # same frame, not stretched


def test_above_the_top_bucket_serves_the_native_image_and_never_upscales(client, ids):
    r = client.get(f"/api/preview/{ids['pentax.dng']}", params={"w": 6000})
    assert _pil(r).size == (4288, 2848)
    small = _pil(client.get(f"/api/preview/{ids['small.jpg']}", params={"w": 3840}))
    assert small.size == (300, 200)  # smaller than every bucket: native, not enlarged
    assert max(_pil(client.get(f"/api/preview/{ids['chart.jpg']}")).size) == 2048  # default bucket


def test_box_request_works_out_the_edge_from_the_image_aspect(client, ids):
    # a 3000x2000 chart in a 1400x1400 px box needs a 1400 px long edge -> 2048; in 1200x600: 900 -> 1280
    a = _pil(client.get(f"/api/preview/{ids['chart.jpg']}", params={"bw": 1400, "bh": 1400}))
    b = _pil(client.get(f"/api/preview/{ids['chart.jpg']}", params={"bw": 1200, "bh": 600}))
    assert max(a.size) == 2048 and max(b.size) == 1280
    # portrait: orient6 shows 600x400 (landscape after rotation) in a 3440x1440 px box: native size
    c = _pil(client.get(f"/api/preview/{ids['orient6.jpg']}", params={"bw": 3440, "bh": 1440}))
    assert c.size == (600, 400)


@pytest.mark.parametrize("name", ["chart.jpg", "pentax.dng", "pana.dng", "orient6.jpg"])
def test_quality_subsampling_and_srgb_profile(client, ids, name):
    r = client.get(f"/api/preview/{ids[name]}", params={"w": 1280})
    assert viewer.jpeg_quality(r.content) >= 90
    im = _pil(r)
    assert JpegImagePlugin.get_sampling(im) == 0  # 4:4:4: no chroma smearing
    desc = viewer.jpeg_icc_description(r.content)
    assert desc and "srgb" in desc.lower()
    full = client.get(f"/api/full/{ids[name]}")
    assert "srgb" in (viewer.jpeg_icc_description(full.content) or "").lower()
    assert viewer.jpeg_quality(full.content) >= 90


def test_orientation_is_applied_once(client, ids, shoot):
    r = client.get(f"/api/preview/{ids['orient6.jpg']}", params={"w": 2048})
    got = np.asarray(_pil(r).convert("RGB"))
    assert got.shape[:2] == (400, 600)  # stored 600x400 portrait, displayed landscape
    ref = np.asarray(Image.open(shoot / "orient6.jpg").transpose(Image.ROTATE_270).convert("RGB"))
    assert ref.shape == got.shape and np.abs(ref.astype(int) - got.astype(int)).mean() < 6
    full = client.get(f"/api/full/{ids['orient6.jpg']}")
    fa = np.asarray(_pil(full).convert("RGB"))
    assert np.abs(fa.astype(int) - got.astype(int)).mean() < 6


def test_downscale_keeps_a_sharp_chart_sharp(client, ids, shoot):
    r = client.get(f"/api/preview/{ids['chart.jpg']}", params={"w": 1280})
    got = np.asarray(_pil(r).convert("RGB"))
    assert got.shape[1] == 1280
    src = Image.open(shoot / "chart.jpg").convert("RGB")
    ref = np.asarray(src.resize((got.shape[1], got.shape[0]), Image.LANCZOS))
    assert _lapvar(got) >= 0.9 * _lapvar(ref)  # as sharp as a Lanczos downscale of the original
    # and clearly sharper than the old 2-step path (INTER_AREA, q88, then a browser resize)
    legacy = load_preview(shoot / "chart.jpg", 1600)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(legacy, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 88])  # fmt: skip
    old = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    old = cv2.resize(old, (got.shape[1], got.shape[0]), interpolation=cv2.INTER_LINEAR)
    assert _lapvar(got) > 1.15 * _lapvar(old)
    mse = np.mean((got.astype(float) - ref.astype(float)) ** 2)
    assert 10 * np.log10(255**2 / mse) > 30  # PSNR vs the reference


@pytest.mark.skipif(not P3_ICC.exists(), reason="no Display P3 profile on this system")
def test_wide_gamut_file_is_converted_to_srgb_with_its_profile(client, ids):
    got = np.asarray(_pil(client.get(f"/api/preview/{ids['p3.jpg']}")).convert("RGB"))
    p3 = ImageCms.ImageCmsProfile(str(P3_ICC))
    srgb = ImageCms.createProfile("sRGB")
    exp = ImageCms.profileToProfile(
        Image.new("RGB", (1, 1), (230, 60, 40)), p3, srgb, outputMode="RGB"
    ).getpixel((0, 0))  # fmt: skip
    mid = got[got.shape[0] // 2, got.shape[1] // 2]
    assert np.abs(mid.astype(int) - np.array(exp)).max() <= 4, (mid, exp)
    assert abs(int(mid[1]) - 60) > 10  # i.e. it really changed: P3 (230,60,40) is not sRGB (230,60,40)


def test_raw_demosaic_is_in_the_frame_of_the_embedded_jpeg_and_of_full(client, ids):
    # pana.dng has no JPEG thumb (demosaic); pentax.dng has a full-size embedded JPEG
    for name in ("pana.dng", "pentax.dng"):
        iid = ids[name]
        size = client.get(f"/api/full/{iid}/size").json()
        p = _pil(client.get(f"/api/preview/{iid}", params={"w": 2048}))
        assert abs(p.size[0] / p.size[1] - size["w"] / size["h"]) < 0.005, name


def test_tone_matching_pulls_a_flat_demosaic_towards_the_camera_look():
    rng = np.random.default_rng(1)
    base = rng.integers(0, 256, (80, 120, 3)).astype(np.float64)
    flat = (base * 0.6 + 30).astype(np.uint8)  # low contrast
    ref = np.clip((base - 128) * 1.2 + 128, 0, 255).astype(np.uint8)  # punchy "camera" look
    out = cv2.LUT(flat, viewer.match_tone(flat, ref))
    assert np.abs(out.astype(int) - ref.astype(int)).mean() < 0.5 * np.abs(
        flat.astype(int) - ref.astype(int)
    ).mean()


def test_demosaic_crop_follows_the_camera_frame():
    img = np.zeros((4024, 6024, 3), np.uint8)
    out = viewer.center_crop_to_aspect(img, 16 / 9)
    assert out.shape[:2] == (3388, 6024)  # centred 16:9 band of a 3:2 sensor
    assert viewer.crop_dims(6000, 4000, 1.5) == (6000, 4000)  # already that aspect: untouched


def test_jpeg_quality_estimator_roundtrip():
    rgb = np.random.default_rng(0).integers(0, 255, (64, 64, 3), np.uint8)
    for q in (60, 75, 88, 92, 95):
        assert abs(viewer.jpeg_quality(viewer.encode_jpeg(rgb, q)) - q) <= 1


def test_originals_untouched_and_cache_is_versioned(client, ids, shoot):
    before = {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in shoot.iterdir() if p.is_file()}
    client.get(f"/api/preview/{ids['chart.jpg']}", params={"w": 1280})
    client.get(f"/api/thumb/{ids['chart.jpg']}", params={"size": 400})
    after = {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in shoot.iterdir() if p.is_file()}
    assert before == after
    cached = list((shoot / ".shotsieve" / "thumbs").glob("v_*_1280_*_v2.jpg"))
    assert len(cached) == 1
