# ruff: noqa: E501
"""T26 RawNIND through /api/denoise/*: the engine is chosen per image (Bayer RAW -> rawnind, everything
else NAFNet / basic), the cache key carries the backend id and version, Fit / region / full agree, a
failing model falls back, originals stay untouched. The network is a stand-in (see test_denoise_raw)."""

from __future__ import annotations

import hashlib
import io
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from shotsieve.config import Settings
from shotsieve.models import denoise as engine
from shotsieve.models import rawnind
from shotsieve.models.idle import IdleLoaded
from shotsieve.pipeline.cache import Cache
from shotsieve.server import denoise as dsrv
from shotsieve.server.app import create_app
from tests.test_api import _row, shoot_of
from tests.test_denoise_raw import PANA, PENTAX, fake_tile

pytestmark = pytest.mark.filterwarnings("ignore")


class FakeNet:
    provider = "cpu"

    def tile(self, t):
        return fake_tile(t)


@pytest.fixture()
def shoot(tmp_path, gen_dir):
    folder = tmp_path / "shoot"
    folder.mkdir()
    shutil.copy(PENTAX, folder / "bggr.dng")
    shutil.copy(PANA, folder / "rggb.dng")
    src = Image.open(gen_dir / "shoot" / "landscape.jpg").convert("RGB").resize((1200, 800))
    src.save(folder / "plain.jpg", quality=95)
    names = ["bggr.dng", "rggb.dng", "plain.jpg"]
    with Cache.for_folder(folder) as c:
        c.save_decisions([_row(folder / n, "keep", i, True) for i, n in enumerate(names)])
    return folder


@pytest.fixture()
def client(shoot, monkeypatch):
    monkeypatch.setattr(rawnind, "available", lambda: True)
    monkeypatch.setattr(rawnind, "_holder", IdleLoaded(FakeNet, 60.0))
    rawnind.release_scene()
    with TestClient(create_app(Settings(), default_folder=shoot)) as c:
        yield c
    dsrv.release_all()
    monkeypatch.undo()
    rawnind.release()


@pytest.fixture()
def ids(client, shoot):
    return {im["name"]: im["id"] for im in shoot_of(client, shoot)[0]["images"]}


def _img(r) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"))


def _tree(folder: Path) -> dict[str, str]:
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() + str(p.stat().st_mtime_ns)
        for p in sorted(folder.iterdir())
        if p.is_file()
    }


def test_status_names_the_engine_per_image(client, ids):
    d = client.get("/api/denoise/status").json()
    assert d["raw_model"]["name"] == "rawnind_utnet2" and d["raw_model"]["license"] == "CC-BY-4.0"
    assert d["raw_model"]["installed"] is True and d["raw_model"]["group"] == "rawdenoise"
    assert (
        client.get("/api/denoise/status", params={"image_id": ids["bggr.dng"]}).json()["engine"]
        == "rawnind"
    )
    assert (
        client.get("/api/denoise/status", params={"image_id": ids["rggb.dng"]}).json()["engine"]
        == "rawnind"
    )
    other = client.get("/api/denoise/status", params={"image_id": ids["plain.jpg"]}).json()[
        "engine"
    ]
    assert other in ("ai", "basic")  # a JPEG never gets RawNIND


def test_a_raw_never_needs_the_nafnet_download(client, ids, monkeypatch):
    monkeypatch.setattr(engine, "available", lambda: False)  # NAFNet not downloaded
    st = client.get("/api/denoise/status", params={"image_id": ids["rggb.dng"]}).json()
    assert st["engine"] == "rawnind" and st["model"]["installed"] is False
    r = client.get(f"/api/denoise/{ids['rggb.dng']}", params={"w": 1280})
    assert r.status_code == 200 and r.headers["x-denoise-backend"] == "rawnind"
    # ... while a JPEG without NAFNet is Basic, as before
    assert (
        client.get("/api/denoise/status", params={"image_id": ids["plain.jpg"]}).json()["engine"]
        == "basic"
    )


def test_without_the_bundled_model_a_raw_falls_back_to_the_old_engines(client, ids, monkeypatch):
    monkeypatch.setattr(rawnind, "available", lambda: False)
    monkeypatch.setattr(engine, "available", lambda: False)
    assert (
        client.get("/api/denoise/status", params={"image_id": ids["rggb.dng"]}).json()["engine"]
        == "basic"
    )
    r = client.get(f"/api/denoise/{ids['rggb.dng']}", params={"w": 1280})
    assert r.status_code == 200 and r.headers["x-denoise-backend"] == "basic"


def test_non_bayer_raw_keeps_nafnet(client, ids, monkeypatch):
    monkeypatch.setattr(rawnind, "supports", lambda *a, **k: False)  # X-Trans, Foveon, a linear DNG
    st = client.get("/api/denoise/status", params={"image_id": ids["rggb.dng"]}).json()
    assert st["engine"] in ("ai", "basic")


def test_fit_view_cache_key_has_backend_id_and_versions_and_is_reused(
    client, ids, shoot, monkeypatch
):
    before = _tree(shoot)
    calls = []
    real = rawnind.render
    monkeypatch.setattr(rawnind, "render", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    r = client.get(f"/api/denoise/{ids['bggr.dng']}", params={"w": 1280})
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert r.headers["x-denoise-backend"] == "rawnind"
    files = list((shoot / ".shotsieve" / "denoise").glob("v_*.jpg"))
    assert len(files) == 1
    assert f"_v{engine.VERSION}_medium_rawnind{rawnind.VERSION}_" in files[0].name
    assert engine.VERSION >= 3 and rawnind.VERSION >= 1
    plain = _img(client.get(f"/api/preview/{ids['bggr.dng']}", params={"w": 1280}))
    out = _img(r)
    assert out.shape == plain.shape  # same frame as the picture it replaces
    again = client.get(f"/api/denoise/{ids['bggr.dng']}", params={"w": 1280})
    assert again.content == r.content and len(calls) == 1  # served from .shotsieve/denoise/
    # another strength re-blends the SAME model render: no second model run
    hi = client.get(f"/api/denoise/{ids['bggr.dng']}", params={"w": 1280, "level": "high"})
    assert hi.status_code == 200 and hi.content != r.content and len(calls) == 1
    after = _tree(shoot)
    assert {k: v for k, v in after.items() if k in before} == before and set(after) == set(before)
    assert not list(shoot.glob("*.xmp"))


def test_strength_levels_are_monotonic(client, ids):
    """Low keeps most of the (noisier) base, High is the pure RawNIND render: the distance to High
    shrinks Low > Medium > High = 0."""
    i = ids["rggb.dng"]
    out = {
        lv: _img(client.get(f"/api/denoise/{i}", params={"w": 1280, "level": lv})).astype(np.float32)
        for lv in ("low", "medium", "high")
    }
    d = {lv: float(np.abs(out[lv] - out["high"]).mean()) for lv in ("low", "medium")}
    assert d["low"] > d["medium"] > 0, d


def test_region_is_the_crop_of_the_full_image(client, ids):
    i = ids["rggb.dng"]
    r = client.get(
        f"/api/denoise/{i}/region",
        params={"x0": 0.25, "y0": 0.25, "x1": 0.5, "y1": 0.5, "level": "high"},
    )
    assert r.status_code == 200 and r.headers["x-denoise-backend"] == "rawnind"
    x0, y0, x1, y1 = (int(v) for v in r.headers["x-region"].split(","))
    reg = _img(r)
    assert reg.shape[:2] == (y1 - y0, x1 - x0)
    full = client.get(f"/api/denoise/{i}/full", params={"level": "high"})
    assert full.status_code == 200 and full.headers["x-denoise-backend"] == "rawnind"
    crop = _img(full)[y0:y1, x0:x1]
    assert crop.shape == reg.shape
    assert np.abs(crop.astype(int) - reg.astype(int)).mean() < 2.0  # two JPEG encodes apart


def test_a_failing_model_falls_back_and_is_filed_under_the_engine_that_made_it(
    client, ids, shoot, monkeypatch
):
    def boom(*a, **k):
        raise RuntimeError("model blew up")

    monkeypatch.setattr(rawnind, "render", boom)
    r = client.get(f"/api/denoise/{ids['rggb.dng']}", params={"w": 1280})
    assert r.status_code == 200 and r.headers["x-denoise-backend"] in ("ai", "basic")
    names = [p.name for p in (shoot / ".shotsieve" / "denoise").glob("v_*.jpg")]
    assert names and not any("rawnind" in n for n in names)


def test_warm_warms_rawnind_too(client, monkeypatch):
    calls = []
    monkeypatch.setattr(rawnind, "warm", lambda: calls.append("raw") or True)
    monkeypatch.setattr(rawnind, "loaded", lambda: False)
    monkeypatch.setattr(engine, "available", lambda: False)
    r = client.post("/api/denoise/warm")
    assert r.status_code == 202 and r.json()["warming"] is True
    for _ in range(100):
        if calls:
            break
        time.sleep(0.05)
    assert calls == ["raw"]
    monkeypatch.setattr(rawnind, "loaded", lambda: True)
    time.sleep(0.2)
    assert client.post("/api/denoise/warm").json() == {"warming": False, "reason": "loaded"}


def test_a_raw_plus_jpeg_pair_is_denoised_as_the_jpeg(tmp_path, gen_dir, monkeypatch):
    """The viewer's picture of a RAW+JPEG pair is the JPEG: that is what is denoised (NAFNet / basic)."""
    folder = tmp_path / "pair"
    folder.mkdir()
    shutil.copy(PANA, folder / "p1.dng")
    Image.open(gen_dir / "shoot" / "landscape.jpg").convert("RGB").resize((1200, 800)).save(
        folder / "p1.jpg", quality=95
    )
    from shotsieve.io.formats import preview_source_for

    if preview_source_for(folder / "p1.dng").suffix.lower() != ".jpg":
        pytest.skip("pair detection does not pair these names")
    monkeypatch.setattr(rawnind, "available", lambda: True)
    with Cache.for_folder(folder) as c:
        c.save_decisions([_row(folder / "p1.dng", "keep", 0, True)])
    with TestClient(create_app(Settings(), default_folder=folder)) as c:
        i = shoot_of(c, folder)[0]["images"][0]["id"]
        assert c.get("/api/denoise/status", params={"image_id": i}).json()["engine"] in (
            "ai",
            "basic",
        )
    dsrv.release_all()


def test_a_zoom_region_never_deadlocks_with_a_running_raw_render(client, ids, monkeypatch):
    """The whole-image RawNIND render pauses between tiles for a waiting region request; the region
    request must not need anything the render holds (it used to wait for the same lock: both hung)."""

    class Slow(FakeNet):
        def tile(self, t):
            time.sleep(0.12)
            return fake_tile(t)

    monkeypatch.setattr(rawnind, "_holder", IdleLoaded(Slow, 60.0))
    i = ids["rggb.dng"]
    done = {}

    def run_full():
        done["full"] = client.get(f"/api/denoise/{i}/full", params={"level": "high"})

    th = threading.Thread(target=run_full)
    th.start()
    time.sleep(0.8)  # the render is under way (it holds the whole-image job lock)
    assert th.is_alive()
    t0 = time.time()
    r = client.get(
        f"/api/denoise/{i}/region",
        params={"x0": 0.1, "y0": 0.1, "x1": 0.3, "y1": 0.3, "level": "high"},
    )
    assert r.status_code == 200 and r.headers["x-denoise-backend"] == "rawnind"
    assert time.time() - t0 < 20  # answered while the full render was still going, not after it
    th.join(120)
    assert not th.is_alive() and done["full"].status_code == 200


# ---- the blend base has the geometry of the RawNIND render, not of a lens-corrected picture ---------


def _warp(img: np.ndarray, px: float) -> np.ndarray:
    """A radial remap: no shift in the centre, `px` pixels at the corners (an in-camera distortion
    correction applied to the picture the viewer shows without denoising)."""
    import cv2

    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w - 1) / 2, (h - 1) / 2
    dx, dy = xx - cx, yy - cy
    r2 = (dx * dx + dy * dy) / (cx * cx + cy * cy)
    k = 1.0 + (px / np.sqrt(cx * cx + cy * cy)) * r2
    return cv2.remap(
        img,
        (cx + dx * k).astype(np.float32),
        (cy + dy * k).astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def _corner_shift(a: np.ndarray, b: np.ndarray) -> float:
    import cv2

    la = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float32)
    n = 256
    worst = 0.0
    for sl in ((slice(0, n), slice(0, n)), (slice(-n, None), slice(-n, None))):
        pa, pb = cv2.GaussianBlur(la[sl], (0, 0), 1.0), cv2.GaussianBlur(lb[sl], (0, 0), 1.0)
        (dx, dy), _ = cv2.phaseCorrelate(pa, pb, cv2.createHanningWindow((n, n), cv2.CV_32F))
        worst = max(worst, float(np.hypot(dx, dy)))
    return worst


def test_low_and_medium_blend_against_our_own_demosaic_not_a_warped_picture(
    client, ids, monkeypatch
):
    """If the picture shown without denoising is distortion-corrected (a few px at the corners) the
    Low / Medium blend must not mix that geometry with RawNIND's: no doubled edges."""
    real = dsrv.load_full
    monkeypatch.setattr(dsrv, "load_full", lambda p: _warp(real(p), 6.0))
    i = ids["rggb.dng"]
    out = {
        lv: _img(client.get(f"/api/denoise/{i}/full", params={"level": lv}))
        for lv in ("low", "medium", "high")
    }
    assert _corner_shift(out["low"], out["high"]) < 0.8
    assert _corner_shift(out["medium"], out["high"]) < 0.8
    # a region of the same photo is blended against the same base
    r = client.get(
        f"/api/denoise/{i}/region",
        params={"x0": 0.0, "y0": 0.0, "x1": 0.35, "y1": 0.35, "level": "low"},
    )
    hi = client.get(
        f"/api/denoise/{i}/region",
        params={"x0": 0.0, "y0": 0.0, "x1": 0.35, "y1": 0.35, "level": "high"},
    )
    a, b = _img(r), _img(hi)
    assert a.shape == b.shape and _corner_shift(a, b) < 0.8


def test_the_blend_base_is_the_demosaic_render_in_the_frame_of_the_output(client, ids):
    from shotsieve.io.preview import load_full
    from tests.test_denoise_raw import PANA

    off = load_full(PANA)
    base = rawnind.render_base(PANA, off)
    den = rawnind.render(PANA, off)
    assert base.shape == den.shape == off.shape and base.dtype == np.uint8
    assert np.abs(base.reshape(-1, 3).mean(0) - den.reshape(-1, 3).mean(0)).max() < 6
