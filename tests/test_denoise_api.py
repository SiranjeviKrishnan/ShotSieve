# ruff: noqa: E501
"""T23f denoise (view only): /api/denoise/*, cache, fallback, regions, idle release, quality, RSS."""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import psutil
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from shotsieve.config import Settings
from shotsieve.models import denoise as engine
from shotsieve.pipeline.cache import Cache
from shotsieve.server import denoise as dsrv
from shotsieve.server.app import create_app
from tests.test_api import _row, shoot_of
from tests.test_viewer_quality import _chart

needs_model = pytest.mark.skipif(
    not engine.available(), reason="denoise model not installed (shotsieve models download --extra denoise)"
)  # fmt: skip


def _noisy(clean: np.ndarray, sigma: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(clean.astype(np.float32) + rng.normal(0, sigma, clean.shape), 0, 255).astype(
        np.uint8
    )


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    return float(10 * np.log10(255**2 / np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)))


def _photo(gen_dir, w=1200, h=800) -> np.ndarray:
    src = cv2.cvtColor(cv2.imread(str(gen_dir / "shoot" / "landscape.jpg")), cv2.COLOR_BGR2RGB)
    return cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA)


@pytest.fixture()
def shoot(tmp_path, gen_dir):
    folder = tmp_path / "shoot"
    folder.mkdir()
    clean = _photo(gen_dir, 2400, 1600)
    Image.fromarray(_noisy(clean, 14)).save(folder / "noisy.jpg", quality=95, subsampling=0)
    Image.fromarray(clean).save(folder / "clean.jpg", quality=95)
    Image.fromarray(_chart(1800, 1200)).save(folder / "chart.jpg", quality=98, subsampling=0)
    clip = folder / "clip.mp4"
    clip.write_bytes(b"\0" * 64)
    names = ["noisy.jpg", "clean.jpg", "chart.jpg"]
    rows = [_row(folder / n, "keep", i, True) for i, n in enumerate(names)]
    rows.append(_row(f"{clip}#t=1.000", "keep", 9, True))
    with Cache.for_folder(folder) as c:
        c.save_decisions(rows)
    return folder


@pytest.fixture()
def client(shoot):
    with TestClient(create_app(Settings(), default_folder=shoot)) as c:
        yield c
    dsrv.release_all()


@pytest.fixture()
def ids(client, shoot):
    return {im["name"]: im["id"] for im in shoot_of(client, shoot)[0]["images"]}


def _tree_hash(folder: Path) -> dict[str, str]:
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() + str(p.stat().st_mtime_ns)
        for p in sorted(folder.iterdir())
        if p.is_file()
    }


def _img(r) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"))


# ---- endpoints ---------------------------------------------------------------------------------


def test_status_reports_model_and_backend(client):
    d = client.get("/api/denoise/status").json()
    assert d["model"]["name"] == "nafnet_sidd" and d["model"]["license"] == "MIT"
    assert d["backend"] in ("ai", "basic") and d["levels"] == ["low", "medium", "high"]
    assert d["backend"] == ("ai" if d["model"]["installed"] else "basic")


def test_view_image_is_denoised_at_the_bucket_size_and_cached(client, ids, shoot, monkeypatch):
    before = _tree_hash(shoot)
    calls = []
    real = engine.denoise
    monkeypatch.setattr(engine, "denoise", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    r = client.get(f"/api/denoise/{ids['noisy.jpg']}", params={"w": 1280})
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert r.headers["x-denoise-backend"] in ("ai", "basic")
    out = _img(r)
    assert max(out.shape[:2]) == 1280
    plain = _img(client.get(f"/api/preview/{ids['noisy.jpg']}", params={"w": 1280}))
    assert plain.shape == out.shape  # the same frame as the picture it replaces
    assert out.astype(float).std(axis=None) > 0 and np.abs(out.astype(int) - plain.astype(int)).mean() > 0.5
    again = client.get(f"/api/denoise/{ids['noisy.jpg']}", params={"w": 1280})
    assert again.content == r.content and len(calls) == 1  # served from .shotsieve/denoise/
    files = list((shoot / ".shotsieve" / "denoise").glob("v_*.jpg"))
    assert len(files) == 1 and not list((shoot / ".shotsieve" / "denoise").glob("*.tmp"))
    # a different strength is a different file; the originals never change
    client.get(f"/api/denoise/{ids['noisy.jpg']}", params={"w": 1280, "level": "high"})
    assert len(list((shoot / ".shotsieve" / "denoise").glob("v_*.jpg"))) == 2
    after = _tree_hash(shoot)
    assert {k: v for k, v in after.items() if k in before} == before
    assert set(after) == set(before)  # no stray file next to the originals
    assert not list(shoot.glob("*.xmp"))


def test_bad_level_unknown_image_and_video_are_refused(client, ids, shoot):
    assert client.get(f"/api/denoise/{ids['noisy.jpg']}", params={"level": "extreme"}).status_code == 422
    assert client.get("/api/denoise/424242").status_code == 404
    clip = next(i for i in shoot_of(client, shoot)[0]["images"] if i["name"].startswith("clip.mp4"))
    assert client.get(f"/api/denoise/{clip['id']}").status_code == 422


def test_without_the_model_the_basic_fallback_runs(client, ids, shoot, monkeypatch):
    monkeypatch.setenv("SHOTSIEVE_DENOISE_BACKEND", "basic")
    d = client.get("/api/denoise/status").json()
    assert d["backend"] == "basic" and not d["model"]["installed"]
    r = client.get(f"/api/denoise/{ids['noisy.jpg']}", params={"w": 1280})
    assert r.status_code == 200 and r.headers["x-denoise-backend"] == "basic"
    assert list((shoot / ".shotsieve" / "denoise").glob("v_*_basic_*.jpg"))
    # installing the model later is a different cache key, never a stale "basic" picture
    monkeypatch.delenv("SHOTSIEVE_DENOISE_BACKEND")
    if engine.available():
        r2 = client.get(f"/api/denoise/{ids['noisy.jpg']}", params={"w": 1280})
        assert r2.headers["x-denoise-backend"] == "ai"


def test_region_is_denoised_on_full_resolution_pixels_and_cached(client, ids, shoot):
    iid = ids["noisy.jpg"]
    q = {"x0": 0.1, "y0": 0.2, "x1": 0.4, "y1": 0.5}
    r = client.get(f"/api/denoise/{iid}/region", params=q)
    assert r.status_code == 200
    x0, y0, x1, y1 = (int(v) for v in r.headers["x-region"].split(","))
    assert r.headers["x-image-size"] == "2400,1600"
    assert x0 % 64 == 0 and y0 % 64 == 0 and (x0, y0) <= (240, 320) and x1 >= 960 and y1 >= 800
    out = _img(r)
    assert out.shape[:2] == (y1 - y0, x1 - x0)  # native pixels: far more than the 1280 px preview has
    full = np.asarray(Image.open(shoot / "noisy.jpg").convert("RGB"))[y0:y1, x0:x1]
    assert np.abs(out.astype(int) - full.astype(int)).mean() > 1.0  # it really changed the pixels
    n = len(list((shoot / ".shotsieve" / "denoise").glob("r_*.jpg")))
    r2 = client.get(f"/api/denoise/{iid}/region", params={**q, "x0": 0.101})  # same snapped box
    assert r2.content == r.content
    assert len(list((shoot / ".shotsieve" / "denoise").glob("r_*.jpg"))) == n
    assert client.get(f"/api/denoise/{iid}/region", params={**q, "x1": 0.1}).status_code in (200, 422)
    tiny = client.get(f"/api/denoise/{iid}/region", params={"x0": 0.5, "y0": 0.5, "x1": 0.501, "y1": 0.501})
    assert tiny.status_code == 422


def test_region_has_the_same_noise_level_as_the_whole_image(client, ids, shoot):
    """A crop is denoised with the noise level of the WHOLE picture, so it matches the full result."""
    iid = ids["noisy.jpg"]
    r = client.get(f"/api/denoise/{iid}/region", params={"x0": 0.25, "y0": 0.25, "x1": 0.75, "y1": 0.75})
    x0, y0, x1, y1 = (int(v) for v in r.headers["x-region"].split(","))
    f = client.get(f"/api/denoise/{iid}/full")
    assert f.status_code == 200 and f.headers["x-denoise-backend"] in ("ai", "basic")
    whole = _img(f)
    assert whole.shape[:2] == (1600, 2400)
    inner = whole[y0 + 48 : y1 - 48, x0 + 48 : x1 - 48].astype(int)
    reg = _img(r)[48 : y1 - y0 - 48, 48 : x1 - x0 - 48].astype(int)
    assert np.abs(inner - reg).mean() < 3.0  # tile seams / JPEG only


def test_a_region_request_goes_before_a_running_full_job(client, ids, monkeypatch):
    seen = []
    real = engine.denoise

    def slow(rgb, level="medium", **k):
        pause = k.get("pause")
        if pause is not None:  # the whole-image job: ask to be paused 5 times, note the order
            for _ in range(5):
                pause()
            seen.append("full-ran")
        else:
            seen.append("region-ran")
        return real(rgb, level, **{x: y for x, y in k.items() if x != "pause"})

    monkeypatch.setattr(engine, "denoise", slow)
    iid = ids["noisy.jpg"]
    monkeypatch.setattr(dsrv, "_urgent", 1)  # a region is "waiting": the full job must block
    t = threading.Thread(target=lambda: client.get(f"/api/denoise/{iid}/full"))
    t.start()
    time.sleep(0.6)
    assert "full-ran" not in seen  # paused on the waiting region request
    monkeypatch.setattr(dsrv, "_urgent", 0)
    t.join(60)
    assert "full-ran" in seen


def test_cache_is_lru_bounded(client, ids, shoot, monkeypatch):
    d = shoot / ".shotsieve" / "denoise"
    client.get(f"/api/denoise/{ids['clean.jpg']}", params={"w": 1280})
    first = next(d.glob("v_*.jpg"))
    one = first.stat().st_size
    os.utime(first, ns=(1, 1))  # clearly the least recently used
    monkeypatch.setattr(dsrv, "CACHE_CAP", int(one * 2.5))
    client.get(f"/api/denoise/{ids['clean.jpg']}", params={"w": 2048})
    client.get(f"/api/denoise/{ids['chart.jpg']}", params={"w": 1280})
    left = list(d.glob("*.jpg"))
    assert first not in left and left  # the LRU file went, the newest one is kept
    newest = max(left, key=lambda f: f.stat().st_mtime_ns)
    assert "chart" not in newest.name and newest.name.startswith(f"v_{ids['chart.jpg']}_")


def test_changed_original_is_a_new_cache_entry(client, ids, shoot):
    iid = ids["clean.jpg"]
    client.get(f"/api/denoise/{iid}", params={"w": 1280})
    p = shoot / "clean.jpg"
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    client.get(f"/api/denoise/{iid}", params={"w": 1280})
    assert len(list((shoot / ".shotsieve" / "denoise").glob("v_*.jpg"))) == 2


# ---- the model is released when idle ---------------------------------------------------------


@needs_model
def test_model_session_is_released_after_the_idle_time():
    engine.release()
    engine._holder = None  # a fresh holder with a short idle time
    h = engine.holder(0.4)
    try:
        with h.use() as net:
            net.tile(np.zeros((256, 256, 3), np.uint8))
        assert engine.loaded()
        time.sleep(1.2)
        assert not engine.loaded()
        with h.use():  # and it loads again on demand
            assert engine.loaded()
    finally:
        engine.release()
        engine._holder = None


# ---- quality (synthetic noise on a clean reference) -------------------------------------------


@pytest.mark.parametrize("backend", ["ai", "basic"])
@pytest.mark.parametrize("sigma", [8, 15, 25])
def test_psnr_improves_by_at_least_3_db(gen_dir, backend, sigma):
    if backend == "ai" and not engine.available():
        pytest.skip("model not installed")
    clean = _photo(gen_dir, 1000, 700)
    noisy = _noisy(clean, sigma)
    out, used = engine.denoise(noisy, "medium", backend=backend)
    assert used == backend
    gain = _psnr(out, clean) - _psnr(noisy, clean)
    assert gain >= 3.0, f"{backend} sigma {sigma}: only +{gain:.2f} dB"
    engine.release()


_REAL = ["cc0_football_18.jpg", "cc0_football_47.jpg", "cc0_football_64.jpg"]


def _lap(a: np.ndarray) -> float:
    return float(cv2.Laplacian(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())


def _edge_gain(a: np.ndarray, ref: np.ndarray) -> float:
    """Mean edge gradient on the reference's Canny edges, relative to the reference."""
    g = lambda x: np.abs(cv2.Sobel(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), cv2.CV_32F, 1, 1))  # noqa: E731
    edges = cv2.Canny(cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY), 100, 200) > 0
    return float(g(a)[edges].mean() / g(ref)[edges].mean())


@pytest.mark.parametrize("backend", ["ai", "basic"])
@pytest.mark.parametrize("name", _REAL)
def test_sharp_real_photos_are_not_blurred_beyond_a_threshold(backend, name):
    """A sharp, detailed photo (real edges and texture) with sigma-10 noise: PSNR up by >= 3 dB (AI; 2.5 for the fallback), the
    edge contrast stays >= 80 % of the clean photo at Medium (>= 70 % at High), and a clean photo is
    barely touched (Laplacian variance kept >= 65 % AI / 75 % basic)."""
    if backend == "ai" and not engine.available():
        pytest.skip("model not installed")
    root = Path(__file__).parent / "fixtures" / "real"
    img = np.asarray(Image.open(root / name).convert("RGB"))
    img = np.ascontiguousarray(img[:700, :700])
    noisy = _noisy(img, 10, seed=1)
    for level, keep in (("medium", 0.8), ("high", 0.7)):
        out, _ = engine.denoise(noisy, level, backend=backend)
        assert _psnr(out, img) >= _psnr(noisy, img) + {"ai": 3, "basic": 2.5}[backend], (backend, level)
        assert _edge_gain(out, img) >= keep, (backend, level, _edge_gain(out, img))
    clean_out, _ = engine.denoise(img, "medium", backend=backend)
    assert _lap(clean_out) >= {"ai": 0.65, "basic": 0.75}[backend] * _lap(img)
    engine.release()


def test_basic_fallback_keeps_a_binary_chart_sharp():
    """(The network is trained on photographs: a synthetic binary chart is out of its domain, so this
    one runs on the OpenCV fallback only.)"""
    chart = _chart(1200, 800)
    out, _ = engine.denoise(chart, "medium", backend="basic")
    assert _lap(out) >= 0.95 * _lap(chart) and _psnr(out, chart) > 40
    noisy = _noisy(chart, 10)
    out2, _ = engine.denoise(noisy, "high", backend="basic")
    assert _psnr(out2, chart) > _psnr(noisy, chart) + 3
    assert _edge_gain(out2, chart) >= 0.6


def test_tiling_is_seamless_and_handles_odd_sizes():
    rng = np.random.default_rng(0)
    for shape in ((100, 90), (257, 300), (700, 523)):
        img = rng.integers(0, 255, (*shape, 3), dtype=np.uint8)
        out = engine.run_tiled(img, lambda t: t.astype(np.float32), workers=2)
        assert out.shape == img.shape and np.array_equal(out, img)  # identity model: exact
        ramp = engine.run_tiled(img, lambda t: t.astype(np.float32) * 0.5 + 10, workers=2)
        assert np.abs(ramp.astype(int) - (img * 0.5 + 10).round().astype(int)).max() <= 1
    with pytest.raises(engine.Cancelled):
        engine.run_tiled(img, lambda t: t.astype(np.float32), cancel=lambda: True)


def test_noise_estimate_tracks_noise_and_iso_bounds_it(gen_dir):
    clean = _photo(gen_dir, 800, 600)
    s0 = engine.estimate_sigma(clean)
    s15 = engine.estimate_sigma(_noisy(clean, 15))
    assert s15 > s0 + 5 and 9 < s15 < 20
    assert engine.effective_sigma(clean, 100) >= 0.5 * engine.prior_sigma(100)
    assert engine.effective_sigma(clean, None) > 0
    assert engine.ai_alpha("high", 1.0) < engine.ai_alpha("high", 10.0)
    assert engine.nlm_strength("low", 5) < engine.nlm_strength("high", 5)


# ---- memory ---------------------------------------------------------------------------------------


@needs_model
def test_rss_while_denoising_a_6_mp_image_stays_within_budget(gen_dir):
    img = _noisy(_photo(gen_dir, 3000, 2000), 12)
    p = psutil.Process()
    engine.release()
    base = p.memory_info().rss
    peak = [base]
    stop = threading.Event()

    def watch():
        while not stop.is_set():
            peak[0] = max(peak[0], p.memory_info().rss)
            time.sleep(0.03)

    t = threading.Thread(target=watch)
    t.start()
    try:
        out, used = engine.denoise(img, "medium", backend="ai")
    finally:
        stop.set()
        t.join()
        engine.release()
    assert used == "ai" and out.shape == img.shape
    assert (peak[0] - base) / 1e6 <= 600, f"+{(peak[0] - base) / 1e6:.0f} MB while denoising"


def test_originals_are_never_written_by_the_fallback(tmp_path):
    f = tmp_path / "a.jpg"
    Image.fromarray(_noisy(np.full((300, 400, 3), 120, np.uint8), 10)).save(f, quality=95)
    h = hashlib.sha256(f.read_bytes()).hexdigest()
    engine.denoise(np.asarray(Image.open(f).convert("RGB")), "low", backend="basic")
    assert hashlib.sha256(f.read_bytes()).hexdigest() == h
    assert shutil.which("true")  # (keeps `shutil` imported for the tmp helpers above)
