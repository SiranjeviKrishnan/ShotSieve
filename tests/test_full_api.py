"""/api/full/{id} (T23c): native-size image for zoom, cached, bounded, originals untouched."""

from __future__ import annotations

import hashlib
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

from shotsieve.config import Settings
from shotsieve.pipeline.cache import Cache
from shotsieve.server import app as appmod
from shotsieve.server.app import create_app
from tests.test_api import _row, shoot_of


def _dec(r):
    return cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)


def _tree_hash(folder: Path) -> dict[str, tuple[str, int]]:
    out = {}
    for p in sorted(folder.iterdir()):
        if p.is_file():
            out[p.name] = (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
    return out


@pytest.fixture()
def shoot(tmp_path, shoot_dir, gen_dir):
    folder = tmp_path / "shoot"
    folder.mkdir()
    shutil.copy(shoot_dir / "landscape.jpg", folder / "landscape.jpg")
    shutil.copy(gen_dir / "misc" / "orient6.jpg", folder / "orient6.jpg")
    shutil.copy(gen_dir / "raw" / "pentax_kx.dng", folder / "pentax.dng")  # full-size embedded JPEG
    shutil.copy(gen_dir / "raw" / "panasonic_lx1.dng", folder / "pana.dng")  # small thumb: demosaic
    (folder / "junk.jpg").write_bytes(b"not an image")
    names = ["landscape.jpg", "orient6.jpg", "pentax.dng", "pana.dng", "junk.jpg"]
    rows = [_row(folder / n, "keep", i, True) for i, n in enumerate(names)]
    clip = folder / "clip.mp4"
    clip.write_bytes(b"\0" * 64)
    rows.append(_row(f"{clip}#t=1.000", "keep", 9, True))
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


def test_jpeg_native_size_and_size_endpoint(client, shoot, ids):
    r = client.get(f"/api/full/{ids['landscape.jpg']}")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    img = _dec(r)
    ref = cv2.imread(str(shoot / "landscape.jpg"))
    assert img.shape == ref.shape
    assert client.get(f"/api/full/{ids['landscape.jpg']}/size").json() == {
        "w": ref.shape[1], "h": ref.shape[0]
    }  # fmt: skip
    # the preview is the same picture, smaller
    p = _dec(client.get(f"/api/preview/{ids['landscape.jpg']}"))
    small = cv2.resize(img, (p.shape[1], p.shape[0]), interpolation=cv2.INTER_AREA)
    assert np.abs(small.astype(int) - p.astype(int)).mean() < 6


def test_orientation_is_the_display_frame_of_the_preview(client, ids):
    iid = ids["orient6.jpg"]
    img = _dec(client.get(f"/api/full/{iid}"))
    assert img.shape[:2] == (400, 600)  # stored 600x400 portrait, displayed landscape
    assert client.get(f"/api/full/{iid}/size").json() == {"w": 600, "h": 400}
    p = _dec(client.get(f"/api/preview/{iid}"))
    assert p.shape == img.shape and np.abs(p.astype(int) - img.astype(int)).mean() < 8


def test_raw_embedded_and_demosaic_paths(client, ids):
    for name, shape in (("pentax.dng", (2848, 4288)), ("pana.dng", (2170, 3857))):
        iid = ids[name]
        img = _dec(client.get(f"/api/full/{iid}"))
        assert img.shape[:2] == shape, name
        sz = client.get(f"/api/full/{iid}/size").json()
        assert (sz["h"], sz["w"]) == shape
        p = _dec(client.get(f"/api/preview/{iid}"))
        assert abs(p.shape[1] / p.shape[0] - shape[1] / shape[0]) < 0.01  # same frame


def test_cached_on_disk_not_decoded_twice_and_originals_untouched(client, shoot, ids, monkeypatch):
    before = _tree_hash(shoot)
    calls = []
    real = appmod.load_full
    monkeypatch.setattr(appmod, "load_full", lambda p: (calls.append(p), real(p))[1])
    iid = ids["pentax.dng"]
    a = client.get(f"/api/full/{iid}")
    b = client.get(f"/api/full/{iid}")
    assert a.content == b.content and len(calls) == 1
    files = list((shoot / ".shotsieve" / "full").glob("f_*.jpg"))
    assert len(files) == 1 and not list((shoot / ".shotsieve" / "full").glob("*.tmp"))
    assert _tree_hash(shoot) == before  # no original, sidecar or stray file changed or appeared
    assert not list(shoot.glob("*.xmp"))


def test_changed_original_is_a_new_cache_entry(client, shoot, ids):
    iid = ids["landscape.jpg"]
    client.get(f"/api/full/{iid}")
    p = shoot / "landscape.jpg"
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    client.get(f"/api/full/{iid}")
    assert len(list((shoot / ".shotsieve" / "full").glob("f_*.jpg"))) == 2


def test_lru_prune_keeps_the_cache_within_the_cap(client, shoot, ids, monkeypatch):
    first = client.get(f"/api/full/{ids['landscape.jpg']}").content
    one = len(first)
    monkeypatch.setattr(appmod, "FULL_CACHE_CAP", int(one * 2.5))
    full_dir = shoot / ".shotsieve" / "full"
    old = next(full_dir.glob("f_*.jpg"))
    os.utime(old, ns=(1, 1))  # clearly the least recently used
    client.get(f"/api/full/{ids['orient6.jpg']}")
    client.get(f"/api/full/{ids['pana.dng']}")
    left = list(full_dir.glob("f_*.jpg"))
    assert old not in left
    newest = [f for f in left if f.name.startswith(f"f_{ids['pana.dng']}_")]
    assert newest, "the file just written is never pruned"
    sizes = [f.stat().st_size for f in left]
    assert len(left) == 1 or sum(sizes) <= max(appmod.FULL_CACHE_CAP, newest[0].stat().st_size)


def test_hit_refreshes_lru_order(client, shoot, ids):
    client.get(f"/api/full/{ids['landscape.jpg']}")
    f = next((shoot / ".shotsieve" / "full").glob("f_*.jpg"))
    os.utime(f, ns=(1, 1))
    client.get(f"/api/full/{ids['landscape.jpg']}")
    assert f.stat().st_mtime_ns > 10**9


def test_error_cases(client, shoot, ids):
    assert client.get("/api/full/424242").status_code == 404
    assert client.get("/api/full/424242/size").status_code == 404
    assert client.get(f"/api/full/{ids['junk.jpg']}").status_code == 422
    assert client.get(f"/api/full/{ids['junk.jpg']}/size").status_code == 422
    assert not list((shoot / ".shotsieve" / "full").glob("f_*"))  # nothing cached for a failure
    assert client.get(f"/api/full/{ids['clip.mp4#t=1.000']}").status_code == 422  # video frame
    assert client.get("/api/full/abc").status_code == 422
    (shoot / "landscape.jpg").unlink()
    assert client.get(f"/api/full/{ids['landscape.jpg']}").status_code == 404
    empty = shoot.parent / "empty"
    empty.mkdir()
    assert client.get("/api/full/1", params={"folder": str(empty)}).status_code == 404


def test_host_and_folder_rules(client, shoot, ids):
    iid = ids["landscape.jpg"]
    for url in (f"/api/full/{iid}", f"/api/full/{iid}/size"):
        assert client.get(url, headers={"host": "evil.example.com"}).status_code == 400
        assert client.get(url, params={"folder": "/tmp/\x00evil"}).status_code == 400
        assert client.get(url, params={"folder": "relative/dir"}).status_code == 400
        assert client.get(url, params={"folder": str(shoot)}).status_code == 200
    # a state-changing call from a foreign origin is still refused; reads are the same as /preview
    assert client.post("/api/quit", headers={"Origin": "http://evil.example"}).status_code == 403


def test_a_row_outside_the_folder_is_not_served(client, shoot, tmp_path):
    outside = tmp_path / "outside.jpg"
    shutil.copy(shoot / "landscape.jpg", outside)
    with Cache.for_folder(shoot) as c:
        c.save_decisions([_row(outside, "keep", 0, True)])
    images = shoot_of(client, shoot)[0]["images"]
    assert images
    for im in images:
        assert client.get(f"/api/full/{im['id']}").status_code == 404


def test_only_one_full_decode_at_a_time(client, shoot, ids, monkeypatch):
    live, peak = [0], [0]
    gate = threading.Lock()
    real = appmod.load_full

    def slow(p):
        with gate:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        time.sleep(0.15)
        try:
            return real(p)
        finally:
            with gate:
                live[0] -= 1

    monkeypatch.setattr(appmod, "load_full", slow)
    out = []
    ts = [
        threading.Thread(target=lambda i=i: out.append(client.get(f"/api/full/{i}").status_code))
        for i in (ids["landscape.jpg"], ids["orient6.jpg"], ids["pentax.dng"], ids["pana.dng"])
    ]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert out == [200] * 4 and peak[0] == 1


def test_a_big_decode_does_not_retain_memory(tmp_path):
    folder = tmp_path / "big"
    folder.mkdir()
    yy, xx = np.mgrid[0:4000, 0:6000]
    img = np.stack([xx % 256, yy % 256, (xx + yy) % 256], -1).astype(np.uint8)  # 24 MP
    cv2.imwrite(str(folder / "big.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    del img
    with Cache.for_folder(folder) as c:
        c.save_decisions([_row(folder / "big.jpg", "keep", 0, True)])
    proc = psutil.Process()
    with TestClient(create_app(Settings(), default_folder=folder)) as cl:
        iid = shoot_of(cl, folder)[0]["images"][0]["id"]
        base = proc.memory_info().rss
        r = cl.get(f"/api/full/{iid}")
        assert r.status_code == 200
        del r
        after = proc.memory_info().rss
    assert (after - base) / 1e6 < 300, (after - base) / 1e6
