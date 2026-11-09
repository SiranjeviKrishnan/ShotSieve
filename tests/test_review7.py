# ruff: noqa: E501
"""Review #7 minors n1-n4 (n5 is documentation)."""

from __future__ import annotations

import subprocess
import sys
import time
import zlib
from types import SimpleNamespace

import numpy as np
import pytest

from shotsieve.features.color_stats import EDIT_STAT_NAMES, edit_stats
from shotsieve.io import xmp
from shotsieve.io.crs import read_develop_xmp, write_develop
from shotsieve.pipeline import edit


def test_n1_curve_only_edit_is_not_ours(tmp_path):
    raw = tmp_path / "a.CR2"
    raw.write_bytes(b"raw")
    assert write_develop(raw, {"Exposure2012": -0.4}, profile_hash="h").status == "written"
    sp = tmp_path / "a.xmp"
    assert read_develop_xmp(raw).ours
    text = sp.read_text()
    curve = "<crs:ToneCurvePV2012><rdf:Seq><rdf:li>0, 0</rdf:li><rdf:li>255, 255</rdf:li></rdf:Seq></crs:ToneCurvePV2012>"
    head, sep, tail = text.rpartition(" />\n  </rdf:RDF>")  # the Description is self-closing
    assert sep
    sp.write_text(head + ">" + curve + "</rdf:Description>\n  </rdf:RDF>" + tail)
    cur = read_develop_xmp(raw)
    assert cur.tagged and not cur.ours  # a point-curve-only change is the user's
    before = sp.read_bytes()
    assert write_develop(raw, {"Exposure2012": 0.9}, profile_hash="h2").status == "skipped_user_settings"
    assert sp.read_bytes() == before


def test_n2_copy_root_inside_the_shoot_must_be_a_skipped_folder(tmp_path):
    shoot = tmp_path / "shoot"
    shoot.mkdir()
    with pytest.raises(ValueError):
        edit.check_copy_root(shoot / "copies", shoot)
    with pytest.raises(ValueError):
        edit.check_copy_root(shoot / "day1" / "out", shoot)
    edit.check_copy_root(shoot / "shotsieve_edits", shoot)
    edit.check_copy_root(shoot / "shotsieve_edits" / "sub", shoot)
    edit.check_copy_root(tmp_path / "elsewhere", shoot)


def test_n3_sidecar_lock_blocks_another_process(tmp_path):
    p = tmp_path / "a.xmp"
    ready = tmp_path / "ready"
    code = (
        "import sys,time,pathlib;from shotsieve.io import xmp;"
        "l=xmp.sidecar_lock(sys.argv[1]);l.__enter__();pathlib.Path(sys.argv[2]).write_text('1');"
        "time.sleep(1.5);l.__exit__(None,None,None)"
    )
    proc = subprocess.Popen([sys.executable, "-c", code, str(p), str(ready)])
    try:
        for _ in range(200):
            if ready.exists():
                break
            time.sleep(0.05)
        assert ready.exists()
        t0 = time.time()
        with xmp.sidecar_lock(p):
            waited = time.time() - t0
        assert waited >= 0.8, waited  # the other process held it
        with xmp.sidecar_lock(p), xmp.sidecar_lock(p):  # still re-entrant in process
            pass
    finally:
        proc.wait(timeout=10)


def test_n4_camera_bucket_separates_bodies_that_collided_mod_997():
    seen: dict[int, str] = {}
    a = b = ""
    for i in range(5000):
        name = f"Body{i}"
        k = zlib.crc32(f"Make|{name}".encode()) % 997
        if k in seen:
            a, b = seen[k], name
            break
        seen[k] = name
    assert a and b
    img = (np.random.default_rng(0).random((48, 64, 3)) * 255).astype(np.uint8)
    col = EDIT_STAT_NAMES.index("camera_idx")

    def idx(model: str) -> float:
        meta = SimpleNamespace(make="Make", model=model, iso=100, exposure_time=0.01, fnumber=2.8,
                               focal_length=35.0, flash=False, timestamp=None, ts_source="")
        return float(edit_stats(img, meta)[col])

    assert idx(a) != idx(b)
    assert idx(a) == float(np.float32(idx(a)))  # exact in float32
