# ruff: noqa: E501
"""Fix pass after Review #12: M1 (rotated in Lightroom), M2 (catalog guard), m2/m3/m5."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from shotsieve.io import crs
from shotsieve.io.crs import write_crop
from shotsieve.pipeline import crop as pcrop
from shotsieve.pipeline.cache import Cache
from shotsieve.pipeline.crop_geom import Rect

from .test_crop_pipeline import patched_dng

LEFT_HALF = Rect(0.0, 0.0, 0.5, 1.0)
RECT = (0.12, 0.1, 0.9, 0.8)


def _rotated_sidecar(path: Path, o: int) -> None:
    path.write_text(
        '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        f'<rdf:Description rdf:about="" xmlns:tiff="http://ns.adobe.com/tiff/1.0/" tiff:Orientation="{o}"/></rdf:RDF></x:xmpmeta>'
    )


@pytest.mark.parametrize("exif_o,lr_o", [(1, 6), (1, 3), (1, 8), (6, 1), (3, 8)])
def test_rotated_in_lightroom_never_writes_a_different_region(tmp_path, gen_dir, exif_o, lr_o):
    """The reviewer's probe: the left half the user sees is never written as another region."""
    raw = patched_dng(gen_dir / "raw" / "pentax_kx.dng", tmp_path / "r.pef", exif_o)
    sp = tmp_path / "r.xmp"
    _rotated_sidecar(sp, lr_o)
    before = sp.read_bytes()
    fw, fh, got = pcrop.read_image_facts(str(raw))
    assert got == exif_o
    info = pcrop.ImageInfo(str(raw), fw, fh, exif_o, "raw")
    out = pcrop.write_xmp(tmp_path, 1, str(raw), LEFT_HALF, 1, info, pcrop.Settings())
    assert out["status"] == "skipped_rotated" and out["mode"] == "none"
    assert "rotated in Lightroom" in out["reason"]
    assert sp.read_bytes() == before  # nothing written
    # reset is frame independent and still allowed (nothing of ours there: unchanged)
    out = pcrop.write_xmp(tmp_path, 1, str(raw), None, 2, info, pcrop.Settings())
    assert out["status"] == "unchanged"


def test_rotated_crop_is_kept_in_the_db_and_the_state_says_so(tmp_path, gen_dir):
    folder = tmp_path / "s"
    folder.mkdir()
    raw = patched_dng(gen_dir / "raw" / "pentax_kx.dng", folder / "r.pef", 1)
    _rotated_sidecar(folder / "r.xmp", 6)
    with Cache.for_folder(folder) as c:
        i = c.ids_for([str(raw)])[str(raw)]
        info = pcrop._info(folder, c, i, str(raw), pcrop.Settings())
        rect = Rect(0.0, 0.0, 0.5, 1.0)
        # snap to a ratio: use the exact half-width of the frame
        ratio = f"{info.fw // 2}:{info.fh}"
        r = pcrop.accept(folder, c, i, str(raw), pcrop.Settings(), list(rect.as_tuple()), ratio)
        assert r["write"]["status"] == "skipped_rotated"
        assert c.crop_current(i)["write_status"] == "skipped_rotated"
        st = pcrop.crop_state(folder, c, i, str(raw), pcrop.Settings())
        assert st["rotated_in_lightroom"] is True and "rotated in Lightroom" in st["rotated_notice"]
        assert st["orientation"] == 1  # one frame: the EXIF frame the preview uses
        assert crs.read_develop_xmp(folder / "r.xmp").crop == {}


# ---- M2 -------------------------------------------------------------------------------------


def test_our_crop_only_sidecar_does_not_bypass_the_catalog_check(tmp_path):
    raw = tmp_path / "k.CR3"
    raw.write_bytes(b"x")
    assert write_crop(raw, RECT).status == "written"  # no catalog known: crop-only sidecar
    cur = crs.read_develop_xmp(tmp_path / "k.xmp")
    assert cur.only_ours and cur.has_settings
    res = write_crop(raw, RECT, catalog_has=True)  # probe: must not be replaced_own
    assert res.status == "skipped_catalog"
    assert write_crop(raw, None, catalog_has=True).status == "skipped_catalog"  # reset too


def _fake_catalog(tmp_path: Path) -> Path:
    cat = tmp_path / "cat" / "My.lrcat"
    cat.parent.mkdir()
    con = sqlite3.connect(cat)
    con.execute("create table t(x)")
    con.commit()
    con.close()
    return cat


def test_crop_write_defaults_to_the_remembered_catalog_after_a_restart(tmp_path, gen_dir, monkeypatch):
    folder = tmp_path / "s"
    folder.mkdir()
    raw = patched_dng(gen_dir / "raw" / "pentax_kx.dng", folder / "e.pef", 1)
    cat = _fake_catalog(tmp_path)
    with Cache.for_folder(folder) as c:
        c.set_meta("lr_catalog", str(cat))
    seen: list[str] = []

    def state(catalog):
        seen.append(catalog)
        return {str(raw): (True, False)}  # Lightroom holds edits for this RAW

    monkeypatch.setattr(pcrop, "_catalog_state", state)
    with Cache.for_folder(folder) as c:  # "restarted server": a fresh Cache, no catalog in the request
        i = c.ids_for([str(raw)])[str(raw)]
        info = pcrop._info(folder, c, i, str(raw), pcrop.Settings())
        ratio = f"{info.fw // 2}:{info.fh}"
        r = pcrop.accept(folder, c, i, str(raw), pcrop.Settings(), [0, 0, 0.5, 1], ratio)
    assert seen == [str(cat)]
    assert r["write"]["status"] == "skipped_catalog"
    assert not (folder / "e.xmp").exists()


def test_missing_remembered_catalog_is_unknown_and_explicit_bad_one_is_a_400(tmp_path):
    folder = tmp_path / "s"
    folder.mkdir()
    with Cache.for_folder(folder) as c:
        c.set_meta("lr_catalog", str(tmp_path / "gone.lrcat"))
        assert pcrop.resolve_catalog(c, None) is None
        with pytest.raises(pcrop.CropError) as ei:
            pcrop.resolve_catalog(c, "relative.lrcat")
        assert ei.value.status == 400
        with pytest.raises(pcrop.CropError):
            pcrop.resolve_catalog(c, str(tmp_path / "x.txt"))


def test_bad_catalog_stores_no_version_and_unreadable_one_is_skipped_cleanly(tmp_path, gen_dir):
    folder = tmp_path / "s"
    folder.mkdir()
    raw = patched_dng(gen_dir / "raw" / "pentax_kx.dng", folder / "e.pef", 1)
    bad = tmp_path / "Broken.lrcat"
    bad.write_bytes(b"not a database at all")
    with Cache.for_folder(folder) as c:
        i = c.ids_for([str(raw)])[str(raw)]
        info = pcrop._info(folder, c, i, str(raw), pcrop.Settings())
        ratio = f"{info.fw // 2}:{info.fh}"
        with pytest.raises(pcrop.CropError) as ei:
            pcrop.accept(folder, c, i, str(raw), pcrop.Settings(), [0, 0, 0.5, 1], ratio, catalog="nope.lrcat")
        assert ei.value.status == 400 and c.crop_history(i) == []
        r = pcrop.accept(folder, c, i, str(raw), pcrop.Settings(), [0, 0, 0.5, 1], ratio, catalog=str(bad))
        assert r["write"]["status"] == "skipped_catalog" and "could not be read" in r["write"]["reason"]
        assert c.crop_current(i)["write_status"] == "skipped_catalog"  # consistent state
        assert not (folder / "e.xmp").exists()


def test_catalog_cache_key_sees_wal_writes(tmp_path):
    cat = _fake_catalog(tmp_path)
    k1 = pcrop._catalog_key(str(cat))
    Path(str(cat) + "-wal").write_bytes(b"w" * 100)
    k2 = pcrop._catalog_key(str(cat))
    assert k1 != k2
    os.utime(str(cat) + "-wal", ns=(1, 1))
    assert pcrop._catalog_key(str(cat)) != k2


def test_api_bad_catalog_is_a_400_not_a_500(tmp_path):
    from fastapi.testclient import TestClient

    from shotsieve.server.app import create_app

    folder = tmp_path / "s"
    folder.mkdir()
    import cv2
    import numpy as np

    cv2.imwrite(str(folder / "a.jpg"), np.full((200, 300, 3), 99, np.uint8))
    with Cache.for_folder(folder) as c:
        i = c.ids_for([str(folder / "a.jpg")])[str(folder / "a.jpg")]
    client = TestClient(create_app(folder), headers={"Origin": "http://testserver", "Host": "localhost"})
    r = client.post(f"/api/crop/{i}/accept", json={"folder": str(folder), "rect": [0, 0, 1, 1], "ratio": "original", "catalog": "/nope/x.lrcat"})
    assert r.status_code == 400, r.text
    with Cache.for_folder(folder) as c:
        assert c.crop_history(i) == []


def test_crop_only_sidecar_is_not_a_colour_match_reference_or_training_edit(tmp_path):
    from shotsieve.personal import edit_dataset
    from shotsieve.pipeline import match

    raw = tmp_path / "m.CR3"
    raw.write_bytes(b"x")
    carry = {"Exposure2012": 0.3, "Contrast2012": 12.0}
    assert write_crop(raw, RECT, carry=carry).status == "written"
    assert crs.read_develop_xmp(raw).made_by_us
    with pytest.raises(ValueError, match="written by ShotSieve"):
        match.reference_settings(raw)
    assert edit_dataset._xmp_entries(tmp_path) == []
