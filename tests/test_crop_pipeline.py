# ruff: noqa: E501
"""T22 pipeline details: full-resolution dims and orientation (RAW with a patched Orientation tag,
JPEGs with every EXIF orientation), the crs frame end to end, scoring on orientation copies."""

from __future__ import annotations

import shutil
import struct
from pathlib import Path

import cv2
import numpy as np
import piexif
import pytest

from shotsieve.io import crs
from shotsieve.pipeline import crop as pcrop
from shotsieve.pipeline import crop_geom as g
from shotsieve.pipeline.cache import Cache
from shotsieve.pipeline.crop_geom import Rect


def patched_dng(src: Path, dst: Path, orientation: int) -> Path:
    data = bytearray(src.read_bytes())
    e = "<" if data[:2] == b"II" else ">"
    off = struct.unpack(e + "I", data[4:8])[0]
    n = struct.unpack(e + "H", data[off : off + 2])[0]
    for i in range(n):
        pos = off + 2 + i * 12
        if struct.unpack(e + "H", data[pos : pos + 2])[0] == 274:
            data[pos + 8 : pos + 10] = struct.pack(e + "H", orientation)
            dst.write_bytes(bytes(data))
            return dst
    raise AssertionError("no Orientation tag in the fixture DNG")


@pytest.mark.parametrize("o", [1, 3, 6, 8])
def test_raw_dims_and_orientation_follow_the_tag(tmp_path, gen_dir, o):
    p = patched_dng(gen_dir / "raw" / "pentax_kx.dng", tmp_path / "x.pef", o)
    fw, fh, got = pcrop.read_image_facts(str(p))
    assert got == o
    assert (fw, fh) == ((2868, 4309) if o in (6, 8) else (4309, 2868))


@pytest.mark.parametrize("o", range(1, 9))
def test_jpeg_dims_swap_for_orientations_5_to_8(tmp_path, o):
    p = tmp_path / "a.jpg"
    cv2.imwrite(str(p), np.zeros((300, 500, 3), np.uint8))
    piexif.insert(piexif.dump({"0th": {piexif.ImageIFD.Orientation: o}}), str(p))
    fw, fh, got = pcrop.read_image_facts(str(p))
    assert got == o and (fw, fh) == ((300, 500) if o >= 5 else (500, 300))


def test_portrait_raw_sidecar_uses_the_stored_frame(tmp_path, gen_dir):
    """Orientation 6: a display-space crop is written in sensor space and reads back identically."""
    folder = tmp_path / "s"
    folder.mkdir()
    raw = patched_dng(gen_dir / "raw" / "pentax_kx.dng", folder / "p.pef", 6)
    with Cache.for_folder(folder) as c:
        i = c.ids_for([str(raw)])[str(raw)]
        info = pcrop.ImageInfo(str(raw), 2868, 4309, 6, "raw")
        rect = Rect(0.10, 0.20, 0.60, 0.70)
        out = pcrop.write_xmp(folder, i, str(raw), rect, 1, info, pcrop.Settings())
        assert out["status"] == "written"
        cur = crs.read_develop_xmp(raw)
        left, top, right, bottom = (
            float(cur.raw[k]) for k in ("CropLeft", "CropTop", "CropRight", "CropBottom")
        )
        # display (x, y) of a rot90-CW image was stored at (y, 1 - x)
        assert (left, top, right, bottom) == pytest.approx((0.20, 0.40, 0.70, 0.90), abs=1e-6)
        assert crs.read_crop(raw, 6).rect_display == pytest.approx(rect.as_tuple(), abs=1e-6)
        shutil.rmtree(folder / ".shotsieve", ignore_errors=True)


def test_sidecar_rotation_overrides_the_exif_orientation(tmp_path, gen_dir):
    raw = patched_dng(gen_dir / "raw" / "pentax_kx.dng", tmp_path / "r.pef", 1)
    (tmp_path / "r.xmp").write_text(
        '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description rdf:about="" xmlns:tiff="http://ns.adobe.com/tiff/1.0/" tiff:Orientation="6"/></rdf:RDF></x:xmpmeta>'
    )
    assert pcrop.effective_orientation(str(raw), 1) == 6
    assert pcrop.effective_orientation(str(tmp_path / "none.pef"), 3) == 3


def test_every_returned_crop_is_exact_to_a_pixel_for_all_orientations(tmp_path):
    """Acceptance 2 end to end: suggestions on JPEGs with each orientation tag."""
    from shotsieve.config import Settings

    folder = tmp_path / "s"
    folder.mkdir()
    img = np.full((400, 600, 3), 120, np.uint8)
    cv2.rectangle(img, (200, 100), (330, 260), (250, 200, 160), -1)
    names = []
    for o in range(1, 9):
        n = f"o{o}.jpg"
        cv2.imwrite(str(folder / n), img)
        piexif.insert(piexif.dump({"0th": {piexif.ImageIFD.Orientation: o}}), str(folder / n))
        names.append(n)
    with Cache.for_folder(folder) as c:
        ids = c.ids_for([str(folder / n) for n in names])
        for n in names:
            p = str(folder / n)
            for ratio in ("1:1", "4:5", "16:9", "1.91:1"):
                r = pcrop.suggest_crops(
                    folder, c, ids[p], p, Settings(), ratio=ratio, log_event=False
                )
                info = pcrop._info(folder, c, ids[p], p, Settings())
                rt = g.parse_ratio(r["ratio"])
                for cand in r["candidates"]:
                    assert g.ratio_error_px(tuple(cand["rect_px"]), rt, info.fw, info.fh) <= 1.0
                    x0, y0, x1, y1 = cand["rect_px"]
                    assert 0 <= x0 < x1 <= info.fw and 0 <= y0 < y1 <= info.fh
