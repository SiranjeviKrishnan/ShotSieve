# ruff: noqa: E501
"""T15: crs: XMP read/format/write, JPEG/DNG embedded copies, never-overwrite, carry set."""

from __future__ import annotations

import hashlib
import re
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from shotsieve.io import dng_xmp, jpeg_xmp, xmp
from shotsieve.io.crs import CRS, format_value, read_develop_xmp, write_develop
from shotsieve.personal.edit_dataset import carry_set

VALUES = {
    "Exposure2012": 0.354, "Contrast2012": 24.6, "Highlights2012": -40.2, "Shadows2012": 31.0,
    "Whites2012": 10.0, "Blacks2012": -20.0, "Vibrance": 15.0, "Saturation": -5.0,
    "Temperature": 5432.4, "Tint": 7.6,
}  # fmt: skip


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_format_value_uses_lightroom_forms_and_clamps():
    assert format_value("Exposure2012", 0.354) == "+0.35"
    assert format_value("Exposure2012", -1.0) == "-1.00"
    assert format_value("Exposure2012", 0.0) == "+0.00"
    assert format_value("Exposure2012", 9.0) == "+5.00"  # range -5..+5
    assert format_value("Temperature", 5432.4) == "5432"
    assert format_value("Temperature", 900) == "2000" and format_value("Temperature", 9e4) == "50000"
    assert format_value("Tint", 300) == "+150" and format_value("Tint", -151) == "-150"
    assert format_value("Contrast2012", 24.6) == "+25" and format_value("Vibrance", -140) == "-100"
    assert format_value("IncrementalTemperature", 130) == "+100"


def test_sidecar_is_valid_xmp_with_namespace_process_version_and_ranges(tmp_path):
    raw = tmp_path / "IMG_0001.CR2"
    raw.write_bytes(b"not a real raw")
    res = write_develop(raw, VALUES, profile_hash="abc123", confidence={"Exposure2012": 0.91},
                        wb=True)  # fmt: skip
    assert res.status == "written" and res.path == tmp_path / "IMG_0001.xmp"
    text = res.path.read_text()
    assert text.startswith('<?xpacket begin="') and text.rstrip().endswith('<?xpacket end="w"?>')
    root = ET.fromstring(text)  # well-formed XML
    desc = next(root.iter("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description"))
    ns = "{" + CRS + "}"
    assert CRS == "http://ns.adobe.com/camera-raw-settings/1.0/"
    attrs = {k[len(ns):]: v for k, v in desc.attrib.items() if k.startswith(ns)}
    assert attrs["ProcessVersion"] == "11.0" and attrs["HasSettings"] == "True"
    assert attrs["WhiteBalance"] == "Custom"
    assert attrs["Exposure2012"] == "+0.35" and attrs["Contrast2012"] == "+25"
    assert attrs["Temperature"] == "5432" and attrs["Tint"] == "+8"
    ranges = {"Exposure2012": (-5, 5), "Temperature": (2000, 50000), "Tint": (-150, 150)}
    for k, v in attrs.items():
        if k in ("HasSettings", "ProcessVersion", "WhiteBalance"):
            continue
        lo, hi = ranges.get(k, (-100, 100))
        assert re.fullmatch(r"[+-]\d+(\.\d\d)?|\d+", v), (k, v)
        assert lo <= float(v) <= hi, (k, v)
    # the crs prefix is declared with Adobe's URI and our provenance is in our own namespace
    assert f'xmlns:crs="{CRS}"' in text
    assert "shotsieve:EditProfile=\"abc123\"" in text and "shotsieve:EditConfidence" in text


def test_roundtrip_and_golden_string(tmp_path):
    raw = tmp_path / "a.NEF"
    raw.write_bytes(b"x")
    write_develop(raw, {"Exposure2012": 0.35, "Contrast2012": 10.0}, profile_hash="h",
                  confidence={"Exposure2012": 0.5}, wb=False)  # fmt: skip
    got = read_develop_xmp(raw)
    assert got is not None and got.has_settings and got.ours and got.profile == "h"
    assert got.settings["Exposure2012"] == 0.35 and got.settings["Contrast2012"] == 10.0
    assert got.settings["ProcessVersion"] == 11.0
    assert "WhiteBalance" not in got.settings  # only written when WB is written
    body = (tmp_path / "a.xmp").read_text()
    assert "<x:xmpmeta" in body and f'xmlns:crs="{CRS}"' in body
    for frag in ('crs:Exposure2012="+0.35"', 'crs:Contrast2012="+10"', 'crs:HasSettings="True"',
                 'crs:ProcessVersion="11.0"', 'shotsieve:EditProfile="h"'):  # fmt: skip
        assert frag in body


def test_merge_keeps_foreign_content_and_ratings(tmp_path):
    raw = tmp_path / "a.ARW"
    raw.write_bytes(b"x")
    xmp.write_sidecar(raw, rating=4, label="Red", decision="keep")
    xmp.merge_keywords(raw, ["wedding"], ["Ann"])
    write_develop(raw, {"Exposure2012": 0.2}, profile_hash="p1")
    data = xmp.read_sidecar(raw)
    assert data.rating == 4 and data.label == "Red" and data.decision == "keep"
    assert xmp.read_keywords(raw) == (["wedding"], ["Ann"])
    # and the XMP writers keep our crs block when they run afterwards
    xmp.write_sidecar(raw, decision="review")
    assert read_develop_xmp(raw).settings["Exposure2012"] == 0.2


def test_never_overwrites_users_develop_settings_but_replaces_own(tmp_path):
    raw = tmp_path / "a.RAF"
    raw.write_bytes(b"x")
    sp = tmp_path / "a.xmp"
    sp.write_text(
        '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf='
        '"http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description rdf:about="" '
        f'xmlns:crs="{CRS}" crs:HasSettings="True" crs:Exposure2012="+1.00"/></rdf:RDF></x:xmpmeta>'
    )
    before = sp.read_bytes()
    res = write_develop(raw, {"Exposure2012": 0.2}, profile_hash="p")
    assert res.status == "skipped_user_settings" and sp.read_bytes() == before
    # our own earlier write may be replaced (and is not a "user edit")
    raw2 = tmp_path / "b.RAF"
    raw2.write_bytes(b"x")
    assert write_develop(raw2, {"Exposure2012": 0.2, "Contrast2012": 5.0}, profile_hash="p1").written
    r2 = write_develop(raw2, {"Exposure2012": -0.4}, profile_hash="p2")
    assert r2.status == "replaced_own"
    s = read_develop_xmp(raw2).settings
    assert s["Exposure2012"] == -0.4 and "Contrast2012" not in s  # fully replaced
    # a catalog that already has develop settings for the image blocks the write
    raw3 = tmp_path / "c.RAF"
    raw3.write_bytes(b"x")
    assert write_develop(raw3, {"Exposure2012": 0.2}, profile_hash="p", catalog_has=True).status == (
        "skipped_catalog"
    )
    assert not (tmp_path / "c.xmp").exists()
    # unreadable sidecars are skipped untouched
    raw4 = tmp_path / "d.RAF"
    raw4.write_bytes(b"x")
    (tmp_path / "d.xmp").write_text("<<< not xml")
    assert write_develop(raw4, {"Exposure2012": 0.2}, profile_hash="p").status == (
        "skipped_unreadable"
    )
    assert (tmp_path / "d.xmp").read_text() == "<<< not xml"


def test_carry_set_requires_80_percent_agreement():
    rows = [{"Sharpness": 40.0, "LuminanceSmoothing": 10.0, "Exposure2012": float(i)} for i in range(10)]
    rows[0]["Sharpness"] = 25.0  # 9/10 agree
    for r in rows[:4]:
        r["LuminanceSmoothing"] = 0.0  # 6/10: not carried
    rows[1]["Rare"] = 1.0
    out = carry_set(rows)
    assert out == {"Sharpness": 40.0}  # targets, rare keys and non-consensus keys are not carried


def test_carry_is_written_verbatim(tmp_path):
    raw = tmp_path / "a.CR3"
    raw.write_bytes(b"x")
    write_develop(raw, {"Exposure2012": 0.1}, carry={"Sharpness": 40.0, "CameraProfile": "Adobe Standard",
                                                      "SharpenRadius": 1.5}, profile_hash="p")  # fmt: skip
    s = read_develop_xmp(raw).settings
    assert s["Sharpness"] == 40.0 and s["SharpenRadius"] == 1.5
    assert s["CameraProfile"] == "Adobe Standard"
    assert 'crs:Sharpness="40"' in (tmp_path / "a.xmp").read_text()


# ---- JPEG / DNG embedded copies ---------------------------------------------------------------


def _jpeg(path: Path, with_xmp: bool = False) -> Path:
    img = (np.random.default_rng(3).random((96, 128, 3)) * 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if with_xmp:
        packet = b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description rdf:about="" xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmp:Rating="3"/></rdf:RDF></x:xmpmeta>'
        out = path.with_name("seed.jpg")
        jpeg_xmp.write_copy(path, out, packet)
        out.replace(path)
    return path


def _segments(p: Path):
    data = p.read_bytes()
    return [(m, data[s:e]) for m, s, e in jpeg_xmp._segments(data)], data


@pytest.mark.parametrize("with_xmp", [False, True])
def test_jpeg_embedded_copy(tmp_path, with_xmp):
    src = _jpeg(tmp_path / "IMG.jpg", with_xmp)
    before = sha(src)
    res = write_develop(src, VALUES, profile_hash="h", wb=True, mode="embed", out_dir=tmp_path / "out")
    assert res.written and res.path == tmp_path / "out" / "IMG.jpg"
    assert sha(src) == before  # original untouched
    # pixels identical, every non-XMP segment byte-identical, Pillow reads crs from the copy
    a, b = np.asarray(Image.open(src)), np.asarray(Image.open(res.path))
    assert np.array_equal(a, b)
    sa, da = _segments(src)
    sb, db = _segments(res.path)
    strip = lambda segs: [  # noqa: E731
        s for m, s in segs if not (m == 0xE1 and s[4:4 + len(jpeg_xmp.XMP_HEADER)] == jpeg_xmp.XMP_HEADER)
    ]
    assert strip(sa) == strip(sb)
    # the entropy-coded scan data after the last header segment is identical
    assert da[len(da) - len(da) :] == da and da[-200:] == db[-200:]
    assert da.endswith(b"\xff\xd9") and db.endswith(b"\xff\xd9")
    with Image.open(res.path) as im:
        x = im.getxmp()
    desc = x["xmpmeta"]["RDF"]["Description"]
    assert desc["Exposure2012"] == "+0.35" and desc["ProcessVersion"] == "11.0"
    if with_xmp:
        assert desc["Rating"] == "3"  # the file's own XMP content is merged, not replaced
    got = read_develop_xmp(res.path)
    assert got.ours and got.settings["Contrast2012"] == 25.0


def test_jpeg_never_overwrites_user_settings_or_oversize(tmp_path):
    src = _jpeg(tmp_path / "a.jpg")
    first = write_develop(src, {"Exposure2012": 0.3}, profile_hash="h", mode="embed", out_dir=tmp_path / "o1")
    # pretend the user then edited that copy in Lightroom: develop settings without our provenance
    user = tmp_path / "b.jpg"
    packet = (b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
              b'<rdf:Description rdf:about="" xmlns:crs="' + CRS.encode() + b'" crs:HasSettings="True" crs:Exposure2012="+1.00"/></rdf:RDF></x:xmpmeta>')
    jpeg_xmp.write_copy(src, user, packet)
    res = write_develop(user, {"Exposure2012": 0.3}, profile_hash="h", mode="embed", out_dir=tmp_path / "o2")
    assert res.status == "skipped_user_settings" and not (tmp_path / "o2").exists()
    assert first.written
    with pytest.raises(jpeg_xmp.JpegError):
        jpeg_xmp.write_copy(src, tmp_path / "big.jpg", b"x" * 70_000)
    # writing the copy into the original's own folder is refused
    assert write_develop(src, {"Exposure2012": 0.3}, profile_hash="h", mode="embed", out_dir=tmp_path).status == "error"


def _tiff(path: Path, with_700: bool, big_endian: bool = False) -> Path:
    e = ">" if big_endian else "<"
    strip = bytes(range(64))
    entries = [
        (256, 3, 1, 8), (257, 3, 1, 8), (258, 3, 1, 8), (259, 3, 1, 1), (262, 3, 1, 1),
        (273, 4, 1, 8), (278, 3, 1, 8), (279, 4, 1, 64),
    ]  # fmt: skip
    if with_700:
        entries.append((700, 1, 5, 8 + 64 + 8))
    ifd_off = 8 + len(strip)
    buf = bytearray(b"MM\x00\x2a" if big_endian else b"II\x2a\x00")
    buf += struct.pack(e + "I", ifd_off)
    buf += strip
    buf += struct.pack(e + "H", len(entries))
    for t, ty, c, v in entries:
        buf += struct.pack(e + "HHII", t, ty, c, v) if ty not in (3,) else (
            struct.pack(e + "HHI", t, ty, c) + struct.pack(e + "H", v) + b"\x00\x00")
    buf += struct.pack(e + "I", 0)
    if with_700:
        buf += b"<old>"
    path.write_bytes(bytes(buf))
    return path


@pytest.mark.parametrize("with_700", [True, False])
@pytest.mark.parametrize("big", [False, True])
def test_dng_xmp_copy_synthetic_tiff(tmp_path, with_700, big):
    src = _tiff(tmp_path / "t.dng", with_700, big)
    before = sha(src)
    packet = b"<x:xmpmeta>hello</x:xmpmeta>"
    dst = dng_xmp.write_copy(src, tmp_path / "out" / "t.dng", packet)
    assert sha(src) == before
    assert dng_xmp.read_packet(dst) == packet
    data_a, data_b = src.read_bytes(), dst.read_bytes()
    assert data_b[8:72] == data_a[8:72]  # the strip data did not move
    tags = dng_xmp.ifd0_tags(dst)
    assert sorted(tags) == sorted([256, 257, 258, 259, 262, 273, 278, 279, 700])
    assert [t for t in dng_xmp.ifd0_tags(dst)] == sorted(dng_xmp.ifd0_tags(dst))


@pytest.mark.parametrize("name", ["pentax_kx.dng", "panasonic_lx1.dng"])
def test_dng_embedded_copy_real(gen_dir, tmp_path, name):
    import rawpy

    src = gen_dir / "raw" / name
    before = sha(src)
    res = write_develop(src, VALUES, profile_hash="h", wb=True, mode="embed", out_dir=tmp_path)
    assert res.written and sha(src) == before
    with rawpy.imread(str(src)) as a, rawpy.imread(str(res.path)) as b:
        assert np.array_equal(a.raw_image, b.raw_image)
    pkt = dng_xmp.read_packet(res.path)
    root = ET.fromstring(pkt.strip())  # tag 700 parses as XML
    assert "Exposure2012" in pkt.decode()
    assert root is not None
    got = read_develop_xmp(res.path)
    assert got.ours and got.settings["Exposure2012"] == 0.35 and got.settings["Temperature"] == 5432.0
    # a second write replaces our own packet and keeps the DNG readable
    res2 = write_develop(res.path, {"Exposure2012": -0.5}, profile_hash="h2", mode="embed", out_dir=tmp_path / "again")
    assert res2.status == "replaced_own"
    assert read_develop_xmp(res2.path).settings["Exposure2012"] == -0.5
    with rawpy.imread(str(src)) as a2, rawpy.imread(str(res2.path)) as c:
        assert np.array_equal(c.raw_image, a2.raw_image)


def test_dng_preview_application_detection(gen_dir):
    # PreviewApplicationName: the Pentax original has none, the LX1 file went through Adobe DNG Converter
    assert dng_xmp.preview_application(gen_dir / "raw" / "pentax_kx.dng") is None
    assert "adobe" in dng_xmp.preview_application(gen_dir / "raw" / "panasonic_lx1.dng").lower()
