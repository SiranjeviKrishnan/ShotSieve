# ruff: noqa: E501
"""T22 crop XMP: the T15 provenance interaction (first, written before the fix), then write_crop /
read_crop / digest v2 / only_ours (11 section 5)."""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

from shotsieve.io import crs, xmp
from shotsieve.io.crs import read_develop_xmp, write_develop

CROP_KEYS = (
    "HasCrop",
    "CropTop",
    "CropLeft",
    "CropBottom",
    "CropRight",
    "CropAngle",
    "CropConstrainToWarp",
)
VALUES = {"Exposure2012": 0.35, "Contrast2012": 10.0}


def _crop_lines(vals: dict[str, str]) -> str:
    return "\n".join(f"{k}={vals[k]}" for k in sorted(vals))


def _add_crop_by_hand(sidecar: Path, *, provenance: bool = True) -> None:
    """What ANY crop writer does to a sidecar: add crs:Crop* attributes (+ our crop provenance)."""
    crop = {
        "HasCrop": "True",
        "CropTop": "0.100000",
        "CropLeft": "0.200000",
        "CropBottom": "0.900000",
        "CropRight": "0.800000",
        "CropAngle": "0",
        "CropConstrainToWarp": "0",
    }
    xmp._register(None)
    tree = ET.parse(sidecar)
    desc = next(tree.getroot().iter(xmp.RDF + "Description"))
    for k, v in crop.items():
        desc.set(xmp._q("crs:" + k), v)
    if provenance:
        desc.set(
            xmp._q("shotsieve:CropDigest"), hashlib.sha256(_crop_lines(crop).encode()).hexdigest()
        )
        desc.set(xmp._q("shotsieve:CropVersion"), "1")
    sidecar.write_bytes(crs.packet_bytes(tree))


def _crop_only_sidecar(sidecar: Path) -> None:
    crop = {
        "HasCrop": "True",
        "CropTop": "0.100000",
        "CropLeft": "0.200000",
        "CropBottom": "0.900000",
        "CropRight": "0.800000",
        "CropAngle": "0",
        "CropConstrainToWarp": "0",
    }
    digest = hashlib.sha256(_crop_lines(crop).encode()).hexdigest()
    attrs = " ".join(f'crs:{k}="{v}"' for k, v in crop.items())
    sidecar.write_text(
        '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description rdf:about="" xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/" '
        'xmlns:shotsieve="http://ns.shotsieve.app/1.0/" crs:HasSettings="True" crs:ProcessVersion="11.0" '
        f'{attrs} shotsieve:CropDigest="{digest}" shotsieve:CropVersion="1"/></rdf:RDF></x:xmpmeta>'
    )


# ---- 1. regression: the T15 / crop provenance collision (must fail on the pre-fix code) --------------


def test_edit_stays_ours_after_a_crop_is_added(tmp_path):
    raw = tmp_path / "IMG_1.CR2"
    raw.write_bytes(b"x")
    assert write_develop(raw, VALUES, profile_hash="h").status == "written"
    sp = xmp.sidecar_path(raw)
    assert read_develop_xmp(raw).ours
    _add_crop_by_hand(sp)
    cur = read_develop_xmp(raw)
    assert cur.has_settings and cur.ours, (
        "a crop must not make ShotSieve's own edit look like the user's"
    )
    res = write_develop(raw, {"Exposure2012": 0.5}, profile_hash="h2")
    assert res.status == "replaced_own"
    after = read_develop_xmp(raw)
    assert after.raw["CropLeft"] == "0.200000" and after.raw["HasCrop"] == "True"  # crop untouched
    assert after.raw["Exposure2012"] == "+0.50" and after.ours


def test_crop_only_sidecar_does_not_block_edit_apply(tmp_path):
    raw = tmp_path / "IMG_2.CR2"
    raw.write_bytes(b"x")
    sp = xmp.sidecar_path(raw)
    _crop_only_sidecar(sp)
    ok, why = crs.writable(raw, "sidecar")
    assert ok, why
    res = write_develop(raw, VALUES, profile_hash="h")
    assert res.status == "written", res
    cur = read_develop_xmp(raw)
    assert cur.raw["CropLeft"] == "0.200000" and cur.raw["Exposure2012"] == "+0.35"


# ---- 2. write_crop / read_crop (11 section 5, acceptance 6 and 7) -------------------------------

import re  # noqa: E402
import threading  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402

from shotsieve.io.crs import CROP_DEFAULTS, read_crop, write_crop  # noqa: E402
from shotsieve.pipeline import crop_geom as geom  # noqa: E402
from tests.test_crs import _jpeg, _tiff, sha  # noqa: E402

RECT = (0.12, 0.08, 0.88, 0.95)  # left, top, right, bottom (crs frame)
USER_SIDECAR = (
    '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
    '<rdf:Description rdf:about="" xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/" '
    'xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmlns:dc="http://purl.org/dc/elements/1.1/" '
    'xmlns:tiff="http://ns.adobe.com/tiff/1.0/" xmp:Rating="4" crs:HasSettings="True" crs:Version="15.0" '
    'crs:ProcessVersion="11.0" crs:Exposure2012="+1.00" crs:Temperature="5200" '
    'crs:CameraProfile="Adobe Standard" crs:HasCrop="False" crs:CropTop="0" crs:CropLeft="0" '
    'crs:CropBottom="1" crs:CropRight="1" crs:CropAngle="0">'
    '<dc:subject><rdf:Bag><rdf:li>wedding</rdf:li></rdf:Bag></dc:subject>'
    '<crs:ToneCurvePV2012><rdf:Seq><rdf:li>0, 0</rdf:li><rdf:li>255, 255</rdf:li></rdf:Seq></crs:ToneCurvePV2012>'
    '</rdf:Description></rdf:RDF></x:xmpmeta>'
)


def _noncrop_attrs(path: Path) -> dict[str, str]:
    root = ET.parse(path).getroot()
    desc = next(root.iter(xmp.RDF + "Description"))
    drop = {"{" + crs.CRS + "}" + k for k in CROP_KEYS}
    return {k: v for k, v in desc.attrib.items() if k not in drop and "shotsieve" not in k}


def test_new_sidecar_has_settings_pv_crop_keys_and_provenance(tmp_path):
    raw = tmp_path / "a.CR2"
    raw.write_bytes(b"x")
    res = write_crop(raw, RECT, version=3)
    assert res.status == "written" and res.path == tmp_path / "a.xmp"
    text = res.path.read_text()
    for frag in ('crs:HasSettings="True"', 'crs:ProcessVersion="11.0"', 'crs:HasCrop="True"',
                 'crs:CropLeft="0.120000"', 'crs:CropTop="0.080000"', 'crs:CropRight="0.880000"',
                 'crs:CropBottom="0.950000"', 'crs:CropAngle="0"', 'crs:CropConstrainToWarp="0"',
                 'shotsieve:CropVersion="3"', "shotsieve:CropDigest="):  # fmt: skip
        assert frag in text, frag
    cur = read_develop_xmp(raw)
    assert cur.crop_ours and cur.has_crop and cur.only_ours and not cur.ours and cur.crop_version == 3
    got = read_crop(raw)
    assert got.ours and got.source == "shotsieve" and got.version == 3
    assert got.rect_crs == pytest.approx(RECT, abs=1e-6)


def test_new_sidecar_carries_the_preset_and_stays_only_ours(tmp_path):
    raw = tmp_path / "a.NEF"
    raw.write_bytes(b"x")
    carry = {"Sharpness": 40.0, "CameraProfile": "Adobe Standard", "HasCrop": "True"}
    assert write_crop(raw, RECT, carry=carry).written
    cur = read_develop_xmp(raw)
    assert cur.raw["Sharpness"] == "40" and cur.raw["CameraProfile"] == "Adobe Standard"
    assert cur.raw["HasCrop"] == "True"  # a carried crop key never wins over the real crop
    assert cur.only_ours and crs.writable(raw, "sidecar")[0]
    assert write_develop(raw, {"Exposure2012": 0.4}, carry=carry, profile_hash="p").status == "written"
    after = read_develop_xmp(raw)
    assert after.ours and after.crop_ours and after.raw["CropLeft"] == "0.120000"


def test_user_develop_settings_are_merged_byte_for_byte_and_backed_up_once(tmp_path):
    raw = tmp_path / "u.ARW"
    raw.write_bytes(b"x")
    sp = tmp_path / "u.xmp"
    sp.write_text(USER_SIDECAR)
    before_attrs = _noncrop_attrs(sp)
    res = write_crop(raw, RECT, version=1)
    assert res.status == "written"
    assert _noncrop_attrs(sp) == before_attrs  # every non-crop attribute byte-for-byte equal
    text = sp.read_text()
    for frag in ("<rdf:li>wedding</rdf:li>", "<rdf:li>0, 0</rdf:li>", "<rdf:li>255, 255</rdf:li>"):
        assert frag in text  # foreign children survive
    cur = read_develop_xmp(raw)
    assert cur.crop_ours and cur.raw["CropLeft"] == "0.120000" and not cur.ours
    baks = sorted(tmp_path.glob("u.xmp.bak-*"))
    assert len(baks) == 1 and baks[0].read_text() == USER_SIDECAR  # the user's original, once
    # one .bak per distinct content: a different crop backs up the first write ...
    write_crop(raw, (0.2, 0.2, 0.7, 0.9), version=2)
    assert len(list(tmp_path.glob("u.xmp.bak-*"))) == 2
    write_crop(raw, (0.2, 0.2, 0.7, 0.9), version=2)  # ... the same content again adds one more,
    assert len(list(tmp_path.glob("u.xmp.bak-*"))) == 3
    write_crop(raw, (0.2, 0.2, 0.7, 0.9), version=2)  # ... but a repeat of that content adds none
    assert len(list(tmp_path.glob("u.xmp.bak-*"))) == 3


def test_users_own_crop_is_never_overwritten(tmp_path):
    raw = tmp_path / "c.RAF"
    raw.write_bytes(b"x")
    sp = tmp_path / "c.xmp"
    sp.write_text(USER_SIDECAR.replace('crs:HasCrop="False" crs:CropTop="0" crs:CropLeft="0"',
                                       'crs:HasCrop="True" crs:CropTop="0.1" crs:CropLeft="0.2"'))
    before = sp.read_bytes()
    assert write_crop(raw, RECT).status == "skipped_user_crop" and sp.read_bytes() == before
    assert write_crop(raw, None).status == "skipped_user_crop" and sp.read_bytes() == before
    got = read_crop(raw, 1)
    assert got.has_crop and not got.ours and got.source == "lightroom"
    assert got.rect_display == pytest.approx((0.2, 0.1, 1.0, 1.0), abs=1e-9)
    # a crop edited by the user after ShotSieve wrote it is theirs too
    raw2 = tmp_path / "d.RAF"
    raw2.write_bytes(b"x")
    write_crop(raw2, RECT)
    s2 = (tmp_path / "d.xmp").read_text().replace('CropLeft="0.120000"', 'CropLeft="0.300000"')
    (tmp_path / "d.xmp").write_text(s2)
    assert not read_develop_xmp(raw2).crop_ours
    assert write_crop(raw2, RECT).status == "skipped_user_crop"


def test_catalog_skips(tmp_path):
    raw = tmp_path / "k.CR3"
    raw.write_bytes(b"x")
    assert write_crop(raw, RECT, catalog_has_crop=True).status == "skipped_user_crop"
    res = write_crop(raw, RECT, catalog_has=True)
    assert res.status == "skipped_catalog" and "Crop saved in ShotSieve only" in res.reason
    assert not (tmp_path / "k.xmp").exists()
    # the sidecar already carries the user's settings: merging is safe, so the catalog flag is moot
    (tmp_path / "k.xmp").write_text(USER_SIDECAR)
    assert write_crop(raw, RECT, catalog_has=True).status == "written"


def test_reset_writes_lightroom_defaults_and_reads_back_no_crop(tmp_path):
    raw = tmp_path / "r.DNG.CR2"
    raw.write_bytes(b"x")
    assert write_crop(raw, None).status == "unchanged" and not (tmp_path / "r.DNG.xmp").exists()
    write_crop(raw, RECT, version=1)
    res = write_crop(raw, None, version=2)
    assert res.status == "replaced_own"
    cur = read_develop_xmp(raw)
    assert {k: cur.raw[k] for k in CROP_KEYS} == CROP_DEFAULTS
    assert cur.crop_ours and cur.crop_version == 2 and not cur.has_crop
    got = read_crop(raw)
    assert not got.has_crop and got.rect_display is None and got.source == "none"
    assert write_crop(raw, RECT, version=3).written  # and a new crop can follow a reset


@pytest.mark.parametrize("o", range(1, 9))
def test_roundtrip_all_eight_orientations_to_1e6(tmp_path, o):
    raw = tmp_path / f"o{o}.CR2"
    raw.write_bytes(b"x")
    for rect in (geom.Rect(0.1234567, 0.0765432, 0.6543219, 0.8123456), geom.Rect(0, 0, 1, 1)):
        c = geom.to_crs(rect, o)
        assert write_crop(raw, c.as_tuple(), version=1).written
        got = read_crop(raw, o)
        assert got.rect_display == pytest.approx(rect.as_tuple(), abs=1e-6)
    # a tiff:Orientation in the sidecar (a Lightroom rotation) wins over the EXIF value
    sp = tmp_path / f"o{o}.xmp"
    tiff = 'xmlns:tiff="http://ns.adobe.com/tiff/1.0/" tiff:Orientation="6"'
    sp.write_text(sp.read_text().replace("<rdf:Description", f"<rdf:Description {tiff}", 1))
    got = read_crop(raw, o)
    assert got.orientation == 6
    want = geom.from_crs(geom.Rect(*got.rect_crs), 6).as_tuple()
    assert got.rect_display == pytest.approx(want, abs=1e-9)


def test_legacy_v1_edit_digest_is_still_ours_and_is_resigned_by_write_crop(tmp_path):
    raw = tmp_path / "v1.CR2"
    raw.write_bytes(b"x")
    assert write_develop(raw, VALUES, profile_hash="h").written
    sp = tmp_path / "v1.xmp"
    # downgrade to what an older build wrote: no DigestV, digest over the whole crs set
    root = ET.parse(sp).getroot()
    desc = next(root.iter(xmp.RDF + "Description"))
    del desc.attrib["{" + xmp.NS["shotsieve"] + "}DigestV"]
    cur = read_develop_xmp(raw)
    desc.set(xmp._q("shotsieve:EditDigest"), crs.crs_digest(cur.raw, cur.struct))
    xmp._register(None)
    sp.write_bytes(crs.packet_bytes(ET.ElementTree(root)))
    v1 = read_develop_xmp(raw)
    assert v1.ours and v1.digest_v == "1"
    assert write_crop(raw, RECT, version=1).written  # re-signs as v2 BEFORE the crop is added
    v2 = read_develop_xmp(raw)
    assert v2.ours and v2.digest_v == "2" and v2.crop_ours
    assert v2.raw["Exposure2012"] == "+0.35"
    assert write_develop(raw, {"Exposure2012": 0.1}, profile_hash="h3").status == "replaced_own"


def test_user_edit_after_a_crop_still_makes_the_edit_theirs(tmp_path):
    raw = tmp_path / "e.CR2"
    raw.write_bytes(b"x")
    write_develop(raw, VALUES, profile_hash="h")
    write_crop(raw, RECT)
    sp = tmp_path / "e.xmp"
    sp.write_text(sp.read_text().replace('Exposure2012="+0.35"', 'Exposure2012="+0.90"'))
    cur = read_develop_xmp(raw)
    assert not cur.ours and cur.crop_ours  # the edit changed (theirs); the crop did not
    assert write_develop(raw, VALUES, profile_hash="h").status == "skipped_user_settings"


def test_crop_edit_apply_crop_apply_cycle_keeps_the_crop(tmp_path):
    raw = tmp_path / "cy.CR2"
    raw.write_bytes(b"x")
    assert write_develop(raw, VALUES, profile_hash="a").status == "written"
    assert write_crop(raw, RECT, version=1).status == "written"
    assert write_develop(raw, {"Exposure2012": 0.7}, profile_hash="b").status == "replaced_own"
    assert write_crop(raw, (0.3, 0.2, 0.8, 0.9), version=2).status == "replaced_own"
    assert write_develop(raw, {"Exposure2012": 0.2}, profile_hash="c").status == "replaced_own"
    cur = read_develop_xmp(raw)
    assert cur.ours and cur.crop_ours and cur.raw["CropLeft"] == "0.300000"
    assert cur.raw["Exposure2012"] == "+0.20" and "Contrast2012" not in cur.raw


def test_concurrent_write_decision_and_write_crop_lose_nothing(tmp_path):
    raw = tmp_path / "t.CR2"
    raw.write_bytes(b"x")
    errors: list[BaseException] = []

    def decisions():
        try:
            for i in range(15):
                xmp.write_sidecar(raw, rating=(i % 5) + 1, decision="keep")
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    def crops():
        try:
            for i in range(15):
                write_crop(raw, (0.05 + i * 0.01, 0.1, 0.9, 0.95), version=i + 1)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    ts = [threading.Thread(target=decisions), threading.Thread(target=crops)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errors, errors
    assert xmp.read_sidecar(raw).decision == "keep"
    cur = read_develop_xmp(raw)
    assert cur.crop_ours and cur.crop_version == 15
    ET.parse(tmp_path / "t.xmp")  # still well-formed XML


def test_jpeg_embed_copy_writes_crop_and_never_touches_the_original(tmp_path):
    src = _jpeg(tmp_path / "IMG.jpg")
    before = sha(src)
    res = write_crop(src, RECT, mode="embed", out_dir=tmp_path / "out", version=1)
    assert res.written and sha(src) == before
    assert np.array_equal(np.asarray(Image.open(src)), np.asarray(Image.open(res.path)))
    with Image.open(res.path) as im:
        desc = im.getxmp()["xmpmeta"]["RDF"]["Description"]
    assert desc["CropTop"] == "0.080000" and desc["HasCrop"] == "True"
    # an existing ShotSieve copy with an edit is merged, not replaced
    ed = write_develop(src, VALUES, profile_hash="h", mode="embed", out_dir=tmp_path / "out2")
    assert ed.written
    res2 = write_crop(src, RECT, mode="embed", out_dir=tmp_path / "out2", version=2)
    assert res2.written
    cur = read_develop_xmp(res2.path)
    assert cur.ours and cur.crop_ours and cur.raw["Exposure2012"] == "+0.35"
    assert sha(src) == before
    # reset of the copy
    res3 = write_crop(src, None, mode="embed", out_dir=tmp_path / "out2", version=3)
    assert res3.written and not read_crop(res3.path).has_crop
    assert write_crop(src, RECT, mode="embed", out_dir=tmp_path).status == "error"


def test_dng_embed_copy_and_unsupported_formats(tmp_path):
    src = _tiff(tmp_path / "t.dng", False)
    before = sha(src)
    res = write_crop(src, RECT, mode="embed", out_dir=tmp_path / "o", version=1)
    assert res.written and sha(src) == before
    assert read_crop(res.path).rect_crs == pytest.approx(RECT, abs=1e-6)
    png = tmp_path / "a.png"
    png.write_bytes(b"x")
    assert write_crop(png, RECT, mode="embed", out_dir=tmp_path / "o").status == "error"
    with pytest.raises(ValueError):
        write_crop(png, RECT, mode="nope")


def test_unreadable_sidecar_is_skipped_untouched(tmp_path):
    raw = tmp_path / "bad.CR2"
    raw.write_bytes(b"x")
    (tmp_path / "bad.xmp").write_text("<<< not xml")
    assert write_crop(raw, RECT).status == "skipped_unreadable"
    assert (tmp_path / "bad.xmp").read_text() == "<<< not xml"


def test_crs_values_use_lightroom_number_format(tmp_path):
    raw = tmp_path / "f.CR2"
    raw.write_bytes(b"x")
    write_crop(raw, (1e-9, 0.5, 1.2, 0.99999999))
    text = (tmp_path / "f.xmp").read_text()
    assert re.search(r'crs:CropLeft="0\.000000"', text) and 'crs:CropRight="1.000000"' in text
