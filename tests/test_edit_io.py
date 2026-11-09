# ruff: noqa: E501
"""T15: Lightroom develop-settings reader, as-shot WB, edit statistics, dataset targets."""

from __future__ import annotations

import hashlib
import shutil
import sqlite3
import struct
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from shotsieve.features import raw_wb
from shotsieve.features.color_stats import EDIT_STAT_NAMES, N_STATS, edit_stats
from shotsieve.io import dng_xmp, lrcat
from shotsieve.io.exif import ImageMeta
from shotsieve.personal import edit_dataset
from shotsieve.personal.edit_profile import FMT_DNG, FMT_OTHER, FMT_RAW

ROOT = Path(__file__).resolve().parents[1]

LUA = """s = {
\tAutoLateralCA = 1,
\tBlacks2012 = -12,
\tContrast2012 = +25,
\tExposure2012 = 0.35,
\tToneCurvePV2012 = {
\t\t0,
\t\t0,
\t\t255,
\t\t255,
\t},
\tToneCurveName2012 = "Medium {Contrast}",
\tLook = {
\t\tName = "x \\" } y",
\t\tParameters = {
\t\t\tAmount = 1,
\t\t},
\t},
\tWhiteBalance = "Custom",
\tTemperature = 5500,
\tProcessVersion = "11.0",
\tHasCrop = false,
\tCropTop = 0.0625,
}"""


def test_lua_parser_nested_tables_strings_negatives_and_bools():
    d = lrcat.parse_lua_table(LUA)
    assert d["Blacks2012"] == -12.0 and d["Contrast2012"] == 25.0 and d["Exposure2012"] == 0.35
    assert d["ToneCurveName2012"] == "Medium {Contrast}"  # braces inside a string
    assert d["WhiteBalance"] == "Custom" and d["Temperature"] == 5500.0
    assert d["ProcessVersion"] == "11.0" and d["HasCrop"] == "False" and d["CropTop"] == 0.0625
    assert "ToneCurvePV2012" not in d and "Look" not in d and "Parameters" not in d
    assert "Amount" not in d and "Name" not in d  # keys of nested tables never leak out
    assert lrcat.parse_lua_table("") == {} and lrcat.parse_lua_table("s = { }") == {}


def _cat(path: Path, rows: list[dict], virtual: bool = False) -> Path:
    spec_mod = pytest.importorskip("importlib.util")
    spec = spec_mod.spec_from_file_location("mes_io", ROOT / "scripts" / "make_edit_styles.py")
    import sys

    mod = spec_mod.module_from_spec(spec)
    sys.modules["mes_io"] = mod
    spec.loader.exec_module(mod)
    mod.build_catalog(path, rows)
    if virtual:
        con = sqlite3.connect(path)
        con.execute("INSERT INTO Adobe_images VALUES (99, 1, 0, NULL, '', 1)")  # a virtual copy
        con.execute("INSERT INTO Adobe_imageDevelopSettings VALUES (99, 99, ?, '11.0')", (LUA,))
        con.commit()
        con.close()
    return path


def _row(base, ext, develop, root="/photos"):
    return {"root": root, "rel": "2024/", "base": base, "ext": ext, "develop": develop}


def test_read_develop_filters_pv_unedited_and_virtual_copies(tmp_path):
    rows = [
        _row("edited", "CR2", {"Exposure2012": 0.5, "ProcessVersion": "11.0", "Sharpness": 40}),
        _row("only_contrast", "jpg", {"Contrast2012": 10, "ProcessVersion": "11.0"}),
        _row("pv2010", "NEF", {"Exposure": 0.5, "Brightness": 20, "ProcessVersion": "5.7"}),
        _row("untouched", "ARW", {"Exposure2012": 0, "Contrast2012": 0, "ProcessVersion": "11.0",
                                  "WhiteBalance": "As Shot"}),  # fmt: skip
        _row("custom_wb", "dng", {"Exposure2012": 0, "WhiteBalance": "Custom", "Temperature": 4800,
                                  "Tint": 5, "ProcessVersion": "11.0"}),  # fmt: skip
    ]
    cat = _cat(tmp_path / "c.lrcat", rows, virtual=True)
    before = hashlib.sha256(cat.read_bytes()).hexdigest()
    got = lrcat.read_develop(cat)
    assert hashlib.sha256(cat.read_bytes()).hexdigest() == before  # read-only
    names = {Path(e.path).name: e for e in got}
    assert set(names) == {"edited.CR2", "only_contrast.jpg", "custom_wb.dng"}
    assert names["edited.CR2"].fmt == "raw" and names["only_contrast.jpg"].fmt == "jpeg"
    assert names["custom_wb.dng"].fmt == "dng" and names["edited.CR2"].pv == 11.0
    assert names["edited.CR2"].settings["Sharpness"] == 40.0
    assert names["edited.CR2"].path == "/photos/2024/edited.CR2"


def test_read_develop_always_reads_a_private_copy(tmp_path, monkeypatch):
    """n1: SQLite never opens the user's catalog, locked or not: every read is on the copy."""
    cat = _cat(tmp_path / "c.lrcat", [_row("a", "CR2", {"Exposure2012": 1, "ProcessVersion": "11.0"})])
    real = lrcat._develop_rows
    calls = []

    def spy(db):
        calls.append(db)
        return real(db)

    monkeypatch.setattr(lrcat, "_develop_rows", spy)
    got = lrcat.read_develop(cat)
    assert len(got) == 1 and len(calls) == 1 and calls[0].parent != cat.parent


# ---- as-shot white balance ----------------------------------------------------------------------


def test_cct_of_standard_illuminants():
    for xyz, cct in (((0.9505, 1.0, 1.0888), 6504), ((1.0985, 1.0, 0.3558), 2856),
                     ((0.9642, 1.0, 0.8251), 5003)):  # fmt: skip
        got, duv = raw_wb.xyz_to_cct_duv(np.array(xyz))
        assert abs(got - cct) < 0.03 * cct and abs(duv) < 0.01, (cct, got, duv)
    assert raw_wb.xyz_to_cct_duv(np.array([0.0, 0.0, 0.0])) is None


def _patched_neutral(src: Path, dst: Path, neutral: tuple[float, float, float]) -> Path:
    """Copy a DNG and overwrite its AsShotNeutral RATIONALs (a synthetic lighting change)."""
    shutil.copy(src, dst)
    data = bytearray(dst.read_bytes())
    e = "<" if data[:2] == b"II" else ">"
    ent = dng_xmp.ifd0_tags(dst)[dng_xmp.TAG_AS_SHOT_NEUTRAL]
    off = struct.unpack(e + "I", ent.raw)[0]
    for i, v in enumerate(neutral):
        data[off + 8 * i : off + 8 * i + 8] = struct.pack(e + "II", round(v * 100000), 100000)
    dst.write_bytes(bytes(data))
    return dst


def test_as_shot_mired_is_monotonic_with_the_scene_neutral(gen_dir, tmp_path):
    src = gen_dir / "raw" / "pentax_kx.dng"
    base = raw_wb.as_shot(src)
    assert base is not None and 100 < base[0] < 500
    # a tungsten-lit frame has a *blue-starved* camera neutral (low B/G), daylight a high one
    mired = []
    for i, b in enumerate((0.25, 0.32, 0.45, 0.60, 0.80)):
        p = _patched_neutral(src, tmp_path / f"n{i}.dng", (0.77, 0.97, b))
        r = raw_wb.as_shot(p)
        assert r is not None
        mired.append(r[0])
    assert all(a > b for a, b in zip(mired, mired[1:], strict=False)), mired  # warmer -> higher
    assert dng_xmp.as_shot_neutral(src) == pytest.approx((0.7688, 0.9697, 0.3232), abs=1e-3)
    assert raw_wb.as_shot(tmp_path / "missing.cr2") is None


# ---- statistics ----------------------------------------------------------------------------------


def test_edit_stats_shape_nan_tolerance_and_speed(rgb):
    img = rgb("lena.jpg")
    v = edit_stats(img)
    assert v.shape == (N_STATS,) == (len(EDIT_STAT_NAMES),) and v.dtype == np.float32
    idx = {n: i for i, n in enumerate(EDIT_STAT_NAMES)}
    assert 40 <= N_STATS <= 50
    assert np.isnan(v[idx["wb_mired"]]) and np.isnan(v[idx["log2_iso"]])  # JPEG, no EXIF
    assert np.isfinite(v[idx["ev"]]) and np.isfinite(v[idx["gw_rg"]])
    meta = ImageMeta(timestamp=1.7e9, ts_source="exif", make="Canon", model="R5", iso=400,
                     exposure_time=1 / 250, fnumber=2.8, focal_length=50.0, flash_fired=False)  # fmt: skip
    w = edit_stats(img, meta, (190.0, 0.003))
    assert w[idx["log2_iso"]] == pytest.approx(np.log2(400)) and w[idx["wb_mired"]] == 190.0
    assert w[idx["ev100"]] == pytest.approx(np.log2(2.8**2 * 250) - np.log2(4.0), abs=1e-4)
    # +1 EV (linear gain) moves the exposure statistic by about one stop and the hist upward
    lin = (img / 255.0) ** 2.2
    up = (np.clip(lin * 2.0, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
    u = edit_stats(up)
    assert 0.5 < u[idx["ev"]] - v[idx["ev"]] < 1.3
    small = cv2.resize(img, (256, 170))
    t = time.perf_counter()
    for _ in range(20):
        edit_stats(small)
    assert (time.perf_counter() - t) / 20 < 0.006  # spec: <= 3 ms (generous for CI noise)
    # a colour cast shows up in grey-world ln R/G
    warm = img.astype(np.float32)
    warm[:, :, 0] *= 1.25
    assert edit_stats(np.clip(warm, 0, 255).astype(np.uint8))[idx["gw_rg"]] > v[idx["gw_rg"]] + 0.1


# ---- targets, carry set, format classes -----------------------------------------------------------


def test_targets_raw_wb_is_relative_to_as_shot():
    wb = (200.0, 0.002)
    t = edit_dataset.targets_from_settings(
        {"WhiteBalance": "Custom", "Temperature": 4000.0, "Tint": 10.0, "Exposure2012": 0.5},
        FMT_RAW, wb)  # fmt: skip
    assert t["Exposure2012"] == 0.5 and t["Contrast2012"] == 0.0
    assert t["raw_temp"] == pytest.approx(1e6 / 4000.0 - 200.0)  # 250 - 200 = +50 mired
    assert t["raw_tint"] == pytest.approx(10.0 - edit_dataset.TINT_PER_DUV * 0.002)
    assert np.isnan(t["jpeg_temp"]) and np.isnan(t["jpeg_tint"])
    asshot = edit_dataset.targets_from_settings({"WhiteBalance": "As Shot"}, FMT_DNG, wb)
    assert asshot["raw_temp"] == 0.0 and asshot["raw_tint"] == 0.0  # unchanged WB: target 0
    nowb = edit_dataset.targets_from_settings({"Temperature": 4000.0}, FMT_RAW, None)
    assert np.isnan(nowb["raw_temp"])  # no as-shot metadata: row not used for the WB heads


def test_targets_jpeg_uses_incremental_wb():
    t = edit_dataset.targets_from_settings(
        {"IncrementalTemperature": -12.0, "IncrementalTint": 3.0, "Vibrance": 20.0},
        FMT_OTHER, None)  # fmt: skip
    assert t["jpeg_temp"] == -12.0 and t["jpeg_tint"] == 3.0 and t["Vibrance"] == 20.0
    assert np.isnan(t["raw_temp"])


def test_format_class_and_adobe_dng_preview_is_skipped(gen_dir, tmp_path, monkeypatch):
    assert edit_dataset.format_class("a.CR2") == FMT_RAW
    assert edit_dataset.format_class("a.DNG") == FMT_DNG and edit_dataset.format_class("a.jpg") == 2
    # DNGs whose preview was rendered by Adobe software have the edit baked in: skipped
    d = tmp_path / "shoot"
    d.mkdir()
    for n in ("pentax_kx.dng", "panasonic_lx1.dng"):
        shutil.copy(gen_dir / "raw" / n, d / n)
    dev = {"Exposure2012": 0.5, "ProcessVersion": "11.0"}
    cat = _cat(tmp_path / "c.lrcat", [
        {"root": str(d), "rel": "", "base": "pentax_kx", "ext": "dng", "develop": dev},
        {"root": str(d), "rel": "", "base": "panasonic_lx1", "ext": "dng", "develop": dev},
    ])  # fmt: skip
    seen = {}

    def fake(paths, settings, progress, dng_render=False, store=None):
        seen["paths"] = [Path(p).name for p in paths]
        seen["render"] = dng_render
        return {}

    monkeypatch.setattr(edit_dataset, "featurise", fake)
    S = edit_dataset.build_edit_set(cat, [], None)
    assert seen["paths"] == ["pentax_kx.dng"] and S.notes["adobe_preview_skipped"] == 1
    edit_dataset.build_edit_set(cat, [], None, dng_render=True)  # --dng-render keeps both
    assert sorted(seen["paths"]) == ["panasonic_lx1.dng", "pentax_kx.dng"] and seen["render"]
