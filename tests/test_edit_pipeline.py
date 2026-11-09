# ruff: noqa: E501
"""T15: `edit apply` end to end (JPEG/DNG copies, RAW sidecars), `edits` table, provenance."""

from __future__ import annotations

import hashlib
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from shotsieve.config import Settings
from shotsieve.features.vector import FeatureVector
from shotsieve.io import crs
from shotsieve.personal.edit_dataset import Feat, build_edit_set
from shotsieve.pipeline import edit
from shotsieve.pipeline.cache import Cache


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _row(path, decision, group):
    return dict(path=str(path), decision=decision, reasons=[], score=0.7, prior=0.7,
                group_id=group, burst_id=group, is_best=True, audit=False, audit_prob=None,
                features=FeatureVector(n_faces=1.0).to_dict())  # fmt: skip


@pytest.fixture()
def jpeg_shoot(tmp_path, shoot_dir):
    folder = tmp_path / "shoot"
    folder.mkdir()
    names = ["burst0_0", "landscape", "black", "burst1_3"]
    for n in names:
        shutil.copy(shoot_dir / f"{n}.jpg", folder / f"{n}.jpg")
    rows = [_row(folder / f"{n}.jpg", d, i) for i, (n, d) in
            enumerate(zip(names, ["keep", "keep", "reject", "review"], strict=True))]  # fmt: skip
    with Cache.for_folder(folder) as c:
        c.save_decisions(rows)
    return folder


def test_apply_jpeg_report_only_by_default_with_lightroom_warning(edit_env, jpeg_shoot):
    before = {p.name: sha(p) for p in jpeg_shoot.glob("*.jpg")}
    res = edit.apply_folder(jpeg_shoot, Settings(), scope="keeps")
    assert res["n"] == 2 and res["predicted"] == 2 and res["report_only"] == 2
    assert res["written_sidecar"] == 0 and res["written_copies"] == 0
    assert any("Lightroom" in w and "embed-copies" in w for w in res["warnings"])
    assert not list(jpeg_shoot.glob("*.xmp")) and not (jpeg_shoot / "shotsieve_edits").exists()
    assert {p.name: sha(p) for p in jpeg_shoot.glob("*.jpg")} == before
    with Cache.for_folder(jpeg_shoot) as c:  # predictions are stored (badge / UI) with mode report
        e = c.get_edits()
    assert len(e) == 2 and all(v["written_mode"] == "report" for v in e.values())


def test_apply_jpeg_embed_copies_are_valid_and_originals_untouched(edit_env, jpeg_shoot):
    before = {p.name: sha(p) for p in jpeg_shoot.glob("*.jpg")}
    res = edit.apply_folder(jpeg_shoot, Settings(), scope="all", embed_copies=True)
    out = jpeg_shoot / "shotsieve_edits"
    assert res["written_copies"] >= 3 and not res["skipped"]
    assert {p.name: sha(p) for p in jpeg_shoot.glob("*.jpg")} == before  # originals sha-identical
    copies = sorted(out.glob("*.jpg"))
    assert len(copies) == res["written_copies"]
    prof_hash = res["profile_hash"]
    for cp in copies:
        got = crs.read_develop_xmp(cp)
        assert got.ours and got.profile == prof_hash and got.has_settings
        assert got.settings["ProcessVersion"] == 11.0
        for k, v in got.settings.items():
            if isinstance(v, float) and k not in ("ProcessVersion", "Sharpness"):
                lo, hi = crs.RANGES.get(k, crs.DEFAULT_RANGE)
                assert lo <= v <= hi, (cp.name, k, v)
        assert np.array_equal(np.asarray(Image.open(cp)), np.asarray(Image.open(jpeg_shoot / cp.name)))
        assert "Contrast2012" in got.settings  # constant sliders are always written
    # carry set from the training preset is written verbatim into every copy
    assert "Sharpness" in crs.read_develop_xmp(copies[0]).settings
    # the edits table records what was written
    with Cache.for_folder(jpeg_shoot) as c:
        e = c.get_edits()
    assert {v["written_mode"] for v in e.values()} == {"copy"}
    assert all(v["profile_hash"] == prof_hash for v in e.values())
    # re-running replaces our own copies (provenance), nothing is "skipped"
    res2 = edit.apply_folder(jpeg_shoot, Settings(), scope="all", embed_copies=True)
    assert res2["written_copies"] == res["written_copies"] and not res2["skipped"]


def test_dry_run_writes_nothing_and_reports_a_diff(edit_env, jpeg_shoot):
    res = edit.apply_folder(jpeg_shoot, Settings(), scope="all", embed_copies=True, dry_run=True)
    assert res["dry_run"] and res["written_copies"] >= 3 and res["diff"]
    assert not (jpeg_shoot / "shotsieve_edits").exists()
    with Cache.for_folder(jpeg_shoot) as c:
        assert c.get_edits() == {}
    d0 = res["diff"][0]
    assert {"name", "status", "values", "omitted", "needs_edit"} <= set(d0)


def test_apply_needs_a_profile_and_a_cull(tmp_path, monkeypatch, jpeg_shoot):
    monkeypatch.setenv("SHOTSIEVE_DATA_DIR", str(tmp_path / "empty"))
    with pytest.raises(edit.NoProfileError):
        edit.apply_folder(jpeg_shoot, Settings())


def test_dng_copies_keep_raw_pixels(edit_env, gen_dir, tmp_path):
    import rawpy

    folder = tmp_path / "dngs"
    folder.mkdir()
    src = folder / "pentax_kx.dng"
    shutil.copy(gen_dir / "raw" / "pentax_kx.dng", src)
    with Cache.for_folder(folder) as c:
        c.save_decisions([_row(src, "keep", 0)])
    before = sha(src)
    res = edit.apply_folder(folder, Settings(), scope="keeps", embed_copies=str(tmp_path / "copies"))
    assert res["written_copies"] == 1 and sha(src) == before
    cp = tmp_path / "copies" / "pentax_kx.dng"
    with rawpy.imread(str(src)) as a, rawpy.imread(str(cp)) as b:
        assert np.array_equal(a.raw_image, b.raw_image)
    got = crs.read_develop_xmp(cp)
    assert got.ours and got.settings["ProcessVersion"] == 11.0
    assert "Temperature" not in got.settings  # this profile has no RAW white-balance head


@pytest.fixture()
def raw_shoot(tmp_path, monkeypatch, edit_sets):
    """Four fake RAW files with cull decisions; featurise/as_shot are stubbed (no RAW decoder)."""
    folder = tmp_path / "raws"
    folder.mkdir()
    train, test, _ = edit_sets["A"]
    paths = []
    for i in range(4):
        p = folder / f"IMG_{i:04d}.CR2"
        p.write_bytes(b"raw")
        paths.append(str(p))
    with Cache.for_folder(folder) as c:
        c.save_decisions([_row(p, "keep", i) for i, p in enumerate(paths)])
    feats = {p: Feat(test.stats[i], test.emb[i], 0) for i, p in enumerate(paths)}
    monkeypatch.setattr(edit, "featurise", lambda ps, *a, **k: {p: feats[p] for p in ps})
    monkeypatch.setattr(edit, "as_shot", lambda p: (180.0, 0.004))
    return folder, paths


def test_raw_sidecars_valid_and_never_overwrite(edit_env, raw_shoot):
    folder, paths = raw_shoot
    # profile A has jpeg WB heads only, so RAW frames get tone sliders and no Temperature
    res = edit.apply_folder(folder, Settings(), scope="keeps")
    assert res["written_sidecar"] == 4 and res["report_only"] == 0
    for p in paths:
        sp = Path(p).with_suffix(".xmp")
        root = ET.fromstring(sp.read_text())
        assert root is not None
        got = crs.read_develop_xmp(p)
        assert got.ours and got.settings["ProcessVersion"] == 11.0
        assert "Temperature" not in got.settings and "WhiteBalance" not in got.settings
    # a sidecar with the photographer's own develop settings is never replaced
    sp1 = Path(paths[1]).with_suffix(".xmp")
    sp1.write_text(
        '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        f'<rdf:Description rdf:about="" xmlns:crs="{crs.CRS}" crs:HasSettings="True" crs:Exposure2012="+0.90"/>'
        "</rdf:RDF></x:xmpmeta>"
    )
    own_before = Path(paths[2]).with_suffix(".xmp").read_bytes()
    res2 = edit.apply_folder(folder, Settings(), scope="keeps")
    assert res2["skipped"] == {"skipped_user_settings": 1} and res2["written_sidecar"] == 3
    assert crs.read_develop_xmp(paths[1]).settings["Exposure2012"] == 0.9
    assert own_before  # (the sidecar existed from the first run)
    # our own earlier sidecar (provenance) was replaced, not skipped
    assert crs.read_develop_xmp(paths[2]).ours


def test_catalog_with_develop_settings_blocks_write(edit_env, raw_shoot, tmp_path, edit_styles):
    import importlib.util
    import sys

    folder, paths = raw_shoot
    spec = importlib.util.spec_from_file_location(
        "mes", Path(__file__).resolve().parents[1] / "scripts" / "make_edit_styles.py"
    )
    mes = importlib.util.module_from_spec(spec)
    sys.modules["mes"] = mes
    spec.loader.exec_module(mes)
    cat = tmp_path / "c.lrcat"
    mes.build_catalog(cat, [{"root": str(folder), "rel": "", "base": "IMG_0000", "ext": "CR2",
                             "develop": {"Exposure2012": 0.5, "ProcessVersion": "11.0"}}])  # fmt: skip
    res = edit.apply_folder(folder, Settings(), scope="keeps", catalog=cat)
    assert res["skipped"] == {"skipped_catalog": 1} and res["written_sidecar"] == 3
    assert not Path(paths[0]).with_suffix(".xmp").exists()


def test_raw_white_balance_is_as_shot_relative(edit_env, raw_shoot, edit_sets, tmp_path):
    """With a profile that has RAW WB heads, Temperature = 1e6 / (as-shot mired + delta)."""
    from shotsieve.personal.edit_profile import Prediction

    pred = Prediction(point={"wb_temp": 25.0, "wb_tint": 3.0},
                      written={"wb_temp": True, "wb_tint": True})  # fmt: skip
    vals, _, wb = edit.to_crs(pred, 0, (180.0, 0.004))
    assert wb and vals["Temperature"] == pytest.approx(1e6 / 205.0)
    assert vals["Tint"] == pytest.approx(-3000 * 0.004 + 3.0)
    vals, _, wb = edit.to_crs(pred, 0, None)  # no as-shot metadata: no WB written
    assert not wb and "Temperature" not in vals
    vals, _, wb = edit.to_crs(pred, 2, None)  # JPEG: Incremental*
    assert wb and vals == {"IncrementalTemperature": 25.0, "IncrementalTint": 3.0}


def test_own_writes_are_excluded_from_training(edit_env, tmp_path, edit_styles):
    """A catalog value that equals what ShotSieve wrote (edits table) is not the user's label."""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "mes2", Path(__file__).resolve().parents[1] / "scripts" / "make_edit_styles.py"
    )
    mes = importlib.util.module_from_spec(spec)
    sys.modules["mes2"] = mes
    spec.loader.exec_module(mes)
    folder = tmp_path / "shoot"
    folder.mkdir()
    import cv2

    names = []
    for i in range(3):
        img = np.full((64, 64, 3), 60 + 40 * i, np.uint8)
        cv2.imwrite(str(folder / f"a{i}.jpg"), img)
        names.append(str(folder / f"a{i}.jpg"))
    with Cache.for_folder(folder) as c:
        c.save_decisions([_row(p, "keep", i) for i, p in enumerate(names)])
        c.put_edits([
            {"path": names[0], "values": {"Exposure2012": 0.35, "Contrast2012": 10.0},
             "confidence": {}, "written_mode": "copy", "profile_hash": "h"},
            {"path": names[1], "values": {"Exposure2012": 0.35}, "confidence": {},
             "written_mode": "report", "profile_hash": "h"},  # never written: a real label
        ])  # fmt: skip
    cat = tmp_path / "c.lrcat"
    dev = {"ProcessVersion": "11.0", "Exposure2012": 0.35, "Contrast2012": 10, "Sharpness": 40}
    mes.build_catalog(cat, [
        {"root": str(folder), "rel": "", "base": f"a{i}", "ext": "jpg", "develop": dev}
        for i in range(3)
    ])  # fmt: skip
    S = build_edit_set(cat, [], Settings(), None, store=None)
    got = sorted(Path(p).name for p in S.paths)
    # a0 equals our write -> excluded; a1 was only reported -> kept; a2 never predicted -> kept
    assert got == ["a1.jpg", "a2.jpg"] and S.notes["own_writes_excluded"] == 1


def test_xmp_sidecar_training_source_skips_own_provenance(edit_env, tmp_path):
    import cv2

    folder = tmp_path / "x"
    folder.mkdir()
    for i, own in enumerate([False, True, False]):
        p = folder / f"IMG_{i}.CR2"
        p.write_bytes(b"raw")
        cv2.imwrite(str(folder / f"IMG_{i}.jpg"), np.full((32, 32, 3), 100, np.uint8))
        if own:
            crs.write_develop(p, {"Exposure2012": 0.3}, profile_hash="mine")
        else:
            crs.write_develop(p, {"Exposure2012": 0.3 + i}, profile_hash="tmp")
            sp = p.with_suffix(".xmp")  # a photographer's sidecar has no ShotSieve provenance
            sp.write_text(re.sub(r'\s*shotsieve:\w+="[^"]*"', "", sp.read_text()))
    from shotsieve.personal import edit_dataset

    got = edit_dataset._xmp_entries(folder)
    assert sorted(Path(e.path).name for e in got) == ["IMG_0.CR2", "IMG_2.CR2"]
    assert all(e.fmt == "raw" and e.settings["Exposure2012"] in (0.3, 2.3) for e in got)


def test_predictions_for_live_and_stored(edit_env, jpeg_shoot):
    with Cache.for_folder(jpeg_shoot) as c:
        ids = list(c.ids_for([str(jpeg_shoot / "landscape.jpg")]).values())
    live = edit.predictions_for(jpeg_shoot, ids, Settings())
    assert live[ids[0]]["applied"] is False and live[ids[0]]["values"]
    edit.apply_folder(jpeg_shoot, Settings(), scope="all", embed_copies=True)
    stored = edit.predictions_for(jpeg_shoot, ids, Settings())
    assert stored[ids[0]]["applied"] is True and stored[ids[0]]["written_mode"] == "copy"


def test_raw_white_balance_written_with_a_raw_profile(tmp_path, monkeypatch, edit_sets, raw_shoot):
    """Relabel the style-A rows as RAW: the RAW WB heads learn, and `apply` writes Kelvin/Tint."""
    from shotsieve.personal.edit_profile import EditProfile, EditSet

    train, _, _ = edit_sets["A"]
    y = dict(train.y)
    y["raw_temp"], y["raw_tint"] = y["jpeg_temp"].copy(), y["jpeg_tint"].copy()
    y["jpeg_temp"] = np.full(len(train), np.nan)
    y["jpeg_tint"] = np.full(len(train), np.nan)
    S = EditSet(train.stats, train.emb, np.zeros(len(train), int), y, train.groups, train.carry)
    d = tmp_path / "rawdata"
    d.mkdir()
    prof = EditProfile().fit(S)
    assert prof.sliders["raw_temp"].mode == "learned" and prof.sliders["jpeg_temp"].mode == "off"
    prof.save(d / "edit_profile.joblib")
    monkeypatch.setenv("SHOTSIEVE_DATA_DIR", str(d))
    folder, paths = raw_shoot
    res = edit.apply_folder(folder, Settings(), scope="keeps")
    assert res["written_sidecar"] == 4
    seen = 0
    for p in paths:
        got = crs.read_develop_xmp(p)
        if "Temperature" in got.settings:
            seen += 1
            assert 2000 <= got.settings["Temperature"] <= 50000 and -150 <= got.settings["Tint"] <= 150
            assert got.settings["WhiteBalance"] == "Custom"
            # as-shot mired is 180 (stubbed); the predicted delta is within the training range
            assert abs(1e6 / got.settings["Temperature"] - 180.0) <= 100.0
    assert seen >= 1
