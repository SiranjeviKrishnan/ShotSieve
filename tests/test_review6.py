# ruff: noqa: E501
"""Regression tests for Review #6 (write safety of T14-T17): B1, M1-M6 and the cheap minors."""

from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from shotsieve.config import Settings
from shotsieve.io import crs, xmp
from shotsieve.io.crs import CRS, read_develop_xmp, write_develop
from shotsieve.io.formats import scan_folder
from shotsieve.pipeline import edit, match
from shotsieve.pipeline.cache import Cache
from shotsieve.server.app import create_app


def _jpeg(path: Path, seed: int = 3) -> Path:
    img = (np.random.default_rng(seed).random((64, 96, 3)) * 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return path


def _raw(path: Path) -> Path:
    path.write_bytes(b"raw")
    return path


def _user_edit(sp: Path, old: str, new: str) -> None:
    """What Lightroom/ACR does: change a value but leave foreign (our) namespace attributes."""
    text = sp.read_text()
    assert old in text
    sp.write_text(text.replace(old, new))


# ---- B1: provenance is a digest of what we wrote, not the presence of a tag --------------------


def test_b1_later_user_edit_with_provenance_kept_is_never_overwritten(tmp_path):
    raw = _raw(tmp_path / "a.CR2")
    assert write_develop(raw, {"Exposure2012": -0.4, "Contrast2012": 5.0}, profile_hash="h").status == "written"
    cur = read_develop_xmp(raw)
    assert cur.ours and cur.tagged
    sp = tmp_path / "a.xmp"
    _user_edit(sp, 'crs:Exposure2012="-0.40"', 'crs:Exposure2012="+1.70"')
    assert "shotsieve:EditProfile" in sp.read_text()  # provenance still there ...
    cur = read_develop_xmp(raw)
    assert cur.tagged and not cur.ours  # ... but no longer "ours"
    before = sp.read_bytes()
    res = write_develop(raw, {"Exposure2012": 0.9}, profile_hash="h2")
    assert res.status == "skipped_user_settings" and sp.read_bytes() == before
    assert crs.writable(raw, "sidecar") == (False, "skipped_user_settings")
    # a changed value, a removed value and an added value all count as the user's edit
    for old, new in (('crs:Contrast2012="+5"', ""), ("<rdf:Description", '<rdf:Description crs:Vibrance="+9"')):
        r2 = _raw(tmp_path / "b.CR2")
        write_develop(r2, {"Exposure2012": 0.2, "Contrast2012": 5.0}, profile_hash="h")
        _user_edit(tmp_path / "b.xmp", old, new)
        assert write_develop(r2, {"Exposure2012": 0.3}, profile_hash="h").status == "skipped_user_settings"
        (tmp_path / "b.xmp").unlink()


def test_b1_user_edited_frames_are_training_rows_own_writes_are_not(tmp_path):
    from shotsieve.personal.edit_dataset import _xmp_entries

    mine, edited = _raw(tmp_path / "mine.CR2"), _raw(tmp_path / "edited.CR2")
    for r in (mine, edited):
        write_develop(r, {"Exposure2012": -0.4, "Contrast2012": 12.0}, profile_hash="h")
    _user_edit(tmp_path / "edited.xmp", 'crs:Exposure2012="-0.40"', 'crs:Exposure2012="+1.70"')
    got = {Path(e.path).stem: e.settings for e in _xmp_entries(tmp_path)}
    assert "mine" not in got  # an untouched prediction is not the photographer's label
    assert got["edited"]["Exposure2012"] == pytest.approx(1.7)  # the corrected frame is the best label


def test_b1_legacy_tag_without_digest_is_treated_as_the_users(tmp_path):
    raw = _raw(tmp_path / "a.NEF")
    (tmp_path / "a.xmp").write_text(
        '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        f'<rdf:Description rdf:about="" xmlns:crs="{CRS}" xmlns:shotsieve="http://ns.shotsieve.app/1.0/" '
        'crs:HasSettings="True" crs:Exposure2012="+1.00" shotsieve:EditProfile="old"/></rdf:RDF></x:xmpmeta>'
    )
    assert write_develop(raw, {"Exposure2012": 0.1}, profile_hash="h").status == "skipped_user_settings"


def test_b1_second_apply_over_own_write_still_replaces(tmp_path):
    raw = _raw(tmp_path / "a.CR2")
    write_develop(raw, {"Exposure2012": 0.1, "Contrast2012": 4.0}, profile_hash="p1")
    xmp.write_sidecar(raw, rating=3, decision="keep")  # a non-crs writer must not break the digest
    assert write_develop(raw, {"Exposure2012": 0.2}, profile_hash="p2").status == "replaced_own"
    s = read_develop_xmp(raw).settings
    assert s["Exposure2012"] == 0.2 and "Contrast2012" not in s


# ---- M3: foreign crs values survive -----------------------------------------------------------


def test_m3_foreign_crs_kept_with_has_settings_false_and_on_replace(tmp_path):
    raw = _raw(tmp_path / "a.CR2")
    (tmp_path / "a.xmp").write_text(
        '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        f'<rdf:Description rdf:about="" xmlns:crs="{CRS}" crs:HasSettings="False" crs:RawFileName="a.CR2" '
        'crs:CameraProfile="Camera Standard"/></rdf:RDF></x:xmpmeta>'
    )
    assert write_develop(raw, {"Exposure2012": 0.2, "Contrast2012": 7.0}, profile_hash="p1").status == "written"
    s = read_develop_xmp(raw)
    assert s.has_settings and s.ours
    assert s.settings["CameraProfile"] == "Camera Standard" and s.settings["RawFileName"] == "a.CR2"
    assert len(list(tmp_path.glob("a.xmp.bak-*"))) == 1  # one .bak of the sidecar we did not create (m6)
    assert write_develop(raw, {"Exposure2012": 0.5}, profile_hash="p2").status == "replaced_own"
    s = read_develop_xmp(raw).settings
    assert s["CameraProfile"] == "Camera Standard" and s["Exposure2012"] == 0.5 and "Contrast2012" not in s
    assert len(list(tmp_path.glob("a.xmp.bak-*"))) == 1  # not one per write


# ---- M1: embedded copies never overwrite what is already there -----------------------------------


def test_m1_existing_unrelated_file_is_not_overwritten(tmp_path):
    (tmp_path / "shoot").mkdir()
    (tmp_path / "other").mkdir()
    src = _jpeg(tmp_path / "shoot" / "IMG_1.jpg")
    other = _jpeg(tmp_path / "other" / "IMG_1.jpg", seed=9)
    before = other.read_bytes()
    res = write_develop(src, {"Exposure2012": 0.3}, profile_hash="h", mode="embed", out_dir=tmp_path / "other")
    assert res.status == "skipped_exists" and not res.written
    assert other.read_bytes() == before
    assert crs.writable(src, "embed", dst=other) == (False, "skipped_exists")


def test_m1_own_copy_replaced_but_user_edited_copy_never(tmp_path):
    src = _jpeg(tmp_path / "IMG.jpg")
    out = tmp_path / "out"
    assert write_develop(src, {"Exposure2012": 0.3}, profile_hash="h", mode="embed", out_dir=out).status == "written"
    assert write_develop(src, {"Exposure2012": 0.6}, profile_hash="h", mode="embed", out_dir=out).status == "written"
    dst = out / "IMG.jpg"
    assert read_develop_xmp(dst).settings["Exposure2012"] == 0.6
    dst.write_bytes(dst.read_bytes().replace(b"+0.60", b"+1.70"))  # LR edit of the imported copy
    before = dst.read_bytes()
    res = write_develop(src, {"Exposure2012": 0.1}, profile_hash="h", mode="embed", out_dir=out)
    assert res.status == "skipped_user_settings" and dst.read_bytes() == before


def test_m1_copy_root_must_not_contain_or_equal_a_culled_shoot(tmp_path):
    shoot, other = tmp_path / "a", tmp_path / "b"
    for f in (shoot, other):
        (f / ".shotsieve").mkdir(parents=True)
        (f / ".shotsieve" / "cull.db").write_bytes(b"")
    with pytest.raises(ValueError):
        edit.check_copy_root(other, shoot)  # another culled shoot
    with pytest.raises(ValueError):
        edit.check_copy_root(tmp_path, shoot)  # contains culled folders
    with pytest.raises(ValueError):
        edit.check_copy_root(shoot, shoot)
    edit.check_copy_root(shoot / "shotsieve_edits", shoot)  # the default is fine
    edit.check_copy_root(tmp_path / "elsewhere", shoot)


# ---- M2: the copy folder is not culled again ------------------------------------------------------


def test_m2_scan_skips_shotsieve_edits(tmp_path):
    _jpeg(tmp_path / "IMG_1.jpg")
    src = _jpeg(tmp_path / "IMG_2.jpg")
    write_develop(src, {"Exposure2012": 0.3}, profile_hash="h", mode="embed", out_dir=tmp_path / "shotsieve_edits")
    assert (tmp_path / "shotsieve_edits" / "IMG_2.jpg").exists()
    for recursive in (True, False):
        names = sorted(e.master.name for e in scan_folder(tmp_path, recursive=recursive))
        assert names == ["IMG_1.jpg", "IMG_2.jpg"]
    assert not any("shotsieve_edits" in e.master.parts for e in scan_folder(tmp_path))


# ---- M4: one lock per sidecar ------------------------------------------------------------------


def test_m4_threaded_writers_lose_nothing(tmp_path):
    raw = _raw(tmp_path / "a.CR2")
    n = 100
    errors: list[BaseException] = []

    def keywords():
        try:
            for i in range(n):
                xmp.merge_keywords(raw, [f"kw{i}"])
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def develop():
        try:
            for i in range(n):
                write_develop(raw, {"Exposure2012": (i % 9) / 10}, profile_hash=f"p{i}")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def decisions():
        try:
            for i in range(n):
                xmp.write_sidecar(raw, rating=2, decision="keep", score=i / 100)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    ts = [threading.Thread(target=f) for f in (keywords, develop, decisions)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errors, errors
    kws, _ = xmp.read_keywords(raw)
    assert {f"kw{i}" for i in range(n)} <= set(kws)  # no keyword lost
    cur = read_develop_xmp(raw)
    assert cur.has_settings and cur.ours  # the crs edit is there and its digest still holds
    assert xmp.read_sidecar(raw).decision == "keep"


# ---- M6: white balance only for the same camera body ----------------------------------------------


def _m(body, raw=True, shot=(200.0, 0.0), emb=0):
    return match.Measure("x", np.eye(4, dtype=np.float32)[emb], 0.2, 0.0, 0.0, 0.0, raw, shot, body)


def test_m6_match_writes_wb_only_for_the_same_body():
    ref = match.Reference("r", {"Exposure2012": 0.1, "Temperature": 5000.0, "Tint": 3.0}, _m("canon eos 5d mark iii"), "x")
    same = match.match_one(ref, _m("canon eos 5d mark iii"))
    assert same.wb and "Temperature" in same.values
    other = match.match_one(ref, _m("sony ilce-7m3"))
    assert not other.wb and not {"Temperature", "Tint"} & set(other.values)
    assert "different body: white balance not matched" in other.notes
    assert "Exposure2012" in other.values  # exposure is still matched
    unknown = match.match_one(ref, _m(""))
    assert not unknown.wb and "different body: white balance not matched" in unknown.notes


# ---- m3, m4 (match) ---------------------------------------------------------------------------


def test_m3_clipped_target_carries_the_reference_exposure():
    ref = match.Reference("r", {"Exposure2012": 0.5}, _m("a"), "x")
    t = _m("a", raw=False)
    t.clip_hi = 0.5
    r = match.match_one(ref, t)
    assert r.values["Exposure2012"] == 0.5 and any("clipped" in n for n in r.notes)


def test_m4_reference_written_by_shotsieve_is_refused(tmp_path):
    raw = _raw(tmp_path / "a.CR2")
    write_develop(raw, {"Exposure2012": 0.3}, profile_hash="h")
    with pytest.raises(ValueError, match="written by ShotSieve"):
        match.reference_settings(raw)
    _user_edit(tmp_path / "a.xmp", 'crs:Exposure2012="+0.30"', 'crs:Exposure2012="+0.80"')
    settings, _ = match.reference_settings(raw)  # after the photographer's edit it is a reference
    assert settings["Exposure2012"] == 0.8


# ---- API: cross-body copy roots, a folder being re-decided ---------------------------------------


def test_api_embed_copies_into_a_culled_shoot_is_rejected_and_busy_folder_is_409(tmp_path, monkeypatch):
    shoot, other = tmp_path / "shoot", tmp_path / "other"
    for f in (shoot, other):
        f.mkdir()
        _jpeg(f / "a.jpg")
    (other / ".shotsieve").mkdir()
    (other / ".shotsieve" / "cull.db").write_bytes(b"")
    with Cache.for_folder(shoot) as c:
        c.save_decisions([dict(path=str(shoot / "a.jpg"), decision="keep", reasons=[], score=0.5, prior=0.5,
                               group_id=0, burst_id=0, is_best=True, audit=False, audit_prob=None, features={})])
    monkeypatch.setattr(edit, "profile_status", lambda s=None: {"trained": True})
    with TestClient(create_app(Settings(), default_folder=shoot)) as c:
        r = c.post("/api/edit/apply", json={"folder": str(shoot), "embed_copies": str(other)})
        assert r.status_code == 400 and "culled shoot" in r.json()["detail"]
        r = c.post("/api/edit/apply", json={"folder": str(shoot), "embed_copies": str(tmp_path)})
        assert r.status_code == 400
        c.app.state.shotsieve.calibrating.add(shoot)
        r = c.post("/api/edit/apply", json={"folder": str(shoot)})
        assert r.status_code == 409


def test_m2_process_version_never_lower_than_11_but_follows_a_newer_reference(tmp_path):
    raw = _raw(tmp_path / "a.CR2")
    write_develop(raw, {"Exposure2012": 0.1}, profile_hash="p", process_version=6.7)
    assert read_develop_xmp(raw).settings["ProcessVersion"] == 11.0
    raw2 = _raw(tmp_path / "b.CR2")
    write_develop(raw2, {"Exposure2012": 0.1}, profile_hash="p", process_version="15.4")
    assert read_develop_xmp(raw2).settings["ProcessVersion"] == 15.4
