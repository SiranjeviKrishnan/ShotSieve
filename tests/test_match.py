# ruff: noqa: E501
"""T16: reference-frame colour match (pipeline/match.py, `shotsieve match`, /api/edit/match)."""

from __future__ import annotations

import hashlib
import itertools
import json
import shutil
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from shotsieve.cli import main
from shotsieve.config import Settings
from shotsieve.features.vector import FeatureVector
from shotsieve.io import crs
from shotsieve.personal.edit_dataset import TINT_PER_DUV
from shotsieve.pipeline import match
from shotsieve.pipeline.cache import Cache
from shotsieve.server.app import create_app

ROOT = Path(__file__).resolve().parents[1]
UNIT = np.eye(4, dtype=np.float32)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _row(path, decision, group):
    return dict(path=str(path), decision=decision, reasons=[], score=0.7, prior=0.7,
                group_id=group, burst_id=group, is_best=True, audit=False, audit_prob=None,
                features=FeatureVector(n_faces=1.0).to_dict())  # fmt: skip


def meas(path="x.jpg", emb=0, lum=0.2, clip=0.0, cast=0.0, duv=0.0, raw=False, shot=None, body="body-a"):
    return match.Measure(path, UNIT[emb], lum, clip, cast, duv, raw, shot, body)


def ref_of(settings=None, **kw):
    return match.Reference("ref.jpg", settings or {"Exposure2012": 0.5}, meas("ref.jpg", **kw), "x")


# ---- the arithmetic (no models) ----------------------------------------------------------------


def test_gate_exposure_and_range():
    ref = ref_of({"Exposure2012": 0.5, "Contrast2012": 12.0, "Sharpness": 40})
    r = match.match_one(ref, meas(emb=1))
    assert r.skipped_reason == "different scene" and not r.values  # cos 0
    r = match.match_one(ref, meas(lum=0.1))  # half the light: +1 EV on top of the reference's +0.5
    assert r.skipped_reason is None and r.values["Exposure2012"] == pytest.approx(1.5, abs=1e-6)
    assert r.delta_ev == pytest.approx(1.0) and r.values["Contrast2012"] == 12.0
    assert r.carried == {"Sharpness": 40}  # every other key is synced verbatim
    r = match.match_one(ref, meas(lum=0.2 / 8))  # 3 EV off
    assert r.skipped_reason and "2 EV" in r.skipped_reason
    r = match.match_one(ref, meas(clip=0.3))  # clipped highlights: exposure is not re-derived (m3)
    assert r.skipped_reason is None and r.values["Exposure2012"] == 0.5 and r.values["Contrast2012"] == 12.0  # the reference's value
    assert any("clipped" in n for n in r.notes)


def test_white_balance_transfer_raw_and_jpeg_off():
    # reference: as-shot 180 mired + cast 10 => scene 190; the photographer chose 5000 K = 200 mired
    ref = ref_of({"Exposure2012": 0.0, "Temperature": 5000.0, "Tint": 7.0}, cast=10.0, duv=0.001, raw=True, shot=(180.0, 0.004))
    # a Sony-like body: as-shot 240, preview cast -20 (scene 220) => +30 mired vs the reference scene
    r = match.match_one(ref, meas(raw=True, shot=(240.0, 0.002), cast=-20.0, duv=0.003))
    assert r.wb and 1e6 / r.values["Temperature"] == pytest.approx(200.0 + 30.0, abs=0.5)
    assert r.values["Tint"] == pytest.approx(7.0 + TINT_PER_DUV * ((0.002 + 0.003) - (0.004 + 0.001)))
    r = match.match_one(ref, meas())  # JPEG target: no WB keys at all
    assert not r.wb and not {"Temperature", "Tint"} & set(r.values) and any("JPEG" in n for n in r.notes)
    # a JPEG reference (incremental units) cannot drive a RAW target either
    ref_j = ref_of({"Exposure2012": 0.0, "IncrementalTemperature": 12.0})
    r = match.match_one(ref_j, meas(raw=True, shot=(200.0, 0.0)))
    assert not r.wb and "IncrementalTemperature" not in r.values and "IncrementalTemperature" not in r.carried


def test_preview_measure_recovers_gain_and_cast(rgb):
    img = rgb("lena.jpg")
    lin = ((np.arange(256) / 255.0) ** 2.2)[img]

    def render(ev, cast):
        g = np.exp(np.array([0.0025 * cast, 0.0, -0.0025 * cast])) * 2.0**ev
        return (np.clip((lin * g) ** (1 / 2.2), 0, 1) * 255 + 0.5).astype(np.uint8)

    a, b = render(0.0, 0.0), render(-0.7, 25.0)
    la, ca, ma = match.preview_measure(a)[:3]
    lb, cb, mb = match.preview_measure(b)[:3]
    assert np.log2(la / lb) == pytest.approx(0.7, abs=0.1)
    assert mb - ma == pytest.approx(25.0, abs=6.0) and ca == 0.0 and cb == 0.0


def test_reference_without_develop_settings_is_refused(tmp_path, shoot_dir):
    p = tmp_path / "plain.jpg"
    shutil.copy(shoot_dir / "burst0_0.jpg", p)
    with pytest.raises(ValueError, match="edit the reference in Lightroom first"):
        match.reference_settings(p)


def test_reference_from_catalog(tmp_path, shoot_dir, edit_styles):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("mes_m", ROOT / "scripts" / "make_edit_styles.py")
    mes = importlib.util.module_from_spec(spec)
    sys.modules["mes_m"] = mes
    spec.loader.exec_module(mes)
    p = tmp_path / "ref.jpg"
    shutil.copy(shoot_dir / "burst0_0.jpg", p)
    cat = tmp_path / "c.lrcat"
    mes.build_catalog(cat, [{"root": str(tmp_path), "rel": "", "base": "ref", "ext": "jpg",
                             "develop": {"Exposure2012": 0.4, "Contrast2012": 9, "ProcessVersion": "11.0"}}])  # fmt: skip
    st, src = match.reference_settings(p, cat)
    assert src == "catalog" and st["Exposure2012"] == 0.4 and st["Contrast2012"] == 9


# ---- synthetic accuracy (edit_styles corpus: known gain / cast between frames) -----------------


@pytest.fixture(scope="module")
def corpus(edit_styles, tmp_path_factory):
    """Measures of 90 perturbed bases (5 each) with cached embeddings, plus the truth records."""
    import os

    d = tmp_path_factory.mktemp("matchdata")
    shutil.copy(edit_styles / "features.db", d / "edit_features.db")
    old = os.environ.get("SHOTSIEVE_DATA_DIR")
    os.environ["SHOTSIEVE_DATA_DIR"] = str(d)
    try:
        truth = json.loads((edit_styles / "truth.json").read_text())
        by: dict[int, list[dict]] = {}
        for t in truth:
            by.setdefault(t["base"], []).append(t)
        bases = sorted(by)[::3][:90]
        real = [b for b in sorted(by) if not by[b][0]["name"].startswith("IMG_")]
        paths = [str(edit_styles / "images" / t["file"]) for b in [*bases, *real] for t in by[b]]
        t0 = time.time()
        ms = match.measure_paths(paths)
        speed = len(paths) / (time.time() - t0)
    finally:
        if old is None:
            os.environ.pop("SHOTSIEVE_DATA_DIR", None)
        else:
            os.environ["SHOTSIEVE_DATA_DIR"] = old
    return {"root": edit_styles / "images", "ms": ms, "by": by, "bases": bases, "real": real, "speed": speed}


def test_synthetic_exposure_and_wb_accuracy(corpus):
    ms, by, root = corpus["ms"], corpus["by"], corpus["root"]
    exp_err, wb_err, n, skipped = [], [], 0, 0
    for b in corpus["bases"]:
        for a, c in itertools.permutations(by[b], 2):
            ma, mc = ms[str(root / a["file"])], ms[str(root / c["file"])]
            ma.is_raw = mc.is_raw = True  # pretend RAW: the body's as-shot WB is constant
            ma.body = mc.body = "synthetic body"  # one body (WB is gated on it, Review #6 M6)
            ma.as_shot = mc.as_shot = (200.0, 0.0)
            ref = match.Reference(ma.path, {"Exposure2012": 0.0, "Temperature": 1e6 / 200.0, "Tint": 5.0}, ma, "x")
            r = match.match_one(ref, mc)
            if r.skipped_reason:
                skipped += 1
                continue
            n += 1
            if "Exposure2012" in r.values:
                exp_err.append(abs(r.values["Exposure2012"] - (a["dev"] - c["dev"])))
            if r.wb:
                wb_err.append(abs(1e6 / r.values["Temperature"] - (200.0 + c["cast"] - a["cast"])))
    assert n > 900 and skipped < 0.2 * (n + skipped)  # same-scene pairs mostly pass the gate
    assert np.mean(exp_err) <= 0.10, np.mean(exp_err)
    assert np.mean(wb_err) <= 8.0, np.mean(wb_err)
    print(f"T16 synthetic: {n} pairs, Exposure MAE {np.mean(exp_err):.3f} EV, WB MAE {np.mean(wb_err):.2f} mired")


def test_different_scene_pairs_are_all_skipped(corpus):
    ms, by, root = corpus["ms"], corpus["by"], corpus["root"]
    real = [b for b in corpus["real"] if not by[b][0]["name"].startswith("cc0_football")]
    football = [b for b in corpus["real"] if by[b][0]["name"].startswith("cc0_football")]
    n = skipped = 0
    for b1, b2 in [*itertools.combinations(real, 2), *itertools.product(football, real)]:
        for a, c in itertools.product(by[b1], by[b2]):
            ref = match.Reference("r", {"Exposure2012": 0.0}, ms[str(root / a["file"])], "x")
            n += 1
            skipped += match.match_one(ref, ms[str(root / c["file"])]).skipped_reason == "different scene"
    assert n >= 500 and skipped == n


def test_match_speed_on_cached_embeddings(corpus):
    assert corpus["speed"] >= 30, corpus["speed"]  # img/s (previews decoded at 256 px, embeddings cached)


# ---- end to end: JPEG copies, RAW sidecars, never-overwrite ------------------------------------


@pytest.fixture()
def shoot(tmp_path, shoot_dir, edit_env):
    """A culled folder; burst0_0.jpg is the reference and already carries the photographer's crs."""
    folder = tmp_path / "shoot"
    folder.mkdir()
    names = ["burst0_0", "burst0_1", "burst0_2", "burst1_0", "black"]
    stage = tmp_path / "stage"
    stage.mkdir()
    for n in names[1:]:  # burst0_0 is written below, never over an existing file (Review #6 M1)
        shutil.copy(shoot_dir / f"{n}.jpg", folder / f"{n}.jpg")
    shutil.copy(shoot_dir / "burst0_0.jpg", stage / "burst0_0.jpg")
    res = crs.write_develop(stage / "burst0_0.jpg", {"Exposure2012": 0.35, "Contrast2012": 18.0, "Saturation": -6.0},
                            {"Sharpness": 40, "LensProfileEnable": 1}, profile_hash="photographer", mode="embed", out_dir=folder)
    assert res.written
    # the photographer saved this copy in Lightroom: develop settings, but not ShotSieve's tag
    ref_copy = folder / "burst0_0.jpg"
    ref_copy.write_bytes(ref_copy.read_bytes().replace(b"shotsieve:EditProfile", b"shotsieve:EditProfilX"))
    assert not crs.read_develop_xmp(ref_copy).ours
    decisions = ["keep", "keep", "keep", "keep", "reject"]
    with Cache.for_folder(folder) as c:
        c.save_decisions([_row(folder / f"{n}.jpg", d, 0 if n.startswith("burst0") else 1 + i)
                          for i, (n, d) in enumerate(zip(names, decisions, strict=True))])
    return folder


def test_match_jpeg_embedded_copies(shoot, edit_env):
    ref = shoot / "burst0_0.jpg"
    before = {p.name: sha(p) for p in shoot.glob("*.jpg")}
    results = match.match(shoot, ref, "keeps", Settings(), embed_copies=True)
    names = {Path(r.path).name: r for r in results}
    assert set(names) == {"burst0_1.jpg", "burst0_2.jpg", "burst1_0.jpg"}  # keeps, reference excluded
    assert {p.name: sha(p) for p in shoot.glob("*.jpg")} == before  # originals untouched
    done = [r for r in results if r.written_mode == "copy"]
    assert len(done) >= 2
    for r in done:
        got = crs.read_develop_xmp(shoot / "shotsieve_edits" / Path(r.path).name)
        assert got.ours and got.profile.startswith("match:") and got.has_settings
        assert got.settings["Contrast2012"] == 18.0 and got.settings["Saturation"] == -6.0
        assert got.settings["Sharpness"] == 40 and got.settings["LensProfileEnable"] == 1  # synced
        assert "Exposure2012" in got.settings and "Temperature" not in got.settings and "WhiteBalance" not in got.settings
        assert abs(got.settings["Exposure2012"] - (0.35 + r.delta_ev)) < 0.006
    assert any(r.skipped_reason == "different scene" for r in results) or len(done) == 3
    with Cache.for_folder(shoot) as c:  # stored for the badge / UI
        e = c.get_edits()
    assert len(e) == len(done) and all(v["profile_hash"].startswith("match:") for v in e.values())
    # re-running replaces our own copies
    again = match.match(shoot, ref, "keeps", Settings(), embed_copies=True)
    assert sum(r.written_mode == "copy" for r in again) == len(done)


def test_match_dry_run_report_only_and_targets(shoot, edit_env):
    ref = shoot / "burst0_0.jpg"
    res = match.match(shoot, ref, "group", Settings(), dry_run=True, embed_copies=True)
    assert {Path(r.path).name for r in res} == {"burst0_1.jpg", "burst0_2.jpg"}
    assert all(r.written_mode == "dry_run" for r in res if not r.skipped_reason)
    assert not (shoot / "shotsieve_edits").exists()
    with Cache.for_folder(shoot) as c:
        assert c.get_edits() == {}
    rep = match.match(shoot, ref, [str(shoot / "burst0_1.jpg")], Settings())
    assert [r.status for r in rep] == ["report_only"] and not list(shoot.glob("*.xmp"))
    with Cache.for_folder(shoot) as c:
        ids = c.ids_for([str(shoot / "burst0_2.jpg")])
    one = match.match(shoot, ref, None, Settings(), ids=list(ids.values()), dry_run=True)
    assert [Path(r.path).name for r in one] == ["burst0_2.jpg"]
    s = match.summarize(rep)
    assert s["report_only"] == 1 and any("Lightroom" in w for w in s["warnings"])
    with pytest.raises(ValueError, match="targets must be"):
        match.match(shoot, ref, "bogus", Settings())


def test_match_refuses_an_unedited_reference(shoot, edit_env):
    with pytest.raises(ValueError, match="edit the reference in Lightroom first"):
        match.match(shoot, shoot / "burst0_1.jpg", "keeps", Settings())


@pytest.fixture()
def raw_shoot(tmp_path, monkeypatch):
    """Four fake RAWs with cull decisions and stubbed measurements (no RAW decoder needed)."""
    folder = tmp_path / "raws"
    folder.mkdir()
    paths = []
    for i, ext in enumerate(["CR2", "CR2", "ARW", "NEF"]):
        p = folder / f"IMG_{i:04d}.{ext}"
        p.write_bytes(b"raw")
        paths.append(str(p))
    with Cache.for_folder(folder) as c:
        c.save_decisions([_row(p, "keep", 0) for p in paths])
    ms = {
        paths[0]: meas(paths[0], lum=0.2, cast=10.0, raw=True, shot=(180.0, 0.004)),
        paths[1]: meas(paths[1], lum=0.1, cast=-30.0, raw=True, shot=(180.0, 0.004)),
        paths[2]: meas(paths[2], lum=0.4, cast=0.0, raw=True, shot=(240.0, 0.002)),
        paths[3]: meas(paths[3], emb=1, raw=True, shot=(200.0, 0.0)),  # another scene
    }
    monkeypatch.setattr(match, "measure_paths", lambda ps, *a, **k: {p: ms[p] for p in ps if p in ms})
    ref = Path(paths[0]).with_suffix(".xmp")
    ref.write_text(
        '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        f'<rdf:Description rdf:about="" xmlns:crs="{crs.CRS}" crs:HasSettings="True" crs:Exposure2012="+0.50" '
        'crs:Temperature="5000" crs:Tint="+7" crs:Contrast2012="+10" crs:Sharpness="40" crs:CropTop="0.1" '
        'crs:WhiteBalance="Custom"/></rdf:RDF></x:xmpmeta>'
    )
    return folder, paths


def test_raw_sidecars_wb_exposure_and_never_overwrite(raw_shoot, edit_env):
    folder, paths = raw_shoot
    res = {Path(r.path).name: r for r in match.match(folder, paths[0], "keeps", Settings())}
    assert set(res) == {"IMG_0001.CR2", "IMG_0002.ARW", "IMG_0003.NEF"}
    assert res["IMG_0003.NEF"].skipped_reason == "different scene"
    assert not Path(paths[3]).with_suffix(".xmp").exists()
    got = crs.read_develop_xmp(paths[1])
    assert got.ours and got.settings["WhiteBalance"] == "Custom"
    # IMG_0001: half the light (+1 EV), scene 150 vs reference scene 190 => 160 mired
    assert got.settings["Exposure2012"] == pytest.approx(1.5, abs=0.006)
    assert 1e6 / got.settings["Temperature"] == pytest.approx(200.0 - 40.0, abs=1.5)
    assert got.settings["Contrast2012"] == 10 and got.settings["Sharpness"] == 40
    assert "CropTop" not in got.settings  # per-image geometry is never synced
    # a second body: as-shot 240 => scene 240, 50 mired above the reference's scene
    got2 = crs.read_develop_xmp(paths[2])
    assert 1e6 / got2.settings["Temperature"] == pytest.approx(250.0, abs=1.5)
    assert got2.settings["Exposure2012"] == pytest.approx(0.5 + np.log2(0.2 / 0.4), abs=0.006)
    ET.fromstring(Path(paths[1]).with_suffix(".xmp").read_text())
    # the reference sidecar is untouched; a target the photographer already edited is skipped
    ref_before = Path(paths[0]).with_suffix(".xmp").read_bytes()
    Path(paths[2]).with_suffix(".xmp").write_text(
        '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        f'<rdf:Description rdf:about="" xmlns:crs="{crs.CRS}" crs:HasSettings="True" crs:Exposure2012="-1.00"/></rdf:RDF></x:xmpmeta>')
    again = {Path(r.path).name: r for r in match.match(folder, paths[0], "keeps", Settings())}
    assert again["IMG_0002.ARW"].status == "skipped_user_settings"
    assert crs.read_develop_xmp(paths[2]).settings["Exposure2012"] == -1.0
    assert Path(paths[0]).with_suffix(".xmp").read_bytes() == ref_before
    assert again["IMG_0001.CR2"].status == "written"  # our own earlier sidecar is replaced


def test_catalog_blocks_target_write(raw_shoot, edit_env, tmp_path, edit_styles):
    import importlib.util
    import sys

    folder, paths = raw_shoot
    spec = importlib.util.spec_from_file_location("mes_c", ROOT / "scripts" / "make_edit_styles.py")
    mes = importlib.util.module_from_spec(spec)
    sys.modules["mes_c"] = mes
    spec.loader.exec_module(mes)
    cat = tmp_path / "c.lrcat"
    mes.build_catalog(cat, [{"root": str(folder), "rel": "", "base": "IMG_0001", "ext": "CR2",
                             "develop": {"Exposure2012": 0.5, "ProcessVersion": "11.0"}}])  # fmt: skip
    res = {Path(r.path).name: r for r in match.match(folder, paths[0], "keeps", Settings(), catalog=cat)}
    assert res["IMG_0001.CR2"].status == "skipped_catalog" and not Path(paths[1]).with_suffix(".xmp").exists()
    assert res["IMG_0002.ARW"].status == "written"


# ---- CLI + API ---------------------------------------------------------------------------------


def wait(client, job_id, timeout=120):
    t = time.time()
    while time.time() - t < timeout:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["state"] in ("done", "error"):
            return j
        time.sleep(0.1)
    raise AssertionError("job timed out")


def test_cli_match(shoot, edit_env, capsys, tmp_path):
    before = {p.name: sha(p) for p in shoot.glob("*.jpg")}
    assert main(["match", str(shoot), "--ref", "burst0_0.jpg", "--dry-run", "--embed-copies", "--json"]) == 0
    res = json.loads(capsys.readouterr().out)
    assert res["dry_run"] and res["n"] == 3 and res["would_write"] >= 2 and not (shoot / "shotsieve_edits").exists()
    out = tmp_path / "copies"
    assert main(["match", str(shoot), "--ref", str(shoot / "burst0_0.jpg"), "--targets", "group", "--embed-copies", str(out)]) == 0
    assert "copy(ies)" in capsys.readouterr().out and list(out.glob("*.jpg"))
    assert {p.name: sha(p) for p in shoot.glob("*.jpg")} == before
    assert main(["match", str(shoot), "--ref", "burst0_1.jpg"]) == 3
    assert "edit the reference in Lightroom first" in capsys.readouterr().err
    assert main(["match", str(shoot), "--ref", "burst0_0.jpg"]) == 0
    cap = capsys.readouterr()
    assert "reported only" in cap.out and "Lightroom" in cap.err


@pytest.fixture()
def client(shoot, edit_env):
    with TestClient(create_app(Settings(), default_folder=shoot)) as c:
        yield c


def _id(client, shoot, name):
    imgs = client.get("/api/shoot", params={"folder": str(shoot)}).json()["images"]
    return next(i["id"] for i in imgs if i["name"] == name)


def test_api_reference_and_match_job(client, shoot):
    rid = _id(client, shoot, "burst0_0.jpg")
    ok = client.get("/api/edit/reference", params={"id": rid, "folder": str(shoot)}).json()
    assert ok["ok"] and ok["source"] == "embedded" and ok["n_settings"] >= 3
    bad = client.get("/api/edit/reference", params={"id": _id(client, shoot, "burst0_1.jpg"), "folder": str(shoot)}).json()
    assert not bad["ok"] and "edit the reference in Lightroom first" in bad["message"]
    r = client.post("/api/edit/match", json={"folder": str(shoot), "ref": rid, "dry_run": True, "embed_copies": True})
    j = wait(client, r.json()["job_id"])
    assert j["state"] == "done" and j["kind"] == "edit_match" and j["result"]["dry_run"] and j["result"]["n"] == 3
    assert not (shoot / "shotsieve_edits").exists()
    tid = _id(client, shoot, "burst0_1.jpg")
    j = wait(client, client.post("/api/edit/match", json={"folder": str(shoot), "ref": rid, "ids": [tid], "embed_copies": True}).json()["job_id"])
    assert j["state"] == "done" and j["result"]["n"] == 1
    assert [p.name for p in (shoot / "shotsieve_edits").glob("*.jpg")] in (["burst0_1.jpg"], [])
    flags = {i["name"]: i["edit"] for i in client.get("/api/shoot", params={"folder": str(shoot)}).json()["images"]}
    assert (flags["burst0_1.jpg"] is not None) == bool(list((shoot / "shotsieve_edits").glob("*.jpg")))


def test_api_match_validation(client, shoot):
    rid = _id(client, shoot, "burst0_0.jpg")
    unedited = _id(client, shoot, "burst0_1.jpg")
    r = client.post("/api/edit/match", json={"folder": str(shoot), "ref": unedited})
    assert r.status_code == 400 and "edit the reference in Lightroom first" in r.json()["detail"]
    assert client.post("/api/edit/match", json={"folder": str(shoot), "ref": rid, "targets": "x"}).status_code == 400
    assert client.post("/api/edit/match", json={"folder": str(shoot), "ref": 99999}).status_code == 404
    assert client.post("/api/edit/match", json={"folder": str(shoot), "ref": rid, "embed_copies": "/a/../b"}).status_code == 400
    assert client.post("/api/edit/match", json={"folder": str(shoot), "ref": rid, "catalog": "relative.lrcat"}).status_code == 400
    assert client.post("/api/edit/match", json={"ref": rid, "folder": "relative"}).status_code == 400


# ---- network: two real bodies (CC0, raw.pixls.us) ------------------------------------------------


@pytest.mark.network
def test_two_real_bodies_end_to_end(tmp_path, edit_env):
    import urllib.request

    px = "https://raw.pixls.us/data"
    files = {"IMG_A.CR2": f"{px}/Canon/EOS%205D%20Mark%20III/5G4A9394.CR2", "IMG_B.ARW": f"{px}/Sony/ILCE-7M3/_DSC0009.ARW"}
    folder = tmp_path / "bodies"
    folder.mkdir()
    for name, url in files.items():
        try:
            urllib.request.urlretrieve(url, folder / name)
        except OSError as exc:  # pragma: no cover
            pytest.skip(f"cannot download {url}: {exc}")
    paths = [str(folder / n) for n in files]
    ms = match.measure_paths(paths)
    assert all(m.is_raw and m.as_shot for m in ms.values())
    # as-shot WB of two different bodies is on one mired scale: tungsten-ish Canon, cooler Sony
    assert 150 < ms[paths[0]].as_shot[0] < 450 and 100 < ms[paths[1]].as_shot[0] < 450
    ref = match.Reference(paths[0], {"Exposure2012": 0.2, "Temperature": 3800.0, "Tint": 5.0}, ms[paths[0]], "x")
    forced = match.Measure(paths[1], ms[paths[0]].emb, ms[paths[1]].lum, ms[paths[1]].clip_hi, ms[paths[1]].cast_mired, ms[paths[1]].cast_duv, True, ms[paths[1]].as_shot, ms[paths[0]].body)
    r = match.match_one(ref, forced)  # force the "same scene" gate: different content, real WB path
    assert r.wb and 2000 <= r.values["Temperature"] <= 50000 and -5 <= r.values["Exposure2012"] <= 5
