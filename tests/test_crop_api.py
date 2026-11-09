# ruff: noqa: E501
"""T22: /api/crop/* and `shotsieve crop ...` (needs the core models; YOLOX when present)."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from shotsieve.cli import main
from shotsieve.config import Settings
from shotsieve.features.vector import FeatureVector
from shotsieve.io import crs
from shotsieve.pipeline.cache import Cache
from shotsieve.server.app import create_app

pytestmark = pytest.mark.usefixtures("models_ready")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _row(path, decision, group, burst):
    return dict(path=str(path), decision=decision, reasons=[], score=0.7, prior=0.7,
                group_id=group, burst_id=burst, is_best=True, audit=False, audit_prob=None,
                features=FeatureVector(n_faces=1.0).to_dict())  # fmt: skip


def two_face_image(gen_dir: Path, shoot_dir: Path, dest: Path) -> Path:
    a = cv2.imread(str(gen_dir / "src" / "lena.jpg"))
    b = cv2.imread(str(shoot_dir / "burst0_0.jpg"))
    a = cv2.resize(a, (640, 640))
    cv2.imwrite(str(dest), np.hstack([a, cv2.resize(b, (640, 640))]), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return dest


@pytest.fixture()
def shoot(tmp_path, gen_dir, shoot_dir):
    folder = tmp_path / "shoot"
    folder.mkdir()
    two_face_image(gen_dir, shoot_dir, folder / "two.jpg")
    for i in range(3):
        shutil.copy(shoot_dir / f"burst0_{i}.jpg", folder / f"b{i}.jpg")
    shutil.copy(shoot_dir / "landscape.jpg", folder / "landscape.jpg")
    shutil.copy(gen_dir / "raw" / "pentax_kx.dng", folder / "frame.pef")  # a decodable RAW master
    names = ["two.jpg", "b0.jpg", "b1.jpg", "b2.jpg", "landscape.jpg", "frame.pef"]
    bursts = [10, 1, 1, 1, 11, 12]
    (folder / "clip.mp4").write_bytes(b"not a video")
    with Cache.for_folder(folder) as c:
        rows = [_row(folder / n, "keep", i, b) for i, (n, b) in enumerate(zip(names, bursts, strict=True))]
        rows.append(_row(str(folder / "clip.mp4") + "#t=1.000", "keep", 99, 99))
        c.save_decisions(rows)
        c.ids_for([r["path"] for r in rows])
    return folder


@pytest.fixture()
def ids(shoot):
    with Cache.for_folder(shoot) as c:
        return {Path(p).name: i for p, i in c.ids_lookup(list(c.load_decisions())).items()}


@pytest.fixture()
def client(shoot, tmp_path, monkeypatch):
    monkeypatch.setenv("SHOTSIEVE_DATA_DIR", str(tmp_path / "data"))
    with TestClient(create_app(Settings(), default_folder=shoot)) as c:
        yield c


def wait(client, job_id, timeout=120):
    t = time.time()
    while time.time() - t < timeout:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["state"] in ("done", "error"):
            return j
        time.sleep(0.1)
    raise AssertionError("job timed out")


def tree(folder: Path):
    return {str(p.relative_to(folder)): sha(p) for p in sorted(folder.rglob("*"))
            if p.is_file() and ".shotsieve" not in p.parts}  # fmt: skip


def test_gets_write_nothing_user_visible_and_subjects_find_two_faces(client, shoot, ids):
    before = tree(shoot)
    d = client.get(f"/api/crop/{ids['two.jpg']}/subjects").json()
    faces = [s for s in d["subjects"] if s["kind"] == "face"]
    assert len(faces) == 2 and d["saliency_ready"] and d["full"] == {"w": 1280, "h": 640}
    assert all(0 <= v <= 1 for s in faces for v in s["box"])
    st = client.get(f"/api/crop/{ids['two.jpg']}").json()
    assert st["current"] is None and st["history"] == [] and st["full"] == {"w": 1280, "h": 640}
    assert st["orientation"] == 1 and "4:5" in st["ratios"] and st["xmp"]["has_crop"] is False
    assert tree(shoot) == before  # a GET never changes a user-visible file (the cache fill is internal)


def test_detector_is_optional(client, ids, monkeypatch):
    from shotsieve.models import object_detect

    monkeypatch.setattr(object_detect, "available", lambda: False)
    d = client.get(f"/api/crop/{ids['b0.jpg']}/subjects").json()
    assert d["detector"] is None and d["saliency_ready"]
    assert [s["kind"] for s in d["subjects"]] == ["face"]
    r = client.post(f"/api/crop/{ids['b0.jpg']}/suggest", json={"ratio": "4:5"}).json()
    assert r["candidates"] and r["auto"]


def test_cache_invalidates_when_the_detector_appears(client, shoot, ids, monkeypatch):
    from shotsieve.models import object_detect

    if not object_detect.available():
        pytest.skip("yolox not downloaded")
    monkeypatch.setattr(object_detect, "available", lambda: False)
    assert client.get(f"/api/crop/{ids['two.jpg']}/subjects").json()["detector"] is None
    monkeypatch.undo()
    monkeypatch.setenv("SHOTSIEVE_EXTRA_HOSTS", "testserver")
    assert client.get(f"/api/crop/{ids['two.jpg']}/subjects").json()["detector"] == "yolox_s"


def test_suggest_writes_nothing_and_is_exact_in_pixels(client, shoot, ids):
    before = tree(shoot)
    i = ids["two.jpg"]
    sub = client.get(f"/api/crop/{i}/subjects").json()
    f1 = sorted((s for s in sub["subjects"] if s["kind"] == "face"), key=lambda s: s["box"][0])[0]
    r = client.post(f"/api/crop/{i}/suggest", json={"ratio": "4:5", "subjects": [f1["sid"]], "n": 3})
    assert r.status_code == 200
    d = r.json()
    assert 1 <= len(d["candidates"]) <= 3 and not d["infeasible"]
    assert d["ratio"] == "5:4"  # a landscape image takes a portrait preset in landscape
    for c in d["candidates"]:
        x0, y0, x1, y1 = c["rect_px"]
        W, H = x1 - x0, y1 - y0
        assert abs(W - H * 5 / 4) <= 1.0 and x0 >= 0 and y0 >= 0 and x1 <= 1280 and y1 <= 640
        assert c["rect"][0] <= f1["box"][0] and c["rect"][2] >= f1["box"][2]  # the face is inside
        assert set(c["terms"]) >= {"thirds", "head", "clear", "sal", "size"}
    assert client.get(f"/api/crop/{i}").json()["history"] == []
    assert tree(shoot) == before
    flip = client.post(f"/api/crop/{i}/suggest", json={"ratio": "4:5", "flip": True}).json()
    assert flip["ratio"] == "4:5"  # flip swaps W:H of the image-oriented ratio
    # a landscape image defaults a portrait ratio to landscape ("3:2 not 2:3") unless flipped
    assert client.post(f"/api/crop/{i}/suggest", json={"ratio": "2:3"}).json()["ratio"] == "3:2"
    free = client.post(f"/api/crop/{i}/suggest", json={"ratio": "free"}).json()
    assert free["ratio_px"] is None and free["candidates"]


def test_point_prompt_adds_a_point_subject(client, ids):
    i = ids["landscape.jpg"]
    p = client.post(f"/api/crop/{i}/point", json={"x": 0.5, "y": 0.5}).json()["subject"]
    assert p["kind"] == "point" and p["box"][0] < 0.5 < p["box"][2] and p["box"][1] < 0.5 < p["box"][3]
    r = client.post(f"/api/crop/{i}/suggest", json={"ratio": "1:1", "points": [{"x": 0.5, "y": 0.5}]}).json()
    assert r["candidates"] and any(s["kind"] == "point" for s in r["selected"])
    assert client.post(f"/api/crop/{i}/point", json={"x": 1.5, "y": 0.5}).status_code == 422


def test_accept_reject_reset_restore_history_and_raw_sidecar(client, shoot, ids):
    raw = shoot / "frame.pef"
    before_raw = sha(raw)
    i = ids["frame.pef"]
    st = client.get(f"/api/crop/{i}").json()
    fw, fh = st["full"]["w"], st["full"]["h"]
    assert (fw, fh) == (4309, 2868) and st["kind"] == "dng" or st["kind"] == "raw"
    s = client.post(f"/api/crop/{i}/suggest", json={"ratio": "4:5", "flip": True}).json()
    cand = s["candidates"][0]
    r1 = client.post(f"/api/crop/{i}/accept", json={"rect": cand["rect"], "ratio": s["ratio"], "source": "ai", "alt_rank": 0})
    assert r1.status_code == 200, r1.text
    d1 = r1.json()
    assert d1["version"] == 1
    if raw.suffix.lower() == ".pef":
        assert d1["write"]["status"] == "written" and d1["write"]["mode"] == "sidecar"
        text = (shoot / "frame.xmp").read_text()
        assert 'crs:HasCrop="True"' in text and "shotsieve:CropVersion=\"1\"" in text
        got = crs.read_crop(raw, 1)
        assert got.rect_display == pytest.approx(tuple(d1["rect"]), abs=1e-6) and got.ours
    assert sha(raw) == before_raw  # the original is never modified
    cur = client.get(f"/api/crop/{i}").json()
    assert cur["current"]["version"] == 1 and cur["current"]["source"] == "ai"
    # reject: keeps the previous crop
    assert client.post(f"/api/crop/{i}/reject", json={"alt_rank": 1}).json() == {"ok": True}
    assert client.get(f"/api/crop/{i}").json()["current"]["version"] == 1
    # accept an adjusted crop
    s2 = client.post(f"/api/crop/{i}/suggest", json={"ratio": "1:1"}).json()
    r2 = client.post(f"/api/crop/{i}/accept", json={"rect": s2["candidates"][0]["rect"], "ratio": "1:1", "source": "ai_adjusted"}).json()
    assert r2["version"] == 2
    # reset: new version, full frame, XMP back to Lightroom's defaults
    r3 = client.post(f"/api/crop/{i}/reset", json={}).json()
    assert r3["version"] == 3
    cur = client.get(f"/api/crop/{i}").json()
    assert cur["current"]["rect"] is None and cur["current"]["source"] == "reset"
    assert crs.read_develop_xmp(raw).raw["HasCrop"] == "False"
    assert not crs.read_crop(raw).has_crop
    # restore v1 reproduces it exactly
    r4 = client.post(f"/api/crop/{i}/restore", json={"version": 1}).json()
    assert r4["version"] == 4 and r4["rect"] == d1["rect"]
    hist = client.get(f"/api/crop/{i}").json()["history"]
    assert [h["version"] for h in hist] == [1, 2, 3, 4]
    assert [h["source"] for h in hist] == ["ai", "ai_adjusted", "reset", "restore"]
    assert hist[3]["rect"] == hist[0]["rect"]
    assert client.post(f"/api/crop/{i}/restore", json={"version": 9}).status_code == 404
    assert sha(raw) == before_raw
    shoot_json = client.get("/api/shoot").json()
    assert next(x for x in shoot_json["images"] if x["id"] == i)["crop"] == hist[3]["rect"]
    assert next(x for x in shoot_json["images"] if x["id"] == ids["b0.jpg"])["crop"] is None
    with Cache.for_folder(shoot) as c:  # events for the future taste model
        acts = [e["action"] for e in c.crop_events(i)]
    assert {"suggest", "accept", "reject", "adjust", "reset", "restore"} <= set(acts)


def test_jpeg_crop_is_db_only_with_a_lightroom_warning_and_embed_copies_work(client, shoot, ids):
    i = ids["b0.jpg"]
    before = sha(shoot / "b0.jpg")
    s = client.post(f"/api/crop/{i}/suggest", json={"ratio": "1:1"}).json()
    r = client.post(f"/api/crop/{i}/accept", json={"rect": s["candidates"][0]["rect"], "ratio": "1:1"}).json()
    assert r["write"]["status"] == "db_only" and "Lightroom" in r["write"]["reason"]
    assert not (shoot / "b0.xmp").exists()
    e = client.post(f"/api/crop/{i}/accept", json={"rect": s["candidates"][0]["rect"], "ratio": "1:1", "write": "embed"}).json()
    assert e["write"]["mode"] == "embed" and e["write"]["status"] in ("written", "replaced_own")
    copy = shoot / "shotsieve_edits" / "b0.jpg"
    assert crs.read_crop(copy).has_crop and sha(shoot / "b0.jpg") == before


def test_errors_404_422_and_ratio_validation(client, shoot, ids):
    assert client.get("/api/crop/99999").status_code == 404
    assert client.get("/api/crop/99999/subjects").status_code == 404
    with Cache.for_folder(shoot) as c:
        vid = next(i for p, i in c.ids_lookup(list(c.load_decisions())).items() if "#t=" in p)
    for url in (f"/api/crop/{vid}/subjects", f"/api/crop/{vid}"):
        assert client.get(url).status_code == 422
    assert client.post(f"/api/crop/{vid}/suggest", json={"ratio": "1:1"}).status_code == 422
    i = ids["b0.jpg"]
    for bad in ("5:1", "0:3", "x", "1.234:1", "4"):
        assert client.post(f"/api/crop/{i}/suggest", json={"ratio": bad}).status_code == 422, bad
    ok = {"rect": [0.1, 0.1, 0.5, 0.5], "ratio": "1:1"}
    assert client.post(f"/api/crop/{i}/accept", json=ok).status_code == 200
    for rect in ([0.1, 0.1, 1.2, 0.5], [0.5, 0.1, 0.4, 0.5], [-0.1, 0, 0.5, 0.5], [0.1, 0.1, 0.5]):
        assert client.post(f"/api/crop/{i}/accept", json={**ok, "rect": rect}).status_code == 422, rect
    assert client.post(f"/api/crop/{i}/accept", json={**ok, "rect": [0.1, 0.1, 0.6, 0.4]}).status_code == 422  # not 1:1
    assert client.post(f"/api/crop/{i}/accept", json={**ok, "source": "nope"}).status_code == 422
    assert client.post(f"/api/crop/{i}/accept", json={**ok, "write": "nope"}).status_code == 422
    assert client.post(f"/api/crop/{i}/suggest", json={"ratio": "1:1", "subjects": ["zz9"]}).status_code == 422
    assert client.post(f"/api/crop/{i}/suggest", json={"ratio": "1:1", "n": 0}).status_code == 422
    assert client.post(f"/api/crop/{i}/suggest", json={"ratio": "1:1", "select_modes": {"f1": "arm"}, "subjects": ["f1"]}).status_code == 422
    assert client.post(f"/api/crop/{i}/accept", json=ok, headers={"Origin": "http://evil.example"}).status_code == 403


def test_users_lightroom_crop_is_kept_and_shown_read_only(client, shoot, ids):
    i = ids["frame.pef"]
    (shoot / "frame.xmp").write_text(
        '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description rdf:about="" xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/" crs:HasSettings="True" '
        'crs:Exposure2012="+0.50" crs:HasCrop="True" crs:CropLeft="0.1" crs:CropTop="0.2" crs:CropRight="0.9" crs:CropBottom="0.8"/>'
        "</rdf:RDF></x:xmpmeta>"
    )
    before = sha(shoot / "frame.xmp")
    st = client.get(f"/api/crop/{i}").json()
    assert st["current"]["source"] == "lightroom" and st["current"]["readonly"]
    assert st["current"]["rect"] == pytest.approx([0.1, 0.2, 0.9, 0.8], abs=1e-6)
    assert st["xmp"] == {"has_crop": True, "ours": False, "source": "lightroom"}
    r = client.post(f"/api/crop/{i}/accept", json={"rect": [0.1, 0.1, 0.5, 0.5], "ratio": "free"}).json()
    assert r["write"]["status"] == "skipped_user_crop" and sha(shoot / "frame.xmp") == before
    assert client.get(f"/api/crop/{i}").json()["current"]["version"] == 1  # still saved in ShotSieve


def test_burst_transfer_is_proposals_only(client, shoot, ids):
    src = ids["b0.jpg"]
    s = client.post(f"/api/crop/{src}/suggest", json={"ratio": "4:5"}).json()
    c = s["candidates"][0]
    r = client.post("/api/crop/burst", json={"id": src, "rect": c["rect"], "ratio": s["ratio"]})
    assert r.status_code == 200, r.text
    j = wait(client, r.json()["job_id"])
    assert j["state"] == "done", j
    res = {x["id"]: x for x in j["result"]["results"]}
    assert set(res) == {ids["b1.jpg"], ids["b2.jpg"]}
    assert all(x["status"] in ("suggested", "needs_selection") for x in res.values())
    assert any(x["status"] == "suggested" for x in res.values())
    for x in res.values():
        if x["status"] == "suggested":
            x0, y0, x1, y1 = x["rect_px"]
            assert abs((x1 - x0) - (y1 - y0) * 4 / 5) <= 1.0
    for k in res:  # nothing accepted automatically
        assert client.get(f"/api/crop/{k}").json()["history"] == []
    acc = client.post("/api/crop/accept_many", json={"items": [
        {"id": k, "rect": x["rect"], "ratio": s["ratio"]} for k, x in res.items() if x["rect"]]}).json()
    assert all(x["ok"] for x in acc["results"])
    assert client.get(f"/api/crop/{ids['b1.jpg']}").json()["history"][0]["source"] == "burst"
    assert client.post("/api/crop/burst", json={"id": ids["two.jpg"], "rect": c["rect"], "ratio": "4:5", "ids": []}).status_code == 422
    assert client.post("/api/crop/burst", json={"id": src, "rect": [0.5, 0.5, 0.4, 0.4], "ratio": "4:5"}).status_code == 422


def test_burst_job_holds_the_one_job_lock(client, ids):
    src = ids["b0.jpg"]
    s = client.post(f"/api/crop/{src}/suggest", json={"ratio": "4:5"}).json()
    body = {"id": src, "rect": s["candidates"][0]["rect"], "ratio": s["ratio"]}
    first = client.post("/api/crop/burst", json=body)
    second = client.post("/api/crop/burst", json=body)
    assert first.status_code == 200
    assert second.status_code in (200, 409)  # 409 while the first one runs
    wait(client, first.json()["job_id"])


# ---- CLI ------------------------------------------------------------------------------------


def test_cli_suggest_writes_nothing_apply_dry_run_writes_nothing(shoot, ids, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SHOTSIEVE_DATA_DIR", str(tmp_path / "data"))
    before = tree(shoot)
    assert main(["crop", "suggest", str(shoot), "--ratio", "4:5", "--ids", str(ids["b0.jpg"]), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["n"] == 1 and out["results"][0]["candidates"]
    with Cache.for_folder(shoot) as c:
        assert c.crops_current_all() == {} and c.crop_events() == []
    assert main(["crop", "apply", str(shoot), "--ratio", "1:1", "--ids", str(ids["b0.jpg"]), "--dry-run", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["results"][0]["dry_run"] is True
    with Cache.for_folder(shoot) as c:
        assert c.crops_current_all() == {}
    assert tree(shoot) == before


def test_cli_apply_status_history_reset_and_exit_codes(shoot, ids, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SHOTSIEVE_DATA_DIR", str(tmp_path / "data"))
    i = ids["frame.pef"]
    assert main(["crop", "apply", str(shoot), "--ratio", "4:5", "--ids", str(i), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["n"] == 1 and out["results"][0]["write"]["status"] == "written"
    assert crs.read_crop(shoot / "frame.pef", 1).ours
    assert main(["crop", "status", str(shoot), "--json"]) == 0
    st = json.loads(capsys.readouterr().out)
    assert st["n"] == 1 and st["cropped"][0]["id"] == i and st["cropped"][0]["ratio"] == "5:4"
    assert main(["crop", "history", str(shoot), "--id", str(i)]) == 0
    assert "v1" in capsys.readouterr().out
    assert main(["crop", "reset", str(shoot), "--all", "--json"]) == 0
    capsys.readouterr()
    assert not crs.read_crop(shoot / "frame.pef").has_crop
    assert main(["crop", "reset", str(shoot), "--all"]) == 3  # nothing left to reset
    # explicit rect
    assert main(["crop", "apply", str(shoot), "--ratio", "1:1", "--id", str(ids["b1.jpg"]), "--rect", "0.1,0.1,0.5,0.5",
                 "--ids", str(ids["b1.jpg"])]) == 0
    capsys.readouterr()
    # exit codes: 2 bad args / ratio, 3 nothing to do
    assert main(["crop", "suggest", str(shoot), "--ratio", "7:1"]) == 2
    assert main(["crop", "apply", str(shoot), "--ratio", "1:1", "--rect", "0,0,1,1"]) == 2
    assert main(["crop", "suggest", str(shoot), "--ratio", "1:1", "--ids", "99999"]) == 3
    assert main(["crop", "history", str(shoot), "--id", "99999"]) == 3
    with pytest.raises(SystemExit) as e:
        main(["crop", "suggest", str(shoot)])
    assert e.value.code == 2
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["crop", "status", str(empty)]) == 3


def test_cli_detect_fills_the_cache(shoot, ids, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SHOTSIEVE_DATA_DIR", str(tmp_path / "data"))
    assert main(["crop", "detect", str(shoot), "--ids", f"{ids['two.jpg']},{ids['b0.jpg']}", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["n"] == 2
    with Cache.for_folder(shoot) as c:
        assert c.get_subjects(str(shoot / "two.jpg")) is not None


def test_edit_apply_is_not_blocked_by_a_crop_only_sidecar(client, shoot, ids, edit_env, monkeypatch):
    """The T15/T22 interaction on the real `edit apply` path: crop first, edit second.

    About crop + edit, not prediction quality: the learned profile gates sliders by confidence and
    out-of-distribution distance, which varies with the execution provider (CoreML vs CPU), so the
    prediction is pinned to a fixed, writable one.
    """
    from shotsieve.pipeline import edit

    fixed = {"Exposure2012": 0.4, "Contrast2012": 10.0}

    def fake_predict(paths, profile, settings=None, progress=None, **kw):
        return {p: edit.ImageEdit(p, dict(fixed), point=dict(fixed)) for p in paths}

    monkeypatch.setattr(edit, "predict_paths", fake_predict)
    i = ids["frame.pef"]
    assert client.post(f"/api/crop/{i}/accept", json={"rect": [0.1, 0.1, 0.9, 0.9], "ratio": "free"}).json()["write"]["status"] == "written"
    cur = crs.read_develop_xmp(shoot / "frame.pef")
    assert cur.only_ours and cur.has_settings and not cur.ours
    ok, why = crs.writable(shoot / "frame.pef", "sidecar")
    assert ok, why
    assert edit._mode_for(str(shoot / "frame.pef"), False) == "sidecar"
    res = edit.apply_folder(shoot, Settings(), scope="all", ids=[i])
    assert res["predicted"] == 1 and res["diff"][0]["values"] == fixed, res
    assert res["diff"][0]["status"] == "written", res
    assert res["written_sidecar"] == 1 and not res["skipped"], res
    cur = crs.read_develop_xmp(shoot / "frame.pef")
    assert cur.ours and cur.crop_ours and float(cur.raw["CropLeft"]) == pytest.approx(0.1, abs=1e-3)
    left = cur.raw["CropLeft"]
    again = edit.apply_folder(shoot, Settings(), scope="all", ids=[i])
    assert again["written_sidecar"] == 1 and not again["skipped"]  # replaced_own, crop untouched
    assert crs.read_develop_xmp(shoot / "frame.pef").raw["CropLeft"] == left
