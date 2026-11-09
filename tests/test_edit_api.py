# ruff: noqa: E501
"""T15: /api/edit/* and `shotsieve edit` (train refusal, apply, status, job lifecycle)."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from shotsieve.cli import main
from shotsieve.config import Settings
from shotsieve.features.vector import FeatureVector
from shotsieve.io import crs
from shotsieve.pipeline.cache import Cache
from shotsieve.server.app import create_app

ROOT = Path(__file__).resolve().parents[1]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _row(path, decision, group):
    return dict(path=str(path), decision=decision, reasons=[], score=0.7, prior=0.7,
                group_id=group, burst_id=group, is_best=True, audit=False, audit_prob=None,
                features=FeatureVector(n_faces=1.0).to_dict())  # fmt: skip


@pytest.fixture()
def shoot(tmp_path, shoot_dir):
    folder = tmp_path / "shoot"
    folder.mkdir()
    names = ["burst0_0", "landscape", "black"]
    for n in names:
        shutil.copy(shoot_dir / f"{n}.jpg", folder / f"{n}.jpg")
    with Cache.for_folder(folder) as c:
        c.save_decisions([_row(folder / f"{n}.jpg", d, i)
                          for i, (n, d) in enumerate(zip(names, ["keep", "keep", "reject"], strict=True))])  # fmt: skip
    return folder


@pytest.fixture()
def client(shoot, edit_env):
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


def test_profile_status_and_delete(client, edit_env):
    d = client.get("/api/edit/profile").json()
    assert d["trained"] and d["n"] >= 1100 and len(d["sliders"]) == 12
    assert {r["mode"] for r in d["sliders"]} <= {"learned", "constant", "off"}
    assert client.delete("/api/edit/profile").json() == {"deleted": True}
    assert client.get("/api/edit/profile").json()["trained"] is False
    assert client.delete("/api/edit/profile").json() == {"deleted": False}
    r = client.post("/api/edit/apply", json={"scope": "keeps"})
    assert r.status_code == 409 and "train" in r.json()["detail"]


def test_apply_job_dry_run_then_write_copies(client, shoot):
    before = {p.name: sha(p) for p in shoot.glob("*.jpg")}
    r = client.post("/api/edit/apply", json={"folder": str(shoot), "scope": "keeps", "dry_run": True,
                                             "embed_copies": True})  # fmt: skip
    assert r.status_code == 200
    j = wait(client, r.json()["job_id"])
    assert j["state"] == "done" and j["kind"] == "edit_apply"
    res = j["result"]
    assert res["dry_run"] and res["n"] == 2 and res["written_copies"] == 2 and res["diff"]
    assert not (shoot / "shotsieve_edits").exists()
    j = wait(client, client.post("/api/edit/apply", json={"folder": str(shoot), "embed_copies": True}).json()["job_id"])
    assert j["state"] == "done" and j["result"]["written_copies"] == 2
    copies = sorted((shoot / "shotsieve_edits").glob("*.jpg"))
    assert [c.name for c in copies] == ["burst0_0.jpg", "landscape.jpg"]
    assert all(crs.read_develop_xmp(c).ours for c in copies)
    assert {p.name: sha(p) for p in shoot.glob("*.jpg")} == before
    # the shoot payload carries the badge data and the predictions endpoint serves stored edits
    shoot_json = client.get("/api/shoot", params={"folder": str(shoot)}).json()
    flags = {i["name"]: i["edit"] for i in shoot_json["images"]}
    assert flags["burst0_0.jpg"] is not None and flags["black.jpg"] is None
    ids = [i["id"] for i in shoot_json["images"] if i["name"] == "landscape.jpg"]
    p = client.get("/api/edit/predictions", params={"ids": ",".join(map(str, ids)),
                                                    "folder": str(shoot)}).json()["predictions"]
    got = p[str(ids[0])]
    assert got["applied"] and got["written_mode"] == "copy" and got["values"]


def test_apply_report_only_for_jpeg_without_embed(client, shoot):
    j = wait(client, client.post("/api/edit/apply", json={"folder": str(shoot)}).json()["job_id"])
    assert j["state"] == "done" and j["result"]["report_only"] == 2
    assert any("Lightroom" in w for w in j["result"]["warnings"])


def test_apply_validation_and_single_job_lock(client, shoot, tmp_path):
    assert client.post("/api/edit/apply", json={"folder": str(shoot), "scope": "x"}).status_code == 400
    assert client.post("/api/edit/apply", json={"folder": "relative/path"}).status_code == 400
    assert client.post("/api/edit/apply", json={"folder": str(shoot), "embed_copies": "/a/../b"}).status_code == 400
    assert client.post("/api/edit/apply", json={"folder": str(shoot), "catalog": str(tmp_path / "no.lrcat")}).status_code == 400
    assert client.get("/api/edit/predictions", params={"ids": "x"}).status_code == 400
    assert client.post("/api/edit/train", json={}).status_code == 400
    assert client.post("/api/edit/train", json={"catalog": "../x.lrcat"}).status_code == 400
    assert client.post("/api/edit/train", json={"xmp": [str(tmp_path / "missing")]}).status_code == 400


def test_train_job_refuses_under_200_edits_with_a_message(client, tmp_path, edit_styles, edit_env, monkeypatch):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("mes_api", ROOT / "scripts" / "make_edit_styles.py")
    mes = importlib.util.module_from_spec(spec)
    sys.modules["mes_api"] = mes
    spec.loader.exec_module(mes)
    truth = json.loads((edit_styles / "truth.json").read_text())[:30]
    cat = tmp_path / "few.lrcat"
    mes.build_catalog(cat, [{"root": str(edit_styles / "images"), "rel": str(Path(t["file"]).parent) + "/",
                             "base": Path(t["file"]).stem, "ext": "jpg",
                             "develop": mes.develop_settings(t["truth"]["A"])} for t in truth])  # fmt: skip
    keep = (edit_env / "edit_profile.joblib").read_bytes()
    r = client.post("/api/edit/train", json={"catalog": str(cat)})
    assert r.status_code == 200
    j = wait(client, r.json()["job_id"])
    assert j["state"] == "error" and j["error"] == "need 200 edits, have 30; use `match` instead"
    assert (edit_env / "edit_profile.joblib").read_bytes() == keep  # a refusal leaves the profile alone


# ---- CLI --------------------------------------------------------------------------------------


def test_cli_status_apply_and_refusal(shoot, edit_env, capsys, tmp_path):
    assert main(["edit", "status", "--json"]) == 0
    st = json.loads(capsys.readouterr().out)
    assert st["trained"] and len(st["sliders"]) == 12
    assert main(["edit", "status"]) == 0 and "Exposure2012" in capsys.readouterr().out
    before = {p.name: sha(p) for p in shoot.glob("*.jpg")}
    assert main(["edit", "apply", str(shoot), "--dry-run", "--embed-copies", "--json"]) == 0
    res = json.loads(capsys.readouterr().out)
    assert res["dry_run"] and res["written_copies"] == 2 and not (shoot / "shotsieve_edits").exists()
    out = tmp_path / "copies"
    assert main(["edit", "apply", str(shoot), "--embed-copies", str(out)]) == 0
    text = capsys.readouterr().out
    assert "wrote 0 sidecar(s) and 2 embedded copy(ies)" in text
    assert sorted(p.name for p in out.glob("*.jpg")) == ["burst0_0.jpg", "landscape.jpg"]
    assert {p.name: sha(p) for p in shoot.glob("*.jpg")} == before
    assert main(["edit", "apply", str(shoot)]) == 0
    cap = capsys.readouterr()
    assert "reported only" in cap.out and "Lightroom" in cap.err
    # `--scope all` also covers the rejected frame
    assert main(["edit", "apply", str(shoot), "--scope", "all", "--dry-run", "--embed-copies", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["n"] == 3
    assert main(["edit", "train"]) == 2 and "--catalog" in capsys.readouterr().err


def test_cli_without_profile_exits_3(shoot, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SHOTSIEVE_DATA_DIR", str(tmp_path / "none"))
    assert main(["edit", "apply", str(shoot)]) == 3
    assert "edit train" in capsys.readouterr().err
    assert main(["edit", "status"]) == 0 and "no editing profile" in capsys.readouterr().out


def test_cli_train_end_to_end_and_refusal(tmp_path, monkeypatch, edit_styles, capsys, edit_sets):
    d = tmp_path / "data"
    d.mkdir()
    monkeypatch.setenv("SHOTSIEVE_DATA_DIR", str(d))
    shutil.copy(edit_styles / "features.db", d / "edit_features.db")  # warm feature cache
    assert main(["edit", "train", "--catalog", str(edit_styles / "styleB.lrcat"), "--json"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["n"] == 1415 and info["size_bytes"] <= 10_000_000 and info["carry_keys"] >= 5
    modes = {r["slider"]: r["mode"] for r in info["sliders"]}
    assert modes["Exposure2012"] == "learned" and modes["jpeg_temp"] == "learned"
    assert modes["Contrast2012"] == "constant" and modes["raw_temp"] == "off"
    assert (d / "edit_profile.joblib").exists()
    assert main(["edit", "train", "--xmp", str(tmp_path)]) == 3  # no edits found there
    assert "need 200 edits, have 0" in capsys.readouterr().err
