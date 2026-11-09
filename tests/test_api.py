"""FastAPI backend: job lifecycle, shoot listing, thumbs, overrides, audit, XMP, path safety."""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from shotsieve.config import Settings
from shotsieve.features.vector import FeatureVector
from shotsieve.io.xmp import read_sidecar
from shotsieve.pipeline.cache import Cache
from shotsieve.server.app import create_app

# ---- a shoot seeded straight into cull.db (no models needed) -------------------------------


def _row(path, decision, group, best=False, audit=False, prob=None, reasons=(), score=0.5):
    return dict(path=str(path), decision=decision, reasons=list(reasons), score=score, prior=score,
                group_id=group, burst_id=group, is_best=best, audit=audit, audit_prob=prob,
                features=FeatureVector(n_faces=1.0, sharp_global_norm=0.8).to_dict())  # fmt: skip


@pytest.fixture()
def seeded(tmp_path, shoot_dir):
    """6 real JPEGs: group 0 = burst0_0 (best), burst0_1 (dup), burst0_2 (dup); singles:
    landscape (keep), black (reject), burst1_3 (reject, audit p=0.25)."""
    folder = tmp_path / "shoot"
    folder.mkdir()
    names = ["burst0_0", "burst0_1", "burst0_2", "landscape", "black", "burst1_3"]
    for n in names:
        shutil.copy(shoot_dir / f"{n}.jpg", folder / f"{n}.jpg")
    rows = [
        _row(folder / "burst0_0.jpg", "keep", 0, True, score=0.9),
        _row(folder / "burst0_1.jpg", "reject", 0, reasons=["duplicate"], score=0.4),
        _row(folder / "burst0_2.jpg", "reject", 0, reasons=["duplicate_tie"], score=0.85),
        _row(folder / "landscape.jpg", "keep", 1, True, score=0.8),
        _row(folder / "black.jpg", "reject", 2, True, reasons=["accidental"], score=0.0),
        _row(
            folder / "burst1_3.jpg", "review", 3, True, True, 0.25, ["audit", "exposure_bad"], 0.2
        ),
    ]
    with Cache.for_folder(folder) as c:
        c.save_decisions(rows)
    return folder


@pytest.fixture()
def client(seeded):
    with TestClient(create_app(Settings(), default_folder=seeded)) as c:
        yield c


def shoot_of(client, folder):
    r = client.get("/api/shoot", params={"folder": str(folder)})
    assert r.status_code == 200, r.text
    d = r.json()
    return d, {im["name"]: im for im in d["images"]}


def test_health(client):
    d = client.get("/api/health").json()
    assert d["ok"] is True and set(d["models"]) == {
        "yunet",
        "sface",
        "face_landmarker",
        "dinov2_small",
    }
    assert isinstance(d["coreml"], bool)


def test_index_and_static_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "ShotSieve" in r.text and "/static/app.js" in r.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200


def test_shoot_listing(client, seeded):
    d, by = shoot_of(client, seeded)
    assert d["stats"]["counts"] == {"keep": 2, "review": 1, "reject": 3}
    assert d["stats"]["n_audit"] == 1 and d["stats"]["n_audit_pending"] == 1
    assert d["stats"]["thresholds"] == {"t_keep": 0.6, "t_reject": 0.3}
    ids = {im["id"] for im in d["images"]}
    assert len(ids) == 6 and all(isinstance(i, int) for i in ids)
    g0 = next(g for g in d["groups"] if len(g["members"]) == 3)
    assert g0["best"] == by["burst0_0.jpg"]["id"] and g0["members"][0] == g0["best"]
    assert by["burst0_0.jpg"]["best"] and not by["burst0_1.jpg"]["best"]
    # plain-language reasons, naming the better frame
    assert "burst0_0.jpg" in by["burst0_1.jpg"]["reason_texts"][0]
    assert "Near-identical to the best frame burst0_0.jpg" in by["burst0_2.jpg"]["reason_texts"][0]
    # the structured pointer to the better frame (T23b), next to the unchanged text
    best_id = by["burst0_0.jpg"]["id"]
    assert by["burst0_1.jpg"]["duplicate_of"] == {"image_id": best_id, "name": "burst0_0.jpg"}
    assert by["burst0_2.jpg"]["duplicate_of"]["image_id"] == best_id
    assert by["burst0_0.jpg"]["duplicate_of"] is None
    assert set(by["burst0_1.jpg"]["metrics"]) == {"sharp", "eyes_open", "face", "n_faces"}
    assert by["burst1_3.jpg"]["audit"] and by["burst1_3.jpg"]["audit_prob"] == 0.25
    assert len(by["burst1_3.jpg"]["reason_texts"]) == 2
    assert d["groups"][0]["id"] in (0, 1, 2, 3)
    # ids are stable across calls
    assert {im["name"]: im["id"] for im in shoot_of(client, seeded)[0]["images"]} == {
        n: i["id"] for n, i in by.items()
    }


def test_shoot_uses_default_folder_and_404_when_not_culled(client, tmp_path):
    assert client.get("/api/shoot").status_code == 200
    empty = tmp_path / "empty"
    empty.mkdir()
    assert client.get("/api/shoot", params={"folder": str(empty)}).status_code == 404


def test_thumb_and_preview(client, seeded):
    _, by = shoot_of(client, seeded)
    iid = by["landscape.jpg"]["id"]
    r = client.get(f"/api/thumb/{iid}", params={"size": 200, "folder": str(seeded)})
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert max(img.shape[:2]) <= 200
    cached = list((seeded / ".shotsieve" / "thumbs").glob("t_*_200_*.jpg"))
    assert len(cached) == 1  # generated once, then served from .shotsieve/thumbs/
    mtime = cached[0].stat().st_mtime_ns
    assert client.get(f"/api/thumb/{iid}", params={"size": 200}).content == r.content
    assert cached[0].stat().st_mtime_ns == mtime
    p = client.get(f"/api/preview/{iid}")
    assert p.status_code == 200
    big = cv2.imdecode(np.frombuffer(p.content, np.uint8), cv2.IMREAD_COLOR)
    assert max(big.shape[:2]) <= 1600
    assert not list((seeded / ".shotsieve" / "thumbs").glob("*.tmp"))


def test_override_propensity_rating_and_clear(client, seeded):
    _, by = shoot_of(client, seeded)
    audit_id, plain_id = by["burst1_3.jpg"]["id"], by["burst0_1.jpg"]["id"]
    r = client.post("/api/override", json={"image_id": plain_id, "decision": "keep",
                                           "folder": str(seeded)}).json()  # fmt: skip
    assert r["decision"] == "keep" and r["propensity"] == 1.0 and r["overridden"]
    r = client.post("/api/override", json={"image_id": audit_id, "decision": "keep"}).json()
    assert r["propensity"] == 0.25  # audit items carry their sampling probability
    with Cache.for_folder(seeded) as c:
        ovr = c.overrides()
    assert ovr[str(seeded / "burst0_1.jpg")]["propensity"] == 1.0
    assert ovr[str(seeded / "burst1_3.jpg")]["propensity"] == 0.25
    _, by = shoot_of(client, seeded)
    assert (
        by["burst0_1.jpg"]["decision"] == "keep"
        and by["burst0_1.jpg"]["model_decision"] == "reject"
    )
    assert by["burst0_1.jpg"]["overridden"]
    # rating only: a rated image counts as wanted; the rating sticks to the override
    black = by["black.jpg"]["id"]
    r = client.post("/api/override", json={"image_id": black, "rating": 4}).json()
    assert r["rating"] == 4 and r["decision"] == "keep"
    r = client.post("/api/override", json={"image_id": black, "decision": "reject"}).json()
    assert r["decision"] == "reject" and r["rating"] == 4
    # U: back to the model's decision
    r = client.post("/api/override", json={"image_id": plain_id, "decision": "model"}).json()
    assert r["decision"] == "reject" and not r["overridden"]
    assert shoot_of(client, seeded)[1]["burst0_1.jpg"]["decision"] == "reject"
    # validation
    bad = [{"image_id": plain_id, "decision": "maybe"}, {"image_id": plain_id, "rating": 9},
           {"image_id": plain_id}]  # fmt: skip
    assert [client.post("/api/override", json=b).status_code for b in bad] == [422, 422, 422]
    assert (
        client.post("/api/override", json={"image_id": 9999, "decision": "keep"}).status_code == 404
    )


def test_audit_complete_without_rescues_keeps_thresholds(client, seeded):
    _, by = shoot_of(client, seeded)
    client.post("/api/override", json={"image_id": by["burst1_3.jpg"]["id"], "decision": "reject"})
    r = client.post("/api/audit/complete", json={"folder": str(seeded)})
    assert r.status_code == 200
    d = r.json()
    assert d["n_audited"] == 1 and d["n_kept"] == 0 and d["missed_keeper_rate"] == 0.0
    assert d["thresholds"]["t_keep"] == 0.6 and d["thresholds"]["t_reject"] == 0.3
    assert d["thresholds"]["hard_margin"] == 0.0 and not d["lowered"]


def test_audit_complete_reports_missed_keepers(client, seeded, monkeypatch):
    called = {}

    def fake_run(folder, settings, *a, **k):
        called["folder"] = folder
        from types import SimpleNamespace

        return SimpleNamespace(stats={"counts": {"keep": 1, "review": 2, "reject": 3}})

    monkeypatch.setattr("shotsieve.pipeline.runner.run_cull", fake_run)
    _, by = shoot_of(client, seeded)
    client.post("/api/override", json={"image_id": by["burst1_3.jpg"]["id"], "decision": "keep"})
    d = client.post("/api/audit/complete", json={"folder": str(seeded)}).json()
    # one rescued reject sampled with p=0.25 -> 4 est. missed keepers vs 2 kept -> lower t_reject
    assert d["n_kept"] == 1 and d["est_missed_keepers"] == pytest.approx(4.0)
    assert d["missed_keeper_rate"] == pytest.approx(4 / 6)
    assert d["lowered"] and d["thresholds"]["t_reject"] < 0.3 and d["redecided"]
    assert Path(called["folder"]) == seeded
    d2, _ = shoot_of(client, seeded)
    assert d2["stats"]["thresholds"]["t_reject"] == d["thresholds"]["t_reject"]  # persisted


def test_xmp_write_uses_final_decisions(client, seeded):
    _, by = shoot_of(client, seeded)
    client.post("/api/override", json={"image_id": by["black.jpg"]["id"], "decision": "keep",
                                       "rating": 5})  # fmt: skip
    r = client.post("/api/xmp/write", json={"folder": str(seeded)})
    assert r.status_code == 200 and r.json()["written"] == 6 and r.json()["failed"] == 0
    black = read_sidecar(seeded / "black.jpg")
    assert black.decision == "keep" and black.label == "Green" and black.rating == 5
    dup = read_sidecar(seeded / "burst0_1.jpg")
    assert dup.decision == "reject" and dup.label == "Red" and "burst0_0.jpg" in dup.reasons[0]
    assert read_sidecar(seeded / "landscape.jpg").rating == 3
    assert read_sidecar(seeded / "burst1_3.jpg").label == "Yellow"
    assert not list(seeded.glob("*.tmp"))


# ---- path safety ---------------------------------------------------------------------------


def test_path_validation_and_traversal_rejected(client, seeded, tmp_path):
    for bad in (
        str(seeded / ".." / ".."),
        "../etc",
        "relative/dir",
        str(tmp_path / "does-not-exist"),
        str(seeded / "landscape.jpg"),  # a file, not a directory
        "/etc/passwd",
        "/tmp/\x00evil",
    ):
        assert client.get("/api/shoot", params={"folder": bad}).status_code == 400, bad
        assert client.post("/api/cull", json={"folder": bad}).status_code == 400, bad
        assert client.post("/api/xmp/write", json={"folder": bad}).status_code == 400, bad
        assert client.get("/api/thumb/1", params={"folder": bad}).status_code == 400, bad
    # image ids are integers resolved server-side; paths cannot be smuggled in
    assert client.get("/api/thumb/..%2f..%2fetc%2fpasswd").status_code in (404, 422)
    assert client.get("/api/thumb/abc").status_code == 422
    assert client.get("/api/thumb/424242", params={"folder": str(seeded)}).status_code == 404
    assert client.get("/api/preview/424242").status_code == 404
    assert client.get("/static/../pyproject.toml").status_code in (404, 400)
    assert client.get("/static/%2e%2e/app.py").status_code == 404


def test_thumb_refuses_ids_pointing_outside_the_folder(client, seeded, tmp_path):
    outside = tmp_path / "secret.jpg"
    outside.write_bytes(b"x")
    with Cache.for_folder(seeded) as c:
        iid = c.ids_for([str(outside)])[str(outside)]
    assert client.get(f"/api/thumb/{iid}", params={"folder": str(seeded)}).status_code == 404


def test_foreign_host_header_rejected(seeded):
    with TestClient(create_app(Settings(), default_folder=seeded)) as c:
        assert c.get("/api/health", headers={"host": "evil.example.com"}).status_code == 400
        assert c.get("/api/health", headers={"host": "127.0.0.1:8765"}).status_code == 200


# ---- jobs ----------------------------------------------------------------------------------


def wait_job(client, job_id, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["state"] in ("done", "error"):
            return j
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_one_job_at_a_time_and_progress(client, seeded, monkeypatch):
    gate = threading.Event()
    reached = threading.Event()

    def slow_run(folder, settings, progress=None, write_xmp=False, **kw):
        progress("features", 1, 4)
        reached.set()  # progress is published; the job now blocks until the test opens the gate
        assert gate.wait(30)
        progress("score", 4, 4)
        from types import SimpleNamespace

        return SimpleNamespace(
            stats={"counts": {"keep": 1, "review": 0, "reject": 0}, "n_images": 1}
        )

    monkeypatch.setattr("shotsieve.pipeline.runner.run_cull", slow_run)
    monkeypatch.setattr("shotsieve.models.registry.verify_model", lambda *a, **k: True)
    r = client.post("/api/cull", json={"folder": str(seeded)})
    assert r.status_code == 200
    jid = r.json()["job_id"]
    assert reached.wait(30)
    j = client.get(f"/api/jobs/{jid}").json()
    assert j["state"] == "running" and (j["stage"], j["done"], j["total"]) == ("features", 1, 4)
    assert client.post("/api/cull", json={"folder": str(seeded)}).status_code == 409
    assert client.post("/api/audit/complete", json={"folder": str(seeded)}).status_code == 409
    gate.set()
    done = wait_job(client, jid)
    assert done["state"] == "done" and done["error"] is None and done["stage"] == "score"
    assert client.get("/api/jobs/nope").status_code == 404
    gate2 = client.post("/api/cull", json={"folder": str(seeded)})
    assert gate2.status_code == 200  # a new job may start once the previous one finished
    wait_job(client, gate2.json()["job_id"])


def test_failing_job_reports_error(client, seeded, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("kaboom")

    monkeypatch.setattr("shotsieve.pipeline.runner.run_cull", boom)
    monkeypatch.setattr("shotsieve.models.registry.verify_model", lambda *a, **k: True)
    j = wait_job(client, client.post("/api/cull", json={"folder": str(seeded)}).json()["job_id"])
    assert j["state"] == "error" and "kaboom" in j["error"]


def test_cull_without_models_is_503(client, seeded, monkeypatch):
    monkeypatch.setattr("shotsieve.models.registry.verify_model", lambda *a, **k: False)
    assert client.post("/api/cull", json={"folder": str(seeded)}).status_code == 503


@pytest.mark.models
@pytest.mark.usefixtures("models_ready")
def test_full_lifecycle_on_fixture_shoot(tmp_path, shoot_dir):
    folder = tmp_path / "shoot"
    shutil.copytree(shoot_dir, folder)
    with TestClient(create_app(Settings(audit_enabled=True), default_folder=None)) as c:
        assert c.get("/api/shoot").status_code == 400  # nothing selected yet
        jid = c.post("/api/cull", json={"folder": str(folder)}).json()["job_id"]
        j = wait_job(c, jid)
        assert j["state"] == "done", j
        assert j["done"] == j["total"] and sum(j["result"]["counts"].values()) == 17
        d = c.get("/api/shoot", params={"folder": str(folder)}).json()
        assert len(d["images"]) == 17 and d["stats"]["run"]["n_images"] == 17
        assert len(d["groups"]) == 5 and d["stats"]["n_audit"] >= 1
        by = {im["name"]: im for im in d["images"]}
        # every reject / review explains itself in plain language
        for im in d["images"]:
            assert len(im["reason_texts"]) == len(im["reasons"])
            if im["decision"] == "reject":
                assert im["reason_texts"]
        iid = by["landscape.jpg"]["id"]
        t = c.get(f"/api/thumb/{iid}", params={"folder": str(folder), "size": 300})
        assert t.status_code == 200 and t.content[:2] == b"\xff\xd8"
        rejected_before = {im["name"] for im in d["images"] if im["model_decision"] == "reject"}
        # override the audit items (P on the first, X on the rest), then finish the audit
        audits = [im for im in d["images"] if im["audit"]]
        for k, im in enumerate(audits):
            r = c.post(
                "/api/override",
                json={"image_id": im["id"], "decision": "keep" if k == 0 else "reject"},
            )
            assert r.json()["propensity"] == pytest.approx(im["audit_prob"])
        ac = c.post("/api/audit/complete", json={"folder": str(folder)}).json()
        assert ac["n_audited"] == len(audits) and ac["n_kept"] == 1
        assert ac["missed_keeper_rate"] > 0.01 and ac["lowered"] and ac["redecided"]
        # B2: recalibration really changes outcomes - the reject set shrinks
        lever = ("t_reject", "hard_margin", "dup_margin")
        assert ac["thresholds"]["t_reject"] < 0.3 or any(ac["thresholds"][k] > 0 for k in lever[1:])
        after = c.get("/api/shoot", params={"folder": str(folder)}).json()
        rejected_after = {im["name"] for im in after["images"] if im["model_decision"] == "reject"}
        assert rejected_before - rejected_after, "recalibration left the rejected set unchanged"
        # the dup lever is budgeted (<= 10 % of 17 frames = 1 review), so compare on the frames
        # that were not audit samples (audit frames show as review regardless of the model)
        aud = {im["name"] for im in audits}
        assert len(rejected_after - aud) < len(rejected_before - aud)
        assert ac["thresholds"]["dup_review_budget"] <= 0.10
        # XMP for all images (final decisions)
        x = c.post("/api/xmp/write", json={"folder": str(folder)}).json()
        assert x["written"] == 17 and x["failed"] == 0
        assert read_sidecar(folder / "landscape.jpg").decision in ("keep", "review")
        # re-listing after the re-decide still has one image per file
        assert len(c.get("/api/shoot", params={"folder": str(folder)}).json()["images"]) == 17


# ---- T18: health identity, quit, server.json -------------------------------------------------


def test_health_identifies_app(client):
    d = client.get("/api/health").json()
    assert d["app"] == "shotsieve" and d["pid"] == os.getpid() and d["version"]


def test_quit_refuses_foreign_origin(client):
    r = client.post("/api/quit", headers={"Origin": "http://evil.example"})
    assert r.status_code == 403
    assert client.get("/api/health").status_code == 200


def test_quit_sets_should_exit_and_removes_server_json(client, tmp_path, monkeypatch):
    import time

    from shotsieve.server.lifecycle import server_json_path, write_server_json

    monkeypatch.setenv("SHOTSIEVE_DATA_DIR", str(tmp_path / "d"))
    write_server_json(8765)
    assert server_json_path().exists()

    class Fake:
        should_exit = False

    fake = Fake()
    client.app.state.uvicorn_server = fake
    r = client.post("/api/quit", headers={"Origin": "http://127.0.0.1:8765"})
    assert r.status_code == 200 and r.json()["quitting"] is True
    for _ in range(30):
        if fake.should_exit:
            break
        time.sleep(0.1)
    assert fake.should_exit and not server_json_path().exists()


def test_serve_writes_and_removes_server_json(tmp_path):
    import json
    import socket
    import subprocess
    import sys
    import time
    import urllib.request

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = {**os.environ, "SHOTSIEVE_DATA_DIR": str(tmp_path)}
    proc = subprocess.Popen(
        [sys.executable, "-m", "shotsieve.cli", "serve", "--port", str(port), "--no-open"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    sj = tmp_path / "server.json"
    try:
        for _ in range(100):
            if sj.exists():
                break
            time.sleep(0.2)
        info = json.loads(sj.read_text())
        assert info["pid"] == proc.pid and info["port"] == port and info["version"]
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/quit", method="POST")
        assert json.load(urllib.request.urlopen(req, timeout=5))["ok"]
        proc.wait(timeout=5)
        assert not sj.exists()
    finally:
        if proc.poll() is None:
            proc.kill()


# ---- T19: onboarding, models lane, file browser, train -------------------------------------


@pytest.fixture()
def ob_env(tmp_path, monkeypatch):
    """Sandbox HOME + data dir + models dir for the onboarding endpoints."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("SHOTSIEVE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SHOTSIEVE_MODELS_DIR", str(tmp_path / "models"))
    return home


@pytest.fixture()
def ob_client(ob_env):
    with TestClient(create_app(Settings())) as c:
        yield c


def _snapshot(path: Path):
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in path.rglob("*")}


def test_onboarding_state_roundtrip(ob_client, tmp_path):
    d = ob_client.get("/api/onboarding").json()
    assert d["show"] is True and d["completed"] is False and d["skipped"] is False
    assert not (tmp_path / "data" / "onboarding.json").exists()  # GET never writes
    r = ob_client.post("/api/onboarding", json={"last_step": 3}).json()
    assert r["last_step"] == 3
    ob_client.post("/api/onboarding", json={"skipped": True})
    d = ob_client.get("/api/onboarding").json()
    assert d["skipped"] is True and d["show"] is False
    ob_client.post("/api/onboarding", json={"skipped": False, "completed": True})
    d = ob_client.get("/api/onboarding").json()
    assert d["completed"] is True and d["completed_at"]
    assert d["show"] is (not d["models_ready"])  # completed but models missing: shown again
    stored = json.loads((tmp_path / "data" / "onboarding.json").read_text())
    assert stored["version"] == 1 and set(stored) == {"version", "completed_at", "skipped",
                                                      "last_step"}


def test_models_listing_sizes_equal_specs(ob_client):
    from shotsieve.models.registry import GROUPS, MODEL_SPECS, group_names

    d = ob_client.get("/api/models").json()
    assert set(d["groups"]) == set(GROUPS) and d["download"] is None
    for g, info in d["groups"].items():
        assert info["bytes"] == sum(MODEL_SPECS[n].size for n in group_names(g))
        assert info["installed"] is False and info["files"] == len(group_names(g))


def _fake_models_download(monkeypatch, payloads=None, fail_first=False):
    """Stub `registry._download` with synthetic tiny models; records Range resumes."""
    import hashlib
    from dataclasses import replace

    from shotsieve.models import registry

    state = {"calls": 0, "ranges": []}
    for name in registry.group_names("core"):
        blob = (name.encode() * 3000)[:5000]
        monkeypatch.setitem(
            registry.MODEL_SPECS, name,
            replace(registry.MODEL_SPECS[name], size=len(blob),
                    sha256=hashlib.sha256(blob).hexdigest()))
        (payloads if payloads is not None else {})[name] = blob

    def fake(url, dest, progress=None):
        state["calls"] += 1
        part = dest.with_name(dest.name + ".part")
        if fail_first and state["calls"] == 1:
            part.write_bytes(b"partial")
            import urllib.error

            raise urllib.error.URLError("offline")
        name = next(n for n, s in registry.MODEL_SPECS.items() if s.url == url)
        blob = (name.encode() * 3000)[:5000]
        have = part.stat().st_size if part.exists() else 0
        if have:
            state["ranges"].append(have)
        for k in range(have, len(blob), 1000):
            part.write_bytes(blob[: k + 1000]) if k else part.write_bytes(blob[:1000])
            if progress:
                progress(min(k + 1000, len(blob)), len(blob))
            time.sleep(0.01)
        part.replace(dest)

    monkeypatch.setattr(registry, "_download", fake)
    return state


def _wait_job(client, jid, timeout=20):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/jobs/{jid}").json()
        if j["state"] in ("done", "error"):
            return j
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_models_download_lane_progress_and_retry(ob_client, monkeypatch):
    st = _fake_models_download(monkeypatch, fail_first=True)
    assert ob_client.post("/api/models/download", json={"groups": ["bogus"]}).status_code == 400
    jid = ob_client.post("/api/models/download", json={"groups": ["core"]}).json()["job_id"]
    j = _wait_job(ob_client, jid)
    assert j["state"] == "error" and "internet connection" in j["error"]
    # Retry resumes from the .part with a Range request
    jid = ob_client.post("/api/models/download", json={"groups": ["core"]}).json()["job_id"]
    seen = set()
    t0 = time.time()
    while time.time() - t0 < 20:
        j = ob_client.get(f"/api/jobs/{jid}").json()
        seen.add(j["done"])
        if j["state"] in ("done", "error"):
            break
        time.sleep(0.01)
    assert j["state"] == "done" and j["done"] == j["total"] > 0
    assert len(seen) >= 3  # progress was observable
    assert st["ranges"] and st["ranges"][0] == len(b"partial")
    assert ob_client.get("/api/health").json()["models_ready"] is True
    assert ob_client.get("/api/onboarding").json()["models_ready"] is True


def test_download_lane_is_separate_from_cull_lock(ob_client, monkeypatch, tmp_path):
    import threading

    from shotsieve.models import registry
    from shotsieve.server.app import Job

    gate = threading.Event()

    def slow(url, dest, progress=None):
        gate.wait(10)
        dest.write_bytes(b"x")

    monkeypatch.setattr(registry, "_download", slow)
    state = ob_client.app.state.shotsieve
    state.active = Job("busy", str(tmp_path), kind="cull", state="running")  # a cull is running
    r = ob_client.post("/api/models/download", json={"groups": ["ocr"]})
    assert r.status_code == 200  # the download lane does not wait for the CPU job lock
    assert ob_client.post("/api/models/download", json={"groups": ["ocr"]}).status_code == 409
    folder = tmp_path / "f"
    folder.mkdir()
    # a second cull is still refused by the cull lock
    monkeypatch.setattr("shotsieve.models.registry.verify_model", lambda *a, **k: True)
    assert ob_client.post("/api/cull", json={"folder": str(folder)}).status_code == 409
    gate.set()
    _wait_job(ob_client, r.json()["job_id"])


def test_fs_list_security_and_filters(ob_client, ob_env, tmp_path):
    home = ob_env
    (home / "Shoots" / "wedding").mkdir(parents=True)
    (home / "Shoots" / "wedding" / "a.jpg").write_bytes(b"x")
    (home / "Shoots" / "wedding" / "b.CR3").write_bytes(b"x")
    (home / "Shoots" / ".hidden").mkdir()
    (home / "Shoots" / "notes.txt").write_text("secret")
    (home / "Shoots" / "cat.lrcat").write_bytes(b"x")
    (home / "Shoots" / "cat.lrcat-wal").write_bytes(b"x")
    outside = tmp_path / "outside"
    outside.mkdir()
    (home / "Shoots" / "escape").symlink_to(outside)
    (home / "Shoots" / "filelink").symlink_to(home / "Shoots" / "notes.txt")
    g = ob_client.get
    d = g("/api/fs/list", params={"dir": str(home / "Shoots")}).json()
    names = [e["name"] for e in d["entries"]]
    assert names == ["wedding"] and d["entries"][0]["n_images"] == 2  # no files, no hidden
    assert d["parent"] == str(home.resolve())
    d = g("/api/fs/list", params={"dir": str(home / "Shoots"), "kind": "lrcat"}).json()
    assert sorted(e["name"] for e in d["entries"]) == ["cat.lrcat", "wedding"]  # only .lrcat files
    assert g("/api/fs/list").json()["dir"] == str(home.resolve())  # default: HOME
    for bad in ("../etc", str(home / ".." / "etc"), "relative/dir", "a\x00b",
                str(home / "Shoots" / "escape"), "/etc", "/",
                str(home / "Shoots" / "notes.txt"), str(home / "missing")):
        r = g("/api/fs/list", params={"dir": bad})
        assert r.status_code in (400, 403), (bad, r.status_code)
        assert g("/api/fs/scan", params={"folder": bad}).status_code in (400, 403), bad
    assert g("/api/fs/scan").status_code == 400  # scan needs an explicit folder
    # the symlink pointing outside is not even offered
    assert "escape" not in [e["name"] for e in g(
        "/api/fs/list", params={"dir": str(home / "Shoots")}).json()["entries"]]
    assert g("/api/fs/list", params={"dir": str(home), "kind": "bogus"}).status_code == 400


def test_fs_gets_leave_data_unchanged(ob_client, ob_env, tmp_path):
    (ob_env / "S").mkdir()
    before = _snapshot(tmp_path / "data") if (tmp_path / "data").exists() else {}
    for url, params in (("/api/fs/list", {"dir": str(ob_env / "S")}),
                        ("/api/fs/scan", {"folder": str(ob_env / "S")}),
                        ("/api/models", {}), ("/api/onboarding", {}), ("/api/recent", {})):
        assert ob_client.get(url, params=params).status_code == 200
    after = _snapshot(tmp_path / "data") if (tmp_path / "data").exists() else {}
    # data_dir() creates the folder itself but no GET may create or touch a file in it
    assert {k: v for k, v in after.items()} == before


def test_fs_scan_counts_formats(ob_client, ob_env):
    d = ob_env / "shoot"
    d.mkdir()
    for n in ("a.CR3", "b.NEF", "c.jpg", "e.heic", "f.dng", "g.png"):
        (d / n).write_bytes(b"x")
    (d / "clip.mp4").write_bytes(b"x")
    r = ob_client.get("/api/fs/scan", params={"folder": str(d)}).json()
    assert r["by_format"] == {"raw": 2, "jpeg": 1, "dng": 1, "heic": 1, "other": 1, "video": 1}
    assert r["n_images"] == 6 and r["culled"] is False


def test_fs_pick_stub_cancel_and_platform(ob_client, ob_env, monkeypatch, tmp_path):
    folder = ob_env / "chosen"
    folder.mkdir()
    stub = tmp_path / "picker.sh"
    stub.write_text(f'#!/bin/bash\necho "{folder}/"\n')
    stub.chmod(0o755)
    monkeypatch.setenv("SHOTSIEVE_PICKER", str(stub))
    r = ob_client.post("/api/fs/pick", json={"kind": "folder"}).json()
    assert r == {"path": str(folder.resolve()), "cancelled": False}
    stub.write_text("#!/bin/bash\nexit 1\n")  # user pressed Cancel
    assert ob_client.post("/api/fs/pick", json={"kind": "folder"}).json()["cancelled"] is True
    stub.write_text("#!/bin/bash\necho /etc\n")  # outside HOME: refused
    assert ob_client.post("/api/fs/pick", json={"kind": "folder"}).status_code == 403
    monkeypatch.delenv("SHOTSIEVE_PICKER")
    monkeypatch.setattr("sys.platform", "linux")
    assert ob_client.post("/api/fs/pick", json={"kind": "folder"}).status_code == 501


def test_recent_folders_recorded_on_cull(ob_client, ob_env, monkeypatch):
    from shotsieve.server import onboarding

    for i in range(12):
        (ob_env / f"s{i}").mkdir()
        onboarding.add_recent(ob_env / f"s{i}")
    names = [f["name"] for f in ob_client.get("/api/recent").json()["folders"]]
    assert names[0] == "s11" and len(names) == 10
    onboarding.add_recent(ob_env / "s5")
    assert ob_client.get("/api/recent").json()["folders"][0]["name"] == "s5"


@pytest.mark.usefixtures("models_ready")
def test_train_job_few_rows_message(
    ob_client, ob_env, monkeypatch, tmp_path, shoot_dir, make_fixtures
):
    real_models = Path(__file__).resolve().parents[1] / "models"
    monkeypatch.setenv("SHOTSIEVE_MODELS_DIR", str(real_models))
    shoot = tmp_path / "shoot_copy"
    shutil.copytree(shoot_dir, shoot)
    cat = tmp_path / "copy.lrcat"
    make_fixtures.build_lrcat(
        cat, [dict(root=str(shoot), base=p.stem, ext="jpg", rating=3, pick=0, label="")
              for p in sorted(shoot.glob("*.jpg"))]
    )  # a catalog of the private copy: the shared fixture folder stays unculled
    assert ob_client.post("/api/train", json={}).status_code == 400
    assert ob_client.post("/api/train", json={"catalog": "/nope.lrcat"}).status_code == 400
    assert ob_client.post("/api/train", json={"catalog": "/a/../b.lrcat"}).status_code == 400
    r = ob_client.post("/api/train", json={"catalog": str(cat)})
    assert r.status_code == 200, r.text
    j = _wait_job(ob_client, r.json()["job_id"], timeout=120)
    assert j["state"] == "done" and j["kind"] == "train", j
    assert j["result"]["trained"] is False and j["result"]["n_rows"] < 40
    assert "Not enough picks yet (need 40)" in j["result"]["message"]


@pytest.mark.usefixtures("models_ready")
def test_search_download_runs_during_cull_and_cull_lock_holds(
    ob_env, monkeypatch, tmp_path, shoot_dir
):
    """Acceptance 4: a model download and a cull run side by side; a second cull gets 409."""
    import hashlib
    from dataclasses import replace

    from shotsieve.models import registry

    real_models = Path(__file__).resolve().parents[1] / "models"
    private = tmp_path / "models_copy"  # NEVER point a synthetic-spec test at the real models/
    private.mkdir()
    for name in registry.group_names("core"):
        shutil.copy(real_models / registry.MODEL_SPECS[name].filename, private)
    monkeypatch.setenv("SHOTSIEVE_MODELS_DIR", str(private))
    for name in registry.group_names("search"):  # tiny synthetic search group, slow download
        blob = name.encode() * 50
        monkeypatch.setitem(registry.MODEL_SPECS, name, replace(
            registry.MODEL_SPECS[name], size=len(blob), sha256=hashlib.sha256(blob).hexdigest()))

    def slow(url, dest, progress=None):
        name = next(n for n, s in registry.MODEL_SPECS.items() if s.url == url)
        time.sleep(1.5)
        dest.write_bytes(name.encode() * 50)

    monkeypatch.setattr(registry, "_download", slow)
    shoot = tmp_path / "shoot"
    shutil.copytree(shoot_dir, shoot, ignore=shutil.ignore_patterns(".shotsieve"))
    with TestClient(create_app(Settings(search_index_auto=False))) as c:
        dl = c.post("/api/models/download", json={"groups": ["search"]}).json()["job_id"]
        cull = c.post("/api/cull", json={"folder": str(shoot)})
        assert cull.status_code == 200
        assert c.post("/api/cull", json={"folder": str(shoot)}).status_code == 409
        assert c.get(f"/api/jobs/{dl}").json()["state"] in ("running", "done")
        j_dl, j_cull = _wait_job(c, dl, 60), _wait_job(c, cull.json()["job_id"], 180)
        assert j_dl["state"] == "done" and j_cull["state"] == "done", (j_dl, j_cull)


def test_testserver_host_is_not_a_production_host(monkeypatch, tmp_path):
    """m5: only loopback names are served unless a test adds `testserver`."""
    from fastapi.testclient import TestClient

    from shotsieve.config import Settings
    from shotsieve.server import app as appmod

    assert appmod.PRODUCTION_HOSTS == ("127.0.0.1", "localhost")
    monkeypatch.delenv("SHOTSIEVE_EXTRA_HOSTS", raising=False)
    assert "testserver" not in appmod.allowed_hosts()
    app = appmod.create_app(Settings())
    assert TestClient(app, base_url="http://127.0.0.1").get("/api/health").status_code == 200
    assert TestClient(app, base_url="http://testserver").get("/api/health").status_code == 400
    r = TestClient(app, base_url="http://127.0.0.1").post(
        "/api/quit", headers={"Origin": "http://testserver"}
    )
    assert r.status_code == 403


def _cull(c, folder, files=None):
    body = {"folder": str(folder)} | ({"files": [str(f) for f in files]} if files else {})
    r = c.post("/api/cull", json=body)
    assert r.status_code == 200, r.text
    j = wait_job(c, r.json()["job_id"])
    assert j["state"] == "done", j
    return j


def _held_back_shoot(tmp_path, shoot_dir):
    """The fixture shoot minus two photos (they wait in `later/`)."""
    folder, later = tmp_path / "shoot", tmp_path / "later"
    shutil.copytree(shoot_dir, folder)
    later.mkdir()
    names = ("landscape.jpg", "black.jpg")
    for n in names:
        shutil.move(folder / n, later / n)
    return folder, [(later / n, folder / n) for n in names]


def test_file_list_cull_is_add_only(tmp_path, shoot_dir):
    folder, held = _held_back_shoot(tmp_path, shoot_dir)
    with TestClient(create_app(Settings(), default_folder=None)) as c:
        j0 = _cull(c, folder)
        d0, by0 = shoot_of(c, folder)
        n0 = len(by0)
        victim = sorted(by0)[0]
        r = c.post("/api/override", json={"image_id": by0[victim]["id"], "decision": "reject",
                                          "rating": 2, "folder": str(folder)})
        assert r.status_code == 200, r.text
        with Cache.for_folder(folder) as ca:
            norm_before = ca.get_meta("shoot_norm")
        for src, dst in held:
            shutil.move(src, dst)
        j = _cull(c, folder, [dst for _, dst in held])  # drop 2 files from this folder
        d, by = shoot_of(c, folder)
        assert len(by) == n0 + 2, (len(by), n0)
        assert set(by0) <= set(by) and {dst.name for _, dst in held} <= set(by)
        assert by[victim]["decision"] == "reject" and by[victim]["overridden"]
        assert by[victim]["id"] == by0[victim]["id"]  # stable ids: nothing was re-created
        assert j["result"].get("n_removed", 0) == 0 and j["result"]["n_images"] == n0 + 2
        assert j["result"].get("redecide") != "full", j["result"]
        with Cache.for_folder(folder) as ca:
            assert ca.overrides()  # the override row survived
            # norm not re-fitted
            assert ca.get_meta("shoot_norm")["lap_ref"] == norm_before["lap_ref"]
        assert j0["result"]["n_images"] == n0


def test_file_list_cull_into_never_culled_folder_then_full_cull(tmp_path, shoot_dir):
    folder = tmp_path / "shoot"
    shutil.copytree(shoot_dir, folder)
    total = len([p for p in folder.glob("*.jpg")])
    pick = [folder / "landscape.jpg", folder / "black.jpg", folder / "burst0_0.jpg"]
    with TestClient(create_app(Settings(), default_folder=None)) as c:
        _cull(c, folder, pick)
        _, by = shoot_of(c, folder)
        assert sorted(by) == sorted(p.name for p in pick)
        j = _cull(c, folder)  # a later full Cull adds the rest
        _, by = shoot_of(c, folder)
        assert len(by) == total and j["result"]["n_images"] == total
        assert j["result"].get("n_removed", 0) == 0


def test_cull_through_api_never_touches_the_real_data_dir(seeded):
    """Isolation: conftest's data dir takes the recents; the project's data/ stays untouched."""
    from shotsieve.config import data_dir, project_root

    real = project_root() / "data" / "recent.json"
    before = real.read_bytes() if real.exists() else None
    assert data_dir().resolve() != (project_root() / "data").resolve()
    settings = Settings(audit_enabled=False, search_index_auto=False)
    with TestClient(create_app(settings, seeded)) as c:
        r = c.post("/api/cull", json={"folder": str(seeded)})
        if r.status_code == 200:  # the job started and the folder was remembered
            for _ in range(300):
                job = c.get("/api/jobs/" + r.json()["job_id"]).json()
                if job["state"] in ("done", "error"):
                    break
                time.sleep(0.2)
            assert str(seeded) in json.loads((data_dir() / "recent.json").read_text())
    assert (real.read_bytes() if real.exists() else None) == before
