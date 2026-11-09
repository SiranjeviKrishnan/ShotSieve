# ruff: noqa: E501
"""T23h: recognise every face (small, noisy, background) in the background; the cull's decision
faces and decisions are untouched. Floors are the measured held-out numbers of tests/faces_eval.py
(thresholds were picked on the train split only; see docs/TASKS.md T23h)."""

from __future__ import annotations

import shutil
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from shotsieve.config import Settings
from shotsieve.pipeline import facescan, people
from shotsieve.pipeline.cache import Cache
from shotsieve.pipeline.runner import run_cull
from shotsieve.server.app import create_app
from tests import faces_eval as fe

NO_AUDIT = Settings(audit_enabled=False)


def test_recall_by_size_and_false_positives_heldout():
    r = fe.evaluate("heldout")
    b, a = r["before"]["recall"], r["after"]["recall"]
    assert b["<24"][0] == 0.0 and b["24-48"][0] < 0.2  # today: small faces are dropped
    assert a["<24"][0] >= 0.8 and a["24-48"][0] >= 0.8 and a[">48"][0] >= 0.88
    assert r["after"]["fp_clean_neg"] <= 0.35  # faceless photos: at most ~1 box per 3 images
    assert r["after"]["fp_synth"] <= 1.2  # the painted / statue faces of the scenes count here
    assert r["t_after"] < 1.0  # background job, still bounded


def test_small_face_fixture_is_found_at_16_px_and_embedded():
    from shotsieve.models.face_detect import get_rec_face_detector
    from shotsieve.models.face_embed import get_face_embedder

    sc = fe.synth_scenes(7, n_faces=4)
    det, emb = get_rec_face_detector(), get_face_embedder()
    boxes, vecs = facescan.scan_image(sc["img"], [], det, emb)
    assert len(boxes) == len(vecs) >= 2
    assert vecs.shape[1] == 128 and np.allclose(np.linalg.norm(vecs, axis=1), 1, atol=1e-3)
    hits, _ = fe.match(
        facescan.scan(sc["img"], det), sc["faces"]
    )
    assert sum(hits) >= 2


def _shoot(tmp_path):
    d = tmp_path / "shoot"
    d.mkdir()
    sc = fe.synth_scenes(7, n_faces=5, size=(1800, 1200))
    cv2.imwrite(str(d / "wide.jpg"), cv2.cvtColor(sc["img"], cv2.COLOR_RGB2BGR))
    shutil.copy(fe.FIX / "crop2" / "real" / "group_03.jpg", d / "crowd.jpg")
    shutil.copy(fe.FIX / "crop2" / "real" / "portrait_01.jpg", d / "portrait.jpg")
    return d


def _tables(c):
    with c._lock:
        return (
            c._con.execute("SELECT path, decision, reasons, features FROM decisions ORDER BY path").fetchall(),
            c._con.execute("SELECT path, json FROM features ORDER BY path").fetchall(),
            c._con.execute("SELECT path, n, data FROM embeddings WHERE kind='face' ORDER BY path").fetchall(),
        )


def test_rescan_adds_recognition_faces_and_never_touches_decisions(tmp_path):
    d = _shoot(tmp_path)
    run_cull(d, NO_AUDIT)
    with Cache.for_folder(d) as c:
        c.set_meta("people_consent", True)
        before = _tables(c)
        n0 = sum(len(v) for v in c.face_embeddings().values())
        assert facescan.needs_rescan(c)
    sc = facescan.FaceScanner(d, Settings(people_consent=True))
    sc.run()
    assert sc.error is None and sc.done == 3
    with Cache.for_folder(d) as c:
        assert _tables(c) == before  # decision faces, features, decisions: byte-identical
        assert not facescan.needs_rescan(c)
        n1 = sum(len(v) for v in c.face_embeddings().values())
        assert n1 > n0  # the crowd and the wide scene gained faces
        info = people.list_people(c)
        assert sum(p["n"] for p in info["people"]) + info["unsorted"] == n1
        # a new detection version rescans again, and a changed face count invalidates a row
        old = facescan.FACE_SCAN_VERSION
    facescan.FACE_SCAN_VERSION = old + 1
    try:
        with Cache.for_folder(d) as c:
            assert facescan.needs_rescan(c)
    finally:
        facescan.FACE_SCAN_VERSION = old


def test_opening_a_shoot_rescans_in_the_background_and_keeps_names(tmp_path, monkeypatch):
    monkeypatch.setenv("SHOTSIEVE_DATA_DIR", str(tmp_path / "data"))
    d = _shoot(tmp_path)
    run_cull(d, NO_AUDIT)
    with Cache.for_folder(d) as c:
        c.set_meta("people_consent", True)
        people.cluster(c)
        top = people.list_people(c)["people"][0]
        people.rename(c, top["id"], "Ada")
        n_before = sum(p["n"] for p in people.list_people(c)["people"])
    client = TestClient(create_app(Settings(people_consent=True)))
    assert client.get("/api/shoot", params={"folder": str(d)}).status_code == 200
    for _ in range(300):
        st = client.get("/api/people/scan", params={"folder": str(d)}).json()
        if not st["running"] and not st["pending"]:
            break
        time.sleep(0.2)
    assert st["pending"] is False and st["error"] is None and st["faces"] > 0
    res = client.get("/api/people", params={"folder": str(d)}).json()
    assert sum(p["n"] for p in res["people"]) > n_before
    assert "Ada" in [p["name"] for p in res["people"]]
    pid = res["people"][0]["id"]
    assert client.get(f"/api/people/{pid}/face", params={"folder": str(d)}).status_code == 200


@pytest.mark.parametrize("pause", [True])
def test_scan_waits_while_a_cull_runs(tmp_path, pause):
    from shotsieve.pipeline.indexer import cull_active

    d = _shoot(tmp_path)
    run_cull(d, NO_AUDIT)
    sc = facescan.FaceScanner(d, Settings())
    with cull_active():
        sc.start()
        time.sleep(0.6)
        assert sc.running and sc.done == 0
    sc._thread.join(60)
    assert sc.done == 3
