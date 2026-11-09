# ruff: noqa: E501
"""T23b: user-drawn subject boxes (`boxes` in the suggest request, persistence, CLI `--box`)."""

from __future__ import annotations

import json

import pytest

from shotsieve.cli import main
from shotsieve.pipeline.cache import Cache
from tests.test_crop_api import client, ids, shoot  # noqa: F401  (fixtures)

pytestmark = pytest.mark.usefixtures("models_ready")


def contains(rect, box, tol=2e-3):
    return rect[0] <= box[0] + tol and rect[1] <= box[1] + tol and rect[2] >= box[2] - tol and rect[3] >= box[3] - tol


def bx(b):
    return {"x0": b[0], "y0": b[1], "x1": b[2], "y1": b[3]}


def test_boxes_are_validated(client, ids):  # noqa: F811
    i = ids["two.jpg"]
    for bad in (
        [{"x0": 0.5, "y0": 0.1, "x1": 0.4, "y1": 0.5}],  # x0 >= x1
        [{"x0": 0.1, "y0": 0.5, "x1": 0.4, "y1": 0.5}],  # y0 >= y1
        [{"x0": -0.1, "y0": 0.1, "x1": 0.4, "y1": 0.5}],  # outside [0, 1]
        [{"x0": 0.1, "y0": 0.1, "x1": 1.2, "y1": 0.5}],
        [bx((0.1 + k * 0.1, 0.1, 0.15 + k * 0.1, 0.3)) for k in range(7)],  # more than 6
    ):
        r = client.post(f"/api/crop/{i}/suggest", json={"ratio": "4:5", "boxes": bad})
        assert r.status_code == 422, bad
        assert client.post(f"/api/crop/{i}/boxes", json={"boxes": bad}).status_code == 422
    r = client.post(f"/api/crop/{i}/suggest", json={"ratio": "4:5", "boxes": [{"x0": 0.1, "y0": 0.1}]})
    assert r.status_code == 422  # malformed
    six = [bx((0.02 + k * 0.15, 0.1, 0.1 + k * 0.15, 0.3)) for k in range(6)]
    assert client.post(f"/api/crop/{i}/suggest", json={"ratio": "free", "boxes": six}).status_code == 200


def test_a_user_box_is_kept_inside_the_top_crop_like_a_selected_subject(client, ids):  # noqa: F811
    i = ids["two.jpg"]
    for box in ((0.80, 0.10, 0.95, 0.40), (0.03, 0.55, 0.20, 0.92), (0.45, 0.30, 0.55, 0.60)):
        for ratio in ("4:5", "1:1", "16:9"):
            d = client.post(f"/api/crop/{i}/suggest", json={"ratio": ratio, "boxes": [bx(box)], "n": 3}).json()
            assert d["auto"] is False and any(s["kind"] == "user" for s in d["selected"])
            top = d["candidates"][0]
            assert contains(top["rect"], box) or "subject_may_be_cut" in top["warnings"] or d["infeasible"], (box, ratio, top)
            assert contains(top["rect"], box), (box, ratio, top)  # these boxes all fit: expect real containment


def test_two_far_apart_boxes_are_flagged_infeasible_not_silently_cut(client, ids):  # noqa: F811
    i = ids["two.jpg"]
    d = client.post(f"/api/crop/{i}/suggest", json={
        "ratio": "1:1", "boxes": [bx((0.02, 0.2, 0.08, 0.5)), bx((0.92, 0.2, 0.98, 0.5))]}).json()
    assert d["infeasible"] is True


def test_boxes_persist_in_the_cache_append_only_and_come_back_with_the_crop_state(client, shoot, ids):  # noqa: F811
    i = ids["two.jpg"]
    assert client.get(f"/api/crop/{i}/boxes").json() == {"boxes": []}
    b1 = [0.1, 0.2, 0.3, 0.6]
    assert client.post(f"/api/crop/{i}/boxes", json={"boxes": [bx(b1)]}).json() == {"boxes": [b1]}
    b2 = [0.6, 0.1, 0.9, 0.5]
    client.post(f"/api/crop/{i}/boxes", json={"boxes": [bx(b1), bx(b2)]})
    assert client.get(f"/api/crop/{i}").json()["user_boxes"] == [b1, b2]  # a reload sees them
    assert client.get(f"/api/crop/{ids['b0.jpg']}").json()["user_boxes"] == []  # per image
    client.post(f"/api/crop/{i}/boxes", json={"boxes": []})  # cleared = a new version, empty
    assert client.get(f"/api/crop/{i}/boxes").json()["boxes"] == []
    with Cache.for_folder(shoot) as c:  # append-only: every version is still there
        with c._lock:
            rows = c._con.execute("SELECT version, boxes_json FROM crop_user_boxes WHERE image_id=? ORDER BY version", (i,)).fetchall()
    assert [r[0] for r in rows] == [1, 2, 3] and json.loads(rows[1][1]) == [b1, b2]
    assert client.post("/api/crop/999999/boxes", json={"boxes": []}).status_code == 404


def test_a_box_survives_a_server_restart(shoot, ids, tmp_path, monkeypatch):  # noqa: F811
    from fastapi.testclient import TestClient

    from shotsieve.config import Settings
    from shotsieve.server.app import create_app

    monkeypatch.setenv("SHOTSIEVE_DATA_DIR", str(tmp_path / "data"))
    i = ids["two.jpg"]
    with TestClient(create_app(Settings(), default_folder=shoot)) as c1:
        c1.post(f"/api/crop/{i}/boxes", json={"boxes": [bx((0.1, 0.2, 0.3, 0.6))]})
    with TestClient(create_app(Settings(), default_folder=shoot)) as c2:
        assert c2.get(f"/api/crop/{i}").json()["user_boxes"] == [[0.1, 0.2, 0.3, 0.6]]


def test_cli_box_option(shoot, ids, tmp_path, monkeypatch, capsys):  # noqa: F811
    monkeypatch.setenv("SHOTSIEVE_DATA_DIR", str(tmp_path / "data"))
    i = str(ids["two.jpg"])
    box = (0.80, 0.10, 0.95, 0.40)
    assert main(["crop", "suggest", str(shoot), "--ratio", "4:5", "--ids", i, "--box", ",".join(map(str, box)), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    top = out["results"][0]["candidates"][0]
    assert any(s["kind"] == "user" for s in out["results"][0]["selected"])
    assert contains(top["rect"], box)
    assert main(["crop", "suggest", str(shoot), "--ratio", "4:5", "--ids", i, "--box", "0.5,0.1,0.4,0.5"]) == 2  # x0 >= x1
    with pytest.raises(SystemExit) as ex:  # argparse rejects a malformed box
        main(["crop", "suggest", str(shoot), "--ratio", "4:5", "--ids", i, "--box", "0.5,0.1"])
    assert ex.value.code == 2
    with Cache.for_folder(shoot) as c:  # suggest writes nothing, not even the boxes
        assert c.user_boxes(int(i)) == [] and c.crops_current_all() == {}
