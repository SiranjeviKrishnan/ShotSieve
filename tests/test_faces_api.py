# ruff: noqa: E501
"""T23d API: faces per image (boxes in the display frame, scores, decision link), the people list
for many clusters, rename / merge / detach / ignore propagation, and POST /api/lightroom/open
(subprocess mocked; consent sheet logic, path validation, the 500 cap)."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import piexif
import pytest
from fastapi.testclient import TestClient

from shotsieve.config import Settings
from shotsieve.features.record import RawFeatures
from shotsieve.features.subject import FaceObs
from shotsieve.features.vector import SCHEMA_VERSION, FeatureVector
from shotsieve.io.preview import apply_orientation
from shotsieve.pipeline import people as ppl
from shotsieve.pipeline.cache import Cache, _pack
from shotsieve.server import lightroom as lr
from shotsieve.server.app import create_app
from tests.test_api import _row, shoot_of

NAN = float("nan")


def put_raw(c: Cache, path: Path, w: int, h: int, faces: list[FaceObs]) -> None:
    raw = RawFeatures(width=w, height=h, faces=faces)
    c._con.execute(
        "INSERT OR REPLACE INTO features VALUES (?,?,?)", (c._rel(str(path)), SCHEMA_VERSION, raw.to_json())
    )
    c._con.commit()


def put_emb(c: Cache, path: Path, n: int) -> None:
    arr = np.random.default_rng(0).normal(size=(n, 128)).astype(np.float32)
    c._con.execute(
        "INSERT OR REPLACE INTO embeddings VALUES (?,?,?,?,?)",
        (c._rel(str(path)), "face", 128, n, _pack(arr)),
    )
    c._con.commit()


def make_person(c: Cache, name, faces: list[tuple[Path, int]]) -> int:
    cur = c._con.execute(
        "INSERT INTO people (name, cover_face, n) VALUES (?,?,?)",
        (name, f"{c._rel(str(faces[0][0]))}#{faces[0][1]}", len(faces)),
    )
    pid = cur.lastrowid
    c._con.executemany(
        "INSERT INTO person_faces VALUES (?,?,?,?)",
        [(pid, c._rel(str(p)), i, 0.9 - 0.01 * k) for k, (p, i) in enumerate(faces)],
    )
    c._con.commit()
    c.set_meta("people_cluster_version", 99)  # fixtures model a current people table (no re-cluster on open)
    return pid


@pytest.fixture()
def shoot(tmp_path):
    """Upright 400x300 image with a red square (face 0), a 300x200 EXIF-6 image (rotated into the
    display frame), plain images for people, a video clip and a RAW-named file."""
    f = tmp_path / "shoot"
    f.mkdir()
    img = np.full((300, 400, 3), 60, np.uint8)
    img[60:140, 100:180] = (30, 30, 230)  # BGR red square: x 100..180, y 60..140
    cv2.imwrite(str(f / "upright.jpg"), img)
    # stored 300x200 with EXIF orientation 6 (display = 200x300 portrait); red square painted in
    # the DISPLAY frame at x 40..100, y 150..230, stored by inverting the rotation
    disp = np.full((300, 200, 3), 60, np.uint8)
    disp[150:230, 40:100] = (30, 30, 230)
    stored = np.rot90(disp, 1).copy()  # inverse of orientation 6 (a 90 degrees CW display rotation)
    cv2.imwrite(str(f / "rot6.jpg"), stored)
    ex = {"0th": {piexif.ImageIFD.Orientation: 6}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
    piexif.insert(piexif.dump(ex), str(f / "rot6.jpg"))
    for n in ("a", "b", "c", "d"):
        cv2.imwrite(str(f / f"{n}.jpg"), np.full((120, 160, 3), 90, np.uint8))
    (f / "clip.mp4").write_bytes(b"\0" * 32)
    rows = [
        _row(f / "upright.jpg", "reject", 0, True, reasons=["eyes_closed"]),
        _row(f / "rot6.jpg", "keep", 1, True),
        _row(f / "a.jpg", "keep", 2, True), _row(f / "b.jpg", "keep", 3, True),
        _row(f / "c.jpg", "reject", 4, True), _row(f / "d.jpg", "review", 5, True),
        _row(f"{f / 'clip.mp4'}#t=1.000", "keep", 6, True),
    ]
    with Cache.for_folder(f) as c:
        c.save_decisions(rows)
        c._con.execute(
            "UPDATE decisions SET features=? WHERE path=?",
            (json.dumps(FeatureVector(n_faces=2.0, sharp_global_norm=0.8, face_sharp_norm_main=0.6).to_dict()),
             "upright.jpg"),
        )
        c._con.commit()
        put_raw(c, f / "upright.jpg", 400, 300, [
            FaceObs((100, 60, 80, 80), 0.99, 100.0, 0.1, 0.1, 0.3, True, True),
            FaceObs((300, 200, 40, 40), 0.9, 40.0, 0.9, 0.2, 0.1, True, True),   # closed eyes, 2nd face
            FaceObs((10, 10, 8, 8), 0.8, NAN),                                  # small, not important
        ])
        put_emb(c, f / "upright.jpg", 2)
        put_raw(c, f / "rot6.jpg", 200, 300, [FaceObs((40, 150, 60, 80), 0.97, 80.0, 0.1, 0.1, 0.0, True, True)])
        put_emb(c, f / "rot6.jpg", 1)
        c.set_meta("people_consent", True)
    return f


@pytest.fixture()
def client(shoot):
    with TestClient(create_app(Settings(), default_folder=shoot)) as c:
        yield c


@pytest.fixture()
def ids(client, shoot):
    return {im["name"]: im["id"] for im in shoot_of(client, shoot)[0]["images"]}


# ---- faces per image ----------------------------------------------------------------------------


def test_faces_boxes_scores_and_the_face_that_drove_the_decision(client, ids):
    d = client.get(f"/api/image/{ids['upright.jpg']}/faces").json()
    f0, f1, f2 = d["faces"]
    assert f0["box"] == [0.25, 0.2, 0.45, pytest.approx(140 / 300, abs=1e-4)]
    assert f0["main"] and f0["important"] and f0["face_idx"] == 0 and f1["face_idx"] == 1
    assert f2["important"] is False and f2["face_idx"] is None
    assert f0["eyes_open"] == pytest.approx(0.9) and f1["eyes_open"] == pytest.approx(0.1)
    assert f0["smile"] == pytest.approx(0.3) and f0["sharp"] == pytest.approx(0.6)
    assert f1["sharp"] == pytest.approx(0.24)  # normalised like the main face (0.6 / 100 * 40)
    assert [f["drove"] for f in (f0, f1, f2)] == [False, True, False]  # reason eyes_closed: the closed face
    assert f1["drove_reason"] == "eyes_closed" and d["people_consent"] is True


def test_boxes_are_in_the_upright_display_frame_and_the_crop_shows_the_same_pixels(client, ids):
    iid = ids["rot6.jpg"]
    d = client.get(f"/api/image/{iid}/faces").json()
    b = d["faces"][0]["box"]
    assert b == pytest.approx([0.2, 0.5, 0.5, 230 / 300], abs=1e-4)
    p = cv2.imdecode(np.frombuffer(client.get(f"/api/preview/{iid}").content, np.uint8), cv2.IMREAD_COLOR)
    assert p.shape[:2] == (300, 200)  # the preview is the upright frame (portrait)
    x0, y0, x1, y1 = (round(b[0] * 200), round(b[1] * 300), round(b[2] * 200), round(b[3] * 300))
    patch = p[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0)  # BGR: the red square
    assert patch[2] > 180 and patch[0] < 90
    t = client.get(f"/api/image/{iid}/face/0?size=64")
    assert t.status_code == 200 and t.headers["content-type"] == "image/jpeg"
    img = cv2.imdecode(np.frombuffer(t.content, np.uint8), cv2.IMREAD_COLOR)
    assert max(img.shape[:2]) == 64
    centre = img[img.shape[0] // 3 : 2 * img.shape[0] // 3, img.shape[1] // 3 : 2 * img.shape[1] // 3]
    assert centre.reshape(-1, 3).mean(axis=0)[2] > 170  # red at the centre of the face thumbnail
    assert apply_orientation(np.zeros((200, 300, 3), np.uint8), 6).shape[:2] == (300, 200)


def test_face_thumb_is_cached_and_errors(client, shoot, ids):
    iid = ids["upright.jpg"]
    assert client.get(f"/api/image/{iid}/face/0").status_code == 200
    assert list((shoot / ".shotsieve" / "faces").glob("fc_*.jpg"))
    assert client.get(f"/api/image/{iid}/face/9").status_code == 404
    assert client.get("/api/image/99999/faces").status_code == 404
    assert client.get(f"/api/image/{ids['a.jpg']}/faces").json()["faces"] == []  # no features: none


def test_names_need_consent_boxes_do_not(client, shoot, ids):
    iid = ids["upright.jpg"]
    with Cache.for_folder(shoot) as c:
        pid = make_person(c, "Lena", [(shoot / "upright.jpg", 0), (shoot / "rot6.jpg", 0), (shoot / "a.jpg", 0)])
        c.set_meta("people_consent", False)
    d = client.get(f"/api/image/{iid}/faces").json()
    assert d["people_consent"] is False and d["faces"][0]["person_id"] is None and d["faces"][0]["name"] is None
    assert d["faces"][0]["box"][0] == 0.25
    assert client.get("/api/people").status_code == 403
    assert client.get(f"/api/people/{pid}/face").status_code == 403
    with Cache.for_folder(shoot) as c:
        c.set_meta("people_consent", True)
    assert client.get(f"/api/image/{iid}/faces").json()["faces"][0]["name"] == "Lena"


# ---- people: list, rename, merge, detach, ignore ------------------------------------------------


def test_people_list_for_many_clusters_with_counts(client, shoot, ids):
    up, rot = shoot / "upright.jpg", shoot / "rot6.jpg"
    with Cache.for_folder(shoot) as c:
        big = make_person(c, "Lena", [(up, 0), (rot, 0), (shoot / "a.jpg", 0)])
        c._con.execute("INSERT OR REPLACE INTO overrides VALUES (?,?,?,?,?,?)", ("a.jpg", "reject", 1.0, 0.0, "", None))
        for i in range(320):
            make_person(c, None if i % 2 else f"P{i:03d}", [(shoot / f"{'bcd'[i % 3]}.jpg", 100 + i)])
        c._con.commit()
    d = client.get("/api/people").json()
    assert len(d["people"]) == 321
    first = d["people"][0]
    assert first["id"] == big and first["name"] == "Lena" and first["n"] == 3
    assert first["photos"] == 3 and first["keeps"] == 1  # upright is rejected, a.jpg overridden to reject, rot6 keep
    assert all(p["photos"] == 1 for p in d["people"][1:])
    assert [p["n"] for p in d["people"]] == sorted((p["n"] for p in d["people"]), reverse=True)
    assert d["ignored"] == 0 and {"cover_id", "cover_idx"} <= set(first)
    # a person with no readable cover face still answers 404 instead of a broken image
    assert client.get(f"/api/people/{d['people'][5]['id']}/face").status_code == 404


def test_rename_and_merge_propagate_everywhere(client, shoot, ids):
    up, rot = shoot / "upright.jpg", shoot / "rot6.jpg"
    with Cache.for_folder(shoot) as c:
        a = make_person(c, None, [(up, 0), (shoot / "a.jpg", 0)])
        b = make_person(c, "Anna", [(rot, 0), (shoot / "b.jpg", 0)])
    iid = ids["upright.jpg"]
    assert client.get(f"/api/image/{iid}/faces").json()["faces"][0]["name"] is None
    r = client.post(f"/api/people/{a}", json={"name": "  Ann  "}).json()
    assert {p["id"]: p["name"] for p in r["people"]}[a] == "Ann"
    assert client.get(f"/api/image/{iid}/faces").json()["faces"][0]["name"] == "Ann"
    assert client.get("/api/people/chips").json()["chips"][str(iid)] == ["Ann"]
    hits = client.get("/api/search", params={"q": 'person:"ann"'}).json()["hits"]
    assert {h["id"] for h in hits} == {iid, ids["a.jpg"]}
    assert client.get(f"/api/people/{a}/images").json()["ids"] == sorted([iid, ids["a.jpg"]])
    assert "Ann" in client.get("/api/people/names").json()["names"]
    m = client.post(f"/api/people/{a}", json={"merge_into": b}).json()
    assert [p["id"] for p in m["people"]] == [b] and m["people"][0]["n"] == 4 and m["people"][0]["photos"] == 4
    assert client.get(f"/api/image/{iid}/faces").json()["faces"][0]["person_id"] == b
    assert client.get(f"/api/image/{iid}/faces").json()["faces"][0]["name"] == "Anna"
    hits = client.get("/api/search", params={"q": 'person:"Anna"'}).json()["hits"]
    assert {h["id"] for h in hits} == {iid, ids["a.jpg"], ids["rot6.jpg"], ids["b.jpg"]}
    assert client.post(f"/api/people/{b}", json={"merge_into": 4242}).status_code == 404
    assert client.post(f"/api/people/{b}", json={}).status_code == 422


def test_not_this_person_detaches_one_face_and_survives_a_recluster(tmp_path, shoot_dir):
    from tests.test_people import NAMES, seed_faces

    d = tmp_path / "recluster"
    d.mkdir()
    for n in NAMES:
        shutil.copy(shoot_dir / f"{n}.jpg", d / f"{n}.jpg")
    seed_faces(d, per_person=(4, 4))
    with Cache.for_folder(d) as c:
        c.set_meta("people_consent", True)
        assert ppl.cluster(c) == 2
        info = ppl.list_people(c)["people"]
        biggest = info[0]
        with c._lock:
            path, idx = c._con.execute("SELECT path, face_idx FROM person_faces WHERE person_id=? LIMIT 1", (biggest["id"],)).fetchone()
        left = ppl.detach_face(c, c._abs(path), idx)
        assert left == biggest["id"]
        after = {p["id"]: p for p in ppl.list_people(c)["people"]}
        assert after[biggest["id"]]["n"] == 3
        assert ppl.cluster(c) == 2  # re-cluster: the face stays out of every person
        with c._lock:
            assert c._con.execute("SELECT COUNT(*) FROM person_faces WHERE path=? AND face_idx=?", (path, idx)).fetchone()[0] == 0
            assert sorted(r[0] for r in c._con.execute("SELECT n FROM people")) == [3, 4]
        assert ppl.list_people(c)["ignored"] == 1
        c.revoke_people()  # a real wipe forgets the exclusions too
        with c._lock:
            assert c._con.execute("SELECT COUNT(*) FROM face_excluded").fetchone()[0] == 0


def test_detach_assign_ignore_restore_over_http(client, shoot, ids):
    up = shoot / "upright.jpg"
    iid = ids["upright.jpg"]
    with Cache.for_folder(shoot) as c:
        pid = make_person(c, "Lena", [(up, 0), (shoot / "a.jpg", 0), (shoot / "b.jpg", 0)])
    r = client.post(f"/api/image/{iid}/faces/0/detach", json={}).json()
    assert r["was"] == pid and r["people"][0]["n"] == 2
    assert client.get(f"/api/image/{iid}/faces").json()["faces"][0]["person_id"] is None
    assert client.post(f"/api/image/{iid}/faces/2/detach", json={}).status_code == 422  # too small: no identity
    new = client.post(f"/api/image/{iid}/faces/1/assign", json={"name": "Max"}).json()
    assert new["people"][-1]["name"] == "Max" or any(p["name"] == "Max" for p in new["people"])
    f1 = client.get(f"/api/image/{iid}/faces").json()["faces"][1]
    assert f1["name"] == "Max"
    back = client.post(f"/api/image/{iid}/faces/0/assign", json={"person_id": pid}).json()
    assert {p["id"]: p["n"] for p in back["people"]}[pid] == 3
    assert client.post(f"/api/image/{iid}/faces/0/assign", json={"person_id": 777}).status_code == 404
    ig = client.post(f"/api/people/{pid}", json={"ignore": True}).json()
    assert pid not in [p["id"] for p in ig["people"]] and ig["ignored"] == 3
    assert client.get(f"/api/image/{iid}/faces").json()["faces"][0]["person_id"] is None
    assert len(ig["ignored_faces"]) == 3
    un = client.post("/api/people/unignore", json={"faces": ig["ignored_faces"][:2]}).json()  # Undo: just these faces
    assert un["restored"] == 2 and un["ignored"] == 1
    rs = client.post("/api/people/restore_ignored", json={}).json()
    assert rs["restored"] == 1 and rs["ignored"] == 0


# ---- Open in Lightroom -------------------------------------------------------------------------


class Spy:
    def __init__(self, rc=0, err=""):
        self.calls, self.rc, self.err = [], rc, err

    def __call__(self, args, **kw):
        self.calls.append((list(args), kw))
        return subprocess.CompletedProcess(args, self.rc, "", self.err)


@pytest.fixture()
def spy(monkeypatch):
    s = Spy()
    monkeypatch.setattr(lr.subprocess, "run", s)
    monkeypatch.setattr(lr, "find_app", lambda bid: f"/Applications/{bid}.app")
    return s


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_open_needs_no_consent_sheet_and_runs_open_with_an_argument_list(client, shoot, ids, spy):
    iid = ids["a.jpg"]
    r = client.post("/api/lightroom/open", json={"ids": [iid]}).json()  # T23g: no question any more
    assert r["ok"] and r["app"] == "cloud" and r["bundle_id"] == "com.adobe.lightroomCC"
    assert "needs_consent" not in r and r["xmp"] is None
    assert (shoot / "a.xmp").exists()  # automatic XMP is on by default: flushed before open
    args, kw = spy.calls[-1]
    assert args == ["open", "-b", "com.adobe.lightroomCC", str(shoot / "a.jpg")]
    assert kw.get("shell") is None and kw["capture_output"]
    # the per-shoot switch: off means nothing is written
    assert client.post("/api/xmp/auto", json={"on": False}).json()["auto"] is False
    r = client.post("/api/lightroom/open", json={"ids": [ids["b.jpg"]]}).json()
    assert r["ok"] and not (shoot / "b.xmp").exists()


def test_write_and_open_puts_the_state_into_the_sidecar_and_never_touches_the_original(client, shoot, ids, spy):
    iid = ids["c.jpg"]
    before = sha(shoot / "c.jpg")
    r = client.post("/api/lightroom/open", json={"ids": [iid], "xmp": "write"}).json()
    assert r["ok"] and r["xmp"]["decisions"] == 1 and r["xmp"]["failed"] == 0
    xmp = (shoot / "c.xmp").read_text()
    assert "shotsieve:Decision" in xmp and "reject" in xmp
    assert sha(shoot / "c.jpg") == before
    # user wins: a foreign rating the photographer set is kept by the same writer
    sp = shoot / "d.xmp"
    sp.write_text(
        '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmp:Rating="4"/></rdf:RDF></x:xmpmeta>'
    )
    client.post("/api/lightroom/open", json={"ids": [ids["d.jpg"]]})
    assert 'Rating="4"' in sp.read_text() or ">4<" in sp.read_text()


def test_write_and_open_also_writes_the_current_crop_for_a_raw(client, shoot, tmp_path, gen_dir, spy):
    shutil.copy(gen_dir / "raw" / "pentax_kx.dng", shoot / "frame.pef")
    with Cache.for_folder(shoot) as c:
        rows = [{**row, "path": p} for p, row in c.load_decisions().items()]
        rows = [{**r, "path": str(shoot / r["path"]) if not Path(r["path"]).is_absolute() else r["path"]} for r in rows]
        c.save_decisions([*rows, _row(shoot / "frame.pef", "keep", 9, True)])
    iid = next(im["id"] for im in client.get("/api/shoot").json()["images"] if im["name"] == "frame.pef")
    info = client.get(f"/api/crop/{iid}").json()
    w, h = info["full"]["w"], info["full"]["h"]
    side = h / w
    rect = [(1 - side) / 2, 0.0, (1 + side) / 2, 1.0]
    acc = client.post(f"/api/crop/{iid}/accept", json={"rect": rect, "ratio": "1:1", "write": "none"}).json()
    assert acc["write"]["status"] == "none" and not (shoot / "frame.xmp").exists()
    r = client.post("/api/lightroom/open", json={"ids": [iid], "xmp": "write"}).json()
    assert r["ok"] and r["xmp"]["crops"]["written"] == 1, r
    assert "HasCrop" in (shoot / "frame.xmp").read_text()
    assert spy.calls[-1][0][-1] == str(shoot / "frame.pef")  # the RAW master, not a preview


def test_video_frames_open_the_video_file_and_duplicates_collapse(client, shoot, ids, spy):
    vid = ids["clip.mp4#t=1.000"]
    r = client.post("/api/lightroom/open", json={"ids": [vid, vid], "xmp": "skip"}).json()
    assert r["n"] == 1 and spy.calls[-1][0][-1] == str(shoot / "clip.mp4")


def test_app_choice_preference_and_installation(client, shoot, ids, spy, monkeypatch):
    iid = ids["a.jpg"]
    client.post("/api/lightroom/open", json={"ids": [iid], "xmp": "skip"})
    apps = client.get("/api/lightroom/apps").json()
    assert [a["id"] for a in apps["apps"]] == ["cloud", "classic"] and apps["preferred"] == "cloud"
    assert apps["classic_sync"] is True
    r = client.post("/api/lightroom/open", json={"ids": [iid], "app": "classic"}).json()
    assert r["app"] == "classic" and spy.calls[-1][0][2] == "com.adobe.LightroomClassicCC7"
    assert client.get("/api/lightroom/apps").json()["preferred"] == "classic"  # the app used last
    assert client.post("/api/lightroom/open", json={"ids": [iid]}).json()["app"] == "classic"
    monkeypatch.setattr(lr, "find_app", lambda bid: "/Applications/x.app" if bid == "com.adobe.lightroomCC" else None)
    a2 = client.get("/api/lightroom/apps").json()
    assert [a["id"] for a in a2["apps"]] == ["cloud"] and a2["classic_sync"] is False
    assert client.post("/api/lightroom/open", json={"ids": [iid], "app": "classic"}).status_code == 409
    assert client.post("/api/lightroom/open", json={"ids": [iid]}).json()["app"] == "cloud"  # falls back
    with Cache.for_folder(shoot) as c:
        c.set_meta("lr_catalog", "/x/My.lrcat")
    assert client.get("/api/lightroom/apps").json()["classic_sync"] is True  # a catalog was used here
    monkeypatch.setattr(lr, "find_app", lambda bid: None)
    assert client.get("/api/lightroom/apps").json()["apps"] == []
    assert client.post("/api/lightroom/open", json={"ids": [iid]}).status_code == 409


def test_validation_path_rules_and_failures(client, shoot, ids, spy, tmp_path):
    for body, code in (
        ({"ids": []}, 422), ({"ids": [1], "app": "word"}, 422), ({"ids": [1], "xmp": "maybe"}, 422),
        ({"ids": [987654]}, 404), ({"ids": [ids["a.jpg"]], "folder": "relative/dir"}, 400),
        ({"ids": [ids["a.jpg"]], "folder": str(shoot / "..")}, 400), ({"ids": ["x"]}, 422),
    ):
        assert client.post("/api/lightroom/open", json=body).status_code == code, body
    # an id whose row points outside the shoot folder is refused, even though the file exists
    outside = tmp_path / "secret.jpg"
    outside.write_bytes(b"x")
    with Cache.for_folder(shoot) as c:
        oid = c.ids_for([str(outside)])[str(outside)]
    assert client.post("/api/lightroom/open", json={"ids": [oid], "xmp": "skip"}).status_code == 404
    # a vanished file: 404 (nothing left to open) or reported as missing next to the good ones
    (shoot / "b.jpg").unlink()
    assert client.post("/api/lightroom/open", json={"ids": [ids["b.jpg"]], "xmp": "skip"}).status_code == 404
    r = client.post("/api/lightroom/open", json={"ids": [ids["b.jpg"], ids["a.jpg"]], "xmp": "skip"}).json()
    assert r["missing"] == [ids["b.jpg"]] and r["n"] == 1
    assert not spy.calls or all("secret.jpg" not in " ".join(c[0]) for c in spy.calls)
    spy.rc, spy.err = 1, "The application cannot be opened"
    e = client.post("/api/lightroom/open", json={"ids": [ids["a.jpg"]]})
    assert e.status_code == 502 and "cannot be opened" in e.json()["detail"]


def test_foreign_origin_and_host_are_refused(client, ids, spy):
    r = client.post("/api/lightroom/open", json={"ids": [ids["a.jpg"], ], "xmp": "skip"}, headers={"Origin": "http://evil.example"})
    assert r.status_code == 403 and not spy.calls
    r = client.post("/api/lightroom/open", json={"ids": [ids["a.jpg"]]}, headers={"Host": "evil.example"})
    assert r.status_code == 400 and not spy.calls


def test_at_most_500_files_per_call(client, shoot, spy):
    rows = []
    for i in range(501):
        p = shoot / f"bulk{i:03d}.jpg"
        p.write_bytes(b"x")
        rows.append(_row(p, "keep", 100 + i, True))
    with Cache.for_folder(shoot) as c:
        old = [{**r, "path": str(shoot / r["path"]) if not Path(r["path"]).is_absolute() else r["path"]} for p, r in c.load_decisions().items() for r in [{**r, "path": p}]]
        c.save_decisions([*old, *rows])
        ids = c.ids_for([r["path"] for r in rows])
    all_ids = [ids[r["path"]] for r in rows]
    assert client.post("/api/lightroom/open", json={"ids": all_ids, "xmp": "skip"}).status_code == 422
    assert not spy.calls
    r = client.post("/api/lightroom/open", json={"ids": all_ids[:500], "xmp": "skip"}).json()
    assert r["ok"] and r["n"] == 500 and len(spy.calls[-1][0]) == 4 + 499


def test_unculled_folder_and_missing_folder(tmp_path, spy):
    d = tmp_path / "empty"
    d.mkdir()
    with TestClient(create_app(Settings(), default_folder=d)) as c:
        assert c.post("/api/lightroom/open", json={"ids": [1]}).status_code == 404
        assert c.get("/api/lightroom/apps").json()["xmp_choice"] is None
