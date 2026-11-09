# ruff: noqa: E501
"""T23i: batch override / label with `ids`, and Move to Trash (+ undo). No file ever reaches the
user's Trash here: the server's backend is a directory (`DirTrash`) or a recording fake."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from shotsieve.config import Settings
from shotsieve.pipeline.cache import Cache
from shotsieve.server import trash as trash_mod
from shotsieve.server.app import create_app
from tests.test_api import _row, shoot_of, wait_job

_STASH: dict[str, dict] = {}


def fake_cull(folder, settings, progress=None, write_xmp=False, **kw):
    """Stands in for the incremental cull: drops the rows of files that are gone from disk and
    re-adds the stashed rows of files that came back (what `removed` / `new` do)."""
    with Cache.for_folder(folder) as c:
        rows = c.load_decisions()
        stash = _STASH.setdefault(str(folder), {})
        for p in [p for p in stash if os.path.exists(p)]:
            rows[p] = stash.pop(p)
        for p in [p for p in rows if not os.path.exists(p)]:
            stash[p] = rows.pop(p)
        c.save_decisions([{**r, "path": p} for p, r in rows.items()])
    return SimpleNamespace(stats={"counts": {}, "n_images": len(rows)})


@pytest.fixture()
def shoot(tmp_path, shoot_dir):
    f = tmp_path / "shoot"
    f.mkdir()
    names = ["burst0_0", "burst0_1", "burst0_2", "landscape", "black", "burst1_3"]
    for n in names:
        shutil.copy(shoot_dir / f"{n}.jpg", f / f"{n}.jpg")
    rows = [
        _row(f / "burst0_0.jpg", "keep", 0, True, score=0.9),
        _row(f / "burst0_1.jpg", "reject", 0, reasons=["duplicate"], score=0.4),
        _row(f / "burst0_2.jpg", "reject", 0, reasons=["duplicate_tie"], score=0.85),
        _row(f / "landscape.jpg", "keep", 1, True, score=0.8),
        _row(f / "black.jpg", "reject", 2, True, reasons=["accidental"], score=0.0),
        _row(f / "burst1_3.jpg", "review", 3, True, score=0.2),
    ]
    with Cache.for_folder(f) as c:
        c.save_decisions(rows)
    return f


@pytest.fixture()
def fake_trash(tmp_path):
    return tmp_path / "fake_trash"


@pytest.fixture()
def client(shoot, fake_trash, monkeypatch):
    monkeypatch.setattr("shotsieve.pipeline.runner.run_cull", fake_cull)
    app = create_app(Settings(), default_folder=shoot)
    app.state.shotsieve.trash_backend = trash_mod.DirTrash(fake_trash)
    with TestClient(app) as c:
        yield c


def db_labels(f):
    with Cache.for_folder(f) as c:
        return c.labels()


def ids_of(client, f, *names):
    _, by = shoot_of(client, f)
    return [by[n]["id"] for n in names]


def trash(client, f, ids, **kw):
    return client.post("/api/trash", json={"folder": str(f), "ids": ids, **kw})


# ---- batch override / label ---------------------------------------------------------------


def test_batch_override_sets_all_in_one_request(client, shoot):
    ids = ids_of(client, shoot, "burst0_0.jpg", "burst0_1.jpg", "black.jpg")
    r = client.post("/api/override", json={"folder": str(shoot), "ids": ids, "decision": "keep"})
    assert r.status_code == 200 and r.json()["n"] == 3
    _, by = shoot_of(client, shoot)
    assert all(by[n]["decision"] == "keep" and by[n]["overridden"] for n in ("burst0_0.jpg", "burst0_1.jpg", "black.jpg"))
    assert by["landscape.jpg"]["overridden"] is False
    # stars on the batch: rating only keeps the decision (rating >= 1 on a model decision -> keep)
    r = client.post("/api/override", json={"folder": str(shoot), "ids": ids, "rating": 4})
    assert [x["rating"] for x in r.json()["results"]] == [4, 4, 4]
    # back to the AI decision
    r = client.post("/api/override", json={"folder": str(shoot), "ids": ids, "decision": "model"})
    assert all(x["overridden"] is False for x in r.json()["results"])
    _, by = shoot_of(client, shoot)
    assert by["black.jpg"]["decision"] == "reject" and by["burst0_0.jpg"]["rating"] is None


def test_batch_override_validation(client, shoot):
    ok = ids_of(client, shoot, "black.jpg")[0]
    post = lambda **b: client.post("/api/override", json={"folder": str(shoot), "decision": "keep", **b})  # noqa: E731
    assert post().status_code == 422  # neither
    assert post(image_id=ok, ids=[ok]).status_code == 422  # both
    assert post(ids=[]).status_code == 422
    assert post(ids=list(range(1, 5002))).status_code == 422
    assert post(ids=[ok, 99999]).status_code == 404
    _, by = shoot_of(client, shoot)
    assert by["black.jpg"]["overridden"] is False  # an unknown id changed nothing
    assert post(image_id=ok).status_code == 200  # the single form still works


def test_batch_label_takes_many_ids(client, shoot):
    ids = ids_of(client, shoot, "burst0_0.jpg", "landscape.jpg")
    r = client.post("/api/label", json={"folder": str(shoot), "image_ids": ids, "label": "red"})
    assert r.status_code == 200 and r.json()["n"] == 2
    assert set(db_labels(shoot).values()) == {"red"} and len(db_labels(shoot)) == 2
    assert client.post("/api/label", json={"folder": str(shoot), "image_ids": list(range(5001)), "label": "red"}).status_code == 422


def test_batch_override_is_one_autoxmp_request(client, shoot):
    calls = []
    st = client.app.state.shotsieve
    st.autoxmp.request = lambda f, paths: calls.append(paths)
    ids = ids_of(client, shoot, "burst0_0.jpg", "burst0_1.jpg", "black.jpg")
    client.post("/api/override", json={"folder": str(shoot), "ids": ids, "decision": "reject"})
    assert len(calls) == 1 and len(calls[0]) == 3


# ---- trash ---------------------------------------------------------------------------------


def test_trash_moves_master_and_sidecar_and_rows_go(client, shoot, fake_trash):
    (shoot / "black.xmp").write_text("<x/>")
    ids = ids_of(client, shoot, "black.jpg")
    r = trash(client, shoot, ids)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["trashed"] == ids and d["n_files"] == 2 and d["token"] and d["job_id"]
    assert {x["kind"] for x in d["files"]} == {"master", "sidecar"}
    assert not (shoot / "black.jpg").exists() and not (shoot / "black.xmp").exists()
    assert sorted(p.name for p in fake_trash.rglob("*") if p.is_file()) == ["black.jpg", "black.xmp"]
    assert wait_job(client, d["job_id"])["state"] == "done"
    _, by = shoot_of(client, shoot)
    assert "black.jpg" not in by and len(by) == 5


def test_trash_pairs_raw_and_jpeg_checkbox(client, shoot, fake_trash, tmp_path):
    # a RAW master with a same-stem JPEG: the master row is the RAW, the JPEG is its pair
    raw = shoot / "pair1.dng"
    raw.write_bytes(b"RAW")
    shutil.copy(shoot / "landscape.jpg", shoot / "pair1.jpg")
    (shoot / "pair1.xmp").write_text("<x/>")
    rows = []
    with Cache.for_folder(shoot) as c:
        for p, r in c.load_decisions().items():
            rows.append({**r, "path": p})
        rows.append({**_row(raw, "keep", 9, True), "klass": ""})
        c.save_decisions(rows)
    iid = ids_of(client, shoot, "pair1.dng")[0]
    pv = client.post("/api/trash/preview", json={"folder": str(shoot), "ids": [iid]}).json()
    assert pv["n_pairs"] == 1 and pv["n_files"] == 3
    # unchecked: only the RAW goes; the JPEG and the shared sidecar stay
    r = trash(client, shoot, [iid], include_pairs=False).json()
    assert [x["kind"] for x in r["files"]] == ["master"]
    assert (shoot / "pair1.jpg").exists() and (shoot / "pair1.xmp").exists() and not raw.exists()
    # undo, then trash with the pair
    wait_job(client, r["job_id"])
    u = client.post("/api/trash/undo", json={"folder": str(shoot), "token": r["token"]}).json()
    assert u["restored"] == 1
    wait_job(client, u["job_id"])
    r = trash(client, shoot, [iid], include_pairs=True)
    assert r.status_code == 200, r.text
    assert sorted(x["kind"] for x in r.json()["files"]) == ["master", "pair", "sidecar"]
    assert not (shoot / "pair1.jpg").exists() and not (shoot / "pair1.xmp").exists()


def test_video_frames_are_not_trashed_unless_asked(client, shoot):
    clip = shoot / "clip.mp4"
    clip.write_bytes(b"VID")
    vp = f"{clip}#t=1.000"
    with Cache.for_folder(shoot) as c:
        rows = [{**r, "path": p} for p, r in c.load_decisions().items()]
        rows.append({**_row(vp, "keep", 20, True), "klass": ""})
        c.save_decisions(rows)
    iid = ids_of(client, shoot, "clip.mp4#t=1.000")[0]
    r = trash(client, shoot, [iid]).json()
    assert r["trashed"] == [] and r["token"] is None and "individually" in r["skipped"][0]["reason"]
    assert clip.exists()
    r = trash(client, shoot, [iid], include_videos=True).json()
    assert r["trashed"] == [iid] and not clip.exists()


def test_path_traversal_and_symlink_escape_rejected(client, shoot, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.jpg"
    secret.write_bytes(b"x")
    link = shoot / "link.jpg"
    link.symlink_to(secret)
    linkdir = shoot / "linkdir"
    linkdir.symlink_to(outside, target_is_directory=True)
    (shoot / ".shotsieve" / "thumbs").mkdir(parents=True, exist_ok=True)
    inner = shoot / ".shotsieve" / "thumbs" / "t.jpg"
    inner.write_bytes(b"x")
    with Cache.for_folder(shoot) as c:
        rows = [{**r, "path": p} for p, r in c.load_decisions().items()]
        for i, p in enumerate((link, linkdir / "secret.jpg", inner)):
            rows.append({**_row(p, "keep", 30 + i, True), "klass": ""})
        rows.append({**_row(secret, "keep", 40, True), "klass": ""})  # a row outside the folder
        c.save_decisions(rows)
    _, by = shoot_of(client, shoot)
    for rel in ("link.jpg", "linkdir/secret.jpg", ".shotsieve/thumbs/t.jpg"):
        iid = next(im["id"] for im in by.values() if im["rel"] == rel)
        r = trash(client, shoot, [iid])
        assert r.status_code == 200
        d = r.json()
        assert d["trashed"] == [] and d["failed"] and d["token"] is None, (rel, d)
    assert secret.exists() and link.is_symlink() and inner.exists()
    with Cache.for_folder(shoot) as c:
        out_id = c.ids_for([str(secret)])[str(secret)]
    assert trash(client, shoot, [out_id]).status_code == 404  # a row outside the shoot is not ours
    assert secret.exists()


def test_unknown_ids_and_other_shoot_rejected(client, shoot, tmp_path, shoot_dir):
    ids = ids_of(client, shoot, "black.jpg")
    other = tmp_path / "other"
    other.mkdir()
    shutil.copy(shoot_dir / "landscape.jpg", other / "landscape.jpg")
    with Cache.for_folder(other) as c:
        c.save_decisions([_row(other / "landscape.jpg", "keep", 0, True)])
    r = trash(client, shoot, [*ids, 424242])
    assert r.status_code == 404
    assert (shoot / "black.jpg").exists()  # nothing moved
    assert trash(client, shoot, []).status_code == 422
    assert trash(client, shoot, list(range(5001))).status_code == 422


def test_failed_trash_is_skipped_never_deleted(client, shoot):
    class Refusing:
        calls: list[str] = []

        def trash(self, path):
            Refusing.calls.append(str(path))
            raise OSError("the volume has no Trash")

        def restore(self, src, dst):
            raise AssertionError

    client.app.state.shotsieve.trash_backend = Refusing()
    ids = ids_of(client, shoot, "black.jpg", "landscape.jpg")
    d = trash(client, shoot, ids).json()
    assert d["trashed"] == [] and len(d["failed"]) == 2 and d["token"] is None and d["job_id"] is None
    assert (shoot / "black.jpg").exists() and (shoot / "landscape.jpg").exists()  # never a permanent delete
    assert "no Trash" in d["failed"][0]["error"]


def test_partial_failure_trashes_the_rest(client, shoot, fake_trash):
    real = trash_mod.DirTrash(fake_trash)

    class Half:
        def trash(self, path):
            if path.name == "black.jpg":
                raise OSError("busy")
            return real.trash(path)

        def restore(self, s, d):
            real.restore(s, d)

    client.app.state.shotsieve.trash_backend = Half()
    ids = ids_of(client, shoot, "black.jpg", "landscape.jpg")
    d = trash(client, shoot, ids).json()
    assert len(d["trashed"]) == 1 and [x["name"] for x in d["failed"]] == ["black.jpg"]
    assert (shoot / "black.jpg").exists() and not (shoot / "landscape.jpg").exists()


def test_busy_shoot_409_and_nothing_moved(client, shoot):
    st = client.app.state.shotsieve
    st.active = SimpleNamespace(state="running", folder=str(shoot))
    r = trash(client, shoot, ids_of(client, shoot, "black.jpg"))
    assert r.status_code == 409 and (shoot / "black.jpg").exists()
    st.active = None


def test_origin_policy_applies(client, shoot):
    r = client.post("/api/trash", json={"folder": str(shoot), "ids": [1]}, headers={"Origin": "http://evil.example"})
    assert r.status_code == 403
    r = client.post("/api/trash/undo", json={"folder": str(shoot), "token": "t"}, headers={"Origin": "http://evil.example"})
    assert r.status_code == 403


def test_undo_restores_files_and_rows_and_keeps_user_choices(client, shoot, fake_trash):
    ids = ids_of(client, shoot, "black.jpg", "landscape.jpg")
    client.post("/api/override", json={"folder": str(shoot), "ids": ids, "decision": "keep", "rating": 3})
    client.post("/api/label", json={"folder": str(shoot), "image_ids": ids, "label": "blue"})
    d = trash(client, shoot, ids).json()
    wait_job(client, d["job_id"])
    _, by = shoot_of(client, shoot)
    assert "black.jpg" not in by and "landscape.jpg" not in by
    # the rows come back through the same incremental path (here the stand-in re-adds them)
    u = client.post("/api/trash/undo", json={"folder": str(shoot), "token": d["token"]})
    assert u.status_code == 200 and u.json()["restored"] == 2 and u.json()["ids"] == d["trashed"]
    wait_job(client, u.json()["job_id"])
    d2, by = shoot_of(client, shoot)
    assert {by["black.jpg"]["id"], by["landscape.jpg"]["id"]} == set(ids)
    assert all(by[n]["rating"] == 3 and by[n]["overridden"] for n in ("black.jpg", "landscape.jpg"))
    assert set(db_labels(shoot).values()) == {"blue"} and len(db_labels(shoot)) == 2
    assert (shoot / "black.jpg").exists() and (shoot / "landscape.jpg").exists()
    assert not [p for p in fake_trash.rglob("*") if p.is_file()]
    # a second undo of the same token is refused
    assert client.post("/api/trash/undo", json={"folder": str(shoot), "token": d["token"]}).status_code == 404


def test_undo_does_not_overwrite_a_new_file(client, shoot):
    ids = ids_of(client, shoot, "black.jpg")
    d = trash(client, shoot, ids).json()
    wait_job(client, d["job_id"])
    (shoot / "black.jpg").write_bytes(b"new file")
    u = client.post("/api/trash/undo", json={"folder": str(shoot), "token": d["token"]}).json()
    assert u["restored"] == 0 and u["failed"] and (shoot / "black.jpg").read_bytes() == b"new file"
    assert client.post("/api/trash/undo", json={"folder": str(shoot), "token": d["token"]}).status_code == 200  # still undoable


def test_trash_and_undo_are_logged(client, shoot, caplog):
    import logging

    logging.getLogger("shotsieve").propagate = True
    with caplog.at_level(logging.INFO, logger="shotsieve"):
        d = trash(client, shoot, ids_of(client, shoot, "black.jpg")).json()
        wait_job(client, d["job_id"])
        client.post("/api/trash/undo", json={"folder": str(shoot), "token": d["token"]})
    text = caplog.text
    assert "trash: " in text and "black.jpg" in text and "trash undo:" in text


def test_dirtrash_backend_is_a_move_not_a_delete(tmp_path):
    b = trash_mod.DirTrash(tmp_path / "t")
    f = tmp_path / "a.jpg"
    f.write_bytes(b"x")
    dest = b.trash(f)
    assert not f.exists() and dest.read_bytes() == b"x"
    b.restore(dest, f)
    assert f.read_bytes() == b"x"


def test_source_has_no_permanent_delete():
    src = Path(trash_mod.__file__).read_text()
    for bad in ("unlink(", "os.remove(", "rmtree(", "os.rmdir(", ".rmdir("):
        assert bad not in src.replace("`rmtree`", "").replace("`shutil.rmtree`", ""), bad


def test_default_backend_is_the_mac_trash():
    assert isinstance(trash_mod.default_backend(), trash_mod.MacTrash) or os.environ.get("SHOTSIEVE_FAKE_TRASH")


def test_mac_trash_uses_nsfilemanager_trash_and_move(monkeypatch):
    """The real backend talks to NSFileManager (never touched here: a fake Foundation module)."""
    import sys
    import types

    calls = []

    class URL:
        def __init__(self, p):
            self.p = p

        def path(self):
            return "/Users/x/.Trash/" + Path(self.p).name

        @classmethod
        def fileURLWithPath_(cls, p):
            return cls(p)

    class FM:
        def trashItemAtURL_resultingItemURL_error_(self, url, res, err):
            calls.append(("trash", url.p))
            return True, URL(url.p), None

        def moveItemAtURL_toURL_error_(self, a, b, err):
            calls.append(("move", a.p, b.p))
            return True, None

    fake = types.SimpleNamespace(NSURL=URL, NSFileManager=types.SimpleNamespace(defaultManager=lambda: FM()))
    monkeypatch.setitem(sys.modules, "Foundation", fake)
    b = trash_mod.MacTrash()
    dest = b.trash(Path("/shoot/a.jpg"))
    b.restore(dest, Path("/shoot/a.jpg"))
    assert calls == [("trash", "/shoot/a.jpg"), ("move", "/Users/x/.Trash/a.jpg", "/shoot/a.jpg")]


def test_mac_trash_failure_raises_never_deletes(monkeypatch):
    import sys
    import types

    class URL:
        @classmethod
        def fileURLWithPath_(cls, p):
            return cls()

    class Err:
        def localizedDescription(self):
            return "no Trash on this volume"

    class FM:
        def trashItemAtURL_resultingItemURL_error_(self, *a):
            return False, None, Err()

    fake = types.SimpleNamespace(NSURL=URL, NSFileManager=types.SimpleNamespace(defaultManager=lambda: FM()))
    monkeypatch.setitem(sys.modules, "Foundation", fake)
    with pytest.raises(OSError, match="no Trash"):
        trash_mod.MacTrash().trash(Path("/shoot/a.jpg"))


# ---- the real incremental cull -------------------------------------------------------------


@pytest.mark.models
@pytest.mark.usefixtures("models_ready")
def test_real_cull_removal_keeps_unrelated_decisions_and_undo_restores(tmp_path, shoot_dir, fake_trash):
    folder = tmp_path / "real"
    shutil.copytree(shoot_dir, folder)
    app = create_app(Settings(), default_folder=None)
    app.state.shotsieve.trash_backend = trash_mod.DirTrash(fake_trash)
    with TestClient(app) as c:
        wait_job(c, c.post("/api/cull", json={"folder": str(folder)}).json()["job_id"])
        d0, by0 = shoot_of(c, folder)
        victim = next(im for im in d0["images"] if im["group_id"] == max(i["group_id"] for i in d0["images"]))
        group = victim["group_id"]
        solo = [im for im in d0["images"] if sum(1 for j in d0["images"] if j["group_id"] == im["group_id"]) == 1 and im["id"] != victim["id"]]
        c.post("/api/override", json={"folder": str(folder), "ids": [solo[0]["id"]], "decision": "keep", "rating": 2})
        t = trash(c, folder, [victim["id"]])
        assert t.status_code == 200, t.text
        j = wait_job(c, t.json()["job_id"])
        assert j["state"] == "done", j
        d1, by1 = shoot_of(c, folder)
        assert victim["name"] not in by1 and len(d1["images"]) == len(d0["images"]) - 1
        # photos outside the removed photo's group keep their decision, reasons and score
        for n, im in by1.items():
            if by0[n]["group_id"] != group:
                assert (im["model_decision"], im["reasons"], im["score"]) == (by0[n]["model_decision"], by0[n]["reasons"], by0[n]["score"]), n
        # undo: the file is back and the same incremental path re-adds it with the same id and choices
        u = c.post("/api/trash/undo", json={"folder": str(folder), "token": t.json()["token"]}).json()
        assert u["restored"] >= 1
        assert wait_job(c, u["job_id"])["state"] == "done"
        d2, by2 = shoot_of(c, folder)
        assert victim["name"] in by2 and by2[victim["name"]]["id"] == victim["id"]
        assert len(d2["images"]) == len(d0["images"])
        assert by2[solo[0]["name"]]["rating"] == 2 and by2[solo[0]["name"]]["overridden"]
