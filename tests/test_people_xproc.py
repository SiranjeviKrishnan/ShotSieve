"""r2: the consent check and the face writes are one cross-process transaction.

Workers are module-level functions run under `spawn`; ordering is forced with Events (the one
bounded `wait(timeout)` only proves that a revoke is *blocked*, it is never the mechanism)."""

from __future__ import annotations

import multiprocessing as mp

import pytest

from shotsieve.pipeline.cache import Cache

ROW = {"number": "23", "conf": 0.9, "x": 0, "y": 0, "w": 1, "h": 1, "face_idx": 0, "person_id": 1}


def _insert_person(c: Cache) -> None:
    cur = c._con.execute("INSERT INTO people (name, cover_face, n) VALUES ('x','a#0',1)")
    c._con.execute("INSERT INTO person_faces VALUES (?,?,?,?)", (cur.lastrowid, "a", 0, 1.0))
    c._commit()


def _write_faces(c: Cache, epoch: int) -> None:
    """What cluster()/finalize()/the indexer write, each behind the same consent gate."""
    with c.people_write(epoch) as ok, c._lock:
        if ok:
            _insert_person(c)
    c.put_jerseys("/nonexistent/a.jpg", "fp", [dict(ROW)], epoch=epoch)


def _worker_a_reads_epoch(folder, ready, go, done):
    with Cache.for_folder(folder) as c:
        epoch = c.people_epoch()
        ready.set()
        go.wait(30)
        _write_faces(c, epoch)
    done.set()


def _worker_a_inside_tx(folder, inside, b_started, b_done, finished):
    with Cache.for_folder(folder) as c:
        epoch = c.people_epoch()
        with c.people_write(epoch) as ok, c._lock:
            assert ok
            inside.set()
            b_started.wait(30)
            b_done.wait(1.0)  # new code: B is blocked on our transaction, so this times out
            _insert_person(c)
    finished.set()


def _worker_b_revoke(folder, started, done):
    with Cache.for_folder(folder) as c:
        started.set()
        c.revoke_people()
    done.set()


def _state(folder):
    with Cache.for_folder(folder) as c:
        q = c._con.execute
        return {
            "people": q("SELECT COUNT(*) FROM people").fetchone()[0],
            "faces": q("SELECT COUNT(*) FROM person_faces").fetchone()[0],
            "links": q(
                "SELECT COUNT(*) FROM jerseys WHERE face_idx IS NOT NULL OR person_id IS NOT NULL"
            ).fetchone()[0],
            "jerseys": q("SELECT COUNT(*) FROM jerseys").fetchone()[0],
            "consent": c.people_consent(True),
        }


@pytest.fixture()
def folder(tmp_path):
    d = tmp_path / "shoot"
    d.mkdir()
    with Cache.for_folder(d) as c:
        c.set_meta("people_consent", True)
    return d


def _run(*procs, timeout=60):
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout)
        assert p.exitcode == 0, f"worker failed: {p}"


def test_control_without_revoke_writes_rows(folder):
    ctx = mp.get_context("spawn")
    ready, go, done = ctx.Event(), ctx.Event(), ctx.Event()
    go.set()
    _run(ctx.Process(target=_worker_a_reads_epoch, args=(folder, ready, go, done)))
    s = _state(folder)
    assert s["people"] == 1 and s["faces"] == 1 and s["links"] == 1


def test_revoke_between_epoch_read_and_write(folder):
    ctx = mp.get_context("spawn")
    ready, go, done = ctx.Event(), ctx.Event(), ctx.Event()
    a = ctx.Process(target=_worker_a_reads_epoch, args=(folder, ready, go, done))
    a.start()
    assert ready.wait(60)
    b_started, b_done = ctx.Event(), ctx.Event()
    _run(ctx.Process(target=_worker_b_revoke, args=(folder, b_started, b_done)))
    go.set()
    a.join(60)
    assert a.exitcode == 0
    s = _state(folder)
    assert s["people"] == 0 and s["faces"] == 0 and s["links"] == 0
    assert s["jerseys"] == 1 and s["consent"] is False


def test_revoke_while_writer_transaction_is_open_waits_then_wipes(folder):
    ctx = mp.get_context("spawn")
    inside, b_started, b_done, finished = (ctx.Event() for _ in range(4))
    a = ctx.Process(target=_worker_a_inside_tx, args=(folder, inside, b_started, b_done, finished))
    a.start()
    assert inside.wait(60)
    b = ctx.Process(target=_worker_b_revoke, args=(folder, b_started, b_done))
    b.start()
    for p in (a, b):
        p.join(60)
        assert p.exitcode == 0
    assert finished.is_set() and b_done.is_set()
    s = _state(folder)
    assert s["people"] == 0 and s["faces"] == 0 and s["links"] == 0
    assert s["consent"] is False
