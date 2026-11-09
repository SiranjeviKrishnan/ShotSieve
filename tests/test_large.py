"""T21: 10k-frame shoots. Model-free (tests/large_synth.py): warm runs, incremental re-cull with
stable decisions, `save_decisions` upsert, fast read-only shoot payload."""

from __future__ import annotations

import bisect
import hashlib
import shutil
import sqlite3
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import large_synth as ls  # noqa: E402
from shotsieve.config import Settings  # noqa: E402
from shotsieve.pipeline import runner  # noqa: E402
from shotsieve.pipeline.cache import Cache  # noqa: E402

pytestmark = pytest.mark.large
N = 10_000
SETTINGS = Settings()
WINDOW = max(SETTINGS.dup_window_s, SETTINGS.flash_window_s)


def _no_extract(*a, **k):  # a warm run must never extract
    raise AssertionError("extraction was called on a warm run")


def _sig(res) -> dict[str, tuple]:
    return {
        r.master_path.name: (r.decision, tuple(r.reasons), round(r.score, 12), r.is_best_in_group)
        for r in res.images
    }


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    """A culled 10k shoot (frames, folder, first result)."""
    folder = tmp_path_factory.mktemp("big") / "shoot"
    frames = ls.build(folder, N, SETTINGS)
    t = time.perf_counter()
    res = runner.run_cull(folder, SETTINGS)
    return frames, folder, res, time.perf_counter() - t


@pytest.fixture
def work(base, tmp_path):
    frames, folder, res, _ = base
    dst = tmp_path / "shoot"
    shutil.copytree(folder, dst)
    return frames, dst, res


def _far_from(frames, names, changed_ts_by_serial, margin) -> set[str]:
    """Names of frames whose body has no changed frame within `margin` seconds."""
    sorted_ts = {k: sorted(v) for k, v in changed_ts_by_serial.items()}
    out = set()
    for f in frames:
        if f.name in names:
            continue
        ts = sorted_ts.get(f.serial, [])
        i = bisect.bisect_left(ts, f.ts)
        near = [abs(ts[j] - f.ts) for j in (i - 1, i) if 0 <= j < len(ts)]
        if not near or min(near) > margin:
            out.add(f.name)
    return out


def test_cold_then_warm_identical(base, monkeypatch):
    frames, folder, first, _ = base
    monkeypatch.setattr(runner, "_process_uncached", _no_extract)
    t = time.perf_counter()
    warm = runner.run_cull(folder, SETTINGS)
    wall = time.perf_counter() - t
    assert wall <= 15.0, wall
    assert warm.stats["n_processed"] == 0 == warm.stats["n_extracted"]
    assert warm.stats["n_new"] == 0 and warm.stats["n_removed"] == 0
    assert warm.stats["n_affected"] == 0
    assert warm.stats["redecide"] == "none"
    assert _sig(warm) == _sig(first)
    assert [(r.group_id, r.burst_id) for r in warm.images] == [
        (r.group_id, r.burst_id) for r in first.images
    ]


def test_add_500_stable_outside_affected(work, monkeypatch):
    frames, folder, first = work
    before = {r.master_path.name: (r.group_id, r.burst_id) for r in first.images}
    old = _sig(first)
    blk = ls.gen_frames(450, serial="C", seed=5, prefix="IMC", t0=ls.T0 + 7200, sharp=2.0)
    mid = ls.gen_frames(50, serial="A", seed=6, prefix="IMX", t0=ls.T0 + 1800, span=30 * 3600.0)
    new = blk + mid
    ls.write_files(folder, new)
    fake = ls.FakeProcess(new)
    monkeypatch.setattr(runner, "_process_uncached", fake)
    t = time.perf_counter()
    res = runner.run_cull(folder, SETTINGS)
    wall = time.perf_counter() - t
    assert len(fake.paths) == 500 == len(set(fake.paths))
    assert wall <= 20.0 + 5, wall  # extraction is fake: only the re-decide cost counts
    changed = {}
    for f in new:
        changed.setdefault(f.serial, []).append(f.ts)
    unaffected = _far_from(frames, {f.name for f in new}, changed, 2 * WINDOW)
    assert len(unaffected) > 0.5 * N
    now = _sig(res)
    bad = [n for n in unaffected if now[n] != old[n]]
    assert not bad, f"{len(bad)} of {len(unaffected)} unaffected frames changed, e.g. {bad[:3]}"
    cur = {r.master_path.name: (r.group_id, r.burst_id) for r in res.images}
    moved = [n for n in unaffected if cur[n][0] != before[n][0]]
    assert not moved, f"{len(moved)} unaffected frames got a new group id"
    s = res.stats
    assert s["n_new"] == 500 and s["n_extracted"] == 500 and s["n_removed"] == 0
    assert s["redecide"] == "incremental"
    assert s["n_drift"] == 0  # the affected set alone explains every change (no verbatim stamping)
    assert 500 <= s["n_affected"] < 0.3 * N
    with Cache.for_folder(folder) as c:
        stored = c.get_meta("shoot_norm")
    assert stored["n"] == N + 500 or stored["n"] == N  # frozen norm keeps its basis
    assert {"lap_ref", "tenen_ref", "face_ref", "af_ref", "n", "extractor"} <= set(stored)


def test_remove_200(work, monkeypatch):
    frames, folder, first = work
    old = _sig(first)
    gone = [f for f in frames[::50]][:200]
    for f in gone:
        (folder / f.name).unlink()
    monkeypatch.setattr(runner, "_process_uncached", _no_extract)
    t = time.perf_counter()
    res = runner.run_cull(folder, SETTINGS)
    assert time.perf_counter() - t <= 15.0
    s = res.stats
    assert s["n_extracted"] == 0 and s["n_removed"] == 200 and s["n_new"] == 0
    assert s["redecide"] == "incremental" and s["n_affected"] >= 1
    assert s["n_drift"] == 0
    with Cache.for_folder(folder) as c:
        rows = {Path(p).name for p in c.load_decisions()}
    assert not rows & {f.name for f in gone}
    assert len(rows) == N - 200
    changed = {}
    for f in gone:
        changed.setdefault(f.serial, []).append(f.ts)
    unaffected = _far_from(frames, {f.name for f in gone}, changed, 2 * WINDOW)
    now = _sig(res)
    assert len(unaffected) > 0.4 * N
    assert all(now[n] == old[n] for n in unaffected)


def test_over_20_percent_is_full(work, monkeypatch):
    frames, folder, first = work
    new = ls.gen_frames(2500, serial="C", seed=9, prefix="IMD", t0=ls.T0 + 9 * 86400, sharp=1.6)
    ls.write_files(folder, new)
    fake = ls.FakeProcess(new)
    monkeypatch.setattr(runner, "_process_uncached", fake)
    with Cache.for_folder(folder) as c:
        old_norm = c.get_meta("shoot_norm")
    res = runner.run_cull(folder, SETTINGS)
    assert len(fake.paths) == 2500
    assert res.stats["redecide"] == "full"
    assert res.stats["redecide_reason"]
    with Cache.for_folder(folder) as c:
        new_norm = c.get_meta("shoot_norm")
    assert new_norm["n"] == N + 2500
    assert new_norm["lap_ref"] != old_norm["lap_ref"]


def test_full_flag_and_settings_change_force_full(work, monkeypatch):
    frames, folder, first = work
    monkeypatch.setattr(runner, "_process_uncached", _no_extract)
    res = runner.run_cull(folder, SETTINGS, full=True)
    assert res.stats["redecide"] == "full" and "full" in res.stats["redecide_reason"]
    other = Settings.from_dict({**SETTINGS.to_dict(), "t_keep": 0.7})
    res = runner.run_cull(folder, other)
    assert res.stats["redecide"] == "full"


def test_save_decisions_is_upsert(tmp_path):
    def row(p, d="keep", score=0.5):
        return {"path": str(tmp_path / p), "decision": d, "reasons": [], "score": score,
                "group_id": 0, "burst_id": 0, "is_best": True, "audit": False,
                "features": {"a": 1.0}}  # fmt: skip

    with Cache.for_folder(tmp_path) as c:
        c.save_decisions([row("a"), row("b"), row("c")])
        con = c._con
        n0 = con.total_changes
        c.save_decisions([row("a"), row("b", "reject"), row("c")])
        assert con.total_changes - n0 == 1  # only the changed row is written
        n0 = con.total_changes
        c.save_decisions([row("a"), row("b", "reject")])
        assert con.total_changes - n0 == 1  # and the removed one deleted
        assert {Path(p).name for p in c.load_decisions()} == {"a", "b"}
        n0 = con.total_changes
        c.save_decisions([row("a"), row("b", "reject")])
        assert con.total_changes - n0 == 0


def test_lookup_many_chunks(base):
    frames, folder, _, _ = base
    stats = {}
    for f in frames[:300]:
        st = (folder / f.name).stat()
        stats[str(folder / f.name)] = (st.st_size, st.st_mtime_ns)
    ext = runner.extractor_fingerprint(SETTINGS)
    with Cache.for_folder(folder) as c:
        a = c.lookup_many(stats, ext, chunk=7)
        b = c.lookup_many(stats, ext)
    assert set(a) == set(b) == set(stats)


def test_shoot_payload_fast_readonly_gzip(base, monkeypatch):
    from fastapi.testclient import TestClient

    from shotsieve.server.app import build_shoot, create_app

    frames, folder, _, _ = base
    build_shoot(folder, SETTINGS)  # warm the OS cache
    t = time.perf_counter()
    shoot = build_shoot(folder, SETTINGS)
    assert time.perf_counter() - t <= 1.0
    assert shoot["stats"]["n_images"] == N

    changes: list[int] = []
    orig = Cache.close

    def spy(self):
        changes.append(self._con.total_changes)
        orig(self)

    monkeypatch.setattr(Cache, "close", spy)
    app = create_app(SETTINGS, folder)
    cl = TestClient(app)
    raw = cl.get("/api/shoot", headers={"Accept-Encoding": "identity"})
    assert raw.status_code == 200
    assert len(raw.content) <= 8_000_000
    gz = cl.get("/api/shoot", headers={"Accept-Encoding": "gzip"})
    assert gz.headers.get("content-encoding") == "gzip"
    assert int(gz.headers["content-length"]) <= 1_500_000
    assert changes and all(c == 0 for c in changes), changes  # GET writes nothing (M5)


def test_get_shoot_does_not_touch_db_bytes(base):
    from shotsieve.server.app import build_shoot

    _, folder, _, _ = base
    db = folder / ".shotsieve" / "cull.db"

    def digest():
        return hashlib.sha256(db.read_bytes()).hexdigest()

    build_shoot(folder, SETTINGS)
    d0 = digest()
    build_shoot(folder, SETTINGS)
    assert digest() == d0
    con = sqlite3.connect(db)
    assert con.execute("SELECT COUNT(*) FROM image_ids").fetchone()[0] >= N
    con.close()


# ---- pure helpers (pipeline/incremental.py) ----------------------------------------


def test_stable_ids_keep_old_ids_and_never_collide():
    from shotsieve.pipeline.incremental import stable_ids

    paths = ["a", "b", "c", "d", "e", "f"]
    old = {"a": 7, "b": 7, "c": 3, "d": 3}  # e, f are new
    new = stable_ids([0, 0, 1, 1, 2, 2], paths, old)
    assert new[:4] == [7, 7, 3, 3] and new[4] == new[5] and new[4] > 7
    # a merge keeps the id of the larger overlap, the smaller stored id is released
    merged = stable_ids([0, 0, 0, 0], ["a", "b", "c", "d"], {"a": 7, "b": 7, "c": 3, "d": 5})
    assert merged == [7, 7, 7, 7]
    # a split: the bigger part keeps the id, the other part gets a fresh one
    split = stable_ids([0, 0, 0, 1], ["a", "b", "c", "d"], {"a": 4, "b": 4, "c": 4, "d": 4})
    assert split[:3] == [4, 4, 4] and split[3] not in (4,) and split[3] > 4


def test_plan_reasons_and_cumulative_drift():
    from shotsieve.features.record import ShootNorm
    from shotsieve.pipeline import incremental as inc

    sig = {"settings": "a", "thresholds": "b", "model": "c", "offsets": "d"}
    rec = inc.norm_record(ShootNorm(1, 2, 3, 4), 1000, "ext", sig, 0)

    def plan(stored, **kw):
        base = dict(extractor="ext", sig=sig, n_prev=1000, n_new=0, n_removed=0, force_full=False)
        return inc.plan(stored, **{**base, **kw})

    assert plan(None, n_prev=0).mode == "full" and "first" in plan(None, n_prev=0).reason
    assert plan(rec).mode == "none" and plan(rec).norm == ShootNorm(1, 2, 3, 4)
    p = plan(rec, n_new=150, n_removed=50)  # exactly 20 %: reuse
    assert p.mode == "incremental" and p.drift == 200
    assert plan(rec, n_new=201).mode == "full"
    assert plan(rec, force_full=True).mode == "full"
    assert plan(rec, extractor="other").reason == "feature extractor changed"
    assert "thresholds" in plan(rec, sig={**sig, "thresholds": "x"}).reason
    # the norm is not refreshed by incremental runs, so the drift accumulates until it is
    again = inc.norm_record(ShootNorm(1, 2, 3, 4), 1000, "ext", sig, 150)
    assert plan(again, n_new=40).mode == "incremental"
    assert plan(again, n_new=60).mode == "full"


def test_extra_seeds_are_closed_over_their_groups():
    """m7: an overridden audit frame's group-mates (post- and pre-change) join the affected set."""
    from shotsieve.pipeline import incremental

    paths = ["a", "b", "c", "d"]
    kw = dict(
        paths=paths, group_ids=[1, 1, 2, 2], burst_ids=[1, 1, 2, 2], serials=["A"] * 4,
        ts=[0.0, 1.0, 1000.0, 1001.0], new=set(), removed={},
        prev_group={"a": 1, "b": 1, "c": 2, "d": 2}, prev_burst={"a": 1, "b": 1, "c": 2, "d": 2},
        window_s=5.0,
    )
    assert incremental.affected_set(**kw) == set()
    assert incremental.affected_set(**kw, extra={"c"}) == {"c", "d"}


def test_filter_bar_facets_10k_fast(base):
    """T23e: the camera / orientation / label side data of a 10k shoot is one cheap read (the
    filtering itself runs in the browser: tests/test_ui_filters.py times it); the lens back-fill
    never touches decisions."""
    from shotsieve.server import filters as fs

    _, folder, _, _ = base
    with Cache.for_folder(folder) as c:
        before = c.load_decisions()
    fs.facets_payload(folder)  # warm
    t = time.perf_counter()
    d = fs.facets_payload(folder)
    dt = time.perf_counter() - t
    assert d["n"] == N and len(d["rows"]) == N
    assert {v["label"] for v in d["vals"]["make"]} == {"Synth"} and d["pending"] == N
    assert dt < 1.0, f"facets took {dt:.2f}s"
    with Cache.for_folder(folder) as c:
        assert c.load_decisions() == before
