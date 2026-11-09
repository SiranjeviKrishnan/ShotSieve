# ruff: noqa: E501
"""T20: Lightroom Classic sync-back. The user's catalog is only ever read.

Model-free: shoots come from tests/large_synth.py (real files + real sidecars, synthetic
features), catalogs from scripts/make_fixtures.build_lrcat."""

from __future__ import annotations

import contextlib
import hashlib
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import large_synth as ls  # noqa: E402
from shotsieve.config import Settings  # noqa: E402
from shotsieve.io import lrcat  # noqa: E402
from shotsieve.io.lrcat import CatalogBusyError, CatalogError, read_catalog  # noqa: E402
from shotsieve.io.xmp import read_sidecar, write_decision  # noqa: E402
from shotsieve.pipeline import lrsync  # noqa: E402
from shotsieve.pipeline.cache import Cache  # noqa: E402
from shotsieve.pipeline.runner import run_cull  # noqa: E402

S = Settings()
from shotsieve.config import data_dir  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%dT%H:%M:%S")


def snap(cat: Path) -> dict[str, tuple[str, int] | None]:
    """sha256 + mtime_ns of the catalog and its side files (None: absent)."""
    out: dict[str, tuple[str, int] | None] = {}
    for suffix in ("", "-wal", "-shm", "-journal"):
        f = cat.with_name(cat.name + suffix)
        out[suffix] = (
            (hashlib.sha256(f.read_bytes()).hexdigest(), f.stat().st_mtime_ns) if f.exists() else None
        )
    return out


class Shoot:
    """A culled model-free shoot with sidecars written (`--write-xmp`)."""

    def __init__(self, folder: Path, n: int = 300) -> None:
        self.folder = folder
        self.frames = ls.build(folder, n, S)
        self.res = run_cull(folder, S, write_xmp=True)
        self.ts = {f.name: f.ts for f in self.frames}
        self.by_dec: dict[str, list] = {"keep": [], "reject": [], "review": []}
        for r in sorted(self.res.images, key=lambda r: r.master_path.name):
            self.by_dec[r.model_decision].append(r)

    def entries(self, root: Path | None = None, **extra) -> list[dict]:
        out = []
        for r in self.res.images:
            d = r.model_decision
            out.append({
                "root": str(root or self.folder), "rel": "", "base": r.master_path.stem,
                "ext": r.master_path.suffix.lstrip("."), "pick": 0,
                "rating": getattr(S, f"rating_{d}"), "label": getattr(S, f"label_{d}"),
                "touch": 1_700_000_000.0, "capture": iso(self.ts[r.master_path.name]),
                "original": r.master_path.name, **extra,
            })  # fmt: skip
        return out

    def catalog(self, path: Path, **kw) -> Path:
        import make_fixtures

        make_fixtures.build_lrcat(path, self.entries(), **kw)
        return path

    def p(self, r) -> str:
        return str(r.master_path)


@pytest.fixture(scope="session")
def mf(make_fixtures):
    return make_fixtures


@pytest.fixture
def shoot(tmp_path) -> Shoot:
    return Shoot(tmp_path / "shoot")


def edit_12(shoot: Shoot, cat: Path, mf, touch: float) -> dict[str, str]:
    """The 12 Lightroom edits of acceptance 2; returns {path: expected override decision}."""
    k, r = shoot.by_dec["keep"], shoot.by_dec["reject"]
    assert len(k) >= 5 and len(r) >= 7, (len(k), len(r))
    want: dict[str, str] = {}
    for x in r[:4]:  # pick=1 on rejects
        mf.lr_edit(cat, shoot.p(x), pick=1, touch=touch)
        want[shoot.p(x)] = "keep"
    for x in k[:3]:  # pick=-1 on keeps
        mf.lr_edit(cat, shoot.p(x), pick=-1, touch=touch)
        want[shoot.p(x)] = "reject"
    for x in r[4:6]:  # 2 stars added
        mf.lr_edit(cat, shoot.p(x), rating=2, touch=touch)
        want[shoot.p(x)] = "keep"
    mf.lr_edit(cat, shoot.p(k[3]), rating=0, touch=touch)  # a star removed from a keep
    want[shoot.p(k[3])] = "reject"
    mf.lr_edit(cat, shoot.p(k[4]), label="Red", touch=touch)
    want[shoot.p(k[4])] = "reject"
    mf.lr_edit(cat, shoot.p(r[6]), label="Green", touch=touch)
    want[shoot.p(r[6])] = "keep"
    assert len(want) == 12
    return want


def lr_overrides(folder: Path) -> dict[str, dict]:
    with Cache.for_folder(folder) as c:
        return {p: o for p, o in c.overrides().items() if o["note"] == "lrcat"}


# ---- mapping table ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rating, pick, label, written, want",
    [
        (None, -1, None, None, ("reject", None, "pick")),
        (3, 1, None, {"rating": 3, "label": "Green"}, ("keep", 3, "pick")),
        (2, 0, "Green", {"rating": 0, "label": "Red"}, ("keep", 2, "rating")),
        (0, 0, "", {"rating": 3, "label": "Green"}, ("reject", None, "downgrade")),
        (3, 0, "Red", {"rating": 3, "label": "Green"}, ("reject", None, "label")),
        (0, 0, "Green", {"rating": 0, "label": "Red"}, ("keep", None, "label")),
        (3, 0, "Green", {"rating": 3, "label": "Green"}, None),  # equal to what we wrote
        (0, 0, "Red", {"rating": 0, "label": "Red"}, None),
        (0, 0, "Yellow", {"rating": 0, "label": "Yellow"}, None),
        (0, 0, "Blue", {"rating": 0, "label": "Red"}, None),  # unmapped label: ignored
        (4, 0, None, None, ("keep", 4, "rating")),  # no baseline: a rating is the user's
        (None, 0, None, None, None),
    ],
)
def test_user_signal_mapping(rating, pick, label, written, want):
    assert lrsync.user_signal(rating, pick, label, written, S) == want


def test_label_map_is_configurable():
    s = Settings(lr_label_map={"Blue": "keep"})
    assert lrsync.user_signal(0, 0, "Blue", {"rating": 0, "label": "Red"}, s)[0] == "keep"
    assert lrsync.user_signal(0, 0, "Green", {"rating": 0, "label": "Red"}, s) is None


# ---- safety: the user's catalog is never written ----------------------------------------------------


def _hold_exclusive(cat: Path):
    """A child process that holds BEGIN EXCLUSIVE until stdin gets a line."""
    code = (
        "import sqlite3, sys\n"
        f"c = sqlite3.connect({str(cat)!r}, isolation_level=None)\n"
        "c.execute('BEGIN EXCLUSIVE')\n"
        "print('ready', flush=True)\n"
        "sys.stdin.readline()\n"
    )
    p = subprocess.Popen([sys.executable, "-c", code], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, text=True)  # fmt: skip
    assert p.stdout.readline().strip() == "ready"
    return p


@pytest.fixture
def spy_connect(monkeypatch):
    calls: list[str] = []
    orig = sqlite3.connect

    def spy(database, *a, **k):
        calls.append(str(database))
        return orig(database, *a, **k)

    monkeypatch.setattr(sqlite3, "connect", spy)
    return calls


def _assert_ro_only(calls: list[str], cat: Path) -> None:
    """n1: SQLite never opens the user's catalog at all (every read is on a private copy)."""
    mine = [c for c in calls if str(cat.parent) in c]  # the user's path (not the private copy)
    assert not mine, mine


@pytest.mark.parametrize(
    "case", ["plain", "wal", "wal_closed", "locked", "readonly_wal", "readonly_plain"]
)
def test_catalog_is_never_written(shoot, mf, tmp_path, spy_connect, case):
    cat = tmp_path / "lr" / "Catalog.lrcat"
    cat.parent.mkdir()
    shoot.catalog(cat, wal=case in ("wal", "readonly_wal"))
    touch = time.time()
    holder = None
    if case in ("plain", "locked", "readonly_plain", "wal_closed"):
        want = edit_12(shoot, cat, mf, touch)
    else:  # WAL catalogs: the edits themselves must be applied through the live WAL
        want = {}
    if case == "wal":
        # a change that exists only in the un-checkpointed -wal: apply it on top of the snapshot
        con = sqlite3.connect(cat)  # (this is the TEST's own catalog, before the sync)
        con.execute("PRAGMA wal_autocheckpoint=0")
        con.execute("UPDATE Adobe_images SET pick=1 WHERE id_local=1")
        con.commit()
        for sfx in ("-wal", "-shm"):  # keep the live side files: snapshot while connected
            assert cat.with_name(cat.name + sfx).exists()
        import shutil

        for sfx in ("", "-wal", "-shm"):
            shutil.copyfile(cat.with_name(cat.name + sfx), cat.with_name("snap" + sfx))
        con.close()
        for sfx in ("", "-wal", "-shm"):
            cat.with_name(cat.name + sfx).unlink(missing_ok=True)
            os.replace(cat.with_name("snap" + sfx), cat.with_name(cat.name + sfx))
        assert snap(cat)["-wal"] is not None
    if case == "wal_closed":
        # Lightroom's normal state after quit: WAL header, cleanly closed, no side files at all
        con = sqlite3.connect(cat)
        assert con.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        con.close()
        assert snap(cat)["-wal"] is None and snap(cat)["-shm"] is None
        assert cat.read_bytes()[18:20] == b"\x02\x02"
    if case == "locked":
        holder = _hold_exclusive(cat)
    if case.startswith("readonly"):
        for f in cat.parent.iterdir():
            f.chmod(0o444)
        cat.parent.chmod(0o555)
    before = snap(cat)
    listing = sorted(p.name for p in cat.parent.iterdir())
    spy_connect.clear()  # only the connections the sync itself makes
    try:
        res = lrsync.sync(shoot.folder, cat, S, dry_run=True, report=False)
    finally:
        if holder:
            holder.stdin.write("\n")
            holder.stdin.flush()
            holder.wait(10)
        cat.parent.chmod(0o755)
        for f in cat.parent.iterdir():
            f.chmod(0o644)
    assert snap(cat) == before  # sha256 + mtime + presence of .lrcat / -wal / -shm / -journal
    assert sorted(p.name for p in cat.parent.iterdir()) == listing  # no new file of any kind
    _assert_ro_only(spy_connect, cat)
    assert res.n_matched == len(shoot.res.images)
    if case == "locked":
        assert res.copied
        assert res.n_new == len(want) == 12  # committed data seen through the copy
    if case == "wal":
        assert res.copied and res.n_matched > 0  # rows exist only in the WAL
        assert res.n_new == 1  # the one un-checkpointed change was seen
    if case == "wal_closed":
        assert res.copied and res.n_new == 12  # WAL catalogs are always read from the copy
    if case in ("plain", "readonly_plain"):
        assert res.copied and res.n_new == 12  # n1: even a plain catalog is read from the copy
    assert not list((data_dir() / "tmp").glob("lrcat-*"))  # temp copies are always deleted


def test_corrupted_copy_is_refused(shoot, tmp_path, monkeypatch):
    cat = shoot.catalog(tmp_path / "Catalog.lrcat", wal=True)
    before = snap(cat)
    monkeypatch.setattr(lrcat, "COPY_WAIT_S", 0.0)
    orig = lrcat._private_copy

    def broken(src):
        tmp, dst = orig(src)
        dst.write_bytes(b"not a database at all" * 100)
        for sfx in ("-wal", "-shm"):
            dst.with_name(dst.name + sfx).unlink(missing_ok=True)
        return tmp, dst

    monkeypatch.setattr(lrcat, "_private_copy", broken)
    with pytest.raises(CatalogBusyError, match="quit Lightroom"):
        lrsync.sync(shoot.folder, cat, S, dry_run=True)
    assert snap(cat) == before
    assert not list((data_dir() / "tmp").glob("lrcat-*"))
    from shotsieve.cli import main

    assert main(["sync", str(shoot.folder), "--lrcat", str(cat), "--dry-run"]) == 4


@pytest.mark.parametrize(
    "bad", ["relative/Catalog.lrcat", "/nonexistent/x.lrcat", "/tmp/../tmp/x.lrcat", "/tmp/a\x00.lrcat"]
)
def test_bad_catalog_paths_refused(bad):
    with pytest.raises(CatalogError):
        lrcat.check_catalog_path(bad)


def test_not_a_lrcat_file_refused(tmp_path):
    f = tmp_path / "x.sqlite"
    f.write_bytes(b"")
    with pytest.raises(CatalogError):
        lrcat.check_catalog_path(f)


def test_reader_columns_and_prefix_filter(shoot, tmp_path):
    cat = shoot.catalog(tmp_path / "Catalog.lrcat", pad_mb=0)
    es = read_catalog(cat)
    e = es[0]
    assert e.touch_ts == 1_700_000_000.0 and e.capture and e.original_filename and e.id_local == 1
    sub = read_catalog(cat, folder_prefix=str(shoot.folder) + "/")
    assert len(sub) == len(es)
    assert read_catalog(cat, folder_prefix=str(tmp_path / "other") + "/") == []
    # '%' and '_' in a folder name are literal, not wildcards
    assert read_catalog(cat, folder_prefix=str(shoot.folder)[:-3] + "%/") == []


# ---- the round trip -----------------------------------------------------------------------------


def test_round_trip_12_edits_idempotent_and_recull_safe(shoot, mf, tmp_path):
    cat = shoot.catalog(tmp_path / "Catalog.lrcat")
    with Cache.for_folder(shoot.folder) as c:
        written = c.written()
    assert len(written) == len(shoot.res.images)  # `lr_written` filled by --write-xmp
    k0 = shoot.by_dec["keep"][0]
    assert written[shoot.p(k0)]["rating"] == S.rating_keep
    # untouched catalog: agreement is not a label
    res0 = lrsync.sync(shoot.folder, cat, S, report=False)
    assert res0.n_new == 0 and res0.n_matched == len(shoot.res.images)
    assert lr_overrides(shoot.folder) == {}
    want = edit_12(shoot, cat, mf, touch=time.time() + 10)
    res = lrsync.sync(shoot.folder, cat, S)
    ovr = lr_overrides(shoot.folder)
    assert set(ovr) == set(want)  # exactly those 12, none for untouched frames
    assert {p: o["decision"] for p, o in ovr.items()} == want
    assert all(o["propensity"] == 1.0 for o in ovr.values())
    assert res.n_new == 12 and res.n_changed == 0
    assert res.n_keeps == 7 and res.n_rejects == 5
    assert res.report and res.report["n_evaluated"] == len(shoot.res.images)
    with Cache.for_folder(shoot.folder) as c:
        assert c.get_meta("lr_report") == res.report
        assert c.get_meta("lr_sync_last")["n_new"] == 12
        assert c.get_meta("lr_catalog") == str(cat)
        rows0 = c._con.execute("SELECT * FROM overrides ORDER BY path").fetchall()
        sync0 = c._con.execute("SELECT * FROM lr_sync ORDER BY path").fetchall()
    # second sync: 0 writes
    res2 = lrsync.sync(shoot.folder, cat, S)
    assert res2.n_new == res2.n_changed == res2.n_cleared == 0
    with Cache.for_folder(shoot.folder) as c:
        assert c._con.execute("SELECT * FROM overrides ORDER BY path").fetchall() == rows0
        assert c._con.execute("SELECT * FROM lr_sync ORDER BY path").fetchall() == sync0
    # a re-cull, with and without XMP + force: overrides stay, sidecar rating / label stay
    side0 = {p: read_sidecar(p) for p in want}
    for kw in ({}, {"write_xmp": True}, {"write_xmp": True, "force_ratings": True}):
        again = run_cull(shoot.folder, S, **kw)
        assert again.stats["n_extracted"] == 0
        assert lr_overrides(shoot.folder).keys() == want.keys()
        assert {p: o["decision"] for p, o in lr_overrides(shoot.folder).items()} == want
        for r in again.images:
            if shoot.p(r) in want:
                assert r.decision == want[shoot.p(r)] and r.overridden
                assert r.model_decision != "" and r.model_decision in ("keep", "reject", "review")
        if kw.get("write_xmp"):
            assert again.stats["kept_lr"] == 12
            for p in want:
                d = read_sidecar(p)
                assert (d.rating, d.label) == (side0[p].rating, side0[p].label)  # LR owns them
    # stored decisions are the model's (refreshed), not the overrides
    with Cache.for_folder(shoot.folder) as c:
        dec = c.decision_summary()
    for r in shoot.res.images:
        assert dec[shoot.p(r)][0] == r.model_decision


def test_lr_written_is_backfilled_from_sidecars(shoot, mf, tmp_path):
    """A shoot culled before T20 has no `lr_written`: it is recovered from the sidecars once."""
    cat = shoot.catalog(tmp_path / "Catalog.lrcat")
    with Cache.for_folder(shoot.folder) as c:
        full = c.written()
        c._con.execute("DELETE FROM lr_written")
        c._con.commit()
        assert c.written() == {}
        assert lrsync.ensure_written(c) == len(full)
        back = c.written()
        c._con.execute("DELETE FROM lr_written")
        c._con.commit()
    assert {p: (v["rating"], v["label"]) for p, v in back.items()} == {
        p: (v["rating"], v["label"]) for p, v in full.items()
    }
    res = lrsync.sync(shoot.folder, cat, S, dry_run=True, report=False)  # in-memory backfill
    assert res.n_new == 0 and res.n_matched == len(full)
    with Cache.for_folder(shoot.folder) as c:
        assert c.written() == {}  # a dry run wrote nothing
    lrsync.sync(shoot.folder, cat, S, report=False)
    with Cache.for_folder(shoot.folder) as c:
        assert len(c.written()) == len(full)


def test_dry_run_changes_nothing(shoot, mf, tmp_path):
    cat = shoot.catalog(tmp_path / "Catalog.lrcat")
    edit_12(shoot, cat, mf, touch=time.time() + 10)
    db = shoot.folder / ".shotsieve" / "cull.db"
    with Cache.for_folder(shoot.folder) as c:
        before = (c.overrides(), c.lr_sync_rows(), c.get_meta("lr_sync_last"))
    res = lrsync.sync(shoot.folder, cat, S, dry_run=True)
    assert res.n_new == 12 and res.n_keeps + res.n_rejects == 12
    with Cache.for_folder(shoot.folder) as c:
        assert (c.overrides(), c.lr_sync_rows(), c.get_meta("lr_sync_last")) == before
    assert db.exists()


def test_revert_in_lightroom_clears_the_override(shoot, mf, tmp_path):
    cat = shoot.catalog(tmp_path / "Catalog.lrcat")
    x = shoot.by_dec["reject"][0]
    mf.lr_edit(cat, shoot.p(x), pick=1, touch=time.time() + 5)
    lrsync.sync(shoot.folder, cat, S)
    assert shoot.p(x) in lr_overrides(shoot.folder)
    mf.lr_edit(cat, shoot.p(x), pick=0, touch=time.time() + 50)  # back to what we wrote
    res = lrsync.sync(shoot.folder, cat, S)
    assert res.n_cleared == 1 and shoot.p(x) not in lr_overrides(shoot.folder)


def test_changing_it_again_in_lightroom_replaces_the_lrcat_override(shoot, mf, tmp_path):
    cat = shoot.catalog(tmp_path / "Catalog.lrcat")
    x = shoot.by_dec["reject"][0]
    mf.lr_edit(cat, shoot.p(x), pick=1, touch=time.time() + 5)
    lrsync.sync(shoot.folder, cat, S)
    mf.lr_edit(cat, shoot.p(x), pick=-1, touch=time.time() + 50)
    res = lrsync.sync(shoot.folder, cat, S)
    assert res.n_changed == 1
    assert lr_overrides(shoot.folder)[shoot.p(x)]["decision"] == "reject"


def test_conflicts_later_wins(shoot, mf, tmp_path):
    cat = shoot.catalog(tmp_path / "Catalog.lrcat")
    a, b = shoot.by_dec["reject"][0], shoot.by_dec["reject"][1]
    t = time.time()
    for x in (a, b):
        mf.lr_edit(cat, shoot.p(x), pick=1, touch=t)  # Lightroom: keep both
    with Cache.for_folder(shoot.folder) as c:
        c.set_override(shoot.p(a), "reject", 1.0, "ui", None, ts=t + 100)  # newer than LR
        c.set_override(shoot.p(b), "reject", 1.0, "ui", None, ts=t - 100)  # older than LR
    res = lrsync.sync(shoot.folder, cat, S)
    assert res.n_conflicts_ui_won == 1 and res.n_conflicts_lr_won == 1
    assert {c["name"]: c["winner"] for c in res.conflicts} == {
        a.master_path.name: "shotsieve", b.master_path.name: "lightroom"
    }
    with Cache.for_folder(shoot.folder) as c:
        o = c.overrides()
    assert (o[shoot.p(a)]["note"], o[shoot.p(a)]["decision"]) == ("ui", "reject")
    assert (o[shoot.p(b)]["note"], o[shoot.p(b)]["decision"]) == ("lrcat", "keep")


def test_agreeing_ui_override_is_not_a_conflict(shoot, mf, tmp_path):
    cat = shoot.catalog(tmp_path / "Catalog.lrcat")
    a = shoot.by_dec["reject"][0]
    mf.lr_edit(cat, shoot.p(a), pick=1, touch=time.time())
    with Cache.for_folder(shoot.folder) as c:
        c.set_override(shoot.p(a), "keep", 1.0, "ui")
    res = lrsync.sync(shoot.folder, cat, S)
    assert res.n_new == 0 and res.n_conflicts_ui_won == res.n_conflicts_lr_won == 0


def test_copy_import_fallback_matches_by_name_and_capture_time(shoot, mf, tmp_path):
    other = tmp_path / "imported_elsewhere"
    import make_fixtures

    es = shoot.entries(root=other)
    # two catalog images share name + capture second (two cards): ambiguous, never guessed
    dup = dict(es[0], root=str(tmp_path / "second_card"))
    cat = tmp_path / "Catalog.lrcat"
    make_fixtures.build_lrcat(cat, [*es, dup])
    mf.lr_edit(cat, str(other / f"{es[1]['base']}.jpg"), pick=1, touch=time.time())
    res = lrsync.sync(shoot.folder, cat, S, report=False)
    n = len(shoot.res.images)
    assert res.n_fallback >= 0.95 * n  # 100 % expected
    assert res.n_matched == n - 1 and res.n_unmatched == 1  # the ambiguous one stays unmatched
    assert res.n_new == 1


def test_video_frames_and_unmatched_are_counted_not_guessed(shoot, tmp_path):
    import make_fixtures

    es = shoot.entries()[:50]
    cat = tmp_path / "Catalog.lrcat"
    make_fixtures.build_lrcat(cat, es)
    res = lrsync.sync(shoot.folder, cat, S, report=False)
    assert res.n_matched == 50 and res.n_unmatched == len(shoot.res.images) - 50


# ---- training uses it -------------------------------------------------------------------------


def test_training_includes_synced_rows_and_ignores_own_output(shoot, mf, tmp_path):
    from shotsieve.personal.dataset import build_training_set, labels_from_overrides

    cat = shoot.catalog(tmp_path / "Catalog.lrcat")
    # a catalog equal to ShotSieve's output teaches nothing (regression: it used to teach 3-star keeps)
    assert len(build_training_set(lrcat=cat, settings=S)) == 0
    want = edit_12(shoot, cat, mf, touch=time.time() + 10)
    lrsync.sync(shoot.folder, cat, S)
    labs = labels_from_overrides(shoot.folder)
    assert len(labs) == 12 and {x.path for x in labs} == set(want)
    ts = build_training_set(override_folders=[shoot.folder], settings=S)
    assert len(ts) == 12
    # train --lrcat on the edited catalog: only the 12 changed frames carry user signal
    ts2 = build_training_set(lrcat=cat, settings=S)
    assert len(ts2) <= 12 + 0 and set(ts2.paths) <= set(want)


def test_synced_audit_items_count_for_recalibration(shoot, mf, tmp_path):
    from shotsieve.personal.calibrate import calibrate_shoot

    with Cache.for_folder(shoot.folder) as c:
        frozen = c.get_audit_sample() or {}
    items = sorted(frozen)
    assert items, "the synthetic shoot has an audit sample"
    cat = shoot.catalog(tmp_path / "Catalog.lrcat")
    for p in items[:3]:
        mf.lr_edit(cat, p, pick=1, touch=time.time() + 10)  # the photographer kept audited rejects
    for p in items[3:]:
        mf.lr_edit(cat, p, pick=-1, touch=time.time() + 10)
    lrsync.sync(shoot.folder, cat, S)
    result, _ = calibrate_shoot(shoot.folder, S, redecide=False)
    assert result.n_audited == len(items) and result.n_pending == 0
    assert result.n_kept == min(3, len(items))


# ---- the report -------------------------------------------------------------------------------------


def test_report_equals_evaluate_on_the_same_inputs(shoot, mf, tmp_path):
    from types import SimpleNamespace

    from shotsieve.pipeline.evaluate import evaluate, truth_from_catalog

    cat = shoot.catalog(tmp_path / "Catalog.lrcat")
    edit_12(shoot, cat, mf, touch=time.time() + 10)
    res = lrsync.sync(shoot.folder, cat, S)
    truth = truth_from_catalog(read_catalog(cat), S.min_rating)
    with Cache.for_folder(shoot.folder) as c:
        dec = c.decision_summary()
    images = [
        SimpleNamespace(path=Path(p), master_path=Path(p), decision=d, model_decision=d,
                        group_id=g, audit=a)
        for p, (d, g, a) in dec.items()
    ]  # fmt: skip
    ev = evaluate(images, truth)
    rep = res.report
    assert rep["group_gate_rate"] == ev.group_gate_rate
    assert rep["missed_keeper_rate"] == ev.missed_keeper_rate
    assert rep["review_fraction"] == ev.review_fraction
    assert rep["keepers_in_review"] == ev.keepers_in_review
    assert rep["n_evaluated"] == ev.n_evaluated
    assert rep["missed_keepers"] == ev.missed_keepers
    assert rep["missed_files"] == [os.path.relpath(p, shoot.folder) for p in ev.missed_paths]
    # the 4 flagged rejects are exactly the missed keepers (plus rated rejects)
    assert rep["missed_keepers"] >= 4


def test_hand_made_truth_known_missed_keepers(shoot, mf, tmp_path):
    cat = shoot.catalog(tmp_path / "Catalog.lrcat")
    missed = shoot.by_dec["reject"][:3]
    for x in missed:
        mf.lr_edit(cat, shoot.p(x), pick=1, touch=time.time())
    res = lrsync.sync(shoot.folder, cat, S, dry_run=True)
    # ShotSieve's own keeps are rated 3 in the catalog (user-kept in the final state)
    n_keeps = len(shoot.by_dec["keep"])
    assert res.report["n_kept_truth"] == n_keeps + 3
    assert res.report["missed_keepers"] == 3
    assert set(res.report["missed_files"]) == {x.master_path.name for x in missed}


# ---- XMP never fights Lightroom --------------------------------------------------------------------


def test_write_decision_keeps_lr_owned_rating_and_label(tmp_path):
    from shotsieve.io.xmp import WriteOutcome

    f = tmp_path / "a.jpg"
    f.write_bytes(b"x")
    write_decision(f, S, "keep", ["r"], 0.9)  # rating 3 / Green written
    out = WriteOutcome()
    write_decision(f, S, "reject", ["r"], 0.1,
                   override={"decision": "reject", "note": "lrcat", "rating": None},
                   force=True, outcome=out)  # fmt: skip
    d = read_sidecar(f)
    assert (d.rating, d.label, d.decision) == (3, "Green", "reject")
    assert out.lr_kept and out.written_rating is None and out.written_label is None
    out2 = WriteOutcome()
    write_decision(f, S, "reject", ["r"], 0.1,
                   override={"decision": "reject", "note": "ui", "rating": None}, outcome=out2)
    assert not out2.lr_kept and read_sidecar(f).label == "Red"


# ---- performance (acceptance 7) ----------------------------------------------------------------------


@pytest.mark.large
def test_perf_100k_catalog_10k_shoot(tmp_path):
    import psutil

    import make_fixtures

    folder = tmp_path / "big"
    frames = ls.build(folder, 10_000, S)
    run_cull(folder, S)
    ents = [
        {"root": str(folder), "rel": "", "base": f.name[:-4], "ext": "jpg", "pick": 0,
         "rating": 0, "label": "", "touch": 1_700_000_000.0, "capture": iso(f.ts)}
        for f in frames
    ]  # fmt: skip
    ents += [
        {"root": f"/Volumes/Archive/{i // 500}", "rel": "", "base": f"OLD_{i:06d}", "ext": "cr3",
         "pick": 0, "rating": i % 6, "label": "", "touch": 1_600_000_000.0}
        for i in range(90_000)
    ]  # fmt: skip
    cat = tmp_path / "Big.lrcat"
    t0 = time.perf_counter()
    make_fixtures.build_lrcat(cat, ents)
    assert time.perf_counter() - t0 <= 5.0
    for f in frames[:50]:
        make_fixtures.lr_edit(cat, folder / f.name, pick=1, touch=time.time())
    before = snap(cat)
    proc = psutil.Process()
    rss0 = proc.memory_info().rss
    t0 = time.perf_counter()
    res = lrsync.sync(folder, cat, S, report=False)
    wall = time.perf_counter() - t0
    assert wall <= 3.0, wall
    assert res.n_catalog == 10_000 and res.n_new == 50
    assert (proc.memory_info().rss - rss0) / 1e6 <= 200
    assert snap(cat) == before


@pytest.mark.large
def test_perf_locked_copy_1gb_catalog(tmp_path):
    import make_fixtures

    folder = tmp_path / "shoot"
    frames = ls.build(folder, 500, S)
    run_cull(folder, S)
    cat = tmp_path / "Huge.lrcat"
    make_fixtures.build_lrcat(
        cat,
        [{"root": str(folder), "rel": "", "base": f.name[:-4], "ext": "jpg", "pick": 0,
          "rating": 0, "label": ""} for f in frames],
        pad_mb=1000,
    )  # fmt: skip
    assert cat.stat().st_size >= 1_000_000_000
    holder = _hold_exclusive(cat)
    try:
        before = snap(cat)
        t0 = time.perf_counter()
        res = lrsync.sync(folder, cat, S, dry_run=True, report=False)
        wall = time.perf_counter() - t0
    finally:
        holder.stdin.write("\n")
        holder.stdin.flush()
        holder.wait(10)
    assert res.copied and res.n_matched == 500
    assert wall <= 10.0, wall
    assert snap(cat) == before


# ---- API + CLI --------------------------------------------------------------------------------------


def _wait(client, job_id, timeout=60):
    t = time.time()
    while time.time() - t < timeout:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["state"] in ("done", "error"):
            return j
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_api_lr_sync_report_catalogs_badge(shoot, mf, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from shotsieve.server.app import create_app

    cat = shoot.catalog(tmp_path / "Catalog.lrcat")
    want = edit_12(shoot, cat, mf, touch=time.time() + 10)
    home = tmp_path / "home"
    (home / "Pictures" / "Lightroom" / "Archive").mkdir(parents=True)
    other = home / "Pictures" / "Lightroom" / "Archive" / "My.lrcat"
    other.write_bytes(b"")
    deep = home / "Pictures" / "Lightroom" / "a" / "b" / "c" / "d"
    deep.mkdir(parents=True)
    (deep / "TooDeep.lrcat").write_bytes(b"")
    monkeypatch.setenv("HOME", str(home))
    before = snap(cat)
    with TestClient(create_app(S, shoot.folder)) as cl:
        listed = cl.get("/api/lr/catalogs").json()
        assert str(other) in listed["catalogs"] and not any("TooDeep" in c for c in listed["catalogs"])
        assert cl.get("/api/lr/report").json()["report"] is None
        for bad in ("relative.lrcat", str(tmp_path / "nope.lrcat"), str(tmp_path / "x.txt")):
            r = cl.post("/api/lr/sync", json={"catalog": bad})
            assert r.status_code == 400, bad
        r = cl.post("/api/lr/sync", json={"catalog": str(cat), "dry_run": True})
        j = _wait(cl, r.json()["job_id"])
        assert j["kind"] == "lr_sync" and j["state"] == "done", j
        assert j["result"]["n_new"] == 12 and j["result"]["dry_run"] is True
        assert lr_overrides(shoot.folder) == {}
        assert not any(i["lr"] for i in cl.get("/api/shoot").json()["images"])
        r = cl.post("/api/lr/sync", json={"catalog": str(cat)})
        j = _wait(cl, r.json()["job_id"])
        assert j["state"] == "done" and j["result"]["n_new"] == 12
        rep = cl.get("/api/lr/report").json()
        assert rep["report"]["n_evaluated"] == len(shoot.res.images) and rep["catalog"] == str(cat)
        assert rep["last"]["n_new"] == 12
        imgs = cl.get("/api/shoot").json()["images"]
        assert {i["rel"] for i in imgs if i["lr"]} == {os.path.relpath(p, shoot.folder) for p in want}
        assert str(cat) in cl.get("/api/lr/catalogs").json()["catalogs"]  # remembered
        # the one-job lock: a second sync while a cull-class job runs is refused
        cl.app.state.shotsieve.active = type("J", (), {"state": "running", "folder": "x"})()
        assert cl.post("/api/lr/sync", json={"catalog": str(cat)}).status_code == 409
    assert snap(cat) == before


def test_cli_sync_exit_codes_and_json(shoot, mf, tmp_path, capsys):
    import json

    from shotsieve.cli import main

    cat = shoot.catalog(tmp_path / "Catalog.lrcat")
    edit_12(shoot, cat, mf, touch=time.time() + 10)
    assert main(["sync", str(shoot.folder), "--lrcat", str(cat), "--dry-run", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["n_new"] == 12 and out["dry_run"] is True and out["report"]["n_evaluated"] > 0
    assert lr_overrides(shoot.folder) == {}
    assert main(["sync", str(shoot.folder), "--lrcat", str(cat)]) == 0
    printed = capsys.readouterr().out
    assert "new overrides" in printed and "this shoot" in printed
    assert len(lr_overrides(shoot.folder)) == 12
    assert main(["sync", str(shoot.folder), "--lrcat", str(cat), "--no-report", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["report"] is None
    assert main(["sync", str(shoot.folder), "--lrcat", str(tmp_path / "missing.lrcat")]) == 2
    assert main(["sync", str(shoot.folder), "--lrcat", str(tmp_path / "x.txt")]) == 2
    with pytest.raises(SystemExit) as e:
        main(["sync", str(tmp_path / "nofolder"), "--lrcat", str(cat)])
    assert e.value.code == 2
    with pytest.raises(SystemExit) as e:
        main(["sync", str(shoot.folder)])
    assert e.value.code == 2


def test_every_catalog_reader_copies_a_closed_wal_catalog(shoot, mf, tmp_path):
    """read_catalog, read_develop and CatalogReader never touch a closed WAL catalog's folder,
    and no other module opens a catalog by itself."""
    (tmp_path / "lr").mkdir()
    cat = shoot.catalog(tmp_path / "lr" / "Catalog.lrcat")
    con = sqlite3.connect(cat)
    con.execute("PRAGMA journal_mode=WAL")
    con.close()
    before, listing = snap(cat), sorted(p.name for p in cat.parent.iterdir())
    assert len(lrcat.read_catalog(cat)) > 0
    with contextlib.suppress(sqlite3.OperationalError):  # the fixture has no develop table
        lrcat.read_develop(cat)
    with lrcat.CatalogReader(cat) as r:
        assert r.copied and r.entries()
    assert snap(cat) == before
    assert sorted(p.name for p in cat.parent.iterdir()) == listing
    src = ROOT / "src" / "shotsieve"
    for f in src.rglob("*.py"):
        if f.name != "lrcat.py":
            assert "open_ro" not in f.read_text(), f


def test_catalog_switching_to_wal_between_check_and_read_leaves_no_side_files(
    shoot, mf, tmp_path, spy_connect
):
    """n1 probe: a rollback-mode catalog is converted to WAL (and closed) after the reader was
    entered and before the query. The old design re-opened the original and created -shm/-wal."""
    (tmp_path / "lr").mkdir()
    cat = shoot.catalog(tmp_path / "lr" / "Catalog.lrcat")
    with lrcat.CatalogReader(cat) as r:  # header said "rollback" here
        con = sqlite3.connect(cat)  # (the TEST plays Lightroom switching the journal mode)
        con.execute("PRAGMA journal_mode=WAL")
        con.close()
        for f in cat.parent.glob("Catalog.lrcat-*"):
            f.unlink()  # a clean close leaves no side files
        before, listing = snap(cat), sorted(p.name for p in cat.parent.iterdir())
        mark = len(spy_connect)  # only connects made by the reader from here on
        assert r.entries()
        assert r.entries(prefix=str(shoot.folder))  # several reads on one reader (as lrsync does)
    assert snap(cat) == before
    assert sorted(p.name for p in cat.parent.iterdir()) == listing
    _assert_ro_only(spy_connect[mark:], cat)


def test_stale_private_copies_are_swept(tmp_path, monkeypatch):
    """m3: `data/tmp/lrcat-*` older than 1 h (a killed sync) goes; fresh ones and other names stay."""
    monkeypatch.setenv("SHOTSIEVE_DATA_DIR", str(tmp_path))
    root = tmp_path / "tmp"
    old, new, other = root / "lrcat-old", root / "lrcat-new", root / "keep-me"
    for d in (old, new, other):
        d.mkdir(parents=True)
        (d / "Catalog.lrcat").write_bytes(b"x")
    past = time.time() - 2 * 3600
    for d in (old, other):
        os.utime(d, (past, past))
    assert lrcat.sweep_stale_copies() == 1
    assert not old.exists() and new.exists() and other.exists()


def test_next_sync_sweeps_stale_copies(shoot, mf, tmp_path):
    stale = data_dir() / "tmp" / "lrcat-stale-test"
    stale.mkdir(parents=True, exist_ok=True)
    (stale / "Catalog.lrcat").write_bytes(b"x")
    past = time.time() - 2 * 3600
    os.utime(stale, (past, past))
    cat = shoot.catalog(tmp_path / "Catalog.lrcat", wal=True)  # WAL: the copy path runs
    lrsync.sync(shoot.folder, cat, S, dry_run=True, report=False)
    assert not stale.exists()


def test_sync_result_carries_summary_line(shoot, mf, tmp_path):
    cat = shoot.catalog(tmp_path / "Catalog.lrcat")
    edit_12(shoot, cat, mf, time.time())
    d = lrsync.sync(shoot.folder, cat, S, dry_run=True, report=False).to_dict()
    assert d["n_writes"] == 12 and d["summary"].startswith("12 changes: ")
