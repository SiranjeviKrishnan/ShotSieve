from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from shotsieve.config import Settings
from shotsieve.features.record import RawFeatures
from shotsieve.features.subject import FaceObs
from shotsieve.io.exif import ImageMeta
from shotsieve.io.lrcat import CatalogEntry
from shotsieve.pipeline import evaluate
from shotsieve.pipeline.cache import Cache, CachedImage, cache_path


def _rec(seed=0):
    rng = np.random.default_rng(seed)
    scene = rng.normal(size=384).astype(np.float32)
    scene /= np.linalg.norm(scene)
    faces = rng.normal(size=(2, 128)).astype(np.float32)
    raw = RawFeatures(width=10, height=8, lap_var=1.5, faces=[FaceObs((1, 2, 3, 4), 0.9)])
    return CachedImage(ImageMeta(timestamp=12.5, camera_serial="X", iso=100), raw, scene, faces)


def test_cache_roundtrip_float16_and_keying(tmp_path):
    path = tmp_path / "x" / "cull.db"
    rec = _rec()
    with Cache(path) as c:
        c.put("/a.jpg", 100, 5, rec)
        c.put("/b.jpg", 100, 5, CachedImage(rec.meta, rec.raw, rec.scene, np.zeros((0, 128))))
        c.commit()
    with Cache(path) as c:
        hit = c.lookup_many({"/a.jpg": (100, 5), "/b.jpg": (100, 5)})
        assert set(hit) == {"/a.jpg", "/b.jpg"}
        got = hit["/a.jpg"]
        assert got.meta.iso == 100 and got.meta.camera_serial == "X" and got.raw.lap_var == 1.5
        assert got.raw.faces[0].bbox == (1, 2, 3, 4)
        assert np.abs(got.scene - rec.scene).max() < 2e-3 and got.faces.shape == (2, 128)
        assert hit["/b.jpg"].faces.shape == (0, 128)
        # changed size or mtime -> miss
        assert c.lookup_many({"/a.jpg": (101, 5)}) == {}
        assert c.lookup_many({"/a.jpg": (100, 6)}) == {}
        assert c.lookup_many({}) == {}


def test_cache_decisions_overrides_thresholds(tmp_path):
    with Cache(tmp_path / "c.db") as c:
        row = dict(path="/a.jpg", decision="reject", reasons=["blur"], score=0.2, prior=0.3,
                   group_id=1, burst_id=2, is_best=False, audit=True, audit_prob=0.25,
                   features={"n_faces": 1.0})  # fmt: skip
        c.save_decisions([row])
        c.save_decisions([row])  # replaces, not appends
        d = c.load_decisions()
        assert list(d) == ["/a.jpg"] and d["/a.jpg"]["reasons"] == ["blur"]
        assert d["/a.jpg"]["audit"] is True and d["/a.jpg"]["audit_prob"] == 0.25
        assert c.overrides() == {}
        c.set_override("/a.jpg", "keep", 0.25, "user")
        assert c.overrides()["/a.jpg"]["decision"] == "keep"
        assert c.overrides()["/a.jpg"]["propensity"] == 0.25
        c.clear_override("/a.jpg")
        assert c.overrides() == {}
        assert c.get_thresholds() is None
        c.set_thresholds({"t_reject": 0.2})
        assert c.get_thresholds() == {"t_reject": 0.2}
        assert c.get_meta("missing", 7) == 7


def test_cache_path_inside_folder(tmp_path):
    assert cache_path(tmp_path) == tmp_path / ".shotsieve" / "cull.db"
    Cache.for_folder(tmp_path).close()
    assert cache_path(tmp_path).exists()


# ---- evaluate ---------------------------------------------------------------------------


def _img(name, decision, root="/shoot"):
    p = Path(root) / name
    return SimpleNamespace(path=p, master_path=p, decision=decision)


def test_missed_keeper_rate_hand_made():
    images = [
        _img("a.jpg", "keep"), _img("b.jpg", "reject"), _img("c.jpg", "review"),
        _img("d.jpg", "reject"), _img("e.jpg", "reject"), _img("f.jpg", "keep"),
    ]  # fmt: skip
    truth = {f"/shoot/{n}.jpg": k for n, k in
             dict(a=True, b=True, c=True, d=False, e=False, f=False, zzz=True).items()}  # fmt: skip
    rep = evaluate.evaluate(images, truth)
    assert rep.n_evaluated == 6 and rep.n_kept_truth == 3  # 'zzz' is not in the shoot
    assert rep.missed_keepers == 1 and rep.missed_keeper_rate == pytest.approx(1 / 3)
    assert rep.keepers_in_review == 1 and rep.keepers_kept == 1
    assert rep.review_fraction == pytest.approx(1 / 6) and rep.reject_rate == 0.5
    assert rep.reject_precision == pytest.approx(2 / 3)
    assert rep.keep_precision == 0.5
    assert [Path(p).name for p in rep.missed_paths] == ["b.jpg"]


def test_evaluate_no_keepers_and_empty():
    rep = evaluate.evaluate([_img("a.jpg", "reject")], {"/shoot/a.jpg": False})
    assert rep.missed_keeper_rate == 0.0 and rep.reject_precision == 1.0
    empty = evaluate.evaluate([], {})
    assert empty.n_images == 0 and empty.missed_keeper_rate == 0.0


def test_truth_from_catalog_and_master_path_match():
    entries = [
        CatalogEntry("/shoot/a.jpg", 4, 0, None),
        CatalogEntry("/shoot/b.jpg", None, 1, None),  # flagged pick
        CatalogEntry("/shoot/c.jpg", 5, -1, None),  # rejected flag wins over rating
        CatalogEntry("/shoot/d.jpg", 0, 0, None),
        CatalogEntry("/shoot/e.jpg", None, 0, None),
    ]
    truth = evaluate.truth_from_catalog(entries)
    assert [Path(k).name for k, v in truth.items() if v] == ["a.jpg", "b.jpg"]
    raw = SimpleNamespace(
        path=Path("/shoot/a.jpg"), master_path=Path("/shoot/a.cr2"), decision="reject"
    )
    rep = evaluate.evaluate([raw], {"/shoot/a.jpg": True})  # falls back to the preview path
    assert rep.missed_keepers == 1


def test_runner_empty_folder_needs_no_models(tmp_path):
    from shotsieve.pipeline.runner import run_cull

    events = []
    res = run_cull(tmp_path, Settings(), lambda *a: events.append(a))
    assert res.images == [] and res.groups == {} and res.stats["n_images"] == 0
    assert any(e[0] == "scan" for e in events)
    assert res.thresholds["t_keep"] == 0.6 and res.thresholds["t_reject"] == 0.3
