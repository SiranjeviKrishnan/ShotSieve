"""End-to-end on the generated fixture shoot (needs the downloaded models)."""

import shutil
import sys
import types
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from shotsieve.config import Settings
from shotsieve.features.vector import FEATURE_NAMES
from shotsieve.io.xmp import read_sidecar, sidecar_path
from shotsieve.pipeline import evaluate
from shotsieve.pipeline.cache import Cache
from shotsieve.pipeline.runner import run_cull

NO_AUDIT = Settings(audit_enabled=False)  # tiny shoots would otherwise audit most rejects

pytestmark = [pytest.mark.models, pytest.mark.usefixtures("models_ready")]


@pytest.fixture(scope="module")
def shoot(tmp_path_factory, shoot_dir) -> Path:
    dest = tmp_path_factory.mktemp("integration") / "shoot"
    shutil.copytree(shoot_dir, dest)
    return dest


@pytest.fixture(scope="module")
def cold(shoot):
    events = []
    res = run_cull(shoot, NO_AUDIT, lambda *a: events.append(a), write_xmp=True)
    return res, events


def by_name(res):
    return {r.path.name: r for r in res.images}


def test_every_image_decided(cold, manifest):
    res, _ = cold
    assert len(res.images) == len(manifest) == 17
    for r in res.images:
        assert r.decision in {"keep", "reject", "review"}
        assert 0.0 <= r.score <= 1.0 and len(r.reason_texts) == len(r.reasons)
        if r.decision == "reject":
            assert r.reasons and all(len(t) > 10 for t in r.reason_texts)
    assert sum(res.stats["counts"].values()) == 17


def test_bad_frames_rejected_with_correct_reasons(cold):
    names = by_name(cold[0])
    assert names["burst0_1.jpg"].decision == "reject" and "blur" in names["burst0_1.jpg"].reasons
    dark = names["burst1_3.jpg"]
    assert dark.decision == "reject" and "exposure_bad" in dark.reasons
    mis = names["burst2_2.jpg"]
    assert mis.decision == "reject" and "flash_misfire" in mis.reasons
    black = names["black.jpg"]
    assert black.decision == "reject" and black.reasons == ["accidental"]
    assert black.features.n_faces == 0 and black.features.exp_mean < 1  # face stage skipped


def test_good_frames_survive(cold):
    names = by_name(cold[0])
    assert names["landscape.jpg"].decision == "keep" and not names["landscape.jpg"].reasons
    for b in range(3):
        frames = [r for n, r in names.items() if n.startswith(f"burst{b}_")]
        assert any(r.decision == "keep" for r in frames), f"burst {b} has no keeper"
        assert sum(r.is_best_in_group for r in frames) == 1


def test_groups_and_bursts(cold, manifest):
    res, _ = cold
    names = by_name(res)
    burst_ids, group_ids = set(), set()
    for b in range(3):
        frames = [names[m["file"]] for m in manifest if m["burst"] == b]
        assert len({f.burst_id for f in frames}) == 1 and len({f.group_id for f in frames}) == 1
        burst_ids.add(frames[0].burst_id)
        group_ids.add(frames[0].group_id)
    assert len(burst_ids) == 3 and len(group_ids) == 3  # three bursts, three duplicate groups
    assert len(res.groups) == 5
    # exactly one best per duplicate group
    for gid, members in res.groups.items():
        assert sum(r.is_best_in_group for r in res.images if r.group_id == gid) == 1
        assert len(members) >= 1
    # duplicates that lost to a better frame say which one
    dup = [r for r in res.images if "duplicate" in r.reasons]
    for r in dup:
        best = next(x for x in res.images if x.group_id == r.group_id and x.is_best_in_group)
        assert best.path.name in r.reason_texts[r.reasons.index("duplicate")]
    # near-duplicate burst frames never merge across different people
    assert len({names["burst0_0.jpg"].group_id, names["burst1_0.jpg"].group_id}) == 2


def test_progress_stages(cold):
    _, events = cold
    stages = [e[0] for e in events]
    for s in ("scan", "features", "faces", "embed", "group", "score", "write"):
        assert s in stages, s
    for stage in ("features", "faces", "embed"):
        seq = [e[1] for e in events if e[0] == stage]
        assert seq == sorted(seq) and seq[-1] == 17
        assert all(e[2] == 17 for e in events if e[0] == stage)
    assert [e[1:] for e in events if e[0] == "write"][-1] == (17, 17)


def test_xmp_written_and_parsable(cold, shoot):
    res, _ = cold
    for r in res.images:
        sp = sidecar_path(r.master_path)
        assert sp.exists()
        ET.parse(sp)  # valid XML
        data = read_sidecar(r.master_path)
        assert data.decision == r.decision
        assert data.label == {"keep": "Green", "review": "Yellow", "reject": "Red"}[r.decision]
        assert data.reasons == r.reason_texts
        assert data.score == pytest.approx(r.score, abs=1e-3)
    assert not list(shoot.glob("*.xmp.tmp"))
    # originals untouched: still 17 jpgs, sidecars are separate files
    assert len(list(shoot.glob("*.jpg"))) == 17


def test_results_persisted_and_cache_rerun_is_fast(cold, shoot):
    first, _ = cold
    db = shoot / ".shotsieve" / "cull.db"
    assert db.exists()
    with Cache(db) as c:
        saved = c.load_decisions()
    assert len(saved) == 17
    assert saved[str(shoot / "burst0_1.jpg")]["decision"] == "reject"
    second = run_cull(shoot, NO_AUDIT)
    assert second.stats["n_cached"] == 17 and second.stats["n_processed"] == 0
    assert first.stats["total_s"] >= 10 * second.stats["total_s"], (
        first.stats["total_s"], second.stats["total_s"])  # fmt: skip
    assert [r.decision for r in first.images] == [r.decision for r in second.images]
    assert [r.reasons for r in first.images] == [r.reasons for r in second.images]


def test_features_have_expected_values(cold):
    names = by_name(cold[0])
    ok, blur = names["burst0_2.jpg"].features, names["burst0_1.jpg"].features
    assert blur.sharp_global_norm < 0.15 < ok.sharp_global_norm
    assert names["burst1_3.jpg"].features.exp_ev < -2
    assert names["burst2_2.jpg"].features.flash_lum_delta_ev < -1.5
    assert names["landscape.jpg"].features.n_faces == 0
    assert ok.n_faces >= 1 and ok.eyes_closed_main < 0.5 and ok.dup_group_size == 5
    assert ok.burst_size == 5 and ok.flash_fired == 1.0 and not np.isnan(ok.iso_log2)


def test_missed_keeper_rate_on_fixture_truth(cold, manifest, shoot):
    # Burst frames are interchangeable near-duplicates, so truth is per moment: some normal
    # frame of every burst must survive, and the landscape (a lone keeper) must not be rejected.
    names = by_name(cold[0])
    for b in range(3):
        normal = [m["file"] for m in manifest if m["burst"] == b and m["variant"] == "normal"]
        assert any(names[f].decision != "reject" for f in normal), f"burst {b} lost every keeper"
    truth = {str(shoot / m["file"]): m["file"] == "landscape.jpg" for m in manifest}
    rep = evaluate.evaluate(cold[0].images, truth)
    assert rep.missed_keeper_rate == 0.0 and rep.n_kept_truth == 1


def test_override_and_threshold_persist(cold, shoot):
    key = str(shoot / "burst0_1.jpg")
    with Cache.for_folder(shoot) as c:
        c.set_override(key, "keep", 1.0)
        c.set_thresholds({"t_keep": 0.9, "t_reject": 0.3})
    res = run_cull(shoot, NO_AUDIT)
    r = by_name(res)["burst0_1.jpg"]
    assert r.decision == "keep" and r.overridden
    assert res.thresholds["t_keep"] == 0.9
    with Cache.for_folder(shoot) as c:
        c.clear_override(key)
        c.set_meta("thresholds", None)
    assert run_cull(shoot, NO_AUDIT).thresholds["t_keep"] == 0.6


class _Always:
    n_train = 200

    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        assert X.shape == (17, len(FEATURE_NAMES))
        return np.full(len(X), self.p)


def test_personal_model_blend(cold, shoot):
    base = by_name(run_cull(shoot, NO_AUDIT))
    boosted = by_name(run_cull(shoot, NO_AUDIT, personal_model=_Always(1.0)))
    for name, r in base.items():
        assert boosted[name].score == pytest.approx(0.5 * r.prior + 0.5, abs=1e-6)
    assert boosted["burst0_1.jpg"].decision == "reject"  # hard-fails override the model
    assert boosted["black.jpg"].decision == "reject"
    assert run_cull(shoot, NO_AUDIT).stats["personal_n_train"] == 0


def test_audit_hook_called_if_module_exists(cold, shoot, monkeypatch):
    calls = []

    def sample_audit(results, seed):
        calls.append((len(results), seed))
        rej = next(r for r in results if r.decision == "reject")
        rej.decision, rej.audit, rej.audit_prob = "review", True, 0.5

    mod = types.ModuleType("shotsieve.personal.audit")
    mod.sample_audit = sample_audit
    monkeypatch.setitem(sys.modules, "shotsieve.personal.audit", mod)
    res = run_cull(shoot, Settings(audit_seed=5, audit_enabled=True))
    assert calls == [(17, 5)] and sum(r.audit for r in res.images) == 1
    with Cache.for_folder(shoot) as c:
        rows = c.load_decisions()
    assert sum(v["audit"] for v in rows.values()) == 1
    audited = next(v for v in rows.values() if v["audit"])
    assert audited["decision"] == "review" and audited["audit_prob"] == 0.5


def test_broken_audit_module_does_not_crash(cold, shoot, monkeypatch):
    mod = types.ModuleType("shotsieve.personal.audit")
    mod.sample_audit = lambda results, seed: 1 / 0
    monkeypatch.setitem(sys.modules, "shotsieve.personal.audit", mod)
    assert len(run_cull(shoot, NO_AUDIT).images) == 17


def test_changed_file_invalidates_cache_and_garbage_is_reviewed(cold, shoot):
    bad = shoot / "garbage.jpg"
    bad.write_bytes(b"this is not a jpeg")
    p = shoot / "landscape.jpg"
    p.write_bytes(p.read_bytes() + b"\x00")  # size changes
    try:
        res = run_cull(shoot, NO_AUDIT)
        assert res.stats["n_images"] == 18 and res.stats["n_processed"] == 2
        g = by_name(res)["garbage.jpg"]
        assert g.decision == "review" and g.reasons == ["unreadable"] and g.reason_texts
        assert by_name(res)["landscape.jpg"].decision in {"keep", "review"}
    finally:
        bad.unlink()
        p.write_bytes(p.read_bytes()[:-1])
