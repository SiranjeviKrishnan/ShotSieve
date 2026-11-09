"""Personal model, dataset labelling / IPW weights, audit sampling and calibration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from shotsieve.config import Settings
from shotsieve.features.vector import FEATURE_NAMES, SCHEMA_VERSION, FeatureVector
from shotsieve.io.lrcat import CatalogEntry
from shotsieve.io.xmp import write_sidecar
from shotsieve.personal import audit, calibrate, dataset
from shotsieve.personal.model import PersonalModel, SchemaMismatchError
from shotsieve.pipeline.cache import Cache, CachedImage
from shotsieve.scoring import selection

NF = len(FEATURE_NAMES)


def synth(n=400, seed=0, noise=0.4):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, NF)).astype(np.float32)
    X[rng.random(X.shape) < 0.05] = np.nan  # missing values are legal
    signal = np.nan_to_num(X[:, 1]) + 0.7 * np.nan_to_num(X[:, 12]) - 0.5 * np.nan_to_num(X[:, 13])
    y = (signal + rng.normal(0, noise, n) > 0).astype(int)
    emb = rng.normal(size=(n, 384)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return X, y, emb


# ---- model ------------------------------------------------------------------------------


def test_fit_reaches_auc_above_0_8_and_is_calibrated():
    X, y, emb = synth(600)
    m = PersonalModel().fit(X[:450], y[:450], embeddings=emb[:450])
    assert m.trained and m.n_train == 450 and m.pca.n_components_ == 16
    p = m.predict_proba(X[450:], embeddings=emb[450:])
    assert p.shape == (150,) and ((p >= 0) & (p <= 1)).all()
    assert roc_auc_score(y[450:], p) > 0.8
    assert abs(p.mean() - y[450:].mean()) < 0.1  # calibrated, not just ranked
    assert m.calibration.startswith(("sigmoid", "isotonic"))
    # still usable without embeddings (PCA columns become NaN)
    assert m.predict_proba(X[450:]).shape == (150,)


def test_runner_protocol_and_embeddings_kwarg():
    X, y, emb = synth(120)
    m = PersonalModel().fit(X, y, embeddings=emb)
    assert isinstance(m, selection.PersonalModelLike)
    a = selection.personal_probabilities(m, X[:5], emb[:5])
    assert np.allclose(a, m.predict_proba(X[:5], embeddings=emb[:5]))


def test_ipw_weights_are_applied():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, NF)).astype(np.float32)
    y = rng.integers(0, 2, 120)  # labels carry no signal: only the weights can matter
    keep_heavy = PersonalModel().fit(X, y, np.where(y == 1, 10.0, 1.0))
    reject_heavy = PersonalModel().fit(X, y, np.where(y == 1, 1.0, 10.0))
    assert keep_heavy.predict_proba(X).mean() > reject_heavy.predict_proba(X).mean() + 0.3


def test_prior_only_when_too_few_rows_or_one_class():
    X, y, _ = synth(200)
    small = PersonalModel().fit(X[:39], y[:39])
    assert not small.trained and small.n_train == 0 and small.n_rows == 39
    assert np.all(small.predict_proba(X[:4]) == 0.5)
    single = PersonalModel().fit(X, np.ones(200, int))
    assert not single.trained and single.n_train == 0
    ok = PersonalModel().fit(X[:40], y[:40])
    assert ok.trained and ok.n_train == 40
    # an untrained model leaves the runner blend weight at zero
    assert selection.blend_weight(small.n_train) == 0.0


def test_save_load_roundtrip_and_schema_mismatch(tmp_path):
    X, y, emb = synth(150)
    m = PersonalModel().fit(X, y, embeddings=emb)
    path = m.save(tmp_path / "m" / "model.joblib")
    loaded = PersonalModel.load(path)
    assert loaded.n_train == 150 and loaded.pca.n_components_ == 16  # PCA saved with the model
    assert np.allclose(loaded.predict_proba(X, embeddings=emb), m.predict_proba(X, embeddings=emb))
    payload = joblib.load(path)
    for field, value in (("schema_version", SCHEMA_VERSION + 1), ("feature_names", ["a"])):
        bad = dict(payload, **{field: value})
        joblib.dump(bad, tmp_path / "bad.joblib")
        with pytest.raises(SchemaMismatchError):
            PersonalModel.load(tmp_path / "bad.joblib")
    joblib.dump([1, 2], tmp_path / "junk.joblib")
    with pytest.raises(SchemaMismatchError):
        PersonalModel.load(tmp_path / "junk.joblib")


def test_runner_ignores_model_with_wrong_schema(tmp_path):
    from shotsieve.pipeline.runner import _load_personal

    X, y, _ = synth(100)
    path = PersonalModel().fit(X, y).save(tmp_path / "m.joblib")
    payload = joblib.load(path)
    payload["schema_version"] = 99
    joblib.dump(payload, path)
    assert _load_personal(Settings(personal_model_path=str(path))) is None


# ---- dataset ----------------------------------------------------------------------------


def test_catalog_labels_and_weights():
    e = CatalogEntry
    entries = [
        e("/s/a/1.jpg", 4, 0, None),  # rating >= 1 -> keep
        e("/s/a/2.jpg", None, 1, None),  # flagged -> keep
        e("/s/a/3.jpg", 5, -1, None),  # rejected flag wins
        e("/s/a/4.jpg", 0, 0, None),  # seen but not selected (shoot has picks) -> weak negative
        e("/s/a/5.jpg", None, 0, None),
        e("/s/b/1.jpg", 0, 0, None),  # shoot b has no picks/ratings at all -> no label
        e("/s/b/2.jpg", None, 0, None),
    ]
    labs = {
        Path(x.path).parent.name + "/" + Path(x.path).name: x
        for x in dataset.labels_from_catalog(entries)
    }
    assert set(labs) == {"a/1.jpg", "a/2.jpg", "a/3.jpg", "a/4.jpg", "a/5.jpg"}
    assert labs["a/1.jpg"].keep and labs["a/2.jpg"].keep and not labs["a/3.jpg"].keep
    assert labs["a/3.jpg"].weight == 1.0
    assert not labs["a/4.jpg"].keep and labs["a/4.jpg"].weight == 0.5
    # rating below min_rating is neither keep nor weak negative
    strict = dataset.labels_from_catalog(
        [e("/s/a/1.jpg", 2, 0, None), e("/s/a/2.jpg", 3, 0, None)], 3
    )
    assert [(Path(x.path).name, x.keep) for x in strict] == [("2.jpg", True)]


def test_override_labels_carry_inverse_propensity(tmp_path):
    with Cache.for_folder(tmp_path) as c:
        c.set_override(str(tmp_path / "a.jpg"), "keep", 0.25)
        c.set_override(str(tmp_path / "b.jpg"), "reject", 1.0)
        c.set_override(str(tmp_path / "c.jpg"), "review", 1.0)  # not a label
    labs = {Path(x.path).name: x for x in dataset.labels_from_overrides(tmp_path)}
    assert set(labs) == {"a.jpg", "b.jpg"}
    assert labs["a.jpg"].weight == pytest.approx(4.0) and labs["a.jpg"].keep
    assert labs["b.jpg"].weight == 1.0 and not labs["b.jpg"].keep
    assert dataset.labels_from_overrides(tmp_path / "nothing") == []


def test_xmp_folder_labels(tmp_path):
    for name in "abcd":
        (tmp_path / f"{name}.jpg").write_bytes(b"x")
    # the photographer's own tool wrote these (no ShotSieve provenance)
    write_sidecar(tmp_path / "a.jpg", rating=4, label="Green", record=False)
    write_sidecar(tmp_path / "b.jpg", rating=-1, label="Red", record=False)
    write_sidecar(tmp_path / "c.jpg", rating=0, record=False)
    labs = {Path(x.path).name: x for x in dataset.labels_from_xmp_folder(tmp_path)}
    assert labs["a.jpg"].keep and not labs["b.jpg"].keep and labs["b.jpg"].weight == 1.0
    assert not labs["c.jpg"].keep and labs["c.jpg"].weight == 0.5
    assert "d.jpg" not in labs  # no sidecar -> no label


def _stored_photos(tmp_path, n=3):
    from shotsieve.features.record import RawFeatures
    from shotsieve.io.exif import ImageMeta

    paths = []
    with Cache.for_folder(tmp_path) as c:
        rows = []
        for i in range(n):
            p = tmp_path / f"{i}.jpg"
            p.write_bytes(b"x")
            paths.append(p)
            scene = np.zeros(384, np.float32)
            scene[i] = 1.0
            c.put(
                str(p),
                1,
                1,
                CachedImage(ImageMeta(timestamp=0.0), RawFeatures(), scene, np.zeros((0, 128))),
            )
            fv = FeatureVector(sharp_global_norm=0.1 * i)
            rows.append(dict(path=str(p), decision="keep", reasons=[], score=0.5, prior=0.5,
                             group_id=i, burst_id=i, is_best=True, audit=False, audit_prob=None,
                             features=fv.to_dict()))  # fmt: skip
        c.save_decisions(rows)
    return paths


def test_featurise_uses_stored_features_without_running_models(tmp_path):
    paths = _stored_photos(tmp_path)
    labels = [
        dataset.Label(str(paths[0]), True),
        dataset.Label(str(paths[1]), False, 0.5, 0.5, "override"),
        dataset.Label(str(tmp_path / "gone.jpg"), True),
    ]

    def boom(*a, **k):
        raise AssertionError("must not re-run the pipeline")

    ts = dataset.featurise(labels, Settings(), run=boom)
    assert (
        len(ts) == 2
        and ts.n_missing == 1
        and ts.X.shape == (2, NF)
        and ts.embeddings.shape == (2, 384)
    )
    assert list(ts.y) == [1, 0] and list(ts.w) == [1.0, 1.0]  # 0.5 / 0.5
    assert ts.X[1, FEATURE_NAMES.index("sharp_global_norm")] == pytest.approx(0.1)


def test_video_frame_overrides_are_counted_not_missing_and_not_trained(tmp_path):
    paths = _stored_photos(tmp_path, 5)
    (tmp_path / "clip.mp4").write_bytes(b"x")
    labels = [dataset.Label(str(p), i % 2 == 0, 1.0, 1.0, "override") for i, p in enumerate(paths)]
    labels += [
        dataset.Label(f"{tmp_path / 'clip.mp4'}#t=1.500", True, 1.0, 1.0, "override"),
        dataset.Label(f"{tmp_path / 'clip.mp4'}#t=2.500", False, 1.0, 1.0, "override"),
        dataset.Label(f"{tmp_path / 'deleted.mp4'}#t=3.000", True, 1.0, 1.0, "override"),
    ]

    def boom(*a, **k):
        raise AssertionError("must not re-run the pipeline")

    ts = dataset.featurise(labels, Settings(), run=boom)
    assert len(ts) == 5 and ts.n_video == 3 and ts.n_missing == 0
    assert not any("#t=" in p for p in ts.paths)


# ---- audit ------------------------------------------------------------------------------


def fake_results(n_reject=100, n_other=20, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_reject + n_other):
        rej = i < n_reject
        out.append(SimpleNamespace(
            master_path=Path(f"/s/{i:04d}.jpg"), decision="reject" if rej else "keep",
            reasons=["low_quality"] if rej else [],
            reason_texts=["Overall quality score is low."] if rej else [],
            score=float(rng.uniform(0, 0.3)) if rej else 0.9, audit=False, audit_prob=None,
            features=FeatureVector(),
        ))  # fmt: skip
    return out


def test_audit_size_rule():
    assert [audit.audit_size(n) for n in (0, 3, 5, 100, 200, 1000, 5000)] == [0, 3, 5, 5, 6, 25, 25]


def test_audit_sample_is_deterministic_and_mutates_in_place():
    a, b, c = fake_results(), fake_results(), fake_results()
    pa, pb, pc = audit.sample_audit(a, 7), audit.sample_audit(b, 7), audit.sample_audit(c, 8)
    names = lambda ps: [str(p.master_path) for p in ps]  # noqa: E731
    assert names(pa) == names(pb) and names(pa) != names(pc)
    assert len(pa) == audit.audit_size(100) == 5
    for r in pa:
        assert r.decision == "review" and r.audit and 0 < r.audit_prob <= 1
        assert r.reasons[0] == "audit" and r.reasons[1:] == ["low_quality"]
        assert len(r.reason_texts) == len(r.reasons)
    assert sum(r.audit for r in a) == 5 and sum(r.decision == "keep" for r in a) == 20


def test_audit_inclusion_probability_is_correct_and_favours_boundary():
    res = fake_results(60)
    probs = {}
    hits = {str(r.master_path): 0 for r in res if r.decision == "reject"}
    trials = 600
    for seed in range(trials):
        sample = audit.sample_audit(fake_results(60), seed)
        for r in sample:
            hits[str(r.master_path)] += 1
            probs[str(r.master_path)] = r.audit_prob
    assert sum(hits.values()) == trials * 5
    for key, h in hits.items():
        if key in probs:
            assert h / trials == pytest.approx(probs[key], abs=0.06)
    rej = [r for r in res if r.decision == "reject"]
    w = np.array([r.score + 0.05 for r in rej])
    pi = audit.inclusion_probabilities(w, 5)
    assert pi.sum() == pytest.approx(5) and pi.max() <= 1
    assert pi[np.argmax(w)] > pi[np.argmin(w)]  # closer to the boundary -> likelier
    capped = audit.inclusion_probabilities(np.array([100.0, 1.0, 1.0, 1.0]), 2)
    assert capped[0] == 1.0 and capped.sum() == pytest.approx(2)


def test_audit_skips_empty_and_small_buckets():
    assert audit.sample_audit(fake_results(0, 5), 0) == []
    small = fake_results(3, 0)
    assert len(audit.sample_audit(small, 0)) == 3


# ---- calibrate --------------------------------------------------------------------------


def test_ipw_estimate_math():
    s = [calibrate.AuditSample(0.5, True), calibrate.AuditSample(0.25, True),
         calibrate.AuditSample(0.1, False)]  # fmt: skip
    assert calibrate.estimate_missed_keepers(s) == pytest.approx(2.0 + 4.0)
    assert calibrate.estimate_missed_keepers([calibrate.AuditSample(0.3, False)]) == 0.0
    assert calibrate.missed_keeper_rate(6.0, 94) == pytest.approx(0.06)
    assert calibrate.missed_keeper_rate(0.0, 0) == 0.0


def test_threshold_lowered_only_when_estimate_exceeds_one_percent():
    th = {"t_keep": 0.6, "t_reject": 0.3}
    none = calibrate.calibrate([calibrate.AuditSample(0.2, False)] * 5, 50, th)
    assert none.thresholds["t_reject"] == 0.3 and not none.lowered
    assert none.missed_keeper_rate == 0.0
    # one kept at pi=0.5 -> 2 missed of 202 keepers = 0.99 % -> unchanged; at pi=0.4 -> 1.2 %
    edge = calibrate.calibrate([calibrate.AuditSample(0.5, True)], 198, th)
    assert not edge.lowered
    hit = calibrate.calibrate([calibrate.AuditSample(0.4, True)], 198, th)
    assert hit.lowered and hit.thresholds["t_reject"] == 0.25 and hit.thresholds["t_keep"] == 0.6
    mid = calibrate.calibrate([calibrate.AuditSample(0.1, True)], 140, th)  # est 10, rate 6.7 %
    assert mid.thresholds["t_reject"] == pytest.approx(0.20)  # two 0.05 steps
    huge = calibrate.calibrate([calibrate.AuditSample(0.1, True)] * 2, 50, th)  # rate 28 %
    assert huge.thresholds["t_reject"] == 0.10  # floored
    floor = calibrate.calibrate(
        [calibrate.AuditSample(0.5, True)], 1, {"t_keep": 0.6, "t_reject": 0.12}
    )
    assert floor.thresholds["t_reject"] == 0.10
    assert calibrate.lower_t_reject(0.10, 0.5) == 0.10


def test_calibrate_shoot_reads_audit_decisions_and_persists(tmp_path):
    rows = []
    for i in range(6):
        rows.append(dict(path=str(tmp_path / f"{i}.jpg"), decision="review" if i < 4 else "keep",
                         reasons=["audit"] if i < 4 else [], score=0.2, prior=0.2, group_id=i,
                         burst_id=i, is_best=True, audit=i < 4, audit_prob=0.25 if i < 4 else None,
                         features={}))  # fmt: skip
    with Cache.for_folder(tmp_path) as c:
        c.save_decisions(rows)
        c.set_override(str(tmp_path / "0.jpg"), "keep", 0.25)  # user rescued one
        c.set_override(str(tmp_path / "1.jpg"), "reject", 0.25)
        c.set_override(str(tmp_path / "2.jpg"), "reject", 0.25)  # item 3 stays undecided
    res, cull = calibrate.calibrate_shoot(tmp_path, Settings(), redecide=False)
    assert res.n_audited == 3 and res.n_kept == 1 and res.n_pending == 1
    assert res.est_missed_keepers == pytest.approx(4.0)
    assert res.missed_keeper_rate == pytest.approx(4 / 6)  # 4 est. missed vs 2 keepers found
    assert res.lowered and res.thresholds["t_reject"] == pytest.approx(0.10)  # rate 67 %: floored
    # m2: one audited reject is still unjudged -> nothing is persisted yet
    assert not res.applied and cull is None
    with Cache.for_folder(tmp_path) as c:
        assert c.get_thresholds() is None
        c.set_override(str(tmp_path / "3.jpg"), "reject", 0.25)  # judge the last one
    res, cull = calibrate.calibrate_shoot(tmp_path, Settings(), redecide=False)
    assert res.n_pending == 0 and res.applied and res.n_audited == 4
    with Cache.for_folder(tmp_path) as c:
        saved = c.get_thresholds()
    assert saved["t_reject"] == res.thresholds["t_reject"] < 0.3 and cull is None
    again, _ = calibrate.calibrate_shoot(tmp_path, Settings(), redecide=False)
    assert again.thresholds == res.thresholds  # idempotent: no ratchet on repeated calls
