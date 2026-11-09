# ruff: noqa: E501
"""T15: the editing profile on the synthetic photographer styles (docs/08_PHASE3.md)."""

from __future__ import annotations

import numpy as np
import pytest

from shotsieve.personal.edit_profile import (
    MIN_EDITS,
    EditProfile,
    EditSet,
    InsufficientEditsError,
    assign_folds,
)

# (MAE target, ratio vs median baseline) per slider name
TARGETS = {
    "Exposure2012": (0.15, 0.5),
    "wb_temp": (10.0, 0.6),
    "wb_tint": (5.0, 0.6),
}
SLIDER_MAE = 6.0
NOISE_FLOOR = 2.4  # MAE of sigma = 3 noise: no model can beat it


@pytest.fixture(scope="module")
def fitted(edit_sets):
    return {s: EditProfile().fit(edit_sets[s][0]) for s in "ABC"}


@pytest.fixture(scope="module")
def evals(edit_sets, fitted):
    return {s: fitted[s].evaluate(edit_sets[s][1]) for s in "ABC"}


@pytest.mark.parametrize("style", ["A", "B"])
def test_per_slider_mae_targets_on_held_out_bases(style, evals, fitted):
    ev = evals[style]
    for name, r in ev.items():
        if name.startswith("_"):
            continue
        assert r["n_pred"] > 0, f"{style}/{name} is off"
        if name in TARGETS:
            mae_t, ratio = TARGETS[name]
            assert r["mae"] <= mae_t, (style, name, r["mae"])
            assert r["mae"] <= ratio * r["base_mae"], (style, name, r["mae"], r["base_mae"])
        else:
            assert r["mae"] <= SLIDER_MAE, (style, name, r["mae"])
            if r["mode"] == ["constant"]:
                assert r["mae"] <= 1.1 * r["base_mae"], (style, name)
            elif r["base_mae"] >= 5.0:  # the style has real signal in this slider
                assert r["mae"] <= 0.6 * r["base_mae"], (style, name, r["mae"], r["base_mae"])
            else:  # signal below the noise (A's Shadows/Highlights): learned, not worse
                assert r["mae"] <= 0.9 * r["base_mae"], (style, name, r["mae"], r["base_mae"])


@pytest.mark.parametrize("style", ["A", "B"])
def test_intervals_gate_and_written_share(style, evals):
    ev = evals[style]
    learned = [r for k, r in ev.items() if not k.startswith("_") and r["mode"] == ["learned"]]
    assert learned, "no learned slider"
    for r in learned:
        assert 0.70 <= r["coverage"] <= 0.90, r  # empirical q10-q90 coverage
        if r["base_mae"] >= 5.0:  # slider with real signal: the gate removes the worse half
            assert r["mae_gated"] <= r["mae"] + 1e-9, r
        else:  # at the noise floor the gate cannot tell good from lucky: allow 0.1
            assert r["mae_gated"] <= r["mae"] + 0.1, r
    assert np.mean([r["mae_gated"] for r in learned]) <= np.mean([r["mae"] for r in learned])
    assert ev["_overall"]["written"] >= 0.75  # (image, slider) pairs written


def test_random_style_writes_almost_nothing_and_every_slider_off_or_constant(evals, fitted):
    ev = evals["C"]
    assert ev["_overall"]["written"] <= 0.10
    for key, m in fitted["C"].sliders.items():
        assert m.mode in ("off", "constant"), (key, m.mode, m.mae_cv, m.mae_median)
    assert all(r["mode"] != ["learned"] for k, r in ev.items() if not k.startswith("_"))


def test_n200_passes_exposure_target_and_n150_refuses(edit_sets):
    train, test, _ = edit_sets["A"]
    rng = np.random.default_rng(0)
    # whole base images (5 frames each) so folds stay grouped: 40 bases = 200 rows
    bases = np.unique(train.groups)
    pick = rng.choice(bases, 40, replace=False)

    def take(S, mask):
        return EditSet(
            S.stats[mask], S.emb[mask], S.fmt[mask], {k: v[mask] for k, v in S.y.items()},
            S.groups[mask], S.carry, [], S.sources, S.notes,
        )

    s200 = take(train, np.isin(train.groups, pick))
    assert len(s200) == MIN_EDITS == 200
    prof = EditProfile().fit(s200)
    r = prof.evaluate(test)["Exposure2012"]
    assert r["mae"] <= 0.15, r
    s150 = take(train, np.isin(train.groups, pick[:30]))
    assert len(s150) == 150
    with pytest.raises(InsufficientEditsError, match="need 200 edits, have 150; use `match`"):
        EditProfile().fit(s150)


def test_gate_withholds_couplings_and_ood(edit_sets, fitted):
    train, test, _ = edit_sets["A"]
    prof = fitted["A"]
    preds = prof.predict(test.stats, test.emb, test.fmt)
    saw_exposure_withheld = saw_wb_withheld = False
    for p in preds:
        if "Exposure2012" in p.written and not p.written["Exposure2012"]:
            saw_exposure_withheld = True
            for k in ("Highlights2012", "Shadows2012", "Whites2012", "Blacks2012"):
                if p.modes.get(k) == "learned":
                    assert not p.written[k]  # Exposure withheld -> its dependants too
        w = [p.written.get(k) for k in ("wb_temp", "wb_tint") if k in p.written]
        assert len(set(w)) <= 1  # temperature and tint are written together or not at all
        saw_wb_withheld |= bool(w) and not w[0]
        assert p.needs_edit == bool(p.omitted)
    assert saw_exposure_withheld and saw_wb_withheld
    # a frame that looks like nothing in training (random embedding) is out of distribution
    rng = np.random.default_rng(1)
    emb = rng.normal(size=(8, test.emb.shape[1])).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    odd = prof.predict(test.stats[:8], emb, test.fmt[:8])
    assert all(p.ood and not p.values() or p.ood for p in odd)
    assert all(not any(p.written.get(k) for k in p.modes if p.modes[k] == "learned") for p in odd)
    # constant sliders are still written for them: that is the user's constant, not a guess
    assert any(p.written for p in odd)


def test_profile_save_load_roundtrip_is_identical_and_small(tmp_path, edit_sets, fitted):
    _, test, _ = edit_sets["B"]
    prof = fitted["B"]
    path = prof.save(tmp_path / "p.joblib")
    assert path.stat().st_size <= 10_000_000
    again = EditProfile.load(path)
    assert again.profile_hash == prof.profile_hash
    a = prof.predict(test.stats, test.emb, test.fmt)
    b = again.predict(test.stats, test.emb, test.fmt)
    for x, y in zip(a, b, strict=True):
        assert x.written == y.written
        for k in x.point:
            assert x.point[k] == pytest.approx(y.point[k], abs=1e-4)


def test_assign_folds_keeps_groups_together():
    g = np.repeat(np.arange(23), 5)
    f = assign_folds(g)
    assert set(f) == set(range(5))
    for grp in range(23):
        assert len(set(f[g == grp])) == 1
    one = assign_folds(np.zeros(50, int))  # a single shoot: contiguous blocks, still 5 folds
    assert set(one) == set(range(5)) and list(one) == sorted(one)


def test_wb_only_for_bodies_seen_in_training_and_always_as_a_pair(edit_sets, fitted):
    """Review #6 M6 (T15) + m1: an unseen camera body gets no Temperature/Tint (with a note) and
    Temperature never comes without Tint."""
    from shotsieve.personal.edit_profile import _CAM_COL, WB_BODY_NOTE

    prof = fitted["A"]
    test = edit_sets["A"][1]
    n = min(60, len(test))
    stats = test.stats[:n].copy()
    base = prof.predict(stats, test.emb[:n], test.fmt[:n])
    assert any("wb_temp" in p.written for p in base), "the fixture should write WB for a known body"
    for p in base:  # the pair invariant on the ordinary path
        assert ("wb_temp" in p.modes) == ("wb_tint" in p.modes)
    stats[:, _CAM_COL] = 777.0  # a body that never appeared in the training rows
    for p in prof.predict(stats, test.emb[:n], test.fmt[:n]):
        assert not {"wb_temp", "wb_tint"} & (set(p.point) | set(p.written))
    assert any(WB_BODY_NOTE in p.notes for p in prof.predict(stats, test.emb[:n], test.fmt[:n]))
    # a body with fewer than MIN_BODY_WB white-balance training rows is gated too
    prof2 = EditProfile(**{**prof.__dict__})
    prof2.wb_bodies = {k: 5 for k in prof.wb_bodies}
    stats2 = test.stats[:n].copy()
    assert all("wb_temp" not in p.written for p in prof2.predict(stats2, test.emb[:n], test.fmt[:n]))


def test_auto_white_balance_is_not_an_unchanged_row():
    """Review #6 m5: WhiteBalance="Auto" is a change, so it gives no WB target."""
    import math

    from shotsieve.personal.edit_dataset import targets_from_settings

    y = targets_from_settings({"WhiteBalance": "Auto", "Exposure2012": 0.2}, 0, (200.0, 0.0))
    assert math.isnan(y["raw_temp"]) and math.isnan(y["raw_tint"])
    y = targets_from_settings({"WhiteBalance": "As Shot"}, 0, (200.0, 0.0))
    assert y["raw_temp"] == 0.0
