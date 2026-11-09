"""T20 regression: `train --lrcat` must not learn ShotSieve's own written ratings / labels."""

from __future__ import annotations

import shutil

from shotsieve.config import Settings
from shotsieve.personal.dataset import build_training_set
from shotsieve.pipeline.runner import run_cull


def test_train_lrcat_learns_nothing_from_our_own_xmp_round_trip(
    shoot_dir, models_ready, make_fixtures, tmp_path
):
    """Cull with --write-xmp, import the result into 'Lightroom' unchanged (ratings and labels
    equal what ShotSieve wrote): the catalog says nothing about the photographer, so training on
    it must yield 0 labels. (Before the fix: every 3-star / Green keep became a user keep.)"""
    folder = tmp_path / "shoot"
    shutil.copytree(shoot_dir, folder, ignore=shutil.ignore_patterns(".shotsieve"))
    s = Settings()
    res = run_cull(folder, s, write_xmp=True)
    entries = []
    for r in res.images:
        d = r.model_decision
        entries.append({
            "root": str(folder), "rel": "", "base": r.master_path.stem,
            "ext": r.master_path.suffix.lstrip("."), "pick": 0,
            "rating": getattr(s, f"rating_{d}"), "label": getattr(s, f"label_{d}"),
        })  # fmt: skip
    assert any(e["rating"] for e in entries)  # there are keeps with stars to (mis)learn
    cat = tmp_path / "round_trip.lrcat"
    make_fixtures.build_lrcat(cat, entries)
    ts = build_training_set(lrcat=cat, settings=s)
    assert len(ts) == 0
