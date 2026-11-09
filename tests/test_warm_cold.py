"""A cached (warm) re-run must reproduce the cold run exactly."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import numpy as np

from shotsieve.config import Settings
from shotsieve.pipeline.cache import Cache
from shotsieve.pipeline.runner import run_cull

ROOT = Path(__file__).resolve().parents[1]


def _sig(res):
    return [
        (r.master_path.name, r.decision, tuple(r.reasons), r.group_id, r.burst_id,
         r.audit, r.is_best_in_group, round(r.score, 12))
        for r in res.images
    ]  # fmt: skip


def _cold_warm(folder: Path):
    shutil.rmtree(folder / ".shotsieve", ignore_errors=True)
    s = Settings()
    cold = run_cull(folder, s)
    warm = run_cull(folder, s)
    assert warm.stats["n_cached"] == warm.stats["n_images"] > 0
    assert _sig(cold) == _sig(warm)
    assert {k: sorted(v) for k, v in cold.groups.items()} == {
        k: sorted(v) for k, v in warm.groups.items()
    }
    assert cold.thresholds == warm.thresholds
    for a, b in zip(cold.images, warm.images, strict=True):
        da, db = a.features.to_dict(), b.features.to_dict()
        assert da.keys() == db.keys()
        for k in da:
            assert da[k] == db[k] or (da[k] != da[k] and db[k] != db[k]), k
        assert a.reason_texts == b.reason_texts
    return cold


def test_warm_equals_cold_fixture_shoot(shoot_dir, tmp_path):
    folder = tmp_path / "shoot"
    shutil.copytree(shoot_dir, folder, ignore=shutil.ignore_patterns(".shotsieve"))
    _cold_warm(folder)


def test_warm_equals_cold_benchmark_like(tmp_path):
    spec = importlib.util.spec_from_file_location("benchmark", ROOT / "scripts" / "benchmark.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["benchmark"] = mod
    spec.loader.exec_module(mod)
    folder = tmp_path / "bench"
    mod.synthesize(60, folder, 7)
    _cold_warm(folder)


def test_cold_scene_vector_is_the_cached_vector(shoot_dir, tmp_path, monkeypatch):
    """Root cause guard: what a cold run groups on is bit-identical to what the cache returns."""
    folder = tmp_path / "shoot"
    shutil.copytree(shoot_dir, folder, ignore=shutil.ignore_patterns(".shotsieve"))
    seen: dict[str, np.ndarray] = {}
    orig = Cache.put

    def spy(self, key, size, mtime, rec, fp, extractor):
        seen[key] = rec.scene.copy()
        return orig(self, key, size, mtime, rec, fp, extractor)

    monkeypatch.setattr(Cache, "put", spy)
    run_cull(folder, Settings())
    assert seen
    with Cache.for_folder(folder) as c:
        stored = c.scene_embeddings()
    for key, v in seen.items():
        assert np.array_equal(v, stored[str(folder / key) if key not in stored else key]), key
