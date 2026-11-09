"""T21: the grid thumbnail is written while the preview is decoded for extraction."""

from __future__ import annotations

import shutil

from fastapi.testclient import TestClient

from shotsieve.config import Settings
from shotsieve.io.formats import preview_source_for
from shotsieve.pipeline.cache import Cache
from shotsieve.pipeline.runner import run_cull
from shotsieve.server.app import create_app


def test_cull_writes_grid_thumbs_the_server_reuses(shoot_dir, models_ready, tmp_path):
    folder = tmp_path / "shoot"
    shutil.copytree(shoot_dir, folder, ignore=shutil.ignore_patterns(".shotsieve"))
    res = run_cull(folder, Settings())
    thumbs = folder / ".shotsieve" / "thumbs"
    with Cache.for_folder(folder) as c:
        ids = c.ids_lookup([str(r.master_path) for r in res.images])
    assert len(ids) == len(res.images)
    for r in res.images:
        mtime = preview_source_for(r.master_path).stat().st_mtime_ns
        assert (thumbs / f"t_{ids[str(r.master_path)]}_400_{mtime}.jpg").exists(), r.master_path
    before = {p.name: p.stat().st_mtime_ns for p in thumbs.glob("t_*_400_*.jpg")}
    cl = TestClient(create_app(Settings(), folder))
    some = next(iter(ids.values()))
    resp = cl.get(f"/api/thumb/{some}")
    assert resp.status_code == 200
    assert resp.headers["cache-control"] == "max-age=86400, immutable"
    after = {p.name: p.stat().st_mtime_ns for p in thumbs.glob("t_*_400_*.jpg")}
    assert before == after  # served as is, nothing re-rendered
    assert not list(thumbs.glob("*.tmp"))
