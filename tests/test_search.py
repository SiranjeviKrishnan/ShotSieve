"""Search / index HTTP API with fake embedders (no model files)."""

from __future__ import annotations

import shutil
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from shotsieve.config import Settings
from shotsieve.pipeline.cache import Cache
from shotsieve.server.app import create_app
from tests.test_indexer import NAMES, FakeText, fake_embed


@pytest.fixture()
def shoot(tmp_path, shoot_dir):
    d = tmp_path / "shoot"
    d.mkdir()
    for n in NAMES:
        shutil.copy(shoot_dir / f"{n}.jpg", d / f"{n}.jpg")
    return d


def _seed_decisions(folder):
    from shotsieve.features.vector import FeatureVector

    rows = [
        dict(path=str(folder / f"{n}.jpg"), decision="keep", reasons=[], score=0.5, prior=0.5,
             group_id=i, burst_id=i, is_best=True, audit=False, features=FeatureVector().to_dict())
        for i, n in enumerate(NAMES)
    ]  # fmt: skip
    with Cache.for_folder(folder) as c:
        c.save_decisions(rows)


def test_index_start_status_search(shoot, monkeypatch):
    from shotsieve.models import text_image
    from shotsieve.pipeline import indexer as indexer_mod

    _seed_decisions(shoot)
    monkeypatch.setattr(text_image, "search_models_ready", lambda: True)
    monkeypatch.setattr(indexer_mod.Indexer, "_get_embedder", lambda self: fake_embed)
    app = create_app(Settings(index_batch=2), default_folder=shoot)
    with TestClient(app) as c:
        st = c.post("/api/index/start", json={"people": False}).json()
        assert "task" in st
        for _ in range(100):
            s = c.get("/api/index/status").json()
            if not s["running"]:
                break
            time.sleep(0.1)
        assert s["done"] == 6 and s["total"] == 6 and s["error"] is None
        assert c.get("/api/index/status").json()["indexed"] == 6
        with Cache.for_folder(shoot) as cache:
            ids, mat = cache.siglip_matrix()
        monkeypatch.setattr(text_image, "get_text", lambda: FakeText(mat[2]))
        body = c.get("/api/search", params={"q": "whatever", "k": 5}).json()
        assert body["hits"][0]["id"] == int(ids[2])
        assert c.get("/api/search", params={"q": "is:keep"}).json()["hits"]
        assert c.post("/api/index/pause").status_code == 200
        assert c.post("/api/index/bogus").status_code == 404


def test_search_503_without_models(shoot, monkeypatch):
    from shotsieve.models import text_image

    _seed_decisions(shoot)
    monkeypatch.setattr(text_image, "search_models_ready", lambda: False)
    with TestClient(create_app(Settings(search_index_auto=False), default_folder=shoot)) as c:
        assert c.get("/api/search", params={"q": "a dog"}).status_code == 503
        assert c.post("/api/index/start").status_code == 503
        assert c.get("/api/health").json()["search_ready"] in (True, False)


def test_search_requires_cull(tmp_path, monkeypatch):
    d = tmp_path / "empty"
    d.mkdir()
    with TestClient(create_app(Settings(), default_folder=d)) as c:
        assert c.get("/api/search", params={"q": "x"}).status_code == 404


def test_recall_helpers():
    from shotsieve.pipeline.search import knn_agreement, recall_at_k

    assert recall_at_k(["a", "b", "c"], {"a", "c"}, 10) == 1.0
    assert recall_at_k(["x", "b"], {"a", "c"}, 10) == 0.0
    assert knn_agreement(np.arange(20), np.arange(20)) == 1.0
