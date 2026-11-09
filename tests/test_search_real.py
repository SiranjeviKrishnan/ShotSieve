"""End-to-end search quality with the real SigLIP2 models: recall@10 on hand-labelled queries."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from shotsieve.config import Settings
from shotsieve.pipeline.cache import Cache
from shotsieve.pipeline.indexer import Indexer
from shotsieve.pipeline.search import search

pytestmark = [pytest.mark.models]
QUERIES = Path(__file__).parent / "fixtures" / "search_queries.json"


def _variants(img: np.ndarray) -> list[np.ndarray]:
    h, w = img.shape[:2]
    return [
        img,
        img[:, ::-1],
        img[int(h * 0.1) : int(h * 0.9), int(w * 0.1) : int(w * 0.9)],
        cv2.convertScaleAbs(img, alpha=0.8, beta=-10),
        img[:, : int(w * 0.75)],
    ]


@pytest.fixture(scope="module")
def corpus(tmp_path_factory, search_ready, search_images):
    folder = tmp_path_factory.mktemp("searchshoot")
    for name, img in search_images.items():
        for i, v in enumerate(_variants(img)):
            cv2.imwrite(
                str(folder / f"{name}__{i}.jpg"),
                cv2.cvtColor(np.ascontiguousarray(v), cv2.COLOR_RGB2BGR),
            )
    ix = Indexer(folder, Settings(), people=False)
    ix.run_blocking()
    assert ix.error is None and ix.done == ix.total == 5 * len(search_images)
    return folder


def test_recall_at_10_on_labelled_queries(corpus):
    queries = json.loads(QUERIES.read_text())["queries"]
    assert len(queries) == 30
    recalls, misses = [], []
    for item in queries:
        hits = search(corpus, item["q"], k=10)
        ranked = [Path(h.path).name.split("__")[0] for h in hits]
        rel = {f"{item['relevant']}"}
        # the corpus has 5 variants per subject: count variants, not subjects
        got = sum(1 for n in ranked[:10] if n in rel)
        r = got / 5
        recalls.append(r)
        if r < 0.6:
            misses.append((item["q"], got))
    mean = float(np.mean(recalls))
    print(f"mean recall@10 = {mean:.3f}; weak: {misses}")
    assert mean >= 0.8, (mean, misses)
    assert (np.array(recalls) >= 0.6).mean() >= 0.8


def test_vec0_matches_numpy_on_real_index(corpus):
    from shotsieve.models.text_image import get_text

    q = get_text().embed("a dog")
    with Cache.for_folder(corpus) as c:
        if not c._vec_ready():
            pytest.skip("sqlite-vec unavailable")
        a = [i for i, _ in c.knn(q, 10, use_vec=True)]
        b = [i for i, _ in c.knn(q, 10, use_vec=False)]
    assert a == b


def test_search_margin_trims_irrelevant(corpus):
    hits = search(corpus, "a red car", k=100)
    assert 1 <= len(hits) < 65
    assert Path(hits[0].path).name.startswith("red_car")
