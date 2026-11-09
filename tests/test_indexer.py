"""Background indexer, sqlite-vec storage and the search engine (fake embedders: no models)."""

from __future__ import annotations

import shutil
import threading
import time

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from shotsieve.config import Settings
from shotsieve.features.vector import FeatureVector
from shotsieve.models import text_image
from shotsieve.pipeline import search as search_mod
from shotsieve.pipeline.cache import SIGLIP_DIM, SIGLIP_KIND, Cache
from shotsieve.pipeline.indexer import Indexer, cull_active, cull_running

NAMES = ["burst0_0", "burst0_1", "burst0_2", "landscape", "black", "burst1_3"]


def fake_embed(images):
    """Deterministic 768-d unit vectors derived from the mean colour of each image."""
    out = []
    for im in images:
        rng = np.random.default_rng(int(im.mean() * 1000) % (2**31))
        v = rng.normal(size=SIGLIP_DIM).astype(np.float32)
        out.append(v / np.linalg.norm(v))
    return np.stack(out)


@pytest.fixture()
def folder(tmp_path, shoot_dir):
    d = tmp_path / "shoot"
    d.mkdir()
    for n in NAMES:
        shutil.copy(shoot_dir / f"{n}.jpg", d / f"{n}.jpg")
    return d


def _settings(**kw):
    return Settings(index_batch=2, **kw)


def test_tokenizer_pad_eos_shape():
    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2, "a": 3, "red": 4, "car": 5}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    ids = text_image.tokenize(tok, "A Red CAR")
    assert ids.shape == (1, 64) and ids.dtype == np.int64
    assert ids[0, :4].tolist() == [3, 4, 5, 1]  # lowercased, EOS appended
    assert (ids[0, 4:] == text_image.PAD_ID).all()
    long = text_image.tokenize(tok, " ".join(["car"] * 200))
    assert long.shape == (1, 64) and long[0, -1] == text_image.EOS_ID  # truncated, EOS kept


def test_vec0_and_numpy_agree(folder):
    rng = np.random.default_rng(0)
    with Cache.for_folder(folder) as c:
        if not c._vec_ready():
            pytest.skip("sqlite-vec cannot load in this Python")
        items = []
        for i in range(200):
            v = rng.normal(size=SIGLIP_DIM)
            items.append((str(folder / f"x{i}.jpg"), f"fp{i}", v / np.linalg.norm(v)))
        c.put_siglip(items)
        q = rng.normal(size=SIGLIP_DIM).astype(np.float32)
        q /= np.linalg.norm(q)
        a = c.knn(q, 10, use_vec=True)
        b = c.knn(q, 10, use_vec=False)
        assert [i for i, _ in a] == [i for i, _ in b]
        assert np.allclose([s for _, s in a], [s for _, s in b], atol=1e-4)
        allowed = {i for i, _ in a[:3]}
        assert {i for i, _ in c.knn(q, 10, allowed)} == allowed


def test_indexer_indexes_and_is_idempotent(folder):
    ix = Indexer(folder, _settings(), people=False, embedder=fake_embed)
    ix.run_blocking()
    assert (ix.done, ix.total, ix.error) == (6, 6, None)
    with Cache.for_folder(folder) as c:
        assert c.siglip_count() == 6
    calls = []
    ix2 = Indexer(
        folder, _settings(), people=False, embedder=lambda im: calls.append(1) or fake_embed(im)
    )
    ix2.run_blocking()
    assert calls == [] and ix2.done == 6  # nothing left to do


def test_indexer_resumes_after_kill(folder):
    seen = []

    def dying(images):
        if len(seen) >= 4:
            raise RuntimeError("killed")
        seen.extend([1] * len(images))
        return fake_embed(images)

    ix = Indexer(folder, _settings(), people=False, embedder=dying)
    ix.run_blocking()
    assert ix.error and "killed" in ix.error
    with Cache.for_folder(folder) as c:
        assert c.siglip_count() == 4
    again = []
    ix2 = Indexer(folder, _settings(), people=False,
                  embedder=lambda im: again.extend([1] * len(im)) or fake_embed(im))  # fmt: skip
    ix2.run_blocking()
    assert len(again) == 2  # only the missing rows
    with Cache.for_folder(folder) as c:
        assert c.siglip_count() == 6


def test_indexer_pauses_during_cull(folder):
    ix = Indexer(folder, _settings(), people=False, embedder=fake_embed)
    with cull_active():  # a (mock) cull job holds the gate
        assert cull_running()
        ix.start()
        time.sleep(0.6)
        assert ix.running and ix.paused and ix.done == 0
        assert ix.status()["paused"] is True
    ix.join(20)
    assert not cull_running() and ix.done == 6 and not ix.paused


def test_user_pause_and_resume(folder):
    ix = Indexer(folder, _settings(), people=False, embedder=fake_embed)
    ix.pause()
    ix._user_pause.set()
    ix.start()
    ix.pause()
    time.sleep(0.4)
    assert ix.done == 0 and ix.paused
    ix.resume()
    ix.join(20)
    assert ix.done == 6


def test_gate_is_reentrant():
    with cull_active(), cull_active():
        assert cull_running()
    assert not cull_running()


def test_variant_change_reindexes(folder):
    Indexer(folder, _settings(), people=False, embedder=fake_embed).run_blocking()
    n = []
    int8 = _settings(siglip_vision="int8")
    Indexer(
        folder, int8, people=False, embedder=lambda im: n.extend([1] * len(im)) or fake_embed(im)
    ).run_blocking()
    assert len(n) == 6


# ---- search ------------------------------------------------------------------------------------


class FakeText:
    """Query -> the vector of one named image, so ranking is exact."""

    def __init__(self, vec):
        self.vec = vec

    def embed(self, text):
        return self.vec


def test_parse_query():
    pq = search_mod.parse_query('bride laughing person:"Ana Lee" is:keep jersey:23 camera:SN1')
    assert pq.text == "bride laughing"
    assert pq.filters == {
        "person": ["Ana Lee"], "is": ["keep"], "jersey": ["23"], "camera": ["SN1"]
    }
    assert search_mod.parse_query("is:review").text == ""


def test_search_ranking_margin_and_filters(folder):
    decisions = ["keep", "reject", "keep", "review", "reject", "keep"]
    rows = [
        dict(path=str(folder / f"{n}.jpg"), decision=d, reasons=[], score=0.5, prior=0.5,
             group_id=i, burst_id=i, is_best=True, audit=False, features=FeatureVector().to_dict())
        for i, (n, d) in enumerate(zip(NAMES, decisions, strict=True))
    ]  # fmt: skip
    Indexer(folder, _settings(), people=False, embedder=fake_embed).run_blocking()
    with Cache.for_folder(folder) as c:
        c.save_decisions(rows)
        ids = c.ids_for([str(folder / f"{n}.jpg") for n in NAMES])
        mat = dict(zip(*[c.siglip_matrix()[0].tolist(), c.siglip_matrix()[1]], strict=True))
    target = ids[str(folder / "landscape.jpg")]
    hits = search_mod.search(folder, "anything", 10, text_embedder=FakeText(mat[target]))
    assert hits[0].id == target and hits[0].score == pytest.approx(1.0, abs=1e-3)
    assert len(hits) == 1  # random 768-d vectors: nothing else within the 0.08 margin
    kept = search_mod.search(folder, "anything is:keep", 10, text_embedder=FakeText(mat[target]))
    assert kept == [] or all(h.id != target for h in kept)  # landscape is "review": filtered out
    only_keep = search_mod.search(folder, "is:keep", 10)
    keep_names = {h.path.rsplit("/", 1)[-1] for h in only_keep}
    assert keep_names == {"burst0_0.jpg", "burst0_2.jpg", "burst1_3.jpg"}
    assert search_mod.search(folder, "jersey:23", 10) == []  # no OCR table yet
    with pytest.raises(PermissionError):
        search_mod.search(folder, 'person:"Ana"', 10)


def test_text_tower_lazy_and_unloads():
    t = text_image.SiglipText(idle_unload_s=0.1)
    assert not t.loaded  # nothing is loaded at construction (no model files needed)
    t.unload()
    assert not t.loaded


def test_text_lru(monkeypatch):
    t = text_image.SiglipText()

    class Sess:
        n = 0

        def run(self, _out, feed):
            Sess.n += 1
            return [np.ones((1, SIGLIP_DIM), np.float32)]

    t._session = Sess()
    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2, "dog": 3}
    t._tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    t._tok.pre_tokenizer = pre_tokenizers.Whitespace()
    a = t.embed("Dog")
    b = t.embed("dog ")
    assert Sess.n == 1 and np.allclose(a, b) and abs(np.linalg.norm(a) - 1) < 1e-5
    for i in range(300):
        t.embed(f"dog {i}")
    assert len(t._cache) == 256
    t.unload()


def test_threadsafe_status_while_running(folder):
    ix = Indexer(folder, _settings(), people=False, embedder=fake_embed)
    ev = threading.Event()
    ix.start()
    for _ in range(50):
        ix.status()
        if not ix.running:
            break
        time.sleep(0.05)
    ev.set()
    ix.join(20)
    assert ix.status()["done"] == 6


def test_start_while_the_worker_is_finishing_is_not_lost(tmp_path, monkeypatch):
    """Review #4 m5: start() during the worker's wind-down used to see a live thread and return,
    so the new cull's images were not indexed until the next trigger."""
    from shotsieve.config import Settings
    from shotsieve.pipeline.indexer import Indexer

    ix = Indexer(tmp_path, Settings(), people=False, embedder=fake_embed)
    runs = []
    monkeypatch.setattr(ix, "_index_images", lambda: runs.append(1))
    real_release = ix._release_models
    fired = []

    def release_then_restart():
        real_release()
        if not fired:
            fired.append(1)
            ix.start()  # the worker is still alive here, in its `finally`

    monkeypatch.setattr(ix, "_release_models", release_then_restart)
    ix.start()
    ix.join(10)
    assert len(runs) == 2 and not ix.running


def test_vec0_is_reconciled_with_the_embedding_blobs(folder):
    """Review #4 m4: rows written while sqlite-vec could not load left vec0 partial, and knn
    trusted any non-empty vec0 result."""
    rng = np.random.default_rng(1)

    def items(lo, hi):
        out = []
        for i in range(lo, hi):
            v = rng.normal(size=SIGLIP_DIM)
            out.append((str(folder / f"y{i}.jpg"), f"fp{i}", v / np.linalg.norm(v)))
        return out

    with Cache.for_folder(folder) as c:
        if not c._vec_ready():
            pytest.skip("sqlite-vec cannot load in this Python")
        c.put_siglip(items(0, 20))
        c._vec_ok = False  # the extension "was not loadable" for the next writes
        c.put_siglip(items(20, 60))
    with Cache.for_folder(folder) as c:  # reopened: vec0 has 20 of 60 rows
        q = rng.normal(size=SIGLIP_DIM).astype(np.float32)
        a = c.knn(q, 60, use_vec=True)
        b = c.knn(q, 60, use_vec=False)
        assert len(a) == len(b) == 60
        assert c._con.execute("SELECT COUNT(*) FROM vec_siglip").fetchone()[0] == 60


class _FakeVision:
    def __init__(self):
        self.batch_size = 2

    def embed_batch(self, images):
        return fake_embed(images)


def _embeddings(folder):
    with Cache.for_folder(folder) as c:
        return {
            k: bytes(v)
            for k, v in c._con.execute(
                "SELECT path, data FROM embeddings WHERE kind=? ORDER BY path", (SIGLIP_KIND,)
            ).fetchall()
        }


def test_pause_releases_models_once_and_resume_is_bit_identical(folder, tmp_path, monkeypatch):
    from shotsieve.models import ocr

    monkeypatch.setattr(text_image, "get_vision", lambda variant="fp32": _FakeVision())
    loads = []
    real_release = Indexer._release_models

    def counting(self):
        loads.append(1)
        real_release(self)

    monkeypatch.setattr(Indexer, "_release_models", counting)
    text_image._vision["fp32"] = _FakeVision()  # resident sessions, as after a first batch
    ocr._det, ocr._rec = object(), object()
    try:
        ix = Indexer(folder, _settings(), people=False)
        with cull_active():
            ix.start()
            deadline = time.time() + 1.0
            while (text_image._vision or ocr._det or ocr._rec) and time.time() < deadline:
                time.sleep(0.02)
            assert not text_image._vision and ocr._det is None and ocr._rec is None
            assert ix._embedder is None
            time.sleep(0.5)  # a long pause releases once, not once per 0.1 s tick
            assert len(loads) == 1
        ix.join(20)
        assert ix.error is None and ix.done == 6
        resumed = _embeddings(folder)

        ref = tmp_path / "ref"
        ref.mkdir()
        for n in NAMES:
            shutil.copy(folder / f"{n}.jpg", ref / f"{n}.jpg")
        Indexer(ref, _settings(), people=False).run_blocking()
        assert len(resumed) == 6 and resumed == _embeddings(ref)
    finally:
        text_image._vision.clear()
        ocr._det = ocr._rec = None


def test_pause_after_first_batch_releases_embedder_and_reload_is_bit_identical(
    folder, tmp_path, monkeypatch
):
    """m2: the gate closes AFTER a batch embedded, so `ix._embedder` was really set."""
    made: list[_FakeVision] = []
    gate = cull_active()

    class Vis(_FakeVision):
        def __init__(self):
            super().__init__()
            made.append(self)

        def embed_batch(self, images):
            out = fake_embed(images)
            if len(made) == 1 and not getattr(Vis, "closed", False):
                Vis.closed = True
                gate.__enter__()  # cull starts right after this batch
            return out

    def get_vision(variant="fp32"):
        v = Vis()
        text_image._vision["fp32"] = v
        return v

    monkeypatch.setattr(text_image, "get_vision", get_vision)
    try:
        ix = Indexer(folder, _settings(), people=False)
        ix.start()
        deadline = time.time() + 5.0
        while not (Vis.__dict__.get("closed") and ix._embedder is None) and time.time() < deadline:
            time.sleep(0.02)
        assert Vis.closed and ix._embedder is None and not text_image._vision
        assert ix.done == 2  # exactly the first batch (batch size 2) was embedded
        gate.__exit__(None, None, None)
        ix.join(20)
        assert ix.error is None and ix.done == 6 and len(made) == 2  # reloaded once after release
        resumed = _embeddings(folder)
        ref = tmp_path / "ref"
        ref.mkdir()
        for n in NAMES:
            shutil.copy(folder / f"{n}.jpg", ref / f"{n}.jpg")
        monkeypatch.setattr(text_image, "get_vision", lambda variant="fp32": _FakeVision())
        Indexer(ref, _settings(), people=False).run_blocking()
        assert len(resumed) == 6 and resumed == _embeddings(ref)
    finally:
        text_image._vision.clear()


def test_pause_during_ocr_frees_the_real_det_and_rec_objects(folder, monkeypatch):
    """m1: `_index_ocr` must not keep det/rec alive across a pause (weakref, not just globals)."""
    import gc
    import weakref

    from shotsieve.features import jersey as jf
    from shotsieve.models import face_detect, ocr

    class Det:
        pass

    class Rec:
        pass

    class FaceDet:
        def detect(self, img, max_faces=8):
            return []

    refs: list[tuple[weakref.ref, weakref.ref]] = []
    used: list[tuple[int, int]] = []
    gate = cull_active()

    def make_det(threads=4):
        return Det()

    def make_rec(threads=4):
        return Rec()

    def read_numbers(img, det, rec, faces):
        used.append((id(det), id(rec)))
        if len(used) == 1:
            refs.append((weakref.ref(det), weakref.ref(rec)))
            gate.__enter__()  # a cull starts while this image is being read
        return []

    monkeypatch.setattr(ocr, "TextDetector", make_det)
    monkeypatch.setattr(ocr, "TextRecognizer", make_rec)
    monkeypatch.setattr(jf, "read_numbers", read_numbers)
    monkeypatch.setattr(face_detect, "get_face_detector", lambda: FaceDet())
    ocr._det = ocr._rec = None
    try:
        ix = Indexer(folder, _settings(), images=False, people=False, ocr=True)
        ix.start()
        deadline = time.time() + 5.0
        while not refs and time.time() < deadline:
            time.sleep(0.02)
        assert refs
        deadline = time.time() + 1.0
        while time.time() < deadline:
            gc.collect()
            if refs[0][0]() is None and refs[0][1]() is None:
                break
            time.sleep(0.02)
        assert ocr._det is None and ocr._rec is None
        assert refs[0][0]() is None and refs[0][1]() is None  # the real objects are gone
        gate.__exit__(None, None, None)
        ix.join(20)
        assert ix.error is None and ix.done == 6
        assert len(used) == 6 and used[1] != used[0]  # reloaded lazily after the release
    finally:
        ocr._det = ocr._rec = None
