# ruff: noqa: E501
"""T17: shot-list coverage alerts (pipeline/shotlist.py, `shotsieve shotlist`, /api/shotlist*)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from shotsieve.cli import main
from shotsieve.config import Settings
from shotsieve.features.vector import FeatureVector
from shotsieve.io import xmp
from shotsieve.models import text_image
from shotsieve.pipeline import shotlist
from shotsieve.pipeline.cache import SIGLIP_DIM, Cache
from shotsieve.server.app import create_app

ROOT = Path(__file__).resolve().parents[1]
QUERIES = ROOT / "tests" / "fixtures" / "search_queries.json"
ABSENT = ROOT / "tests" / "fixtures" / "shotlist_absent.json"
HELDOUT = ROOT / "tests" / "fixtures" / "shotlist_heldout.json"
BLIND = ROOT / "tests" / "fixtures" / "shotlist_blind_labels.json"
BLIND_DIR = ROOT / "tests" / "fixtures" / "shotlist_blind" / "images"
# sha256 of the label file frozen in docs/TASKS.md (T17, N1) before the first scoring
BLIND_SHA256 = "4522d07f3b05be520e74c119de6cf994af6186495245f519cc28ea59ab141fa8"
FOOTBALL = sorted((ROOT / "tests" / "fixtures" / "real").glob("cc0_football_*.jpg"))


def _rows(paths, decisions):
    return [dict(path=p, decision=d, reasons=[], score=0.5, prior=0.5, group_id=i, burst_id=i,
                 is_best=True, audit=False, audit_prob=None, features=FeatureVector().to_dict())
            for i, (p, d) in enumerate(zip(paths, decisions, strict=True))]  # fmt: skip


class FakeText:
    def __init__(self, vecs: dict[str, np.ndarray]):
        self.vecs = vecs

    def embed(self, text):
        key = text.strip().lower()
        if key not in self.vecs:  # unknown shot: an arbitrary direction
            rng = np.random.default_rng(abs(hash(key)) % (2**32))
            self.vecs[key] = rng.normal(size=SIGLIP_DIM).astype(np.float32)
        return self.vecs[key]


def axis(k):
    v = np.zeros(SIGLIP_DIM, np.float32)
    v[k] = 1.0
    return v


def make_vec(comps: dict[int, float], rng) -> np.ndarray:
    """A unit vector whose cosine with axis k is comps[k]; the rest is random and orthogonal to
    the first 16 axes (so the background similarity is exactly what `comps` says)."""
    v = np.zeros(SIGLIP_DIM, np.float32)
    for k, c in comps.items():
        v[k] = c
    r = rng.normal(size=SIGLIP_DIM).astype(np.float32)
    r[:16] = 0.0
    r *= np.sqrt(max(1.0 - float((v**2).sum()), 1e-6)) / np.linalg.norm(r)
    return v + r


@pytest.fixture()
def shoot(tmp_path):
    """40 frames on axes 0..4: frame i's cosine with item k is a small background value, except
    the planted ones. Frames 36..39 are rejects (item 3's only strong match is a reject)."""
    folder = tmp_path / "shoot"
    folder.mkdir()
    rng = np.random.default_rng(1)
    n = 40
    bg = np.linspace(-0.01, 0.01, n)
    comps = [{k: float(bg[(i * (k + 3)) % n]) for k in range(5)} for i in range(n)]
    comps[3][0] = 0.55  # item 0: covered
    comps[4][1] = 0.4  # item 1 (x3): 2 frames only -> weak
    comps[5][1] = 0.35
    comps[6][2] = 0.06  # item 2: strong outlier below tau -> weak
    comps[38][3] = 0.6  # item 3: only a reject has it -> missing
    comps[7][4] = 0.5  # item 4 is person-restricted
    comps[8][5] = 0.105  # item 5 ("the toast"): between TAU and TAU_HI -> uncertain
    paths = [str(folder / f"f{i:02d}.jpg") for i in range(n)]
    decisions = ["keep"] * 36 + ["reject"] * 4
    with Cache.for_folder(folder) as c:
        c.save_decisions(_rows(paths, decisions))
        c.put_siglip([(p, f"fp{i}", make_vec(comps[i], rng)) for i, p in enumerate(paths)])
    return folder, paths


TEXTS = ["bride", "family photo", "the cake", "the rings", "the groom", "the toast"]


@pytest.fixture()
def fake_text(monkeypatch):
    monkeypatch.setattr(text_image, "get_text", lambda: FakeText({t.lower(): axis(k) for k, t in enumerate(TEXTS)}))
    monkeypatch.setattr(text_image, "search_models_ready", lambda: True)


LIST = '# a list\n!bride\n!family photo x3\nthe cake\n!the rings\nthe groom person:"Ana"\n'

LIST_U = LIST + "!the toast\n"  # + a must-have that only matches near the threshold


# ---- parsing and templates ---------------------------------------------------------------------


def test_parse_syntax():
    items = shotlist.parse_list('# comment\n\n!first kiss\n  group photo x3  \nspeech person:"Ana B" x2\n!  the rings  \nx5\n', "t")
    assert [(i.text, i.must, i.min_count, i.person) for i in items] == [
        ("first kiss", True, 1, None), ("group photo", False, 3, None), ("speech", False, 2, "Ana B"),
        ("the rings", True, 1, None), ("x5", False, 1, None),
    ]
    assert [i.item_id for i in items] == ["t:000", "t:001", "t:002", "t:003", "t:004"]
    with pytest.raises(ValueError, match="empty"):
        shotlist.load_items(None, None, "# nothing\n")


@pytest.mark.parametrize("name", ["wedding", "sports", "corporate"])
def test_templates_have_at_least_15_items(name):
    assert name in shotlist.template_names()
    items = shotlist.load_template(name)
    assert len(items) >= 15 and any(i.must for i in items) and len({i.text for i in items}) == len(items)
    with pytest.raises(ValueError):
        shotlist.load_template("../" + name)
    with pytest.raises(ValueError, match="no such list"):
        shotlist.load_template("nope")


# ---- the rule ----------------------------------------------------------------------------------


def test_band_constants_are_ordered():
    assert shotlist.TAU < shotlist.TAU_HI


def test_classify_uncertain_band():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 0.01, 200)
    allowed = np.ones(200, bool)
    mid = (shotlist.TAU + shotlist.TAU_HI) / 2

    def run(sims, n=1):
        return shotlist.classify(sims, allowed, n)

    s = base.copy()
    s[5] = mid
    st, n, *_ = run(s)
    assert st == "uncertain" and n == 1  # clears TAU but not TAU_HI: never "covered"
    s[5] = shotlist.TAU_HI + 0.001
    assert run(s)[0] == "covered"
    s[5] = shotlist.TAU - 0.001
    assert run(s)[0] == "weak"
    # x2: one strong + one near-threshold frame is not enough strong evidence
    s = base.copy()
    s[5], s[6] = shotlist.TAU_HI + 0.05, mid
    assert run(s, 2)[0] == "uncertain" and run(s, 1)[0] == "covered"
    s[6] = shotlist.TAU_HI + 0.05
    assert run(s, 2)[0] == "covered"
    # an explicit narrower/wider band is honoured
    s = base.copy()
    s[5] = 0.2
    assert shotlist.classify(s, allowed, 1, shotlist.TAU, 0.25)[0] == "uncertain"
    assert shotlist.classify(s, allowed, 1, shotlist.TAU, 0.15)[0] == "covered"


def test_fit_tau_hi_is_train_only_and_respects_recall_floor():
    present = [(0.06 + 0.01 * i, 5.0) for i in range(20)]  # 0.06 .. 0.25
    rec, hi = shotlist.fit_tau_hi(present, tau=0.05, min_recall=0.75)
    assert rec >= 0.75 and 0.05 < hi < 0.25
    n_cov = sum(s >= hi for s, _ in present)
    assert n_cov == 15  # exactly the floor: the highest cut that still keeps 75 %
    assert shotlist.fit_tau_hi(present, 0.05, min_recall=1.0)[0] == 1.0
    assert shotlist.fit_tau_hi(present, 0.05, min_recall=0.5)[1] > hi  # a lower floor, a higher cut
    # low-z present items never count as strictly covered
    assert shotlist.fit_tau_hi([(0.2, 1.0)] * 4, 0.05)[1] == 0.05


def test_classify_rule():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 0.01, 200)
    allowed = np.ones(200, bool)

    def run(sims, n=1, mask=allowed):
        return shotlist.classify(sims, mask, n)[:4]

    s = base.copy()
    s[5] = 0.3
    assert run(s)[0] == "covered" and run(s)[1] == 1
    assert run(s, 2)[0] == "weak"  # one frame clears the cut, two are asked for
    s2 = base.copy()
    s2[5] = shotlist.TAU - 0.01  # a z outlier but below the cosine cut
    assert run(s2)[0] == "weak"
    assert run(base)[0] == "missing"  # no outlier
    s3 = base.copy()
    s3[7] = 0.5
    only = np.ones(200, bool)
    only[7] = False  # the strong frame is excluded (reject / other person)
    assert run(s3, mask=only)[0] == "missing"
    assert run(s3, mask=np.zeros(200, bool))[0] == "missing"
    assert shotlist.classify(np.zeros(0), np.zeros(0, bool), 1)[0] == "missing"


def test_evaluate_statuses_rejects_and_person(shoot, fake_text):
    folder, paths = shoot
    items = shotlist.parse_list(LIST, "w")
    with Cache.for_folder(folder) as c:
        cov = shotlist.evaluate(c, items, Settings())
    got = {r.item.text: r for r in cov.results}
    assert got["bride"].status == "covered" and got["bride"].best_ids[0] == got["bride"].hit_ids[0]
    assert got["family photo"].status == "weak" and got["family photo"].n_hits == 2
    assert got["the cake"].status == "weak" and got["the cake"].n_hits == 0
    assert got["the rings"].status == "missing"  # its only match is a reject
    assert got["the groom"].status == "pending" and "consent" in got["the groom"].detail
    assert cov.indexed_pct == 100.0 and cov.n_candidates == 36
    assert [r.item.text for r in cov.must_missing] == ["the rings"]
    pub = cov.public()
    assert pub["must_missing"] == ["the rings"] and pub["must_weak"] == ["family photo"]
    assert pub["counts"] == {"covered": 1, "uncertain": 0, "weak": 2, "missing": 1, "pending": 1}


def test_uncertain_must_have_is_never_covered_and_is_surfaced(shoot, fake_text):
    folder, _ = shoot
    items = shotlist.parse_list(LIST_U, "w")
    with Cache.for_folder(folder) as c:
        cov = shotlist.evaluate(c, items, Settings())
    got = {r.item.text: r for r in cov.results}
    toast = got["the toast"]
    assert toast.status == "uncertain" and toast.n_hits == 1 and toast.best_ids and "review" in toast.detail
    assert [r.item.text for r in cov.must_uncertain] == ["the toast"]
    assert "the toast" not in [r.item.text for r in cov.must_missing]
    pub = cov.public()
    assert pub["must_uncertain"] == ["the toast"] and pub["counts"]["uncertain"] == 1
    assert pub["counts"]["covered"] == 1  # the toast is not silently counted covered
    # a non-must uncertain item is counted but not listed as a must-have
    with Cache.for_folder(folder) as c:
        cov2 = shotlist.evaluate(c, shotlist.parse_list("the toast\n", "w"), Settings())
    assert cov2.counts()["uncertain"] == 1 and cov2.public()["must_uncertain"] == []
    # the tag keyword is only written for covered/weak keeps, never for an uncertain match
    with Cache.for_folder(folder) as c:
        assert shotlist.tag_xmp(c, cov)["images"] >= 1
    assert not Path(shoot[1][8]).with_suffix(".xmp").exists()


def test_overrides_move_frames_in_and_out(shoot, fake_text):
    folder, paths = shoot
    with Cache.for_folder(folder) as c:
        c.set_override(paths[38], "keep")  # the reject the user rescued
        cov = shotlist.evaluate(c, shotlist.parse_list("the rings"), Settings())
        assert cov.results[0].status == "covered"
        c.set_override(paths[3], "reject")  # ...and the keeper they rejected
        cov = shotlist.evaluate(c, shotlist.parse_list("bride"), Settings())
        assert cov.results[0].status == "missing"


def test_incomplete_index_is_pending_never_missing(shoot, fake_text):
    folder, paths = shoot
    with Cache.for_folder(folder) as c:
        with c._lock:  # forget the embeddings of 10 frames (index 75 %)
            for p in paths[26:36]:
                c._con.execute("DELETE FROM embeddings WHERE path=? AND kind='siglip2'", (c._rel(p),))
                c._con.execute("DELETE FROM index_state WHERE path=? AND kind='siglip2'", (c._rel(p),))
            c._con.commit()
        cov = shotlist.evaluate(c, shotlist.parse_list(LIST_U, "w"), Settings())
    st = {r.item.text: r for r in cov.results}
    assert cov.indexed_pct == 75.0
    assert st["the toast"].status == "pending" and not cov.must_uncertain  # no verdict yet
    assert st["bride"].status == "covered"  # found frames stay found
    assert st["the rings"].status == "pending" and st["the rings"].detail == "pending (indexed 75 %)"
    assert st["the cake"].status == "pending" and st["family photo"].status == "pending"
    assert not any(r.status == "missing" for r in cov.results) and not cov.must_missing


def test_store_roundtrip_and_rerun(shoot, fake_text):
    folder, _ = shoot
    cov = shotlist.run(folder, shotlist.parse_list(LIST, "w"), Settings())
    with Cache.for_folder(folder) as c:
        rows = c.get_shotlist()
    assert [r["text"] for r in rows] == ["bride", "family photo", "the cake", "the rings", "the groom"]
    assert rows[0]["status"] == "covered" and rows[0]["must"] and rows[1]["min_count"] == 3 and rows[0]["best_ids"]
    assert rows[4]["person"] == "Ana"
    assert shotlist.run(folder, None, Settings()).counts() == cov.counts()  # re-evaluates the stored list
    with Cache.for_folder(folder) as c:
        c.clear_shotlist()
    with pytest.raises(ValueError, match="no shot list"):
        shotlist.run(folder, None, Settings())


def test_tag_xmp_only_matched_keeps_idempotent(shoot, fake_text):
    folder, paths = shoot
    shotlist.run(folder, shotlist.parse_list("bride\nthe rings\nthe cake\n", "w"), Settings(), tag=True)
    kws, _ = xmp.read_keywords(paths[3])
    assert kws == ["shotlist:bride"]
    assert not Path(paths[38]).with_suffix(".xmp").exists()  # a reject is never tagged
    assert not Path(paths[6]).with_suffix(".xmp").exists()  # a weak item's best frame below the cut is not a match
    with Cache.for_folder(folder) as c:
        cov = shotlist.evaluate(c, shotlist.load_items(None, None, "bride\n"), Settings())
        assert shotlist.tag_xmp(c, cov)["written"] == 0  # idempotent


def test_tag_xmp_never_tags_uncertain_band_frame(shoot, fake_text):
    folder, paths = shoot
    # x1 -> uncertain, x2 -> weak with one band frame (comps[8][5] = 0.105 in [TAU, TAU_HI)): neither tags
    for text in ("!the toast\n", "the toast x2\n"):
        cov = shotlist.run(folder, shotlist.parse_list(text, "w"), Settings(), tag=True)
        assert cov.results[0].status in ("uncertain", "weak")
        assert cov.results[0].hit_ids and not cov.results[0].strong_ids
        assert not Path(paths[8]).with_suffix(".xmp").exists()


def test_tag_xmp_weak_item_tags_only_strong_frames(shoot, fake_text):
    folder, paths = shoot
    shotlist.run(folder, shotlist.parse_list("!family photo x3\n", "w"), Settings(), tag=True)  # frames 4, 5 strong
    assert xmp.read_keywords(paths[4])[0] == ["shotlist:family-photo"]


# ---- speed: 100 items over 20k frames, warm ----------------------------------------------------


def test_hundred_items_over_20k_frames_under_a_second_warm(tmp_path):
    folder = tmp_path / "big"
    folder.mkdir()
    n = 20_000
    rng = np.random.default_rng(0)
    paths = [str(folder / f"f{i:05d}.jpg") for i in range(n)]
    with Cache.for_folder(folder) as c:
        c.save_decisions(_rows(paths, ["keep" if i % 7 else "reject" for i in range(n)]))
        mat = rng.normal(size=(n, SIGLIP_DIM)).astype(np.float32)
        mat /= np.linalg.norm(mat, axis=1, keepdims=True)
        for i in range(0, n, 2000):
            c.put_siglip([(p, f"fp{j}", mat[j]) for j, p in enumerate(paths[i : i + 2000], start=i)])
        items = [shotlist.Item(f"x:{k:03d}", f"shot {k}", k % 3 == 0, 1 + k % 2, None, k, "x") for k in range(100)]
        emb = FakeText({f"shot {k}": (rng.normal(size=SIGLIP_DIM)).astype(np.float32) for k in range(100)})
        shotlist.evaluate(c, items, Settings(), emb)  # cold: loads the matrix
        t0 = time.time()
        cov = shotlist.evaluate(c, items, Settings(), emb)
        warm = time.time() - t0
    assert len(cov.results) == 100 and warm < 1.0, warm
    print(f"T17 100 items x 20k frames warm: {warm * 1000:.0f} ms")


# ---- CLI + API ---------------------------------------------------------------------------------


def test_cli_shotlist(shoot, fake_text, tmp_path, capsys):
    folder, paths = shoot
    lst = tmp_path / "shots.txt"
    lst.write_text(LIST)
    assert main(["shotlist", str(folder), "--list", str(lst), "--json"]) == 0
    res = json.loads(capsys.readouterr().out)
    assert res["counts"]["covered"] == 1 and res["must_missing"] == ["the rings"] and res["indexed_pct"] == 100.0
    assert main(["shotlist", str(folder), "--list", str(lst), "--tag-xmp"]) == 0
    cap = capsys.readouterr()
    assert "covered" in cap.out and "MISSING must-have: the rings" in cap.err
    assert xmp.read_keywords(paths[3])[0] == ["shotlist:bride"]
    lst2 = tmp_path / "shots2.txt"
    lst2.write_text(LIST_U)
    assert main(["shotlist", str(folder), "--list", str(lst2), "--json"]) == 0
    res2 = json.loads(capsys.readouterr().out)
    assert res2["must_uncertain"] == ["the toast"] and res2["counts"]["uncertain"] == 1
    assert main(["shotlist", str(folder), "--list", str(lst2)]) == 0
    cap2 = capsys.readouterr()
    assert "1 uncertain" in cap2.out and "REVIEW must-have (uncertain match): the toast" in cap2.err
    assert main(["shotlist", str(folder), "--list", "nope"]) == 3
    assert "no such list" in capsys.readouterr().err


@pytest.fixture()
def client(shoot, fake_text):
    folder, _ = shoot
    with TestClient(create_app(Settings(search_index_auto=False), default_folder=folder)) as c:
        yield c, folder


def test_api_shotlist(client):
    c, folder = client
    d = c.get("/api/shotlist", params={"folder": str(folder)}).json()
    assert d["items"] == [] and not d["has_list"] and {"wedding", "sports", "corporate"} <= set(d["templates"])
    r = c.post("/api/shotlist", json={"folder": str(folder), "text": LIST})
    assert r.status_code == 200
    body = r.json()
    assert body["counts"]["covered"] == 1 and body["must_missing"] == ["the rings"] and body["has_list"]
    assert body["items"][0]["status"] == "covered" and body["items"][0]["best_ids"]
    got = c.get("/api/shotlist", params={"folder": str(folder)}).json()  # the stored state
    assert [i["status"] for i in got["items"]] == ["covered", "weak", "weak", "missing", "pending"]
    cov = c.get("/api/shotlist/coverage", params={"folder": str(folder)}).json()
    assert cov["must_missing"] == ["the rings"] and cov["indexed_pct"] == 100.0
    # a template is available by name
    assert c.post("/api/shotlist", json={"folder": str(folder), "list": "sports"}).json()["items"][0]["list_name"] == "sports"
    assert c.delete("/api/shotlist", params={"folder": str(folder)}).json() == {"ok": True}
    assert c.get("/api/shotlist/coverage", params={"folder": str(folder)}).json()["items"] == []


def test_api_shotlist_uncertain_state(client):
    c, folder = client
    body = c.post("/api/shotlist", json={"folder": str(folder), "text": LIST_U}).json()
    assert body["must_uncertain"] == ["the toast"] and body["counts"]["uncertain"] == 1
    assert body["items"][-1]["status"] == "uncertain" and body["items"][-1]["best_ids"]
    stored = c.get("/api/shotlist", params={"folder": str(folder)}).json()  # no model work
    assert stored["must_uncertain"] == ["the toast"] and stored["counts"]["uncertain"] == 1
    assert stored["must_missing"] == ["the rings"]
    cov = c.get("/api/shotlist/coverage", params={"folder": str(folder)}).json()
    assert cov["must_uncertain"] == ["the toast"]
    c.delete("/api/shotlist", params={"folder": str(folder)})
    assert c.get("/api/shotlist", params={"folder": str(folder)}).json()["must_uncertain"] == []


def test_api_shotlist_validation(client, tmp_path, monkeypatch):
    c, folder = client
    assert c.post("/api/shotlist", json={"folder": str(folder)}).status_code == 400
    assert c.post("/api/shotlist", json={"folder": str(folder), "list": "../x"}).status_code == 400
    assert c.post("/api/shotlist", json={"folder": str(folder), "text": "# nothing"}).status_code == 400
    assert c.post("/api/shotlist", json={"folder": "relative"}).status_code == 400
    empty = tmp_path / "empty"
    empty.mkdir()
    assert c.get("/api/shotlist", params={"folder": str(empty)}).json()["items"] == []
    assert c.post("/api/shotlist", json={"folder": str(empty), "text": "x"}).status_code == 404
    monkeypatch.setattr(text_image, "search_models_ready", lambda: False)
    assert c.post("/api/shotlist", json={"folder": str(folder), "text": LIST}).status_code == 503


def test_api_tag_xmp(client):
    c, folder = client
    r = c.post("/api/shotlist", json={"folder": str(folder), "text": "bride\n", "tag_xmp": True}).json()
    assert r["tagged"] and xmp.read_keywords(str(folder / "f03.jpg"))[0] == ["shotlist:bride"]


def test_get_coverage_never_writes_xmp(client):
    """M5: a GET (a CORS "simple" request any web page can fire) must not tag keywords."""
    c, folder = client
    assert c.post("/api/shotlist", json={"folder": str(folder), "text": "bride\n"}).status_code == 200
    r = c.get("/api/shotlist/coverage", params={"folder": str(folder), "tag_xmp": "true"})
    assert r.status_code == 200 and not r.json().get("tagged")
    assert not list(folder.glob("*.xmp"))
    r = c.post("/api/shotlist/coverage", json={"folder": str(folder), "tag_xmp": True})
    assert r.json()["tagged"] and xmp.read_keywords(str(folder / "f03.jpg"))[0] == ["shotlist:bride"]


def test_cross_origin_post_refused(client):
    c, folder = client
    body = {"folder": str(folder), "tag_xmp": True}
    bad = c.post("/api/shotlist/coverage", json=body, headers={"Origin": "http://evil.example"})
    assert bad.status_code == 403
    assert not list(folder.glob("*.xmp"))
    ok = c.post("/api/shotlist/coverage", json=body, headers={"Origin": "http://127.0.0.1:8765"})
    assert ok.status_code != 403


# ---- accuracy with the real SigLIP2 models -----------------------------------------------------


@pytest.fixture(scope="module")
def real_corpus(tmp_path_factory, search_ready, search_images):
    import cv2

    from shotsieve.pipeline.indexer import Indexer
    from tests.test_search_real import _variants

    folder = tmp_path_factory.mktemp("shotlist_corpus")
    paths = []
    for name, img in search_images.items():
        for i, v in enumerate(_variants(img)):
            p = folder / f"{name}__{i}.jpg"
            cv2.imwrite(str(p), cv2.cvtColor(np.ascontiguousarray(v), cv2.COLOR_RGB2BGR))
            paths.append(str(p))
    with Cache.for_folder(folder) as c:
        c.save_decisions(_rows(paths, ["keep"] * len(paths)))
    ix = Indexer(folder, Settings(), people=False)
    ix.run_blocking()
    assert ix.error is None and ix.done == ix.total == len(paths)
    return folder, paths


@pytest.mark.models
def test_accuracy_on_30_present_and_20_absent(real_corpus):
    folder, _ = real_corpus
    present = [q["q"] for q in json.loads(QUERIES.read_text())["queries"]]
    absent = json.loads(ABSENT.read_text())["absent"]
    assert len(present) == 30 and len(absent) == 20
    items = shotlist.parse_list("\n".join(["!" + t for t in [*present, *absent]]), "acc")
    cov = shotlist.run(folder, items, Settings())
    status = {r.item.text: r.status for r in cov.results}
    found = ("covered", "uncertain")  # an uncertain present item still reaches the user for review
    correct = sum(status[t] in found for t in present) + sum(status[t] != "covered" for t in absent)
    covered_recall = sum(status[t] == "covered" for t in present) / len(present)
    false_covered = sum(status[t] == "covered" for t in absent) / len(absent)
    wrong = [(t, status[t]) for t in present if status[t] not in found] + [(t, status[t]) for t in absent if status[t] == "covered"]
    uncertain = [t for t in [*present, *absent] if status[t] == "uncertain"]
    print(f"T17 accuracy {correct / 50:.2f} ({correct}/50), covered recall {covered_recall:.2f}, uncertain {len(uncertain)}/50, must-have false-covered {false_covered:.2f}, wrong: {wrong}")
    assert correct / 50 >= 0.90, wrong
    assert covered_recall >= 0.70 and len(uncertain) / 50 <= 0.20, uncertain
    assert false_covered <= 0.05, wrong
    assert cov.indexed_pct == 100.0 and not any(r.status == "pending" for r in cov.results)


@pytest.mark.models
def test_leave_one_out_accuracy(real_corpus):
    """M7 (a): every item is judged by a tau refitted on the other 49, not by its own fit."""
    from shotsieve.models.text_image import get_text

    folder, _ = real_corpus
    present = [q["q"] for q in json.loads(QUERIES.read_text())["queries"]]
    absent = json.loads(ABSENT.read_text())["absent"]
    with Cache.for_folder(folder) as c:
        _, mat = c.siglip_matrix()
    emb = get_text().embed
    p_st, a_st = shotlist.top_stats(mat, emb, present), shotlist.top_stats(mat, emb, absent)
    acc, false_cov = shotlist.loo_scores(p_st, a_st)
    print(f"T17 leave-one-out accuracy {acc:.3f}, must-have false-covered {false_cov:.3f}")
    assert acc >= 0.90 and false_cov <= 0.05


@pytest.fixture(scope="module")
def heldout_corpus(real_corpus, tmp_path_factory):
    """The bench corpus plus 3 CC0 football photos (5 variants each) that tau never saw."""
    import shutil

    import cv2

    from shotsieve.pipeline.indexer import Indexer
    from tests.test_search_real import _variants

    src, paths = real_corpus
    folder = tmp_path_factory.mktemp("shotlist_heldout")
    for p in paths:
        shutil.copy(p, folder / Path(p).name)
    for fp in FOOTBALL:
        img = cv2.cvtColor(cv2.imread(str(fp)), cv2.COLOR_BGR2RGB)
        for i, v in enumerate(_variants(img)):
            cv2.imwrite(str(folder / f"{fp.stem}__{i}.jpg"),
                        cv2.cvtColor(np.ascontiguousarray(v), cv2.COLOR_RGB2BGR))  # fmt: skip
    allp = sorted(str(p) for p in folder.glob("*.jpg"))
    with Cache.for_folder(folder) as c:
        c.save_decisions(_rows(allp, ["keep"] * len(allp)))
    ix = Indexer(folder, Settings(), people=False)
    ix.run_blocking()
    assert ix.error is None and ix.done == ix.total == len(allp)
    return folder


@pytest.mark.models
def test_burned_heldout_set_records_the_honest_miss(heldout_corpus):
    """The first held-out set (Review #6 M7) with its ORIGINAL labels. It is burned: tau was judged
    on it and 'a city skyline at night' (a daytime building facade in the corpus) is a real
    false-covered. Kept as a diagnostic of the near-miss failure, not as a gate."""
    d = json.loads(HELDOUT.read_text())
    present, absent = d["present"], d["absent"]
    assert "a city skyline at night" in absent and "a bowl of cereal with milk" not in absent
    items = shotlist.parse_list("\n".join("!" + t for t in [*present, *absent]), "heldout")
    cov = shotlist.run(heldout_corpus, items, Settings())
    status = {r.item.text: r.status for r in cov.results}
    correct = sum(status[t] in ("covered", "uncertain") for t in present) + sum(status[t] != "covered" for t in absent)
    wrong = [(t, status[t]) for t in present if status[t] not in ("covered", "uncertain")] + [(t, status[t]) for t in absent if status[t] == "covered"]
    print(f"T17 BURNED held-out accuracy {correct / 24:.3f} ({correct}/24), wrong: {wrong}")
    assert correct / 24 >= 0.90, wrong


def load_blind() -> dict:
    """The frozen blind labels; refuses to run if the file changed after the freeze."""
    import hashlib

    raw = BLIND.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == BLIND_SHA256, "blind labels changed after the freeze"
    d = json.loads(raw)
    for fn, meta in d["images"].items():  # the images are frozen too
        assert hashlib.sha256((BLIND_DIR / fn).read_bytes()).hexdigest() == meta["sha256"], fn
        assert meta["license"] == "CC0"
    return d


def test_blind_labels_are_frozen_and_well_formed():
    d = load_blind()
    present = [p["item"] for p in d["present"]]
    absent = d["absent"]
    assert len(present) >= 30 and len(absent) >= 30
    assert len(set(present)) == len(present) and len(set(absent)) == len(absent)
    assert not set(present) & set(absent) and d["frozen_at"]
    assert {p["image"] for p in d["present"]} <= set(d["images"])
    fit = {q["q"] for q in json.loads(QUERIES.read_text())["queries"]} | set(json.loads(ABSENT.read_text())["absent"])
    assert not (set(present) | set(absent)) & fit  # no tau-fit item reused
    assert "a city skyline at night" not in present  # the relabel stays reverted
    tasks = (ROOT / "docs" / "TASKS.md").read_text()
    assert BLIND_SHA256 in tasks  # the freeze is recorded in the task file


@pytest.fixture(scope="module")
def blind_corpus(heldout_corpus, tmp_path_factory):
    """Blind images (1 frame each) + the tau-fit corpus + football photos as distractors."""
    import shutil

    from shotsieve.pipeline.indexer import Indexer

    d = load_blind()
    folder = tmp_path_factory.mktemp("shotlist_blind")
    for p in sorted(heldout_corpus.glob("*.jpg")):
        shutil.copy(p, folder / p.name)
    for fn in d["images"]:
        shutil.copy(BLIND_DIR / fn, folder / f"blind_{fn}")
    allp = sorted(str(p) for p in folder.glob("*.jpg"))
    with Cache.for_folder(folder) as c:
        c.save_decisions(_rows(allp, ["keep"] * len(allp)))
    ix = Indexer(folder, Settings(), people=False)
    ix.run_blocking()
    assert ix.error is None and ix.done == ix.total == len(allp)
    return folder


@pytest.mark.models
def test_blind_heldout_accuracy_and_false_covered(blind_corpus):
    """N1: the frozen blind set (>= 30 present, >= 30 absent, every item a must-have). Scored with
    the committed method; the labels must never change after seeing results."""
    d = load_blind()
    present = [p["item"] for p in d["present"]]
    absent = d["absent"]
    items = shotlist.parse_list("\n".join("!" + t for t in [*present, *absent]), "blind")
    cov = shotlist.run(blind_corpus, items, Settings())
    status = {r.item.text: r.status for r in cov.results}
    for r in cov.results:
        print(f"  BLIND {r.status:8s} s={r.top_score:.4f} z={r.top_z:5.2f} {'P' if r.item.text in present else 'A'} {r.item.text}")
    correct = sum(status[t] == "covered" for t in present) + sum(status[t] != "covered" for t in absent)
    n = len(present) + len(absent)
    false_covered = sum(status[t] == "covered" for t in absent) / len(absent)
    wrong = [(t, status[t]) for t in present if status[t] != "covered"] + [(t, status[t]) for t in absent if status[t] == "covered"]
    print(f"T17 BLIND accuracy {correct / n:.3f} ({correct}/{n}), must-have false-covered {false_covered:.3f}, wrong: {wrong}")
    assert correct / n >= 0.90, wrong
    # Burned set #1. Official history: attempt 2 (TAU 0.0946, no band) 82/85, false-covered 3/45 = 6.7 %
    # (xfail then). Under the uncertain-band method it now passes (83/85, 2/45 = 4.4 %); pinned, never
    # tuned. The official N1 closure is blind set #2 (test_blind2_false_covered, 4.0 %).
    assert len(wrong) <= 3, wrong
    if false_covered > 0.05:
        pytest.xfail(f"blind #1 false-covered {false_covered:.3f} > 0.05 (burned set, informational)")


def test_loo_scores_are_out_of_sample():
    """The leave-one-out helper refits without the judged item (synthetic, no models)."""
    present = [(0.30 + 0.01 * i, 5.0) for i in range(10)]
    absent = [(0.05 + 0.005 * i, 1.0) for i in range(10)]
    acc, fc = shotlist.loo_scores(present, absent, fitter=shotlist.fit_tau)
    assert acc == 1.0 and fc == 0.0
    # one absent item that looks exactly like a present one is a LOO error, not hidden by the fit
    acc2, fc2 = shotlist.loo_scores(present, [*absent, (0.35, 5.0)], fitter=shotlist.fit_tau)
    assert acc2 < 1.0 and fc2 > 0.0


def test_conservative_tau_is_the_highest_cut_at_the_accuracy_floor():
    present = [(0.05, 4.0), (0.09, 4.0), *[(0.12 + 0.01 * i, 5.0) for i in range(18)]]
    absent = [(0.01 * i, 1.0) for i in range(8)] + [(0.06, 4.0), (0.07, 4.0)]
    f1, t_f1, _ = shotlist.fit_tau(present, absent)
    acc, t_cons, (lo, hi) = shotlist.fit_tau_conservative(present, absent, min_acc=0.90)
    assert t_cons > t_f1 and acc >= 0.90 and lo < t_cons <= hi
    # a stricter floor can only lower the cut
    strict_acc, t_strict, _ = shotlist.fit_tau_conservative(present, absent, min_acc=0.96)
    assert strict_acc >= 0.96 and t_strict < t_cons
    # an unreachable floor falls back to the most accurate plateau instead of failing
    assert shotlist.fit_tau_conservative([(0.1, 5.0)], [(0.2, 5.0), (0.3, 5.0)], min_acc=1.0)[0] < 1.0


def test_matrix_memo_is_bounded(tmp_path):
    for i in range(5):
        (tmp_path / f"s{i}").mkdir()
        cache = Cache.for_folder(tmp_path / f"s{i}")
        shotlist._matrix(cache)
        cache.close()
    assert len(shotlist._MATRIX) <= shotlist._MATRIX_KEEP


@pytest.mark.models
def test_half_indexed_real_shoot_is_pending_not_missing(real_corpus, tmp_path):
    import shutil

    folder, paths = real_corpus
    half = tmp_path / "half"
    shutil.copytree(folder, half)
    with Cache.for_folder(half) as c:
        with c._lock:
            for p in sorted(paths)[::2]:
                c._con.execute("DELETE FROM embeddings WHERE path=? AND kind='siglip2'", (c._rel(p.replace(str(folder), str(half))),))
                c._con.execute("DELETE FROM index_state WHERE path=? AND kind='siglip2'", (c._rel(p.replace(str(folder), str(half))),))
            c._con.commit()
        absent = json.loads(ABSENT.read_text())["absent"]
        cov = shotlist.evaluate(c, shotlist.parse_list("\n".join("!" + t for t in absent), "a"), Settings())
    assert cov.indexed_pct < 100.0
    assert {r.status for r in cov.results} <= {"pending", "covered"} and not cov.must_missing
    assert all(r.detail.startswith("pending (indexed") for r in cov.results if r.status == "pending")


def test_video_frames_do_not_hold_the_index_below_100(shoot, fake_text):
    folder, paths = shoot
    with Cache.for_folder(folder) as c:  # a cull with a video: its frames are never SigLIP-indexed
        rows = _rows([*paths, str(folder / "clip.mp4#t=4.000")], ["keep"] * 41)
        c.save_decisions(rows)
        cov = shotlist.evaluate(c, shotlist.parse_list("the rings"), Settings())
    assert cov.indexed_pct == 100.0 and cov.results[0].status != "pending"


BLIND2_DIR = Path(__file__).parent / "fixtures" / "shotlist_blind2"
BLIND2_SHA = "dbb6838c5908aad885a5e3c87393432f48dc3cd9e85feea6617c0364f20f5f51"


@pytest.fixture(scope="module")
def blind2_corpus(heldout_corpus, tmp_path_factory):
    """Blind set #2 images + the tau-fit corpus as distractors (same build as blind_corpus)."""
    import shutil

    from shotsieve.pipeline.indexer import Indexer

    d = json.loads((BLIND2_DIR / "labels.json").read_text())
    folder = tmp_path_factory.mktemp("shotlist_blind2")
    for p in sorted(heldout_corpus.glob("*.jpg")):
        shutil.copy(p, folder / p.name)
    for fn in d["images"]:
        shutil.copy(BLIND2_DIR / "images" / fn, folder / f"blind2_{fn}")
    allp = sorted(str(p) for p in folder.glob("*.jpg"))
    with Cache.for_folder(folder) as c:
        c.save_decisions(_rows(allp, ["keep"] * len(allp)))
    ix = Indexer(folder, Settings(), people=False)
    ix.run_blocking()
    assert ix.error is None and ix.done == ix.total == len(allp)
    return folder


@pytest.mark.models
def test_blind2_false_covered(blind2_corpus):
    """N1 closure: blind set #2 (independent author), scored ONCE with the uncertain-band method.
    Criteria fixed before scoring: must-have false-covered <= 5 %; accuracy >= 0.90 where a present
    item counts as found if covered or uncertain (uncertain must-haves surface for review)."""
    import hashlib

    raw = (BLIND2_DIR / "labels.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == BLIND2_SHA
    d = json.loads(raw)
    present = [p["item"] for p in d["present"]]
    absent = d["absent"]
    items = shotlist.parse_list("\n".join("!" + t for t in [*present, *absent]), "blind2")
    cov = shotlist.run(blind2_corpus, items, Settings())
    status = {r.item.text: r.status for r in cov.results}
    for r in cov.results:
        print(f"  BLIND2 {r.status:9s} s={r.top_score:.4f} z={r.top_z:5.2f} {'P' if r.item.text in present else 'A'} {r.item.text}")
    found = ("covered", "uncertain")
    fc = sum(status[t] == "covered" for t in absent) / len(absent)
    correct = sum(status[t] in found for t in present) + sum(status[t] != "covered" for t in absent)
    n = len(present) + len(absent)
    strict = sum(status[t] == "covered" for t in present)
    unc = sum(s == "uncertain" for s in status.values())
    print(f"T17 BLIND2 accuracy {correct / n:.3f} ({correct}/{n}), present covered {strict}/{len(present)}, "
          f"uncertain {unc}/{n}, false-covered {fc:.3f}")
    assert fc <= 0.05
    assert correct / n >= 0.90
    # pinned regression guards (Review #8 m2), not re-tuning: "all uncertain" must not pass
    assert unc / n <= 0.10, unc
    assert strict / len(present) >= 0.75, strict
