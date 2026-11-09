"""People clustering + the consent gate (synthetic SFace-like vectors; no models)."""

from __future__ import annotations

import shutil
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from shotsieve.config import Settings
from shotsieve.pipeline import people
from shotsieve.pipeline.cache import Cache, _pack
from shotsieve.pipeline.indexer import Indexer
from tests.test_indexer import NAMES, fake_embed


def _unit(v):
    return v / np.linalg.norm(v)


def seed_faces(folder, per_person=(4, 4, 2)):
    """Images carry faces of identity A, B, C (noisy copies of one direction each)."""
    rng = np.random.default_rng(1)
    centres = [_unit(rng.normal(size=128)) for _ in per_person]
    with Cache.for_folder(folder) as c:
        i = 0
        for k, n in enumerate(per_person):
            for _ in range(n):
                path = str(folder / f"{NAMES[i % len(NAMES)]}.jpg")
                v = _unit(centres[k] + 0.15 * _unit(rng.normal(size=128)))
                key = c._rel(path)
                row = c._con.execute(
                    "SELECT data, n FROM embeddings WHERE path=? AND kind='face'", (key,)
                ).fetchone()
                old = np.frombuffer(row[0], np.float16).reshape(row[1], 128) if row else None
                arr = v[None] if old is None else np.vstack([old.astype(np.float32), v[None]])
                c._con.execute(
                    "INSERT OR REPLACE INTO embeddings VALUES (?,?,?,?,?)",
                    (key, "face", 128, len(arr), _pack(arr)),
                )
                i += 1
        c._con.commit()


@pytest.fixture()
def folder(tmp_path, shoot_dir):
    d = tmp_path / "shoot"
    d.mkdir()
    for n in NAMES:
        shutil.copy(shoot_dir / f"{n}.jpg", d / f"{n}.jpg")
    return d


def _count(folder):
    with Cache.for_folder(folder) as c:
        return c._con.execute("SELECT COUNT(*) FROM people").fetchone()[0]


def test_consent_off_means_no_people_rows(folder):
    seed_faces(folder)
    ix = Indexer(folder, Settings(), people=True, embedder=fake_embed)
    ix.run_blocking()
    assert ix.error is None and _count(folder) == 0
    with Cache.for_folder(folder) as c:
        assert c.people_consent(False) is False


def test_consent_on_clusters_and_every_face_gets_an_entry(folder):
    seed_faces(folder)
    with Cache.for_folder(folder) as c:
        c.set_meta("people_consent", True)
    Indexer(folder, Settings(), people=True, embedder=fake_embed).run_blocking()
    with Cache.for_folder(folder) as c:
        info = people.list_people(c)
    assert sorted(p["n"] for p in info["people"]) == [2, 4, 4]  # T23g: nobody is left "unsorted"
    assert info["unsorted"] == 0


def test_names_survive_recluster_and_merge(folder):
    seed_faces(folder)
    with Cache.for_folder(folder) as c:
        assert people.cluster(c) == 3
        pid = people.list_people(c)["people"][0]["id"]
        people.rename(c, pid, "Ana")
        assert people.cluster(c) == 3
        names = [p["name"] for p in people.list_people(c)["people"]]
        assert names.count("Ana") == 1
        ana = people.image_ids_for_person(c, "ana")
        assert ana
        a, b = [p["id"] for p in people.list_people(c)["people"]][:2]
        people.merge(c, b, a)
        assert sorted(p["n"] for p in people.list_people(c)["people"]) == [2, 8]


def test_cluster_vectors_incremental_path(monkeypatch):
    monkeypatch.setattr(people, "FULL_LIMIT", 10)
    rng = np.random.default_rng(3)
    cs = [_unit(rng.normal(size=128)) for _ in range(3)]
    x = np.stack([_unit(cs[i % 3] + 0.1 * _unit(rng.normal(size=128))) for i in range(60)])
    lab = people.cluster_vectors(x)
    assert len(set(lab.tolist())) == 3
    assert all(len(set(lab[i::3].tolist())) == 1 for i in range(3))


# ---- API: consent gate and wipe ------------------------------------------------------------------


@pytest.fixture()
def client(folder, monkeypatch):
    from shotsieve.server import app as app_mod

    monkeypatch.setattr("shotsieve.models.text_image.search_models_ready", lambda: False)
    app = app_mod.create_app(Settings(search_index_auto=False), default_folder=folder)
    with TestClient(app) as c:
        yield c


def test_people_endpoints_403_until_consent(client, folder):
    seed_faces(folder)
    assert client.get("/api/people").status_code == 403
    assert client.post("/api/people/1", json={"name": "x"}).status_code == 403
    assert client.get("/api/search", params={"q": 'person:"Ana"'}).status_code in (403, 404)
    assert client.post("/api/people/consent", json={"on": True}).json()["people_consent"] is True
    with Cache.for_folder(folder) as c:
        people.cluster(c)
    body = client.get("/api/people").json()
    assert len(body["people"]) == 3
    pid = body["people"][0]["id"]
    named = client.post(f"/api/people/{pid}", json={"name": "Ana"}).json()
    assert named["people"][0]["name"] == "Ana"
    assert client.get("/api/index/status").json()["people_consent"] is True


def test_delete_wipes_and_revokes(client, folder):
    seed_faces(folder)
    client.post("/api/people/consent", json={"on": True})
    with Cache.for_folder(folder) as c:
        people.cluster(c)
    assert _count(folder) == 3
    assert client.delete("/api/people").json()["people_consent"] is False
    assert _count(folder) == 0
    with Cache.for_folder(folder) as c:
        n = c._con.execute("SELECT COUNT(*) FROM person_faces").fetchone()[0]
    assert n == 0
    assert client.get("/api/people").status_code == 403


# ---- Review #4 M3: revocation holds even while clustering runs ----------------------------------


def _jersey_links(folder):
    with Cache.for_folder(folder) as c:
        return c._con.execute(
            "SELECT COUNT(*) FROM jerseys WHERE face_idx IS NOT NULL OR person_id IS NOT NULL"
        ).fetchone()[0]


def _seed_jerseys(folder):
    with Cache.for_folder(folder) as c:
        c.put_jerseys(
            str(folder / f"{NAMES[0]}.jpg"), "fp",
            [{"number": "23", "conf": 0.9, "x": 0, "y": 0, "w": 1, "h": 1,
              "face_idx": 0, "person_id": 1}],
        )  # fmt: skip
        c.replace_burst_jerseys(
            [{"path": str(folder / f"{NAMES[0]}.jpg"), "number": "23", "conf": 0.9, "person_id": 1}]
        )


def test_wipe_nulls_face_links_in_jerseys(folder):
    seed_faces(folder)
    _seed_jerseys(folder)
    assert _jersey_links(folder) == 2
    with Cache.for_folder(folder) as c:
        c.wipe_people()
    assert _jersey_links(folder) == 0


def test_revocation_during_clustering_leaves_no_face_data(folder, monkeypatch):
    seed_faces(folder)
    _seed_jerseys(folder)
    with Cache.for_folder(folder) as c:
        c.set_meta("people_consent", True)
    real = people.cluster_vectors

    def revoke_midway(x):
        labels = real(x)
        with Cache.for_folder(folder) as other:  # the DELETE /api/people request
            other.revoke_people()
        return labels

    monkeypatch.setattr(people, "cluster_vectors", revoke_midway)
    with Cache.for_folder(folder) as c:
        assert people.cluster(c) == 0
    with Cache.for_folder(folder) as c:
        n = c._con.execute("SELECT COUNT(*) FROM person_faces").fetchone()[0]
        assert c.people_consent(True) is False
    assert n == 0 and _count(folder) == 0 and _jersey_links(folder) == 0


def test_revocation_during_finalize_does_not_name_people(folder):
    from shotsieve.pipeline import jerseys

    seed_faces(folder)
    with Cache.for_folder(folder) as c:
        c.set_meta("people_consent", True)
        epoch = c.people_epoch()
        assert people.cluster(c) == 3
        c.set_roster([("23", "Ana Test", "")])
        with Cache.for_folder(folder) as other:
            other.revoke_people()
        out = jerseys.finalize(c, True, epoch)
        assert out["people_named"] == 0
        assert c._con.execute("SELECT COUNT(*) FROM people").fetchone()[0] == 0


def test_delete_endpoint_nulls_jersey_links_and_stops_people_task(client, folder):
    seed_faces(folder)
    _seed_jerseys(folder)
    client.post("/api/people/consent", json={"on": True})
    with Cache.for_folder(folder) as c:
        people.cluster(c)
    assert client.delete("/api/people").json()["people_consent"] is False
    assert _jersey_links(folder) == 0 and _count(folder) == 0


def test_consent_off_post_nulls_jersey_links(client, folder):
    seed_faces(folder)
    _seed_jerseys(folder)
    client.post("/api/people/consent", json={"on": True})
    client.post("/api/people/consent", json={"on": False})
    assert _jersey_links(folder) == 0


def test_indexer_people_task_is_cancelled_by_revocation(folder, monkeypatch):
    seed_faces(folder)
    with Cache.for_folder(folder) as c:
        c.set_meta("people_consent", True)
    ix = Indexer(folder, Settings(), people=True, embedder=fake_embed)
    calls = []
    monkeypatch.setattr(people, "cluster", lambda *_a, **_k: calls.append(1) or 0)
    ix.cancel_people()
    ix.run_blocking()
    assert calls == []


def test_single_appearance_faces_are_entries_nameable_and_mergeable(folder):
    """T23g: N distinct one-face people give N entries (no "unsorted" pile); naming two of them the
    same name merges them; a named singleton survives a re-cluster."""
    seed_faces(folder, per_person=(1, 1, 1, 1))
    with Cache.for_folder(folder) as c:
        c.set_meta("people_consent", True)
        assert people.cluster(c) == 4
        info = people.list_people(c)
        assert [p["n"] for p in info["people"]] == [1, 1, 1, 1] and info["unsorted"] == 0
        a, b, c3, _d = (p["id"] for p in info["people"])
        assert people.rename(c, a, "Mia") is None
        assert people.rename(c, b, "mia") == a  # same name: they are one person
        assert sorted(p["n"] for p in people.list_people(c)["people"]) == [1, 1, 2]
        people.merge(c, c3, a)  # "Same person as..."
        assert sorted(p["n"] for p in people.list_people(c)["people"]) == [1, 3]
        assert people.cluster(c) == 4  # re-cluster: the 3 faces split again but the name carries
        assert [p["name"] for p in people.list_people(c)["people"]].count("Mia") <= 1


# ---- T23g: shoots clustered by an older rule set are re-clustered once on open --------------


def test_old_version_shoot_is_reclustered_on_open_keeping_names_and_exclusions(client, folder):
    seed_faces(folder, per_person=(5, 4, 2))
    client.post("/api/people/consent", json={"on": True})
    with Cache.for_folder(folder) as c:
        people.cluster(c)
        ppl = people.list_people(c)["people"]
        big, mid, small = ppl[0], ppl[1], ppl[2]
        people.rename(c, big["id"], "Ana")
        people.ignore(c, mid["id"])  # "Not a person": 4 faces excluded for good
        # an old table (MIN_FACES = 2): the 2-face cluster was never stored; no version recorded
        c._con.execute("DELETE FROM person_faces WHERE person_id=?", (small["id"],))
        c._con.execute("DELETE FROM people WHERE id=?", (small["id"],))
        c._con.execute("DELETE FROM meta WHERE key=?", (people.VERSION_KEY,))
        c._con.commit()
        before = people.list_people(c)
        assert before["unsorted"] == 2 and people.needs_recluster(c)
    body = client.get("/api/people").json()  # opening the shoot starts the background re-cluster
    for _ in range(100):
        if body["unsorted"] == 0:
            break
        time.sleep(0.1)
        body = client.get("/api/people").json()
    assert body["unsorted"] == 0 and body["ignored"] == 4
    assert sorted(p["n"] for p in body["people"]) == [2, 5]
    assert [p["name"] for p in body["people"] if p["n"] == 5] == ["Ana"]
    with Cache.for_folder(folder) as c:
        assert c.get_meta(people.VERSION_KEY) == people.PEOPLE_CLUSTER_VERSION
        assert not people.needs_recluster(c)
    # current version: opening again does not re-cluster
    with Cache.for_folder(folder) as c:
        c._con.execute("DELETE FROM people WHERE n=2")
        c._con.commit()
    assert len(client.get("/api/people").json()["people"]) == 1
