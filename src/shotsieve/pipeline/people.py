"""People clustering over the cached SFace embeddings (opt-in; see Settings.people_consent).

Agglomerative, cosine, average linkage, merge while similarity >= MERGE_SIM. Every face is stored as
a person (MIN_FACES = 1): a face that joined no one is a one-face "Unknown" person that the user can
name or merge. Names are only ever user-given and survive a
re-cluster when most of a named person's faces land in one new cluster.
"""

from __future__ import annotations

import numpy as np

from shotsieve.pipeline.cache import Cache

MERGE_SIM = 0.40
JOIN_SIM = 0.45
MIN_FACES = 1
PEOPLE_CLUSTER_VERSION = 3  # 3 = + recognition-only faces (T23h); 2 = singletons (T23g)
VERSION_KEY = "people_cluster_version"
FULL_LIMIT = 3000  # above this, leader-pass first so the O(n^2) matrix stays small


class ConsentError(PermissionError):
    """People features were used without the user's consent."""


def _agglomerate(x: np.ndarray, merge_sim: float) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering

    if len(x) < 2:
        return np.zeros(len(x), int)
    model = AgglomerativeClustering(
        n_clusters=None, metric="cosine", linkage="average", distance_threshold=1.0 - merge_sim
    )
    return model.fit_predict(x)


def cluster_vectors(x: np.ndarray) -> np.ndarray:
    """Cluster label per row of L2-normalised vectors."""
    if len(x) <= FULL_LIMIT:
        return _agglomerate(x, MERGE_SIM)
    # incremental: join the best centroid at sim >= JOIN_SIM else open a cluster, then merge
    cents: list[np.ndarray] = []
    sums: list[np.ndarray] = []
    assign = np.zeros(len(x), int)
    for i, v in enumerate(x):
        if cents:
            sims = np.stack(cents) @ v
            j = int(np.argmax(sims))
            if sims[j] >= JOIN_SIM:
                assign[i] = j
                sums[j] += v
                cents[j] = sums[j] / np.linalg.norm(sums[j])
                continue
        cents.append(v.copy())
        sums.append(v.copy())
        assign[i] = len(cents) - 1
    top = _agglomerate(np.stack(cents), MERGE_SIM)
    return top[assign]


def cluster(cache: Cache) -> int:
    """Re-cluster all faces in `cache`; returns the number of people stored.

    The result is written in one step under the people lock, after re-checking that consent was
    not revoked while clustering ran (`Cache.revoke_people` bumps the epoch): a revoked run
    writes nothing."""
    epoch = cache.people_epoch()
    faces = cache.face_embeddings()
    alive = set(cache.load_decisions())
    if alive:  # T23i: photos moved to the Trash no longer count (their vectors stay for an undo)
        faces = {p: v for p, v in faces.items() if p in alive}
    gone = _excluded(cache)
    keys = [(p, i) for p in sorted(faces) for i in range(len(faces[p])) if (p, i) not in gone]
    old = _old_names(cache)
    if len(keys) < MIN_FACES:
        with cache.people_write(epoch) as ok:
            if ok:
                cache.wipe_people(keep_face_idx=True)
                _stamp(cache)
        return 0
    x = np.stack([faces[p][i] for p, i in keys]).astype(np.float32)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    labels = cluster_vectors(x)
    clusters: list[list[int]] = []
    for lab in sorted(set(labels.tolist())):
        idx = [i for i, label in enumerate(labels) if label == lab]
        if len(idx) >= MIN_FACES:
            clusters.append(idx)
    clusters.sort(key=len, reverse=True)
    names = _carry_names(old, [[keys[i] for i in idx] for idx in clusters])
    with cache.people_write(epoch) as ok:
        if not ok:
            return 0
        cache.wipe_people(keep_face_idx=True)
        con = cache._con
        for n, idx in enumerate(clusters):
            c = x[idx].mean(axis=0)
            c /= max(float(np.linalg.norm(c)), 1e-12)
            sims = x[idx] @ c
            cover = keys[idx[int(np.argmax(sims))]]
            cur = con.execute(
                "INSERT INTO people (name, cover_face, n) VALUES (?,?,?)",
                (names.get(n), f"{cache._rel(cover[0])}#{cover[1]}", len(idx)),
            )
            pid = cur.lastrowid
            con.executemany(
                "INSERT INTO person_faces VALUES (?,?,?,?)",
                [
                    (pid, cache._rel(keys[i][0]), keys[i][1], float(s))
                    for i, s in zip(idx, sims, strict=True)
                ],
            )
        _stamp(cache)
    return len(clusters)


def _stamp(cache: Cache) -> None:  # inside the caller's people_write transaction
    cache._con.execute(
        "INSERT OR REPLACE INTO meta VALUES (?,?)", (VERSION_KEY, str(PEOPLE_CLUSTER_VERSION))
    )


def needs_recluster(cache: Cache) -> bool:
    """True when this shoot has a people table built by an older clustering (no stored version
    counts as version 1) and stored face vectors to re-cluster from."""
    v = cache.get_meta(VERSION_KEY, 1)
    if isinstance(v, int) and v >= PEOPLE_CLUSTER_VERSION:
        return False
    with cache._lock:
        has_people = cache._con.execute("SELECT 1 FROM people LIMIT 1").fetchone() is not None
    return has_people and bool(cache.face_embeddings())


def _old_names(cache: Cache) -> list[tuple[str, set[tuple[str, int]]]]:
    with cache._lock:
        rows = cache._con.execute(
            "SELECT p.name, f.path, f.face_idx FROM people p JOIN person_faces f "
            "ON f.person_id=p.id WHERE p.name IS NOT NULL AND p.name != ''"
        ).fetchall()
    by: dict[str, set[tuple[str, int]]] = {}
    for name, path, idx in rows:
        by.setdefault(name, set()).add((cache._abs(path), idx))
    return list(by.items())


def _carry_names(old, new_clusters) -> dict[int, str]:
    out: dict[int, str] = {}
    for name, faces in old:
        best, best_n = None, 0
        for n, members in enumerate(new_clusters):
            if n in out:
                continue
            k = len(faces & set(members))
            if k > best_n:
                best, best_n = n, k
        if best is not None and best_n * 2 >= len(faces):
            out[best] = name
    return out


# ---- reads / edits (the API layer) ---------------------------------------------------------
def _excluded(cache: Cache) -> set[tuple[str, int]]:
    with cache._lock:
        rows = cache._con.execute("SELECT path, face_idx FROM face_excluded").fetchall()
    return {(cache._abs(p), i) for p, i in rows}


def list_people(cache: Cache) -> dict:
    """Every person (named or not), biggest first. `n` = faces, `photos` = distinct images,
    `keeps` = images whose final decision (override first) is keep, `cover_idx` = the cover face's
    index among the image's stored face embeddings."""
    with cache._lock:
        con = cache._con
        people = con.execute(
            "SELECT id, name, cover_face, n FROM people ORDER BY n DESC, id"
        ).fetchall()
        n_faces = con.execute("SELECT COUNT(*) FROM person_faces").fetchone()[0]
        stats = {
            r[0]: (r[1], r[2])
            for r in con.execute(
                "SELECT pf.person_id, COUNT(DISTINCT pf.path), "
                "COUNT(DISTINCT CASE WHEN COALESCE(o.decision, d.decision) = 'keep' "
                "THEN pf.path END) FROM person_faces pf "
                "LEFT JOIN decisions d ON d.path = pf.path "
                "LEFT JOIN overrides o ON o.path = pf.path GROUP BY pf.person_id"
            ).fetchall()
        }
        n_ignored = con.execute("SELECT COUNT(*) FROM face_excluded").fetchone()[0]
    alive = set(cache.load_decisions())
    total = sum(len(v) for p, v in cache.face_embeddings().items() if not alive or p in alive)
    covers = [c.rsplit("#", 1)[0] for _, _, c, _ in people]
    ids = cache.ids_for([cache._abs(c) for c in covers]) if covers else {}
    out = []
    for (pid, name, cover, n), cpath in zip(people, covers, strict=True):
        photos, keeps = stats.get(pid, (0, 0))
        try:
            cidx = int(cover.rsplit("#", 1)[1])
        except (IndexError, ValueError):
            cidx = 0
        out.append(
            {
                "id": pid, "name": name, "n": n, "photos": photos, "keeps": keeps,
                "cover_id": ids.get(cache._abs(cpath)), "cover_idx": cidx,
            }  # fmt: skip
        )
    return {"people": out, "unsorted": max(total - n_faces - n_ignored, 0), "ignored": n_ignored}


def rename(cache: Cache, person_id: int, name: str) -> int | None:
    """Name a person. A name another person already has means the same person: merge the two.
    Returns the id that now holds the faces when a merge happened."""
    name = name.strip()
    with cache._lock:
        cache._con.execute("UPDATE people SET name=? WHERE id=?", (name or None, person_id))
        cache._commit()
        other = None
        if name:
            row = cache._con.execute(
                "SELECT id FROM people WHERE id != ? AND LOWER(name)=LOWER(?) "
                "ORDER BY n DESC, id LIMIT 1",
                (person_id, name),
            ).fetchone()
            other = row[0] if row else None
    if other is not None:
        merge(cache, person_id, other)
        return int(other)
    return None


def merge(cache: Cache, person_id: int, into: int) -> None:
    with cache._lock:
        con = cache._con
        if con.execute("SELECT 1 FROM people WHERE id=?", (into,)).fetchone() is None:
            raise KeyError(into)
        con.execute("UPDATE person_faces SET person_id=? WHERE person_id=?", (into, person_id))
        con.execute("DELETE FROM people WHERE id=?", (person_id,))
        n = con.execute("SELECT COUNT(*) FROM person_faces WHERE person_id=?", (into,)).fetchone()
        con.execute("UPDATE people SET n=? WHERE id=?", (n[0], into))
        con.commit()


def _refresh(cache: Cache, person_id: int) -> None:
    """Recount a person; a new cover when the old one left; delete an empty person."""
    con = cache._con
    rows = con.execute(
        "SELECT path, face_idx FROM person_faces WHERE person_id=? ORDER BY sim DESC",
        (person_id,),
    ).fetchall()
    if not rows:
        con.execute("DELETE FROM people WHERE id=?", (person_id,))
        return
    cover = con.execute("SELECT cover_face FROM people WHERE id=?", (person_id,)).fetchone()
    have = {f"{p}#{i}" for p, i in rows}
    new = cover[0] if cover and cover[0] in have else f"{rows[0][0]}#{rows[0][1]}"
    con.execute("UPDATE people SET n=?, cover_face=? WHERE id=?", (len(rows), new, person_id))


def ignore(cache: Cache, person_id: int) -> int:
    """"Not a person": every face of the cluster is excluded for good (a re-cluster skips them)
    and the cluster is removed. Returns the number of faces."""
    return len(ignore_faces(cache, person_id))


def ignore_faces(cache: Cache, person_id: int) -> list[tuple[str, int]]:
    """Same as `ignore`, returning the (path, face_idx) rows that were excluded (for an Undo)."""
    with cache._lock:
        con = cache._con
        rows = con.execute(
            "SELECT path, face_idx FROM person_faces WHERE person_id=?", (person_id,)
        ).fetchall()
        con.executemany("INSERT OR IGNORE INTO face_excluded VALUES (?,?)", rows)
        con.execute("DELETE FROM person_faces WHERE person_id=?", (person_id,))
        con.execute("DELETE FROM people WHERE id=?", (person_id,))
        con.commit()
    return [(r[0], r[1]) for r in rows]


def unignore(cache: Cache, faces: list[tuple[str, int]]) -> int:
    """Undo `ignore` for just these faces; they are grouped again at the next clustering run."""
    with cache._lock:
        cache._con.executemany("DELETE FROM face_excluded WHERE path=? AND face_idx=?", faces)
        cache._con.commit()
    return len(faces)


def restore_ignored(cache: Cache) -> int:
    """Forget every exclusion; the faces return at the next clustering run."""
    with cache._lock:
        n = cache._con.execute("SELECT COUNT(*) FROM face_excluded").fetchone()[0]
        cache._con.execute("DELETE FROM face_excluded")
        cache._con.commit()
    return n


def detach_face(cache: Cache, path: str, face_idx: int) -> int | None:
    """"Not this person": take one face out of its cluster for good. Returns the person id it
    left (None when it was in no cluster)."""
    rel = cache._rel(path)
    with cache._lock:
        con = cache._con
        row = con.execute(
            "SELECT person_id FROM person_faces WHERE path=? AND face_idx=?", (rel, face_idx)
        ).fetchone()
        con.execute("INSERT OR IGNORE INTO face_excluded VALUES (?,?)", (rel, face_idx))
        if row is None:
            con.commit()
            return None
        con.execute("DELETE FROM person_faces WHERE path=? AND face_idx=?", (rel, face_idx))
        _refresh(cache, row[0])
        con.commit()
    return int(row[0])


def assign_face(
    cache: Cache, path: str, face_idx: int, person_id: int | None = None, name: str | None = None
) -> int:
    """Put one face into an existing person, or (person_id None) into a new one named `name`."""
    rel = cache._rel(path)
    with cache._lock:
        con = cache._con
        old = con.execute(
            "SELECT person_id FROM person_faces WHERE path=? AND face_idx=?", (rel, face_idx)
        ).fetchone()
        if person_id is None:
            cur = con.execute(
                "INSERT INTO people (name, cover_face, n) VALUES (?,?,0)",
                ((name or "").strip() or None, f"{rel}#{face_idx}"),
            )
            person_id = int(cur.lastrowid)
        elif con.execute("SELECT 1 FROM people WHERE id=?", (person_id,)).fetchone() is None:
            raise KeyError(person_id)
        con.execute("DELETE FROM face_excluded WHERE path=? AND face_idx=?", (rel, face_idx))
        con.execute(
            "INSERT OR REPLACE INTO person_faces VALUES (?,?,?,?)", (person_id, rel, face_idx, 1.0)
        )
        _refresh(cache, person_id)
        if old is not None and old[0] != person_id:
            _refresh(cache, old[0])
        con.commit()
    return person_id


def faces_of_image(cache: Cache, path: str) -> dict[int, tuple[int, str | None]]:
    """{embedding index: (person id, name)} for the clustered faces of one image."""
    with cache._lock:
        rows = cache._con.execute(
            "SELECT f.face_idx, p.id, p.name FROM person_faces f JOIN people p ON p.id=f.person_id "
            "WHERE f.path=?",
            (cache._rel(path),),
        ).fetchall()
    return {r[0]: (r[1], r[2]) for r in rows}


def image_ids_for_person_id(cache: Cache, person_id: int) -> set[int]:
    with cache._lock:
        rows = cache._con.execute(
            "SELECT DISTINCT i.id FROM person_faces f JOIN image_ids i ON i.path=f.path "
            "WHERE f.person_id=?",
            (person_id,),
        ).fetchall()
    return {r[0] for r in rows}


def names(cache: Cache) -> list[str]:
    with cache._lock:
        rows = cache._con.execute(
            "SELECT DISTINCT name FROM people WHERE name IS NOT NULL AND name != '' ORDER BY name"
        ).fetchall()
    return [r[0] for r in rows]


def image_ids_for_person(cache: Cache, name: str) -> set[int]:
    with cache._lock:
        rows = cache._con.execute(
            "SELECT DISTINCT i.id FROM person_faces f JOIN people p ON p.id=f.person_id "
            "JOIN image_ids i ON i.path=f.path WHERE lower(p.name)=lower(?)",
            (name,),
        ).fetchall()
    return {r[0] for r in rows}
