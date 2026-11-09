"""SQLite cache + result store: <folder>/.shotsieve/cull.db.

Tables: images (EXIF meta, keyed by relative path + content fingerprint), features (raw
measurements), embeddings (float16 BLOBs), decisions (the *model's* decision), overrides (user
decisions with propensity, never touched by a re-run), meta (kv).

Paths are stored relative to the shoot folder (the folder that holds `.shotsieve/`), so moving or
renaming the folder keeps the cache, ids and overrides. The public API still speaks absolute
paths: they are converted at the boundary. A cached row is reused when its extractor
fingerprint matches and the file is unchanged, i.e. (size, mtime) match or, after a copy that
reset mtimes, the content fingerprint (size + hash of the head and tail) matches.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from shotsieve.features.record import RawFeatures
from shotsieve.features.vector import SCHEMA_VERSION
from shotsieve.io.exif import ImageMeta

DB_NAME = "cull.db"
_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS images (
    path TEXT PRIMARY KEY, size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL, meta_json TEXT,
    fp TEXT, extractor TEXT);
CREATE TABLE IF NOT EXISTS features (
    path TEXT PRIMARY KEY, schema INTEGER NOT NULL, json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS embeddings (
    path TEXT NOT NULL, kind TEXT NOT NULL, dim INTEGER NOT NULL, n INTEGER NOT NULL,
    data BLOB NOT NULL, PRIMARY KEY (path, kind));
CREATE TABLE IF NOT EXISTS decisions (
    path TEXT PRIMARY KEY, decision TEXT NOT NULL, reasons TEXT, score REAL, prior REAL,
    group_id INTEGER, burst_id INTEGER, is_best INTEGER, audit INTEGER, audit_prob REAL,
    features TEXT, run_ts REAL, klass TEXT);
CREATE TABLE IF NOT EXISTS overrides (
    path TEXT PRIMARY KEY, decision TEXT NOT NULL, propensity REAL, ts REAL, note TEXT,
    rating INTEGER);
CREATE TABLE IF NOT EXISTS image_ids (
    id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT NOT NULL UNIQUE);
CREATE TABLE IF NOT EXISTS index_state (
    path TEXT NOT NULL, kind TEXT NOT NULL, fp TEXT NOT NULL, PRIMARY KEY (path, kind));
CREATE TABLE IF NOT EXISTS people (
    id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, cover_face TEXT, n INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS person_faces (
    person_id INTEGER NOT NULL, path TEXT NOT NULL, face_idx INTEGER NOT NULL, sim REAL,
    PRIMARY KEY (path, face_idx));
CREATE TABLE IF NOT EXISTS face_excluded (
    path TEXT NOT NULL, face_idx INTEGER NOT NULL, PRIMARY KEY (path, face_idx));
CREATE TABLE IF NOT EXISTS jerseys (
    path TEXT NOT NULL, number TEXT NOT NULL, conf REAL, x REAL, y REAL, w REAL, h REAL,
    source TEXT NOT NULL, face_idx INTEGER, person_id INTEGER);
CREATE INDEX IF NOT EXISTS jerseys_path ON jerseys(path);
CREATE INDEX IF NOT EXISTS jerseys_number ON jerseys(number);
CREATE TABLE IF NOT EXISTS edits (
    image_id INTEGER PRIMARY KEY, values_json TEXT NOT NULL, confidence_json TEXT,
    written_mode TEXT, written_at REAL, profile_hash TEXT);
CREATE TABLE IF NOT EXISTS shotlist (
    item_id TEXT PRIMARY KEY, list_name TEXT NOT NULL DEFAULT '', pos INTEGER NOT NULL DEFAULT 0,
    text TEXT NOT NULL, must INTEGER NOT NULL DEFAULT 0, min_count INTEGER NOT NULL DEFAULT 1,
    person TEXT, status TEXT NOT NULL DEFAULT 'pending', best_ids TEXT, n_hits INTEGER,
    top_score REAL, top_z REAL, detail TEXT, updated REAL);
CREATE TABLE IF NOT EXISTS lr_written (
    path TEXT PRIMARY KEY, rating INTEGER, label TEXT, decision TEXT, ts REAL);
CREATE TABLE IF NOT EXISTS lr_sync (
    path TEXT PRIMARY KEY, lr_id INTEGER, rating INTEGER, pick INTEGER, label TEXT,
    touch_ts REAL, synced_at REAL);
CREATE TABLE IF NOT EXISTS crops (
    image_id INTEGER NOT NULL, version INTEGER NOT NULL,
    x0 REAL, y0 REAL, x1 REAL, y1 REAL,
    ratio TEXT, source TEXT NOT NULL,
    subjects_json TEXT, score REAL, terms_json TEXT, alt_rank INTEGER, det_version TEXT,
    orientation INTEGER, full_w INTEGER, full_h INTEGER,
    written_mode TEXT, write_status TEXT, written_at REAL, created REAL NOT NULL,
    PRIMARY KEY (image_id, version));
CREATE TABLE IF NOT EXISTS subjects (
    path TEXT PRIMARY KEY, fp TEXT NOT NULL, det_version TEXT NOT NULL, json TEXT NOT NULL,
    sal_h INTEGER, sal_w INTEGER, saliency BLOB, created REAL);
CREATE TABLE IF NOT EXISTS crop_events (
    image_id INTEGER NOT NULL, ts REAL NOT NULL, action TEXT NOT NULL,
    rect_json TEXT, ratio TEXT, rank INTEGER);
CREATE TABLE IF NOT EXISTS crop_user_boxes (
    image_id INTEGER NOT NULL, version INTEGER NOT NULL, boxes_json TEXT NOT NULL,
    orientation INTEGER, created REAL NOT NULL, PRIMARY KEY (image_id, version));
CREATE TABLE IF NOT EXISTS labels (
    path TEXT PRIMARY KEY, label TEXT NOT NULL, ts REAL);
CREATE TABLE IF NOT EXISTS exif_extra (
    path TEXT PRIMARY KEY, lens_model TEXT, lens_serial TEXT, lens_ver INTEGER,
    xmp_label TEXT, xmp_mtime_ns INTEGER);
CREATE TABLE IF NOT EXISTS rec_boxes (
    path TEXT PRIMARY KEY, json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS roster2 (
    team TEXT NOT NULL DEFAULT '', number TEXT NOT NULL, name TEXT NOT NULL,
    PRIMARY KEY (team, number));
"""


_PEOPLE_LOCKS: dict[str, threading.RLock] = {}
_PEOPLE_LOCKS_GUARD = threading.Lock()


@dataclass
class CachedImage:
    meta: ImageMeta
    raw: RawFeatures
    scene: np.ndarray  # (384,) float32
    faces: np.ndarray  # (K, 128) float32, row 0 = main face (K may be 0)


def cache_path(folder: str | Path) -> Path:
    return Path(folder) / ".shotsieve" / DB_NAME


def _pack(a: np.ndarray) -> bytes:
    return np.ascontiguousarray(a, dtype=np.float16).tobytes()


def _under(p: str, root: str | Path) -> bool:
    r = str(root).rstrip(os.sep) + os.sep
    return os.path.abspath(p).startswith(r)


def _unpack(blob: bytes, n: int, dim: int) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float16).astype(np.float32).reshape(n, dim)


FP_CHUNK = 64 * 1024
LOOKUP_CHUNK = 2000
_PATH_TABLES = (
    "images", "features", "embeddings", "decisions", "overrides", "image_ids", "lr_written",
    "lr_sync",
)
SIGLIP_KIND = "siglip2"
REC_KIND = "face_rec"  # T23h: recognition-only faces (embeddings + index_state fp stamp + rec_boxes)  # noqa: E501
REC_STATE = "facescan"
SIGLIP_DIM = 768


def _same_row(old: tuple | None, new: tuple) -> bool:
    if old is None or len(old) != len(new):
        return False
    for a, b in zip(old, new, strict=True):
        if a != b and not (a is None and isinstance(b, float) and b != b):  # NaN is stored NULL
            return False
    return True


def fingerprint(path: str | Path) -> str:
    """Cheap content identity: size + blake2b of the first and last 64 KiB."""
    p = Path(path)
    size = p.stat().st_size
    h = hashlib.blake2b(str(size).encode(), digest_size=16)
    with open(p, "rb") as fh:
        h.update(fh.read(FP_CHUNK))
        if size > 2 * FP_CHUNK:
            fh.seek(size - FP_CHUNK)
            h.update(fh.read(FP_CHUNK))
    return h.hexdigest()


class Cache:
    """Thread-safe (one connection, one lock) feature cache and decision store."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.root = self.path.parent.parent  # the shoot folder
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._in_tx = False  # inside an explicit BEGIN IMMEDIATE: helpers must not commit
        self._con = sqlite3.connect(self.path, check_same_thread=False)
        self._con.execute("PRAGMA busy_timeout=10000")  # a second writer waits, not "locked"
        self._con.execute("PRAGMA journal_mode=WAL")
        self._con.execute("PRAGMA synchronous=NORMAL")
        self._con.executescript(_SCHEMA)
        old = self._con.execute(  # rosters keyed by number alone -> (team, number)
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='roster'"
        ).fetchone()
        if old:
            self._con.execute(
                "INSERT OR IGNORE INTO roster2 SELECT COALESCE(team,''), number, name FROM roster"
            )
            self._con.execute("DROP TABLE roster")
            self._con.commit()
        cols = {r[1] for r in self._con.execute("PRAGMA table_info(overrides)")}
        if "rating" not in cols:  # databases created before the review UI
            self._con.execute("ALTER TABLE overrides ADD COLUMN rating INTEGER")
        for table, col, decl in (
            ("images", "fp", "TEXT"),
            ("images", "extractor", "TEXT"),
            ("decisions", "klass", "TEXT"),
        ):
            have = {r[1] for r in self._con.execute(f"PRAGMA table_info({table})")}
            if col not in have:
                self._con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
        self._con.commit()
        self._migrate_absolute_paths()

    # ---- path identity --------------------------------------------------------------------
    def _rel(self, p: str) -> str:
        """Path relative to the shoot folder (unchanged if outside it or already relative)."""
        if os.path.isabs(p):
            try:
                return os.path.relpath(p, self.root) if _under(p, self.root) else p
            except ValueError:
                return p
        return p

    def _abs(self, r: str) -> str:
        return r if os.path.isabs(r) else str(self.root / r)

    def _migrate_absolute_paths(self) -> None:
        """Databases written before relative keys: strip the folder prefix they were made in."""
        with self._lock:
            old = self.get_meta("root")
            base = Path(old) if old else self.root
            for table in _PATH_TABLES:
                rows = self._con.execute(
                    f"SELECT path FROM {table} WHERE path LIKE '/%'"  # noqa: S608
                ).fetchall()
                for (p,) in rows:
                    if _under(p, base):
                        rel = os.path.relpath(p, base)
                        self._con.execute(
                            f"UPDATE OR REPLACE {table} SET path=? WHERE path=?",  # noqa: S608
                            (rel, p),
                        )
            self._con.commit()
            if not old:
                self.set_meta("root", str(self.root))

    @classmethod
    def for_folder(cls, folder: str | Path) -> Cache:
        return cls(cache_path(folder))

    def close(self) -> None:
        with self._lock:
            self._con.commit()
            self._con.close()

    def __enter__(self) -> Cache:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- features -------------------------------------------------------------------------
    def lookup_many(
        self,
        stats: dict[str, tuple[int, int]],
        extractor: str | None = None,
        fp_of: Callable[[str], str] | None = None,
        chunk: int = LOOKUP_CHUNK,
    ) -> dict[str, CachedImage]:
        """Cached records for the files in `stats` (path -> (size, mtime_ns)).

        A record is reused when the schema (and `extractor` fingerprint, if given) match and
        the file is unchanged: (size, mtime_ns) equal, or - e.g. after the folder was copied and
        mtimes reset - size equal and `fp_of(path)` equal to the stored content fingerprint.
        The 4-table join is read in chunks of `chunk` rows (keyset on the images rowid), so a
        10k-frame shoot never holds the whole join in memory at once.
        """
        out: dict[str, CachedImage] = {}
        last = -1
        while True:
            with self._lock:
                rows = self._con.execute(
                    "SELECT i.path, i.size, i.mtime_ns, i.meta_json, f.schema, f.json, "
                    "s.n, s.dim, s.data, fa.n, fa.dim, fa.data, i.fp, i.extractor, i.rowid "
                    "FROM images i JOIN features f ON f.path = i.path "
                    "JOIN embeddings s ON s.path = i.path AND s.kind = 'scene' "
                    "LEFT JOIN embeddings fa ON fa.path = i.path AND fa.kind = 'face' "
                    "WHERE i.rowid > ? ORDER BY i.rowid LIMIT ?",
                    (last, max(1, chunk)),
                ).fetchall()
            if not rows:
                break
            last = rows[-1][14]
            for row in rows:
                rel, size, mtime, meta_json, schema, fjson, sn, sdim, sdata, fn, fdim, fdata = row[
                    :12
                ]
                fp, ext = row[12], row[13]
                path = self._abs(rel)
                want = stats.get(path)
                if want is None or schema != SCHEMA_VERSION:
                    continue
                if extractor is not None and ext != extractor:
                    continue
                if want != (size, mtime):
                    if fp_of is None or not fp or want[0] != size:
                        continue
                    try:
                        if fp_of(path) != fp:
                            continue
                    except OSError:
                        continue
                try:
                    meta = ImageMeta(**json.loads(meta_json))
                    raw = RawFeatures.from_json(fjson)
                except (TypeError, ValueError):
                    continue
                faces = _unpack(fdata, fn, fdim) if fdata else np.zeros((0, 128), np.float32)
                out[path] = CachedImage(meta, raw, _unpack(sdata, sn, sdim)[0], faces)
        return out

    def put(
        self,
        path: str,
        size: int,
        mtime_ns: int,
        rec: CachedImage,
        fp: str = "",
        extractor: str = "",
    ) -> None:
        key = self._rel(path)
        with self._lock:
            c = self._con
            c.execute(
                "INSERT OR REPLACE INTO images (path, size, mtime_ns, meta_json, fp, extractor) "
                "VALUES (?,?,?,?,?,?)",
                (key, size, mtime_ns, json.dumps(asdict(rec.meta)), fp, extractor),
            )
            c.execute(
                "INSERT OR REPLACE INTO features VALUES (?,?,?)",
                (key, SCHEMA_VERSION, rec.raw.to_json()),
            )
            c.execute(
                "INSERT OR REPLACE INTO embeddings VALUES (?,?,?,?,?)",
                (key, "scene", rec.scene.shape[-1], 1, _pack(rec.scene)),
            )
            if len(rec.faces):
                c.execute(
                    "INSERT OR REPLACE INTO embeddings VALUES (?,?,?,?,?)",
                    (key, "face", rec.faces.shape[1], len(rec.faces), _pack(rec.faces)),
                )
            else:
                c.execute("DELETE FROM embeddings WHERE path=? AND kind='face'", (key,))

    def commit(self) -> None:
        with self._lock:
            self._con.commit()

    def metas_of(self, paths: list[str]) -> dict[str, ImageMeta]:
        """Cached EXIF meta of the given files (e.g. removed ones that are no longer on disk)."""
        out: dict[str, ImageMeta] = {}
        with self._lock:
            for i in range(0, len(paths), 500):
                part = paths[i : i + 500]
                rel = {self._rel(p): p for p in part}
                q = ",".join("?" * len(rel))
                rows = self._con.execute(
                    f"SELECT path, meta_json FROM images WHERE path IN ({q})",  # noqa: S608
                    list(rel),
                ).fetchall()
                for r, mj in rows:
                    try:
                        out[rel[r]] = ImageMeta(**json.loads(mj))
                    except (TypeError, ValueError):
                        pass
        return out

    # ---- edits (T15: predicted develop settings) ---------------------------------------------
    def put_edits(self, rows: list[dict[str, Any]]) -> None:
        """Upsert predicted edits: dicts with path, values, confidence, written_mode,
        profile_hash (written_at = now)."""
        ids = self.ids_for([r["path"] for r in rows])
        now = time.time()
        with self._lock:
            self._con.executemany(
                "INSERT OR REPLACE INTO edits VALUES (?,?,?,?,?,?)",
                [
                    (
                        ids[r["path"]], json.dumps(r["values"]), json.dumps(r.get("confidence")),
                        r.get("written_mode"), now, r.get("profile_hash"),
                    )
                    for r in rows if r["path"] in ids
                ],
            )
            self._con.commit()

    def get_edits(self, paths: list[str] | None = None) -> dict[str, dict[str, Any]]:
        """Stored edits by master path (all, or only `paths`)."""
        with self._lock:
            rows = self._con.execute(
                "SELECT i.path, e.values_json, e.confidence_json, e.written_mode, e.written_at, "
                "e.profile_hash FROM edits e JOIN image_ids i ON i.id = e.image_id"
            ).fetchall()
        out = {}
        want = None if paths is None else set(paths)
        for rel, vj, cj, mode, at, ph in rows:
            p = self._abs(rel)
            if want is not None and p not in want:
                continue
            out[p] = {
                "values": json.loads(vj), "confidence": json.loads(cj) if cj else None,
                "written_mode": mode, "written_at": at, "profile_hash": ph,
            }  # fmt: skip
        return out

    def clear_edits(self) -> None:
        with self._lock:
            self._con.execute("DELETE FROM edits")
            self._con.commit()

    # ---- crops (T22) ----------------------------------------------------------------------
    _CROP_COLS = (
        "image_id", "version", "x0", "y0", "x1", "y1", "ratio", "source", "subjects_json", "score",
        "terms_json", "alt_rank", "det_version", "orientation", "full_w", "full_h",
        "written_mode", "write_status", "written_at", "created",
    )

    def _crop_row(self, row: tuple) -> dict[str, Any]:
        d = dict(zip(self._CROP_COLS, row, strict=True))
        d["rect"] = None if d["x0"] is None else [d["x0"], d["y0"], d["x1"], d["y1"]]
        return d

    def crop_current(self, image_id: int) -> dict[str, Any] | None:
        """The newest crop version of an image (rect None = full frame after a reset)."""
        with self._lock:
            row = self._con.execute(
                f"SELECT {','.join(self._CROP_COLS)} FROM crops WHERE image_id=? "  # noqa: S608
                "ORDER BY version DESC LIMIT 1", (image_id,)
            ).fetchone()
        return self._crop_row(row) if row else None

    def crop_history(self, image_id: int) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._con.execute(
                f"SELECT {','.join(self._CROP_COLS)} FROM crops WHERE image_id=? "  # noqa: S608
                "ORDER BY version", (image_id,)
            ).fetchall()
        return [self._crop_row(r) for r in rows]

    def crop_add(self, image_id: int, rect: list[float] | None, **kw: Any) -> int:
        """Append a crop version (never updates an old one); returns its number."""
        with self._lock:
            v = self._con.execute(
                "SELECT COALESCE(MAX(version), 0) + 1 FROM crops WHERE image_id=?", (image_id,)
            ).fetchone()[0]
            r = rect or [None] * 4
            self._con.execute(
                "INSERT INTO crops VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    image_id, v, r[0], r[1], r[2], r[3], kw.get("ratio"),
                    kw.get("source", "manual"),
                    kw.get("subjects_json"), kw.get("score"), kw.get("terms_json"),
                    kw.get("alt_rank"), kw.get("det_version"), kw.get("orientation"),
                    kw.get("full_w"), kw.get("full_h"), kw.get("written_mode"),
                    kw.get("write_status"), kw.get("written_at"), time.time(),
                ),
            )
            self._con.commit()
        return int(v)

    def crop_set_write(
        self, image_id: int, version: int, mode: str | None, status: str | None
    ) -> None:
        with self._lock:
            self._con.execute(
                "UPDATE crops SET written_mode=?, write_status=?, written_at=? "
                "WHERE image_id=? AND version=?", (mode, status, time.time(), image_id, version),
            )
            self._con.commit()

    def crop_log(
        self, image_id: int, action: str, rect: Any = None, ratio: str | None = None,
        rank: int | None = None,
    ) -> None:
        with self._lock:
            self._con.execute(
                "INSERT INTO crop_events VALUES (?,?,?,?,?,?)",
                (image_id, time.time(), action, json.dumps(rect) if rect is not None else None,
                 ratio, rank),
            )
            self._con.commit()

    def crop_events(self, image_id: int | None = None) -> list[dict[str, Any]]:
        with self._lock:
            q = "SELECT image_id, ts, action, rect_json, ratio, rank FROM crop_events"
            rows = self._con.execute(
                q + (" WHERE image_id=?" if image_id is not None else "") + " ORDER BY ts",
                (image_id,) if image_id is not None else (),
            ).fetchall()
        return [
            {"image_id": i, "ts": t, "action": a, "rect": json.loads(rj) if rj else None,
             "ratio": ra, "rank": rk}
            for i, t, a, rj, ra, rk in rows
        ]  # fmt: skip

    def user_boxes(self, image_id: int) -> list[list[float]]:
        """The newest saved set of user-drawn subject boxes (display-normalised x0,y0,x1,y1)."""
        with self._lock:
            row = self._con.execute(
                "SELECT boxes_json FROM crop_user_boxes WHERE image_id=? "
                "ORDER BY version DESC LIMIT 1", (image_id,)
            ).fetchone()
        return json.loads(row[0]) if row else []

    def user_boxes_add(
        self, image_id: int, boxes: list[list[float]], orientation: int | None = None
    ) -> int:
        """Append a version of the user boxes (an empty list = cleared); never updates a row."""
        with self._lock:
            v = self._con.execute(
                "SELECT COALESCE(MAX(version), 0) + 1 FROM crop_user_boxes WHERE image_id=?",
                (image_id,),
            ).fetchone()[0]
            self._con.execute(
                "INSERT INTO crop_user_boxes VALUES (?,?,?,?,?)",
                (image_id, v, json.dumps(boxes), orientation, time.time()),
            )
            self._con.commit()
        return int(v)

    def crops_current_all(self) -> dict[int, list[float]]:
        """{image id: [x0, y0, x1, y1]} for every image whose newest crop version is a real crop
        (one indexed query: `build_shoot` calls this)."""
        with self._lock:
            rows = self._con.execute(
                "SELECT c.image_id, c.x0, c.y0, c.x1, c.y1 FROM crops c JOIN "
                "(SELECT image_id, MAX(version) v FROM crops GROUP BY image_id) m "
                "ON m.image_id = c.image_id AND m.v = c.version WHERE c.x0 IS NOT NULL"
            ).fetchall()
        return {int(i): [a, b, c, d] for i, a, b, c, d in rows}

    def get_subjects(self, path: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._con.execute(
                "SELECT fp, det_version, json, sal_h, sal_w, saliency FROM subjects WHERE path=?",
                (self._rel(path),),
            ).fetchone()
        if not row:
            return None
        fp, det, js, sh, sw, blob = row
        sal = None
        if blob and sh and sw:
            sal = np.frombuffer(blob, dtype=np.float16).astype(np.float32).reshape(sh, sw)
        return {"fp": fp, "det_version": det, "data": json.loads(js), "saliency": sal}

    def put_subjects(
        self, path: str, fp: str, det_version: str, data: dict[str, Any], sal: np.ndarray | None
    ) -> None:
        blob = None if sal is None else np.ascontiguousarray(sal, dtype=np.float16).tobytes()
        with self._lock:
            self._con.execute(
                "INSERT OR REPLACE INTO subjects VALUES (?,?,?,?,?,?,?,?)",
                (self._rel(path), fp, det_version, json.dumps(data),
                 None if sal is None else int(sal.shape[0]),
                 None if sal is None else int(sal.shape[1]), blob, time.time()),
            )
            self._con.commit()

    # ---- shot list (T17) ------------------------------------------------------------------
    def put_shotlist(self, rows: list[dict[str, Any]]) -> None:
        """Replace the stored shot list: dicts with item_id, list_name, pos, text, must, min_count,
        person, status, best_ids (list of image ids), n_hits, top_score, top_z, detail."""
        now = time.time()
        with self._lock:
            self._con.execute("DELETE FROM shotlist")
            self._con.executemany(
                "INSERT INTO shotlist VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        r["item_id"], r.get("list_name", ""), r.get("pos", 0), r["text"],
                        int(bool(r.get("must"))), int(r.get("min_count", 1)), r.get("person"),
                        r.get("status", "pending"), json.dumps(r.get("best_ids", [])),
                        r.get("n_hits"), r.get("top_score"), r.get("top_z"), r.get("detail"), now,
                    )
                    for r in rows
                ],
            )
            self._con.commit()

    def get_shotlist(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._con.execute(
                "SELECT item_id, list_name, pos, text, must, min_count, person, status, best_ids, "
                "n_hits, top_score, top_z, detail, updated FROM shotlist ORDER BY pos, item_id"
            ).fetchall()
        keys = ("item_id", "list_name", "pos", "text", "must", "min_count", "person", "status",
                "best_ids", "n_hits", "top_score", "top_z", "detail", "updated")  # fmt: skip
        out = []
        for r in rows:
            d = dict(zip(keys, r, strict=True))
            d["must"], d["best_ids"] = bool(d["must"]), json.loads(d["best_ids"] or "[]")
            out.append(d)
        return out

    def clear_shotlist(self) -> None:
        with self._lock:
            self._con.execute("DELETE FROM shotlist")
            self._con.commit()

    # ---- decisions ------------------------------------------------------------------------
    def save_decisions(self, rows: list[dict[str, Any]]) -> int:
        """Make the stored *model* decisions equal `rows` (dicts with the decisions columns): an
        upsert of the rows that changed plus a delete of the paths that are gone, in one
        transaction. Returns the number of rows written or deleted. User overrides live in their
        own table and are never written here."""
        now = time.time()
        new: dict[str, tuple] = {}
        for r in rows:
            new[self._rel(r["path"])] = (
                r["decision"], json.dumps(r["reasons"]), r["score"], r.get("prior"),
                r["group_id"], r["burst_id"], int(r["is_best"]), int(r["audit"]),
                r.get("audit_prob"), json.dumps(r["features"]), r.get("klass", "") or "",
            )  # fmt: skip
        with self._lock:
            old = {
                row[0]: tuple(row[1:])
                for row in self._con.execute(
                    "SELECT path, decision, reasons, score, prior, group_id, burst_id, is_best, "
                    "audit, audit_prob, features, COALESCE(klass, '') FROM decisions"
                )
            }
            changed = [(k, v) for k, v in new.items() if not _same_row(old.get(k), v)]
            gone = [(k,) for k in old if k not in new]
            self._con.executemany(
                "INSERT OR REPLACE INTO decisions (path, decision, reasons, score, prior, "
                "group_id, burst_id, is_best, audit, audit_prob, features, klass, run_ts) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [(k, *v, now) for k, v in changed],
            )
            self._con.executemany("DELETE FROM decisions WHERE path=?", gone)
            self._con.commit()
        return len(changed) + len(gone)

    def load_decisions(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            rows = self._con.execute(
                "SELECT path, decision, reasons, score, prior, group_id, burst_id, is_best, "
                "audit, audit_prob, features, klass FROM decisions"
            ).fetchall()
        return {
            self._abs(r[0]): {
                "decision": r[1],
                "reasons": json.loads(r[2] or "[]"),
                "score": r[3],
                "prior": r[4],
                "group_id": r[5],
                "burst_id": r[6],
                "is_best": bool(r[7]),
                "audit": bool(r[8]),
                "audit_prob": r[9],
                "features": json.loads(r[10] or "{}"),
                "klass": r[11] or "",
            }
            for r in rows
        }

    # ---- overrides ------------------------------------------------------------------------
    def set_override(
        self,
        path: str,
        decision: str,
        propensity: float = 1.0,
        note: str = "",
        rating: int | None = None,
        ts: float | None = None,
        commit: bool = True,
    ) -> None:
        with self._lock:
            self._con.execute(
                "INSERT OR REPLACE INTO overrides VALUES (?,?,?,?,?,?)",
                (self._rel(path), decision, propensity, time.time() if ts is None else ts, note,
                 rating),
            )  # fmt: skip
            if commit:
                self._con.commit()

    def clear_override(self, path: str, commit: bool = True) -> None:
        with self._lock:
            self._con.execute("DELETE FROM overrides WHERE path=?", (self._rel(path),))
            if commit:
                self._con.commit()

    def overrides(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            rows = self._con.execute(
                "SELECT path, decision, propensity, ts, note, rating FROM overrides"
            ).fetchall()
        return {
            self._abs(r[0]): {
                "decision": r[1], "propensity": r[2], "ts": r[3], "note": r[4], "rating": r[5]
            }
            for r in rows
        }  # fmt: skip

    # ---- colour labels (T23e): the user's own choice per image; "" = explicitly none --------
    def set_labels(self, items: list[tuple[str, str]]) -> None:
        now = time.time()
        with self._lock:
            self._con.executemany(
                "INSERT OR REPLACE INTO labels VALUES (?,?,?)",
                [(self._rel(p), lab, now) for p, lab in items],
            )
            self._con.commit()

    def labels(self) -> dict[str, str]:
        with self._lock:
            rows = self._con.execute("SELECT path, label FROM labels").fetchall()
        return {self._abs(r[0]): r[1] for r in rows}

    def exif_extra_rows(self) -> dict[str, dict[str, Any]]:
        """Lens fields and sidecar colour label of the facets back-fill (not a cull input)."""
        with self._lock:
            rows = self._con.execute(
                "SELECT path, lens_model, lens_serial, lens_ver, xmp_label, xmp_mtime_ns "
                "FROM exif_extra"
            ).fetchall()
        return {
            self._abs(r[0]): {"lens_model": r[1] or "", "lens_serial": r[2] or "", "ver": r[3],
                              "xmp_label": r[4] or "", "xmp_mtime_ns": r[5]}
            for r in rows
        }  # fmt: skip

    def put_exif_extra(self, rows: list[tuple[str, str, str, int, str, int | None]]) -> None:
        """(path, lens_model, lens_serial, lens_ver, xmp_label, xmp_mtime_ns)."""
        with self._lock:
            self._con.executemany(
                "INSERT OR REPLACE INTO exif_extra VALUES (?,?,?,?,?,?)",
                [(self._rel(r[0]), *r[1:]) for r in rows],
            )
            self._con.commit()

    def facet_rows(self) -> list[tuple[str, str, str, int, int]]:
        """(path, make, model, width, height) of every cached image, read with SQL json_extract
        (no per-image JSON parse in Python). Width/height are the upright preview frame."""
        with self._lock:
            rows = self._con.execute(
                "SELECT i.path, json_extract(i.meta_json,'$.make'), "
                "json_extract(i.meta_json,'$.model'), json_extract(f.json,'$.width'), "
                "json_extract(f.json,'$.height') FROM images i "
                "LEFT JOIN features f ON f.path=i.path"
            ).fetchall()
        return [
            (self._abs(r[0]), r[1] or "", r[2] or "", int(r[3] or 0), int(r[4] or 0))
            for r in rows
        ]

    # ---- Lightroom sync (T20) ---------------------------------------------------------------
    def put_written(self, rows: list[tuple[str, int | None, str | None, str]]) -> None:
        """Remember what ShotSieve itself wrote to a sidecar: (path, rating, label, decision).
        A field that was not written this time (None: a foreign value was kept) keeps its old
        value. This is what Lightroom imports from us, so equal values are our opinion, not the
        photographer's."""
        now = time.time()
        with self._lock:
            for path, rating, label, decision in rows:
                self._con.execute(
                    "INSERT INTO lr_written (path, rating, label, decision, ts) VALUES (?,?,?,?,?) "
                    "ON CONFLICT(path) DO UPDATE SET rating=COALESCE(excluded.rating, rating), "
                    "label=COALESCE(excluded.label, label), decision=excluded.decision, "
                    "ts=excluded.ts",
                    (self._rel(path), rating, label, decision, now),
                )
            self._con.commit()

    def written(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            rows = self._con.execute(
                "SELECT path, rating, label, decision, ts FROM lr_written"
            ).fetchall()
        return {
            self._abs(r[0]): {"rating": r[1], "label": r[2], "decision": r[3], "ts": r[4]}
            for r in rows
        }

    def lr_sync_rows(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            rows = self._con.execute(
                "SELECT path, lr_id, rating, pick, label, touch_ts, synced_at FROM lr_sync"
            ).fetchall()
        return {
            self._abs(r[0]): {
                "lr_id": r[1], "rating": r[2], "pick": r[3], "label": r[4], "touch_ts": r[5],
                "synced_at": r[6],
            }
            for r in rows
        }  # fmt: skip

    def put_lr_sync(self, rows: list[tuple[str, int, int | None, int, str | None, float | None]]):
        """Upsert the last seen Lightroom state: (path, lr_id, rating, pick, label, touch_ts)."""
        now = time.time()
        with self._lock:
            self._con.executemany(
                "INSERT OR REPLACE INTO lr_sync VALUES (?,?,?,?,?,?,?)",
                [(self._rel(p), i, r, k, lab, t, now) for p, i, r, k, lab, t in rows],
            )
            self._con.commit()

    def clear_lr_sync(self) -> None:
        with self._lock:
            self._con.execute("DELETE FROM lr_sync")
            self._con.commit()

    # ---- stable ids / lookups ------------------------------------------------------------
    def ids_for(self, paths: list[str]) -> dict[str, int]:
        """Stable integer id per path (assigned on first sight, never reused)."""
        rel = {p: self._rel(p) for p in paths}
        with self._lock:
            self._con.executemany(
                "INSERT OR IGNORE INTO image_ids (path) VALUES (?)", [(r,) for r in rel.values()]
            )
            self._con.commit()
            rows = dict(self._con.execute("SELECT path, id FROM image_ids").fetchall())
        return {p: rows[r] for p, r in rel.items() if r in rows}

    def ids_lookup(self, paths: list[str]) -> dict[str, int]:
        """Existing ids only: a read (ids are assigned at cull time). Any path without an id
        (a database from before T21) is assigned one then, once."""
        rel = {p: self._rel(p) for p in paths}
        with self._lock:
            rows = dict(self._con.execute("SELECT path, id FROM image_ids").fetchall())
        out = {p: rows[r] for p, r in rel.items() if r in rows}
        if len(out) < len(rel):
            out.update(self.ids_for([p for p in rel if p not in out]))
        return out

    def final_decisions_by_id(self) -> tuple[dict[int, str], int, int]:
        """({image id: final decision (your override, else the model's)}, indexable culled images,
        how many of them have a SigLIP2 embedding): one SQL pass, no path handling (T17). Video
        frames (`clip.mp4#t=..`) are never indexed, so they do not count towards completeness."""
        with self._lock:
            self._con.execute(
                "INSERT OR IGNORE INTO image_ids (path) SELECT path FROM decisions"
            )
            self._con.commit()
            rows = self._con.execute(
                "SELECT i.id, COALESCE(o.decision, d.decision) FROM decisions d "
                "JOIN image_ids i ON i.path = d.path LEFT JOIN overrides o ON o.path = d.path"
            ).fetchall()
            done = self._con.execute(
                "SELECT COUNT(*) FROM decisions d JOIN embeddings e ON e.path = d.path "
                "AND e.kind = ? WHERE d.path NOT LIKE '%#t=%'",
                (SIGLIP_KIND,),
            ).fetchone()[0]
            total = self._con.execute(
                "SELECT COUNT(*) FROM decisions WHERE path NOT LIKE '%#t=%'"
            ).fetchone()[0]
        return {int(i): dec for i, dec in rows}, int(total), int(done)

    def siglip_stamp(self) -> tuple[int, int]:
        """(row count, newest rowid) of the SigLIP2 embeddings: changes when the index does."""
        with self._lock:
            n, top = self._con.execute(
                "SELECT COUNT(*), COALESCE(MAX(rowid), 0) FROM embeddings WHERE kind = ?",
                (SIGLIP_KIND,),
            ).fetchone()
        return int(n), int(top)

    def raw_features(self, path: str) -> RawFeatures | None:
        """The stored per-image measurements (faces with boxes and scores), or None."""
        with self._lock:
            row = self._con.execute(
                "SELECT json FROM features WHERE path=?", (self._rel(path),)
            ).fetchone()
        if not row:
            return None
        try:
            return RawFeatures.from_json(row[0])
        except (TypeError, ValueError):
            return None

    def path_for_id(self, image_id: int) -> str | None:
        with self._lock:
            row = self._con.execute("SELECT path FROM image_ids WHERE id=?", (image_id,)).fetchone()
        return self._abs(row[0]) if row else None

    def timestamps(self) -> dict[str, float | None]:
        """Capture timestamp per cached image (None when unknown)."""
        with self._lock:
            rows = self._con.execute("SELECT path, meta_json FROM images").fetchall()
        out: dict[str, float | None] = {}
        for path, meta_json in rows:
            try:
                out[self._abs(path)] = json.loads(meta_json).get("timestamp")
            except (TypeError, ValueError):
                out[self._abs(path)] = None
        return out

    def exif_times(self) -> dict[str, float]:
        """Capture time per image whose timestamp really came from EXIF (not the file mtime)."""
        with self._lock:
            rows = self._con.execute("SELECT path, meta_json FROM images").fetchall()
        out: dict[str, float] = {}
        for path, meta_json in rows:
            try:
                m = json.loads(meta_json)
            except (TypeError, ValueError):
                continue
            if m.get("ts_source") == "exif" and m.get("timestamp") is not None:
                out[self._abs(path)] = float(m["timestamp"])
        return out

    def decision_summary(self) -> dict[str, tuple[str, int, bool]]:
        """{path: (model decision, group id, audit)}: the light read (no feature parsing)."""
        with self._lock:
            rows = self._con.execute(
                "SELECT path, decision, group_id, audit FROM decisions"
            ).fetchall()
        return {self._abs(r[0]): (r[1], int(r[2] or 0), bool(r[3])) for r in rows}

    def scene_embeddings(self) -> dict[str, np.ndarray]:
        """DINOv2 scene embeddings (float32, 384-d) by master path."""
        with self._lock:
            rows = self._con.execute(
                "SELECT path, n, dim, data FROM embeddings WHERE kind='scene'"
            ).fetchall()
        return {self._abs(p): _unpack(blob, n, dim)[0] for p, n, dim, blob in rows}

    def face_embeddings(self) -> dict[str, np.ndarray]:
        """SFace embeddings (K, 128) by master path (faceless: absent): the important faces the
        cull embedded (indices 0..n-1, what person_faces.face_idx has always meant), then the
        recognition-only faces of the T23h scan (indices n..), when that scan is still valid."""
        with self._lock:
            rows = self._con.execute(
                "SELECT path, n, dim, data FROM embeddings WHERE kind='face'"
            ).fetchall()
            rec = self._con.execute(
                "SELECT e.path, e.n, e.dim, e.data, s.fp FROM embeddings e "
                "JOIN index_state s ON s.path=e.path AND s.kind=? WHERE e.kind=?",
                (REC_STATE, REC_KIND),
            ).fetchall()
        out = {self._abs(p): _unpack(blob, n, dim) for p, n, dim, blob in rows}
        for p, n, dim, blob, stamp in rec:
            ap = self._abs(p)
            base = out.get(ap)
            n_imp = 0 if base is None else len(base)
            if not n or not str(stamp).endswith(f"|{n_imp}"):
                continue  # the cull's face count changed since the scan: ignore its faces
            x = _unpack(blob, n, dim)
            out[ap] = x if base is None else np.concatenate([base, x])
        return out

    def image_sigs(self) -> dict[str, str]:
        """{abs path: "size-mtime_ns"} of every culled image (recognition scan key)."""
        with self._lock:
            rows = self._con.execute("SELECT path, size, mtime_ns FROM images").fetchall()
        return {self._abs(p): f"{s}-{m}" for p, s, m in rows}

    def rec_stamps(self) -> dict[str, str]:
        """{abs path: stamp} of the finished recognition scans (stamp = version|fp|n_important)."""
        with self._lock:
            rows = self._con.execute(
                "SELECT path, fp FROM index_state WHERE kind=?", (REC_STATE,)
            ).fetchall()
        return {self._abs(p): fp for p, fp in rows}

    def n_important_faces(self) -> dict[str, int]:
        with self._lock:
            rows = self._con.execute("SELECT path, n FROM embeddings WHERE kind='face'").fetchall()
        return {self._abs(p): int(n) for p, n in rows}

    def put_rec_faces(
        self, path: str, stamp: str, boxes: list[list[float]], emb: np.ndarray, epoch: int | None
    ) -> bool:
        """Store the recognition faces of one image: `boxes` = normalised [x0,y0,x1,y1,score] of
        the display frame, `emb` (K,128). Dropped when people consent was revoked since `epoch`."""
        key = self._rel(path)
        with self.people_write(epoch) as ok:
            if not ok:
                return False
            c = self._con
            if len(emb):
                c.execute(
                    "INSERT OR REPLACE INTO embeddings VALUES (?,?,?,?,?)",
                    (key, REC_KIND, emb.shape[1], len(emb), _pack(emb)),
                )
            else:
                c.execute("DELETE FROM embeddings WHERE path=? AND kind=?", (key, REC_KIND))
            c.execute("INSERT OR REPLACE INTO rec_boxes VALUES (?,?)", (key, json.dumps(boxes)))
            c.execute("INSERT OR REPLACE INTO index_state VALUES (?,?,?)", (key, REC_STATE, stamp))
        return True

    def rec_boxes(self, path: str) -> list[list[float]]:
        with self._lock:
            row = self._con.execute(
                "SELECT json FROM rec_boxes WHERE path=?", (self._rel(path),)
            ).fetchone()
        return json.loads(row[0]) if row else []

    def clock_inputs(self) -> list:
        """One ClockImage per cached image (timestamp, body serial, EXIF time-zone offset)."""
        from shotsieve.grouping.clock import ClockImage, parse_tz

        with self._lock:
            rows = self._con.execute("SELECT path, meta_json FROM images").fetchall()
        out = []
        for path, meta_json in rows:
            try:
                m = json.loads(meta_json)
            except (TypeError, ValueError):
                continue
            if m.get("timestamp") is None:
                continue
            out.append(
                ClockImage(
                    self._abs(path), str(m.get("camera_serial", "unknown")), float(m["timestamp"]),
                    str(m.get("ts_source", "mtime")), parse_tz(m.get("tz_offset")),
                )
            )  # fmt: skip
        return out

    def get_clock_offsets(self) -> dict[str, dict]:
        """meta.clock_offsets: {serial: {offset_s, source: auto|manual|exif[, confidence]}}."""
        return self.get_meta("clock_offsets") or {}

    def set_clock_offset(
        self, serial: str, offset_s: float, source: str, confidence: float | None = None
    ) -> dict[str, dict]:
        """Persist one body's offset; a manual offset is never replaced by an automatic one."""
        from shotsieve.grouping.clock import set_offset

        new = set_offset(self.get_clock_offsets(), serial, offset_s, source, confidence)
        self.set_meta("clock_offsets", new)
        return new

    def clear_clock_offset(self, serial: str) -> None:
        cur = self.get_clock_offsets()
        if cur.pop(serial, None) is not None:
            self.set_meta("clock_offsets", cur)

    def adjusted_timestamps(self) -> dict[str, float | None]:
        """Capture time per image with the persisted body offsets applied (t_adj)."""
        from shotsieve.grouping.clock import adjust, offsets_from_meta

        offsets = offsets_from_meta(self.get_clock_offsets())
        return {c.path: adjust(c.ts, c.serial, offsets) for c in self.clock_inputs()}

    # ---- search index (SigLIP2 embeddings, sqlite-vec) ---------------------------------------
    def _vec_ready(self) -> bool:
        """Load sqlite-vec once per connection and create the vec0 table; False -> numpy path."""
        if getattr(self, "_vec_ok", None) is None:
            self._vec_ok = False
            try:
                import sqlite_vec

                self._con.enable_load_extension(True)
                sqlite_vec.load(self._con)
                self._con.enable_load_extension(False)
                self._con.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS vec_siglip USING vec0("
                    f"id INTEGER PRIMARY KEY, emb float[{SIGLIP_DIM}] distance_metric=cosine)"
                )
                self._vec_ok = True
                self._reconcile_vec()
            except (ImportError, AttributeError, sqlite3.Error) as exc:
                from shotsieve.log import get_logger

                get_logger("cache").info("sqlite-vec unavailable (%s); numpy kNN", exc)
        return self._vec_ok

    def _reconcile_vec(self) -> None:
        """vec0 is derived data: when its row count differs from the embedding blobs (rows were
        written while sqlite-vec could not load), rebuild it from them."""
        n_vec = self._con.execute("SELECT COUNT(*) FROM vec_siglip").fetchone()[0]
        n_blob = self._con.execute(
            "SELECT COUNT(*) FROM embeddings e JOIN image_ids i ON i.path=e.path WHERE e.kind=?",
            (SIGLIP_KIND,),
        ).fetchone()[0]
        if n_vec == n_blob:
            return
        self._con.execute("DELETE FROM vec_siglip")
        rows = self._con.execute(
            "SELECT i.id, e.data FROM embeddings e JOIN image_ids i ON i.path=e.path "
            "WHERE e.kind=?",
            (SIGLIP_KIND,),
        ).fetchall()
        self._con.executemany(
            "INSERT INTO vec_siglip(id, emb) VALUES (?, ?)",
            [(i, _unpack(d, 1, SIGLIP_DIM)[0].tobytes()) for i, d in rows],
        )
        self._con.commit()

    def siglip_current(self, paths_fp: dict[str, str]) -> set[str]:
        """Subset of `paths_fp` (abs path -> content fingerprint) already embedded for that fp."""
        with self._lock:
            rows = dict(
                self._con.execute(
                    "SELECT path, fp FROM index_state WHERE kind=?", (SIGLIP_KIND,)
                ).fetchall()
            )
        return {p for p, fp in paths_fp.items() if rows.get(self._rel(p)) == fp}

    def put_siglip(self, items: list[tuple[str, str, np.ndarray]]) -> None:
        """Store (abs path, fingerprint, 768-d vector) rows: embeddings blob + vec0 row."""
        ids = self.ids_for([p for p, _, _ in items])
        with self._lock:
            vec = self._vec_ready()
            for path, fp, v in items:
                key = self._rel(path)
                v = _unpack(_pack(np.asarray(v).reshape(-1)), 1, SIGLIP_DIM)[0]  # fp16 round trip
                self._con.execute(
                    "INSERT OR REPLACE INTO embeddings VALUES (?,?,?,?,?)",
                    (key, SIGLIP_KIND, SIGLIP_DIM, 1, _pack(v)),
                )
                self._con.execute(
                    "INSERT OR REPLACE INTO index_state VALUES (?,?,?)", (key, SIGLIP_KIND, fp)
                )
                if vec:
                    self._con.execute("DELETE FROM vec_siglip WHERE id=?", (ids[path],))
                    self._con.execute(
                        "INSERT INTO vec_siglip(id, emb) VALUES (?, ?)", (ids[path], v.tobytes())
                    )
            self._con.commit()

    def clear_siglip(self) -> None:
        with self._lock:
            self._con.execute("DELETE FROM embeddings WHERE kind=?", (SIGLIP_KIND,))
            self._con.execute("DELETE FROM index_state WHERE kind=?", (SIGLIP_KIND,))
            if self._vec_ready():
                self._con.execute("DELETE FROM vec_siglip")
            self._con.commit()

    def siglip_count(self) -> int:
        with self._lock:
            return self._con.execute(
                "SELECT COUNT(*) FROM embeddings WHERE kind=?", (SIGLIP_KIND,)
            ).fetchone()[0]

    def siglip_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """(image ids int64 (N,), embeddings float32 (N, 768)) from the source-of-truth blobs."""
        with self._lock:
            rows = self._con.execute(
                "SELECT i.id, e.data FROM embeddings e JOIN image_ids i ON i.path=e.path "
                "WHERE e.kind=? ORDER BY i.id",
                (SIGLIP_KIND,),
            ).fetchall()
        if not rows:
            return np.zeros(0, np.int64), np.zeros((0, SIGLIP_DIM), np.float32)
        ids = np.array([r[0] for r in rows], np.int64)
        return ids, np.stack([_unpack(r[1], 1, SIGLIP_DIM)[0] for r in rows])

    def knn(
        self, q: np.ndarray, k: int, allowed: set[int] | None = None, use_vec: bool = True
    ) -> list[tuple[int, float]]:
        """Top-k (image id, cosine similarity), best first. sqlite-vec when available (and no
        `allowed` filter), else a numpy brute-force matmul."""
        q = np.asarray(q, np.float32).reshape(-1)
        if use_vec and allowed is None:
            with self._lock:
                if self._vec_ready():
                    rows = self._con.execute(
                        "SELECT id, distance FROM vec_siglip WHERE emb MATCH ? AND k=? "
                        "ORDER BY distance",
                        (q.tobytes(), k),
                    ).fetchall()
                    if rows:
                        return [(int(i), 1.0 - float(d)) for i, d in rows]
        ids, mat = self.siglip_matrix()
        if allowed is not None:
            keep = np.fromiter((int(i) in allowed for i in ids), bool, len(ids))
            ids, mat = ids[keep], mat[keep]
        if not len(ids):
            return []
        sims = mat @ q / max(float(np.linalg.norm(q)), 1e-12)
        top = np.argsort(-sims)[:k]
        return [(int(ids[i]), float(sims[i])) for i in top]

    # ---- jerseys (T11) ----------------------------------------------------------------------
    def ocr_current(self, paths_fp: dict[str, str]) -> set[str]:
        """Subset of `paths_fp` already OCR'd at that fingerprint."""
        with self._lock:
            rows = dict(
                self._con.execute("SELECT path, fp FROM index_state WHERE kind='ocr'").fetchall()
            )
        return {p for p, fp in paths_fp.items() if rows.get(self._rel(p)) == fp}

    def put_jerseys(
        self, path: str, fp: str, rows: list[dict[str, Any]], epoch: int | None = None
    ) -> None:
        """Replace one image's directly-read numbers (`source='ocr'`) and mark it OCR'd. With
        `epoch`, face links are dropped when people consent was revoked since it was read."""
        key = self._rel(path)
        with self.people_lock(), self._immediate():
            if epoch is not None and (
                self.people_epoch() != epoch or self.get_meta("people_consent") is False
            ):
                rows = [{**r, "face_idx": None, "person_id": None} for r in rows]
            self._con.execute("DELETE FROM jerseys WHERE path=? AND source='ocr'", (key,))
            self._con.executemany(
                "INSERT INTO jerseys VALUES (?,?,?,?,?,?,?,?,?,?)",
                [
                    (key, r["number"], r["conf"], r["x"], r["y"], r["w"], r["h"], "ocr",
                     r.get("face_idx"), r.get("person_id"))
                    for r in rows
                ],
            )  # fmt: skip
            self._con.execute("INSERT OR REPLACE INTO index_state VALUES (?,?,?)", (key, "ocr", fp))

    def replace_burst_jerseys(self, rows: list[dict[str, Any]]) -> None:
        with self._lock:
            self._con.execute("DELETE FROM jerseys WHERE source='burst'")
            self._con.executemany(
                "INSERT INTO jerseys VALUES (?,?,?,?,?,?,?,?,?,?)",
                [
                    (self._rel(r["path"]), r["number"], r["conf"], -1, -1, -1, -1, "burst",
                     None, r.get("person_id"))
                    for r in rows
                ],
            )  # fmt: skip
            self._con.commit()

    def jerseys(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._con.execute(
                "SELECT path, number, conf, x, y, w, h, source, face_idx, person_id FROM jerseys"
            ).fetchall()
        keys = ("path", "number", "conf", "x", "y", "w", "h", "source", "face_idx", "person_id")
        out = [dict(zip(keys, r, strict=True)) for r in rows]
        for r in out:
            r["path"] = self._abs(r["path"])
        return out

    def clear_jerseys(self) -> None:
        with self._lock:
            self._con.execute("DELETE FROM jerseys")
            self._con.execute("DELETE FROM index_state WHERE kind='ocr'")
            self._con.commit()

    def set_roster(self, entries: list[tuple[str, str, str]]) -> None:
        with self._lock:
            self._con.execute("DELETE FROM roster2")
            self._con.executemany(
                "INSERT OR REPLACE INTO roster2 (number, name, team) VALUES (?,?,?)",
                [(n, name, team or "") for n, name, team in entries],
            )
            self._con.commit()

    def roster(self) -> dict[str, list[tuple[str, str]]]:
        """number -> [(name, team), ...] (a number can be on several teams)."""
        with self._lock:
            rows = self._con.execute(
                "SELECT number, name, team FROM roster2 ORDER BY team, number"
            ).fetchall()
        out: dict[str, list[tuple[str, str]]] = {}
        for n, name, team in rows:
            out.setdefault(n, []).append((name, team or ""))
        return out

    def roster_team(self) -> str:
        """The folder's team (a per-folder setting: whose frames these are)."""
        return str(self.get_meta("roster_team", "") or "")

    # ---- people ---------------------------------------------------------------------------
    def people_lock(self) -> threading.RLock:
        """One lock per database file (a Cache per request means several connections): holds
        revocation and the final write of a clustering run apart."""
        with _PEOPLE_LOCKS_GUARD:
            return _PEOPLE_LOCKS.setdefault(str(self.path.resolve()), threading.RLock())

    def people_epoch(self) -> int:
        """Bumped on every revocation; a face job compares it before it writes."""
        return int(self.get_meta("people_epoch", 0) or 0)

    @contextmanager
    def _immediate(self, commit: bool = True) -> Iterator[None]:
        """One cross-process write transaction (`BEGIN IMMEDIATE`; the busy timeout makes a
        second writer wait). Re-entrant: an inner block joins the outer transaction. Commits on
        a clean exit when `commit`, rolls back on an exception. Helpers called inside use
        `_commit`, which does nothing while the transaction is open."""
        with self._lock:
            if self._in_tx:
                yield
                return
            self._con.commit()  # close any implicit transaction first
            self._con.execute("BEGIN IMMEDIATE")
            self._in_tx = True
            try:
                yield
            except BaseException:
                self._in_tx = False
                self._con.rollback()
                raise
            self._in_tx = False
            if commit:
                self._con.commit()
            else:
                self._con.rollback()

    @contextmanager
    def people_write(self, epoch: int | None) -> Iterator[bool]:
        """Yields False (write nothing) when consent was revoked since `epoch` was read, or is
        explicitly off. The check and the caller's writes share one `BEGIN IMMEDIATE`
        transaction, so a revocation in another process cannot interleave."""
        with self.people_lock(), self._immediate(commit=True) as _:
            ok = (epoch is None or self.people_epoch() == epoch) and (
                self.get_meta("people_consent") is not False
            )
            yield ok

    def wipe_people(self, keep_face_idx: bool = False) -> None:
        """Delete everything derived from faces: clusters, names and the jersey -> face links
        (a re-cluster passes `keep_face_idx`: the face a number was read next to is still true,
        only the person ids it pointed at are stale)."""
        with self.people_lock(), self._immediate():
            self._con.execute("DELETE FROM person_faces")
            self._con.execute("DELETE FROM people")
            if not keep_face_idx:  # a real wipe (revocation / delete) also forgets the exclusions
                self._con.execute("DELETE FROM face_excluded")
                self._con.execute("DELETE FROM embeddings WHERE kind=?", (REC_KIND,))
                self._con.execute("DELETE FROM rec_boxes")
                self._con.execute("DELETE FROM index_state WHERE kind=?", (REC_STATE,))
            if keep_face_idx:
                self._con.execute("UPDATE jerseys SET person_id=NULL")
            else:
                self._con.execute("UPDATE jerseys SET face_idx=NULL, person_id=NULL")

    def revoke_people(self) -> None:
        """Consent off / DELETE: cancels in-flight face jobs (epoch), then wipes."""
        with self.people_lock(), self._immediate():
            self.set_meta("people_consent", False)
            self.set_meta("people_epoch", self.people_epoch() + 1)
            self.wipe_people()

    def people_consent(self, default: bool = False) -> bool:
        return bool(self.get_meta("people_consent", default))

    # ---- kv -------------------------------------------------------------------------------
    def _commit(self) -> None:
        if not self._in_tx:
            self._con.commit()

    def set_meta(self, key: str, value: Any) -> None:
        with self._lock:
            self._con.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, json.dumps(value)))
            self._commit()

    def get_meta(self, key: str, default: Any = None) -> Any:
        with self._lock:
            row = self._con.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def get_thresholds(self) -> dict[str, float] | None:
        """Per-shoot thresholds (t_keep / t_reject / hard_margin / dup_margin) persisted by
        calibration, if any."""
        return self.get_meta("thresholds")

    def set_thresholds(self, thresholds: dict[str, float]) -> None:
        self.set_meta("thresholds", thresholds)

    def get_audit_sample(self) -> dict[str, dict] | None:
        """The frozen audit sample {absolute path: {"p": inclusion prob, "k": stratum}}."""
        raw = self.get_meta("audit_sample")
        return None if raw is None else {self._abs(k): v for k, v in raw.items()}

    def set_audit_sample(self, sample: dict[str, dict] | None) -> None:
        if sample is None:
            with self._lock:
                self._con.execute("DELETE FROM meta WHERE key='audit_sample'")
                self._con.commit()
        else:
            self.set_meta("audit_sample", {self._rel(k): v for k, v in sample.items()})
