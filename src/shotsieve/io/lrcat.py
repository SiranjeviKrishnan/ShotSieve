"""Read a Lightroom catalog (.lrcat, SQLite) into catalog entries. Strictly read-only.

Safety rules (docs/10_PHASE4.md T20):
* the user's catalog is NEVER opened by SQLite (no TOCTOU gap): every read goes through a private
  copy under `data/tmp/` (APFS clone when possible, plain file reads of the original, with its
  -wal / -shm / -journal copied alongside). The *copy* is opened normally (folding its WAL in)
  and checked with `PRAGMA quick_check`; a failing copy (torn by a live Lightroom) is retried
  3 times, then refused (`CatalogBusyError`). The temp dir is always deleted;
* the original's bytes, mtimes and sidecar files are never touched.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

from shotsieve.config import data_dir

CORE_EPOCH = 978307200  # Lightroom times count seconds from 2001-01-01 UTC
COPY_RETRIES = 3
COPY_WAIT_S = 0.5
SIDE_SUFFIXES = ("", "-wal", "-shm", "-journal")
BUSY_MESSAGE = "Lightroom is writing the catalog; quit Lightroom and sync again."


class CatalogError(ValueError):
    """The catalog path is unusable (not absolute, not a .lrcat file, `..` or NUL in it)."""


class CatalogBusyError(RuntimeError):
    """The catalog is locked and a private copy failed the integrity check (3 tries)."""


@dataclass(frozen=True)
class CatalogEntry:
    path: str
    rating: int | None
    pick: int  # 1 flagged, 0 none, -1 rejected
    label: str | None
    touch_ts: float | None = None  # Adobe_images.touchTime as a Unix timestamp
    capture: str | None = None  # Adobe_images.captureTime ("2024-06-01T14:00:00")
    id_local: int = 0
    original_filename: str | None = None
    sidecar_exts: str | None = None


def check_catalog_path(path: str | Path) -> Path:
    """An absolute existing file ending in `.lrcat`, without `..` segments or NUL bytes."""
    raw = str(path)
    if "\x00" in raw:
        raise CatalogError("invalid catalog path")
    p = Path(raw).expanduser()
    if ".." in p.parts:
        raise CatalogError("path traversal is not allowed in the catalog path")
    if not p.is_absolute():
        raise CatalogError("the catalog path must be absolute")
    if p.suffix.lower() != ".lrcat":
        raise CatalogError("not a Lightroom catalog (.lrcat)")
    if not p.is_file():
        raise CatalogError(f"catalog not found: {raw}")
    return p


def ro_uri(db: Path) -> str:
    """`file:<abs path>?mode=ro` with the path URI-escaped."""
    return "file:" + quote(os.path.abspath(db)) + "?mode=ro"


def open_ro(db: Path) -> sqlite3.Connection:
    return sqlite3.connect(ro_uri(db), uri=True, timeout=1.0)


_BASE = """
SELECT rf.absolutePath, f.pathFromRoot, fi.baseName, fi.extension,
       i.rating, i.pick, i.colorLabels, i.id_local{extra}
FROM Adobe_images i
JOIN AgLibraryFile fi ON fi.id_local = i.rootFile
JOIN AgLibraryFolder f ON f.id_local = fi.folder
JOIN AgLibraryRootFolder rf ON rf.id_local = f.rootFolder
"""


def _like_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _query_rows(
    db: Path, prefix: str | None = None, names: list[str] | None = None
) -> tuple[list[tuple], list[str]]:
    """(rows, optional column names). `prefix` filters on the full folder path in SQL, `names`
    on the lower-cased file name (copy-import fallback)."""
    con = open_ro(db)
    try:
        cols = {r[1] for r in con.execute("PRAGMA table_info(Adobe_images)")}
        fcols = {r[1] for r in con.execute("PRAGMA table_info(AgLibraryFile)")}
        extra, have = "", []
        for col, table in (
            ("touchTime", "i"), ("captureTime", "i"), ("originalFilename", "fi"),
            ("sidecarExtensions", "fi"),
        ):  # fmt: skip
            present = col in (cols if table == "i" else fcols)
            extra += f", {table}.{col}" if present else ", NULL"
            have.append(col if present else "")
        where, params = [], []
        if "masterImage" in cols:
            where.append("i.masterImage IS NULL")
        if prefix:
            where.append("(rf.absolutePath || f.pathFromRoot) LIKE ? ESCAPE '\\'")
            params.append(_like_escape(prefix) + "%")
        sql = _BASE.format(extra=extra)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY i.id_local"
        if names is None:
            return con.execute(sql, params).fetchall(), have
        want = {n.casefold() for n in names}
        out = []
        for row in con.execute(sql, params):
            nm = f"{row[2]}.{row[3]}" if row[3] else str(row[2])
            if nm.casefold() in want:
                out.append(row)
        return out, have
    finally:
        con.close()


def _clone(src: Path, dst: Path) -> None:
    """Copy `src` to `dst`: APFS clone (`cp -c`) when possible, else a plain copy. The copy is
    made writable so SQLite can fold its WAL in."""
    try:
        r = subprocess.run(
            ["cp", "-c", str(src), str(dst)], capture_output=True, timeout=600, check=False
        )
        ok = r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        ok = False
    if not ok:
        shutil.copyfile(src, dst)
    os.chmod(dst, 0o600)


STALE_COPY_S = 3600.0  # a private copy older than this belongs to a killed sync


def sweep_stale_copies(max_age_s: float = STALE_COPY_S) -> int:
    """Delete `data/tmp/lrcat-*` directories older than `max_age_s` (only our prefix). Returns the
    number removed."""
    root = data_dir() / "tmp"
    n = 0
    try:
        entries = list(root.glob("lrcat-*"))
    except OSError:
        return 0
    now = time.time()
    for d in entries:
        try:
            if d.is_dir() and not d.is_symlink() and now - d.stat().st_mtime > max_age_s:
                shutil.rmtree(d, ignore_errors=True)
                n += 1
        except OSError:
            pass
    return n


def _private_copy(src: Path) -> tuple[Path, Path]:
    """(temp dir, copy of the catalog) with its -wal / -shm / -journal copied alongside."""
    root = data_dir() / "tmp"
    root.mkdir(parents=True, exist_ok=True)
    sweep_stale_copies()
    tmp = Path(tempfile.mkdtemp(prefix="lrcat-", dir=root))
    try:
        for suffix in SIDE_SUFFIXES:
            f = src.with_name(src.name + suffix)
            if f.exists():
                _clone(f, tmp / f.name)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return tmp, tmp / src.name


def _copy_checked(src: Path) -> Iterator[tuple[Path, Path]]:
    """Yield (tmpdir, intact copy), retrying; raise CatalogBusyError when none passes."""
    for attempt in range(COPY_RETRIES):
        tmp, dst = _private_copy(src)
        try:
            con = sqlite3.connect(dst, timeout=1.0)  # the COPY, opened normally (WAL folded in)
            try:
                ok = con.execute("PRAGMA quick_check").fetchone()[0] == "ok"
            finally:
                con.close()
        except sqlite3.DatabaseError:
            ok = False
        if ok:
            yield tmp, dst
            return
        shutil.rmtree(tmp, ignore_errors=True)
        if attempt + 1 < COPY_RETRIES:
            time.sleep(COPY_WAIT_S)
    raise CatalogBusyError(BUSY_MESSAGE)


def is_wal_catalog(src: Path) -> bool:
    """True when the SQLite header says WAL (file-format version bytes 18 and 19 are 2), or the
    header cannot be read. A plain binary read: SQLite never opens the user's file for this."""
    try:
        with open(src, "rb") as fh:
            head = fh.read(20)
    except OSError:
        return True
    return len(head) < 20 or head[18] == 2 or head[19] == 2


def _has_side_files(src: Path) -> bool:
    for suffix in SIDE_SUFFIXES[1:]:
        f = src.with_name(src.name + suffix)
        try:
            if f.exists() and f.stat().st_size > 0:
                return True
        except OSError:
            pass
    return False


class CatalogReader:
    """Read-only access to one catalog (a context manager).

    `with CatalogReader(path) as cat: cat.entries(prefix=...)`: a
    private checked copy is made (always, see the module docstring) and deleted on exit.
    `cat.copied` is always True; it is kept for the sync report.
    """

    def __init__(self, path: str | Path) -> None:
        self.src = check_catalog_path(path).resolve()
        self.copied = False
        self._tmp: Path | None = None
        self._db: Path | None = None

    def __enter__(self) -> CatalogReader:
        # always a private copy: even a mode=ro open of a closed WAL db creates -shm/-wal beside
        # it, and a catalog can switch journal mode between a check and an open (TOCTOU)
        self._switch_to_copy()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._tmp is not None:
            shutil.rmtree(self._tmp, ignore_errors=True)
            self._tmp = None

    def _switch_to_copy(self) -> None:
        for tmp, dst in _copy_checked(self.src):
            self._tmp, self._db, self.copied = tmp, dst, True

    def query(self, fn: Callable[[Path], Any]) -> Any:
        """Run `fn(db_path)` on the private checked copy (never on the original)."""
        assert self._db is not None, "use CatalogReader as a context manager"
        return fn(self._db)

    def entries(
        self, prefix: str | None = None, names: Iterable[str] | None = None
    ) -> list[CatalogEntry]:
        """Catalog entries (virtual copies skipped when the column exists). `prefix` (a folder
        path) is pushed into the SQL; `names` keeps only files with those names."""
        nm = None if names is None else list(names)
        rows, have = self.query(lambda db: _query_rows(db, prefix, nm))
        out = []
        for root, rel, base, ext, rating, pick, label, idl, *opt in rows:
            full = f"{root or ''}{rel or ''}{base}" + (f".{ext}" if ext else "")
            touch, capture, orig, side = opt
            out.append(
                CatalogEntry(
                    full,
                    None if rating is None else int(float(rating)),
                    int(float(pick or 0)),
                    label or None,
                    None if touch is None else float(touch) + CORE_EPOCH,
                    capture or None,
                    int(idl or 0),
                    orig or None,
                    side or None,
                )
            )
        return out


@contextmanager
def _reader(path: str | Path) -> Iterator[CatalogReader]:
    with CatalogReader(path) as r:
        yield r


def _read_locked(src: Path, query: Callable[[Path], Any]) -> Any:
    """Run `query(db_path)` on the catalog (see `CatalogReader.query`)."""
    with CatalogReader(src) as r:
        return r.query(query)


def read_catalog(path: str | Path, folder_prefix: str | None = None) -> list[CatalogEntry]:
    """Parse a catalog read-only (`CatalogReader`). Optional columns (`touchTime`, `captureTime`,
    `originalFilename`, `sidecarExtensions`) are read when present; `folder_prefix` filters in SQL
    so a 100k-image catalog is not fully materialised."""
    with CatalogReader(path) as r:
        return r.entries(prefix=folder_prefix)


# ---- develop settings (T15) -----------------------------------------------------------------


@dataclass(frozen=True)
class DevelopEntry:
    path: str
    fmt: str  # raw | dng | jpeg | other
    settings: dict[str, float | str]  # top-level scalar keys of the develop text
    pv: float  # process version (e.g. 11.0), 0.0 when unknown


_DEVELOP_QUERY = """
SELECT rf.absolutePath, f.pathFromRoot, fi.baseName, fi.extension, d.text, {pv}
FROM Adobe_imageDevelopSettings d
JOIN Adobe_images i ON i.id_local = d.image
JOIN AgLibraryFile fi ON fi.id_local = i.rootFile
JOIN AgLibraryFolder f ON f.id_local = fi.folder
JOIN AgLibraryRootFolder rf ON rf.id_local = f.rootFolder
"""
_PV2012_KEYS = (
    "Exposure2012", "Contrast2012", "Highlights2012", "Shadows2012", "Whites2012", "Blacks2012"
)  # fmt: skip


def parse_lua_table(text: str) -> dict[str, float | str]:
    """Top-level scalar entries of a Lightroom develop-settings Lua table (`s = { K = v, ... }`).

    Nested tables (`ToneCurvePV2012 = { ... }`) are skipped by a brace-depth scanner that is
    aware of quoted strings. Numbers become floats, strings stay strings, booleans `"True"` /
    `"False"`.
    """
    i = text.find("{")
    if i < 0:
        return {}
    n, i, out = len(text), i + 1, {}
    depth = 1
    while i < n and depth > 0:
        c = text[i]
        if c in " \t\r\n,;":
            i += 1
            continue
        if c == "}":
            depth -= 1
            i += 1
            continue
        if depth != 1:
            i += 1
            continue
        # a key: identifier or ["string"]
        if c == "[":
            j = text.find("]", i)
            if j < 0:
                break
            key = text[i + 1 : j].strip().strip("\"'")
            i = j + 1
        else:
            j = i
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            if j == i:  # unexpected character: skip it
                i += 1
                continue
            key, i = text[i:j], j
        while i < n and text[i] in " \t\r\n":
            i += 1
        if i >= n or text[i] != "=":
            continue  # a bare list item
        i += 1
        while i < n and text[i] in " \t\r\n":
            i += 1
        if i >= n:
            break
        c = text[i]
        if c == "{":  # nested table: skip to its matching brace (strings may hold braces)
            d, i = 1, i + 1
            while i < n and d > 0:
                if text[i] == '"':
                    i += 1
                    while i < n and text[i] != '"':
                        i += 2 if text[i] == "\\" else 1
                elif text[i] == "{":
                    d += 1
                elif text[i] == "}":
                    d -= 1
                i += 1
        elif c == '"':
            i += 1
            buf = []
            while i < n and text[i] != '"':
                if text[i] == "\\" and i + 1 < n:
                    i += 1
                buf.append(text[i])
                i += 1
            i += 1
            out[key] = "".join(buf)
        else:
            j = i
            while j < n and text[j] not in ",;}\r\n":
                j += 1
            raw = text[i:j].strip()
            i = j
            if raw in ("true", "false"):
                out[key] = raw.capitalize()
            else:
                try:
                    out[key] = float(raw)
                except ValueError:
                    out[key] = raw
    return out


def _fmt_of(ext: str) -> str:
    from shotsieve.io.formats import JPEG_EXTS, RAW_EXTS

    e = "." + ext.lower().lstrip(".")
    if e == ".dng":
        return "dng"
    if e in RAW_EXTS:
        return "raw"
    if e in JPEG_EXTS:
        return "jpeg"
    return "other"


def _pv_float(v: object) -> float:
    try:
        return float(str(v))
    except (TypeError, ValueError):
        return 0.0


def _develop_rows(db: Path) -> list[tuple]:
    con = open_ro(db)
    try:
        cols = {r[1] for r in con.execute("PRAGMA table_info(Adobe_images)")}
        dcols = {r[1] for r in con.execute("PRAGMA table_info(Adobe_imageDevelopSettings)")}
        pv = "d.processVersion" if "processVersion" in dcols else "NULL"
        query = _DEVELOP_QUERY.format(pv=pv)
        if "masterImage" in cols:
            query += " WHERE i.masterImage IS NULL"
        return con.execute(query + " ORDER BY i.id_local").fetchall()
    finally:
        con.close()


def read_develop(path: str | Path) -> list[DevelopEntry]:
    """Develop settings of every image in a catalog (read-only, virtual copies skipped).

    Only PV2012+ rows (Exposure2012 or another *2012 key present, or processVersion >= 6.7) with
    at least one edited value are kept: a target that differs from its default (sliders 0,
    white balance other than "As Shot").
    """
    rows = _read_locked(path, _develop_rows)
    out: list[DevelopEntry] = []
    for root, rel, base, ext, text, pv_col in rows:
        if not text:
            continue
        s = parse_lua_table(text)
        pv = _pv_float(s.get("ProcessVersion", pv_col))
        if not (any(k in s for k in _PV2012_KEYS) or pv >= 6.7):
            continue
        if not edited(s):
            continue
        full = f"{root or ''}{rel or ''}{base}" + (f".{ext}" if ext else "")
        out.append(DevelopEntry(full, _fmt_of(ext or ""), s, pv))
    return out


def read_develop_state(path: str | Path) -> dict[str, tuple[bool, bool]]:
    """{absolute path: (has edited develop settings, has a crop)} for every catalog image that has
    a develop row (read-only copy; virtual copies skipped). T22 uses it to decide whether a
    sidecar crop write could collide with what Lightroom holds."""
    rows = _read_locked(path, _develop_rows)
    out: dict[str, tuple[bool, bool]] = {}
    for root, rel, base, ext, text, _pv in rows:
        if not text:
            continue
        s = parse_lua_table(text)
        full = f"{root or ''}{rel or ''}{base}" + (f".{ext}" if ext else "")
        crop = str(s.get("HasCrop", "False")).lower() == "true"
        out[full] = (edited(s), crop)
    return out


_TARGET_KEYS = (
    *_PV2012_KEYS, "Vibrance", "Saturation", "IncrementalTemperature", "IncrementalTint"
)


def edited(settings: dict[str, float | str]) -> bool:
    """At least one target differs from its default."""
    for k in _TARGET_KEYS:
        v = settings.get(k)
        if isinstance(v, float) and abs(v) > 1e-9:
            return True
    return str(settings.get("WhiteBalance", "As Shot")) not in ("As Shot", "Auto") and any(
        k in settings for k in ("Temperature", "Tint")
    )
