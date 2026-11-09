"""Training rows for the editing profile (T15): catalog / XMP develop settings -> features, targets.

Features come from the embedded/camera preview (what prediction sees) and are cached in
`edit_features.db` keyed by `cache.fingerprint`; images of shoots that were already culled reuse
their DINOv2 embedding and EXIF from `<shoot>/.shotsieve/cull.db`. Labels that ShotSieve itself
wrote (sidecar provenance, or a catalog value equal to a stored prediction) are excluded.
"""

from __future__ import annotations

import sqlite3
import threading
from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from shotsieve.config import Settings, data_dir
from shotsieve.features.color_stats import EDIT_STATS_VERSION, N_STATS, edit_stats
from shotsieve.features.raw_wb import as_shot
from shotsieve.io import crs, dng_xmp, lrcat
from shotsieve.io.exif import ImageMeta, read_meta
from shotsieve.io.formats import RAW_EXTS, SUPPORTED_EXTS, preview_source_for
from shotsieve.io.preview import PreviewError, load_preview
from shotsieve.log import get_logger
from shotsieve.personal.edit_profile import (
    ALL_KEYS,
    FMT_DNG,
    FMT_OTHER,
    FMT_RAW,
    TONE_KEYS,
    EditSet,
)
from shotsieve.pipeline.cache import DB_NAME, Cache, fingerprint

log = get_logger("edit_dataset")

TINT_PER_DUV = -3000.0  # as-shot tint proxy: tint ~ -3000 x Delta-uv (used only as a difference)
CARRY_FRACTION = 0.8
# keys the carry set never copies: our own targets, white balance, and per-image geometry
_NO_CARRY = frozenset(
    {
        *TONE_KEYS, "Temperature", "Tint", "WhiteBalance", "IncrementalTemperature",
        "IncrementalTint", "ProcessVersion", "Version", "HasSettings", "CameraProfileDigest",
        "CropTop", "CropLeft", "CropBottom", "CropRight", "CropAngle", "CropConstrainToWarp",
        "HasCrop", "RawFileName", "GrainSeed", "PostCropVignetteSeed", "UprightVersion",
        "UprightCenterMode", "AlreadyApplied", "OverrideLookVignette", "ToneCurveName2012",
    }
)  # fmt: skip


def format_class(path: str | Path) -> int:
    ext = Path(path).suffix.lower()
    if ext == ".dng":
        return FMT_DNG
    return FMT_RAW if ext in RAW_EXTS else FMT_OTHER


def features_db_path() -> Path:
    return data_dir() / "edit_features.db"


# ---- targets ----------------------------------------------------------------------------------


def _f(v: object, default: float = 0.0) -> float:
    return float(v) if isinstance(v, int | float) else default


def targets_from_settings(
    s: dict[str, float | str], fmt: int, wb: tuple[float, float] | None
) -> dict[str, float]:
    """Target per key (NaN = not applicable) from a develop-settings dict.

    RAW/DNG white balance is relative to the as-shot: mired(Temperature) - as-shot mired and
    Tint - as-shot tint proxy ("As Shot" rows give 0); JPEG uses IncrementalTemperature/Tint.
    """
    y = {k: float("nan") for k in ALL_KEYS}
    for k in TONE_KEYS:
        y[k] = _f(s.get(k))
    if fmt <= FMT_DNG:
        if str(s.get("WhiteBalance")) == "Auto":
            pass  # Auto is a change we cannot express relative to as-shot: no WB row
        elif str(s.get("WhiteBalance", "As Shot")) == "As Shot" or "Temperature" not in s:
            if wb is not None:
                y["raw_temp"] = y["raw_tint"] = 0.0
        elif wb is not None:
            t = _f(s.get("Temperature"), 0.0)
            if t > 0:
                y["raw_temp"] = 1e6 / t - wb[0]
                y["raw_tint"] = _f(s.get("Tint")) - TINT_PER_DUV * wb[1]
    else:
        y["jpeg_temp"] = _f(s.get("IncrementalTemperature"))
        y["jpeg_tint"] = _f(s.get("IncrementalTint"))
    return y


def carry_set(rows: Iterable[dict[str, float | str]]) -> dict[str, float | str]:
    """crs keys (profile, lens corrections, sharpening, NR, HSL...) where >= 80 % of the user's
    edits share one value. Written verbatim so `HasSettings=True` does not wipe the user's
    import preset."""
    rows = list(rows)
    if not rows:
        return {}
    out: dict[str, float | str] = {}
    keys = {k for r in rows for k in r if k not in _NO_CARRY}
    for k in sorted(keys):
        vals = [r[k] for r in rows if k in r]
        if len(vals) < CARRY_FRACTION * len(rows):
            continue
        v, n = Counter(vals).most_common(1)[0]
        if n >= CARRY_FRACTION * len(rows):
            out[k] = v
    return out


# ---- feature store ----------------------------------------------------------------------------


class FeatureStore:
    """`edit_features.db`: statistics + DINOv2 embedding per content fingerprint."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else features_db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._con = sqlite3.connect(self.path, check_same_thread=False)
        self._con.execute("PRAGMA journal_mode=WAL")
        self._con.execute(
            "CREATE TABLE IF NOT EXISTS feats (key TEXT PRIMARY KEY, ver INTEGER NOT NULL, "
            "stats BLOB NOT NULL, emb BLOB NOT NULL, fmt INTEGER NOT NULL)"
        )
        self._con.commit()

    def get_many(self, keys: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray, int]]:
        out: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
        with self._lock:
            for i in range(0, len(keys), 500):
                chunk = keys[i : i + 500]
                q = ",".join("?" * len(chunk))
                for k, ver, st, em, fm in self._con.execute(
                    f"SELECT key, ver, stats, emb, fmt FROM feats WHERE key IN ({q})", chunk
                ):
                    if ver == EDIT_STATS_VERSION:
                        out[k] = (
                            np.frombuffer(st, np.float32).copy(),
                            np.frombuffer(em, np.float16).astype(np.float32),
                            int(fm),
                        )
        return out

    def put_many(self, rows: list[tuple[str, np.ndarray, np.ndarray, int]]) -> None:
        with self._lock:
            self._con.executemany(
                "INSERT OR REPLACE INTO feats VALUES (?,?,?,?,?)",
                [
                    (k, EDIT_STATS_VERSION, np.asarray(st, np.float32).tobytes(),
                     np.asarray(em, np.float16).tobytes(), fm)
                    for k, st, em, fm in rows
                ],
            )  # fmt: skip
            self._con.commit()

    def close(self) -> None:
        with self._lock:
            self._con.close()


@dataclass
class Feat:
    stats: np.ndarray
    emb: np.ndarray
    fmt: int


def _shoot_root(path: Path) -> Path | None:
    for parent in list(path.parents)[:4]:
        if (parent / ".shotsieve" / DB_NAME).exists():
            return parent
    return None


def _render_preview(path: Path) -> np.ndarray:
    """rawpy half-size render (`--dng-render`) for DNGs whose embedded preview carries an edit."""
    import rawpy

    with rawpy.imread(str(path)) as raw:
        return raw.postprocess(half_size=True, use_camera_wb=True, output_bps=8)


def featurise(
    paths: list[str],
    settings: Settings | None = None,
    progress: Callable[[str, int, int], None] | None = None,
    *,
    dng_render: bool = False,
    store: FeatureStore | None = None,
) -> dict[str, Feat]:
    """Features of each readable path (unreadable ones are absent from the result).

    Order of lookups: the feature store (by content fingerprint), then the DINOv2 embedding/EXIF
    of an already-culled shoot, then a fresh decode of the preview (embedder loaded lazily).
    """
    settings = settings or Settings()
    own_store = store is None
    store = store or FeatureStore()
    out: dict[str, Feat] = {}
    try:
        keys: dict[str, str] = {}
        for p in paths:
            try:
                tag = "|render" if dng_render and p.lower().endswith(".dng") else ""
                keys[p] = fingerprint(p) + tag
            except OSError:
                continue
        hits = store.get_many(list(keys.values()))
        todo: list[str] = []
        for p, k in keys.items():
            if k in hits:
                st, em, fm = hits[k]
                out[p] = Feat(st, em, fm)
            else:
                todo.append(p)
        total = len(paths)
        if progress:
            progress("features", len(out), total)
        if not todo:
            return out
        # cached embeddings / EXIF of culled shoots
        cached: dict[str, tuple[ImageMeta, np.ndarray]] = {}
        by_root: dict[Path, list[str]] = {}
        for p in todo:
            r = _shoot_root(Path(p))
            if r is not None:
                by_root.setdefault(r, []).append(p)
        for root, ps in by_root.items():
            try:
                stats = {}
                for p in ps:
                    stt = Path(p).stat()
                    stats[p] = (stt.st_size, stt.st_mtime_ns)
                with Cache.for_folder(root) as c:
                    for p, ci in c.lookup_many(stats, fp_of=fingerprint).items():
                        cached[p] = (ci.meta, ci.scene)
            except Exception as exc:  # noqa: BLE001 - a broken cache only costs recomputation
                log.warning("cull cache of %s unreadable (%s)", root, exc)

        def prep(p: str) -> tuple[str, np.ndarray, ImageMeta, tuple[float, float] | None] | None:
            path = Path(p)
            try:
                if dng_render and path.suffix.lower() == ".dng":
                    img = _render_preview(path)
                    import cv2

                    s = settings.preview_max_edge / max(img.shape[:2])
                    if s < 1:
                        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                else:
                    img = load_preview(preview_source_for(path), settings.preview_max_edge)
            except (PreviewError, OSError, Exception) as exc:  # noqa: BLE001
                log.warning("cannot decode %s: %s", p, exc)
                return None
            meta = cached[p][0] if p in cached else read_meta(path)
            wb = as_shot(path) if path.suffix.lower() in RAW_EXTS else None
            return p, img, meta, wb

        embedder = None
        new_rows: list[tuple[str, np.ndarray, np.ndarray, int]] = []
        batch = max(1, settings.embed_batch)
        with ThreadPoolExecutor(max_workers=max(1, settings.preview_workers)) as pool:
            for i in range(0, len(todo), batch * 4):
                chunk = [r for r in pool.map(prep, todo[i : i + batch * 4]) if r is not None]
                need = [r for r in chunk if r[0] not in cached]
                embs: dict[str, np.ndarray] = {
                    r[0]: cached[r[0]][1] for r in chunk if r[0] in cached
                }
                if need:
                    if embedder is None:
                        from shotsieve.models.image_embed import get_image_embedder

                        embedder = get_image_embedder(
                            threads=settings.ort_threads, batch_size=settings.embed_batch,
                            use_coreml=settings.use_coreml,
                        )
                    vecs = embedder.embed_batch([r[1] for r in need])
                    # same fp16 round-trip + renormalisation as the cull cache (warm == cold)
                    vecs = vecs.astype(np.float16).astype(np.float32)
                    for r, v in zip(need, vecs, strict=True):
                        embs[r[0]] = v
                for p, img, meta, wb in chunk:
                    st = edit_stats(img, meta, wb)
                    fm = format_class(p)
                    out[p] = Feat(st, embs[p], fm)
                    new_rows.append((keys[p], st, embs[p], fm))
                if progress:
                    progress("features", len(out), total)
        store.put_many(new_rows)
    finally:
        if own_store:
            store.close()
    return out


# ---- entries ----------------------------------------------------------------------------------


def _xmp_entries(folder: Path) -> list[lrcat.DevelopEntry]:
    out: list[lrcat.DevelopEntry] = []
    for xp in sorted(folder.rglob("*.xmp")):
        if xp.name.startswith("."):
            continue
        d = crs.read_develop_xmp(xp)
        if d is None or d.made_by_us or not d.settings:
            continue
        pv = lrcat._pv_float(d.settings.get("ProcessVersion"))
        if not (any(k in d.settings for k in lrcat._PV2012_KEYS) or pv >= 6.7):
            continue
        if not lrcat.edited(d.settings):
            continue
        master = next(
            (
                m
                for m in sorted(xp.parent.iterdir())
                if m.stem == xp.stem and m.suffix.lower() in SUPPORTED_EXTS
            ),
            None,
        )
        if master is None:
            continue
        out.append(
            lrcat.DevelopEntry(
                str(master), lrcat._fmt_of(master.suffix), dict(d.settings), pv
            )
        )
    return out


def _is_own_catalog_value(entry: lrcat.DevelopEntry, written: dict) -> bool:
    """Every key ShotSieve wrote for this image still has that value in the catalog."""
    vals = written.get("values") or {}
    if not vals or written.get("written_mode") not in ("sidecar", "copy"):
        return False
    for k, v in vals.items():
        cur = entry.settings.get(k)
        if not isinstance(cur, float):
            return False
        tol = 0.011 if k == "Exposure2012" else 0.6
        if k == "Temperature":
            tol = 2.0
        if abs(cur - float(v)) > tol:
            return False
    return True


def _drop_own_writes(entries: list[lrcat.DevelopEntry]) -> tuple[list[lrcat.DevelopEntry], int]:
    kept, dropped = [], 0
    caches: dict[Path, dict] = {}
    for e in entries:
        root = _shoot_root(Path(e.path))
        written = None
        if root is not None:
            if root not in caches:
                try:
                    with Cache.for_folder(root) as c:
                        caches[root] = c.get_edits()
                except Exception:  # noqa: BLE001
                    caches[root] = {}
            written = caches[root].get(e.path)
        if written and _is_own_catalog_value(e, written):
            dropped += 1
            continue
        kept.append(e)
    return kept, dropped


def build_edit_set(
    catalog: str | Path | None,
    xmp_dirs: Iterable[str | Path] = (),
    settings: Settings | None = None,
    progress: Callable[[str, int, int], None] | None = None,
    *,
    dng_render: bool = False,
    group_of: Callable[[str], str] | None = None,
    store: FeatureStore | None = None,
) -> EditSet:
    """Edited frames of a catalog and/or XMP folders as an EditSet (features + targets)."""
    entries: list[lrcat.DevelopEntry] = []
    sources: list[str] = []
    notes: dict[str, int] = {}
    if catalog:
        entries += lrcat.read_develop(catalog)
        sources.append("lrcat")
    for d in xmp_dirs:
        entries += _xmp_entries(Path(d))
        sources.append("xmp")
    entries, notes["own_writes_excluded"] = _drop_own_writes(entries)
    seen: set[str] = set()
    existing = []
    for e in entries:  # catalog rows win over sidecars of the same file
        if e.path in seen:
            continue
        seen.add(e.path)
        if Path(e.path).is_file():
            existing.append(e)
    notes["missing_files"] = len(entries) - len(existing) - (len(entries) - len(seen))
    keep: list[lrcat.DevelopEntry] = []
    notes["adobe_preview_skipped"] = 0
    for e in existing:
        if e.fmt == "dng" and not dng_render:
            app = dng_xmp.preview_application(e.path)
            if app and "adobe" in app.lower():
                notes["adobe_preview_skipped"] += 1  # the preview has the edit baked in (leak)
                continue
        keep.append(e)
    feats = featurise(
        [e.path for e in keep], settings, progress, dng_render=dng_render, store=store
    )
    rows = [e for e in keep if e.path in feats]
    notes["undecodable"] = len(keep) - len(rows)
    stats = np.vstack([feats[e.path].stats for e in rows]) if rows else np.zeros((0, N_STATS))
    emb = np.vstack([feats[e.path].emb for e in rows]) if rows else np.zeros((0, 384))
    fmt = np.array([feats[e.path].fmt for e in rows], dtype=int)
    wb_cache = {
        e.path: as_shot(e.path) if feats[e.path].fmt <= FMT_DNG else None for e in rows
    }
    ys = [targets_from_settings(e.settings, feats[e.path].fmt, wb_cache[e.path]) for e in rows]
    y = {k: np.array([r[k] for r in ys], dtype=float) for k in ALL_KEYS}
    gf = group_of or (lambda p: str(Path(p).parent))
    gid: dict[str, int] = {}
    groups = np.array([gid.setdefault(gf(e.path), len(gid)) for e in rows], dtype=int)
    return EditSet(
        stats=stats.astype(np.float32), emb=emb.astype(np.float32), fmt=fmt, y=y, groups=groups,
        carry=carry_set(e.settings for e in rows), paths=[e.path for e in rows],
        sources=sources, notes=notes,
    )  # fmt: skip
