"""FastAPI backend for the review UI (bind 127.0.0.1 only; see docs/04_API_UI.md)."""

from __future__ import annotations

import contextlib
import gc
import math
import os
import threading
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.gzip import GZipMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from shotsieve import __version__
from shotsieve.config import Settings
from shotsieve.features.vector import FeatureVector
from shotsieve.io.formats import RAW_EXTS, preview_source_for, split_virtual
from shotsieve.io.preview import PreviewError, full_size, load_full
from shotsieve.io.viewer import (
    THUMB_QUALITY,
    VIEW_QUALITY,
    bucket_for,
    encode_jpeg,
    has_foreign_profile,
    load_view,
)
from shotsieve.io.xmp import (
    LABEL_XMP,
    LIGHTROOM_NOTE,
    WriteOutcome,
    lightroom_warning,
    write_decision,
)
from shotsieve.log import get_logger
from shotsieve.pipeline.cache import Cache, cache_path
from shotsieve.scoring import reasons as reason_text
from shotsieve.server import crop as crop_server
from shotsieve.server import denoise as denoise_server
from shotsieve.server import faces as faces_server
from shotsieve.server import filters as filters_server
from shotsieve.server import lightroom as lightroom_server
from shotsieve.server import onboarding
from shotsieve.server import subject as subject_server
from shotsieve.server import trash as trash_server

log = get_logger("server")

STATIC_DIR = Path(__file__).parent / "static"
DECISIONS = ("keep", "review", "reject")
PRODUCTION_HOSTS = ("127.0.0.1", "localhost")


def allowed_hosts() -> list[str]:
    """Hosts the server answers to (TrustedHost + Origin). Production: loopback names only; the
    test suite adds Starlette's `testserver` through SHOTSIEVE_EXTRA_HOSTS (tests/conftest.py)."""
    extra = [h for h in os.environ.get("SHOTSIEVE_EXTRA_HOSTS", "").split(",") if h.strip()]
    return [*PRODUCTION_HOSTS, *extra]



# ---- request models ---------------------------------------------------------------------


class CullRequest(BaseModel):
    folder: str
    write_xmp: bool = False
    video: bool = False  # T12 opt-in: also cull frame candidates of the videos in the folder
    files: list[str] | None = None  # cull only these files (absolute, directly inside `folder`)


class GrabRequest(BaseModel):
    image_id: int
    folder: str | None = None


class OverrideRequest(BaseModel):
    image_id: int | None = None
    ids: list[int] | None = None  # T23i: the same change for up to 5000 photos in one request
    decision: str | None = None  # keep | review | reject | model (clear the override)
    rating: int | None = None  # 0-5
    folder: str | None = None


class FolderRequest(BaseModel):
    folder: str | None = None


class ClockRequest(BaseModel):
    folder: str | None = None
    serial: str
    offset_s: float | None = None  # seconds added to this body's timestamps
    pair: list[int] | None = None  # [image id, image id] showing the same moment, one per body
    clear: bool = False


class IndexRequest(BaseModel):
    folder: str | None = None
    people: bool = True
    ocr: bool | None = None  # jersey OCR (opt-in); None = the shoot's saved choice / Settings


class ShotlistRequest(BaseModel):
    folder: str | None = None
    list: str | None = None  # a template name (wedding, sports, corporate)
    text: str | None = None  # or the list itself, one item per line: [!]text [person:"Name"] [xN]
    tag_xmp: bool = False  # opt-in: write `shotlist:<item>` keywords on the matched keeps


class RosterRequest(BaseModel):
    folder: str | None = None
    csv: str  # `number,name[,team]` text
    team: str | None = None  # this folder's team (settles numbers shared by two teams)


class PersonRequest(BaseModel):
    folder: str | None = None
    name: str | None = None
    merge_into: int | None = None
    ignore: bool = False  # "not a person": exclude every face of this cluster


class ConsentRequest(BaseModel):
    folder: str | None = None
    on: bool


class EditTrainRequest(BaseModel):
    catalog: str | None = None  # Lightroom catalog (.lrcat)
    xmp: list[str] = []  # folders of XMP sidecars with develop settings
    dng_render: bool = False


class EditApplyRequest(BaseModel):
    folder: str | None = None
    scope: str = "keeps"  # keeps | all
    ids: list[int] | None = None  # explicit images instead of a scope
    dry_run: bool = False
    embed_copies: str | bool | None = None  # JPEG/DNG: write embedded-XMP copies (dir or true)
    catalog: str | None = None


class EditMatchRequest(BaseModel):
    folder: str | None = None
    ref: int  # image id of the reference frame (it must carry develop settings)
    targets: str = "keeps"  # keeps | group | camera:MODEL
    ids: list[int] | None = None  # explicit images instead of `targets`
    dry_run: bool = False
    embed_copies: str | bool | None = None
    catalog: str | None = None


class LrSyncRequest(BaseModel):
    folder: str | None = None
    catalog: str
    dry_run: bool = False


class XmpAutoRequest(BaseModel):
    folder: str | None = None
    on: bool


class XmpRequest(BaseModel):
    folder: str | None = None
    force_ratings: bool = False  # also replace ratings/labels the photographer already set


# ---- jobs -------------------------------------------------------------------------------


@dataclass
class Job:
    id: str
    folder: str
    write_xmp: bool = False
    video: bool = False
    only: list[str] | None = None  # cull job restricted to these files
    kind: str = "cull"  # cull | edit_train | edit_apply | edit_match | lr_sync | crop_burst
    payload: dict[str, Any] = field(default_factory=dict)
    state: str = "queued"  # queued | running | done | error
    stage: str = ""
    done: int = 0
    total: int = 0
    error: str | None = None
    result: dict[str, Any] = field(default_factory=dict)

    def public(self) -> dict[str, Any]:
        return {
            "job_id": self.id, "state": self.state, "stage": self.stage, "done": self.done,
            "total": self.total, "error": self.error, "folder": self.folder,
            "result": self.result, "kind": self.kind,
        }  # fmt: skip


class AppState:
    def __init__(self, settings: Settings, default_folder: Path | None) -> None:
        self.settings = settings
        self.default_folder = default_folder
        self.current_folder: Path | None = default_folder
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        self.active: Job | None = None
        self.calibrating: set[Path] = set()  # folders being re-decided (guarded by `lock`)
        self.indexers: dict[Path, Any] = {}  # folder -> Indexer (guarded by `lock`)
        from shotsieve.server.autoxmp import AutoXmp

        self.autoxmp = AutoXmp(self)  # T23g: automatic XMP sidecars (background)

    def _cull_running(self, folder: Path) -> bool:  # caller holds `lock`
        a = self.active
        return a is not None and a.state in ("queued", "running") and Path(a.folder) == folder

    def busy_with(self, folder: Path) -> bool:
        with self.lock:
            return self._cull_running(folder) or folder in self.calibrating


def validate_folder(raw: str | None, state: AppState) -> Path:
    """Existing directory, absolute, without `..` segments or NUL bytes."""
    if raw is None or raw == "":
        if state.current_folder is None:
            raise HTTPException(400, "no folder given and none selected")
        return state.current_folder
    if "\x00" in raw:
        raise HTTPException(400, "invalid folder path")
    p = Path(raw).expanduser()
    if ".." in p.parts:
        raise HTTPException(400, "path traversal is not allowed")
    if not p.is_absolute():
        raise HTTPException(400, "folder must be an absolute path")
    if not p.is_dir():
        raise HTTPException(400, f"not a directory: {raw}")
    return Path(os.path.abspath(p))


def _load(folder: Path) -> tuple[dict, dict, dict, dict, dict]:
    """(decisions, overrides, ids, timestamps, thresholds-meta) from the shoot database."""
    if not cache_path(folder).exists():
        raise HTTPException(404, "this folder has not been culled yet")
    with Cache.for_folder(folder) as c:
        decisions = c.load_decisions()
        if not decisions:
            raise HTTPException(404, "this folder has not been culled yet")
        overrides = c.overrides()
        ids = c.ids_lookup(list(decisions))  # ids are assigned at cull time: a GET only reads
        ts = c.adjusted_timestamps()  # t_adj: per-body clock offsets applied
        meta = {
            "thresholds": c.get_thresholds(),
            "last_stats": c.get_meta("last_stats"),
            "calibration": c.get_meta("calibration"),
        }
    return decisions, overrides, ids, ts, meta


def _sug_paths(s: Any) -> list[str]:
    ps = ([s.pair] if s.pair else []) + list(s.alternatives)
    return [x for p in ps for x in (p.ref_path, p.path)]


def _final(row: dict, ovr: dict | None) -> str:
    return ovr["decision"] if ovr else row["decision"]


def _texts(row: dict, fv: FeatureVector | None, ctx: dict[str, Any]) -> list[str]:
    """Plain-language sentence per reason code (the code itself if it is unknown)."""
    out = []
    for code in row["reasons"]:
        try:
            out.append(reason_text.describe(code, fv, ctx))
        except KeyError:
            out.append(code)
    return out


def _num(v: Any, nd: int = 3) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else round(f, nd)


def _metrics(fv: FeatureVector | None) -> dict[str, Any]:
    """The few per-frame numbers the compare view shows (None = not measured)."""
    if fv is None:
        return {"sharp": None, "eyes_open": None, "face": None, "n_faces": 0}
    ec = _num(fv.eyes_closed_main)
    return {
        "sharp": _num(fv.sharp_global_norm), "eyes_open": None if ec is None else round(1 - ec, 3),
        "face": _num(fv.face_sharp_norm_main), "n_faces": int(fv.n_faces or 0),
    }  # fmt: skip


def build_shoot(folder: Path, settings: Settings) -> dict[str, Any]:
    decisions, overrides, ids, ts, meta = _load(folder)
    by_group: dict[int, list[str]] = {}
    for path, row in decisions.items():
        by_group.setdefault(row["group_id"], []).append(path)
    group_rows = []
    for gid, paths in by_group.items():
        paths.sort(key=lambda p: (not decisions[p]["is_best"], Path(p).name))
        best = next((p for p in paths if decisions[p]["is_best"]), paths[0])
        times = [ts.get(p) for p in paths if ts.get(p) is not None]
        group_rows.append((min(times) if times else float("inf"), min(paths), gid, paths, best))
    group_rows.sort(key=lambda g: (g[0], g[1]))

    edit_flags: dict[str, bool] = {}
    with Cache.for_folder(folder) as c:  # T15: images with predicted edits (badge / filter)
        for ep, er in c.get_edits().items():
            edit_flags[ep] = bool((er["confidence"] or {}).get("needs_edit"))
        crops = c.crops_current_all()  # T22: current crop per image id (one indexed query)
    images, groups = [], []
    counts = {d: 0 for d in DECISIONS}
    n_audit = n_audit_pending = 0
    for _, _, gid, paths, best in group_rows:
        best_name = Path(best).name
        for p in paths:
            row, ovr = decisions[p], overrides.get(p)
            final = _final(row, ovr)
            counts[final] = counts.get(final, 0) + 1
            fv = FeatureVector.from_dict(row["features"]) if row["features"] else None
            ctx = {"best": best_name, "best_uncertain": row["decision"] == "review"}
            texts = _texts(row, fv, ctx)
            if row["audit"]:
                n_audit += 1
                n_audit_pending += ovr is None or ovr["decision"] == "review"
            rel = os.path.relpath(p, folder)
            images.append(
                {
                    "id": ids[p], "name": Path(p).name, "rel": rel, "decision": final,
                    "model_decision": row["decision"], "overridden": ovr is not None,
                    "rating": ovr["rating"] if ovr else None, "reasons": row["reasons"],
                    "reason_texts": texts, "score": row["score"], "prior": row["prior"],
                    "group_id": gid, "burst_id": row["burst_id"], "audit": row["audit"],
                    "audit_prob": row["audit_prob"], "best": bool(row["is_best"]),
                    "n_faces": fv.n_faces if fv else 0,
                    "video_t": vt[1] if (vt := split_virtual(p)) else None,
                    "lr": bool(ovr and ovr.get("note") == "lrcat"),
                    "edit": None if p not in edit_flags else {"needs_edit": edit_flags[p]},
                    "crop": crops.get(ids[p]),
                    "duplicate_of": (
                        {"image_id": ids[best], "name": best_name}
                        if p != best and {"duplicate", "duplicate_tie"} & set(row["reasons"])
                        else None
                    ),
                    "metrics": _metrics(fv),
                }
            )  # fmt: skip
        groups.append({"id": gid, "members": [ids[p] for p in paths], "best": ids[best]})
    thresholds = {"t_keep": settings.t_keep, "t_reject": settings.t_reject}
    thresholds.update({k: float(v) for k, v in (meta["thresholds"] or {}).items()})
    return {
        "folder": str(folder),
        "images": images,
        "groups": groups,
        "stats": {
            "n_images": len(images), "counts": counts, "n_audit": n_audit,
            "n_audit_pending": n_audit_pending, "thresholds": thresholds,
            "run": meta["last_stats"], "calibration": meta["calibration"],
        },
    }  # fmt: skip


def _path_of(folder: Path, image_id: int) -> str:
    if not cache_path(folder).exists():
        raise HTTPException(404, "unknown image")
    with Cache.for_folder(folder) as c:
        path = c.path_for_id(image_id)
    if path is None:
        raise HTTPException(404, "unknown image")
    root = Path(os.path.abspath(folder))
    if not Path(os.path.abspath(path)).is_relative_to(root):
        raise HTTPException(404, "unknown image")
    return path


def _write_atomic(dest: Path, data: bytes) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{uuid.uuid4().hex}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, dest)


def _render_jpeg(source: Path, dest: Path, max_edge: int, quality: int, icc: bool) -> None:
    """Viewer / thumbnail JPEG (T23f): colour-managed to sRGB, one Lanczos downscale, 4:4:4 and an
    embedded sRGB profile for q >= 90. The analysis path (`load_preview`) is not used here."""
    _write_atomic(dest, encode_jpeg(load_view(source, max_edge), quality, icc))


def _video_frame_file(
    folder: Path, video: Path, t: float, image_id: int, edge: int, kind: str, quality: int
) -> Path:
    from shotsieve.io.video import VideoError, VideoReader

    try:
        mtime = video.stat().st_mtime_ns
    except OSError as exc:
        raise HTTPException(404, "video file is missing") from exc
    dest = folder / ".shotsieve" / "thumbs" / f"{kind}_{image_id}_{edge}_{mtime}.jpg"
    if not dest.exists():
        try:
            with VideoReader(video, edge) as vr:
                rgb = vr.frame_at(t, edge)
            if rgb is None:
                raise VideoError("frame not readable")
            ok, buf = cv2.imencode(
                ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality]
            )
            if not ok:
                raise VideoError("jpeg encode failed")
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(f".{uuid.uuid4().hex}.tmp")
            tmp.write_bytes(buf.tobytes())
            os.replace(tmp, dest)
        except (VideoError, OSError) as exc:
            raise HTTPException(422, f"cannot decode video frame: {exc}") from exc
    return dest


def _image_file(folder: Path, image_id: int, edge: int, kind: str, quality: int) -> Path:
    master = _path_of(folder, image_id)
    sv = split_virtual(master)
    if sv is not None:
        return _video_frame_file(folder, sv[0], sv[1], image_id, edge, kind, quality)
    source = preview_source_for(master)
    try:
        mtime = source.stat().st_mtime_ns
    except OSError as exc:
        raise HTTPException(404, "image file is missing") from exc
    if kind == "t":  # the cull writes 400 px grid thumbs (pipeline/thumbs.py): reuse them, unless
        legacy = folder / ".shotsieve" / "thumbs" / f"t_{image_id}_{edge}_{mtime}.jpg"
        if legacy.exists() and not (  # the source has a wide-gamut profile they would ignore
            source.suffix.lower() not in RAW_EXTS and has_foreign_profile(source)
        ):
            return legacy
    # "_v2": images rendered by the colour-managed viewer pipeline (T23f); older files are ignored
    dest = folder / ".shotsieve" / "thumbs" / f"{kind}_{image_id}_{edge}_{mtime}_v2.jpg"
    if not dest.exists():
        try:
            _render_jpeg(source, dest, edge, quality, icc=quality >= 90)
        except (PreviewError, OSError, ValueError) as exc:
            raise HTTPException(422, f"cannot decode image: {exc}") from exc
    return dest


# ---- full resolution for zoom (T23c) ---------------------------------------------------------

FULL_CACHE_CAP = 2 * 1024**3  # bytes of full-size JPEGs kept per shoot (LRU-pruned)
FULL_QUALITY = 92
_FULL_LOCK = threading.Lock()  # one full decode at a time: a 45 MP RAW is ~300 MB transient
_SIZE_CACHE: dict[tuple[str, int], tuple[int, int]] = {}


def _release_heap() -> None:
    """Hand freed decode buffers back to macOS (the allocator otherwise keeps them resident)."""
    with contextlib.suppress(Exception):
        import ctypes

        ctypes.CDLL("/usr/lib/libSystem.B.dylib").malloc_zone_pressure_relief(None, 0)


def _prune_full(full_dir: Path, keep: Path, cap: int) -> None:
    """Delete the least recently used full-size files until the folder is within `cap`."""
    try:
        files = [(f, f.stat()) for f in full_dir.glob("f_*.jpg") if f != keep]
    except OSError:
        return
    total = sum(st.st_size for _, st in files)
    try:
        total += keep.stat().st_size
    except OSError:
        pass
    for f, st in sorted(files, key=lambda x: x[1].st_mtime_ns):  # hits touch mtime: oldest = LRU
        if total <= cap:
            break
        try:
            f.unlink()
            total -= st.st_size
        except OSError:
            pass


def _full_source(folder: Path, image_id: int) -> tuple[Path, int]:
    master = _path_of(folder, image_id)
    if split_virtual(master) is not None:
        raise HTTPException(422, "videos have no full-resolution view")
    source = preview_source_for(master)
    try:
        return source, source.stat().st_mtime_ns
    except OSError as exc:
        raise HTTPException(404, "image file is missing") from exc


def _image_size(f: Path, image_id: int) -> tuple[int, int]:
    """Displayed pixel size of the full-resolution image (cached per file + mtime)."""
    source, mtime = _full_source(f, image_id)
    key = (str(source), mtime)
    if key not in _SIZE_CACHE:
        try:
            _SIZE_CACHE[key] = full_size(source)
        except (PreviewError, OSError, ValueError) as exc:
            raise HTTPException(422, f"cannot read image: {exc}") from exc
        if len(_SIZE_CACHE) > 20000:
            _SIZE_CACHE.clear()
    return _SIZE_CACHE[key]


def _full_file(folder: Path, image_id: int) -> Path:
    source, mtime = _full_source(folder, image_id)
    full_dir = folder / ".shotsieve" / "full"
    dest = full_dir / f"f_{image_id}_{mtime}.jpg"
    if dest.exists():
        with contextlib.suppress(OSError):
            os.utime(dest)  # LRU touch
        return dest
    with _FULL_LOCK:
        if dest.exists():
            return dest
        try:
            img = load_full(source)
            try:
                data = encode_jpeg(img, FULL_QUALITY, icc=True)  # Pillow shares img's buffer
            finally:
                del img
            _write_atomic(dest, data)
            del data
        except (PreviewError, OSError, ValueError) as exc:
            raise HTTPException(422, f"cannot decode image: {exc}") from exc
        finally:
            gc.collect()
            _release_heap()
        _prune_full(full_dir, dest, FULL_CACHE_CAP)
    return dest


def write_xmp_sidecars(
    f: Path, settings: Settings, force: bool = False, only: set[str] | None = None
) -> dict[str, Any]:
    """Write the final decision of every image (or of `only`) into its XMP sidecar."""
    decisions, overrides, _, _, _ = _load(f)
    with Cache.for_folder(f) as c:
        user_labels = c.labels()  # T23e: colour labels the user set ("" = cleared)
    written = failed = kept_rating = kept_label = kept_lr = 0
    wrote: list[tuple[str, int | None, str | None, str]] = []
    best_of = {r["group_id"]: p for p, r in decisions.items() if r["is_best"]}
    for path, row in decisions.items():
        if only is not None and path not in only:
            continue
        if split_virtual(path):  # video frame candidates have no file to annotate
            continue
        ovr = overrides.get(path)
        final = _final(row, ovr)
        fv = FeatureVector.from_dict(row["features"]) if row["features"] else None
        best = Path(best_of[row["group_id"]]).name if row["group_id"] in best_of else None
        texts = _texts(row, fv, {"best": best, "best_uncertain": row["decision"] == "review"})
        outcome = WriteOutcome()
        try:
            write_decision(
                path, settings, final, texts, row["score"],
                override=ovr, force=force, outcome=outcome,
                user_label=LABEL_XMP.get(user_labels[path], "") if path in user_labels else None,
            )  # fmt: skip
            written += 1
            kept_rating += outcome.rating_kept
            kept_label += outcome.label_kept
            kept_lr += outcome.lr_kept
            if outcome.written_rating is not None or outcome.written_label is not None:
                wrote.append((path, outcome.written_rating, outcome.written_label, final))
        except Exception as exc:  # noqa: BLE001 - one odd sidecar must not abort the rest
            log.warning("xmp write failed for %s: %s", path, exc)
            failed += 1
    if wrote:  # T20: remember what Lightroom will import from us
        with Cache.for_folder(f) as c:
            c.put_written(wrote)
    warnings = []
    if kept_rating or kept_label:
        warnings.append(
            f"kept your existing rating on {kept_rating} and label on {kept_label} image(s); "
            "override an image in ShotSieve (or pass force_ratings) to replace them"
        )
    lr = lightroom_warning(
        p for p in decisions if not split_virtual(p) and (only is None or p in only)
    )
    if lr:
        warnings.append(lr)
    return {"written": written, "failed": failed, "kept_user_rating": kept_rating,
            "kept_user_label": kept_label, "kept_lr": kept_lr,
            "warnings": warnings}  # fmt: skip



# ---- app --------------------------------------------------------------------------------


MAX_BATCH = 5000  # photos per batch request (override / label / trash)


def _apply_override(
    c: Cache, path: str, row: dict, existing: dict | None, req: OverrideRequest, commit: bool = True
) -> dict[str, Any]:
    """One image's override (decision and / or rating) -> the response fields."""
    if req.decision == "model":
        c.clear_override(path, commit=commit)
        return {"decision": row["decision"], "overridden": False, "rating": None,
                "propensity": None}  # fmt: skip
    # chance the user was shown this image: the sampling probability for audit items
    propensity = float(row["audit_prob"] or 1.0) if row["audit"] else 1.0
    decision = req.decision
    if decision is None:  # rating only: a rated image is one the user wants (rating >= 1)
        decision = (
            existing["decision"] if existing else ("keep" if req.rating else row["decision"])
        )
    rating = req.rating if req.rating is not None else (existing or {}).get("rating")
    c.set_override(path, decision, propensity, "ui", rating, commit=commit)
    return {"decision": decision, "overridden": True, "rating": rating, "propensity": propensity}


def create_app(
    settings: Settings | None = None, default_folder: str | Path | None = None
) -> FastAPI:
    folder0 = None
    if default_folder is not None:
        folder0 = Path(os.path.abspath(Path(default_folder).expanduser()))
        if not folder0.is_dir():
            raise ValueError(f"not a directory: {default_folder}")
    state = AppState(settings or Settings(), folder0)
    app = FastAPI(title="ShotSieve", docs_url=None, redoc_url=None)
    hosts = allowed_hosts()
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=hosts)
    app.add_middleware(GZipMiddleware, minimum_size=4096)  # a 10k-frame /api/shoot: ~4 MB -> <1 MB

    @app.middleware("http")
    async def _local_origin_only(request: Request, call_next: Any) -> Any:
        """Defence in depth: a state-changing request from a page on another origin is refused."""
        origin = request.headers.get("origin")
        if origin and request.method not in ("GET", "HEAD", "OPTIONS"):
            host = urlparse(origin).hostname
            if host not in hosts:
                return JSONResponse({"detail": "cross-origin request refused"}, status_code=403)
        return await call_next(request)
    app.state.shotsieve = state

    # ---- health ---------------------------------------------------------------------------
    @app.get("/api/health")
    def health() -> dict[str, Any]:
        from shotsieve.models.registry import group_names, verify_model
        from shotsieve.models.runtime import coreml_available
        from shotsieve.personal.model import default_model_path

        models = {n: verify_model(n) for n in group_names("core")}
        search_models = all(verify_model(n) for n in group_names("search"))
        return {
            "ok": True,
            "app": "shotsieve",
            "version": __version__,
            "pid": os.getpid(),
            "models": models,
            "models_ready": all(models.values()),
            "search_ready": search_models,
            "coreml": bool(coreml_available()),
            "personal_model": default_model_path(state.settings).exists(),
            "default_folder": str(state.default_folder) if state.default_folder else None,
        }

    @app.post("/api/quit")
    def quit_server() -> dict[str, Any]:
        """Graceful shutdown after the response (Origin policy: the middleware above)."""
        from shotsieve.server.lifecycle import remove_server_json_if_ours

        def _stop() -> None:
            remove_server_json_if_ours()
            srv = getattr(app.state, "uvicorn_server", None)
            if srv is not None:
                srv.should_exit = True

        threading.Timer(0.2, _stop).start()
        return {"ok": True, "quitting": True}

    @app.post("/api/focus")
    def focus_window() -> dict[str, Any]:
        """A second `shotsieve app` launch asks the running window to come forward."""
        cb = getattr(app.state, "focus", None)
        if cb is None:
            return {"ok": True, "focused": False}
        threading.Thread(target=cb, daemon=True).start()
        return {"ok": True, "focused": True}

    # ---- search index ---------------------------------------------------------------------
    def _maybe_start_indexer(folder: Path, people: bool = True, ocr: bool | None = None) -> Any:
        from shotsieve.models.ocr import ocr_models_ready
        from shotsieve.models.text_image import search_models_ready
        from shotsieve.pipeline.indexer import Indexer

        want_ocr = state.settings.jersey_ocr
        if cache_path(folder).exists():
            with Cache.for_folder(folder) as c:
                if ocr is not None:
                    c.set_meta("jersey_ocr", bool(ocr))
                want_ocr = bool(c.get_meta("jersey_ocr", want_ocr))
        elif ocr is not None:
            want_ocr = ocr
        images, run_ocr = search_models_ready(), want_ocr and ocr_models_ready()
        if not images and not run_ocr:
            return None
        with state.lock:
            ix = state.indexers.get(folder)
            if ix is None:
                ix = state.indexers[folder] = Indexer(folder, state.settings, people=people)
            ix.people, ix.images, ix.ocr = people, images, run_ocr
        ix.start()
        return ix

    def _consent(folder: Path) -> bool:
        if not cache_path(folder).exists():
            return state.settings.people_consent
        with Cache.for_folder(folder) as c:
            return c.people_consent(state.settings.people_consent)

    reclustering: set[Path] = set()
    scanners: dict[Path, Any] = {}

    def _maybe_recluster(folder: Path) -> None:
        """Background people maintenance when a shoot is opened (never in the cull path):
        (1) T23h: images without a current recognition scan (face-detection version bump, new
        files, a changed face count) are re-scanned for ALL faces (small / noisy / background);
        the scan re-clusters when it ends. (2) A shoot clustered by an older rule set is
        re-clustered once from the stored face vectors. Names and exclusions survive both;
        decisions are untouched (the scan writes only recognition faces)."""
        from shotsieve.pipeline import facescan, people

        if not cache_path(folder).exists():
            return
        with state.lock:
            if folder in reclustering:
                return
        try:
            with Cache.for_folder(folder) as c:
                if not c.people_consent(state.settings.people_consent):
                    return
                rescan = facescan.needs_rescan(c)
                if not rescan and not people.needs_recluster(c):
                    return
        except Exception:  # noqa: BLE001 - a read problem must never break opening a shoot
            return
        with state.lock:
            if folder in reclustering:
                return
            reclustering.add(folder)

        def done() -> None:
            with state.lock:
                reclustering.discard(folder)

        if rescan:
            sc = facescan.FaceScanner(folder, state.settings, on_done=done)
            with state.lock:
                scanners[folder] = sc
            sc.start()
            return

        def work() -> None:
            try:
                with Cache.for_folder(folder) as c:
                    people.cluster(c)
            except Exception:  # noqa: BLE001 - best effort; retried on the next open
                pass
            finally:
                done()

        threading.Thread(target=work, name="people-recluster", daemon=True).start()

    def _people_after_cull(folder: Path) -> None:
        """Photos moved to the Trash (rows gone) or put back: re-cluster once from the stored face
        vectors (names and exclusions carried) so the people lists match the shoot. No-op when
        nothing changed, when people features are off, or when the shoot has no people yet."""
        from shotsieve.pipeline import people

        try:
            with Cache.for_folder(folder) as c:
                if not c.people_consent(state.settings.people_consent):
                    return
                alive = set(c.load_decisions())
                with c._lock:
                    has = c._con.execute("SELECT 1 FROM people LIMIT 1").fetchone() is not None
                    rows = c._con.execute("SELECT DISTINCT path FROM person_faces").fetchall()
                in_p = {c._abs(r[0]) for r in rows}
                if not has:
                    return
                have = {p for p, v in c.face_embeddings().items() if len(v)}
                stale = bool(in_p - alive) or bool((have & alive) - in_p)
            if not stale:
                return
            with state.lock:
                if folder in reclustering:
                    return
                reclustering.add(folder)

            def work() -> None:
                try:
                    with Cache.for_folder(folder) as c:
                        people.cluster(c)
                except Exception:  # noqa: BLE001 - best effort; the next open retries
                    log.warning("people re-cluster after a cull failed", exc_info=True)
                finally:
                    with state.lock:
                        reclustering.discard(folder)

            threading.Thread(target=work, name="people-after-cull", daemon=True).start()
        except Exception:  # noqa: BLE001 - never fail a cull job over people housekeeping
            log.warning("people check after a cull failed", exc_info=True)

    @app.get("/api/people/scan")
    def people_scan_status(folder: str | None = None) -> dict[str, Any]:
        """Progress of the background all-faces recognition scan (T23h)."""
        from shotsieve.pipeline import facescan

        f = validate_folder(folder, state)
        sc = scanners.get(f)
        out = sc.status() if sc else {"running": False, "done": 0, "total": 0, "faces": 0,
                                      "error": None}  # fmt: skip
        out["version"] = facescan.FACE_SCAN_VERSION
        if not out["running"] and cache_path(f).exists():
            with Cache.for_folder(f) as c:
                out["pending"] = facescan.needs_rescan(c)
        return out

    def _need_consent(folder: Path) -> None:
        if not _consent(folder):
            raise HTTPException(403, "people features are off; enable them with consent first")

    @app.get("/api/index/status")
    def index_status(folder: str | None = None) -> dict[str, Any]:
        from shotsieve.models.ocr import ocr_models_ready
        from shotsieve.models.text_image import search_models_ready

        f = validate_folder(folder, state)
        ix = state.indexers.get(f)
        out = ix.status() if ix else {"task": "idle", "done": 0, "total": 0, "paused": False,
                                      "img_s": 0.0, "error": None, "running": False}  # fmt: skip
        out["models_ready"] = search_models_ready()
        out["ocr_models_ready"] = ocr_models_ready()
        out["people_consent"] = _consent(f)
        if not out["running"] and cache_path(f).exists():
            with Cache.for_folder(f) as c:
                out["indexed"] = c.siglip_count()
        return out

    @app.post("/api/index/{action}")
    def index_control(action: str, req: IndexRequest | None = None) -> dict[str, Any]:
        req = req or IndexRequest()
        f = validate_folder(req.folder, state)
        if action == "start":
            ix = _maybe_start_indexer(f, req.people, req.ocr)
            if ix is None:
                raise HTTPException(
                    503, "search models missing; run `shotsieve models download --extra search`"
                    " (jersey OCR: `--extra ocr`)"
                )
            return ix.status()
        if action == "pause":
            ix = state.indexers.get(f)
            if ix is None:
                raise HTTPException(404, "no indexer for this folder")
            ix.pause()
            return ix.status()
        if action == "resume":
            ix = state.indexers.get(f)
            if ix is None:
                raise HTTPException(404, "no indexer for this folder")
            ix.resume()
            return ix.status()
        raise HTTPException(404, "unknown action (start | pause | resume)")

    @app.get("/api/search")
    def get_search(q: str = "", k: int = Query(100, ge=1, le=500), folder: str | None = None):
        from shotsieve.models.text_image import search_models_ready
        from shotsieve.pipeline import search as search_mod
        from shotsieve.pipeline.people import ConsentError

        f = validate_folder(folder, state)
        if not cache_path(f).exists():
            raise HTTPException(404, "this folder has not been culled yet")
        pq = search_mod.parse_query(q)
        if pq.text and not search_models_ready():
            raise HTTPException(503, "search models missing; run `shotsieve models download "
                                "--extra search`")  # fmt: skip
        warnings: list[str] = []
        try:
            hits = search_mod.search(f, q, k, state.settings, warnings=warnings)
        except ConsentError as exc:
            raise HTTPException(403, str(exc)) from exc
        ix = state.indexers.get(f)
        return {"q": q, "hits": [{"id": h.id, "score": h.score, "why": h.why} for h in hits],
                "indexing": bool(ix and ix.running),
                "warnings": warnings}  # fmt: skip

    # ---- shot list coverage (T17) ---------------------------------------------------------
    def _shotlist_payload(f: Path, cov: Any | None = None) -> dict[str, Any]:
        from shotsieve.pipeline import shotlist

        if cov is not None:
            out = cov.public()
        else:
            with Cache.for_folder(f) as c:
                stored = c.get_shotlist()
            counts = {"covered": 0, "uncertain": 0, "weak": 0, "missing": 0, "pending": 0}
            for r in stored:
                counts[r["status"]] = counts.get(r["status"], 0) + 1
            missing = [r["text"] for r in stored if r["must"] and r["status"] == "missing"]
            unc = [r["text"] for r in stored if r["must"] and r["status"] == "uncertain"]
            out = {"items": [dict(r) for r in stored], "counts": counts, "must_missing": missing,
                   "must_uncertain": unc}
        out["templates"] = shotlist.template_names()
        out["has_list"] = bool(out["items"])
        return out

    def _run_shotlist(f: Path, items: Any, tag: bool = False) -> dict[str, Any]:
        from shotsieve.models.text_image import search_models_ready
        from shotsieve.pipeline import shotlist

        if not search_models_ready():
            raise HTTPException(503, "search models missing; run `shotsieve models download "
                                "--extra search`")  # fmt: skip
        try:
            cov = shotlist.run(f, items, state.settings, tag=tag)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        payload = _shotlist_payload(f, cov)
        payload["tagged"] = tag
        return payload

    @app.get("/api/shotlist")
    def get_shotlist(folder: str | None = None) -> dict[str, Any]:
        f = validate_folder(folder, state)
        if not cache_path(f).exists():  # nothing stored yet: an empty list, not an error
            from shotsieve.pipeline import shotlist

            return {"items": [], "counts": {}, "must_missing": [], "must_uncertain": [],
                    "has_list": False, "templates": shotlist.template_names()}  # fmt: skip
        return _shotlist_payload(f)

    @app.post("/api/shotlist")
    def post_shotlist(req: ShotlistRequest) -> dict[str, Any]:
        from shotsieve.pipeline import shotlist

        f = validate_folder(req.folder, state)
        _need_cull(f)
        try:
            items = shotlist.load_items(req.list, None, req.text)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        return _run_shotlist(f, items, req.tag_xmp)

    def _rescore(f: Path, tag: bool) -> dict[str, Any]:
        _need_cull(f)
        with Cache.for_folder(f) as c:
            have = bool(c.get_shotlist())
        if not have:
            return _shotlist_payload(f)
        return _run_shotlist(f, None, tag)

    @app.get("/api/shotlist/coverage")
    def get_shotlist_coverage(folder: str | None = None) -> dict[str, Any]:
        """Re-score the stored list against the current index (after a cull / index finishes).
        A GET never writes: tagging XMP keywords is the POST below."""
        return _rescore(validate_folder(folder, state), False)

    @app.post("/api/shotlist/coverage")
    def post_shotlist_coverage(req: ShotlistRequest) -> dict[str, Any]:
        """Re-score the stored list, optionally tagging the matched keeps (`tag_xmp`)."""
        return _rescore(validate_folder(req.folder, state), req.tag_xmp)

    @app.delete("/api/shotlist")
    def delete_shotlist(folder: str | None = None) -> dict[str, Any]:
        f = validate_folder(folder, state)
        _need_cull(f)
        with Cache.for_folder(f) as c:
            c.clear_shotlist()
        return {"ok": True}

    # ---- jersey roster (T11) --------------------------------------------------------------
    def _need_cull(f: Path) -> None:
        if not cache_path(f).exists():
            raise HTTPException(404, "this folder has not been culled yet")

    @app.get("/api/roster")
    def get_roster(folder: str | None = None) -> dict[str, Any]:
        from shotsieve.pipeline import jerseys

        f = validate_folder(folder, state)
        _need_cull(f)
        with Cache.for_folder(f) as c:
            return jerseys.roster_summary(c)

    @app.post("/api/roster")
    def post_roster(req: RosterRequest) -> dict[str, Any]:
        from shotsieve.io.roster import RosterError, parse_roster
        from shotsieve.pipeline import jerseys

        f = validate_folder(req.folder, state)
        _need_cull(f)
        try:
            entries = parse_roster(req.csv)
        except RosterError as exc:
            raise HTTPException(400, str(exc)) from exc
        with Cache.for_folder(f) as c:
            return jerseys.set_roster(c, entries, req.team)

    @app.delete("/api/roster")
    def delete_roster(folder: str | None = None) -> dict[str, Any]:
        from shotsieve.pipeline import jerseys

        f = validate_folder(folder, state)
        _need_cull(f)
        with Cache.for_folder(f) as c:
            c.set_roster([])
            return jerseys.roster_summary(c)

    @app.get("/api/jerseys")
    def get_jerseys(folder: str | None = None) -> dict[str, Any]:
        from shotsieve.pipeline import jerseys

        f = validate_folder(folder, state)
        _need_cull(f)
        with Cache.for_folder(f) as c:
            return {"jerseys": {str(k): v for k, v in jerseys.jerseys_by_image(c).items()}}

    @app.post("/api/jerseys/xmp")
    def post_jerseys_xmp(req: FolderRequest) -> dict[str, Any]:
        from shotsieve.pipeline import jerseys

        f = validate_folder(req.folder, state)
        _need_cull(f)
        with Cache.for_folder(f) as c:
            res = jerseys.write_xmp(c)
        return {**res, "warnings": [LIGHTROOM_NOTE] if res["images"] else []}

    # ---- people (consent-gated) -----------------------------------------------------------
    filters_server.register(app, state)  # T23e: labels, facets (camera / lens), people map
    faces_server.register(app, state)  # T23d: faces per image, face thumbnails, people helpers

    @app.get("/api/people")
    def get_people(folder: str | None = None) -> dict[str, Any]:
        from shotsieve.pipeline import people

        f = validate_folder(folder, state)
        _need_consent(f)
        _maybe_recluster(f)
        with Cache.for_folder(f) as c:
            return people.list_people(c)

    def _stop_people(folder: Path) -> None:
        with state.lock:
            ix = state.indexers.get(folder)
        if ix is not None:
            ix.cancel_people()

    @app.post("/api/people/consent")
    def post_consent(req: ConsentRequest) -> dict[str, Any]:
        f = validate_folder(req.folder, state)
        with Cache.for_folder(f) as c:
            if req.on:
                c.set_meta("people_consent", True)
            else:
                _stop_people(f)
                c.revoke_people()
        if req.on and cache_path(f).exists() and state.settings.search_index_auto:
            _maybe_start_indexer(f, True)
        return {"ok": True, "people_consent": bool(req.on)}

    @app.post("/api/people/{person_id}")
    def post_person(person_id: int, req: PersonRequest) -> dict[str, Any]:
        from shotsieve.pipeline import people

        f = validate_folder(req.folder, state)
        _need_consent(f)
        if req.name is None and req.merge_into is None and not req.ignore:
            raise HTTPException(422, "give name, merge_into and/or ignore")
        with Cache.for_folder(f) as c:
            if req.ignore:
                gone = people.ignore_faces(c, person_id)
                return {**people.list_people(c), "ignored_faces": [list(g) for g in gone]}
            merged = None
            if req.name is not None:
                merged = people.rename(c, person_id, req.name)  # same name: merged
            if merged is None and req.merge_into is not None and req.merge_into != person_id:
                try:
                    people.merge(c, person_id, req.merge_into)
                except KeyError:
                    raise HTTPException(404, "unknown person to merge into") from None
            return people.list_people(c)

    @app.delete("/api/people")
    def delete_people(folder: str | None = None) -> dict[str, Any]:
        f = validate_folder(folder, state)
        _stop_people(f)
        with Cache.for_folder(f) as c:
            c.revoke_people()
        return {"ok": True, "people_consent": False}

    # ---- jobs -----------------------------------------------------------------------------
    def run_job(job: Job) -> None:
        from shotsieve.pipeline.runner import run_cull

        def progress(stage: str, done: int, total: int) -> None:
            job.stage, job.done, job.total = stage, done, total

        job.state = "running"
        if job.kind == "lr_sync":
            _run_lr_job(job, progress)
            return
        if job.kind == "crop_burst":
            crop_server.run_burst_job(job, progress, state)
            return
        if job.kind != "cull":
            _run_edit_job(job, progress)
            return
        try:
            cfg = replace(state.settings, video=True) if job.video else state.settings
            res = run_cull(
                Path(job.folder), cfg, progress, write_xmp=job.write_xmp,
                only=frozenset(job.only) if job.only is not None else None,
            )
            with Cache.for_folder(job.folder) as c:
                c.set_meta("last_stats", res.stats)
            st = res.stats
            job.result = {"counts": st["counts"], "n_images": st["n_images"]}
            for k in ("redecide", "redecide_reason", "n_new", "n_removed", "n_affected",
                      "n_extracted", "n_cached"):  # fmt: skip
                if k in st:
                    job.result[k] = st[k]
            job.state = "done"
            state.autoxmp.request(Path(job.folder), None)  # after the decisions are committed
            _people_after_cull(Path(job.folder))  # T23i: photos that left / came back (Trash, Undo)
            if state.settings.search_index_auto:
                _maybe_start_indexer(Path(job.folder))
        except Exception as exc:  # noqa: BLE001 - report to the client
            log.exception("cull job failed")
            job.error, job.state = f"{type(exc).__name__}: {exc}", "error"

    def _run_lr_job(job: Job, progress: Any) -> None:
        from shotsieve.io.lrcat import CatalogBusyError, CatalogError
        from shotsieve.pipeline import lrsync

        try:
            p = job.payload
            res = lrsync.sync(
                job.folder, p["catalog"], state.settings, dry_run=bool(p.get("dry_run")),
                progress=progress,
            )  # fmt: skip
            job.result = res.to_dict()
            job.state = "done"
        except (CatalogError, CatalogBusyError, ValueError) as exc:
            job.error, job.state = str(exc), "error"
        except Exception as exc:  # noqa: BLE001 - report to the client
            log.exception("lightroom sync failed")
            job.error, job.state = f"{type(exc).__name__}: {exc}", "error"

    def _run_edit_job(job: Job, progress: Any) -> None:
        from shotsieve.personal.edit_profile import InsufficientEditsError
        from shotsieve.pipeline import edit

        try:
            if job.kind == "edit_train":
                _, info = edit.train_profile(
                    job.payload.get("catalog"), job.payload.get("xmp") or [], state.settings,
                    progress, dng_render=bool(job.payload.get("dng_render")),
                )
                job.result = info
            elif job.kind == "edit_match":
                from shotsieve.pipeline import match

                p = job.payload
                res = match.match(
                    job.folder, p["ref"], p.get("targets") or "keeps",
                    state.settings, ids=p.get("ids"), dry_run=bool(p.get("dry_run")),
                    embed_copies=p.get("embed_copies"), catalog=p.get("catalog"),
                    progress=progress,
                )
                job.result = match.summarize(res, bool(p.get("dry_run")))
            else:
                p = job.payload
                job.result = edit.apply_folder(
                    job.folder, state.settings, scope=p.get("scope", "keeps"), ids=p.get("ids"),
                    dry_run=bool(p.get("dry_run")), embed_copies=p.get("embed_copies"),
                    catalog=p.get("catalog"), progress=progress,
                )
            job.state = "done"
        except (InsufficientEditsError, edit.NoProfileError, ValueError) as exc:
            job.error, job.state = str(exc), "error"
        except Exception as exc:  # noqa: BLE001 - report to the client
            log.exception("edit job failed")
            job.error, job.state = f"{type(exc).__name__}: {exc}", "error"

    def _start_job(job: Job) -> str:
        with state.lock:
            if Path(job.folder) in state.calibrating:
                raise HTTPException(409, "this folder is being re-decided; try again shortly")
            if state.active is not None and state.active.state in ("queued", "running"):
                raise HTTPException(409, "a job is already running")
            state.jobs[job.id] = job
            state.active = job
            threading.Thread(
                target=run_job, args=(job,), name=f"{job.kind}-job", daemon=True
            ).start()
        return job.id

    def _embed_arg(embed: str | bool | None, folder: Path) -> str | bool | None:
        from shotsieve.pipeline import edit

        if isinstance(embed, str) and embed:
            ep = Path(embed).expanduser()
            if "\x00" in embed or ".." in ep.parts or not ep.is_absolute():
                raise HTTPException(400, "embed_copies must be an absolute path without '..'")
            try:
                edit.check_copy_root(ep, folder)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from None
        return embed

    def _file_arg(raw: str | None, what: str) -> str | None:
        if not raw:
            return None
        if "\x00" in raw or ".." in Path(raw).parts or not Path(raw).expanduser().is_absolute():
            raise HTTPException(400, f"{what} must be an absolute path without '..'")
        p = Path(raw).expanduser()
        if not p.is_file():
            raise HTTPException(400, f"not a file: {raw}")
        return str(p)

    # ---- Lightroom sync-back (T20): the catalog is only ever read ---------------------------
    def _catalog_arg(raw: str) -> str:
        from shotsieve.io.lrcat import CatalogError, check_catalog_path

        try:
            return str(check_catalog_path(raw))
        except CatalogError as exc:
            raise HTTPException(400, str(exc)) from None

    @app.post("/api/lr/sync")
    def lr_sync(req: LrSyncRequest) -> dict[str, str]:
        f = validate_folder(req.folder, state)
        if not cache_path(f).exists():
            raise HTTPException(404, "this folder has not been culled yet")
        job = Job(
            uuid.uuid4().hex[:12], str(f), kind="lr_sync",
            payload={"catalog": _catalog_arg(req.catalog), "dry_run": req.dry_run},
        )  # fmt: skip
        return {"job_id": _start_job(job)}

    @app.get("/api/lr/report")
    def lr_report(folder: str | None = None) -> dict[str, Any]:
        f = validate_folder(folder, state)
        if not cache_path(f).exists():
            raise HTTPException(404, "this folder has not been culled yet")
        with Cache.for_folder(f) as c:
            return {
                "report": c.get_meta("lr_report"), "last": c.get_meta("lr_sync_last"),
                "catalog": c.get_meta("lr_catalog"),
            }  # fmt: skip

    @app.get("/api/lr/catalogs")
    def lr_catalogs(folder: str | None = None) -> dict[str, Any]:
        from shotsieve.pipeline import lrsync

        remembered: list[str] = []
        try:
            f = validate_folder(folder, state)
            if cache_path(f).exists():
                with Cache.for_folder(f) as c:
                    remembered = [c.get_meta("lr_catalog")] if c.get_meta("lr_catalog") else []
        except HTTPException:
            pass
        found = lrsync.detect_catalogs(remembered=remembered)
        return {"catalogs": found, "remembered": remembered[0] if remembered else None}

    # ---- editing profile (T15) --------------------------------------------------------------
    @app.get("/api/edit/profile")
    def get_edit_profile() -> dict[str, Any]:
        from shotsieve.pipeline import edit

        return edit.profile_status(state.settings)

    @app.delete("/api/edit/profile")
    def delete_edit_profile() -> dict[str, Any]:
        from shotsieve.pipeline import edit

        return {"deleted": edit.delete_profile(state.settings)}

    @app.post("/api/edit/train")
    def post_edit_train(req: EditTrainRequest) -> dict[str, str]:
        cat = _file_arg(req.catalog, "catalog")
        xmp_dirs = [str(validate_folder(x, state)) for x in req.xmp]
        if not cat and not xmp_dirs:
            raise HTTPException(400, "give a catalog and/or XMP folders")
        job = Job(uuid.uuid4().hex[:12], "", kind="edit_train",
                  payload={"catalog": cat, "xmp": xmp_dirs,
                           "dng_render": req.dng_render})  # fmt: skip
        return {"job_id": _start_job(job)}

    @app.post("/api/edit/apply")
    def post_edit_apply(req: EditApplyRequest) -> dict[str, str]:
        from shotsieve.pipeline import edit

        folder = validate_folder(req.folder, state)
        if req.scope not in ("keeps", "all"):
            raise HTTPException(400, "scope must be 'keeps' or 'all'")
        if not edit.profile_status(state.settings)["trained"]:
            raise HTTPException(409, "no editing profile yet: train one first")
        embed = _embed_arg(req.embed_copies, folder)
        job = Job(uuid.uuid4().hex[:12], str(folder), kind="edit_apply",
                  payload={"scope": req.scope, "ids": req.ids, "dry_run": req.dry_run,
                           "embed_copies": embed,
                           "catalog": _file_arg(req.catalog, "catalog")})  # fmt: skip
        return {"job_id": _start_job(job)}

    def _ref_path(folder: Path, image_id: int) -> str:
        with Cache.for_folder(folder) as c:
            rp = c.path_for_id(image_id)
        if not rp or split_virtual(rp):
            raise HTTPException(404, "no such image (or a video frame)")
        return rp

    @app.get("/api/edit/reference")
    def get_edit_reference(id: int, folder: str | None = None, catalog: str | None = None) -> Any:
        """Can this frame be a colour-match reference? (it must carry develop settings)"""
        from shotsieve.pipeline import match

        f = validate_folder(folder, state)
        rp = _ref_path(f, id)
        try:
            st, source = match.reference_settings(rp, _file_arg(catalog, "catalog"))
        except ValueError as exc:
            return {"ok": False, "message": str(exc), "id": id}
        keys = [k for k in st if k not in ("ProcessVersion", "Version", "HasSettings")]
        return {
            "ok": True, "id": id, "source": source, "n_settings": len(keys),
            "exposure": st.get("Exposure2012"), "temperature": st.get("Temperature"),
        }

    @app.post("/api/edit/match")
    def post_edit_match(req: EditMatchRequest) -> dict[str, str]:
        from shotsieve.pipeline import match

        folder = validate_folder(req.folder, state)
        rp = _ref_path(folder, req.ref)
        if req.targets not in ("keeps", "group") and not req.targets.lower().startswith("camera:"):
            raise HTTPException(400, "targets must be keeps, group or camera:MODEL")
        cat = _file_arg(req.catalog, "catalog")
        try:
            match.reference_settings(rp, cat)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        embed = _embed_arg(req.embed_copies, folder)
        job = Job(uuid.uuid4().hex[:12], str(folder), kind="edit_match",
                  payload={"ref": rp, "targets": req.targets, "ids": req.ids,
                           "dry_run": req.dry_run, "embed_copies": embed,
                           "catalog": cat})  # fmt: skip
        return {"job_id": _start_job(job)}

    @app.get("/api/edit/predictions")
    def get_edit_predictions(ids: str = "", folder: str | None = None) -> dict[str, Any]:
        from shotsieve.pipeline import edit

        f = validate_folder(folder, state)
        try:
            want = [int(x) for x in ids.split(",") if x.strip()][:200]
        except ValueError:
            raise HTTPException(400, "ids must be comma-separated integers") from None
        if not cache_path(f).exists():
            raise HTTPException(404, "this folder has not been culled yet")
        got = edit.predictions_for(f, want, state.settings, 0 if state.busy_with(f) else 1)
        return {"predictions": {str(k): v for k, v in got.items()}}

    @app.post("/api/cull")
    def start_cull(req: CullRequest) -> dict[str, str]:
        from shotsieve.models.registry import group_names, verify_model

        folder = validate_folder(req.folder or None, state)
        missing = [n for n in group_names("core") if not verify_model(n)]
        if missing:
            raise HTTPException(
                503, f"models missing: {', '.join(missing)}; run `shotsieve models download`"
            )
        only: list[str] | None = None
        video = req.video
        if req.files is not None:
            from shotsieve.io.formats import is_supported, is_video

            only = []
            for raw in req.files:
                f = Path(os.path.abspath(Path(raw).expanduser()))
                if "\x00" in raw or ".." in Path(raw).parts or not f.is_absolute() or (
                        f.parent != folder or not f.is_file()):
                    raise HTTPException(400, f"not a file directly inside the folder: {raw}")
                if not (is_supported(f) or is_video(f)):
                    raise HTTPException(400, f"unsupported file type: {raw}")
                only.append(str(f))
            if not only:
                raise HTTPException(400, "no files given")
            video = video or any(is_video(f) for f in only)
        with state.lock:
            if state.active is not None and state.active.state in ("queued", "running"):
                raise HTTPException(409, "a cull job is already running")
            if folder in state.calibrating:
                raise HTTPException(409, "this folder is being recalibrated")
            job = Job(uuid.uuid4().hex[:12], str(folder), req.write_xmp, video, only=only)
            state.jobs[job.id] = job
            state.active = job
            state.current_folder = folder
            threading.Thread(target=run_job, args=(job,), name="cull-job", daemon=True).start()
        onboarding.add_recent(folder)
        return {"job_id": job.id}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        job = state.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "unknown job")
        return job.public()

    # ---- shoot ----------------------------------------------------------------------------
    @app.get("/api/shoot")
    def get_shoot(folder: str | None = None) -> dict[str, Any]:
        f = validate_folder(folder, state)  # read-only: never changes the server's folder
        out = build_shoot(f, state.settings)
        _maybe_recluster(f)
        return out

    @app.get("/api/thumb/{image_id}")
    def get_thumb(
        image_id: int, size: int = Query(400, ge=64, le=1600), folder: str | None = None
    ) -> FileResponse:
        f = validate_folder(folder, state)
        dest = _image_file(f, image_id, size, "t", THUMB_QUALITY)
        return FileResponse(  # the name carries the file's mtime: safe to cache for good
            dest, media_type="image/jpeg",
            headers={"Cache-Control": "max-age=86400, immutable"},
        )  # fmt: skip

    @app.get("/api/preview/{image_id}")
    def get_preview(
        image_id: int,
        w: int | None = Query(None, ge=64, le=100000),
        bw: int | None = Query(None, ge=64, le=100000),
        bh: int | None = Query(None, ge=64, le=100000),
        folder: str | None = None,
    ) -> FileResponse:
        """The picture for the loupe / compare at the size the screen needs. `w` = the long edge
        in device pixels; or `bw` x `bh` = the box it is shown in (device pixels), from which the
        server works out the long edge at Fit with the image's own aspect. It snaps up to 1280 /
        2048 / 2880 / 3840 (cache-friendly); above that the native-size image is served (as
        /api/full). Without any: 2048. Never upscaled; q92, 4:4:4, sRGB profile."""
        f = validate_folder(folder, state)
        virtual = split_virtual(_path_of(f, image_id)) is not None
        if w is None and bw and bh:
            if virtual:
                w = max(bw, bh)
            else:
                iw, ih = _image_size(f, image_id)
                w = round(max(iw, ih) * min(1.0, bw / iw, bh / ih))
        edge = bucket_for(w)
        if edge is None and virtual:
            edge = 3840  # a video frame has no native-size view
        dest = _full_file(f, image_id) if edge is None else _image_file(
            f, image_id, edge, "v", VIEW_QUALITY
        )
        return FileResponse(
            dest, media_type="image/jpeg",
            headers={"Cache-Control": "max-age=86400, immutable"},
        )  # fmt: skip

    @app.get("/api/full/{image_id}/size")
    def get_full_size(image_id: int, folder: str | None = None) -> dict[str, Any]:
        """Pixel size /api/full/{id} has (orientation applied), without decoding the image."""
        f = validate_folder(folder, state)
        w, h = _image_size(f, image_id)
        return {"w": w, "h": h}

    @app.get("/api/full/{image_id}")
    def get_full(image_id: int, folder: str | None = None) -> FileResponse:
        """The image at native size (RAW demosaiced or its full-size embedded JPEG), cached
        under <shoot>/.shotsieve/full/. The original is never touched."""
        f = validate_folder(folder, state)
        dest = _full_file(f, image_id)
        return FileResponse(
            dest, media_type="image/jpeg",
            headers={"Cache-Control": "max-age=86400, immutable"},
        )  # fmt: skip

    @app.post("/api/video/grab")
    def video_grab(req: GrabRequest) -> dict[str, Any]:
        """Export the full-resolution frame behind a video candidate to <shoot>/frame_grabs/."""
        from shotsieve.pipeline.video import export_frame

        f = validate_folder(req.folder, state)
        virtual = _path_of(f, req.image_id)
        if split_virtual(virtual) is None:
            raise HTTPException(422, "not a video frame")
        try:
            dest = export_frame(f, virtual)
        except (OSError, ValueError) as exc:
            raise HTTPException(422, f"cannot export frame: {exc}") from exc
        return {"ok": True, "path": str(dest), "rel": os.path.relpath(dest, f)}

    # ---- overrides / audit / xmp ------------------------------------------------------------
    @app.post("/api/override")
    def post_override(req: OverrideRequest) -> dict[str, Any]:
        f = validate_folder(req.folder, state)
        if req.decision is not None and req.decision not in (*DECISIONS, "model"):
            raise HTTPException(422, "decision must be keep, review, reject or model")
        if req.rating is not None and not 0 <= req.rating <= 5:
            raise HTTPException(422, "rating must be 0-5")
        if req.decision is None and req.rating is None:
            raise HTTPException(422, "give a decision and/or a rating")
        if (req.image_id is None) == (req.ids is None):
            raise HTTPException(422, "give image_id or ids")
        if req.ids is not None:
            if not 1 <= len(req.ids) <= MAX_BATCH:
                raise HTTPException(422, f"give 1 to {MAX_BATCH} image ids")
            ids = list(dict.fromkeys(req.ids))
            paths = {i: _path_of(f, i) for i in ids}  # any unknown id: nothing is changed
            results = []
            with Cache.for_folder(f) as c:
                rows, ovrs = c.load_decisions(), c.overrides()
                for i in ids:
                    row = rows.get(paths[i])
                    if row is None:
                        raise HTTPException(404, f"unknown image {i}")
                for i in ids:
                    results.append({"id": i, **_apply_override(c, paths[i], rows[paths[i]],
                                                               ovrs.get(paths[i]), req,
                                                               commit=False)})
                c.commit()
            state.autoxmp.request(f, set(paths.values()))  # one batched write for all of them
            return {"ok": True, "n": len(ids), "results": results}
        path = _path_of(f, req.image_id)
        with Cache.for_folder(f) as c:
            row = c.load_decisions().get(path)
            if row is None:
                raise HTTPException(404, "unknown image")
            out = _apply_override(c, path, row, c.overrides().get(path), req)
        state.autoxmp.request(f, {path})
        return {"ok": True, "id": req.image_id, **out}

    @app.post("/api/audit/complete")
    def audit_complete(req: FolderRequest) -> dict[str, Any]:
        from shotsieve.personal.calibrate import calibrate_shoot

        f = validate_folder(req.folder, state)
        if not cache_path(f).exists():
            raise HTTPException(404, "this folder has not been culled yet")
        with state.lock:  # check-and-claim atomically so a new cull cannot slip in
            if state._cull_running(f) or f in state.calibrating:
                raise HTTPException(409, "a cull job is running on this folder")
            state.calibrating.add(f)
        try:
            result, cull = calibrate_shoot(f, state.settings, redecide=True)
        finally:
            with state.lock:
                state.calibrating.discard(f)
        if result.n_pending:
            raise HTTPException(
                409,
                f"{result.n_pending} audited reject(s) are not judged yet; keep or reject each "
                "of them (or click the same button again) before completing the audit",
            )
        out = result.to_dict()
        out["redecided"] = cull is not None
        if cull is not None:
            out["counts"] = cull.stats["counts"]
        return out

    @app.get("/api/clock")
    def get_clock(folder: str | None = None) -> dict[str, Any]:
        from shotsieve.grouping import clock

        f = validate_folder(folder, state)
        if not cache_path(f).exists():
            raise HTTPException(404, "this folder has not been culled yet")
        with Cache.for_folder(f) as c:
            sugs, offsets = clock.analyse(c, auto_apply=False)
            items = c.clock_inputs()
            paths = {p for s in sugs for p in _sug_paths(s)}
            ids = c.ids_for(sorted(paths))
        cams: dict[str, int] = {}
        for i in items:
            cams[i.serial] = cams.get(i.serial, 0) + 1

        def pair(p: Any) -> dict[str, Any]:
            return {"ref_id": ids[p.ref_path], "id": ids[p.path], "ref_name": Path(p.ref_path).name,
                    "name": Path(p.path).name, "cos": p.cos, "delta_s": p.delta_s}  # fmt: skip

        out = []
        for s in sugs:
            d = s.to_dict()
            d["pair"] = pair(s.pair) if s.pair else None
            d["alternatives"] = [pair(p) for p in s.alternatives]
            out.append(d)
        return {"cameras": [{"serial": k, "n": v} for k, v in sorted(cams.items())],
                "offsets": offsets, "suggestions": out}  # fmt: skip

    @app.post("/api/clock")
    def post_clock(req: ClockRequest) -> dict[str, Any]:
        f = validate_folder(req.folder, state)
        if not cache_path(f).exists():
            raise HTTPException(404, "this folder has not been culled yet")
        from shotsieve.grouping import clock

        with Cache.for_folder(f) as c:
            if req.clear:
                c.clear_clock_offset(req.serial)
                return {"ok": True, "offsets": c.get_clock_offsets()}
            offset = req.offset_s
            if req.pair is not None:
                if len(req.pair) != 2:
                    raise HTTPException(400, "pair needs two image ids")
                pa, pb = (c.path_for_id(i) for i in req.pair)
                by = {i.path: i for i in c.clock_inputs()}
                if pa not in by or pb not in by:
                    raise HTTPException(404, "unknown image id")
                a, b = by[pa], by[pb]
                if a.serial == b.serial or req.serial not in (a.serial, b.serial):
                    raise HTTPException(
                        400, f"the pair must show one frame from camera {req.serial} and one other"
                    )
                this, other = (a, b) if a.serial == req.serial else (b, a)
                # land `this` where the other body's *adjusted* frame is
                other_off = clock.offsets_from_meta(c.get_clock_offsets()).get(other.serial, 0.0)
                offset = other.ts + other_off - this.ts
            if offset is None:
                raise HTTPException(400, "give offset_s or pair")
            offsets = c.set_clock_offset(req.serial, float(offset), "manual")
        return {"ok": True, "serial": req.serial, "offset_s": float(offset), "offsets": offsets}

    @app.get("/api/xmp/status")
    def xmp_status(folder: str | None = None) -> dict[str, Any]:
        return state.autoxmp.status(validate_folder(folder, state))

    @app.post("/api/xmp/auto")
    def xmp_auto(req: XmpAutoRequest) -> dict[str, Any]:
        f = validate_folder(req.folder, state)
        if not cache_path(f).exists():
            raise HTTPException(404, "this folder has not been culled yet")
        state.autoxmp.set_enabled(f, req.on)
        return state.autoxmp.status(f)

    @app.post("/api/xmp/write")
    def xmp_write(req: XmpRequest) -> dict[str, Any]:
        f = validate_folder(req.folder, state)
        return write_xmp_sidecars(f, state.settings, req.force_ratings)

    # ---- UI -------------------------------------------------------------------------------
    onboarding.register(app, state)  # T19: /api/onboarding, /api/models*, /api/fs/*, /api/train
    crop_server.register(app, state, _start_job)  # T22: /api/crop/*
    subject_server.register(app, state)  # T23h: /api/subject/{id}/score
    denoise_server.register(app, state)  # T23f: /api/denoise/* (view-only denoise)
    lightroom_server.register(app, state)  # T23d: /api/lightroom/*
    trash_server.register(app, state, _start_job)  # T23i: /api/trash, /api/trash/undo

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
