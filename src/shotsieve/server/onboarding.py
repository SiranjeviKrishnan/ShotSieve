"""First-run onboarding endpoints (T19): setup state, model downloads, file browser, training.

Registered from `create_app` (`register(app, state)`), all under the app's M5 rules: GETs never
write, POSTs only from the local Origin (enforced by the app middleware).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from shotsieve._ort import user_home
from shotsieve.config import data_dir
from shotsieve.io.formats import JPEG_EXTS, RAW_EXTS, VIDEO_EXTS, scan_folder
from shotsieve.log import get_logger
from shotsieve.pipeline.cache import cache_path

log = get_logger("onboarding")

ONBOARDING_VERSION = 1
FS_LIMIT = 500
PICK_TIMEOUT_S = 120
MIN_TRAIN_ROWS = 40
DOWNLOAD_GROUPS = ("core", "rawdenoise", "search", "ocr", "crop", "denoise")
HEIC_EXTS = frozenset({".heic", ".heif"})


# ---- request models ---------------------------------------------------------------------


class OnboardingRequest(BaseModel):
    completed: bool | None = None
    skipped: bool | None = None
    last_step: int | None = None


class DownloadRequest(BaseModel):
    groups: list[str] = ["core"]


class PickRequest(BaseModel):
    kind: str = "folder"  # folder | lrcat


class RecentRequest(BaseModel):
    folder: str


class TrainRequest(BaseModel):
    catalog: str | None = None
    xmp_folders: list[str] = []
    from_overrides: list[str] = []


# ---- small state files ------------------------------------------------------------------


def _onboarding_path() -> Path:
    return data_dir() / "onboarding.json"


def read_onboarding() -> dict[str, Any]:
    base = {"version": ONBOARDING_VERSION, "completed_at": None, "skipped": False,
            "last_step": 0}
    try:
        data = json.loads(_onboarding_path().read_text())
        if isinstance(data, dict):
            base.update({k: data[k] for k in base if k in data})
    except (OSError, ValueError):
        pass
    return base


def _atomic_write(path: Path, obj: Any) -> None:
    tmp = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(obj))
    tmp.replace(path)


def _recent_path() -> Path:
    return data_dir() / "recent.json"


def read_recent() -> list[str]:
    try:
        data = json.loads(_recent_path().read_text())
        return [p for p in data if isinstance(p, str)][:10]
    except (OSError, ValueError):
        return []


def add_recent(folder: str | Path) -> None:
    """Remember a shoot folder (most recent first, 10 entries). Best effort."""
    try:
        f = str(folder)
        items = [f] + [p for p in read_recent() if p != f]
        _atomic_write(_recent_path(), items[:10])
    except OSError:
        pass


# ---- file system browser (HOME and /Volumes only) ---------------------------------------


def _roots() -> list[Path]:
    home = Path(os.path.realpath(user_home()))
    return [home, Path("/Volumes")]


def _under(real: Path, root: Path) -> bool:
    return real == root or root in real.parents


def fs_path(raw: str | None, *, want: str = "dir") -> Path:  # want: dir | lrcat
    """Validate a browser path: absolute, no `..`/NUL, realpath inside HOME or /Volumes."""
    if not raw or "\x00" in raw:
        raise HTTPException(400, "invalid path")
    p = Path(raw)
    if not p.is_absolute():
        raise HTTPException(400, "path must be absolute")
    if ".." in p.parts:
        raise HTTPException(400, "path traversal is not allowed")
    real = Path(os.path.realpath(p))
    if not any(_under(real, r) for r in _roots()):
        raise HTTPException(403, "ShotSieve only browses your home folder and /Volumes")
    if want == "dir" and not real.is_dir():
        raise HTTPException(400, f"not a directory: {raw}")
    if want == "lrcat" and not (real.is_file() and real.suffix.lower() == ".lrcat"):
        raise HTTPException(400, "choose a Lightroom catalog (.lrcat file)")
    return real


def _count_images(d: Path) -> int:
    n = 0
    try:
        with os.scandir(d) as it:
            for e in it:
                ext = os.path.splitext(e.name)[1].lower()
                if ext in RAW_EXTS or ext in JPEG_EXTS or ext in HEIC_EXTS or ext in (
                    ".png", ".tif", ".tiff", ".webp"
                ):
                    n += 1
    except OSError:
        pass
    return n


def fs_list(real: Path, kind: str) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    truncated = False
    try:
        scan = sorted(os.scandir(real), key=lambda e: e.name.lower())
    except OSError as exc:
        raise HTTPException(403, f"cannot read this folder ({exc.strerror})") from None
    for e in scan:
        if e.name.startswith("."):
            continue
        try:
            if e.is_dir():
                er = Path(os.path.realpath(e.path))
                if not any(_under(er, r) for r in _roots()) or er == Path("/Volumes"):
                    continue  # a symlink that leaves the allowed roots
                entries.append({"name": e.name, "path": os.path.join(str(real), e.name),
                                "dir": True, "n_images": _count_images(Path(e.path))})
            elif kind == "lrcat" and e.name.lower().endswith(".lrcat") and e.is_file():
                entries.append({"name": e.name, "path": os.path.join(str(real), e.name),
                                "dir": False, "size": e.stat().st_size})
        except OSError:
            continue
        if len(entries) >= FS_LIMIT:
            truncated = True
            break
    parent = real.parent
    if not any(_under(Path(os.path.realpath(parent)), r) for r in _roots()):
        parent_s = None
    else:
        parent_s = str(parent)
    return {"dir": str(real), "parent": parent_s, "entries": entries, "truncated": truncated,
            "roots": [str(r) for r in _roots() if r.exists()]}


def fs_scan(real: Path) -> dict[str, Any]:
    by: Counter[str] = Counter()
    for en in scan_folder(real):
        ext = en.master.suffix.lower()
        if ext == ".dng":
            by["dng"] += 1
        elif ext in RAW_EXTS:
            by["raw"] += 1
        elif ext in JPEG_EXTS:
            by["jpeg"] += 1
        elif ext in HEIC_EXTS:
            by["heic"] += 1
        else:
            by["other"] += 1
    videos = 0
    try:
        for _dp, dn, fn in os.walk(real):
            dn[:] = [d for d in dn if not d.startswith(".") and d != "shotsieve_edits"]
            videos += sum(1 for f in fn if os.path.splitext(f)[1].lower() in VIDEO_EXTS)
    except OSError:
        pass
    keys = ("raw", "jpeg", "dng", "heic", "other")
    out = {k: by.get(k, 0) for k in keys}
    out["video"] = videos
    return {"folder": str(real), "n_images": sum(by.values()), "by_format": out,
            "culled": cache_path(real).exists()}


def run_picker(kind: str) -> str | None:
    """Native macOS chooser via osascript (or the SHOTSIEVE_PICKER test stub). None = cancelled."""
    stub = os.environ.get("SHOTSIEVE_PICKER")
    if stub:
        cmd = [stub, kind]
    elif sys.platform == "darwin":
        what = "choose folder with prompt \"Choose your shoot folder\"" if kind == "folder" else (
            "choose file with prompt \"Choose your Lightroom catalog\" of type {\"lrcat\"}"
        )
        cmd = ["/usr/bin/osascript", "-e", f"POSIX path of ({what})"]
    else:
        raise HTTPException(501, "the native chooser is only available on macOS")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=PICK_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return None
    except OSError as exc:
        raise HTTPException(501, f"cannot start the chooser: {exc}") from None
    out = r.stdout.strip()
    if r.returncode != 0 or not out:
        return None  # cancelled
    return out.rstrip("/") if len(out) > 1 else out


# ---- registration -----------------------------------------------------------------------


def _plain_error(exc: BaseException) -> str:
    if isinstance(exc, urllib.error.URLError | TimeoutError | ConnectionError):
        return ("Could not reach the download server. Check your internet connection, "
                "then press Retry.")
    return str(exc) or type(exc).__name__


def register(app: FastAPI, state: Any) -> None:
    from shotsieve.server.app import Job, validate_folder

    state.download = None  # the single download slot: a Job while one is running

    # ---- setup state ----
    @app.get("/api/onboarding")
    def get_onboarding() -> dict[str, Any]:
        from shotsieve.models.registry import bundled, group_names, verify_model

        ob = read_onboarding()
        models_ready = all(verify_model(n) for n in group_names("core"))
        done = ob["completed_at"] is not None
        show = not ob["skipped"] and (not done or not models_ready)
        paths = {getattr(r, "path", "") for r in app.routes}
        features = {"lr": "/api/lr/catalogs" in paths and "/api/lr/sync" in paths}
        return {**ob, "completed": done, "models_ready": models_ready, "show": show,
                "features": features, "bundled": bundled()}

    @app.post("/api/onboarding")
    def post_onboarding(req: OnboardingRequest) -> dict[str, Any]:
        ob = read_onboarding()
        if req.completed is True:
            ob["completed_at"] = time.time()
            ob["skipped"] = False
        elif req.completed is False:
            ob["completed_at"] = None
        if req.skipped is not None:
            ob["skipped"] = req.skipped
        if req.last_step is not None:
            ob["last_step"] = max(0, min(int(req.last_step), 9))
        _atomic_write(_onboarding_path(), ob)
        return {**ob, "completed": ob["completed_at"] is not None}

    @app.get("/api/recent")
    def get_recent() -> dict[str, Any]:
        out = []
        for p in read_recent():
            if Path(p).is_dir():
                out.append({"path": p, "name": Path(p).name or p})
        return {"folders": out}

    # ---- models ----
    @app.get("/api/models")
    def get_models() -> dict[str, Any]:
        from shotsieve.models.registry import models_status

        dl = state.download
        return {"groups": models_status(), "download": dl.public() if dl else None}

    def _run_download(job: Job, groups: tuple[str, ...]) -> None:
        from shotsieve.models.registry import ensure_models

        t0 = time.monotonic()
        first = {"done": None}

        def cb(ev: dict) -> None:
            if first["done"] is None:
                first["done"] = ev["done"] - ev["file_done"] if ev["file_done"] else ev["done"]
            job.stage, job.done, job.total = ev["name"], ev["done"], ev["total"]
            moved = max(ev["done"] - (first["done"] or 0), 0)
            el = max(time.monotonic() - t0, 1e-3)
            rate = moved / el
            eta = (ev["total"] - ev["done"]) / rate if rate > 0 else None
            job.result = {"file": ev["name"], "files_done": ev["files_done"],
                          "files": ev["files"], "mb_s": round(rate / 1e6, 2),
                          "eta_s": None if eta is None else round(eta, 1),
                          "groups": list(groups)}

        job.state = "running"
        try:
            ensure_models(groups=groups, progress=cb)
            if "core" in groups and "rawdenoise" not in groups:
                try:  # T26: the RAW denoiser comes with the core models, but never blocks the cull
                    ensure_models(groups=("rawdenoise",))
                except Exception as exc:  # noqa: BLE001
                    log.warning("RawNIND download failed (RAW denoise falls back): %s", exc)
            job.done = job.total
            job.state = "done"
        except Exception as exc:  # noqa: BLE001 - shown to the user with a Retry button
            log.warning("model download failed: %s", exc)
            job.error, job.state = _plain_error(exc), "error"

    @app.post("/api/models/download")
    def post_models_download(req: DownloadRequest) -> dict[str, str]:
        groups = tuple(dict.fromkeys(req.groups))
        if not groups or any(g not in DOWNLOAD_GROUPS for g in groups):
            raise HTTPException(400, f"groups must be a subset of {list(DOWNLOAD_GROUPS)}")
        with state.lock:
            cur = state.download
            if cur is not None and cur.state in ("queued", "running"):
                raise HTTPException(409, "a download is already running")
            job = Job(uuid.uuid4().hex[:12], "", kind="models_download")
            job.payload = {"groups": list(groups)}
            state.jobs[job.id] = job
            state.download = job
            threading.Thread(target=_run_download, args=(job, groups), name="model-download",
                             daemon=True).start()
        return {"job_id": job.id}

    # ---- file browser ----
    @app.get("/api/fs/list")
    def get_fs_list(dir: str | None = None, kind: str = Query("folder")) -> dict[str, Any]:
        if kind not in ("folder", "lrcat"):
            raise HTTPException(400, "kind must be folder or lrcat")
        if not dir:
            dir = str(_roots()[0])
        return fs_list(fs_path(dir), kind)

    @app.get("/api/fs/scan")
    def get_fs_scan(folder: str | None = None) -> dict[str, Any]:
        return fs_scan(fs_path(folder))

    @app.post("/api/fs/pick")
    def post_fs_pick(req: PickRequest) -> dict[str, Any]:
        if req.kind not in ("folder", "lrcat"):
            raise HTTPException(400, "kind must be folder or lrcat")
        chosen = run_picker(req.kind)
        if chosen is None:
            return {"path": None, "cancelled": True}
        real = fs_path(chosen, want="dir" if req.kind == "folder" else "lrcat")
        return {"path": str(real), "cancelled": False}

    # ---- training ----
    def _run_train(job: Job) -> None:
        from shotsieve.personal.dataset import build_training_set
        from shotsieve.personal.model import PersonalModel, default_model_path

        p = job.payload
        t0 = time.monotonic()
        job.state = "running"

        def progress(stage: str, done: int, total: int) -> None:
            if stage != "images":
                return
            job.stage, job.done, job.total = "images", done, total
            el = max(time.monotonic() - t0, 1e-3)
            rate = done / el
            eta = (total - done) / rate if rate > 0 and total else None
            job.result = {"img_s": round(rate, 2), "eta_s": None if eta is None else round(eta, 1)}

        try:
            ts = build_training_set(
                p.get("catalog"), p.get("xmp_folders") or [], p.get("from_overrides") or [],
                state.settings, progress,
            )
            n = len(ts)
            model = PersonalModel().fit(ts.X, ts.y, ts.w, ts.embeddings)
            model.meta = {"sources": sorted(set(ts.sources)), "trained_at": time.time()}
            trained = bool(model.trained)
            if trained:
                model.save(default_model_path(state.settings))
            msg = (f"Learned from {n:,} of your picks" if trained else
                   f"Not enough picks yet (need {MIN_TRAIN_ROWS}); "
                   "ShotSieve uses its built-in judgement")
            job.result = {"trained": trained, "n_rows": n, "min_rows": MIN_TRAIN_ROWS,
                          "n_missing": ts.n_missing, "message": msg}
            job.state = "done"
        except Exception as exc:  # noqa: BLE001
            log.exception("train job failed")
            job.error, job.state = f"{type(exc).__name__}: {exc}", "error"

    def _catalog_arg(raw: str | None) -> str | None:
        if not raw:
            return None
        if "\x00" in raw or ".." in Path(raw).parts or not Path(raw).is_absolute():
            raise HTTPException(400, "catalog must be an absolute path without '..'")
        if not raw.lower().endswith(".lrcat") or not Path(raw).is_file():
            raise HTTPException(400, "catalog must be an existing .lrcat file")
        return raw

    @app.post("/api/train")
    def post_train(req: TrainRequest) -> dict[str, str]:
        from shotsieve.models.registry import group_names, verify_model

        catalog = _catalog_arg(req.catalog)
        xmp = [str(validate_folder(f, state)) for f in req.xmp_folders]
        ovr = [str(validate_folder(f, state)) for f in req.from_overrides]
        if not (catalog or xmp or ovr):
            raise HTTPException(400, "give a catalog, an XMP folder or folders with overrides")
        if any(not verify_model(n) for n in group_names("core")):
            raise HTTPException(503, "models missing; download them first")
        with state.lock:
            if state.active is not None and state.active.state in ("queued", "running"):
                raise HTTPException(409, "a job is already running")
            job = Job(uuid.uuid4().hex[:12], "", kind="train")
            job.payload = {"catalog": catalog, "xmp_folders": xmp, "from_overrides": ovr}
            state.jobs[job.id] = job
            state.active = job
            threading.Thread(target=_run_train, args=(job,), name="train-job",
                             daemon=True).start()
        return {"job_id": job.id}

