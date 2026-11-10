"""Denoise for the loupe / compare (T23f): `/api/denoise/*`, registered from `create_app`.

View only: the original is never written. Denoised copies are JPEGs under
`<shoot>/.shotsieve/denoise/` (LRU-pruned like `/api/full`), keyed by image id + source mtime +
level + backend + engine VERSION (a changed blend never reuses an old result), so a changed
original, a new strength or a freshly installed model starts clean.

  GET /api/denoise/status                    model installed? backend, provider, memory state
  POST /api/denoise/warm                     load + compile the model in the background (no-op
                                             while a cull runs or when it is not installed)
  GET /api/denoise/{id}?level&w|bw&bh        the Fit picture (the viewer bucket image) denoised
  GET /api/denoise/{id}/region?level&x0&y0&x1&y1   a viewport region of the FULL-resolution image
                                             (0..1 of the frame), denoised on the native pixels
  GET /api/denoise/{id}/full?level           the whole full-resolution image denoised (background)

Region requests jump the queue: a running whole-image job pauses between tiles while one is
waiting. The cull and its decisions are not involved."""

from __future__ import annotations

import contextlib
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from PIL import Image

from shotsieve.io.formats import split_virtual
from shotsieve.io.preview import PreviewError, load_full
from shotsieve.io.viewer import VIEW_QUALITY, bucket_for, encode_jpeg
from shotsieve.log import get_logger
from shotsieve.models import denoise as engine

log = get_logger("denoise")

CACHE_CAP = 1024**3  # bytes of denoised JPEGs kept per shoot (LRU)
REGION_SNAP = 64  # region boxes are widened to this grid so a small pan re-uses the cache
REGION_MAX_PX = 6_000_000  # a viewport region above this is cropped around its centre
REGION_MARGIN = 32  # context pixels denoised around a region and discarded
ARRAY_KEEP_S = 45.0  # the decoded full-resolution array of the last image stays this long

_LEVEL_Q = Query("medium", pattern="^(low|medium|high)$")

_full_lock = threading.Lock()  # one whole-image denoise at a time
_decode_lock = threading.Lock()
_urgent = 0  # region requests waiting / running: whole-image jobs pause for them
_urgent_lock = threading.Lock()
_recent_full: deque[tuple] = deque(maxlen=2)  # the (compare) full jobs still wanted
_ARRAY: dict[str, Any] = {"key": None, "rgb": None, "t": 0.0, "timer": None}
_SIGMA: dict[tuple, float] = {}
_ISO: dict[tuple, int | None] = {}
_warm_lock = threading.Lock()


def _backend() -> str:
    return "ai" if engine.available() else "basic"


def _pause() -> None:
    while _urgent:
        time.sleep(0.01)


def _full_rgb(source: Path, mtime: int) -> np.ndarray:
    """The decoded full-resolution image (same frame as /api/full), kept briefly in memory so the
    region requests of one viewing session do not decode a 24 MP RAW each time."""
    key = (str(source), mtime)
    with _decode_lock:
        if _ARRAY["key"] == key and _ARRAY["rgb"] is not None:
            _ARRAY["t"] = time.monotonic()
            return _ARRAY["rgb"]
        _ARRAY.update(key=None, rgb=None)
        rgb = load_full(source)
        _ARRAY.update(key=key, rgb=rgb, t=time.monotonic())
        _arm_release()
        return rgb


def prefetch(source: Path, mtime: int) -> None:
    """Decode the full-resolution image in the background (a RAW demosaic is ~1 s) while the Fit
    picture is being denoised, so the first viewport region after a zoom does not wait for it."""
    if _ARRAY["key"] == (str(source), mtime):
        return

    def run() -> None:
        with contextlib.suppress(Exception):
            _full_rgb(source, mtime)

    threading.Thread(target=run, name="denoise-prefetch", daemon=True).start()


def _arm_release() -> None:
    if _ARRAY["timer"] is not None:
        _ARRAY["timer"].cancel()
    t = threading.Timer(ARRAY_KEEP_S, _release_array)
    t.daemon = True
    _ARRAY["timer"] = t
    t.start()


def _release_array() -> None:
    with _decode_lock:
        if time.monotonic() - _ARRAY["t"] < ARRAY_KEEP_S - 0.5:
            _arm_release()
            return
        _ARRAY.update(key=None, rgb=None, timer=None)
    from shotsieve.models.idle import _trim

    _trim()


def release_all() -> None:
    """Drop the cached array and the model session (tests, shutdown)."""
    with _decode_lock:
        if _ARRAY["timer"] is not None:
            _ARRAY["timer"].cancel()
        _ARRAY.update(key=None, rgb=None, timer=None)
    engine.release()


def _iso_of(master: str, mtime: int) -> int | None:
    key = (master, mtime)
    if key not in _ISO:
        try:
            from shotsieve.io.exif import read_meta

            _ISO[key] = read_meta(master).iso
        except Exception:  # noqa: BLE001
            _ISO[key] = None
        if len(_ISO) > 5000:
            _ISO.clear()
    return _ISO[key]


def _sigma_for(rgb: np.ndarray, source: Path, mtime: int, master: str) -> float:
    key = (str(source), mtime)
    if key not in _SIGMA:
        _SIGMA[key] = engine.effective_sigma(rgb, _iso_of(master, mtime))
        if len(_SIGMA) > 5000:
            _SIGMA.clear()
    return _SIGMA[key]


def _prune(cache_dir: Path, keep: Path, cap: int) -> None:
    try:
        files = [(f, f.stat()) for f in cache_dir.glob("*.jpg") if f != keep]
    except OSError:
        return
    total = sum(st.st_size for _, st in files)
    with contextlib.suppress(OSError):
        total += keep.stat().st_size
    for f, st in sorted(files, key=lambda x: x[1].st_mtime_ns):  # hits touch mtime: oldest = LRU
        if total <= cap:
            break
        with contextlib.suppress(OSError):
            f.unlink()
            total -= st.st_size


def _write(dest: Path, data: bytes) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, dest)
    _prune(dest.parent, dest, CACHE_CAP)


def _hit(dest: Path) -> bool:
    if dest.exists():
        with contextlib.suppress(OSError):
            os.utime(dest)  # LRU touch
        return True
    return False


def _jpeg_response(dest: Path, backend: str, extra: dict[str, str] | None = None) -> FileResponse:
    headers = {"Cache-Control": "max-age=86400, immutable", "X-Denoise-Backend": backend}
    headers.update(extra or {})
    return FileResponse(dest, media_type="image/jpeg", headers=headers)


def register(app: FastAPI, state: Any) -> None:
    from shotsieve.server.app import (
        _full_source,
        _image_file,
        _image_size,
        _path_of,
        validate_folder,
    )

    def prep(folder: str | None, image_id: int) -> tuple[Path, Path, int, str]:
        f = validate_folder(folder, state)
        master = _path_of(f, image_id)
        if split_virtual(master) is not None:
            raise HTTPException(422, "videos cannot be denoised")
        source, mtime = _full_source(f, image_id)
        return f, source, mtime, master

    def guard(fn):  # decode / model errors become 422, a superseded job 409
        try:
            return fn()
        except engine.Cancelled as exc:
            raise HTTPException(409, "superseded by a newer request") from exc
        except (PreviewError, OSError, ValueError) as exc:
            raise HTTPException(422, f"cannot denoise: {exc}") from exc

    @app.get("/api/denoise/status")
    def status() -> dict[str, Any]:
        from shotsieve.models.registry import MODEL_SPECS

        spec = MODEL_SPECS["nafnet_sidd"]
        return {
            "model": {"name": spec.name, "installed": engine.available(), "bytes": spec.size,
                      "license": spec.license, "group": spec.group},
            "backend": _backend(), "loaded": engine.loaded(),
            "levels": list(engine.LEVELS),
        }  # fmt: skip

    @app.post("/api/denoise/warm", status_code=202)
    def warm() -> dict[str, Any]:
        """Warm the model (CoreML compile) in the background when the loupe opens, so the first
        press of D is fast. Never while a cull runs; the session idle-unloads as usual."""
        if not engine.available():
            return {"warming": False, "reason": "no model"}
        with state.lock:
            if state.active is not None and state.active.state in ("queued", "running"):
                return {"warming": False, "reason": "cull running"}
        if engine.loaded():
            return {"warming": False, "reason": "loaded"}
        if _warm_lock.acquire(blocking=False):
            def run() -> None:
                try:
                    engine.warm()
                finally:
                    _warm_lock.release()

            threading.Thread(target=run, name="denoise-warm", daemon=True).start()
        return {"warming": True}

    @app.get("/api/denoise/{image_id}/region")
    def region(
        image_id: int,
        x0: float = Query(..., ge=0, le=1), y0: float = Query(..., ge=0, le=1),
        x1: float = Query(..., ge=0, le=1), y1: float = Query(..., ge=0, le=1),
        level: str = _LEVEL_Q, folder: str | None = None,
    ) -> FileResponse:  # fmt: skip
        """A box of the full-resolution frame (fractions 0..1) denoised on its native pixels."""
        global _urgent
        f, source, mtime, master = prep(folder, image_id)
        iw, ih = _image_size(f, image_id)
        px0, px1 = sorted((int(x0 * iw), int(np.ceil(x1 * iw))))
        py0, py1 = sorted((int(y0 * ih), int(np.ceil(y1 * ih))))
        if px1 - px0 < 8 or py1 - py0 < 8:
            raise HTTPException(422, "empty region")
        if (px1 - px0) * (py1 - py0) > REGION_MAX_PX:  # keep the middle of an over-large request
            k = (REGION_MAX_PX / ((px1 - px0) * (py1 - py0))) ** 0.5
            cx, cy = (px0 + px1) / 2, (py0 + py1) / 2
            hw, hh = (px1 - px0) * k / 2, (py1 - py0) * k / 2
            px0, px1, py0, py1 = int(cx - hw), int(cx + hw), int(cy - hh), int(cy + hh)
        g = REGION_SNAP  # widen to the grid
        px0, py0 = px0 // g * g, py0 // g * g
        px1, py1 = min(iw, -(-px1 // g) * g), min(ih, -(-py1 // g) * g)
        be = _backend()
        dest = (f / ".shotsieve" / "denoise"
                / f"r_{image_id}_{mtime}_v{engine.VERSION}_{level}_{be}_{px0}_{py0}_{px1}_{py1}.jpg"
                )  # fmt: skip
        extra = {"X-Region": f"{px0},{py0},{px1},{py1}", "X-Image-Size": f"{iw},{ih}"}
        if _hit(dest):
            return _jpeg_response(dest, be, extra)
        with _urgent_lock:
            _urgent += 1
        try:
            def work() -> tuple[bytes, str]:
                full = _full_rgb(source, mtime)
                sigma = _sigma_for(full, source, mtime, master)
                ex0, ey0 = max(0, px0 - REGION_MARGIN), max(0, py0 - REGION_MARGIN)
                ex1, ey1 = min(iw, px1 + REGION_MARGIN), min(ih, py1 + REGION_MARGIN)
                crop = np.ascontiguousarray(full[ey0:ey1, ex0:ex1])
                out, used = engine.denoise(crop, level, sigma=sigma)
                res = out[py0 - ey0 : py1 - ey0, px0 - ex0 : px1 - ex0]
                return encode_jpeg(np.ascontiguousarray(res), VIEW_QUALITY, icc=False), used

            data, used = guard(work)
        finally:
            with _urgent_lock:
                _urgent -= 1
        if used != be:  # the model failed at run time: file it under what really made it
            dest = dest.with_name(dest.name.replace(f"_{be}_", f"_{used}_"))
        _write(dest, data)
        return _jpeg_response(dest, used, extra)

    @app.get("/api/denoise/{image_id}/full")
    def full(image_id: int, level: str = _LEVEL_Q, folder: str | None = None) -> FileResponse:
        """The whole full-resolution image denoised (cached; a minute-scale background job)."""
        f, source, mtime, master = prep(folder, image_id)
        be = _backend()
        name = f"f_{image_id}_{mtime}_v{engine.VERSION}_{level}_{be}.jpg"
        dest = f / ".shotsieve" / "denoise" / name
        if _hit(dest):
            return _jpeg_response(dest, be)
        key = (image_id, mtime, level)
        if key not in _recent_full:
            _recent_full.append(key)
        with _full_lock:
            if _hit(dest):
                return _jpeg_response(dest, be)

            def work() -> tuple[bytes, str]:
                rgb = _full_rgb(source, mtime)
                sigma = _sigma_for(rgb, source, mtime, master)
                out, used = engine.denoise(
                    rgb, level, sigma=sigma, pause=_pause, cancel=lambda: key not in _recent_full
                )
                return encode_jpeg(out, VIEW_QUALITY, icc=True), used

            data, used = guard(work)
            if used != be:
                dest = dest.with_name(dest.name.replace(f"_{be}.", f"_{used}."))
            _write(dest, data)
        return _jpeg_response(dest, used)

    @app.get("/api/denoise/{image_id}")
    def view(
        image_id: int,
        level: str = _LEVEL_Q,
        w: int | None = Query(None, ge=64, le=100000),
        bw: int | None = Query(None, ge=64, le=100000),
        bh: int | None = Query(None, ge=64, le=100000),
        folder: str | None = None,
    ) -> FileResponse:
        """The Fit picture (the viewer image of that size) denoised. Above the top viewer bucket
        this is the whole full-resolution image (same as /full)."""
        f, source, mtime, master = prep(folder, image_id)
        prefetch(source, mtime)  # the zoomed-in regions will need the full-resolution pixels
        if w is None and bw and bh:
            iw, ih = _image_size(f, image_id)
            w = round(max(iw, ih) * min(1.0, bw / iw, bh / ih))
        edge = bucket_for(w)
        if edge is None:
            return full(image_id, level, folder)
        be = _backend()
        name = f"v_{image_id}_{mtime}_v{engine.VERSION}_{level}_{be}_{edge}.jpg"
        dest = f / ".shotsieve" / "denoise" / name
        if _hit(dest):
            return _jpeg_response(dest, be)

        def work() -> tuple[bytes, str]:
            # one noise level per photo, measured on the full-resolution pixels (a downscaled
            # picture hides its noise): Fit, region and full all use the same strength
            sigma = _sigma_for(_full_rgb(source, mtime), source, mtime, master)
            src = _image_file(f, image_id, edge, "v", VIEW_QUALITY)  # the picture the loupe shows
            with Image.open(src) as im:
                rgb = np.asarray(im.convert("RGB"))
            out, used = engine.denoise(rgb, level, sigma=sigma, pause=_pause)
            return encode_jpeg(out, VIEW_QUALITY, icc=True), used

        data, used = guard(work)
        if used != be:
            dest = dest.with_name(dest.name.replace(f"_{be}_", f"_{used}_"))
        _write(dest, data)
        return _jpeg_response(dest, used)

