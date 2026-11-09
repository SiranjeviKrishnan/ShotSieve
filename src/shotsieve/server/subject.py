"""Subject mode (T23h): the sharpness / in-focus score of a user-drawn subject box.

The boxes themselves are the crop mode's user boxes (`/api/crop/{id}/boxes`, display-normalised,
EXIF orientation respected); this module only measures them. The measure is the cull's face
sharpness (`features.sharpness`: Laplacian variance at the canonical 96 px scale) computed on the
FULL-resolution region and normalised like the face value (`clip(lap / shoot reference, 0, 1)`
with the shoot's p90 face reference, floor 30), so "0.82" means the same thing as a face's
sharpness in the Faces row."""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from shotsieve.features import sharpness
from shotsieve.pipeline.cache import Cache

IN_FOCUS = 0.25  # the same cut-off the Faces row uses for "soft"
CACHE_MAX = 512
_cache: OrderedDict[tuple, dict[str, Any]] = OrderedDict()
_refs: dict[tuple, float] = {}
_lock = threading.Lock()


class ScoreBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float


class ScoreRequest(BaseModel):
    folder: str | None = None
    boxes: list[ScoreBox] = []


def shoot_face_ref(cache: Cache) -> float:
    """The shoot's face-sharpness reference (p90 of the main faces' raw values, floor 30)."""
    with cache._lock:
        n = cache._con.execute("SELECT COUNT(*) FROM features").fetchone()[0]
    key = (str(cache.path), n)
    with _lock:
        if key in _refs:
            return _refs[key]
    import json

    vals: list[float] = []
    with cache._lock:
        rows = cache._con.execute("SELECT json FROM features").fetchall()
    for (js,) in rows:
        try:
            faces = json.loads(js).get("faces") or []
        except (TypeError, ValueError):
            continue
        if faces and faces[0].get("sharp") is not None:
            vals.append(faces[0]["sharp"])
    ref = sharpness.shoot_reference(vals, sharpness.FACE_FLOOR)
    with _lock:
        _refs[key] = ref
    return ref


def label_of(norm: float) -> str:
    return "in focus" if norm >= IN_FOCUS else "soft"


def score_boxes(rgb_gray, boxes: list[tuple[float, float, float, float]], ref: float) -> list[dict]:
    out = []
    for b in boxes:
        raw = sharpness.region_sharpness(rgb_gray, b)
        if raw != raw:
            out.append({"box": list(b), "sharp": None, "raw": None, "in_focus": None, "label": "n/a"})  # noqa: E501
            continue
        norm = sharpness.normalise(raw, ref)
        out.append({
            "box": list(b), "sharp": round(norm, 3), "raw": round(raw, 2),
            "in_focus": bool(norm >= IN_FOCUS), "label": label_of(norm),
        })  # fmt: skip
    return out


def register(app: FastAPI, state: Any) -> None:
    from shotsieve.io.preview import PreviewError, load_full
    from shotsieve.pipeline.crop import CropError, check_boxes
    from shotsieve.server.app import _path_of, validate_folder

    @app.post("/api/subject/{image_id}/score")
    def post_score(image_id: int, req: ScoreRequest) -> dict[str, Any]:
        """Sharpness of each subject box, measured on the full-resolution pixels. Cached by
        image (path + mtime) and box."""
        f: Path = validate_folder(req.folder, state)
        path = _path_of(f, image_id)
        try:
            boxes = check_boxes([[b.x0, b.y0, b.x1, b.y1] for b in req.boxes])
            mtime = Path(path).stat().st_mtime_ns
        except CropError as exc:
            raise HTTPException(exc.status, str(exc)) from None
        except OSError:
            raise HTTPException(404, "image file is missing") from None
        with Cache.for_folder(f) as c:
            ref = shoot_face_ref(c)
        keys = [(str(f), path, mtime, tuple(round(v, 4) for v in b), round(ref, 2)) for b in boxes]
        results: list[dict | None] = []
        with _lock:
            for k in keys:
                hit = _cache.get(k)
                if hit is not None:
                    _cache.move_to_end(k)
                results.append(hit)
        todo = [i for i, r in enumerate(results) if r is None]
        if todo:
            try:
                import cv2

                gray = cv2.cvtColor(load_full(path), cv2.COLOR_RGB2GRAY)
            except (PreviewError, ValueError) as exc:
                raise HTTPException(422, str(exc)) from None
            fresh = score_boxes(gray, [boxes[i] for i in todo], ref)
            del gray
            with _lock:
                for i, r in zip(todo, fresh, strict=True):
                    results[i] = r
                    _cache[keys[i]] = r
                while len(_cache) > CACHE_MAX:
                    _cache.popitem(last=False)
        return {"scores": results, "ref": round(ref, 2), "in_focus_at": IN_FOCUS}
