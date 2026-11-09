"""Faces on each picture and the people list (T23d), registered from `create_app`.

Boxes are the face detector's boxes stored at cull time (`RawFeatures.faces`, preview pixels of
the EXIF-upright preview), returned normalised to 0..1 of that display frame: the same frame as
`/api/preview`, `/api/full` and the crop subjects. Names / person ids need the people consent;
the boxes and scores do not (they are part of the cull)."""

from __future__ import annotations

import math
import os
import uuid
from pathlib import Path
from typing import Any

import cv2
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from shotsieve.features.eyes import closed_score, is_closed
from shotsieve.features.vector import FeatureVector
from shotsieve.pipeline import people as ppl
from shotsieve.pipeline.cache import Cache, cache_path

FACE_PAD = 0.45  # margin around a face box in its thumbnail (share of the box size)


class AssignRequest(BaseModel):
    folder: str | None = None
    person_id: int | None = None
    name: str | None = None


class FolderBody(BaseModel):
    folder: str | None = None


class UnignoreBody(BaseModel):
    folder: str | None = None
    faces: list[list[Any]] = []


def _num(v: Any, nd: int = 3) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else round(f, nd)


def faces_payload(cache: Cache, path: str, settings: Any, consent: bool) -> dict[str, Any]:
    """Per-face boxes, person, scores and whether the face drove the decision."""
    raw = cache.raw_features(path)
    out: list[dict[str, Any]] = []
    if raw is None or not raw.width or not raw.height:
        return {"faces": out, "people_consent": consent}
    with cache._lock:
        row = cache._con.execute(
            "SELECT reasons, features FROM decisions WHERE path=?", (cache._rel(path),)
        ).fetchone()
    import json

    reasons = json.loads(row[0] or "[]") if row else []
    fv = FeatureVector.from_dict(json.loads(row[1] or "{}")) if row and row[1] else None
    owners = ppl.faces_of_image(cache, path) if consent else {}
    main = raw.faces[0] if raw.faces else None
    factor = None
    if fv is not None and main is not None and main.sharp == main.sharp and main.sharp > 0:
        nm = _num(fv.face_sharp_norm_main, 6)
        factor = None if nm is None else nm / main.sharp
    imp = [f for f in raw.faces if f.important]
    min_imp = min(imp, key=lambda f: f.sharp if f.sharp == f.sharp else 1e18) if imp else None
    emb = 0
    thresh = getattr(settings, "eyes_closed_thresh", 0.55)
    both = getattr(settings, "eyes_closed_both_thresh", 0.45)
    for k, f in enumerate(raw.faces):
        j = None
        if f.important:
            j, emb = emb, emb + 1
        x, y, w, h = f.bbox
        box = [
            max(0.0, x / raw.width), max(0.0, y / raw.height),
            min(1.0, (x + w) / raw.width), min(1.0, (y + h) / raw.height),
        ]  # fmt: skip
        drove = None
        if "eyes_closed" in reasons and f.important and (
            is_closed(f.blink_left, f.blink_right, thresh, both)
            or (_num(closed_score(f.blink_left, f.blink_right)) or 0) > thresh
        ):
            drove = "eyes_closed"
        elif {"face_out_of_focus", "blur"} & set(reasons) and (f is main or f is min_imp):
            drove = "face_out_of_focus" if "face_out_of_focus" in reasons else "blur"
        closed = _num(closed_score(f.blink_left, f.blink_right))
        pid, name = owners.get(j, (None, None)) if j is not None else (None, None)
        out.append(
            {
                "idx": k, "box": [round(v, 5) for v in box], "important": bool(f.important),
                "main": k == 0, "person_id": pid, "name": name, "face_idx": j,
                "det_score": _num(f.score),
                "sharp": None if factor is None or f.sharp != f.sharp else _num(f.sharp * factor),
                "eyes_open": None if closed is None else round(1 - closed, 3),
                "smile": _num(f.smile) if f.found else None,
                "drove": drove is not None, "drove_reason": drove,
            }  # fmt: skip
        )
    return {"faces": out, "people_consent": consent}


def _face_file(
    folder: Path, cache: Cache, image_id: int, path: str, k: int, size: int, rec: int | None = None
) -> Path:
    """Thumbnail of decision face `k`, or (rec = index among the recognition-only faces of the
    T23h scan) of that face; k is ignored then."""
    from shotsieve.pipeline import crop as pcrop

    raw = cache.raw_features(path)
    if raw is None or not raw.width:
        raise HTTPException(404, "unknown face")
    if rec is None and not (0 <= k < len(raw.faces)):
        raise HTTPException(404, "unknown face")
    rbox = None
    if rec is not None:
        rb = cache.rec_boxes(path)
        if not (0 <= rec < len(rb)):
            raise HTTPException(404, "unknown face")
        n = rb[rec]
        rbox = (n[0] * raw.width, n[1] * raw.height, (n[2] - n[0]) * raw.width,
                (n[3] - n[1]) * raw.height)  # fmt: skip
    tag = f"{k}" if rec is None else f"r{rec}"
    try:
        mtime = Path(path).stat().st_mtime_ns
    except OSError:
        mtime = 0
    dest = folder / ".shotsieve" / "faces" / f"fc_{image_id}_{tag}_{size}_{mtime}.jpg"
    if dest.exists():
        return dest
    try:
        rgb = pcrop.preview_rgb(folder, image_id, path)
    except pcrop.CropError as exc:
        raise HTTPException(exc.status, str(exc)) from None
    ih, iw = rgb.shape[:2]
    x, y, w, h = rbox if rbox is not None else raw.faces[k].bbox
    sx, sy = iw / raw.width, ih / raw.height
    cx, cy = (x + w / 2) * sx, (y + h / 2) * sy
    half = max(w * sx, h * sy) * (1 + FACE_PAD) / 2  # a square crop around the face
    x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
    x1, y1 = int(min(iw, cx + half)), int(min(ih, cy + half))
    if x1 - x0 < 2 or y1 - y0 < 2:
        raise HTTPException(404, "face outside the picture")
    crop = rgb[y0:y1, x0:x1]
    s = size / max(crop.shape[:2])
    if s != 1.0:
        crop = cv2.resize(
            crop, (max(1, round(crop.shape[1] * s)), max(1, round(crop.shape[0] * s))),
            interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR,
        )
    ok, buf = cv2.imencode(
        ".jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 86]
    )
    if not ok:
        raise HTTPException(422, "cannot encode face")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{uuid.uuid4().hex}.tmp")
    tmp.write_bytes(buf.tobytes())
    os.replace(tmp, dest)
    return dest


def register(app: FastAPI, state: Any) -> None:
    from shotsieve.server.app import _path_of, validate_folder

    def consent(f: Path) -> bool:
        with Cache.for_folder(f) as c:
            return c.people_consent(state.settings.people_consent)

    def need(f: Path) -> None:
        if not consent(f):
            raise HTTPException(403, "people features are off; enable them with consent first")

    def k_of(c: Cache, path: str, k: int) -> int | None:
        """The embedding index (person_faces.face_idx) of face k, None if it has none."""
        raw = c.raw_features(path)
        if raw is None or not (0 <= k < len(raw.faces)) or not raw.faces[k].important:
            return None
        return sum(1 for f in raw.faces[:k] if f.important)

    @app.get("/api/image/{image_id}/faces")
    def get_faces(image_id: int, folder: str | None = None) -> dict[str, Any]:
        f = validate_folder(folder, state)
        path = _path_of(f, image_id)
        with Cache.for_folder(f) as c:
            ok = c.people_consent(state.settings.people_consent)
            return faces_payload(c, path, state.settings, ok)

    @app.get("/api/image/{image_id}/face/{k}")
    def get_face_thumb(
        image_id: int, k: int, size: int = Query(96, ge=24, le=512), folder: str | None = None
    ) -> FileResponse:
        f = validate_folder(folder, state)
        path = _path_of(f, image_id)
        with Cache.for_folder(f) as c:
            dest = _face_file(f, c, image_id, path, k, size)
        return FileResponse(dest, media_type="image/jpeg",
                            headers={"Cache-Control": "max-age=86400, immutable"})  # fmt: skip

    @app.get("/api/people/{person_id}/face")
    def get_person_face(
        person_id: int, size: int = Query(96, ge=24, le=512), folder: str | None = None
    ) -> FileResponse:
        f = validate_folder(folder, state)
        need(f)
        with Cache.for_folder(f) as c:
            with c._lock:
                row = c._con.execute(
                    "SELECT cover_face FROM people WHERE id=?", (person_id,)
                ).fetchone()
            if row is None:
                raise HTTPException(404, "unknown person")
            rel, _, j = row[0].rpartition("#")
            path = c._abs(rel)
            ids = c.ids_for([path])
            raw = c.raw_features(path)
            if raw is None or path not in ids:
                raise HTTPException(404, "unknown face")
            imp = [i for i, fc in enumerate(raw.faces) if fc.important]
            try:
                jj = int(j)
            except ValueError:
                raise HTTPException(404, "unknown face") from None
            if jj >= len(imp):  # a recognition-only face of the T23h scan
                dest = _face_file(f, c, ids[path], path, 0, size, rec=jj - len(imp))
            else:
                dest = _face_file(f, c, ids[path], path, imp[jj], size)
        return FileResponse(dest, media_type="image/jpeg", headers={"Cache-Control": "no-cache"})

    @app.get("/api/people/{person_id}/images")
    def get_person_images(person_id: int, folder: str | None = None) -> dict[str, Any]:
        f = validate_folder(folder, state)
        need(f)
        with Cache.for_folder(f) as c:
            return {"ids": sorted(ppl.image_ids_for_person_id(c, person_id))}

    @app.get("/api/people/chips")
    def get_chips(folder: str | None = None) -> dict[str, Any]:
        """{image id: [names]} of the named people on each image (grid name chips)."""
        f = validate_folder(folder, state)
        need(f)
        with Cache.for_folder(f) as c:
            with c._lock:
                rows = c._con.execute(
                    "SELECT DISTINCT i.id, p.name FROM person_faces f "
                    "JOIN people p ON p.id=f.person_id JOIN image_ids i ON i.path=f.path "
                    "WHERE p.name IS NOT NULL AND p.name != '' ORDER BY i.id, p.name"
                ).fetchall()
        out: dict[str, list[str]] = {}
        for iid, name in rows:
            out.setdefault(str(iid), []).append(name)
        return {"chips": out}

    @app.get("/api/people/names")
    def get_names(folder: str | None = None) -> dict[str, Any]:
        f = validate_folder(folder, state)
        need(f)
        with Cache.for_folder(f) as c:
            return {"names": ppl.names(c)}

    @app.post("/api/people/restore_ignored")
    def post_restore(req: FolderBody) -> dict[str, Any]:
        f = validate_folder(req.folder, state)
        need(f)
        with Cache.for_folder(f) as c:
            n = ppl.restore_ignored(c)
            return {"restored": n, **ppl.list_people(c)}

    @app.post("/api/people/unignore")
    def post_unignore(req: UnignoreBody) -> dict[str, Any]:
        f = validate_folder(req.folder, state)
        need(f)
        with Cache.for_folder(f) as c:
            n = ppl.unignore(c, [(str(p), int(i)) for p, i in req.faces])
            return {"restored": n, **ppl.list_people(c)}

    @app.post("/api/image/{image_id}/faces/{k}/detach")
    def post_detach(image_id: int, k: int, req: FolderBody) -> dict[str, Any]:
        f = validate_folder(req.folder, state)
        path = _path_of(f, image_id)
        need(f)
        with Cache.for_folder(f) as c:
            j = k_of(c, path, k)
            if j is None:
                raise HTTPException(422, "this face has no identity to detach")
            left = ppl.detach_face(c, path, j)
            return {"ok": True, "was": left, **ppl.list_people(c)}

    @app.post("/api/image/{image_id}/faces/{k}/assign")
    def post_assign(image_id: int, k: int, req: AssignRequest) -> dict[str, Any]:
        f = validate_folder(req.folder, state)
        path = _path_of(f, image_id)
        need(f)
        with Cache.for_folder(f) as c:
            j = k_of(c, path, k)
            if j is None:
                raise HTTPException(422, "this face is too small to be recognised")
            try:
                pid = ppl.assign_face(c, path, j, req.person_id, req.name)
            except KeyError:
                raise HTTPException(404, "unknown person") from None
            return {"ok": True, "person_id": pid, **ppl.list_people(c)}


__all__ = ["register", "faces_payload", "cache_path"]
