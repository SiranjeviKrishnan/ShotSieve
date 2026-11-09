"""Crop endpoints (T22, docs/11_CROP.md section 7), registered from `create_app`.

All take `folder` like the existing endpoints; ids are `image_id`; GETs never write user-visible
state (the `subjects` cache fill is derived data, like thumbnails); POSTs only from the local
Origin (enforced by the app middleware)."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from shotsieve.pipeline import crop as pcrop
from shotsieve.pipeline.cache import Cache, cache_path


class PointModel(BaseModel):
    x: float
    y: float


class BoxModel(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float

    def as_list(self) -> list[float]:
        return [self.x0, self.y0, self.x1, self.y1]


class BoxesRequest(BaseModel):
    folder: str | None = None
    boxes: list[BoxModel] = []


class SuggestRequest(BaseModel):
    folder: str | None = None
    subjects: list[str] = []
    points: list[PointModel] = []
    boxes: list[BoxModel] = []
    select_modes: dict[str, str] = {}
    ratio: str = "3:2"
    flip: bool = False
    n: int = 3


class PointRequest(BaseModel):
    folder: str | None = None
    x: float
    y: float


class AcceptRequest(BaseModel):
    folder: str | None = None
    rect: list[float]
    ratio: str
    source: str = "ai"
    subjects: list[str] = []
    alt_rank: int | None = None
    score: float | None = None
    write: str = "auto"  # auto | none | embed
    embed_copies: str | None = None
    catalog: str | None = None


class RejectRequest(BaseModel):
    folder: str | None = None
    alt_rank: int | None = None


class ResetRequest(BaseModel):
    folder: str | None = None
    write: str = "auto"
    embed_copies: str | None = None
    catalog: str | None = None


class RestoreRequest(ResetRequest):
    version: int


class BurstRequest(BaseModel):
    folder: str | None = None
    id: int
    ids: list[int] | None = None
    rect: list[float]
    ratio: str
    subjects: list[str] = []
    points: list[PointModel] = []
    select_modes: dict[str, str] = {}


class ManyItem(BaseModel):
    id: int
    rect: list[float]
    ratio: str


class AcceptManyRequest(BaseModel):
    folder: str | None = None
    items: list[ManyItem]
    write: str = "auto"
    embed_copies: str | None = None
    catalog: str | None = None


def burst_members(cache: Cache, image_id: int) -> list[tuple[int, str]]:
    """The other frames of the same burst as (image id, path)."""
    path = cache.path_for_id(image_id)
    if path is None:
        raise pcrop.CropError("unknown image", 404)
    rows = cache.load_decisions()
    row = rows.get(path)
    if row is None or row.get("burst_id") is None:
        raise pcrop.CropError("this frame is not part of a burst", 422)
    members = [p for p, r in rows.items() if r.get("burst_id") == row["burst_id"] and p != path]
    members = [p for p in sorted(members) if "#t=" not in p]
    ids = cache.ids_lookup(members)
    return [(ids[p], p) for p in members if p in ids]


def register(app: FastAPI, state: Any, start_job: Callable[[Any], str]) -> None:
    from shotsieve.server.app import Job, _path_of, validate_folder

    def folder_of(raw: str | None) -> Path:
        return validate_folder(raw, state)

    def guard(fn: Callable[[], Any]) -> Any:
        try:
            return fn()
        except pcrop.CropError as exc:
            raise HTTPException(exc.status, str(exc)) from None

    def copies(raw: str | None, folder: Path) -> str | None:
        if not raw:
            return None
        from shotsieve.server.app import _embed_arg

        return _embed_arg(raw, folder)  # type: ignore[return-value]

    def catalog(raw: str | None) -> str | None:
        """An explicit catalog is validated like T20's (a clean 400); None means "the shoot's
        remembered catalog" (resolved inside `pcrop`)."""
        if not raw:
            return None
        from shotsieve.io.lrcat import CatalogError, check_catalog_path

        try:
            return str(check_catalog_path(raw))
        except CatalogError as exc:
            raise HTTPException(400, str(exc)) from None

    def subjects_payload(sset: pcrop.SubjectSet) -> dict[str, Any]:
        return {
            "subjects": [pcrop._subject_json(s) for s in sset.subjects],
            "saliency_ready": sset.saliency is not None,
            "detector": sset.detector,
            "full": {"w": sset.fw, "h": sset.fh}, "orientation": sset.orientation,
            "peak": list(sset.peak) if sset.peak else None,
            "horizon": list(sset.horizon) if sset.horizon else None,
        }  # fmt: skip

    @app.get("/api/crop/{image_id}")
    def get_crop(image_id: int, folder: str | None = None) -> dict[str, Any]:
        f = folder_of(folder)
        path = _path_of(f, image_id)
        with Cache.for_folder(f) as c:
            return guard(lambda: pcrop.crop_state(f, c, image_id, path, state.settings))

    @app.get("/api/crop/{image_id}/subjects")
    def get_subjects(image_id: int, folder: str | None = None) -> dict[str, Any]:
        f = folder_of(folder)
        path = _path_of(f, image_id)
        with Cache.for_folder(f) as c:
            sset = guard(lambda: pcrop.detect_subjects(f, c, image_id, path, state.settings))
        return subjects_payload(sset)

    @app.post("/api/crop/{image_id}/point")
    def post_point(image_id: int, req: PointRequest) -> dict[str, Any]:
        f = folder_of(req.folder)
        path = _path_of(f, image_id)
        with Cache.for_folder(f) as c:
            s = guard(
                lambda: pcrop.point_subject(f, c, image_id, path, req.x, req.y, state.settings)
            )
        return {"subject": pcrop._subject_json(s)}

    @app.post("/api/crop/{image_id}/suggest")
    def post_suggest(image_id: int, req: SuggestRequest) -> dict[str, Any]:
        f = folder_of(req.folder)
        path = _path_of(f, image_id)
        with Cache.for_folder(f) as c:
            return guard(
                lambda: pcrop.suggest_crops(
                    f,
                    c,
                    image_id,
                    path,
                    state.settings,
                    subjects=req.subjects,
                    points=[(p.x, p.y) for p in req.points],
                    select_modes=req.select_modes,
                    ratio=req.ratio,
                    flip=req.flip,
                    n=req.n,
                    boxes=[b.as_list() for b in req.boxes],
                )
            )

    @app.get("/api/crop/{image_id}/boxes")
    def get_boxes(image_id: int, folder: str | None = None) -> dict[str, Any]:
        f = folder_of(folder)
        _path_of(f, image_id)
        with Cache.for_folder(f) as c:
            return {"boxes": c.user_boxes(image_id)}

    @app.post("/api/crop/{image_id}/boxes")
    def post_boxes(image_id: int, req: BoxesRequest) -> dict[str, Any]:
        f = folder_of(req.folder)
        path = _path_of(f, image_id)
        with Cache.for_folder(f) as c:
            return guard(
                lambda: pcrop.save_user_boxes(
                    f, c, image_id, path, [b.as_list() for b in req.boxes], state.settings
                )
            )

    @app.post("/api/crop/{image_id}/accept")
    def post_accept(image_id: int, req: AcceptRequest) -> dict[str, Any]:
        f = folder_of(req.folder)
        path = _path_of(f, image_id)
        emb = copies(req.embed_copies, f)
        with Cache.for_folder(f) as c:
            out = guard(
                lambda: pcrop.accept(
                    f,
                    c,
                    image_id,
                    path,
                    state.settings,
                    req.rect,
                    req.ratio,
                    source=req.source,
                    subjects=req.subjects,
                    alt_rank=req.alt_rank,
                    score=req.score,
                    write=req.write,
                    embed_dir=emb,
                    catalog=catalog(req.catalog),
                )
            )
        state.autoxmp.request(f, {path})
        return out

    @app.post("/api/crop/{image_id}/reject")
    def post_reject(image_id: int, req: RejectRequest) -> dict[str, Any]:
        f = folder_of(req.folder)
        _path_of(f, image_id)
        with Cache.for_folder(f) as c:
            return pcrop.reject(c, image_id, req.alt_rank)

    @app.post("/api/crop/{image_id}/reset")
    def post_reset(image_id: int, req: ResetRequest) -> dict[str, Any]:
        f = folder_of(req.folder)
        path = _path_of(f, image_id)
        emb = copies(req.embed_copies, f)
        with Cache.for_folder(f) as c:
            out = guard(
                lambda: pcrop.reset(
                    f,
                    c,
                    image_id,
                    path,
                    state.settings,
                    write=req.write,
                    embed_dir=emb,
                    catalog=catalog(req.catalog),
                )
            )
        state.autoxmp.request(f, {path})
        return out

    @app.post("/api/crop/{image_id}/restore")
    def post_restore(image_id: int, req: RestoreRequest) -> dict[str, Any]:
        f = folder_of(req.folder)
        path = _path_of(f, image_id)
        emb = copies(req.embed_copies, f)
        with Cache.for_folder(f) as c:
            out = guard(
                lambda: pcrop.restore(
                    f,
                    c,
                    image_id,
                    path,
                    req.version,
                    state.settings,
                    write=req.write,
                    embed_dir=emb,
                    catalog=catalog(req.catalog),
                )
            )
        state.autoxmp.request(f, {path})
        return out

    @app.post("/api/crop/burst")
    def post_burst(req: BurstRequest) -> dict[str, str]:
        f = folder_of(req.folder)
        if not cache_path(f).exists():
            raise HTTPException(404, "this folder has not been culled yet")
        path = _path_of(f, req.id)
        with Cache.for_folder(f) as c:
            members = guard(lambda: burst_members(c, req.id))
        if req.ids is not None:
            want = set(req.ids)
            members = [m for m in members if m[0] in want]
        if not members:
            raise HTTPException(422, "no other frames to transfer to")
        try:
            pcrop._check_rect(req.rect)
            pcrop.g.parse_ratio(req.ratio)
        except (pcrop.CropError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from None
        job = Job(
            uuid.uuid4().hex[:12], str(f), kind="crop_burst",
            payload={
                "id": req.id, "path": path, "members": members, "rect": req.rect,
                "ratio": req.ratio, "subjects": req.subjects,
                "points": [(p.x, p.y) for p in req.points], "select_modes": req.select_modes,
            },
        )  # fmt: skip
        return {"job_id": start_job(job)}

    @app.post("/api/crop/accept_many")
    def post_accept_many(req: AcceptManyRequest) -> dict[str, Any]:
        f = folder_of(req.folder)
        emb = copies(req.embed_copies, f)
        cat = catalog(req.catalog)
        out: list[dict[str, Any]] = []
        with Cache.for_folder(f) as c:
            for it in req.items:
                path = _path_of(f, it.id)
                try:
                    r = pcrop.accept(
                        f,
                        c,
                        it.id,
                        path,
                        state.settings,
                        it.rect,
                        it.ratio,
                        source="burst",
                        write=req.write,
                        embed_dir=emb,
                        catalog=cat,
                    )
                    out.append({"id": it.id, "ok": True, **r})
                except pcrop.CropError as exc:
                    out.append({"id": it.id, "ok": False, "error": str(exc)})
        return {"results": out}


def run_burst_job(job: Any, progress: Callable[[str, int, int], None], state: Any) -> None:
    """Job body for kind `crop_burst` (called from the app's `run_job`)."""
    p = job.payload
    try:
        with Cache.for_folder(job.folder) as c:
            items = pcrop.burst_transfer(
                job.folder,
                c,
                p["id"],
                p["path"],
                p["members"],
                p["rect"],
                p["ratio"],
                p["subjects"],
                p["select_modes"],
                p["points"],
                state.settings,
                progress,
            )
        job.result = {
            "source": p["id"], "ratio": p["ratio"],
            "results": [
                {"id": it.id, "rect": it.rect, "rect_px": it.rect_px, "status": it.status,
                 "warnings": it.warnings}
                for it in items
            ],
        }  # fmt: skip
        job.state = "done"
    except (pcrop.CropError, ValueError) as exc:
        job.error, job.state = str(exc), "error"
    except Exception as exc:  # noqa: BLE001 - report to the client
        from shotsieve.log import get_logger

        get_logger("server").exception("crop burst job failed")
        job.error, job.state = f"{type(exc).__name__}: {exc}", "error"
