# ruff: noqa: E501
"""Move to Trash (T23i), registered from `create_app`: POST /api/trash/preview, /api/trash and
/api/trash/undo.

Safety rules (each has a test):
- RECOVERABLE only. Files go to the macOS Trash through NSFileManager
  `trashItemAtURL:resultingItemURL:error:` (Finder's "Put Back" works). There is no unlink, no
  `rmtree`, no `shutil.rmtree`, and NO fallback to a permanent delete: when the Trash refuses a
  file (a volume without a Trash, a network drive, a non-Mac host) that file is skipped and reported.
- Only files strictly inside the shoot folder: no `..`, no symlinks (a link could point out of the
  folder), the real path must stay under the real shoot root, nothing under `.shotsieve/`.
- Per photo: the master, its XMP sidecar, and (checkbox, default on) the paired JPEG/RAW with the
  same stem. A video frame has no file of its own: nothing is trashed for it unless the caller
  explicitly asks for the video file (`include_videos`).
- The shoot is brought up to date through the existing incremental cull (the files are gone from
  disk, so `run_cull` sees them as `removed`; the frozen norm / affected-set logic of T21 decides
  what is re-decided). Overrides, colour labels and image ids are kept in their tables, so an undo
  (files moved back, same incremental path, the new files are "new") brings the photos back with
  their ids, stars, labels and picks.
- Tests inject a backend (`state.trash_backend`, or SHOTSIEVE_FAKE_TRASH=<dir> for a server
  subprocess: files go to that directory instead of the user's Trash, still recoverable)."""

from __future__ import annotations

import os
import shutil
import threading
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from shotsieve.io.formats import is_raw, preview_source_for, split_virtual
from shotsieve.io.xmp import sidecar_path
from shotsieve.log import get_logger
from shotsieve.pipeline.cache import Cache

log = get_logger("trash")
MAX_IDS = 5000
MAX_UNDO = 20  # undo records kept per server (until the shoot is closed / the app quits)


class TrashError(OSError):
    pass


class MacTrash:
    """The real thing: NSFileManager (pyobjc comes with pywebview). Never deletes."""

    def _fm(self) -> Any:
        try:
            from Foundation import NSURL, NSFileManager  # type: ignore[import-not-found]
        except ImportError as exc:
            raise TrashError("Move to Trash needs macOS (Foundation is not available)") from exc
        return NSFileManager.defaultManager(), NSURL

    def trash(self, path: Path) -> Path:
        fm, url_cls = self._fm()
        ok, result, err = fm.trashItemAtURL_resultingItemURL_error_(
            url_cls.fileURLWithPath_(str(path)), None, None
        )
        if not ok:
            raise TrashError(str(err.localizedDescription()) if err else "the Trash refused it")
        return Path(str(result.path())) if result is not None else Path()

    def restore(self, src: Path, dst: Path) -> None:
        fm, url_cls = self._fm()
        ok, err = fm.moveItemAtURL_toURL_error_(
            url_cls.fileURLWithPath_(str(src)), url_cls.fileURLWithPath_(str(dst)), None
        )
        if not ok:
            raise TrashError(str(err.localizedDescription()) if err else "could not put it back")


class DirTrash:
    """Test double: moves into a directory (still recoverable, never leaves the machine)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def trash(self, path: Path) -> Path:
        dest_dir = self.root / uuid.uuid4().hex[:8]
        dest_dir.mkdir()
        dest = dest_dir / path.name
        shutil.move(str(path), str(dest))  # a move, never a delete
        return dest

    def restore(self, src: Path, dst: Path) -> None:
        shutil.move(str(src), str(dst))


def default_backend() -> Any:
    fake = os.environ.get("SHOTSIEVE_FAKE_TRASH")
    return DirTrash(fake) if fake else MacTrash()


# ---- planning ------------------------------------------------------------------------------


class TrashRequest(BaseModel):
    folder: str | None = None
    ids: list[int]
    include_pairs: bool = True
    include_videos: bool = False  # only when the user explicitly picks the video file


class UndoRequest(BaseModel):
    folder: str | None = None
    token: str


def _ids(ids: list[int]) -> list[int]:
    if not 1 <= len(ids) <= MAX_IDS:
        raise HTTPException(422, f"give 1 to {MAX_IDS} image ids")
    return list(dict.fromkeys(ids))


def check_inside(root: Path, p: Path) -> str | None:
    """None when `p` is a regular file strictly inside `root` (real paths, no symlink); else why not."""
    try:
        rel = Path(os.path.abspath(p)).relative_to(os.path.abspath(root))
    except ValueError:
        return "outside the shoot folder"
    if ".." in rel.parts or ".shotsieve" in rel.parts:
        return "ShotSieve's own folder is never touched"
    if os.path.islink(p):
        return "a symbolic link is not moved"
    # a symlinked directory on the way could lead out of the folder
    try:
        real_root, real = Path(os.path.realpath(root)), Path(os.path.realpath(p))
    except OSError as exc:
        return str(exc)
    if not real.is_relative_to(real_root) or real != real_root / rel:
        return "outside the shoot folder (symbolic link)"
    return None


def _photo_files(
    root: Path, path: str, include_pairs: bool, include_videos: bool
) -> dict[str, Any]:
    """{"files": [(kind, Path)], "skip": reason | None} for one image row."""
    sv = split_virtual(path)
    if sv is not None:
        if include_videos:
            return {"files": [("video", sv[0])], "skip": None}
        return {"files": [], "skip": "Video frames can't be trashed individually"}
    master = Path(path)
    files: list[tuple[str, Path]] = [("master", master)]
    pair = preview_source_for(master) if is_raw(master) else master
    has_pair = pair != master
    if has_pair and include_pairs:
        files.append(("pair", pair))
    side = sidecar_path(master)
    # the sidecar `stem.xmp` is shared by the RAW and its JPEG: leave it while the other stays
    if side.is_file() and (include_pairs or not has_pair):
        files.append(("sidecar", side))
    return {"files": files, "skip": None, "has_pair": has_pair}


def _resolve(root: Path, c: Cache, ids: list[int]) -> list[tuple[int, str]]:
    out = []
    for i in ids:
        path = c.path_for_id(i)
        if path is None:
            raise HTTPException(404, f"unknown image {i}")
        real = split_virtual(path)[0] if split_virtual(path) else Path(path)
        if check_inside(root, real) == "outside the shoot folder":
            raise HTTPException(404, f"unknown image {i}")
        out.append((i, path))
    return out


def register(app: FastAPI, state: Any, start_job: Callable[[Any], str]) -> None:
    from shotsieve.pipeline.cache import cache_path
    from shotsieve.server.app import Job, validate_folder

    state.trash_backend = default_backend()
    state.trash_undo = {}  # token -> record (insertion ordered)
    lock = threading.Lock()

    def need_free(f: Path) -> None:
        with state.lock:
            if state._cull_running(f) or f in state.calibrating:
                raise HTTPException(409, "the shoot is being updated; try again in a moment")

    def redecide(f: Path) -> str | None:
        """The incremental cull that drops / re-adds the rows (the files are gone / back)."""
        with Cache.for_folder(f) as c:
            video = any(split_virtual(p) for p in c.load_decisions())
        job = Job(uuid.uuid4().hex[:12], str(f), video=video)
        try:
            return start_job(job)
        except HTTPException as exc:
            log.warning("trash: could not start the incremental cull: %s", exc.detail)
            return None

    def open_cache(f: Path) -> None:
        if not cache_path(f).exists():
            raise HTTPException(404, "this folder has not been culled yet")

    @app.post("/api/trash/preview")
    def preview(req: TrashRequest) -> dict[str, Any]:
        f = validate_folder(req.folder, state)
        open_cache(f)
        ids = _ids(req.ids)
        n_files = n_pairs = n_video = n_unsafe = 0
        names: list[str] = []
        with Cache.for_folder(f) as c:
            rows = _resolve(f, c, ids)
        for _i, path in rows:
            plan = _photo_files(f, path, req.include_pairs, req.include_videos)
            names.append(Path(path).name)
            if plan["skip"]:
                n_video += 1
                continue
            if any(check_inside(f, p) for _k, p in plan["files"]):
                n_unsafe += 1
                continue
            n_files += len([1 for _k, p in plan["files"] if p.exists()])
            n_pairs += bool(plan.get("has_pair"))
        return {"n": len(rows), "names": names[:6], "n_files": n_files, "n_pairs": n_pairs,
                "n_video_frames": n_video, "n_unsafe": n_unsafe}  # fmt: skip

    @app.post("/api/trash")
    def post_trash(req: TrashRequest) -> dict[str, Any]:
        f = validate_folder(req.folder, state)
        open_cache(f)
        ids = _ids(req.ids)
        backend = state.trash_backend
        with lock:
            need_free(f)
            with Cache.for_folder(f) as c:
                rows = _resolve(f, c, ids)  # any unknown id: nothing is touched
            files_out: list[dict[str, Any]] = []
            trashed: list[int] = []
            failed: list[dict[str, Any]] = []
            skipped: list[dict[str, Any]] = []
            moved: list[dict[str, str]] = []
            done: set[str] = set()
            for iid, path in rows:
                name = Path(path).name
                plan = _photo_files(f, path, req.include_pairs, req.include_videos)
                if plan["skip"]:
                    skipped.append({"id": iid, "name": name, "reason": plan["skip"]})
                    continue
                ok_master = False
                for kind, p in plan["files"]:
                    key = str(p)
                    rel = os.path.relpath(p, f)
                    if key in done:  # a file already moved for another selected photo
                        ok_master = ok_master or kind in ("master", "video")
                        continue
                    why = check_inside(f, p)
                    if why is None and not os.path.lexists(p):
                        files_out.append({"id": iid, "rel": rel, "kind": kind, "status": "missing"})
                        ok_master = ok_master or kind in ("master", "video")  # already gone
                        continue
                    if why is None:
                        try:
                            dest = backend.trash(p)
                        except Exception as exc:  # noqa: BLE001 - skip it, never delete it
                            why = f"{type(exc).__name__}: {exc}"
                        else:
                            done.add(key)
                            moved.append({"orig": key, "trashed": str(dest)})
                            files_out.append({"id": iid, "rel": rel, "kind": kind, "status": "trashed"})
                            ok_master = ok_master or kind in ("master", "video")
                            log.info("trash: %s -> %s", key, dest)
                            continue
                    files_out.append({"id": iid, "rel": rel, "kind": kind, "status": "failed",
                                      "error": why})  # fmt: skip
                    log.warning("trash: not moved %s: %s", key, why)
                    if kind in ("master", "video"):
                        failed.append({"id": iid, "name": name, "error": why})
                        break  # no companions of a photo whose master stayed
                if ok_master:
                    trashed.append(iid)
            token = job_id = None
            if moved:
                token = uuid.uuid4().hex
                state.trash_undo[token] = {"folder": str(f), "moved": moved, "ids": trashed}
                while len(state.trash_undo) > MAX_UNDO:
                    state.trash_undo.pop(next(iter(state.trash_undo)))
                job_id = redecide(f)
            log.info("trash: %d photo(s), %d file(s) moved, %d failed, %d skipped (token %s)",
                     len(trashed), len(moved), len(failed), len(skipped), token)  # fmt: skip
        return {"ok": True, "token": token, "job_id": job_id, "trashed": trashed, "failed": failed,
                "skipped": skipped, "files": files_out, "n_files": len(moved)}  # fmt: skip

    @app.post("/api/trash/undo")
    def post_undo(req: UndoRequest) -> dict[str, Any]:
        f = validate_folder(req.folder, state)
        backend = state.trash_backend
        with lock:
            rec = state.trash_undo.get(req.token)
            if rec is None or rec["folder"] != str(f):
                raise HTTPException(404, "nothing to undo for that token")
            need_free(f)
            restored = 0
            failed: list[dict[str, str]] = []
            left: list[dict[str, str]] = []
            for m in rec["moved"]:
                orig = Path(m["orig"])
                if os.path.lexists(orig):
                    why = "a file with that name is back already"  # never overwrite
                else:
                    why = check_inside(f, orig)
                if why is None:
                    try:
                        orig.parent.mkdir(parents=True, exist_ok=True)
                        backend.restore(Path(m["trashed"]), orig)
                        restored += 1
                        log.info("trash undo: %s <- %s", orig, m["trashed"])
                        continue
                    except Exception as exc:  # noqa: BLE001
                        why = f"{type(exc).__name__}: {exc}"
                failed.append({"rel": os.path.relpath(orig, f), "error": why})
                left.append(m)
            if left:
                rec["moved"] = left
            else:
                state.trash_undo.pop(req.token, None)
            job_id = redecide(f) if restored else None
            log.info("trash undo: %d restored, %d failed (token %s)", restored, len(failed), req.token)
        return {"ok": True, "restored": restored, "failed": failed, "ids": rec["ids"],
                "job_id": job_id}  # fmt: skip
