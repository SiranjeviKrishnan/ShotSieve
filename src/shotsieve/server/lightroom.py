"""Open photos in Adobe Lightroom (T23d), registered from `create_app`.

Lightroom (the cloud app, bundle id `com.adobe.lightroomCC`) has no .lrcat catalogue and reads XMP
sidecars only when it imports a file, so "open in Lightroom" = make sure the sidecar carries
ShotSieve's state, then `open -b <bundle> <files>` (Lightroom shows its Add Photos import).
Lightroom Classic (`com.adobe.LightroomClassicCC7`) is offered when installed. Originals are never
written; the only files written are `.xmp` sidecars, through the same writers (and the same
user-wins / consent / .bak rules) as `Write XMP` and the crop accept."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from shotsieve.io.formats import split_virtual
from shotsieve.log import get_logger
from shotsieve.pipeline.cache import Cache, cache_path

log = get_logger("lightroom")

MAX_FILES = 500
APPS = {
    "cloud": {"name": "Lightroom", "bundle_ids": ["com.adobe.lightroomCC"]},
    "classic": {
        "name": "Lightroom Classic",
        "bundle_ids": ["com.adobe.LightroomClassicCC7", "com.adobe.LightroomClassicCC"],
    },
}
OPEN_TIMEOUT = 30


class OpenRequest(BaseModel):
    folder: str | None = None
    ids: list[int]
    app: str | None = None  # cloud | classic (default: the one used last, else what is installed)
    xmp: str | None = None  # write | skip: answer to the one-time sheet (remembered)
    remember: bool = True


def _fake_apps() -> set[str] | None:
    """Test seam: `SHOTSIEVE_FAKE_LR_APPS=cloud,classic` pretends those are installed and `open` is
    not run (the UI tests drive the real server without launching Lightroom)."""
    v = os.environ.get("SHOTSIEVE_FAKE_LR_APPS")
    return None if v is None else {x.strip() for x in v.split(",") if x.strip()}


def find_app(bundle_id: str) -> str | None:
    """Path of the installed app with this bundle id (NSWorkspace, else Spotlight), or None."""
    fake = _fake_apps()
    if fake is not None:
        hit = any(bundle_id in APPS[k]["bundle_ids"] for k in fake if k in APPS)
        return "/Applications/Fake Lightroom.app" if hit else None
    try:
        from AppKit import NSWorkspace  # type: ignore[import-not-found]

        url = NSWorkspace.sharedWorkspace().URLForApplicationWithBundleIdentifier_(bundle_id)
        if url is not None:
            return str(url.path())
        return None
    except Exception:  # noqa: BLE001 - no pyobjc (tests, other platforms): ask Spotlight
        pass
    try:
        r = subprocess.run(
            ["mdfind", f"kMDItemCFBundleIdentifier == '{bundle_id}'"],
            capture_output=True, text=True, timeout=5, check=False,
        )  # fmt: skip
        line = next((x for x in r.stdout.splitlines() if x.endswith(".app")), "")
        return line or None
    except (OSError, subprocess.SubprocessError):
        return None


def installed_apps() -> list[dict[str, Any]]:
    out = []
    for key, spec in APPS.items():
        for bid in spec["bundle_ids"]:
            path = find_app(bid)
            if path:
                out.append({"id": key, "name": spec["name"], "bundle_id": bid, "path": path})
                break
    return out


def prepare_xmp(
    folder: Path, ids: list[int], paths: dict[int, str], settings: Any
) -> dict[str, Any]:
    """Bring the sidecars of `paths` up to date: decision (rating / pick / label), the current
    crop. Edit settings that were already written stay in the sidecar (nothing is predicted or
    re-applied here). Every writer applies its own safety rules."""
    from shotsieve.pipeline import crop as pcrop
    from shotsieve.pipeline.crop_geom import Rect
    from shotsieve.server.app import write_xmp_sidecars

    res = write_xmp_sidecars(folder, settings, False, only=set(paths.values()))
    crops = {"written": 0, "skipped": {}, "failed": 0}
    with Cache.for_folder(folder) as c:
        for iid, path in paths.items():
            hist = c.crop_history(iid)
            cur = hist[-1] if hist else None
            if cur is None or not cur.get("rect"):
                continue
            try:
                info = pcrop.info_for(c, path, (c.get_subjects(path) or {}).get("data"))
                out = pcrop.write_xmp(
                    folder, iid, path, Rect.of(cur["rect"]), cur["version"], info, settings,
                    write="auto", catalog=pcrop.resolve_catalog(c, None),
                )  # fmt: skip
            except Exception as exc:  # noqa: BLE001 - one odd file must not stop the rest
                log.warning("crop write for Lightroom failed for %s: %s", path, exc)
                crops["failed"] += 1
                continue
            if out["status"] in ("written", "replaced_own"):
                crops["written"] += 1
            else:
                crops["skipped"][out["status"]] = crops["skipped"].get(out["status"], 0) + 1
        edits = c.get_edits(list(paths.values()))
    return {
        "decisions": res["written"], "failed": res["failed"], "crops": crops,
        "kept_user_rating": res["kept_user_rating"], "kept_user_label": res["kept_user_label"],
        "kept_lr": res["kept_lr"],
        "edits_in_sidecar": sum(1 for e in edits.values() if e.get("written_mode") == "sidecar"),
        "warnings": res["warnings"],
    }  # fmt: skip


def register(app: FastAPI, state: Any) -> None:
    from shotsieve.server.app import validate_folder

    def meta(f: Path, key: str) -> Any:
        if not cache_path(f).exists():
            return None
        with Cache.for_folder(f) as c:
            return c.get_meta(key)

    @app.get("/api/lightroom/apps")
    def get_apps(folder: str | None = None) -> dict[str, Any]:
        f = validate_folder(folder, state)
        apps = installed_apps()
        ids = {a["id"] for a in apps}
        pref = meta(f, "lr_app")
        if pref not in ids:
            pref = "cloud" if "cloud" in ids else (apps[0]["id"] if apps else None)
        return {
            "apps": apps, "preferred": pref, "xmp_choice": meta(f, "lr_xmp_choice"),
            "classic_sync": "classic" in ids or bool(meta(f, "lr_catalog")),
        }

    @app.post("/api/lightroom/open")
    def open_in_lightroom(req: OpenRequest) -> dict[str, Any]:
        f = validate_folder(req.folder, state)
        if not req.ids:
            raise HTTPException(422, "give at least one image id")
        if len(req.ids) > MAX_FILES:
            raise HTTPException(422, f"at most {MAX_FILES} files per call")
        if req.app not in (None, "cloud", "classic"):
            raise HTTPException(422, "app must be 'cloud' or 'classic'")
        if req.xmp not in (None, "write", "skip"):
            raise HTTPException(422, "xmp must be 'write' or 'skip'")
        if not cache_path(f).exists():
            raise HTTPException(404, "this folder has not been culled yet")
        root = Path(f).resolve()
        files: list[str] = []
        by_id: dict[int, str] = {}
        missing: list[int] = []
        with Cache.for_folder(f) as c:
            for iid in dict.fromkeys(req.ids):
                path = c.path_for_id(iid)
                if path is None:
                    raise HTTPException(404, f"unknown image {iid}")
                sv = split_virtual(path)
                master = Path(sv[0] if sv else path)
                try:
                    inside = master.resolve().is_relative_to(root)
                except OSError:
                    inside = False
                if not inside:
                    raise HTTPException(404, f"unknown image {iid}")
                if not master.is_file():
                    missing.append(iid)
                    continue
                if str(master) not in files:
                    files.append(str(master))
                if sv is None:
                    by_id[iid] = path
        if not files:
            raise HTTPException(404, "the files are missing from the folder")
        apps = {a["id"]: a for a in installed_apps()}
        if not apps:
            raise HTTPException(409, "Adobe Lightroom is not installed")
        want = req.app or meta(f, "lr_app")
        if want not in apps:
            if req.app:
                raise HTTPException(409, f"{APPS[req.app]['name']} is not installed")
            want = "cloud" if "cloud" in apps else next(iter(apps))
        chosen = apps[want]
        auto = state.autoxmp.enabled(f)  # T23g: standing consent; no question any more
        choice = req.xmp or ("write" if auto else "skip")
        xmp: dict[str, Any] | None = None
        if choice == "write" and by_id:
            if auto and not req.xmp:  # the sidecars are kept current in the background: flush
                state.autoxmp.request(f, set(by_id.values()))
                state.autoxmp.flush(f)
            else:
                xmp = prepare_xmp(f, list(by_id), by_id, state.settings)
        if _fake_apps() is not None:  # test seam: nothing is launched
            r = subprocess.CompletedProcess([], 0, "", "")
        else:
            try:
                r = subprocess.run(
                    ["open", "-b", chosen["bundle_id"], *files],
                    capture_output=True, text=True, timeout=OPEN_TIMEOUT, check=False,
                )  # fmt: skip
            except (OSError, subprocess.SubprocessError) as exc:
                raise HTTPException(502, f"could not start {chosen['name']}: {exc}") from None
        if r.returncode != 0:
            raise HTTPException(
                502, f"{chosen['name']} did not open the files: {(r.stderr or '').strip()[:200]}"
            )
        with Cache.for_folder(f) as c:
            c.set_meta("lr_app", want)
        return {
            "ok": True, "app": want, "name": chosen["name"], "bundle_id": chosen["bundle_id"],
            "n": len(files), "missing": missing, "xmp": xmp, "xmp_choice": choice,
        }  # fmt: skip
