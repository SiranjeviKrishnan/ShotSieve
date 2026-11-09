"""Editing-profile jobs (T15): train from a catalog / XMP folders, apply to a shoot.

Nothing here runs inside `cull`. `apply` predicts develop settings from cached embeddings plus
cheap statistics and writes `crs:` XMP: a sidecar for RAW, an embedded-XMP copy (in a separate
folder) for JPEG/DNG; originals are never modified.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from shotsieve.config import Settings
from shotsieve.features.raw_wb import as_shot
from shotsieve.io import crs, lrcat
from shotsieve.io.formats import EXCLUDED_DIRS, RAW_EXTS, split_virtual
from shotsieve.log import get_logger
from shotsieve.personal import edit_dataset
from shotsieve.personal.edit_dataset import TINT_PER_DUV, Feat, featurise, format_class
from shotsieve.personal.edit_profile import (
    FMT_DNG,
    TONE_KEYS,
    EditProfile,
    Prediction,
    default_profile_path,
)
from shotsieve.pipeline.cache import Cache

log = get_logger("edit")
Progress = Callable[[str, int, int], None]

LIGHTROOM_EMBED_NOTE = (
    "Lightroom Classic ignores XMP sidecars for JPEG and DNG files: no sidecar was written for "
    "them. Pass --embed-copies DIR (UI: 'embed copies') to write embedded-XMP copies into DIR for "
    "re-import; the originals are never modified."
)


class NoProfileError(RuntimeError):
    """No trained editing profile yet."""


def load_profile(settings: Settings | None = None) -> EditProfile:
    path = default_profile_path(settings)
    if not path.exists():
        raise NoProfileError("no editing profile yet: run `shotsieve edit train` first")
    return EditProfile.load(path)


# ---- train -----------------------------------------------------------------------------------


def train_profile(
    catalog: str | Path | None,
    xmp_dirs: list[str | Path] | None = None,
    settings: Settings | None = None,
    progress: Progress | None = None,
    *,
    dng_render: bool = False,
    out: str | Path | None = None,
) -> tuple[EditProfile, dict[str, Any]]:
    """Read edits, featurise, fit and save. Raises InsufficientEditsError under 200 edits."""
    settings = settings or Settings()
    t0 = time.time()
    S = edit_dataset.build_edit_set(
        catalog, xmp_dirs or [], settings, progress, dng_render=dng_render
    )
    prof = EditProfile().fit(S, progress)
    path = Path(out) if out else default_profile_path(settings)
    prof.save(path)
    info = {
        "n": len(S), "sources": S.sources, "notes": S.notes, "saved_to": str(path),
        "profile_hash": prof.profile_hash, "seconds": round(time.time() - t0, 1),
        "train_seconds": round(prof.meta.get("train_seconds", 0.0), 1),
        "carry_keys": len(S.carry), "sliders": prof.report(),
        "size_bytes": path.stat().st_size,
    }  # fmt: skip
    return prof, info


def profile_status(settings: Settings | None = None) -> dict[str, Any]:
    path = default_profile_path(settings)
    if not path.exists():
        return {"trained": False, "path": str(path)}
    try:
        prof = EditProfile.load(path)
    except Exception as exc:  # noqa: BLE001
        return {"trained": False, "path": str(path), "error": str(exc)}
    return {
        "trained": True, "path": str(path), "n": prof.n, "profile_hash": prof.profile_hash,
        "trained_at": prof.meta.get("trained_at"), "sources": prof.meta.get("sources", []),
        "sliders": prof.report(), "carry_keys": len(prof.carry),
        "n_formats": prof.meta.get("n_formats"), "size_bytes": path.stat().st_size,
    }  # fmt: skip


def delete_profile(settings: Settings | None = None) -> bool:
    path = default_profile_path(settings)
    if path.exists():
        path.unlink()
        return True
    return False


# ---- predict ---------------------------------------------------------------------------------


@dataclass
class ImageEdit:
    path: str
    values: dict[str, float] = field(default_factory=dict)  # crs key -> value (written ones)
    wb: bool = False
    confidence: dict[str, float] = field(default_factory=dict)
    point: dict[str, float] = field(default_factory=dict)  # crs key -> ungated prediction
    omitted: list[str] = field(default_factory=list)
    needs_edit: bool = False
    ood: bool = False
    notes: list[str] = field(default_factory=list)


def to_crs(
    pred: Prediction, fmt: int, wb: tuple[float, float] | None
) -> tuple[dict[str, float], dict[str, float], bool]:
    """(written crs values, ungated crs values, wb written) of a prediction.

    RAW/DNG white balance is as-shot-relative: Temperature K = 1e6 / (as-shot mired + delta),
    Tint = as-shot tint proxy + delta. JPEG uses IncrementalTemperature/Tint.
    """

    def convert(src: dict[str, float]) -> dict[str, float]:
        out = {k: src[k] for k in TONE_KEYS if k in src}
        if "wb_temp" in src or "wb_tint" in src:
            dt, dn = src.get("wb_temp"), src.get("wb_tint")
            if fmt <= FMT_DNG:
                if wb is not None:
                    mired = max(wb[0] + (dt or 0.0), 20.0)
                    out["Temperature"] = 1e6 / mired
                    out["Tint"] = TINT_PER_DUV * wb[1] + (dn or 0.0)
            else:
                if dt is not None:
                    out["IncrementalTemperature"] = dt
                if dn is not None:
                    out["IncrementalTint"] = dn
        return out

    written = {k: v for k, v in pred.point.items() if pred.written.get(k)}
    vals = convert(written)
    wb_written = any(k in vals for k in ("Temperature", "IncrementalTemperature"))
    return vals, convert(pred.point), wb_written


def _crs_conf(pred: Prediction, fmt: int) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in pred.conf.items():
        if k == "wb_temp":
            out["Temperature" if fmt <= FMT_DNG else "IncrementalTemperature"] = v
        elif k == "wb_tint":
            out["Tint" if fmt <= FMT_DNG else "IncrementalTint"] = v
        else:
            out[k] = v
    return out


def predict_paths(
    paths: list[str],
    profile: EditProfile,
    settings: Settings | None = None,
    progress: Progress | None = None,
    *,
    dng_render: bool = False,
) -> dict[str, ImageEdit]:
    """Predictions for the readable files among `paths`."""
    feats: dict[str, Feat] = featurise(paths, settings, progress, dng_render=dng_render)
    ok = [p for p in paths if p in feats]
    if not ok:
        return {}
    preds = profile.predict(
        np.vstack([feats[p].stats for p in ok]),
        np.vstack([feats[p].emb for p in ok]),
        np.array([feats[p].fmt for p in ok]),
    )
    out: dict[str, ImageEdit] = {}
    for p, pred in zip(ok, preds, strict=True):
        fmt = feats[p].fmt
        wb = as_shot(p) if fmt <= FMT_DNG and "wb_temp" in pred.point else None
        vals, point, wbw = to_crs(pred, fmt, wb)
        out[p] = ImageEdit(
            p, vals, wbw, _crs_conf(pred, fmt), point, list(pred.omitted), pred.needs_edit,
            pred.ood, list(pred.notes),
        )
    return out


# ---- apply -----------------------------------------------------------------------------------


def check_copy_root(copy_root: str | Path, folder: str | Path) -> None:
    """Refuse an `embed_copies` folder that is, or contains, a culled shoot (it would write over
    that shoot's originals); a ValueError says why."""
    root = Path(os.path.abspath(copy_root))
    folder = Path(os.path.abspath(folder))
    if root == folder or folder.is_relative_to(root):
        raise ValueError("the copy folder must not be the shoot folder or contain it")
    inside = root.relative_to(folder).parts if root.is_relative_to(folder) else ()
    if inside and not any(p in EXCLUDED_DIRS for p in inside):
        # inside the shoot, only a folder the scanner skips is safe (it would be re-culled)
        raise ValueError("a copy folder inside the shoot must be 'shotsieve_edits' (or sit in it)")
    seen = 0
    for dirpath, dirnames, _ in os.walk(root):
        if ".shotsieve" in dirnames and Path(dirpath, ".shotsieve", "cull.db").exists():
            raise ValueError(f"the copy folder holds a culled shoot ({dirpath}); pick another")
        depth = Path(dirpath).relative_to(root).parts
        seen += 1
        dirnames[:] = [] if len(depth) >= 3 or seen > 5000 else [
            d for d in dirnames if not d.startswith(".")
        ]


def _mode_for(path: str, embed: bool) -> str:
    ext = Path(path).suffix.lower()
    if ext in RAW_EXTS and ext != ".dng":
        return "sidecar"
    if ext in (".jpg", ".jpeg", ".dng"):
        return "embed" if embed else "report"
    return "unsupported"


def _final(row: dict, ovr: dict | None) -> str:
    return ovr["decision"] if ovr else row["decision"]


def apply_folder(
    folder: str | Path,
    settings: Settings | None = None,
    *,
    scope: str = "keeps",
    ids: list[int] | None = None,
    dry_run: bool = False,
    embed_copies: str | Path | bool | None = None,
    catalog: str | Path | None = None,
    progress: Progress | None = None,
    max_diff: int = 500,
) -> dict[str, Any]:
    """Predict and write develop settings for a culled shoot (see module docstring)."""
    settings = settings or Settings()
    folder = Path(os.path.abspath(folder))
    profile = load_profile(settings)
    with Cache.for_folder(folder) as c:
        decisions = c.load_decisions()
        if not decisions:
            raise ValueError("this folder has not been culled yet")
        overrides = c.overrides()
        by_id = c.ids_for(list(decisions))
    paths = [p for p in sorted(decisions) if not split_virtual(p)]
    if ids is not None:
        want = set(ids)
        paths = [p for p in paths if by_id.get(p) in want]
    elif scope == "keeps":
        paths = [p for p in paths if _final(decisions[p], overrides.get(p)) == "keep"]
    elif scope != "all":
        raise ValueError("scope must be 'keeps' or 'all'")
    embed = bool(embed_copies)
    copy_root = (
        Path(embed_copies) if isinstance(embed_copies, str | Path) and str(embed_copies)
        else folder / "shotsieve_edits"
    )
    if embed:
        check_copy_root(copy_root, folder)
    cat_paths: set[str] = set()
    if catalog:
        cat_paths = {e.path for e in lrcat.read_develop(catalog)}
    edits = predict_paths(paths, profile, settings, progress)

    summary: dict[str, Any] = {
        "n": len(paths), "predicted": len(edits), "undecodable": len(paths) - len(edits),
        "written_sidecar": 0, "written_copies": 0, "report_only": 0, "needs_edit": 0,
        "skipped": {}, "sliders": {}, "dry_run": dry_run, "profile_hash": profile.profile_hash,
        "warnings": [], "diff": [],
    }  # fmt: skip
    rows: list[dict[str, Any]] = []
    n_done = 0
    for p in paths:
        e = edits.get(p)
        if e is None:
            continue
        n_done += 1
        if progress:
            progress("write", n_done, len(edits))
        summary["needs_edit"] += e.needs_edit
        mode = _mode_for(p, embed)
        status = "written"
        if not e.values:
            status = "nothing_to_write"
        elif mode == "report":
            status = "report_only"
            summary["report_only"] += 1
        elif mode == "unsupported":
            status = "unsupported"
        else:
            dst = None
            if mode == "embed":
                rel = Path(os.path.relpath(Path(p).parent, folder))
                dst = copy_root / rel / Path(p).name
            ok, why = crs.writable(
                p, "sidecar" if mode == "sidecar" else "embed", p in cat_paths, dst
            )
            if not ok:
                status = why
        if status == "written" and not dry_run:
            kw: dict[str, Any] = dict(
                carry=profile.carry, profile_hash=profile.profile_hash,
                confidence=e.confidence, wb=e.wb, catalog_has=p in cat_paths,
            )  # fmt: skip
            if mode == "embed":
                rel_parent = Path(os.path.relpath(Path(p).parent, folder))
                kw.update(mode="embed", out_dir=copy_root / rel_parent)
            res = crs.write_develop(p, e.values, **kw)
            status = res.status if not res.written else "written"
        if status == "written":
            summary["written_sidecar" if mode == "sidecar" else "written_copies"] += 1
            for k in e.values:
                summary["sliders"][k] = summary["sliders"].get(k, 0) + 1
        elif status not in ("report_only", "nothing_to_write"):
            summary["skipped"][status] = summary["skipped"].get(status, 0) + 1
        written_mode = {"sidecar": "sidecar", "embed": "copy"}.get(mode, "report")
        if status != "written":
            written_mode = status if status != "report_only" else "report"
        rows.append(
            {
                "path": p, "values": e.values,
                "confidence": {"conf": e.confidence, "omitted": e.omitted,
                               "needs_edit": e.needs_edit, "point": e.point},
                "written_mode": written_mode, "profile_hash": profile.profile_hash,
            }  # fmt: skip
        )
        if len(summary["diff"]) < max_diff:
            summary["diff"].append(
                {"name": Path(p).name, "status": status, "values": e.values,
                 "omitted": e.omitted, "needs_edit": e.needs_edit, "notes": e.notes}  # fmt: skip
            )
    if summary["report_only"]:
        summary["warnings"].append(LIGHTROOM_EMBED_NOTE)
    if not dry_run and rows:
        with Cache.for_folder(folder) as c:
            c.put_edits(rows)
    return summary


def predictions_for(
    folder: str | Path, ids: list[int], settings: Settings | None = None, max_live: int = 1
) -> dict[int, dict[str, Any]]:
    """Stored edits of images (by image id), plus a fresh prediction for at most `max_live` that
    were never applied (uncached featurisation is slow; the loupe asks for one id)."""
    settings = settings or Settings()
    folder = Path(os.path.abspath(folder))
    out: dict[int, dict[str, Any]] = {}
    with Cache.for_folder(folder) as c:
        paths = {i: c.path_for_id(i) for i in ids}
        stored = c.get_edits([p for p in paths.values() if p])
    missing = [p for p in paths.values() if p and p not in stored and not split_virtual(p)]
    missing = missing[: max(max_live, 0)]
    live: dict[str, ImageEdit] = {}
    if missing:
        try:
            live = predict_paths(missing, load_profile(settings), settings)
        except NoProfileError:
            live = {}
    for i, p in paths.items():
        if not p:
            continue
        if p in stored:
            s = stored[p]
            conf = s["confidence"] or {}
            out[i] = {
                "values": s["values"], "confidence": conf.get("conf", {}),
                "omitted": conf.get("omitted", []), "needs_edit": conf.get("needs_edit", False),
                "written_mode": s["written_mode"], "profile_hash": s["profile_hash"],
                "applied": True,
            }  # fmt: skip
        elif p in live:
            e = live[p]
            out[i] = {
                "values": e.values, "confidence": e.confidence, "omitted": e.omitted,
                "needs_edit": e.needs_edit, "written_mode": None, "profile_hash": None,
                "applied": False,
            }  # fmt: skip
    return out


__all__ = [
    "LIGHTROOM_EMBED_NOTE", "NoProfileError", "apply_folder", "check_copy_root", "delete_profile",
    "format_class", "load_profile", "predict_paths", "predictions_for", "profile_status",
    "train_profile",
]  # fmt: skip
