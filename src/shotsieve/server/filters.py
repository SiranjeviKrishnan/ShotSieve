"""Filter bar data (T23e), registered from `create_app`: colour labels, the camera / lens facets
and the person -> images map. Filtering itself is client-side (static/filters.js) over the
`GET /api/shoot` payload plus these small side tables; nothing here touches the cull's decisions,
features or cache version."""

from __future__ import annotations

import os
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from shotsieve.io.exif import _LENS_VER, read_lens
from shotsieve.io.formats import split_virtual
from shotsieve.io.xmp import LABEL_XMP, label_key, read_sidecar, sidecar_path, user_values
from shotsieve.log import get_logger
from shotsieve.pipeline.cache import Cache

log = get_logger("filters")
SQUARE_TOL = 0.02  # |w - h| / max(w, h) below this is "square"
_SCANS: dict[str, threading.Thread] = {}
_SCANS_GUARD = threading.Lock()


class LabelRequest(BaseModel):
    folder: str | None = None
    image_ids: list[int]
    label: str  # red | yellow | green | blue | purple | none


class FolderBody(BaseModel):
    folder: str | None = None


def _norm(v: str) -> str:
    return " ".join(v.split()).casefold()


def _vals(raw: list[str], is_make: bool = False) -> tuple[list[dict[str, Any]], list[int]]:
    """Distinct values (case / spacing-insensitive) and the index of each row's value (-1 =
    missing). The label is the commonest spelling (a make written all upper-case is title-cased);
    every original spelling is kept in `raw` for the tooltip."""
    groups: dict[str, Counter] = {}
    for v in raw:
        if v:
            groups.setdefault(_norm(v), Counter())[" ".join(v.split())] += 1
    keys = sorted(groups, key=lambda k: (-sum(groups[k].values()), k))
    index = {k: i for i, k in enumerate(keys)}
    out = []
    for k in keys:
        spellings = groups[k]
        label = spellings.most_common(1)[0][0]
        if is_make and label.isupper() and len(label) > 3:
            label = label.title()
        out.append({"label": label, "raw": sorted(spellings), "n": sum(spellings.values())})
    return out, [index[_norm(v)] if v else -1 for v in raw]


def _orient(w: int, h: int) -> int:
    """0 landscape, 1 portrait, 2 square, -1 unknown (upright preview frame)."""
    if w <= 0 or h <= 0:
        return -1
    if abs(w - h) / max(w, h) < SQUARE_TOL:
        return 2
    return 0 if w > h else 1


def facets_payload(f: Path) -> dict[str, Any]:
    with Cache.for_folder(f) as c:
        rows = c.facet_rows()
        extra = c.exif_extra_rows()
        user = c.labels()
        ids = c.ids_lookup([r[0] for r in rows])
    rows = [r for r in rows if r[0] in ids]
    makes, mk = _vals([r[1] for r in rows], is_make=True)
    models, md = _vals([r[2] for r in rows])
    lens, ln = _vals([(extra.get(r[0]) or {}).get("lens_model", "") for r in rows])
    serial, sn = _vals([(extra.get(r[0]) or {}).get("lens_serial", "") for r in rows])
    labels: dict[str, str] = {}
    for r in rows:  # the user's choice (a key, "" = cleared) wins over the sidecar's label
        key = user[r[0]] if r[0] in user else (extra.get(r[0]) or {}).get("xmp_label", "")
        if key:
            labels[str(ids[r[0]])] = key
    pending = sum(1 for r in rows if r[0] not in extra)
    with _SCANS_GUARD:
        th = _SCANS.get(str(f))
        scanning = bool(th and th.is_alive())
    return {
        "n": len(rows), "pending": pending, "scanning": scanning,
        "vals": {"make": makes, "model": models, "lens": lens, "lens_serial": serial},
        "rows": {str(ids[r[0]]): [mk[i], md[i], ln[i], sn[i], _orient(r[3], r[4])]
                 for i, r in enumerate(rows)},
        "labels": labels,
    }  # fmt: skip


def scan(f: Path, settings: Any) -> None:
    """Back-fill the lens fields (once per image) and the sidecar colour label (when the sidecar
    changed). Runs in a background thread; only reads EXIF / sidecars, writes `exif_extra`."""
    try:
        with Cache.for_folder(f) as c:
            paths = [r[0] for r in c.facet_rows()]
            extra = c.exif_extra_rows()
        batch: list[tuple[str, str, str, int, str, int | None]] = []
        for p in paths:
            e = extra.get(p)
            virtual = split_virtual(p) is not None
            try:
                mt = None if virtual else sidecar_path(p).stat().st_mtime_ns
            except OSError:
                mt = None
            need_lens = e is None or e["ver"] != _LENS_VER
            need_xmp = e is None or e["xmp_mtime_ns"] != mt
            if not need_lens and not need_xmp:
                continue
            lm, ls = (e["lens_model"], e["lens_serial"]) if e and not need_lens else ("", "")
            if need_lens and not virtual and os.path.exists(p):
                lm, ls = read_lens(p)
            xl = e["xmp_label"] if e and not need_xmp else ""
            if need_xmp and mt is not None:
                data = read_sidecar(p)
                if data is not None:
                    xl = label_key(user_values(data, settings)[1])
            batch.append((p, lm, ls, _LENS_VER, xl, mt))
            if len(batch) >= 200:
                with Cache.for_folder(f) as c:
                    c.put_exif_extra(batch)
                batch = []
                time.sleep(0.02)  # a background chore: leave the server's other requests room
        if batch:
            with Cache.for_folder(f) as c:
                c.put_exif_extra(batch)
    except Exception as exc:  # noqa: BLE001 - a back-fill must never take the server down
        log.warning("facet scan failed for %s: %s", f, exc)


def register(app: FastAPI, state: Any) -> None:
    from shotsieve.server.app import _path_of, validate_folder

    @app.post("/api/label")
    def post_label(req: LabelRequest) -> dict[str, Any]:
        """Set (or clear, `none`) the colour label of images. Stored per image, newest wins; the
        sidecar is only written by Write XMP / Open in Lightroom (user-wins, .bak per hash)."""
        f = validate_folder(req.folder, state)
        if req.label not in LABEL_XMP:
            raise HTTPException(422, "label must be red, yellow, green, blue, purple or none")
        if not 1 <= len(req.image_ids) <= 5000:
            raise HTTPException(422, "give 1 to 5000 image ids")
        paths = [_path_of(f, i) for i in req.image_ids]
        value = "" if req.label == "none" else req.label
        with Cache.for_folder(f) as c:
            c.set_labels([(p, value) for p in paths])
        state.autoxmp.request(f, set(paths))
        return {"ok": True, "n": len(paths), "label": req.label}

    @app.get("/api/shoot/facets")
    def get_facets(folder: str | None = None) -> dict[str, Any]:
        f = validate_folder(folder, state)
        return facets_payload(f)

    @app.post("/api/shoot/facets/scan")
    def post_scan(req: FolderBody) -> dict[str, Any]:
        f = validate_folder(req.folder, state)
        with _SCANS_GUARD:
            th = _SCANS.get(str(f))
            if not (th and th.is_alive()):
                th = threading.Thread(target=scan, args=(f, state.settings), daemon=True)
                _SCANS[str(f)] = th
                th.start()
        return {"ok": True}

    @app.get("/api/people/map")
    def get_people_map(folder: str | None = None) -> dict[str, Any]:
        """{person id: [image ids]} for the People filter (any / all done in the browser)."""
        f = validate_folder(folder, state)
        with Cache.for_folder(f) as c:
            if not c.people_consent(state.settings.people_consent):
                raise HTTPException(403, "people features are off; enable them with consent first")
            with c._lock:
                rows = c._con.execute(
                    "SELECT DISTINCT f.person_id, i.id FROM person_faces f "
                    "JOIN people p ON p.id=f.person_id JOIN image_ids i ON i.path=f.path"
                ).fetchall()
        out: dict[str, list[int]] = {}
        for pid, iid in rows:
            out.setdefault(str(pid), []).append(iid)
        return {"people": out}
