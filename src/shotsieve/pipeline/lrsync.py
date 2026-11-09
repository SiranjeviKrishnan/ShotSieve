"""Lightroom Classic sync-back (T20): what the photographer finally picked in Lightroom flows
back into ShotSieve as overrides (so the personal model and the per-shoot calibration learn from
real use, and a re-cull never fights the user).

The catalog is only ever read (`io/lrcat.py`: `mode=ro`, or a private checked copy). Nothing is
ever written to Lightroom.

Agreement is not a label: a Lightroom rating / label equal to what ShotSieve itself wrote
(`lr_written`) is ShotSieve's opinion, never the photographer's.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from shotsieve._ort import user_home
from shotsieve.config import Settings
from shotsieve.io.formats import split_virtual
from shotsieve.io.lrcat import CatalogEntry, CatalogReader
from shotsieve.io.xmp import read_sidecar
from shotsieve.log import get_logger
from shotsieve.pipeline.cache import Cache
from shotsieve.pipeline.evaluate import evaluate, norm_path, truth_from_catalog

log = get_logger("lrsync")

ProgressFn = Callable[[str, int, int], None]
NOTE = "lrcat"
MAX_LISTED = 500  # changes / conflicts echoed back to the UI


@dataclass
class SyncResult:
    dry_run: bool = False
    catalog: str = ""
    folder: str = ""
    copied: bool = False  # the catalog was read from a private copy (Lightroom had it open)
    n_catalog: int = 0  # catalog images under the folder
    n_matched: int = 0
    n_unmatched: int = 0
    n_fallback: int = 0  # matched by file name + capture time (copy imports)
    n_new: int = 0
    n_changed: int = 0
    n_cleared: int = 0
    n_keeps: int = 0
    n_rejects: int = 0
    n_conflicts_ui_won: int = 0
    n_conflicts_lr_won: int = 0
    changes: list[dict[str, Any]] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    report: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["n_writes"] = self.n_new + self.n_changed + self.n_cleared
        d["summary"] = self.describe()
        return d

    def describe(self) -> str:
        """One line for the UI (main Lightroom sheet and setup wizard share it)."""
        n = self.n_new + self.n_changed + self.n_cleared
        t = f"{n} change{'' if n == 1 else 's'}: {self.n_keeps} keeps, {self.n_rejects} rejects"
        c = self.n_conflicts_ui_won + self.n_conflicts_lr_won
        if c:
            t += f"; {c} conflicts"
            if self.n_conflicts_ui_won:
                t += f", your ShotSieve choice kept on {self.n_conflicts_ui_won}"
            if self.n_conflicts_lr_won:
                t += f", Lightroom won on {self.n_conflicts_lr_won}"
        return t + f". Matched {self.n_matched} of {self.n_matched + self.n_unmatched} frames."

    def summary(self) -> dict[str, Any]:
        """The numbers stored as `meta.lr_sync_last`."""
        keys = ("catalog", "n_catalog", "n_matched", "n_unmatched", "n_new", "n_changed",
                "n_conflicts_ui_won", "n_conflicts_lr_won")  # fmt: skip
        out = {k: getattr(self, k) for k in keys}
        out["ts"] = time.time()
        return out


# ---- baseline of ShotSieve's own output ----------------------------------------------------------


def backfill_rows(paths: Any) -> list[tuple[str, int | None, str | None, str]]:
    """`lr_written` rows recovered from the sidecars' `shotsieve:WrittenRating/Label`."""
    rows = []
    for p in paths:
        if split_virtual(p):
            continue
        d = read_sidecar(p)
        if d is None or (d.written_rating is None and d.written_label is None):
            continue
        rows.append((p, d.written_rating, d.written_label, d.decision or ""))
    return rows


def ensure_written(cache: Cache, decisions: dict[str, Any] | None = None) -> int:
    """Backfill `lr_written` once when the table is empty (shoots culled before T20). Returns the
    number of rows added."""
    if cache.written():
        return 0
    rows = backfill_rows(decisions if decisions is not None else cache.decision_summary())
    if rows:
        cache.put_written(rows)
    return len(rows)


# ---- the user's signal -----------------------------------------------------------------------


def label_decision(label: str | None, settings: Settings) -> str | None:
    """keep / reject for a Lightroom colour-label name via `Settings.lr_label_map`."""
    if not label:
        return None
    m = {k.casefold(): v for k, v in settings.lr_label_map.items()}
    d = m.get(label.casefold())
    return d if d in ("keep", "reject") else None


def user_signal(
    rating: int | None,
    pick: int,
    label: str | None,
    written: dict[str, Any] | None,
    settings: Settings,
) -> tuple[str, int | None, str] | None:
    """(decision, rating, reason) the photographer's Lightroom state implies, or None.

    Order: pick > rating > label; everything equal to what ShotSieve wrote is no signal."""
    r = int(rating or 0)
    w_r = None if not written else written.get("rating")
    w_l = None if not written else written.get("label")
    if pick == -1:
        return "reject", None, "pick"
    if pick == 1:
        return "keep", r or None, "pick"
    if r >= settings.min_rating and (w_r is None or r != w_r):
        return "keep", r, "rating"
    if w_r is not None and w_r >= settings.min_rating and r < settings.min_rating:
        return "reject", None, "downgrade"
    dec = label_decision(label, settings)
    if dec and (w_l is None or str(label).casefold() != str(w_l).casefold()):
        return dec, None, "label"
    return None


# ---- matching ----------------------------------------------------------------------------------


def _capture_second(s: str | None) -> int | None:
    if not s:
        return None
    try:
        return int(datetime.fromisoformat(s[:19]).timestamp())
    except ValueError:
        return None


def match_entries(
    cat: CatalogReader,
    entries: list[CatalogEntry],
    frames: list[str],
    cache: Cache,
) -> tuple[dict[str, CatalogEntry], int]:
    """({frame path: catalog entry}, how many matched by the fallback key).

    Exact `norm_path` match first; then (casefold(file name), capture time to the second), unique
    on both sides in the catalog subset (Lightroom "Copy" imports to another folder). RAW+JPEG
    pairs match on the RAW master (Lightroom lists the JPEG as a sidecar extension). Ambiguous
    keys stay unmatched: never guessed."""
    by_norm = {norm_path(p): p for p in frames}
    matched: dict[str, CatalogEntry] = {}
    used: set[int] = set()
    for e in entries:
        p = by_norm.get(norm_path(e.path))
        if p is not None and p not in matched:
            matched[p] = e
            used.add(e.id_local)
    rest = [p for p in frames if p not in matched]
    if not rest:
        return matched, 0
    times = cache.exif_times()
    fkey: dict[tuple[str, int], list[str]] = defaultdict(list)
    for p in rest:
        t = times.get(p)
        if t is not None:
            fkey[(os.path.basename(p).casefold(), int(t))].append(p)
    if not fkey:
        return matched, 0
    names = {os.path.basename(p) for p in rest}
    cands = [e for e in cat.entries(names=names) if e.id_local not in used]
    ckey: dict[tuple[str, int], list[CatalogEntry]] = defaultdict(list)
    for e in cands:
        sec = _capture_second(e.capture)
        if sec is not None:
            ckey[(os.path.basename(e.path).casefold(), sec)].append(e)
    n = 0
    for k, fs in fkey.items():
        cs = ckey.get(k)
        if len(fs) == 1 and cs and len(cs) == 1:
            matched[fs[0]] = cs[0]
            n += 1
    return matched, n


# ---- the sync --------------------------------------------------------------------------------


def _same_state(prev: dict[str, Any] | None, e: CatalogEntry) -> bool:
    if prev is None:
        return False
    return (prev["rating"], prev["pick"], prev["label"]) == (e.rating, e.pick, e.label)


def sync(
    folder: str | Path,
    catalog: str | Path,
    settings: Settings | None = None,
    *,
    dry_run: bool = False,
    report: bool = True,
    progress: ProgressFn | None = None,
) -> SyncResult:
    """Read the catalog, match it to the shoot's decided frames, store the photographer's
    Lightroom choices as `note="lrcat"` overrides. `dry_run` changes nothing at all."""
    settings = settings or Settings()
    folder = Path(os.path.abspath(Path(folder).expanduser()))
    emit = progress or (lambda *_: None)
    res = SyncResult(dry_run=dry_run, catalog=str(catalog), folder=str(folder))
    with Cache.for_folder(folder) as cache:
        dec = cache.decision_summary()
        if not dec:
            raise ValueError("this folder has not been culled yet")
        frames = sorted(p for p in dec if not split_virtual(p))
        written = cache.written()
        if not written:  # shoots culled before T20: recover our own output from the sidecars
            rows = backfill_rows(frames)
            if rows and not dry_run:
                cache.put_written(rows)
            written = {p: {"rating": r, "label": lab, "decision": d} for p, r, lab, d in rows}
        emit("read", 0, 1)
        with CatalogReader(catalog) as cat:
            entries: list[CatalogEntry] = []
            seen_ids: set[int] = set()
            for pth in dict.fromkeys((folder, folder.resolve())):  # a symlinked path too
                for e in cat.entries(prefix=str(pth).rstrip("/") + "/"):
                    if e.id_local not in seen_ids:
                        seen_ids.add(e.id_local)
                        entries.append(e)
            res.n_catalog = len(entries)
            emit("read", 1, 1)
            emit("match", 0, len(frames))
            matched, res.n_fallback = match_entries(cat, entries, frames, cache)
            res.copied = cat.copied
        res.n_matched = len(matched)
        res.n_unmatched = len(frames) - len(matched)
        emit("match", len(frames), len(frames))

        ovr = cache.overrides()
        synced = cache.lr_sync_rows()
        now = time.time()
        sets: list[tuple[str, str, int | None, float, str]] = []  # path, decision, rating, ts, why
        clears: list[str] = []
        state_rows = []
        for p, e in matched.items():
            prev = synced.get(p)
            changed = not _same_state(prev, e)
            if changed:
                state_rows.append((p, e.id_local, e.rating, e.pick, e.label, e.touch_ts))
            sig = user_signal(e.rating, e.pick, e.label, written.get(p), settings)
            o = ovr.get(p)
            lr_o = o is not None and o.get("note") == NOTE
            if sig is None:
                if lr_o and changed:  # the photographer reverted it in Lightroom
                    clears.append(p)
                    res.changes.append(_change(folder, p, o["decision"], "cleared", "reverted"))
                continue
            decision, rating, why = sig
            ts = e.touch_ts or now
            if o is None:
                kind = "new"
            elif lr_o:
                if not changed and o["decision"] == decision:
                    continue
                kind = "changed"
            else:  # a ShotSieve (UI) override exists
                if o["decision"] == decision:
                    continue  # agreement: nothing to decide
                if e.touch_ts is not None:
                    lr_wins = e.touch_ts > (o.get("ts") or 0.0)
                else:
                    lr_wins = prev is not None and changed
                conflict = {
                    "path": os.path.relpath(p, folder), "name": os.path.basename(p),
                    "lightroom": decision, "shotsieve": o["decision"],
                    "winner": "lightroom" if lr_wins else "shotsieve",
                }  # fmt: skip
                if len(res.conflicts) < MAX_LISTED:
                    res.conflicts.append(conflict)
                if not lr_wins:
                    res.n_conflicts_ui_won += 1
                    continue
                res.n_conflicts_lr_won += 1
                kind = "changed"
            sets.append((p, decision, rating, ts, why))
            if kind == "new":
                res.n_new += 1
            else:
                res.n_changed += 1
            if decision == "keep":
                res.n_keeps += 1
            else:
                res.n_rejects += 1
            res.changes.append(_change(folder, p, decision, kind, why))
        res.n_cleared = len(clears)
        res.changes = res.changes[:MAX_LISTED]

        emit("write", 0, len(sets) + len(clears))
        if not dry_run:
            for p, decision, rating, ts, _why in sets:
                cache.set_override(
                    p, decision, 1.0, NOTE, rating, ts=ts, commit=False
                )  # propensity 1.0: the selection bias is documented, the audit stays unbiased
            for p in clears:
                cache.clear_override(p)
            cache.commit()
            if state_rows:
                cache.put_lr_sync(state_rows)
        emit("write", len(sets) + len(clears), len(sets) + len(clears))

        if report:
            emit("report", 0, 1)
            res.report = build_report(folder, dec, matched, settings, cache)
            emit("report", 1, 1)
        if not dry_run:
            cache.set_meta("lr_catalog", str(catalog))
            cache.set_meta("lr_sync_last", res.summary())
            if res.report is not None:
                cache.set_meta("lr_report", res.report)
    return res


def _change(folder: Path, p: str, decision: str, kind: str, why: str) -> dict[str, Any]:
    return {"path": os.path.relpath(p, folder), "name": os.path.basename(p),
            "decision": decision, "kind": kind, "reason": why}  # fmt: skip


# ---- "this shoot" report ---------------------------------------------------------------------


def build_report(
    folder: Path,
    dec: dict[str, tuple[str, int, bool]],
    matched: dict[str, CatalogEntry],
    settings: Settings,
    cache: Cache | None = None,
) -> dict[str, Any]:
    """How the stored *model* decisions fared against the photographer's final Lightroom state
    (`pipeline/evaluate.evaluate`). Truth: kept = pick 1 or rating >= min_rating, not rejected;
    unrated, unflagged frames count as not kept. Predictions: the model's decisions, never the
    overrides. Selection bias: the photographer mostly looked at what ShotSieve kept, so read the
    audit numbers for the unbiased miss rate."""
    truth_entries = [
        CatalogEntry(p, e.rating, e.pick, e.label) for p, e in matched.items()
    ]
    truth = truth_from_catalog(truth_entries, settings.min_rating)
    images = [
        SimpleNamespace(
            path=Path(p), master_path=Path(p), decision=d, model_decision=d, group_id=g, audit=a
        )
        for p, (d, g, a) in dec.items()
        if not split_virtual(p)
    ]
    ev = evaluate(images, truth)

    def num(x: float) -> float | None:
        return None if x != x else float(x)

    return {
        "group_gate_rate": ev.group_gate_rate,
        "missed_keeper_rate": ev.missed_keeper_rate,
        "review_fraction": ev.review_fraction,
        "keepers_in_review": ev.keepers_in_review,
        "reject_precision": num(ev.reject_precision),
        "n_evaluated": ev.n_evaluated,
        "n_kept_truth": ev.n_kept_truth,
        "missed_keepers": ev.missed_keepers,
        "missed_files": [os.path.relpath(p, folder) for p in ev.missed_paths][:MAX_LISTED],
        "ts": time.time(),
        "note": "Selection bias: you mostly looked at what ShotSieve kept; the audit is the "
        "unbiased signal.",
    }


# ---- catalog detection (listing only) ---------------------------------------------------------


def detect_catalogs(
    home: str | Path | None = None, remembered: list[str] | None = None
) -> list[str]:
    """`~/Pictures/Lightroom/**/*.lrcat` down to depth 3, plus remembered catalog paths. Listing
    only: nothing is opened."""
    root = Path(home).expanduser() if home else user_home()
    base = root / "Pictures" / "Lightroom"
    found: list[str] = []
    if base.is_dir():
        for dirpath, dirs, files in os.walk(base):
            depth = len(Path(dirpath).relative_to(base).parts)
            if depth >= 3:
                dirs[:] = []
            found += [
                os.path.join(dirpath, f) for f in sorted(files) if f.lower().endswith(".lrcat")
            ]
    for r in remembered or []:
        if r and r not in found and Path(r).is_file():
            found.append(r)
    return found
