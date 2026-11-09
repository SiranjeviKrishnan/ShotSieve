"""Incremental re-cull with stable decisions (T21).

A re-cull of a shoot that gained or lost files must not flip frames the user already reviewed.
Three things make that so:

* the shoot-wide sharpness references (`ShootNorm`) are frozen in `meta.shoot_norm` and reused
  while the accumulated change stays <= 20 % of the shoot (else a full re-decide, reported);
* the *affected set* A: frames in a duplicate group or burst (before or after the change) that
  holds a new / removed frame, plus frames on the same body within the dup / flash window of
  one. Everything is still computed for the whole shoot (cheap), but a frame outside A keeps
  its stored decision row verbatim;
* group / burst ids of unaffected groups stay stable (new -> old id by member overlap).

Shoot-global couplings handled here: the norm (frozen), the calibrated thresholds and the
personal model, the settings that shape grouping / scoring, per-body clock offsets (any change
-> full re-decide), the frozen audit sample (new frames never enter it) and
`dup_review_budget` (its cut is shoot-wide, but frames outside A keep their stored status).
"""

from __future__ import annotations

import bisect
import hashlib
import json
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shotsieve.config import Settings
from shotsieve.features.record import ShootNorm

CHANGE_LIMIT = 0.20  # reuse the stored norm while (new + removed) <= 20 % of its basis
# settings that shape grouping, scoring and decisions (extraction ones live in the extractor fp)
DECIDE_KEYS = (
    "burst_gap_s", "dup_window_s", "dup_cos_dist", "face_same_person_cos", "keep_per_group",
    "tie_margin", "t_keep", "t_reject", "personal_prior_n", "t_soft", "eyes_closed_thresh",
    "eyes_closed_both_thresh", "flash_window_s", "flash_dark_ev_nofire", "flash_dark_ev_fired",
    "audit_seed", "audit_enabled",
)  # fmt: skip


def _h(obj: Any) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def signature(
    settings: Settings, thresholds: dict | None, personal: str, offsets: dict | None
) -> dict[str, str]:
    """Hashes of everything (besides the file set) that a stored decision depends on."""
    return {
        "settings": _h({k: getattr(settings, k) for k in DECIDE_KEYS}),
        "thresholds": _h(thresholds or {}),
        "model": personal,
        "offsets": _h(offsets or {}),
    }


def personal_id(settings: Settings, given: Any, default_path: Path) -> str:
    if given is not None:
        return f"obj:{type(given).__name__}:{getattr(given, 'n_train', 0)}:" + _h(
            getattr(given, "thresholds", None)
        )
    path = Path(settings.personal_model_path) if settings.personal_model_path else default_path
    try:
        st = path.stat()
    except OSError:
        return "none"
    return f"file:{st.st_size}:{st.st_mtime_ns}"


@dataclass
class Plan:
    mode: str  # full | incremental | none
    reason: str = ""
    norm: ShootNorm | None = None  # the frozen norm to reuse (None: compute and store a new one)
    drift: int = 0  # accumulated change since the norm was computed
    n_new: int = 0
    n_removed: int = 0
    stored: dict = field(default_factory=dict)


def plan(
    stored: dict | None,
    *,
    extractor: str,
    sig: dict[str, str],
    n_prev: int,
    n_new: int,
    n_removed: int,
    force_full: bool,
) -> Plan:
    """Decide between reusing the stored norm (incremental) and a full re-decide."""
    p = Plan("full", n_new=n_new, n_removed=n_removed)
    if force_full:
        p.reason = "full re-decide requested (cull --full / recalibration)"
    elif not stored or not n_prev:
        p.reason = "first cull of this folder" if not n_prev else "no stored shoot norm"
    elif stored.get("extractor") != extractor:
        p.reason = "feature extractor changed"
    elif stored.get("sig") != sig:
        changed = [k for k in sig if (stored.get("sig") or {}).get(k) != sig[k]]
        p.reason = "changed since the last cull: " + ", ".join(changed)
    else:
        basis = max(int(stored.get("n", 0)), 1)
        drift = int(stored.get("drift", 0)) + n_new + n_removed
        if drift > CHANGE_LIMIT * basis:
            p.reason = (
                f"{drift} frames added or removed since the shoot norm was set "
                f"(> {int(CHANGE_LIMIT * 100)} % of {basis})"
            )
        else:
            p.mode = "none" if not (n_new or n_removed) else "incremental"
            p.norm = ShootNorm(
                float(stored["lap_ref"]), float(stored["tenen_ref"]),
                float(stored["face_ref"]), float(stored["af_ref"]),
            )  # fmt: skip
            p.drift = drift
    p.stored = stored or {}
    return p


def norm_record(norm: ShootNorm, n: int, extractor: str, sig: dict[str, str], drift: int) -> dict:
    return {
        "lap_ref": norm.lap_ref, "tenen_ref": norm.tenen_ref, "face_ref": norm.face_ref,
        "af_ref": norm.af_ref, "n": n, "extractor": extractor, "sig": sig, "drift": drift,
    }  # fmt: skip


def affected_set(
    *,
    paths: list[str],
    group_ids: list[int],
    burst_ids: list[int],
    serials: list[str],
    ts: list[float | None],
    new: set[str],
    removed: dict[str, tuple[str, float | None]],
    prev_group: dict[str, int],
    prev_burst: dict[str, int],
    window_s: float,
    extra: set[str] | None = None,
) -> set[str]:
    """The affected set A (paths; may include removed ones, which are no longer present).

    `extra` seeds are frames that must be re-judged for another reason (overridden audit frames);
    like new/removed ones they are closed over their post- and pre-change groups and bursts."""
    idx = {p: k for k, p in enumerate(paths)}
    post_g: dict[int, list[int]] = defaultdict(list)
    post_b: dict[int, list[int]] = defaultdict(list)
    for k, (g, b) in enumerate(zip(group_ids, burst_ids, strict=True)):
        post_g[g].append(k)
        post_b[b].append(k)
    prev_g: dict[int, list[str]] = defaultdict(list)
    prev_b: dict[int, list[str]] = defaultdict(list)
    for p, g in prev_group.items():
        prev_g[g].append(p)
    for p, b in prev_burst.items():
        prev_b[b].append(p)

    a: set[str] = set()
    work: deque[str] = deque()

    def add(p: str) -> None:
        if p not in a:
            a.add(p)
            work.append(p)

    # same-body frames within the window of a new / removed frame
    by_body: dict[str, tuple[list[float], list[int]]] = {}
    tmp: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for k, (s, t) in enumerate(zip(serials, ts, strict=True)):
        if t is not None:
            tmp[s].append((t, k))
    for s, rows in tmp.items():
        rows.sort()
        by_body[s] = ([r[0] for r in rows], [r[1] for r in rows])

    def near(serial: str, t: float | None) -> None:
        if t is None or serial not in by_body:
            return
        arr, ks = by_body[serial]
        lo = bisect.bisect_left(arr, t - window_s)
        hi = bisect.bisect_right(arr, t + window_s)
        for k in ks[lo:hi]:
            add(paths[k])

    for p in new:
        add(p)
        k = idx.get(p)
        if k is not None:
            near(serials[k], ts[k])
    for p, (serial, t) in removed.items():
        add(p)
        near(serial, t)
    for p in extra or ():
        add(p)
    while work:
        p = work.popleft()
        k = idx.get(p)
        if k is not None:
            for j in post_g[group_ids[k]]:
                add(paths[j])
            for j in post_b[burst_ids[k]]:
                add(paths[j])
        og, ob = prev_group.get(p), prev_burst.get(p)
        if og is not None:
            for q in prev_g[og]:
                add(q)
        if ob is not None:
            for q in prev_b[ob]:
                add(q)
    return a


def stable_ids(new_ids: list[int], paths: list[str], old_of: dict[str, int]) -> list[int]:
    """Renumber `new_ids` so a group keeps the id of the stored group it overlaps most (each
    stored id is claimed once); groups with no counterpart get ids above every stored one."""
    members: dict[int, list[int]] = defaultdict(list)
    for k, g in enumerate(new_ids):
        members[g].append(k)
    votes: list[tuple[int, int, int]] = []
    for g, ks in members.items():
        for old, c in Counter(old_of[paths[k]] for k in ks if paths[k] in old_of).items():
            votes.append((-c, old, g))
    votes.sort()
    mapping: dict[int, int] = {}
    claimed: set[int] = set()
    for _, old, g in votes:
        if g not in mapping and old not in claimed:
            mapping[g] = old
            claimed.add(old)
    fresh = max(old_of.values(), default=-1) + 1
    for g in sorted(members):
        if g not in mapping:
            mapping[g] = fresh
            fresh += 1
    return [mapping[g] for g in new_ids]
