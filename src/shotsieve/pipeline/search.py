"""Natural-language + filtered search over the SigLIP2 index.

Query syntax: free text plus filters `person:"Name"` (needs people consent), `jersey:23`,
`is:keep|review|reject`, `camera:SERIAL`. Text -> text tower -> cosine kNN; hits are kept
while cos >= top1 - SCORE_MARGIN. Filters only -> time order.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from shotsieve.config import Settings
from shotsieve.features import jersey as jf
from shotsieve.io.roster import normalize_number
from shotsieve.pipeline.cache import Cache
from shotsieve.pipeline.people import ConsentError, image_ids_for_person

SCORE_MARGIN = 0.08
_FILTER = re.compile(r'(person|jersey|is|camera):(?:"([^"]*)"|(\S+))', re.I)


@dataclass
class Hit:
    id: int
    score: float
    why: str = ""
    path: str = ""


@dataclass
class ParsedQuery:
    text: str
    filters: dict[str, list[str]] = field(default_factory=dict)


def parse_query(q: str) -> ParsedQuery:
    filters: dict[str, list[str]] = {}

    def take(m: re.Match) -> str:
        filters.setdefault(m.group(1).lower(), []).append(m.group(2) or m.group(3) or "")
        return " "

    text = _FILTER.sub(take, q)
    return ParsedQuery(" ".join(text.split()), filters)


def _all_paths(cache: Cache) -> list[str]:
    return sorted(set(cache.timestamps()) | set(cache.load_decisions()))


def _filter_ids(
    cache: Cache, pq: ParsedQuery, settings: Settings, warnings: list[str] | None = None
) -> set[int] | None:
    """Image ids passing every filter (None when there are no filters)."""
    if not pq.filters:
        return None
    ids_of = cache.ids_for(_all_paths(cache))
    allowed = set(ids_of.values())
    f = pq.filters
    if "person" in f:
        consent = cache.people_consent(settings.people_consent)
        roster = cache.roster()
        team = cache.roster_team()
        for name in f["person"]:
            want = name.strip().lower()
            on_roster = any(rn.strip().lower() == want for v in roster.values() for rn, _ in v)
            ids: set[int] = set()
            if consent:
                ids |= image_ids_for_person(cache, name)
            elif not on_roster:
                raise ConsentError("people search needs consent (POST /api/people/consent)")
            if on_roster:  # roster names resolve to (team, number): no face data involved
                nums = []
                for n, v in roster.items():
                    if not any(rn.strip().lower() == want for rn, _ in v):
                        continue
                    status, rname, _ = jf.resolve(roster, n, team)
                    if status == "matched" and rname.strip().lower() == want:
                        nums.append(n)
                    elif status == "ambiguous" and warnings is not None:
                        teams = {t.strip().lower() for _, t in roster[n] if t.strip()}
                        if len(teams) > 1:
                            msg = f"#{n} is on {len(teams)} teams; set the folder team"
                        else:  # teamless entries: a folder team cannot settle it
                            names = len({rn.strip().lower() for rn, _ in roster[n]})
                            msg = f"#{n} has {names} names on the roster; give them teams"
                        if msg not in warnings:
                            warnings.append(msg)
                if nums:
                    ids |= _jersey_ids(cache, nums, ids_of)
            allowed &= ids
    if "is" in f:
        dec, ovr = cache.load_decisions(), cache.overrides()
        want = {v.lower() for v in f["is"]}
        keep = {
            ids_of[p] for p, r in dec.items()
            if p in ids_of and (ovr[p]["decision"] if p in ovr else r["decision"]) in want
        }  # fmt: skip
        allowed &= keep
    if "camera" in f:
        with cache._lock:
            rows = cache._con.execute("SELECT path, meta_json FROM images").fetchall()
        cams = {c.lower() for c in f["camera"]}
        ok = set()
        for p, mj in rows:
            try:
                if str(json.loads(mj).get("camera_serial", "")).lower() in cams:
                    ok.add(ids_of.get(cache._abs(p)))
            except (TypeError, ValueError):
                pass
        allowed &= ok
    if "jersey" in f:
        allowed &= _jersey_ids(cache, f["jersey"], ids_of)
    return allowed


def _jersey_ids(cache: Cache, numbers: list[str], ids_of: dict[str, int]) -> set[int]:
    """Images with a recognised jersey number (direct OCR reads and burst-propagated ones)."""
    with cache._lock:
        have = cache._con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='jerseys'"
        ).fetchone()
        if not have:
            return set()
        numbers = [normalize_number(n) or n for n in numbers]
        qs = ",".join("?" * len(numbers))
        rows = cache._con.execute(
            f"SELECT path FROM jerseys WHERE number IN ({qs})", numbers  # noqa: S608
        ).fetchall()
    return {ids_of[cache._abs(r[0])] for r in rows if cache._abs(r[0]) in ids_of}


def search(
    folder: str | Path,
    q: str,
    k: int = 100,
    settings: Settings | None = None,
    text_embedder: Any = None,
    use_vec: bool = True,
    warnings: list[str] | None = None,
) -> list[Hit]:
    """Search one shoot; `text_embedder.embed(str) -> (768,)` defaults to the lazy SigLIP2 text.
    Notices for the caller (an ambiguous roster number) are appended to `warnings`."""
    settings = settings or Settings()
    pq = parse_query(q)
    with Cache.for_folder(folder) as cache:
        allowed = _filter_ids(cache, pq, settings, warnings)
        if not pq.text:
            if allowed is None:
                return []
            ts = cache.adjusted_timestamps()
            paths = _all_paths(cache)
            ids_of = cache.ids_for(paths)
            order = sorted(
                (p for p in paths if ids_of.get(p) in allowed),
                key=lambda p: (ts.get(p) is None, ts.get(p) or 0.0, p),
            )
            return [Hit(ids_of[p], 0.0, "filter", p) for p in order[:k]]
        if text_embedder is None:
            from shotsieve.models.text_image import get_text

            text_embedder = get_text()
        vec = text_embedder.embed(pq.text)
        # an unfiltered kNN can use vec0; a filtered one scans the allowed subset
        raw = cache.knn(vec, k, allowed, use_vec=use_vec)
        if not raw:
            return []
        floor = raw[0][1] - SCORE_MARGIN
        hits = [Hit(i, s, f'matches "{pq.text}"') for i, s in raw if s >= floor][:k]
        for h in hits:
            h.path = cache.path_for_id(h.id) or ""
        return hits


def recall_at_k(ranked: list[str], relevant: set[str], k: int = 10) -> float:
    if not relevant:
        return 1.0
    return len(set(ranked[:k]) & relevant) / min(k, len(relevant))


def knn_agreement(a: np.ndarray, b: np.ndarray, k: int = 20) -> float:
    return len(set(a[:k].tolist()) & set(b[:k].tolist())) / k
