"""Jersey table maintenance: burst vote, roster mapping, person naming, XMP/IPTC writeback."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from shotsieve.features import jersey as jf
from shotsieve.io.roster import RosterEntry
from shotsieve.io.xmp import merge_keywords
from shotsieve.log import get_logger
from shotsieve.pipeline import people as people_mod
from shotsieve.pipeline.cache import Cache

log = get_logger("jerseys")


def finalize(
    cache: Cache, consent: bool = False, epoch: int | None = None
) -> dict[str, int]:
    """Re-run the burst vote over all OCR'd frames and (with people consent) name unnamed face
    clusters whose dominant jersey number is on the roster."""
    rows = [r for r in cache.jerseys() if r["source"] == "ocr"]
    reads: dict[str, dict[str, float]] = {}
    with cache._lock:
        ocrd = [cache._abs(r[0]) for r in cache._con.execute(
            "SELECT path FROM index_state WHERE kind='ocr'")]  # fmt: skip
    for p in ocrd:
        reads[p] = {}
    for r in rows:
        reads.setdefault(r["path"], {})[r["number"]] = max(
            r["conf"], reads.get(r["path"], {}).get(r["number"], 0.0)
        )
    dec = cache.load_decisions()
    bursts = {p: d["burst_id"] for p, d in dec.items() if d.get("burst_id") is not None}
    prop = jf.burst_vote(reads, bursts)
    cache.replace_burst_jerseys(prop)
    named = 0
    roster = cache.roster()
    team = cache.roster_team()
    if consent and roster:
        for pid, num in jf.dominant_numbers(rows).items():
            status, rname, _ = jf.resolve(roster, num, team)
            if status == "matched":  # an ambiguous number never names a person
                # re-check consent in the same critical section as the write (Review #4 M3)
                with cache.people_write(epoch) as ok:
                    if not ok:
                        break
                    cur = cache._con.execute(
                        "SELECT name FROM people WHERE id=?", (pid,)
                    ).fetchone()
                    if cur is not None and not (cur[0] or "").strip():
                        people_mod.rename(cache, pid, rname)
                        named += 1
    return {"direct": len(rows), "burst": len(prop), "people_named": named}


def set_roster(
    cache: Cache, entries: list[RosterEntry], team: str | None = None
) -> dict[str, Any]:
    cache.set_roster([(e.number, e.name, e.team) for e in entries])
    if team is not None:
        cache.set_meta("roster_team", team.strip())
    return roster_summary(cache)


def set_team(cache: Cache, team: str) -> dict[str, Any]:
    """Per-folder team setting: settles numbers that two teams share."""
    cache.set_meta("roster_team", team.strip())
    return roster_summary(cache)


def roster_summary(cache: Cache) -> dict[str, Any]:
    roster = cache.roster()
    team = cache.roster_team()
    rows = jf.annotate(cache.jerseys(), roster, team)
    imgs = {r["path"] for r in rows}
    matched = {r["path"] for r in rows if r["status"] == "matched"}
    amb = {r["path"] for r in rows if r["status"] == "ambiguous"}
    return {
        "entries": sum(len(v) for v in roster.values()),
        "team": team,
        "roster": [
            {"number": n, "name": name, "team": t}
            for n, v in sorted(roster.items())
            for name, t in v
        ],
        "images_with_numbers": len(imgs),
        "images_matched": len(matched),
        "unmatched_numbers": sorted({r["number"] for r in rows if r["status"] == "unmatched"}),
        # shared numbers with no team to settle them: no name written, frame needs a look
        "ambiguous_numbers": sorted({r["number"] for r in rows if r["status"] == "ambiguous"}),
        "images_ambiguous": len(amb),
        "review_images": sorted(Path(p).name for p in amb),
    }


def jerseys_by_image(cache: Cache) -> dict[int, list[dict[str, Any]]]:
    """image id -> [{number, name, status, conf, source}]."""
    rows = jf.annotate(cache.jerseys(), cache.roster(), cache.roster_team())
    ids = cache.ids_for(sorted({r["path"] for r in rows}))
    out: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        if r["path"] in ids:
            out.setdefault(ids[r["path"]], []).append(
                {k: r[k] for k in ("number", "name", "team", "status", "conf", "source")}
            )
    return out


def write_xmp(cache: Cache) -> dict[str, int]:
    """Merge `#N` / roster names into each image's sidecar (dc:subject + PersonInImage). Foreign
    keywords and people are never removed; originals are never touched."""
    tags = jf.xmp_tags(cache.jerseys(), cache.roster(), cache.roster_team())
    written = failed = added = 0
    for path, t in tags.items():
        try:
            n = merge_keywords(Path(path), t["keywords"], t["persons"])
            added += n
            written += n > 0
        except Exception as exc:  # noqa: BLE001 - one odd sidecar must not abort the rest
            log.warning("jersey xmp failed for %s: %s", path, exc)
            failed += 1
    return {"images": len(tags), "written": written, "failed": failed, "items_added": added}
