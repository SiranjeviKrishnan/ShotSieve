"""Shot-list coverage alerts (T17): which of the shots you must get are in the cull?

Each list line is `[!]text [person:"Name"] [xN]` (`!` = must-have, `xN` = at least N frames,
default 1). An item is scored against every indexed keep/review frame with the raw SigLIP2 cosine
`s` (rejects excluded; `person:` restricts the candidates through the T10 people / roster filter).
The robust z-score of a frame is z = (s - median_s) / (1.4826 * MAD_s) over the shoot's frames:

  covered    >= N frames with s >= TAU_HI and z >= Z_MIN
  uncertain  not covered, but >= N frames with s in [TAU, TAU_HI) and z >= Z_MIN: a near-threshold
             match, the band where absent look-alikes live. Never counted as covered: a must-have
             here is surfaced for review (banner / API `must_uncertain` / CLI), like a missing one
  weak       the best frame has z >= Z_MIN, but fewer than N frames reach TAU
  missing  otherwise
  pending  the index is incomplete (never reported as missing)

TAU_HI (N1 band, Review #7) is the highest cut that keeps >= 75 % of the train present items
strictly "covered" (`fit_tau_hi`, train only); items between TAU and TAU_HI are "uncertain".

TAU is the highest cut that keeps accuracy >= 0.94 on the train items ("an item with a relevant
frame present"): the 30 labelled queries of `tests/fixtures/search_queries.json`, 20 absent ones and
the 24 items of the burned first held-out set (`scripts/fit_shotlist.py`); frozen here. (Review #7
N1: the earlier max-F1 midpoint 0.0751 let near-miss items such as "a mosque with a golden dome"
through as covered; an item between the old and new cut is now "weak", i.e. review.)
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from shotsieve.config import Settings, data_dir
from shotsieve.io.xmp import merge_keywords
from shotsieve.log import get_logger
from shotsieve.pipeline.cache import Cache
from shotsieve.pipeline.people import ConsentError
from shotsieve.pipeline.search import ParsedQuery, _filter_ids, parse_query

log = get_logger("shotlist")

TAU = 0.0946  # raw SigLIP2 cosine; fit_tau_conservative (scripts/fit_shotlist.py), frozen
TAU_HI = 0.1106  # top of the uncertain band; fit_tau_hi (scripts/fit_shotlist.py), frozen
Z_MIN = 3.0
MIN_COVERED_RECALL = 0.75  # train present items that must stay strictly covered (band width)
MIN_TRAIN_ACC = 0.94  # train accuracy floor of the conservative tau (spec 0.90 + margin)
N_BEST = 6  # thumbnails per item
TAG_PREFIX = "shotlist:"
_COUNT = re.compile(r"\s+x(\d+)\s*$", re.I)


@dataclass
class Item:
    item_id: str
    text: str
    must: bool = False
    min_count: int = 1
    person: str | None = None
    pos: int = 0
    list_name: str = ""


@dataclass
class ItemResult:
    item: Item
    status: str = "pending"  # covered | uncertain | weak | missing | pending
    n_hits: int = 0
    top_score: float | None = None
    top_z: float | None = None
    best_ids: list[int] = field(default_factory=list)  # best frames, qualifying ones first
    hit_ids: list[int] = field(default_factory=list)  # every qualifying frame
    strong_ids: list[int] = field(default_factory=list)  # hits with s >= TAU_HI: all tag_xmp tags
    detail: str = ""

    def public(self) -> dict[str, Any]:
        it = self.item
        return {
            "item_id": it.item_id, "text": it.text, "must": it.must, "min_count": it.min_count,
            "person": it.person, "status": self.status, "detail": self.detail,
            "n_hits": self.n_hits, "top_score": self.top_score, "top_z": self.top_z,
            "best_ids": self.best_ids, "list_name": it.list_name,
        }  # fmt: skip


# ---- list files ------------------------------------------------------------------------------


def builtin_templates_dir() -> Path:
    """Templates shipped inside the package."""
    return Path(__file__).resolve().parent / "shotlist_templates"


def templates_dir() -> Path:
    """User templates: `<data dir>/shotlists/*.txt` (override a built-in of the same name)."""
    return data_dir() / "shotlists"


def _template_dirs() -> list[Path]:
    return [templates_dir(), builtin_templates_dir()]


def template_names() -> list[str]:
    names: set[str] = set()
    for d in _template_dirs():
        if d.is_dir():
            names.update(p.stem for p in d.glob("*.txt"))
    return sorted(names)


def parse_list(text: str, list_name: str = "custom") -> list[Item]:
    """Items of a list file (`#` comments and blank lines skipped)."""
    items: list[Item] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        must = line.startswith("!")
        if must:
            line = line[1:].strip()
        m = _COUNT.search(line)
        n = max(1, int(m.group(1))) if m else 1
        if m:
            line = line[: m.start()]
        pq = parse_query(line)
        person = (pq.filters.get("person") or [None])[0]
        if not pq.text:
            continue
        pos = len(items)
        items.append(Item(f"{list_name}:{pos:03d}", pq.text, must, n, person, pos, list_name))
    return items


def load_template(name: str) -> list[Item]:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name or ""):
        raise ValueError("invalid list name")
    p = next((d / f"{name}.txt" for d in _template_dirs() if (d / f"{name}.txt").is_file()),
             templates_dir() / f"{name}.txt")
    if not p.is_file():
        raise ValueError(f"no such list: {name} (have: {', '.join(template_names()) or 'none'})")
    return parse_list(p.read_text(encoding="utf-8"), name)


def load_items(list_name: str | None = None, path: str | Path | None = None,
               text: str | None = None) -> list[Item]:  # fmt: skip
    """Items from a template name, a list file, or raw text."""
    if text is not None:
        items = parse_list(text, "custom")
    elif path is not None:
        p = Path(path).expanduser()
        items = parse_list(p.read_text(encoding="utf-8"), p.stem or "custom")
    elif list_name:
        return load_template(list_name)
    else:
        raise ValueError("give a list name, a list file or the list text")
    if not items:
        raise ValueError("the shot list is empty")
    return items


# ---- the rule --------------------------------------------------------------------------------


def robust_z(s: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """z of scores `s` against `baseline` scores: (s - median) / (1.4826 MAD)."""
    med = float(np.median(baseline))
    mad = float(np.median(np.abs(baseline - med)))
    return (s - med) / max(1.4826 * mad, 1e-4)


def classify(
    sims: np.ndarray, allowed: np.ndarray, min_count: int, tau: float = TAU,
    tau_hi: float | None = None,
) -> tuple[str, int, float | None, float | None, np.ndarray, np.ndarray]:
    """(status, n_hits, top s, top z, order, hits) of one item.

    `sims`: cosines of every candidate frame (the baseline); `allowed`: bool mask of the frames the
    item may match (a `person:` filter narrows it). `order` = allowed frame indices, best first;
    `hits` = those with s >= tau and z >= Z_MIN (covered needs `min_count` of them at s >= tau_hi;
    fewer strong ones but enough hits is "uncertain").
    """
    tau_hi = max(TAU_HI if tau_hi is None else tau_hi, tau)
    if not len(sims) or not allowed.any():
        return "missing", 0, None, None, np.zeros(0, np.int64), np.zeros(0, np.int64)
    z = robust_z(sims, sims)
    idx = np.nonzero(allowed)[0]
    order = idx[np.argsort(-sims[idx], kind="stable")]
    ok = (sims[order] >= tau) & (z[order] >= Z_MIN)
    hits = order[ok]
    top = order[0]
    n_strong = int((sims[hits] >= tau_hi).sum())
    if n_strong >= min_count:
        status = "covered"
    elif len(hits) >= min_count:
        status = "uncertain"
    elif z[top] >= Z_MIN:
        status = "weak"
    else:
        status = "missing"
    return status, int(len(hits)), float(sims[top]), float(z[top]), order, hits


# ---- fitting and out-of-sample evidence ------------------------------------------------------

Stat = tuple[float, float]  # (top cosine, top robust z) of one query against a corpus


def top_stats(mat: np.ndarray, embed: Any, queries: list[str]) -> list[Stat]:
    """(top cosine, its z) of each query text over the frames' embedding matrix `mat`."""
    out: list[Stat] = []
    for q in queries:
        s = mat @ embed(q)
        z = robust_z(s, s)
        out.append((float(s.max()), float(z[int(np.argmax(s))])))
    return out


def _f1(present: list[Stat], absent: list[Stat], tau: float, z_min: float) -> float:
    tp = sum(s >= tau and z >= z_min for s, z in present)
    fp = sum(s >= tau and z >= z_min for s, z in absent)
    return 2 * tp / max(2 * tp + fp + (len(present) - tp), 1)


def fit_tau(
    present: list[Stat], absent: list[Stat], z_min: float = Z_MIN
) -> tuple[float, float, tuple[float, float]]:
    """(best F1, midpoint of the widest plateau of cuts reaching it, (lo, hi) of that plateau).

    F1 only changes when tau crosses a query's top score, so every cut in (previous score, score]
    is equivalent; the midpoint of the widest such gap is the most robust choice."""
    cands = sorted({s for s, _ in [*present, *absent]})
    f1s = [_f1(present, absent, t, z_min) for t in cands]
    top = max(f1s)
    gaps = [(cands[k] - (cands[k - 1] if k else cands[k] - 0.05), k) for k, f in enumerate(f1s)
            if f >= top - 1e-9]  # fmt: skip
    _, k = max(gaps)
    lo, hi = (cands[k - 1] if k else cands[k] - 0.05), cands[k]
    return top, (lo + hi) / 2, (lo, hi)


def fit_tau_conservative(
    present: list[Stat], absent: list[Stat], z_min: float = Z_MIN, min_acc: float = MIN_TRAIN_ACC
) -> tuple[float, float, tuple[float, float]]:
    """(train accuracy, midpoint, (lo, hi)) of the HIGHEST cut plateau whose accuracy on the
    labelled train items is still >= `min_acc`.

    Max-F1 (`fit_tau`) treats a missed shot and a falsely "covered" one alike; for a must-have
    alert the second is the worse error (the photographer stops looking), so tau is pushed as high
    as the train items allow while accuracy stays at the floor. Train data only."""
    n = len(present) + len(absent)
    cands = sorted({s for s, _ in [*present, *absent]})

    def acc(t: float) -> float:
        tp = sum(s >= t and z >= z_min for s, z in present)
        tn = sum(not (s >= t and z >= z_min) for s, z in absent)
        return (tp + tn) / n

    rows = []
    for k, c in enumerate(cands):
        lo = cands[k - 1] if k else c - 0.05
        rows.append((acc(c), (lo + c) / 2, (lo, c)))
    ok = [r for r in rows if r[0] >= min_acc - 1e-9]
    if ok:
        return ok[-1]  # the highest qualifying plateau
    top = max(r[0] for r in rows)  # floor unreachable: the most accurate, highest plateau
    return [r for r in rows if r[0] >= top - 1e-9][-1]


def fit_tau_hi(
    present: list[Stat], tau: float = TAU, z_min: float = Z_MIN,
    min_recall: float = MIN_COVERED_RECALL,
) -> tuple[float, float]:
    """(strict recall, midpoint) of the HIGHEST cut (>= tau) at which at least `min_recall` of the
    train present items still count as covered (s >= cut and z >= z_min). Train data only.

    Absent items carry no information about where the band must end (the train absents all score
    below the tau plateau), so the band is sized by how much present recall we can afford to
    send to review."""
    ok = sorted((s for s, z in present if z >= z_min and s >= tau), reverse=True)
    need = max(1, int(np.ceil(min_recall * len(present) - 1e-9)))
    if len(ok) < need:
        return len(ok) / max(len(present), 1), tau
    hi = ok[need - 1]
    lower = [s for s, _ in present if s < hi]
    lo = max(lower) if lower else hi - 0.02
    cut = (lo + hi) / 2
    return sum(s >= cut and z >= z_min for s, z in present) / max(len(present), 1), max(cut, tau)


def loo_scores(
    present: list[Stat], absent: list[Stat], z_min: float = Z_MIN, fitter: Any = None
) -> tuple[float, float]:
    """Leave-one-out (accuracy, false-covered rate of the absent items): each item is judged by a
    tau refitted on all the others, so no item helps decide its own verdict."""
    correct = false_cov = 0
    for kind, rows in (("p", present), ("a", absent)):
        for j, (s, z) in enumerate(rows):
            pp = [r for i, r in enumerate(present) if not (kind == "p" and i == j)]
            aa = [r for i, r in enumerate(absent) if not (kind == "a" and i == j)]
            _, tau, _ = (fitter or fit_tau_conservative)(pp, aa, z_min)
            covered = s >= tau and z >= z_min
            correct += covered if kind == "p" else not covered
            false_cov += covered and kind == "a"
    return correct / (len(present) + len(absent)), false_cov / max(len(absent), 1)


# ---- the shoot -------------------------------------------------------------------------------

_MATRIX: dict[str, tuple[tuple, np.ndarray, np.ndarray]] = {}
_MATRIX_KEEP = 2  # shoots whose embedding matrix stays memoised


def _matrix(cache: Cache) -> tuple[np.ndarray, np.ndarray]:
    """(ids, embeddings) of the shoot's index, memoised until the database changes."""
    key, stamp = str(cache.path), cache.siglip_stamp()
    hit = _MATRIX.get(key)
    if hit and hit[0] == stamp:
        return hit[1], hit[2]
    ids, mat = cache.siglip_matrix()
    while len(_MATRIX) >= _MATRIX_KEEP and key not in _MATRIX:
        _MATRIX.pop(next(iter(_MATRIX)))  # oldest first: bounded, not every shoot forever
    _MATRIX[key] = (stamp, ids, mat)
    return ids, mat


@dataclass
class Coverage:
    results: list[ItemResult]
    indexed_pct: float
    n_candidates: int
    seconds: float = 0.0

    @property
    def must_missing(self) -> list[ItemResult]:
        return [r for r in self.results if r.item.must and r.status == "missing"]

    @property
    def must_uncertain(self) -> list[ItemResult]:
        """Must-haves matched only near the threshold: needs a human look, not "covered"."""
        return [r for r in self.results if r.item.must and r.status == "uncertain"]

    def counts(self) -> dict[str, int]:
        out = {"covered": 0, "uncertain": 0, "weak": 0, "missing": 0, "pending": 0}
        for r in self.results:
            out[r.status] += 1
        return out

    def public(self) -> dict[str, Any]:
        return {
            "items": [r.public() for r in self.results], "indexed_pct": self.indexed_pct,
            "n_candidates": self.n_candidates, "counts": self.counts(),
            "must_missing": [r.item.text for r in self.must_missing],
            "must_uncertain": [r.item.text for r in self.must_uncertain],
            "must_weak": [r.item.text for r in self.results
                          if r.item.must and r.status == "weak"],
            "seconds": round(self.seconds, 3),
        }  # fmt: skip


def _final(row: dict, ovr: dict | None) -> str:
    return ovr["decision"] if ovr else row["decision"]


def evaluate(
    cache: Cache,
    items: list[Item],
    settings: Settings | None = None,
    text_embedder: Any = None,
    tau: float = TAU,
    tau_hi: float | None = None,
) -> Coverage:
    """Coverage of `items` over the keep + review frames of a culled, indexed shoot."""
    t0 = time.time()
    settings = settings or Settings()
    final, n_total, n_done = cache.final_decisions_by_id()
    if not final:
        raise ValueError("this folder has not been culled yet")
    ids, mat = _matrix(cache)
    complete = n_done >= n_total
    pct = 100.0 if complete else min(100.0 * n_done / max(n_total, 1), 99.9)
    cand_ids = {i for i, d in final.items() if d in ("keep", "review")}
    sel = np.fromiter((int(i) in cand_ids for i in ids), bool, len(ids))
    cids, cmat = ids[sel], mat[sel]
    if text_embedder is None and len(cids):
        from shotsieve.models.text_image import get_text

        text_embedder = get_text()
    results = [ItemResult(it) for it in items]
    persons: dict[str, np.ndarray | None | str] = {}
    sims_all = None
    if len(cids):
        Q = np.stack([text_embedder.embed(it.text) for it in items]).astype(np.float32)
        Q /= np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12)
        sims_all = cmat @ Q.T  # (n_frames, n_items)
    for k, (it, res) in enumerate(zip(items, results, strict=True)):
        allowed = np.ones(len(cids), bool)
        if it.person:
            if it.person not in persons:
                try:
                    pq = ParsedQuery("", {"person": [it.person]})
                    persons[it.person] = _filter_ids(cache, pq, settings) or set()
                except ConsentError as exc:
                    persons[it.person] = str(exc)
            got = persons[it.person]
            if isinstance(got, str):
                res.status, res.detail = "pending", got
                continue
            allowed = np.fromiter((int(i) in got for i in cids), bool, len(cids))
        if sims_all is None:
            status, n, top_s, top_z, order, hits = "missing", 0, None, None, [], []
        else:
            status, n, top_s, top_z, order, hits = classify(
                sims_all[:, k], allowed, it.min_count, tau, tau_hi
            )
        res.n_hits, res.top_score, res.top_z = n, top_s, top_z
        res.best_ids = [int(cids[i]) for i in order[:N_BEST]]
        res.hit_ids = [int(cids[i]) for i in hits]
        hi = max(TAU_HI if tau_hi is None else tau_hi, tau)
        res.strong_ids = [int(cids[i]) for i in hits if sims_all[i, k] >= hi]
        res.status = status
        if status != "covered" and not complete:
            res.status = "pending"
            res.detail = f"pending (indexed {pct:.0f} %)"
        elif status == "uncertain":
            res.detail = f"near-threshold match (best s {top_s:.3f}): review the frames"
        elif status == "weak":
            res.detail = f"{n} of {it.min_count} frame(s) clear the cut; best z {top_z:.1f}"
    return Coverage(results, round(pct, 1), len(cids), time.time() - t0)


def store(cache: Cache, cov: Coverage) -> None:
    cache.put_shotlist([
        {
            "item_id": r.item.item_id, "list_name": r.item.list_name, "pos": r.item.pos,
            "text": r.item.text, "must": r.item.must, "min_count": r.item.min_count,
            "person": r.item.person, "status": r.status, "best_ids": r.best_ids,
            "n_hits": r.n_hits, "top_score": r.top_score, "top_z": r.top_z, "detail": r.detail,
        }
        for r in cov.results
    ])  # fmt: skip


def stored_items(cache: Cache) -> list[Item]:
    return [
        Item(r["item_id"], r["text"], r["must"], r["min_count"], r["person"], r["pos"],
             r["list_name"])
        for r in cache.get_shotlist()
    ]  # fmt: skip


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48]


def tag_xmp(cache: Cache, cov: Coverage, limit: int = 50) -> dict[str, int]:
    """Opt-in: merge a `shotlist:<item>` dc:subject keyword into the sidecar of each matched keep
    (a frame with s >= TAU_HI for a covered / weak item; never an uncertain-band frame).
    Foreign keywords are never removed and originals are never modified."""
    decisions, overrides = cache.load_decisions(), cache.overrides()
    final = {p: _final(r, overrides.get(p)) for p, r in decisions.items()}
    kw: dict[str, list[str]] = {}
    for r in cov.results:
        if r.status not in ("covered", "weak"):
            continue
        for i in r.strong_ids[:limit]:  # never a near-threshold (uncertain-band) frame
            p = cache.path_for_id(i)
            if p and final.get(p) == "keep" and "#t=" not in p:
                kw.setdefault(p, []).append(TAG_PREFIX + slug(r.item.text))
    written = failed = 0
    for p, tags in kw.items():
        try:
            written += merge_keywords(Path(p), tags) > 0
        except Exception as exc:  # noqa: BLE001 - one odd sidecar must not abort the rest
            log.warning("shotlist xmp failed for %s: %s", p, exc)
            failed += 1
    return {"images": len(kw), "written": written, "failed": failed}


def run(
    folder: str | Path,
    items: list[Item] | None = None,
    settings: Settings | None = None,
    *,
    tag: bool = False,
    save: bool = True,
    text_embedder: Any = None,
) -> Coverage:
    """Evaluate (and store) a shot list for a shoot; `items=None` re-evaluates the stored list."""
    with Cache.for_folder(folder) as cache:
        its = items if items is not None else stored_items(cache)
        if not its:
            raise ValueError("no shot list yet: load one first")
        cov = evaluate(cache, its, settings, text_embedder)
        if save:
            store(cache, cov)
        if tag:
            tag_xmp(cache, cov)
    return cov
