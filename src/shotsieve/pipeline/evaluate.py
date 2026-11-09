"""Evaluation against ground truth: missed-keeper rate and companion metrics."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from shotsieve.io.lrcat import CatalogEntry


class _HasDecision(Protocol):
    path: Path
    master_path: Path
    decision: str
    group_id: int


def norm_path(p: str | Path) -> str:
    """Case-insensitive, symlink-free absolute path used for matching (macOS and Windows
    volumes are case-insensitive, and `normcase` is a no-op on macOS)."""
    try:
        out = os.path.normcase(str(Path(p).resolve()))
    except OSError:
        out = os.path.normcase(str(p))
    return out.casefold() if sys.platform in ("darwin", "win32") else out


def truth_from_catalog(entries: Iterable[CatalogEntry], min_rating: int = 1) -> dict[str, bool]:
    """Ground truth {path: kept}. Kept = flagged (pick 1) or rating >= min_rating.

    Rejected (pick -1) and unrated/unflagged images count as not kept. Images explicitly
    unrated in a catalog with no picks at all cannot be told apart, so callers should only
    evaluate shoots the user actually culled.
    """
    out: dict[str, bool] = {}
    for e in entries:
        kept = e.pick == 1 or (e.pick >= 0 and e.rating is not None and e.rating >= min_rating)
        out[norm_path(e.path)] = bool(kept)
    return out


def truth_from_xmp(folder: str | Path, settings=None) -> dict[str, bool]:
    """Ground truth from the photographer's own sidecar ratings / labels in `folder`.

    ShotSieve-written values are excluded (see `io.xmp.user_values`): a sidecar that only holds
    ShotSieve's output carries no ground truth and is left out.
    """
    from shotsieve.config import Settings
    from shotsieve.io.formats import scan_folder
    from shotsieve.io.xmp import read_sidecar, user_values

    settings = settings or Settings()
    truth: dict[str, bool] = {}
    for e in scan_folder(folder):
        d = read_sidecar(e.master)
        if d is None:
            continue
        rating, label = user_values(d, settings)
        if rating is None and not label and d.decision is not None:
            continue
        lab = (label or "").lower()
        rejected = (rating is not None and rating < 0) or lab == "red"
        truth[str(e.master)] = not rejected and (
            (rating or 0) >= settings.min_rating or lab == "green"
        )
    return truth


@dataclass
class EvalReport:
    n_images: int = 0
    n_evaluated: int = 0  # images present in the ground truth
    n_kept_truth: int = 0
    missed_keepers: int = 0  # user kept, ShotSieve rejected
    missed_keeper_rate: float = 0.0
    keepers_in_review: int = 0
    keepers_kept: int = 0
    review_fraction: float = 0.0  # over all images
    reject_precision: float = float("nan")  # of rejects, fraction the user also did not keep
    keep_precision: float = float("nan")  # of keeps, fraction the user also kept
    reject_rate: float = 0.0
    counts: dict[str, int] = field(default_factory=dict)
    missed_paths: list[str] = field(default_factory=list)
    # Group-level metrics (the product's success metric). A user-kept frame is only "missed"
    # when ShotSieve kept none of its near-duplicates either: keeping an equally good sibling
    # is correct de-duplication, not a miss.
    group_missed_keepers: int = 0  # user-kept frames whose duplicate group has no ShotSieve keep
    group_missed_keeper_rate: float = 0.0  # / n_kept_truth: STRICT form (review = miss), secondary
    group_gate_rate: float = 0.0  # (group_missed_keepers - group_review_rescuable) / n_kept_truth
    #                               <- the gated metric: review counts as saved
    group_review_rescuable: int = 0  # of those, frames whose group at least has a review
    n_kept_groups: int = 0  # duplicate groups holding at least one user keep
    moment_misses: int = 0  # ... of which have no ShotSieve keep
    moment_miss_rate: float = 0.0
    moment_review_rescuable: int = 0


def _dec(im: _HasDecision) -> str:
    """The model's decision (not a user override that may have been applied on top)."""
    return getattr(im, "model_decision", "") or im.decision


def evaluate(images: Iterable[_HasDecision], truth: Mapping[str, bool]) -> EvalReport:
    """Compare decisions with ground truth {path: user_kept}.

    Truth keys are matched by `norm_path` against master_path and path; pass keys through
    `norm_path` (as `truth_from_catalog` does) or use plain resolved absolute paths.
    """
    t = {norm_path(k): v for k, v in truth.items()}
    rep = EvalReport(counts={"keep": 0, "reject": 0, "review": 0})
    rejects_total = rejects_ok = keeps_total = keeps_ok = 0
    images = list(images)
    has: dict[int, set[str]] = {}
    for k, im in enumerate(images):
        # an audit review is a model *reject* shown for sampling, so it never rescues a group (N3)
        d = _dec(im)
        if d == "review" and getattr(im, "audit", False):
            d = "audit_review"
        has.setdefault(getattr(im, "group_id", -1 - k), set()).add(d)
    kept_groups: dict[int, int] = {}
    for k, im in enumerate(images):
        gid = getattr(im, "group_id", -1 - k)
        im_key = next((q for q in (norm_path(im.master_path), norm_path(im.path)) if q in t), None)
        if im_key is not None and t[im_key]:
            kept_groups[gid] = kept_groups.get(gid, 0) + 1
            if "keep" not in has[gid]:
                rep.group_missed_keepers += 1
                rep.group_review_rescuable += "review" in has[gid]
    rep.n_kept_groups = len(kept_groups)
    for gid in kept_groups:
        if "keep" not in has[gid]:
            rep.moment_misses += 1
            rep.moment_review_rescuable += "review" in has[gid]
    for im in images:
        rep.n_images += 1
        rep.counts[_dec(im)] = rep.counts.get(_dec(im), 0) + 1
        key = next((k for k in (norm_path(im.master_path), norm_path(im.path)) if k in t), None)
        if key is None:
            continue
        rep.n_evaluated += 1
        kept = t[key]
        if _dec(im) == "reject":
            rejects_total += 1
            rejects_ok += not kept
        elif _dec(im) == "keep":
            keeps_total += 1
            keeps_ok += kept
        if kept:
            rep.n_kept_truth += 1
            if _dec(im) == "reject":
                rep.missed_keepers += 1
                rep.missed_paths.append(str(im.master_path))
            elif _dec(im) == "review":
                rep.keepers_in_review += 1
            else:
                rep.keepers_kept += 1
    if rep.n_kept_truth:
        rep.missed_keeper_rate = rep.missed_keepers / rep.n_kept_truth
        rep.group_missed_keeper_rate = rep.group_missed_keepers / rep.n_kept_truth
        rep.group_gate_rate = (
            rep.group_missed_keepers - rep.group_review_rescuable
        ) / rep.n_kept_truth
    if rep.n_kept_groups:
        rep.moment_miss_rate = rep.moment_misses / rep.n_kept_groups
    if rep.n_images:
        rep.review_fraction = rep.counts["review"] / rep.n_images
        rep.reject_rate = rep.counts["reject"] / rep.n_images
    if rejects_total:
        rep.reject_precision = rejects_ok / rejects_total
    if keeps_total:
        rep.keep_precision = keeps_ok / keeps_total
    return rep
