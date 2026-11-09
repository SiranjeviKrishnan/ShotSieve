"""Best-pick per duplicate group and keep / reject / review decisions."""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from shotsieve.config import Settings
from shotsieve.features import banding
from shotsieve.features.vector import FeatureVector
from shotsieve.scoring.quality import FIRM, SOFT, QualityResult, weakest_area


@runtime_checkable
class PersonalModelLike(Protocol):
    """What the engine needs from a personal model (implemented in `shotsieve.personal`).

    `predict_proba(X)` receives the (N, len(FEATURE_NAMES)) feature matrix (FEATURE_NAMES order)
    and returns either a (N,) vector of P(keep) or sklearn-style (N, 2) probabilities.
    If the signature also accepts an `embeddings` keyword, the (N, 384) DINOv2 scene embeddings
    are passed as well (for PCA features). `n_train` is the number of training rows.
    """

    n_train: int

    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


def blend_weight(n_train: int, prior_n: int = 200) -> float:
    """w = n / (n + prior_n): how much to trust the personal model."""
    return n_train / (n_train + prior_n) if n_train > 0 else 0.0


def personal_probabilities(
    model: PersonalModelLike, X: np.ndarray, embeddings: np.ndarray | None = None
) -> np.ndarray:
    """P(keep) from a personal model as a (N,) float array."""
    kwargs: dict[str, Any] = {}
    try:
        if (
            embeddings is not None
            and "embeddings" in inspect.signature(model.predict_proba).parameters
        ):
            kwargs["embeddings"] = embeddings
    except (TypeError, ValueError):
        pass
    p = np.asarray(model.predict_proba(X, **kwargs), dtype=np.float64)
    if p.ndim == 2:
        p = p[:, -1]
    return p.reshape(-1)


def blend(prior: np.ndarray, personal: np.ndarray | None, n_train: int, prior_n: int = 200):
    """p = w * personal + (1 - w) * prior with w = n / (n + prior_n)."""
    if personal is None or n_train <= 0:
        return np.asarray(prior, dtype=np.float64)
    w = blend_weight(n_train, prior_n)
    return w * personal + (1.0 - w) * np.asarray(prior, dtype=np.float64)


@dataclass
class Candidate:
    name: str  # file name (used in 'duplicate' explanations and for stable ordering)
    group_id: int
    fv: FeatureVector
    quality: QualityResult
    p: float  # final keep probability (blend of personal model and prior)
    flash_delta_ev: float = float("nan")


@dataclass
class Decision:
    decision: str  # keep | reject | review
    reasons: list[str] = field(default_factory=list)
    ctx: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    is_best: bool = False
    rank: int = 0  # 0 = best within its group
    klass: str = ""  # reject stratum (see reject_class)


def keep_count(group_size: int, settings: Settings) -> int:
    return max(settings.keep_per_group, 2) if group_size >= 8 else settings.keep_per_group


def rank_groups(
    group_ids: list[int], scores: list[float], failed: list[bool], names: list[str]
) -> tuple[list[int], list[int]]:
    """Per-image (rank within group, group size). Hard-failed images rank after healthy ones."""
    members: dict[int, list[int]] = {}
    for i, g in enumerate(group_ids):
        members.setdefault(g, []).append(i)
    rank = [0] * len(group_ids)
    size = [1] * len(group_ids)
    for idxs in members.values():
        order = sorted(idxs, key=lambda i: (failed[i], -scores[i], names[i]))
        for r, i in enumerate(order):
            rank[i] = r
            size[i] = len(idxs)
    return rank, size


HARD_MARGIN_MAX = 0.30
DUP_MARGIN_MAX = 0.30


def reject_class(d: Decision, cand: Candidate) -> str:
    """Audit stratum of a reject: certain | hard | dup | score ('' for non-rejects)."""
    if d.decision != "reject":
        return ""
    if cand.quality.hard_fails:
        return "certain" if cand.quality.certain else "hard"
    if d.reasons and d.reasons[0] in ("duplicate", "duplicate_tie"):
        return "dup"
    return "score"


def decide(
    cands: list[Candidate],
    settings: Settings,
    t_keep: float,
    t_reject: float,
    *,
    hard_margin: float = 0.0,
    dup_margin: float = 0.0,
    dup_review_budget: float = 1.0,
    trained: bool = False,
) -> list[Decision]:
    """Decisions aligned with `cands`. Every reject carries at least one reason code.

    Levers (the ones a per-shoot calibration moves):
      t_reject    - low_quality / no_subject rejects below it (score stratum)
      hard_margin - failed frames (firm / soft tiers, never certain) whose keep-probability is
                    within the margin of the review bar go to review instead of reject
      dup_margin  - duplicates (and failed siblings) within the margin of their group's best go
                    to review instead of reject
      dup_review_budget - at most this share of all frames may be moved to review by dup_margin;
                    the closest siblings to their best win (bounds the lever, which is a cliff on
                    burst frames that are near-identical)
    `trained` says a personal model produced p, which lets a firm fail (e.g. both eyes closed)
    be overridden once p >= t_keep.
    """
    failed = [bool(c.quality.hard_fails) for c in cands]
    rank, size = rank_groups(
        [c.group_id for c in cands], [c.p for c in cands], failed, [c.name for c in cands]
    )
    best_of: dict[int, int] = {}
    for i, c in enumerate(cands):
        if rank[i] == 0:
            best_of[c.group_id] = i
    t_soft = settings.t_soft - hard_margin
    t_firm = t_keep - hard_margin

    def overridable(c: Candidate) -> bool:
        """May the model / calibration move this failed frame from reject to review?"""
        tier = c.quality.tier
        if tier == SOFT:
            return c.p >= t_soft
        if tier == FIRM:
            return (trained or hard_margin > 0) and c.p >= t_firm
        return False

    out: list[Decision] = []
    margin_hits: list[tuple[float, str, int]] = []  # (gap to best, name, index)
    for i, c in enumerate(cands):
        d = Decision("review", score=c.p, rank=rank[i], is_best=rank[i] == 0)
        ctx = {"flash_delta_ev": c.flash_delta_ev}
        best_i = best_of[c.group_id]
        best = cands[best_i]
        has_better = size[i] > 1 and rank[i] >= keep_count(size[i], settings)
        if failed[i]:
            d.reasons = list(c.quality.hard_fails)
            if d.reasons == ["banding"]:
                # review, unless the banding is severe and a clean sibling exists in the group
                clean_alt = has_better and not failed[best_i]
                severe = _nan0(c.fv.banding_amp_ev) >= banding.AMP_EV_SEVERE
                # scene stripes (blinds, fences, textiles) are never rejected: only broad,
                # flicker-like bands, and only against a sibling that was measured and is clean
                # (an unmeasured sibling, e.g. a slow shutter, proves nothing)
                severe = severe and c.quality.flicker
                clean_alt = clean_alt and not math.isnan(best.fv.banding_amp_ev)
                if clean_alt:
                    ctx["best"] = best.name
                d.decision = "reject" if (severe and clean_alt) else "review"
            elif has_better and not failed[best_i]:
                ctx["best"] = best.name
                # a failed sibling of a healthy best: reject unless the model likes it or it
                # sits within the dup margin of the best (calibration lever)
                near = dup_margin > 0 and c.p >= best.p - dup_margin
                ok = (near or overridable(c)) and not c.quality.certain
                d.decision = "review" if ok else "reject"
                if ok and not overridable(c):
                    margin_hits.append((best.p - c.p, c.name, i))
            else:
                # a singleton is also the only copy of its moment: a firm fail the model rates
                # at or above t_soft goes to review, like the moment safeguard (N4)
                lone_firm = size[i] == 1 and c.quality.tier == FIRM and c.p >= t_soft
                d.decision = "review" if (overridable(c) or lone_firm) else "reject"
        elif has_better and best.p >= t_reject:
            ctx["best"] = best.name
            tie = c.p >= best.p - settings.tie_margin
            d.reasons = ["duplicate_tie" if tie else "duplicate"]
            if dup_margin > 0 and c.p >= best.p - dup_margin:
                d.decision = "review"  # within the calibrated margin of the best frame
                if not (tie and t_reject <= best.p < t_keep and not failed[best_i]):
                    margin_hits.append((best.p - c.p, c.name, i))  # review only by the margin
            elif tie:
                # Near-identical to the best frame: a clear duplicate, unless the best frame
                # itself is borderline, in which case the user should look at the pair.
                if t_reject <= best.p < t_keep and not failed[best_i]:
                    d.decision = "review"
                    ctx["best_uncertain"] = True
                else:
                    d.decision = "reject"
            else:
                d.decision = "reject"
        elif c.p >= t_keep:
            d.decision = "keep"
        elif c.p < t_reject:
            ctx["weakest"] = weakest_area(c.quality.subs)
            d.decision = "reject"
            d.reasons = ["no_subject" if c.fv.n_faces <= 0 else "low_quality"]
        else:
            d.reasons = ["uncertain"]
        if d.decision == "reject" and size[i] > 1 and rank[i] == 0 and not c.quality.certain:
            # Moment safeguard: the best frame of a duplicate group is the user's only copy of
            # that moment. Only a CERTAIN fail (black, extreme blur, misfire) may drop it; any
            # other reject becomes a soft review that says why the frame is weak.
            ctx["issue"] = d.reasons[0] if d.reasons else "low_quality"
            ctx["weakest"] = ctx.get("weakest") or weakest_area(c.quality.subs)
            d.reasons = ["best_available", *d.reasons]
            d.decision = "review"
        d.ctx = ctx
        out.append(d)
    allowed = int(max(dup_review_budget, 0.0) * len(cands))
    for _gap, _name, i in sorted(margin_hits)[allowed:]:
        out[i].decision = "reject"  # over budget: the farthest siblings stay rejected
    for d, c in zip(out, cands, strict=True):
        d.klass = reject_class(d, c)
    return out


def _nan0(x: float) -> float:
    return 0.0 if x is None or math.isnan(x) else x


def is_finite(x: float) -> bool:
    return x is not None and not math.isnan(x)
