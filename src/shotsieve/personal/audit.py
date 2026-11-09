"""Propensity sampling of the reject bucket into the review queue (audit)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from shotsieve.scoring import reasons as reason_text

AUDIT_MIN = 5
AUDIT_MAX = 25
AUDIT_FRACTION = 0.03
SCORE_FLOOR = 0.05


def audit_size(n_rejects: int) -> int:
    """audit_n = min(25, max(5, 3% of rejects)), never more than the bucket itself."""
    if n_rejects <= 0:
        return 0
    return min(n_rejects, AUDIT_MAX, max(AUDIT_MIN, math.ceil(AUDIT_FRACTION * n_rejects)))


def inclusion_probabilities(weights: np.ndarray, n: int) -> np.ndarray:
    """First-order inclusion probabilities of a size-n PPS sample: pi_i = min(1, c * w_i) with
    c chosen so that sum(pi) == n (certainty units are capped at 1 and the rest rescaled)."""
    w = np.asarray(weights, dtype=np.float64)
    pi = np.zeros_like(w)
    certain = np.zeros(len(w), bool)
    for _ in range(len(w) + 1):
        free = ~certain
        remaining = n - certain.sum()
        if remaining <= 0 or not free.any():
            break
        pi[free] = remaining * w[free] / w[free].sum()
        over = free & (pi >= 1.0)
        if not over.any():
            break
        certain |= over
        pi[over] = 1.0
    pi[certain] = 1.0
    return np.clip(pi, 0.0, 1.0)


def _systematic_pps(pi: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Systematic sampling in a random order: P(i selected) == pi_i exactly, sample size sum(pi)."""
    order = rng.permutation(len(pi))
    cum = np.cumsum(pi[order])
    start = rng.uniform(0.0, 1.0)
    n = int(round(cum[-1]))
    marks = start + np.arange(n)
    idx = np.searchsorted(cum, marks, side="right")
    chosen = np.zeros(len(pi), bool)
    chosen[order[np.clip(idx, 0, len(pi) - 1)]] = True
    return chosen


def mark_audit(r: Any, prob: float) -> None:
    """Turn a reject into an audit item: review, with its inclusion probability and the
    original reject reasons kept behind an 'audit' reason."""
    old_codes, old_texts = list(r.reasons), list(r.reason_texts)
    if len(old_texts) != len(old_codes):
        old_texts = [reason_text.describe(c, r.features) for c in old_codes]
    r.decision, r.audit, r.audit_prob = "review", True, float(prob)
    r.reasons = ["audit", *old_codes]
    r.reason_texts = [reason_text.describe("audit"), *old_texts]


def sample_audit(images: list[Any], seed: int = 0) -> list[Any]:
    """Pick rejects for a spot-check; mutates the chosen results in place.

    Sampling probability is proportional to `score + 0.05` (rejects closest to the boundary are
    likelier). Each chosen image becomes decision='review', audit=True with its inclusion
    probability in `audit_prob`; the original reject reasons are kept behind an 'audit' reason.
    Deterministic for a given seed and set of rejects. Returns the chosen images.
    """
    pool = sorted(
        (
            r
            for r in images
            if r.decision == "reject" and not r.audit and not getattr(r, "overridden", False)
        ),
        key=lambda r: str(r.master_path),
    )
    n = audit_size(len(pool))
    if n == 0:
        return []
    weights = np.array([max(float(r.score), 0.0) + SCORE_FLOOR for r in pool])
    pi = inclusion_probabilities(weights, n)
    rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, len(pool)])
    chosen = _systematic_pps(pi, rng)
    picked = []
    for r, p, take in zip(pool, pi, chosen, strict=True):
        if not take:
            continue
        mark_audit(r, float(p))
        picked.append(r)
    return picked
