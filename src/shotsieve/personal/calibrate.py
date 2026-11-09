"""Audit-based threshold recalibration (IPW estimate of the missed-keeper rate).

The audit is stratified by *why* a frame was rejected, and each stratum has its own lever:

    score  (low_quality / no_subject)   t_reject   lowered
    hard   (firm / soft hard fails)     hard_margin raised: failed frames near the review bar
                                         go to review
    dup    (duplicates, failed siblings) dup_margin raised: frames within the margin of their
                                         group's best go to review
    certain (black frame, extreme blur, flash misfire) no lever, reported only
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from shotsieve.config import Settings

MISSED_RATE_LIMIT = 0.01
STEP = 0.05
MIN_T_REJECT = 0.10
MAX_MARGIN = 0.30
# dup lever: burst siblings sit within ~0.03 of their best, so a coarse step sweeps in whole
# groups. Fine, proportional steps, and the review share is bounded per round (see calibrate).
DUP_STEP = 0.01
DUP_MAX_STEP = 0.05
MAX_REVIEW = 0.20  # hard ceiling on the total review share
MAX_REVIEW_GROWTH = 0.10  # cap on how much one round may add (absolute share)
STRATA = ("score", "hard", "dup", "certain")
LEVER = {"score": "t_reject", "hard": "hard_margin", "dup": "dup_margin"}


@dataclass
class AuditSample:
    audit_prob: float  # inclusion probability recorded when the image was sampled
    kept: bool  # the user kept (did not agree to reject) this audited reject
    stratum: str = "score"  # why it was rejected: score | hard | dup | certain


@dataclass
class CalibrationResult:
    thresholds: dict[str, float]
    est_missed_keepers: float = 0.0
    missed_keeper_rate: float = 0.0
    n_audited: int = 0
    n_kept: int = 0
    n_pending: int = 0
    lowered: bool = False
    details: dict[str, float] = field(default_factory=dict)
    by_stratum: dict[str, dict[str, float]] = field(default_factory=dict)
    rate_ci: tuple[float, float] = (0.0, 0.0)  # approx. 95 % interval of missed_keeper_rate
    applied: bool = False  # thresholds were persisted (needs audited rejects, none pending)

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_missed_keepers(samples: Sequence[AuditSample]) -> float:
    """Horvitz-Thompson estimate of the keepers hiding in the reject bucket: sum of 1/pi over
    the audited rejects the user kept."""
    return float(sum(1.0 / max(s.audit_prob, 1e-6) for s in samples if s.kept))


def missed_keeper_rate(est_missed: float, n_keepers_found: int) -> float:
    """Estimated missed keepers as a share of all keepers (found + missed)."""
    total = est_missed + n_keepers_found
    return est_missed / total if total > 0 else 0.0


def lower_t_reject(t_reject: float, rate: float) -> float:
    """Lower t_reject in 0.05 steps (one step, plus one per further 5 points of estimated
    rate), never below 0.10."""
    steps = 1 + int(rate // STEP)
    return max(MIN_T_REJECT, round(t_reject - STEP * steps, 4))


def _rate_ci(samples: Sequence[AuditSample], n_found: int, z: float = 1.96) -> tuple[float, float]:
    """Rough 95 % interval (Poisson variance of the HT sum, delta method on the rate)."""
    est = estimate_missed_keepers(samples)
    var = sum((1.0 - s.audit_prob) / max(s.audit_prob, 1e-6) ** 2 for s in samples if s.kept)
    sd = var**0.5
    return (
        missed_keeper_rate(max(est - z * sd, 0.0), n_found),
        missed_keeper_rate(est + z * sd, n_found),
    )


def calibrate(
    samples: Sequence[AuditSample],
    n_keepers_found: int,
    thresholds: dict[str, float],
    n_pending: int = 0,
    simulate: Callable[[dict[str, float]], float] | None = None,
) -> CalibrationResult:
    """Estimate the missed-keeper rate; when it exceeds 1 %, move the lever of every stratum in
    which audited rejects turned out to be keepers (t_reject down, hard/dup margins up).

    The dup step is proportional to the stratum's share of keepers (DUP_STEP..DUP_MAX_STEP). When
    `simulate(thresholds) -> review share` is given it measures the review share before the lever
    moves; `dup_review_budget` (<= MAX_REVIEW_GROWTH and <= MAX_REVIEW - share) then caps how many
    frames the margin may send to review."""
    est = estimate_missed_keepers(samples)
    rate = missed_keeper_rate(est, n_keepers_found)
    new = dict(thresholds)
    new.setdefault("hard_margin", 0.0)
    new.setdefault("dup_margin", 0.0)
    new.setdefault("dup_review_budget", 1.0)
    by: dict[str, dict[str, float]] = {}
    for st in STRATA:
        sub = [s for s in samples if s.stratum == st]
        e = estimate_missed_keepers(sub)
        by[st] = {
            "n_audited": float(len(sub)), "n_kept": float(sum(s.kept for s in sub)),
            "est_missed": e, "share_of_keepers": missed_keeper_rate(e, n_keepers_found + est - e),
        }  # fmt: skip
    lowered = False
    details: dict[str, float] = {"n_keepers_found": float(n_keepers_found)}
    if rate > MISSED_RATE_LIMIT:
        for st, lever in LEVER.items():
            if by[st]["est_missed"] <= 0:
                continue
            steps = 1 + int(by[st]["share_of_keepers"] // STEP)
            if st == "dup":
                cur = float(thresholds.get(lever, 0.0))
                step = min(DUP_MAX_STEP, max(DUP_STEP, round(by[st]["share_of_keepers"] / 4, 2)))
                new[lever] = min(MAX_MARGIN, round(cur + step, 4))
                details["dup_margin_proposed"] = new[lever]
                # bound: the margin may add at most MAX_REVIEW_GROWTH of review this round and
                # never push the total past MAX_REVIEW (decide() keeps the closest siblings)
                budget = MAX_REVIEW_GROWTH
                if simulate is not None:
                    r0 = simulate({**new, "dup_margin": cur, "dup_review_budget": 1.0})
                    budget = min(budget, max(0.0, MAX_REVIEW - r0))
                    details["review_before"] = r0
                new["dup_review_budget"] = round(budget, 4)
                if budget <= 0:
                    new[lever] = cur
                lowered |= new[lever] > cur
            elif st == "score":
                t = lower_t_reject(float(thresholds["t_reject"]), by[st]["share_of_keepers"])
                lowered |= t < thresholds["t_reject"]
                new["t_reject"] = t
            else:
                m = min(MAX_MARGIN, round(float(thresholds.get(lever, 0.0)) + STEP * steps, 4))
                lowered |= m > float(thresholds.get(lever, 0.0))
                new[lever] = m
    return CalibrationResult(
        thresholds=new,
        est_missed_keepers=est,
        missed_keeper_rate=rate,
        n_audited=len(samples),
        n_kept=sum(s.kept for s in samples),
        n_pending=n_pending,
        lowered=lowered,
        details=details,
        by_stratum=by,
        rate_ci=_rate_ci(samples, n_keepers_found),
    )


def _simulator(cache, folder: str | Path, settings: Settings):
    """Review share of the shoot under candidate thresholds, from cached features (no decode)."""
    from shotsieve.pipeline.runner import run_cull

    def simulate(th: dict[str, float]) -> float:
        cache.set_thresholds(
            {k: v for k, v in th.items() if k in ("hard_margin", "dup_margin", "dup_review_budget")}
        )
        st = run_cull(Path(folder), settings).stats
        return st["counts"].get("review", 0) / max(st["n_images"], 1)

    return simulate


def calibrate_shoot(folder: str | Path, settings: Settings, redecide: bool = True):
    """Calibrate a shoot from its audit decisions in cull.db, persist the thresholds and (by
    default) re-decide from cached features. Returns (CalibrationResult, CullResult | None).

    Levers are always derived from the *default* thresholds and the full audit sample, so
    calling this twice gives the same answer (it does not ratchet). Audited rejects the user has
    not judged yet are counted in `n_pending`; the thresholds are only changed once none is
    pending, because an unjudged audit item would otherwise count as "not a keeper". With
    `redecide`, candidate dup margins are simulated on the cached features so one round can add at
    most MAX_REVIEW_GROWTH of review and never exceed MAX_REVIEW (the simulation re-decides, so the
    stored decisions are refreshed at the end).
    """
    from shotsieve.pipeline.cache import Cache

    with Cache.for_folder(folder) as cache:
        rows = cache.load_decisions()
        ovr = cache.overrides()
        base = {
            "t_keep": settings.t_keep, "t_reject": settings.t_reject,
            "hard_margin": 0.0, "dup_margin": 0.0, "dup_review_budget": 1.0,
        }  # fmt: skip
        samples: list[AuditSample] = []
        pending = found = 0
        frozen = cache.get_audit_sample()
        for path, row in rows.items():
            o = ovr.get(path)
            final = o["decision"] if o else row["decision"]
            if frozen is not None:  # frozen sample ⨝ overrides: survives re-decides
                fz = frozen.get(path)
                if fz is None:
                    found += final == "keep"
                elif o is None or o["decision"] == "review":
                    pending += 1
                else:
                    samples.append(
                        AuditSample(float(fz.get("p") or 1.0), o["decision"] == "keep",
                                    fz.get("k") or "score")
                    )  # fmt: skip
            elif row["audit"]:
                if o is None or o["decision"] == "review":
                    pending += "audit" in row["reasons"]
                else:
                    samples.append(
                        AuditSample(
                            float(row["audit_prob"] or 1.0),
                            o["decision"] == "keep",
                            row.get("klass") or "score",
                        )
                    )
            elif final == "keep":
                found += 1
        if not samples:  # nothing audited: report persisted thresholds, leave calibration alone
            persisted = {**base, **(cache.get_thresholds() or {})}
            empty = CalibrationResult(thresholds=persisted, n_pending=pending, applied=False)
            return empty, None
        sim = _simulator(cache, folder, settings) if redecide and pending == 0 else None
        result = calibrate(samples, found, base, pending, simulate=sim)
        result.applied = pending == 0
        if result.applied:
            # only the levers calibration moved (the rest may come from the personal model)
            cache.set_thresholds({k: v for k, v in result.thresholds.items() if v != base.get(k)})
        cache.set_meta("calibration", result.to_dict())
    cull = None
    if redecide and result.applied and (result.lowered or sim is not None):
        from shotsieve.pipeline.runner import run_cull

        cull = run_cull(Path(folder), settings)
    return result, cull
