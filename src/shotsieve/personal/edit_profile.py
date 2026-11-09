"""EditProfile: per-slider models of the photographer's own develop settings (T15).

Per slider: HistGradientBoosting point model + q10/q90 models on edit statistics, a DINOv2
PCA-24 and the format class. Exposure is chained into the tone and white-balance sliders
(out-of-fold at training time). Output is shrunk towards the user's median, w = n / (n + 100).

Train-time mode per slider comes from 5-fold CV grouped by shoot folder (bursts must not leak):
`learned` if MAE_cv <= 0.85 x MAE_median, else `constant` (write the median) when the user's IQR
is within tolerance, else `off`. Per image a two-part gate decides whether a learned slider is
written: (a) the interval width (calibrated q10-q90, split-conformal so empirical coverage is
~0.80) <= its CV 80th percentile; (b) the image's DINOv2 is not out of distribution (mean cosine
distance to its 5 nearest training rows <= the CV p95).
"""

from __future__ import annotations

import hashlib
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from shotsieve.config import Settings, data_dir
from shotsieve.features.color_stats import EDIT_STAT_NAMES, EDIT_STATS_VERSION
from shotsieve.log import get_logger

log = get_logger("edit_profile")

PROFILE_FORMAT = 1
MIN_BODY_WB = 40  # training rows with a WB target a camera body needs before its WB is written
WB_BODY_NOTE = "different body: white balance not matched"
_CAM_COL = EDIT_STAT_NAMES.index("camera_idx")


def body_key(stats: np.ndarray) -> np.ndarray:
    """Camera-body bucket of each stats row (-1 = no EXIF make/model)."""
    c = np.asarray(stats, np.float64)[:, _CAM_COL]
    return np.where(np.isfinite(c), c, -1.0).astype(int)
MIN_EDITS = 200
N_FOLDS = 5
PCA_DIM = 24
BANK_CAP = 3000
KNN = 5
LEARNED_RATIO = 0.85
WIDTH_Q = 0.80
OOD_Q = 0.95
QUANTILE_LR = 0.1  # q10/q90 models: lr 0.1, <= 150 iterations (speed and the 10 MB file cap)
QUANTILE_ITERS = 150
CHANGED_MIN_FRAC = 0.05
MIN_KEY_ROWS = 40

TONE_KEYS = (
    "Exposure2012", "Contrast2012", "Highlights2012", "Shadows2012", "Whites2012", "Blacks2012",
    "Vibrance", "Saturation",
)  # fmt: skip
WB_KEYS = ("raw_temp", "raw_tint", "jpeg_temp", "jpeg_tint")
ALL_KEYS = TONE_KEYS + WB_KEYS
TOL = {"Exposure2012": 0.15, "raw_temp": 10.0, "jpeg_temp": 10.0, "raw_tint": 5.0, "jpeg_tint": 5.0}
DEFAULT_TOL = 5.0
COUPLED_TO_EXPOSURE = ("Highlights2012", "Shadows2012", "Whites2012", "Blacks2012")
FMT_RAW, FMT_DNG, FMT_OTHER = 0, 1, 2  # format class column


class InsufficientEditsError(ValueError):
    """Fewer than MIN_EDITS edited frames."""


def wb_group(fmt: np.ndarray | int) -> np.ndarray | str:
    """Head group of a format class: raw+DNG share the as-shot-relative head."""
    if isinstance(fmt, np.ndarray):
        return np.where(fmt <= FMT_DNG, 0, 1)
    return "raw" if fmt <= FMT_DNG else "jpeg"


@dataclass
class EditSet:
    """Training rows: statistics, embeddings, format class, targets (NaN = not applicable)."""

    stats: np.ndarray  # (n, len(EDIT_STAT_NAMES))
    emb: np.ndarray  # (n, 384)
    fmt: np.ndarray  # (n,) int
    y: dict[str, np.ndarray]  # key -> (n,) float
    groups: np.ndarray  # (n,) int shoot-folder ids
    carry: dict[str, float | str] = field(default_factory=dict)
    paths: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    notes: dict[str, int] = field(default_factory=dict)  # skipped counts, etc.

    def __len__(self) -> int:
        return len(self.stats)


@dataclass
class SliderModel:
    key: str
    mode: str = "off"  # learned | constant | off
    n: int = 0
    median: float = 0.0
    iqr: float = 0.0
    mae_cv: float = float("nan")
    mae_median: float = float("nan")
    tol: float = DEFAULT_TOL
    q_cal: float = 0.0  # conformal widening of [q10, q90] on each side
    width_thr: float = float("inf")
    point: Any = None
    lo: Any = None
    hi: Any = None
    clf: Any = None  # WB heads: P(user changes WB)
    p_changed: float = 0.0


@dataclass
class Prediction:
    point: dict[str, float] = field(default_factory=dict)  # every non-off slider, ungated
    lo: dict[str, float] = field(default_factory=dict)
    hi: dict[str, float] = field(default_factory=dict)
    written: dict[str, bool] = field(default_factory=dict)  # after gate and couplings
    conf: dict[str, float] = field(default_factory=dict)
    modes: dict[str, str] = field(default_factory=dict)
    omitted: list[str] = field(default_factory=list)  # learned sliders withheld by the gate
    notes: list[str] = field(default_factory=list)  # e.g. "different body: white balance ..."
    ood: bool = False
    needs_edit: bool = False

    def values(self) -> dict[str, float]:
        return {k: v for k, v in self.point.items() if self.written.get(k)}


class _LinHGB:
    """Ridge (on median-imputed, standardised columns) + HistGradientBoosting on its residuals.
    Output shrinkage: ridge + w x boosted residual, w = n / (n + 100).

    The ridge stage carries the smooth, near-linear part of a slider (exposure vs. luminance
    statistics) so that ~200 edits suffice; boosting adds the non-linear and scene-dependent
    part. `quantile` fits the q-th quantile of the residuals (loss="quantile")."""

    def __init__(self, loss: str = "squared_error", q: float | None = None) -> None:
        self.loss, self.q = loss, q
        self.lin = make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            RidgeCV(alphas=np.logspace(-1, 3.5, 10)),
        )  # fmt: skip
        kw: dict[str, Any] = {} if q is None else {"quantile": q}
        self.gb = HistGradientBoostingRegressor(
            loss=loss, max_iter=300 if q is None else QUANTILE_ITERS,
            learning_rate=0.05 if q is None else QUANTILE_LR, max_leaf_nodes=15,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=12, random_state=0, **kw,
        )  # fmt: skip

    def fit(self, X: np.ndarray, y: np.ndarray) -> _LinHGB:
        self.lin.fit(X, y)
        self.gb.fit(X, y - self.lin.predict(X))
        # shrinkage w = n / (n + 100) on the boosted part (point models); the quantile offsets
        # are not shrunk
        self.w = 1.0 if self.q is not None else _shrink_w(len(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.lin.predict(X) + self.w * self.gb.predict(X)


def _hgb(loss: str = "squared_error", q: float | None = None) -> _LinHGB:
    return _LinHGB(loss, q)


def assign_folds(groups: np.ndarray, k: int = N_FOLDS, seed: int = 0) -> np.ndarray:
    """Fold per row, whole groups (shoot folders) together, sizes balanced greedily. One group:
    contiguous chunks."""
    groups = np.asarray(groups)
    n = len(groups)
    uniq, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    if len(uniq) < 2:
        return np.minimum(np.arange(n) * k // max(n, 1), k - 1)
    k = min(k, len(uniq))
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(len(uniq)), -counts))
    load = np.zeros(k)
    fold_of = np.zeros(len(uniq), dtype=int)
    for g in order:
        f = int(np.argmin(load))
        fold_of[g] = f
        load[f] += counts[g]
    return fold_of[inv]


def _ood_dist(q: np.ndarray, bank: np.ndarray) -> np.ndarray:
    """Mean cosine distance of each row of `q` to its KNN nearest rows of `bank` (unit rows)."""
    out = np.empty(len(q), dtype=np.float64)
    k = min(KNN, len(bank))
    for i in range(0, len(q), 1024):
        d = 1.0 - q[i : i + 1024] @ bank.T
        out[i : i + 1024] = np.partition(d, k - 1, axis=1)[:, :k].mean(axis=1)
    return out


def _bank(emb: np.ndarray, seed: int = 0) -> np.ndarray:
    if len(emb) <= BANK_CAP:
        return emb.astype(np.float32)
    idx = np.random.default_rng(seed).choice(len(emb), BANK_CAP, replace=False)
    return emb[np.sort(idx)].astype(np.float32)


def _shrink_w(n: int) -> float:
    """Weight of the boosted part: w = n / (n + 100)."""
    return n / (n + 100.0)


def default_profile_path(settings: Settings | None = None) -> Path:
    return data_dir() / "edit_profile.joblib"


@dataclass
class EditProfile:
    n: int = 0
    trained: bool = False
    pca: PCA | None = None
    sliders: dict[str, SliderModel] = field(default_factory=dict)
    bank: np.ndarray | None = None
    ood_thr: float = float("inf")
    carry: dict[str, float | str] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    profile_hash: str = ""
    stats_version: int = EDIT_STATS_VERSION
    cols: np.ndarray | None = None  # design columns with >= 2 distinct values in training
    # "<head group>:<body bucket>" -> training rows with a white-balance target
    wb_bodies: dict[str, int] = field(default_factory=dict)

    # ---- design -------------------------------------------------------------------------
    def _design(self, stats: np.ndarray, emb: np.ndarray, fmt: np.ndarray) -> np.ndarray:
        stats = np.asarray(stats, np.float32)
        if stats.shape[1] != len(EDIT_STAT_NAMES):
            raise ValueError(f"expected {len(EDIT_STAT_NAMES)} statistics, got {stats.shape[1]}")
        z = self.pca.transform(np.asarray(emb, np.float32)).astype(np.float32)
        X = np.hstack([stats, z, np.asarray(fmt, np.float32)[:, None]])
        return X if self.cols is None else X[:, self.cols]

    # ---- training -----------------------------------------------------------------------
    def fit(
        self, s: EditSet, progress: Callable[[str, int, int], None] | None = None
    ) -> EditProfile:
        n = len(s)
        if n < MIN_EDITS:
            raise InsufficientEditsError(
                f"need {MIN_EDITS} edits, have {n}; use `match` instead"
            )
        t0 = time.time()
        emb = np.asarray(s.emb, np.float32)
        self.pca = PCA(n_components=min(PCA_DIM, n - 1), random_state=0).fit(emb)
        self.cols = None
        X = self._design(s.stats, emb, s.fmt)
        with warnings.catch_warnings():  # all-NaN / constant columns (no EXIF, one camera)
            warnings.simplefilter("ignore", RuntimeWarning)
            lo_, hi_ = np.nanmin(X, axis=0), np.nanmax(X, axis=0)
        self.cols = np.nonzero(np.isfinite(lo_) & (hi_ > lo_))[0]
        X = X[:, self.cols]
        folds = assign_folds(s.groups)
        k = int(folds.max()) + 1
        # OOD statistic in CV: held-out rows against a (capped) bank of the other folds
        cv_ood = np.zeros(n)
        for f in range(k):
            te = folds == f
            tr = ~te
            if tr.sum() >= KNN and te.any():
                cv_ood[te] = _ood_dist(emb[te], _bank(emb[tr], seed=f))
        self.ood_thr = float(np.quantile(cv_ood, OOD_Q))
        self.bank = _bank(emb)
        self.n = n
        self.carry = dict(s.carry)
        self.wb_bodies = {}
        bodies, grp_ = body_key(s.stats), wb_group(s.fmt)
        for g, ykey in ((0, "raw_temp"), (1, "jpeg_temp")):
            have = (grp_ == g) & np.isfinite(s.y[ykey])
            for b in np.unique(bodies[have]):
                self.wb_bodies[f"{g}:{int(b)}"] = int((have & (bodies == b)).sum())

        done = [0]
        total = len(ALL_KEYS)

        def tick() -> None:
            done[0] += 1
            if progress:
                progress("train", done[0], total)

        # exposure first (its out-of-fold prediction is a chain feature), then the other
        # sliders; every (slider, fold) fit is an independent single-threaded task
        ctx_e = _prepare("Exposure2012", X, s.y["Exposure2012"], np.ones(n, bool), folds, None)
        self.sliders = {}
        exp_ctx = [ctx_e]
        _run_cv(exp_ctx)
        _finish(ctx_e)
        tick()
        Xc = np.hstack([X, ctx_e["oof_full"][:, None]])
        ctxs = []
        for key in ALL_KEYS:
            if key == "Exposure2012":
                continue
            y = s.y[key]
            if key in WB_KEYS:
                g = 0 if key.startswith("raw") else 1
                rows = (wb_group(s.fmt) == g) & np.isfinite(y)
            else:
                rows = np.isfinite(y)
            ctxs.append(_prepare(key, Xc, y, rows, folds, s))
        _run_cv(ctxs)
        for c in ctxs:
            _finish(c)
            tick()
        allc = [ctx_e, *ctxs]
        _fit_finals(allc)
        _release_workers()
        self.sliders = {c["model"].key: c["model"] for c in allc}
        # drop heavy models of sliders that will never be used (Exposure is kept for the chain)
        for key, m in self.sliders.items():
            if m.mode != "learned" and key != "Exposure2012":
                m.point = m.lo = m.hi = None
        self.trained = True
        self.meta = {
            "trained_at": time.time(), "train_seconds": time.time() - t0, "n": n,
            "sources": sorted(set(s.sources)), "notes": dict(s.notes),
            "n_formats": {
                "raw": int((s.fmt <= FMT_DNG).sum()), "jpeg": int((s.fmt > FMT_DNG).sum()),
            },
        }  # fmt: skip
        sig = repr(
            [(k_, m.mode, round(m.mae_cv, 4), m.n) for k_, m in self.sliders.items()]
        ) + str(self.meta["trained_at"])
        self.profile_hash = hashlib.sha1(sig.encode()).hexdigest()[:10]
        return self

    # ---- inference ----------------------------------------------------------------------
    def predict(self, stats: np.ndarray, emb: np.ndarray, fmt: np.ndarray) -> list[Prediction]:
        if not self.trained or self.pca is None:
            raise RuntimeError("edit profile is not trained")
        n = len(stats)
        if n == 0:
            return []
        fmt = np.asarray(fmt, int)
        emb = np.asarray(emb, np.float32)
        X = self._design(stats, emb, fmt)
        ood = _ood_dist(emb, self.bank) > self.ood_thr
        exp = self.sliders["Exposure2012"]
        with threadpool_limits(limits=2):
            e_pt = _chain_point(exp, X)
            Xc = np.hstack([X, e_pt[:, None]])
            raw = {k: _predict_key(m, Xc, None) for k, m in self.sliders.items()
                   if k != "Exposure2012" and m.mode != "off"}  # fmt: skip
            raw["Exposure2012"] = _predict_key(exp, X, None)
        grp = wb_group(fmt)
        bodies = body_key(stats)
        out: list[Prediction] = []
        for i in range(n):
            p = Prediction(ood=bool(ood[i]))
            for key in TONE_KEYS:
                self._one(p, key, key, raw, i)
            g = int(grp[i])
            for kind in ("temp", "tint"):
                head = f"{'raw' if g == 0 else 'jpeg'}_{kind}"
                self._one(p, head, f"wb_{kind}", raw, i, wb=True)
            if self.wb_bodies.get(f"{g}:{int(bodies[i])}", 0) < MIN_BODY_WB:
                if self._drop_wb(p):  # a body the training edits never covered
                    p.notes.append(WB_BODY_NOTE)
            elif ("wb_temp" in p.modes) != ("wb_tint" in p.modes):
                self._drop_wb(p)  # Temperature and Tint come as a pair, never one alone
            self._gate(p)
            out.append(p)
        return out

    def _one(
        self, p: Prediction, key: str, name: str, raw: dict, i: int, wb: bool = False
    ) -> None:
        m = self.sliders.get(key)
        if m is None or m.mode == "off":
            return
        pt, lo, hi, pc = raw[key]
        if wb and m.clf is not None and pc[i] < 0.5:
            return  # P(user changes WB) < 0.5: write nothing for white balance
        p.modes[name] = m.mode
        if m.mode == "constant":
            p.point[name], p.lo[name], p.hi[name] = m.median, m.median, m.median
            p.written[name] = True
            p.conf[name] = 1.0
            return
        p.point[name], p.lo[name], p.hi[name] = float(pt[i]), float(lo[i]), float(hi[i])
        width = float(hi[i] - lo[i])
        p.conf[name] = float(max(0.0, 1.0 - width / (2.0 * max(m.width_thr, 1e-9))))
        p.written[name] = (width <= m.width_thr) and not p.ood

    @staticmethod
    def _drop_wb(p: Prediction) -> bool:
        had = False
        for name in ("wb_temp", "wb_tint"):
            for d in (p.point, p.lo, p.hi, p.written, p.conf, p.modes):
                had |= d.pop(name, None) is not None
        return had

    def _gate(self, p: Prediction) -> None:
        exp = self.sliders.get("Exposure2012")
        if (
            exp is not None
            and exp.mode == "learned"
            and "Exposure2012" in p.written
            and not p.written["Exposure2012"]
        ):
            for k in COUPLED_TO_EXPOSURE:
                if k in p.written and p.modes.get(k) == "learned":
                    p.written[k] = False
        wb = [k for k in ("wb_temp", "wb_tint") if k in p.written]
        if any(not p.written[k] for k in wb):  # temperature and tint: together or not at all
            for k in wb:
                p.written[k] = False
        p.omitted = sorted(
            k for k, mode in p.modes.items() if mode == "learned" and not p.written[k]
        )
        p.needs_edit = bool(p.omitted)

    # ---- reporting ----------------------------------------------------------------------
    def report(self) -> list[dict[str, Any]]:
        rows = []
        for key in ALL_KEYS:
            m = self.sliders.get(key)
            if m is None:
                continue
            rows.append({
                "slider": key, "mode": m.mode, "n": m.n, "mae_cv": _r(m.mae_cv),
                "mae_median": _r(m.mae_median), "median": _r(m.median), "iqr": _r(m.iqr),
                "tol": m.tol,
            })  # fmt: skip
        return rows

    def evaluate(self, s: EditSet) -> dict[str, Any]:
        """Held-out metrics per slider name (TONE_KEYS + wb_temp, wb_tint) and overall:
        ungated MAE, MAE of the written predictions, median-baseline MAE, q10-q90 coverage and the
        fraction of (image, slider) pairs written."""
        preds = self.predict(s.stats, s.emb, s.fmt)
        grp = wb_group(s.fmt)
        names = (*TONE_KEYS, "wb_temp", "wb_tint")
        res: dict[str, Any] = {}
        pairs = written_pairs = 0
        for name in names:
            ys, pt, lo, hi, wr, base = [], [], [], [], [], []
            n_rows = 0
            for i, p in enumerate(preds):
                if name.startswith("wb_"):
                    key = f"{'raw' if grp[i] == 0 else 'jpeg'}_{name[3:]}"
                else:
                    key = name
                y = s.y[key][i]
                if not np.isfinite(y):
                    continue
                n_rows += 1
                m = self.sliders.get(key)
                base.append(abs(y - (m.median if m else 0.0)))
                if name in p.point:
                    ys.append(y)
                    pt.append(p.point[name])
                    lo.append(p.lo[name])
                    hi.append(p.hi[name])
                    wr.append(p.written.get(name, False))
            ys, pt, lo, hi, wr = (np.array(a) for a in (ys, pt, lo, hi, wr))
            r: dict[str, Any] = {"n": n_rows, "n_pred": len(ys)}
            r["base_mae"] = float(np.mean(base)) if base else float("nan")
            if len(ys):
                r["mae"] = float(np.abs(pt - ys).mean())
                r["coverage"] = float(((ys >= lo) & (ys <= hi)).mean())
                r["mae_gated"] = float(np.abs(pt - ys)[wr].mean()) if wr.any() else float("nan")
                r["written"] = float(wr.mean())
                # MAE of the unwritten sliders' median fallback is not part of the table
            r["mode"] = sorted({p.modes.get(name, "off") for p in preds})
            res[name] = r
            pairs += n_rows
            written_pairs += int(wr.sum()) if len(ys) else 0
        res["_overall"] = {"written_pairs": written_pairs, "pairs": pairs,
                           "written": written_pairs / max(pairs, 1)}  # fmt: skip
        return res

    # ---- persistence --------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        joblib.dump(
            {
                "format": PROFILE_FORMAT, "stats_version": self.stats_version, "n": self.n,
                "pca": self.pca, "sliders": self.sliders, "cols": self.cols,
                "bank": None if self.bank is None else self.bank.astype(np.float16),
                "ood_thr": self.ood_thr, "carry": self.carry, "meta": self.meta,
                "hash": self.profile_hash, "wb_bodies": self.wb_bodies,
            },
            tmp, compress=6,
        )  # fmt: skip
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> EditProfile:
        d = joblib.load(path)
        if not isinstance(d, dict) or d.get("format") != PROFILE_FORMAT:
            raise ValueError(f"{path}: not a ShotSieve edit profile (retrain)")
        if d.get("stats_version") != EDIT_STATS_VERSION:
            raise ValueError(f"{path}: statistics changed (v{d.get('stats_version')}); retrain")
        return cls(
            n=int(d["n"]), trained=True, pca=d["pca"], sliders=d["sliders"],
            bank=None if d["bank"] is None else np.asarray(d["bank"], np.float32),
            ood_thr=float(d["ood_thr"]), carry=dict(d["carry"]), meta=dict(d["meta"]),
            profile_hash=str(d["hash"]), stats_version=int(d["stats_version"]), cols=d["cols"],
            wb_bodies=dict(d.get("wb_bodies", {})),
        )  # fmt: skip


def _r(x: float) -> float | None:
    return None if x is None or not np.isfinite(x) else round(float(x), 4)


# ---- per-slider fitting -----------------------------------------------------------------------


def _workers() -> int:
    import os

    return max(1, min(os.cpu_count() or 1, 8))


def _parallel(calls: list[tuple]) -> list:
    """Run (function, *args) calls on worker processes (HistGradientBoosting holds the GIL too
    much for threads to scale); each call is single-threaded."""
    from joblib import Parallel, delayed

    if len(calls) < 4:
        return [fn(*args) for fn, *args in calls]
    return Parallel(n_jobs=_workers(), backend="loky")(delayed(fn)(*args) for fn, *args in calls)


def _release_workers() -> None:
    """Shut the worker processes down: each holds ~170 MB and would idle for minutes."""
    try:
        from joblib.externals.loky import get_reusable_executor

        get_reusable_executor().shutdown(wait=True)
    except Exception:  # noqa: BLE001 - best effort
        pass


def _quant(x: np.ndarray, q: float) -> float:
    return float(np.quantile(x, min(max(q, 0.0), 1.0))) if len(x) else 0.0


def _prepare(
    key: str, X: np.ndarray, y: np.ndarray, rows: np.ndarray, folds: np.ndarray,
    s: EditSet | None,
) -> dict[str, Any]:  # fmt: skip
    """Per-slider context: rows, WB change classifier and training arrays (no CV yet)."""
    m = SliderModel(key=key, tol=TOL.get(key, DEFAULT_TOL))
    ctx: dict[str, Any] = {"model": m, "X": X, "key": key, "n_all": len(y), "ready": False,
                           "oof_full": np.zeros(len(y))}  # fmt: skip
    ridx = np.nonzero(rows)[0]
    if len(ridx) < MIN_KEY_ROWS:
        return ctx
    Xr, yr, fr = X[ridx], y[ridx], folds[ridx]
    if key in WB_KEYS:
        grp = key.split("_")[0]
        other = f"{grp}_{'tint' if key.endswith('temp') else 'temp'}"
        yo = s.y[other][ridx] if s is not None else yr
        changed = (np.abs(yr) > 0.5) | (np.abs(np.nan_to_num(yo)) > 0.5)
        m.p_changed = float(changed.mean())
        if m.p_changed < CHANGED_MIN_FRAC or changed.sum() < MIN_KEY_ROWS:
            return ctx
        n_min = int(min(changed.sum(), (~changed).sum()))
        if m.p_changed > 0.97:  # the user (almost) always changes WB: no gate needed
            m.clf = None
        else:
            clf = HistGradientBoostingClassifier(
                max_iter=150, learning_rate=0.08, max_leaf_nodes=15, l2_regularization=1.0,
                early_stopping=n_min >= 10, n_iter_no_change=10, random_state=0,
            )
            with threadpool_limits(limits=1):
                clf.fit(Xr, changed.astype(int))
            m.clf = clf
        sel = np.nonzero(changed)[0]
        Xr, yr, fr, ridx = Xr[sel], yr[sel], fr[sel], ridx[sel]
    m.n = len(yr)
    m.median = float(np.median(yr))
    m.iqr = float(np.subtract(*np.percentile(yr, [75, 25])))
    ctx.update(ridx=ridx, Xr=Xr, yr=yr, fr=fr, ready=True)
    return ctx


def _cv_fold(Xr: np.ndarray, yr: np.ndarray, fr: np.ndarray, f: int):
    with threadpool_limits(limits=1):
        te, tr = fr == f, fr != f
        if tr.sum() < 20 or not te.any():
            return None
        pt = _hgb().fit(Xr[tr], yr[tr])
        lo = _hgb("quantile", 0.1).fit(Xr[tr], yr[tr])
        hi = _hgb("quantile", 0.9).fit(Xr[tr], yr[tr])
        med = float(np.median(yr[tr]))
        a, b = np.sort(np.vstack([lo.predict(Xr[te]), hi.predict(Xr[te])]), axis=0)
        return te, pt.predict(Xr[te]), a, b, med


def _final_fit(Xr: np.ndarray, yr: np.ndarray, q: float | None):
    with threadpool_limits(limits=1):
        return _hgb("quantile", q).fit(Xr, yr) if q else _hgb().fit(Xr, yr)


def _run_cv(ctxs: list[dict[str, Any]]) -> None:
    calls, owners = [], []
    for c in ctxs:
        if not c["ready"]:
            continue
        for f in np.unique(c["fr"]):
            calls.append((_cv_fold, c["Xr"], c["yr"], c["fr"], int(f)))
            owners.append(c)
    for c, res in zip(owners, _parallel(calls), strict=True):
        c.setdefault("folds_out", []).append(res)


def _finish(ctx: dict[str, Any]) -> None:
    """Mode, conformal widening and width gate from the out-of-fold predictions."""
    m: SliderModel = ctx["model"]
    if not ctx["ready"]:
        return
    yr = ctx["yr"]
    pt_o = np.full(len(yr), np.nan)
    lo_o, hi_o, med_o = pt_o.copy(), pt_o.copy(), pt_o.copy()
    for res in ctx.get("folds_out", []):
        if res is None:
            continue
        te, p, a, b, med = res
        pt_o[te], lo_o[te], hi_o[te], med_o[te] = p, a, b, med
    ok = np.isfinite(pt_o)
    ctx["ok"] = ok
    if ok.sum() < MIN_KEY_ROWS:
        ctx["ready"] = False
        return
    m.mae_cv = float(np.abs(pt_o[ok] - yr[ok]).mean())
    m.mae_median = float(np.abs(med_o[ok] - yr[ok]).mean())
    if m.mae_cv <= LEARNED_RATIO * m.mae_median:
        m.mode = "learned"
    elif m.iqr <= m.tol:
        m.mode = "constant"
    else:
        m.mode = "off"
    # conformal widening so [lo, hi] covers ~80 % (CQR), then the 80th percentile width gate
    e = np.maximum(lo_o[ok] - yr[ok], yr[ok] - hi_o[ok])
    m.q_cal = _quant(e, 0.8 * (1 + 1.0 / len(e)))
    m.width_thr = _quant(hi_o[ok] - lo_o[ok] + 2 * m.q_cal, WIDTH_Q)
    ctx["oof_full"][ctx["ridx"]] = np.where(ok, pt_o, m.median)


def _fit_finals(ctxs: list[dict[str, Any]]) -> None:
    """Final models on all rows: learned sliders (point + q10 + q90) and Exposure's point model
    (chain feature, whatever its mode)."""
    calls, owners = [], []
    for c in ctxs:
        m: SliderModel = c["model"]
        if not c["ready"]:
            continue
        if m.mode == "learned":
            kinds = (("point", None), ("lo", 0.1), ("hi", 0.9))
        elif m.key == "Exposure2012":
            kinds = (("point", None),)
        else:
            continue
        for name, q in kinds:
            calls.append((_final_fit, c["Xr"], c["yr"], q))
            owners.append((m, name))
    for (m, name), model in zip(owners, _parallel(calls), strict=True):
        setattr(m, name, model)


def _chain_point(m: SliderModel, X: np.ndarray) -> np.ndarray:
    """Exposure as a chain feature, whatever the slider's mode."""
    return m.point.predict(X) if m.point is not None else np.full(len(X), m.median)


def _predict_key(m: SliderModel, X: np.ndarray, _unused: None):
    """(point, lo, hi, p_changed) arrays for a slider on design matrix X (chain column appended
    by the caller). Constant sliders return their median."""
    n = len(X)
    if m.mode == "off" or (m.point is None and m.mode != "constant"):
        z = np.zeros(n)
        pc = np.ones(n) if m.clf is None else m.clf.predict_proba(X)[:, 1]
        return z, z, z, pc
    pc = np.ones(n)
    if m.clf is not None:
        proba = m.clf.predict_proba(X)
        pc = proba[:, list(m.clf.classes_).index(1)] if 1 in m.clf.classes_ else np.zeros(n)
    if m.mode == "constant":
        c = np.full(n, m.median)
        return c, c, c, pc
    a, b = np.sort(np.vstack([m.lo.predict(X), m.hi.predict(X)]), axis=0)
    return m.point.predict(X), a - m.q_cal, b + m.q_cal, pc
