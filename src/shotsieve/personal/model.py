"""PersonalModel: weighted HistGradientBoosting + calibration on features + PCA-16 of DINOv2."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from shotsieve.config import Settings, data_dir
from shotsieve.features.vector import FEATURE_NAMES, SCHEMA_VERSION
from shotsieve.log import get_logger

log = get_logger("personal")

MIN_ROWS = 40
PCA_DIM = 16
MODEL_FORMAT = 1


class SchemaMismatchError(ValueError):
    """A saved model was trained against a different feature schema."""


def default_model_path(settings: Settings | None = None) -> Path:
    if settings is not None and settings.personal_model_path:
        return Path(settings.personal_model_path)
    return data_dir() / "personal_model.joblib"


def _make_hgb() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.08, l2_regularization=1.0, random_state=0
    )


def _wquantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    order = np.argsort(x)
    cw = np.cumsum(w[order])
    return float(x[order][min(int(np.searchsorted(cw, q * cw[-1])), len(x) - 1)])


def derive_thresholds(
    p: np.ndarray, y: np.ndarray, w: np.ndarray, t_reject: float = 0.30
) -> dict[str, float]:
    """Per-user reject bar from the training-set quantiles of P(keep) on the user's own keepers:
    it sits at their weighted 1st percentile, so about 1 % of keepers fall below it (the
    product's missed-keeper target). Never stricter than the default: a photographer who keeps
    ~20 % of frames has a calibrated P(keep) that rarely reaches 0.3 for a keeper, and a fixed bar
    would send those frames to `low_quality` rejects. The keep bar is left alone, because moving
    it down would dissolve a picky user's preference (the cost of a too-high keep bar is only
    review work, the cost of a too-high reject bar is lost photos)."""
    keepers = np.asarray(y) == 1
    if keepers.sum() < 5:
        return {"t_reject": t_reject}
    q01 = _wquantile(np.asarray(p)[keepers], np.asarray(w, dtype=np.float64)[keepers], 0.01)
    return {"t_reject": round(min(t_reject, max(0.02, q01)), 4)}


@dataclass
class PersonalModel:
    """P(keep) from the 25 feature columns plus PCA-16 of the DINOv2 embedding.

    `n_train` is the number of training rows when the model is trained and 0 when it stayed on
    the prior (too few rows or a single class) so the runner's blend weight is 0 in that case.
    """

    n_train: int = 0
    n_rows: int = 0  # rows offered to fit(), trained or not
    trained: bool = False
    schema_version: int = SCHEMA_VERSION
    feature_names: tuple[str, ...] = FEATURE_NAMES
    pca: PCA | None = None
    clf: Any = None
    calibration: str = "none"
    meta: dict[str, Any] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)  # per-user t_reject
    empty_cols: list[int] = field(default_factory=list)  # all-NaN in training: zero-filled

    # ---- features -----------------------------------------------------------------------
    def _design(self, X: np.ndarray, embeddings: np.ndarray | None) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        n = len(self.feature_names)
        if X.ndim != 2 or X.shape[1] < n:
            raise ValueError(f"expected (N, {n}) features, got {X.shape}")
        X = X[:, :n]  # an older-schema model ignores features appended since it was trained
        if self.empty_cols:  # e.g. AF features of a shoot without maker-note data
            X = X.copy()
            X[:, self.empty_cols] = 0.0
        if self.pca is None:
            return X
        k = self.pca.n_components_
        if embeddings is None or len(embeddings) != len(X):
            z = np.full((len(X), k), np.nan, np.float32)  # HistGradientBoosting handles NaN
        else:
            z = self.pca.transform(np.asarray(embeddings, dtype=np.float32)).astype(np.float32)
        return np.hstack([X, z])

    # ---- training -----------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        w: np.ndarray | None = None,
        embeddings: np.ndarray | None = None,
    ) -> PersonalModel:
        """Fit on (X, y, w). Fewer than 40 rows or one class leaves the model untrained."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(int)
        w = np.ones(len(y)) if w is None else np.asarray(w, dtype=np.float64)
        self.n_rows = len(y)
        self.trained, self.n_train, self.pca, self.clf, self.calibration = (
            False,
            0,
            None,
            None,
            "none",
        )
        self.thresholds = {}
        X = X[:, : len(self.feature_names)]
        self.empty_cols = [int(i) for i in np.nonzero(np.isnan(X).all(axis=0))[0]] if len(X) else []
        if len(y) < MIN_ROWS or len(np.unique(y)) < 2:
            log.info("personal model: %d rows / %d classes -> prior only", len(y), len(set(y)))
            return self
        if embeddings is not None and len(embeddings) == len(X):
            k = min(PCA_DIM, len(X) - 1, np.asarray(embeddings).shape[1])
            self.pca = PCA(n_components=k, random_state=0).fit(
                np.asarray(embeddings, dtype=np.float32)
            )
        D = self._design(X, embeddings)
        minority = int(min(np.sum(y == 0), np.sum(y == 1)))
        base = _make_hgb()
        if len(y) >= 60 and minority >= 3:
            method = "isotonic" if len(y) >= 300 and minority >= 30 else "sigmoid"
            clf = CalibratedClassifierCV(base, method=method, cv=3)
            self.calibration = f"{method}-cv3"
        elif minority >= 2:
            clf = CalibratedClassifierCV(base, method="sigmoid", cv=2)
            self.calibration = "sigmoid-cv2"
        else:
            clf, self.calibration = base, "none"
        # small data: OpenMP over every core is slower than a couple of threads
        limits = 2 if len(y) < 5000 else None
        try:
            with threadpool_limits(limits=limits):
                clf.fit(D, y, sample_weight=w)
        except ValueError as exc:  # e.g. a fold lost a class
            log.warning("calibrated fit failed (%s); fitting uncalibrated", exc)
            clf, self.calibration = _make_hgb(), "none"
            with threadpool_limits(limits=limits):
                clf.fit(D, y, sample_weight=w)
        self.clf = clf
        self.trained, self.n_train = True, len(y)
        self.thresholds = derive_thresholds(self.predict_proba(X, embeddings), y, w)
        return self

    # ---- inference ----------------------------------------------------------------------
    def predict_proba(self, X: np.ndarray, embeddings: np.ndarray | None = None) -> np.ndarray:
        """(N,) P(keep). An untrained model returns 0.5 (and n_train == 0 disables the blend)."""
        X = np.asarray(X)
        if not self.trained or self.clf is None:
            return np.full(len(X), 0.5)
        D = self._design(X, embeddings)
        with threadpool_limits(limits=2 if len(D) < 5000 else None):
            proba = self.clf.predict_proba(D)
        col = list(self.clf.classes_).index(1)
        return np.asarray(proba[:, col], dtype=np.float64)

    # ---- persistence --------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": MODEL_FORMAT,
            "schema_version": self.schema_version,
            "feature_names": list(self.feature_names),
            "n_train": self.n_train,
            "n_rows": self.n_rows,
            "trained": self.trained,
            "pca": self.pca,
            "clf": self.clf,
            "calibration": self.calibration,
            "meta": self.meta,
            "thresholds": self.thresholds,
            "empty_cols": self.empty_cols,
        }
        tmp = path.with_name(path.name + ".tmp")
        joblib.dump(payload, tmp)
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> PersonalModel:
        """Load a saved model; raises SchemaMismatchError if its feature schema differs."""
        payload = joblib.load(path)
        if not isinstance(payload, dict):
            raise SchemaMismatchError(f"{path}: not a ShotSieve personal model")
        names = list(payload.get("feature_names", []))
        ver = payload.get("schema_version")
        if (
            payload.get("format") != MODEL_FORMAT
            or not isinstance(ver, int)
            or not 1 <= ver <= SCHEMA_VERSION
            or not names
            or names != list(FEATURE_NAMES[: len(names)])  # append-only schema: prefix still valid
        ):
            raise SchemaMismatchError(
                f"{path}: trained for feature schema {payload.get('schema_version')!r} "
                f"(format {payload.get('format')!r}), current is {SCHEMA_VERSION}; retrain"
            )
        return cls(
            n_train=int(payload["n_train"]),
            n_rows=int(payload.get("n_rows", payload["n_train"])),
            trained=bool(payload["trained"]),
            schema_version=int(payload["schema_version"]),
            feature_names=tuple(payload["feature_names"]),
            pca=payload["pca"],
            clf=payload["clf"],
            calibration=payload.get("calibration", "none"),
            meta=dict(payload.get("meta", {})),
            thresholds=dict(payload.get("thresholds", {})),
            empty_cols=[int(i) for i in payload.get("empty_cols", [])],
        )
