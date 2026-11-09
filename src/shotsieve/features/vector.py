"""FeatureVector: the fixed-order numeric feature record used by scoring and the personal model."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import numpy as np

SCHEMA_VERSION = 2

# Order is part of the contract (personal models are trained against it). Append only, and bump
# SCHEMA_VERSION on any change.
FEATURE_NAMES: tuple[str, ...] = (
    "sharp_global",
    "sharp_global_norm",
    "tenengrad_norm",
    "exp_mean",
    "exp_clip_hi",
    "exp_clip_lo",
    "exp_ev",
    "n_faces",
    "main_face_area",
    "main_face_centrality",
    "face_sharp_main",
    "face_sharp_min_important",
    "face_sharp_norm_main",
    "eyes_closed_main",
    "eyes_closed_max_important",
    "n_eyes_closed_important",
    "smile_main",
    "face_det_score_main",
    "flash_fired",
    "flash_lum_delta_ev",
    "burst_size",
    "dup_group_size",
    "rank_in_group_norm",
    "iso_log2",
    "shutter_log2",
    # schema 2
    "af_known",
    "af_sharp_norm",
    "af_peak_ratio",
    "af_on_face",
    "banding_amp_ev",
    "banding_coherence",
)

NAN = float("nan")


@dataclass
class FeatureVector:
    """Per-image features. Missing values are NaN.

    Notes: `n_faces` counts *important* faces (area >= 25% of the largest and >= 0.3% of the
    image, max 6). `face_sharp_min_important` is the minimum *normalised* sharpness over the
    important faces. `rank_in_group_norm` is 0 for the best (and for singletons), 1 for the worst.
    """

    sharp_global: float = NAN
    sharp_global_norm: float = NAN
    tenengrad_norm: float = NAN
    exp_mean: float = NAN
    exp_clip_hi: float = NAN
    exp_clip_lo: float = NAN
    exp_ev: float = NAN
    n_faces: float = 0.0
    main_face_area: float = 0.0
    main_face_centrality: float = NAN
    face_sharp_main: float = NAN
    face_sharp_min_important: float = NAN
    face_sharp_norm_main: float = NAN
    eyes_closed_main: float = NAN
    eyes_closed_max_important: float = NAN
    n_eyes_closed_important: float = 0.0
    smile_main: float = NAN
    face_det_score_main: float = NAN
    flash_fired: float = NAN
    flash_lum_delta_ev: float = NAN
    burst_size: float = 1.0
    dup_group_size: float = 1.0
    rank_in_group_norm: float = 0.0
    iso_log2: float = NAN
    shutter_log2: float = NAN
    af_known: float = 0.0
    af_sharp_norm: float = NAN
    af_peak_ratio: float = NAN
    af_on_face: float = 0.0
    banding_amp_ev: float = NAN
    banding_coherence: float = NAN

    def to_array(self) -> np.ndarray:
        """float32 vector in FEATURE_NAMES order."""
        return np.array([getattr(self, n) for n in FEATURE_NAMES], dtype=np.float32)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> FeatureVector:
        if len(arr) != len(FEATURE_NAMES):
            raise ValueError(f"expected {len(FEATURE_NAMES)} features, got {len(arr)}")
        return cls(**{n: float(v) for n, v in zip(FEATURE_NAMES, arr, strict=True)})

    @classmethod
    def from_array_v1(cls, arr: np.ndarray) -> FeatureVector:
        """From a (possibly shorter) older-schema row; missing trailing features take defaults."""
        return cls(**{n: float(v) for n, v in zip(FEATURE_NAMES, arr, strict=False)})

    def to_dict(self) -> dict[str, float]:
        return {n: getattr(self, n) for n in FEATURE_NAMES}

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> FeatureVector:
        return cls(**{n: float(data[n]) for n in FEATURE_NAMES if n in data})


assert [f.name for f in fields(FeatureVector)] == list(FEATURE_NAMES)


def stack(vectors: list[FeatureVector]) -> np.ndarray:
    """(N, len(FEATURE_NAMES)) float32 matrix."""
    if not vectors:
        return np.zeros((0, len(FEATURE_NAMES)), np.float32)
    return np.stack([v.to_array() for v in vectors])


def isnan(x: float) -> bool:
    return x is None or math.isnan(x)
