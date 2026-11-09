"""Raw per-image measurements (cached) and their conversion into FeatureVectors."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np

from shotsieve.features import af_focus, sharpness
from shotsieve.features.eyes import closed_score, is_closed
from shotsieve.features.subject import FaceObs, centrality
from shotsieve.features.vector import FeatureVector

NAN = float("nan")


@dataclass
class RawFeatures:
    """Everything measured from one preview. Normalisation happens later (needs the whole shoot)."""

    width: int = 0
    height: int = 0
    lap_var: float = NAN
    tenengrad: float = NAN
    mean_luma: float = NAN
    lin_mean: float = NAN
    clip_hi: float = NAN
    clip_lo: float = NAN
    ev: float = NAN
    p98: float = NAN
    faces: list[FaceObs] = field(default_factory=list)
    faces_skipped: bool = False  # face stage skipped (near-black frame)
    af_boxes: list[list[float]] = field(default_factory=list)  # [cx, cy, w, h], upright preview
    af_lap: float = NAN  # Laplacian variance in the AF ROI
    af_peak_ratio: float = NAN
    af_peak_adjacent: bool = False
    band_amp_ev: float = NAN  # e-shutter banding (NaN: not measured)
    band_coherence: float = NAN
    band_peak_frac: float = NAN
    band_k: float = NAN  # bands per frame height (flicker bands are broad: k <= 12)

    def to_json(self) -> str:
        d = {k: v for k, v in self.__dict__.items() if k != "faces"}
        d["faces"] = [f.to_dict() for f in self.faces]
        return json.dumps(d)

    @classmethod
    def from_json(cls, s: str) -> RawFeatures:
        d = json.loads(s)
        faces = [FaceObs.from_dict(f) for f in d.pop("faces", [])]
        return cls(faces=faces, **d)

    @property
    def important_faces(self) -> list[FaceObs]:
        return [f for f in self.faces if f.important]


@dataclass(frozen=True)
class ShootNorm:
    """Per-shoot sharpness references (p90 with absolute floors)."""

    lap_ref: float = sharpness.LAP_FLOOR
    tenen_ref: float = sharpness.TENEN_FLOOR
    face_ref: float = sharpness.FACE_FLOOR
    af_ref: float = sharpness.LAP_FLOOR


def compute_norm(records: list[RawFeatures]) -> ShootNorm:
    return ShootNorm(
        sharpness.shoot_reference(
            [r.lap_var for r in records], sharpness.LAP_FLOOR, cap=sharpness.LAP_REF_CAP
        ),
        sharpness.shoot_reference([r.tenengrad for r in records], sharpness.TENEN_FLOOR),
        sharpness.shoot_reference(
            [r.faces[0].sharp for r in records if r.faces], sharpness.FACE_FLOOR
        ),
        sharpness.shoot_reference([r.af_lap for r in records], sharpness.LAP_FLOOR),
    )


GROUP_REF_FLOOR = 0.25  # a group's own reference never drops below this fraction of the shoot p90


def group_norms(
    records: list[RawFeatures], group_ids: list[int], norm: ShootNorm
) -> list[ShootNorm]:
    """Per-image references: for duplicate groups the sharpest sibling (clamped to
    [0.25 * shoot p90, shoot p90]) so blur is judged against the same scene; singletons use
    the shoot-wide p90."""
    best: dict[int, list[float]] = {}
    for r, g in zip(records, group_ids, strict=True):
        cur = best.setdefault(g, [0.0, 0.0, 0.0])
        for k, v in enumerate((r.lap_var, r.tenengrad, r.faces[0].sharp if r.faces else NAN)):
            if v is not None and math.isfinite(v):
                cur[k] = max(cur[k], v)
    size: dict[int, int] = {}
    for g in group_ids:
        size[g] = size.get(g, 0) + 1
    out = []
    for g in group_ids:
        if size[g] < 2:
            out.append(norm)
            continue
        refs = []
        for v, shoot in zip(best[g], (norm.lap_ref, norm.tenen_ref, norm.face_ref), strict=True):
            refs.append(min(shoot, max(v, GROUP_REF_FLOOR * shoot)))
        out.append(ShootNorm(*refs, af_ref=norm.af_ref))
    return out


def _log2(x: float | None) -> float:
    return math.log2(x) if x and x > 0 else NAN


def build_vector(
    raw: RawFeatures,
    norm: ShootNorm,
    *,
    flash_fired: bool | None = None,
    flash_delta_ev: float = NAN,
    iso: int | None = None,
    exposure_time: float | None = None,
    burst_size: int = 1,
    dup_group_size: int = 1,
    rank_norm: float = 0.0,
    eyes_thresh: float = 0.55,
    eyes_both_thresh: float = 0.45,
) -> FeatureVector:
    """Assemble the FeatureVector for one image (sharpness normalised with `norm`)."""
    fv = FeatureVector(
        sharp_global=raw.lap_var,
        sharp_global_norm=sharpness.normalise(raw.lap_var, norm.lap_ref),
        tenengrad_norm=sharpness.normalise(raw.tenengrad, norm.tenen_ref),
        exp_mean=raw.mean_luma,
        exp_clip_hi=raw.clip_hi,
        exp_clip_lo=raw.clip_lo,
        exp_ev=raw.ev,
        flash_fired=NAN if flash_fired is None else float(flash_fired),
        flash_lum_delta_ev=flash_delta_ev,
        burst_size=float(burst_size),
        dup_group_size=float(dup_group_size),
        rank_in_group_norm=rank_norm,
        iso_log2=_log2(iso),
        shutter_log2=_log2(exposure_time),
    )
    if raw.af_boxes:
        fv.af_known = 1.0
        fv.af_sharp_norm = sharpness.normalise(raw.af_lap, norm.af_ref)
        fv.af_peak_ratio = raw.af_peak_ratio
        fv.af_on_face = float(
            af_focus.on_face(
                [tuple(b) for b in raw.af_boxes],
                [f.bbox for f in raw.important_faces],
                raw.width,
                raw.height,
            )
        )
        if math.isnan(fv.af_sharp_norm):
            fv.af_known = 0.0
    fv.banding_amp_ev, fv.banding_coherence = raw.band_amp_ev, raw.band_coherence
    faces = raw.faces
    imp = raw.important_faces
    fv.n_faces = float(len(imp))
    if faces:
        main = faces[0]
        fv.main_face_area = main.area / max(1.0, raw.width * raw.height)
        fv.main_face_centrality = centrality(main.bbox, raw.width, raw.height)
        fv.face_sharp_main = main.sharp
        fv.face_sharp_norm_main = sharpness.normalise(main.sharp, norm.face_ref)
        fv.eyes_closed_main = closed_score(main.blink_left, main.blink_right)
        fv.smile_main = main.smile if main.found else NAN
        fv.face_det_score_main = main.score
    if imp:
        sharp_n = [sharpness.normalise(f.sharp, norm.face_ref) for f in imp]
        sharp_n = [s for s in sharp_n if not math.isnan(s)]
        fv.face_sharp_min_important = min(sharp_n) if sharp_n else NAN
        closed = [closed_score(f.blink_left, f.blink_right) for f in imp]
        closed = [c for c in closed if not math.isnan(c)]
        fv.eyes_closed_max_important = max(closed) if closed else NAN
        fv.n_eyes_closed_important = float(
            sum(is_closed(f.blink_left, f.blink_right, eyes_thresh, eyes_both_thresh) for f in imp)
        )
    return fv


def stack_embeddings(embs: list[np.ndarray | None], dim: int) -> np.ndarray:
    out = np.zeros((len(embs), dim), np.float32)
    for i, e in enumerate(embs):
        if e is not None:
            out[i] = e
    return out
