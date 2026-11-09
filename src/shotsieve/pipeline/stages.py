"""Per-image stage functions: Stage A (cheap measurements) and Stage B (faces)."""

from __future__ import annotations

import cv2
import numpy as np

from shotsieve.config import Settings
from shotsieve.features import af_focus, banding, exposure, sharpness
from shotsieve.features.record import RawFeatures
from shotsieve.features.subject import FaceObs, mark_important

BLACK_LUMA = 8.0  # near-black frames skip the face stage
THUMB_EDGE = 256


def stage_a(
    img_rgb: np.ndarray,
    af_boxes: list[list[float]] | None = None,
    exposure_time: float | None = None,
    orientation: int = 1,
) -> RawFeatures:
    """Exposure + global sharpness (+ AF-ROI focus when `af_boxes` - normalised cx,cy,w,h in the
    upright preview - are known); pure NumPy/OpenCV, safe to run in many threads."""
    h, w = img_rgb.shape[:2]
    gray = sharpness.to_luma(img_rgb)
    s = THUMB_EDGE / max(h, w)
    thumb = cv2.resize(
        gray, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA
    )
    ex = exposure.analyse(thumb)
    sh = sharpness.global_sharpness(gray)
    band = banding.measure(gray, orientation, exposure_time)
    af = af_focus.measure(gray, [tuple(b) for b in af_boxes]) if af_boxes else None
    return RawFeatures(
        af_boxes=[list(b) for b in af_boxes or []],
        af_lap=af.lap if af else float("nan"),
        af_peak_ratio=af.peak_ratio if af else float("nan"),
        af_peak_adjacent=af.peak_adjacent if af else False,
        band_amp_ev=band.amp_ev,
        band_coherence=band.coherence,
        band_peak_frac=band.peak_frac,
        band_k=band.k,
        width=w,
        height=h,
        lap_var=sh.lap_var,
        tenengrad=sh.tenengrad,
        mean_luma=ex.mean,
        lin_mean=ex.lin_mean,
        clip_hi=ex.clip_hi,
        clip_lo=ex.clip_lo,
        ev=ex.ev,
        p98=ex.p98,
    )


def is_near_black(raw: RawFeatures) -> bool:
    return raw.mean_luma < BLACK_LUMA


def stage_faces(
    img_rgb: np.ndarray, raw: RawFeatures, settings: Settings
) -> tuple[list[FaceObs], np.ndarray]:
    """Detect faces, measure them and embed the important ones.

    Returns (faces sorted by area, (K,128) SFace embeddings of the important faces, main first).
    Model access is serialised inside the wrappers; call from a single worker thread.
    """
    from shotsieve.models.face_detect import get_face_detector
    from shotsieve.models.face_embed import EMBED_DIM, get_face_embedder
    from shotsieve.models.face_landmarks import get_face_landmarker

    detected = get_face_detector().detect(img_rgb, max_faces=settings.max_faces)
    if not detected:
        return [], np.zeros((0, EMBED_DIM), np.float32)
    gray = sharpness.to_luma(img_rgb)
    obs = [
        FaceObs(
            f.bbox, f.score, sharpness.face_sharpness(gray, f.bbox),
            luma=sharpness.face_luma(gray, f.bbox),
        )
        for f in detected
    ]  # fmt: skip
    obs = mark_important(obs, raw.width, raw.height)
    # mark_important sorts by area like the detector does, so indices line up
    emb, lm = get_face_embedder(), get_face_landmarker()
    vecs = []
    for o, face in zip(obs, detected, strict=False):
        if not o.important:
            continue
        ex = lm.analyze(img_rgb, face)
        o.found = ex.found
        o.blink_left, o.blink_right, o.smile = ex.blink_left, ex.blink_right, ex.smile
        vecs.append(emb.embed(img_rgb, face))
    faces = np.stack(vecs) if vecs else np.zeros((0, EMBED_DIM), np.float32)
    return obs, faces
