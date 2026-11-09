"""Subject presence and importance from detected faces."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

MAX_IMPORTANT = 6
REL_AREA_MIN = 0.25  # of the largest face
IMG_AREA_MIN = 0.003  # of the image


@dataclass
class FaceObs:
    """One detected face with its measurements (coordinates in preview pixels)."""

    bbox: tuple[float, float, float, float]
    score: float
    sharp: float = float("nan")  # raw face-ROI Laplacian variance
    blink_left: float = float("nan")
    blink_right: float = float("nan")
    smile: float = float("nan")
    found: bool = False  # landmarker produced blendshapes
    important: bool = False
    luma: float = float("nan")  # mean luma of the face crop (0-255)

    @property
    def area(self) -> float:
        return self.bbox[2] * self.bbox[3]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bbox"] = list(self.bbox)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> FaceObs:
        d = dict(d)
        d["bbox"] = tuple(d["bbox"])
        return cls(**d)


def mark_important(faces: list[FaceObs], width: int, height: int) -> list[FaceObs]:
    """Sort by area (largest first), flag important faces, keep at most MAX_IMPORTANT total."""
    faces = sorted(faces, key=lambda f: f.area, reverse=True)[:MAX_IMPORTANT]
    if not faces:
        return faces
    largest, img_area = faces[0].area, float(width * height)
    for f in faces:
        f.important = f.area >= REL_AREA_MIN * largest and f.area >= IMG_AREA_MIN * img_area
    return faces


def centrality(bbox: tuple[float, float, float, float], width: int, height: int) -> float:
    """1 at the image centre, 0 at a corner."""
    cx, cy = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
    d = math.hypot((cx - width / 2) / (width / 2), (cy - height / 2) / (height / 2))
    return max(0.0, 1.0 - d / math.sqrt(2))


def subject_score(n_important: int, main_area_frac: float, main_centrality: float) -> float:
    """0.3 baseline when no faces; faces raise it with size and centrality."""
    if n_important <= 0:
        return 0.3
    size = min(1.0, main_area_frac / 0.08)
    cen = 0.0 if math.isnan(main_centrality) else main_centrality
    return 0.6 + 0.2 * size + 0.2 * cen
