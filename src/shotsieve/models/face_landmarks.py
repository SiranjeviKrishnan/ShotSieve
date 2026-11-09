"""MediaPipe FaceLandmarker blendshapes on expanded face crops."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import cv2
import numpy as np

from shotsieve.models.face_detect import Face
from shotsieve.models.registry import model_path

CROP_SCALE = 1.6
CROP_MAX_EDGE = 256
NAN = float("nan")


@dataclass
class FaceExpression:
    """Per-face expression values; NaN everywhere when `found` is False."""

    found: bool
    blink_left: float = NAN
    blink_right: float = NAN
    smile: float = NAN
    jaw_open: float = NAN
    blendshapes: dict[str, float] = field(default_factory=dict)

    @property
    def eyes_closed(self) -> float:
        """max(blink_left, blink_right); NaN if not found."""
        if not self.found:
            return NAN
        return max(self.blink_left, self.blink_right)

    @property
    def eye_open_left(self) -> float:
        return 1.0 - self.blink_left

    @property
    def eye_open_right(self) -> float:
        return 1.0 - self.blink_right


def expand_crop(img: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    """Crop `bbox` scaled by CROP_SCALE (clamped to the image), long edge <= 256."""
    h, w = img.shape[:2]
    x, y, bw, bh = bbox
    cx, cy = x + bw / 2, y + bh / 2
    half_w, half_h = bw * CROP_SCALE / 2, bh * CROP_SCALE / 2
    x0, x1 = max(0, int(round(cx - half_w))), min(w, int(round(cx + half_w)))
    y0, y1 = max(0, int(round(cy - half_h))), min(h, int(round(cy + half_h)))
    crop = img[y0:y1, x0:x1]
    longest = max(crop.shape[:2]) if crop.size else 0
    if longest > CROP_MAX_EDGE:
        s = CROP_MAX_EDGE / longest
        crop = cv2.resize(
            crop,
            (max(1, round(crop.shape[1] * s)), max(1, round(crop.shape[0] * s))),
            interpolation=cv2.INTER_AREA,
        )
    return np.ascontiguousarray(crop)


class FaceLandmarkerWrapper:
    """Lazy MediaPipe FaceLandmarker (IMAGE mode, num_faces=1, blendshapes on).

    Thread-safety: a MediaPipe task object is not documented as thread-safe, so every call
    is serialised with a lock; use from a single worker thread (others block, not crash).
    """

    def __init__(self) -> None:
        import os

        if "MPLCONFIGDIR" not in os.environ:  # mediapipe imports matplotlib: keep ~/.matplotlib out
            from shotsieve.config import data_dir

            os.environ["MPLCONFIGDIR"] = str(data_dir() / "mpl")
        from mediapipe.tasks.python import BaseOptions, vision

        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path("face_landmarker"))),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=True,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
        )
        self._lm = vision.FaceLandmarker.create_from_options(opts)
        self._lock = threading.Lock()

    def analyze_crop(self, crop_rgb: np.ndarray) -> FaceExpression:
        """Run on an already-cropped RGB face image."""
        import mediapipe as mp

        if crop_rgb.size == 0:
            return FaceExpression(False)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop_rgb))
        with self._lock:
            result = self._lm.detect(image)
        if not result.face_blendshapes:
            return FaceExpression(False)
        bs = {c.category_name: float(c.score) for c in result.face_blendshapes[0]}
        return FaceExpression(
            True,
            bs.get("eyeBlinkLeft", NAN),
            bs.get("eyeBlinkRight", NAN),
            (bs.get("mouthSmileLeft", 0.0) + bs.get("mouthSmileRight", 0.0)) / 2,
            bs.get("jawOpen", NAN),
            bs,
        )

    def analyze(self, img_rgb: np.ndarray, face: Face) -> FaceExpression:
        """Crop the face (bbox x1.6) from `img_rgb` and analyse it."""
        return self.analyze_crop(expand_crop(img_rgb, face.bbox))

    def close(self) -> None:
        with self._lock:
            self._lm.close()


_singleton: FaceLandmarkerWrapper | None = None
_singleton_lock = threading.Lock()


def get_face_landmarker() -> FaceLandmarkerWrapper:
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = FaceLandmarkerWrapper()
        return _singleton
