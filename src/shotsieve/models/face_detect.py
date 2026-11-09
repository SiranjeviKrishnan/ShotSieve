"""YuNet face detection via cv2.FaceDetectorYN."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import cv2
import numpy as np

from shotsieve.models.registry import model_path

DETECT_EDGE = 640


@dataclass(frozen=True)
class Face:
    """bbox = (x, y, w, h) and landmarks (5x2: r-eye, l-eye, nose, r-mouth, l-mouth) in the
    coordinates of the image passed to `detect`."""

    bbox: tuple[float, float, float, float]
    landmarks: np.ndarray
    score: float

    @property
    def area(self) -> float:
        return self.bbox[2] * self.bbox[3]

    def to_row(self) -> np.ndarray:
        """15-float YuNet row (x,y,w,h, 5 landmarks, score) as FaceRecognizerSF expects."""
        return np.concatenate(
            [np.array(self.bbox, np.float32), self.landmarks.reshape(-1), [self.score]]
        ).astype(np.float32)


class FaceDetector:
    """Thread-safety: the underlying cv2 detector is stateful (input size); calls are locked."""

    def __init__(self, score_threshold: float = 0.7, nms_threshold: float = 0.3) -> None:
        self._det = cv2.FaceDetectorYN.create(
            str(model_path("yunet")), "", (320, 320), score_threshold, nms_threshold, 5000
        )
        self._score_threshold = score_threshold
        self._lock = threading.Lock()

    def detect(self, img_rgb: np.ndarray, max_faces: int | None = None) -> list[Face]:
        """Detect faces; boxes are returned in original image coordinates, largest first."""
        h, w = img_rgb.shape[:2]
        scale = DETECT_EDGE / max(h, w)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        bgr = cv2.cvtColor(cv2.resize(img_rgb, (nw, nh), interpolation=interp), cv2.COLOR_RGB2BGR)
        with self._lock:
            self._det.setInputSize((nw, nh))
            _, rows = self._det.detect(bgr)
        if rows is None:
            return []
        sx, sy = w / nw, h / nh
        faces = []
        for r in rows:
            lm = r[4:14].reshape(5, 2) * np.array([sx, sy], np.float32)
            bbox = (float(r[0] * sx), float(r[1] * sy), float(r[2] * sx), float(r[3] * sy))
            faces.append(Face(bbox, lm.astype(np.float32), float(r[14])))
        faces.sort(key=lambda f: f.area, reverse=True)
        return faces[:max_faces] if max_faces else faces


    def detect_scaled(self, img_rgb: np.ndarray, scale: float, score: float) -> list[Face]:
        """Detect on `img_rgb` resized by `scale` (>1 upsamples) with a per-call score threshold
        (T23h, recognition scans). Boxes come back in the coordinates of `img_rgb`; unsorted."""
        h, w = img_rgb.shape[:2]
        nw, nh = max(8, round(w * scale)), max(8, round(h * scale))
        if (nw, nh) != (w, h):
            interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
            img_rgb = cv2.resize(img_rgb, (nw, nh), interpolation=interp)
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        with self._lock:
            self._det.setInputSize((nw, nh))
            self._det.setScoreThreshold(float(score))
            try:
                _, rows = self._det.detect(bgr)
            finally:
                self._det.setScoreThreshold(self._score_threshold)
        if rows is None:
            return []
        sx, sy = w / nw, h / nh
        out = []
        for r in rows:
            lm = r[4:14].reshape(5, 2) * np.array([sx, sy], np.float32)
            bbox = (float(r[0] * sx), float(r[1] * sy), float(r[2] * sx), float(r[3] * sy))
            out.append(Face(bbox, lm.astype(np.float32), float(r[14])))
        return out


_singleton: FaceDetector | None = None
_singleton_lock = threading.Lock()


_rec_singleton: FaceDetector | None = None


def get_rec_face_detector() -> FaceDetector:
    """A second detector for the recognition scans (the cull detector is never touched)."""
    global _rec_singleton
    with _singleton_lock:
        if _rec_singleton is None:
            _rec_singleton = FaceDetector(score_threshold=0.3, nms_threshold=0.3)
        return _rec_singleton


def get_face_detector() -> FaceDetector:
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = FaceDetector()
        return _singleton
