"""SFace identity embeddings via cv2.FaceRecognizerSF."""

from __future__ import annotations

import threading

import cv2
import numpy as np

from shotsieve.models.face_detect import Face
from shotsieve.models.registry import model_path

EMBED_DIM = 128


class FaceEmbedder:
    """Thread-safety: calls are serialised with a lock."""

    def __init__(self) -> None:
        self._rec = cv2.FaceRecognizerSF.create(str(model_path("sface")), "")
        self._lock = threading.Lock()

    def embed(self, img_rgb: np.ndarray, face: Face) -> np.ndarray:
        """Align with YuNet landmarks and return a (128,) L2-normalised float32 vector."""
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        with self._lock:
            aligned = self._rec.alignCrop(bgr, face.to_row())
            feat = np.asarray(self._rec.feature(aligned), np.float32).reshape(-1)
        return feat / max(float(np.linalg.norm(feat)), 1e-12)

    def embed_small(self, img_rgb: np.ndarray, face: Face, target: int = 112) -> np.ndarray:
        """Embedding for a small / noisy face (T23h, recognition only): the neighbourhood of the
        face is cut out, lightly denoised and Lanczos-upscaled until the face is ~`target` px, so
        SFace's 112 px alignment warp does not interpolate a 20 px face from the full frame."""
        x, y, w, h = face.bbox
        side = max(w, h)
        if side >= target:
            return self.embed(img_rgb, face)
        k = target / max(side, 1.0)
        ih, iw = img_rgb.shape[:2]
        pad = side * 0.8
        x0, y0 = int(max(0, x - pad)), int(max(0, y - pad))
        x1, y1 = int(min(iw, x + w + pad)), int(min(ih, y + h + pad))
        crop = img_rgb[y0:y1, x0:x1]
        if side < 40:  # sensor noise dominates tiny faces: a mild blur before upscaling
            crop = cv2.GaussianBlur(crop, (0, 0), 0.6)
        up = cv2.resize(
            crop, (round(crop.shape[1] * k), round(crop.shape[0] * k)), interpolation=cv2.INTER_LANCZOS4  # noqa: E501
        )
        lm = (face.landmarks - np.array([x0, y0], np.float32)) * k
        bb = ((x - x0) * k, (y - y0) * k, w * k, h * k)
        return self.embed(up, Face(bb, lm.astype(np.float32), face.score))

    def embed_faces(self, img_rgb: np.ndarray, faces: list[Face]) -> np.ndarray:
        if not faces:
            return np.zeros((0, EMBED_DIM), np.float32)
        return np.stack([self.embed(img_rgb, f) for f in faces])


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two (already normalised or not) vectors."""
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


_singleton: FaceEmbedder | None = None
_singleton_lock = threading.Lock()


def get_face_embedder() -> FaceEmbedder:
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = FaceEmbedder()
        return _singleton
