"""DINOv2-small scene embeddings (384-d, L2-normalised) via ONNX Runtime."""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from shotsieve._ort import import_onnxruntime
from shotsieve.models.registry import model_path
from shotsieve.models.runtime import create_session, np_dtype

ort = import_onnxruntime()

EMBED_DIM = 384
INPUT_SIZE = 224
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(img_rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 (H,W,3) -> normalised float32 CHW (3,224,224); whole frame is squashed."""
    small = cv2.resize(img_rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    x = (small.astype(np.float32) / 255.0 - _MEAN) / _STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


class ImageEmbedder:
    """Batched DINOv2 CLS-token embedder (fp32 model on the CPU provider; see docs/02_MODELS.md).

    Input/output names and the input dtype (fp32/fp16) are read from the session at load.
    Thread-safety: `run` is guarded by a lock, so it may be called from any thread.
    """

    def __init__(self, use_coreml: bool = False, threads: int = 6, batch_size: int = 8) -> None:
        self.batch_size = batch_size
        self._lock = threading.Lock()
        self.session, self.provider = create_session(
            model_path("dinov2_small"), self._warmup, use_coreml, threads
        )
        self._bind(self.session)

    def _bind(self, sess: ort.InferenceSession) -> None:
        inp, out = sess.get_inputs()[0], sess.get_outputs()[0]
        self.input_name, self.output_name = inp.name, out.name
        self.input_dtype = np_dtype(inp.type)

    def _warmup(self, sess: ort.InferenceSession) -> None:
        self._bind(sess)
        x = np.zeros((2, 3, INPUT_SIZE, INPUT_SIZE), dtype=self.input_dtype)
        out = sess.run([self.output_name], {self.input_name: x})[0]
        if out.ndim != 3 or out.shape[-1] != EMBED_DIM or not np.isfinite(out).all():
            raise RuntimeError(f"bad warmup output shape/values {out.shape}")

    def _run(self, batch: np.ndarray) -> np.ndarray:
        x = batch.astype(self.input_dtype, copy=False)
        with self._lock:
            hidden = self.session.run([self.output_name], {self.input_name: x})[0]
        cls = hidden[:, 0, :].astype(np.float32)
        return cls / np.maximum(np.linalg.norm(cls, axis=1, keepdims=True), 1e-12)

    def embed_batch(self, images: list[np.ndarray]) -> np.ndarray:
        """Embed RGB uint8 images -> (N, 384) float32, rows L2-normalised."""
        if not images:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)
        chunks = []
        for i in range(0, len(images), self.batch_size):
            batch = np.stack([preprocess(im) for im in images[i : i + self.batch_size]])
            chunks.append(self._run(batch))
        return np.concatenate(chunks)

    def embed(self, image: np.ndarray) -> np.ndarray:
        """Embed one RGB image -> (384,) vector."""
        return self.embed_batch([image])[0]

    def benchmark(self, image: np.ndarray, n: int = 10) -> float:
        """Mean seconds per single-image embed."""
        self.embed(image)
        t = time.perf_counter()
        for _ in range(n):
            self.embed(image)
        return (time.perf_counter() - t) / n


_singleton: ImageEmbedder | None = None
_singleton_lock = threading.Lock()


def get_image_embedder(
    threads: int | None = None, batch_size: int | None = None, use_coreml: bool | None = None
) -> ImageEmbedder:
    """Lazy process-wide embedder (the arguments only matter for the first creation)."""
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            kwargs = {"threads": threads, "batch_size": batch_size, "use_coreml": use_coreml}
            _singleton = ImageEmbedder(**{k: v for k, v in kwargs.items() if v is not None})
        return _singleton
