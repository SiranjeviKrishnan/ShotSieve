"""SigLIP2-B/16 image + text towers (768-d, L2-normalised) via ONNX Runtime, CPU provider.

The vision tower is loaded by the background indexer; the text tower (and tokenizer) only on
the first query and unloaded after `IDLE_UNLOAD_S` without use.
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from shotsieve.models.registry import model_path
from shotsieve.models.runtime import create_session

DIM = 768
INPUT_SIZE = 224
TEXT_LEN = 64
EOS_ID, PAD_ID = 1, 0
IDLE_UNLOAD_S = 600.0


def preprocess(img_rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 (H,W,3) -> float32 CHW 224x224, squash-resized (bilinear), (x/255-0.5)/0.5."""
    small = cv2.resize(img_rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    x = small.astype(np.float32) / 127.5 - 1.0
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def _norm(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


class SiglipVision:
    def __init__(self, variant: str = "fp32", threads: int = 4, batch_size: int = 8) -> None:
        name = "siglip2_vision_int8" if variant == "int8" else "siglip2_vision"
        self.variant, self.batch_size = variant, batch_size
        self.session, self.provider = create_session(
            model_path(name), None, use_coreml=False, threads=threads, arena=False
        )
        self._in = self.session.get_inputs()[0].name
        self._lock = threading.Lock()

    def embed_batch(self, images: list[np.ndarray]) -> np.ndarray:
        """RGB uint8 images -> (N, 768) float32, rows L2-normalised."""
        if not images:
            return np.zeros((0, DIM), np.float32)
        out = []
        for i in range(0, len(images), self.batch_size):
            x = np.stack([preprocess(im) for im in images[i : i + self.batch_size]])
            with self._lock:
                pooled = self.session.run(["pooler_output"], {self._in: x})[0]
            out.append(_norm(pooled))
        return np.concatenate(out)


def tokenize(tokenizer, text: str) -> np.ndarray:
    """Lowercase, encode, append EOS, truncate and pad with <pad> to 64 -> int64 (1, 64)."""
    tokenizer.no_padding()
    ids = tokenizer.encode(text.lower(), add_special_tokens=False).ids[: TEXT_LEN - 1]
    ids = ids + [EOS_ID]
    ids += [PAD_ID] * (TEXT_LEN - len(ids))
    return np.asarray([ids], dtype=np.int64)


class SiglipText:
    """Lazy text tower with an LRU of query embeddings; `unload()` frees the session."""

    def __init__(self, threads: int = 2, idle_unload_s: float = IDLE_UNLOAD_S) -> None:
        self.threads, self.idle_unload_s = threads, idle_unload_s
        self._lock = threading.Lock()
        self._session = None
        self._tok = None
        self._last = 0.0
        self._cache: dict[str, np.ndarray] = {}
        self._timer: threading.Timer | None = None

    @property
    def loaded(self) -> bool:
        return self._session is not None

    def _load(self) -> None:
        from tokenizers import Tokenizer

        self._tok = Tokenizer.from_file(str(model_path("siglip2_tokenizer")))
        self._session, _ = create_session(
            model_path("siglip2_text"), None, use_coreml=False, threads=self.threads
        )

    def embed(self, text: str) -> np.ndarray:
        """Query text -> (768,) L2-normalised vector."""
        key = text.strip().lower()
        with self._lock:
            hit = self._cache.pop(key, None)
            if hit is not None:
                self._cache[key] = hit  # LRU refresh
                self._touch()
                return hit
            if self._session is None:
                self._load()
            ids = tokenize(self._tok, key)
            pooled = self._session.run(["pooler_output"], {"input_ids": ids})[0]
            vec = _norm(pooled)[0]
            self._cache[key] = vec
            while len(self._cache) > 256:
                self._cache.pop(next(iter(self._cache)))
            self._touch()
            return vec

    def _touch(self) -> None:  # caller holds the lock
        self._last = time.monotonic()
        if self._timer is not None:
            self._timer.cancel()
        self._timer = threading.Timer(self.idle_unload_s + 1.0, self._maybe_unload)
        self._timer.daemon = True
        self._timer.start()

    def _maybe_unload(self) -> None:
        with self._lock:
            if self._session is not None and time.monotonic() - self._last >= self.idle_unload_s:
                self._unload()

    def _unload(self) -> None:
        self._session = None
        self._tok = None

    def unload(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._unload()


_text: SiglipText | None = None
_vision: dict[str, SiglipVision] = {}
_lock = threading.Lock()


def get_text() -> SiglipText:
    global _text
    with _lock:
        if _text is None:
            _text = SiglipText()
        return _text


def get_vision(variant: str = "fp32") -> SiglipVision:
    with _lock:
        if variant not in _vision:
            _vision[variant] = SiglipVision(variant)
        return _vision[variant]


def release_vision() -> None:
    """Drop the SigLIP vision sessions (hundreds of MB) once indexing is idle."""
    with _lock:
        _vision.clear()


def search_models_ready() -> bool:
    from shotsieve.models.registry import group_names, verify_model

    return all(verify_model(n) for n in group_names("search"))
