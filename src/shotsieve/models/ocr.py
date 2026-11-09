"""PaddleOCR on ONNX Runtime: DB text detector (PP-OCRv3) + CTC recogniser (PP-OCRv5 en).

`TextDetector.detect(img_rgb)` -> rotated boxes (4x2 float32 points in image coordinates, with a
score); `TextRecognizer.recognize(crops)` -> [(text, per-char probabilities)]. Both are lazy,
thread-safe (a lock per session) and run on the CPU provider (tiny models; CoreML is slower).
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

import cv2
import numpy as np

from shotsieve._ort import import_onnxruntime
from shotsieve.models.registry import model_path

ort = import_onnxruntime()

DET_LONG_EDGE = 608  # B2.2: 960 = +210 MB isolated, 608 = +121 MB (576 loses small digits)
DET_BINARIZE = 0.3
DET_BOX_SCORE = 0.6
DET_UNCLIP = 1.6
REC_HEIGHT = 48
REC_MAX_WIDTH = 320
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _session(path, threads: int = 2) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.log_severity_level = 3
    so.enable_cpu_mem_arena = False  # dynamic shapes: the arena only grows (RSS budget)
    so.enable_mem_pattern = False
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


@dataclass(frozen=True)
class TextBox:
    points: np.ndarray  # (4, 2) float32 ordered tl, tr, br, bl
    score: float

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Axis-aligned (x, y, w, h)."""
        x0, y0 = self.points.min(axis=0)
        x1, y1 = self.points.max(axis=0)
        return float(x0), float(y0), float(x1 - x0), float(y1 - y0)

    @property
    def sides(self) -> tuple[float, float]:
        """(width along the top edge, height along the left edge)."""
        p = self.points
        return float(np.linalg.norm(p[1] - p[0])), float(np.linalg.norm(p[3] - p[0]))

    @property
    def area(self) -> float:
        w, h = self.sides
        return w * h


def order_points(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).reshape(-1)
    return np.stack([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]])


class TextDetector:
    """DB text detector with the PaddleOCR post-process (binarize, min-area rect, unclip)."""

    def __init__(self, threads: int = 2) -> None:
        self._sess = _session(model_path("ppocr_det"), threads)
        self._in = self._sess.get_inputs()[0].name
        self._lock = threading.Lock()

    def _prep(self, img_rgb: np.ndarray) -> tuple[np.ndarray, float, float]:
        h, w = img_rgb.shape[:2]
        scale = min(1.0, DET_LONG_EDGE / max(h, w))
        nh = max(32, int(round(h * scale / 32)) * 32)
        nw = max(32, int(round(w * scale / 32)) * 32)
        im = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else
                        cv2.INTER_LINEAR)
        x = (im[:, :, ::-1].astype(np.float32) / 255.0 - _MEAN) / _STD  # PaddleOCR feeds BGR
        return x.transpose(2, 0, 1)[None].astype(np.float32), w / nw, h / nh

    def detect(
        self,
        img_rgb: np.ndarray,
        binarize: float = DET_BINARIZE,
        box_score: float = DET_BOX_SCORE,
        unclip: float = DET_UNCLIP,
        min_side: float = 3.0,
    ) -> list[TextBox]:
        x, sx, sy = self._prep(img_rgb)
        with self._lock:
            prob = self._sess.run(None, {self._in: x})[0][0, 0]
        mask = (prob > binarize).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        H, W = img_rgb.shape[:2]
        out: list[TextBox] = []
        for c in contours[:200]:
            if len(c) < 4:
                continue
            rect = cv2.minAreaRect(c)
            if min(rect[1]) < min_side:
                continue
            x0, y0, w, h = cv2.boundingRect(c)
            sub = np.zeros((h, w), np.uint8)
            cv2.drawContours(sub, [c - [x0, y0]], -1, 1, -1)
            score = float(cv2.mean(prob[y0 : y0 + h, x0 : x0 + w], sub)[0])
            if score < box_score:
                continue
            (cx, cy), (rw, rh), ang = rect
            dist = rw * rh * unclip / max(2.0 * (rw + rh), 1e-6)
            big = ((cx, cy), (rw + 2 * dist, rh + 2 * dist), ang)
            pts = order_points(cv2.boxPoints(big)) * np.array([sx, sy], np.float32)
            pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
            out.append(TextBox(pts, score))
        return out


def crop_box(img_rgb: np.ndarray, box: TextBox) -> np.ndarray:
    """Perspective-rectify a box to an upright crop (width along its top edge)."""
    w, h = box.sides
    w, h = max(int(round(w)), 2), max(int(round(h)), 2)
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    m = cv2.getPerspectiveTransform(box.points.astype(np.float32), dst)
    return cv2.warpPerspective(img_rgb, m, (w, h), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)


def load_charset() -> list[str]:
    lines = model_path("ppocr_dict").read_text(encoding="utf-8").split("\n")
    chars = [ln.rstrip("\r") for ln in lines if ln.rstrip("\r") != ""]
    return ["<blank>", *chars, " "]


class TextRecognizer:
    """PP-OCRv5 english recogniser with greedy CTC decoding."""

    def __init__(self, threads: int = 2) -> None:
        self._sess = _session(model_path("ppocr_rec"), threads)
        self._in = self._sess.get_inputs()[0].name
        self._chars = load_charset()
        self._lock = threading.Lock()

    @staticmethod
    def _prep(crop: np.ndarray) -> np.ndarray:
        h, w = crop.shape[:2]
        nw = int(min(REC_MAX_WIDTH, max(16, math.ceil(REC_HEIGHT * w / max(h, 1)))))
        im = cv2.resize(crop, (nw, REC_HEIGHT), interpolation=cv2.INTER_CUBIC)
        return ((im[:, :, ::-1].astype(np.float32) / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)

    def recognize(self, crops: list[np.ndarray]) -> list[tuple[str, list[float]]]:
        if not crops:
            return []
        arrs = [self._prep(c) for c in crops]
        width = max(a.shape[2] for a in arrs)
        width = max(width, 48)
        batch = np.zeros((len(arrs), 3, REC_HEIGHT, width), np.float32)
        for i, a in enumerate(arrs):
            batch[i, :, :, : a.shape[2]] = a
        with self._lock:
            out = self._sess.run(None, {self._in: batch})[0]
        res = []
        for i, a in enumerate(arrs):
            steps = max(1, int(math.ceil(out.shape[1] * a.shape[2] / width)))
            res.append(self.ctc_decode(out[i, :steps]))
        return res

    def ctc_decode(self, probs: np.ndarray) -> tuple[str, list[float]]:
        idx, conf = probs.argmax(axis=1), probs.max(axis=1)
        text, confs, prev = [], [], 0
        for k, c in zip(idx.tolist(), conf.tolist(), strict=True):
            if k != 0 and k != prev and k < len(self._chars):
                text.append(self._chars[k])
                confs.append(float(c))
            prev = k
        return "".join(text), confs


_det: TextDetector | None = None
_rec: TextRecognizer | None = None
_lock = threading.Lock()


def get_detector(threads: int = 4) -> TextDetector:
    global _det
    with _lock:
        if _det is None:
            _det = TextDetector(threads)
        return _det


def get_recognizer(threads: int = 4) -> TextRecognizer:
    global _rec
    with _lock:
        if _rec is None:
            _rec = TextRecognizer(threads)
        return _rec


def release_ocr() -> None:
    """Drop the OCR sessions once the indexer is idle."""
    global _det, _rec
    with _lock:
        _det = _rec = None


def ocr_models_ready() -> bool:
    from shotsieve.models.registry import group_names, verify_model

    return all(verify_model(n) for n in group_names("ocr"))
