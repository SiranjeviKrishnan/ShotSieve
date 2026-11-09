"""YOLOX-s (Megvii, Apache-2.0) object detector on ONNX Runtime (T22, optional model group `crop`).

Pre/post-processing follows opencv_zoo `models/object_detection_yolox` at the pinned commit:
letterbox keep-ratio to 640x640 with pad value 114 (top-left aligned), BGR, float32 0-255 (no
mean/std), HWC -> CHW; decode the `[1, 8400, 85]` output with grids for strides 8/16/32
(`xy = (out + grid) * stride`, `wh = exp(out) * stride`), `score = obj * cls`, confidence >= 0.35,
class-wise NMS (IoU 0.45), then undo the letterbox. CPU provider, 4 threads, arena off."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import cv2
import numpy as np

from shotsieve._ort import import_onnxruntime
from shotsieve.models.idle import IdleLoaded
from shotsieve.models.registry import model_path, verify_model
from shotsieve.models.runtime import create_session

ort = import_onnxruntime()

INPUT = 640
STRIDES = (8, 16, 32)
CONF = 0.35
NMS_IOU = 0.45
COCO = (
    "person bicycle car motorcycle airplane bus train truck boat traffic_light fire_hydrant "
    "stop_sign parking_meter bench bird cat dog horse sheep cow elephant bear zebra giraffe "
    "backpack umbrella handbag tie suitcase frisbee skis snowboard sports_ball kite baseball_bat "
    "baseball_glove skateboard surfboard tennis_racket bottle wine_glass cup fork knife spoon "
    "bowl banana apple sandwich orange broccoli carrot hot_dog pizza donut cake chair couch "
    "potted_plant bed dining_table toilet tv laptop mouse remote keyboard cell_phone microwave "
    "oven toaster sink refrigerator book clock vase scissors teddy_bear hair_drier toothbrush"
).split()
ANIMALS = frozenset(
    {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}
)


@dataclass(frozen=True)
class Detection:
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in the pixels of the input image
    score: float
    cls: int

    @property
    def name(self) -> str:
        return COCO[self.cls]


def _grids() -> tuple[np.ndarray, np.ndarray]:
    grids, strides = [], []
    for s in STRIDES:
        n = INPUT // s
        xv, yv = np.meshgrid(np.arange(n), np.arange(n))
        g = np.stack((xv, yv), 2).reshape(-1, 2)
        grids.append(g)
        strides.append(np.full((g.shape[0], 1), s))
    return np.concatenate(grids, 0).astype(np.float32), np.concatenate(strides, 0).astype(
        np.float32
    )


_GRIDS, _EXP_STRIDES = _grids()


def letterbox(img_rgb: np.ndarray) -> tuple[np.ndarray, float]:
    """RGB uint8 (H, W, 3) -> (float32 CHW 3x640x640 BGR 0-255, ratio)."""
    h, w = img_rgb.shape[:2]
    ratio = min(INPUT / h, INPUT / w)
    rw, rh = max(1, int(w * ratio)), max(1, int(h * ratio))
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    resized = cv2.resize(bgr, (rw, rh), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    padded = np.full((INPUT, INPUT, 3), 114.0, np.float32)
    padded[:rh, :rw] = resized
    return np.ascontiguousarray(padded.transpose(2, 0, 1)), float(ratio)


def decode(
    out: np.ndarray, ratio: float, conf: float = CONF, nms: float = NMS_IOU
) -> list[Detection]:
    """Decode a raw YOLOX output `[1, 8400, 85]` into detections in input-image pixels."""
    d = np.array(out, dtype=np.float32, copy=True).reshape(-1, out.shape[-1])
    d[:, :2] = (d[:, :2] + _GRIDS) * _EXP_STRIDES
    d[:, 2:4] = np.exp(d[:, 2:4]) * _EXP_STRIDES
    xywh = np.empty((d.shape[0], 4), np.float32)
    xywh[:, 0] = d[:, 0] - d[:, 2] / 2.0
    xywh[:, 1] = d[:, 1] - d[:, 3] / 2.0
    xywh[:, 2:] = d[:, 2:4]
    scores = d[:, 4:5] * d[:, 5:]
    best = scores.max(axis=1)
    cls = scores.argmax(axis=1)
    cand = np.nonzero(best >= conf)[0]  # NMSBoxesBatched is O(n^2) on its input: pre-filter
    if cand.size == 0:
        return []
    keep = cv2.dnn.NMSBoxesBatched(
        xywh[cand].tolist(), best[cand].tolist(), cls[cand].tolist(), conf, nms
    )
    out_list: list[Detection] = []
    for i in np.asarray(keep).reshape(-1):
        j = cand[int(i)]
        x, y, w, h = (xywh[j] / ratio).tolist()
        out_list.append(Detection((x, y, x + w, y + h), float(best[j]), int(cls[j])))
    out_list.sort(key=lambda t: -t.score)
    return out_list


class ObjectDetector:
    def __init__(self, threads: int = 4) -> None:
        self.session, self.provider = create_session(
            model_path("yolox_s"), None, use_coreml=False, threads=threads, arena=False
        )
        inp, outp = self.session.get_inputs()[0], self.session.get_outputs()[0]
        self.input_name, self.output_name = inp.name, outp.name
        self._lock = threading.Lock()

    def detect(self, img_rgb: np.ndarray) -> list[Detection]:
        """Detections (boxes clipped to the image) in the pixels of `img_rgb`, best first."""
        h, w = img_rgb.shape[:2]
        x, ratio = letterbox(img_rgb)
        with self._lock:
            out = self.session.run([self.output_name], {self.input_name: x[None]})[0]
        dets = decode(out, ratio)
        res = []
        for d in dets:
            b = d.box
            nb = (max(0.0, b[0]), max(0.0, b[1]), min(float(w), b[2]), min(float(h), b[3]))
            if nb[2] - nb[0] > 1 and nb[3] - nb[1] > 1:
                res.append(Detection(nb, d.score, d.cls))
        return res


def available() -> bool:
    return verify_model("yolox_s")


_holder: IdleLoaded[ObjectDetector] | None = None
_holder_lock = threading.Lock()


def holder(threads: int = 4, idle_s: float = 120.0) -> IdleLoaded[ObjectDetector]:
    global _holder
    with _holder_lock:
        if _holder is None:
            _holder = IdleLoaded(lambda: ObjectDetector(threads), idle_s)
        return _holder
