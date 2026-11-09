"""Recognition-only face scan (T23h): find every face, including small / noisy / background ones.

The cull keeps using `FaceDetector.detect` (one 640 px pass, score >= 0.7, at most `max_faces`
faces) for every decision. This module runs in the background, never in the cull path: YuNet over
several scales and overlapping tiles with a low score threshold, merged by NMS, then a second check
rejects false positives (stricter score for tiny boxes, landmark geometry, optional MediaPipe
landmarker). The faces found go to recognition / People only (`Cache.put_rec_faces`)."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from shotsieve.models.face_detect import Face

FACE_SCAN_VERSION = 1  # bump when the scan or its thresholds change: shoots are re-scanned
REC_EDGE = 4096  # long edge of the image the scan works on
TILE = 1280
OVERLAP = 0.15
# plan entries: ("whole", long edge) or ("tiles", scale); tile windows are TILE / scale px of source
DEFAULT_PLAN: tuple[tuple[str, float], ...] = (("whole", 640), ("whole", 1280), ("tiles", 1.0),
                                               ("tiles", 2.0))  # fmt: skip
CAND_SCORE = 0.3  # candidates below this are never kept


@dataclass(frozen=True)
class ScanCfg:
    """Acceptance thresholds. A face is kept when score >= the entry of the first size band whose
    `max_side` it does not exceed (long side of the box in pixels of the scanned image)."""

    bands: tuple[tuple[float, float], ...] = ((24.0, 0.6), (48.0, 0.8), (1e9, 0.8))
    plan: tuple[tuple[str, float], ...] = DEFAULT_PLAN
    geometry: bool = True
    nms_iou: float = 0.3


def windows(w: int, h: int, win: float, overlap: float = OVERLAP) -> list[tuple[int, int, int, int]]:  # noqa: E501
    """Overlapping square windows (side `win`) covering a w x h image; one window if it fits."""
    win = max(32, int(round(win)))
    if w <= win * 1.1 and h <= win * 1.1:
        return [(0, 0, w, h)]
    step = max(1, int(win * (1 - overlap)))

    def axis(n: int) -> list[int]:
        if n <= win:
            return [0]
        xs = list(range(0, n - win, step))
        xs.append(n - win)
        return xs

    return [(x, y, min(w, x + win), min(h, y + win)) for y in axis(h) for x in axis(w)]


def _shift(f: Face, dx: float, dy: float) -> Face:
    x, y, w, h = f.bbox
    return Face((x + dx, y + dy, w, h), f.landmarks + np.array([dx, dy], np.float32), f.score)


def candidates(img_rgb: np.ndarray, det, plan=DEFAULT_PLAN, score: float = CAND_SCORE) -> list[Face]:  # noqa: E501
    """Every detection of every pass (unmerged), in the coordinates of `img_rgb`."""
    h, w = img_rgb.shape[:2]
    long_edge = max(h, w)
    done: list[float] = []
    out: list[Face] = []
    for kind, v in plan:
        scale = v / long_edge if kind == "whole" else float(v)
        if kind == "whole" and scale >= 1.0:
            scale = 1.0
            kind, v = "tiles", 1.0
        if any(abs(scale - d) < 0.05 * d for d in done):
            continue  # the same effective scale was already run (small image, "whole" >= native)
        done.append(scale)
        if kind == "whole":
            out += det.detect_scaled(img_rgb, scale, score)
            continue
        for x0, y0, x1, y1 in windows(w, h, TILE / scale):
            out += [_shift(f, x0, y0) for f in det.detect_scaled(img_rgb[y0:y1, x0:x1], scale, score)]  # noqa: E501
    return out


def _iou(a, b) -> float:
    iw = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    ih = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    i = iw * ih
    return i / (a[2] * a[3] + b[2] * b[3] - i)


def _contain(a, b) -> float:
    """Share of the smaller box inside the other."""
    iw = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    ih = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    return iw * ih / max(min(a[2] * a[3], b[2] * b[3]), 1e-9)


def nms(faces: list[Face], iou: float = 0.3) -> list[Face]:
    """Greedy NMS by score; a box 80 % inside a kept, higher-scoring box is a duplicate too."""
    keep: list[Face] = []
    for f in sorted(faces, key=lambda f: f.score, reverse=True):
        if any(_iou(f.bbox, k.bbox) > iou or _contain(f.bbox, k.bbox) > 0.8 for k in keep):
            continue
        keep.append(f)
    return keep


def side(f: Face) -> float:
    return max(f.bbox[2], f.bbox[3])


def plausible(f: Face) -> bool:
    """Loose landmark geometry (a strict one rejected real profile / tilted faces on the train
    split): sane aspect, eyes above the mouth, every landmark within 25 % of the box."""
    x, y, w, h = f.bbox
    lm = f.landmarks
    if w < 4 or h < 4 or not 0.5 <= w / h <= 2.0:
        return False
    if not (lm[0][1] + lm[1][1]) / 2 < (lm[3][1] + lm[4][1]) / 2:
        return False
    return bool(
        np.all(lm[:, 0] >= x - 0.25 * w) and np.all(lm[:, 0] <= x + 1.25 * w)
        and np.all(lm[:, 1] >= y - 0.25 * h) and np.all(lm[:, 1] <= y + 1.25 * h)
    )  # fmt: skip


def accept(f: Face, cfg: ScanCfg) -> bool:
    s = side(f)
    for max_side, thr in cfg.bands:
        if s <= max_side:
            if f.score < thr:
                return False
            break
    return plausible(f) if cfg.geometry else True


def scan(img_rgb: np.ndarray, det, cfg: ScanCfg | None = None) -> list[Face]:
    """All accepted faces of one image, largest first (coordinates of `img_rgb`)."""
    cfg = cfg or ScanCfg()
    cands = candidates(img_rgb, det, cfg.plan)
    kept = nms([f for f in cands if f.score >= min(b[1] for b in cfg.bands)], cfg.nms_iou)
    kept = [f for f in kept if accept(f, cfg)]
    kept.sort(key=lambda f: f.bbox[2] * f.bbox[3], reverse=True)
    return kept


def timed(fn, *a):
    t0 = time.perf_counter()
    r = fn(*a)
    return r, time.perf_counter() - t0


# ---- the background job ----------------------------------------------------------------------
SMALL_EMBED = 24  # faces below this many px get the denoise + Lanczos pre-step before SFace
MAX_REC_FACES = 60
RSS_BUDGET_MB = 700  # the scan may grow the process by at most this (it frees and waits above)
BLACK_LUMA = 8.0


def stamp_of(fp: str, n_imp: int) -> str:
    return f"{FACE_SCAN_VERSION}|{fp}|{n_imp}"


def _norm_iou(a, b) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    i = iw * ih
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def scan_image(img_rgb: np.ndarray, known: list[tuple[float, float, float, float]], det, emb,
               cfg: ScanCfg | None = None) -> tuple[list[list[float]], np.ndarray]:
    """Recognition faces of one image that are NOT one of the `known` (normalised x0,y0,x1,y1)
    decision faces already embedded by the cull: (normalised [x0,y0,x1,y1,score] boxes, (K,128))."""
    cfg = cfg or ScanCfg()
    h, w = img_rgb.shape[:2]
    found = scan(img_rgb, det, cfg)
    boxes: list[list[float]] = []
    vecs = []
    for f in found:
        x, y, bw, bh = f.bbox
        nb = (max(0.0, x / w), max(0.0, y / h), min(1.0, (x + bw) / w), min(1.0, (y + bh) / h))
        if any(_norm_iou(nb, k) > 0.3 for k in known):
            continue
        boxes.append([round(v, 5) for v in nb] + [round(f.score, 3)])
        vecs.append(emb.embed_small(img_rgb, f) if side(f) < SMALL_EMBED else emb.embed(img_rgb, f))
        if len(boxes) >= MAX_REC_FACES:
            break
    return boxes, (np.stack(vecs).astype(np.float32) if vecs else np.zeros((0, 128), np.float32))


class FaceScanner:
    """Background recognition scan of one shoot (its own daemon thread, paused while a cull runs,
    RSS-budgeted). `run()` does the same inline."""

    def __init__(self, folder, settings, on_done=None) -> None:
        import threading

        self.folder = folder
        self.settings = settings
        self.on_done = on_done
        self.done = self.total = 0
        self.faces = 0
        self.error: str | None = None
        self.t_work = 0.0
        self.peak_delta_mb = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start(self) -> None:
        import threading

        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self.run, name="facescan", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def status(self) -> dict:
        return {"running": self.running, "done": self.done, "total": self.total,
                "faces": self.faces, "error": self.error, "peak_delta_mb": self.peak_delta_mb}  # fmt: skip  # noqa: E501

    def _wait(self) -> bool:
        from shotsieve.pipeline.indexer import cull_running

        while cull_running() and not self._stop.is_set():
            time.sleep(0.1)
        return not self._stop.is_set()

    def run(self) -> None:
        import gc

        import psutil

        from shotsieve.io.formats import scan_folder
        from shotsieve.io.preview import PreviewError, load_preview
        from shotsieve.models.face_detect import get_rec_face_detector
        from shotsieve.models.face_embed import get_face_embedder
        from shotsieve.pipeline import people
        from shotsieve.pipeline.cache import Cache
        from shotsieve.pipeline.indexer import _return_freed_pages

        proc = psutil.Process()
        base = proc.memory_info().rss
        try:
            with Cache.for_folder(self.folder) as cache:
                epoch = cache.people_epoch()
                stamps = cache.rec_stamps()
                n_imp = cache.n_important_faces()
                sig = cache.image_sigs()
                todo = []
                for e in scan_folder(self.folder):
                    if ".shotsieve" in e.master.parts:
                        continue
                    p = str(e.master)
                    fp = sig.get(p)
                    if fp is None:
                        continue  # not culled (no cache row): nothing to recognise yet
                    if stamps.get(p) != stamp_of(fp, n_imp.get(p, 0)):
                        todo.append((p, e.preview_source, fp))
                self.total = len(todo)
                det, emb = get_rec_face_detector(), get_face_embedder()
                for p, src, fp in todo:
                    if not self._wait():
                        return
                    t0 = time.perf_counter()
                    raw = cache.raw_features(p)
                    black = raw is not None and raw.mean_luma < BLACK_LUMA
                    boxes, vecs = [], np.zeros((0, 128), np.float32)
                    if not black:
                        try:
                            img = load_preview(src, REC_EDGE)
                        except PreviewError:
                            img = None
                        if img is not None:
                            known = []
                            if raw is not None and raw.width and raw.height:
                                known = [
                                    (f.bbox[0] / raw.width, f.bbox[1] / raw.height,
                                     (f.bbox[0] + f.bbox[2]) / raw.width,
                                     (f.bbox[1] + f.bbox[3]) / raw.height)
                                    for f in raw.faces if f.important
                                ]  # fmt: skip
                            boxes, vecs = scan_image(img, known, det, emb)
                            del img
                    cache.put_rec_faces(p, stamp_of(fp, n_imp.get(p, 0)), boxes, vecs, epoch)
                    self.faces += len(boxes)
                    self.done += 1
                    self.t_work += time.perf_counter() - t0
                    delta = (proc.memory_info().rss - base) / 2**20
                    self.peak_delta_mb = max(self.peak_delta_mb, delta)
                    if delta > RSS_BUDGET_MB:  # over budget: hand pages back before going on
                        gc.collect()
                        _return_freed_pages()
                if todo and not self._stop.is_set() and cache.people_consent(
                    self.settings.people_consent
                ):
                    people.cluster(cache)
        except Exception as exc:  # noqa: BLE001 - surfaced through status()
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            gc.collect()
            _return_freed_pages()
            if self.on_done:
                self.on_done()


def needs_rescan(cache) -> bool:
    """True when some culled image has no valid recognition scan (cheap: no decoding)."""
    stamps = cache.rec_stamps()
    n_imp = cache.n_important_faces()
    return any(stamps.get(p) != stamp_of(fp, n_imp.get(p, 0)) for p, fp in cache.image_sigs().items())  # noqa: E501
