"""DINOv2-small patch tokens at dynamic size: saliency + point prompts (T22, no new weights).

A session of its own (not the cull embedder's), CPU provider, arena off, idle unload. The preview is
resized so the long edge is 448 and both sides are multiples of 14 (short side >= 224), aspect kept.
Tokens `P (gh, gw, 384)` are L2-normalised.

* saliency: background contrast on deep features. Border-ring patches -> k-means (k = 4, seeded)
  -> centroids `c_k`; `d_i = 1 - max_k cos(P_i, c_k)`; `s = blur(d, sigma = 1 patch)` scaled to
  [0, 1]; the peak region is the flood fill of `s >= 0.6` from the argmax.
* point prompt: 4-connected flood fill over patches with `cos(P_i, P_p) >= TAU_PT` or
  (`cos >= TAU_PT_LOW` and `s_i >= 0.5`), closed by one patch (dilate + erode); a region over
  25 % of the frame, or more than 4x wider/taller than the strict seed (over-segmentation: a band
  behind a tiny object), falls back to a 15 %-area box centred on the click (united with the seed).
  Subject extension / halo: `subject_extent` (B1, Review #12). The region -> box step
  is the only backend-specific part (a SAM backend would replace `point_region` only).
"""

from __future__ import annotations

import threading
from collections import OrderedDict

import cv2
import numpy as np

from shotsieve._ort import import_onnxruntime
from shotsieve.models.idle import IdleLoaded
from shotsieve.models.image_embed import _MEAN, _STD
from shotsieve.models.registry import model_path, verify_model
from shotsieve.models.runtime import create_session, np_dtype

ort = import_onnxruntime()

PATCH = 14
LONG_EDGE = 448
MIN_SHORT = 224
DIM = 384
TAU_PT = 0.60  # fitted on the train split only (scripts/make_crop_fixtures.py fit), frozen here
TAU_PT_LOW = 0.30  # train fit (4 big objects, grid 0.3-0.45): 3 of 4 reach IoU 0.5
SAL_PEAK = 0.60
SAL_SEED = 0.5
FALLBACK_AREA = 0.15
# B1 (Review #12), fitted on the TRAIN split only (scripts/fit_crop_b1.py): subject extension and
# the over-segmentation guard of the point prompt.
TAU_EXT = 0.60  # patch joins the subject when cos(token, mean token of the subject) >= TAU_EXT
GROW_CAP = 2.0  # the grown box may be at most GROW_CAP x the detected box area
TAU_HALO = 0.45  # looser similarity for the "may be cut" halo
SAL_HALO = 0.50  # ... or saliency >= SAL_HALO connected to the subject
HALO_CAP = 3.0  # halo bbox <= HALO_CAP x the detected box area
PT_SEED_MULT = 4.0  # point region wider/taller than 4 x the strict seed -> over-segmented
PT_MAX_AREA = 0.35  # ... or larger than 35 % of the frame (train fit: 0.25 lost object_03)


def grid_size(h: int, w: int) -> tuple[int, int, int, int]:
    """(gh, gw, H, W): patch grid and the resized pixel size for an image of h x w."""
    s = LONG_EDGE / max(h, w)
    rh, rw = h * s, w * s
    if min(rh, rw) < MIN_SHORT:
        s2 = MIN_SHORT / min(h, w)
        rh, rw = h * s2, w * s2
        if max(rh, rw) > 560:  # extreme aspect ratios: cap the long edge (patch count)
            k = 560 / max(rh, rw)
            rh, rw = rh * k, rw * k
    gh = max(1, int(round(rh / PATCH)))
    gw = max(1, int(round(rw / PATCH)))
    return gh, gw, gh * PATCH, gw * PATCH


class PatchExtractor:
    def __init__(self, threads: int = 4) -> None:
        self.session, self.provider = create_session(
            model_path("dinov2_small"), None, use_coreml=False, threads=threads, arena=False
        )
        inp, out = self.session.get_inputs()[0], self.session.get_outputs()[0]
        self.input_name, self.output_name = inp.name, out.name
        self.dtype = np_dtype(inp.type)
        self._lock = threading.Lock()

    def tokens(self, img_rgb: np.ndarray) -> np.ndarray:
        """(gh, gw, 384) float32 L2-normalised patch tokens."""
        h, w = img_rgb.shape[:2]
        gh, gw, H, W = grid_size(h, w)
        small = cv2.resize(img_rgb, (W, H), interpolation=cv2.INTER_AREA)
        x = (small.astype(np.float32) / 255.0 - _MEAN) / _STD
        x = np.ascontiguousarray(x.transpose(2, 0, 1))[None].astype(self.dtype, copy=False)
        with self._lock:
            hidden = self.session.run([self.output_name], {self.input_name: x})[0]
        p = hidden[0, 1:, :].astype(np.float32)
        if p.shape[0] != gh * gw:  # a model with register tokens: keep the last gh*gw
            p = p[-gh * gw :]
        p = p.reshape(gh, gw, -1)
        return p / np.maximum(np.linalg.norm(p, axis=-1, keepdims=True), 1e-12)


def available() -> bool:
    return verify_model("dinov2_small")


_holder: IdleLoaded[PatchExtractor] | None = None
_holder_lock = threading.Lock()


def holder(threads: int = 4, idle_s: float = 120.0) -> IdleLoaded[PatchExtractor]:
    global _holder
    with _holder_lock:
        if _holder is None:
            _holder = IdleLoaded(lambda: PatchExtractor(threads), idle_s)
        return _holder


# ---- numpy-only analysis (testable with synthetic tokens) -------------------------------------


def _kmeans(x: np.ndarray, k: int, iters: int = 12) -> np.ndarray:
    """Deterministic k-means (farthest-point initialisation from the first sample)."""
    k = min(k, len(x))
    cent = [x[0]]
    for _ in range(1, k):
        d = np.min([1.0 - x @ c for c in cent], axis=0)
        cent.append(x[int(np.argmax(d))])
    c = np.stack(cent)
    for _ in range(iters):
        lab = np.argmax(x @ c.T, axis=1)
        for j in range(k):
            m = lab == j
            if m.any():
                v = x[m].mean(0)
                c[j] = v / max(np.linalg.norm(v), 1e-12)
    return c


def saliency(tokens: np.ndarray) -> np.ndarray:
    """(gh, gw) float32 in [0, 1]: how unlike the border (background) each patch is."""
    gh, gw, _ = tokens.shape
    ring = np.zeros((gh, gw), bool)
    ring[0, :] = ring[-1, :] = True
    ring[:, 0] = ring[:, -1] = True
    if gh > 4 and gw > 4:
        ring[1, :] = ring[-2, :] = True
        ring[:, 1] = ring[:, -2] = True
    cent = _kmeans(tokens[ring], 4)
    d = 1.0 - np.max(tokens.reshape(-1, tokens.shape[-1]) @ cent.T, axis=1).reshape(gh, gw)
    d = cv2.GaussianBlur(d.astype(np.float32), (0, 0), 1.0)
    d -= d.min()
    m = d.max()
    return (d / m).astype(np.float32) if m > 1e-9 else np.zeros((gh, gw), np.float32)


def _component(mask: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=4)
    if not mask[seed]:
        return np.zeros_like(mask)
    return lab == lab[seed]


def _box_of(region: np.ndarray) -> tuple[float, float, float, float]:
    """Normalised bbox of a patch mask, padded by half a patch."""
    gh, gw = region.shape
    ys, xs = np.nonzero(region)
    return (
        float(max(0.0, (xs.min() - 0.5) / gw)), float(max(0.0, (ys.min() - 0.5) / gh)),
        float(min(1.0, (xs.max() + 1.5) / gw)), float(min(1.0, (ys.max() + 1.5) / gh)),
    )  # fmt: skip


def peak_region(sal: np.ndarray) -> tuple[float, float, float, float] | None:
    """Box of the flood fill of `sal >= 0.6` from the saliency maximum."""
    if sal.max() <= 0:
        return None
    seed = np.unravel_index(int(np.argmax(sal)), sal.shape)
    region = _component(sal >= SAL_PEAK, (int(seed[0]), int(seed[1])))
    return _box_of(region) if region.any() else None


def point_region(
    tokens: np.ndarray, sal: np.ndarray, x: float, y: float
) -> tuple[tuple[float, float, float, float], str]:
    """Box (normalised) around the object under the point (x, y), and how it was made
    ("region" | "fallback")."""
    gh, gw, _ = tokens.shape
    pj = min(gw - 1, max(0, int(x * gw)))
    pi = min(gh - 1, max(0, int(y * gh)))
    cos = tokens @ tokens[pi, pj]
    mask = (cos >= TAU_PT) | ((cos >= TAU_PT_LOW) & (sal >= SAL_SEED))
    region = _component(mask, (pi, pj))
    k = np.ones((3, 3), np.uint8)  # close one-patch holes (dilate then erode), keep the extent
    region = cv2.erode(cv2.dilate(region.astype(np.uint8), k), k).astype(bool)
    strict = _component(cos >= TAU_PT, (pi, pj))
    if strict.any() and region.any():
        sb, rb = _exact_box(strict), _exact_box(region)
        sw, sh = max(sb[2] - sb[0], 1.0 / gw), max(sb[3] - sb[1], 1.0 / gh)
        if (rb[2] - rb[0]) > PT_SEED_MULT * sw or (rb[3] - rb[1]) > PT_SEED_MULT * sh:
            return _fallback_box(x, y, _box_of(strict)), "fallback"  # over-segmented (a band)
    if region.mean() > PT_MAX_AREA or not region.any():
        return _fallback_box(x, y, None), "fallback"
    return _box_of(region), "region"


def _fallback_box(
    x: float, y: float, seed_box: tuple[float, float, float, float] | None
) -> tuple[float, float, float, float]:
    """The 15 %-area box centred on the click, united with the strict seed's box (a tiny object
    is then still fully inside)."""
    s = FALLBACK_AREA**0.5
    b = (max(0.0, x - s / 2), max(0.0, y - s / 2), min(1.0, x + s / 2), min(1.0, y + s / 2))
    if seed_box is None:
        return b
    return (min(b[0], seed_box[0]), min(b[1], seed_box[1]), max(b[2], seed_box[2]),
            max(b[3], seed_box[3]))  # fmt: skip


def _patch_mask(box: tuple[float, float, float, float], gh: int, gw: int) -> np.ndarray:
    """Patches whose centre lies inside `box` (at least the patch under the box centre)."""
    ys = (np.arange(gh) + 0.5) / gh
    xs = (np.arange(gw) + 0.5) / gw
    m = ((ys >= box[1]) & (ys <= box[3]))[:, None] & ((xs >= box[0]) & (xs <= box[2]))[None, :]
    if not m.any():
        ci = min(gh - 1, max(0, int((box[1] + box[3]) / 2 * gh)))
        cj = min(gw - 1, max(0, int((box[0] + box[2]) / 2 * gw)))
        m[ci, cj] = True
    return m


def _exact_box(region: np.ndarray) -> tuple[float, float, float, float]:
    gh, gw = region.shape
    ys, xs = np.nonzero(region)
    return (float(xs.min()) / gw, float(ys.min()) / gh, float(xs.max() + 1) / gw,
            float(ys.max() + 1) / gh)  # fmt: skip


def _grow(
    seed: np.ndarray, allowed: np.ndarray, max_area: float
) -> np.ndarray:
    """4-connected growth of `seed` inside `allowed`, one ring at a time, stopping before the
    bbox of the region would exceed `max_area` (normalised)."""
    k = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    region = seed.copy()
    gh, gw = seed.shape
    for _ in range(max(gh, gw)):
        nxt = cv2.dilate(region.astype(np.uint8), k).astype(bool) & (allowed | region)
        if (nxt == region).all():
            break
        b = _exact_box(nxt)
        if (b[2] - b[0]) * (b[3] - b[1]) > max_area:
            break
        region = nxt
    return region


def pack_mask(mask: np.ndarray) -> tuple[int, int, str]:
    return (mask.shape[0], mask.shape[1], np.packbits(mask.ravel()).tobytes().hex())


def unpack_mask(packed: tuple[int, int, str]) -> np.ndarray:
    gh, gw, hx = packed
    bits = np.unpackbits(np.frombuffer(bytes.fromhex(hx), np.uint8))[: gh * gw]
    return bits.reshape(gh, gw).astype(bool)


def subject_extent(
    tokens: np.ndarray,
    sal: np.ndarray | None,
    box: tuple[float, float, float, float],
    blockers: list[tuple[float, float, float, float]] = (),
    grow: bool = True,
) -> tuple[tuple[float, float, float, float], tuple[int, int, str]]:
    """(grown box, halo mask) of a detected subject.

    The grown box adds the patches 4-connected to the subject whose token is within `TAU_EXT`
    cosine of the subject's mean token (capped at `GROW_CAP` x the box area, never into a blocker
    box): a dog's chest or a cat's tail the detector left out. The halo is the same growth at the
    looser `TAU_HALO`, plus saliency >= `SAL_HALO` connected to the subject (cap `HALO_CAP`): the
    region a crop edge should not pass through without a warning."""
    gh, gw, _ = tokens.shape
    base = _patch_mask(box, gh, gw)
    area = max((box[2] - box[0]) * (box[3] - box[1]), 1.0 / (gh * gw))
    blocked = np.zeros((gh, gw), bool)
    for b in blockers:
        blocked |= _patch_mask(b, gh, gw)
    blocked &= ~base
    proto = tokens[base].mean(0)
    proto /= max(float(np.linalg.norm(proto)), 1e-12)
    cos = tokens @ proto
    ext_box = box
    region = base
    if grow:
        region = _grow(base, (cos >= TAU_EXT) & ~blocked, GROW_CAP * area)
        eb = _exact_box(region)
        ext_box = (min(box[0], eb[0]), min(box[1], eb[1]), max(box[2], eb[2]), max(box[3], eb[3]))
    allowed = ((cos >= TAU_HALO) | ((sal >= SAL_HALO) if sal is not None else False)) & ~blocked
    halo = _grow(region, allowed, HALO_CAP * area)
    return ext_box, pack_mask(halo | region)


class TokenLRU:
    """The last few images' tokens (about 540 KB each): point prompts after the first are warm."""

    def __init__(self, n: int = 8) -> None:
        self.n = n
        self._d: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> np.ndarray | None:
        with self._lock:
            v = self._d.get(key)
            if v is not None:
                self._d.move_to_end(key)
            return v

    def put(self, key: str, v: np.ndarray) -> None:
        with self._lock:
            self._d[key] = v
            self._d.move_to_end(key)
            while len(self._d) > self.n:
                self._d.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._d.clear()
