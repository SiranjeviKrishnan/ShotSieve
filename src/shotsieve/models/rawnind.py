"""RawNIND UtNet2 raw denoiser (T26): the primary, bundled denoiser for Bayer RAW files.

The network (Brummer and De Vleeschouwer, arXiv:2501.08924; weights CC-BY-4.0, converted to ONNX by
`scripts/export_rawnind.py`, see docs/02_MODELS.md) takes the Bayer mosaic itself, packed as four
planes (R, G1, G2, B), and returns demosaiced, denoised linear camera RGB at twice the pack size
(= the sensor size). So the denoiser works on the raw data before any tone curve, where the noise is
still simple, instead of on a rendered 8-bit picture.

Pipeline (every part of the engine is in this file; the server only asks for pixels):

1. `load_scene` reads `raw_image_visible`, subtracts the black levels, normalises by the white level
   and packs the mosaic (any of RGGB / BGGR / GRBG / GBRG is shifted by at most one pixel so the
   pack starts on an R site). A camera -> sRGB matrix is fitted against libraw's half-size render.
   X-Trans, Foveon, linear DNGs and 4-colour sensors are not Bayer: `supports` is False and the
   caller keeps NAFNet.
2. The pack is cut into 512 px tiles (32 px margin, feathered, blended band by band so a 24 MP image
   never holds more than about 150 MB of accumulator) and every tile is run through the model.
   One output gain per image (the model's output level is only defined up to ~3 %: it is matched to
   the input's mean over a few probe tiles) is shared by Fit, region and full, so a region is the
   same picture as the crop of the full image.
3. The linear RGB goes through the fitted matrix, the BT.709 curve libraw uses and a per-channel
   tone curve matched to the picture the viewer shows without denoising (the same `match_tone` that
   `finish_demosaic` uses), so the denoised view has the colours and brightness of the normal view.
4. The server blends that render with the normal view (luma level-scaled, chroma full) exactly like
   NAFNet (`models.denoise.blend_luma_chroma`).

CoreML (all units, 0.06 s/MP) runs the tiles; every tile is checked for NaN/Inf and an implausible
range (the network's activations reach ~6.5e4, near fp16's limit) and one bad tile moves the whole
process to the CPU provider (0.3 s/MP), which is remembered. Nothing here writes a file.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from shotsieve._ort import import_onnxruntime
from shotsieve.io.formats import RAW_EXTS
from shotsieve.log import get_logger
from shotsieve.models.denoise import Cancelled, blend_luma_chroma, strength_scale  # noqa: F401
from shotsieve.models.idle import IdleLoaded
from shotsieve.models.registry import model_path, verify_model

ort = import_onnxruntime()
log = get_logger("rawnind")

MODEL = "rawnind_utnet2"
BACKEND = "rawnind"  # the backend id in cache keys, headers and the UI
VERSION = 1  # bump when the look of a RawNIND result changes (it is part of every cache key)
TILE = 512  # the model's fixed input size, in pack pixels (the output tile is 2x2 larger)
MARGIN = 32  # pack pixels of each tile discarded in favour of its neighbour (feathered)
WORKERS = 3  # tiles in flight on CoreML
IDLE_S = 60.0
OUT_MAX = 1e8  # the network's raw output is ~1e5..1e6 (gain ~1e-6 later); beyond this: a failed run
FRAME_TOL = 0.03  # the displayed frame may be this much smaller than the sensor (a camera crop)
CHUNK_ROWS = 128  # output rows per piece handed to the caller
PROBES = (1 / 3, 2 / 3)  # probe tile centres for the output gain (fractions of the pack)

_PATTERNS = ((0, 0), (0, 1), (1, 0), (1, 1))  # (dy, dx) of the planes R, G1, G2, B from the R site


class FrameMismatch(ValueError):
    """The picture the viewer shows is not a plain crop of the sensor frame (cannot register)."""


def available() -> bool:
    """True when the model file is present (it ships in the app). SHOTSIEVE_DENOISE_BACKEND=basic or
    =nafnet switches RawNIND off (tests, a machine where it misbehaves)."""
    if os.environ.get("SHOTSIEVE_DENOISE_BACKEND", "").lower() in ("basic", "nafnet"):
        return False
    return verify_model(MODEL)


# ---- Bayer pack ------------------------------------------------------------------------------


def bayer_offset(colors: np.ndarray) -> tuple[int, int] | None:
    """(oy, ox) of the first R site if `colors` (libraw colour index per pixel: 0 R, 1 G, 2 B, 3 G2)
    is a plain 2x2 Bayer mosaic (RGGB, BGGR, GRBG or GBRG; the two greens may carry either label),
    else None (X-Trans 6x6, 4-colour sensors, Foveon)."""
    if colors.ndim != 2 or min(colors.shape) < 4:
        return None
    c = colors[:8, :8]
    p = c[:2, :2]
    if not (np.array_equal(c[2:, :], c[:-2, :]) and np.array_equal(c[:, 2:], c[:, :-2])):
        return None  # not period 2
    reds = np.argwhere(p == 0)
    blues = np.argwhere(p == 2)
    if len(reds) != 1 or len(blues) != 1:
        return None
    (ry, rx), (by, bx) = reds[0], blues[0]
    if (ry + by) != 1 or (rx + bx) != 1:  # B must sit on the other diagonal of the 2x2 cell
        return None
    return int(ry), int(rx)


def pack_bayer(
    img: np.ndarray, colors: np.ndarray, black: np.ndarray | list[float], white: float
) -> tuple[np.ndarray, tuple[int, int]]:
    """(4, h, w) float32 planes (R, G1, G2, B) in 0..1 from a Bayer mosaic, and the (oy, ox) of the
    first R site: the pack covers sensor pixels [oy, oy + 2h) x [ox, ox + 2w). `black` is indexed by
    libraw colour index. Raises ValueError for a non-Bayer layout."""
    off = bayer_offset(colors)
    if off is None:
        raise ValueError("not a 2x2 Bayer mosaic")
    oy, ox = off
    H, W = img.shape
    h, w = (H - oy) // 2, (W - ox) // 2
    out = np.empty((4, h, w), np.float32)
    bl = np.asarray(black, np.float32)
    for k, (dy, dx) in enumerate(_PATTERNS):
        ch = int(colors[oy + dy, ox + dx])
        plane = img[oy + dy : oy + dy + 2 * h : 2, ox + dx : ox + dx + 2 * w : 2]
        out[k] = (plane.astype(np.float32) - bl[ch]) / max(1.0, float(white) - float(bl[ch]))
    np.clip(out, 0.0, 1.0, out=out)
    return out, off


def unpack_bayer(pack: np.ndarray) -> np.ndarray:
    """The inverse of the plane split: an (2h, 2w) RGGB-ordered mosaic of the pack."""
    _, h, w = pack.shape
    m = np.empty((2 * h, 2 * w), pack.dtype)
    for k, (dy, dx) in enumerate(_PATTERNS):
        m[dy::2, dx::2] = pack[k]
    return m


def binned_rgb(pack: np.ndarray) -> np.ndarray:
    """(h, w, 3) camera RGB at half the sensor size: R, mean of the greens, B."""
    return np.stack([pack[0], (pack[1] + pack[2]) * 0.5, pack[3]], -1)


# ---- display frame ---------------------------------------------------------------------------


def _swap(orient: int, w: int, h: int) -> tuple[int, int]:
    return (h, w) if orient in (5, 6, 7, 8) else (w, h)


def sensor_point(orient: int, u: int, v: int, H: int, W: int) -> tuple[int, int]:
    """Sensor (y, x) of pixel (u, v) of the oriented image (EXIF orientation 1..8 applied to an
    H x W array by `preview.apply_orientation`)."""
    return {
        1: (v, u), 2: (v, W - 1 - u), 3: (H - 1 - v, W - 1 - u), 4: (H - 1 - v, u),
        5: (u, v), 6: (H - 1 - u, v), 7: (H - 1 - u, W - 1 - v), 8: (u, W - 1 - v),
    }[orient]  # fmt: skip


def sensor_box(
    orient: int, box: tuple[int, int, int, int], H: int, W: int
) -> tuple[int, int, int, int]:
    """Bounding box (y0, y1, x0, x1), half-open, in sensor pixels of the box (u0, v0, u1, v1) of
    the oriented image."""
    u0, v0, u1, v1 = box
    (ya, xa), (yb, xb) = (
        sensor_point(orient, u0, v0, H, W),
        sensor_point(orient, u1 - 1, v1 - 1, H, W),
    )
    return min(ya, yb), max(ya, yb) + 1, min(xa, xb), max(xa, xb) + 1


@dataclass(frozen=True)
class Frame:
    """How the displayed picture (`ref_hw`, what the viewer shows without denoising) sits in the
    oriented sensor render: a centred crop starting at (cx, cy)."""

    orient: int
    sensor_hw: tuple[int, int]
    ref_hw: tuple[int, int]
    cx: int
    cy: int


def make_frame(orient: int, sensor_hw: tuple[int, int], ref_hw: tuple[int, int]) -> Frame:
    H, W = sensor_hw
    Wo, Ho = _swap(orient, W, H)
    rh, rw = ref_hw
    from shotsieve.io.viewer import crop_dims

    Wc, Hc = crop_dims(Wo, Ho, rw / rh)  # a 16:9 JPEG of a 3:2 sensor: centred aspect crop first
    if rw > Wc or rh > Hc or (Wc - rw) > FRAME_TOL * Wc or (Hc - rh) > FRAME_TOL * Hc:
        raise FrameMismatch(f"displayed {rw}x{rh} does not fit the sensor frame {Wo}x{Ho}")
    return Frame(orient, sensor_hw, ref_hw, (Wo - rw) // 2, (Ho - rh) // 2)


# ---- model session ---------------------------------------------------------------------------

_cpu_only = False  # CoreML produced a bad tile in this process: stay on the CPU provider
_cpu_lock = threading.Lock()


def _bad(y: np.ndarray) -> bool:
    """A failed run: NaN / Inf, an absurd magnitude, or values flat-lined at fp16's maximum (a half
    precision overflow that did not reach Inf)."""
    if not np.isfinite(y).all():
        return True
    m = float(np.abs(y).max())
    return m > OUT_MAX or (m == 65504.0 and int(np.count_nonzero(np.abs(y) == 65504.0)) > 8)


class _Net:
    """The ONNX session: tile in (4, T, T) float32 pack -> (3, 2T, 2T) float32 linear camera RGB."""

    def __init__(self, sessions: list | None = None) -> None:
        self.provider = "cpu"
        self.sessions: list = []
        if sessions is not None:  # tests: ready-made sessions, first one preferred
            self.sessions = list(sessions)
            self.provider = getattr(sessions[0], "provider", "cpu")
        else:
            path = str(model_path(MODEL))
            so = ort.SessionOptions()
            so.log_severity_level = 3
            so.enable_cpu_mem_arena = False  # the arena would stay resident after a 24 MP image
            so.enable_mem_pattern = False
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            forced = os.environ.get("SHOTSIEVE_RAWNIND_EP", "").lower()
            if (
                "CoreMLExecutionProvider" in ort.get_available_providers()
                and not _cpu_only
                and forced != "cpu"
            ):
                try:
                    s = ort.InferenceSession(
                        path, so,
                        providers=[("CoreMLExecutionProvider",
                                    {"MLComputeUnits": "ALL", "ModelFormat": "MLProgram"}),
                                   "CPUExecutionProvider"],
                    )  # fmt: skip
                    if s.get_providers()[0] == "CoreMLExecutionProvider":
                        self.sessions.append(s)
                        self.provider = "coreml"
                except Exception as exc:  # noqa: BLE001
                    log.warning("CoreML RawNIND session failed (%s); using the CPU", exc)
            so2 = ort.SessionOptions()
            so2.log_severity_level = 3
            so2.enable_cpu_mem_arena = False
            so2.enable_mem_pattern = False
            so2.intra_op_num_threads = min(8, os.cpu_count() or 4)
            self._cpu_opts, self._path = so2, path
            if not self.sessions:
                self.sessions.append(self._cpu_session())
        self.name = self.sessions[0].get_inputs()[0].name

    def _cpu_session(self):
        return ort.InferenceSession(self._path, self._cpu_opts, providers=["CPUExecutionProvider"])

    def _fall_back(self) -> None:
        global _cpu_only
        with _cpu_lock:
            _cpu_only = True
            if len(self.sessions) > 1 or self.provider == "cpu":
                self.sessions = self.sessions[-1:]
            else:
                self.sessions = [self._cpu_session()]
            self.provider = "cpu"

    def tile(self, t: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(t[None], dtype=np.float32)
        y = self.sessions[0].run(None, {self.name: x})[0][0]
        if _bad(y):
            if self.provider == "cpu" or len(self.sessions) == 0:
                raise RuntimeError("RawNIND produced a non-finite tile on the CPU")
            log.warning(
                "RawNIND tile failed the NaN/range check on %s: using the CPU", self.provider
            )
            self._fall_back()
            y = self.sessions[0].run(None, {self.name: x})[0][0]
            if _bad(y):
                raise RuntimeError("RawNIND produced a non-finite tile on the CPU")
        return y


_holder: IdleLoaded[_Net] | None = None
_holder_lock = threading.Lock()


def holder(idle_s: float = IDLE_S) -> IdleLoaded[_Net]:
    global _holder
    with _holder_lock:
        if _holder is None:
            _holder = IdleLoaded(_Net, idle_s)
        return _holder


def release() -> None:
    """Drop the session and the cached scene now (tests; memory pressure)."""
    with _holder_lock:
        h = _holder
    if h is not None:
        h.unload()
    release_scene()


def loaded() -> bool:
    with _holder_lock:
        return _holder is not None and _holder.loaded


def warm() -> bool:
    """Load the model and run one tile so CoreML compiles now. False when it is not installed."""
    if not available():
        return False
    try:
        with holder().use() as net:
            net.tile(np.full((4, TILE, TILE), 0.1, np.float32))
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("RawNIND warm-up failed: %s", exc)
        return False


# ---- tiling ----------------------------------------------------------------------------------


def _origins(n: int, tile: int, stride: int) -> list[int]:
    if n <= tile:
        return [0]
    out = list(range(0, n - tile, stride))
    out.append(n - tile)
    return out


def _ramp(n: int, edge: int) -> np.ndarray:
    w = np.ones(n, np.float32)
    r = (np.arange(edge, dtype=np.float32) + 1.0) / (edge + 1.0)
    w[:edge] = r
    w[-edge:] = r[::-1]
    return w


def _window(lo: int, hi: int, n: int, tile: int, margin: int) -> tuple[int, int]:
    """The part of an axis of length n to run: [lo, hi) plus a margin each side, widened to a full
    tile (of real pixels) when the image is big enough, clamped to [0, n)."""
    a, b = max(0, lo - margin), min(n, hi + margin)
    if b - a < tile <= n:
        grow = tile - (b - a)
        a = max(0, a - grow // 2)
        b = min(n, a + tile)
        a = max(0, b - tile)
    return a, b


def iter_bands(
    pack: np.ndarray,
    run_tile: Callable[[np.ndarray], np.ndarray],
    rect: tuple[int, int, int, int] | None = None,
    tile: int = TILE,
    margin: int = MARGIN,
    workers: int = 1,
    pause: Callable[[], None] | None = None,
    cancel: Callable[[], bool] | None = None,
    progress: Callable[[float], None] | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Run `run_tile` ((4,T,T) pack -> (3,2T,2T)) over `rect` = (py0, py1, px0, px1) of the pack
    (default: all of it) and yield (y_out0, band (3, n, 2*(px1-px0)) float32): finished output rows
    in order, y_out0 in output pixels (2x pack). Tiles overlap by 2*margin and are blended with
    feathered weights, band by band (the weights are separable, so a horizontal pass per band and a
    vertical blend between bands is exactly the 2-D weighted blend). Images smaller than a tile are
    reflected up to one. A rect is widened by `margin` and to a whole tile so its pixels never come
    from padding."""
    _, H, W = pack.shape
    py0, py1, px0, px1 = rect or (0, H, 0, W)
    wy0, wy1 = _window(py0, py1, H, tile, margin)
    wx0, wx1 = _window(px0, px1, W, tile, margin)
    wh, ww = wy1 - wy0, wx1 - wx0
    stride = tile - 2 * margin
    ys, xs = _origins(wh, tile, stride), _origins(ww, tile, stride)
    ramp = _ramp(2 * tile, 2 * margin)
    w2 = 2 * max(ww, tile)  # a window narrower than a tile is reflected up to one
    out_y0, out_y1 = 2 * py0, 2 * py1
    cx0, cx1 = 2 * (px0 - wx0), 2 * (px1 - wx0)
    total, done = len(ys) * len(xs), 0

    def read(y0: int, x0: int) -> np.ndarray:
        t = pack[:, wy0 + y0 : wy0 + y0 + tile, wx0 + x0 : wx0 + x0 + tile]
        ph, pw = tile - t.shape[1], tile - t.shape[2]
        if ph or pw:
            t = np.pad(
                t, ((0, 0), (0, ph), (0, pw)), mode="reflect" if min(t.shape[1:]) > 1 else "edge"
            )
        return np.ascontiguousarray(t)

    def job(y0: int, x0: int) -> np.ndarray:
        if cancel is not None and cancel():
            raise Cancelled
        if pause is not None:
            pause()
        return run_tile(read(y0, x0))

    limit = 2 * wh  # valid output rows of the window (a reflected tile has more)
    pend: tuple[int, np.ndarray, np.ndarray] | None = None  # (base row, weighted acc, weights)
    pool = ThreadPoolExecutor(max_workers=max(1, workers))

    def finish(base: int, acc: np.ndarray, w: np.ndarray) -> Iterator[tuple[int, np.ndarray]]:
        n = min(acc.shape[1], limit - base)
        a, b = max(base + 2 * wy0, out_y0), min(base + n + 2 * wy0, out_y1)
        for r in range(a, b, CHUNK_ROWS):  # small pieces: the caller's temporaries stay small
            r0, r1 = r - base - 2 * wy0, min(r + CHUNK_ROWS, b) - base - 2 * wy0
            yield r, acc[:, r0:r1, cx0:cx1] / np.maximum(w[r0:r1], 1e-6)[None, :, None]

    try:
        for y0 in ys:
            acc = np.zeros((3, 2 * tile, w2), np.float32)
            wsum = np.zeros(w2, np.float32)
            results = pool.map(lambda x0, y0=y0: job(y0, x0), xs)
            for x0, res in zip(xs, results, strict=True):
                acc[:, :, 2 * x0 : 2 * x0 + 2 * tile] += res * ramp[None, None, :]
                wsum[2 * x0 : 2 * x0 + 2 * tile] += ramp
                done += 1
            acc /= np.maximum(wsum, 1e-6)[None, None, :]
            acc *= ramp[None, :, None]  # vertical weights; `cur_w` keeps their sum
            cur_w = ramp.copy()
            base = 2 * y0
            if pend is not None:
                pbase, pacc, pw = pend
                n_emit = base - pbase
                ov = pbase + pacc.shape[1] - base
                if ov > 0:
                    acc[:, :ov] += pacc[:, n_emit : n_emit + ov]
                    cur_w[:ov] += pw[n_emit : n_emit + ov]
                yield from finish(pbase, pacc[:, :n_emit], pw[:n_emit])
            pend = (base, acc, cur_w)
            if progress:
                progress(done / total)
        if pend is not None:
            yield from finish(*pend)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


# ---- scene (one RAW file, read once) ----------------------------------------------------------


@dataclass
class Scene:
    path: str
    mtime: int
    pack: np.ndarray  # (4, h, w) float32
    offset: tuple[int, int]
    sensor_hw: tuple[int, int]  # visible sensor size (H, W)
    orient: int
    M: np.ndarray  # (3, 3) float32: binned camera RGB -> linear sRGB


_SCENE: dict = {"key": None, "scene": None, "t": 0.0}
_scene_lock = threading.Lock()
_MATRIX: dict[tuple, np.ndarray] = {}
_SUPPORT: dict[tuple, bool] = {}


def _open(path: str):
    import rawpy

    return rawpy.imread(path)


def _is_bayer(raw) -> bool:
    import rawpy

    return (
        raw.raw_type == rawpy.RawType.Flat
        and raw.raw_image_visible is not None
        and bayer_offset(raw.raw_colors_visible) is not None
    )


def supports(
    path: str | Path, ref_hw: tuple[int, int] | None = None, mtime: int | None = None
) -> bool:
    """True when `path` is a Bayer RAW the engine can denoise and (with `ref_hw`, the displayed
    size) whose displayed frame is a crop of the sensor frame. Cached per file version."""
    p = Path(path)
    if p.suffix.lower() not in RAW_EXTS:
        return False
    try:
        mt = p.stat().st_mtime_ns if mtime is None else mtime
    except OSError:
        return False
    key = (str(p), mt, ref_hw)
    if key not in _SUPPORT:
        ok = False
        try:
            with _open(str(p)) as raw:
                if _is_bayer(raw):
                    ok = True
                    if ref_hw is not None:
                        from shotsieve.io.preview import _RAW_FLIP_TO_EXIF

                        make_frame(
                            _RAW_FLIP_TO_EXIF.get(raw.sizes.flip, 1),
                            tuple(raw.raw_image_visible.shape),
                            ref_hw,
                        )
        except FrameMismatch:
            ok = False
        except Exception as exc:  # noqa: BLE001  (unreadable / truncated file: not ours)
            log.debug("rawnind.supports(%s): %s", p.name, exc)
            ok = False
        if len(_SUPPORT) > 20000:
            _SUPPORT.clear()
        _SUPPORT[key] = ok
    return _SUPPORT[key]


def _fit_matrix(raw, pack: np.ndarray) -> np.ndarray:
    """3x3 matrix taking the binned camera RGB (R, mean G, B; black-subtracted, 0..1, no white
    balance) to libraw's linear sRGB at camera white balance: least squares against libraw's own
    half-size render of the same file, unclipped samples only."""
    ref16 = raw.postprocess(
        half_size=True, use_camera_wb=True, no_auto_bright=True, gamma=(1, 1),
        output_bps=16, user_flip=0,
    )  # fmt: skip
    h, w = min(ref16.shape[0], pack.shape[1]), min(ref16.shape[1], pack.shape[2])
    k = 3  # a sparse grid of samples is plenty for nine unknowns, and keeps the temporaries small
    R = ref16[:h:k, :w:k].reshape(-1, 3).astype(np.float32) / 65535.0
    A = binned_rgb(pack[:, :h:k, :w:k]).reshape(-1, 3)
    del ref16
    ok = (A.max(1) < 0.9) & (R.max(1) < 0.9) & (A.max(1) > 0.005)
    if ok.sum() < 100:
        ok = np.ones(len(A), bool)
    M, *_ = np.linalg.lstsq(A[ok], R[ok], rcond=None)
    return np.ascontiguousarray(M.T, dtype=np.float32)  # dst = M @ src


def load_scene(path: str | Path, mtime: int | None = None) -> Scene:
    p = Path(path)
    mt = p.stat().st_mtime_ns if mtime is None else mtime
    with _open(str(p)) as raw:
        if not _is_bayer(raw):
            raise ValueError("not a Bayer RAW")
        pack, off = pack_bayer(
            raw.raw_image_visible,
            raw.raw_colors_visible,
            np.array(raw.black_level_per_channel, np.float32),
            float(raw.white_level),
        )
        sensor_hw = tuple(raw.raw_image_visible.shape)
        from shotsieve.io.preview import _RAW_FLIP_TO_EXIF

        orient = _RAW_FLIP_TO_EXIF.get(raw.sizes.flip, 1)
        key = (str(p), mt)
        M = _MATRIX.get(key)
        if M is None:
            M = _fit_matrix(raw, pack)
            if len(_MATRIX) > 5000:
                _MATRIX.clear()
            _MATRIX[key] = M
    return Scene(str(p), mt, pack, off, sensor_hw, orient, M)  # type: ignore[arg-type]


def scene_for(path: str | Path, mtime: int | None = None) -> Scene:
    """The scene of `path`, kept in memory for a short while (a viewing session asks for many
    regions of one image)."""
    p = Path(path)
    mt = p.stat().st_mtime_ns if mtime is None else mtime
    key = (str(p), mt)
    with _scene_lock:
        if _SCENE["key"] == key and _SCENE["scene"] is not None:
            _SCENE["t"] = time.monotonic()
            return _SCENE["scene"]
        _SCENE.update(key=None, scene=None)
    sc = load_scene(p, mt)
    with _scene_lock:
        _SCENE.update(key=key, scene=sc, t=time.monotonic())
    return sc


def release_scene() -> None:
    with _scene_lock:
        _SCENE.update(key=None, scene=None)


# ---- strength --------------------------------------------------------------------------------

ALPHA = {"low": 0.5, "medium": 0.8, "high": 1.0}  # luma share of the RawNIND render (tuned, T26)
CHROMA = {
    "low": 1.0,
    "medium": 1.0,
    "high": 1.0,
}  # chroma share: colour mottling goes at every level


def alphas(level: str, sigma: float) -> tuple[float, float]:
    """(luma, chroma) blend of the RawNIND render into the normal view. The luma share scales with
    the level and the noise of the photo (a clean image is barely touched, like NAFNet's blend);
    chroma is taken (almost) fully once there is visible noise."""
    from shotsieve.models.denoise import ai_chroma_alpha

    return float(ALPHA[level] * strength_scale(sigma)), ai_chroma_alpha(level, sigma) * CHROMA[
        level
    ]


# ---- colour ----------------------------------------------------------------------------------


def _gamma_table() -> np.ndarray:
    v = np.arange(4096, dtype=np.float64) / 4095.0
    g = np.where(v <= 0.018, 4.5 * v, 1.099 * np.power(np.maximum(v, 1e-12), 0.45) - 0.099)
    return np.clip(np.round(g * 255.0), 0, 255).astype(np.uint8)


_GAMMA = _gamma_table()


def to_srgb8(rows: np.ndarray, M: np.ndarray, tone: np.ndarray | None) -> np.ndarray:
    """(3, n, w) linear camera RGB -> (n, w, 3) uint8 sRGB: matrix, BT.709 curve, tone curve."""
    a = np.ascontiguousarray(rows.transpose(1, 2, 0), dtype=np.float32)
    lin = cv2.transform(a, M)
    np.clip(lin, 0.0, 1.0, out=lin)
    u8 = _GAMMA[(lin * 4095.0 + 0.5).astype(np.uint16)]
    return cv2.LUT(u8, tone) if tone is not None else u8


@dataclass
class Params:
    gain: float
    tone: np.ndarray  # (256, 1, 3) uint8


_PARAMS: dict[tuple, Params] = {}
_params_lock = threading.Lock()


def _gamma_small(pack_small: np.ndarray, M: np.ndarray) -> np.ndarray:
    b = binned_rgb(pack_small)
    lin = np.clip(cv2.transform(np.ascontiguousarray(b), M), 0.0, 1.0)
    return _GAMMA[(lin * 4095.0 + 0.5).astype(np.uint16)]


def tone_curve(scene: Scene, frame: Frame, ref: np.ndarray) -> np.ndarray:
    """The per-channel curve (same `match_tone` as `finish_demosaic`) that takes this raw's own
    sRGB render, from the NOISY pack, to the picture the viewer shows without denoising. Fitted on a
    320 px copy of the noisy half-size render, so it does not depend on the region asked for."""
    from shotsieve.io.preview import apply_orientation
    from shotsieve.io.viewer import match_tone, resize_rgb

    pk = scene.pack
    k = max(1, round(max(pk.shape[1:]) / 640))
    small_pack = np.stack(
        [cv2.resize(pl, None, fx=1 / k, fy=1 / k, interpolation=cv2.INTER_AREA) for pl in pk]
    )
    img = apply_orientation(_gamma_small(small_pack, scene.M), frame.orient)
    H, W = scene.sensor_hw
    Wo, Ho = _swap(frame.orient, W, H)
    sx, sy = img.shape[1] / Wo, img.shape[0] / Ho  # the display frame inside the small render
    rh, rw = frame.ref_hw
    x0, y0 = round(frame.cx * sx), round(frame.cy * sy)
    x1, y1 = round((frame.cx + rw) * sx), round((frame.cy + rh) * sy)
    crop = np.ascontiguousarray(img[y0 : max(y0 + 1, y1), x0 : max(x0 + 1, x1)])
    small = resize_rgb(crop, 320)
    refs = cv2.resize(ref, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA)
    return match_tone(small, refs)


def probe_gain(scene: Scene, run_tile: Callable[[np.ndarray], np.ndarray]) -> float:
    """One output gain per image: sum(input) / sum(output) over a few probe tiles (binned camera RGB
    means, so the three output channels are compared like with like). The model's output level is
    only defined up to ~3 % between tiles; one number per image keeps Fit, region and full equal."""
    pk = scene.pack
    _, H, W = pk.shape
    th, tw = min(TILE, H), min(TILE, W)
    sin = sout = 0.0
    for fy in PROBES:
        for fx in PROBES:
            y0 = int(np.clip(fy * H - th / 2, 0, H - th))
            x0 = int(np.clip(fx * W - tw / 2, 0, W - tw))
            t = pk[:, y0 : y0 + th, x0 : x0 + tw]
            if th < TILE or tw < TILE:
                t = np.pad(t, ((0, 0), (0, TILE - th), (0, TILE - tw)), mode="reflect")
            t = np.ascontiguousarray(t)
            y = run_tile(t)
            sin += float(binned_rgb(t).sum()) * 4.0  # per output pixel: 4 of them per binned one
            sout += float(y.sum())
    return float(sin / sout) if sout > 1e-9 and sin > 0 else 1.0


_TONES: dict[tuple, np.ndarray] = {}


def tone_for(scene: Scene, frame: Frame, ref: np.ndarray) -> np.ndarray:
    key = (scene.path, scene.mtime, ref.shape[:2])
    with _params_lock:
        t = _TONES.get(key)
    if t is None:
        t = tone_curve(scene, frame, ref)
        with _params_lock:
            if len(_TONES) > 200:
                _TONES.clear()
            _TONES[key] = t
    return t


def params_for(
    scene: Scene, frame: Frame, ref: np.ndarray, run_tile: Callable[[np.ndarray], np.ndarray]
) -> Params:
    key = (scene.path, scene.mtime, ref.shape[:2])
    with _params_lock:
        p = _PARAMS.get(key)
    if p is None:
        p = Params(probe_gain(scene, run_tile), tone_for(scene, frame, ref))
        with _params_lock:
            if len(_PARAMS) > 200:
                _PARAMS.clear()
            _PARAMS[key] = p
    return p


# ---- rendering -------------------------------------------------------------------------------


def render(
    path: str | Path,
    ref: np.ndarray,
    box: tuple[int, int, int, int] | None = None,
    *,
    mtime: int | None = None,
    run_tile: Callable[[np.ndarray], np.ndarray] | None = None,
    pause: Callable[[], None] | None = None,
    cancel: Callable[[], bool] | None = None,
    progress: Callable[[float], None] | None = None,
) -> np.ndarray:
    """The denoised sRGB uint8 picture of a Bayer RAW in the frame of `ref` (the picture the viewer
    shows without denoising: its size, orientation and tone), whole or the pixel box
    (x0, y0, x1, y1)
    of it. `run_tile` replaces the model (tests); by default the loaded network runs."""
    if run_tile is not None:
        return _render(path, ref, box, mtime, run_tile, 1, pause, cancel, progress)
    with holder().use() as net:
        gain_run = net.tile
        return _render(
            path, ref, box, mtime, gain_run, WORKERS if net.provider == "coreml" else 1,
            pause, cancel, progress,
        )  # fmt: skip


def _render(path, ref, box, mtime, raw_tile, workers, pause, cancel, progress) -> np.ndarray:
    scene = scene_for(path, mtime)
    frame = make_frame(scene.orient, scene.sensor_hw, ref.shape[:2])
    params = params_for(scene, frame, ref, raw_tile)
    g = params.gain

    def run_tile(t: np.ndarray) -> np.ndarray:
        return raw_tile(t) * g

    rh, rw = ref.shape[:2]
    u0, v0, u1, v1 = box if box is not None else (0, 0, rw, rh)
    u0, v0, u1, v1 = max(0, u0), max(0, v0), min(rw, u1), min(rh, v1)
    if u1 <= u0 or v1 <= v0:
        raise ValueError("empty box")
    H, W = scene.sensor_hw
    # display box -> oriented sensor render -> sensor pixels
    sy0, sy1, sx0, sx1 = sensor_box(
        frame.orient, (u0 + frame.cx, v0 + frame.cy, u1 + frame.cx, v1 + frame.cy), H, W
    )
    oy, ox = scene.offset
    h, w = scene.pack.shape[1:]
    # sensor px -> output px (pack coordinates x2), clamped to what the pack covers
    ay0, ay1 = max(sy0 - oy, 0), min(sy1 - oy, 2 * h)
    ax0, ax1 = max(sx0 - ox, 0), min(sx1 - ox, 2 * w)
    py0, py1, px0, px1 = ay0 // 2, -(-ay1 // 2), ax0 // 2, -(-ax1 // 2)
    sens = np.empty((sy1 - sy0, sx1 - sx0, 3), np.uint8)
    got = np.zeros(sy1 - sy0, bool)
    for y_out, band in iter_bands(
        scene.pack,
        run_tile,
        (py0, py1, px0, px1),
        workers=workers,
        pause=pause,
        cancel=cancel,
        progress=progress,
    ):
        n = band.shape[1]
        # band columns are [2*px0, 2*px1) output px; rows start at y_out
        cols = band[:, :, ax0 - 2 * px0 : ax1 - 2 * px0]
        rows_y0 = max(y_out, ay0)
        rows_y1 = min(y_out + n, ay1)
        if rows_y1 <= rows_y0:
            continue
        cols = cols[:, rows_y0 - y_out : rows_y1 - y_out]
        px = to_srgb8(cols, scene.M, params.tone)
        ty0 = rows_y0 + oy - sy0
        sens[ty0 : ty0 + px.shape[0], ax0 + ox - sx0 : ax0 + ox - sx0 + px.shape[1]] = px
        got[ty0 : ty0 + px.shape[0]] = True
    if not got.all() or ax0 + ox > sx0 or ax1 + ox < sx1:
        _fill_edges(sens, got, ax0 + ox - sx0, ax1 + ox - sx0)
    from shotsieve.io.preview import apply_orientation

    # the sensor box is exactly the display box, oriented: an isometry, so no further cropping
    return apply_orientation(sens, frame.orient)


def render_base(
    path: str | Path,
    ref: np.ndarray,
    box: tuple[int, int, int, int] | None = None,
    *,
    mtime: int | None = None,
) -> np.ndarray:
    """The picture Low / Medium blend the RawNIND render INTO: our own demosaic of the same raw
    data (libraw, camera white balance, BT.709 curve, no auto brightness) through the same tone
    curve and the same frame mapping as `render`, so both have exactly the same geometry. It is
    never the picture the viewer shows without denoising: a camera's embedded JPEG is usually
    distortion-corrected (a few pixels at the corners), and a blend of two geometries doubles
    edges. `box` is a pixel box (x0, y0, x1, y1) of the frame of `ref`, as in `render`."""
    from shotsieve.io.preview import apply_orientation

    scene = scene_for(path, mtime)
    frame = make_frame(scene.orient, scene.sensor_hw, ref.shape[:2])
    tone = tone_for(scene, frame, ref)
    with _open(scene.path) as raw:
        img = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=8, user_flip=0)
    H, W = scene.sensor_hw
    if (
        img.shape[0] != H or img.shape[1] != W
    ):  # libraw's active area vs the visible area: <= a few px
        img = np.ascontiguousarray(
            cv2.copyMakeBorder(
                img[:H, :W],
                0,
                max(0, H - img.shape[0]),
                0,
                max(0, W - img.shape[1]),
                cv2.BORDER_REPLICATE,
            )
        )
    img = apply_orientation(cv2.LUT(img, tone), frame.orient)
    rh, rw = ref.shape[:2]
    u0, v0, u1, v1 = box if box is not None else (0, 0, rw, rh)
    u0, v0, u1, v1 = max(0, u0), max(0, v0), min(rw, u1), min(rh, v1)
    if u1 <= u0 or v1 <= v0:
        raise ValueError("empty box")
    return np.ascontiguousarray(img[v0 + frame.cy : v1 + frame.cy, u0 + frame.cx : u1 + frame.cx])


def _fill_edges(sens: np.ndarray, got: np.ndarray, c0: int, c1: int) -> None:
    """Rows / columns of the sensor box the pack does not cover (a BGGR-style 1 px offset, an odd
    size) take their nearest covered neighbour."""
    if not got.any():
        return
    idx = np.flatnonzero(got)
    for y in np.flatnonzero(~got):
        sens[y] = sens[idx[np.abs(idx - y).argmin()]]
    if c0 > 0:
        sens[:, :c0] = sens[:, c0 : c0 + 1]
    if c1 < sens.shape[1]:
        sens[:, c1:] = sens[:, max(c1 - 1, 0) : max(c1, 1)]
