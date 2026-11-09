"""View-only denoising (T23f): NAFNet-SIDD-width32 (MIT) on ONNX Runtime, OpenCV NLM fallback.

Nothing here touches an original or the cull: the loupe shows a denoised copy that the server caches
under `<shoot>/.shotsieve/denoise/`.

* **AI** (`nafnet_sidd`, optional model group `denoise`, ~124 MB): the published ONNX is a tf2onnx
  conversion with a fixed 1x3x256x256 input. Its Conv attributes are repaired in memory
  (`onnx_conv_fix`) so the CoreML provider accepts it (GPU + Neural Engine, 3 tiles in flight);
  the image is cut into overlapping 256 px tiles (16 px margin, linear feathering, reflection at the
  image border), run, and blended band by band, so a 24 MP image never needs more than a few tile
  rows of float32 accumulators.
* **Basic** (no model, or CoreML/ORT failing): `cv2.fastNlMeansDenoisingColored` per tile with a
  strength scaled to the noise estimated from the image (and its ISO).
* Strength Low / Medium / High: AI blends the model output with the input (residual scaling),
  scaled by the estimated noise so a clean image is barely touched; Basic scales NLM's `h`.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from shotsieve._ort import import_onnxruntime
from shotsieve.log import get_logger
from shotsieve.models.idle import IdleLoaded
from shotsieve.models.onnx_conv_fix import fix_conv_file
from shotsieve.models.registry import bundled, model_path, models_dir, verify_model

ort = import_onnxruntime()
log = get_logger("denoise")

TILE = 256  # the model's fixed input size
MARGIN = 16  # px of each tile discarded in favour of its neighbour (feathered)
LEVELS = ("low", "medium", "high")
AI_ALPHA = {"low": 0.45, "medium": 0.75, "high": 1.0}  # blend towards the model output
NLM_K = {"low": 0.7, "medium": 1.0, "high": 1.4}  # NLM h = k * estimated sigma
WORKERS = 3  # tiles in flight: ~30 % faster than 1 on the M1 Pro (CoreML queues them)
IDLE_S = 60.0


class Cancelled(RuntimeError):
    """The caller asked to stop (a newer request superseded this one)."""


def available() -> bool:
    """True when the AI model file is installed (size check only). SHOTSIEVE_DENOISE_BACKEND=basic
    switches the model off (tests, a machine where the Neural Engine misbehaves)."""
    if os.environ.get("SHOTSIEVE_DENOISE_BACKEND", "").lower() == "basic":
        return False
    return verify_model("nafnet_sidd")


# ---- noise estimate --------------------------------------------------------

_LAP = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)


def estimate_sigma(rgb: np.ndarray) -> float:
    """Noise standard deviation in 8-bit levels: median of the Immerkaer (1996) response on the
    luma of up to ~2 MP of the image, which stays robust to edges and texture. Typical: ISO 100 low
    noise ~ 0.8, ISO 1600 ~ 3, ISO 12800 ~ 9."""
    h, w = rgb.shape[:2]
    step = max(1, int(np.sqrt(h * w / 2_000_000)))
    sub = rgb[::step, ::step] if step > 1 else rgb
    luma = cv2.cvtColor(sub, cv2.COLOR_RGB2GRAY).astype(np.float32)
    r = np.abs(cv2.filter2D(luma, -1, _LAP))[2:-2, 2:-2]
    if r.size == 0:
        return 0.0
    return float(np.median(r) / (0.6745 * 6.0))


def prior_sigma(iso: int | None) -> float | None:
    return None if not iso or iso <= 0 else 0.9 * float(np.sqrt(iso / 100.0))


def effective_sigma(rgb: np.ndarray, iso: int | None = None) -> float:
    """The noise level used for strength scaling: the image estimate kept within 0.5x..2.5x of what
    the ISO predicts (a textured scene fools the estimate upwards, a smooth one downwards)."""
    est = estimate_sigma(rgb)
    prior = prior_sigma(iso)
    if prior is None:
        return max(est, 0.3)
    return float(np.clip(est, 0.5 * prior, 2.5 * prior))


# ---- tiling ----------------------------------------------------------------


def _origins(n: int, tile: int, stride: int) -> list[int]:
    if n <= tile:
        return [0]
    out = list(range(0, n - tile, stride))
    out.append(n - tile)
    return out


def _ramp(tile: int, margin: int) -> np.ndarray:
    w = np.ones(tile, np.float32)
    ramp = (np.arange(margin, dtype=np.float32) + 1.0) / (margin + 1.0)
    w[:margin] = ramp
    w[-margin:] = ramp[::-1]
    return w


def run_tiled(
    rgb: np.ndarray,
    run_tile: Callable[[np.ndarray], np.ndarray],
    workers: int = WORKERS,
    pause: Callable[[], None] | None = None,
    cancel: Callable[[], bool] | None = None,
    progress: Callable[[float], None] | None = None,
) -> np.ndarray:
    """Apply `run_tile` (uint8 TILExTILEx3 RGB -> float32 RGB 0..255, same shape) over the image
    with overlapping tiles blended by feathered weights. Images smaller than a tile are reflected
    up to one. `pause()` is called between tiles (lets an urgent request go first); `cancel()`
    returning True raises `Cancelled`."""
    h0, w0 = rgb.shape[:2]
    pad_h, pad_w = max(0, TILE - h0), max(0, TILE - w0)
    if pad_h or pad_w:
        rgb = cv2.copyMakeBorder(rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    h, w = rgb.shape[:2]
    stride = TILE - 2 * MARGIN
    ys, xs = _origins(h, TILE, stride), _origins(w, TILE, stride)
    wy, wx = _ramp(TILE, MARGIN), _ramp(TILE, MARGIN)
    win = wy[:, None] * wx[None, :]
    out = np.empty((h, w, 3), np.uint8)
    cap = TILE + stride + 1  # rows an accumulator window has to hold (two neighbouring bands)
    acc = np.zeros((cap, w, 3), np.float32)
    wacc = np.zeros((cap, w), np.float32)
    base = 0
    total = len(ys) * len(xs)
    done = 0

    def flush(upto: int) -> None:  # rows [base, upto) are final
        n = upto - base
        if n <= 0:
            return
        ww = np.maximum(wacc[:n], 1e-6)[..., None]
        out[base:upto] = np.clip(acc[:n] / ww + 0.5, 0, 255).astype(np.uint8)

    def job(x0: int, y0: int) -> np.ndarray:
        if cancel is not None and cancel():
            raise Cancelled
        if pause is not None:
            pause()
        return run_tile(np.ascontiguousarray(rgb[y0 : y0 + TILE, x0 : x0 + TILE]))

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for y0 in ys:
            flush(y0)
            keep = base + cap - y0  # rows of the window still needed by this band
            if y0 > base:
                shift = y0 - base
                acc[:keep] = acc[shift : shift + keep]
                wacc[:keep] = wacc[shift : shift + keep]
                acc[keep:] = 0
                wacc[keep:] = 0
                base = y0
            for x0, res in zip(xs, pool.map(lambda x0, y0=y0: job(x0, y0), xs), strict=True):
                ty = y0 - base
                acc[ty : ty + TILE, x0 : x0 + TILE] += res * win[..., None]
                wacc[ty : ty + TILE, x0 : x0 + TILE] += win
                done += 1
            if progress:
                progress(done / total)
        flush(base + min(cap, h - base))
    return out[:h0, :w0]


# ---- AI engine -------------------------------------------------------------


def _coreml_ready_path() -> str:
    """The repaired copy of the verified model (built once, streamed from a memory map)."""
    src = model_path("nafnet_sidd")
    if bundled():  # the release bundle ships the repaired file itself (build_release.sh)
        return str(src)
    dst = models_dir() / "nafnet_sidd_width32_coreml.onnx"
    if not dst.is_file() or dst.stat().st_mtime < src.stat().st_mtime:
        fix_conv_file(src, dst)
    return str(dst)


class _NafNet:
    def __init__(self) -> None:
        fixed = _coreml_ready_path()
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.enable_cpu_mem_arena = False  # with 3 tiles in flight the arena adds ~230 MB resident
        so.enable_mem_pattern = False
        self.provider = "cpu"
        self.session = None
        if "CoreMLExecutionProvider" in ort.get_available_providers():
            try:
                s = ort.InferenceSession(
                    fixed, so,
                    providers=[("CoreMLExecutionProvider",
                                {"MLComputeUnits": "ALL", "ModelFormat": "MLProgram"}),
                               "CPUExecutionProvider"],
                )  # fmt: skip
                if s.get_providers()[0] == "CoreMLExecutionProvider":
                    self.session, self.provider = s, "coreml"
            except Exception as exc:  # noqa: BLE001
                log.warning("CoreML denoise session failed (%s); using the CPU", exc)
        if self.session is None:
            so.intra_op_num_threads = 6
            self.session = ort.InferenceSession(fixed, so, providers=["CPUExecutionProvider"])
        self.name = self.session.get_inputs()[0].name

    def tile(self, rgb: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0)
        y = self.session.run(None, {self.name: x})[0][0]
        return np.clip(y.transpose(1, 2, 0), 0.0, 1.0) * 255.0


_holder: IdleLoaded[_NafNet] | None = None
_holder_lock = threading.Lock()


def holder(idle_s: float = IDLE_S) -> IdleLoaded[_NafNet]:
    """The lazily loaded model; the session is dropped `idle_s` seconds after the last use."""
    global _holder
    with _holder_lock:
        if _holder is None:
            _holder = IdleLoaded(_NafNet, idle_s)
        return _holder


def release() -> None:
    """Drop the session now (tests; memory pressure)."""
    with _holder_lock:
        h = _holder
    if h is not None:
        h.unload()


def loaded() -> bool:
    with _holder_lock:
        return _holder is not None and _holder.loaded


# ---- strength --------------------------------------------------------------


def ai_alpha(level: str, sigma: float) -> float:
    """Share of the model output mixed into the image: the level, scaled down for a clean image."""
    return float(AI_ALPHA[level] * np.clip(sigma / 5.0, 0.25, 1.0))


def nlm_strength(level: str, sigma: float) -> float:
    return float(np.clip(NLM_K[level] * sigma, 1.5, 18.0))


# ---- public API ------------------------------------------------------------


def denoise(
    rgb: np.ndarray,
    level: str = "medium",
    iso: int | None = None,
    backend: str | None = None,
    sigma: float | None = None,
    pause: Callable[[], None] | None = None,
    cancel: Callable[[], bool] | None = None,
    progress: Callable[[float], None] | None = None,
) -> tuple[np.ndarray, str]:
    """Denoised copy of an (H, W, 3) uint8 RGB image and the backend that made it ("ai" | "basic").

    `backend` None picks "ai" when the model is installed and loads, else "basic". `sigma` (8-bit
    noise level) is estimated from the image and ISO when not given - pass the value measured on the
    full image when denoising a crop of it, so a crop is treated like the whole picture."""
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}")
    sig = effective_sigma(rgb, iso) if sigma is None else sigma
    if backend in (None, "ai") and available():
        try:
            with holder().use() as net:
                alpha = ai_alpha(level, sig)

                def run(t: np.ndarray) -> np.ndarray:
                    y = net.tile(t)
                    return t.astype(np.float32) + alpha * (y - t.astype(np.float32))

                return run_tiled(rgb, run, WORKERS if net.provider == "coreml" else 1,
                                 pause, cancel, progress), "ai"  # fmt: skip
        except Cancelled:
            raise
        except Exception as exc:  # noqa: BLE001
            if backend == "ai":
                raise
            log.warning("AI denoise failed (%s); using the basic denoiser", exc)
    h = nlm_strength(level, sig)

    def basic(t: np.ndarray) -> np.ndarray:
        bgr = cv2.cvtColor(t, cv2.COLOR_RGB2BGR)
        y = cv2.fastNlMeansDenoisingColored(bgr, None, h, h * 0.8, 7, 21)
        return cv2.cvtColor(y, cv2.COLOR_BGR2RGB).astype(np.float32)

    return run_tiled(rgb, basic, 2, pause, cancel, progress), "basic"


def timed(fn: Callable[[], object]) -> tuple[object, float]:
    t = time.perf_counter()
    r = fn()
    return r, time.perf_counter() - t
