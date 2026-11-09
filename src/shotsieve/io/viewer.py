"""Viewer images (T23f): what the loupe / compare / grid actually show.

The cull's analysis path (`io.preview.load_preview`) is NOT touched; this module only feeds the UI.

Rules (found by `scripts/viewer_quality_report.py`, see docs/TASKS.md T23f):
  * the picture is rendered at the size the screen needs (long edge in device pixels, bucketed to
    `BUCKETS`) and never upscaled;
  * one high-quality downscale (Lanczos, draft/DCT scaling first for JPEG) straight from the
    source, not a decode at 1/2..1/8 followed by a second resize;
  * colour is managed: an embedded ICC profile (Display P3 HEIC/JPEG, Adobe RGB, ...) is
    converted to sRGB and the file is written WITH an sRGB profile, 4:4:4, quality >= 90;
  * RAW: the camera's embedded JPEG when it is big enough, else a camera-WB demosaic whose tone is
    matched to the embedded JPEG (so the viewer keeps the camera's look at full sensor detail),
    cropped to the same frame as the embedded JPEG (a camera's 16:9 mode).
"""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageCms

from shotsieve.io.formats import RAW_EXTS

BUCKETS = (1280, 2048, 2880, 3840)  # long edge in px a viewer image can have
VIEW_QUALITY = 92  # loupe / compare / full
THUMB_QUALITY = 85  # grid thumbnails
DEFAULT_BUCKET = 2048

_SRGB = ImageCms.createProfile("sRGB")
SRGB_ICC: bytes = ImageCms.ImageCmsProfile(_SRGB).tobytes()


def bucket_for(w: int | None) -> int | None:
    """Smallest bucket >= w (a 3 % shortfall is tolerated); None when w is above the top bucket
    (callers serve the full-resolution image then)."""
    if w is None or w <= 0:
        return DEFAULT_BUCKET
    for b in BUCKETS:
        if w <= b * 1.03:
            return b
    return None


# ---- colour ---------------------------------------------------------------------------------


def _is_srgb(profile: ImageCms.ImageCmsProfile) -> bool:
    try:
        return "srgb" in ImageCms.getProfileDescription(profile).lower()
    except Exception:  # noqa: BLE001
        return False


def to_srgb(im: Image.Image) -> Image.Image:
    """RGB image in sRGB: converted from its embedded ICC profile when it has a non-sRGB one."""
    icc = im.info.get("icc_profile")
    if im.mode != "RGB":
        im = im.convert("RGB")
    if not icc:
        return im
    try:
        src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        if _is_srgb(src):
            return im
        return ImageCms.profileToProfile(im, src, _SRGB, outputMode="RGB")
    except Exception:  # noqa: BLE001 - a broken profile: show the pixels as they are
        return im


def has_foreign_profile(src: Path | bytes) -> bool:
    """True when the file carries an ICC profile that is not sRGB (header read only)."""
    try:
        with Image.open(io.BytesIO(src) if isinstance(src, bytes) else src) as im:
            icc = im.info.get("icc_profile")
        return bool(icc) and not _is_srgb(ImageCms.ImageCmsProfile(io.BytesIO(icc)))
    except Exception:  # noqa: BLE001
        return False


# ---- resize / encode -------------------------------------------------------------------------


def fit_size(w: int, h: int, edge: int) -> tuple[int, int]:
    longest = max(w, h)
    if longest <= edge:
        return w, h
    s = edge / longest
    return max(1, round(w * s)), max(1, round(h * s))


def resize_rgb(rgb: np.ndarray, edge: int) -> np.ndarray:
    """Downscale so the long edge is `edge` (never upscales). Lanczos with proper antialiasing."""
    h, w = rgb.shape[:2]
    nw, nh = fit_size(w, h, edge)
    if (nw, nh) == (w, h):
        return rgb
    im = Image.fromarray(rgb).resize((nw, nh), Image.LANCZOS)
    return np.asarray(im)


def encode_jpeg(rgb: np.ndarray, quality: int = VIEW_QUALITY, icc: bool = True) -> bytes:
    """JPEG with 4:4:4 chroma (kept for quality >= 90) and, by default, an sRGB profile."""
    buf = io.BytesIO()
    Image.fromarray(rgb).save(
        buf, "JPEG", quality=quality, subsampling=0 if quality >= 90 else 2,
        icc_profile=SRGB_ICC if icc else None,
    )  # fmt: skip
    return buf.getvalue()


_STD_LUMA_Q = np.array(
    [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69,
     56, 14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104,
     113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99],
    np.float64,
)  # fmt: skip


def jpeg_quality(data: bytes) -> int | None:
    """libjpeg quality (1-100) estimated from the luminance quantisation table."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            tabs = getattr(im, "quantization", None)
    except Exception:  # noqa: BLE001
        return None
    if not tabs:
        return None
    t = np.array(tabs[0], np.float64)[:64]
    best, err = 0, 1e18
    for q in range(1, 101):
        s = 5000 / q if q < 50 else 200 - 2 * q
        e = float(np.abs(np.clip(np.floor((_STD_LUMA_Q * s + 50) / 100), 1, 255) - t).sum())
        if e < err:
            best, err = q, e
    return best


def jpeg_icc_description(data: bytes) -> str | None:
    try:
        with Image.open(io.BytesIO(data)) as im:
            icc = im.info.get("icc_profile")
        if not icc:
            return None
        return ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(io.BytesIO(icc))).strip()
    except Exception:  # noqa: BLE001
        return None


# ---- decoding --------------------------------------------------------------------------------


def _register_heif() -> None:
    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
    except ImportError:
        pass


def _open_reduced(src: Path | bytes, edge: int) -> Image.Image:
    """Open an image; a JPEG is DCT-scaled so the long edge stays >= `edge` (fast, exact)."""
    _register_heif()
    im = Image.open(io.BytesIO(src) if isinstance(src, bytes) else src)
    if im.format == "JPEG":
        w, h = im.size
        s = edge / max(w, h)
        if s < 1:
            im.draft("RGB", (max(1, round(w * s)), max(1, round(h * s))))
    return im


def _pil_to_view(im: Image.Image, edge: int) -> np.ndarray:
    from shotsieve.io.preview import apply_orientation

    im.load()
    orient = int(im.getexif().get(0x0112, 1) or 1)
    icc = im.info.get("icc_profile")
    arr = apply_orientation(resize_rgb(np.asarray(im.convert("RGB")), edge), orient)
    if icc:
        small = Image.fromarray(arr)
        small.info["icc_profile"] = icc
        arr = np.asarray(to_srgb(small))
    return np.ascontiguousarray(arr)


def decode_full_srgb(src: Path | bytes, orientation: int | None = None) -> np.ndarray:
    """Full-size RGB uint8 in sRGB through Pillow (ICC-aware). `orientation` None: the file's own
    EXIF orientation is applied; an int applies that value (an embedded RAW JPEG)."""
    from shotsieve.io.preview import apply_orientation

    _register_heif()
    with Image.open(io.BytesIO(src) if isinstance(src, bytes) else src) as im:
        im.load()
        orient = int(im.getexif().get(0x0112, 1) or 1) if orientation is None else orientation
        arr = np.asarray(to_srgb(im.convert("RGB") if not im.info.get("icc_profile") else im))
    return np.ascontiguousarray(apply_orientation(arr, orient))


def match_tone(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Per-channel LUT (256, 1, 3) uint8 that maps `src`'s histogram onto `ref`'s (same frame)."""
    lut = np.empty((256, 3), np.float64)
    levels = np.arange(256, dtype=np.float64)
    for c in range(3):
        a = np.bincount(src[..., c].ravel(), minlength=256).cumsum().astype(np.float64)
        b = np.bincount(ref[..., c].ravel(), minlength=256).cumsum().astype(np.float64)
        a /= a[-1]
        b /= b[-1]
        m = np.maximum.accumulate(np.interp(a, b, levels))
        for _ in range(3):  # light smoothing keeps the curve free of banding steps
            p = np.pad(m, 2, mode="edge")
            m = (p[:-4] + 2 * p[1:-3] + 3 * p[2:-2] + 2 * p[3:-1] + p[4:]) / 9
        lut[:, c] = np.clip(np.maximum.accumulate(m), 0, 255)
    return np.round(lut).astype(np.uint8).reshape(256, 1, 3)


def crop_dims(w: int, h: int, aspect: float) -> tuple[int, int]:
    """Size of the centred w/h = aspect crop of a w x h frame (unchanged within 1 %)."""
    if abs(w / h - aspect) <= 0.01 * aspect:
        return w, h
    if w / h > aspect:
        return round(h * aspect), h
    return w, round(w / aspect)


def center_crop_to_aspect(img: np.ndarray, aspect: float) -> np.ndarray:
    """Centre crop to w/h = aspect (a camera's 16:9 JPEG of a 3:2 sensor); no-op within 1 %."""
    h, w = img.shape[:2]
    nw, nh = crop_dims(w, h, aspect)
    if (nw, nh) == (w, h):
        return img
    x0, y0 = (w - nw) // 2, (h - nh) // 2
    return np.ascontiguousarray(img[y0 : y0 + nh, x0 : x0 + nw])


def finish_demosaic(
    rgb: np.ndarray, thumb_bytes: bytes | None, orient: int, edge: int | None = None
) -> np.ndarray:
    """A camera-WB demosaic in the frame and tone of the camera's embedded JPEG: centre-cropped to
    its aspect and passed through a per-channel curve fitted to it (the camera's look at full
    sensor detail). Without an embedded JPEG: unchanged. `edge` also downsizes (Lanczos)."""
    from shotsieve.io.preview import apply_orientation

    if thumb_bytes is not None:
        im = _open_reduced(thumb_bytes, 512)
        icc = im.info.get("icc_profile")
        im.load()
        ref = apply_orientation(np.asarray(to_srgb(im) if icc else im.convert("RGB")), orient)
        rgb = center_crop_to_aspect(rgb, ref.shape[1] / ref.shape[0])
        if edge is not None:
            rgb = resize_rgb(rgb, edge)
        small = resize_rgb(rgb, 320)
        ref = cv2.resize(ref, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA)
        rgb = cv2.LUT(rgb, match_tone(small, ref))
    elif edge is not None:
        rgb = resize_rgb(rgb, edge)
    return np.ascontiguousarray(rgb)


def _raw_view(path: Path, edge: int) -> np.ndarray:
    import rawpy

    from shotsieve.io.preview import _RAW_FLIP_TO_EXIF, _thumb_orientation, apply_orientation

    with rawpy.imread(str(path)) as raw:
        thumb_bytes: bytes | None = None
        tw = th = 0
        try:
            t = raw.extract_thumb()
            if t.format == rawpy.ThumbFormat.JPEG:
                thumb_bytes = bytes(t.data)
        except (rawpy.LibRawNoThumbnailError, rawpy.LibRawUnsupportedThumbnailError):
            thumb_bytes = None
        orient = _RAW_FLIP_TO_EXIF.get(raw.sizes.flip, 1)
        if thumb_bytes is not None:
            try:
                with Image.open(io.BytesIO(thumb_bytes)) as im:
                    tw, th = im.size
                orient = _thumb_orientation(thumb_bytes) or orient
            except Exception:  # noqa: BLE001
                thumb_bytes = None
        sensor_long = max(raw.sizes.width, raw.sizes.height)
        if thumb_bytes is not None and max(tw, th) >= min(edge, sensor_long):
            im = _open_reduced(thumb_bytes, edge)
            icc = im.info.get("icc_profile")
            im.load()
            arr = apply_orientation(np.asarray(im.convert("RGB")), orient)
            arr = resize_rgb(arr, edge)
            if icc:
                small = Image.fromarray(arr)
                small.info["icc_profile"] = icc
                arr = np.asarray(to_srgb(small))
            return np.ascontiguousarray(arr)
        # demosaic (camera WB, sRGB output): half size is a true 2x2 binning, no demosaic artefacts
        half = max(raw.sizes.width, raw.sizes.height) // 2 >= edge
        rgb = raw.postprocess(half_size=half, use_camera_wb=True, output_bps=8)
        return finish_demosaic(rgb, thumb_bytes, orient, edge)


def load_view(path: str | Path, edge: int) -> np.ndarray:
    """(H, W, 3) uint8 sRGB, EXIF orientation applied, long edge <= `edge` (never upscaled)."""
    from shotsieve.io.preview import PreviewError

    path = Path(path)
    try:
        if path.suffix.lower() in RAW_EXTS:
            return _raw_view(path, edge)
        with _open_reduced(path, edge) as im:
            return _pil_to_view(im, edge)
    except PreviewError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise PreviewError(f"cannot decode {path}: {exc}") from exc
