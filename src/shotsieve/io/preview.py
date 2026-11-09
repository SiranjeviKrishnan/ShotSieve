"""Fast preview loading: embedded JPEG for RAW, reduced decode for big rasters."""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np

from shotsieve.io.formats import RAW_EXTS

MIN_THUMB_EDGE = 640


class PreviewError(RuntimeError):
    """The file could not be decoded."""


def _downscale(img: np.ndarray, max_edge: int) -> np.ndarray:
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_edge:
        return img
    s = max_edge / longest
    return cv2.resize(
        img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA
    )


def apply_orientation(img: np.ndarray, orientation: int) -> np.ndarray:
    """Apply an EXIF orientation value (1-8) to an array."""
    ops = {
        2: lambda a: a[:, ::-1],
        3: lambda a: a[::-1, ::-1],
        4: lambda a: a[::-1, :],
        5: lambda a: np.rot90(a, 1)[::-1, :],
        6: lambda a: np.rot90(a, -1),
        7: lambda a: np.rot90(a, -1)[::-1, :],
        8: lambda a: np.rot90(a, 1),
    }
    op = ops.get(orientation)
    return np.ascontiguousarray(op(img)) if op else img


def _load_raster(path: Path, max_edge: int) -> np.ndarray:
    ext = path.suffix.lower()
    if ext in {".heic", ".heif"}:
        return _load_pillow(path)
    flags = cv2.IMREAD_COLOR
    try:
        from PIL import Image

        with Image.open(path) as im:
            longest = max(im.size)
    except Exception:  # noqa: BLE001
        longest = 0
    if ext in {".jpg", ".jpeg"} and longest >= 4 * max_edge:
        flags = cv2.IMREAD_REDUCED_COLOR_4
    elif ext in {".jpg", ".jpeg"} and longest >= 2 * max_edge:
        flags = cv2.IMREAD_REDUCED_COLOR_2
    bgr = cv2.imread(str(path), flags)  # applies EXIF orientation for JPEG
    if bgr is None:
        return _load_pillow(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _load_pillow(path: Path) -> np.ndarray:
    from PIL import Image, ImageOps

    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
    except ImportError:
        pass
    with Image.open(path) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert("RGB"))


def decode_embedded_jpeg(data: bytes, max_edge: int) -> np.ndarray | None:
    """Decode an embedded JPEG (BGR, orientation ignored), at 1/2, 1/4 or 1/8 scale when the
    result would still be at least `max_edge` px - a 45 MP preview then costs a fraction of the
    memory and time of a full decode."""
    flags = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as im:
            longest = max(im.size)
    except Exception:  # noqa: BLE001
        longest = 0
    for div, flag in ((8, cv2.IMREAD_REDUCED_COLOR_8), (4, cv2.IMREAD_REDUCED_COLOR_4),
                      (2, cv2.IMREAD_REDUCED_COLOR_2)):  # fmt: skip
        if longest // div >= max_edge:
            flags |= flag
            break
    return cv2.imdecode(np.frombuffer(data, np.uint8), flags)


def _thumb_orientation(data: bytes) -> int | None:
    try:
        import exifread

        tags = exifread.process_file(io.BytesIO(data), details=False, stop_tag="Orientation")
        if "Image Orientation" in tags:
            return int(tags["Image Orientation"].values[0])
    except Exception:  # noqa: BLE001
        pass
    return None


_RAW_FLIP_TO_EXIF = {0: 1, 3: 3, 5: 8, 6: 6}


def _load_raw(path: Path, max_edge: int) -> np.ndarray:
    import rawpy

    with rawpy.imread(str(path)) as raw:
        img: np.ndarray | None = None
        try:
            thumb = raw.extract_thumb()
            if thumb.format == rawpy.ThumbFormat.JPEG:
                data = bytes(thumb.data)
                arr = decode_embedded_jpeg(data, max_edge)
                if arr is not None:
                    img = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                    orient = _thumb_orientation(data) or _RAW_FLIP_TO_EXIF.get(raw.sizes.flip, 1)
                    img = apply_orientation(img, orient)
            elif thumb.format == rawpy.ThumbFormat.BITMAP:
                img = apply_orientation(
                    np.asarray(thumb.data), _RAW_FLIP_TO_EXIF.get(raw.sizes.flip, 1)
                )
        except (rawpy.LibRawNoThumbnailError, rawpy.LibRawUnsupportedThumbnailError):
            img = None
        if img is None or max(img.shape[:2]) < min(MIN_THUMB_EDGE, max_edge):
            img = raw.postprocess(half_size=True, use_camera_wb=True, output_bps=8)
        return img


def load_preview(path: str | Path, max_edge: int = 1280) -> np.ndarray:
    """Return an (H, W, 3) uint8 RGB preview with orientation applied and long edge <= max_edge."""
    path = Path(path)
    try:
        if path.suffix.lower() in RAW_EXTS:
            img = _load_raw(path, max_edge)
        else:
            img = _load_raster(path, max_edge)
    except PreviewError:
        raise
    except Exception as exc:
        raise PreviewError(f"cannot decode {path}: {exc}") from exc
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return _downscale(np.ascontiguousarray(img[:, :, :3]), max_edge)


# ---- full resolution (zoom, T23c) -------------------------------------------------------------

FULL_EMBEDDED_MIN = 0.9  # an embedded RAW JPEG this close to the sensor size is the full view


def _swap(orient: int, w: int, h: int) -> tuple[int, int]:
    return (h, w) if orient in (5, 6, 7, 8) else (w, h)


def _raw_full_plan(raw: object) -> tuple[str, bytes | None, int, int, int]:
    """How the full view of an open RAW is made: ("thumb", jpeg bytes, w, h, orient) for an
    embedded JPEG >= 90 % of the sensor's long edge, else ("demosaic", the embedded JPEG or None,
    w, h, orient). w x h is the displayed size (orientation applied; a demosaic is centre-cropped to
    the embedded JPEG's aspect, e.g. a camera's 16:9 mode, so zoom shows the same frame as the
    preview and the face / crop overlays)."""
    import rawpy

    from shotsieve.io.viewer import crop_dims

    sizes = raw.sizes  # type: ignore[attr-defined]
    flip_orient = _RAW_FLIP_TO_EXIF.get(sizes.flip, 1)
    sensor_long = max(sizes.width, sizes.height)
    data: bytes | None = None
    orient, tw, th = flip_orient, 0, 0
    try:
        thumb = raw.extract_thumb()  # type: ignore[attr-defined]
        if thumb.format == rawpy.ThumbFormat.JPEG:
            data = bytes(thumb.data)
            from PIL import Image

            with Image.open(io.BytesIO(data)) as im:
                tw, th = im.size
            orient = _thumb_orientation(data) or flip_orient
            if max(tw, th) >= FULL_EMBEDDED_MIN * sensor_long:
                w, h = _swap(orient, tw, th)
                return "thumb", data, w, h, orient
    except (rawpy.LibRawNoThumbnailError, rawpy.LibRawUnsupportedThumbnailError):
        data = None
    except Exception:  # noqa: BLE001  (unreadable thumbnail: demosaic instead)
        data = None
    w, h = _swap(flip_orient, sizes.width, sizes.height)
    if data is not None and tw and th:
        sw, sh = _swap(orient, tw, th)
        w, h = crop_dims(w, h, sw / sh)
    return "demosaic", data, w, h, orient


def full_size(path: str | Path) -> tuple[int, int]:
    """Displayed (orientation applied) pixel size `load_full` will produce, without decoding."""
    path = Path(path)
    try:
        if path.suffix.lower() in RAW_EXTS:
            import rawpy

            with rawpy.imread(str(path)) as raw:
                _kind, _data, w, h, _o = _raw_full_plan(raw)
                return w, h
        from PIL import Image

        try:
            import pillow_heif

            pillow_heif.register_heif_opener()
        except ImportError:
            pass
        with Image.open(path) as im:
            w, h = im.size
            orient = int(im.getexif().get(0x0112, 1) or 1)
        return _swap(orient, w, h)
    except Exception as exc:
        raise PreviewError(f"cannot read {path}: {exc}") from exc


def load_full(path: str | Path) -> np.ndarray:
    """(H, W, 3) uint8 RGB at native size, orientation applied (same display frame as
    `load_preview`). RAW: the embedded JPEG when >= 90 % of the sensor, else a full demosaic."""
    from shotsieve.io.viewer import decode_full_srgb, has_foreign_profile

    path = Path(path)
    try:
        if path.suffix.lower() in RAW_EXTS:
            import rawpy

            with rawpy.imread(str(path)) as raw:
                kind, data, _w, _h, orient = _raw_full_plan(raw)
                if kind == "thumb" and data is not None:
                    if has_foreign_profile(data):
                        img = decode_full_srgb(data, orient)
                    else:
                        arr = cv2.imdecode(
                            np.frombuffer(data, np.uint8),
                            cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
                        )
                        if arr is None:
                            raise PreviewError("embedded JPEG not decodable")
                        img = apply_orientation(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB), orient)
                    del data
                else:
                    from shotsieve.io.viewer import finish_demosaic

                    img = finish_demosaic(
                        raw.postprocess(use_camera_wb=True, output_bps=8), data, orient
                    )
        elif path.suffix.lower() in {".heic", ".heif"} or has_foreign_profile(path):
            img = decode_full_srgb(path)  # colour-managed: Display P3 / Adobe RGB -> sRGB
        else:
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)  # EXIF orientation applied by OpenCV
            img = _load_pillow(path) if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except PreviewError:
        raise
    except Exception as exc:
        raise PreviewError(f"cannot decode {path}: {exc}") from exc
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return np.ascontiguousarray(img[:, :, :3])
