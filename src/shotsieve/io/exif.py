"""EXIF metadata extraction. Never raises on bad input; returns defaults."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from shotsieve.io.formats import RAW_EXTS

_EXIFREAD_EXTS = RAW_EXTS | {".jpg", ".jpeg", ".tif", ".tiff"}
_SERIAL_KEYS = (
    "EXIF BodySerialNumber",
    "EXIF SerialNumber",
    "MakerNote SerialNumber",
    "MakerNote InternalSerialNumber",
)


@dataclass
class ImageMeta:
    timestamp: float
    ts_source: str = "mtime"  # "exif" | "mtime"
    camera_serial: str = "unknown"
    make: str = ""
    model: str = ""
    flash_fired: bool | None = None
    iso: int | None = None
    exposure_time: float | None = None
    fnumber: float | None = None
    focal_length: float | None = None
    orientation: int = 1
    width: int = 0
    height: int = 0
    burst_seq: int | None = None
    af_points: list[list[float]] = field(default_factory=list)  # [cx, cy, w, h] sensor orient.
    tz_offset: str = ""  # EXIF OffsetTimeOriginal ("+02:00"), when present


def _ratio(value: Any) -> float | None:
    try:
        num, den = getattr(value, "num", None), getattr(value, "den", None)
        if num is not None and den:
            return float(num) / float(den)
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _first(tag: Any) -> Any:
    vals = getattr(tag, "values", tag)
    if isinstance(vals, list | tuple):
        return vals[0] if vals else None
    return vals


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_timestamp(dt: str | None, subsec: str | None) -> float | None:
    if not dt:
        return None
    try:
        ts = datetime.strptime(dt.strip()[:19], "%Y:%m:%d %H:%M:%S").timestamp()
    except ValueError:
        return None
    sub = (subsec or "").strip().strip("\x00")
    if sub.isdigit():
        ts += float("0." + sub)
    return ts


def _from_exifread(path: Path, meta: ImageMeta) -> bool:
    import exifread

    with open(path, "rb") as fh:
        tags = exifread.process_file(fh, details=False)
    if not tags:
        return False
    s = lambda k: str(tags[k]).strip() if k in tags else None  # noqa: E731
    ts = _parse_timestamp(
        s("EXIF DateTimeOriginal") or s("EXIF DateTimeDigitized") or s("Image DateTime"),
        s("EXIF SubSecTimeOriginal") or s("EXIF SubSecTimeDigitized") or s("EXIF SubSecTime"),
    )
    if ts is not None:
        meta.timestamp, meta.ts_source = ts, "exif"
    meta.tz_offset = s("EXIF OffsetTimeOriginal") or ""
    meta.make, meta.model = s("Image Make") or "", s("Image Model") or ""
    for key in _SERIAL_KEYS:
        if s(key) and s(key).strip("0 "):
            meta.camera_serial = s(key)
            break
    else:
        if meta.make or meta.model:
            meta.camera_serial = f"{meta.make}-{meta.model}".strip("-")
    if "EXIF Flash" in tags:
        flash = _to_int(_first(tags["EXIF Flash"]))
        meta.flash_fired = None if flash is None else bool(flash & 1)
    if "EXIF ISOSpeedRatings" in tags:
        meta.iso = _to_int(_first(tags["EXIF ISOSpeedRatings"]))
    if "EXIF ExposureTime" in tags:
        meta.exposure_time = _ratio(_first(tags["EXIF ExposureTime"]))
    if "EXIF FNumber" in tags:
        meta.fnumber = _ratio(_first(tags["EXIF FNumber"]))
    if "EXIF FocalLength" in tags:
        meta.focal_length = _ratio(_first(tags["EXIF FocalLength"]))
    if "Image Orientation" in tags:
        meta.orientation = _to_int(_first(tags["Image Orientation"])) or 1
    meta.width = (
        _to_int(_first(tags["EXIF ExifImageWidth"])) or 0 if "EXIF ExifImageWidth" in tags else 0
    )
    meta.height = (
        _to_int(_first(tags["EXIF ExifImageLength"])) or 0 if "EXIF ExifImageLength" in tags else 0
    )
    return True


def _from_pillow(path: Path, meta: ImageMeta) -> bool:
    from PIL import Image

    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
    except ImportError:
        pass
    with Image.open(path) as im:
        meta.width, meta.height = im.size
        exif = im.getexif()
        if not exif:
            return False
        ifd = exif.get_ifd(0x8769)
        meta.make, meta.model = str(exif.get(0x010F, "")).strip(), str(exif.get(0x0110, "")).strip()
        meta.orientation = _to_int(exif.get(0x0112)) or 1
        ts = _parse_timestamp(
            ifd.get(0x9003) or exif.get(0x0132), str(ifd.get(0x9291)) if 0x9291 in ifd else None
        )
        if ts is not None:
            meta.timestamp, meta.ts_source = ts, "exif"
        meta.tz_offset = str(ifd.get(0x9011) or "").strip("\x00 ")
        serial = ifd.get(0xA431)
        meta.camera_serial = (
            str(serial) if serial else f"{meta.make}-{meta.model}".strip("-") or "unknown"
        )
        if 0x9209 in ifd:
            flash = _to_int(ifd[0x9209])
            meta.flash_fired = None if flash is None else bool(flash & 1)
        meta.iso = _to_int(ifd.get(0x8827))
        meta.exposure_time = _ratio(ifd.get(0x829A))
        meta.fnumber = _ratio(ifd.get(0x829D))
        meta.focal_length = _ratio(ifd.get(0x920A))
    return True


def _from_exiftool(path: Path, meta: ImageMeta) -> bool:
    exe = shutil.which("exiftool")
    if not exe:
        return False
    out = subprocess.run(
        [exe, "-j", "-n", "-api", "largefilesupport=1", str(path)],
        capture_output=True,
        timeout=20,
        check=True,
    ).stdout
    d = json.loads(out)[0]
    ts = _parse_timestamp(d.get("DateTimeOriginal"), str(d.get("SubSecTimeOriginal", "")))
    if ts is not None:
        meta.timestamp, meta.ts_source = ts, "exif"
    meta.tz_offset = str(d.get("OffsetTimeOriginal", "") or "")
    meta.make, meta.model = str(d.get("Make", "")), str(d.get("Model", ""))
    serial = d.get("SerialNumber") or d.get("InternalSerialNumber") or d.get("BodySerialNumber")
    meta.camera_serial = (
        str(serial) if serial else f"{meta.make}-{meta.model}".strip("-") or "unknown"
    )
    if "Flash" in d:
        meta.flash_fired = bool(int(d["Flash"]) & 1)
    meta.iso = _to_int(d.get("ISO"))
    meta.exposure_time = _ratio(d.get("ExposureTime"))
    meta.fnumber = _ratio(d.get("FNumber"))
    meta.focal_length = _ratio(d.get("FocalLength"))
    meta.orientation = _to_int(d.get("Orientation")) or 1
    meta.width = _to_int(d.get("ImageWidth")) or 0
    meta.height = _to_int(d.get("ImageHeight")) or 0
    return True


def read_meta(path: str | Path) -> ImageMeta:
    """Read EXIF metadata; any failure yields defaults with the file mtime as timestamp."""
    path = Path(path)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    meta = ImageMeta(timestamp=mtime)
    ext = path.suffix.lower()
    readers = []
    if ext in _EXIFREAD_EXTS:
        readers.append(_from_exifread)
    if ext not in RAW_EXTS or ext == ".cr3":
        readers.append(_from_pillow)
    if ext in RAW_EXTS:
        readers.append(_from_exiftool)
    for reader in readers:
        try:
            if reader(path, meta):
                break
        except Exception:  # noqa: BLE001 - metadata is best-effort by contract
            continue
    if not meta.width and ext not in RAW_EXTS:
        try:
            from PIL import Image

            with Image.open(path) as im:
                meta.width, meta.height = im.size
        except Exception:  # noqa: BLE001
            pass
    return meta


_LENS_VER = 1  # bump to re-read the lens fields (the facets back-fill; never a cull input)


def _lens_clean(v: Any) -> str:
    s = str(v or "").strip().strip("\x00").strip()
    return "" if not s or not s.strip("0 ") or s.lower() in ("n/a", "unknown", "none") else s


def read_lens(path: str | Path) -> tuple[str, str]:
    """(lens model, lens serial number) from the EXIF, or "" for what is not there. Reads only
    the Exif IFD (exifread without MakerNotes, Pillow for HEIC), so it is cheap and never raises;
    exiftool is a last resort for a RAW whose lens is only in the MakerNote."""
    path = Path(path)
    ext = path.suffix.lower()
    model = serial = ""
    if ext in _EXIFREAD_EXTS:
        try:
            import exifread

            with open(path, "rb") as fh:
                tags = exifread.process_file(fh, details=False)
            model = _lens_clean(tags.get("EXIF LensModel"))
            serial = _lens_clean(tags.get("EXIF LensSerialNumber"))
        except Exception:  # noqa: BLE001
            pass
    if not model and ext not in RAW_EXTS:
        try:
            from PIL import Image

            try:
                import pillow_heif

                pillow_heif.register_heif_opener()
            except ImportError:
                pass
            with Image.open(path) as im:
                ifd = im.getexif().get_ifd(0x8769)
            model = _lens_clean(ifd.get(0xA434))
            serial = serial or _lens_clean(ifd.get(0xA435))
        except Exception:  # noqa: BLE001
            pass
    if not model and ext in RAW_EXTS and (exe := shutil.which("exiftool")):
        try:
            out = subprocess.run(
                [exe, "-j", "-fast", "-LensModel", "-LensID", "-LensSerialNumber", str(path)],
                capture_output=True, timeout=20, check=True,
            ).stdout  # fmt: skip
            d = json.loads(out)[0]
            model = _lens_clean(d.get("LensModel") or d.get("LensID"))
            serial = serial or _lens_clean(d.get("LensSerialNumber"))
        except Exception:  # noqa: BLE001
            pass
    return model, serial
