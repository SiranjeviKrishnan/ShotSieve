"""Supported file formats and folder scanning."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

RAW_EXTS = frozenset(
    ".cr2 .cr3 .nef .nrw .arw .srf .sr2 .raf .orf .rw2 .pef .dng .srw .3fr .iiq .erf".split()
)
RASTER_EXTS = frozenset(".jpg .jpeg .png .tif .tiff .heic .heif .webp".split())
SUPPORTED_EXTS = RAW_EXTS | RASTER_EXTS
VIDEO_EXTS = frozenset(".mp4 .mov .m4v".split())  # opt-in frame-grab culling (T12); not images
VIRTUAL_SEP = "#t="  # `clip.mp4#t=12.345`: a candidate frame of a video (decisions/UI only)
JPEG_EXTS = frozenset({".jpg", ".jpeg"})


@dataclass(frozen=True)
class ImageEntry:
    """One logical image: the master file and the file used for previews."""

    master: Path
    preview_source: Path
    raw_jpeg_pair: bool = False


def is_raw(path: str | Path) -> bool:
    return Path(path).suffix.lower() in RAW_EXTS


def is_raster(path: str | Path) -> bool:
    return Path(path).suffix.lower() in RASTER_EXTS


def is_supported(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_EXTS


# ShotSieve's own output folders: never culled again (embedded-XMP copies are pixel-identical)
EXCLUDED_DIRS = frozenset({"shotsieve_edits"})


def scan_folder(folder: str | Path, recursive: bool = True) -> list[ImageEntry]:
    """List logical images under `folder`, sorted by path.

    A RAW+JPEG pair with the same stem in the same directory is one image: the RAW is the
    master (XMP target) and the JPEG is the preview source. Hidden files/dirs are skipped.
    """
    root = Path(folder)
    files: list[Path] = []
    if recursive:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(
                d for d in dirnames if not d.startswith(".") and d not in EXCLUDED_DIRS
            )
            files.extend(
                Path(dirpath, f) for f in filenames if not f.startswith(".") and is_supported(f)
            )
    else:
        files = [p for p in root.iterdir() if p.is_file() and not p.name.startswith(".")]
        files = [p for p in files if is_supported(p)]

    by_key: dict[tuple[Path, str], list[Path]] = {}
    for p in files:
        by_key.setdefault((p.parent, p.stem.lower()), []).append(p)

    entries: list[ImageEntry] = []
    for group in by_key.values():
        raws = sorted(p for p in group if is_raw(p))
        jpegs = sorted(p for p in group if p.suffix.lower() in JPEG_EXTS)
        if raws and jpegs:
            entries.append(ImageEntry(raws[0], jpegs[0], True))
            others = [p for p in group if p not in raws[:1] and p not in jpegs[:1]]
        else:
            others = group
        entries.extend(ImageEntry(p, p) for p in others if not (raws and jpegs and p in raws[:1]))
    entries.sort(key=lambda e: str(e.master))
    return entries


def preview_source_for(master: str | Path) -> Path:
    """The file previews are decoded from: a same-stem JPEG next to a RAW master, else itself."""
    p = Path(master)
    if not is_raw(p):
        return p
    try:
        for sib in sorted(p.parent.iterdir()):
            if sib.stem.lower() == p.stem.lower() and sib.suffix.lower() in JPEG_EXTS:
                return sib
    except OSError:
        pass
    return p


def is_video(path: str | Path) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTS


def virtual_path(video: str | Path, t: float) -> str:
    return f"{video}{VIRTUAL_SEP}{t:.3f}"


def split_virtual(path: str | Path) -> tuple[Path, float] | None:
    """(video file, seconds) of a virtual `clip.mp4#t=12.345` path, else None."""
    head, sep, tail = str(path).rpartition(VIRTUAL_SEP)
    if not sep:
        return None
    try:
        return Path(head), float(tail)
    except ValueError:
        return None


def scan_videos(folder: str | Path) -> list[Path]:
    """Video files under `folder` (hidden files/dirs skipped), sorted."""
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(folder):
        dirnames[:] = sorted(
            d for d in dirnames if not d.startswith(".") and d not in EXCLUDED_DIRS
        )
        out.extend(
            Path(dirpath, f) for f in filenames if not f.startswith(".") and is_video(f)
        )
    return sorted(out)
