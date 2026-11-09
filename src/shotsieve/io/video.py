"""Video decoding for frame-grab culling (T12): the backend is chosen per codec.

HEVC (hvc1/hev1/hevc) on macOS goes through cv2's AVFoundation backend first: it decodes with
VideoToolbox (4K30 HEVC ~6x realtime against ~1.5-2.4x for the wheel's software FFmpeg, which
ignores CAP_PROP_HW_ACCELERATION). Everything else (H.264, ...) keeps FFmpeg first: it skips the
BGR conversion of frames in between and is faster there (~7-14x against ~7x). Each is the
fallback of the other when a file cannot be opened or decoded. `SHOTSIEVE_VIDEO_BACKEND`
(ffmpeg | avfoundation) forces one.

Decoded frames are downscaled to `max_edge` immediately and at most two full-size frames are
alive at once. Video files are never modified.

Codecs/licensing: decoding goes through the platform decoder (AVFoundation / VideoToolbox) or the
FFmpeg bundled in the `opencv-contrib-python` wheel; ShotSieve ships and re-encodes no codec.
H.264/HEVC decoding on macOS is covered by Apple's system licence. The wheel's FFmpeg is a GPL-3.0
build (see THIRD_PARTY_NOTICES.md).
Exported frame grabs are plain JPEGs.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


class VideoError(Exception):
    pass


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    n_frames: int
    duration: float
    width: int
    height: int


HEVC_CODECS = frozenset({"hevc", "hvc1", "hev1", "h265", "hevc "})


def codec_of(path: str | Path) -> str:
    """FOURCC of the first video stream as reported by the FFmpeg backend ('' if unknown)."""
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    try:
        if not cap.isOpened():
            return ""
        code = int(cap.get(cv2.CAP_PROP_FOURCC))
        return code.to_bytes(4, "little").decode("latin-1").strip("\x00").lower() if code else ""
    finally:
        cap.release()


def _backends(path: str | Path | None = None) -> list[int]:
    order = [cv2.CAP_FFMPEG, cv2.CAP_AVFOUNDATION]
    if sys.platform == "darwin" and path is not None and codec_of(path) in HEVC_CODECS:
        order = [cv2.CAP_AVFOUNDATION, cv2.CAP_FFMPEG]
    forced = os.environ.get("SHOTSIEVE_VIDEO_BACKEND", "").lower()
    if forced == "ffmpeg":
        order = [cv2.CAP_FFMPEG]
    elif forced == "avfoundation":
        order = [cv2.CAP_AVFOUNDATION]
    return order


def _shrink(frame_bgr: np.ndarray, max_edge: int) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    s = max_edge / max(h, w)
    if s < 1.0:
        frame_bgr = cv2.resize(
            frame_bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA
        )
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


class VideoReader:
    """Sequential (`frames`) and random-access (`frame_at`) reads of one video."""

    def __init__(self, path: str | Path, max_edge: int = 1280) -> None:
        self.path, self.max_edge = Path(path), max_edge
        self._cap: cv2.VideoCapture | None = None
        self.backend = ""
        for be in _backends(self.path):
            cap = cv2.VideoCapture(str(self.path), be)
            if cap.isOpened() and cap.grab():  # opens AND decodes a frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._cap = cap
                self.backend = "avfoundation" if be == cv2.CAP_AVFOUNDATION else "ffmpeg"
                break
            cap.release()
        if self._cap is None:
            raise VideoError(f"cannot open video: {self.path.name}")
        fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or 0.0
        n = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if not 1.0 <= fps <= 480.0:
            fps = 30.0
        self.info = VideoInfo(
            fps, n, n / fps if n > 0 else 0.0,
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )  # fmt: skip

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def samples(self, step_s: float):
        """Yield (t_seconds, rgb) about every `step_s`, decoding sequentially (grab() skips the
        colour conversion of the frames in between). Memory: one frame at a time."""
        assert self._cap is not None
        cap = self._cap
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        step = max(1, round(step_s * self.info.fps))
        i = 0
        while cap.grab():
            if i % step == 0:
                ok, frame = cap.retrieve()
                if ok:
                    yield i / self.info.fps, _shrink(frame, self.max_edge)
                    del frame
            i += 1

    def frame_at(self, t: float, max_edge: int | None = None) -> np.ndarray | None:
        """RGB frame nearest `t` seconds (None when unreadable). `max_edge=0` keeps full size."""
        assert self._cap is not None
        cap = self._cap
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, round(t * self.info.fps)))
        ok, frame = cap.read()
        if not ok:
            return None
        edge = self.max_edge if max_edge is None else max_edge
        return _shrink(frame, edge) if edge else cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


_MAC_EPOCH_OFFSET = 2082844800  # seconds between 1904-01-01 (QuickTime) and 1970-01-01


def _walk_boxes(f, start: int, end: int, want: bytes):
    """Yield (payload_start, payload_end) of boxes named `want` in [start, end)."""
    pos = start
    while pos + 8 <= end:
        f.seek(pos)
        hdr = f.read(8)
        if len(hdr) < 8:
            return
        size, name = int.from_bytes(hdr[:4], "big"), hdr[4:8]
        head = 8
        if size == 1:
            size, head = int.from_bytes(f.read(8), "big"), 16
        elif size == 0:
            size = end - pos
        if size < head:
            return
        if name == want:
            yield pos + head, min(pos + size, end)
        pos += size


def mvhd_creation_time(path: str | Path) -> float | None:
    """`mvhd` creation_time of an MP4/MOV (QuickTime epoch 1904) as epoch seconds; None when
    absent, unreadable or zero (cameras without a clock write 0)."""
    try:
        size = Path(path).stat().st_size
        with open(path, "rb") as f:
            for ms, me in _walk_boxes(f, 0, size, b"moov"):
                for hs, _ in _walk_boxes(f, ms, me, b"mvhd"):
                    f.seek(hs)
                    version = f.read(1)[0]
                    f.read(3)
                    raw = int.from_bytes(f.read(8 if version == 1 else 4), "big")
                    if raw <= 0:
                        return None
                    ts = raw - _MAC_EPOCH_OFFSET
                    return float(ts) if ts > 0 else None
    except (OSError, IndexError, ValueError):
        return None
    return None


def creation_time(path: str | Path) -> float:
    """Recording start (epoch s): the container's `mvhd` creation time (the camera's clock; the
    file's birth time is only the copy time). Falls back to birth time on macOS, else mtime."""
    mv = mvhd_creation_time(path)
    if mv is not None:
        return mv
    st = Path(path).stat()
    return float(getattr(st, "st_birthtime", 0.0) or st.st_mtime)
