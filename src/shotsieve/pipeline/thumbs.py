"""Grid thumbnails written at cull time from the already decoded preview (decode once, T21).

The file name is exactly what the server's `_image_file` looks for, so a culled shoot serves its
grid without decoding anything again: `thumbs/t_{image_id}_400_{mtime_ns}.jpg`.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import cv2
import numpy as np

GRID_EDGE = 400
GRID_QUALITY = 80


def thumb_path(
    folder: str | Path, image_id: int, mtime_ns: int, edge: int = GRID_EDGE, kind: str = "t"
) -> Path:
    return Path(folder) / ".shotsieve" / "thumbs" / f"{kind}_{image_id}_{edge}_{mtime_ns}.jpg"


def write_thumb(
    rgb: np.ndarray, dest: Path, edge: int = GRID_EDGE, quality: int = GRID_QUALITY
) -> bool:
    """Resize `rgb` (long edge `edge`, never upscaled) and write it atomically; False on failure."""
    if dest.exists():
        return True
    h, w = rgb.shape[:2]
    s = min(1.0, edge / max(h, w))
    if s < 1.0:
        rgb = cv2.resize(
            rgb, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA
        )
    ok, buf = cv2.imencode(
        ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    if not ok:
        return False
    tmp = dest.with_name(f".{uuid.uuid4().hex}.tmp")
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(buf.tobytes())
        os.replace(tmp, dest)
    except OSError:
        tmp.unlink(missing_ok=True)
        return False
    return True
