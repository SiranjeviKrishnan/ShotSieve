"""Synthetic videos for the T12 tests (cv2.VideoWriter, mp4v), generated at test time."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

FPS = 30
SCENE_S = 8.0
# spans inside every scene (seconds from its start): sharp+open, blurred, closed eyes, sharp+open
SPANS = (("open", 0.0, 2.5), ("blur", 2.5, 4.0), ("closed", 4.0, 5.5), ("open", 5.5, 8.0))
SIZE = 640
MARK = 24  # the "eyes closed" marker patch (the fake landmarker reads it)


def _scene_images(src_dir: Path) -> list[np.ndarray]:
    a = cv2.imread(str(src_dir / "portrait2.jpg"))
    b = cv2.imread(str(src_dir / "lena.jpg"))
    imgs = [cv2.resize(x, (SIZE, SIZE), interpolation=cv2.INTER_AREA) for x in (a, b)]
    imgs.append(imgs[0][:, :, ::-1].copy())  # channel-swapped: a clearly different colour scene
    return imgs


def make_three_scene_video(path: Path, src_dir: Path) -> list[dict]:
    """3 scenes x 8 s @ 30 fps. Returns [{start, kind, t0, t1}] spans in video seconds."""
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (SIZE, SIZE))
    spans = []
    for si, img in enumerate(_scene_images(src_dir)):
        base = si * SCENE_S
        for kind, t0, t1 in SPANS:
            spans.append({"kind": kind, "t0": base + t0, "t1": base + t1})
        for i in range(int(SCENE_S * FPS)):
            t = i / FPS
            kind = next(k for k, a, b in SPANS if a <= t < b)
            frame = img.copy()
            if kind == "blur":
                frame = cv2.GaussianBlur(frame, (0, 0), 7)
            elif kind == "closed":
                frame[2 : 2 + MARK, 2 : 2 + MARK] = (0, 0, 255)  # BGR red
            w.write(frame)
    w.release()
    return spans


def make_index_video(path: Path, n: int = 90, fps: int = FPS) -> None:
    """Frame i carries its index as 8 black/white 32 px blocks (robust to lossy coding)."""
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (256, 64))
    for i in range(n):
        f = np.zeros((64, 256, 3), np.uint8)
        for b in range(8):
            if (i >> b) & 1:
                f[:, b * 32 : (b + 1) * 32] = 255
        w.write(f)
    w.release()


def read_index(rgb: np.ndarray) -> int:
    g = rgb.astype(np.float32).mean(-1)
    h, wd = g.shape
    sx = wd / 256
    return sum(
        1 << b for b in range(8) if g[h // 2, int((b * 32 + 16) * sx)] > 127
    )
