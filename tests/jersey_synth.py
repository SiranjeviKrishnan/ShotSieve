"""Synthetic sports frames for the jersey tests: numbers rendered on shirt-like crops (varied
fonts, rotation, blur, noise) and pasted on a pitch-like background."""

from __future__ import annotations

import cv2
import numpy as np

FONTS = [cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_SIMPLEX,
         cv2.FONT_HERSHEY_COMPLEX]  # fmt: skip


def draw_number(
    canvas: np.ndarray,
    number: str,
    center: tuple[int, int],
    height: int,
    font: int = cv2.FONT_HERSHEY_DUPLEX,
    angle: float = 0.0,
    color=(255, 255, 255),
) -> tuple[int, int, int, int]:
    """Draw `number` with digit height ~`height` px, rotated about its centre; returns the
    axis-aligned (x, y, w, h) it covers."""
    scale = 1.0
    (tw, th), _ = cv2.getTextSize(number, font, scale, 1)
    scale = height / th
    thick = max(2, int(round(height / 7)))
    (tw, th), base = cv2.getTextSize(number, font, scale, thick)
    pad = int(height * 0.5)
    layer = np.zeros((th + base + 2 * pad, tw + 2 * pad, 4), np.uint8)
    cv2.putText(layer, number, (pad, pad + th), font, scale, (*color, 255), thick, cv2.LINE_AA)
    m = cv2.getRotationMatrix2D((layer.shape[1] / 2, layer.shape[0] / 2), angle, 1.0)
    layer = cv2.warpAffine(layer, m, (layer.shape[1], layer.shape[0]))
    ys, xs = np.nonzero(layer[:, :, 3])
    x0, y0 = center[0] - layer.shape[1] // 2, center[1] - layer.shape[0] // 2
    H, W = canvas.shape[:2]
    sx0, sy0 = max(0, -x0), max(0, -y0)
    ex, ey = min(layer.shape[1], W - x0), min(layer.shape[0], H - y0)
    sub = layer[sy0:ey, sx0:ex]
    a = sub[:, :, 3:4].astype(np.float32) / 255
    dst = canvas[y0 + sy0 : y0 + ey, x0 + sx0 : x0 + ex]
    dst[:] = (dst * (1 - a) + sub[:, :, :3] * a).astype(np.uint8)
    return (x0 + int(xs.min()), y0 + int(ys.min()), int(xs.max() - xs.min()),
            int(ys.max() - ys.min()))  # fmt: skip


def shirt(
    number: str,
    size: tuple[int, int] = (480, 360),
    num_h_frac: float = 0.10,
    seed: int = 0,
    angle: float = 0.0,
    blur: float = 0.0,
    font: int | None = None,
    color=(40, 70, 190),
    ink=(255, 255, 255),
) -> np.ndarray:
    """RGB image (h, w) of a shirt-like fabric (gradient + noise) with `number` on it; the
    digit height is `num_h_frac` of the image height."""
    h, w = size
    rng = np.random.default_rng(seed)
    base = np.zeros((h, w, 3), np.float32)
    base[:] = color
    base *= (0.8 + 0.4 * np.linspace(0, 1, h)[:, None, None])
    base += rng.normal(0, 6, (h, w, 3))
    img = np.clip(base, 0, 255).astype(np.uint8)
    f = FONTS[seed % len(FONTS)] if font is None else font
    draw_number(img, number, (w // 2, h // 2), int(h * num_h_frac), f, angle, ink)
    if blur > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    return img


def sports_frame(
    players: list[tuple[str, tuple[int, int]]],
    size: tuple[int, int] = (720, 1080),
    num_h_frac: float = 0.06,
    seed: int = 0,
    angle: float = 0.0,
    blur: float = 0.0,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """A pitch-coloured frame with a shirt-coloured torso rectangle and a number per player
    (`(number, (cx, cy))`); returns (RGB image, number boxes)."""
    h, w = size
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (50, 120, 55)
    img = np.clip(img + rng.normal(0, 7, img.shape), 0, 255).astype(np.uint8)
    boxes = []
    for i, (num, (cx, cy)) in enumerate(players):
        th = int(h * 0.30)
        cv2.rectangle(img, (cx - th // 3, cy - th // 3), (cx + th // 3, cy + th // 2),
                      (40, 60, 180) if i % 2 == 0 else (220, 200, 40), -1)  # fmt: skip
        ink = (255, 255, 255) if i % 2 == 0 else (20, 20, 20)
        boxes.append(draw_number(img, num, (cx, cy), int(h * num_h_frac),
                                 FONTS[(seed + i) % len(FONTS)], angle, ink))  # fmt: skip
    if blur > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    return img, boxes


def labelled_set(n: int = 100, seed: int = 0):
    """`n` labelled frames: ~70 % with 1-2 numbered players (3.5-10 % height, +-15 deg, blur),
    ~15 % scoreboards / banners (long text, no jersey), ~15 % empty pitch. Yields
    (image, [numbers])."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        kind = rng.random()
        h, w = 720, 1080
        if kind < 0.70:
            k = int(rng.integers(1, 3))
            nums = [str(int(rng.integers(2, 100))) for _ in range(k)]
            pos = [(int(w * (0.3 + 0.4 * j) + rng.integers(-40, 40)), h // 2) for j in range(k)]
            img, _ = sports_frame(
                list(zip(nums, pos, strict=True)), (h, w), float(rng.uniform(0.045, 0.10)),
                seed=i, angle=float(rng.uniform(-15, 15)),
                blur=float(rng.choice([0.0, 0.8, 1.5])),
            )  # fmt: skip
            out.append((img, nums))
        elif kind < 0.85:
            img, _ = sports_frame([], (h, w), seed=i)
            cv2.rectangle(img, (150, 40), (930, 120), (20, 20, 20), -1)
            cv2.putText(img, f"HOME {int(rng.integers(0, 9))} : {int(rng.integers(0, 9))} AWAY",
                        (180, 98), cv2.FONT_HERSHEY_DUPLEX, 1.6, (255, 255, 255), 3, cv2.LINE_AA)
            out.append((img, []))
        else:
            img, _ = sports_frame([], (h, w), seed=i)
            out.append((img, []))
    return out
