"""Synthetic scenes for the T22 crop tests (docs/11_CROP.md section 10), with exact boxes.

`uv run python scripts/make_crop_fixtures.py` writes `tests/fixtures/crop/synth/*.jpg` and
`synth_labels.json` (deterministic: fixed seeds, no downloads). The scenes are rendered shapes
(ellipse "faces" with eye dots, rectangle "bodies", textured objects, a horizon line): the unit
tests for detection helpers (saliency, point prompts, horizon) and the saliency-peak check use
them; the geometry / scoring tests need no images at all."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures" / "crop" / "synth"


def _bg(h: int, w: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.zeros((h, w, 3), np.float32)
    base[:] = (150, 160, 150)
    grad = np.linspace(-20, 20, w, dtype=np.float32)[None, :, None]
    noise = rng.normal(0, 3.0, (h, w, 1)).astype(np.float32)
    return np.clip(base + grad + noise, 0, 255).astype(np.uint8)


def _textured(img: np.ndarray, box: tuple[int, int, int, int], seed: int) -> None:
    """A high-contrast checker / stripe patch: unlike the smooth background."""
    x0, y0, x1, y1 = box
    rng = np.random.default_rng(seed)
    h, w = y1 - y0, x1 - x0
    yy, xx = np.mgrid[0:h, 0:w]
    check = (((yy // 8) + (xx // 8)) % 2) * 200 + 30
    col = np.array([rng.integers(40, 220), rng.integers(40, 220), rng.integers(40, 220)])
    patch = np.stack([check * c / 255 for c in col], axis=-1)
    img[y0:y1, x0:x1] = np.clip(patch, 0, 255).astype(np.uint8)


def single_object(w: int = 640, h: int = 480, box=(300, 160, 480, 360), seed: int = 1):
    """One textured object on a smooth background; returns (rgb, normalised box)."""
    img = _bg(h, w, seed)
    _textured(img, box, seed)
    return img, (box[0] / w, box[1] / h, box[2] / w, box[3] / h)


def horizon_scene(w: int = 800, h: int = 500, y: int = 300, tilt_deg: float = 0.0):
    """Sky over ground separated by a (possibly tilted) horizon line."""
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (170, 200, 235)
    yy, xx = np.mgrid[0:h, 0:w]
    line = y + (xx - w / 2) * np.tan(np.radians(tilt_deg))
    img[yy > line] = (60, 90, 50)
    return img, y / h


def face_scene(w: int = 640, h: int = 480):
    """Two ellipse faces on bodies. Returns (rgb, [{kind, box}]) in normalised coordinates."""
    img = _bg(h, w, 3)
    subs = []
    for cx, cy, fw in ((210, 190, 70), (430, 210, 62)):
        fh = int(fw * 1.3)
        cv2.rectangle(img, (cx - fw, cy + fh), (cx + fw, h - 10), (40, 60, 160), -1)
        cv2.ellipse(img, (cx, cy), (fw // 2 + 8, fh // 2 + 8), 0, 0, 360, (225, 180, 150), -1)
        for ex in (-fw // 5, fw // 5):
            cv2.circle(img, (cx + ex, cy - fh // 8), 4, (30, 30, 30), -1)
        subs.append(
            {
                "kind": "face",
                "box": [
                    (cx - fw // 2 - 8) / w,
                    (cy - fh // 2 - 8) / h,
                    (cx + fw // 2 + 8) / w,
                    (cy + fh // 2 + 8) / h,
                ],
            }
        )
        subs.append(
            {
                "kind": "person",
                "box": [(cx - fw) / w, (cy - fh // 2 - 8) / h, (cx + fw) / w, (h - 10) / h],
            }
        )
    return img, subs


def main(argv: list[str] | None = None) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    labels: dict[str, object] = {}
    img, box = single_object()
    cv2.imwrite(str(OUT / "object.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    labels["object.jpg"] = {"subjects": [{"kind": "object", "box": list(box), "must": True}]}
    img, y = horizon_scene()
    cv2.imwrite(str(OUT / "horizon.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    labels["horizon.jpg"] = {"subjects": [], "horizon_y": y}
    img, subs = face_scene()
    cv2.imwrite(str(OUT / "faces.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    labels["faces.jpg"] = {"subjects": subs}
    (OUT / "synth_labels.json").write_text(json.dumps(labels, indent=2))
    print(f"wrote {len(labels)} synthetic scenes to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
