# ruff: noqa: E501
"""T23f: measure what the loupe really shows against ground truth (reproducible, read-only).

    uv run python scripts/viewer_quality_report.py SHOOT_DIR OUT_DIR [--label before|after]

For every image in SHOOT_DIR (RAW / JPEG / HEIC) and each display geometry (3440x1440 at
devicePixelRatio 1, 1728x1117 at 2) it opens the loupe in Chromium at Fit, screenshots the picture
and compares it with
  (b) the camera's embedded JPEG / the original JPEG, converted to sRGB with its ICC profile and
      resized to the displayed size with Lanczos, and
  (c) macOS `sips` (Apple's own decoder, sRGB-matched) resized the same way: the ground truth.
Metrics: pixel size served vs displayed size (device px), Laplacian variance ratio vs (c), mean
CIEDE2000 vs (c), JPEG quality of the served file, ICC profile of the served file.
Crops (loupe | embedded | sips) are saved as PNG side-by-side strips. Nothing in SHOOT_DIR is modified.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageCms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from shotsieve.io.viewer import jpeg_quality  # noqa: E402

GEOMETRIES = {"3440x1440@1x": (3440, 1440, 1), "1728x1117@2x": (1728, 1117, 2)}
SRGB = ImageCms.createProfile("sRGB")
SRGB_ICC = "/System/Library/ColorSync/Profiles/sRGB Profile.icc"


# ---- metrics --------------------------------------------------------------------------------


def lap_var(rgb: np.ndarray) -> float:
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def deltae2000(a: np.ndarray, b: np.ndarray) -> float:
    """Mean CIEDE2000 between two sRGB uint8 images (vectorised)."""
    la = cv2.cvtColor(a.astype(np.float32) / 255, cv2.COLOR_RGB2Lab).reshape(-1, 3).astype(np.float64)
    lb = cv2.cvtColor(b.astype(np.float32) / 255, cv2.COLOR_RGB2Lab).reshape(-1, 3).astype(np.float64)
    L1, a1, b1 = la.T
    L2, a2, b2 = lb.T
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    Cb = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(Cb**7 / (Cb**7 + 25.0**7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1 = np.degrees(np.arctan2(b1, a1p)) % 360
    h2 = np.degrees(np.arctan2(b2, a2p)) % 360
    dL, dC = L2 - L1, C2p - C1p
    dh = h2 - h1
    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))
    dh = np.where(C1p * C2p == 0, 0, dh)
    dH = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh) / 2)
    Lb, Cpb = (L1 + L2) / 2, (C1p + C2p) / 2
    hs = h1 + h2
    hb = np.where(np.abs(h1 - h2) > 180, (hs + 360) / 2, hs / 2)
    hb = np.where(C1p * C2p == 0, hs, hb)
    T = (1 - 0.17 * np.cos(np.radians(hb - 30)) + 0.24 * np.cos(np.radians(2 * hb))
         + 0.32 * np.cos(np.radians(3 * hb + 6)) - 0.20 * np.cos(np.radians(4 * hb - 63)))  # fmt: skip
    dth = 30 * np.exp(-(((hb - 275) / 25) ** 2))
    Rc = 2 * np.sqrt(Cpb**7 / (Cpb**7 + 25.0**7))
    Sl = 1 + 0.015 * (Lb - 50) ** 2 / np.sqrt(20 + (Lb - 50) ** 2)
    Sc, Sh = 1 + 0.045 * Cpb, 1 + 0.015 * Cpb * T
    Rt = -np.sin(np.radians(2 * dth)) * Rc
    d = np.sqrt((dL / Sl) ** 2 + (dC / Sc) ** 2 + (dH / Sh) ** 2 + Rt * (dC / Sc) * (dH / Sh))
    return float(np.mean(d))


def to_srgb(im: Image.Image) -> np.ndarray:
    """RGB uint8 in sRGB: converts from the embedded ICC profile when there is one."""
    icc = im.info.get("icc_profile")
    if icc:
        try:
            src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            im = ImageCms.profileToProfile(im.convert("RGB"), src, SRGB, outputMode="RGB")
        except Exception:  # noqa: BLE001
            pass
    return np.asarray(im.convert("RGB"))


def lf(rgb: np.ndarray) -> np.ndarray:
    """1/8 size: compares tone and colour only (no noise / detail)."""
    h, w = rgb.shape[:2]
    return cv2.resize(rgb, (max(1, w // 8), max(1, h // 8)), interpolation=cv2.INTER_AREA)


def lanczos(rgb: np.ndarray, w: int, h: int) -> np.ndarray:
    """Centre-crop to the aspect of w x h (a camera's 16:9 JPEG vs the 3:2 sensor), then resize."""
    ih, iw = rgb.shape[:2]
    if abs(iw / ih - w / h) > 0.01:
        if iw / ih > w / h:
            nw = round(ih * w / h)
            rgb = rgb[:, (iw - nw) // 2 : (iw - nw) // 2 + nw]
        else:
            nh = round(iw * h / w)
            rgb = rgb[(ih - nh) // 2 : (ih - nh) // 2 + nh]
    return np.asarray(Image.fromarray(rgb).resize((w, h), Image.LANCZOS))


def reference_embedded(path: Path) -> np.ndarray:
    """Camera embedded JPEG (RAW) or the original raster, orientation applied, sRGB."""
    import rawpy

    if path.suffix.lower() in {".arw", ".dng", ".cr2", ".nef"}:
        with rawpy.imread(str(path)) as raw:
            t = raw.extract_thumb()
            data = bytes(t.data)
        im = Image.open(io.BytesIO(data))
        from PIL import ImageOps

        return to_srgb(ImageOps.exif_transpose(im))
    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
    except ImportError:
        pass
    from PIL import ImageOps

    im = Image.open(path)
    return to_srgb(ImageOps.exif_transpose(im))


def reference_sips(path: Path, work: Path) -> np.ndarray:
    out = work / f"{path.stem}_sips.png"
    subprocess.run(
        ["sips", "-s", "format", "png", "--matchTo", SRGB_ICC, str(path), "--out", str(out)],
        check=True, capture_output=True,
    )  # fmt: skip
    return np.asarray(Image.open(out).convert("RGB"))


# ---- driving the app ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def seed(folder: Path) -> None:
    from shotsieve.features.vector import FeatureVector  # noqa: PLC0415
    from shotsieve.pipeline.cache import Cache

    rows = []
    for i, p in enumerate(sorted(x for x in folder.iterdir() if x.is_file())):
        rows.append(dict(path=str(p), decision="keep", reasons=[], score=0.5, prior=0.5,
                         group_id=i, burst_id=i, is_best=True, audit=False, audit_prob=None,
                         features=FeatureVector(n_faces=0.0).to_dict()))  # fmt: skip
    with Cache.for_folder(folder) as c:
        c.save_decisions(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("shoot")
    ap.add_argument("out")
    ap.add_argument("--label", default="run")
    a = ap.parse_args(argv)
    shoot, out = Path(a.shoot).resolve(), Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    seed(shoot)
    port = _free_port()
    data = out / "_data"
    data.mkdir(exist_ok=True)
    (data / "onboarding.json").write_text(json.dumps({"version": 1, "completed_at": 1.0}))
    env = {**os.environ, "SHOTSIEVE_DATA_DIR": str(data)}
    proc = subprocess.Popen(
        [sys.executable, "-m", "shotsieve.cli", "serve", str(shoot), "--port", str(port), "--no-open"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=ROOT, env=env,
    )  # fmt: skip
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            urllib.request.urlopen(base + "/api/health", timeout=1).read()
            break
        except OSError:
            time.sleep(0.2)
    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(ROOT / ".playwright"))
    from playwright.sync_api import sync_playwright

    results: list[dict] = []
    work = out / "_sips"
    work.mkdir(exist_ok=True)
    try:
        with sync_playwright() as pw:
            br = pw.chromium.launch()
            for gname, (vw, vh, dpr) in GEOMETRIES.items():
                ctx = br.new_context(viewport={"width": vw, "height": vh}, device_scale_factor=dpr)
                page = ctx.new_page()
                page.goto(base + "/")
                page.wait_for_selector(".cell")
                for cell in page.locator(".cell").all():
                    name = cell.get_attribute("title")
                    cell.click()
                    page.keyboard.press("Space")
                    page.wait_for_function(
                        "(() => { const i = document.getElementById('big'); return i.complete && i.naturalWidth > 0; })()"
                    )
                    page.wait_for_timeout(1200)  # let a resize-driven reload settle
                    page.wait_for_function(
                        "(() => { const i = document.getElementById('big'); return i.complete && i.naturalWidth > 0; })()"
                    )
                    info = page.evaluate(
                        """() => { const i = document.getElementById('big'); const r = i.getBoundingClientRect();
                        const nw = i.naturalWidth, nh = i.naturalHeight;
                        const f = Math.min(r.width / nw, r.height / nh);
                        const cw = nw * f, ch = nh * f;
                        return {src: i.src, nw, nh, x: r.x + (r.width - cw) / 2, y: r.y + (r.height - ch) / 2, cw, ch,
                                dpr: window.devicePixelRatio, fit: r.width}; }"""
                    )
                    dw, dh = round(info["cw"] * dpr), round(info["ch"] * dpr)
                    shot = page.screenshot(clip={"x": info["x"], "y": info["y"], "width": info["cw"], "height": info["ch"]})
                    loupe = np.asarray(Image.open(io.BytesIO(shot)).convert("RGB"))
                    dh, dw = loupe.shape[:2]
                    served = urllib.request.urlopen(info["src"]).read()
                    sim = Image.open(io.BytesIO(served))
                    icc = sim.info.get("icc_profile")
                    icc_name = None
                    if icc:
                        try:
                            icc_name = ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(io.BytesIO(icc))).strip()
                        except Exception:  # noqa: BLE001
                            icc_name = "unreadable"
                    path = shoot / name
                    emb = lanczos(reference_embedded(path), dw, dh)
                    sps = lanczos(reference_sips(path, work), dw, dh)
                    served_rs = lanczos(to_srgb(sim), dw, dh)  # the served file alone (no browser resample)
                    # compare on the same frame: sips shows the uncropped sensor, embedded may be a crop
                    row = {
                        "file": name, "geometry": gname, "dpr": dpr,
                        "served_px": [info["nw"], info["nh"]],
                        "displayed_device_px": [dw, dh],
                        "upscale_factor": round(max(dw / info["nw"], dh / info["nh"]), 3),
                        "served_url": re.sub(r"[?&]folder=[^&]*", "", info["src"].replace(base, "")),  # no local paths in the report
                        "served_bytes": len(served),
                        "jpeg_quality": jpeg_quality(served),
                        "icc": icc_name or ("none" if not icc else "present"),
                        "sharp_loupe": round(lap_var(loupe), 3),
                        "sharp_embedded_ref": round(lap_var(emb), 3),
                        "sharp_sips_ref": round(lap_var(sps), 3),
                        "sharp_ratio_vs_sips": round(lap_var(loupe) / max(lap_var(sps), 1e-9), 3),
                        "sharp_ratio_vs_embedded": round(lap_var(loupe) / max(lap_var(emb), 1e-9), 3),
                        "dE2000_loupe_vs_sips": round(deltae2000(loupe, sps), 3),
                        "dE2000_loupe_vs_embedded": round(deltae2000(loupe, emb), 3),
                        "dE2000_embedded_vs_sips": round(deltae2000(emb, sps), 3),
                        "dE2000_served_vs_embedded": round(deltae2000(served_rs, emb), 3),
                        "sharp_ratio_served_vs_embedded": round(lap_var(served_rs) / max(lap_var(emb), 1e-9), 3),
                        "dE2000_lowfreq_loupe_vs_sips": round(deltae2000(lf(loupe), lf(sps)), 3),
                        "dE2000_lowfreq_loupe_vs_embedded": round(deltae2000(lf(loupe), lf(emb)), 3),
                        "dE2000_lowfreq_embedded_vs_sips": round(deltae2000(lf(emb), lf(sps)), 3),
                        "aspect_loupe": round(dw / dh, 4),
                        "aspect_sips": round(np.asarray(Image.open(work / f"{path.stem}_sips.png")).shape[1] / np.asarray(Image.open(work / f"{path.stem}_sips.png")).shape[0], 4),
                    }  # fmt: skip
                    results.append(row)
                    c = 360  # centre crops, same area of the three, side by side
                    lap = np.abs(cv2.Laplacian(cv2.cvtColor(sps, cv2.COLOR_RGB2GRAY), cv2.CV_32F))
                    box = cv2.boxFilter(lap, -1, (c, c), normalize=True)  # most detailed c x c window
                    yy, xx = np.unravel_index(int(box.argmax()), box.shape)
                    y0 = int(np.clip(yy - c // 2, 0, max(0, dh - c)))
                    x0 = int(np.clip(xx - c // 2, 0, max(0, dw - c)))
                    strip = np.hstack([x[y0 : y0 + c, x0 : x0 + c] for x in (loupe, emb, sps)])
                    Image.fromarray(strip).save(out / f"{a.label}_{gname}_{path.stem}.png")
                    page.keyboard.press("Escape")
                ctx.close()
            br.close()
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
    (out / f"{a.label}.json").write_text(json.dumps(results, indent=1))
    for r in results:
        print(
            f"{r['geometry']:13s} {r['file']:20s} served {r['served_px']} -> shown {r['displayed_device_px']} "
            f"x{r['upscale_factor']:.2f} q={r['jpeg_quality']} icc={r['icc']} sharp/sips={r['sharp_ratio_vs_sips']:.2f} "
            f"dE={r['dE2000_loupe_vs_sips']:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
