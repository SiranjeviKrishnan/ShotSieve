"""Reproducible README screenshots: `uv run python scripts/make_screenshots.py [--out docs/images]`.

Builds a demo shoot from the CC0 test fixtures only (tests/fixtures/crop/real and crop2/real, the
closed-eyes fixture; see tests/fixtures/*/LICENSES.md), plus synthetic burst / blur / dark
variants made from those same photos. It culls and indexes the shoot with a throw-away data
folder, drives the real web UI with Playwright (Chromium, 1440x900, dark mode) and writes
optimised WebP images (each <= 400 KB) and a short animated WebP (<= 3 MB).

Needs the models (`uv run python scripts/fetch_models.py`) and Chromium
(`PLAYWRIGHT_BROWSERS_PATH=.playwright uv run playwright install chromium`).
Nothing of your own is ever used: the shoot lives in a temp folder.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import piexif
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"
W, H = 1440, 900
MAX_KB = 400
T0 = datetime(2026, 6, 13, 14, 0, 0)  # an arbitrary demo timeline


def stamp(path: Path, when: datetime, serial: str = "DEMO0001") -> None:
    ex = {"0th": {}, "Exif": {
        piexif.ExifIFD.DateTimeOriginal: when.strftime("%Y:%m:%d %H:%M:%S").encode(),
        piexif.ExifIFD.BodySerialNumber: serial.encode()}, "GPS": {}, "1st": {}}
    try:
        old = piexif.load(str(path))
        old["Exif"].update(ex["Exif"])
        ex = old
    except Exception:  # noqa: BLE001 - no EXIF yet
        pass
    ex["thumbnail"] = None
    ex["1st"] = {}
    piexif.insert(piexif.dump(ex), str(path))


def variant(src: Path, dst: Path, kind: str, i: int = 0) -> None:
    im = cv2.imread(str(src))
    h, w = im.shape[:2]
    if kind == "burst":  # a neighbouring frame of a burst: slight shift, scale and exposure
        dx, dy = (i * 5) % 17 - 6, (i * 3) % 11 - 4
        M = np.float32([[1.0 + 0.004 * i, 0, dx], [0, 1.0 + 0.004 * i, dy]])
        im = cv2.warpAffine(im, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        im = cv2.convertScaleAbs(im, alpha=1.0 + 0.02 * (i % 3 - 1), beta=2 * i)
    elif kind == "blur":
        im = cv2.GaussianBlur(im, (0, 0), 7)
    elif kind == "dark":
        im = cv2.convertScaleAbs(im, alpha=0.18, beta=0)
    cv2.imwrite(str(dst), im, [cv2.IMWRITE_JPEG_QUALITY, 90])


PICKS = [  # (fixture folder, file) in shoot order; no football crowds (they flood the People view)
    ("crop2", "portrait_02"), ("crop2", "portrait_03"), ("crop2", "group_04"),
    ("crop", "portrait_01"), ("crop2", "animal_01"), ("crop2", "horizon_01"),
    ("crop", "portrait_04"), ("crop2", "group_02"), ("crop2", "object_01"),
    ("crop", "animal_02"), ("crop", "portrait_06"), ("crop2", "horizon_02"),
    ("crop2", "animal_02"), ("crop", "group_03"), ("crop2", "portrait_04"),
    ("crop", "horizon_01"), ("crop2", "object_02"), ("crop", "portrait_08"),
    ("crop2", "group_03"), ("crop", "animal_03"), ("crop2", "portrait_05"),
    ("crop", "object_01"), ("crop", "portrait_09"), ("crop2", "animal_03"),
    ("crop", "horizon_02"), ("crop", "group_05"), ("crop2", "horizon_03"),
    ("crop", "portrait_10"), ("crop2", "object_03"), ("crop", "animal_04"),
]
BURSTS = {1, 6, 14}  # picks that get three near-identical neighbours
BLURS = (4, 9)  # picks that also get a blurred copy; 12 gets a dark one


def build_shoot(dest: Path) -> tuple[Path, dict[str, str]]:
    """Returns (folder, {role: file name}) with roles used by the screenshot steps."""
    dest.mkdir(parents=True)
    t, n = T0, 0
    roles: dict[str, str] = {}

    def next_name() -> str:
        nonlocal n
        n += 1
        return f"DSC_{1000 + n:04d}.jpg"

    for k, (d, stem) in enumerate(PICKS):
        src = FIX / d / "real" / f"{stem}.jpg"
        name = next_name()
        shutil.copy(src, dest / name)
        stamp(dest / name, t)
        roles.setdefault(f"pick{k}", name)
        roles[f"{d}/{stem}"] = name
        if k in BURSTS:  # near-duplicates for the compare view
            roles.setdefault("burst", name)
            for j in range(1, 4):
                name = next_name()
                variant(src, dest / name, "burst", j)
                stamp(dest / name, t + timedelta(milliseconds=350 * j))
        t += timedelta(seconds=40)
        if k in BLURS or k == 12:
            name = next_name()
            variant(src, dest / name, "dark" if k == 12 else "blur")
            stamp(dest / name, t)
            roles.setdefault("blur" if k != 12 else "dark", name)
            t += timedelta(seconds=40)
    name = next_name()
    shutil.copy(FIX / "closed_eyes_cc0" / "closed_eyes.jpg", dest / name)
    stamp(dest / name, t)
    roles["closed"] = name
    return dest, roles


class Server:
    def __init__(self, folder: Path, data: Path) -> None:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}"
        data.mkdir(exist_ok=True)
        (data / "onboarding.json").write_text(
            json.dumps({"version": 1, "completed_at": 1.0, "skipped": False, "last_step": 5}))
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "shotsieve.cli", "serve", str(folder), "--port", str(self.port),
             "--no-open"], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            env={**os.environ, "SHOTSIEVE_DATA_DIR": str(data)})
        for _ in range(200):
            try:
                urllib.request.urlopen(self.url + "/api/health", timeout=1).read()
                return
            except OSError:
                time.sleep(0.25)
        raise RuntimeError("server did not start")

    def call(self, path: str, body: dict | None = None, timeout: float = 120):
        req = urllib.request.Request(
            self.url + path, data=None if body is None else json.dumps(body).encode(),
            method="GET" if body is None else "POST",
            headers={"Content-Type": "application/json", "Origin": self.url})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")

    def wait_job(self, job_id: str, timeout: float = 600) -> None:
        t0 = time.time()
        while time.time() - t0 < timeout:
            j = self.call(f"/api/jobs/{job_id}")
            if j["state"] in ("done", "error"):
                assert j["state"] == "done", j
                return
            time.sleep(0.5)
        raise TimeoutError(job_id)

    def stop(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(10)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def save_webp(img: Image.Image, path: Path, max_kb: int = MAX_KB) -> int:
    for q in (88, 82, 76, 70, 62, 55, 48):
        buf = io.BytesIO()
        img.save(buf, "WEBP", quality=q, method=6)
        if buf.tell() <= max_kb * 1024:
            path.write_bytes(buf.getvalue())
            return buf.tell()
    raise RuntimeError(f"{path.name} does not fit in {max_kb} KB")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "images"))
    ap.add_argument("--keep", action="store_true", help="keep the temp shoot and print its path")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(ROOT / ".playwright"))
    from playwright.sync_api import sync_playwright

    tmp = Path(tempfile.mkdtemp(prefix="shotsieve-shots-"))
    shoot, roles = build_shoot(tmp / "Demo shoot")
    srv = Server(shoot, tmp / "data")
    frames: list[Image.Image] = []
    try:
        print(f"culling {len(list(shoot.glob('*.jpg')))} photos ...")
        srv.wait_job(srv.call("/api/cull", {"folder": str(shoot)})["job_id"])
        srv.call("/api/people/consent", {"folder": str(shoot), "on": True})
        srv.call("/api/index/start", {"folder": str(shoot), "people": True, "ocr": False})
        for _ in range(600):
            st = srv.call("/api/index/status?folder=" + urllib.parse.quote(str(shoot)))
            if not st.get("running") and st.get("indexed", 0) >= 1:
                break
            time.sleep(0.5)
        with sync_playwright() as pw:
            b = pw.chromium.launch()
            ctx = b.new_context(viewport={"width": W, "height": H}, color_scheme="dark",
                                device_scale_factor=1)
            pg = ctx.new_page()
            errors: list[str] = []
            pg.on("pageerror", lambda e: errors.append(str(e)))
            pg.goto(srv.url + "/")
            pg.wait_for_selector(".cell", timeout=60000)
            pg.wait_for_timeout(1500)

            def shot(name: str, add_frame: bool = False) -> None:
                png = pg.screenshot()
                im = Image.open(io.BytesIO(png)).convert("RGB")
                kb = save_webp(im, out / f"{name}.webp") // 1024
                print(f"  {name}.webp {kb} KB")
                if add_frame:
                    frames.append(im.resize((960, 600), Image.LANCZOS))

            def settle(ms: int = 900) -> None:
                pg.wait_for_timeout(ms)

            pg.evaluate("document.querySelector('#folder').value = '~/Pictures/Demo shoot'")

            def pick(role: str) -> None:
                c = pg.locator(f'.cell[title="{roles[role]}"]')
                c.scroll_into_view_if_needed()
                c.click()

            # 1. grid + filters; a blurry frame selected: the side panel shows the reason
            pick("blur")
            settle()
            shot("grid", add_frame=True)

            # 2. loupe at 100 % zoom
            pick("pick3")
            pg.keyboard.press("Space")
            settle(1200)
            shot("loupe", add_frame=True)  # demo-clip frame only (README: loupe-zoom)
            (out / "loupe.webp").unlink(missing_ok=True)
            pg.keyboard.press("z")
            settle(1800)
            shot("loupe-zoom", add_frame=True)
            pg.keyboard.press("Escape")
            settle()

            # 3. compare (a burst frame against the best of its group)
            pick("burst")
            pg.keyboard.press("c")
            settle(2200)
            shot("compare", add_frame=True)
            pg.keyboard.press("Escape")
            settle()

            # 4. crop suggestions with a subject box
            pick("crop2/animal_02")
            pg.keyboard.press("r")
            try:
                pg.wait_for_selector('#crop-ui[data-busy="0"]', timeout=60000)
            except Exception:  # noqa: BLE001
                pass
            settle(1500)
            box = pg.locator("#crop-subjects .subj").first
            if box.count():
                bb = box.bounding_box()
                pg.mouse.click(bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2)
                settle(2200)
            shot("crop", add_frame=True)
            pg.keyboard.press("Escape")
            pg.keyboard.press("Escape")
            settle()

            # 5. People
            pg.keyboard.press("Shift+P")
            settle(2500)
            shot("people", add_frame=True)
            pg.keyboard.press("Escape")
            if errors:
                print("JS errors:", errors, file=sys.stderr)
            b.close()
        if frames:
            order = [0, 1, 2, 3, 4, 5, 0][: len(frames) + 1]
            seq = [frames[i % len(frames)] for i in order]
            for q in (70, 60, 50, 40):
                buf = io.BytesIO()
                seq[0].save(buf, "WEBP", save_all=True, append_images=seq[1:], duration=1500,
                            loop=0, quality=q, method=6)
                if buf.tell() <= 3 * 1024 * 1024:
                    (out / "demo.webp").write_bytes(buf.getvalue())
                    print(f"  demo.webp {buf.tell() // 1024} KB ({len(seq)} frames)")
                    break
            else:
                raise RuntimeError("demo.webp does not fit in 3 MB")
    finally:
        srv.stop()
        if args.keep:
            print("kept", tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
