"""Build ShotSieve test fixtures into tests/fixtures/generated/ (idempotent).

Downloads freely usable sample images (OpenCV sample data and opencv_extra test data, both
Apache-2.0/BSD repositories; raw.pixls.us DNGs are CC0), then synthesizes a shoot folder with
EXIF, an oriented JPEG and a synthetic Lightroom catalog.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import struct
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import piexif
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "tests" / "fixtures" / "generated"

_CV = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data"
_EX = "https://raw.githubusercontent.com/opencv/opencv_extra/4.x/testdata/cv"
_PX = "https://raw.pixls.us/data"

# name -> (list of candidate URLs, min bytes)
IMAGES: dict[str, tuple[list[str], int]] = {
    "lena.jpg": ([f"{_CV}/lena.jpg", f"{_EX}/shared/lena.png"], 10_000),
    "messi5.jpg": ([f"{_CV}/messi5.jpg"], 10_000),
    "building.jpg": ([f"{_CV}/building.jpg"], 10_000),
    "portrait2.jpg": ([f"{_EX}/face/detect.jpg"], 10_000),
    "portrait3.jpg": ([f"{_EX}/face/david1.jpg"], 5_000),
    "group.png": ([f"{_EX}/cascadeandhog/images/class57.png"], 100_000),
}
RAWS: dict[str, tuple[list[str], int]] = {
    "pentax_kx.dng": ([f"{_PX}/Pentax/K-x/IMGP6431.DNG"], 5_000_000),
    "panasonic_lx1.dng": ([f"{_PX}/Panasonic/DMC-LX1/P1080763.dng"], 5_000_000),
}
# Subjects for the natural-language search tests (tests/fixtures/search_queries.json).
_GH = "https://raw.githubusercontent.com"
SEARCH_IMAGES: dict[str, tuple[list[str], int]] = {
    "dog.jpg": (["https://github.com/pytorch/hub/raw/master/images/dog.jpg",
                 f"{_GH}/pjreddie/darknet/master/data/dog.jpg"], 20_000),
    "red_car.jpg": (["https://commons.wikimedia.org/wiki/Special:FilePath/"
                     "Red_Ferrari_F40.jpg?width=640"], 20_000),
    "bus.jpg": ([f"{_GH}/ultralytics/yolov5/master/data/images/bus.jpg"], 20_000),
    "fruits.jpg": ([f"{_CV}/fruits.jpg"], 10_000),
    "butterfly.jpg": ([f"{_CV}/butterfly.jpg"], 10_000),
    "starry_night.jpg": ([f"{_CV}/starry_night.jpg"], 10_000),
    "baboon.jpg": ([f"{_CV}/baboon.jpg"], 10_000),
    "home.jpg": ([f"{_CV}/home.jpg"], 10_000),
}  # fmt: skip
SERIAL = "SN123456"
BASE_TIME = datetime(2024, 6, 1, 14, 0, 0)


def download(name: str, urls: list[str], min_bytes: int, dest_dir: Path) -> Path:
    """Fetch the first working URL to dest_dir/name; fail loudly if none works."""
    dest = dest_dir / name
    if dest.exists() and dest.stat().st_size >= min_bytes:
        return dest
    dest_dir.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "shotsieve-fixtures"})
            data = urllib.request.urlopen(req, timeout=120).read()
            if len(data) < min_bytes:
                raise ValueError(f"only {len(data)} bytes")
            tmp = dest.with_name(dest.name + ".part")
            tmp.write_bytes(data)
            if dest.suffix.lower() in {".jpg", ".png"}:
                if cv2.imread(str(tmp)) is None:
                    raise ValueError("not a decodable image")
            tmp.replace(dest)
            return dest
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{url}: {exc}")
    raise RuntimeError(f"could not fetch {name}:\n" + "\n".join(errors))


def adjust_ev(rgb: np.ndarray, ev: float) -> np.ndarray:
    """Scale linear-ish light by 2**ev (gamma 2.2)."""
    lin = (rgb.astype(np.float32) / 255.0) ** 2.2 * (2.0**ev)
    return np.clip(lin ** (1 / 2.2) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def exif_bytes(
    ts: float,
    flash: int | None,
    serial: str = SERIAL,
    orientation: int = 1,
    exposure: tuple[int, int] = (1, 125),
) -> bytes:
    sec = int(ts)
    sub = f"{int(round((ts - sec) * 1000)):03d}"
    stamp = datetime.fromtimestamp(sec).strftime("%Y:%m:%d %H:%M:%S").encode()
    exif_ifd = {
        piexif.ExifIFD.DateTimeOriginal: stamp,
        piexif.ExifIFD.SubSecTimeOriginal: sub.encode(),
        piexif.ExifIFD.BodySerialNumber: serial.encode(),
        piexif.ExifIFD.ISOSpeedRatings: 400,
        piexif.ExifIFD.ExposureTime: exposure,
        piexif.ExifIFD.FNumber: (28, 10),
        piexif.ExifIFD.FocalLength: (50, 1),
    }
    if flash is not None:
        exif_ifd[piexif.ExifIFD.Flash] = flash
    zeroth = {
        piexif.ImageIFD.Make: b"ShotCam",
        piexif.ImageIFD.Model: b"S1",
        piexif.ImageIFD.Orientation: orientation,
    }
    return piexif.dump({"0th": zeroth, "Exif": exif_ifd})


def save_jpeg(rgb: np.ndarray, path: Path, exif: bytes) -> None:
    Image.fromarray(rgb).save(path, quality=92, exif=exif)


def load_rgb(path: Path) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def burst_frame(base: np.ndarray, i: int, size: tuple[int, int] = (640, 640)) -> np.ndarray:
    """Frame i of a burst: small shift/crop of `base`, resized to a common size."""
    h, w = base.shape[:2]
    m = int(min(h, w) * 0.04)
    dx, dy = (i * 3) % (2 * m), (i * 2) % (2 * m)
    crop = base[dy : dy + h - 2 * m, dx : dx + w - 2 * m]
    return cv2.resize(crop, size, interpolation=cv2.INTER_CUBIC)


def build_shoot(src: Path, out: Path) -> list[dict]:
    """3 bursts x 5 frames + variants + landscape + black frame; returns the manifest."""
    out.mkdir(parents=True, exist_ok=True)
    faces = [load_rgb(src / n) for n in ("lena.jpg", "portrait2.jpg", "portrait3.jpg")]
    t0 = BASE_TIME.timestamp()
    flash_fired, flash_off = 0x1, 0x10
    burst_flash = [flash_fired, flash_off, flash_fired]
    manifest: list[dict] = []

    def add(name: str, rgb: np.ndarray, ts: float, flash: int | None, **info: object) -> None:
        save_jpeg(rgb, out / name, exif_bytes(ts, flash))
        manifest.append({"file": name, "timestamp": ts, "flash": flash, **info})

    for b in range(3):
        for i in range(5):
            frame = burst_frame(faces[b], i)
            variant = "normal"
            flash = burst_flash[b]
            if (b, i) == (0, 1):
                frame, variant = cv2.GaussianBlur(frame, (0, 0), 6), "blur"
            elif (b, i) == (1, 3):
                frame, variant = adjust_ev(frame, -3), "underexposed"
            elif (b, i) == (2, 2):
                frame, variant = adjust_ev(frame, -2), "flash_misfire"
            add(
                f"burst{b}_{i}.jpg",
                frame,
                t0 + 60 * b + 0.3 * i,
                flash,
                burst=b,
                frame=i,
                variant=variant,
            )
    land = cv2.resize(load_rgb(src / "building.jpg"), (960, 640), interpolation=cv2.INTER_AREA)
    add("landscape.jpg", land, t0 + 240, flash_off, burst=3, frame=0, variant="landscape")
    add(
        "black.jpg",
        np.zeros((640, 640, 3), np.uint8),
        t0 + 300,
        flash_off,
        burst=4,
        frame=0,
        variant="black",
    )
    return manifest


def build_oriented(src: Path, out: Path) -> None:
    """lena rotated so that it is stored sideways with EXIF Orientation=6 (needs 90 deg CW)."""
    out.mkdir(parents=True, exist_ok=True)
    img = cv2.resize(load_rgb(src / "messi5.jpg"), (600, 400))
    stored = np.ascontiguousarray(np.rot90(img, 1))  # undone by orientation 6 (rotate CW)
    save_jpeg(stored, out / "orient6.jpg", exif_bytes(BASE_TIME.timestamp(), None, orientation=6))


CORE_EPOCH = 978307200  # Lightroom's touchTime counts seconds from 2001-01-01 UTC


def build_lrcat(path: Path, entries: list[dict], *, wal: bool = False, pad_mb: int = 0) -> None:
    """Write a minimal Lightroom-style catalog.

    Each entry: {root, rel, base, ext, rating, pick, label}; root is an absolute folder path.
    Optional per-entry keys add Lightroom columns: `touch` (Unix seconds -> Adobe_images.touchTime),
    `capture` ("2024-06-01T14:00:00" -> captureTime), `original` (originalFilename), `sidecar`
    (sidecarExtensions). `wal=True` leaves the rows un-checkpointed in `<path>-wal` (a catalog that
    Lightroom still has open: the base file alone holds nothing); `pad_mb` adds a blob table so
    the file is that many MB (large-catalog timing).
    """
    path = Path(path)
    for suffix in ("", "-wal", "-shm", "-journal"):
        path.with_name(path.name + suffix).unlink(missing_ok=True)
    stage = path.with_name("stage-" + path.name) if wal else path
    con = sqlite3.connect(stage)
    if wal:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA wal_autocheckpoint=0")
    opt = {k for e in entries for k in ("touch", "capture", "original", "sidecar") if k in e}
    con.executescript(
        f"""
        CREATE TABLE AgLibraryRootFolder (
            id_local INTEGER PRIMARY KEY, absolutePath TEXT, name TEXT);
        CREATE TABLE AgLibraryFolder (
            id_local INTEGER PRIMARY KEY, rootFolder INTEGER, pathFromRoot TEXT);
        CREATE TABLE AgLibraryFile (
            id_local INTEGER PRIMARY KEY, folder INTEGER, baseName TEXT, extension TEXT
            {", originalFilename TEXT" if "original" in opt else ""}
            {", sidecarExtensions TEXT" if "sidecar" in opt else ""});
        CREATE TABLE Adobe_images (id_local INTEGER PRIMARY KEY, rootFile INTEGER, pick REAL,
                                   rating REAL, colorLabels TEXT, masterImage INTEGER
            {", touchTime REAL" if "touch" in opt else ""}
            {", captureTime TEXT" if "capture" in opt else ""});
        """
    )
    roots: dict[str, int] = {}
    folders: dict[tuple[int, str], int] = {}
    files, images = [], []
    for fid, e in enumerate(entries, 1):
        root = e["root"].rstrip("/") + "/"
        if root not in roots:
            roots[root] = len(roots) + 1
            con.execute(
                "INSERT INTO AgLibraryRootFolder VALUES (?,?,?)",
                (roots[root], root, Path(root).name),
            )
        key = (roots[root], e.get("rel", ""))
        if key not in folders:
            folders[key] = len(folders) + 1
            con.execute(
                "INSERT INTO AgLibraryFolder VALUES (?,?,?)", (folders[key], key[0], key[1])
            )
        frow = [fid, folders[key], e["base"], e["ext"]]
        if "original" in opt:
            frow.append(e.get("original"))
        if "sidecar" in opt:
            frow.append(e.get("sidecar"))
        files.append(frow)
        irow = [fid, fid, e.get("pick", 0), e.get("rating"), e.get("label", ""), None]
        if "touch" in opt:
            irow.append(None if e.get("touch") is None else e["touch"] - CORE_EPOCH)
        if "capture" in opt:
            irow.append(e.get("capture"))
        images.append(irow)
    if files:
        con.executemany(
            f"INSERT INTO AgLibraryFile VALUES ({','.join('?' * len(files[0]))})", files
        )
        con.executemany(
            f"INSERT INTO Adobe_images VALUES ({','.join('?' * len(images[0]))})", images
        )
    if pad_mb:
        con.execute("CREATE TABLE pad (b BLOB)")
        con.executemany("INSERT INTO pad VALUES (zeroblob(1000000))", [()] * pad_mb)
    con.commit()
    if wal:  # copy the live files while the connection is open: the WAL stays un-checkpointed
        for suffix in ("", "-wal", "-shm"):
            src = stage.with_name(stage.name + suffix)
            if src.exists():
                shutil.copyfile(src, path.with_name(path.name + suffix))
    con.close()
    if wal:
        for suffix in ("", "-wal", "-shm"):
            stage.with_name(stage.name + suffix).unlink(missing_ok=True)


def lr_edit(
    cat: Path, path: str | Path, *, pick: int | None = None, rating: int | None = None,
    label: str | None = None, touch: float | None = None, clear_rating: bool = False,
) -> None:  # fmt: skip
    """Change one image's state the way Lightroom would (always on a test's own copy) and bump
    `touchTime` (to `touch` Unix seconds, else now). `path` is the image's full path."""
    p = Path(path)
    con = sqlite3.connect(cat)
    try:
        row = con.execute(
            "SELECT i.id_local FROM Adobe_images i JOIN AgLibraryFile f ON f.id_local = i.rootFile "
            "JOIN AgLibraryFolder d ON d.id_local = f.folder "
            "JOIN AgLibraryRootFolder r ON r.id_local = d.rootFolder "
            "WHERE r.absolutePath || d.pathFromRoot || f.baseName || '.' || f.extension = ?",
            (str(p),),
        ).fetchone()
        assert row, f"{p} is not in {cat}"
        sets, vals = [], []
        if pick is not None:
            sets.append("pick=?")
            vals.append(pick)
        if rating is not None or clear_rating:
            sets.append("rating=?")
            vals.append(rating)
        if label is not None:
            sets.append("colorLabels=?")
            vals.append(label)
        cols = {r[1] for r in con.execute("PRAGMA table_info(Adobe_images)")}
        if "touchTime" in cols:
            sets.append("touchTime=?")
            vals.append((touch if touch is not None else time.time()) - CORE_EPOCH)
        con.execute(f"UPDATE Adobe_images SET {', '.join(sets)} WHERE id_local=?", (*vals, row[0]))
        con.commit()
    finally:
        con.close()


# ---- synthetic maker-note containers (no CC0 sample ships a *labelled* AF layout we control) ----
def _ifd(entries: list[tuple[int, int, bytes, int]], start: int, e: str = "<") -> bytes:
    """IFD at absolute offset `start` (relative to the TIFF header). entries = (tag, type,
    payload bytes, count); payloads > 4 bytes are appended after the table."""
    n = len(entries)
    table = 2 + 12 * n + 4
    out, data = bytearray(struct.pack(e + "H", n)), bytearray()
    for tag, typ, payload, cnt in sorted(entries):
        if len(payload) <= 4:
            val = payload.ljust(4, b"\0")
        else:
            val = struct.pack(e + "I", start + table + len(data))
            data += payload + (b"\0" if len(payload) % 2 else b"")
        out += struct.pack(e + "HHI", tag, typ, cnt) + val
    return bytes(out) + struct.pack(e + "I", 0) + bytes(data)


def canon_afinfo2(
    points: list[tuple[int, int, int, int]], img: tuple[int, int], in_focus: list[int],
    selected: list[int] | None = None, e: str = "<",
) -> bytes:
    """AFInfo2 payload; points = (w, h, x, y) with centre-origin, Y-up coordinates."""
    n = len(points)
    words = (n + 15) // 16

    def mask(idx: list[int]) -> list[int]:
        m = [0] * words
        for i in idx:
            m[i // 16] |= 1 << (i % 16)
        return [v - 65536 if v > 32767 else v for v in m]

    vals = [0, 1, n, len(in_focus), img[0], img[1], img[0], img[1]]
    vals += [p[0] for p in points] + [p[1] for p in points]
    vals += [p[2] for p in points] + [p[3] for p in points]
    vals += mask(in_focus) + mask(selected or [])
    vals[0] = len(vals) * 2
    return struct.pack(f"{e}{len(vals)}h", *vals)


def sony_focus_location(w: int, h: int, x: int, y: int, e: str = "<") -> bytes:
    return struct.pack(e + "4h", w, h, x, y)


def tiff_with_makernote(
    make: bytes, mn_entries: list[tuple[int, int, bytes, int]], *, sony_header: bool = False,
    orientation: int = 1, e: str = "<",
) -> bytes:
    """TIFF (header, IFD0 {Make, Orientation, ExifIFD}, ExifIFD {MakerNote})."""
    hdr = (b"II" if e == "<" else b"MM") + struct.pack(e + "HI", 42, 8)

    def ifd0(exif_off: int) -> bytes:
        return _ifd(
            [(0x010F, 2, make + b"\0", len(make) + 1),
             (0x0112, 3, struct.pack(e + "H", orientation), 1),
             (0x8769, 4, struct.pack(e + "I", exif_off), 1)], 8, e)  # fmt: skip

    exif_start = 8 + len(ifd0(0))
    mn_start = exif_start + 2 + 12 + 4  # the single MakerNote entry's data follows the table
    magic = b"SONY DSC \0\0\0" if sony_header else b""
    mn = magic + _ifd(mn_entries, mn_start + len(magic), e)
    exif = _ifd([(0x927C, 7, mn, len(mn))], exif_start, e)
    return hdr + ifd0(exif_start) + exif


def jpeg_with_tiff(tiff: bytes, rgb: np.ndarray, quality: int = 92) -> bytes:
    """JPEG bytes of `rgb` whose Exif APP1 holds `tiff`."""
    import io

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=quality, exif=b"Exif\x00\x00" + tiff)
    return buf.getvalue()


def cr3_like(tiff: bytes) -> bytes:
    """ftyp + moov{CMT3} skeleton: enough for the CMT3 box scan."""
    ftyp = struct.pack(">I4s", 24, b"ftyp") + b"crx \0\0\0\1crx isom"
    cmt3 = struct.pack(">I4s", 8 + len(tiff), b"CMT3") + tiff
    return ftyp + struct.pack(">I4s", 8 + len(cmt3), b"moov") + cmt3


def fetch_search_images() -> Path:
    """Download the search subjects (+ scikit-learn's bundled china/flower, plus the portrait,
    football and building fixtures) into generated/search and return the folder."""
    import shutil

    dest = GEN / "search"
    for name, (urls, mn) in SEARCH_IMAGES.items():
        download(name, urls, mn, dest)
    from sklearn import datasets

    sk = Path(datasets.__file__).parent / "images"
    for name in ("china.jpg", "flower.jpg"):
        if not (dest / name).exists():
            shutil.copy(sk / name, dest / name)
    src = GEN / "src"
    for name in ("building.jpg", "messi5.jpg", "lena.jpg"):
        if not (dest / name).exists():
            download(name, IMAGES[name][0], IMAGES[name][1], src)
            shutil.copy(src / name, dest / name)
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="rebuild synthesized files")
    ap.add_argument("--skip-raw", action="store_true")
    args = ap.parse_args(argv)

    src = GEN / "src"
    for name, (urls, mn) in IMAGES.items():
        download(name, urls, mn, src)
    group = src / "group_rgb.jpg"
    if not group.exists():
        g = cv2.imread(str(src / "group.png"))
        cv2.imwrite(str(group), g, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not args.skip_raw:
        for name, (urls, mn) in RAWS.items():
            download(name, urls, mn, GEN / "raw")

    shoot = GEN / "shoot"
    manifest_path = GEN / "shoot_manifest.json"
    if args.force or not manifest_path.exists() or not (shoot / "black.jpg").exists():
        manifest = build_shoot(src, shoot)
        manifest_path.write_text(json.dumps(manifest, indent=2))
    manifest = json.loads(manifest_path.read_text())
    if args.force or not (GEN / "misc" / "orient6.jpg").exists():
        build_oriented(src, GEN / "misc")

    cat = GEN / "catalog.lrcat"
    if args.force or not cat.exists():
        entries = []
        for m in manifest:
            base = m["file"].rsplit(".", 1)[0]
            good = m["variant"] == "normal" and m["frame"] == 0
            bad = m["variant"] in {"blur", "underexposed", "black"}
            entries.append(
                {
                    "root": str(GEN),
                    "rel": "shoot/",
                    "base": base,
                    "ext": "jpg",
                    "pick": 1 if good else -1 if bad else 0,
                    "rating": 4 if good else None,
                    "label": "Green" if good else "",
                }
            )
        build_lrcat(cat, entries)
    print(f"fixtures ready in {GEN}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
