"""Cheap model-free synthetic shoots of thousands of frames (T21).

Writes tiny JPEG files (so scan / stat work on real files) and fills `cull.db` directly with
synthetic features and scene embeddings (unit vectors = moment centroid + small noise, so
`group_duplicates` forms realistic groups). A `run_cull` on such a folder is warm and needs no
models; files that are added afterwards go through a monkeypatched `runner._process_uncached`
(`fake_process`) that serves synthetic jobs and records the paths it was handed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from shotsieve.config import Settings
from shotsieve.features.record import RawFeatures
from shotsieve.io.exif import ImageMeta
from shotsieve.pipeline import runner
from shotsieve.pipeline.cache import Cache, CachedImage

T0 = 1_717_250_000.0  # 2024-06-01
DIM = 384


@dataclass
class Frame:
    name: str
    serial: str
    ts: float
    moment: int
    raw: RawFeatures
    scene: np.ndarray = field(repr=False)


def _raw(rng: np.random.Generator, lap: float, kind: str) -> RawFeatures:
    r = RawFeatures(width=1280, height=853)
    r.lap_var = lap
    r.tenengrad = lap * 5.0
    r.mean_luma, r.lin_mean, r.clip_hi, r.clip_lo = 112.0, 0.18, 0.002, 0.002
    r.ev, r.p98 = 0.0, 220.0
    if kind == "dark":
        r.mean_luma, r.lin_mean, r.clip_lo, r.ev, r.p98 = 22.0, 0.012, 0.55, -3.4, 60.0
    elif kind == "blur":
        r.lap_var, r.tenengrad = 6.0, 40.0
    return r


def gen_frames(
    n: int,
    *,
    seed: int = 1,
    serial: str = "A",
    t0: float = T0,
    moment0: int = 0,
    prefix: str = "IMG",
    start: int = 0,
    sharp: float = 1.0,
    gap: tuple[float, float] = (5.0, 90.0),
    span: float | None = None,
) -> list[Frame]:
    """`n` frames in bursts of 1..6 (0.25 s apart) separated by `gap` seconds on body `serial`.
    With `span`, moment start times are spread evenly over that many seconds instead."""
    rng = np.random.default_rng(seed)
    out: list[Frame] = []
    t, moment, i = t0, moment0, 0
    n_moments = max(1, n // 3)
    while i < n:
        burst = min(int(rng.integers(1, 7)) if rng.random() < 0.7 else 1, n - i)
        centre = rng.normal(size=DIM)
        centre /= np.linalg.norm(centre)
        base = float(rng.lognormal(np.log(95.0 * sharp), 0.55))
        for k in range(burst):
            v = centre + rng.normal(scale=0.01, size=DIM)
            v = (v / np.linalg.norm(v)).astype(np.float32)
            roll = rng.random()
            kind = "blur" if roll < 0.04 else "dark" if roll < 0.07 else "ok"
            lap = base * float(rng.uniform(0.85, 1.15))
            out.append(
                Frame(f"{prefix}_{start + i + k:05d}.jpg", serial, t + 0.25 * k, moment,
                      _raw(rng, lap, kind), v)
            )
        i += burst
        moment += 1
        if span is not None:
            t += span / n_moments
        else:
            t += float(rng.uniform(*gap))
    return out


def write_files(folder: Path, frames: list[Frame]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (43, 64, 3), dtype=np.uint8)
    for f in frames:
        p = folder / f.name
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 60])
        img = np.roll(img, 1, axis=1)  # distinct bytes (and sizes) without per-file RNG cost


def meta_of(f: Frame) -> ImageMeta:
    return ImageMeta(
        timestamp=f.ts, ts_source="exif", camera_serial=f.serial, make="Synth", model="S1",
        flash_fired=False, iso=400, exposure_time=1 / 250, fnumber=2.8, focal_length=50.0,
        width=1280, height=853,
    )


def record_of(f: Frame) -> CachedImage:
    scene = f.scene / max(float(np.linalg.norm(f.scene)), 1e-12)
    scene = scene.astype(np.float16).astype(np.float32)  # the canonical (cache) form
    return CachedImage(meta_of(f), f.raw, scene, np.zeros((0, 128), np.float32))


def populate(folder: Path, frames: list[Frame], settings: Settings) -> None:
    """Put every frame into <folder>/.shotsieve/cull.db with the file's real (size, mtime_ns)."""
    ext = runner.extractor_fingerprint(settings)
    with Cache.for_folder(folder) as c:
        for f in frames:
            st = (folder / f.name).stat()
            c.put(str(folder / f.name), st.st_size, st.st_mtime_ns, record_of(f), "", ext)
        c.commit()


def build(folder: Path, n: int, settings: Settings | None = None, **kw) -> list[Frame]:
    """Files + cache for a model-free `n`-frame shoot (two bodies: A for ~85 % of the moments)."""
    settings = settings or Settings()
    a = gen_frames(n - n // 7, serial="A", **kw)
    b = gen_frames(n // 7, serial="B", seed=kw.get("seed", 1) + 100, prefix="IMB",
                   t0=T0 + 7 * 86400.0)  # a second body, shooting other moments
    frames = a + b
    write_files(folder, frames)
    populate(folder, frames, settings)
    return frames


class FakeProcess:
    """Stand-in for `runner._process_uncached`: serves synthetic jobs, records the paths."""

    def __init__(self, frames: list[Frame]) -> None:
        self.by_name = {f.name: f for f in frames}
        self.paths: list[str] = []

    def __call__(self, jobs, settings, prog, n_total, n_cached):
        for j in jobs:
            self.paths.append(str(j.entry.master))
            f = self.by_name[j.entry.master.name]
            j.meta, j.raw = meta_of(f), f.raw
            j.scene = f.scene
            j.face_emb = np.zeros((0, 128), np.float32)
            j.error = None
        return list(jobs)
