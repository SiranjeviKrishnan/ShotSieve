"""Video frame-grab culling (T12, opt-in): candidate frames from a video, reusing the photo stack.

1. coarse pass at `Settings.video_fps` (default 2): Stage A + faces per sampled frame;
2. scene cuts (HSV histogram chi-square > CUT_CHI2 between consecutive samples) split segments;
3. per segment the rule prior (same `quality.assess` as photos) is the frame score; greedy NMS
   over a 1.0 s window keeps the top `max(1, 6/min)` peaks, each refined +-0.25 s at 10 fps to
   the sharpest frame (the dense pass);
4. picks are persisted (raw features + scene embedding, float16 round-trip like photos), so a
   cached re-run decides identically; decisions are made by the photo `selection.decide`, one
   group per (video, segment).
"""

from __future__ import annotations

import base64
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from shotsieve.config import Settings
from shotsieve.features.record import RawFeatures, ShootNorm, build_vector, compute_norm
from shotsieve.io.formats import split_virtual, virtual_path
from shotsieve.io.video import VideoReader, creation_time
from shotsieve.log import get_logger
from shotsieve.pipeline import stages
from shotsieve.scoring import quality

log = get_logger("video")

CUT_CHI2 = 0.5
NMS_S = 1.0
REFINE_S = 0.25
REFINE_FPS = 10.0
PICKS_PER_MIN = 6.0
EXPORT_QUALITY = 95


@dataclass
class Pick:
    video: Path
    segment: int
    t: float
    raw: RawFeatures
    scene: np.ndarray  # (384,) float32, unit norm, float16 round-tripped (== photo cache form)
    norm: ShootNorm  # sharpness references of the whole video
    n_in_segment: int

    @property
    def path(self) -> str:
        return virtual_path(self.video, self.t)


@dataclass
class _Sample:
    t: float
    raw: RawFeatures
    hist: np.ndarray


def _hist(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    s = 160 / max(h, w)
    size = (max(1, round(w * s)), max(1, round(h * s)))
    small = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
    return (hist / max(float(hist.sum()), 1.0)).astype(np.float32)


def chi2(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric chi-square distance of two normalised histograms, in [0, 1]."""
    return float(0.5 * np.sum((a - b) ** 2 / (a + b + 1e-9)))


def _flags(raw: RawFeatures, settings: Settings) -> list[str]:
    from shotsieve.pipeline.runner import _eye_flags

    return _eye_flags(raw, settings)


def _analyse(rgb: np.ndarray, settings: Settings) -> RawFeatures:
    raw = stages.stage_a(rgb)
    if stages.is_near_black(raw):
        raw.faces_skipped = True
    else:
        try:
            raw.faces, _ = stages.stage_faces(rgb, raw, settings)
        except Exception as exc:  # noqa: BLE001 - a bad frame must not stop the video
            log.warning("face stage failed on a video frame: %s", exc)
    return raw


def segment_samples(samples: list[_Sample]) -> list[list[int]]:
    """Indices of consecutive samples, split where the histogram jumps (a scene cut)."""
    segs: list[list[int]] = []
    for i, s in enumerate(samples):
        if i == 0 or chi2(samples[i - 1].hist, s.hist) > CUT_CHI2:
            segs.append([])
        segs[-1].append(i)
    return segs


def nms_peaks(ts: list[float], scores: list[float], k: int) -> list[int]:
    """Greedy non-maximum suppression: best score first, nothing else within NMS_S of a pick."""
    order = sorted(range(len(ts)), key=lambda i: (-scores[i], ts[i]))
    keep: list[int] = []
    for i in order:
        if all(abs(ts[i] - ts[j]) >= NMS_S for j in keep):
            keep.append(i)
            if len(keep) >= k:
                break
    return sorted(keep, key=lambda i: ts[i])


def _canon_scene(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.float32)
    v = v / max(float(np.linalg.norm(v)), 1e-12)
    return v.astype(np.float16).astype(np.float32)


def scan_video(
    path: Path, settings: Settings, tick: Callable[[], None] | None = None
) -> tuple[list[dict], dict]:
    """Candidate frames of one video as JSON-able dicts (the persisted form) plus stats."""
    from shotsieve.models.image_embed import get_image_embedder

    t0 = time.perf_counter()
    samples: list[_Sample] = []
    with VideoReader(path, settings.preview_max_edge) as vr:
        info = vr.info
        for t, rgb in vr.samples(1.0 / settings.video_fps):
            samples.append(_Sample(t, _analyse(rgb, settings), _hist(rgb)))
            if tick:
                tick()
        if not samples:
            return [], {"frames": 0, "seconds": info.duration, "wall_s": time.perf_counter() - t0}
        norm = compute_norm([s.raw for s in samples])
        scores = []
        for s in samples:
            fv = build_vector(
                s.raw, norm, eyes_thresh=settings.eyes_closed_thresh,
                eyes_both_thresh=settings.eyes_closed_both_thresh,
            )  # fmt: skip
            scores.append(quality.assess(fv, settings, _flags(s.raw, settings)).score)
        step = 1.0 / settings.video_fps
        embedder = get_image_embedder(
            settings.ort_threads, settings.embed_batch, settings.use_coreml
        )
        out: list[dict] = []
        for seg_no, idx in enumerate(segment_samples(samples)):
            ts = [samples[i].t for i in idx]
            dur = ts[-1] - ts[0] + step
            k = max(1, round(PICKS_PER_MIN * dur / 60.0))
            peaks = nms_peaks(ts, [scores[i] for i in idx], k)
            for pk in peaks:
                t_peak = ts[pk]
                best_t, best_rgb, best_lap = t_peak, None, -1.0
                n_ref = int(round(REFINE_S * REFINE_FPS))
                grid = [t_peak + j / REFINE_FPS for j in range(-n_ref, n_ref + 1)]
                for tt in grid:
                    if tt < ts[0] - 1e-9 or tt > ts[-1] + 1e-9:
                        continue  # stay inside the segment: a cut lies just outside its samples
                    rgb = vr.frame_at(tt)
                    if rgb is None:
                        continue
                    lap = stages.stage_a(rgb).lap_var
                    if lap == lap and lap > best_lap:
                        best_t, best_rgb, best_lap = tt, rgb, lap
                if best_rgb is None:
                    continue
                raw = _analyse(best_rgb, settings)
                scene = _canon_scene(embedder.embed_batch([best_rgb])[0])
                out.append({
                    "segment": seg_no, "t": round(best_t, 3), "raw": raw.to_json(),
                    "scene": base64.b64encode(scene.astype(np.float16).tobytes()).decode(),
                })  # fmt: skip
                del best_rgb
    n_in: dict[int, int] = {}
    for d in out:
        n_in[d["segment"]] = n_in.get(d["segment"], 0) + 1
    for d in out:
        d["n_in_segment"] = n_in[d["segment"]]
        d["norm"] = [norm.lap_ref, norm.tenen_ref, norm.face_ref, norm.af_ref]
    wall = time.perf_counter() - t0
    stats = {
        "frames": len(samples), "src_frames": info.n_frames, "seconds": info.duration,
        "wall_s": wall,
        "segments": len(n_in), "backend": vr.backend,
    }  # fmt: skip
    return out, stats


def pick_from_dict(video: Path, d: dict) -> Pick:
    scene = np.frombuffer(base64.b64decode(d["scene"]), np.float16).astype(np.float32)
    return Pick(
        video, int(d["segment"]), float(d["t"]), RawFeatures.from_json(d["raw"]), scene,
        ShootNorm(*d["norm"]), int(d["n_in_segment"]),
    )  # fmt: skip


def cached_or_scan(
    cache, video: Path, stat: tuple[int, int], extractor: str, settings: Settings,
    tick: Callable[[], None] | None = None,
) -> tuple[list[Pick], dict | None]:
    """Picks of `video` from the cache (same file + extractor + fps) or by scanning it. The
    cold path goes through the same serialised form as the warm one."""
    key = f"video:{cache._rel(str(video))}"  # noqa: SLF001
    ident = [stat[0], stat[1], extractor, settings.video_fps]
    hit = cache.get_meta(key)
    if hit and hit.get("id") == ident:
        return [pick_from_dict(video, d) for d in hit["picks"]], None
    try:
        picks, stats = scan_video(video, settings, tick)
    except Exception as exc:  # noqa: BLE001 - one bad video must not stop the cull
        log.warning("video %s skipped: %s", video.name, exc)
        return [], {"error": f"{type(exc).__name__}: {exc}"}
    cache.set_meta(key, {"id": ident, "picks": picks})
    return [pick_from_dict(video, d) for d in picks], stats


# ---- export -----------------------------------------------------------------------------


def grab_name(video: Path, t: float) -> str:
    return f"{video.stem}_t{round(t * 1000):06d}.jpg"


def export_frame(folder: Path, virtual: str | Path) -> Path:
    """Full-resolution frame of a virtual `clip.mp4#t=12.345` -> <folder>/frame_grabs/*.jpg
    (quality 95, EXIF DateTimeOriginal = video creation time + t). The video is never modified."""
    import piexif

    sv = split_virtual(virtual)
    if sv is None:
        raise ValueError("not a video frame")
    video, t = sv
    with VideoReader(video, 0) as vr:
        rgb = vr.frame_at(t, 0)
    if rgb is None:
        raise OSError(f"cannot decode frame at {t:.3f}s")
    dest = Path(folder) / "frame_grabs" / grab_name(video, t)
    dest.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(
        ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, EXPORT_QUALITY]
    )
    if not ok:
        raise OSError("jpeg encode failed")
    dest.write_bytes(buf.tobytes())
    ts = creation_time(video) + t
    dt = datetime.fromtimestamp(ts)
    stamp = dt.strftime("%Y:%m:%d %H:%M:%S").encode()
    exif = {"Exif": {
        piexif.ExifIFD.DateTimeOriginal: stamp,
        piexif.ExifIFD.DateTimeDigitized: stamp,
        piexif.ExifIFD.SubSecTimeOriginal: f"{dt.microsecond // 1000:03d}".encode(),
    }, "0th": {piexif.ImageIFD.DateTime: stamp}}  # fmt: skip
    piexif.insert(piexif.dump(exif), str(dest))
    return dest
