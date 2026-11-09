"""Subject-aware crop suggestions (T22, docs/11_CROP.md): subject detection (+ cache), point
prompts, suggestions, versions, XMP writes and burst transfer.

Nothing here runs inside `cull`. Pixels are never modified: a crop is a row in cull.db
(`crops`, append-only) and optionally `crs:Crop*` in a RAW sidecar (`io.crs.write_crop`)."""

from __future__ import annotations

import json
import math
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from shotsieve.config import Settings
from shotsieve.features.subject import IMG_AREA_MIN, REL_AREA_MIN
from shotsieve.io import crs
from shotsieve.io import xmp as _xmp
from shotsieve.io.formats import RAW_EXTS, preview_source_for, split_virtual
from shotsieve.io.preview import PreviewError, load_preview
from shotsieve.log import get_logger
from shotsieve.models import object_detect, patch_features
from shotsieve.pipeline import crop_geom as g
from shotsieve.pipeline import crop_score as cs
from shotsieve.pipeline.cache import Cache, fingerprint
from shotsieve.pipeline.crop_geom import Rect
from shotsieve.pipeline.crop_score import Subject
from shotsieve.pipeline.thumbs import thumb_path, write_thumb

log = get_logger("crop")

CROP_DET_VERSION = "2"  # 2: subject extension + halo (Review #12 B1)
PREVIEW_EDGE = 1600
FACE_SCORE = 0.6
OBJ_SCORE = 0.45
OBJ_MIN_AREA = 0.003
AUTO_OBJ_SCORE = 0.5
AUTO_OBJ_AREA = 0.01
LINK_CONTAIN = 0.90
LINK_TOP = 0.25
BURST_IOU = 0.30
BURST_SIM = 0.60
SOURCES = ("ai", "ai_adjusted", "manual", "reset", "restore", "lightroom", "burst")
PRESET_RATIOS = (
    "original",
    "1:1",
    "4:5",
    "5:4",
    "2:3",
    "3:2",
    "5:7",
    "7:5",
    "16:9",
    "9:16",
    "free",
)
LIGHTROOM_CROP_NOTE = (
    "Lightroom Classic ignores XMP sidecars for JPEG/TIFF/PNG/HEIC and DNG files: the crop is "
    "saved in ShotSieve only (use 'embed copies' to write a copy with the crop)."
)


ROTATED_NOTE = (
    "This photo was rotated in Lightroom: the crop is kept in ShotSieve only (writing it would "
    "frame a different region). Reset the rotation in Lightroom to write crops to the sidecar."
)


class CropError(ValueError):
    """A request the crop code refuses; `status` is the HTTP status the API maps it to."""

    def __init__(self, msg: str, status: int = 422) -> None:
        super().__init__(msg)
        self.status = status


# ---- image facts ---------------------------------------------------------------------------


@dataclass
class ImageInfo:
    path: str
    fw: int  # full-resolution DISPLAY width (orientation applied)
    fh: int
    orientation: int  # EXIF orientation (or the sidecar's tiff:Orientation)
    kind: str  # raw | dng | raster


def _kind(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".dng":
        return "dng"
    return "raw" if ext in RAW_EXTS else "raster"


_RAW_FLIP_TO_EXIF = {0: 1, 3: 3, 5: 8, 6: 6}


def read_image_facts(path: str) -> tuple[int, int, int]:
    """(display width, display height, EXIF orientation) of the master at full resolution:
    RAW from rawpy's visible area, rasters from the header (swapped for orientations 5-8)."""
    p = Path(path)
    orientation, sw, sh = 1, 0, 0
    if p.suffix.lower() in RAW_EXTS:
        import rawpy

        try:
            with rawpy.imread(str(p)) as raw:
                sw, sh = int(raw.sizes.width), int(raw.sizes.height)
                orientation = _RAW_FLIP_TO_EXIF.get(int(raw.sizes.flip), 1)
        except Exception as exc:  # noqa: BLE001
            raise CropError(f"cannot read the RAW file: {exc}", 422) from exc
        # the embedded preview's orientation (when present) is what Lightroom shows
        try:
            from shotsieve.io.exif import read_meta

            meta = read_meta(p)
            if meta.orientation in range(1, 9):
                orientation = meta.orientation
        except Exception:  # noqa: BLE001
            pass
    else:
        try:
            from PIL import Image

            with Image.open(p) as im:
                sw, sh = im.size
                orientation = int(im.getexif().get(0x0112, 1) or 1)
        except Exception as exc:  # noqa: BLE001
            raise CropError(f"cannot read the image header: {exc}", 422) from exc
    if orientation not in range(1, 9):
        orientation = 1
    fw, fh = g.display_dims(sw, sh, orientation)
    return fw, fh, orientation


def effective_orientation(path: str, exif_orientation: int) -> int:
    """A `tiff:Orientation` in the sidecar (a Lightroom rotation) wins over the EXIF value."""
    try:
        sp = _xmp.sidecar_path(path)
        if sp.exists():
            cur = crs.read_develop_xmp(sp)
            if cur is not None and cur.orientation in range(1, 9):
                return int(cur.orientation)  # type: ignore[arg-type]
    except OSError:
        pass
    return exif_orientation


# ---- preview -------------------------------------------------------------------------------


def preview_rgb(folder: str | Path, image_id: int, master: str) -> np.ndarray:
    """The cached 1600 px preview (the file `/api/preview/{id}` serves), orientation applied."""
    if split_virtual(master) is not None:
        raise CropError("not for video frames", 422)
    source = preview_source_for(master)
    try:
        mtime = source.stat().st_mtime_ns
    except OSError as exc:
        raise CropError("image file is missing", 404) from exc
    dest = thumb_path(folder, image_id, mtime, PREVIEW_EDGE, "p")
    if not dest.exists():
        try:
            rgb = load_preview(source, PREVIEW_EDGE)
        except (PreviewError, OSError, ValueError) as exc:
            raise CropError(f"cannot decode image: {exc}", 422) from exc
        write_thumb(rgb, dest, PREVIEW_EDGE, 88)
        return rgb
    bgr = cv2.imread(str(dest), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if bgr is None:
        raise CropError("cannot decode the cached preview", 422)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ---- detection -----------------------------------------------------------------------------


@dataclass
class SubjectSet:
    subjects: list[Subject]
    saliency: np.ndarray | None
    detector: str | None
    fw: int
    fh: int
    orientation: int
    horizon: tuple[float, float] | None = None
    peak: tuple[float, float, float, float] | None = None
    preview_wh: tuple[int, int] = (0, 0)
    det_version: str = ""

    def by_sid(self) -> dict[str, Subject]:
        return {s.sid: s for s in self.subjects}


_face_lock = threading.Lock()
_faces: Any = None
_tokens = patch_features.TokenLRU(8)


def _face_detector() -> Any:
    global _faces
    with _face_lock:
        if _faces is None:
            from shotsieve.models.face_detect import FaceDetector

            _faces = FaceDetector(score_threshold=FACE_SCORE)
        return _faces


def det_version() -> str:
    return f"{CROP_DET_VERSION}:{'yolox' if object_detect.available() else 'none'}"


def detect_horizon(rgb: np.ndarray) -> tuple[float, float] | None:
    """(y of the dominant near-horizontal line, tilt in degrees), or None. Canny + HoughLinesP on a
    640 px copy; the line must span >= 50 % of the width and be within 5 degrees of level."""
    h, w = rgb.shape[:2]
    s = 640 / max(h, w)
    small = cv2.resize(
        rgb, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA
    )
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    sh, sw = gray.shape
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        1,
        math.pi / 360,
        threshold=60,
        minLineLength=int(0.5 * sw),
        maxLineGap=int(0.03 * sw),
    )
    if lines is None:
        return None
    best, best_len = None, 0.0
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if abs(ang) >= 5.0:
            continue
        ln = math.hypot(x2 - x1, y2 - y1)
        if ln > best_len:
            best, best_len = (((y1 + y2) / 2) / sh, ang), ln
    return best


def _link_faces(subjects: list[Subject]) -> None:
    people = [s for s in subjects if s.kind == "person"]
    for f in (s for s in subjects if s.kind == "face"):
        fb = f.box
        for p in sorted(people, key=lambda s: s.area):
            pb = p.box
            iw = min(fb[2], pb[2]) - max(fb[0], pb[0])
            ih = min(fb[3], pb[3]) - max(fb[1], pb[1])
            if iw <= 0 or ih <= 0 or p.linked:
                continue
            if iw * ih / max(f.area, 1e-12) < LINK_CONTAIN:
                continue
            if abs(fb[1] - pb[1]) <= LINK_TOP * (pb[3] - pb[1]):
                f.linked, p.linked = p.sid, f.sid
                break


def _norm_box(
    b: tuple[float, float, float, float], w: int, h: int
) -> tuple[float, float, float, float]:
    return (
        max(0.0, b[0] / w), max(0.0, b[1] / h), min(1.0, b[2] / w), min(1.0, b[3] / h),
    )  # fmt: skip


def _detect(
    rgb: np.ndarray, settings: Settings
) -> tuple[list[Subject], str | None, np.ndarray | None, Any]:
    ph, pw = rgb.shape[:2]
    subjects: list[Subject] = []
    for f in _face_detector().detect(rgb):
        box = _norm_box(
            (f.bbox[0], f.bbox[1], f.bbox[0] + f.bbox[2], f.bbox[1] + f.bbox[3]), pw, ph
        )
        lm = tuple((float(x) / pw, float(y) / ph) for x, y in f.landmarks)
        subjects.append(Subject("", "face", box, float(f.score), "yunet", "face", lm))
    detector = None
    if object_detect.available():
        try:
            with object_detect.holder(
                settings.crop_threads, settings.crop_idle_unload_s
            ).use() as det:
                dets = det.detect(rgb)
            detector = "yolox_s"
            for d in dets:
                if d.score < OBJ_SCORE:
                    continue
                box = _norm_box(d.box, pw, ph)
                area = (box[2] - box[0]) * (box[3] - box[1])
                if area < OBJ_MIN_AREA:
                    continue
                if d.name == "person":
                    kind, label = "person", "person"
                elif d.name in object_detect.ANIMALS:
                    kind, label = "animal", d.name
                else:
                    kind, label = "object", d.name
                subjects.append(Subject("", kind, box, d.score, "yolox", label))
        except Exception as exc:  # noqa: BLE001 - crop mode works without the detector
            log.warning("object detector failed: %s", exc)
            detector = None
    subjects.sort(key=lambda s: (s.kind, s.box[0], s.box[1]))
    counters: dict[str, int] = {}
    for s in subjects:
        k = {"face": "f", "person": "p", "animal": "a", "object": "o"}[s.kind]
        counters[k] = counters.get(k, 0) + 1
        s.sid = f"{k}{counters[k]}"
    _link_faces(subjects)
    sal, tokens = None, None
    if patch_features.available():
        try:
            with patch_features.holder(
                settings.crop_threads, settings.crop_idle_unload_s
            ).use() as ex:
                tokens = ex.tokens(rgb)
            sal = patch_features.saliency(tokens)
            _extend_subjects(subjects, tokens, sal)
        except Exception as exc:  # noqa: BLE001
            log.warning("patch features failed: %s", exc)
    return subjects, detector, sal, tokens


def _overlap_small(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    return iw * ih / max(min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])), 1e-12)


def _extend_subjects(subjects: list[Subject], tokens: np.ndarray, sal: np.ndarray) -> None:
    """B1: grown box + halo for persons (halo only), animals and objects. Patches inside another,
    mostly separate subject are never absorbed."""
    if sal.shape != tokens.shape[:2]:
        sal = cv2.resize(sal, (tokens.shape[1], tokens.shape[0]), interpolation=cv2.INTER_LINEAR)
    for s in subjects:
        if s.kind not in ("person", "animal", "object"):
            continue
        blockers = [
            o.box for o in subjects
            if o is not s and o.kind != "face" and _overlap_small(o.box, s.box) <= 0.3
        ]  # fmt: skip
        s.ext, s.halo = patch_features.subject_extent(
            tokens, sal, s.box, blockers, grow=s.kind != "person"
        )
        if s.kind == "person":
            s.ext = None


def _subject_json(s: Subject) -> dict[str, Any]:
    return {
        "sid": s.sid, "kind": s.kind, "box": [round(v, 5) for v in s.box],
        "score": round(float(s.score), 4), "source": s.source, "label": s.label,
        "landmarks": None
        if s.landmarks is None
        else [[round(x, 5), round(y, 5)] for x, y in s.landmarks],
        "linked": s.linked,
        "ext": None if s.ext is None else [round(v, 5) for v in s.ext],
        "halo": None if s.halo is None else list(s.halo),
    }  # fmt: skip


def _subject_from_json(d: dict[str, Any]) -> Subject:
    lm = d.get("landmarks")
    return Subject(
        d["sid"],
        d["kind"],
        tuple(d["box"]),
        float(d.get("score", 1.0)),
        d.get("source", ""),
        d.get("label", ""),
        None if not lm else tuple((float(x), float(y)) for x, y in lm),
        d.get("linked"),
        None,
        tuple(d["ext"]) if d.get("ext") else None,
        (int(d["halo"][0]), int(d["halo"][1]), str(d["halo"][2])) if d.get("halo") else None,
    )


def info_for(cache: Cache, path: str, data: dict[str, Any] | None = None) -> ImageInfo:
    if data is not None and "fw" in data:
        return ImageInfo(path, data["fw"], data["fh"], data["orientation"], _kind(path))
    fw, fh, o = read_image_facts(path)
    return ImageInfo(path, fw, fh, o, _kind(path))


def detect_subjects(
    folder: str | Path, cache: Cache, image_id: int, path: str, settings: Settings | None = None
) -> SubjectSet:
    """Subjects of one image in normalised display coordinates (cached per file fingerprint and
    `CROP_DET_VERSION`; works without the YOLOX model: faces + saliency + point prompts)."""
    settings = settings or Settings()
    if split_virtual(path) is not None:
        raise CropError("not for video frames", 422)
    try:
        fp = fingerprint(path)
    except OSError as exc:
        raise CropError("image file is missing", 404) from exc
    dv = det_version()
    hit = cache.get_subjects(path)
    if hit and hit["fp"] == fp and hit["det_version"] == dv:
        d = hit["data"]
        return SubjectSet(
            [_subject_from_json(s) for s in d["subjects"]], hit["saliency"], d.get("detector"),
            d["fw"], d["fh"], d["orientation"],
            tuple(d["horizon"]) if d.get("horizon") else None,
            tuple(d["peak"]) if d.get("peak") else None, tuple(d.get("preview_wh", (0, 0))), dv,
        )  # fmt: skip
    rgb = preview_rgb(folder, image_id, path)
    fw, fh, o = read_image_facts(path)
    subjects, detector, sal, tokens = _detect(rgb, settings)
    horizon = detect_horizon(rgb)
    peak = patch_features.peak_region(sal) if sal is not None else None
    if tokens is not None:
        _tokens.put(f"{path}|{fp}", tokens)
    data = {
        "subjects": [_subject_json(s) for s in subjects], "detector": detector, "fw": fw, "fh": fh,
        "orientation": o, "horizon": list(horizon) if horizon else None,
        "peak": list(peak) if peak else None, "preview_wh": [rgb.shape[1], rgb.shape[0]],
    }  # fmt: skip
    cache.put_subjects(path, fp, dv, data, sal)
    return SubjectSet(
        subjects, sal, detector, fw, fh, o, horizon, peak, (rgb.shape[1], rgb.shape[0]), dv
    )


def detect_many(
    folder: str | Path,
    cache: Cache,
    items: list[tuple[int, str]],
    settings: Settings | None = None,
    progress: Any = None,
    workers: int = 3,
) -> list[tuple[int, SubjectSet | None]]:
    """`detect_subjects` for many images with a few pipelined workers: the face, YOLOX and DINOv2
    sessions are separate, so different images overlap (about 1.8x the serial rate)."""
    from concurrent.futures import ThreadPoolExecutor

    settings = settings or Settings()
    done = 0
    out: list[tuple[int, SubjectSet | None]] = []

    def one(it: tuple[int, str]) -> tuple[int, SubjectSet | None]:
        try:
            return it[0], detect_subjects(folder, cache, it[0], it[1], settings)
        except CropError:
            return it[0], None

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for r in ex.map(one, items):
            out.append(r)
            done += 1
            if progress:
                progress("crop", done, len(items))
    return out


def _tokens_for(
    folder: str | Path, image_id: int, path: str, settings: Settings
) -> np.ndarray | None:
    try:
        fp = fingerprint(path)
    except OSError:
        return None
    key = f"{path}|{fp}"
    tok = _tokens.get(key)
    if tok is not None:
        return tok
    if not patch_features.available():
        return None
    rgb = preview_rgb(folder, image_id, path)
    with patch_features.holder(settings.crop_threads, settings.crop_idle_unload_s).use() as ex:
        tok = ex.tokens(rgb)
    _tokens.put(key, tok)
    return tok


def point_subject(
    folder: str | Path,
    cache: Cache,
    image_id: int,
    path: str,
    x: float,
    y: float,
    settings: Settings | None = None,
) -> Subject:
    """The object under the point (x, y) (normalised display coordinates) as a `point` subject.
    Backend-agnostic: a SAM backend would replace only `patch_features.point_region`."""
    settings = settings or Settings()
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        raise CropError("point must be inside [0, 1]")
    sset = detect_subjects(folder, cache, image_id, path, settings)
    tokens = _tokens_for(folder, image_id, path, settings)
    sid = f"pt:{x:.4f},{y:.4f}"
    if tokens is None or sset.saliency is None:
        s = math.sqrt(patch_features.FALLBACK_AREA)
        box = (max(0.0, x - s / 2), max(0.0, y - s / 2), min(1.0, x + s / 2), min(1.0, y + s / 2))
        return Subject(sid, "point", box, 1.0, "fallback", "point")
    sal = sset.saliency
    if sal.shape != tokens.shape[:2]:
        sal = cv2.resize(sal, (tokens.shape[1], tokens.shape[0]), interpolation=cv2.INTER_LINEAR)
    box, how = patch_features.point_region(tokens, sal, x, y)
    pt = Subject(sid, "point", box, 1.0, "dino" if how == "region" else "fallback", "point")
    try:
        pt.ext, pt.halo = patch_features.subject_extent(
            tokens, sal, box, [], grow=how == "region"
        )
    except Exception as exc:  # noqa: BLE001 - the plain region box still works
        log.warning("point extent failed: %s", exc)
    return pt


# ---- suggestions -----------------------------------------------------------------------------


def auto_selection(sset: SubjectSet) -> list[Subject]:
    """No selection = auto: important faces + persons/animals (score >= 0.5, area >= 1 %) + the
    saliency peak region when nothing else exists."""
    faces = [s for s in sset.subjects if s.kind == "face"]
    largest = max((f.area for f in faces), default=0.0)
    chosen: list[Subject] = []
    taken: set[str] = set()
    for p in sset.subjects:
        if p.kind in ("person", "animal") and p.score >= AUTO_OBJ_SCORE and p.area >= AUTO_OBJ_AREA:
            q = Subject(**{**p.__dict__, "select_mode": "person" if p.kind == "person" else None})
            chosen.append(q)
            if p.linked:
                taken.add(p.linked)
    for f in faces:
        if f.sid in taken:
            continue
        if f.area >= REL_AREA_MIN * largest and f.area >= IMG_AREA_MIN:
            chosen.append(f)
    if not chosen and sset.peak is not None:
        chosen.append(Subject("sal", "point", sset.peak, 1.0, "saliency", "salient"))
    return chosen


def _selection(
    sset: SubjectSet, sids: list[str], points: list[Subject], modes: dict[str, str]
) -> list[Subject]:
    by = sset.by_sid()
    out: list[Subject] = []
    for sid in sids:
        s = by.get(sid)
        if s is None:
            raise CropError(f"unknown subject {sid!r}")
        mode = modes.get(sid)
        if mode not in (None, "person", "face"):
            raise CropError("select_mode must be 'person' or 'face'")
        if s.kind == "face" and mode == "person" and s.linked:
            s = by[s.linked]  # a click on a linked face selects the person, face = must-keep core
        q = Subject(**{**s.__dict__})
        if q.kind == "person":
            q.select_mode = "face" if mode == "face" else "person"
        out.append(q)
    out.extend(points)
    seen: set[str] = set()
    return [s for s in out if not (s.sid in seen or seen.add(s.sid))]


MAX_USER_BOXES = 6


def check_boxes(boxes: list[Any] | None) -> list[tuple[float, float, float, float]]:
    """Validate user-drawn boxes (display-normalised): inside [0, 1], x0 < x1, y0 < y1, max 6."""
    out: list[tuple[float, float, float, float]] = []
    for b in boxes or []:
        try:
            x0, y0, x1, y1 = (float(v) for v in b)
        except (TypeError, ValueError):
            raise CropError("a box must be four numbers x0, y0, x1, y1", 422) from None
        if not all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in (x0, y0, x1, y1)):
            raise CropError("box coordinates must be inside [0, 1]", 422)
        if not (x0 < x1 and y0 < y1):
            raise CropError("a box needs x0 < x1 and y0 < y1", 422)
        out.append((x0, y0, x1, y1))
    if len(out) > MAX_USER_BOXES:
        raise CropError(f"at most {MAX_USER_BOXES} boxes", 422)
    return out


def user_box_subject(
    folder: str | Path,
    image_id: int,
    path: str,
    box: tuple[float, float, float, float],
    settings: Settings,
) -> Subject:
    """A user-drawn box as a must-keep subject (kind "user"). Like a point/selected subject it gets
    the patch halo (never grown: the user set the extent); coordinates are the display frame."""
    sid = "box:" + ",".join(f"{v:.4f}" for v in box)
    s = Subject(sid, "user", box, 1.0, "user", "your box")
    try:
        tokens = _tokens_for(folder, image_id, path, settings)
        if tokens is not None:
            s.ext, s.halo = patch_features.subject_extent(tokens, None, box, [], grow=False)
    except Exception as exc:  # noqa: BLE001 - the plain box still works
        log.warning("user box halo failed: %s", exc)
    return s


def resolve_ratio(ratio: str | g.Ratio, flip: bool, fw: int, fh: int) -> g.Ratio:
    try:
        r = g.parse_ratio(ratio)
    except ValueError as exc:
        raise CropError(str(exc)) from exc
    r = r.oriented(fw, fh)
    return r.flipped() if flip else r


def run_suggest(
    sset: SubjectSet,
    selected: list[Subject],
    ratio: g.Ratio,
    n: int,
    settings: Settings,
    extra: list[Subject] | None = None,
    same_ref: tuple[float, float] | None = None,
) -> cs.SuggestResult:
    scene = cs.Scene(
        selected,
        [*sset.subjects, *(extra or [])],
        sset.fw,
        sset.fh,
        sset.saliency,
        sset.horizon,
        same_ref,
    )
    return cs.suggest(scene, ratio, n, settings.crop_min_long_edge_px)


def _candidate_json(c: cs.Candidate, sset: SubjectSet) -> dict[str, Any]:
    return {
        "rect": [round(v, 6) for v in c.rect.as_tuple()], "rect_px": list(c.rect_px or ()),
        "score": round(c.score, 5), "terms": {k: round(v, 4) for k, v in c.terms.items()},
        "warnings": c.warnings,
    }  # fmt: skip


def suggest_crops(
    folder: str | Path,
    cache: Cache,
    image_id: int,
    path: str,
    settings: Settings | None = None,
    *,
    subjects: list[str] | None = None,
    points: list[tuple[float, float]] | None = None,
    select_modes: dict[str, str] | None = None,
    ratio: str = "3:2",
    flip: bool = False,
    n: int = 3,
    log_event: bool = True,
    boxes: list[Any] | None = None,
) -> dict[str, Any]:
    """Top `n` diverse crop suggestions. Writes nothing (only a `suggest` event row)."""
    settings = settings or Settings()
    if not 1 <= n <= 6:
        raise CropError("n must be between 1 and 6")
    ubox = check_boxes(boxes)
    sset = detect_subjects(folder, cache, image_id, path, settings)
    r = resolve_ratio(ratio, flip, sset.fw, sset.fh)
    pts = [point_subject(folder, cache, image_id, path, x, y, settings) for x, y in points or []]
    pts += [user_box_subject(folder, image_id, path, b, settings) for b in ubox]
    selected = _selection(sset, subjects or [], pts, select_modes or {})
    auto = not selected
    if auto:
        selected = auto_selection(sset)
    res = run_suggest(sset, selected, r, n, settings, extra=pts)
    dropped = 0
    if auto and res.infeasible and len(selected) > 1:
        # automatic selection only: when everyone cannot fit, keep the most prominent subjects
        order = sorted(selected, key=lambda s: -s.area)
        for k in range(len(order) - 1, 0, -1):
            res2 = run_suggest(sset, order[:k], r, n, settings, extra=pts)
            if not res2.infeasible:
                res, selected, dropped = res2, order[:k], len(order) - k
                break
    if dropped:
        for c in res.candidates:
            c.warnings.append(f"auto_dropped_{dropped}")
    if log_event:
        cache.crop_log(image_id, "suggest", None, r.text(), None)
    return {
        "candidates": [_candidate_json(c, sset) for c in res.candidates],
        "ratio": r.text(), "ratio_px": list(res.ratio_px) if res.ratio_px else None,
        "infeasible": res.infeasible, "auto": auto,
        "selected": [_subject_json(s) for s in selected],
        "horizon_tilt": None if not sset.horizon else round(sset.horizon[1], 2),
    }  # fmt: skip


# ---- versions + XMP --------------------------------------------------------------------------


def _write_kind(path: str, write: str) -> str:
    ext = Path(path).suffix.lower()
    if write == "none":
        return "none"
    if ext in RAW_EXTS and ext != ".dng":
        return "sidecar"
    if write == "embed" and ext in (".jpg", ".jpeg", ".dng"):
        return "embed"
    return "db_only"


def _carry(settings: Settings) -> dict[str, float | str]:
    try:
        from shotsieve.pipeline import edit

        return dict(edit.load_profile(settings).carry)
    except Exception:  # noqa: BLE001 - no edit profile (or unreadable): nothing to carry
        return {}


_CAT_CACHE: dict[tuple, dict[str, tuple[bool, bool]]] = {}


def _catalog_state(catalog: str) -> dict[str, tuple[bool, bool]]:
    """{path: (has edits, has crop)} of a Lightroom catalog, read once per catalog version (the
    read copies the file: a burst accept would otherwise copy it for every frame)."""
    from shotsieve.io import lrcat

    key = _catalog_key(catalog)
    if key not in _CAT_CACHE:
        _CAT_CACHE.clear()
        _CAT_CACHE[key] = lrcat.read_develop_state(catalog)
    return _CAT_CACHE[key]


def _catalog_key(catalog: str) -> tuple:
    """Catalog version: mtime/size of the .lrcat AND its -wal/-shm (Lightroom's WAL writes do not
    touch the main file until a checkpoint)."""
    parts: list[Any] = [str(catalog)]
    for suffix in ("", "-wal", "-shm"):
        try:
            st = Path(str(catalog) + suffix).stat()
            parts += [st.st_mtime_ns, st.st_size]
        except OSError:
            parts += [None, None]
    return tuple(parts)


def resolve_catalog(cache: Cache | None, catalog: str | None) -> str | None:
    """The catalog a crop write is checked against: the explicit one (validated like T20's, a bad
    path is a 400) else the shoot's remembered `lr_catalog`. A remembered catalog that no longer
    exists means "unknown" (None)."""
    from shotsieve.io import lrcat

    if catalog:
        try:
            return str(lrcat.check_catalog_path(catalog))
        except lrcat.CatalogError as exc:
            raise CropError(str(exc), 400) from None
    remembered = cache.get_meta("lr_catalog") if cache is not None else None
    if remembered:
        try:
            return str(lrcat.check_catalog_path(remembered))
        except lrcat.CatalogError:
            return None
    return None


def write_xmp(
    folder: str | Path,
    image_id: int,
    path: str,
    rect: Rect | None,
    version: int,
    info: ImageInfo,
    settings: Settings,
    *,
    write: str = "auto",
    embed_dir: str | Path | None = None,
    catalog: str | None = None,
) -> dict[str, Any]:
    """Write (or reset, `rect=None`) the crop into the master's XMP. Never raises for a normal
    skip: the result says what happened."""
    kind = _write_kind(path, write)
    if kind == "none" or (kind == "sidecar" and not settings.crop_write_xmp and write == "auto"):
        return {"mode": "none", "status": "none", "reason": "not written to XMP", "path": None}
    if kind == "db_only":
        return {
            "mode": "none", "status": "db_only", "path": None,
            "reason": LIGHTROOM_CROP_NOTE if Path(path).suffix.lower() in _xmp.SIDECAR_BLIND_EXTS
            else "this format has no sidecar crop",
        }  # fmt: skip
    has, has_crop = False, False
    if catalog:
        try:
            has, has_crop = _catalog_state(catalog).get(path, (False, False))
        except Exception as exc:  # noqa: BLE001 - locked/corrupt catalog: keep the crop in the DB
            return {
                "mode": "none", "status": "skipped_catalog", "path": None,
                "reason": f"the Lightroom catalog could not be read ({exc}); crop saved in "
                "ShotSieve only.",
            }  # fmt: skip
    o = info.orientation
    if kind == "sidecar" and rect is not None:
        lr_o = effective_orientation(path, info.orientation)
        if lr_o != info.orientation:
            # The preview, subjects and the crop the user saw are in the EXIF frame; the sidecar
            # (rotated in Lightroom) would convert it with another one: a different region.
            return {
                "mode": "none", "status": "skipped_rotated", "path": None,
                "reason": ROTATED_NOTE, "lightroom_orientation": lr_o,
            }  # fmt: skip
    rect_crs = None if rect is None else g.to_crs(rect, o).as_tuple()
    kw: dict[str, Any] = dict(
        catalog_has=has, catalog_has_crop=has_crop, version=version, carry=_carry(settings)
    )
    if kind == "embed":
        from shotsieve.pipeline import edit

        root = Path(embed_dir) if embed_dir else Path(folder) / "shotsieve_edits"
        edit.check_copy_root(root, folder)
        kw.update(mode="embed", out_dir=root / Path(os.path.relpath(Path(path).parent, folder)))
    res = crs.write_crop(path, rect_crs, **kw)
    return {
        "mode": "embed" if kind == "embed" else "sidecar", "status": res.status,
        "reason": res.reason, "path": str(res.path) if res.path else None,
    }  # fmt: skip


def _check_rect(rect: list[float] | tuple[float, ...]) -> Rect:
    try:
        r = Rect.of(rect)
    except (TypeError, ValueError, IndexError) as exc:
        raise CropError("rect must be [x0, y0, x1, y1]") from exc
    if not all(math.isfinite(v) for v in r.as_tuple()) or not r.valid():
        raise CropError("rect must satisfy 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1")
    return r


def _commit(
    folder: str | Path,
    cache: Cache,
    image_id: int,
    path: str,
    rect: Rect | None,
    info: ImageInfo,
    settings: Settings,
    *,
    ratio: str | None,
    source: str,
    subjects: list[str] | None,
    score: float | None = None,
    terms: dict | None = None,
    alt_rank: int | None = None,
    write: str = "auto",
    embed_dir: str | Path | None = None,
    catalog: str | None = None,
) -> dict[str, Any]:
    catalog = resolve_catalog(cache, catalog)  # validate BEFORE a version is stored
    version = cache.crop_add(
        image_id, None if rect is None else list(rect.as_tuple()), ratio=ratio, source=source,
        subjects_json=json.dumps(subjects) if subjects else None, score=score,
        terms_json=json.dumps(terms) if terms else None, alt_rank=alt_rank,
        det_version=det_version(), orientation=info.orientation, full_w=info.fw, full_h=info.fh,
    )  # fmt: skip
    out = write_xmp(
        folder,
        image_id,
        path,
        rect,
        version,
        info,
        settings,
        write=write,
        embed_dir=embed_dir,
        catalog=catalog,
    )
    cache.crop_set_write(image_id, version, out["mode"], out["status"])
    return {"version": version, "write": out}


def _info(
    folder: str | Path, cache: Cache, image_id: int, path: str, settings: Settings
) -> ImageInfo:
    if split_virtual(path) is not None:
        raise CropError("not for video frames", 422)
    hit = cache.get_subjects(path)
    return info_for(cache, path, hit["data"] if hit else None)


def accept(
    folder: str | Path,
    cache: Cache,
    image_id: int,
    path: str,
    settings: Settings | None,
    rect: list[float],
    ratio: str,
    *,
    source: str = "ai",
    subjects: list[str] | None = None,
    alt_rank: int | None = None,
    score: float | None = None,
    terms: dict | None = None,
    write: str = "auto",
    embed_dir: str | Path | None = None,
    catalog: str | None = None,
) -> dict[str, Any]:
    """Store `rect` as a new crop version (snapped to the exact ratio in full-resolution pixels)
    and write the XMP. An explicit accept is the user's decision."""
    settings = settings or Settings()
    if source not in SOURCES:
        raise CropError(f"source must be one of {list(SOURCES)}")
    if write not in ("auto", "none", "embed"):
        raise CropError("write must be 'auto', 'none' or 'embed'")
    info = _info(folder, cache, image_id, path, settings)
    r = _check_rect(rect)
    try:
        rt = g.parse_ratio(ratio)
    except ValueError as exc:
        raise CropError(str(exc)) from exc
    if not g.matches_ratio(r, rt, info.fw, info.fh):
        raise CropError("rect does not match the ratio to within 1 px")
    snapped, _ = g.snap_rect(r, rt, info.fw, info.fh)
    out = _commit(
        folder,
        cache,
        image_id,
        path,
        snapped,
        info,
        settings,
        ratio=rt.text(),
        source=source,
        subjects=subjects,
        score=score,
        terms=terms,
        alt_rank=alt_rank,
        write=write,
        embed_dir=embed_dir,
        catalog=catalog,
    )
    cache.crop_log(image_id, "adjust" if source == "ai_adjusted" else "accept",
                   list(snapped.as_tuple()), rt.text(), alt_rank)  # fmt: skip
    out["rect"] = list(snapped.as_tuple())
    return out


def reject(cache: Cache, image_id: int, alt_rank: int | None = None) -> dict[str, Any]:
    """Dismiss a suggestion: the image keeps its previous crop state; the event is logged."""
    cache.crop_log(image_id, "reject", None, None, alt_rank)
    return {"ok": True}


def reset(
    folder: str | Path,
    cache: Cache,
    image_id: int,
    path: str,
    settings: Settings | None = None,
    *,
    write: str = "auto",
    embed_dir: str | Path | None = None,
    catalog: str | None = None,
) -> dict[str, Any]:
    """A new version with no crop (full frame); the XMP is set back to Lightroom's defaults."""
    settings = settings or Settings()
    info = _info(folder, cache, image_id, path, settings)
    out = _commit(
        folder,
        cache,
        image_id,
        path,
        None,
        info,
        settings,
        ratio=None,
        source="reset",
        subjects=None,
        write=write,
        embed_dir=embed_dir,
        catalog=catalog,
    )
    cache.crop_log(image_id, "reset")
    return out


def restore(
    folder: str | Path,
    cache: Cache,
    image_id: int,
    path: str,
    version: int,
    settings: Settings | None = None,
    *,
    write: str = "auto",
    embed_dir: str | Path | None = None,
    catalog: str | None = None,
) -> dict[str, Any]:
    """A new version that copies an old one (`source="restore"`). Nothing is ever deleted."""
    settings = settings or Settings()
    info = _info(folder, cache, image_id, path, settings)
    old = next((h for h in cache.crop_history(image_id) if h["version"] == version), None)
    if old is None:
        raise CropError(f"no crop version {version}", 404)
    rect = None if old["rect"] is None else Rect.of(old["rect"])
    out = _commit(
        folder,
        cache,
        image_id,
        path,
        rect,
        info,
        settings,
        ratio=old["ratio"],
        source="restore",
        subjects=json.loads(old["subjects_json"]) if old["subjects_json"] else None,
        score=old["score"],
        write=write,
        embed_dir=embed_dir,
        catalog=catalog,
    )
    cache.crop_log(image_id, "restore", old["rect"], old["ratio"], version)
    out["rect"] = old["rect"]
    return out


def crop_state(
    folder: str | Path, cache: Cache, image_id: int, path: str, settings: Settings | None = None
) -> dict[str, Any]:
    """Current crop, history, full-resolution dims, XMP state and the ratio presets."""
    settings = settings or Settings()
    info = _info(folder, cache, image_id, path, settings)
    hist = cache.crop_history(image_id)
    cur = hist[-1] if hist else None
    lr_o = effective_orientation(path, info.orientation)
    o = info.orientation  # one frame end to end: the one the preview and overlays use
    xm = crs.read_crop(path, o)
    current = None
    if cur is not None:
        current = _hist_json(cur)
    elif xm.has_crop and not xm.ours and xm.rect_display:
        current = {
            "version": 0, "rect": [round(v, 6) for v in xm.rect_display], "ratio": None,
            "source": "lightroom", "readonly": True,
        }  # fmt: skip
    return {
        "current": current if current and current["rect"] else (current if current else None),
        "history": [_hist_json(h) for h in hist],
        "full": {"w": info.fw, "h": info.fh}, "orientation": o,
        "xmp": {"has_crop": xm.has_crop, "ours": xm.ours, "source": xm.source},
        "ratios": list(PRESET_RATIOS), "kind": info.kind,
        "user_boxes": cache.user_boxes(image_id),
        "rotated_in_lightroom": lr_o != info.orientation,
        "rotated_notice": ROTATED_NOTE if lr_o != info.orientation else None,
    }  # fmt: skip


def _hist_json(h: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": h["version"], "rect": h["rect"], "ratio": h["ratio"], "source": h["source"],
        "score": h["score"], "alt_rank": h["alt_rank"], "created": h["created"],
        "write_status": h["write_status"], "written_mode": h["written_mode"],
    }  # fmt: skip


# ---- burst transfer --------------------------------------------------------------------------


@dataclass
class BurstItem:
    id: int
    rect: list[float] | None
    status: str  # suggested | needs_selection
    rect_px: list[int] | None = None
    warnings: list[str] = field(default_factory=list)


def _center(b: tuple[float, float, float, float]) -> tuple[float, float]:
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return Rect.of(a).iou(Rect.of(b)) if a[2] > a[0] and b[2] > b[0] else 0.0


def _median_shift(a: SubjectSet, b: SubjectSet) -> tuple[float, float]:
    dx, dy = [], []
    for s in a.subjects:
        if s.kind not in ("face", "person"):
            continue
        cands = [
            t for t in b.subjects if t.kind == s.kind and 0.5 <= t.area / max(s.area, 1e-9) <= 2
        ]
        if not cands:
            continue
        ca = _center(s.box)
        t = min(cands, key=lambda t: math.dist(_center(t.box), ca))
        cb = _center(t.box)
        dx.append(cb[0] - ca[0])
        dy.append(cb[1] - ca[1])
    return (float(np.median(dx)), float(np.median(dy))) if dx else (0.0, 0.0)


def match_subject(
    s: Subject,
    a: SubjectSet,
    b: SubjectSet,
    shift: tuple[float, float],
    tok_a: np.ndarray | None,
    tok_b: np.ndarray | None,
) -> Subject | None:
    """The subject of frame B that is `s` of frame A: faces/persons by box IoU >= 0.3 after the
    burst's median shift (and a similar size); objects/points by DINOv2 token similarity at the
    anchor (best patch within 15 % of the frame)."""
    if s.kind in ("face", "person"):
        moved = (s.box[0] + shift[0], s.box[1] + shift[1], s.box[2] + shift[0], s.box[3] + shift[1])
        best, best_iou = None, BURST_IOU
        for t in b.subjects:
            if t.kind != s.kind:
                continue
            v = _iou(moved, t.box)
            if v >= best_iou and 0.5 <= t.area / max(s.area, 1e-9) <= 2:
                best, best_iou = t, v
        if best is None:
            return None
        q = Subject(**{**best.__dict__})
        q.select_mode = s.select_mode
        return q
    if tok_a is None or tok_b is None:
        return None
    gh, gw, _ = tok_a.shape
    cx, cy = _center(s.box)
    ai, aj = min(gh - 1, int(cy * gh)), min(gw - 1, int(cx * gw))
    ref = tok_a[ai, aj]
    bh, bw, _ = tok_b.shape
    sim = tok_b @ ref
    yy, xx = np.mgrid[0:bh, 0:bw]
    near = (np.abs((xx + 0.5) / bw - cx) <= 0.15) & (np.abs((yy + 0.5) / bh - cy) <= 0.15)
    sim = np.where(near, sim, -1.0)
    i = int(np.argmax(sim))
    if sim.ravel()[i] < BURST_SIM:
        return None
    pi, pj = divmod(i, bw)
    sal = b.saliency
    if sal is None or sal.shape != (bh, bw):
        sal = np.zeros((bh, bw), np.float32) if sal is None else cv2.resize(sal, (bw, bh))
    box, _ = patch_features.point_region(tok_b, sal, (pj + 0.5) / bw, (pi + 0.5) / bh)
    return Subject(s.sid, s.kind, box, 1.0, "burst", s.label)


def burst_transfer(
    folder: str | Path,
    cache: Cache,
    src_id: int,
    src_path: str,
    targets: list[tuple[int, str]],
    rect: list[float],
    ratio: str,
    selected_sids: list[str],
    select_modes: dict[str, str],
    points: list[tuple[float, float]],
    settings: Settings | None = None,
    progress: Any = None,
) -> list[BurstItem]:
    """Transfer the accepted crop of frame A to the other frames of its burst as PROPOSALS (never
    accepted automatically). Every selected subject must match, else `needs_selection`."""
    settings = settings or Settings()
    a = detect_subjects(folder, cache, src_id, src_path, settings)
    r = _check_rect(rect)
    rt = g.parse_ratio(ratio)
    pts = [point_subject(folder, cache, src_id, src_path, x, y, settings) for x, y in points]
    sel = _selection(a, selected_sids, pts, select_modes) or auto_selection(a)
    tok_a = _tokens_for(folder, src_id, src_path, settings)
    # where the subject sits inside A's crop: the offset to keep in the other frames
    sc_a = cs.Scene(sel, [*a.subjects, *pts], a.fw, a.fh)
    anchor = sc_a.anchor or (0.5, 0.5)
    same = ((anchor[0] - r.x0) / r.w, (anchor[1] - r.y0) / r.h)
    out: list[BurstItem] = []
    for k, (bid, bpath) in enumerate(targets):
        if progress:
            progress("burst", k, len(targets))
        try:
            b = detect_subjects(folder, cache, bid, bpath, settings)
        except CropError:
            out.append(BurstItem(bid, None, "needs_selection"))
            continue
        shift = _median_shift(a, b)
        tok_b = None
        if any(s.kind not in ("face", "person") for s in sel):
            tok_b = _tokens_for(folder, bid, bpath, settings)
        matched = [match_subject(s, a, b, shift, tok_a, tok_b) for s in sel]
        if any(m is None for m in matched):
            out.append(BurstItem(bid, None, "needs_selection"))
            continue
        rtb = rt.oriented(b.fw, b.fh) if rt.kind != "fixed" else rt
        res = run_suggest(
            b,
            [m for m in matched if m is not None],
            rtb,
            1,
            settings,
            extra=[m for m in matched if m is not None and m.kind == "point"],
            same_ref=same,
        )
        if not res.candidates:
            out.append(BurstItem(bid, None, "needs_selection"))
            continue
        c = res.candidates[0]
        out.append(BurstItem(bid, [round(v, 6) for v in c.rect.as_tuple()], "suggested",
                             list(c.rect_px or ()), c.warnings))  # fmt: skip
    if progress:
        progress("burst", len(targets), len(targets))
    return out


def save_user_boxes(
    folder: str | Path, cache: Cache, image_id: int, path: str, boxes: list[Any],
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Persist the user-drawn subject boxes of an image (a new append-only version)."""
    ub = check_boxes(boxes)
    info = _info(folder, cache, image_id, path, settings or Settings())
    cache.user_boxes_add(image_id, [list(b) for b in ub], info.orientation)
    return {"boxes": cache.user_boxes(image_id)}
