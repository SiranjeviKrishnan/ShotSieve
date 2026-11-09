"""Staged, overlapped, cached culling pipeline: `run_cull`."""

from __future__ import annotations

import importlib
import os
import queue
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from shotsieve.config import Settings, data_dir
from shotsieve.features import banding
from shotsieve.features.flash import FlashInput, detect_flash
from shotsieve.features.record import (
    RawFeatures,
    build_vector,
    compute_norm,
    group_norms,
)
from shotsieve.features.vector import FeatureVector, stack
from shotsieve.grouping import clock
from shotsieve.grouping.bursts import TimeItem, assign_bursts
from shotsieve.grouping.duplicates import group_duplicates
from shotsieve.io import makernotes
from shotsieve.io.exif import ImageMeta, read_meta
from shotsieve.io.formats import ImageEntry, scan_folder, split_virtual
from shotsieve.io.preview import load_preview
from shotsieve.io.xmp import WriteOutcome, write_decision
from shotsieve.log import get_logger
from shotsieve.pipeline import incremental, stages, thumbs
from shotsieve.pipeline.cache import Cache, CachedImage, fingerprint
from shotsieve.scoring import quality, reasons, selection

log = get_logger("runner")

ProgressFn = Callable[[str, int, int], None]
_SENTINEL = object()
EXTRACTOR_VERSION = "4"  # bump when extraction code changes (face sharpness scale, luma, ...)
THRESHOLD_KEYS = ("t_keep", "t_reject", "hard_margin", "dup_margin", "dup_review_budget")


def extractor_fingerprint(settings: Settings) -> str:
    """Identity of everything that shapes the cached raw features: code version, the settings
    that feed extraction and the model files (size + mtime)."""
    import hashlib

    from shotsieve.models.registry import group_names, model_path

    parts = [EXTRACTOR_VERSION, str(settings.preview_max_edge), str(settings.max_faces)]
    for name in sorted(group_names("core")):
        try:
            st = model_path(name).stat()
            parts.append(f"{name}:{st.st_size}:{st.st_mtime_ns}")
        except OSError:
            parts.append(f"{name}:missing")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


@dataclass
class ImageResult:
    path: Path  # file the preview came from
    master_path: Path  # file that XMP is written next to (RAW if paired with a JPEG)
    group_id: int
    burst_id: int
    decision: str  # keep | reject | review
    reasons: list[str]  # reason codes (see scoring/reasons.py)
    score: float  # final P(keep): personal/prior blend
    features: FeatureVector
    audit: bool = False
    is_best_in_group: bool = False
    reason_texts: list[str] = field(default_factory=list)  # plain-language, aligned with reasons
    prior: float = 0.0  # rule-based quality score
    audit_prob: float | None = None  # inclusion probability when audit=True
    overridden: bool = False
    sub_scores: dict[str, float] = field(default_factory=dict)
    model_decision: str = ""  # decision before any user override (== decision if not overridden)
    reject_class: str = ""  # audit stratum of a reject: certain | hard | dup | score


@dataclass
class CullResult:
    images: list[ImageResult]
    groups: dict[int, list[str]]  # group_id -> file paths (singletons included)
    thresholds: dict[str, float]
    stats: dict[str, Any]


class _Job:
    __slots__ = (
        "idx", "entry", "meta", "raw", "preview", "faces", "face_emb", "scene", "error",
        "pending", "lock", "thumb",
    )  # fmt: skip

    def __init__(self, idx: int, entry: ImageEntry) -> None:
        self.idx, self.entry = idx, entry
        self.meta: ImageMeta | None = None
        self.raw: RawFeatures | None = None
        self.preview: np.ndarray | None = None
        self.faces: list = []
        self.face_emb = np.zeros((0, 128), np.float32)
        self.scene: np.ndarray | None = None
        self.error: str | None = None
        self.pending = 2
        self.lock = threading.Lock()
        self.thumb: Path | None = None  # where the grid thumbnail goes (written at decode time)


class _Rss:
    """Samples process RSS in a thread to record the peak."""

    def __init__(self, interval: float = 0.1) -> None:
        import psutil

        self._proc = psutil.Process()
        self._stop = threading.Event()
        self.peak = self._proc.memory_info().rss
        self._interval = interval
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            self.peak = max(self.peak, self._proc.memory_info().rss)

    def __enter__(self) -> _Rss:
        self._t.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._t.join()
        self.peak = max(self.peak, self._proc.memory_info().rss)

    @property
    def peak_mb(self) -> float:
        return self.peak / 1e6


class _Progress:
    def __init__(self, cb: ProgressFn | None) -> None:
        self.cb, self._lock = cb, threading.Lock()
        self.busy: dict[str, float] = {}

    def emit(self, stage: str, done: int, total: int) -> None:
        if self.cb is not None:
            with self._lock:
                self.cb(stage, done, total)

    def add_busy(self, stage: str, seconds: float) -> None:
        with self._lock:
            self.busy[stage] = self.busy.get(stage, 0.0) + seconds


def _stat(entry: ImageEntry) -> tuple[int, int]:
    st = entry.preview_source.stat()
    return st.st_size, st.st_mtime_ns


def _process_uncached(
    jobs_in: list[_Job], settings: Settings, prog: _Progress, n_total: int, n_cached: int
) -> list[_Job]:
    """Run decode+StageA (thread pool), faces (1 worker) and embeddings (1 worker) overlapped."""
    from shotsieve.models.image_embed import get_image_embedder

    threads = min(settings.ort_threads, max(2, (os.cpu_count() or 4) - 2))
    embedder = get_image_embedder(threads, settings.embed_batch, settings.use_coreml)
    face_q: queue.Queue = queue.Queue(maxsize=settings.queue_size)
    embed_q: queue.Queue = queue.Queue(maxsize=settings.queue_size)
    done_q: queue.Queue = queue.Queue()
    counts = {"features": n_cached, "faces": n_cached, "embed": n_cached}
    clock = threading.Lock()

    def tick(stage: str) -> None:
        with clock:
            counts[stage] += 1
            c = counts[stage]
        prog.emit(stage, c, n_total)

    def finish(job: _Job, stage: str) -> None:
        tick(stage)
        with job.lock:
            job.pending -= 1
            complete = job.pending == 0
        if complete:
            job.preview = None
            done_q.put(job)

    def fail(job: _Job, msg: str) -> None:
        with job.lock:
            first = job.error is None
            job.error = job.error or msg
        if first:
            job.preview = None
            done_q.put(job)
            tick("faces")
            tick("embed")

    def decode(job: _Job) -> None:
        t0 = time.perf_counter()
        try:
            job.meta = read_meta(job.entry.master)
            job.meta.af_points = [
                list(makernotes.to_oriented(p, job.meta.orientation))
                for p in makernotes.read_af_points(
                    job.entry.master, job.meta, settings.af_use_exiftool
                )
            ]  # stored upright, i.e. in preview orientation
            job.preview = load_preview(job.entry.preview_source, settings.preview_max_edge)
            job.raw = stages.stage_a(
                job.preview, job.meta.af_points, job.meta.exposure_time, job.meta.orientation
            )
            if job.thumb is not None:  # decode once: the grid thumbnail comes from this preview
                thumbs.write_thumb(job.preview, job.thumb)
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the shoot
            fail(job, f"{type(exc).__name__}: {exc}")
            tick("features")
            return
        finally:
            prog.add_busy("features", time.perf_counter() - t0)
        tick("features")
        if stages.is_near_black(job.raw):  # skip the model-heavy face stage
            job.raw.faces_skipped = True
            finish(job, "faces")
        else:
            face_q.put(job)
        embed_q.put(job)

    def face_worker() -> None:
        while (job := face_q.get()) is not _SENTINEL:
            t0 = time.perf_counter()
            try:
                job.raw.faces, job.face_emb = stages.stage_faces(job.preview, job.raw, settings)
            except Exception as exc:  # noqa: BLE001
                log.warning("face stage failed for %s: %s", job.entry.master, exc)
                job.raw.faces, job.face_emb = [], np.zeros((0, 128), np.float32)
            prog.add_busy("faces", time.perf_counter() - t0)
            finish(job, "faces")

    def embed_worker() -> None:
        stop = False
        while not stop:
            batch = [embed_q.get()]
            if batch[0] is _SENTINEL:
                return
            while len(batch) < settings.embed_batch:
                try:
                    nxt = embed_q.get_nowait()
                except queue.Empty:
                    break
                if nxt is _SENTINEL:
                    stop = True
                    break
                batch.append(nxt)
            t0 = time.perf_counter()
            try:
                vecs = embedder.embed_batch([j.preview for j in batch])
                for j, v in zip(batch, vecs, strict=True):
                    j.scene = v
            except Exception as exc:  # noqa: BLE001
                for j in batch:
                    j.error = j.error or f"embedding failed: {exc}"
            prog.add_busy("embed", time.perf_counter() - t0)
            for j in batch:
                finish(j, "embed")

    threads = [
        threading.Thread(target=face_worker, name="faces", daemon=True),
        threading.Thread(target=embed_worker, name="embed", daemon=True),
    ]
    for t in threads:
        t.start()
    pool = ThreadPoolExecutor(max_workers=settings.preview_workers, thread_name_prefix="decode")
    futures = [pool.submit(decode, j) for j in jobs_in]
    done: list[_Job] = []
    try:
        for _ in jobs_in:
            done.append(done_q.get())
    finally:
        for f in futures:
            f.cancel()
        pool.shutdown(wait=True)
        face_q.put(_SENTINEL)
        embed_q.put(_SENTINEL)
        for t in threads:
            t.join()
    return done


def _load_personal(settings: Settings) -> selection.PersonalModelLike | None:
    path = (
        Path(settings.personal_model_path)
        if settings.personal_model_path
        else (data_dir() / "personal_model.joblib")
    )
    if not path.exists():
        return None
    try:
        mod = importlib.import_module("shotsieve.personal.model")
        return mod.PersonalModel.load(path)
    except Exception as exc:  # noqa: BLE001 - personal model is optional
        log.warning("personal model not loaded (%s); using prior only", exc)
        return None


def _call_audit(images: list[ImageResult], seed: int) -> list[ImageResult]:
    try:
        mod = importlib.import_module("shotsieve.personal.audit")
    except ImportError:
        return []
    fn = getattr(mod, "sample_audit", None)
    if fn is None:
        return []
    try:
        return list(fn(images, seed))
    except Exception as exc:  # noqa: BLE001
        log.warning("audit sampling failed: %s", exc)
        return []


def _apply_frozen_audit(images: list[ImageResult], frozen: dict[str, dict]) -> None:
    from shotsieve.personal.audit import mark_audit

    for r in images:
        f = frozen.get(str(r.master_path))
        if f is None or r.overridden:
            continue
        if r.decision == "reject":
            r.reject_class = r.reject_class or f.get("k", "")
            mark_audit(r, float(f["p"]))
        else:  # no longer a reject (recalibrated): stays in the sample, keeps its stratum
            r.audit, r.audit_prob, r.reject_class = True, float(f["p"]), f.get("k", "")


def _eye_flags(raw: RawFeatures, settings: Settings) -> list[str]:
    """Context flags for the quality rules: single-eye closure and a well-exposed subject."""
    flags: list[str] = []
    if not raw.faces:
        return flags
    main = raw.faces[0]
    bl, br = main.blink_left, main.blink_right
    if bl == bl and br == br:  # not NaN
        if (
            max(bl, br) > settings.eyes_closed_thresh
            and min(bl, br) <= settings.eyes_closed_both_thresh
        ):
            flags.append("eyes_single")
    lo, hi = quality.WELL_EXPOSED_LUMA
    if main.luma == main.luma and lo <= main.luma <= hi:
        flags.append("subject_well_exposed")
    return flags


def _video_results(
    folder: Path, settings: Settings, cache: Cache, extractor: str, thresholds: dict[str, float],
    model: selection.PersonalModelLike | None, n_train: int, next_group: int, next_burst: int,
    prog: _Progress, vstats: dict[str, Any], only: frozenset[str] | None = None,
) -> list[ImageResult]:
    """Candidate frames of every video under `folder` as virtual `clip.mp4#t=..` results (T12).
    One duplicate group / burst per (video, scene segment); decided by the photo rules."""
    from shotsieve.io.formats import scan_videos
    from shotsieve.pipeline import video as pv

    videos = scan_videos(folder)
    if only is not None:
        videos = [v for v in videos if str(v) in only]
    vstats.update(
        n_videos=len(videos), n_scanned=0, frames=0, src_frames=0, seconds=0.0, scan_s=0.0, picks=0
    )
    picks: list[pv.Pick] = []
    for vi, vp in enumerate(videos):
        prog.emit("video", vi, len(videos))
        try:
            st = vp.stat()
        except OSError:
            continue
        got, vs = pv.cached_or_scan(cache, vp, (st.st_size, st.st_mtime_ns), extractor, settings)
        picks.extend(got)
        if vs and "error" not in vs:
            vstats["n_scanned"] += 1
            vstats["frames"] += vs["frames"]
            vstats["src_frames"] += vs.get("src_frames", 0)
            vstats["seconds"] += vs["seconds"]
            vstats["scan_s"] += vs["wall_s"]
            vstats["backend"] = vs.get("backend", "")
    prog.emit("video", len(videos), len(videos))
    vstats["picks"] = len(picks)
    if not picks:
        return []
    gid: dict[tuple[Path, int], int] = {}
    for p in picks:
        gid.setdefault((p.video, p.segment), len(gid))
    group_ids = [next_group + gid[(p.video, p.segment)] for p in picks]
    fvs, quals = [], []
    for p in picks:
        fv = build_vector(
            p.raw, p.norm, burst_size=p.n_in_segment, dup_group_size=p.n_in_segment,
            eyes_thresh=settings.eyes_closed_thresh,
            eyes_both_thresh=settings.eyes_closed_both_thresh,
        )  # fmt: skip
        flags = _eye_flags(p.raw, settings)
        flags += banding.flags(p.raw.band_peak_frac, p.raw.band_k)
        fvs.append(fv)
        quals.append(quality.assess(fv, settings, flags))
    priors = [q.score for q in quals]
    names = [Path(p.path).name for p in picks]
    rank, size = selection.rank_groups(
        group_ids, priors, [bool(q.hard_fails) for q in quals], names
    )
    for fv, rk, sz in zip(fvs, rank, size, strict=True):
        fv.rank_in_group_norm = rk / (sz - 1) if sz > 1 else 0.0
    p_final = np.asarray(priors, dtype=np.float64)
    if model is not None and n_train:
        try:
            pp = selection.personal_probabilities(
                model, stack(fvs), np.stack([p.scene for p in picks])
            )
            p_final = selection.blend(p_final, pp, n_train, settings.personal_prior_n)
        except Exception as exc:  # noqa: BLE001
            log.warning("personal model failed on video frames (%s); prior only", exc)
    cands = [
        selection.Candidate(names[k], group_ids[k], fvs[k], quals[k], float(p_final[k]))
        for k in range(len(picks))
    ]
    decisions = selection.decide(
        cands, settings, thresholds["t_keep"], thresholds["t_reject"],
        hard_margin=thresholds["hard_margin"], dup_margin=thresholds["dup_margin"],
        dup_review_budget=thresholds["dup_review_budget"], trained=n_train > 0,
    )  # fmt: skip
    out = []
    for k, p in enumerate(picks):
        d = decisions[k]
        vpath = Path(p.path)
        out.append(
            ImageResult(
                path=vpath, master_path=vpath, group_id=group_ids[k],
                burst_id=next_burst + gid[(p.video, p.segment)], decision=d.decision,
                reasons=list(d.reasons), score=float(p_final[k]), features=fvs[k],
                is_best_in_group=d.is_best,
                reason_texts=[reasons.describe(c, fvs[k], d.ctx) for c in d.reasons],
                prior=priors[k], sub_scores=quals[k].subs, model_decision=d.decision,
                reject_class=d.klass,
            )  # fmt: skip
        )
    return out


def _run_cull(
    folder: Path,
    settings: Settings,
    progress: ProgressFn | None = None,
    write_xmp: bool = False,
    *,
    personal_model: selection.PersonalModelLike | None = None,
    redraw_audit: bool = False,
    force_ratings: bool = False,
    full: bool = False,
    only: frozenset[str] | None = None,
) -> CullResult:
    """Cull every image under `folder`; results are persisted in <folder>/.shotsieve/cull.db.
    `only` (drag and drop): add-only cull of those absolute file paths inside `folder` on top of
    whatever is already culled there.

    Progress callbacks receive (stage, done, total) for stage in scan|features|faces|embed|
    group|score|write; features/faces/embed callbacks may arrive from worker threads (serialised).
    `personal_model` (optional) blends with the rule-based prior with w = n / (n + 200); if None
    one is loaded from `settings.personal_model_path` (or data/personal_model.joblib) when present.

    `ImageResult.decision` is the final decision (user overrides applied at read time);
    `model_decision` is what the model decided and is what gets persisted, so an override can
    always be undone and survives re-runs. The audit sample is drawn once per shoot and frozen
    (`redraw_audit=True` draws a new one). With `write_xmp`, an existing foreign rating/label is
    only replaced for images the user overrode in ShotSieve (or with `force_ratings`).

    A re-run only re-decides what changed (T21, `pipeline/incremental.py`): the shoot norm is
    frozen while <= 20 % of the frames changed and frames outside the affected set keep their
    stored decision. `full=True` (CLI `cull --full`) forces a complete re-decide.
    """
    folder = Path(folder)
    t_start = time.perf_counter()
    prog = _Progress(progress)
    timings: dict[str, float] = {}
    with _Rss() as rss, Cache.for_folder(folder) as cache:
        # ---- scan ----------------------------------------------------------------------
        t0 = time.perf_counter()
        entries = scan_folder(folder)
        entries = [e for e in entries if ".shotsieve" not in e.master.parts]
        if only is not None:  # explicit file list (drag and drop): ADD-ONLY
            # everything already culled stays in the scan (never "removed", rows/overrides/norm
            # untouched); the listed files are added, or refreshed when already known
            only = frozenset(only) | frozenset(
                str(split_virtual(p)[0]) if split_virtual(p) else p
                for p in cache.load_decisions()
            )
            entries = [
                e for e in entries if str(e.master) in only or str(e.preview_source) in only
            ]
        n = len(entries)
        stats_by_path: dict[str, tuple[int, int]] = {}
        source_of = {str(e.master): e.preview_source for e in entries}
        for e in entries:
            try:
                stats_by_path[str(e.master)] = _stat(e)
            except OSError:
                pass
        extractor = extractor_fingerprint(settings)
        cached = cache.lookup_many(stats_by_path, extractor, lambda k: fingerprint(source_of[k]))
        timings["scan_s"] = time.perf_counter() - t0
        prog.emit("scan", n, n)

        # ---- feature extraction (overlapped stages) -------------------------------------
        records: dict[str, CachedImage] = {k: v for k, v in cached.items()}
        errors: dict[str, str] = {}
        todo = [
            _Job(i, e) for i, e in enumerate(entries)
            if str(e.master) not in cached and str(e.master) in stats_by_path
        ]  # fmt: skip
        for e in entries:
            if str(e.master) not in stats_by_path:
                errors[str(e.master)] = "file not accessible"
        t0 = time.perf_counter()
        n_cached = len(cached)
        if todo:  # ids are assigned here (never by a GET); thumbnails are written while decoding
            ids = cache.ids_for([str(j.entry.master) for j in todo])
            for j in todo:
                key = str(j.entry.master)
                if key in ids and not split_virtual(key):
                    j.thumb = thumbs.thumb_path(folder, ids[key], stats_by_path[key][1])
        for stage in ("features", "faces", "embed"):
            prog.emit(stage, n_cached, n)
        if todo:
            for job in _process_uncached(todo, settings, prog, n, n_cached):
                key = str(job.entry.master)
                if job.error or job.raw is None or job.scene is None or job.meta is None:
                    errors[key] = job.error or "incomplete"
                    continue
                # Canonical form = exactly what the cache returns: unit-norm, THEN fp16. (Rounding
                # first and normalising after left the cold vector != the warm one, flipping
                # borderline duplicate links on a cached re-run.)
                scene = job.scene.astype(np.float32)
                scene /= max(float(np.linalg.norm(scene)), 1e-12)
                scene = scene.astype(np.float16).astype(np.float32)
                rec = CachedImage(
                    job.meta, job.raw, scene, job.face_emb.astype(np.float16).astype(np.float32)
                )
                records[key] = rec
                size, mtime = stats_by_path[key]
                try:
                    fp = fingerprint(job.entry.preview_source)
                except OSError:
                    fp = ""
                cache.put(key, size, mtime, rec, fp, extractor)
            cache.commit()
        timings["extract_wall_s"] = time.perf_counter() - t0
        for stage, busy in prog.busy.items():
            timings[f"{stage}_busy_s"] = busy

        # ---- clock offsets between camera bodies (post-embed step) -----------------------
        t0 = time.perf_counter()
        try:
            clock.analyse(cache, auto_apply=True)
        except Exception as exc:  # noqa: BLE001 - never let the optional step fail a cull
            log.warning("clock offset step failed: %s", exc)
        offsets = clock.offsets_from_meta(cache.get_clock_offsets())
        timings["clock_s"] = time.perf_counter() - t0

        # ---- grouping --------------------------------------------------------------------
        t0 = time.perf_counter()
        prog.emit("group", 0, n)
        ok = [i for i, e in enumerate(entries) if str(e.master) in records]
        recs = [records[str(entries[i].master)] for i in ok]
        titems = [
            TimeItem(
                r.meta.camera_serial,
                clock.adjust(r.meta.timestamp, r.meta.camera_serial, offsets),
                r.meta.ts_source,
            )
            for r in recs
        ]  # t_adj; bursts stay per body, so a constant per-body shift never changes them
        burst_ids = assign_bursts(titems, settings.burst_gap_s) if recs else []
        scene = np.stack([r.scene for r in recs]) if recs else np.zeros((0, 384), np.float32)
        main_faces = [r.faces[0] if len(r.faces) else None for r in recs]
        dup = group_duplicates(
            scene, titems, main_faces,
            window_s=settings.dup_window_s,
            cos_dist=settings.dup_cos_dist,
            same_person_cos=settings.face_same_person_cos,
        )  # fmt: skip
        group_ids = dup.group_ids

        # ---- incremental plan: frozen norm, affected set, stable ids (T21) ----------------------
        paths_ok = [str(entries[i].master) for i in ok]
        ok_set = set(paths_ok)
        prev = cache.load_decisions()
        prev_real = {p: r for p, r in prev.items() if not split_virtual(p)}
        now_all = {str(e.master) for e in entries}
        removed_paths = sorted(set(prev_real) - now_all)
        new_set = {str(j.entry.master) for j in todo} | (ok_set - set(prev_real))
        sig = incremental.signature(
            settings, cache.get_thresholds(),
            incremental.personal_id(
                settings, personal_model, data_dir() / "personal_model.joblib"
            ),
            cache.get_clock_offsets(),
        )  # fmt: skip
        plan = incremental.plan(
            cache.get_meta("shoot_norm"), extractor=extractor, sig=sig, n_prev=len(prev_real),
            n_new=len(new_set), n_removed=len(removed_paths), force_full=full or redraw_audit,
        )  # fmt: skip
        affected: set[str] = set()
        if plan.mode != "full":
            ovr0 = cache.overrides()  # overridden audit items are re-judged (no longer sampled)
            metas = cache.metas_of(removed_paths)
            removed_info = {
                p: (
                    (m.camera_serial, clock.adjust(m.timestamp, m.camera_serial, offsets))
                    if (m := metas.get(p))
                    else ("", None)
                )
                for p in removed_paths
            }
            affected = incremental.affected_set(
                paths=paths_ok, group_ids=group_ids, burst_ids=burst_ids,
                serials=[r.meta.camera_serial for r in recs], ts=[t.timestamp for t in titems],
                new=new_set, removed=removed_info,
                prev_group={p: r["group_id"] for p, r in prev_real.items()},
                prev_burst={p: r["burst_id"] for p, r in prev_real.items()},
                window_s=max(settings.dup_window_s, settings.flash_window_s),
                extra={p for p, r in prev_real.items() if r["audit"] and p in ovr0},
            )  # fmt: skip
            group_ids = incremental.stable_ids(
                group_ids, paths_ok, {p: r["group_id"] for p, r in prev_real.items()}
            )
            burst_ids = incremental.stable_ids(
                burst_ids, paths_ok, {p: r["burst_id"] for p, r in prev_real.items()}
            )
        burst_size: dict[int, int] = {}
        group_size: dict[int, int] = {}
        for b, g in zip(burst_ids, group_ids, strict=True):
            burst_size[b] = burst_size.get(b, 0) + 1
            group_size[g] = group_size.get(g, 0) + 1
        timings["group_s"] = time.perf_counter() - t0
        prog.emit("group", n, n)

        # ---- scoring ---------------------------------------------------------------------
        t0 = time.perf_counter()
        prog.emit("score", 0, n)
        flash = detect_flash(
            [
                FlashInput(
                    r.meta.camera_serial, r.meta.timestamp, r.meta.flash_fired,
                    r.raw.lin_mean, r.meta.ts_source == "exif",
                )
                for r in recs
            ],
            settings.flash_window_s, settings.flash_dark_ev_nofire, settings.flash_dark_ev_fired,
        )  # fmt: skip
        if plan.norm is not None:
            norm = plan.norm
            norm_rec = incremental.norm_record(
                norm, int(plan.stored["n"]), extractor, sig, plan.drift
            )
        else:
            norm = compute_norm([r.raw for r in recs])
            norm_rec = incremental.norm_record(norm, len(recs), extractor, sig, 0)
        norms = group_norms([r.raw for r in recs], group_ids, norm) if recs else []
        fvs: list[FeatureVector] = []
        quals: list[quality.QualityResult] = []
        for k, r in enumerate(recs):
            fv = build_vector(
                r.raw, norms[k],
                flash_fired=r.meta.flash_fired, flash_delta_ev=flash[k].delta_ev,
                iso=r.meta.iso, exposure_time=r.meta.exposure_time,
                burst_size=burst_size[burst_ids[k]], dup_group_size=group_size[group_ids[k]],
                eyes_thresh=settings.eyes_closed_thresh,
                eyes_both_thresh=settings.eyes_closed_both_thresh,
            )  # fmt: skip
            flags = (["flash_misfire"] if flash[k].misfire else []) + _eye_flags(r.raw, settings)
            if r.raw.af_peak_adjacent:
                flags.append("af_peak_adjacent")
            flags += banding.flags(r.raw.band_peak_frac, r.raw.band_k)
            fvs.append(fv)
            quals.append(quality.assess(fv, settings, flags))
        priors = [q.score for q in quals]
        names = [Path(entries[i].master).name for i in ok]
        failed = [bool(q.hard_fails) for q in quals]
        rank, size = selection.rank_groups(group_ids, priors, failed, names)
        for fv, rk, sz in zip(fvs, rank, size, strict=True):
            fv.rank_in_group_norm = rk / (sz - 1) if sz > 1 else 0.0

        model = personal_model if personal_model is not None else _load_personal(settings)
        p_final = np.asarray(priors, dtype=np.float64)
        n_train = 0
        if model is not None and recs:
            try:
                n_train = int(model.n_train)
                pp = selection.personal_probabilities(model, stack(fvs), scene)
                p_final = selection.blend(p_final, pp, n_train, settings.personal_prior_n)
            except Exception as exc:  # noqa: BLE001
                log.warning("personal model failed (%s); using prior only", exc)
                n_train = 0

        thresholds = {
            "t_keep": settings.t_keep, "t_reject": settings.t_reject,
            "hard_margin": 0.0, "dup_margin": 0.0, "dup_review_budget": 1.0,
        }  # fmt: skip
        mt = getattr(model, "thresholds", None) if n_train else None
        if mt and "t_reject" in mt:
            # the model's own scale for P(keep), blended like the scores themselves
            w = selection.blend_weight(n_train, settings.personal_prior_n)
            thresholds["t_reject"] = min(
                thresholds["t_reject"],
                w * float(mt["t_reject"]) + (1.0 - w) * thresholds["t_reject"],
            )
        saved = cache.get_thresholds()
        if saved:
            thresholds.update({k: float(v) for k, v in saved.items() if k in THRESHOLD_KEYS})
        cands = [
            selection.Candidate(names[k], group_ids[k], fvs[k], quals[k], float(p_final[k]),
                                flash[k].delta_ev)
            for k in range(len(recs))
        ]  # fmt: skip
        decisions = selection.decide(
            cands, settings, thresholds["t_keep"], thresholds["t_reject"],
            hard_margin=thresholds["hard_margin"], dup_margin=thresholds["dup_margin"],
            dup_review_budget=thresholds["dup_review_budget"], trained=n_train > 0,
        )  # fmt: skip

        results: list[ImageResult] = []
        for k, i in enumerate(ok):
            e, d = entries[i], decisions[k]
            results.append(
                ImageResult(
                    path=e.preview_source,
                    master_path=e.master,
                    group_id=group_ids[k],
                    burst_id=burst_ids[k],
                    decision=d.decision,
                    reasons=list(d.reasons),
                    score=float(p_final[k]),
                    features=fvs[k],
                    is_best_in_group=d.is_best,
                    reason_texts=[reasons.describe(c, fvs[k], d.ctx) for c in d.reasons],
                    prior=priors[k],
                    sub_scores=quals[k].subs,
                    model_decision=d.decision,
                    reject_class=d.klass,
                )  # fmt: skip
            )
        next_group = (max(group_ids) + 1) if group_ids else 0
        next_burst = (max(burst_ids) + 1) if burst_ids else 0
        for e in entries:  # unreadable files still get a decision
            if str(e.master) in records:
                continue
            msg = errors.get(str(e.master), "")
            results.append(
                ImageResult(
                    path=e.preview_source,
                    master_path=e.master,
                    group_id=next_group,
                    burst_id=next_burst,
                    decision="review",
                    reasons=["unreadable"],
                    score=0.0,
                    features=FeatureVector(),
                    reason_texts=[reasons.describe("unreadable") + (f" ({msg})" if msg else "")],
                    model_decision="review",
                )  # fmt: skip
            )
            next_group += 1
            next_burst += 1
        results.sort(key=lambda r: str(r.master_path))

        ovr = cache.overrides()
        for r in results:  # which images the user already decided (they never enter the audit)
            r.overridden = str(r.master_path) in ovr
        if redraw_audit:
            cache.set_audit_sample(None)
        if settings.audit_enabled:
            frozen = cache.get_audit_sample()
            if frozen is None:
                picked = _call_audit(results, settings.audit_seed)
                if picked:
                    cache.set_audit_sample(
                        {str(r.master_path): {"p": r.audit_prob, "k": r.reject_class}
                         for r in picked}
                    )  # fmt: skip
            else:
                _apply_frozen_audit(results, frozen)
        vstats: dict[str, Any] = {}
        if settings.video:  # opt-in; after the photos, never part of the audit sample
            t0 = time.perf_counter()
            vres = _video_results(
                folder, settings, cache, extractor, thresholds, model, n_train,
                next_group, next_burst, prog, vstats, only,
            )  # fmt: skip
            for r in vres:
                r.overridden = str(r.master_path) in ovr
            results.extend(vres)
            results.sort(key=lambda r: str(r.master_path))
            timings["video_s"] = time.perf_counter() - t0
        n_drift = 0
        if plan.mode != "full":
            for r in results:  # frames outside the affected set keep their stored row verbatim
                key = str(r.master_path)
                if key in affected or key not in ok_set or key not in prev_real:
                    continue
                st = prev_real[key]
                if (
                    r.decision == st["decision"] and r.reasons == st["reasons"]
                    and abs(r.score - st["score"]) <= 1e-9 and r.is_best_in_group == st["is_best"]
                ):  # fmt: skip
                    continue
                n_drift += 1
                r.decision, r.reasons, r.score = st["decision"], list(st["reasons"]), st["score"]
                r.prior, r.is_best_in_group = st["prior"], st["is_best"]
                r.reject_class = st["klass"]
                if st["features"]:
                    r.features = FeatureVector.from_dict(st["features"])
                r.reason_texts = [
                    reasons.describe(c, r.features) if c in reasons.REASON_LABELS else c
                    for c in r.reasons
                ]
            if n_drift:
                msg = f"{n_drift} frame(s) outside the affected set would have changed; kept"
                if os.environ.get("SHOTSIEVE_DEBUG"):
                    raise AssertionError(msg)
                log.warning(msg)
        for r in results:
            r.model_decision = r.decision  # includes the audit's move of a reject into review
            o = ovr.get(str(r.master_path))
            if o:
                r.decision = o["decision"]
        for r in results:  # audit/override may have touched reasons
            if len(r.reason_texts) != len(r.reasons):
                r.reason_texts = [
                    reasons.describe(c, r.features) if c in reasons.REASON_LABELS else c
                    for c in r.reasons
                ]
            if r.model_decision == "reject" and not r.reasons:
                r.reasons = ["low_quality"]
                r.reason_texts = [reasons.describe("low_quality")]
        timings["score_s"] = time.perf_counter() - t0
        prog.emit("score", n, n)

        cache.save_decisions(
            [
                {
                    "path": str(r.master_path), "decision": r.model_decision, "reasons": r.reasons,
                    "score": r.score, "prior": r.prior, "group_id": r.group_id,
                    "burst_id": r.burst_id, "is_best": r.is_best_in_group, "audit": r.audit,
                    "audit_prob": r.audit_prob, "features": r.features.to_dict(),
                    "klass": r.reject_class,
                }
                for r in results
            ]
        )  # fmt: skip
        cache.ids_for([str(r.master_path) for r in results])  # ids exist before any GET (M5)
        if plan.mode != "none":
            cache.set_meta("shoot_norm", norm_rec)

        # ---- XMP -------------------------------------------------------------------------
        t0 = time.perf_counter()
        kept_lr = 0
        if write_xmp:
            prog.emit("write", 0, len(results))
            written: list[tuple[str, int | None, str | None, str]] = []
            for j, r in enumerate(results, 1):
                if split_virtual(r.master_path):  # video frame candidates have no file
                    continue
                o = ovr.get(str(r.master_path))
                oc = WriteOutcome()
                try:
                    write_decision(
                        r.master_path, settings, r.decision, r.reason_texts, r.score,
                        override=o, force=force_ratings, outcome=oc,
                    )  # fmt: skip
                    kept_lr += oc.lr_kept
                    if oc.written_rating is not None or oc.written_label is not None:
                        written.append(
                            (str(r.master_path), oc.written_rating, oc.written_label, r.decision)
                        )
                except Exception as exc:  # noqa: BLE001 - one odd sidecar must not stop the run
                    log.warning("xmp write failed for %s: %s", r.master_path, exc)
                prog.emit("write", j, len(results))
            cache.put_written(written)  # T20: what Lightroom will import from us
        timings["write_s"] = time.perf_counter() - t0

    groups: dict[int, list[str]] = {}
    for r in results:
        groups.setdefault(r.group_id, []).append(str(r.path))
    total = time.perf_counter() - t_start
    counts = {d: sum(r.decision == d for r in results) for d in ("keep", "reject", "review")}
    n_new = len(todo)
    stats: dict[str, Any] = {
        "n_images": n,
        "n_cached": n_cached,
        "n_processed": n_new,
        "n_new": len(new_set),
        "n_removed": len(removed_paths),
        "n_extracted": n_new,
        "n_affected": len(affected & now_all) if plan.mode != "full" else len(results),
        "redecide": plan.mode,
        "redecide_reason": plan.reason,
        "n_drift": n_drift,
        "kept_lr": kept_lr,
        "total_s": total,
        "imgs_per_s": n / total if total else 0.0,
        "peak_rss_mb": rss.peak_mb,
        "timings": timings,
        "counts": counts,
        "n_bursts": len(set(r.burst_id for r in results)),
        "n_groups": len(groups),
        "dup_pairs_compared": dup.pairs_compared if recs else 0,
        "personal_n_train": n_train,
        "ort_threads": settings.ort_threads,
        "cpu_count": os.cpu_count(),
    }
    if settings.video:
        v = dict(vstats)
        if v.get("scan_s"):
            v["realtime_x"] = v["seconds"] / v["scan_s"]
            v["frames_per_s"] = v["src_frames"] / v["scan_s"]  # source frames decoded per second
        stats["video"] = v
    if n_new:
        wall = timings["extract_wall_s"]
        par = {"features": settings.preview_workers, "faces": 1, "embed": 1}
        stats["stage_imgs_per_s"] = {
            s: n_new / (prog.busy[s] / par[s]) for s in par if prog.busy.get(s, 0) > 0
        }
        stats["extract_imgs_per_s"] = n_new / wall if wall else 0.0
    return CullResult(results, groups, thresholds, stats)


def run_cull(*args: Any, **kwargs: Any) -> CullResult:
    """Cull a folder (see `_run_cull`); the background indexer is paused for the duration."""
    from shotsieve.pipeline.indexer import cull_active

    with cull_active():
        return _run_cull(*args, **kwargs)


run_cull.__doc__ = (run_cull.__doc__ or "") + "\n\n" + (_run_cull.__doc__ or "")
