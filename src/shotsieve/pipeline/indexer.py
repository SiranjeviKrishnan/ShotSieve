"""Background SigLIP2 indexer: one low-priority daemon thread per shoot, resumable, and paused
while any cull runs (`cull_active`, entered by `run_cull`)."""

from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from shotsieve.config import Settings
from shotsieve.io.formats import scan_folder
from shotsieve.io.preview import PreviewError, load_preview
from shotsieve.log import get_logger
from shotsieve.pipeline.cache import Cache, fingerprint

log = get_logger("indexer")

# ---- the cull gate (process-wide) ------------------------------------------------------------
_gate_lock = threading.Lock()
_gate_depth = 0
_gate_idle = threading.Event()
_gate_idle.set()


@contextmanager
def cull_active():
    """Mark a cull as running; indexers pause between batches until the last one finishes."""
    global _gate_depth
    with _gate_lock:
        _gate_depth += 1
        _gate_idle.clear()
    try:
        yield
    finally:
        with _gate_lock:
            _gate_depth -= 1
            if _gate_depth == 0:
                _gate_idle.set()


def cull_running() -> bool:
    return not _gate_idle.is_set()


def _return_freed_pages() -> None:
    """macOS keeps freed malloc pages resident; ask the allocator to hand them back (RSS)."""
    if sys.platform != "darwin":
        return
    try:
        import ctypes

        ctypes.CDLL(None).malloc_zone_pressure_relief(None, 0)
    except (OSError, AttributeError):  # pragma: no cover - best effort
        pass


# ---- the indexer -------------------------------------------------------------------------------
class Indexer:
    """`start()` runs the queue (siglip, then people when consent is on) on a daemon thread;
    `run_blocking()` does the same inline (CLI). Work = images without a current embedding."""

    def __init__(
        self,
        shoot_dir: str | Path,
        settings: Settings,
        people: bool = True,
        embedder: Callable | None = None,
        images: bool = True,
        ocr: bool | None = None,
    ) -> None:
        self.dir = Path(shoot_dir)
        self.settings = settings
        self.people = people
        self.images = images  # the SigLIP task (off for an OCR-only run)
        self.ocr = settings.jersey_ocr if ocr is None else ocr  # T11, opt-in
        self.ocr_n = 0
        self.ocr_t = 0.0
        self._embedder_hook = embedder
        self._embedder = embedder  # test hook: list[RGB array] -> (N, 768)
        self._stop = threading.Event()
        self._user_pause = threading.Event()
        self._people_off = threading.Event()
        self._start_lock = threading.Lock()
        self._rerun = False
        self._exiting = False
        self._thread: threading.Thread | None = None
        self.task = "idle"
        self.done = self.total = 0
        self.error: str | None = None
        self.t_start = 0.0
        self.t_work = 0.0  # seconds spent actually embedding (excludes pauses)
        self.n_embedded = 0
        self.paused_batches = 0

    # -- control
    def start(self) -> None:
        with self._start_lock:
            if self._thread and self._thread.is_alive() and not self._exiting:
                self._user_pause.clear()
                self._rerun = True  # picked up by the worker before it exits (no lost wakeup)
                return
            self._stop.clear()
            self._user_pause.clear()
            self._people_off.clear()
            self._exiting = False
            self._rerun = False
            self._thread = threading.Thread(target=self._run, name="indexer", daemon=True)
            self._thread.start()

    def cancel_people(self) -> None:
        """Consent was revoked: skip / abandon the people task (the epoch guards writes)."""
        self._people_off.set()

    def pause(self) -> None:
        self._user_pause.set()

    def resume(self) -> None:
        self._user_pause.clear()

    def stop(self, join: bool = True) -> None:
        self._stop.set()
        self._user_pause.clear()
        if join and self._thread:
            self._thread.join(timeout=30)

    def join(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout)

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    @property
    def paused(self) -> bool:
        return self.running and (self._user_pause.is_set() or cull_running())

    def status(self) -> dict:
        img_s = self.n_embedded / self.t_work if self.t_work > 0 else 0.0
        ocr_s = self.ocr_n / self.ocr_t if self.ocr_t > 0 else 0.0
        if self.task == "ocr":
            img_s = ocr_s
        return {
            "task": self.task if self.running else "idle", "done": self.done, "total": self.total,
            "paused": self.paused, "img_s": round(img_s, 2), "error": self.error,
            "running": self.running, "ocr": self.ocr, "ocr_img_s": round(ocr_s, 2),
        }  # fmt: skip

    def run_blocking(self) -> None:
        self._stop.clear()
        self._run()

    # -- worker
    def _wait_if_paused(self, drop: Callable[[], None] | None = None) -> bool:
        """Block while a cull runs or the user paused; False when stopped. `drop` is called
        before the models are released so the caller can let go of its own references."""
        released = False
        while not self._stop.is_set():
            if not cull_running() and not self._user_pause.is_set():
                return True
            if not released:  # once per pause: the cull needs the RAM, resume reloads lazily
                if drop is not None:
                    drop()
                self._release_models()
                released = True
            self.paused_batches += 1
            time.sleep(0.1)
        return False

    def _run(self) -> None:
        while True:
            self._run_once()
            with self._start_lock:
                if self._rerun and not self._stop.is_set():
                    self._rerun = False
                    continue
                self._exiting = True
                return

    def _run_once(self) -> None:
        self.error = None
        try:
            if self.images:
                self.task = "siglip"
                self._index_images()
            if self.people and self.images and not self._stop.is_set():
                self.task = "people"
                self._index_people()
            if self.ocr and not self._stop.is_set():
                self.task = "ocr"
                self._index_ocr()
        except Exception as exc:  # noqa: BLE001 - surfaced through status()
            log.exception("indexer failed")
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.task = "idle"
            self._release_models()

    def _release_models(self) -> None:
        """Free SigLIP / OCR sessions when idle (peak RSS: they are not needed by the cull)."""
        import gc

        self._embedder = self._embedder_hook  # a test hook is not a model: keep it
        from shotsieve.models.ocr import release_ocr
        from shotsieve.models.text_image import release_vision

        release_vision()
        release_ocr()
        gc.collect()
        _return_freed_pages()

    def _get_embedder(self) -> Callable:
        if self._embedder is None:
            from shotsieve.models.text_image import get_vision

            vis = get_vision(self.settings.siglip_vision)
            vis.batch_size = self.settings.index_batch
            self._embedder = vis.embed_batch
        return self._embedder

    def _index_images(self) -> None:
        entries = [e for e in scan_folder(self.dir) if ".shotsieve" not in e.master.parts]
        with Cache.for_folder(self.dir) as cache:
            variant = self.settings.siglip_vision
            if cache.get_meta("siglip_variant") not in (None, variant):
                cache.clear_siglip()  # embeddings of another tower are not comparable
            cache.set_meta("siglip_variant", variant)
            fps: dict[str, str] = {}
            source: dict[str, Path] = {}
            for e in entries:
                try:
                    fps[str(e.master)] = fingerprint(e.preview_source)
                    source[str(e.master)] = e.preview_source
                except OSError:
                    continue
            todo = [p for p in fps if p not in cache.siglip_current(fps)]
            self.total, self.done = len(fps), len(fps) - len(todo)
            self.t_start = time.perf_counter()
            bs = max(1, self.settings.index_batch)
            workers = max(1, min(3, (os.cpu_count() or 2) // 2))
            max_edge = min(self.settings.preview_max_edge, 1280)
            with ThreadPoolExecutor(workers, thread_name_prefix="index-decode") as pool:

                def decode(p: str):
                    try:
                        return load_preview(source[p], max_edge)
                    except PreviewError as exc:
                        log.warning("index: %s", exc)
                        return None

                for i in range(0, len(todo), bs):
                    if not self._wait_if_paused():
                        return
                    batch = todo[i : i + bs]
                    t0 = time.perf_counter()
                    imgs = list(pool.map(decode, batch))
                    ok = [(p, im) for p, im in zip(batch, imgs, strict=True) if im is not None]
                    if ok:
                        vecs = self._get_embedder()([im for _, im in ok])
                        cache.put_siglip(
                            [(p, fps[p], v) for (p, _), v in zip(ok, vecs, strict=True)]
                        )
                    self.t_work += time.perf_counter() - t0
                    self.n_embedded += len(ok)
                    self.done += len(batch)

    def _index_people(self) -> None:
        from shotsieve.pipeline import people

        with Cache.for_folder(self.dir) as cache:
            if self._people_off.is_set() or not cache.people_consent(self.settings.people_consent):
                return
            if not self._wait_if_paused() or self._people_off.is_set():
                return
            people.cluster(cache)

    def _index_ocr(self) -> None:
        """Read jersey numbers (opt-in), then burst-vote and name face clusters from the roster."""
        from shotsieve.features import jersey as jf
        from shotsieve.models.face_detect import get_face_detector
        from shotsieve.models.ocr import get_detector, get_recognizer
        from shotsieve.pipeline import jerseys

        entries = [e for e in scan_folder(self.dir) if ".shotsieve" not in e.master.parts]
        with Cache.for_folder(self.dir) as cache:
            consent = cache.people_consent(self.settings.people_consent)
            epoch = cache.people_epoch()
            fps: dict[str, str] = {}
            source: dict[str, Path] = {}
            for e in entries:
                try:
                    fps[str(e.master)] = fingerprint(e.preview_source)
                    source[str(e.master)] = e.preview_source
                except OSError:
                    continue
            todo = [p for p in fps if p not in cache.ocr_current(fps)]
            self.total, self.done = len(fps), len(fps) - len(todo)
            if todo:
                held: dict[str, Any] = {}  # det/rec live here so a pause can drop them

                def ocr_models():
                    if not held:
                        held["det"] = get_detector(self.settings.ocr_threads)
                        held["rec"] = get_recognizer(self.settings.ocr_threads)
                    return held["det"], held["rec"]

                face_det = get_face_detector()
                face_emb: dict[str, Any] = {}
                person_of: dict[tuple[str, int], int] = {}
                embedder = None
                if consent:
                    from shotsieve.models.face_embed import get_face_embedder

                    embedder = get_face_embedder()
                    face_emb = cache.face_embeddings()
                    with cache._lock:
                        for pid, path, idx in cache._con.execute(
                            "SELECT person_id, path, face_idx FROM person_faces"
                        ):
                            person_of[(cache._abs(path), idx)] = pid
                workers = max(1, min(3, (os.cpu_count() or 2) // 2))
                max_edge = min(self.settings.preview_max_edge, 1280)

                def decode(p: str):
                    try:
                        return load_preview(source[p], max_edge)
                    except PreviewError as exc:
                        log.warning("ocr: %s", exc)
                        return None

                with ThreadPoolExecutor(workers, thread_name_prefix="ocr-decode") as pool:
                    futs = {p: pool.submit(decode, p) for p in todo[: workers * 2]}
                    for i, p in enumerate(todo):
                        if not self._wait_if_paused(held.clear):
                            return
                        nxt = i + workers * 2
                        if nxt < len(todo):
                            futs[todo[nxt]] = pool.submit(decode, todo[nxt])
                        img = futs.pop(p).result()
                        t0 = time.perf_counter()
                        if img is not None:
                            rows = self._ocr_one(  # no det/rec locals: they would outlive a pause
                                img, p, jf, *ocr_models(), face_det, embedder, face_emb, person_of
                            )
                            cache.put_jerseys(p, fps[p], rows, epoch)
                            self.ocr_n += 1
                        self.ocr_t += time.perf_counter() - t0
                        self.done += 1
            jerseys.finalize(cache, consent and not self._people_off.is_set(), epoch)

    def _ocr_one(self, img, path, jf, det, rec, face_det, embedder, face_emb, person_of):
        h, w = img.shape[:2]
        detected = face_det.detect(img, max_faces=self.settings.max_faces)
        faces = [f.bbox for f in detected]
        rows = []
        for r in jf.read_numbers(img, det, rec, faces):
            face_idx = person_id = None
            fi = jf.associate_face(r.box, faces)
            if fi is not None and embedder is not None and path in face_emb:
                from shotsieve.models.face_embed import cosine

                vec = embedder.embed(img, detected[fi])
                sims = [cosine(vec, v) for v in face_emb[path]]
                if sims and max(sims) >= 0.8:
                    face_idx = int(np.argmax(sims))
                    person_id = person_of.get((path, face_idx))
            x, y, bw, bh = r.box
            rows.append({
                "number": r.number, "conf": r.conf, "x": x / w, "y": y / h, "w": bw / w,
                "h": bh / h, "face_idx": face_idx, "person_id": person_id,
            })  # fmt: skip
        return rows
