"""T23g: automatic XMP sidecars. ON by default for every shoot (standing consent; the per-shoot
meta `auto_xmp` = "0" turns it off). After a cull and after every user change the sidecars are
written in a background thread, debounced and batched, through the SAME writers and safety
rules as before
(user-wins, one .bak per content hash, atomic write, sidecar lock, crop catalog / rotation skips,
originals never touched). Nothing here can change a decision: it only reads the committed state."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

from shotsieve.io.formats import split_virtual
from shotsieve.pipeline.cache import Cache, cache_path

log = logging.getLogger("shotsieve")
DELAY = 0.8  # seconds of calm before a batch is written
RETRY = 3.0  # a cull / re-decision is running: try again later


class AutoXmp:
    def __init__(self, state: Any, delay: float = DELAY) -> None:
        self.state = state
        self.delay = delay
        self._lock = threading.Lock()
        self._run_lock = threading.Lock()  # one writer at a time
        self._pending: dict[Path, set[str] | None] = {}  # None = every image
        self._timers: dict[Path, threading.Timer] = {}
        self._status: dict[Path, dict[str, Any]] = {}

    # -- setting ---------------------------------------------------------------------------
    def enabled(self, f: Path) -> bool:
        if not cache_path(f).exists():
            return True
        with Cache.for_folder(f) as c:
            return c.get_meta("auto_xmp") != "0"

    def set_enabled(self, f: Path, on: bool) -> None:
        with Cache.for_folder(f) as c:
            c.set_meta("auto_xmp", "1" if on else "0")
        if not on:
            with self._lock:
                self._pending.pop(f, None)
                t = self._timers.pop(f, None)
            if t is not None:
                t.cancel()
        else:
            self.request(f, None)

    # -- requests --------------------------------------------------------------------------
    def request(self, f: Path, paths: set[str] | None) -> None:
        """Queue sidecar writes for `paths` (None = everything). No-op when automatic XMP is off."""
        try:
            if not self.enabled(f):
                return
        except Exception:  # noqa: BLE001
            return
        with self._lock:
            cur = self._pending.get(f, set())
            if paths is None or cur is None:
                self._pending[f] = None
            else:
                self._pending[f] = cur | set(paths)
            t = self._timers.pop(f, None)
            if t is not None:
                t.cancel()
            t = threading.Timer(self.delay, self._fire, (f,))
            t.daemon = True
            self._timers[f] = t
            t.start()

    def _fire(self, f: Path) -> None:
        with self._lock:
            self._timers.pop(f, None)
        self._write(f)

    def flush(self, f: Path) -> None:
        """Write what is pending for `f` now and wait for it (Open in Lightroom)."""
        with self._lock:
            t = self._timers.pop(f, None)
        if t is not None:
            t.cancel()
        self._write(f)

    def pending(self, f: Path) -> bool:
        with self._lock:
            return f in self._pending

    # -- the writer ------------------------------------------------------------------------
    def _write(self, f: Path) -> None:
        with self._run_lock:
            if self.state.busy_with(f):  # decisions are still being made: write afterwards
                with self._lock:
                    if f in self._pending and f not in self._timers:
                        t = threading.Timer(RETRY, self._fire, (f,))
                        t.daemon = True
                        self._timers[f] = t
                        t.start()
                return
            with self._lock:
                if f not in self._pending:
                    return
                only = self._pending.pop(f)
            try:
                from shotsieve.server.lightroom import prepare_xmp

                with Cache.for_folder(f) as c:
                    allp = [p for p in c.load_decisions() if not split_virtual(p)]
                    paths = allp if only is None else [p for p in allp if p in only]
                    ids = c.ids_for(paths)
                mapping = {ids[p]: p for p in paths if p in ids}
                res = prepare_xmp(f, list(mapping), mapping, self.state.settings)
                crops = res["crops"]
                skipped = (res.get("kept_user_rating", 0) + res.get("kept_user_label", 0)
                           + res.get("kept_lr", 0) + sum(crops["skipped"].values()))
                self._status[f] = {
                    "written": res["decisions"], "skipped": skipped,
                    "failed": res["failed"] + crops["failed"], "warnings": res["warnings"],
                    "at": time.time(),
                }
            except Exception as exc:  # noqa: BLE001 - never raise into a request or the GUI
                log.warning("automatic xmp write failed for %s: %s", f, exc)
                self._status[f] = {"written": 0, "skipped": 0, "failed": 1,
                                   "warnings": [f"{type(exc).__name__}: {exc}"], "at": time.time()}

    def status(self, f: Path) -> dict[str, Any]:
        s = dict(self._status.get(f, {"written": 0, "skipped": 0, "failed": 0, "warnings": [],
                                      "at": None}))
        s["auto"] = self.enabled(f)
        s["pending"] = self.pending(f)
        return s
