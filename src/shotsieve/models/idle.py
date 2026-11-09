"""A lazily loaded model that unloads itself after a quiet period (RSS budget, T22).

`holder.use()` is a context manager returning the loaded object; the object cannot be unloaded
while any `use()` is open. After `idle_s` seconds without use a daemon timer drops it and runs the
garbage collector. The sessions are really freed, but on macOS the freed ORT heap is NOT handed
back to the OS: the process RSS keeps a plateau (about +300..360 MB after crop's models, reused by
the next load, no unbounded growth). Review #12 amended the budget accordingly (retained <= +400 MB,
stable across load/unload cycles, whole-process peak <= 3 GB, all PASS/FAIL lines in the bench)."""

from __future__ import annotations

import gc
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Generic, TypeVar

T = TypeVar("T")


def _trim() -> None:
    """Collect garbage and ask macOS to release freed heap pages (best effort: it usually keeps
    them resident, see the module docstring)."""
    gc.collect()
    if sys.platform == "darwin":
        try:
            import ctypes

            ctypes.CDLL(None).malloc_zone_pressure_relief(None, 0)
        except Exception:  # noqa: BLE001 - best effort
            pass


class IdleLoaded(Generic[T]):
    def __init__(
        self, factory: Callable[[], T], idle_s: Callable[[], float] | float = 120.0
    ) -> None:
        self._factory = factory
        self._idle_s = idle_s if callable(idle_s) else (lambda: float(idle_s))
        self._lock = threading.RLock()
        self._obj: T | None = None
        self._busy = 0
        self._last = 0.0
        self._timer: threading.Timer | None = None

    @property
    def loaded(self) -> bool:
        return self._obj is not None

    def _arm(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
        delay = max(0.05, self._idle_s())
        self._timer = threading.Timer(delay, self._maybe_unload)
        self._timer.daemon = True
        self._timer.start()

    def _maybe_unload(self) -> None:
        with self._lock:
            if self._obj is None:
                return
            quiet = time.monotonic() - self._last
            if self._busy or quiet < self._idle_s() - 0.02:
                self._arm()
                return
            self._obj = None
            self._timer = None
        _trim()

    @contextmanager
    def use(self) -> Iterator[T]:
        with self._lock:
            if self._obj is None:
                self._obj = self._factory()
            self._busy += 1
            obj = self._obj
        try:
            yield obj
        finally:
            with self._lock:
                self._busy -= 1
                self._last = time.monotonic()
                self._arm()

    def unload(self) -> None:
        with self._lock:
            if self._busy:
                return
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            self._obj = None
        _trim()
