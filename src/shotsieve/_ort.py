"""Import onnxruntime without it writing outside the ShotSieve folder (T18 footprint rule).

onnxruntime >= 1.31 creates `~/Library/Application Support/Microsoft/DeveloperTools/.onnxruntime`
(a device id plus an event store for its telemetry) as a side effect of `import onnxruntime`. We
import it with HOME pointed at `data/ort-home` (inside the install) and then switch its telemetry
events off. HOME is restored immediately. While the import runs, other threads must not see the
redirect: they ask `user_home()`, which answers with the real HOME during a swap. The server also
imports onnxruntime once at start, before any request thread exists (`serve`).
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

_lock = threading.Lock()
_real_home: str | None = None  # the user's HOME while the swap is active, else None


def user_home() -> Path:
    """The user's home directory, immune to the temporary HOME swap of `import_onnxruntime`."""
    a = _real_home
    h = os.path.expanduser("~")
    b = _real_home
    return Path(a or b or h)


def import_onnxruntime():
    if "onnxruntime" in sys.modules:
        return sys.modules["onnxruntime"]
    with _lock:
        if "onnxruntime" in sys.modules:
            return sys.modules["onnxruntime"]
        global _real_home
        old = os.environ.get("HOME")
        _real_home = os.path.expanduser("~")
        try:
            from shotsieve.config import data_dir

            home = data_dir() / "ort-home"
            home.mkdir(parents=True, exist_ok=True)
            os.environ["HOME"] = str(home)
        except OSError:
            pass  # unwritable data dir: import as is (doctor reports it)
        try:
            import onnxruntime as ort
        finally:
            if old is None:
                os.environ.pop("HOME", None)
            else:
                os.environ["HOME"] = old
            _real_home = None
        try:
            ort.disable_telemetry_events()
        except Exception:  # noqa: BLE001 - older builds have no such switch
            pass
        return ort
