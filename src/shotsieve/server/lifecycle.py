"""`data/server.json`: how the launcher finds a running server (T18)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from shotsieve import __version__
from shotsieve.config import data_dir


def server_json_path() -> Path:
    return data_dir() / "server.json"


def write_server_json(port: int) -> Path:
    path = server_json_path()
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps({"pid": os.getpid(), "port": port, "version": __version__,
                    "started": time.time()})
    )
    tmp.replace(path)
    return path


def remove_server_json_if_ours() -> bool:
    path = server_json_path()
    try:
        if json.loads(path.read_text()).get("pid") != os.getpid():
            return False
        path.unlink()
        return True
    except (OSError, ValueError):
        return False
