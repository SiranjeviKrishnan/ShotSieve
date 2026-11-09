"""Card ingest: copy to primary (+ backup) with streaming blake2b verification."""

from __future__ import annotations

import hashlib
import os
import tempfile
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from shotsieve.io.exif import read_meta
from shotsieve.io.formats import EXCLUDED_DIRS, is_supported

CHUNK = 4 * 1024 * 1024


@dataclass
class IngestReport:
    copied: list[Path] = field(default_factory=list)
    skipped: list[Path] = field(default_factory=list)
    failed: list[tuple[Path, str]] = field(default_factory=list)
    bytes_copied: int = 0
    seconds: float = 0.0


def _bypass_page_cache(fd: int) -> None:
    """Best effort: make reads come from the medium, not the page cache (macOS F_NOCACHE,
    Linux POSIX_FADV_DONTNEED), so a verify pass can notice bytes that never landed."""
    try:
        import fcntl

        if hasattr(fcntl, "F_NOCACHE"):
            fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
    except (ImportError, OSError):
        pass
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass


def _hash_file(path: Path, nocache: bool = False) -> str:
    h = hashlib.blake2b()
    with open(path, "rb") as fh:
        if nocache:
            _bypass_page_cache(fh.fileno())
        while chunk := fh.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _suffixed(path: Path, n: int) -> Path:
    return path.with_name(f"{path.stem}_{n}{path.suffix}")


def _claim(target: Path, size: int, claimed: set[str]) -> Path:
    """First of target, target_1, target_2 ... that no earlier source claimed and that does not
    already hold a different file (a same-size file is left to the hash check so re-running an
    ingest recognises its own earlier copies)."""
    cand, n = target, 0
    while True:
        key = str(cand).casefold()  # card readers and APFS/NTFS volumes ignore case
        if key not in claimed and not (cand.exists() and cand.stat().st_size != size):
            claimed.add(key)
            return cand
        n += 1
        cand = _suffixed(target, n)


def _iter_sources(src: Path) -> Iterator[Path]:
    for dirpath, dirnames, filenames in os.walk(src):
        dirnames[:] = sorted(
            d for d in dirnames if not d.startswith(".") and d not in EXCLUDED_DIRS
        )
        for f in sorted(filenames):
            if not f.startswith(".") and is_supported(f):
                yield Path(dirpath, f)


def _target(
    path: Path, src: Path, dest: Path, by_date: bool, template: str | None, idx: int
) -> Path:
    rel = path.relative_to(src)
    name = rel.name
    day = None
    if by_date or template:
        day = datetime.fromtimestamp(read_meta(path).timestamp)
    if template:
        name = template.format(
            stem=path.stem,
            ext=path.suffix.lstrip("."),
            name=path.name,
            index=idx,
            date=day.strftime("%Y-%m-%d"),
            time=day.strftime("%H%M%S"),
        )
        if "." not in name:
            name += path.suffix
    if by_date:
        return dest / day.strftime("%Y-%m-%d") / name
    return dest / rel.parent / name


def _copy_one(src_file: Path, targets: list[Path]) -> tuple[str, int]:
    """Stream-copy to every target; returns (status, bytes). Raises on any mismatch."""
    size = src_file.stat().st_size
    states = []
    for t in targets:
        states.append(t.exists() and t.stat().st_size == size)
    h = hashlib.blake2b()
    with open(src_file, "rb") as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
    digest = h.hexdigest()

    todo = []
    for t, present in zip(targets, states, strict=True):
        if present and _hash_file(t) == digest:
            continue
        if t.exists():
            raise FileExistsError(f"{t} exists with different content; not overwriting")
        todo.append(t)
    if not todo:
        return "skipped", 0

    for t in todo:
        t.parent.mkdir(parents=True, exist_ok=True)
    # unique temp names: two jobs (or two cards) never share one `.part` file
    parts = []
    for t in todo:
        fd, tmp = tempfile.mkstemp(dir=t.parent, prefix=f".{t.name}.", suffix=".part")
        os.close(fd)
        parts.append((t, Path(tmp)))
    outs = [open(p, "wb") for _, p in parts]
    h2 = hashlib.blake2b()
    try:
        with open(src_file, "rb") as fh:
            while chunk := fh.read(CHUNK):
                h2.update(chunk)
                for o in outs:
                    o.write(chunk)
        for o in outs:
            o.flush()
            os.fsync(o.fileno())
    finally:
        for o in outs:
            o.close()
    try:
        if h2.hexdigest() != digest:
            raise OSError(f"{src_file} changed while copying")
        for t, p in parts:
            if _hash_file(p, nocache=True) != digest:
                raise OSError(f"verify failed for {t}")
    except OSError:
        for _, p in parts:
            p.unlink(missing_ok=True)
        raise
    for t, p in parts:
        if t.exists():  # lost a race with another writer: never overwrite
            p.unlink(missing_ok=True)
            raise FileExistsError(f"{t} appeared while copying; not overwriting")
        os.replace(p, t)
    return "copied", size * len(todo)


def ingest(
    src: str | Path,
    dest: str | Path,
    backup: str | Path | None = None,
    rename_template: str | None = None,
    by_date: bool = False,
    workers: int = 4,
) -> IngestReport:
    """Copy supported files from `src` to `dest` (and `backup`), verifying blake2b.

    Existing identical files are skipped; existing files with different content fail rather
    than being overwritten. Sources that map to the same target name (e.g. 100CANON/IMG_0001 and
    101CANON/IMG_0001 with `by_date`) get a `_1`, `_2` suffix; the source is never
    modified. Template fields: {stem} {ext} {name} {index} {date} {time}.
    """
    t0 = time.perf_counter()
    src, dest = Path(src), Path(dest)
    report = IngestReport()
    jobs = []
    claimed: set[str] = set()
    for i, f in enumerate(_iter_sources(src), 1):
        primary = _claim(
            _target(f, src, dest, by_date, rename_template, i), f.stat().st_size, claimed
        )
        targets = [primary]
        if backup is not None:
            targets.append(Path(backup) / primary.relative_to(dest))
        jobs.append((f, targets))

    def run(job: tuple[Path, list[Path]]) -> tuple[Path, str, int, str]:
        f, targets = job
        try:
            status, n = _copy_one(f, targets)
            return f, status, n, ""
        except Exception as exc:  # noqa: BLE001
            return f, "failed", 0, str(exc)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for f, status, n, err in pool.map(run, jobs):
            if status == "copied":
                report.copied.append(f)
                report.bytes_copied += n
            elif status == "skipped":
                report.skipped.append(f)
            else:
                report.failed.append((f, err))
    report.seconds = time.perf_counter() - t0
    return report
