"""Isolated-process OCR memory measurement (B2.2).

Fresh process: import, record baseline RSS, load the detector + recogniser, read 50 bench frames
(decoded at the indexer's preview size), report peak RSS minus baseline. Run it several times
for a stable number; options override the settings under test without touching the code:

    uv run python scripts/bench_ocr_rss.py [--det-edge 736] [--mem-pattern] [--arena] [--n 50]
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def measure(folder: Path, n: int, det_edge: int | None, threads: int = 4) -> dict:
    from shotsieve.config import Settings
    from shotsieve.features import jersey as jf
    from shotsieve.io.preview import load_preview
    from shotsieve.models import ocr

    if det_edge:
        ocr.DET_LONG_EDGE = det_edge
    real = sorted((ROOT / "tests" / "fixtures" / "real").glob("cc0_football_*.jpg"))
    files = (real + sorted(folder.glob("*.jpg")))[:n]  # 3 real jersey photos + bench frames
    max_edge = min(Settings().preview_max_edge, 1280)
    proc = psutil.Process()
    base = proc.memory_info().rss / 1e6
    peak = [base]
    stop = threading.Event()

    def sample() -> None:
        while not stop.wait(0.01):
            peak[0] = max(peak[0], proc.memory_info().rss / 1e6)

    th = threading.Thread(target=sample, daemon=True)
    th.start()
    det, rec = ocr.get_detector(threads), ocr.get_recognizer(threads)
    n_read = 0
    for f in files:
        img = load_preview(f, max_edge)
        n_read += len(jf.read_numbers(img, det, rec, []))
    stop.set()
    th.join()
    end = proc.memory_info().rss / 1e6
    return {
        "det_long_edge": ocr.DET_LONG_EDGE, "frames": len(files), "numbers": n_read,
        "baseline_mb": round(base), "peak_mb": round(peak[0]), "end_mb": round(end),
        "added_mb": round(peak[0] - base), "threads": threads,
    }  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=ROOT / "data" / "bench" / "shoot_300_7")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--det-edge", type=int, default=None)
    ap.add_argument("--threads", type=int, default=4, help="ORT intra-op threads (indexer: 4)")
    args = ap.parse_args(argv)
    res = measure(args.dir, args.n, args.det_edge, args.threads)
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
