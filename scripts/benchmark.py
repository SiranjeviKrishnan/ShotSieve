# ruff: noqa: E501
"""Throughput + peak-RSS benchmark on a synthetic shoot.

Synthesises N JPEGs (fixture portraits composited onto varied backgrounds, burst structure,
a few blur/dark/black variants), runs the full cull cold, then again with a warm cache, and
writes data/benchmark.md.

    uv run python scripts/benchmark.py --n 300
    uv run python scripts/benchmark.py --raw --n 40    # copies of the fixture DNGs
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shotsieve.config import Settings, data_dir  # noqa: E402

GEN = ROOT / "tests" / "fixtures" / "generated"
CANVAS = (2560, 1707)  # typical camera JPEG preview class (decoded at 1/2 scale)


def _make_fixtures():
    spec = importlib.util.spec_from_file_location(
        "make_fixtures", ROOT / "scripts" / "make_fixtures.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_fixtures"] = mod
    spec.loader.exec_module(mod)
    return mod


def _background(rng: np.random.Generator, building: np.ndarray) -> np.ndarray:
    w, h = CANVAS
    kind = rng.integers(0, 3)
    if kind == 0:  # colour gradient
        a, b = rng.integers(20, 235, 3), rng.integers(20, 235, 3)
        t = np.linspace(0, 1, w, dtype=np.float32)[None, :, None]
        bg = (a * (1 - t) + b * t) * np.ones((h, 1, 1), np.float32)
        bg = bg + rng.normal(0, 3, bg.shape)
    elif kind == 1:  # blurred crop of the building photo
        bh, bw = building.shape[:2]
        x0, y0 = rng.integers(0, bw // 3), rng.integers(0, bh // 3)
        crop = building[y0 : y0 + bh * 2 // 3, x0 : x0 + bw * 2 // 3]
        bg = cv2.GaussianBlur(cv2.resize(crop, CANVAS), (0, 0), float(rng.uniform(4, 14)))
    else:  # low-frequency noise
        small = rng.integers(40, 220, (9, 14, 3)).astype(np.uint8)
        bg = cv2.GaussianBlur(cv2.resize(small, CANVAS, interpolation=cv2.INTER_CUBIC), (0, 0), 20)
    return np.clip(bg, 0, 255).astype(np.uint8)


def _compose(
    bg: np.ndarray, face_img: np.ndarray, scale: float, cx: float, cy: float
) -> np.ndarray:
    w, h = CANVAS
    fw = int(w * scale)
    fh = int(face_img.shape[0] * fw / face_img.shape[1])
    fh = min(fh, h - 4)
    patch = cv2.resize(face_img, (fw, fh), interpolation=cv2.INTER_CUBIC)
    x0 = int(np.clip(cx * w - fw / 2, 0, w - fw))
    y0 = int(np.clip(cy * h - fh / 2, 0, h - fh))
    out = bg.copy()
    mask = np.zeros((fh, fw), np.float32)
    m = max(2, min(fh, fw) // 25)
    mask[m:-m, m:-m] = 1.0
    mask = cv2.GaussianBlur(mask, (0, 0), m / 2)[..., None]
    roi = out[y0 : y0 + fh, x0 : x0 + fw].astype(np.float32)
    out[y0 : y0 + fh, x0 : x0 + fw] = (patch * mask + roi * (1 - mask)).astype(np.uint8)
    return out


TRUTH_FILE = "ground_truth.json"
HARD_CODES = frozenset(
    {"blur", "face_out_of_focus", "eyes_closed", "exposure_bad", "flash_misfire", "accidental"}
)


def synthesize(n: int, out: Path, seed: int = 7) -> list[dict]:
    """Write n JPEGs (+EXIF) into `out` and `ground_truth.json` next to them.

    Ground truth per frame: {"file", "moment", "degraded"} where `degraded` is "" for a clean
    frame and "blur" / "dark" / "black" for the frames deliberately damaged below.
    """
    from datetime import datetime

    mf = _make_fixtures()
    src = GEN / "src"
    if not (src / "lena.jpg").exists():
        mf.main([])
    faces = [
        mf.load_rgb(src / nme)
        for nme in ("lena.jpg", "messi5.jpg", "portrait2.jpg", "portrait3.jpg")
    ]
    faces.append(mf.load_rgb(src / "group_rgb.jpg"))
    building = mf.load_rgb(src / "building.jpg")
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    t = datetime(2024, 6, 1, 14, 0, 0).timestamp()
    i = 0
    truth: list[dict] = []
    moment = 0
    while i < n:
        burst = int(rng.integers(1, 7)) if rng.random() < 0.7 else 1
        burst = min(burst, n - i)
        bg = _background(rng, building)
        face = faces[int(rng.integers(0, len(faces)))]
        scale = float(rng.uniform(0.28, 0.5))
        cx, cy = rng.uniform(0.3, 0.7), rng.uniform(0.4, 0.6)
        flash = int(rng.choice([0x1, 0x10]))
        for k in range(burst):
            jx, jy = rng.normal(0, 0.004, 2)
            frame = _compose(bg, face, scale * (1 + rng.normal(0, 0.004)), cx + jx, cy + jy)
            frame = cv2.add(
                frame, rng.normal(0, 2, frame.shape).astype(np.int16).clip(0, 255).astype(np.uint8)
            )
            r = rng.random()
            degraded = ""
            if r < 0.05:
                frame, degraded = cv2.GaussianBlur(frame, (0, 0), 7), "blur"
            elif r < 0.08:
                frame, degraded = mf.adjust_ev(frame, -3), "dark"
            elif r < 0.09:
                frame, degraded = np.zeros_like(frame), "black"
            ts = t + 0.25 * k
            mf.save_jpeg(frame, out / f"IMG_{i + k:05d}.jpg", mf.exif_bytes(ts, flash))
            truth.append({"file": f"IMG_{i + k:05d}.jpg", "moment": moment, "degraded": degraded})
        i += burst
        moment += 1
        t += float(rng.uniform(5, 90))
    (out / TRUTH_FILE).write_text(json.dumps(truth))
    return truth


def score_truth(images, truth: list[dict]) -> dict:
    """Judge a cull against the synthesis ground truth.

    - `clean_hard_rejects`: clean (never degraded) frames rejected for a hard-fail reason
      (target 0). `clean_rejects` lists every clean frame rejected, by reason.
    - moment-level miss: a moment holding at least one clean frame must end with a clean frame
      kept. `moment_no_keep` counts the strict form (review counts as a miss, target <= 1 %),
      `moment_no_keep_no_review` the lenient one (a review at least lets the user rescue it).
    """
    by_file = {Path(r.master_path).name: r for r in images}
    clean = [t for t in truth if not t["degraded"] and t["file"] in by_file]
    hard, rejected = [], []
    for t in clean:
        r = by_file[t["file"]]
        if r.decision == "reject":
            rejected.append((t["file"], list(r.reasons)))
            if set(r.reasons) & HARD_CODES:
                hard.append((t["file"], sorted(set(r.reasons) & HARD_CODES)))
    moments: dict[int, list[str]] = {}
    for t in clean:
        moments.setdefault(t["moment"], []).append(by_file[t["file"]].decision)
    group_keep: dict[int, bool] = {}
    group_save: dict[int, bool] = {}  # keep OR review: the moment can still be rescued
    for r in images:
        group_keep[r.group_id] = group_keep.get(r.group_id, False) or r.decision == "keep"
        group_save[r.group_id] = group_save.get(r.group_id, False) or (
            r.decision == "keep" or (r.decision == "review" and not getattr(r, "audit", False))
        )
    moment_groups: dict[int, set[int]] = {}
    for t in clean:
        moment_groups.setdefault(t["moment"], set()).add(by_file[t["file"]].group_id)
    group_no_keep = sum(not any(group_keep[g] for g in gs) for gs in moment_groups.values())
    group_lost = sum(not any(group_save[g] for g in gs) for gs in moment_groups.values())
    no_keep = sum("keep" not in d for d in moments.values())
    no_keep_no_review = sum("keep" not in d and "review" not in d for d in moments.values())
    nm = max(len(moments), 1)
    return {
        "n_clean": len(clean),
        "n_degraded": sum(1 for t in truth if t["degraded"]),
        "clean_hard_rejects": len(hard),
        "clean_hard_reject_detail": hard,
        "clean_rejects": len(rejected),
        "n_moments": len(moments),
        "moment_no_keep": no_keep,
        "moment_no_keep_rate": no_keep / nm,
        "moment_group_no_keep": group_no_keep,
        "moment_group_no_keep_rate": group_no_keep / max(len(moments), 1),
        "moment_group_lost": group_lost,
        "moment_group_lost_rate": group_lost / max(len(moments), 1),
        "moment_no_keep_no_review": no_keep_no_review,
        "moment_no_keep_no_review_rate": no_keep_no_review / nm,
    }


STAGES: list[tuple[str, float, float]] = []  # (stage, peak MB during, RSS MB at end)
_stage_peak = [0.0]


def _rss_sampler(stop: threading.Event, peak: list[float]) -> None:
    proc = psutil.Process()
    while not stop.wait(0.05):
        r = proc.memory_info().rss / 1e6
        peak[0] = max(peak[0], r)
        _stage_peak[0] = max(_stage_peak[0], r)


def mark(name: str) -> None:
    """Record the peak RSS since the previous mark and the RSS now (idle, after the stage)."""
    now = psutil.Process().memory_info().rss / 1e6
    STAGES.append((name, max(_stage_peak[0], now), now))
    _stage_peak[0] = now


def stage_lines() -> list[str]:
    out = ["## RSS per stage (peak during stage / RSS at end of stage, MB)", "",
           "| Stage | Peak | End |", "|---|---|---|"]
    out += [f"| {n} | {p:.0f} | {e:.0f} |" for n, p, e in STAGES]
    return [*out, ""]


def phase2_lines(cold, gt) -> list[str]:
    import math

    imgs = cold.images
    n = len(imgs)
    band = [r for r in imgs if "banding" in r.reasons]
    af = [r for r in imgs if "af_missed" in r.reasons]
    measured = sum(not math.isnan(r.features.banding_amp_ev) for r in imgs)
    af_known = sum(r.features.af_known >= 1.0 for r in imgs)
    t = cold.stats["timings"]
    return [
        f"- Banding measured on {measured}/{n} frames; **flagged {len(band)}** "
        f"({100 * len(band) / max(n, 1):.1f} %, target <= 1 % false-flag); "
        f"clean frames hard-rejected overall: {gt['clean_hard_rejects']} (target 0)",
        f"- AF maker notes known on {af_known}/{n} frames (synthetic JPEGs carry none); "
        f"`af_missed` flagged {len(af)}",
        f"- Clock-offset step: {t.get('clock_s', 0.0) * 1000:.1f} ms (single body: skipped)",
    ]


def t10_lines(t10: dict | None, peak_cull: float, peak_all: float) -> list[str]:
    if t10 is None:
        return ["## T10 search (skipped: search models not downloaded)", ""]
    ok_c = t10["cull_imgs_per_s"] >= 10.0 and t10["embedded_during_cull"] == 0
    ok_r = peak_all <= 3000
    return [
        "## T10 indexer + search",
        "",
        f"- **Cull with indexer enabled-but-paused (work pending): "
        f"{t10['cull_imgs_per_s']:.1f} img/s** (target >= 10) {'PASS' if ok_c else 'FAIL'}; "
        f"images embedded while the cull ran: {t10['embedded_during_cull']} (paused="
        f"{t10['paused']})",
        f"- Index drain after the cull: {t10['indexed']} in {t10['index_wall_s']:.1f} s = "
        f"**{t10['index_imgs_per_s']:.1f} img/s** (target >= 5; error: {t10['index_error']})",
        f"- Query latency: cold (text tower load) {t10['query_cold_s']:.2f} s (target < 3), "
        f"warm median {t10['query_warm_s'] * 1000:.0f} ms (target < 300)",
        f"- **Peak RSS** cull only {peak_cull:.0f} MB; with indexer + text query "
        f"**{peak_all:.0f} MB** (target <= 3000) {'PASS' if ok_r else 'FAIL'}",
        "",
    ]


def t11_lines(t11: dict | None, peak_all: float) -> list[str]:
    if t11 is None:
        return ["## T11 jersey OCR (skipped: OCR models not downloaded)", ""]
    ok_c = t11["cull_imgs_per_s"] >= 10.0 and t11["read_during_cull"] == 0
    return [
        "## T11 jersey OCR (opt-in)",
        "",
        f"- **Cull with OCR OFF (default): {t11['cull_off_imgs_per_s']:.1f} img/s** "
        f"(target >= 10) {'PASS' if t11['cull_off_imgs_per_s'] >= 10 else 'FAIL'}",
        f"- Cull with OCR enabled-but-paused: {t11['cull_imgs_per_s']:.1f} img/s; images read "
        f"while the cull ran: {t11['read_during_cull']} {'PASS' if ok_c else 'FAIL'}",
        f"- **OCR-ON indexer drain: {t11['done']} in {t11['wall_s']:.1f} s = "
        f"{t11['ocr_imgs_per_s']:.1f} img/s** (target >= 8; error: {t11['error']}); "
        f"numbers found on the (jersey-free) synthetic frames: {t11['n_numbers']}",
        f"- Peak RSS during OCR {peak_all:.0f} MB (target <= 3000); OCR models added "
        f"+{t11['rss_added_mb']:.0f} MB over the pre-OCR level",
        "",
    ]


def run_t11(folder: Path, settings: Settings, peak: list[float]) -> dict | None:
    """Cull throughput with OCR off and with the OCR task enabled-but-paused, then the OCR-only
    indexer drain rate."""
    import dataclasses

    from shotsieve.models.ocr import ocr_models_ready
    from shotsieve.pipeline.cache import Cache
    from shotsieve.pipeline.indexer import Indexer, cull_active
    from shotsieve.pipeline.runner import run_cull

    if not ocr_models_ready():
        return None
    out: dict = {}
    shutil.rmtree(folder / ".shotsieve", ignore_errors=True)
    off = dataclasses.replace(settings, jersey_ocr=False)
    out["cull_off_imgs_per_s"] = run_cull(folder, off).stats["imgs_per_s"]
    shutil.rmtree(folder / ".shotsieve", ignore_errors=True)
    on = dataclasses.replace(settings, jersey_ocr=True)
    ix = Indexer(folder, on, people=False, images=False, ocr=True)
    with cull_active():
        ix.start()
        res = run_cull(folder, on)
        out["cull_imgs_per_s"] = res.stats["imgs_per_s"]
        out["read_during_cull"] = ix.ocr_n
    before = psutil.Process().memory_info().rss / 1e6
    t0 = time.perf_counter()
    top = before
    while ix.running and time.perf_counter() - t0 < 900:
        top = max(top, psutil.Process().memory_info().rss / 1e6)
        time.sleep(0.05)
    out["wall_s"] = time.perf_counter() - t0
    st = ix.status()
    out["ocr_imgs_per_s"] = st["ocr_img_s"]
    out["done"] = f"{st['done']}/{st['total']}"
    out["error"] = st["error"]
    out["rss_added_mb"] = max(0.0, top - before)
    with Cache.for_folder(folder) as c:
        out["n_numbers"] = len(c.jerseys())
    return out


RAW_MARK = "## RAW benchmark"
RAW_SRC = GEN / "raw"


def run_t10(folder: Path, settings: Settings, peak: list[float]) -> dict | None:
    """Cull throughput with the indexer enabled but paused (work pending), then the index
    drain rate and query latency; the shared RSS sampler covers all of it."""
    from shotsieve.models.text_image import get_text, search_models_ready
    from shotsieve.pipeline.indexer import Indexer, cull_active
    from shotsieve.pipeline.runner import run_cull
    from shotsieve.pipeline.search import search

    if not search_models_ready():
        return None
    shutil.rmtree(folder / ".shotsieve", ignore_errors=True)
    out: dict = {}
    ix = Indexer(folder, settings, people=False)
    with cull_active():  # a cull is running: the indexer must stay paused throughout
        ix.start()
        res = run_cull(folder, settings)
        out["cull_imgs_per_s"] = res.stats["imgs_per_s"]
        out["embedded_during_cull"] = ix.n_embedded
        out["paused"] = ix.paused
    out["peak_after_cull_mb"] = peak[0]
    t0 = time.perf_counter()
    ix.join(600)
    out["index_wall_s"] = time.perf_counter() - t0
    st = ix.status()
    out["index_imgs_per_s"] = st["img_s"]
    out["indexed"] = f"{st['done']}/{st['total']}"
    out["index_error"] = st["error"]
    lat = []
    for i, qtext in enumerate(["a person smiling", "a red car", "a blurry photo", "a building"]):
        t = time.perf_counter()
        hits = search(folder, qtext, 100)
        lat.append(time.perf_counter() - t)
        out["n_hits_" + str(i)] = len(hits)
    out["query_cold_s"], out["query_warm_s"] = lat[0], float(np.median(lat[1:]))
    get_text().unload()
    return out


PAUSED_RSS_BUDGET_MB = 250  # cull with a mid-run-paused indexer vs cull only (B2.1 / m6)


def _paused_child(mode: str, folder: Path) -> dict:
    """One isolated process: preload the cull models; `paused` also has a *running* indexer
    (SigLIP resident) when the cull gate closes. Reports the peak RSS during the cull."""
    from shotsieve.models.face_detect import get_face_detector
    from shotsieve.models.face_embed import get_face_embedder
    from shotsieve.models.face_landmarks import get_face_landmarker
    from shotsieve.models.image_embed import get_image_embedder
    from shotsieve.models.text_image import _vision
    from shotsieve.pipeline.indexer import Indexer, cull_active
    from shotsieve.pipeline.runner import run_cull

    settings = Settings()
    get_image_embedder(settings.ort_threads, settings.embed_batch, settings.use_coreml)
    get_face_detector(), get_face_embedder(), get_face_landmarker()
    shutil.rmtree(folder / ".shotsieve", ignore_errors=True)
    proc = psutil.Process()
    out: dict = {"mode": mode}
    ix = Indexer(folder, settings, people=False) if mode == "paused" else None
    if ix:
        ix.start()
        t0 = time.perf_counter()
        while ix.n_embedded == 0 and ix.running and time.perf_counter() - t0 < 120:
            time.sleep(0.05)
        out["resident_mb"] = proc.memory_info().rss / 1e6
        out["embedded_before"] = ix.n_embedded
    seen: list[float] = [proc.memory_info().rss / 1e6]
    stop = threading.Event()

    def sample() -> None:
        while not stop.wait(0.05):
            seen.append(proc.memory_info().rss / 1e6)

    th = threading.Thread(target=sample, daemon=True)
    with cull_active():
        if ix:
            t1 = time.perf_counter()
            while _vision and time.perf_counter() - t1 < 5:
                time.sleep(0.02)
            out["released_s"] = time.perf_counter() - t1
            out["sessions_left"] = len(_vision)
            out["after_release_mb"] = proc.memory_info().rss / 1e6
            seen.append(out["after_release_mb"])
            n0 = ix.n_embedded
        th.start()
        res = run_cull(folder, settings)
        stop.set()
        th.join()
        if ix:
            out["embedded_during_cull"] = ix.n_embedded - n0
    out["cull_peak_mb"] = max(seen)
    out["cull_imgs_per_s"] = res.stats["imgs_per_s"]
    if ix:
        ix.join(600)
        out["index_error"] = ix.status()["error"]
    return out


def run_paused_midrun(folder: Path, settings: Settings, peak_cull: float) -> dict | None:
    """The indexer is *running* (SigLIP resident) when a cull starts: the pause must release
    the SigLIP + OCR sessions, so the cull's peak RSS stays within +250 MB of cull-only.
    Both runs are fresh subprocesses (the main process keeps freed pages from earlier stages)."""
    import subprocess

    from shotsieve.models.text_image import search_models_ready

    if not search_models_ready():
        return None
    res = {}
    for mode in ("cull", "paused"):
        cp = subprocess.run(
            [sys.executable, __file__, "--paused-child", mode, "--dir", str(folder)],
            capture_output=True, text=True, timeout=900, check=True,
        )  # fmt: skip
        res[mode] = json.loads(cp.stdout.strip().splitlines()[-1])
    p = res["paused"]
    p["cull_only_mb"] = res["cull"]["cull_peak_mb"]
    p["over_mb"] = p["cull_peak_mb"] - p["cull_only_mb"]
    return p


def paused_lines(p: dict | None) -> list[str]:
    if p is None:
        return ["## Cull with indexer paused mid-run (skipped: search models not downloaded)", ""]
    ok = p["over_mb"] <= PAUSED_RSS_BUDGET_MB and p["sessions_left"] == 0
    return [
        "## Cull with indexer paused mid-run (m6)",
        "",
        f"- SigLIP running ({p['embedded_before']} embedded, RSS {p['resident_mb']:.0f} MB), then "
        f"the cull gate closes: sessions released after {p['released_s'] * 1000:.0f} ms "
        f"(left: {p['sessions_left']}); embedded during the cull: {p['embedded_during_cull']}",
        f"- Cull peak RSS (isolated subprocesses) **{p['cull_peak_mb']:.0f} MB** vs cull-only "
        f"{p['cull_only_mb']:.0f} MB = "
        f"**{p['over_mb']:+.0f} MB** (target <= +{PAUSED_RSS_BUDGET_MB}) "
        f"{'PASS' if ok else 'FAIL'}; cull {p['cull_imgs_per_s']:.1f} img/s",
        "",
    ]


HEVC_RSS_BUDGET_MB = 350  # HW-decoded 4K HEVC (Review #5); H.264 stays 300


def hevc_lines(h: dict | None) -> list[str]:
    if h is None:
        return ["- HEVC 4K30: skipped (no HEVC writer/decoder on this machine)", ""]
    return [
        f"- HEVC 4K30 clip, {h['seconds']:.0f} s ({h['src_frames']} frames), backend "
        f"{h['backend']}: **{h['realtime_x']:.1f}x realtime = {h['src_fps']:.0f} source "
        f"frames/s decoded** (target >= 4x), {h['frames']} scored = {h['scored_fps']:.1f} "
        f"scored frames/s, "
        f"{'PASS' if h['realtime_x'] >= 4.0 else 'FAIL'}",
        f"- HEVC RSS added: +{h['rss_added_mb']:.0f} MB "
        f"(target <= {HEVC_RSS_BUDGET_MB} MB: "
        f"{'PASS' if h['rss_added_mb'] <= HEVC_RSS_BUDGET_MB else 'OVER'}; cv2's "
        "AVFoundation/VideoToolbox keeps "
        "~300 MB of decoder surfaces per open 4K stream, transient, released on close)",
        "",
    ]


def _tree_mb(kind: str = "uss") -> float:
    """Memory of this process and its children: `uss` (unique pages: what training really adds)
    or `rss` (double counts the libraries and memory maps the worker processes share)."""
    me = psutil.Process()
    total = 0
    for p in [me, *me.children(recursive=True)]:
        try:
            total += p.memory_full_info().uss if kind == "uss" else p.memory_info().rss
        except psutil.Error:
            pass
    return total / 1e6


def run_t15(folder: Path, settings: Settings) -> dict:
    """T15 editing profile: train time / RSS on 10k rows, `edit apply` speed on the bench shoot."""
    import tempfile

    from shotsieve.personal.edit_dataset import FeatureStore, build_edit_set
    from shotsieve.personal.edit_profile import EditProfile, EditSet
    from shotsieve.pipeline import edit

    ms = _load_mod("make_edit_styles", ROOT / "scripts" / "make_edit_styles.py")
    fx = ms.build()
    store = FeatureStore(fx / "features.db")
    S = build_edit_set(fx / "styleA.lrcat", [], settings, None, store=store,
                       group_of=lambda p: Path(p).parent.name)  # fmt: skip
    store.close()
    rng = np.random.default_rng(0)
    idx = np.tile(np.arange(len(S)), 8)[:10000]
    n = len(idx)
    jitter = rng.normal(size=(n, S.stats.shape[1])).astype(np.float32) * 0.01
    big = EditSet(
        np.nan_to_num(S.stats[idx]) + jitter,
        S.emb[idx], S.fmt[idx], {k: v[idx] + rng.normal(size=n) * 0.5 for k, v in S.y.items()},
        S.groups[idx] + (np.arange(n) // len(S)) * 1000, S.carry,
    )  # fmt: skip
    base = {"uss": _tree_mb("uss"), "rss": _tree_mb("rss")}
    peak_tree = dict(base)
    stop = threading.Event()

    def sample() -> None:
        while not stop.is_set():
            for k in peak_tree:
                peak_tree[k] = max(peak_tree[k], _tree_mb(k))
            time.sleep(0.1)

    th = threading.Thread(target=sample, daemon=True)
    th.start()
    t0 = time.perf_counter()
    EditProfile().fit(big)
    train_s = time.perf_counter() - t0
    stop.set()
    th.join()
    out: dict = {
        "train_rows": n, "train_s": train_s, "train_rss_mb": peak_tree["uss"] - base["uss"],
        "train_rss_shared_mb": peak_tree["rss"] - base["rss"],
    }
    with tempfile.TemporaryDirectory(prefix="shotsieve-t15-") as tmp:
        old = os.environ.get("SHOTSIEVE_DATA_DIR")
        os.environ["SHOTSIEVE_DATA_DIR"] = tmp
        try:
            prof = EditProfile().fit(S)
            path = prof.save(Path(tmp) / "edit_profile.joblib")
            out["profile_mb"] = path.stat().st_size / 1e6
            r0 = psutil.Process().memory_info().rss / 1e6
            t0 = time.perf_counter()
            cold = edit.apply_folder(folder, settings, scope="all", dry_run=True)
            out["apply_cold_s"] = time.perf_counter() - t0
            out["apply_n"] = cold["predicted"]
            t0 = time.perf_counter()
            warm = edit.apply_folder(folder, settings, scope="all", dry_run=True)
            out["apply_warm_s"] = time.perf_counter() - t0
            out["apply_rss_mb"] = psutil.Process().memory_info().rss / 1e6 - r0
            out["needs_edit"] = warm["needs_edit"]
            # featurising uncached images: stats + DINOv2 (no cull cache) from a fresh copy
            sub = Path(tmp) / "plain"
            sub.mkdir()
            files = sorted(folder.glob("*.jpg"))[:100]
            for f in files:
                shutil.copy(f, sub / f.name)
            from shotsieve.personal.edit_dataset import featurise

            fresh = FeatureStore(Path(tmp) / "fresh.db")  # nothing cached, no cull cache
            t0 = time.perf_counter()
            featurise([str(f) for f in sorted(sub.glob("*.jpg"))], settings, store=fresh)
            fresh.close()
            out["featurise_s"] = time.perf_counter() - t0
            out["featurise_n"] = len(files)
        finally:
            if old is None:
                os.environ.pop("SHOTSIEVE_DATA_DIR", None)
            else:
                os.environ["SHOTSIEVE_DATA_DIR"] = old
    return out


def run_t16(folder: Path, settings: Settings) -> dict:
    """T16 colour match: `measure + match_one` throughput on the bench shoot (cached embeddings)."""
    import tempfile

    from shotsieve.pipeline import match

    paths = [str(p) for p in sorted(folder.glob("*.jpg"))]
    out: dict = {"n": len(paths)}
    with tempfile.TemporaryDirectory(prefix="shotsieve-t16-") as tmp:
        old = os.environ.get("SHOTSIEVE_DATA_DIR")
        os.environ["SHOTSIEVE_DATA_DIR"] = tmp
        try:
            r0 = psutil.Process().memory_info().rss / 1e6
            t0 = time.perf_counter()
            match.measure_paths(paths, settings)  # embeddings come from the cull cache
            out["cold_s"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            ms = match.measure_paths(paths, settings)  # feature store warm
            ref = match.Reference(paths[0], {"Exposure2012": 0.3, "Contrast2012": 10.0},
                                  ms[paths[0]], "bench")  # fmt: skip
            res = [match.match_one(ref, m) for m in ms.values()]
            out["warm_s"] = time.perf_counter() - t0
            out["matched"] = sum(r.skipped_reason is None for r in res)
            out["rss_mb"] = psutil.Process().memory_info().rss / 1e6 - r0
        finally:
            if old is None:
                os.environ.pop("SHOTSIEVE_DATA_DIR", None)
            else:
                os.environ["SHOTSIEVE_DATA_DIR"] = old
    return out


def run_t17(folder: Path, settings: Settings) -> dict | None:
    """T17 shot list: 100 items over the indexed bench shoot (real SigLIP2 text tower), plus a
    100-item x 20k-frame synthetic run (warm = matrix and text vectors resident)."""
    import tempfile

    from shotsieve.models.text_image import search_models_ready
    from shotsieve.pipeline import shotlist
    from shotsieve.pipeline.cache import SIGLIP_DIM, Cache

    out: dict = {}
    if search_models_ready():
        items = [i for n in shotlist.template_names() for i in shotlist.load_template(n)]
        items += [shotlist.Item(f"x:{k:03d}", f"a photo of {w}", False, 1, None, k, "x")
                  for k, w in enumerate(("a dog", "a car", "a tree", "a child", "a cake", "a boat",
                                         "a house", "a bird", "a bridge", "a crowd"))]  # fmt: skip
        items = (items * 2)[:100]
        from shotsieve.pipeline.indexer import Indexer

        with Cache.for_folder(folder) as c:
            n_idx = c.siglip_count()
        if n_idx < len(list(folder.glob("*.jpg"))):  # T10 released the index: rebuild it once
            Indexer(folder, settings, people=False).run_blocking()
        r0 = psutil.Process().memory_info().rss / 1e6
        t0 = time.perf_counter()
        cold = shotlist.run(folder, items, settings, save=False)
        out["real_cold_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        warm = shotlist.run(folder, items, settings, save=False)
        out["real_warm_s"] = time.perf_counter() - t0
        out["real_rss_mb"] = psutil.Process().memory_info().rss / 1e6 - r0
        out["real_indexed"] = cold.indexed_pct
        out["real_counts"] = warm.counts()
    n = 20_000
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory(prefix="shotsieve-t17-") as tmp:
        big = Path(tmp)
        paths = [str(big / f"f{i:05d}.jpg") for i in range(n)]
        mat = rng.normal(size=(n, SIGLIP_DIM)).astype(np.float32)
        mat /= np.linalg.norm(mat, axis=1, keepdims=True)
        rows = [dict(path=p, decision="keep" if i % 7 else "reject", reasons=[], score=0.5,
                     prior=0.5, group_id=i, burst_id=i, is_best=True, audit=False,
                     audit_prob=None, features={}) for i, p in enumerate(paths)]  # fmt: skip
        with Cache.for_folder(big) as c:
            c.save_decisions(rows)
            for i in range(0, n, 2000):
                c.put_siglip([(p, f"fp{j}", mat[j]) for j, p in enumerate(paths[i : i + 2000], i)])
            its = [shotlist.Item(f"x:{k:03d}", f"shot {k}", k % 3 == 0, 1 + k % 2, None, k, "x")
                   for k in range(100)]  # fmt: skip

            class Rand:
                def __init__(self) -> None:
                    self.v: dict[str, np.ndarray] = {}

                def embed(self, text: str) -> np.ndarray:
                    return self.v.setdefault(text, rng.normal(size=SIGLIP_DIM).astype(np.float32))

            emb = Rand()
            shotlist.evaluate(c, its, settings, emb)
            t0 = time.perf_counter()
            shotlist.evaluate(c, its, settings, emb)
            out["synthetic_warm_s"] = time.perf_counter() - t0
    return out


def _load_mod(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def t15_lines(t: dict | None, peak_all: float) -> list[str]:
    if t is None:
        return []
    apply_ips = t["apply_n"] / t["apply_warm_s"]
    return [
        "## T15 editing profile (`--phase3`)",
        "",
        f"- `edit train` on {t['train_rows']} rows: **{t['train_s']:.1f} s** (target <= 60 s) "
        f"{'PASS' if t['train_s'] <= 60 else 'FAIL'}; memory added incl. the worker processes "
        f"+{t['train_rss_mb']:.0f} MB unique (USS; target <= +400 MB) "
        f"{'PASS' if t['train_rss_mb'] <= 400 else 'FAIL'} (summed RSS "
        f"+{t['train_rss_shared_mb']:.0f} MB counts shared library pages once per worker)",
        f"- profile file {t['profile_mb']:.1f} MB (target <= 10 MB) "
        f"{'PASS' if t['profile_mb'] <= 10 else 'FAIL'}",
        f"- `edit apply` (dry run, {t['apply_n']} images, cached embeddings): cold features "
        f"{t['apply_n'] / t['apply_cold_s']:.0f} img/s, **warm {apply_ips:.0f} img/s** "
        f"(target >= 50) {'PASS' if apply_ips >= 50 else 'FAIL'}; RSS +{t['apply_rss_mb']:.0f} MB "
        f"(target <= +150 MB) {'PASS' if t['apply_rss_mb'] <= 150 else 'FAIL'}",
        f"- featurising {t['featurise_n']} uncached images (DINOv2 + statistics): "
        f"**{t['featurise_n'] / t['featurise_s']:.1f} img/s** (target >= 15) "
        f"{'PASS' if t['featurise_n'] / t['featurise_s'] >= 15 else 'FAIL'}",
        f"- process peak with everything above {peak_all:.0f} MB (target <= 3000)",
        "",
    ]


def t16_lines(t: dict | None) -> list[str]:
    if t is None:
        return []
    ips = t["n"] / t["warm_s"]
    return [
        "## T16 reference colour match (`--phase3`)",
        "",
        f"- `match` on {t['n']} images (256 px previews, embeddings cached): cold "
        f"{t['n'] / t['cold_s']:.0f} img/s, **warm {ips:.0f} img/s** (target >= 30) "
        f"{'PASS' if ips >= 30 else 'FAIL'}; {t['matched']} passed the scene gate; "
        f"RSS +{t['rss_mb']:.0f} MB",
        "",
    ]


def t17_lines(t: dict | None, peak_all: float) -> list[str]:
    if t is None:
        return []
    lines = ["## T17 shot-list coverage (`--phase3`)", ""]
    if "real_warm_s" in t:
        lines.append(
            f"- 100 items over the {t['real_indexed']:.0f} %-indexed bench shoot, real SigLIP2 "
            f"text tower: cold {t['real_cold_s']:.2f} s (text tower load + 100 text embeddings), "
            f"**warm {t['real_warm_s'] * 1000:.0f} ms** (target < 1 s) "
            f"{'PASS' if t['real_warm_s'] < 1 else 'FAIL'}; RSS +{t['real_rss_mb']:.0f} MB "
            f"(text tower budget +450 MB); statuses {t['real_counts']}"
        )
    lines += [
        f"- 100 items over 20,000 frames (random embeddings): **warm "
        f"{t['synthetic_warm_s'] * 1000:.0f} ms** (target < 1 s) "
        f"{'PASS' if t['synthetic_warm_s'] < 1 else 'FAIL'}",
        f"- process peak with everything above {peak_all:.0f} MB (target <= 3000)",
        "",
    ]
    return lines


def t12_lines(t12: dict | None, peak_all: float) -> list[str]:
    if t12 is None:
        return ["## T12 video frame grabs (skipped: no H.264 writer on this machine)", ""]
    ok = t12["realtime_x"] >= 4.0 and t12["rss_added_mb"] <= 300
    return [
        "## T12 video frame-grab culling (opt-in)",
        "",
        f"- 4K30 H.264 clip, {t12['seconds']:.0f} s ({t12['src_frames']} frames), backend "
        f"{t12['backend']}: **{t12['realtime_x']:.1f}x realtime = {t12['src_fps']:.0f} source "
        f"frames/s decoded** (target >= 4x), {t12['frames']} frames scored (2 fps) "
        f"= {t12['scored_fps']:.1f} scored frames/s, {t12['picks']} picks "
        f"{'PASS' if ok else 'FAIL'}",
        f"- RSS added by video processing (models already loaded): +{t12['rss_added_mb']:.0f} MB "
        f"(target <= 300); process peak {peak_all:.0f} MB (target <= 3000)",
        "",
    ]


def make_clip(clip: Path, fourcc: str = "avc1") -> None:
    """A 30 s 4K30 H.264 pan over a fixture photo (benchmark input for T12)."""
    src = cv2.imread(str(Path(__file__).resolve().parents[1] / "tests" / "fixtures"
                         / "generated" / "src" / "group_rgb.jpg"))
    if src is None:
        return
    src = cv2.resize(src, (3840, 2160))
    w = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*fourcc), 30, (3840, 2160))
    if not w.isOpened():
        return
    for i in range(900):
        m = np.float32([[1, 0, i % 40], [0, 1, (i // 2) % 30]])
        w.write(cv2.warpAffine(src, m, (3840, 2160)))
    w.release()


def run_t12(settings: Settings, name: str = "video_4k30.mp4", fourcc: str = "avc1") -> dict | None:
    from shotsieve.pipeline import video as pv

    clip = data_dir() / "bench" / name
    if not clip.exists():
        clip.parent.mkdir(parents=True, exist_ok=True)
        # encoded in a child process: the hardware encoder buffers GBs that are not the cull's
        cmd = [sys.executable, __file__, "--make-clip", str(clip), "--fourcc", fourcc]
        subprocess.run(cmd, check=False)
        if not clip.exists():
            return None
    proc = psutil.Process()
    base, peak, stop = proc.memory_info().rss, [0.0], threading.Event()
    sampler = threading.Thread(target=_rss_sampler, args=(stop, peak), daemon=True)
    sampler.start()
    vs = replace(settings, video=True)
    pv.scan_video(clip, vs)  # warm-up: loads face/landmark/embedding models, if not yet loaded
    base = proc.memory_info().rss
    peak[0] = base / 1e6
    picks, st = pv.scan_video(clip, vs)
    stop.set()
    sampler.join()
    return {
        "seconds": st["seconds"], "src_frames": st["src_frames"], "frames": st["frames"],
        "backend": st["backend"], "picks": len(picks), "realtime_x": st["seconds"] / st["wall_s"],
        "src_fps": st["src_frames"] / st["wall_s"], "scored_fps": st["frames"] / st["wall_s"],
        "rss_added_mb": max(0.0, peak[0] - base / 1e6),
    }


def make_raw_shoot(n: int, out: Path, seed: int = 7) -> list[str]:
    """n copies of the fixture DNGs with varied names; EXIF timestamps are patched in place.

    Every `YYYY:MM:DD HH:MM:SS` string in the copy (IFD0/Exif DateTime* tags and the embedded
    preview's EXIF) is replaced by a per-copy time (same length, so offsets stay valid); the
    mtime is set to the same instant as a fallback. Bursts of 1-5 frames 0.4 s apart is not
    representable at 1 s EXIF resolution, so frames in a burst are 1 s apart.
    """
    from datetime import datetime

    srcs = sorted(RAW_SRC.glob("*.dng"))
    if not srcs:
        raise SystemExit(f"no fixture DNGs in {RAW_SRC}; run scripts/make_fixtures.py")
    blobs = [p.read_bytes() for p in srcs]
    stamp = re.compile(rb"\d{4}:\d\d:\d\d \d\d:\d\d:\d\d")
    rng = np.random.default_rng(seed)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    t = datetime(2024, 6, 1, 14, 0, 0).timestamp()
    names, i = [], 0
    while i < n:
        burst = min(int(rng.integers(1, 6)), n - i)
        for _ in range(burst):
            blob = blobs[i % len(blobs)]
            ts = datetime.fromtimestamp(t).strftime("%Y:%m:%d %H:%M:%S").encode()
            data = stamp.sub(ts, blob)
            name = f"{('DSC', 'IMG', 'P')[i % 3]}_{1000 + i * 7:04d}.dng"
            f = out / name
            f.write_bytes(data)
            os.utime(f, (t, t))
            names.append(name)
            t += 1.0
            i += 1
        t += float(rng.uniform(10, 90))
    return names


def _sig(res) -> list:
    return [
        (str(r.master_path), r.decision, tuple(r.reasons), r.group_id, r.burst_id, r.audit, r.score)
        for r in res.images
    ]


def run_raw(args) -> int:
    from shotsieve.pipeline.runner import run_cull

    folder = args.dir or data_dir() / "bench" / f"raw_{args.n}_{args.seed}"
    make_raw_shoot(args.n, folder, args.seed)
    settings = Settings()
    peak = [psutil.Process().memory_info().rss / 1e6]
    stop = threading.Event()
    sampler = threading.Thread(target=_rss_sampler, args=(stop, peak), daemon=True)
    sampler.start()
    t0 = time.perf_counter()
    res = run_cull(folder, settings)
    wall = time.perf_counter() - t0
    stop.set()
    sampler.join()
    warm = run_cull(folder, settings)
    identical = _sig(res) == _sig(warm)
    cs = res.stats
    rate = cs["n_images"] / wall
    size_mb = sum(f.stat().st_size for f in folder.glob("*.dng")) / 1e6 / max(args.n, 1)
    sec = [
        RAW_MARK,
        "",
        f"- Date: {time.strftime('%Y-%m-%d %H:%M')}; {cs['n_images']} copies of "
        f"{len(list(RAW_SRC.glob('*.dng')))} fixture DNGs (avg {size_mb:.1f} MB each), varied "
        f"names and patched EXIF timestamps (seed {args.seed})",
        f"- Cold run incl. model load on first use: {wall:.2f} s total, "
        f"engine-reported {cs['total_s']:.2f} s",
        f"- **Throughput: {rate:.1f} img/s** ({cs['imgs_per_s']:.1f} img/s engine-reported); "
        f"**peak RSS {peak[0]:.0f} MB** (whole process, sampled 50 ms)",
        f"- Warm re-run decisions identical to cold: {identical}",
        f"- Decisions {cs['counts']}, {cs['n_groups']} duplicate groups, {cs['n_bursts']} bursts "
        "(all frames derive from 2 source images, so most are duplicates by construction)",
        "",
    ]
    out = args.out or data_dir() / "benchmark.md"
    _replace_section(out, RAW_MARK, sec)
    print("\n".join(sec))
    print(f"appended to {out}")
    return 0 if rate >= 1.0 and peak[0] <= 3000 and identical else 1


LARGE_MARK = "## T21 10k-frame shoot"
P4_MARK = "## Phase 4 checks"
P5_MARK = "## T22 subject-aware crop"
CROP_RSS_LOADED_MB = 300
CROP_RSS_IDLE_MB = 400  # retained after unload (Review #12 ruling; macOS keeps the ORT heap)


def _replace_section(out: Path, mark: str, section: list[str]) -> None:
    """Put `section` (starting with `mark`) into benchmark.md, replacing an older copy."""
    base = out.read_text() if out.exists() else ""
    blocks = re.split(r"(?m)^(?=## )", base)
    kept = [b for b in blocks if b.strip() and not b.startswith(mark)]
    out.write_text("".join(kept).rstrip() + "\n\n" + "\n".join(section))


def _first_grid_s(folder: Path, port: int) -> float:
    """Seconds from navigation to every visible grid thumbnail loaded (Chromium 1500x900)."""
    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(ROOT / ".playwright"))
    from playwright.sync_api import sync_playwright

    js = (
        "(() => { const g = document.getElementById('grid').getBoundingClientRect();"
        " const im = [...document.querySelectorAll('.cell img')].filter(i => {"
        "   const r = i.getBoundingClientRect(); return r.bottom > g.top && r.top < g.bottom; });"
        " return im.length > 0 && im.every(i => i.complete && i.naturalWidth > 0); })()"
    )
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        page = b.new_context(viewport={"width": 1500, "height": 900}).new_page()
        t0 = time.perf_counter()
        page.goto(f"http://127.0.0.1:{port}/")
        page.wait_for_selector(".cell", timeout=60_000)
        page.wait_for_function(js, timeout=60_000)
        dt = time.perf_counter() - t0
        b.close()
    return dt


def _unaffected(frames: list, changed: dict[str, list[float]], margin: float) -> set[str]:
    import bisect

    srt = {k: sorted(v) for k, v in changed.items()}
    out = set()
    for path, serial, ts in frames:
        arr = srt.get(serial, [])
        i = bisect.bisect_left(arr, ts)
        near = [abs(arr[j] - ts) for j in (i - 1, i) if 0 <= j < len(arr)]
        if not near or min(near) > margin:
            out.add(path)
    return out


def run_large(args) -> int:
    """T21: synthetic 10k shoot: cold / warm / incremental / grid timing -> benchmark.md."""
    import socket
    import urllib.request

    import piexif

    from shotsieve.pipeline.cache import Cache
    from shotsieve.pipeline.runner import run_cull
    from shotsieve.server.app import build_shoot

    n = args.large
    free_gb = shutil.disk_usage(data_dir()).free / 2**30
    if free_gb < 10:
        print(f"refusing: only {free_gb:.1f} GiB free (need >= 10 GiB for the {n}-frame shoot)")
        return 2
    folder = args.dir or data_dir() / "bench" / f"shoot_{n}_{args.seed}"
    if len(list(folder.glob("IMG_*.jpg"))) != n or not (folder / TRUTH_FILE).exists():
        shutil.rmtree(folder, ignore_errors=True)
        t0 = time.perf_counter()
        synthesize(n, folder, args.seed)
        print(f"synthesised {n} images in {time.perf_counter() - t0:.0f}s")
    shutil.rmtree(folder / ".shotsieve", ignore_errors=True)
    settings = Settings()
    window = max(settings.dup_window_s, settings.flash_window_s)
    from shotsieve.models.face_detect import get_face_detector
    from shotsieve.models.face_embed import get_face_embedder
    from shotsieve.models.face_landmarks import get_face_landmarker
    from shotsieve.models.image_embed import get_image_embedder

    get_image_embedder(settings.ort_threads, settings.embed_batch, settings.use_coreml)
    get_face_detector(), get_face_embedder(), get_face_landmarker()

    peak = [psutil.Process().memory_info().rss / 1e6]
    stop = threading.Event()
    threading.Thread(target=_rss_sampler, args=(stop, peak), daemon=True).start()
    cold = run_cull(folder, settings)
    cold_rss = peak[0]
    t0 = time.perf_counter()
    warm = run_cull(folder, settings)
    warm_s = time.perf_counter() - t0
    identical = _sig(cold) == _sig(warm)

    # server: payload + RSS + first grid with / without thumbnails
    with socket.socket() as sk:
        sk.bind(("127.0.0.1", 0))
        port = sk.getsockname()[1]
    srv = subprocess.Popen(
        [sys.executable, "-m", "shotsieve.cli", "serve", str(folder), "--port", str(port),
         "--no-open"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=ROOT,
    )  # fmt: skip
    try:
        for _ in range(100):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=1).read()
                break
            except OSError:
                time.sleep(0.3)
        sp = psutil.Process(srv.pid)
        idle = sp.memory_info().rss / 1e6
        grid_warm = _first_grid_s(folder, port)
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/shoot", headers={"Accept-Encoding": "gzip"}
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            gz_bytes = len(r.read())
        build_t = min(
            (lambda t: (build_shoot(folder, settings), time.perf_counter() - t)[1])(
                time.perf_counter()
            )
            for _ in range(3)
        )
        shutil.rmtree(folder / ".shotsieve" / "thumbs", ignore_errors=True)
        grid_cold = _first_grid_s(folder, port)
        srv_rss = max(sp.memory_info().rss / 1e6, idle)
    finally:
        srv.terminate()

    # +500 (450 second-shooter block, +2 h, serial B; 50 interleaved, serial A)
    def frames_now() -> list[tuple[str, str, float]]:
        with Cache.for_folder(folder) as c:
            return [(c._abs(i.path), i.serial, i.ts) for i in c.clock_inputs()]

    base_sig = _sig(cold)
    extra = folder.parent / f".extra_{n}"
    shutil.rmtree(extra, ignore_errors=True)
    synthesize(500, extra, seed=args.seed + 4)
    added: list[Path] = []
    changed: dict[str, list[float]] = {}
    for k, src in enumerate(sorted(extra.glob("IMG_*.jpg"))):
        dst = folder / f"X_{k:04d}.jpg"
        shutil.copy2(src, dst)
        ex = piexif.load(str(dst))
        raw = ex["Exif"][piexif.ExifIFD.DateTimeOriginal].decode()
        t = datetime.strptime(raw, "%Y:%m:%d %H:%M:%S")
        serial = "B" if k < 450 else "A"
        if k < 450:
            t = t + timedelta(hours=2)
        ex["Exif"][piexif.ExifIFD.DateTimeOriginal] = t.strftime("%Y:%m:%d %H:%M:%S").encode()
        ex["Exif"][piexif.ExifIFD.BodySerialNumber] = serial.encode()
        piexif.insert(piexif.dump(ex), str(dst))
        changed.setdefault(serial, []).append(t.timestamp())
        added.append(dst)
    old_rows = {r.master_path.name: r for r in cold.images}
    t0 = time.perf_counter()
    inc = run_cull(folder, settings)
    inc_wall = time.perf_counter() - t0
    new_names = {p.name for p in added}
    stable_set = {Path(p).name for p in _unaffected(frames_now(), changed, 2 * window)} - new_names
    now_rows = {r.master_path.name: r for r in inc.images}

    def same(a, b) -> bool:
        return (a.decision, tuple(a.reasons), round(a.score, 12), a.is_best_in_group) == (
            b.decision, tuple(b.reasons), round(b.score, 12), b.is_best_in_group)  # fmt: skip

    stable = sum(same(old_rows[k], now_rows[k]) for k in stable_set if k in old_rows)
    n_stable = sum(1 for k in stable_set if k in old_rows)
    inc_img_s = inc.stats["extract_imgs_per_s"] or cold.stats["extract_imgs_per_s"]
    budget_s = 500 / max(cold.stats["extract_imgs_per_s"], 1e-9) + 20

    # -200
    gone = sorted(folder.glob("IMG_*.jpg"))[100::50][:200]
    gone_meta = {p.name: p for p in gone}
    stash = folder.parent / f".stash_{n}"
    stash.mkdir(exist_ok=True)
    pre = {r.master_path.name: r for r in inc.images}
    gone_frames = {f[0]: f for f in frames_now() if Path(f[0]).name in gone_meta}
    rchanged: dict[str, list[float]] = {}
    for f in gone_frames.values():
        rchanged.setdefault(f[1], []).append(f[2])
    for p in gone:
        shutil.move(str(p), stash / p.name)
    t0 = time.perf_counter()
    rem = run_cull(folder, settings)
    rem_wall = time.perf_counter() - t0
    rest = {Path(p).name for p in _unaffected(frames_now(), rchanged, 2 * window)}
    rest -= set(gone_meta)
    rem_rows = {r.master_path.name: r for r in rem.images}
    rem_stable = sum(same(pre[k], rem_rows[k]) for k in rest if k in pre and k in rem_rows)
    rem_n = sum(1 for k in rest if k in pre and k in rem_rows)
    with Cache.for_folder(folder) as c:
        rows_gone = sum(1 for p in c.load_decisions() if Path(p).name in gone_meta)

    # > 20 % change: stash 2,700 more files
    more = sorted(folder.glob("IMG_*.jpg"))[10::3][: int(0.27 * n)]
    for p in more:
        shutil.move(str(p), stash / p.name)
    big = run_cull(folder, settings)

    # restore the bench folder
    for p in added:
        p.unlink(missing_ok=True)
    for p in stash.iterdir():
        shutil.move(str(p), folder / p.name)
    stash.rmdir()
    shutil.rmtree(extra, ignore_errors=True)
    shutil.rmtree(folder / ".shotsieve", ignore_errors=True)
    stop.set()

    cs = cold.stats
    ok = {
        "cold": cs["imgs_per_s"] >= 10.0 and cold_rss <= 3000,
        "warm": warm_s <= 15.0 and warm.stats["n_extracted"] == 0 and identical,
        "grid_warm": grid_warm <= 2.0,
        "grid_cold": grid_cold <= 5.0,
        "shoot": build_t <= 1.0 and gz_bytes <= 1_500_000,
        "add": inc.stats["n_extracted"] == 500 and inc_wall <= budget_s and stable == n_stable,
        "remove": rem.stats["n_extracted"] == 0 and rem_wall <= 15.0 and rows_gone == 0
        and rem_stable == rem_n,
        "full": big.stats["redecide"] == "full",
        "rss": srv_rss - idle <= 300,
    }  # fmt: skip

    def v(k: str) -> str:
        return "PASS" if ok[k] else "FAIL"

    sec = [
        LARGE_MARK,
        "",
        f"- Date: {time.strftime('%Y-%m-%d %H:%M')}; {n} synthetic JPEGs ({cs['n_groups']} groups, "
        f"{cs['n_bursts']} bursts), seed {args.seed}",
        "",
        "| Metric | Value | Budget |",
        "|---|---|---|",
        f"| Cold full cull | {cs['imgs_per_s']:.1f} img/s ({cs['total_s']:.0f} s), peak RSS "
        f"{cold_rss:.0f} MB {v('cold')} | >= 10 img/s, <= 3000 MB |",
        f"| Warm re-cull, nothing changed | {warm_s:.1f} s, {warm.stats['n_extracted']} extracted, "
        f"decisions identical {identical} {v('warm')} | <= 15 s, 0, identical |",
        f"| First grid, warm (thumbs from cull) | {grid_warm:.2f} s {v('grid_warm')} | <= 2.0 s |",
        f"| First grid, thumbs deleted | {grid_cold:.2f} s {v('grid_cold')} | <= 5.0 s |",
        f"| build_shoot / gzip payload | {build_t:.2f} s / {gz_bytes / 1e6:.2f} MB {v('shoot')} | "
        "<= 1.0 s / <= 1.5 MB |",
        f"| Server RSS serving the grid | +{srv_rss - idle:.0f} MB over idle ({idle:.0f} MB) "
        f"{v('rss')} | <= +300 MB |",
        f"| Add 500 files | {inc.stats['n_extracted']} extracted, {inc_wall:.0f} s "
        f"(budget {budget_s:.0f} s at {cs['extract_imgs_per_s']:.1f} img/s), frames outside A "
        f"identical {stable}/{n_stable}, n_affected {inc.stats['n_affected']}, redecide "
        f"{inc.stats['redecide']} {v('add')} | 500, <= budget, 100 % |",
        f"| Remove 200 files | {rem.stats['n_extracted']} extracted, {rem_wall:.1f} s, rows gone "
        f"{rows_gone == 0}, outside A identical {rem_stable}/{rem_n}, n_affected "
        f"{rem.stats['n_affected']} {v('remove')} | 0, <= 15 s, 100 % |",
        f"| > 20 % changed | redecide = {big.stats['redecide']} "
        f"({big.stats['redecide_reason']}) {v('full')} | full |",
        "",
    ]
    _ = (base_sig, inc_img_s)
    out = args.out or data_dir() / "benchmark.md"
    _replace_section(out, LARGE_MARK, sec)
    print("\n".join(sec))
    print(f"written to {out}")
    return 0 if all(ok.values()) else 1


CROP_CYCLES = 4
CROP_RSS_GROWTH_MB = 10
CROP_PEAK_MB = 3000


def _crop_child(folder: Path) -> dict:
    """One isolated process: RSS over the idle baseline with the crop sessions loaded, then
    CROP_CYCLES load -> idle-unload cycles (3 s idle) with the retained RSS after each unload. Also
    the first-detection time (model loads included)."""
    import gc

    from shotsieve.pipeline import crop as pcrop
    from shotsieve.pipeline.cache import Cache

    settings = Settings(crop_idle_unload_s=3)
    proc = psutil.Process()
    imgs = sorted(folder.glob("*.jpg"))[:12]
    c = Cache.for_folder(folder)
    ids = c.ids_for([str(p) for p in imgs])
    for p in imgs:  # previews cached first (the budget assumes `/api/preview` was served)
        pcrop.preview_rgb(folder, ids[str(p)], str(p))
    base = proc.memory_info().rss / 1e6
    loaded, idle_cycles, first_s = base, [], 0.0
    for cycle in range(CROP_CYCLES):
        with c._lock:  # the parent already filled the cache: every detection must be a cold one
            c._con.execute("DELETE FROM subjects")
            c._con.commit()
        t0 = time.perf_counter()
        pcrop.detect_subjects(folder, c, ids[str(imgs[0])], str(imgs[0]), settings)
        if cycle == 0:
            first_s = time.perf_counter() - t0
        for p in imgs[1:]:
            pcrop.detect_subjects(folder, c, ids[str(p)], str(p), settings)
            pcrop.point_subject(folder, c, ids[str(p)], str(p), 0.5, 0.5, settings)
            loaded = max(loaded, proc.memory_info().rss / 1e6)
        time.sleep(settings.crop_idle_unload_s + 4)
        gc.collect()
        idle_cycles.append(proc.memory_info().rss / 1e6)
    c.close()
    return {"base_mb": base, "loaded_mb": loaded, "idle_mb": idle_cycles[-1],
            "idle_cycles_mb": idle_cycles, "first_detect_s": first_s}  # fmt: skip


def run_crop(folder: Path, settings: Settings, n_apply: int = 300) -> dict:
    """T22: subjects cold / point / suggest latency, detect and apply throughput, RSS (child process)."""
    from shotsieve.pipeline import crop as pcrop
    from shotsieve.pipeline.cache import Cache

    imgs = sorted(folder.glob("*.jpg"))[:n_apply]
    c = Cache.for_folder(folder)
    ids = c.ids_for([str(p) for p in imgs])
    for p in imgs:
        pcrop.preview_rgb(folder, ids[str(p)], str(p))
    pcrop.detect_subjects(folder, c, ids[str(imgs[0])], str(imgs[0]), settings)  # models loaded
    with c._lock:
        c._con.execute("DELETE FROM subjects")
        c._con.commit()
    cold: list[float] = []
    for p in imgs[:40]:  # serial latency of one cold detection
        c._con.execute("DELETE FROM subjects WHERE path=?", (c._rel(str(p)),))
        t0 = time.perf_counter()
        pcrop.detect_subjects(folder, c, ids[str(p)], str(p), settings)
        cold.append(time.perf_counter() - t0)
    with c._lock:
        c._con.execute("DELETE FROM subjects")
        c._con.commit()
    t_all = time.perf_counter()  # the `crop detect` batch path (pipelined workers)
    pcrop.detect_many(folder, c, [(ids[str(p)], str(p)) for p in imgs], settings)
    detect_s = time.perf_counter() - t_all
    pt_cold, pt_warm, sug = [], [], []
    for p in imgs[:30]:
        pcrop._tokens.clear()
        t0 = time.perf_counter()
        pcrop.point_subject(folder, c, ids[str(p)], str(p), 0.5, 0.5, settings)
        pt_cold.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        pcrop.point_subject(folder, c, ids[str(p)], str(p), 0.45, 0.55, settings)
        pt_warm.append(time.perf_counter() - t0)
    ratios = ["4:5", "3:2", "1:1", "16:9", "5:7"]
    for i, p in enumerate(imgs):
        t0 = time.perf_counter()
        pcrop.suggest_crops(folder, c, ids[str(p)], str(p), settings, ratio=ratios[i % 5], log_event=False)
        sug.append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    for p in imgs:
        r = pcrop.suggest_crops(folder, c, ids[str(p)], str(p), settings, ratio="4:5", n=1, log_event=False)
        pcrop.accept(folder, c, ids[str(p)], str(p), settings, r["candidates"][0]["rect"], r["ratio"],
                     write="none")  # fmt: skip
    apply_s = time.perf_counter() - t0
    c.close()
    child = json.loads(subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--crop-child", "--dir", str(folder)],
        capture_output=True, text=True, check=True, cwd=ROOT,
    ).stdout.strip().splitlines()[-1])  # fmt: skip
    pct = lambda a, q: float(np.percentile(a, q)) * 1000  # noqa: E731
    return {
        "n": len(imgs), "cold_p50_ms": pct(cold, 50), "cold_p95_ms": pct(cold, 95),
        "detect_img_s": len(imgs) / detect_s, "apply_img_s": len(imgs) / apply_s,
        "point_cold_ms": pct(pt_cold, 50), "point_warm_ms": pct(pt_warm, 95),
        "suggest_p50_ms": pct(sug, 50), "suggest_p95_ms": pct(sug, 95), **child,
    }  # fmt: skip


def crop_lines(t: dict | None, cull_imgs_s: float, base_cull: float | None = None, whole_peak: float | None = None) -> list[str]:
    if not t:
        return []
    d_loaded = t["loaded_mb"] - t["base_mb"]
    cyc = t.get("idle_cycles_mb") or [t["idle_mb"]]
    d_idle_all = [x - t["base_mb"] for x in cyc]
    growth = (cyc[-1] - cyc[-2]) if len(cyc) > 1 else 0.0
    rows = [
        ("first detection (models loading included)", f"{t['first_detect_s'] * 1000:.0f} ms", "<= 700 ms", t["first_detect_s"] <= 0.7),
        ("subjects cold p50 / p95 (previews cached)", f"{t['cold_p50_ms']:.0f} / {t['cold_p95_ms']:.0f} ms", "p50 <= 450, p95 <= 700", t["cold_p50_ms"] <= 450 and t["cold_p95_ms"] <= 700),
        ("point cold p50 / warm p95", f"{t['point_cold_ms']:.0f} / {t['point_warm_ms']:.0f} ms", "<= 300 / <= 30 ms", t["point_cold_ms"] <= 300 and t["point_warm_ms"] <= 30),
        ("suggest warm p50 / p95 (3 alternatives, 5 ratios)", f"{t['suggest_p50_ms']:.0f} / {t['suggest_p95_ms']:.0f} ms", "p95 <= 60 ms", t["suggest_p95_ms"] <= 60),
        ("`crop detect` cold throughput", f"{t['detect_img_s']:.1f} img/s", ">= 6", t["detect_img_s"] >= 6),
        ("`crop apply` warm throughput (suggest + accept, no XMP)", f"{t['apply_img_s']:.0f} img/s", ">= 30", t["apply_img_s"] >= 30),
        ("RSS with the sessions loaded", f"+{d_loaded:.0f} MB", f"<= +{CROP_RSS_LOADED_MB} MB", d_loaded <= CROP_RSS_LOADED_MB),
        (f"RSS retained after idle unload, worst of {len(cyc)} cycles", "+" + ", +".join(f"{x - t['base_mb']:.0f}" for x in cyc) + " MB", f"<= +{CROP_RSS_IDLE_MB} MB", max(d_idle_all) <= CROP_RSS_IDLE_MB),
        ("retained RSS stable (last two cycles)", f"{growth:+.0f} MB", f"<= {CROP_RSS_GROWTH_MB} MB", growth <= CROP_RSS_GROWTH_MB),
        ("whole-process peak (cull + indexer + query + OCR + T15-T17 + crop, one process)", "n/a" if whole_peak is None else f"{whole_peak:.0f} MB", f"<= {CROP_PEAK_MB} MB", None if whole_peak is None else whole_peak <= CROP_PEAK_MB),
        ("cull throughput (crop is not in cull)", f"{cull_imgs_s:.1f} img/s", ">= 10 and within 5 % of the last run", cull_imgs_s >= 10 and (base_cull is None or cull_imgs_s >= 0.95 * base_cull)),
    ]  # fmt: skip
    out = [P5_MARK, "", f"{t['n']}-image bench shoot, previews cached, M-series CPU.", "",
           "| Metric | Value | Target | |", "|---|---|---|---|"]  # fmt: skip
    for name, val, tgt, ok in rows:
        verdict = "FAIL (not measured)" if ok is None else ("PASS" if ok else "FAIL")
        out.append(f"| {name} | {val} | {tgt} | {verdict} |")
    return out + [""]


def run_phase4_t20() -> list[str]:
    """T20 Lightroom sync: 100k-image catalog + 10k-frame model-free shoot, then the locked-copy
    path on a 1 GB catalog. Returns the benchmark.md block."""
    import tempfile

    sys.path.insert(0, str(ROOT / "tests"))
    import large_synth as ls
    import make_fixtures
    from shotsieve.pipeline import lrsync
    from shotsieve.pipeline.runner import run_cull

    st = Settings()
    root = data_dir() / "tmp"
    root.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="bench-t20-", dir=root))
    try:
        folder = work / "shoot"
        frames = ls.build(folder, 10_000, st)
        run_cull(folder, st)
        iso = lambda t: datetime.fromtimestamp(t).strftime("%Y-%m-%dT%H:%M:%S")  # noqa: E731
        ents = [
            {"root": str(folder), "rel": "", "base": f.name[:-4], "ext": "jpg", "pick": 0,
             "rating": 0, "label": "", "touch": 1_700_000_000.0, "capture": iso(f.ts)}
            for f in frames
        ]  # fmt: skip
        ents += [
            {"root": f"/Volumes/Archive/{i // 500}", "rel": "", "base": f"OLD_{i:06d}",
             "ext": "cr3", "pick": 0, "rating": i % 6, "label": "", "touch": 1_600_000_000.0}
            for i in range(90_000)
        ]  # fmt: skip
        cat = work / "Big.lrcat"
        t0 = time.perf_counter()
        make_fixtures.build_lrcat(cat, ents)
        build_s = time.perf_counter() - t0
        for f in frames[:50]:
            make_fixtures.lr_edit(cat, folder / f.name, pick=1, touch=time.time())
        proc = psutil.Process()
        rss0 = proc.memory_info().rss
        peak = [rss0 / 1e6]
        stop = threading.Event()
        threading.Thread(target=_rss_sampler, args=(stop, peak), daemon=True).start()
        walls = []
        for _ in range(3):
            t0 = time.perf_counter()
            res = lrsync.sync(folder, cat, st, dry_run=True, report=False)
            walls.append(time.perf_counter() - t0)
        stop.set()
        d_rss = peak[0] - rss0 / 1e6
        huge = work / "Huge.lrcat"
        make_fixtures.build_lrcat(
            huge,
            [{"root": str(folder), "rel": "", "base": f.name[:-4], "ext": "jpg", "pick": 0,
              "rating": 0, "label": ""} for f in frames[:500]],
            pad_mb=1000,
        )  # fmt: skip
        code = (
            "import sqlite3, sys\n"
            f"c = sqlite3.connect({str(huge)!r}, isolation_level=None)\n"
            "c.execute('BEGIN EXCLUSIVE'); print('ready', flush=True); sys.stdin.readline()\n"
        )
        holder = subprocess.Popen([sys.executable, "-c", code], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True)  # fmt: skip
        holder.stdout.readline()
        try:
            t0 = time.perf_counter()
            r2 = lrsync.sync(folder, huge, st, dry_run=True, report=False)
            copy_s = time.perf_counter() - t0
        finally:
            holder.stdin.write("\n")
            holder.stdin.flush()
            holder.wait(10)
        size_gb = huge.stat().st_size / 2**30
    finally:
        shutil.rmtree(work, ignore_errors=True)
    ok_a = min(walls) <= 3.0 and res.n_new == 50  # n1: always read from a private copy
    ok_b = copy_s <= 10.0 and r2.copied
    ok_c = d_rss <= 200
    v = lambda b: "PASS" if b else "FAIL"  # noqa: E731
    return [
        "### T20 Lightroom sync-back",
        "",
        "| Metric | Value | Budget |",
        "|---|---|---|",
        f"| 100k-image catalog built (executemany) | {build_s:.1f} s | <= 5 s |",
        f"| Sync, 10k-frame shoot (private copy, n1) | {min(walls):.2f} s (runs: "
        f"{', '.join(f'{w:.2f}' for w in walls)}), {res.n_catalog} catalog rows read, "
        f"{res.n_new} overrides {v(ok_a)} | <= 3 s |",
        f"| Sync via the locked-copy path, {size_gb:.1f} GiB catalog | {copy_s:.1f} s "
        f"(copied={r2.copied}) {v(ok_b)} | <= 10 s |",
        f"| RSS added by the sync | +{d_rss:.0f} MB {v(ok_c)} | <= +200 MB |",
        "",
    ] + ([] if ok_a and ok_b and ok_c else ["FAIL"])


GATE_QUERIES = [
    "a portrait of a person", "a man smiling", "a woman laughing", "people standing together",
    "a group photo", "a close-up of a face", "someone with closed eyes", "a blurry photo",
    "a dark photo", "a bright sunny day", "a building", "a city street", "a landscape",
    "trees and grass", "a person outdoors", "a person indoors", "a black and white photo",
    "a wedding", "a bride", "a child playing", "a dog", "a red car", "a sunset", "a beach",
    "mountains", "a crowd", "a person looking at the camera", "a person looking away",
    "a football match", "a night scene",
]  # fmt: skip


def run_siglip_gate(args) -> int:
    """int8 vs fp32 SigLIP2 vision on the bench: mean top-20 overlap >= 0.90 and median
    per-image cosine >= 0.98 -> ship int8 (docs/07_PHASE2.md T10)."""
    from shotsieve.io.preview import load_preview
    from shotsieve.models.registry import ensure_models
    from shotsieve.models.text_image import SiglipText, SiglipVision

    ensure_models(groups=("search", "search-int8"))
    folder = args.dir or data_dir() / "bench" / f"shoot_{args.n}_{args.seed}"
    if len(list(folder.glob("*.jpg"))) != args.n:
        synthesize(args.n, folder, args.seed)
    files = sorted(folder.glob("*.jpg"))
    imgs = [load_preview(f, 1280) for f in files]
    res = {}
    for variant in ("fp32", "int8"):
        v = SiglipVision(variant)
        t0 = time.perf_counter()
        res[variant] = v.embed_batch(imgs)
        res[variant + "_s"] = time.perf_counter() - t0
        del v
    txt = SiglipText()
    qv = np.stack([txt.embed(q) for q in GATE_QUERIES])
    overlaps = []
    for row in qv:
        a = np.argsort(-(res["fp32"] @ row))[:20]
        b = np.argsort(-(res["int8"] @ row))[:20]
        overlaps.append(len(set(a.tolist()) & set(b.tolist())) / 20)
    cos = np.sum(res["fp32"] * res["int8"], axis=1)
    overlap, med = float(np.mean(overlaps)), float(np.median(cos))
    ok = overlap >= 0.90 and med >= 0.98
    lines = [
        f"int8 gate on {len(files)} bench images x {len(GATE_QUERIES)} queries: "
        f"mean top-20 overlap {overlap:.3f} (>= 0.90), median cosine {med:.4f} (>= 0.98), "
        f"min cosine {float(cos.min()):.4f}",
        f"embed time: fp32 {len(files) / res['fp32_s']:.1f} img/s, "
        f"int8 {len(files) / res['int8_s']:.1f} img/s",
        f"verdict: {'PASS -> ship int8' if ok else 'FAIL -> ship fp32'}",
    ]
    print("\n".join(lines))
    return 0 if ok else 2


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dir", type=Path, default=None, help="reuse/create images here")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--threads", type=int, default=None, help="ORT intra-op threads")
    ap.add_argument("--workers", type=int, default=None, help="preview decode workers")
    ap.add_argument("--batch", type=int, default=None, help="embedding batch size")
    ap.add_argument("--raw", action="store_true", help="benchmark copies of the fixture DNGs")
    ap.add_argument("--phase3", action="store_true", help="also run the T15-T17 checks")
    ap.add_argument("--large", type=int, default=0, help="T21: synthetic N-frame shoot (10000)")
    ap.add_argument("--phase4", action="store_true", help="also run the T18-T20 checks")
    ap.add_argument("--crop", action="store_true", help="T22 crop stage (also run by --phase4)")
    ap.add_argument("--crop-child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--siglip-gate", action="store_true", help="int8 vs fp32 SigLIP2 quality gate")
    ap.add_argument("--paused-child", choices=["cull", "paused"], help=argparse.SUPPRESS)
    ap.add_argument("--make-clip", type=Path, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--fourcc", default="avc1", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.make_clip:
        make_clip(args.make_clip, args.fourcc)
        return 0
    if args.crop_child:
        print(json.dumps(_crop_child(args.dir)))
        return 0
    if args.paused_child:
        print(json.dumps(_paused_child(args.paused_child, args.dir)))
        return 0
    if args.siglip_gate:
        return run_siglip_gate(args)
    if args.large:
        return run_large(args)
    if args.raw:
        if args.n == 300:
            args.n = 40
        return run_raw(args)

    from shotsieve.pipeline.cache import cache_path
    from shotsieve.pipeline.runner import run_cull

    folder = args.dir or data_dir() / "bench" / f"shoot_{args.n}_{args.seed}"
    if len(list(folder.glob("*.jpg"))) != args.n or not (folder / TRUTH_FILE).exists():
        shutil.rmtree(folder, ignore_errors=True)
        t0 = time.perf_counter()
        synthesize(args.n, folder, args.seed)
        print(f"synthesised {args.n} images in {time.perf_counter() - t0:.1f}s")
    truth = json.loads((folder / TRUTH_FILE).read_text())
    shutil.rmtree(folder / ".shotsieve", ignore_errors=True)

    settings = Settings()
    if args.threads:
        settings.ort_threads = args.threads
    if args.workers:
        settings.preview_workers = args.workers
    if args.batch:
        settings.embed_batch = args.batch

    peak = [psutil.Process().memory_info().rss / 1e6]
    stop = threading.Event()
    sampler = threading.Thread(target=_rss_sampler, args=(stop, peak), daemon=True)
    sampler.start()

    # model loading is measured separately so the throughput figure is steady-state
    t0 = time.perf_counter()
    from shotsieve.models.face_detect import get_face_detector
    from shotsieve.models.face_embed import get_face_embedder
    from shotsieve.models.face_landmarks import get_face_landmarker
    from shotsieve.models.image_embed import get_image_embedder

    emb = get_image_embedder(settings.ort_threads, settings.embed_batch, settings.use_coreml)
    get_face_detector(), get_face_embedder(), get_face_landmarker()
    load_s = time.perf_counter() - t0
    mark("models loaded")

    cold = run_cull(folder, settings)
    t1 = time.perf_counter()
    warm = run_cull(folder, settings)
    warm_s = time.perf_counter() - t1
    mark("cull cold + warm")
    peak_cull = peak[0]
    gt = score_truth(cold.images, truth)
    t10 = run_t10(folder, settings, peak)
    mark("cull+indexer(paused), index SigLIP, query")
    paused = run_paused_midrun(folder, settings, peak_cull)
    mark("indexer paused mid-run")
    t11 = run_t11(folder, settings, peak)
    mark("OCR cull + OCR index")
    t12 = run_t12(settings)
    mark("video")
    hevc = run_t12(settings, "video_4k30_hevc.mp4", "hvc1")
    mark("video HEVC")
    t15 = t16 = t17 = None
    if args.phase3:
        t15 = run_t15(folder, settings)
        mark("T15 editing profile")
        t16 = run_t16(folder, settings)
        mark("T16 colour match")
        t17 = run_t17(folder, settings)
        mark("T17 shot list")
    stop.set()
    sampler.join()

    identical = _sig(cold) == _sig(warm)
    cs = cold.stats
    stage = cs.get("stage_imgs_per_s", {})
    speedup = cs["total_s"] / max(warm_s, 1e-9)
    ok_speed = cs["imgs_per_s"] >= 10.0
    ok_review = cs["counts"]["review"] / cs["n_images"] <= 0.20
    ok_rss = peak[0] <= 3000
    verdict_speed = "PASS" if ok_speed else "FAIL"
    verdict_rss = "PASS" if ok_rss else "FAIL"
    lines = [
        "# ShotSieve benchmark",
        "",
        f"- Date: {time.strftime('%Y-%m-%d %H:%M')}  |  machine: "
        f"{psutil.cpu_count(logical=True)} logical CPUs, "
        f"{psutil.virtual_memory().total / 2**30:.0f} GiB RAM",
        f"- Images: {cs['n_images']} synthetic JPEGs {CANVAS[0]}x{CANVAS[1]} (seed {args.seed}); "
        f"{cold.stats['n_groups']} duplicate groups, {cs['n_bursts']} bursts",
        f"- Settings: ort_threads={settings.ort_threads}, "
        f"preview_workers={settings.preview_workers}, "
        f"embed_batch={settings.embed_batch}, DINOv2 provider={emb.provider} "
        f"({'fp32' if emb.input_dtype == np.float32 else 'fp16'})",
        "",
        "## Cold run (no cache, models pre-loaded)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Total wall time | {cs['total_s']:.2f} s |",
        f"| **Throughput** | **{cs['imgs_per_s']:.1f} img/s** (target >= 10) {verdict_speed} |",
        f"| Peak RSS (whole process, sampled 50 ms) | cull {peak_cull:.0f} MB; whole run incl. "
        f"indexer + search query {peak[0]:.0f} MB (target <= 3000) {verdict_rss} |",
        f"| Model load (one-off, excluded above) | {load_s:.2f} s |",
        f"| Decisions | {cs['counts']} (review "
        f"{100 * cs['counts']['review'] / cs['n_images']:.1f} %, target <= 20 %) |",
        "",
        "## Ground truth (frames deliberately degraded by the synthesiser are recorded)",
        "",
        f"{gt['n_clean']} clean frames, {gt['n_degraded']} degraded "
        "(blur / dark / black); a *moment* is one synthesised burst.",
        "",
        "| Metric | Value | Target |",
        "|---|---|---|",
        f"| Clean frames hard-rejected (false positives) | {gt['clean_hard_rejects']} | 0 |",
        f"| Clean frames rejected for any reason | {gt['clean_rejects']} | (duplicates only) |",
        f"| Moments with a clean frame | {gt['n_moments']} | |",
        f"| **Moment no-keep rate** (no clean frame kept; review counts as a miss) | "
        f"**{100 * gt['moment_no_keep_rate']:.1f} %** ({gt['moment_no_keep']}) | <= 1 % |",
        f"| Moment no-keep-no-review rate (nothing kept, nothing to review) | "
        f"{100 * gt['moment_no_keep_no_review_rate']:.1f} % "
        f"({gt['moment_no_keep_no_review']}) | <= 1 % |",
        f"| **Moment group-level miss** (the moment's duplicate group(s) hold no keep and no "
        f"review: the best frame was rejected; the T5 safeguard target) | "
        f"**{100 * gt['moment_group_lost_rate']:.1f} %** ({gt['moment_group_lost']}) | <= 1 % |",
        f"| Moment group-level no-keep (strict: review counts as a miss; borderline singletons) | "
        f"{100 * gt['moment_group_no_keep_rate']:.1f} % ({gt['moment_group_no_keep']}) | info |",
        "",
        "Per-stage capacity (stages run concurrently, so wall time ~ slowest stage, not the sum):",
        "",
        "| Stage | Workers | Busy time | Capacity (img/s) |",
        "|---|---|---|---|",
    ]
    par = {"features": settings.preview_workers, "faces": 1, "embed": 1}
    label = {
        "features": "decode + EXIF + stage A (exposure, sharpness)",
        "faces": "faces (YuNet + SFace + MediaPipe)",
        "embed": "DINOv2 embeddings",
    }
    for s in ("features", "faces", "embed"):
        busy = cs["timings"].get(f"{s}_busy_s", 0.0)
        cap = stage.get(s, float("nan"))
        lines.append(f"| {label[s]} | {par[s]} | {busy:.1f} s | {cap:.1f} |")
    t = cs["timings"]
    lines += [
        f"| group (bursts + duplicates, {cs['dup_pairs_compared']} pair comparisons) | 1 | "
        f"{t['group_s']:.2f} s | {cs['n_images'] / max(t['group_s'], 1e-9):.0f} |",
        f"| score + select | 1 | {t['score_s']:.2f} s | "
        f"{cs['n_images'] / max(t['score_s'], 1e-9):.0f} |",
        "",
        "## Phase 2 checks (T7 AF focus, T8 banding, T9 clock)",
        "",
        *phase2_lines(cold, gt),
        "",
        *t10_lines(t10, peak_cull, peak[0]),
        *paused_lines(paused),
        *t11_lines(t11, peak[0]),
        *t12_lines(t12, peak[0]),
        *hevc_lines(hevc),
        *t15_lines(t15, peak[0]),
        *t16_lines(t16),
        *t17_lines(t17, peak[0]),
        *stage_lines(),
        "## Cache re-run",
        "",
        f"- Warm re-run: {warm_s:.2f} s ({cs['n_images'] / warm_s:.0f} img/s), "
        f"{speedup:.0f}x faster than cold; "
        f"{warm.stats['n_cached']}/{warm.stats['n_images']} cache hits.",
        f"- Decisions identical to cold run: {identical} "
        "(decision, reasons, group, burst, audit flag and score of every image)",
        f"- Cache DB size: {cache_path(folder).stat().st_size / 1e6:.1f} MB",
        "",
    ]
    report = "\n".join(lines)
    out = args.out or data_dir() / "benchmark.md"
    keep_raw = ""
    if out.exists():
        old_blocks = re.split(r"(?m)^(?=## )", out.read_text())
        keep_raw = "\n" + "".join(
            b for b in old_blocks if b.startswith((RAW_MARK, LARGE_MARK, P4_MARK, P5_MARK))
        )
    out.write_text(report + (keep_raw if keep_raw.strip() else ""))
    print(report)
    print(f"written to {out}")
    p4_ok = True
    if args.phase4 or args.crop:
        prior = None
        if out.exists():
            m = re.search(r"cull throughput[^|]*\| ([\d.]+) img/s", out.read_text().split(P5_MARK)[-1]) if P5_MARK in out.read_text() else None
            prior = float(m.group(1)) if m else None
        whole = [max(peak[0], psutil.Process().memory_info().rss / 1e6)]
        stop2 = threading.Event()
        threading.Thread(target=_rss_sampler, args=(stop2, whole), daemon=True).start()
        t22 = run_crop(folder, settings)
        stop2.set()
        blk = crop_lines(t22, cs["imgs_per_s"], prior, whole[0])
        p4_ok = "FAIL" not in "\n".join(blk)
        _replace_section(out, P5_MARK, blk)
        print("\n".join(blk))
    if args.phase4:
        block = run_phase4_t20()
        p4_ok = p4_ok and "FAIL" not in block
        _replace_section(out, P4_MARK, [P4_MARK, "", *[b for b in block if b != "FAIL"]])
        print("\n".join(block))
    return 0 if (ok_speed and ok_rss and ok_review and identical and p4_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
