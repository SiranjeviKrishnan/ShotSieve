# ruff: noqa: E501
"""Face recall / false-positive evaluation for the T23h recognition scan.

Ground truth: the labelled `face` boxes of tests/fixtures/crop and crop2 (people in the background
are labelled with must=false) plus deterministic SYNTHETIC wide scenes (faces pasted small, with
sensor noise, onto a faceless photo) because the real fixtures have no face under 24 px.

    uv run python tests/faces_eval.py            # prints the tables (before/after, by size)

Splits: crop `train` images + synthetic scenes seeded 0-5 are the TRAIN split (thresholds are
picked there only); crop `held-out`, every crop2 image and synthetic seeds 100-105 are HELD-OUT.
False positives on the real fixtures are measured two ways: on the images with no person/face
label at all (clean negatives, exact) and on all images against the labels (an upper bound: not
every visible face is labelled). Synthetic scenes give an exact FP count."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
FIX = ROOT / "tests" / "fixtures"
BINS = (("<24", 0, 24), ("24-48", 24, 48), (">48", 48, 1e9))


def real_images() -> list[dict]:
    out = []
    for d in ("crop", "crop2"):
        labels = json.loads((FIX / d / "crop_labels.json").read_text())
        man = json.loads((FIX / d / "manifest.json").read_text())
        for name, v in labels.items():
            p = FIX / d / "real" / name
            if not p.exists():
                continue
            faces = [s["box_px"] for s in v["subjects"] if s["kind"] == "face"]
            neg = not any(s["kind"] in ("face", "person") for s in v["subjects"])
            split = "train" if man[name]["split"] == "train" else "heldout"
            out.append({"id": f"{d}/{name}", "path": p, "faces": faces, "neg": neg, "split": split})
    return out


def load_rgb(p: Path) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)


def _face_sources() -> list[tuple[np.ndarray, list[float]]]:
    """Large labelled frontal faces: (image, [x0, y0, x1, y1]) from the fixtures."""
    out = []
    for r in real_images():
        for b in r["faces"]:
            if max(b[2] - b[0], b[3] - b[1]) >= 150:
                out.append((r["id"], r["path"], b, r["split"]))
    return out


def synth_scenes(seed: int, n_faces: int = 8, size=(2400, 1600)) -> dict:
    """One wide scene: a faceless photo as background, `n_faces` small faces (12-64 px) pasted at
    random places, Gaussian sensor noise. Returns {"img", "faces": [[x0,y0,x1,y1]], "src": [...]}."""
    rng = np.random.default_rng(seed)
    heldout = seed >= 100
    split = "heldout" if heldout else "train"
    bases = [r for r in real_images() if r["neg"] and r["split"] == split]
    srcs = [s for s in _face_sources() if s[3] == split]
    base = load_rgb(bases[int(rng.integers(len(bases)))]["path"])
    W, H = size
    s = max(W / base.shape[1], H / base.shape[0])
    big = cv2.resize(base, (round(base.shape[1] * s), round(base.shape[0] * s)), cv2.INTER_CUBIC)
    canvas = big[:H, :W].astype(np.float32)
    boxes: list[list[float]] = []
    meta = []
    tries = 0
    while len(boxes) < n_faces and tries < 400:
        tries += 1
        sid, sp, sb, _ = srcs[int(rng.integers(len(srcs)))]
        img = load_rgb(sp)
        side = float(np.exp(rng.uniform(np.log(12), np.log(64))))
        fs = max(sb[2] - sb[0], sb[3] - sb[1])
        k = side / fs
        cx, cy = (sb[0] + sb[2]) / 2, (sb[1] + sb[3]) / 2
        half = fs * 0.9  # face plus context, so the pasted patch has a plausible surrounding
        x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
        x1, y1 = int(min(img.shape[1], cx + half)), int(min(img.shape[0], cy + half))
        patch = img[y0:y1, x0:x1]
        pw, ph = max(4, round(patch.shape[1] * k)), max(4, round(patch.shape[0] * k))
        patch = cv2.resize(patch, (pw, ph), interpolation=cv2.INTER_AREA).astype(np.float32)
        px = int(rng.integers(0, W - pw))
        py = int(rng.integers(0, H - ph))
        fb = [px + (sb[0] - x0) * k, py + (sb[1] - y0) * k, px + (sb[2] - x0) * k,
              py + (sb[3] - y0) * k]  # fmt: skip
        if any(
            min(fb[2], b[2]) - max(fb[0], b[0]) > -side and min(fb[3], b[3]) - max(fb[1], b[1]) > -side
            for b in boxes
        ):
            continue
        mask = np.ones((ph, pw), np.float32)
        m = max(1, min(pw, ph) // 8)
        mask[:m], mask[-m:], mask[:, :m], mask[:, -m:] = 0.0, 0.0, 0.0, 0.0
        mask = cv2.GaussianBlur(mask, (0, 0), max(0.8, m / 2))[..., None]
        reg = canvas[py : py + ph, px : px + pw]
        canvas[py : py + ph, px : px + pw] = reg * (1 - mask) + patch * mask
        boxes.append(fb)
        meta.append(sid)
    noise = rng.normal(0, 4.0, canvas.shape).astype(np.float32)
    img = np.clip(canvas + noise, 0, 255).astype(np.uint8)
    return {"img": img, "faces": boxes, "src": meta, "id": f"synth{seed}", "split": split,
            "neg": False}  # fmt: skip


def match(dets: list, gts: list) -> tuple[list[bool], int]:
    """Greedy one-to-one match (centre of the detection inside the label, size ratio 0.5-2.5);
    returns (per-gt hit flags, number of unmatched detections)."""
    used: set[int] = set()
    hits = []
    for g in gts:
        gw, gh = g[2] - g[0], g[3] - g[1]
        best, bd = None, 1e18
        for i, f in enumerate(dets):
            if i in used:
                continue
            x, y, w, h = f.bbox
            cx, cy = x + w / 2, y + h / 2
            if not (g[0] - 0.15 * gw <= cx <= g[2] + 0.15 * gw and g[1] - 0.15 * gh <= cy <= g[3] + 0.15 * gh):
                continue
            r = max(w, h) / max(gw, gh)
            if not 0.45 <= r <= 2.5:
                continue
            d = (cx - (g[0] + g[2]) / 2) ** 2 + (cy - (g[1] + g[3]) / 2) ** 2
            if d < bd:
                best, bd = i, d
        hits.append(best is not None)
        if best is not None:
            used.add(best)
    return hits, len(dets) - len(used)


def score_set(items: list[dict], dets_of) -> dict:
    """items: dicts with faces/neg; dets_of(item) -> [Face]. Returns recall by bin + FP stats."""
    tot = {b[0]: [0, 0] for b in BINS}
    fp_all = fp_neg = fp_synth = 0
    n_all = n_neg = n_synth = 0
    for it in items:
        dets = dets_of(it)
        hits, fp = match(dets, it["faces"])
        for g, h in zip(it["faces"], hits, strict=True):
            s = max(g[2] - g[0], g[3] - g[1])
            for name, lo, hi in BINS:
                if lo <= s < hi:
                    tot[name][0] += int(h)
                    tot[name][1] += 1
        if it["id"].startswith("synth"):
            fp_synth += fp
            n_synth += 1
        else:
            fp_all += fp
            n_all += 1
            if it["neg"]:
                fp_neg += fp
                n_neg += 1
    rec = {k: (v[0] / v[1] if v[1] else None, v[0], v[1]) for k, v in tot.items()}
    return {"recall": rec, "fp_real_all": fp_all / max(n_all, 1), "fp_clean_neg": fp_neg / max(n_neg, 1),
            "fp_synth": fp_synth / max(n_synth, 1), "n_real": n_all, "n_neg": n_neg,
            "n_synth": n_synth}  # fmt: skip


def baseline_dets(img: np.ndarray, det, max_faces: int = 6):
    """What the cull sees today: one 640 px pass, score >= 0.7, at most max_faces."""
    return det.detect(img, max_faces=max_faces)


def fmt(res: dict) -> str:
    r = res["recall"]
    parts = [f"{k}: {v[1]}/{v[2]}" + (f" = {v[0]:.2f}" if v[0] is not None else "") for k, v in r.items()]
    return (" | ".join(parts) + f" | FP/img real-all {res['fp_real_all']:.2f}, clean-neg "
            f"{res['fp_clean_neg']:.2f} (n={res['n_neg']}), synth {res['fp_synth']:.2f}")


def evaluate(split: str) -> dict:
    """Before (the cull's detector) and after (the recognition scan) on one split."""
    from shotsieve.models.face_detect import FaceDetector, get_rec_face_detector
    from shotsieve.pipeline import facescan as fs

    det, rdet = FaceDetector(), get_rec_face_detector()
    items = real_images()
    seeds = range(0, 6) if split == "train" else range(100, 106)
    items = [i for i in items if i["split"] == split] + [synth_scenes(s) for s in seeds]
    cache: dict[str, dict] = {}
    for it in items:
        img = it["img"] if "img" in it else load_rgb(it["path"])
        import time

        t0 = time.perf_counter()
        b = baseline_dets(img, det)
        t1 = time.perf_counter()
        a = fs.scan(img, rdet)
        t2 = time.perf_counter()
        cache[it["id"]] = {"before": b, "after": a, "tb": t1 - t0, "ta": t2 - t1}
    out = {
        "before": score_set(items, lambda i: cache[i["id"]]["before"]),
        "after": score_set(items, lambda i: cache[i["id"]]["after"]),
        "t_before": float(np.mean([v["tb"] for v in cache.values()])),
        "t_after": float(np.mean([v["ta"] for v in cache.values()])),
        "n": len(items),
    }
    return out


if __name__ == "__main__":
    for sp in ("train", "heldout"):
        r = evaluate(sp)
        print(f"== {sp} ({r['n']} images; per-image detect before {r['t_before']*1000:.0f} ms, after {r['t_after']*1000:.0f} ms)")
        print("before:", fmt(r["before"]))
        print("after: ", fmt(r["after"]))
