"""Multi-camera clock offset: find the constant shift between two bodies' clocks from frames
that show the same moment.

`suggest_offsets(images, emb, face_emb)` takes the camera with the most EXIF-timestamped frames
as the reference and, for every other body, finds mutual nearest neighbours by DINOv2 cosine
(chunked: never an n x m matrix), turns each into a time difference `delta = t_ref - t_cam`, and
looks for a mode in the delta histogram. Applied as `t_adj = t + offset[serial]`.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

CHUNK = 1024
COS_MIN = 0.65
FACE_BONUS = 0.05
FACE_SIM_MIN = 0.40
BIN_S = 2.0
RANGE_S = 26 * 3600.0
MODE_HALF_S = 3.0
MIN_SUPPORT = 3
MIN_SUPPORT_FRAC = 0.20
AUTO_CONFIDENCE = 0.6
AUTO_SUPPORT = 10
TOP_ALTERNATIVES = 5


@dataclass(frozen=True)
class ClockImage:
    path: str
    serial: str
    ts: float
    ts_source: str = "exif"
    tz_offset_s: float | None = None  # EXIF OffsetTimeOriginal in seconds east of UTC


@dataclass
class Pair:
    ref_path: str
    path: str
    cos: float
    delta_s: float

    def to_dict(self) -> dict:
        return {"ref": self.ref_path, "cam": self.path, "cos": self.cos, "delta_s": self.delta_s}


@dataclass
class OffsetSuggestion:
    serial: str
    offset_s: float | None  # None: no consistent offset, pick a pair from `alternatives`
    confidence: float = 0.0
    support: int = 0
    n_candidates: int = 0
    pair: Pair | None = None  # best-matching candidate inside the mode
    alternatives: list[Pair] = field(default_factory=list)
    ref_serial: str = ""
    source: str = "auto"  # auto | tz
    auto_apply: bool = False  # confidence >= 0.6 and support >= 10

    def to_dict(self) -> dict:
        return {
            "serial": self.serial, "offset_s": self.offset_s, "confidence": self.confidence,
            "support": self.support, "n_candidates": self.n_candidates,
            "pair": self.pair.to_dict() if self.pair else None,
            "alternatives": [p.to_dict() for p in self.alternatives],
            "ref_serial": self.ref_serial, "source": self.source, "auto_apply": self.auto_apply,
        }  # fmt: skip


_TZ = re.compile(r"^\s*([+-])(\d{1,2}):?(\d{2})?\s*$")


def parse_tz(text: str | None) -> float | None:
    """'+02:00' / '-0530' -> seconds east of UTC; None when absent or unparseable."""
    if not text:
        return None
    m = _TZ.match(str(text))
    if not m:
        return None
    s = int(m.group(2)) * 3600 + int(m.group(3) or 0) * 60
    return -float(s) if m.group(1) == "-" else float(s)


def parse_offset(text: str) -> float:
    """'+3725', '-90.5' or '+1:02:05' / '-0:00:30' (h:mm:ss) -> seconds."""
    t = text.strip()
    sign = -1.0 if t.startswith("-") else 1.0
    t = t.lstrip("+-")
    if ":" in t:
        parts = [float(x) for x in t.split(":")]
        while len(parts) < 3:
            parts.insert(0, 0.0)
        h, m, s = parts[-3:]
        return sign * (h * 3600 + m * 60 + s)
    return sign * float(t)


def mutual_nn(
    a: np.ndarray, b: np.ndarray, chunk: int = CHUNK
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mutual nearest neighbours (by dot product of unit vectors) between rows of `a` and `b`.

    Processes `a` in chunks of `chunk` rows, keeping a running best per column of `b`, so the
    memory is O(chunk x len(b)). Returns (index into a, index into b, cosine)."""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    na, nb = len(a), len(b)
    if not na or not nb:
        z = np.zeros(0, np.int64)
        return z, z, np.zeros(0, np.float32)
    best_ab = np.empty(na, np.int64)
    val_ab = np.empty(na, np.float32)
    best_ba = np.zeros(nb, np.int64)
    val_ba = np.full(nb, -np.inf, np.float32)
    for s in range(0, na, chunk):
        sims = a[s : s + chunk] @ b.T  # (chunk, nb)
        j = sims.argmax(axis=1)
        best_ab[s : s + chunk] = j
        val_ab[s : s + chunk] = sims[np.arange(len(j)), j]
        i = sims.argmax(axis=0)
        v = sims[i, np.arange(nb)]
        better = v > val_ba
        val_ba[better] = v[better]
        best_ba[better] = i[better] + s
    ia = np.nonzero(best_ba[best_ab] == np.arange(na))[0]
    return ia, best_ab[ia], val_ab[ia]


def _first_tz(images: Sequence[ClockImage], idx: np.ndarray) -> float | None:
    return next((images[i].tz_offset_s for i in idx if images[i].tz_offset_s is not None), None)


def _faces_match(fa: np.ndarray | None, fb: np.ndarray | None) -> bool:
    if fa is None or fb is None or not len(fa) or not len(fb):
        return False
    return bool((np.asarray(fa) @ np.asarray(fb).T).max() >= FACE_SIM_MIN)


def _mode(delta: np.ndarray) -> tuple[float, int] | None:
    """(median delta in the mode +- 3 s, support) from a 2 s histogram over +-26 h."""
    d = delta[np.abs(delta) <= RANGE_S]
    if not len(d):
        return None
    edges = np.arange(-RANGE_S, RANGE_S + BIN_S, BIN_S)
    hist, _ = np.histogram(d, bins=edges)
    win = hist[:-2] + hist[1:-1] + hist[2:] if len(hist) > 2 else hist  # 3 bins = 6 s wide
    k = int(np.argmax(win)) + 1
    center = edges[k] + BIN_S / 2
    near = d[np.abs(d - center) <= MODE_HALF_S + BIN_S]
    if not len(near):
        return None
    med = float(np.median(near))
    inside = d[np.abs(d - med) <= MODE_HALF_S]
    return med, int(len(inside))


def suggest_offsets(
    images: Sequence[ClockImage],
    emb: np.ndarray,
    face_emb: Sequence[np.ndarray | None] | None = None,
    offsets: dict[str, float] | None = None,
    ref: str | None = None,
) -> list[OffsetSuggestion]:
    """One suggestion per non-reference camera (empty when fewer than two EXIF-clocked bodies).

    Offsets are *absolute* corrections keyed by body serial, derived from the raw EXIF times:
    `offsets` are the corrections already stored, and the reference body's own one is part of
    the time domain the others are aligned to (`delta = t_ref + off_ref - t_cam`). The result
    therefore does not depend on which body is the reference, so a reference that changes between
    runs (`ref` pins the persisted one) cannot make an offset get applied twice."""
    offsets = offsets or {}
    idx_by: dict[str, list[int]] = {}
    for i, im in enumerate(images):
        if im.ts_source == "exif" and im.serial and im.serial != "unknown":
            idx_by.setdefault(im.serial, []).append(i)
    if len(idx_by) < 2:
        return []
    if ref not in idx_by:
        ref = max(idx_by, key=lambda s: (len(idx_by[s]), s))
    off_ref = offsets.get(ref, 0.0)
    ri = np.array(idx_by[ref])
    tz_ref = _first_tz(images, ri)
    out = []
    for serial in sorted(s for s in idx_by if s != ref):
        ci = np.array(idx_by[serial])
        ia, ib, cos = mutual_nn(emb[ri], emb[ci])
        keep = cos >= COS_MIN
        ia, ib, cos = ia[keep], ib[keep], cos[keep]
        pairs, score = [], []
        for a, b, c in zip(ia, ib, cos, strict=True):
            ga, gb = ri[a], ci[b]
            bonus = 0.0
            if face_emb is not None and _faces_match(face_emb[ga], face_emb[gb]):
                bonus = FACE_BONUS
            pairs.append(
                Pair(
                    images[ga].path, images[gb].path, float(c),
                    images[ga].ts + off_ref - images[gb].ts,
                )  # fmt: skip
            )
            score.append(float(c) + bonus)
        sug = _suggest_one(serial, ref, pairs, np.array(score))
        tz_cam = _first_tz(images, ci)
        if tz_ref is not None and tz_cam is not None and tz_ref != tz_cam:
            sug = _apply_tz_prior(sug, pairs, score, tz_ref - tz_cam + off_ref)
        out.append(sug)
    return out


def _suggest_one(serial: str, ref: str, pairs: list[Pair], score: np.ndarray) -> OffsetSuggestion:
    n = len(pairs)
    sug = OffsetSuggestion(serial, None, n_candidates=n, ref_serial=ref)
    order = np.argsort(-score) if n else np.zeros(0, int)
    sug.alternatives = [pairs[i] for i in order[:TOP_ALTERNATIVES]]
    if not n:
        return sug
    delta = np.array([p.delta_s for p in pairs])
    mode = _mode(delta)
    if mode is None:
        return sug
    off, support = mode
    if support < max(MIN_SUPPORT, MIN_SUPPORT_FRAC * n):
        return sug
    inside = [i for i in order if abs(delta[i] - off) <= MODE_HALF_S]
    sug.offset_s, sug.support, sug.confidence = off, support, support / n
    sug.pair = pairs[inside[0]]
    sug.alternatives = []
    sug.auto_apply = sug.confidence >= AUTO_CONFIDENCE and support >= AUTO_SUPPORT
    return sug


def _apply_tz_prior(
    sug: OffsetSuggestion, pairs: list[Pair], score: list[float], prior_s: float
) -> OffsetSuggestion:
    """A time-zone mismatch (OffsetTimeOriginal differs) is a candidate offset: honour it when
    matching frames sit on it and the histogram mode did not already find the same thing."""
    near = [i for i, p in enumerate(pairs) if abs(p.delta_s - prior_s) <= MODE_HALF_S]
    if not near or (sug.offset_s is not None and abs(sug.offset_s - prior_s) <= MODE_HALF_S):
        return sug
    if sug.offset_s is not None and sug.support >= len(near):
        return sug  # a stronger mode elsewhere wins over the prior
    best = max(near, key=lambda i: score[i])
    med = float(np.median([pairs[i].delta_s for i in near]))
    n = len(pairs)
    return OffsetSuggestion(
        sug.serial, med, confidence=len(near) / max(n, 1), support=len(near), n_candidates=n,
        pair=pairs[best], ref_serial=sug.ref_serial, source="tz", auto_apply=False,
    )  # fmt: skip


# ---- persistence / application ------------------------------------------------------------


def offsets_from_meta(raw: dict | None) -> dict[str, float]:
    """{serial: seconds} from the persisted `meta.clock_offsets`."""
    return {
        s: float(v["offset_s"]) for s, v in (raw or {}).items() if v.get("offset_s") is not None
    }


def set_offset(
    raw: dict | None, serial: str, offset_s: float, source: str, confidence: float | None = None
) -> dict:
    """New `clock_offsets` mapping; a manual offset is never replaced by an auto one."""
    cur = dict(raw or {})
    if source != "manual" and cur.get(serial, {}).get("source") == "manual":
        return cur
    cur[serial] = {"offset_s": float(offset_s), "source": source}
    if confidence is not None:
        cur[serial]["confidence"] = float(confidence)
    return cur


def adjust(ts: float | None, serial: str, offsets: dict[str, float]) -> float | None:
    """t_adj = t + offset[serial]."""
    return None if ts is None else ts + offsets.get(serial, 0.0)


def analyse(cache, auto_apply: bool = True) -> tuple[list[OffsetSuggestion], dict[str, dict]]:
    """Suggestions for a cached shoot. With `auto_apply`, a confident suggestion (confidence >=
    0.6 and support >= 10) is persisted as an `auto` offset (manual offsets win). The suggestions
    are stored in `meta.clock_suggestions`. Returns (suggestions, current offsets)."""
    items = cache.clock_inputs()
    scene = cache.scene_embeddings()
    items = [i for i in items if i.path in scene]
    if len({i.serial for i in items if i.ts_source == "exif"}) < 2:
        cache.set_meta("clock_suggestions", [])
        return [], cache.get_clock_offsets()
    faces = cache.face_embeddings()
    emb = np.stack([scene[i.path] for i in items])
    stored = offsets_from_meta(cache.get_clock_offsets())
    serials = {i.serial for i in items if i.ts_source == "exif"}
    ref = cache.get_meta("clock_ref")
    sugs = suggest_offsets(
        items, emb, [faces.get(i.path) for i in items], offsets=stored, ref=ref
    )
    if sugs:
        new_ref = sugs[0].ref_serial
        if new_ref != ref and new_ref in serials:
            cache.set_meta("clock_ref", new_ref)  # keep the reference stable across runs
    if auto_apply:
        for sg in sugs:
            if sg.auto_apply and sg.offset_s is not None:
                cache.set_clock_offset(sg.serial, sg.offset_s, "auto", sg.confidence)
    cache.set_meta("clock_suggestions", [sg.to_dict() for sg in sugs])
    return sugs, cache.get_clock_offsets()
