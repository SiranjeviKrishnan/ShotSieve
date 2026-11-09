"""Subject-aware crop search (docs/11_CROP.md section 3.4-3.5). Pure numpy, no models.

All rectangles are display-normalised (x, y in [0, 1] per axis). A candidate crop is a rect `C`;
its aspect in full-resolution pixels is fixed by the ratio, so in normalised units
`h_n = w_n * k` with `k = fw*b / (fh*a)`.

Hard constraints (a violating candidate is never returned): inside the frame, exact ratio, every
selected face's safe box fully inside, every other selected safe box >= 98 % inside, minimum size.
When no candidate can satisfy them at this ratio the result is flagged `infeasible` and carries
the largest crop that covers the most safe-box area (faces first): never a silent cut.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from shotsieve.pipeline.crop_geom import Ratio, Rect, snap_rect

CROP_WEIGHTS: dict[str, float] = {
    "thirds": 0.24, "head": 0.12, "lead": 0.06, "cut": 0.12, "strangers": 0.12,
    "clear": 0.08, "sal": 0.12, "size": 0.10, "horizon": 0.04,
}  # fmt: skip
SAME_WEIGHT = 0.15
N_WIDTHS = 24
N_POS = 25
MIN_AREA = 0.15
MIN_LONG_EDGE_PX = 1200
DIVERSITY_IOU = 0.70
CONTAIN_NONFACE = 0.98
PAD_PX = 1.5  # search padding so the integer snap can never cut a safe box
FREE_PRESETS = ("original", "1:1", "4:5", "5:4", "2:3", "3:2", "16:9", "9:16")

X_LINES = ((1 / 3, 1.0), (1 / 2, 0.85), (2 / 3, 1.0))
Y_LINES_PERSON = ((1 / 3, 1.0), (1 / 2, 0.7))
Y_LINES_OTHER = ((1 / 3, 1.0), (1 / 2, 0.85), (2 / 3, 1.0))
THIRDS_SIGMA = 0.07
HALO_MIN_OUT = 1  # patches outside the crop before `subject_may_be_cut` (fitted on train)
HALO_MIN_FRAC = 0.0


@dataclass
class Subject:
    sid: str
    kind: str  # face | person | animal | object | point
    box: tuple[float, float, float, float]
    score: float = 1.0
    source: str = ""
    label: str = ""
    # YuNet order: right eye, left eye, nose tip, right mouth corner, left mouth corner
    landmarks: tuple[tuple[float, float], ...] | None = None
    linked: str | None = None  # face -> person sid, person -> face sid
    select_mode: str | None = None  # "person" | "face" for a selected face/person
    ext: tuple[float, float, float, float] | None = None  # box grown over same-subject patches
    halo: tuple[int, int, str] | None = None  # packed patch mask a crop edge should not cross

    @property
    def w(self) -> float:
        return self.box[2] - self.box[0]

    @property
    def h(self) -> float:
        return self.box[3] - self.box[1]

    @property
    def area(self) -> float:
        return max(self.w, 0.0) * max(self.h, 0.0)


@dataclass
class Candidate:
    rect: Rect
    score: float
    terms: dict[str, float]
    warnings: list[str] = field(default_factory=list)
    rect_px: tuple[int, int, int, int] | None = None


@dataclass
class SuggestResult:
    candidates: list[Candidate]
    infeasible: bool
    ratio_px: tuple[float, float] | None
    ratio: str = ""


# ---- safe boxes ------------------------------------------------------------------------------


def _clip(b: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return (max(0.0, b[0]), max(0.0, b[1]), min(1.0, b[2]), min(1.0, b[3]))


def face_safe(b: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Face box + 25 % of w each side, 45 % of h on top (hair), 20 % below (chin)."""
    w, h = b[2] - b[0], b[3] - b[1]
    return _clip((b[0] - 0.25 * w, b[1] - 0.45 * h, b[2] + 0.25 * w, b[3] + 0.20 * h))


def grown(b: tuple[float, float, float, float], frac: float) -> tuple[float, float, float, float]:
    w, h = b[2] - b[0], b[3] - b[1]
    return _clip((b[0] - frac * w, b[1] - frac * h, b[2] + frac * w, b[3] + frac * h))


@dataclass
class Core:
    box: tuple[float, float, float, float]
    is_face: bool
    sid: str


def cores(selected: list[Subject], everything: list[Subject]) -> list[Core]:
    """Must-keep boxes of the selection: face -> face safe box; person (mode "person") -> person
    box + 2 % plus its linked face's safe box; "face" mode -> the face safe box only; animal /
    object / point -> box + 3 %."""
    by_sid = {s.sid: s for s in everything}
    out: list[Core] = []
    for s in selected:
        if s.kind == "face":
            out.append(Core(face_safe(s.box), True, s.sid))
        elif s.kind == "person":
            if s.select_mode == "face":
                f = by_sid.get(s.linked or "")
                if f is not None and f.kind == "face":
                    out.append(Core(face_safe(f.box), True, f.sid))
                continue
            out.append(Core(grown(s.box, 0.02), False, s.sid))
            f = by_sid.get(s.linked or "")
            if f is not None and f.kind == "face":
                out.append(Core(face_safe(f.box), True, f.sid))
        else:
            out.append(Core(grown(s.ext or s.box, 0.03 if s.ext is None else 0.01), False, s.sid))
    return out


# ---- anchors -----------------------------------------------------------------------------------


def anchor_of(s: Subject, everything: list[Subject]) -> tuple[float, float]:
    b = s.box
    if s.kind == "face":
        return _face_anchor(s)
    if s.kind == "person":
        f = next((x for x in everything if x.sid == s.linked and x.kind == "face"), None)
        if f is not None:
            return _face_anchor(f)
        return ((b[0] + b[2]) / 2, b[1] + 0.07 * (b[3] - b[1]))
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def _face_anchor(f: Subject) -> tuple[float, float]:
    b = f.box
    if f.landmarks and len(f.landmarks) >= 2:
        (ax, ay), (bx, by) = f.landmarks[0], f.landmarks[1]
        return ((ax + bx) / 2, (ay + by) / 2)
    return ((b[0] + b[2]) / 2, b[1] + 0.38 * (b[3] - b[1]))


def _facing(f: Subject) -> int:
    """+1 looks to the right of the frame, -1 left, 0 frontal/unknown (nose vs eye midpoint)."""
    if not f.landmarks or len(f.landmarks) < 3:
        return 0
    ex = (f.landmarks[0][0] + f.landmarks[1][0]) / 2
    off = f.landmarks[2][0] - ex
    w = f.box[2] - f.box[0]
    if abs(off) <= 0.08 * w:
        return 0
    return 1 if off > 0 else -1


# ---- saliency integral image -----------------------------------------------------------------


class _Sat:
    def __init__(self, sal: np.ndarray) -> None:
        s = np.asarray(sal, dtype=np.float64)
        self.gh, self.gw = s.shape
        self.cum = np.zeros((self.gh + 1, self.gw + 1))
        self.cum[1:, 1:] = s.cumsum(0).cumsum(1)
        self.total = float(self.cum[-1, -1])

    def _at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        gx = np.clip(x, 0.0, 1.0) * self.gw
        gy = np.clip(y, 0.0, 1.0) * self.gh
        x0 = np.minimum(np.floor(gx).astype(int), self.gw - 1)
        y0 = np.minimum(np.floor(gy).astype(int), self.gh - 1)
        fx, fy = gx - x0, gy - y0
        c = self.cum
        return (
            c[y0, x0] * (1 - fx) * (1 - fy) + c[y0, x0 + 1] * fx * (1 - fy)
            + c[y0 + 1, x0] * (1 - fx) * fy + c[y0 + 1, x0 + 1] * fx * fy
        )  # fmt: skip

    def mass(self, x0, y0, x1, y1) -> np.ndarray:
        return self._at(x1, y1) - self._at(x0, y1) - self._at(x1, y0) + self._at(x0, y0)


# ---- the scorer ------------------------------------------------------------------------------


@dataclass
class Scene:
    """Everything the scorer needs, precomputed once per (image, selection)."""

    selected: list[Subject]
    everything: list[Subject]
    fw: int
    fh: int
    saliency: np.ndarray | None = None
    horizon: tuple[float, float] | None = None  # (y normalised, tilt degrees)
    same_ref: tuple[float, float] | None = None  # burst: anchor offset (u, v) inside the crop of A

    def __post_init__(self) -> None:
        self.cores = cores(self.selected, self.everything)
        face_cores = [c for c in self.cores if c.is_face]
        self.face_boxes = np.array([c.box for c in face_cores]).reshape(-1, 4)
        other = [c for c in self.cores if not c.is_face]
        self.other_boxes = np.array([c.box for c in other]).reshape(-1, 4)
        self.other_area = np.maximum(
            (self.other_boxes[:, 2] - self.other_boxes[:, 0])
            * (self.other_boxes[:, 3] - self.other_boxes[:, 1]), 1e-12)  # fmt: skip
        allb = np.array([c.box for c in self.cores]).reshape(-1, 4)
        if len(allb):
            self.R = (allb[:, 0].min(), allb[:, 1].min(), allb[:, 2].max(), allb[:, 3].max())
        else:
            self.R = None
        sel_sids = {s.sid for s in self.selected}
        for s in self.selected:
            if s.linked:
                sel_sids.add(s.linked)
        self.strangers = [
            s for s in self.everything
            if s.kind in ("face", "person") and s.sid not in sel_sids and s.area > 0
        ]  # fmt: skip
        self.sat = (
            _Sat(self.saliency) if self.saliency is not None and np.size(self.saliency) else None
        )
        if self.sat is not None and self.sat.total <= 0:
            self.sat = None
        # anchors, area weighted
        anc = [(anchor_of(s, self.everything), max(s.area, 1e-6), s) for s in self.selected]
        self.anchor = None
        self.anchor_person = False
        if anc:
            tw = sum(a[1] for a in anc)
            self.anchor = (
                sum(a[0][0] * a[1] for a in anc) / tw,
                sum(a[0][1] * a[1] for a in anc) / tw,
            )
            self.anchor_person = all(s.kind in ("face", "person") for _, _, s in anc)
        heads = [c for c in self.cores if c.is_face] + [
            c for c in self.cores
            if not c.is_face and any(s.sid == c.sid and s.kind == "person" for s in self.selected)
        ]  # fmt: skip
        self.head_top = min((c.box[1] for c in heads), default=None)
        self.persons = [
            s for s in self.selected
            if s.kind == "person" and s.select_mode != "face"
            and s.h * self.fh > 1.4 * s.w * self.fw
        ]  # fmt: skip
        self.halos: list[tuple[np.ndarray, int]] = []
        from shotsieve.models.patch_features import unpack_mask

        for s in self.selected:
            if s.halo is not None and not (s.kind == "person" and s.select_mode == "face"):
                m = unpack_mask(s.halo)
                self.halos.append((m, int(m.sum())))
        self.lead = None
        if len(self.selected) == 1 and self.selected[0].kind == "face":
            self.lead = self.selected[0]

    # -- hard constraints --
    def feasible_mask(self, x0, y0, x1, y1) -> np.ndarray:
        ok = np.ones(x0.shape, dtype=bool)
        e = 1e-9
        for b in self.face_boxes:
            ok &= (x0 <= b[0] + e) & (y0 <= b[1] + e) & (x1 >= b[2] - e) & (y1 >= b[3] - e)
        for b, a in zip(self.other_boxes, self.other_area, strict=True):
            iw = np.clip(np.minimum(x1, b[2]) - np.maximum(x0, b[0]), 0, None)
            ih = np.clip(np.minimum(y1, b[3]) - np.maximum(y0, b[1]), 0, None)
            ok &= iw * ih / a >= CONTAIN_NONFACE - 1e-9
        return ok

    def covered(self, x0, y0, x1, y1) -> np.ndarray:
        """Infeasible-case objective: covered safe area, faces first (lexicographic weights); each
        box counts by sqrt(its area), so a large subject outweighs a distant small one."""
        tot = np.zeros(x0.shape)
        for b in self.face_boxes:
            a = max((b[2] - b[0]) * (b[3] - b[1]), 1e-12)
            iw = np.clip(np.minimum(x1, b[2]) - np.maximum(x0, b[0]), 0, None)
            ih = np.clip(np.minimum(y1, b[3]) - np.maximum(y0, b[1]), 0, None)
            tot += 1000.0 * math.sqrt(a) * iw * ih / a
        for b, a in zip(self.other_boxes, self.other_area, strict=True):
            iw = np.clip(np.minimum(x1, b[2]) - np.maximum(x0, b[0]), 0, None)
            ih = np.clip(np.minimum(y1, b[3]) - np.maximum(y0, b[1]), 0, None)
            tot += math.sqrt(a) * iw * ih / a
        return tot

    def halo_cut(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        """A crop edge passes through a selected subject's grown region (the patches that look like
        the subject): some of them lie outside the crop while others lie inside."""
        for m, n in self.halos:
            gh, gw = m.shape
            ys = (np.arange(gh) + 0.5) / gh
            xs = (np.arange(gw) + 0.5) / gw
            inside = ((ys >= y0) & (ys <= y1))[:, None] & ((xs >= x0) & (xs <= x1))[None, :]
            n_in = int((m & inside).sum())
            n_out = n - n_in
            if n_in > 0 and n_out >= max(HALO_MIN_OUT, HALO_MIN_FRAC * n):
                return True
        return False

    # -- soft terms --
    def terms(self, x0, y0, x1, y1) -> dict[str, np.ndarray]:
        w, h = x1 - x0, y1 - y0
        n = x0.shape
        one = np.ones(n)
        t: dict[str, np.ndarray] = {}
        # thirds
        if self.anchor is None:
            t["thirds"] = one.copy()
        else:
            u = (self.anchor[0] - x0) / w
            v = (self.anchor[1] - y0) / h
            ex = _line_score(u, X_LINES)
            ey = _line_score(v, Y_LINES_PERSON if self.anchor_person else Y_LINES_OTHER)
            t["thirds"] = (ex + ey) / 2
        # headroom
        if self.head_top is None:
            t["head"] = one.copy()
        else:
            hr = (self.head_top - y0) / h
            t["head"] = np.where(
                hr < 0.04, np.clip(hr / 0.04, 0, 1),
                np.where(hr <= 0.18, 1.0, np.clip((0.40 - hr) / (0.40 - 0.18), 0, 1)),
            )  # fmt: skip
        # lead room
        t["lead"] = one.copy()
        if self.lead is not None:
            d = _facing(self.lead)
            if d != 0:
                fcx = (self.lead.box[0] + self.lead.box[2]) / 2
                ahead = (x1 - fcx) if d > 0 else (fcx - x0)
                behind = (fcx - x0) if d > 0 else (x1 - fcx)
                t["lead"] = np.where(ahead >= behind, 1.0, 0.5)
        # cut limbs / sides through selected persons
        cut = one.copy()
        for p in self.persons:
            ph = p.h
            for edge in (y0, y1):
                crossing = (edge > p.box[1] + 1e-9) & (edge < p.box[3] - 1e-9) & (ph / h >= 0.4)
                rel = (edge - p.box[1]) / ph
                bad = (
                    ((rel >= 0.12) & (rel <= 0.20))
                    | ((rel >= 0.70) & (rel <= 0.80))
                    | (rel >= 0.92)
                )
                cut = np.where(crossing & bad, 0.0, cut)
            for e_ in (x0, x1):
                crossing = (e_ > p.box[0] + 1e-9) & (e_ < p.box[2] - 1e-9)
                cut = np.where(crossing, np.minimum(cut, 0.3), cut)
        t["cut"] = cut
        # strangers
        ncut = np.zeros(n)
        for s in self.strangers:
            b = s.box
            iw = np.clip(np.minimum(x1, b[2]) - np.maximum(x0, b[0]), 0, None)
            ih = np.clip(np.minimum(y1, b[3]) - np.maximum(y0, b[1]), 0, None)
            f = iw * ih / s.area
            ncut += ((f > 0.05) & (f < 0.95)).astype(float)
        t["strangers"] = np.clip(1.0 - 0.5 * ncut, 0.0, 1.0)
        # clearance
        if self.R is None:
            t["clear"] = one.copy()
        else:
            r = self.R
            eps = 1e-6
            gaps = [
                np.where(r[0] <= eps, 1.0, (r[0] - x0) / w),
                np.where(r[2] >= 1 - eps, 1.0, (x1 - r[2]) / w),
                np.where(r[1] <= eps, 1.0, (r[1] - y0) / h),
                np.where(r[3] >= 1 - eps, 1.0, (y1 - r[3]) / h),
            ]
            t["clear"] = np.clip(np.minimum.reduce(gaps) / 0.04, 0.0, 1.0)
        # saliency
        if self.sat is None:
            t["sal"] = one.copy()
        else:
            inside = self.sat.mass(x0, y0, x1, y1)
            frac = inside / self.sat.total
            sx, sy = 0.02 * w, 0.02 * h
            strip = np.zeros(n)
            eps = 1e-6
            strip += np.where(x0 > eps, self.sat.mass(x0, y0, x0 + sx, y1), 0.0)
            strip += np.where(x1 < 1 - eps, self.sat.mass(x1 - sx, y0, x1, y1), 0.0)
            strip += np.where(y0 > eps, self.sat.mass(x0, y0, x1, y0 + sy), 0.0)
            strip += np.where(y1 < 1 - eps, self.sat.mass(x0, y1 - sy, x1, y1), 0.0)
            ratio = np.clip(strip / np.maximum(inside, 1e-12), 0.0, 1.0)
            t["sal"] = np.clip(frac * (1.0 - ratio), 0.0, 1.0)
        t["size"] = np.sqrt(w * h)
        # horizon
        if self.horizon is None:
            t["horizon"] = one.copy()
        else:
            yh = (self.horizon[0] - y0) / h
            t["horizon"] = np.where(np.abs(yh - 0.5) <= 0.04, 0.3, 1.0)
        if self.same_ref is not None and self.anchor is not None:
            u = (self.anchor[0] - x0) / w
            v = (self.anchor[1] - y0) / h
            d2 = (u - self.same_ref[0]) ** 2 + (v - self.same_ref[1]) ** 2
            t["same"] = np.exp(-d2 / (2 * 0.05**2))
        return t

    def total(self, t: dict[str, np.ndarray]) -> np.ndarray:
        wts = dict(CROP_WEIGHTS)
        if "same" in t:
            wts = {k: v * (1 - SAME_WEIGHT) for k, v in wts.items()}
            wts["same"] = SAME_WEIGHT
        out = np.zeros(next(iter(t.values())).shape)
        for k, v in wts.items():
            out += v * t[k]
        return out


def _line_score(u: np.ndarray, lines) -> np.ndarray:
    best = np.zeros(u.shape)
    for pos, wt in lines:
        best = np.maximum(best, wt * np.exp(-((u - pos) ** 2) / (2 * THIRDS_SIGMA**2)))
    return best


# ---- search ----------------------------------------------------------------------------------


def _grid(lo: np.ndarray, hi: np.ndarray, n: int) -> np.ndarray:
    """(len(lo), n) positions between lo and hi (a degenerate interval gives one repeated value)."""
    t = np.linspace(0.0, 1.0, n)
    return lo[:, None] + (hi - lo)[:, None] * t[None, :]


def _candidates(
    scene: Scene, a: float, b: float, min_long_px: int, free: tuple[float, float] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, float, float]:
    """Candidate (x0, y0, w) arrays for one ratio. Returns also `infeasible` and the width step."""
    fw, fh = scene.fw, scene.fh
    k = fw * b / (fh * a)  # h_n = w_n * k
    wmax = min(1.0, 1.0 / k)
    # padded R (so the integer snap cannot cut a safe box)
    R = scene.R
    if R is not None:
        px, py = PAD_PX / fw, PAD_PX / fh
        Rp = (max(0.0, R[0] - px), max(0.0, R[1] - py), min(1.0, R[2] + px), min(1.0, R[3] + py))
        w_need = max(Rp[2] - Rp[0], (Rp[3] - Rp[1]) / k)
    else:
        Rp, w_need = None, 0.0
    # minimum size (area >= 15 % and long edge >= min_long_px, relaxed on small images)
    w_area = math.sqrt(MIN_AREA / k)
    long_frame = max(fw, fh)
    need_px = min(min_long_px, 0.4 * long_frame)
    # crop pixel dims: W = w_n * fw, H = w_n * k * fh; long edge = w_n * max(fw, k*fh)
    w_long = need_px / max(fw, k * fh)
    w_min = min(max(w_area, w_long), wmax)
    infeasible = bool(Rp is not None and w_need > wmax + 1e-9)
    if infeasible:
        ws = np.array([wmax])
    else:
        lo = min(max(w_min, w_need), wmax)
        ws = np.geomspace(wmax, lo, N_WIDTHS) if lo < wmax - 1e-9 else np.array([wmax])
    hs = ws * k
    if Rp is None or infeasible:
        xlo = np.zeros_like(ws)
        xhi = 1.0 - ws
        ylo = np.zeros_like(ws)
        yhi = 1.0 - hs
    else:
        xlo = np.maximum(0.0, Rp[2] - ws)
        xhi = np.minimum(Rp[0], 1.0 - ws)
        ylo = np.maximum(0.0, Rp[3] - hs)
        yhi = np.minimum(Rp[1], 1.0 - hs)
        xlo, xhi = np.minimum(xlo, xhi), np.maximum(xlo, xhi)
        ylo, yhi = np.minimum(ylo, yhi), np.maximum(ylo, yhi)
    gx = _grid(xlo, xhi, N_POS)
    gy = _grid(ylo, yhi, N_POS)
    nw = len(ws)
    X0 = np.broadcast_to(gx[:, :, None], (nw, N_POS, N_POS)).ravel()
    Y0 = np.broadcast_to(gy[:, None, :], (nw, N_POS, N_POS)).ravel()
    W = np.broadcast_to(ws[:, None, None], (nw, N_POS, N_POS)).ravel()
    step = float(ws[0] - ws[1]) if nw > 1 else 0.0
    return X0, Y0, W, infeasible, k, step


def _diverse(order: np.ndarray, x0, y0, x1, y1, n: int, thr: float = DIVERSITY_IOU) -> list[int]:
    picks: list[int] = []
    for i in order:
        ok = True
        for j in picks:
            iw = min(x1[i], x1[j]) - max(x0[i], x0[j])
            ih = min(y1[i], y1[j]) - max(y0[i], y0[j])
            inter = max(iw, 0.0) * max(ih, 0.0)
            union = (x1[i] - x0[i]) * (y1[i] - y0[i]) + (x1[j] - x0[j]) * (y1[j] - y0[j]) - inter
            if union > 0 and inter / union >= thr:
                ok = False
                break
        if ok:
            picks.append(int(i))
            if len(picks) >= n:
                break
    return picks


def _refine(scene: Scene, k: float, x0: float, y0: float, w: float, wstep: float, infeasible: bool):
    """Local search around one candidate: +-1/2 width step and +-1/2 position step (3x3x3)."""
    dw = np.array([-0.5, 0.0, 0.5]) * wstep
    dp = np.array([-0.5, 0.0, 0.5]) * max(wstep, 0.01)
    W = np.clip(w + dw, 1e-3, None)
    X = x0 + dp
    Y = y0 + dp
    WW, XX, YY = np.meshgrid(W, X, Y, indexing="ij")
    WW, XX, YY = WW.ravel(), XX.ravel(), YY.ravel()
    HH = WW * k
    inside = (XX >= 0) & (YY >= 0) & (XX + WW <= 1 + 1e-12) & (YY + HH <= 1 + 1e-12)
    XX, YY, WW, HH = XX[inside], YY[inside], WW[inside], HH[inside]
    x1, y1 = XX + WW, YY + HH
    t = scene.terms(XX, YY, x1, y1)
    sc = scene.total(t)
    if infeasible:
        sc = scene.covered(XX, YY, x1, y1) * 1000.0 + sc
    else:
        sc = np.where(scene.feasible_mask(XX, YY, x1, y1), sc, -np.inf)
    i = int(np.argmax(sc))
    return float(XX[i]), float(YY[i]), float(WW[i]), float(sc[i])


def suggest_ratio(
    scene: Scene, ratio: Ratio, n: int = 3, min_long_px: int = MIN_LONG_EDGE_PX
) -> SuggestResult:
    """Top `n` diverse crops at one fixed or `original` ratio (never `free`: see `suggest`)."""
    fw, fh = scene.fw, scene.fh
    ab = ratio.resolve(fw, fh)
    assert ab is not None
    a, b = ab
    X0, Y0, W, infeasible, k, step = _candidates(scene, a, b, min_long_px)
    H = W * k
    X1, Y1 = X0 + W, Y0 + H
    t = scene.terms(X0, Y0, X1, Y1)
    sc = scene.total(t)
    if infeasible:
        obj = scene.covered(X0, Y0, X1, Y1) * 1000.0 + sc
    else:
        obj = np.where(scene.feasible_mask(X0, Y0, X1, Y1), sc, -np.inf)
    keep = min(len(obj), 600)
    top = np.argpartition(-obj, keep - 1)[:keep] if keep < len(obj) else np.arange(len(obj))
    order = top[np.argsort(-obj[top], kind="stable")]
    order = order[np.isfinite(obj[order])]
    picks = _diverse(order, X0, Y0, X1, Y1, n)
    ranked: list[tuple[float, Candidate]] = []
    for i in picks:
        x0, y0, w = float(X0[i]), float(Y0[i]), float(W[i])
        x0, y0, w, o = _refine(scene, k, x0, y0, w, step, infeasible)
        ranked.append((o, _finish(scene, ratio, x0, y0, w, k, infeasible)))
    # refinement can only improve the objective; order by it (covered area first when infeasible)
    ranked.sort(key=lambda t: -t[0])
    out = [c for _, c in ranked]
    return SuggestResult(out, infeasible, (a, b), ratio.text())


def _finish(
    scene: Scene, ratio: Ratio, x0: float, y0: float, w: float, k: float, infeasible: bool
) -> Candidate:
    fw, fh = scene.fw, scene.fh
    rect = Rect(x0, y0, min(1.0, x0 + w), min(1.0, y0 + w * k))
    snapped, px = snap_rect(rect, ratio, fw, fh)
    t = scene.terms(
        np.array([snapped.x0]),
        np.array([snapped.y0]),
        np.array([snapped.x1]),
        np.array([snapped.y1]),
    )
    sc = float(scene.total(t)[0])
    terms = {name: float(v[0]) for name, v in t.items()}
    warns: list[str] = []
    if infeasible:
        warns.append("subjects_do_not_fit")
    if scene.halo_cut(snapped.x0, snapped.y0, snapped.x1, snapped.y1):
        warns.append("subject_may_be_cut")
    if scene.strangers and terms["strangers"] < 1.0:
        warns.append("unselected_person_cut")
    if scene.persons and terms["cut"] < 1.0:
        warns.append("limb_cut")
    if scene.horizon is not None and abs(scene.horizon[1]) >= 1.0:
        warns.append(f"horizon_tilted_{abs(scene.horizon[1]):.1f}deg")
    return Candidate(snapped, sc, terms, warns, px)


def suggest(
    scene: Scene, ratio: Ratio, n: int = 3, min_long_px: int = MIN_LONG_EDGE_PX
) -> SuggestResult:
    """Crop suggestions for any ratio; `free` merges the best crops of the preset ratios."""
    if ratio.kind != "free":
        return suggest_ratio(scene, ratio, n, min_long_px)
    from shotsieve.pipeline.crop_geom import parse_ratio

    pool: list[Candidate] = []
    for p in FREE_PRESETS:
        pool += suggest_ratio(scene, parse_ratio(p), 1, min_long_px).candidates
    feas = [c for c in pool if "subjects_do_not_fit" not in c.warnings]
    use = feas or pool
    use.sort(key=lambda c: -c.score)
    out: list[Candidate] = []
    for c in use:
        if all(c.rect.iou(o.rect) < DIVERSITY_IOU for o in out):
            out.append(c)
        if len(out) >= n:
            break
    return SuggestResult(out, not feas, None, "free")
