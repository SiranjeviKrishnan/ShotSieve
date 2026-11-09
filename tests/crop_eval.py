"""Evaluation helpers for the T22 quality tests (docs/11_CROP.md section 11).

Labels come from an INDEPENDENT labeller (`tests/fixtures/crop/crop_labels.json`, pixel boxes of the
file as displayed): `{image: {w, h, subjects: [{kind, box_px, must}], horizon_y_px, reference_px:
{"4:5": [...], "16:9": [...]}}}`. The simulated user "clicks" the centre of every must-keep subject
exactly as the UI does (smallest detected box containing the point; a face linked to a person
selects the person; empty space = point prompt).

Harness v2 (Review #12 m6, FROZEN by sha256 in TASKS.md before any new label was written):
(i) the click lands inside the intended subject but outside every SMALLER labelled subject nested in
it (the body, not the face, of a bust); (ii) misses are counted as SILENT (top-1 leaves a must-keep
< 98 % inside and carries no warning) or FLAGGED (`subjects_do_not_fit` / `subject_may_be_cut` /
`infeasible`); containment scores only the silent ones and `no_cut_without_warning` must be 1.0
over EVERY pair; the baseline comparison is top-1 vs centre crop (IoU and raw containment)."""

from __future__ import annotations

import json
from pathlib import Path

from shotsieve.config import Settings
from shotsieve.pipeline import crop as pcrop
from shotsieve.pipeline import crop_geom as g
from shotsieve.pipeline.cache import Cache
from shotsieve.pipeline.crop_geom import Rect

ROOT = Path(__file__).resolve().parents[1]
CROP = ROOT / "tests" / "fixtures" / "crop"
RATIOS = ["1:1", "4:5", "5:4", "2:3", "3:2", "16:9", "9:16", "7:5"]


def load_labels() -> dict:
    return json.loads((CROP / "crop_labels.json").read_text())


def split_of(name: str, manifest: dict) -> str:
    return manifest[name]["split"]


def norm_box(b, w, h):
    return (b[0] / w, b[1] / h, b[2] / w, b[3] / h)


def inside_frac(rect: Rect, box) -> float:
    iw = min(rect.x1, box[2]) - max(rect.x0, box[0])
    ih = min(rect.y1, box[3]) - max(rect.y0, box[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    return iw * ih / ((box[2] - box[0]) * (box[3] - box[1]))


def iou(a, b) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    i = iw * ih
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


FLAG_WARNINGS = ("subjects_do_not_fit", "subject_may_be_cut")


def click_point(box, others) -> tuple[float, float]:
    """A point inside `box` (normalised) and outside every smaller labelled box nested in it
    (>= 50 % of that box lies inside `box`); the centre when nothing is nested; else the free
    grid point nearest the centre (the centre if none is free)."""
    def nested(o) -> bool:
        if o == box:
            return False
        a_o = (o[2] - o[0]) * (o[3] - o[1])
        a_b = (box[2] - box[0]) * (box[3] - box[1])
        if a_o >= a_b:
            return False
        iw = min(box[2], o[2]) - max(box[0], o[0])
        ih = min(box[3], o[3]) - max(box[1], o[1])
        return iw > 0 and ih > 0 and iw * ih >= 0.5 * a_o

    inner = [o for o in others if nested(o)]
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    if not inner:
        return cx, cy

    def free(x, y) -> bool:
        return not any(o[0] <= x <= o[2] and o[1] <= y <= o[3] for o in inner)

    if free(cx, cy):
        return cx, cy
    best, bd = None, 9.0
    n = 40
    for i in range(1, n):
        for j in range(1, n):
            x = box[0] + (box[2] - box[0]) * i / n
            y = box[1] + (box[3] - box[1]) * j / n
            if free(x, y):
                d = (x - cx) ** 2 + (y - cy) ** 2
                if d < bd:
                    best, bd = (x, y), d
    return best or (cx, cy)


class Env:
    """One folder + cache holding the fixture images (detection runs as in the product)."""

    def __init__(self, folder: Path, names: list[str], settings: Settings | None = None) -> None:
        self.folder = folder
        self.settings = settings or Settings()
        self.cache = Cache.for_folder(folder)
        self.paths = {n: str(folder / n) for n in names}
        self.ids = self.cache.ids_for(list(self.paths.values()))
        self._sets: dict[str, pcrop.SubjectSet] = {}

    def subjects(self, name: str) -> pcrop.SubjectSet:
        if name not in self._sets:
            p = self.paths[name]
            self._sets[name] = pcrop.detect_subjects(
                self.folder, self.cache, self.ids[p], p, self.settings
            )
        return self._sets[name]

    def click(self, name: str, x: float, y: float):
        """What a click at (x, y) selects: (kind, Subject), like crop.js."""
        sset = self.subjects(name)
        best, area = None, 9.0
        for s in sset.subjects:
            b = s.box
            if b[0] <= x <= b[2] and b[1] <= y <= b[3]:
                a = (b[2] - b[0]) * (b[3] - b[1])
                if a < area:
                    best, area = s, a
        if best is None:
            p = self.paths[name]
            pt = pcrop.point_subject(self.folder, self.cache, self.ids[p], p, x, y, self.settings)
            return "point", pt
        by = sset.by_sid()
        if best.kind == "face" and best.linked:
            q = by[best.linked]
            sel = pcrop.Subject(**{**q.__dict__})
            sel.select_mode = "person"
            return "person", sel
        sel = pcrop.Subject(**{**best.__dict__})
        if sel.kind == "person":
            sel.select_mode = "person"
        return best.kind, sel

    def select(self, name: str, musts: list[tuple[float, float, float, float]], everything=()):
        sel, pts = [], []
        seen: set[str] = set()
        for b in musts:
            kind, s = self.click(name, *click_point(b, list(everything) or musts))
            if s.sid in seen:
                continue
            seen.add(s.sid)
            sel.append(s)
            if kind == "point":
                pts.append(s)
        return sel, pts

    def suggest(self, name: str, sel, pts, ratio: str, n: int = 3):
        sset = self.subjects(name)
        r = g.parse_ratio(ratio)  # exact orientation as given (no image-orientation default)
        return pcrop.run_suggest(sset, sel, r, n, self.settings, extra=pts)

    def close(self) -> None:
        self.cache.close()


def feasible(rects_norm, ratio: str, fw: int, fh: int) -> bool:
    """Can ONE crop at this ratio contain every box (tight, no margin)? Closed form."""
    r = g.parse_ratio(ratio)
    a, b = r.resolve(fw, fh)
    k = fw * b / (fh * a)
    wmax = min(1.0, 1.0 / k)
    if not rects_norm:
        return True
    x0 = min(b_[0] for b_ in rects_norm)
    y0 = min(b_[1] for b_ in rects_norm)
    x1 = max(b_[2] for b_ in rects_norm)
    y1 = max(b_[3] for b_ in rects_norm)
    return max(x1 - x0, (y1 - y0) / k) <= wmax + 1e-9


def centre_crop(ratio: str, fw: int, fh: int) -> Rect:
    r = g.parse_ratio(ratio)
    a, b = r.resolve(fw, fh)
    k = fw * b / (fh * a)
    w = min(1.0, 1.0 / k)
    h = w * k
    return Rect((1 - w) / 2, (1 - h) / 2, (1 + w) / 2, (1 + h) / 2)


# ---- the acceptance numbers (docs/11_CROP.md section 11, items 1, 3, 4, 5) ----------------------


def evaluate(env: Env, names: list[str], labels: dict) -> dict:
    """All crop-quality metrics over the images `names` (see the module docstring)."""
    from shotsieve.models import object_detect

    st = {"pairs": 0, "feasible": 0, "contain_ok": 0, "silent_miss": 0, "flagged_miss": 0,
          "centre_contain_ok": 0, "cut_pairs": 0, "cut_flagged": 0, "face_cut": 0,
          "infeasible": 0, "infeasible_flagged": 0, "alt_free_pairs": 0, "stranger_cut_top1": 0,
          "ref_cases": 0, "ref_best_ok": 0, "ref_iou_top1": [], "ref_iou_centre": [],
          "person_total": 0, "person_found": 0, "click_total": 0, "click_ok": 0,
          "point_total": 0, "point_ok": 0, "fails": []}  # fmt: skip
    for name in names:
        lab = labels[name]
        w, h = lab["w"], lab["h"]
        subs = [(s["kind"], norm_box(s["box_px"], w, h), bool(s["must"])) for s in lab["subjects"]]
        musts = [b for _, b, m in subs if m]
        must_faces = [b for k, b, m in subs if m and k == "face"]
        strangers = [b for k, b, m in subs if not m and k in ("face", "person")]
        sset = env.subjects(name)
        fw, fh = sset.fw, sset.fh
        sel, pts = (
            env.select(name, musts, [b for _, b, _m in subs]) if musts else ([], [])
        )
        for ratio in RATIOS:
            st["pairs"] += 1
            res = env.suggest(name, sel, pts, ratio)
            ok_feasible = feasible(musts, ratio, fw, fh)
            top = res.candidates[0].rect
            good = all(inside_frac(top, b) >= 0.98 for b in musts)
            flagged = bool(res.infeasible) or any(
                w in FLAG_WARNINGS for w in res.candidates[0].warnings
            )
            if not good:  # a cut of a must-keep, feasible or not: it needs a warning
                st["cut_pairs"] += 1
                st["cut_flagged"] += flagged
            if not ok_feasible:
                st["infeasible"] += 1
                st["infeasible_flagged"] += flagged
                continue
            st["feasible"] += 1
            st["contain_ok"] += good
            cc = centre_crop(ratio, fw, fh)
            st["centre_contain_ok"] += all(inside_frac(cc, b) >= 0.98 for b in musts)
            if not good:
                st["flagged_miss" if flagged else "silent_miss"] += 1
                st["fails"].append(
                    f"{name} {ratio}: must-keep outside the top-1 crop"
                    f" ({'flagged' if flagged else 'SILENT'})"
                )
            for c in res.candidates:
                if any(0.02 < inside_frac(c.rect, b) < 0.98 for b in must_faces):
                    st["face_cut"] += 1
                    st["fails"].append(f"{name} {ratio}: a face is cut")
                    break
            def cut(r, strangers=strangers):
                return any(0.05 < inside_frac(r, b) < 0.95 for b in strangers)

            if strangers and any(not cut(c.rect) for c in res.candidates):
                st["alt_free_pairs"] += 1
                st["stranger_cut_top1"] += cut(top)
        # reference crops (4:5 and 16:9 exactly as given, not image-oriented)
        for rt, ref in (lab.get("reference_px") or {}).items():
            rb = norm_box(ref, w, h)
            res = env.suggest(name, sel, pts, rt)
            ious = [iou(c.rect.as_tuple(), rb) for c in res.candidates]
            st["ref_cases"] += 1
            st["ref_best_ok"] += max(ious) >= 0.75
            st["ref_iou_top1"].append(ious[0])
            st["ref_iou_centre"].append(iou(centre_crop(rt, fw, fh).as_tuple(), rb))
        # detection / pointing
        for kind, b, _m in subs:
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            if kind == "person" and object_detect.available():
                st["person_total"] += 1
                dets = [s for s in sset.subjects if s.kind == "person"]
                st["person_found"] += any(iou(s.box, b) >= 0.5 for s in dets)
            inside = [
                s for s in sset.subjects
                if s.box[0] <= cx <= s.box[2] and s.box[1] <= cy <= s.box[3]
            ]
            if inside:
                st["click_total"] += 1
                _, s = env.click(name, *click_point(b, [x[1] for x in subs]))
                st["click_ok"] += s.kind != "point"
            if kind == "object" and _m and (b[2] - b[0]) * (b[3] - b[1]) >= 0.03:
                p = env.paths[name]
                px, py = click_point(b, [x[1] for x in subs])
                pt = pcrop.point_subject(env.folder, env.cache, env.ids[p], p, px, py, env.settings)
                st["point_total"] += 1
                st["point_ok"] += iou(pt.box, b) >= 0.5
    def rate(a: str, b: str) -> float:
        return st[a] / st[b] if st[b] else float("nan")

    st["containment_raw"] = rate("contain_ok", "feasible")
    st["containment"] = (
        1.0 - st["silent_miss"] / st["feasible"] if st["feasible"] else float("nan")
    )  # only unwarned misses count against it
    st["centre_containment"] = rate("centre_contain_ok", "feasible")
    st["contain_gain"] = st["containment_raw"] - st["centre_containment"]
    st["no_cut_without_warning"] = rate("cut_flagged", "cut_pairs")
    st["infeasible_flag_rate"] = rate("infeasible_flagged", "infeasible")
    st["stranger_pairs"] = st["alt_free_pairs"]
    st["stranger_cut_rate"] = rate("stranger_cut_top1", "alt_free_pairs")
    st["ref_best_rate"] = rate("ref_best_ok", "ref_cases")
    st["ref_gap"] = (
        sum(st["ref_iou_top1"]) / len(st["ref_iou_top1"])
        - sum(st["ref_iou_centre"]) / len(st["ref_iou_centre"])
        if st["ref_iou_top1"] else float("nan")
    )  # top-1 minus centre crop, mean IoU against the labelled reference crops  # fmt: skip
    st["person_recall"] = rate("person_found", "person_total")
    st["click_rate"] = rate("click_ok", "click_total")
    st["point_rate"] = rate("point_ok", "point_total")
    return st


ACCEPT = {
    "containment": (">=", 0.95), "no_cut_without_warning": ("==", 1.0), "face_cut": ("==", 0),
    "stranger_cut_rate": ("<=", 0.10), "ref_best_rate": (">=", 0.70), "ref_gap": (">=", -0.01),
    "contain_gain": (">=", 0.05), "person_recall": (">=", 0.85), "click_rate": (">=", 1.0),
    "point_rate": (">=", 0.70),
}  # fmt: skip
# `stranger_cut_rate` only counts with >= 20 qualifying pairs (`stranger_pairs`); `ref_gap` and
# `contain_gain` are the Review #12 replacement for the ill-posed "+0.10 IoU over the centre crop".


def verdicts(st: dict) -> dict[str, bool]:
    import math

    out = {}
    for k, (op, v) in ACCEPT.items():
        x = st[k]
        if isinstance(x, float) and math.isnan(x):
            out[k] = True  # nothing to measure in this split
        else:
            out[k] = x >= v if op == ">=" else x <= v if op == "<=" else x == v
    return out
