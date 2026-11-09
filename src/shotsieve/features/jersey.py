"""Jersey numbers: torso/box OCR with digit filtering, burst voting, face association, roster tags.

Everything here is pure (arrays and dicts in, rows out) so it is testable without the indexer.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from shotsieve.io.roster import normalize_number
from shotsieve.models.ocr import TextBox, TextDetector, TextRecognizer, crop_box

MAX_BOXES = 12
MIN_BOX_H_FRAC = 0.025  # box height / image height
MAX_ASPECT = 4.0  # long/short side: longer boxes are scoreboards, ads, banners
MIN_CHAR_PROB = 0.80
MIN_MEAN_PROB = 0.90
BURST_FRAC = 0.30
BURST_MIN_FRAMES = 2
PERSON_MIN_READS = 2
PERSON_MIN_SHARE = 0.60
_DIGITS = re.compile(r"^\d{1,3}$")

Box = tuple[float, float, float, float]  # x, y, w, h


@dataclass
class Read:
    number: str
    conf: float
    box: Box  # pixels in the image given to `read_numbers`
    face_idx: int | None = None
    person_id: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def accept_text(text: str, probs: list[float]) -> str | None:
    """The normalised number if `text` is 1-3 digits read confidently, else None."""
    if not _DIGITS.match(text) or not probs or len(probs) != len(text):
        return None
    if min(probs) < MIN_CHAR_PROB or float(np.mean(probs)) < MIN_MEAN_PROB:
        return None
    return normalize_number(text) or None


def _inside(cx: float, cy: float, boxes: list[Box]) -> bool:
    return any(x <= cx <= x + w and y <= cy <= y + h for x, y, w, h in boxes)


def _iou(a: Box, b: Box) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return inter / max(a[2] * a[3] + b[2] * b[3] - inter, 1e-9)


def _read_boxes(
    img: np.ndarray,
    boxes: list[TextBox],
    rec: TextRecognizer,
    faces: list[Box],
    ref_h: float,
) -> list[Read]:
    crops: list[np.ndarray] = []
    owner: list[int] = []
    kept: list[TextBox] = []
    for b in sorted(boxes, key=lambda t: t.area, reverse=True)[:MAX_BOXES]:
        x, y, w, h = b.bounds
        sw, sh = b.sides
        if max(sw, sh) / max(min(sw, sh), 1e-6) > MAX_ASPECT:
            continue
        if h < MIN_BOX_H_FRAC * ref_h:
            continue
        if _inside(x + w / 2, y + h / 2, faces):
            continue
        crop = crop_box(img, b)
        kept.append(b)
        crops.append(crop)
        owner.append(len(kept) - 1)
        if sh >= 1.5 * sw:  # tall box: upright single digit, or vertical text
            crops.append(cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE))
            owner.append(len(kept) - 1)
    best: dict[int, tuple[str, float]] = {}
    for k, (text, probs) in zip(owner, rec.recognize(crops), strict=True):
        num = accept_text(text, probs)
        if num and (k not in best):  # the upright variant comes first and wins when valid
            best[k] = (num, float(np.mean(probs)))
    return [Read(n, c, kept[k].bounds) for k, (n, c) in best.items()]


def _dedupe(reads: list[Read]) -> list[Read]:
    out: list[Read] = []
    for r in sorted(reads, key=lambda r: -r.conf):
        if all(_iou(r.box, o.box) < 0.5 for o in out):
            out.append(r)
    return out


def torso_region(face: Box, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """Crop (x0, y0, x1, y1) of the torso below a face: 3 face-widths wide, 1.1-4.5 faces down."""
    fx, fy, fw, fh = face
    h, w = shape
    x0, x1 = int(max(0, fx - fw)), int(min(w, fx + 2 * fw))
    y0, y1 = int(max(0, fy + 1.1 * fh)), int(min(h, fy + 4.5 * fh))
    return x0, y0, x1, y1


def read_numbers(
    img_rgb: np.ndarray,
    det: TextDetector,
    rec: TextRecognizer,
    faces: list[Box] | None = None,
    torso_pass: bool = True,
) -> list[Read]:
    """Jersey-number reads for one image. Pass 1 is the whole frame; when it finds nothing and
    faces are known, pass 2 re-runs detection on up-scaled torso crops below the largest faces
    (small, distant players)."""
    faces = faces or []
    H = img_rgb.shape[0]
    reads = _read_boxes(img_rgb, det.detect(img_rgb), rec, faces, H)
    if not reads and torso_pass and faces:
        for face in sorted(faces, key=lambda f: -f[2] * f[3])[:2]:
            x0, y0, x1, y1 = torso_region(face, img_rgb.shape[:2])
            if x1 - x0 < 24 or y1 - y0 < 24:
                continue
            crop = img_rgb[y0:y1, x0:x1]
            up = max(1.0, 480.0 / max(crop.shape[:2]))
            if up > 1.0:
                crop = cv2.resize(crop, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
            for r in _read_boxes(crop, det.detect(crop), rec, [], H * up):
                bx, by, bw, bh = r.box
                r.box = (x0 + bx / up, y0 + by / up, bw / up, bh / up)
                reads.append(r)
    return _dedupe(reads)


def associate_face(box: Box, faces: list[Box]) -> int | None:
    """Index of the face above a jersey box (a face whose bottom is just above the box top and
    whose centre is within the box's horizontal span), nearest first; None if there is none."""
    bx, by, bw, bh = box
    bcx = bx + bw / 2
    best, best_d = None, 1e18
    for i, (fx, fy, fw, fh) in enumerate(faces):
        fcx = fx + fw / 2
        gap = by - (fy + fh)  # < 0: the box overlaps the face's vertical extent
        if gap < -0.5 * fh or gap > 3.0 * fh:
            continue
        if abs(fcx - bcx) > max(bw, 1.5 * fw) * 0.75 + 0.5 * fw:
            continue
        d = abs(fcx - bcx) + max(gap, 0.0)
        if d < best_d:
            best, best_d = i, d
    return best


def burst_vote(
    reads: dict[str, dict[str, float]], bursts: dict[str, int]
) -> list[dict[str, Any]]:
    """Temporal vote. `reads`: path -> {number: conf} for every OCR'd frame, `bursts`: path ->
    burst id. A number seen in >= 30 % of a burst's frames (and >= 2 frames) is propagated to
    the burst's frames that lack it. Returns rows `{path, number, conf}` for those frames."""
    by_burst: dict[int, list[str]] = {}
    for p in reads:
        if p in bursts:
            by_burst.setdefault(bursts[p], []).append(p)
    out: list[dict[str, Any]] = []
    for frames in by_burst.values():
        n = len(frames)
        if n < 2:
            continue
        seen: Counter[str] = Counter()
        conf: dict[str, list[float]] = {}
        for p in frames:
            for num, c in reads[p].items():
                seen[num] += 1
                conf.setdefault(num, []).append(c)
        for num, k in seen.items():
            if k >= BURST_MIN_FRAMES and k / n >= BURST_FRAC:
                for p in frames:
                    if num not in reads[p]:
                        out.append({"path": p, "number": num, "conf": float(np.mean(conf[num]))})
    return out


def dominant_numbers(rows: list[dict[str, Any]]) -> dict[int, str]:
    """person_id -> the number most of that person's face-linked reads agree on."""
    votes: dict[int, Counter[str]] = {}
    for r in rows:
        if r.get("person_id") is not None and r["source"] == "ocr":
            votes.setdefault(r["person_id"], Counter())[r["number"]] += 1
    out = {}
    for pid, c in votes.items():
        num, k = c.most_common(1)[0]
        if k >= PERSON_MIN_READS and k / sum(c.values()) >= PERSON_MIN_SHARE:
            out[pid] = num
    return out


# number -> [(name, team), ...]: a number repeats across teams (home #23 and away #23). The legacy
# single-tuple form {number: (name, team)} is still accepted.
Roster = dict[str, "list[tuple[str, str]] | tuple[str, str]"]


def _cands(roster: Roster, number: str, team: str = "") -> list[tuple[str, str]]:
    """Distinct (name, team) entries for `number` that can apply in a folder shot for `team`
    (entries without a team apply to every team)."""
    hit = roster.get(number) or []
    if isinstance(hit, tuple):
        hit = [hit]
    out: list[tuple[str, str]] = []
    for name, t in hit:
        if team and t and t.strip().lower() != team.strip().lower():
            continue
        if (name, t) not in out:
            out.append((name, t))
    return out


def resolve(roster: Roster, number: str, team: str = "") -> tuple[str, str, str]:
    """(status, name, team): matched | unmatched | ambiguous | '' (no roster). A number shared by
    different people on several teams is `ambiguous` unless the folder `team` settles it; no name
    is ever guessed."""
    if not roster:
        return "", "", ""
    c = _cands(roster, number, team)
    if not c:
        return "unmatched", "", ""
    if len({n.strip().lower() for n, _ in c}) > 1:
        return "ambiguous", "", ""
    return "matched", c[0][0], c[0][1]


def annotate(rows: list[dict[str, Any]], roster: Roster, team: str = "") -> list[dict[str, Any]]:
    """Add `name`, `team`, `status` (matched | unmatched | ambiguous | '' when no roster)."""
    out = []
    for r in rows:
        status, name, t = resolve(roster, r["number"], team)
        out.append({**r, "name": name, "team": t, "status": status})
    return out


def xmp_tags(rows: list[dict[str, Any]], roster: Roster, team: str = "") -> dict[str, dict]:
    """path -> {'keywords': ['#23', 'Jane Doe'], 'persons': ['Jane Doe']}. With a roster loaded,
    numbers not on it are kept in the table but not written; an ambiguous number (shared by two
    teams) is written as `#N` only, with no name."""
    out: dict[str, dict[str, list[str]]] = {}
    for r in annotate(rows, roster, team):
        if r["status"] == "unmatched":
            continue
        t = out.setdefault(r["path"], {"keywords": [], "persons": []})
        for kw in (f"#{r['number']}", r["name"]):
            if kw and kw not in t["keywords"]:
                t["keywords"].append(kw)
        if r["name"] and r["name"] not in t["persons"]:
            t["persons"].append(r["name"])
    return out
