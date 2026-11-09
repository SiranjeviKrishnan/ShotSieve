"""Training data for the personal model: Lightroom catalogs, XMP folders and UI overrides.

Every row has `weight = base_weight / propensity`. Plain catalog / XMP rows have propensity 1;
override rows carry the probability with which the image was shown (audit rows carry their
sampling probability), which corrects the selection bias of what the user got to see.
Weak negatives ("seen but not selected") have base weight 0.5.
"""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from shotsieve.config import Settings
from shotsieve.features.vector import FEATURE_NAMES, stack
from shotsieve.io.formats import scan_folder, split_virtual
from shotsieve.io.lrcat import CatalogEntry, read_catalog
from shotsieve.io.xmp import read_sidecar, user_values
from shotsieve.log import get_logger
from shotsieve.pipeline.evaluate import norm_path

log = get_logger("dataset")

WEAK_NEGATIVE_WEIGHT = 0.5
MIN_PROPENSITY = 0.02  # caps IPW weights at 50x


@dataclass
class Label:
    path: str
    keep: bool
    base_weight: float = 1.0
    propensity: float = 1.0
    source: str = ""

    @property
    def weight(self) -> float:
        return self.base_weight / max(self.propensity, MIN_PROPENSITY)


@dataclass
class TrainingSet:
    X: np.ndarray  # (N, 25)
    y: np.ndarray  # (N,) 1 = keep
    w: np.ndarray  # (N,) IPW weights
    embeddings: np.ndarray  # (N, 384)
    paths: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    n_missing: int = 0  # labelled files that do not exist / could not be featurised
    n_video: int = 0  # `clip.mp4#t=..` frame labels: excluded on purpose (no EXIF/AF, other blur)

    def __len__(self) -> int:
        return len(self.y)


# ---- label sources ----------------------------------------------------------------------


def _label_verdict(label: str | None, label_map: dict[str, str] | None) -> str | None:
    if not label or not label_map:
        return None
    m = {k.casefold(): v for k, v in label_map.items()}
    d = m.get(label.casefold())
    return d if d in ("keep", "reject") else None


def labels_from_catalog(
    entries: Iterable[CatalogEntry],
    min_rating: int = 1,
    label_map: dict[str, str] | None = None,
) -> list[Label]:
    """Lightroom rules. keep: pick == 1 or rating >= min_rating (and not rejected).
    reject: pick == -1. With a `label_map` (colour-label name -> keep / reject, e.g.
    `Settings.lr_label_map`) a mapped label counts after pick and rating. Unpicked, unrated images
    in a folder where the user picked or rated something are weak negatives ("seen but not
    selected", weight 0.5)."""
    entries = list(entries)
    active: set[str] = set()
    for e in entries:
        if e.pick != 0 or (e.rating or 0) > 0 or _label_verdict(e.label, label_map):
            active.add(os.path.dirname(e.path))
    out: list[Label] = []
    for e in entries:
        rating = e.rating or 0
        verdict = _label_verdict(e.label, label_map)
        if e.pick == -1:
            out.append(Label(e.path, False, 1.0, source="lrcat"))
        elif e.pick == 1 or rating >= min_rating:
            out.append(Label(e.path, True, 1.0, source="lrcat"))
        elif verdict is not None:
            out.append(Label(e.path, verdict == "keep", 1.0, source="lrcat"))
        elif rating == 0 and e.pick == 0 and os.path.dirname(e.path) in active:
            out.append(Label(e.path, False, WEAK_NEGATIVE_WEIGHT, source="lrcat"))
    return out


def _shoot_root(path: str, cache: dict[str, Path | None]) -> Path | None:
    """The culled shoot folder (the one holding `.shotsieve/cull.db`) above `path`, if any."""
    from shotsieve.pipeline.cache import cache_path

    d = os.path.dirname(path)
    if d in cache:
        return cache[d]
    cur, found = Path(d), None
    for cand in (cur, *cur.parents):
        if cache_path(cand).exists():
            found = cand
            break
    cache[d] = found
    return found


def drop_own_output(entries: Iterable[CatalogEntry]) -> list[CatalogEntry]:
    """Remove what ShotSieve itself wrote from catalog entries of culled shoots.

    A catalog rating / label equal to `lr_written` (recorded when XMP was written, or recovered
    from the sidecars' `shotsieve:WrittenRating/Label`) is ShotSieve's opinion that Lightroom
    imported, not the photographer's: it is cleared, and an entry with nothing else left (no flag,
    no differing rating or label) is dropped. Without this, `train --lrcat` after an XMP
    round-trip learned its own 3-star / Green output as user keeps."""
    from shotsieve.pipeline.cache import Cache
    from shotsieve.pipeline.lrsync import ensure_written

    roots: dict[str, Path | None] = {}
    written: dict[Path, dict[str, dict]] = {}
    out: list[CatalogEntry] = []
    n_dropped = 0
    for e in entries:
        root = _shoot_root(e.path, roots)
        if root is None:
            out.append(e)
            continue
        if root not in written:
            with Cache.for_folder(root) as c:
                ensure_written(c)
                written[root] = {norm_path(p): row for p, row in c.written().items()}
        row = written[root].get(norm_path(e.path))
        if row is None:
            out.append(e)
            continue
        rating_eq = row["rating"] is not None and (e.rating or 0) == row["rating"]
        label_eq = (
            row["label"] is not None and bool(e.label)
            and str(e.label).casefold() == str(row["label"]).casefold()
        )  # fmt: skip
        if not (rating_eq or label_eq):
            out.append(e)
            continue
        new = replace(
            e, rating=None if rating_eq else e.rating, label=None if label_eq else e.label
        )
        if new.pick == 0 and not (new.rating or 0) and not new.label:
            n_dropped += 1
            continue
        out.append(new)
    if n_dropped:
        log.info("%d catalog entries equal ShotSieve's own output; ignored as labels", n_dropped)
    return out


def labels_from_xmp_folder(
    folder: str | Path, min_rating: int = 1, settings: Settings | None = None
) -> list[Label]:
    """Sidecars of the user's tool: xmp:Rating (-1 = rejected) and xmp:Label.

    Values ShotSieve wrote itself (tracked in `shotsieve:WrittenRating/Label`) are removed first,
    so training never learns from its own output. Rating >= min_rating or a Green label -> keep;
    rating -1 or a Red label -> reject; a foreign sidecar that selects nothing, in a folder where
    something is selected, is a weak negative. A sidecar holding only ShotSieve's output says
    nothing about the user and is skipped.
    """
    settings = settings or Settings()
    found = []
    n_own = 0
    for e in scan_folder(folder):
        data = read_sidecar(e.master)
        if data is None:
            continue
        rating, label = user_values(data, settings)
        if rating is None and not label and data.decision is not None:
            n_own += 1
            continue
        found.append((str(e.master), rating, (label or "").lower()))
    if n_own:
        log.warning(
            "%s: %d sidecar(s) contain only ShotSieve's own decisions; ignored as labels",
            folder, n_own,
        )  # fmt: skip
    verdicts: list[tuple[str, bool | None]] = []
    for path, rating, label in found:
        if (rating is not None and rating < 0) or label == "red":
            verdicts.append((path, False))
        elif (rating or 0) >= min_rating or label == "green":
            verdicts.append((path, True))
        else:
            verdicts.append((path, None))
    any_selected = any(v is True for _, v in verdicts)
    out = []
    for path, v in verdicts:
        if v is None:
            if any_selected:
                out.append(Label(path, False, WEAK_NEGATIVE_WEIGHT, source="xmp"))
        else:
            out.append(Label(path, v, 1.0, source="xmp"))
    return out


def labels_from_overrides(folder: str | Path) -> list[Label]:
    """UI overrides stored in <folder>/.shotsieve/cull.db (keep / reject only)."""
    from shotsieve.pipeline.cache import Cache, cache_path

    if not cache_path(folder).exists():
        return []
    with Cache.for_folder(folder) as cache:
        ovr = cache.overrides()
    return [
        Label(path, o["decision"] == "keep", 1.0, float(o["propensity"] or 1.0), "override")
        for path, o in ovr.items()
        if o["decision"] in ("keep", "reject")
    ]


# ---- features ---------------------------------------------------------------------------


def _stored_features(folder: Path, wanted: dict[str, Label]):
    """Features already persisted by a previous cull of `folder`, if they cover every label."""
    from shotsieve.features.vector import FeatureVector
    from shotsieve.pipeline.cache import Cache, cache_path

    if not cache_path(folder).exists():
        return None
    with Cache.for_folder(folder) as cache:
        rows = cache.load_decisions()
    got = {norm_path(p): (p, r) for p, r in rows.items()}
    if not all(k in got for k in wanted):
        return None
    return [
        (p, FeatureVector.from_dict(r["features"]), r["reasons"])
        for k, (p, r) in got.items()
        if k in wanted
    ]


def featurise(
    labels: list[Label],
    settings: Settings,
    progress: Callable[[str, int, int], None] | None = None,
    run=None,
) -> TrainingSet:
    """Feature matrix for labelled images by running the culling stages (cached per folder).

    Labels are grouped by parent directory (a directory is one shoot, so burst / duplicate-group
    features are the same ones the runner will use later). Missing files are counted and dropped.
    When one file has several labels the later one wins (so overrides beat catalogs).
    """
    from shotsieve.pipeline.cache import Cache
    from shotsieve.pipeline.runner import run_cull

    run = run or run_cull
    latest: dict[str, Label] = {}
    for lab in labels:
        latest[norm_path(lab.path)] = lab
    by_dir: dict[Path, dict[str, Label]] = defaultdict(dict)
    n_missing = 0
    n_video = 0
    for key, lab in latest.items():
        if split_virtual(lab.path) is not None:  # even when the base clip is gone
            n_video += 1
            continue
        p = Path(lab.path)
        if not p.exists():
            n_missing += 1
            continue
        by_dir[p.parent][key] = lab
    # model-free settings: never let the existing personal model colour training features
    quiet = Settings.from_dict({**settings.to_dict(), "personal_model_path": os.devnull + ".none"})
    rows, ys, ws, embs, paths, srcs = [], [], [], [], [], []
    done_dirs = 0
    # per-image progress (T19): ("images", done, total) over the folders that need the stages
    img_total = img_done = 0
    if progress:
        img_total = sum(
            len(scan_folder(f)) for f, w in by_dir.items() if _stored_features(f, w) is None
        )
        progress("images", 0, img_total)

    def _sub_progress(base: int):
        def cb(stage: str, done: int, total: int) -> None:
            if stage == "embed" and progress:
                progress("images", min(base + done, img_total), img_total)

        return cb

    for folder, wanted in sorted(by_dir.items()):
        stored = _stored_features(folder, wanted)
        if stored is None:  # not (fully) analysed yet: run the stages (cached per folder)
            res = run(folder, quiet, _sub_progress(img_done) if progress else None, False)
            img_done += len(res.images)
            stored = [(str(r.master_path), r.features, r.reasons) for r in res.images]
        with Cache.for_folder(folder) as cache:
            scene = {norm_path(k): v for k, v in cache.scene_embeddings().items()}
        for path, fv, rs in stored:
            lab = wanted.get(norm_path(path))
            if lab is None:
                continue
            e = scene.get(norm_path(path))
            if e is None or "unreadable" in rs:
                n_missing += 1
                continue
            rows.append(fv)
            ys.append(int(lab.keep))
            ws.append(lab.weight)
            embs.append(e)
            paths.append(path)
            srcs.append(lab.source)
        done_dirs += 1
        if progress:
            progress("features", done_dirs, len(by_dir))
    return TrainingSet(
        stack(rows) if rows else np.zeros((0, len(FEATURE_NAMES)), np.float32),
        np.array(ys, dtype=int),
        np.array(ws, dtype=np.float64),
        np.stack(embs) if embs else np.zeros((0, 384), np.float32),
        paths,
        srcs,
        n_missing,
        n_video,
    )


def build_training_set(
    lrcat: str | Path | None = None,
    xmp_folders: Iterable[str | Path] = (),
    override_folders: Iterable[str | Path] = (),
    settings: Settings | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> TrainingSet:
    """Combine any mix of catalog, XMP and override sources into one weighted training set."""
    settings = settings or Settings()
    labels: list[Label] = []
    if lrcat:
        labels += labels_from_catalog(
            drop_own_output(read_catalog(lrcat)), settings.min_rating, settings.lr_label_map
        )
    for f in xmp_folders:
        labels += labels_from_xmp_folder(f, settings.min_rating, settings)
    for f in override_folders:
        labels += labels_from_overrides(f)
    log.info("%d labels collected", len(labels))
    return featurise(labels, settings, progress)
