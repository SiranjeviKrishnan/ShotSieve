"""Reference-frame colour match (T16): copy a photographer's edit of one frame to look-alikes.

The reference's `crs:` develop settings (sidecar, embedded XMP or Lightroom catalog) are synced to
each target with exposure and white balance *re-derived* for the target, from 256-px previews:

* gate: DINOv2 cosine >= `SCENE_COS` ("same scene"), else skipped;
* exposure: dEV = log2(L_R / L_t) on the linear-light median of luma p5-p95 (|dEV| <= 2);
* white balance (RAW targets only): scene mired s = as-shot mired + c, c = residual cast of the
  preview (median log-ratio of unclipped pixels, see `preview_measure`).
  mired_t = s_t + (mired_R - s_R).

Nothing here runs inside `cull`. Writes go through `crs.write_develop` (never-overwrite, sidecar for
RAW, embedded-XMP copies for JPEG/DNG); originals are never modified.
"""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from shotsieve.config import Settings
from shotsieve.features.raw_wb import as_shot
from shotsieve.io import crs, lrcat
from shotsieve.io.exif import read_meta
from shotsieve.io.formats import RAW_EXTS, preview_source_for, split_virtual
from shotsieve.io.preview import load_preview
from shotsieve.log import get_logger
from shotsieve.personal.edit_dataset import _NO_CARRY, TINT_PER_DUV, featurise
from shotsieve.personal.edit_profile import TONE_KEYS
from shotsieve.pipeline.cache import Cache
from shotsieve.pipeline.edit import LIGHTROOM_EMBED_NOTE, _mode_for, check_copy_root

log = get_logger("match")
Progress = Callable[[str, int, int], None]

SCENE_COS = 0.60  # DINOv2 cosine: "same scene"
CLIP_SKIP = 0.25  # clip_hi above this: exposure is not matched
MAX_DEV = 2.0  # |delta EV| above this: skipped
PREVIEW_EDGE = 256
# ln(R/B) -> mired and ln(G/sqrt(RB)) -> delta-uv, fitted once on the synthetic perturbation set
# (scripts/make_edit_styles.py: R gain exp(+0.0025 m), B gain exp(-0.0025 m), G gain exp(-6 duv))
# and frozen here. Real cameras differ somewhat; matches only use differences.
MIRED_PER_LN_RB = 200.0
GREEN_PER_DUV = 6.0
_LIN = (np.arange(256, dtype=np.float64) / 255.0) ** 2.2

REFUSE_OURS = "the reference's settings were written by ShotSieve, not edited by you"
REFUSE_UNEDITED = "the reference has no develop settings: edit the reference in Lightroom first"


@dataclass
class Measure:
    """What `match` needs to know about one frame (preview statistics, embedding, as-shot WB)."""

    path: str
    emb: np.ndarray | None
    lum: float  # linear-light median luma over p5..p95
    clip_hi: float
    cast_mired: float  # residual cast of the preview, mired (warm positive)
    cast_duv: float
    is_raw: bool
    as_shot: tuple[float, float] | None = None  # (mired, duv)
    body: str = ""  # camera Make+Model ("" = unknown)

    @property
    def scene_wb(self) -> tuple[float, float] | None:
        if self.as_shot is None:
            return None
        return self.as_shot[0] + self.cast_mired, self.as_shot[1] + self.cast_duv


@dataclass
class MatchResult:
    path: str
    values: dict[str, float] = field(default_factory=dict)  # crs values re-derived for the target
    carried: dict[str, float | str] = field(default_factory=dict)  # copied verbatim
    skipped_reason: str | None = None
    cos: float | None = None
    delta_ev: float | None = None
    wb: bool = False
    status: str = ""  # written | report_only | skipped_* | ... (set when writing)
    written_mode: str = ""
    notes: list[str] = field(default_factory=list)
    process_version: float | str | None = None  # the reference's PV (never written below 11.0)


@dataclass
class Reference:
    path: str
    settings: dict[str, float | str]
    measure: Measure
    source: str  # sidecar | embedded | catalog

    @property
    def temp_mired(self) -> float | None:
        t = self.settings.get("Temperature")
        return 1e6 / t if isinstance(t, float | int) and t > 0 else None

    @property
    def tint(self) -> float | None:
        t = self.settings.get("Tint")
        return float(t) if isinstance(t, float | int) else None


# ---- measurement -----------------------------------------------------------------------------


def preview_measure(img: np.ndarray) -> tuple[float, float, float, float]:
    """(linear-light luma median over p5-p95, clip_hi, cast mired, cast duv) of an RGB preview."""
    h, w = img.shape[:2]
    s = PREVIEW_EDGE / max(h, w)
    if s < 1:
        img = cv2.resize(
            img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA
        )
    img = np.ascontiguousarray(img[:, :, :3])
    luma = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lo, hi = np.percentile(luma, (5, 95))
    sel = (luma >= lo) & (luma <= hi)
    lum = float(np.median(_LIN[luma[sel]])) if sel.any() else float(np.median(_LIN[luma]))
    clip_hi = float((luma >= 253).mean())
    # cast: median ln(R/B) and ln(G/sqrt(RB)) over unclipped, non-black pixels whose linear luma is
    # within [0.25, 2] x the frame's own median (the same scene pixels at any exposure; the
    # median is cast-equivariant and robust to the clipped / dark tail)
    flat = img.reshape(-1, 3)
    lin = _LIN[flat]
    ratio = _LIN[luma.ravel()] / max(lum, 1e-6)
    ok = (flat.max(axis=1) < 254) & (flat.min(axis=1) > 4) & (ratio >= 0.25) & (ratio <= 2.0)
    if ok.sum() < 64:
        return lum, clip_hi, 0.0, 0.0
    lc = np.log(lin[ok])
    rb = float(np.median(lc[:, 0] - lc[:, 2]))
    gg = float(np.median(lc[:, 1] - 0.5 * (lc[:, 0] + lc[:, 2])))
    return lum, clip_hi, MIRED_PER_LN_RB * rb, -gg / GREEN_PER_DUV


def _body_of(path: str) -> str:
    """Normalised camera Make+Model of a file ("" when it cannot be read)."""
    try:
        m = read_meta(path)
    except Exception:  # noqa: BLE001 - an unreadable header just means "unknown body"
        return ""
    return f"{m.make} {m.model}".strip().lower()


def measure_paths(
    paths: list[str], settings: Settings | None = None, progress: Progress | None = None
) -> dict[str, Measure]:
    """Measures of the readable files among `paths` (embeddings from the feature store/cache)."""
    settings = settings or Settings()
    paths = [p for p in dict.fromkeys(paths) if not split_virtual(p)]
    feats = featurise(paths, settings)
    ok = [p for p in paths if p in feats]

    def one(p: str) -> Measure | None:
        try:
            img = load_preview(preview_source_for(p), PREVIEW_EDGE)
        except Exception as exc:  # noqa: BLE001
            log.warning("cannot decode %s: %s", p, exc)
            return None
        lum, clip, cm, cd = preview_measure(img)
        is_raw = Path(p).suffix.lower() in RAW_EXTS
        return Measure(
            p, feats[p].emb, lum, clip, cm, cd, is_raw, as_shot(p) if is_raw else None,
            _body_of(p),
        )

    out: dict[str, Measure] = {}
    with ThreadPoolExecutor(max_workers=max(1, settings.preview_workers)) as pool:
        for i, m in enumerate(pool.map(one, ok), start=1):
            if m is not None:
                out[m.path] = m
            if progress and i % 8 == 0:
                progress("measure", i, len(ok))
    return out


# ---- the match itself ------------------------------------------------------------------------


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def sync_keys(ref: dict[str, float | str]) -> tuple[dict[str, float], dict[str, float | str]]:
    """Split a reference's settings into (tone values, verbatim carried keys) of a sync."""
    tone = {
        k: float(v) for k, v in ref.items()
        if k in TONE_KEYS and k != "Exposure2012" and isinstance(v, float | int)
    }  # fmt: skip
    carried = {k: v for k, v in ref.items() if k not in _NO_CARRY}
    return tone, carried


def match_one(ref: Reference, t: Measure) -> MatchResult:
    """Develop settings for target `t` that make it look like the reference `ref`."""
    res = MatchResult(t.path)
    R = ref.measure
    if R.emb is None or t.emb is None:
        res.skipped_reason = "no embedding"
        return res
    res.cos = _cos(R.emb, t.emb)
    if res.cos < SCENE_COS:
        res.skipped_reason = "different scene"
        return res
    tone, carried = sync_keys(ref.settings)
    values: dict[str, float] = dict(tone)
    if t.clip_hi > CLIP_SKIP or R.clip_hi > CLIP_SKIP:
        res.notes.append("exposure not matched: highlights clipped (the reference's value is used)")
        e_ref = ref.settings.get("Exposure2012")
        if isinstance(e_ref, float | int):  # HasSettings=True would otherwise mean 0 EV
            values["Exposure2012"] = float(e_ref)
    elif t.lum > 0 and R.lum > 0:
        dev = math.log2(R.lum / t.lum)
        res.delta_ev = dev
        if abs(dev) > MAX_DEV:
            res.skipped_reason = "exposure differs by more than 2 EV"
            return res
        e_ref = ref.settings.get("Exposure2012")
        values["Exposure2012"] = (float(e_ref) if isinstance(e_ref, float | int) else 0.0) + dev
    # white balance: RAW targets with a physical (Kelvin) reference only
    s_t, s_r = t.scene_wb, R.scene_wb
    m_ref = ref.temp_mired
    if t.clip_hi > CLIP_SKIP or R.clip_hi > CLIP_SKIP:
        res.notes.append("white balance not matched: highlights clipped")
    elif t.is_raw and not (t.body and t.body == R.body):
        # the ShotSieve (rawpy matrix / McCamy) scale and Adobe's Kelvin differ per body; the
        # offset cancels only when reference and target are the same body (Review #6 M6)
        res.notes.append("different body: white balance not matched")
    elif t.is_raw and s_t and s_r and m_ref is not None:
        values["Temperature"] = 1e6 / max(s_t[0] + (m_ref - s_r[0]), 20.0)
        tint = ref.tint
        if tint is not None:
            values["Tint"] = tint + TINT_PER_DUV * (s_t[1] - s_r[1])
        res.wb = True
    elif not t.is_raw:
        res.notes.append("white balance not matched: JPEG target")
    else:
        res.notes.append("white balance not matched: no Kelvin reference or as-shot data")
    res.values, res.carried = values, carried
    res.process_version = ref.settings.get("ProcessVersion")
    return res


def match_measures(ref: Reference, targets: list[Measure]) -> list[MatchResult]:
    return [match_one(ref, t) for t in targets]


# ---- reference + targets ---------------------------------------------------------------------


def reference_settings(
    ref: str | Path, catalog: str | Path | None = None
) -> tuple[dict[str, float | str], str]:
    """(crs settings, source) of the reference; ValueError when it has none."""
    dev = crs.read_develop_xmp(ref)
    if dev is not None and dev.made_by_us:
        raise ValueError(REFUSE_OURS)
    if dev is not None and dev.has_settings and dev.settings:
        packet_src = "sidecar" if Path(ref).suffix.lower() in RAW_EXTS else "embedded"
        return dict(dev.settings), packet_src
    if catalog:
        want = os.path.abspath(ref)
        for e in lrcat.read_develop(catalog):
            if os.path.abspath(e.path) == want and e.settings:
                return dict(e.settings), "catalog"
    raise ValueError(REFUSE_UNEDITED)


def _final(row: dict, ovr: dict | None) -> str:
    return ovr["decision"] if ovr else row["decision"]


def resolve_targets(
    folder: Path,
    ref: str,
    targets: str | list[str] | None,
    ids: list[int] | None = None,
) -> list[str]:
    """Target paths: 'keeps' | 'group' | 'camera:MODEL' | explicit ids/paths (never the ref)."""
    with Cache.for_folder(folder) as c:
        decisions = c.load_decisions()
        if not decisions:
            raise ValueError("this folder has not been culled yet")
        overrides = c.overrides()
        by_id = c.ids_for(list(decisions))
        meta: dict[str, dict] = {}
        spec = targets if isinstance(targets, str) else ""
        if spec.lower().startswith("camera:"):
            import json

            with c._lock:
                rows = c._con.execute("SELECT path, meta_json FROM images").fetchall()
            for p, mj in rows:
                try:
                    meta[c._abs(p)] = json.loads(mj)
                except (TypeError, ValueError):
                    pass
    paths = [p for p in sorted(decisions) if not split_virtual(p) and p != ref]
    if ids is not None:
        want = set(ids)
        return [p for p in paths if by_id.get(p) in want]
    if isinstance(targets, list):
        explicit = {os.path.abspath(p) for p in targets}
        return [p for p in paths if p in explicit]
    spec = targets or "keeps"
    if spec == "keeps":
        return [p for p in paths if _final(decisions[p], overrides.get(p)) == "keep"]
    if spec == "group":
        g = decisions.get(ref, {}).get("group_id")
        if g is None:
            raise ValueError("the reference is not part of this cull")
        return [p for p in paths if decisions[p]["group_id"] == g]
    if spec.lower().startswith("camera:"):
        cam = spec.split(":", 1)[1].strip().lower()
        out = []
        for p in paths:
            m = meta.get(p, {})
            if cam in (
                str(m.get("model", "")).lower(), str(m.get("camera_serial", "")).lower(),
                f"{m.get('make', '')}-{m.get('model', '')}".strip("-").lower(),
            ):
                out.append(p)
        return out
    raise ValueError("targets must be keeps, group, camera:MODEL or a list of images")


def match(
    folder: str | Path,
    ref: str | Path,
    targets: str | list[str] | None = "keeps",
    settings: Settings | None = None,
    *,
    ids: list[int] | None = None,
    dry_run: bool = False,
    embed_copies: str | Path | bool | None = None,
    catalog: str | Path | None = None,
    progress: Progress | None = None,
) -> list[MatchResult]:
    """Match `targets` of a culled shoot to the reference frame and write `crs:` XMP.

    Returns one MatchResult per target. The reference's own develop settings come from its
    sidecar, embedded XMP or `catalog`; without any, ValueError("edit the reference ...").
    """
    settings = settings or Settings()
    folder = Path(os.path.abspath(folder))
    ref_path = os.path.abspath(ref if os.path.isabs(str(ref)) else folder / ref)
    ref_settings, source = reference_settings(ref_path, catalog)
    paths = resolve_targets(folder, ref_path, targets, ids)
    cat_paths: set[str] = set()
    if catalog:
        cat_paths = {e.path for e in lrcat.read_develop(catalog)}
    ms = measure_paths([ref_path, *paths], settings, progress)
    if ref_path not in ms:
        raise ValueError(f"cannot read the reference image: {ref_path}")
    reference = Reference(ref_path, ref_settings, ms[ref_path], source)
    tag = "match:" + hashlib.sha256(
        repr(sorted((k, str(v)) for k, v in ref_settings.items())).encode()
    ).hexdigest()[:10]
    embed = bool(embed_copies)
    copy_root = (
        Path(embed_copies) if isinstance(embed_copies, str | Path) and str(embed_copies)
        else folder / "shotsieve_edits"
    )
    if embed:
        check_copy_root(copy_root, folder)
    results: list[MatchResult] = []
    rows: list[dict[str, Any]] = []
    for n, p in enumerate(paths, start=1):
        if progress:
            progress("write", n, len(paths))
        t = ms.get(p)
        if t is None:
            results.append(MatchResult(p, skipped_reason="unreadable"))
            continue
        r = match_one(reference, t)
        if r.skipped_reason is None:
            _write(r, p, tag, embed, copy_root, folder, cat_paths, dry_run)
        results.append(r)
        if r.skipped_reason is None and not dry_run:
            rows.append(
                {
                    "path": p, "values": r.values,
                    "confidence": {"match": {"cos": r.cos, "delta_ev": r.delta_ev,
                                             "reference": ref_path, "notes": r.notes},
                                   "needs_edit": False},
                    "written_mode": r.written_mode, "profile_hash": tag,
                }  # fmt: skip
            )
    if rows:
        with Cache.for_folder(folder) as c:
            c.put_edits(rows)
    return results


def _write(
    r: MatchResult, p: str, tag: str, embed: bool, copy_root: Path, folder: Path,
    cat_paths: set[str], dry_run: bool,
) -> None:  # fmt: skip
    mode = _mode_for(p, embed)
    if not r.values and not r.carried:
        r.status, r.written_mode = "nothing_to_write", "report"
        return
    if mode == "unsupported":
        r.status, r.written_mode = "unsupported", "unsupported"
        return
    if mode == "report":
        r.status, r.written_mode = "report_only", "report"
        return
    dst = None
    if mode == "embed":
        dst = copy_root / Path(os.path.relpath(Path(p).parent, folder)) / Path(p).name
    ok, why = crs.writable(p, "sidecar" if mode == "sidecar" else "embed", p in cat_paths, dst)
    if not ok:
        r.status, r.written_mode = why, why
        return
    if dry_run:
        r.status, r.written_mode = "written", "dry_run"
        return
    kw: dict[str, Any] = dict(
        carry=r.carried, profile_hash=tag, confidence={"cos": r.cos or 0.0}, wb=r.wb,
        catalog_has=p in cat_paths, process_version=r.process_version,
    )  # fmt: skip
    if mode == "embed":
        kw.update(mode="embed", out_dir=copy_root / Path(os.path.relpath(Path(p).parent, folder)))
    res = crs.write_develop(p, r.values, **kw)
    r.status = "written" if res.written else res.status
    r.written_mode = ("sidecar" if mode == "sidecar" else "copy") if res.written else res.status


def summarize(results: list[MatchResult], dry_run: bool = False) -> dict[str, Any]:
    """Counts and a short per-image diff for CLI / API output."""
    skipped: dict[str, int] = {}
    for r in results:
        why = r.skipped_reason or (r.status if r.status not in ("written", "report_only",
                                                                 "nothing_to_write", "") else "")
        if why:
            skipped[why] = skipped.get(why, 0) + 1
    return {
        "n": len(results), "dry_run": dry_run,
        "written_sidecar": sum(r.written_mode == "sidecar" for r in results),
        "written_copies": sum(r.written_mode == "copy" for r in results),
        "would_write": sum(r.written_mode == "dry_run" for r in results),
        "report_only": sum(r.status == "report_only" for r in results),
        "skipped": skipped,
        "warnings": ([LIGHTROOM_EMBED_NOTE]
                     if any(r.status == "report_only" for r in results) else []),
        "diff": [
            {"name": Path(r.path).name, "status": r.skipped_reason or r.status,
             "values": {k: round(v, 3) for k, v in r.values.items()},
             "delta_ev": None if r.delta_ev is None else round(r.delta_ev, 3),
             "cos": None if r.cos is None else round(r.cos, 3), "wb": r.wb, "notes": r.notes}
            for r in results[:500]
        ],
    }  # fmt: skip
