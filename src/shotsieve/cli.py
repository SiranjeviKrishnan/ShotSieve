"""Command line entrypoint: models, ingest, cull, train, eval, serve."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TextIO

from shotsieve import __version__
from shotsieve.config import Settings, load_settings
from shotsieve.io.formats import split_virtual

STAGE_ORDER = ("scan", "features", "faces", "embed", "group", "score", "write")


class ProgressBar:
    """Progress bar on stderr (carriage-return redraw on a TTY, plain lines otherwise)."""

    def __init__(self, stream: TextIO | None = None, width: int = 28, min_interval: float = 0.1):
        self.stream = stream if stream is not None else sys.stderr
        self.width, self.min_interval = width, min_interval
        self._lock = threading.Lock()
        self._last = 0.0
        self._state: dict[str, tuple[int, int]] = {}
        self._tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self._printed: set[str] = set()

    def __call__(self, stage: str, done: int, total: int) -> None:
        with self._lock:
            self._state[stage] = (done, total)
            now = time.monotonic()
            finished = total > 0 and done >= total
            if not finished and now - self._last < self.min_interval:
                return
            self._last = now
            if self._tty:
                self.stream.write("\r" + self._render(stage, done, total))
                if finished:
                    self.stream.write("\n")
            elif finished and stage not in self._printed:
                self._printed.add(stage)
                self.stream.write(self._render(stage, done, total) + "\n")
            self.stream.flush()

    def _render(self, stage: str, done: int, total: int) -> str:
        frac = 1.0 if total <= 0 else min(1.0, done / total)
        fill = int(round(frac * self.width))
        return f"{stage:<9s}[{'#' * fill}{'.' * (self.width - fill)}] {done}/{total}  "

    def close(self) -> None:
        if self._tty:
            self.stream.write("\n")
            self.stream.flush()


def print_table(rows: Sequence[Sequence[Any]], headers: Sequence[str] | None = None) -> None:
    """Plain-text table on stdout."""
    body = [[str(c) for c in r] for r in rows]
    head = [str(h) for h in headers] if headers else None
    cols = len(head or body[0]) if (head or body) else 0
    widths = [max(len(r[i]) for r in ([head] if head else []) + body) for i in range(cols)]
    if head:
        print("  ".join(h.ljust(w) for h, w in zip(head, widths, strict=True)))
        print("  ".join("-" * w for w in widths))
    for r in body:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)))


def _settings(args: argparse.Namespace) -> Settings:
    return load_settings(getattr(args, "config", None))


def _check_models() -> None:
    from shotsieve.models.registry import group_names, verify_model

    missing = [n for n in group_names("core") if not verify_model(n)]
    if missing:
        raise RuntimeError(
            f"model files missing: {', '.join(missing)}. Run `shotsieve models download`."
        )


def _dir_arg(value: str) -> Path:
    p = Path(value).expanduser()
    if not p.is_dir():
        raise argparse.ArgumentTypeError(f"not a directory: {value}")
    return p


# ---- subcommands ------------------------------------------------------------------------


def cmd_models(args: argparse.Namespace) -> int:
    from shotsieve.models.registry import MODEL_SPECS, ensure_models, models_dir, verify_model

    if args.models_cmd == "download":
        extra = tuple(g for g in (args.extra or "").split(",") if g)
        last = {"name": None, "pct": -10}

        def show(ev: dict) -> None:
            pct = int(100 * ev["done"] / max(ev["total"], 1))
            if ev["name"] != last["name"] or pct >= last["pct"] + 10:
                last.update(name=ev["name"], pct=pct)
                print(f"  {ev['name']}: {pct}% of {ev['total'] / 1e6:.0f} MB", flush=True)

        paths = ensure_models(groups=("core", *extra), progress=show)
        print_table(
            [(n, f"{p.stat().st_size:,} B", str(p)) for n, p in paths.items()],
            ["model", "size", "path"],
        )
        return 0
    rows, ok = [], True
    for name, spec in MODEL_SPECS.items():
        good = verify_model(name, check_hash=True)
        if spec.group == "core":
            ok &= good
        rows.append((name, "ok" if good else ("MISSING/CORRUPT" if spec.group == "core"
                                              else "not installed"), spec.filename))
    print_table(rows, ["model", "status", "file"])
    print(f"models dir: {models_dir()}")
    return 0 if ok else 1


def cmd_ingest(args: argparse.Namespace) -> int:
    from shotsieve.ingest.copier import ingest

    rep = ingest(args.src, args.dest, backup=args.backup, by_date=args.by_date)
    print_table(
        [
            ("copied", len(rep.copied)),
            ("skipped (already present)", len(rep.skipped)),
            ("failed", len(rep.failed)),
            ("bytes copied", f"{rep.bytes_copied:,}"),
            ("seconds", f"{rep.seconds:.2f}"),
        ]
    )
    for path, err in rep.failed:
        print(f"FAILED {path}: {err}", file=sys.stderr)
    return 1 if rep.failed else 0


def result_to_json(res: Any, folder: Path) -> dict[str, Any]:
    images = []
    for r in res.images:
        images.append(
            {
                "path": str(r.master_path),
                "preview_path": str(r.path),
                "decision": r.decision,
                "reasons": r.reasons,
                "reason_texts": r.reason_texts,
                "score": round(r.score, 4),
                "prior": round(r.prior, 4),
                "group_id": r.group_id,
                "burst_id": r.burst_id,
                "best_in_group": r.is_best_in_group,
                "audit": r.audit,
                "audit_prob": r.audit_prob,
                "overridden": r.overridden,
                "model_decision": r.model_decision,
            }
        )
    return {
        "folder": str(folder),
        "thresholds": res.thresholds,
        "stats": res.stats,
        "images": images,
    }


def print_cull_summary(res: Any) -> None:
    s = res.stats
    c = s["counts"]
    n = max(sum(c.values()), 1)  # photos + video frame candidates
    v = s.get("video")
    print_table(
        [
            ("images", s["n_images"]),
            *(
                [("video", f"{v['picks']} frame(s) from {v['n_videos']} file(s)"
                  + (f", {v['realtime_x']:.1f}x realtime" if v.get("realtime_x") else ""))]
                if v else []
            ),
            ("keep", f"{c['keep']}  ({100 * c['keep'] / n:.0f}%)"),
            ("review", f"{c['review']}  ({100 * c['review'] / n:.0f}%)"),
            ("reject", f"{c['reject']}  ({100 * c['reject'] / n:.0f}%)"),
            ("duplicate groups", s["n_groups"]),
            ("cached / processed", f"{s['n_cached']} / {s['n_processed']}"),
            *(
                [("re-decide", f"{s['redecide']}"
                  + (f": {s['n_new']} new, {s['n_removed']} removed, {s['n_affected']} affected"
                     if s["redecide"] == "incremental" else "")
                  + (f" ({s['redecide_reason']})" if s["redecide"] == "full" and s["n_cached"]
                     else ""))]
                if s.get("redecide") and s["redecide"] != "none" else []
            ),
            ("throughput", f"{s['imgs_per_s']:.1f} img/s"),
            ("peak RSS", f"{s['peak_rss_mb']:.0f} MB"),
            ("personal model rows", s["personal_n_train"]),
            (
                "thresholds",
                f"keep>={res.thresholds['t_keep']:.2f} reject<{res.thresholds['t_reject']:.2f}",
            ),
        ],
        ["metric", "value"],
    )


def cmd_cull(args: argparse.Namespace) -> int:
    from shotsieve.pipeline.runner import run_cull

    _check_models()
    settings = _settings(args)
    if args.video or args.export_keeps:
        settings.video = True
    bar = ProgressBar()
    res = run_cull(
        args.folder, settings, bar, write_xmp=args.write_xmp,
        redraw_audit=args.redraw_audit, force_ratings=args.force_ratings, full=args.full,
    )  # fmt: skip
    bar.close()
    if args.write_xmp:
        from shotsieve.io.xmp import lightroom_warning

        warn = lightroom_warning(
            r.master_path for r in res.images if not split_virtual(r.master_path)
        )
        if warn:
            print(f"warning: {warn}", file=sys.stderr)
    if args.json:
        Path(args.json).write_text(json.dumps(result_to_json(res, args.folder), indent=2))
        print(f"wrote {args.json}", file=sys.stderr)
    print_cull_summary(res)
    if args.export_keeps:
        from shotsieve.pipeline.video import export_frame

        n = 0
        for r in res.images:
            if split_virtual(r.master_path) and r.decision == "keep":
                try:
                    export_frame(Path(args.folder), r.master_path)
                    n += 1
                except (OSError, ValueError) as exc:
                    print(f"warning: cannot export {r.master_path.name}: {exc}", file=sys.stderr)
        print(f"exported {n} frame grab(s) to {Path(args.folder) / 'frame_grabs'}")
    return 0


def _fmt_offset(s: float) -> str:
    sign = "-" if s < 0 else "+"
    s = abs(round(s))
    return f"{sign}{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d} ({sign}{s} s)"


def cmd_clock(args: argparse.Namespace) -> int:
    from datetime import datetime

    from shotsieve.grouping import clock
    from shotsieve.io.xmp import write_sidecar
    from shotsieve.pipeline.cache import Cache, cache_path

    if not cache_path(args.folder).exists():
        print("error: cull the folder first (shotsieve cull)", file=sys.stderr)
        return 1
    with Cache.for_folder(args.folder) as c:
        for item in args.set or []:
            serial, _, value = item.rpartition("=")
            if not serial:
                print(f"error: --set wants SERIAL=+3725 (got {item!r})", file=sys.stderr)
                return 2
            c.set_clock_offset(serial, clock.parse_offset(value), "manual")
        sugs, offsets = clock.analyse(c, auto_apply=args.apply)
        items = c.clock_inputs()
        cams: dict[str, int] = {}
        for i in items:
            cams[i.serial] = cams.get(i.serial, 0) + 1
        print_table([(s, n, _fmt_offset(offsets[s]["offset_s"]) + f" [{offsets[s]['source']}]"
                      if s in offsets else "-") for s, n in sorted(cams.items())],
                    ["camera", "frames", "offset"])  # fmt: skip
        for s in sugs:
            if s.offset_s is None:
                print(f"\n{s.serial}: no consistent offset; candidate pairs to pick from:")
                for p in s.alternatives:
                    print(f"  {Path(p.ref_path).name} <-> {Path(p.path).name}  cos {p.cos:.2f}  "
                          f"offset {_fmt_offset(p.delta_s)}")  # fmt: skip
            else:
                tag = "applied" if (args.apply and s.auto_apply) else "suggested"
                print(
                    f"\n{s.serial}: {tag} {_fmt_offset(s.offset_s)}  confidence "
                    f"{s.confidence:.2f}  ({s.support}/{s.n_candidates} matching frames)"
                )
                if s.pair:
                    print(f"  pair: {Path(s.pair.ref_path).name} <-> {Path(s.pair.path).name}")
                if s.auto_apply is False and not args.apply:
                    print("  re-run with --apply to use it (or --set SERIAL=+seconds)")
        if args.apply:  # explicit confirmation: store every suggestion that found an offset
            for s in sugs:
                if s.offset_s is not None:
                    c.set_clock_offset(s.serial, s.offset_s, "auto", s.confidence)
            offsets = c.get_clock_offsets()
        if args.write_xmp_time:
            by_serial = clock.offsets_from_meta(offsets)
            n = 0
            for i in items:
                if i.serial in by_serial and i.ts_source == "exif":
                    t = datetime.fromtimestamp(clock.adjust(i.ts, i.serial, by_serial))
                    stamp = t.strftime("%Y-%m-%dT%H:%M:%S")
                    write_sidecar(i.path, date_created=stamp, record=False)
                    n += 1
            print(f"\nwrote photoshop:DateCreated to {n} sidecar(s)")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from shotsieve.personal.dataset import build_training_set
    from shotsieve.personal.model import PersonalModel, default_model_path

    if not (args.lrcat or args.xmp_folder or args.from_overrides):
        print(
            "error: give at least one of --lrcat, --xmp-folder, --from-overrides", file=sys.stderr
        )
        return 2
    _check_models()
    settings = _settings(args)
    bar = ProgressBar()
    ts = build_training_set(
        args.lrcat, args.xmp_folder or [], args.from_overrides or [], settings, bar
    )
    bar.close()
    model = PersonalModel().fit(ts.X, ts.y, ts.w, ts.embeddings)
    model.meta = {"sources": sorted(set(ts.sources)), "trained_at": time.time()}
    out = Path(args.out) if args.out else default_model_path(settings)
    rows = [
        ("labelled rows", len(ts)),
        ("keep / not keep", f"{int(ts.y.sum())} / {int(len(ts) - ts.y.sum())}"),
        ("labelled files missing on disk", ts.n_missing),
        ("video frame labels (not used for training)", ts.n_video),
        ("sources", ", ".join(sorted(set(ts.sources))) or "-"),
    ]
    if model.trained:
        model.save(out)
        rows += [
            ("model", "trained"),
            ("calibration", model.calibration),
            (
                "blend weight w",
                f"{model.n_train / (model.n_train + settings.personal_prior_n):.2f}",
            ),
            ("saved to", str(out)),
        ]
    else:
        rows += [
            ("model", "NOT trained: need >= 40 rows and both classes; using the quality prior"),
        ]
    print_table(rows, ["training", "value"])
    return 0


def cmd_sync(args: argparse.Namespace) -> int:
    """Lightroom sync-back (T20): exit 0 ok, 2 bad arguments, 4 catalog busy and the copy failed."""
    from shotsieve.io.lrcat import CatalogBusyError, CatalogError
    from shotsieve.pipeline import lrsync

    settings = _settings(args)
    args.lrcat = os.path.abspath(Path(args.lrcat).expanduser())  # a relative path is fine here
    try:
        res = lrsync.sync(
            args.folder, args.lrcat, settings, dry_run=args.dry_run, report=not args.no_report
        )
    except CatalogError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except CatalogBusyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 4
    if args.json:
        print(json.dumps(res.to_dict(), indent=2, default=str))
        return 0
    head = "DRY RUN (nothing was changed)" if args.dry_run else "synced"
    print(f"{head}: {args.lrcat}" + ("  [read from a private copy]" if res.copied else ""))
    print_table(
        [
            ("catalog images in folder", res.n_catalog),
            ("matched / unmatched", f"{res.n_matched} / {res.n_unmatched}"
             + (f" ({res.n_fallback} by file name + capture time)" if res.n_fallback else "")),
            ("new overrides", f"{res.n_new}  ({res.n_keeps} keeps, {res.n_rejects} rejects)"),
            ("changed / cleared", f"{res.n_changed} / {res.n_cleared}"),
            ("conflicts, your ShotSieve choice kept", res.n_conflicts_ui_won),
            ("conflicts, Lightroom won", res.n_conflicts_lr_won),
        ],
        ["sync", "value"],
    )
    r = res.report
    if r:
        pct = lambda x: f"{100 * x:.1f} %"  # noqa: E731
        print()
        print_table(
            [
                ("frames compared", r["n_evaluated"]),
                ("missed keepers (frames)", f"{r['missed_keepers']} of {r['n_kept_truth']} "
                 f"({pct(r['missed_keeper_rate'])})"),
                ("moment gate rate (missed, review counts as saved)", pct(r["group_gate_rate"])),
                ("sent to review", pct(r["review_fraction"])),
                ("keepers in review", r["keepers_in_review"]),
                ("reject precision", "-" if r["reject_precision"] is None
                 else pct(r["reject_precision"])),
            ],
            ["this shoot", "value"],
        )
        for name in r["missed_files"][:20]:
            print(f"  missed: {name}")
    return 0


def _print_sliders(rows: list[dict[str, Any]]) -> None:
    print_table(
        [
            (r["slider"], r["mode"], r["n"], "-" if r["mae_cv"] is None else r["mae_cv"],
             "-" if r["mae_median"] is None else r["mae_median"])
            for r in rows
        ],
        ["slider", "mode", "n", "CV MAE", "median-baseline MAE"],
    )  # fmt: skip


def cmd_edit(args: argparse.Namespace) -> int:
    from shotsieve.personal.edit_profile import InsufficientEditsError
    from shotsieve.pipeline import edit

    settings = _settings(args)
    if args.edit_cmd == "train":
        if not (args.catalog or args.xmp):
            print("error: give --catalog and/or --xmp DIR", file=sys.stderr)
            return 2
        _check_models()
        bar = ProgressBar()
        try:
            _, info = edit.train_profile(
                args.catalog, args.xmp or [], settings, bar, dng_render=args.dng_render
            )
        except InsufficientEditsError as exc:
            bar.close()
            print(f"error: {exc}", file=sys.stderr)
            return 3
        bar.close()
        if args.json:
            print(json.dumps(info, indent=2))
            return 0
        print(f"trained on {info['n']} edited frame(s) in {info['seconds']} s "
              f"({info['carry_keys']} carried preset keys) -> {info['saved_to']}")  # fmt: skip
        _print_sliders(info["sliders"])
        for k, v in info["notes"].items():
            if v:
                print(f"note: {k} = {v}")
        return 0
    if args.edit_cmd == "status":
        st = edit.profile_status(settings)
        if args.json:
            print(json.dumps(st, indent=2))
            return 0
        if not st["trained"]:
            print("no editing profile yet (shotsieve edit train --catalog X.lrcat)")
            return 0
        print(f"profile {st['profile_hash']}: {st['n']} edits, sources {st['sources']}")
        _print_sliders(st["sliders"])
        return 0
    _check_models()
    embed: str | bool | None = args.embed_copies
    if embed == "":
        embed = True
    bar = ProgressBar()
    try:
        res = edit.apply_folder(
            args.folder, settings, scope=args.scope, dry_run=args.dry_run, embed_copies=embed,
            catalog=args.catalog, progress=bar,
        )
    except edit.NoProfileError as exc:
        bar.close()
        print(f"error: {exc}", file=sys.stderr)
        return 3
    bar.close()
    if args.json:
        print(json.dumps(res, indent=2))
        return 0
    verb = "would write" if args.dry_run else "wrote"
    print(f"{verb} {res['written_sidecar']} sidecar(s) and {res['written_copies']} embedded "
          f"copy(ies); {res['needs_edit']} image(s) need your edit")  # fmt: skip
    if res["report_only"]:
        print(f"{res['report_only']} JPEG/DNG file(s) reported only")
    for why, n in res["skipped"].items():
        print(f"skipped {n}: {why}")
    for w in res["warnings"]:
        print(f"warning: {w}", file=sys.stderr)
    return 0


def cmd_match(args: argparse.Namespace) -> int:
    from shotsieve.pipeline import match

    settings = _settings(args)
    _check_models()
    embed: str | bool | None = args.embed_copies
    if embed == "":
        embed = True
    ref = Path(args.ref).expanduser()
    if not ref.exists() and (args.folder / ref).exists():
        ref = args.folder / ref
    bar = ProgressBar()
    try:
        results = match.match(
            args.folder, os.path.abspath(ref), args.targets, settings,
            dry_run=args.dry_run, embed_copies=embed, catalog=args.catalog, progress=bar,
        )
    except ValueError as exc:
        bar.close()
        print(f"error: {exc}", file=sys.stderr)
        return 3
    bar.close()
    res = match.summarize(results, args.dry_run)
    if args.json:
        print(json.dumps(res, indent=2))
        return 0
    verb = "would write" if args.dry_run else "wrote"
    n = res["would_write"] if args.dry_run else res["written_sidecar"] + res["written_copies"]
    print(f"{verb} {n} of {res['n']} target(s) matched to {ref.name} "
          f"({res['written_sidecar']} sidecar(s), {res['written_copies']} copy(ies))")  # fmt: skip
    for why, k in res["skipped"].items():
        print(f"skipped {k}: {why}")
    if res["report_only"]:
        print(f"{res['report_only']} JPEG/DNG file(s) reported only")
    for w in res["warnings"]:
        print(f"warning: {w}", file=sys.stderr)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from shotsieve.io.lrcat import read_catalog
    from shotsieve.pipeline import evaluate
    from shotsieve.pipeline.runner import run_cull

    settings = _settings(args)
    if args.truth == "lrcat":
        if not args.lrcat:
            print("error: --truth lrcat needs --lrcat CAT", file=sys.stderr)
            return 2
        truth = evaluate.truth_from_catalog(read_catalog(args.lrcat), settings.min_rating)
    else:
        truth = evaluate.truth_from_xmp(args.folder, settings)
    _check_models()
    bar = ProgressBar()
    res = run_cull(args.folder, settings, bar)
    bar.close()
    rep = evaluate.evaluate(res.images, truth)
    nan = lambda v: "n/a" if v != v else f"{v:.3f}"  # noqa: E731
    print_table(
        [
            ("images", rep.n_images),
            ("with ground truth", rep.n_evaluated),
            ("keepers in truth", rep.n_kept_truth),
            ("missed keepers (rejected)", rep.missed_keepers),
            ("GROUP missed-keeper rate (gate, review = saved)", f"{rep.group_gate_rate:.3f}"),
            ("  strict form (review = miss, secondary)", f"{rep.group_missed_keeper_rate:.3f}"),
            (
                "  user keeps with no ShotSieve keep in their group",
                f"{rep.group_missed_keepers}  ({rep.group_review_rescuable} rescuable in review)",
            ),
            (
                "moment miss rate (groups with a user keep)",
                f"{rep.moment_miss_rate:.3f}  ({rep.moment_misses}/{rep.n_kept_groups}, "
                f"{rep.moment_review_rescuable} rescuable)",
            ),
            ("frame-level missed-keeper rate", f"{rep.missed_keeper_rate:.3f}"),
            ("keepers in review", rep.keepers_in_review),
            ("keepers kept", rep.keepers_kept),
            ("review fraction", f"{rep.review_fraction:.3f}"),
            ("reject rate", f"{rep.reject_rate:.3f}"),
            ("reject precision", nan(rep.reject_precision)),
            ("keep precision", nan(rep.keep_precision)),
            ("decisions", json.dumps(rep.counts)),
        ],
        ["metric", "value"],
    )
    for p in rep.missed_paths:
        print(f"missed keeper: {p}", file=sys.stderr)
    if args.gate is not None and rep.group_gate_rate > args.gate:
        print(
            f"GATE FAILED: group-level missed-keeper rate (review counted as saved) "
            f"{rep.group_gate_rate:.3f} > {args.gate} (strict {rep.group_missed_keeper_rate:.3f})",
            file=sys.stderr,
        )
        return 3
    return 0


def _check_search_models() -> None:
    from shotsieve.models.text_image import search_models_ready

    if not search_models_ready():
        raise RuntimeError(
            "search models missing. Run `shotsieve models download --extra search`."
        )


def cmd_index(args: argparse.Namespace) -> int:
    from shotsieve.models.ocr import ocr_models_ready
    from shotsieve.pipeline.indexer import Indexer

    if args.ocr and not ocr_models_ready():
        raise RuntimeError("OCR models missing. Run `shotsieve models download --extra ocr`.")
    if not args.no_search:
        _check_search_models()
    ix = Indexer(
        args.folder, _settings(args), people=not args.no_people,
        images=not args.no_search, ocr=True if args.ocr else None,
    )
    t = threading.Thread(target=ix.run_blocking, daemon=True)
    t.start()
    last = ""
    while t.is_alive():
        t.join(1.0)
        st = ix.status()
        line = f"{st['task']}: {st['done']}/{st['total']}  {st['img_s']:.1f} img/s"
        if line != last and sys.stderr.isatty():
            sys.stderr.write("\r" + line.ljust(50))
            last = line
    if sys.stderr.isatty():
        sys.stderr.write("\n")
    if ix.error:
        print(f"indexing failed: {ix.error}", file=sys.stderr)
        return 1
    print(f"indexed {ix.done}/{ix.total} images ({ix.n_embedded} new)")
    return 0


def cmd_roster(args: argparse.Namespace) -> int:
    from shotsieve.io.roster import RosterError, parse_roster
    from shotsieve.pipeline import jerseys
    from shotsieve.pipeline.cache import Cache, cache_path

    if not cache_path(args.folder).exists():
        print("error: this folder has not been culled yet", file=sys.stderr)
        return 1
    with Cache.for_folder(args.folder) as c:
        if args.csv:
            try:
                entries = parse_roster(Path(args.csv))
            except (RosterError, OSError) as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
            jerseys.set_roster(c, entries)
        if args.team is not None:
            jerseys.set_team(c, args.team)
        summ = jerseys.roster_summary(c)
        if args.write_xmp:
            res = jerseys.write_xmp(c)
            summ["xmp"] = res
    if args.json:
        print(json.dumps(summ, indent=2))
        return 0
    print(f"roster: {summ['entries']} players; {summ['images_matched']} of "
          f"{summ['images_with_numbers']} images with a number match the roster")
    if summ["ambiguous_numbers"]:
        print("numbers shared by two teams (no name written; pass --team): "
              + ", ".join(f"#{n}" for n in summ["ambiguous_numbers"])
              + f" - {summ['images_ambiguous']} frame(s) to review")
    if summ["unmatched_numbers"]:
        print("numbers not on the roster: " + ", ".join(f"#{n}" for n in summ["unmatched_numbers"]))
    if args.write_xmp:
        x = summ["xmp"]
        print(f"xmp: {x['written']} sidecar(s) updated, {x['items_added']} tag(s) added")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    from shotsieve.pipeline.people import ConsentError
    from shotsieve.pipeline.search import parse_query, search

    if parse_query(args.query).text:
        _check_search_models()
    warnings: list[str] = []
    try:
        hits = search(args.folder, args.query, args.k, _settings(args), warnings=warnings)
    except ConsentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)
    if args.json:
        print(json.dumps([{"id": h.id, "score": h.score, "path": h.path, "why": h.why}
                          for h in hits], indent=2))  # fmt: skip
        return 0
    print_table(
        [(i + 1, f"{h.score:.3f}", os.path.relpath(h.path, args.folder))
         for i, h in enumerate(hits)],
        ["#", "score", "image"],
    )
    return 0


def cmd_shotlist(args: argparse.Namespace) -> int:
    from shotsieve.pipeline import shotlist

    _check_search_models()
    spec = args.list
    try:
        if Path(spec).expanduser().is_file():
            items = shotlist.load_items(None, spec)
        else:
            items = shotlist.load_items(spec)
        cov = shotlist.run(args.folder, items, _settings(args), tag=args.tag_xmp)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    if args.json:
        print(json.dumps(cov.public(), indent=2))
        return 0
    marks = {"covered": "ok ", "uncertain": "?  ", "weak": "~  ", "missing": "NO ",
             "pending": "...  "}  # fmt: skip
    rows = []
    for r in cov.results:
        it = r.item
        label = ("! " if it.must else "") + it.text
        label += f" x{it.min_count}" if it.min_count > 1 else ""
        rows.append((marks[r.status] + r.status, label, r.n_hits, r.detail))
    print_table(rows, ["status", "shot", "frames", "note"])
    c = cov.counts()
    print(f"{c['covered']} covered, {c['uncertain']} uncertain, {c['weak']} weak, "
          f"{c['missing']} missing, "
          f"{c['pending']} pending (index {cov.indexed_pct:.0f} %)")  # fmt: skip
    for r in cov.must_missing:
        print(f"MISSING must-have: {r.item.text}", file=sys.stderr)
    for r in cov.must_uncertain:
        print(f"REVIEW must-have (uncertain match): {r.item.text}", file=sys.stderr)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from shotsieve.server.app import create_app

    settings = _settings(args)
    from shotsieve.server.lifecycle import remove_server_json_if_ours, write_server_json

    try:  # once, before any request thread exists: the import's HOME redirect is then never raced
        from shotsieve._ort import import_onnxruntime

        import_onnxruntime()
    except ImportError:
        pass
    app = create_app(settings=settings, default_folder=args.folder)
    url = f"http://127.0.0.1:{args.port}"
    print(f"ShotSieve review UI at {url}", file=sys.stderr)
    if not args.no_open:
        threading.Timer(1.0, _open_browser, args=(url,)).start()
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=args.port, log_level="warning")
    )
    app.state.uvicorn_server = server  # POST /api/quit flips should_exit

    # uvicorn binds in startup(); publish server.json only once the socket is listening
    original_startup = server.startup

    async def startup(sockets: Any = None) -> None:
        await original_startup(sockets)
        if not server.should_exit:
            write_server_json(args.port)

    server.startup = startup  # type: ignore[method-assign]
    try:
        server.run()
    finally:
        remove_server_json_if_ours()
    return 0


def cmd_app(args: argparse.Namespace) -> int:
    """The native Mac window (pywebview); the process is the app."""
    from shotsieve.desktop import run_app

    try:
        rc = run_app(args.folder)
    except ImportError as exc:
        print(f"The ShotSieve window needs pywebview ({exc}). Run the install again.",
              file=sys.stderr)
        return 1
    # a stray non-daemon thread must not leave a ghost app process behind
    t = threading.Timer(8.0, os._exit, args=(rc,))
    t.daemon = True
    t.start()
    return rc


PORT_RANGE = range(8765, 8776)


def _port_free(port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # TIME_WAIT is not "busy"
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def cmd_doctor(args: argparse.Namespace) -> int:
    """Check the install; one plain sentence per failure, exit 1 if any."""
    import shutil

    from shotsieve.config import data_dir
    from shotsieve.models.registry import group_names, verify_model

    problems: list[str] = []
    if sys.version_info[:2] != (3, 11):
        problems.append(
            f"Python {sys.version_info[0]}.{sys.version_info[1]} is in use but ShotSieve needs "
            "3.11: run the install again."
        )
    bad = [n for n in group_names("core") if not verify_model(n, check_hash=True)]
    if bad:
        problems.append(
            f"The core models are missing or damaged ({', '.join(bad)}): run the install again."
        )
    try:
        d = data_dir()
        probe = d / f".doctor-{os.getpid()}"
        probe.write_text("ok")
        probe.unlink()
    except OSError as exc:
        problems.append(f"ShotSieve cannot write to its data folder ({exc.strerror or exc}).")
        d = Path.cwd()
    if not any(_port_free(p) for p in PORT_RANGE):
        problems.append(
            f"No free port between {PORT_RANGE.start} and {PORT_RANGE.stop - 1}: close the "
            "program using them, or quit another ShotSieve."
        )
    try:
        free = shutil.disk_usage(d).free
        if free < 1_000_000_000:
            problems.append(
                f"Only {free / 1e9:.1f} GB of disk space is free; ShotSieve needs at least 1 GB."
            )
    except OSError:
        pass
    for line in problems:
        print(line)
    if not problems:
        print("ShotSieve is ready.")
    return 1 if problems else 0


def _open_browser(url: str) -> None:
    import webbrowser

    try:
        webbrowser.open(url)
    except Exception:  # noqa: BLE001 - headless machines
        pass


# ---- parser -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="shotsieve", description="Local AI photo culling")
    parser.add_argument("--version", action="store_true", help="print version")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    p = sub.add_parser("models", help="download / verify model files")
    msub = p.add_subparsers(dest="models_cmd", required=True, metavar="ACTION")
    d = msub.add_parser("download", help="fetch the core models into ./models")
    d.add_argument("--extra", default="", help="extra model groups, comma-separated (search)")
    msub.add_parser("check", help="verify presence + sha256")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("ingest", help="copy a card to disk with verification")
    p.add_argument("src", type=Path)
    p.add_argument("dest", type=Path)
    p.add_argument("--backup", type=Path, help="second destination written in parallel")
    p.add_argument("--by-date", action="store_true", help="flatten into YYYY-MM-DD/ folders")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("cull", help="cull a folder of photos")
    p.add_argument("folder", type=_dir_arg)
    p.add_argument(
        "--write-xmp", action="store_true",
        help="write XMP sidecars next to the files (originals are never modified). An existing "
        "rating/label of yours is kept unless you overrode the image in ShotSieve. NOTE: "
        "Lightroom Classic reads sidecars only for proprietary RAW; for JPEG/DNG it reads the "
        "embedded metadata and will not see them (Bridge/Capture One/darktable do)",
    )  # fmt: skip
    p.add_argument(
        "--force-ratings", action="store_true",
        help="with --write-xmp: also replace ratings/labels you already set",
    )  # fmt: skip
    p.add_argument(
        "--full", action="store_true",
        help="re-decide the whole shoot (a re-run normally re-decides only what changed and "
        "keeps the shoot norm while <= 20 %% of the frames changed)",
    )  # fmt: skip
    p.add_argument(
        "--redraw-audit", action="store_true",
        help="draw a new reject spot-check sample (frozen per shoot otherwise)",
    )  # fmt: skip
    p.add_argument(
        "--video", action="store_true",
        help="also cull frame candidates of .mp4/.mov/.m4v files (virtual `clip.mp4#t=12.3` "
        "items; opt-in, runs after the photos)",
    )  # fmt: skip
    p.add_argument(
        "--export-keeps", action="store_true",
        help="with --video: export the kept video frames to <folder>/frame_grabs/ as JPEGs "
        "(q95, EXIF time = video creation + t); implies --video",
    )  # fmt: skip
    p.add_argument("--json", metavar="OUT.json", help="write full results as JSON")
    p.add_argument("--config", metavar="CFG.toml", help="settings file")
    p.set_defaults(func=cmd_cull)

    p = sub.add_parser("clock", help="find / set the clock offset between camera bodies")
    p.add_argument("folder", type=_dir_arg)
    p.add_argument("--apply", action="store_true", help="store the suggested offsets")
    p.add_argument(
        "--set", action="append", metavar="SERIAL=+3725",
        help="manual offset: seconds or +h:mm:ss (repeatable; wins over auto)",
    )  # fmt: skip
    p.add_argument("--write-xmp-time", action="store_true",
                   help="write the corrected capture time to sidecars (photoshop:DateCreated)")
    p.set_defaults(func=cmd_clock)

    p = sub.add_parser("index", help="build the natural-language search index (SigLIP2)")
    p.add_argument("folder", type=_dir_arg)
    p.add_argument("--no-people", action="store_true", help="skip face clustering")
    p.add_argument("--ocr", action="store_true",
                   help="also read jersey numbers (opt-in; needs `models download --extra ocr`)")
    p.add_argument("--no-search", action="store_true", help="skip the SigLIP search index")
    p.add_argument("--config", metavar="CFG.toml", help="settings file")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("roster", help="map jersey numbers to names from a number,name[,team] CSV")
    p.add_argument("folder", type=_dir_arg)
    p.add_argument("csv", nargs="?", help="roster CSV (omit to just show the current mapping)")
    p.add_argument("--team", default=None,
                   help="this folder's team: settles numbers that two teams share")
    p.add_argument("--write-xmp", action="store_true",
                   help="merge #number and names into sidecars (dc:subject, PersonInImage)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_roster)

    p = sub.add_parser("search", help='search a shoot, e.g. "bride laughing at sunset"')
    p.add_argument("folder", type=_dir_arg)
    p.add_argument("query", help='text plus filters: person:"Ana" is:keep camera:SN1 jersey:23')
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--json", action="store_true")
    p.add_argument("--config", metavar="CFG.toml", help="settings file")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("shotlist", help="which shots on your list are covered by the cull?")
    p.add_argument("folder", type=_dir_arg)
    p.add_argument("--list", required=True, metavar="LIST",
                   help="a list file (one item per line: [!]text [person:\"Name\"] [xN]) or "
                        "a template name: wedding, sports, corporate")  # fmt: skip
    p.add_argument("--json", action="store_true")
    p.add_argument("--tag-xmp", action="store_true",
                   help="opt-in: add shotlist:<item> keywords to the matched keeps' sidecars")
    p.add_argument("--config", metavar="CFG.toml", help="settings file")
    p.set_defaults(func=cmd_shotlist)

    p = sub.add_parser("train", help="train the personal model from your past choices")
    p.add_argument("--lrcat", metavar="CAT.lrcat", help="Lightroom catalog")
    p.add_argument("--xmp-folder", action="append", type=_dir_arg, metavar="DIR",
                   help="folder of XMP sidecars (repeatable)")  # fmt: skip
    p.add_argument("--from-overrides", action="append", type=_dir_arg, metavar="FOLDER",
                   help="shoot folder whose UI overrides are used (repeatable)")  # fmt: skip
    p.add_argument(
        "--out", metavar="MODEL.joblib", help="output (default data/personal_model.joblib)"
    )
    p.add_argument("--config", metavar="CFG.toml", help="settings file")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser(
        "sync",
        help="read your picks / stars / labels from a Lightroom Classic catalog (read-only) and "
        "keep them as overrides, so ShotSieve learns from what you finally chose",
    )
    p.add_argument("folder", type=_dir_arg, help="a culled shoot folder")
    p.add_argument("--lrcat", required=True, metavar="CAT.lrcat", help="Lightroom catalog")
    p.add_argument("--dry-run", action="store_true", help="show what would change, change nothing")
    p.add_argument("--json", action="store_true")
    p.add_argument("--no-report", action="store_true", help="skip the 'this shoot' report")
    p.add_argument("--config", metavar="CFG.toml", help="settings file")
    p.set_defaults(func=cmd_sync)

    p = sub.add_parser("edit", help="learn your editing style, predict develop settings")
    esub = p.add_subparsers(dest="edit_cmd", required=True, metavar="ACTION")
    e = esub.add_parser("train", help="learn from a Lightroom catalog / XMP folders (>= 200 edits)")
    e.add_argument("--catalog", metavar="CAT.lrcat", help="Lightroom catalog")
    e.add_argument("--xmp", action="append", type=_dir_arg, metavar="DIR",
                   help="folder of XMP sidecars with develop settings (repeatable)")  # fmt: skip
    e.add_argument("--dng-render", action="store_true",
                   help="render DNGs whose preview has an edit baked in")  # fmt: skip
    e.add_argument("--json", action="store_true")
    e.add_argument("--config", metavar="CFG.toml", help="settings file")
    e = esub.add_parser("apply", help="write predicted crs: develop settings for a culled shoot")
    e.add_argument("folder", type=_dir_arg)
    e.add_argument("--scope", choices=["keeps", "all"], default="keeps")
    e.add_argument("--dry-run", action="store_true", help="show what would be written")
    e.add_argument("--embed-copies", nargs="?", const="", default=None, metavar="DIR",
                   help="JPEG/DNG: write embedded-XMP copies into DIR "
                        "(default <shoot>/shotsieve_edits/); originals never change")  # fmt: skip
    e.add_argument("--catalog", metavar="CAT.lrcat",
                   help="skip images the catalog already has develop settings for")  # fmt: skip
    e.add_argument("--json", action="store_true")
    e.add_argument("--config", metavar="CFG.toml", help="settings file")
    e = esub.add_parser("status", help="the trained profile: per-slider mode and error")
    e.add_argument("--json", action="store_true")
    e.add_argument("--config", metavar="CFG.toml", help="settings file")
    p.set_defaults(func=cmd_edit)

    p = sub.add_parser(
        "match", help="copy the edit of a reference frame to look-alike frames (exposure + WB)"
    )
    p.add_argument("folder", type=_dir_arg)
    p.add_argument("--ref", required=True, metavar="IMG",
                   help="the edited reference (sidecar, XMP, --catalog)")  # fmt: skip
    p.add_argument("--targets", default="keeps", metavar="SPEC",
                   help="keeps (default) | group | camera:MODEL")  # fmt: skip
    p.add_argument("--dry-run", action="store_true", help="show what would be written")
    p.add_argument("--embed-copies", nargs="?", const="", default=None, metavar="DIR",
                   help="JPEG/DNG: write embedded-XMP copies into DIR "
                        "(default <shoot>/shotsieve_edits/); originals never change")  # fmt: skip
    p.add_argument("--catalog", metavar="CAT.lrcat",
                   help="reference settings from / edits respected in this catalog")  # fmt: skip
    p.add_argument("--json", action="store_true")
    p.add_argument("--config", metavar="CFG.toml", help="settings file")
    p.set_defaults(func=cmd_match)

    from shotsieve import cli_crop

    cli_crop.add_parser(sub)  # T22: shotsieve crop detect|suggest|apply|reset|status|history

    p = sub.add_parser("eval", help="missed-keeper rate against your own ratings")
    p.add_argument("folder", type=_dir_arg)
    p.add_argument("--truth", choices=["xmp", "lrcat"], required=True)
    p.add_argument("--lrcat", metavar="CAT.lrcat")
    p.add_argument("--gate", type=float, metavar="RATE",
                   help="exit 3 if group missed-keeper rate (review=saved) > RATE")  # fmt: skip
    p.add_argument("--config", metavar="CFG.toml", help="settings file")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("doctor", help="check that the install is complete and healthy")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("app", help="open ShotSieve in its own Mac window (what ShotSieve.app runs)")
    p.add_argument("folder", nargs="?", type=_dir_arg)
    p.set_defaults(func=cmd_app)

    p = sub.add_parser("serve", help="open the review UI (127.0.0.1 only)")
    p.add_argument("folder", nargs="?", type=_dir_arg)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-open", action="store_true", help="do not open a browser")
    p.add_argument("--config", metavar="CFG.toml", help="settings file")
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        print(__version__)
        return 0
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - surface any failure as a non-zero exit
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
