"""`shotsieve crop detect|suggest|apply|reset|status|history` (T22, docs/11_CROP.md section 8).

Exit codes: 0 ok, 2 bad arguments / ratio, 3 nothing to do. `suggest` writes nothing; an explicit
`apply` is the user's acceptance (top-1 or `--rect` stored as a version, XMP written)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from shotsieve.config import Settings
from shotsieve.io.formats import split_virtual
from shotsieve.pipeline import crop as pcrop
from shotsieve.pipeline.cache import Cache, cache_path


def _err(msg: str, code: int = 2) -> int:
    print(f"error: {msg}", file=sys.stderr)
    return code


def _scope(
    args: argparse.Namespace, cache: Cache, *, allow_all: bool = False
) -> list[tuple[int, str]]:
    """(image id, path) of the images to work on: --ids, or every kept image."""
    dec = cache.load_decisions()
    if not dec:
        raise ValueError("this folder has not been culled yet")
    paths = [p for p in sorted(dec) if not split_virtual(p)]
    ids = cache.ids_lookup(paths)
    rows = [(ids[p], p) for p in paths if p in ids]
    if getattr(args, "ids", None):
        want = set(args.ids)
        return [r for r in rows if r[0] in want]
    if allow_all and getattr(args, "all", False):
        have = cache.crops_current_all()
        return [r for r in rows if r[0] in have]
    ovr = cache.overrides()
    keep = []
    for i, p in rows:
        final = ovr[p]["decision"] if p in ovr else dec[p]["decision"]
        if final == "keep":
            keep.append((i, p))
    return keep


def _parse_ids(text: str) -> list[int]:
    try:
        return [int(t) for t in text.split(",") if t.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"bad id list {text!r}") from exc


def _parse_box(text: str) -> list[float]:
    try:
        v = [float(t) for t in text.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"bad box {text!r}: want x0,y0,x1,y1") from None
    if len(v) != 4:
        raise argparse.ArgumentTypeError(f"bad box {text!r}: want x0,y0,x1,y1")
    return v


def cmd_crop(args: argparse.Namespace) -> int:
    from shotsieve.cli import ProgressBar, _check_models, _settings

    settings: Settings = _settings(args)
    folder = Path(args.folder).expanduser().resolve()
    if not cache_path(folder).exists():
        return _err("this folder has not been culled yet", 3)
    action = args.crop_cmd
    emb = getattr(args, "embed_copies", None)
    if action in ("detect", "suggest", "apply"):
        _check_models()
    if action in ("suggest", "apply"):
        try:
            from shotsieve.pipeline import crop_geom

            crop_geom.parse_ratio(args.ratio)
        except ValueError as exc:
            return _err(str(exc))
    boxes = getattr(args, "box", None) or []
    if boxes:
        try:
            pcrop.check_boxes(boxes)
        except pcrop.CropError as exc:
            return _err(str(exc))
    if action == "apply" and bool(args.rect) != bool(args.id):
        return _err("--rect and --id go together")
    out: dict[str, Any] = {}
    with Cache.for_folder(folder) as c:
        try:
            targets = (
                _scope(args, c, allow_all=action == "reset")
                if action not in ("status", "history")
                else []
            )
        except ValueError as exc:
            return _err(str(exc), 3)
        if action == "history":
            path = c.path_for_id(args.id)
            if path is None:
                return _err(f"unknown image id {args.id}", 3)
            hist = c.crop_history(args.id)
            if args.json:
                print(json.dumps([pcrop._hist_json(h) for h in hist], indent=2))
            elif not hist:
                print("no crop versions")
            for h in hist:
                rect = (
                    "full frame" if h["rect"] is None else ",".join(f"{v:.4f}" for v in h["rect"])
                )
                print(f"v{h['version']}  {h['source']:<11s} {h['ratio'] or '-':<9s} {rect}")
            return 0 if hist else 3
        if action == "status":
            have = c.crops_current_all()
            ids = list(have)
            rows = []
            for i in sorted(ids):
                p = c.path_for_id(i)
                cur = c.crop_current(i)
                rows.append({"id": i, "name": Path(p).name if p else "", "rect": have[i],
                             "ratio": cur["ratio"] if cur else None,
                             "source": cur["source"] if cur else None,
                             "write": cur["write_status"] if cur else None})  # fmt: skip
            if args.json:
                print(json.dumps({"cropped": rows, "n": len(rows)}, indent=2))
            else:
                print(f"{len(rows)} image(s) with a crop")
                for r in rows:
                    print(
                        f"{r['id']:>6d}  {r['name']:<28s} {r['ratio'] or '-':<8s} "
                        f"{r['source']}  {r['write']}"
                    )
            return 0
        if not targets:
            return _err("no images to work on", 3)
        bar = ProgressBar()
        t0 = time.perf_counter()
        results: list[dict[str, Any]] = []
        n_skipped = 0
        if action == "detect":  # pipelined workers: >= 6 img/s cold
            for (i, s), (_, p) in zip(
                pcrop.detect_many(folder, c, targets, settings, bar), targets, strict=True
            ):
                if s is None:
                    results.append({"id": i, "name": Path(p).name, "error": "cannot decode"})
                else:
                    results.append({"id": i, "name": Path(p).name, "n_subjects": len(s.subjects),
                                    "detector": s.detector})  # fmt: skip
            targets = []
        for k, (i, p) in enumerate(targets):
            bar("crop", k, len(targets))
            try:
                if action == "detect":
                    s = pcrop.detect_subjects(folder, c, i, p, settings)
                    results.append({"id": i, "name": Path(p).name, "n_subjects": len(s.subjects),
                                    "detector": s.detector})  # fmt: skip
                elif action == "suggest":
                    r = pcrop.suggest_crops(folder, c, i, p, settings, ratio=args.ratio,
                                            flip=args.flip, n=args.n, log_event=False,
                                            boxes=boxes)  # fmt: skip
                    results.append({"id": i, "name": Path(p).name, **r})
                elif action == "apply":
                    if args.rect:
                        if i != args.id:
                            continue
                        try:
                            rect = [float(v) for v in args.rect.split(",")]
                        except ValueError:
                            return _err("--rect must be x0,y0,x1,y1 (numbers)")
                        source, alt, ratio_txt = "manual", None, args.ratio
                    else:
                        r = pcrop.suggest_crops(folder, c, i, p, settings, ratio=args.ratio,
                                                flip=args.flip, n=1, log_event=False,
                                                boxes=boxes)  # fmt: skip
                        if not r["candidates"]:
                            n_skipped += 1
                            continue
                        rect, source, alt, ratio_txt = (
                            r["candidates"][0]["rect"],
                            "ai",
                            0,
                            r["ratio"],
                        )
                    if args.dry_run:
                        results.append({"id": i, "name": Path(p).name, "rect": rect,
                                        "ratio": ratio_txt, "dry_run": True})  # fmt: skip
                    else:
                        a = pcrop.accept(folder, c, i, p, settings, rect, ratio_txt, source=source,
                                         alt_rank=alt, write=args.write, embed_dir=emb,
                                         catalog=args.catalog)  # fmt: skip
                        results.append({"id": i, "name": Path(p).name, **a})
                elif action == "reset":
                    cur = c.crop_current(i)
                    if cur is None or cur["rect"] is None:
                        n_skipped += 1
                        continue
                    a = pcrop.reset(folder, c, i, p, settings, write=args.write, embed_dir=emb,
                                    catalog=args.catalog)  # fmt: skip
                    results.append({"id": i, "name": Path(p).name, **a})
            except pcrop.CropError as exc:
                results.append({"id": i, "name": Path(p).name, "error": str(exc)})
        bar("crop", len(targets), len(targets))
        bar.close()
        dt = time.perf_counter() - t0
    out = {"action": action, "n": len(results), "skipped": n_skipped, "seconds": round(dt, 2),
           "results": results}  # fmt: skip
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        _print_human(action, out, args)
    if not results:
        return 3
    return 0


def _print_human(action: str, out: dict[str, Any], args: argparse.Namespace) -> None:
    res = out["results"]
    if action == "detect":
        print(f"detected subjects in {out['n']} image(s) in {out['seconds']} s")
        return
    if action == "suggest":
        for r in res:
            if "error" in r:
                print(f"{r['name']}: error: {r['error']}")
                continue
            for k, c in enumerate(r["candidates"], 1):
                warn = f"  [{', '.join(c['warnings'])}]" if c["warnings"] else ""
                print(f"{r['name']}  #{k}  {r['ratio']}  score {c['score']:.3f}  "
                      f"rect {','.join(f'{v:.4f}' for v in c['rect'])}{warn}")  # fmt: skip
        return
    verb = {"apply": "would crop" if getattr(args, "dry_run", False) else "cropped",
            "reset": "reset"}[action]  # fmt: skip
    print(f"{verb} {out['n']} image(s); {out['skipped']} skipped")
    for r in res:
        if "error" in r:
            print(f"  {r['name']}: error: {r['error']}")
        elif "write" in r:
            w = r["write"]
            print(f"  {r['name']}: v{r['version']}  xmp {w['status']}"
                  + (f" ({w['reason']})" if w.get("reason") else ""))  # fmt: skip


def add_parser(sub: Any) -> None:
    p = sub.add_parser("crop", help="subject-aware crop suggestions (non-destructive)")
    csub = p.add_subparsers(dest="crop_cmd", required=True, metavar="ACTION")

    def common(e: argparse.ArgumentParser, ids: bool = True) -> None:
        from shotsieve.cli import _dir_arg

        e.add_argument("folder", type=_dir_arg)
        if ids:
            e.add_argument(
                "--ids", type=_parse_ids, metavar="N,N", help="image ids (default: keeps)"
            )
            e.add_argument("--keeps", action="store_true", help="the kept images (default)")
        e.add_argument("--json", action="store_true")
        e.add_argument("--config", metavar="CFG.toml", help="settings file")

    e = csub.add_parser("detect", help="fill the subject cache (faces, people, saliency)")
    common(e)
    e = csub.add_parser("suggest", help="print crop candidates (writes nothing)")
    common(e)
    e.add_argument("--ratio", required=True, metavar="W:H", help="4:5, 3:2, 1.91:1, original, free")
    e.add_argument("--flip", action="store_true", help="swap W:H")
    e.add_argument("--n", type=int, default=3, help="candidates per image (default 3)")
    e.add_argument("--box", type=_parse_box, action="append", metavar="x0,y0,x1,y1",
                   help="a must-keep subject box (display-normalised 0..1; repeat, max 6)")
    e = csub.add_parser("apply", help="store the top suggestion (or --rect) as the crop")
    common(e)
    e.add_argument("--ratio", required=True, metavar="W:H")
    e.add_argument("--flip", action="store_true")
    e.add_argument("--box", type=_parse_box, action="append", metavar="x0,y0,x1,y1",
                   help="a must-keep subject box for the suggestion (repeat, max 6)")
    e.add_argument(
        "--rect", metavar="x0,y0,x1,y1", help="explicit display-normalised rect (with --id)"
    )
    e.add_argument("--id", type=int, help="image id for --rect")
    e.add_argument("--write", choices=["auto", "none", "embed"], default="auto")
    e.add_argument("--embed-copies", nargs="?", const=None, default=None, metavar="DIR",
                   help="JPEG/DNG: write embedded-XMP copies into DIR "
                        "(default <shoot>/shotsieve_edits/); originals never change")  # fmt: skip
    e.add_argument("--catalog", metavar="CAT.lrcat",
                   help="skip images where the catalog already has a crop / edits")  # fmt: skip
    e.add_argument("--dry-run", action="store_true", help="show what would be written")
    e = csub.add_parser("reset", help="back to the full frame (a new version)")
    common(e)
    e.add_argument("--all", action="store_true", help="every image that has a crop")
    e.add_argument("--write", choices=["auto", "none", "embed"], default="auto")
    e.add_argument("--embed-copies", nargs="?", const=None, default=None, metavar="DIR")
    e.add_argument("--catalog", metavar="CAT.lrcat")
    e = csub.add_parser("status", help="images with a crop")
    common(e, ids=False)
    e = csub.add_parser("history", help="every crop version of one image")
    common(e, ids=False)
    e.add_argument("--id", type=int, required=True)
    p.set_defaults(func=cmd_crop)
