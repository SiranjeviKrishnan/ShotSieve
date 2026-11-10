"""Fetch every model that ships in the release app into ./models (or SHOTSIEVE_MODELS_DIR).

Each file comes from its pinned URL and is checked against the sha256 recorded in
`src/shotsieve/models/registry.py` (a mismatch deletes the file and fails). Already-present,
verified files are kept.

    uv run python scripts/fetch_models.py             # every group below
    uv run python scripts/fetch_models.py --core      # just the 131 MB core group
"""

from __future__ import annotations

import argparse
import sys

from shotsieve.models.registry import MODEL_SPECS, ensure_models, group_names, models_dir

# "search-int8" is a rejected alternative
ALL_GROUPS = ("core", "rawdenoise", "search", "ocr", "crop", "denoise")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--core", action="store_true", help="only the core group")
    args = ap.parse_args()
    groups = ("core",) if args.core else ALL_GROUPS
    last = {"name": None}

    def show(ev: dict) -> None:
        if ev["name"] != last["name"]:
            last["name"] = ev["name"]
            print(f"  {ev['name']}: {ev['file_total'] / 1e6:.0f} MB", flush=True)

    try:
        paths = ensure_models(groups=groups, progress=show)
    except Exception as exc:  # noqa: BLE001
        print(f"fetch failed: {exc}", file=sys.stderr)
        return 1
    total = sum(p.stat().st_size for p in paths.values())
    print(f"{len(paths)} models verified (sha256) in {models_dir()}, {total / 1e6:.0f} MB")
    assert set(paths) == set(group_names(*groups)) <= set(MODEL_SPECS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
