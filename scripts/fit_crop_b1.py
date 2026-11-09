"""Fit/record the B1 (Review #12) thresholds on the TRAIN split only.

Runs the frozen harness (`tests/crop_eval.py`) over the train images for a grid of
(TAU_EXT, GROW_CAP, PT_SEED_MULT, PT_MAX_AREA) values (module attributes of
`shotsieve.models.patch_features`) and prints the verdicts of each setting. It never reads the
held-out splits (`tests/fixtures/crop` held-out names, `heldout_result.json`).

    uv run python scripts/fit_crop_b1.py                 # the recorded values
    uv run python scripts/fit_crop_b1.py TAU_EXT=0.55,0.6 GROW_CAP=1.5,2.0
"""

from __future__ import annotations

import itertools
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests import crop_eval as ce  # noqa: E402

from shotsieve.config import Settings  # noqa: E402
from shotsieve.models import patch_features as pf  # noqa: E402


def train_names() -> list[str]:
    m = json.loads((ce.CROP / "manifest.json").read_text())
    return sorted(n for n, v in m.items() if v.get("split") == "train" and "copy_of" not in v)


def run(params: dict[str, float]) -> dict:
    for k, v in params.items():
        setattr(pf, k, v)
    labels = ce.load_labels()
    names = train_names()
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "shoot"
        f.mkdir()
        for n in names:
            shutil.copy(ce.CROP / "real" / n, f / n)
        env = ce.Env(f, names, Settings())
        try:
            st = ce.evaluate(env, names, labels)
        finally:
            env.close()
    return st


def main() -> None:
    grid = {k: [float(x) for x in v.split(",")]
            for k, v in (a.split("=") for a in sys.argv[1:])}
    keys = list(grid)
    for combo in itertools.product(*grid.values()) if grid else [()]:
        params = dict(zip(keys, combo, strict=True))
        st = run(params)
        v = ce.verdicts(st)
        print(json.dumps({
            "params": params,
            "silent_miss": st["silent_miss"], "flagged_miss": st["flagged_miss"],
            "containment": round(st["containment"], 3),
            "no_cut_without_warning": round(st["no_cut_without_warning"], 3),
            "cut": f"{st['cut_flagged']}/{st['cut_pairs']}",
            "stranger": f"{st['stranger_cut_top1']}/{st['alt_free_pairs']}",
            "point": f"{st['point_ok']}/{st['point_total']}",
            "ref_gap": round(st["ref_gap"], 3), "contain_gain": round(st["contain_gain"], 3),
            "fail": [k for k, ok in v.items() if not ok], "fails": st["fails"][:10],
        }), flush=True)


if __name__ == "__main__":
    main()
