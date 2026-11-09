# ruff: noqa: E501
"""Fit the T17 shot-list cut TAU (max F1 on "an item with a relevant frame present").

Builds the 13-subject search corpus (5 variants each, as tests/test_search_real.py), indexes it
with the real SigLIP2 models, scores the 30 labelled queries (present) of
tests/fixtures/search_queries.json and the 20 absent queries of tests/fixtures/shotlist_absent.json,
and prints the best TAU plus the metrics. Paste the value into `pipeline/shotlist.py: TAU`.

    uv run python scripts/fit_shotlist.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))


def variants(img: np.ndarray) -> list[np.ndarray]:
    h, w = img.shape[:2]
    return [
        img, img[:, ::-1], img[int(h * 0.1) : int(h * 0.9), int(w * 0.1) : int(w * 0.9)],
        cv2.convertScaleAbs(img, alpha=0.8, beta=-10), img[:, : int(w * 0.75)],
    ]  # fmt: skip


def build_corpus(folder: Path, search_images: dict[str, np.ndarray]) -> None:
    for name, img in search_images.items():
        for i, v in enumerate(variants(img)):
            cv2.imwrite(str(folder / f"{name}__{i}.jpg"),
                        cv2.cvtColor(np.ascontiguousarray(v), cv2.COLOR_RGB2BGR))  # fmt: skip


def query_stats(mat: np.ndarray, queries: list[str]) -> list[tuple[float, float]]:
    from shotsieve.models.text_image import get_text
    from shotsieve.pipeline.shotlist import top_stats

    return top_stats(mat, get_text().embed, queries)


def main() -> int:
    from conftest import _load_make_fixtures  # tests/conftest.py
    from shotsieve.config import Settings
    from shotsieve.pipeline.cache import Cache
    from shotsieve.pipeline.indexer import Indexer
    from shotsieve.pipeline.shotlist import Z_MIN, fit_tau, loo_scores

    mk = _load_make_fixtures()
    images = {
        p.stem: cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        for p in sorted(mk.fetch_search_images().glob("*.jpg"))
    }
    queries = json.loads((ROOT / "tests/fixtures/search_queries.json").read_text())["queries"]
    absent = json.loads((ROOT / "tests/fixtures/shotlist_absent.json").read_text())["absent"]
    with tempfile.TemporaryDirectory() as td:
        folder = Path(td)
        build_corpus(folder, images)
        ix = Indexer(folder, Settings(), people=False)
        ix.run_blocking()
        with Cache.for_folder(folder) as c:
            _, mat = c.siglip_matrix()
        present_s = query_stats(mat, [q["q"] for q in queries])
        absent_s = query_stats(mat, absent)
    if "-v" in sys.argv:
        for name, rows, qs in (("present", present_s, [q["q"] for q in queries]), ("absent", absent_s, absent)):
            for q, (s_, z_) in sorted(zip(qs, rows, strict=True), key=lambda t: t[1][0]):
                print(f"  {name:8s} s={s_:.4f} z={z_:5.2f}  {q}")
    # the burned first held-out set (Review #6 M7) is now train data too (never the blind set)
    football = sorted((ROOT / "tests/fixtures/real").glob("cc0_football_*.jpg"))
    ho = json.loads((ROOT / "tests/fixtures/shotlist_heldout.json").read_text())
    with tempfile.TemporaryDirectory() as td:
        folder = Path(td)
        build_corpus(folder, images)
        build_corpus(folder, {p.stem: cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in football})
        Indexer(folder, Settings(), people=False).run_blocking()
        with Cache.for_folder(folder) as c:
            _, mat = c.siglip_matrix()
        ho_p, ho_a = query_stats(mat, ho["present"]), query_stats(mat, ho["absent"])
    if "-v" in sys.argv:
        for name, rows, qs in (("burned-p", ho_p, ho["present"]), ("burned-a", ho_a, ho["absent"])):
            for q, (s_, z_) in sorted(zip(qs, rows, strict=True), key=lambda t: t[1][0]):
                print(f"  {name:8s} s={s_:.4f} z={z_:5.2f}  {q}")
    from shotsieve.pipeline.shotlist import fit_tau_conservative

    for label, pp, aa in (("tau-fit 50", present_s, absent_s), ("pooled 74", [*present_s, *ho_p], [*absent_s, *ho_a])):
        a, t, sp = fit_tau_conservative(pp, aa, Z_MIN)
        print(f"conservative ({label}): train accuracy {a:.3f}, plateau {sp[0]:.4f}..{sp[1]:.4f}, TAU = {t:.4f}")
    from shotsieve.pipeline.shotlist import MIN_COVERED_RECALL, TAU, TAU_HI, fit_tau_hi

    pp, aa = [*present_s, *ho_p], [*absent_s, *ho_a]
    rec, hi = fit_tau_hi(pp, TAU, Z_MIN)
    print(f"N1 band (pooled 74, covered recall floor {MIN_COVERED_RECALL}): TAU_HI = {hi:.4f} (strict recall {rec:.3f})")
    for name, t_hi in (("fitted", hi), ("committed", TAU_HI)):
        cov_p = sum(s >= t_hi and z >= Z_MIN for s, z in pp)
        unc_p = sum(TAU <= s < t_hi and z >= Z_MIN for s, z in pp)
        cov_a = sum(s >= t_hi and z >= Z_MIN for s, z in aa)
        unc_a = sum(TAU <= s < t_hi and z >= Z_MIN for s, z in aa)
        n = len(pp) + len(aa)
        print(f"  [{name} TAU_HI {t_hi:.4f}] train accuracy (covered=present) {(cov_p + len(aa) - cov_a) / n:.3f}; "
              f"present: covered {cov_p / len(pp):.3f}, uncertain {unc_p / len(pp):.3f}, "
              f"covered+uncertain {(cov_p + unc_p) / len(pp):.3f}; absent false-covered {cov_a / len(aa):.3f}, "
              f"absent uncertain {unc_a / len(aa):.3f}; uncertain rate {(unc_p + unc_a) / n:.3f}")
    f1, tau, span = fit_tau(present_s, absent_s, Z_MIN)
    loo_acc, loo_fc = loo_scores(present_s, absent_s, Z_MIN)
    print(f"leave-one-out accuracy {loo_acc:.3f}, false-covered {loo_fc:.3f}")
    print(f"present top s: min {min(s for s, _ in present_s):.4f}  median {np.median([s for s, _ in present_s]):.4f}")
    print(f"absent  top s: max {max(s for s, _ in absent_s):.4f}  median {np.median([s for s, _ in absent_s]):.4f}")
    print(f"present top z: min {min(z for _, z in present_s):.2f}  absent top z: max {max(z for _, z in absent_s):.2f}")
    print(f"best F1 {f1:.3f}; TAU plateau {span[0]:.4f}..{span[1]:.4f}; chosen TAU = {tau:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
