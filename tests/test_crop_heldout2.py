"""T22 acceptance held-out #2 (`tests/fixtures/crop2`, independent labeller, scored ONCE).

The result was recorded by `scripts/score_crop_heldout2.py` in
`tests/fixtures/crop2_heldout_result.json` with the frozen harness `tests/crop_eval.py`. This set is
now BURNED: no retuning on it. These tests verify both shas and compare the RECORDED numbers with
the Review #12 targets. Targets MISSED on the recorded run are listed in `RECORDED_MISSES` and
pinned as floors (a recorded miss, not a pass); they must not silently get worse."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tests import crop_eval as ce

pytestmark = pytest.mark.models

ROOT = Path(__file__).resolve().parents[1]
CROP2 = ROOT / "tests" / "fixtures" / "crop2"
RESULT = ROOT / "tests" / "fixtures" / "crop2_heldout_result.json"
HARNESS_SHA = "da83cde8a95ae2f8cefde874dc85edd941d4cdbc4dfac6d403289c56b0cfa2ed"
LABELS_SHA = "a5e3c801450be6c50f8d67a00dc44f0e441532d634c814710847e0f116abc4a5"

# RECORDED MISS (held-out #2, scored once 2026-11-08): metric -> (op, recorded floor/ceiling)
RECORDED_MISSES = {
    "no_cut_without_warning": (">=", 78 / 79),  # target 1.0: one SILENT cut (small_01 1:1)
    "stranger_cut_rate": ("<=", 5 / 46),  # target <= 0.10 with n = 46 pairs: 5 cut
    "point_rate": (">=", 3 / 7),  # target >= 0.70: 3 of 7 objects
}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _result() -> dict:
    if not RESULT.exists():
        pytest.skip("crop2 held-out not scored yet")
    return json.loads(RESULT.read_text())


def test_labels_and_harness_are_frozen():
    assert _sha(CROP2 / "crop_labels.json") == LABELS_SHA
    assert (CROP2 / "crop_labels.sha256").read_text().split()[0] == LABELS_SHA
    assert _sha(ROOT / "tests" / "crop_eval.py") == HARNESS_SHA


def test_recorded_result_belongs_to_these_labels_and_harness():
    r = _result()
    assert r["labels_sha256"] == LABELS_SHA
    assert r["harness_sha256"] == HARNESS_SHA
    assert r["n_images"] == 25
    for key in ("date", "code_fingerprint", "metrics", "verdicts"):
        assert key in r


def test_recorded_result_vs_targets():
    m = _result()["metrics"]
    ver = ce.verdicts(m)
    if m["stranger_pairs"] < 20:
        ver["stranger_cut_rate"] = True  # report only below 20 qualifying pairs
    assert ver == _result()["verdicts"] or m["stranger_pairs"] < 20
    missed = {k for k, ok in ver.items() if not ok}
    # every target not in RECORDED_MISSES must hold; the misses are pinned below, not forgiven
    assert missed == set(RECORDED_MISSES), f"targets missed: {sorted(missed)}"
    assert m["face_cut"] == 0
    assert m["silent_miss"] == 1 and m["containment"] >= 0.95  # silent misses only


def test_recorded_misses_are_pinned_as_floors():
    m = _result()["metrics"]
    for k, (op, v) in RECORDED_MISSES.items():
        assert (m[k] >= v - 1e-9) if op == ">=" else (m[k] <= v + 1e-9), (k, m[k], v)
