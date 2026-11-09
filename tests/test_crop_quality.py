"""T22 crop quality on the independently labelled CC0 set (docs/11_CROP.md section 11: 1, 3, 4, 5).

Labels were written by a SEPARATE agent (see TASKS.md T22); `crop_labels.json` is frozen by
`crop_labels.sha256`. The train split may be used for tuning; the held-out split is scored ONCE
(`SHOTSIEVE_CROP_HELDOUT=1`), the result is stored in `heldout_result.json` and later runs only
re-check that stored result."""

from __future__ import annotations

import hashlib
import json
import os
import shutil

import pytest

from shotsieve.config import Settings
from tests import crop_eval as ce

CROP = ce.CROP
LABELS = CROP / "crop_labels.json"
SHA = CROP / "crop_labels.sha256"
RESULT = CROP / "heldout_result.json"

pytestmark = pytest.mark.models


def _manifest():
    return json.loads((CROP / "manifest.json").read_text())


def _names(split: str) -> list[str]:
    m = _manifest()
    return sorted(n for n, v in m.items() if v.get("split") == split and "copy_of" not in v)


@pytest.fixture(scope="module")
def labels():
    if not LABELS.exists():
        pytest.skip("crop_labels.json not written yet")
    return ce.load_labels()


@pytest.fixture(scope="module")
def env(tmp_path_factory, models_ready, labels):
    f = tmp_path_factory.mktemp("cropq") / "shoot"
    f.mkdir()
    for n in labels:
        shutil.copy(CROP / "real" / n, f / n)
    e = ce.Env(f, sorted(labels), Settings())
    yield e
    e.close()


def test_fixtures_have_licences_hashes_and_the_requested_mix():
    m = _manifest()
    lic = (CROP / "LICENSES.md").read_text()
    base = {n: v for n, v in m.items() if "copy_of" not in v}
    assert len(base) >= 40
    cats = {}
    for n, v in base.items():
        cats[v["category"]] = cats.get(v["category"], 0) + 1
        data = (CROP / "real" / n).read_bytes()
        assert hashlib.sha256(data).hexdigest() == v["sha256"], n
        assert n in lic and v["sha256"] in lic and ("CC0" in lic or "Public domain" in lic)
    assert cats == {"portrait": 10, "group": 8, "animal": 6, "object": 6, "horizon": 5, "action": 5}
    assert sum(1 for v in base.values() if v["split"] == "held-out") >= 15
    assert sum(len(p) for _, _, fs in os.walk(CROP / "real") for p in fs) < 8_000_000 * 1  # <= 8 MB
    total = sum(f.stat().st_size for f in (CROP / "real").iterdir())
    assert total <= 8_000_000


def test_labels_are_frozen_and_well_formed(labels):
    assert SHA.exists(), "crop_labels.sha256 missing: the labels must be frozen before scoring"
    assert hashlib.sha256(LABELS.read_bytes()).hexdigest() == SHA.read_text().split()[0]
    from PIL import Image

    m = _manifest()
    assert set(labels) == {n for n, v in m.items() if "copy_of" not in v}
    refs = 0
    for n, lab in labels.items():
        w, h = Image.open(CROP / "real" / n).size
        assert (lab["w"], lab["h"]) == (w, h)
        for s in lab["subjects"]:
            x0, y0, x1, y1 = s["box_px"]
            assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h and s["kind"] in (
                "face", "person", "animal", "object")  # fmt: skip
        for rt, r in (lab.get("reference_px") or {}).items():
            a, b = (int(v) for v in rt.split(":"))
            W, H = r[2] - r[0], r[3] - r[1]
            assert W * b == H * a and 0 <= r[0] and 0 <= r[1] and r[2] <= w and r[3] <= h, (n, rt)
        refs += bool(lab.get("reference_px"))
    assert refs >= 20


def test_orientation_copies_display_like_their_originals(env):
    import cv2
    import numpy as np

    from shotsieve.io.preview import load_preview

    m = _manifest()
    for n, v in m.items():
        if "copy_of" not in v:
            continue
        a = load_preview(CROP / "real" / n, 1280).astype(np.float32)
        b = cv2.cvtColor(cv2.imread(str(CROP / "real" / v["copy_of"])), cv2.COLOR_BGR2RGB)
        b = b.astype(np.float32)
        assert a.shape == b.shape, (n, a.shape, b.shape)
        assert float(np.abs(a - b).mean()) < 6.0, n  # JPEG re-encoding noise only


def test_train_split_meets_the_acceptance_numbers(env, labels):
    st = ce.evaluate(env, _names("train"), labels)
    report = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in st.items()
              if k not in ("fails", "ref_iou_top1", "ref_iou_centre")}  # fmt: skip
    print("TRAIN", json.dumps(report), st["fails"][:12])
    bad = [k for k, ok in ce.verdicts(st).items() if not ok]
    assert not bad, f"{bad}: {report} {st['fails'][:12]}"


def test_burned_first_heldout_result_is_kept_as_history():
    """The first held-out split (scored once before Review #12, old harness and key names) is
    burned: its recorded result stays as history and is never re-scored or tuned on. The
    acceptance held-out is `tests/fixtures/crop2` (test_crop_heldout2.py)."""
    if not RESULT.exists():
        pytest.skip("no recorded result")
    r = json.loads(RESULT.read_text())
    assert r["labels_sha256"] == hashlib.sha256(LABELS.read_bytes()).hexdigest()
    assert r["face_cut"] == 0
