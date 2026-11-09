"""Score the independent T22 crop held-out set (tests/fixtures/crop2) EXACTLY ONCE.

Refuses to run if tests/fixtures/crop2_heldout_result.json already exists. Verifies the label and
harness sha256 first, calls the frozen `tests/crop_eval.py` unchanged, and records everything."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests import crop_eval as ce  # noqa: E402

from shotsieve.config import Settings  # noqa: E402

CROP2 = ROOT / "tests" / "fixtures" / "crop2"
RESULT = ROOT / "tests" / "fixtures" / "crop2_heldout_result.json"
HARNESS_SHA = "da83cde8a95ae2f8cefde874dc85edd941d4cdbc4dfac6d403289c56b0cfa2ed"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def code_fingerprint() -> str:
    src = ROOT / "src" / "shotsieve"
    files = sorted((src / "pipeline").glob("crop*.py")) + [
        src / "models" / "patch_features.py", src / "models" / "object_detect.py"]
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode() + b"\0" + f.read_bytes())
    return h.hexdigest()


def main() -> None:
    if RESULT.exists():
        sys.exit("held-out #2 already scored: refusing to score again")
    lab_sha = sha(CROP2 / "crop_labels.json")
    assert lab_sha == (CROP2 / "crop_labels.sha256").read_text().split()[0], "labels changed"
    assert sha(ROOT / "tests" / "crop_eval.py") == HARNESS_SHA, "harness changed"
    labels = json.loads((CROP2 / "crop_labels.json").read_text())
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "shoot"
        f.mkdir()
        for n in labels:
            shutil.copy(CROP2 / "real" / n, f / n)
        env = ce.Env(f, sorted(labels), Settings())
        try:
            st = ce.evaluate(env, sorted(labels), labels)
        finally:
            env.close()
    ver = ce.verdicts(st)
    out = {
        "date": dt.date.today().isoformat(), "labels_sha256": lab_sha,
        "harness_sha256": HARNESS_SHA,
        "code_fingerprint": code_fingerprint(), "n_images": len(labels),
        "metrics": st, "verdicts": ver,
        "stranger_counts_only_if_pairs_ge_20": st["stranger_pairs"] >= 20,
    }
    RESULT.write_text(json.dumps(out, indent=1, default=float))
    skip = ("fails", "ref_iou_top1", "ref_iou_centre")
    print(json.dumps({k: v for k, v in st.items() if k not in skip}, indent=1, default=float))
    print(ver)
    print(*st["fails"], sep="\n")


if __name__ == "__main__":
    main()
