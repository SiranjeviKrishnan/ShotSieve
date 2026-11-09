"""Integrity only for blind set #2 (never scored here): labels.json and image hashes are frozen."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

D = Path(__file__).parent / "fixtures" / "shotlist_blind2"
LABELS_SHA256 = "dbb6838c5908aad885a5e3c87393432f48dc3cd9e85feea6617c0364f20f5f51"


def test_blind2_labels_frozen():
    raw = (D / "labels.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == LABELS_SHA256


def test_blind2_images_match_recorded_hashes():
    d = json.loads((D / "labels.json").read_text())
    assert d["frozen_at"] == "2026-11-08"
    assert len(d["present"]) == 41 and len(d["absent"]) == 50
    on_disk = {p.name for p in (D / "images").iterdir()}
    assert on_disk == set(d["images"])
    for name, meta in d["images"].items():
        got = hashlib.sha256((D / "images" / name).read_bytes()).hexdigest()
        assert got == meta["sha256"], name
    assert {p["image"] for p in d["present"]} <= set(d["images"])
