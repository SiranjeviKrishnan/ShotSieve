# ruff: noqa: E501, F811
"""T23h: POST /api/subject/{id}/score (sharpness of a user subject box on the full-resolution
pixels, cached by image + box) and the shared storage with crop mode's user boxes."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from shotsieve.features import sharpness
from shotsieve.server import subject as subj
from tests.test_crop_api import client, ids, shoot  # noqa: F401

pytestmark = pytest.mark.usefixtures("models_ready")


def score(client, shoot, iid, boxes):
    return client.post(f"/api/subject/{iid}/score", json={"folder": str(shoot), "boxes": boxes})


def test_score_is_normalised_labelled_and_cached(client, shoot, ids):
    iid = ids["landscape.jpg"]
    box = {"x0": 0.1, "y0": 0.1, "x1": 0.6, "y1": 0.7}
    r = score(client, shoot, iid, [box])
    assert r.status_code == 200, r.text
    s = r.json()["scores"][0]
    assert 0.0 <= s["sharp"] <= 1.0 and s["label"] in ("in focus", "soft")
    assert s["in_focus"] == (s["sharp"] >= subj.IN_FOCUS) and s["raw"] > 0
    n = len(subj._cache)
    again = score(client, shoot, iid, [box]).json()["scores"][0]
    assert again == s and len(subj._cache) == n  # served from the cache


def test_blurred_region_is_softer_than_the_sharp_one(client, shoot, ids, tmp_path):
    iid = ids["two.jpg"]
    sharp = {"x0": 0.0, "y0": 0.0, "x1": 0.5, "y1": 1.0}
    a = score(client, shoot, iid, [sharp]).json()["scores"][0]
    yy, xx = np.mgrid[0:400, 0:400]
    gray = (((yy // 16 + xx // 16) % 2) * 255).astype(np.uint8)  # a checkerboard: crisp edges
    blur = cv2.GaussianBlur(gray, (0, 0), 12)
    full = (0.0, 0.0, 1.0, 1.0)
    assert sharpness.region_sharpness(blur, full) < 0.05 * sharpness.region_sharpness(gray, full)
    assert a["sharp"] is not None
    ref = sharpness.region_sharpness(gray, full) / 2
    out = subj.score_boxes(blur, [full], ref)[0]
    assert out["in_focus"] is False and out["label"] == "soft"
    out = subj.score_boxes(gray, [full], ref)[0]
    assert out["in_focus"] is True


def test_the_boxes_are_the_crop_mode_boxes_and_scored_in_display_coordinates(client, shoot, ids):
    iid = ids["two.jpg"]
    b = {"x0": 0.05, "y0": 0.2, "x1": 0.45, "y1": 0.8}
    assert client.post(f"/api/crop/{iid}/boxes", json={"folder": str(shoot), "boxes": [b]}).status_code == 200
    stored = client.get(f"/api/crop/{iid}/boxes", params={"folder": str(shoot)}).json()["boxes"]
    assert stored == [[0.05, 0.2, 0.45, 0.8]]
    r = score(client, shoot, iid, [dict(zip(("x0", "y0", "x1", "y1"), stored[0], strict=True))])
    assert r.status_code == 200 and r.json()["scores"][0]["sharp"] is not None


def test_validation(client, shoot, ids):
    iid = ids["landscape.jpg"]
    assert score(client, shoot, iid, [{"x0": 0.5, "y0": 0.1, "x1": 0.4, "y1": 0.7}]).status_code == 422
    assert score(client, shoot, iid, [{"x0": -0.1, "y0": 0.1, "x1": 0.4, "y1": 0.7}]).status_code == 422
    seven = [{"x0": 0.1 * i, "y0": 0.1, "x1": 0.1 * i + 0.05, "y1": 0.5} for i in range(7)]
    assert score(client, shoot, iid, seven).status_code == 422
    assert score(client, shoot, 99999, []).status_code == 404
    assert score(client, shoot, iid, []).json()["scores"] == []
