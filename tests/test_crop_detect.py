# ruff: noqa: E501
"""T22 step 3: YOLOX decode, subject linking, saliency / point prompts, cache, idle unload."""

from __future__ import annotations

import importlib.util
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from shotsieve.config import Settings
from shotsieve.models import object_detect as od
from shotsieve.models import patch_features as pf
from shotsieve.models.idle import IdleLoaded
from shotsieve.pipeline import crop as pcrop
from shotsieve.pipeline.cache import Cache
from shotsieve.pipeline.crop_score import Subject

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"


def _synth():
    spec = importlib.util.spec_from_file_location("make_crop_fixtures", ROOT / "scripts" / "make_crop_fixtures.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_crop_fixtures"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---- YOLOX decode / NMS (network-free) ---------------------------------------------------------


def canned(boxes):
    """A `[1, 8400, 85]` raw output with the given (anchor index, cx, cy, w, h, cls, score)."""
    out = np.full((1, 8400, 85), -8.0, np.float32)
    out[0, :, 4] = 0.0  # objectness 0
    out[0, :, 5:] = 0.0
    for idx, cx, cy, w, h, cls, score in boxes:
        stride = od._EXP_STRIDES[idx, 0]
        gx, gy = od._GRIDS[idx]
        out[0, idx, 0] = cx / stride - gx
        out[0, idx, 1] = cy / stride - gy
        out[0, idx, 2] = np.log(w / stride)
        out[0, idx, 3] = np.log(h / stride)
        out[0, idx, 4] = 1.0
        out[0, idx, 5 + cls] = score
    return out


def test_grids_cover_8400_anchors_in_zoo_order():
    assert od._GRIDS.shape == (8400, 2) and od._EXP_STRIDES.shape == (8400, 1)
    assert tuple(od._GRIDS[0]) == (0, 0) and tuple(od._GRIDS[1]) == (1, 0) and tuple(od._GRIDS[80]) == (0, 1)
    assert od._EXP_STRIDES[0, 0] == 8 and od._EXP_STRIDES[6400, 0] == 16 and od._EXP_STRIDES[8399, 0] == 32
    assert tuple(od._GRIDS[6400]) == (0, 0) and tuple(od._GRIDS[8000]) == (0, 0)


def test_decode_boxes_nms_and_class_names():
    out = canned([
        (100, 200, 300, 120, 240, 0, 0.9),      # person
        (101, 205, 302, 118, 236, 0, 0.8),      # duplicate (suppressed by NMS)
        (6500, 400, 100, 60, 60, 16, 0.7),       # dog
        (5, 50, 50, 30, 30, 0, 0.2),             # below confidence
    ])
    dets = od.decode(out, 1.0)
    assert [d.name for d in dets] == ["person", "dog"]
    p = dets[0]
    assert p.box == pytest.approx((140, 180, 260, 420), abs=0.5) and p.score == pytest.approx(0.9, abs=1e-4)
    assert od.COCO[0] == "person" and od.COCO[16] == "dog" and len(od.COCO) == 80
    assert "dog" in od.ANIMALS and "person" not in od.ANIMALS


def test_letterbox_is_top_left_with_pad_114_and_the_ratio_undoes_it():
    img = np.zeros((300, 600, 3), np.uint8)
    img[:, :, 0] = 255  # pure red RGB
    x, ratio = od.letterbox(img)
    assert x.shape == (3, 640, 640) and x.dtype == np.float32 and ratio == pytest.approx(640 / 600)
    assert x[0, 0, 0] == 0 and x[2, 0, 0] == 255  # BGR order, 0-255 range
    assert x[0, 639, 0] == 114 and x[1, 639, 5] == 114  # padding below the resized content
    # a box at (60, 30)-(120, 90) in the original appears scaled by `ratio` in the letterboxed frame
    out = canned([(100, 100 * 0 + 128, 128, 64, 64, 0, 0.9)])
    d = od.decode(out, ratio)[0]
    assert d.box[0] == pytest.approx((128 - 32) / ratio, abs=0.5)


@pytest.mark.models
def test_yolox_matches_an_independent_opencv_dnn_decode_within_2px():
    if not od.available():
        pytest.skip("yolox not downloaded (models download --extra crop)")
    from shotsieve.models.registry import model_path

    img = cv2.cvtColor(cv2.imread(str(FIX / "real" / "cc0_football_18.jpg")), cv2.COLOR_BGR2RGB)
    ours = od.ObjectDetector().detect(img)
    # reference: opencv_zoo's YoloX (readNet + the zoo's per-anchor post-processing, written out
    # independently with plain loops instead of our vectorised decode)
    net = cv2.dnn.readNet(str(model_path("yolox_s")))
    h, w = img.shape[:2]
    ratio = min(640 / h, 640 / w)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    pad = np.ones((640, 640, 3), np.float32) * 114.0
    pad[: int(h * ratio), : int(w * ratio)] = cv2.resize(bgr, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_LINEAR)
    net.setInput(np.transpose(pad, (2, 0, 1))[None])
    raw = net.forward(net.getUnconnectedOutLayersNames())[0][0]
    cand = []
    idx = 0
    for stride in (8, 16, 32):
        n = 640 // stride
        for gy in range(n):
            for gx in range(n):
                row = raw[idx]
                idx += 1
                sc = row[4] * row[5:]
                c = int(np.argmax(sc))
                if sc[c] >= 0.35:
                    cx, cy = (row[0] + gx) * stride, (row[1] + gy) * stride
                    bw, bh = np.exp(row[2]) * stride, np.exp(row[3]) * stride
                    cand.append([cx - bw / 2, cy - bh / 2, bw, bh, float(sc[c]), c])
    keep = cv2.dnn.NMSBoxesBatched([c[:4] for c in cand], [c[4] for c in cand], [c[5] for c in cand], 0.35, 0.45)
    ref = [cand[int(k)] for k in np.asarray(keep).reshape(-1)]
    ref_boxes = sorted((c[0] / ratio, c[1] / ratio, (c[0] + c[2]) / ratio, (c[1] + c[3]) / ratio) for c in ref)
    assert len(ref_boxes) >= 3
    got = sorted(
        (max(0.0, d.box[0]), max(0.0, d.box[1]), min(float(w), d.box[2]), min(float(h), d.box[3])) for d in ours
    )
    assert len(got) == len(ref_boxes)
    for a, b in zip(got, ref_boxes, strict=True):
        b = (max(0.0, b[0]), max(0.0, b[1]), min(float(w), b[2]), min(float(h), b[3]))
        assert max(abs(x - y) for x, y in zip(a, b, strict=True)) <= 2.0, (a, b)
    # golden: the football frame has several people
    assert sum(d.name == "person" for d in ours) >= 3


# ---- linking, saliency, point prompts (numpy only) -------------------------------------------


def test_face_person_link_rules():
    face = Subject("f1", "face", (0.45, 0.10, 0.55, 0.20))
    person = Subject("p1", "person", (0.40, 0.08, 0.60, 0.95))
    other = Subject("p2", "person", (0.70, 0.30, 0.90, 0.95))  # face is not inside
    low = Subject("p3", "person", (0.40, 0.50, 0.60, 0.95))  # contains the face? no (face above)
    pcrop._link_faces([face, person, other, low])
    assert face.linked == "p1" and person.linked == "f1" and other.linked is None and low.linked is None
    # a face in the lower half of a person box is not that person's face (top must be within 0.25 h)
    f2 = Subject("f2", "face", (0.45, 0.60, 0.55, 0.70))
    p4 = Subject("p4", "person", (0.40, 0.08, 0.60, 0.95))
    pcrop._link_faces([f2, p4])
    assert f2.linked is None


def _tokens(h=20, w=28, obj=(8, 6, 16, 14), seed=0):
    rng = np.random.default_rng(seed)
    bg = rng.normal(size=(384,)).astype(np.float32)
    ob = rng.normal(size=(384,)).astype(np.float32)
    t = bg + 0.1 * rng.normal(size=(h, w, 384)).astype(np.float32)
    x0, y0, x1, y1 = obj
    t[y0:y1, x0:x1] = ob + 0.1 * rng.normal(size=(y1 - y0, x1 - x0, 384)).astype(np.float32)
    return t / np.linalg.norm(t, axis=-1, keepdims=True)


def test_saliency_peak_and_point_region_on_synthetic_tokens():
    t = _tokens()
    sal = pf.saliency(t)
    assert sal.shape == (20, 28) and sal.min() >= 0 and sal.max() == pytest.approx(1.0)
    pk = pf.peak_region(sal)
    cx, cy = (pk[0] + pk[2]) / 2, (pk[1] + pk[3]) / 2
    assert 8 / 28 < cx < 16 / 28 and 6 / 20 < cy < 14 / 20  # inside the object block
    box, how = pf.point_region(t, sal, 11 / 28, 11 / 20)
    assert how == "region"
    want = (8 / 28, 6 / 20, 16 / 28, 14 / 20)
    iw = min(box[2], want[2]) - max(box[0], want[0])
    ih = min(box[3], want[3]) - max(box[1], want[1])
    inter = iw * ih
    iou = inter / ((box[2] - box[0]) * (box[3] - box[1]) + (want[2] - want[0]) * (want[3] - want[1]) - inter)
    assert iou > 0.7, (box, want, iou)
    # a point on the background selects (almost) everything: fall back to a 15 % box on the click
    box, how = pf.point_region(t, sal, 0.8, 0.8)
    assert how == "fallback" and (box[2] - box[0]) * (box[3] - box[1]) == pytest.approx(0.15, abs=0.05)


def test_grid_size_is_multiples_of_14_with_long_edge_448():
    for h, w in ((480, 640), (640, 480), (1000, 1000), (300, 1600), (1600, 300)):
        gh, gw, H, W = pf.grid_size(h, w)
        assert H == gh * 14 and W == gw * 14 and max(H, W) <= 560
        if max(h, w) / min(h, w) < 2:
            assert min(H, W) >= 224 - 14
    assert pf.grid_size(480, 640)[2:] == (336, 448)


def test_token_lru_keeps_the_last_eight():
    lru = pf.TokenLRU(3)
    for i in range(5):
        lru.put(str(i), np.zeros((1, 1, 384), np.float32))
    assert lru.get("0") is None and lru.get("1") is None and lru.get("4") is not None
    lru.clear()
    assert lru.get("4") is None


def test_idle_loaded_unloads_after_the_idle_time_and_never_while_busy():
    made = []
    h = IdleLoaded(lambda: made.append(1) or object(), 0.25)
    with h.use() as a:
        time.sleep(0.4)  # busy: must stay loaded
        assert h.loaded
    with h.use() as b:
        assert a is b and len(made) == 1
    time.sleep(0.7)
    assert not h.loaded
    with h.use() as c:
        assert c is not a and len(made) == 2
    h.unload()
    assert not h.loaded


def test_horizon_detector_on_level_and_tilted_scenes():
    m = _synth()
    img, y = m.horizon_scene(tilt_deg=0.0)
    got = pcrop.detect_horizon(img)
    assert got is not None and abs(got[0] - y) < 0.03 and abs(got[1]) < 0.6
    img, y = m.horizon_scene(tilt_deg=2.5)
    got = pcrop.detect_horizon(img)
    assert got is not None and 1.5 < abs(got[1]) < 3.5
    img, _ = m.horizon_scene(tilt_deg=12.0)
    assert pcrop.detect_horizon(img) is None  # not near-level: ignored
    assert pcrop.detect_horizon(np.full((200, 300, 3), 128, np.uint8)) is None


# ---- real models ---------------------------------------------------------------------------


@pytest.mark.models
def test_saliency_peak_lies_inside_a_single_object_image(models_ready):
    m = _synth()
    img, box = m.single_object()
    with pf.holder(2, 5).use() as ex:
        t = ex.tokens(img)
    sal = pf.saliency(t)
    pk = pf.peak_region(sal)
    ay, ax = np.unravel_index(int(np.argmax(sal)), sal.shape)
    assert box[0] <= (ax + 0.5) / sal.shape[1] <= box[2] and box[1] <= (ay + 0.5) / sal.shape[0] <= box[3]
    assert pk is not None


@pytest.fixture()
def folder(tmp_path):
    f = tmp_path / "shoot"
    f.mkdir()
    return f


def _cache_with(folder: Path, names: list[str]):
    c = Cache.for_folder(folder)
    ids = c.ids_for([str(folder / n) for n in names])
    return c, ids


@pytest.mark.models
def test_detect_caches_by_fingerprint_and_det_version(folder, models_ready, monkeypatch):
    shutil.copy(FIX / "crop" / "synth" / "faces.jpg", folder / "a.jpg")
    c, ids = _cache_with(folder, ["a.jpg"])
    p = str(folder / "a.jpg")
    calls = []
    real = pcrop._detect
    monkeypatch.setattr(pcrop, "_detect", lambda *a, **k: calls.append(1) or real(*a, **k))
    s1 = pcrop.detect_subjects(folder, c, ids[p], p, Settings())
    s2 = pcrop.detect_subjects(folder, c, ids[p], p, Settings())
    assert len(calls) == 1  # the second call is served from the cache
    assert [s.sid for s in s1.subjects] == [s.sid for s in s2.subjects]
    assert s1.saliency is not None and s2.saliency is not None
    assert np.allclose(s1.saliency, s2.saliency, atol=2e-3)
    assert (s2.fw, s2.fh, s2.orientation) == (640, 480, 1)
    # the file changes -> fingerprint changes -> detection runs again
    img = cv2.imread(str(folder / "a.jpg"))
    cv2.imwrite(str(folder / "a.jpg"), cv2.flip(img, 1))
    pcrop.detect_subjects(folder, c, ids[p], p, Settings())
    assert len(calls) == 2
    # a CROP_DET_VERSION bump invalidates too
    monkeypatch.setattr(pcrop, "CROP_DET_VERSION", "999")
    pcrop.detect_subjects(folder, c, ids[p], p, Settings())
    assert len(calls) == 3
    c.close()


@pytest.mark.models
def test_without_the_object_detector_faces_and_saliency_remain(folder, models_ready, monkeypatch):
    shutil.copy(FIX / "crop" / "synth" / "object.jpg", folder / "o.jpg")
    c, ids = _cache_with(folder, ["o.jpg"])
    monkeypatch.setattr(od, "available", lambda: False)
    p = str(folder / "o.jpg")
    s = pcrop.detect_subjects(folder, c, ids[p], p, Settings())
    assert s.detector is None and all(x.kind == "face" for x in s.subjects)
    assert s.saliency is not None and s.peak is not None
    pt = pcrop.point_subject(folder, c, ids[p], p, 0.58, 0.55, Settings())
    assert pt.kind == "point" and pt.box[0] < 0.58 < pt.box[2]
    assert pcrop.auto_selection(s)[0].kind == "point"  # nothing detected: the saliency peak region
    c.close()


@pytest.mark.models
def test_a_missing_preview_source_and_video_items_are_refused(folder, models_ready):
    c, ids = _cache_with(folder, ["gone.jpg"])
    with pytest.raises(pcrop.CropError) as e:
        pcrop.detect_subjects(folder, c, ids[str(folder / "gone.jpg")], str(folder / "gone.jpg"))
    assert e.value.status == 404
    with pytest.raises(pcrop.CropError) as e:
        pcrop.detect_subjects(folder, c, 1, str(folder / "clip.mp4") + "#t=1.000")
    assert e.value.status == 422
    c.close()
