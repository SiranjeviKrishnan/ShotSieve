"""AF-ROI focus measure, the af_missed rule, orientation and v1-model compatibility."""

import cv2
import numpy as np
import pytest

from shotsieve.config import Settings
from shotsieve.features import af_focus, sharpness
from shotsieve.features.record import build_vector, compute_norm
from shotsieve.features.subject import FaceObs
from shotsieve.features.vector import FEATURE_NAMES, FeatureVector
from shotsieve.io import makernotes
from shotsieve.io.preview import apply_orientation, load_preview
from shotsieve.pipeline import stages
from shotsieve.scoring import quality

S = Settings()


@pytest.fixture(scope="module")
def scene(gen_dir):
    img = cv2.cvtColor(cv2.imread(str(gen_dir / "src" / "building.jpg")), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (960, 640), interpolation=cv2.INTER_AREA)
    return img


def tile_vars(img):
    lap = cv2.Laplacian(sharpness.to_luma(img), cv2.CV_32F)
    h, w = lap.shape
    th, tw = h // 8, w // 8
    return lap[: th * 8, : tw * 8].reshape(8, th, 8, tw).var(axis=(1, 3))


@pytest.fixture(scope="module")
def af_box(scene):
    """A textured tile far from the sharpest tile, as a normalised (cx, cy, w, h) box."""
    tv = tile_vars(scene)
    py, px = np.unravel_index(int(np.argmax(tv)), tv.shape)
    best = None
    for y in range(1, 7):
        for x in range(1, 7):
            if max(abs(y - py), abs(x - px)) >= 3 and tv[y, x] >= 0.15 * tv[py, px]:
                if best is None or tv[y, x] > tv[best]:
                    best = (y, x)
    assert best is not None, "fixture has no textured tile away from the peak"
    return ((best[1] + 0.5) / 8, (best[0] + 0.5) / 8, 0.1, 0.1)


def blur_box(img, box, sigma=4.0, pad=24):
    h, w = img.shape[:2]
    m = af_focus.roi_mask((h, w), [box])
    ys, xs = np.nonzero(m)
    y0, y1, x0, x1 = max(0, ys.min() - pad), ys.max() + pad, max(0, xs.min() - pad), xs.max() + pad
    out = img.copy()
    out[y0:y1, x0:x1] = cv2.GaussianBlur(img[y0:y1, x0:x1], (0, 0), sigma)
    return out


def fails_for(img, box, ref_img=None, faces=()):
    """Run stage A + vector + classify; the shoot reference comes from `ref_img` (the sharp
    shot of the same scene) so a lone blurred frame is judged against its sharp siblings."""
    raw = stages.stage_a(img, [list(box)])
    ref = stages.stage_a(ref_img if ref_img is not None else img, [list(box)])
    raw.faces = list(faces)
    fv = build_vector(raw, compute_norm([ref]))
    flags = ["af_peak_adjacent"] if raw.af_peak_adjacent else []
    return fv, quality.classify_fails(fv, S, flags)


def test_sharp_at_af_box_is_not_flagged(scene, af_box):
    fv, fails = fails_for(scene, af_box)
    assert fv.af_known == 1.0 and fv.af_sharp_norm == pytest.approx(1.0)
    assert "af_missed" not in fails


def test_blurred_af_box_with_sharp_rest_is_af_missed(scene, af_box):
    blurred = blur_box(scene, af_box)
    fv, fails = fails_for(blurred, af_box, ref_img=scene)
    assert fv.af_sharp_norm < 0.35 and fv.af_peak_ratio >= 2.5
    assert fails.get("af_missed") == quality.SOFT
    assert quality.CERTAIN != fails["af_missed"]


def test_af_on_face_defers_to_face_sharpness(scene, af_box):
    h, w = scene.shape[:2]
    cx, cy = af_box[0] * w, af_box[1] * h
    face = FaceObs((cx - 60, cy - 60, 120, 120), 0.9, important=True)
    blurred = blur_box(scene, af_box)
    fv, fails = fails_for(blurred, af_box, ref_img=scene, faces=[face])
    assert fv.af_on_face == 1.0 and "af_missed" not in fails


def test_unknown_af_never_flags(scene):
    raw = stages.stage_a(scene)
    fv = build_vector(raw, compute_norm([raw]))
    assert fv.af_known == 0.0 and np.isnan(fv.af_sharp_norm)
    assert "af_missed" not in quality.classify_fails(fv, S)


def test_adjacent_peak_is_not_a_miss(scene):
    tv = tile_vars(scene)
    py, px = np.unravel_index(int(np.argmax(tv)), tv.shape)
    box = ((min(px + 1, 7) + 0.5) / 8, (py + 0.5) / 8, 0.1, 0.1)  # next to the sharpest tile
    blurred = blur_box(scene, box, pad=4)
    raw = stages.stage_a(blurred, [list(box)])
    assert raw.af_peak_adjacent


def test_on_face_iou_and_centre():
    assert af_focus.on_face([(0.5, 0.5, 0.1, 0.1)], [(400, 400, 200, 200)], 1000, 1000)
    assert not af_focus.on_face([(0.1, 0.1, 0.05, 0.05)], [(700, 700, 100, 100)], 1000, 1000)


def test_orientation_6_jpeg_maps_box_to_same_scene_spot(scene, af_box, make_fixtures, tmp_path):
    """A sensor-orientation AF point, stored in a rotated JPEG with Orientation=6, lands on the
    same scene content in the upright preview."""
    mf = make_fixtures
    h, w = scene.shape[:2]
    sensor = np.ascontiguousarray(np.rot90(scene, 1))  # inverse of orientation 6 (rot 90 cw)
    sh, sw = sensor.shape[:2]
    # the scene box expressed in sensor coords (inverse of the mapping under test)
    cx, cy, bw, bh = af_box
    sx, sy = cy, 1 - cx  # rot90 ccw: (x, y) -> (y, 1-x)
    sensor_box = (sx, sy, bh, bw)
    afw, afh = 6000, 4000
    pt = (round(sensor_box[2] * afw), round(sensor_box[3] * afh),
          round((sensor_box[0] - 0.5) * afw), round((0.5 - sensor_box[1]) * afh))  # fmt: skip
    af = mf.canon_afinfo2([pt], (afw, afh), [0])
    tiff = mf.tiff_with_makernote(b"Canon", [(0x0026, 3, af, len(af) // 2)], orientation=6)
    f = tmp_path / "rot.jpg"
    f.write_bytes(mf.jpeg_with_tiff(tiff, sensor, quality=98))
    (p,) = makernotes.read_af_points(f)
    up = makernotes.to_oriented(p, 6)
    img = load_preview(f, 1280)
    assert img.shape[:2] == (sw, sh) == (h, w)
    got = af_focus.measure(sharpness.to_luma(img), [tuple(up)])
    want = af_focus.measure(sharpness.to_luma(scene), [af_box])
    assert got.lap == pytest.approx(want.lap, rel=0.15)
    assert np.array_equal(apply_orientation(sensor, 6).shape, scene.shape)


def test_v1_models_still_load_and_predict_on_v2_features(tmp_path):
    from shotsieve.personal.model import PersonalModel

    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 25)).astype(np.float32)
    y = (X[:, 1] > 0).astype(int)
    m = PersonalModel(feature_names=FEATURE_NAMES[:25], schema_version=1).fit(X, y)
    path = m.save(tmp_path / "m.joblib")
    loaded = PersonalModel.load(path)
    assert loaded.schema_version == 1 and len(loaded.feature_names) == 25
    wide = np.hstack([X, np.zeros((120, len(FEATURE_NAMES) - 25), np.float32)])
    assert np.allclose(loaded.predict_proba(wide), loaded.predict_proba(X))
    assert FeatureVector.from_array_v1(X[0]).af_known == 0.0
