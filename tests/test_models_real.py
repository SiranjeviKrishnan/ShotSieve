import time

import cv2
import numpy as np
import pytest

from shotsieve.models.face_detect import Face, get_face_detector
from shotsieve.models.face_embed import EMBED_DIM, cosine, get_face_embedder
from shotsieve.models.face_landmarks import expand_crop, get_face_landmarker
from shotsieve.models.image_embed import EMBED_DIM as IMG_DIM
from shotsieve.models.image_embed import ImageEmbedder, get_image_embedder

pytestmark = [pytest.mark.models, pytest.mark.usefixtures("models_ready")]


def test_detect_single_faces(rgb):
    det = get_face_detector()
    for name in ("lena.jpg", "messi5.jpg", "portrait2.jpg"):
        faces = det.detect(rgb(name))
        assert len(faces) >= 1, name
        assert faces[0].landmarks.shape == (5, 2) and 0 < faces[0].score <= 1


def test_detect_group(rgb):
    assert len(get_face_detector().detect(rgb("group_rgb.jpg"))) >= 3


def test_detect_none_on_landscape(rgb):
    assert get_face_detector().detect(rgb("building.jpg")) == []


def test_bbox_in_original_coordinates(rgb):
    det = get_face_detector()
    img = rgb("lena.jpg")
    big = cv2.resize(img, None, fx=3, fy=3)
    a, b = det.detect(img)[0], det.detect(big)[0]
    h, w = img.shape[:2]
    x, y, bw, bh = a.bbox
    assert 0 <= x and 0 <= y and x + bw <= w + 2 and y + bh <= h + 2
    assert np.allclose(np.array(b.bbox) / 3, a.bbox, atol=8)
    faces = det.detect(rgb("group_rgb.jpg"), max_faces=4)
    assert len(faces) == 4 and faces[0].area >= faces[-1].area


def test_landmarks_eyes_open(rgb):
    det, lm = get_face_detector(), get_face_landmarker()
    for name in ("lena.jpg", "portrait2.jpg"):
        img = rgb(name)
        ex = lm.analyze(img, det.detect(img)[0])
        assert ex.found and ex.eyes_closed < 0.5
        assert 0 <= ex.blink_left <= 1 and 0 <= ex.smile <= 1
        assert "eyeBlinkLeft" in ex.blendshapes and len(ex.blendshapes) >= 50


def test_landmarks_no_face_is_nan():
    ex = get_face_landmarker().analyze_crop(np.full((128, 128, 3), 127, np.uint8))
    assert not ex.found and np.isnan(ex.eyes_closed) and np.isnan(ex.smile)


def test_expand_crop_bounds(rgb):
    img = rgb("lena.jpg")
    crop = expand_crop(img, (0, 0, 400, 400))
    assert max(crop.shape[:2]) <= 256 and crop.size > 0


def test_face_embed_same_vs_different(shoot_dir, rgb):
    det, emb = get_face_detector(), get_face_embedder()

    def vec(img):
        return emb.embed(img, det.detect(img)[0])

    def load(n):
        return cv2.cvtColor(cv2.imread(str(shoot_dir / n)), cv2.COLOR_BGR2RGB)

    a, b, c = vec(load("burst0_0.jpg")), vec(load("burst0_3.jpg")), vec(load("burst1_0.jpg"))
    assert a.shape == (EMBED_DIM,) and abs(np.linalg.norm(a) - 1) < 1e-4
    assert cosine(a, b) >= 0.6
    assert cosine(a, c) < cosine(a, b) and cosine(a, c) < 0.36


def test_image_embed_burst_and_scene(shoot_dir):
    emb = get_image_embedder()

    def load(n):
        return cv2.cvtColor(cv2.imread(str(shoot_dir / n)), cv2.COLOR_BGR2RGB)

    v = [emb.embed(load(f"burst0_{i}.jpg")) for i in (0, 2, 3, 4)]
    for x in v[1:]:
        assert 1 - float(v[0] @ x) < 0.12
    land = emb.embed(load("landscape.jpg"))
    assert v[0].shape == (IMG_DIM,) and abs(np.linalg.norm(v[0]) - 1) < 1e-4
    assert 1 - float(v[0] @ land) > 0.3


def test_image_embed_batch_equals_single(shoot_dir):
    emb = ImageEmbedder(batch_size=3)
    names = ["burst0_0.jpg", "burst1_0.jpg", "burst2_0.jpg", "landscape.jpg", "black.jpg"]
    imgs = [cv2.cvtColor(cv2.imread(str(shoot_dir / n)), cv2.COLOR_BGR2RGB) for n in names]
    batch = emb.embed_batch(imgs)
    single = np.stack([emb.embed(i) for i in imgs])
    assert batch.shape == (5, IMG_DIM)
    assert np.abs(batch - single).max() < 2e-3
    assert emb.embed_batch([]).shape == (0, IMG_DIM)


def test_image_embed_dtype_and_provider():
    emb = get_image_embedder()
    assert emb.input_dtype in (np.float32, np.float16)
    assert emb.provider in {"CoreMLExecutionProvider", "CPUExecutionProvider"}
    assert (emb.input_name, emb.output_name) == ("pixel_values", "last_hidden_state")


def test_face_row_roundtrip():
    f = Face((1, 2, 3, 4), np.arange(10, dtype=np.float32).reshape(5, 2), 0.9)
    assert f.to_row().shape == (15,) and f.area == 12


def test_report_latency(rgb, capsys):
    """Informational: per-model latency on a fixture image."""
    img = rgb("lena.jpg")
    det, emb, lm, ie = (
        get_face_detector(),
        get_face_embedder(),
        get_face_landmarker(),
        get_image_embedder(),
    )
    face = det.detect(img)[0]
    for fn in (
        lambda: det.detect(img),
        lambda: emb.embed(img, face),
        lambda: lm.analyze(img, face),
        lambda: ie.embed(img),
    ):
        fn()
    out = {}
    for name, fn in [
        ("yunet", lambda: det.detect(img)),
        ("sface", lambda: emb.embed(img, face)),
        ("landmarker", lambda: lm.analyze(img, face)),
        ("dinov2", lambda: ie.embed(img)),
    ]:
        t = time.perf_counter()
        for _ in range(10):
            fn()
        out[name] = (time.perf_counter() - t) * 100
    with capsys.disabled():
        print("\nlatency ms:", {k: round(v, 1) for k, v in out.items()}, "provider:", ie.provider)


# ---- SigLIP2 (T10) ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def siglip_scores(search_ready, search_images):
    from shotsieve.models.text_image import SiglipText, SiglipVision

    names = sorted(search_images)
    vis = SiglipVision("fp32")
    emb = vis.embed_batch([search_images[n] for n in names])
    assert emb.shape == (len(names), 768) and np.allclose(np.linalg.norm(emb, axis=1), 1, atol=1e-4)
    txt = SiglipText()
    assert not txt.loaded  # lazy: nothing loaded until the first query

    def rank(query: str) -> list[str]:
        sims = emb @ txt.embed(query)
        return [names[i] for i in np.argsort(-sims)]

    yield rank
    assert txt.loaded
    txt.unload()
    assert not txt.loaded


def test_siglip_a_dog_ranks_dog_top3(siglip_scores):
    assert "dog" in siglip_scores("a dog")[:3]


def test_siglip_a_red_car_picks_red_car(siglip_scores):
    assert siglip_scores("a red car")[0] == "red_car"


def test_siglip_other_subjects(siglip_scores):
    assert siglip_scores("a butterfly")[0] == "butterfly"
    assert siglip_scores("a soccer player kicking a ball")[0] == "messi5"
    assert "building" in siglip_scores("a modern apartment building")[:2]
