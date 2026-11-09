import cv2
import numpy as np
import pytest

from shotsieve.io.preview import PreviewError, apply_orientation, load_preview


def test_jpeg_downscale(shoot_dir):
    img = load_preview(shoot_dir / "landscape.jpg", max_edge=320)
    assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3
    assert max(img.shape[:2]) == 320 and img.shape[:2] == (213, 320)
    small = load_preview(shoot_dir / "burst0_0.jpg", max_edge=1280)
    assert small.shape[:2] == (640, 640)


def test_big_jpeg_reduced_decode(tmp_path):
    big = np.random.default_rng(0).integers(0, 255, (2400, 3600, 3), np.uint8)
    p = tmp_path / "big.jpg"
    cv2.imwrite(str(p), big)
    img = load_preview(p, max_edge=800)
    assert max(img.shape[:2]) <= 800 and max(img.shape[:2]) > 400


def test_orientation_applied(gen_dir):
    img = load_preview(gen_dir / "misc" / "orient6.jpg")
    assert img.shape[:2] == (400, 600)  # stored 600x400 portrait, displayed landscape
    ref = cv2.resize(
        cv2.cvtColor(cv2.imread(str(gen_dir / "src" / "messi5.jpg")), cv2.COLOR_BGR2RGB), (600, 400)
    )
    assert np.abs(img.astype(int) - ref.astype(int)).mean() < 12


def test_apply_orientation_shapes():
    a = np.zeros((2, 4, 3), np.uint8)
    for o in (5, 6, 7, 8):
        assert apply_orientation(a, o).shape[:2] == (4, 2)
    for o in (1, 2, 3, 4):
        assert apply_orientation(a, o).shape[:2] == (2, 4)


def test_raw_embedded_thumb(gen_dir):
    img = load_preview(gen_dir / "raw" / "pentax_kx.dng", max_edge=1280)
    assert img.dtype == np.uint8 and img.shape[2] == 3
    assert 640 <= max(img.shape[:2]) <= 1280


def test_raw_small_thumb_falls_back_to_postprocess(gen_dir):
    img = load_preview(gen_dir / "raw" / "panasonic_lx1.dng", max_edge=1280)
    assert max(img.shape[:2]) > 640 and max(img.shape[:2]) <= 1280


def test_max_edge_respected_raw(gen_dir):
    assert max(load_preview(gen_dir / "raw" / "pentax_kx.dng", max_edge=400).shape[:2]) == 400


def test_bad_file(tmp_path):
    p = tmp_path / "bad.jpg"
    p.write_bytes(b"junk")
    with pytest.raises(PreviewError):
        load_preview(p)
    with pytest.raises(PreviewError):
        load_preview(tmp_path / "missing.nef")
