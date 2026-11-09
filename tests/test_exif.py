import pytest

from shotsieve.io.exif import read_meta


def test_burst_timestamps_serial_flash(shoot_dir, manifest):
    by = {m["file"]: m for m in manifest}
    m0 = read_meta(shoot_dir / "burst0_0.jpg")
    m4 = read_meta(shoot_dir / "burst0_4.jpg")
    assert m0.ts_source == "exif"
    assert m0.camera_serial == "SN123456" and (m0.make, m0.model) == ("ShotCam", "S1")
    assert m0.flash_fired is True
    assert read_meta(shoot_dir / "burst1_0.jpg").flash_fired is False
    assert m4.timestamp - m0.timestamp == pytest.approx(1.2, abs=0.002)
    assert m0.timestamp == pytest.approx(by["burst0_0.jpg"]["timestamp"], abs=0.002)
    b1 = read_meta(shoot_dir / "burst1_0.jpg")
    assert b1.timestamp - m0.timestamp == pytest.approx(60, abs=0.002)
    assert m0.iso == 400 and m0.exposure_time == pytest.approx(1 / 125)
    assert m0.fnumber == pytest.approx(2.8) and m0.focal_length == 50
    assert m0.orientation == 1 and m0.width == 640 and m0.height == 640


def test_orientation_tag(gen_dir):
    assert read_meta(gen_dir / "misc" / "orient6.jpg").orientation == 6


def test_garbage_file_defaults(tmp_path):
    p = tmp_path / "bad.jpg"
    p.write_bytes(b"not an image at all")
    m = read_meta(p)
    assert m.ts_source == "mtime" and m.timestamp == pytest.approx(p.stat().st_mtime)
    assert m.camera_serial == "unknown" and m.flash_fired is None and m.orientation == 1


def test_missing_and_png(tmp_path):
    assert read_meta(tmp_path / "nope.dng").ts_source == "mtime"
    from PIL import Image

    Image.new("RGB", (10, 8)).save(tmp_path / "a.png")
    m = read_meta(tmp_path / "a.png")
    assert (m.width, m.height) == (10, 8) and m.ts_source == "mtime"


def test_raw_meta(gen_dir):
    m = read_meta(gen_dir / "raw" / "pentax_kx.dng")
    assert "PENTAX" in m.make.upper() and m.iso


def test_serial_fallback_make_model(tmp_path):
    import piexif
    from PIL import Image

    p = tmp_path / "x.jpg"
    exif = piexif.dump({"0th": {piexif.ImageIFD.Make: b"Acme", piexif.ImageIFD.Model: b"Z9"}})
    Image.new("RGB", (8, 8)).save(p, exif=exif)
    assert read_meta(p).camera_serial == "Acme-Z9"
