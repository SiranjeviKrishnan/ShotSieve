"""T12 video frame-grab culling on small synthetic videos (needs the downloaded models)."""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from shotsieve.config import Settings
from shotsieve.io import video as iov
from shotsieve.io.exif import read_meta
from shotsieve.io.formats import VIDEO_EXTS, scan_folder, scan_videos, split_virtual, virtual_path
from shotsieve.models.face_landmarks import FaceExpression
from shotsieve.pipeline import video as pv
from shotsieve.pipeline.cache import Cache
from shotsieve.pipeline.runner import run_cull
from tests.video_synth import (
    FPS,
    SCENE_S,
    make_index_video,
    make_three_scene_video,
    read_index,
)

NO_AUDIT = Settings(audit_enabled=False, video=True)


class _FakeLandmarker:
    """Reads the 'eyes closed' marker patch instead of a real blendshape model."""

    def analyze(self, img, face):
        # `face` is in full-frame pixels; the marker lives at the frame's top-left corner
        return FaceExpression(True, *self._blink(img), 0.1)

    @staticmethod
    def _blink(img):
        patch = img[2:26, 2:26].reshape(-1, 3).mean(0)
        closed = patch[0] > 200 and patch[1] < 90
        v = 0.9 if closed else 0.05
        return v, v


@pytest.fixture()
def fake_landmarks(monkeypatch):
    """Only for the closed-eye *span* test: painted eyes cannot drive real blendshapes."""
    import shotsieve.models.face_landmarks as fl

    monkeypatch.setattr(fl, "get_face_landmarker", lambda *a, **k: _FakeLandmarker())


@pytest.fixture(scope="module")
def sources(gen_dir) -> Path:
    return gen_dir / "src"


@pytest.fixture(scope="module")
def clip_dir(tmp_path_factory, sources):
    d = tmp_path_factory.mktemp("videoshoot")
    spans = make_three_scene_video(d / "clip.mp4", sources)
    return d, spans


@pytest.fixture
def shoot(clip_dir, tmp_path):
    d, spans = clip_dir
    dest = tmp_path / "shoot"
    dest.mkdir()
    shutil.copy(d / "clip.mp4", dest / "clip.mp4")
    return dest, spans


def _in_open_span(t, spans):
    return any(s["kind"] == "open" and s["t0"] <= t < s["t1"] for s in spans)


# ---- pure logic -----------------------------------------------------------------------------


def test_virtual_paths_and_scan(tmp_path):
    assert VIDEO_EXTS == {".mp4", ".mov", ".m4v"}
    v = virtual_path(tmp_path / "a.mp4", 12.3456)
    assert v.endswith("a.mp4#t=12.346")
    assert split_virtual(v) == (tmp_path / "a.mp4", 12.346)
    assert split_virtual(tmp_path / "plain.jpg") is None
    (tmp_path / "a.mp4").write_bytes(b"x")
    (tmp_path / ".hidden.mov").write_bytes(b"x")
    assert scan_videos(tmp_path) == [tmp_path / "a.mp4"]
    assert scan_folder(tmp_path) == []  # videos are never photo entries


def test_nms_and_chi2():
    ts = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    sc = [0.2, 0.9, 0.8, 0.3, 0.7, 0.95]
    assert pv.nms_peaks(ts, sc, 1) == [5]
    # 0.95@2.5, 0.9@0.5; 0.8@1.0 and 0.7@2.0 are within 1 s of a pick; 0.3@1.5 is 1 s from both
    assert pv.nms_peaks(ts, sc, 3) == [1, 3, 5]
    assert pv.nms_peaks(ts, sc, 9) == [1, 3, 5]
    a = np.array([1.0, 0.0], np.float32)
    assert pv.chi2(a, a) == 0.0
    assert pv.chi2(a, a[::-1]) == pytest.approx(1.0, abs=1e-6)


def test_seek_accuracy_within_one_frame(tmp_path):
    make_index_video(tmp_path / "idx.mp4", 90)
    with iov.VideoReader(tmp_path / "idx.mp4", 256) as vr:
        assert abs(vr.info.fps - FPS) < 0.5 and vr.info.n_frames == 90
        for want in (0, 7, 31, 45, 60, 88):
            got = read_index(vr.frame_at(want / FPS))
            assert abs(got - want) <= 1, (want, got)
        seq = [read_index(f) for _, f in vr.samples(0.5)]
        assert len(seq) == 6 and all(abs(g - 15 * k) <= 1 for k, g in enumerate(seq))


def test_unreadable_video_raises(tmp_path):
    (tmp_path / "bad.mp4").write_bytes(b"not a video")
    with pytest.raises(iov.VideoError):
        iov.VideoReader(tmp_path / "bad.mp4")


# ---- pipeline -------------------------------------------------------------------------------


@pytest.mark.models
@pytest.mark.usefixtures("models_ready", "fake_landmarks")
def test_picks_fall_in_sharp_open_spans(shoot):
    folder, spans = shoot
    res = run_cull(folder, NO_AUDIT)
    vids = [r for r in res.images if split_virtual(r.master_path)]
    assert len(vids) >= 3  # >= 1 per scene (3 scene cuts found)
    assert {int(split_virtual(r.master_path)[1] // SCENE_S) for r in vids} == {0, 1, 2}
    for r in vids:
        t = split_virtual(r.master_path)[1]
        assert _in_open_span(t, spans), (t, r.reasons)
    # one group per scene segment, one best per group, burst_id == segment
    assert len({r.group_id for r in vids}) == 3
    assert sum(r.is_best_in_group for r in vids) == 3
    assert all(r.decision == "keep" for r in vids if r.is_best_in_group)
    assert res.stats["video"]["picks"] == len(vids)
    assert res.stats["video"]["n_videos"] == 1


@pytest.mark.models
@pytest.mark.usefixtures("models_ready")
def test_video_warm_run_identical_and_cached(shoot):
    folder, _ = shoot
    cold = run_cull(folder, NO_AUDIT)
    warm = run_cull(folder, NO_AUDIT)
    sig = lambda res: [  # noqa: E731
        (r.master_path.name, r.decision, tuple(r.reasons), r.group_id, r.burst_id, r.score)
        for r in res.images
    ]
    assert sig(cold) == sig(warm)
    assert warm.stats["video"]["n_scanned"] == 0  # served from the cache
    with Cache.for_folder(folder) as c:
        assert any(p for p in c.load_decisions() if split_virtual(p))


@pytest.mark.models
@pytest.mark.usefixtures("models_ready")
def test_video_is_opt_in_and_photos_unchanged(shoot, shoot_dir):
    folder, _ = shoot
    for f in shoot_dir.glob("*.jpg"):
        shutil.copy(f, folder / f.name)
    off = run_cull(folder, replace(NO_AUDIT, video=False))
    assert not any(split_virtual(r.master_path) for r in off.images)
    assert "video" not in off.stats
    on = run_cull(folder, NO_AUDIT)
    photos = lambda res: [  # noqa: E731
        (r.master_path.name, r.decision, tuple(r.reasons), r.group_id)
        for r in res.images
        if not split_virtual(r.master_path)
    ]
    assert photos(off) == photos(on)


@pytest.mark.models
@pytest.mark.usefixtures("models_ready")
def test_export_frame_exif_time(shoot, monkeypatch):
    folder, _ = shoot
    monkeypatch.setattr(pv, "creation_time", lambda p: 1717221600.0)
    dest = pv.export_frame(folder, virtual_path(folder / "clip.mp4", 12.345))
    assert dest == folder / "frame_grabs" / "clip_t012345.jpg"
    meta = read_meta(dest)
    assert meta.ts_source == "exif"
    assert meta.timestamp == pytest.approx(1717221600.0 + 12.345, abs=0.01)
    assert (meta.width, meta.height) == (640, 640)  # full-resolution decode
    assert [e.master.name for e in scan_folder(folder)] == ["clip_t012345.jpg"]  # a normal image
    assert (folder / "clip.mp4").exists()


@pytest.mark.models
@pytest.mark.usefixtures("models_ready")
def test_api_and_cli(shoot, monkeypatch):
    from fastapi.testclient import TestClient

    from shotsieve.cli import main
    from shotsieve.server.app import create_app

    folder, _ = shoot
    assert main(["cull", str(folder), "--video", "--export-keeps"]) == 0
    grabs = sorted((folder / "frame_grabs").glob("*.jpg"))
    assert len(grabs) == 3  # one keep per scene segment

    client = TestClient(create_app(Settings(audit_enabled=False), folder))
    shoot_json = client.get("/api/shoot", params={"folder": str(folder)}).json()
    vids = [i for i in shoot_json["images"] if i["video_t"] is not None]
    assert len(vids) >= 3
    t = client.get(f"/api/thumb/{vids[0]['id']}", params={"folder": str(folder)})
    assert t.status_code == 200 and t.headers["content-type"] == "image/jpeg"
    g = client.post("/api/video/grab", json={"image_id": vids[0]["id"], "folder": str(folder)})
    assert g.status_code == 200 and (folder / g.json()["rel"]).exists()
    # a photo id is not a video frame
    r = client.post("/api/video/grab", json={"image_id": 10_000_000, "folder": str(folder)})
    assert r.status_code == 404
    # XMP is never written for virtual items
    x = client.post("/api/xmp/write", json={"folder": str(folder)})
    assert x.status_code == 200 and not list(folder.glob("*#t=*"))



@pytest.mark.models
@pytest.mark.usefixtures("models_ready")
def test_api_cull_with_video_flag(shoot):
    import time

    from fastapi.testclient import TestClient

    from shotsieve.server.app import create_app

    folder, _ = shoot
    client = TestClient(create_app(Settings(audit_enabled=False, search_index_auto=False), folder))
    job = client.post("/api/cull", json={"folder": str(folder), "video": True}).json()["job_id"]
    for _ in range(300):
        j = client.get(f"/api/jobs/{job}").json()
        if j["state"] in ("done", "error"):
            break
        time.sleep(0.2)
    assert j["state"] == "done", j
    images = client.get("/api/shoot", params={"folder": str(folder)}).json()["images"]
    assert sum(i["video_t"] is not None for i in images) >= 3


# ---- Review #4 M5: HEVC goes through AVFoundation (VideoToolbox) on macOS ---------------------
import sys  # noqa: E402
import time  # noqa: E402

import cv2  # noqa: E402

darwin_only = pytest.mark.skipif(sys.platform != "darwin", reason="AVFoundation backend")


def _write_av(path: Path, fourcc: str, n: int = 90, size=(256, 64), textured=False, fps=FPS):
    """Encode a clip through cv2's AVFoundation writer (the only encoder in the wheel)."""
    w = cv2.VideoWriter(str(path), cv2.CAP_AVFOUNDATION, cv2.VideoWriter_fourcc(*fourcc), fps, size)
    assert w.isOpened(), fourcc
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (size[1] // 8, size[0] // 8, 3), dtype=np.uint8)
    base = cv2.resize(base, size, interpolation=cv2.INTER_CUBIC)
    for i in range(n):
        if textured:
            f = np.roll(base, i * 7, axis=1)
        else:
            f = np.zeros((size[1], size[0], 3), np.uint8)
            for b in range(8):
                if (i >> b) & 1:
                    f[:, b * 32 : (b + 1) * 32] = 255
        w.write(f)
    w.release()


@darwin_only
def test_codec_probe_and_backend_choice(tmp_path):
    _write_av(tmp_path / "h.mp4", "hvc1")
    _write_av(tmp_path / "a.mp4", "avc1")
    assert iov.codec_of(tmp_path / "h.mp4") in iov.HEVC_CODECS
    assert iov.codec_of(tmp_path / "a.mp4") not in iov.HEVC_CODECS
    with iov.VideoReader(tmp_path / "h.mp4", 256) as vr:
        assert vr.backend == "avfoundation"
    with iov.VideoReader(tmp_path / "a.mp4", 256) as vr:
        assert vr.backend == "ffmpeg"  # FFmpeg stays first where it is faster (H.264)


@darwin_only
def test_hevc_seek_and_sequential_accuracy(tmp_path):
    _write_av(tmp_path / "h.mp4", "hvc1", 90)
    with iov.VideoReader(tmp_path / "h.mp4", 256) as vr:
        assert vr.backend == "avfoundation" and vr.info.n_frames == 90
        for want in (0, 7, 31, 45, 60, 88):
            got = read_index(vr.frame_at(want / FPS))
            assert abs(got - want) <= 1, (want, got)
        seq = [read_index(f) for _, f in vr.samples(0.5)]
        assert len(seq) == 6 and all(abs(g - 15 * k) <= 1 for k, g in enumerate(seq))


@darwin_only
def test_backend_env_override_still_wins(tmp_path, monkeypatch):
    _write_av(tmp_path / "h.mp4", "hvc1")
    monkeypatch.setenv("SHOTSIEVE_VIDEO_BACKEND", "ffmpeg")
    with iov.VideoReader(tmp_path / "h.mp4", 256) as vr:
        assert vr.backend == "ffmpeg"


BENCH = Path(__file__).resolve().parents[1] / "data" / "bench"


@darwin_only
@pytest.mark.parametrize("name", ["video_4k30_hevc.mp4", "video_4k30.mp4"])
def test_bench_4k30_clips_decode_at_least_4x_realtime(name):
    clip = BENCH / name
    if not clip.exists():
        pytest.skip("run scripts/benchmark.py once to generate the 4K30 clips")
    with iov.VideoReader(clip, 640) as vr:
        t0 = time.perf_counter()
        n = sum(1 for _ in vr.samples(1.0))
        dt = time.perf_counter() - t0
        secs = vr.info.duration
    assert n >= 3 and secs / dt >= 4.0, f"{name} [{vr.backend}]: {secs / dt:.1f}x realtime"


@pytest.mark.models
@pytest.mark.usefixtures("models_ready")
def test_real_landmarker_on_decoded_video_frame(tmp_path, sources):
    """No fake: a real face photo encoded to mp4 is decoded and read by the real detector and
    blendshape model. (The repo has no real closed-eye photo, so the closed case stays covered
    by the fake-landmarker span test above.)"""
    import cv2

    from shotsieve.models.face_detect import get_face_detector
    from shotsieve.models.face_landmarks import get_face_landmarker

    img = cv2.resize(cv2.imread(str(sources / "portrait2.jpg")), (640, 640))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = cv2.VideoWriter(str(tmp_path / "face.mp4"), fourcc, FPS, (640, 640))
    for _ in range(FPS):
        w.write(img)
    w.release()
    with iov.VideoReader(tmp_path / "face.mp4", 640) as vr:
        frame = vr.frame_at(0.5)
    face = get_face_detector().detect(frame)[0]
    ex = get_face_landmarker().analyze(frame, face)
    assert ex.found and ex.eyes_closed < 0.5


# ---- Review #5 m2: capture time from the container's mvhd, not the copy time -----------------
def _set_mvhd(path: Path, epoch: int, version: int = 0) -> None:
    data = bytearray(path.read_bytes())
    i = data.index(b"mvhd") + 4
    assert data[i] == 0, "synthetic mp4v writes a version-0 mvhd"
    mac = epoch + iov._MAC_EPOCH_OFFSET
    data[i + 4 : i + 8] = mac.to_bytes(4, "big")
    path.write_bytes(bytes(data))


def test_mvhd_creation_time_beats_file_birth_time(tmp_path):
    make_index_video(tmp_path / "c.mp4", 30)
    assert iov.creation_time(tmp_path / "c.mp4") > 1.75e9 - 1e8  # just written: copy time
    _set_mvhd(tmp_path / "c.mp4", 1717221600)  # 2024-06-01 06:00:00 UTC
    assert iov.mvhd_creation_time(tmp_path / "c.mp4") == 1717221600.0
    assert iov.creation_time(tmp_path / "c.mp4") == 1717221600.0


def test_mvhd_zero_or_garbage_falls_back(tmp_path):
    make_index_video(tmp_path / "z.mp4", 30)
    data = bytearray((tmp_path / "z.mp4").read_bytes())
    i = data.index(b"mvhd") + 8
    data[i : i + 4] = b"\0\0\0\0"
    (tmp_path / "z.mp4").write_bytes(bytes(data))
    assert iov.mvhd_creation_time(tmp_path / "z.mp4") is None
    (tmp_path / "bad.mp4").write_bytes(b"not a video")
    assert iov.mvhd_creation_time(tmp_path / "bad.mp4") is None
    assert iov.creation_time(tmp_path / "z.mp4") > 0


def test_export_frame_exif_uses_mvhd_plus_offset(tmp_path):
    """No monkeypatch: the real mvhd of the file drives DateTimeOriginal and SubSec."""
    make_index_video(tmp_path / "m.mp4", 90)
    _set_mvhd(tmp_path / "m.mp4", 1717221600)
    dest = pv.export_frame(tmp_path, virtual_path(tmp_path / "m.mp4", 1.5))
    meta = read_meta(dest)
    assert meta.ts_source == "exif"
    assert meta.timestamp == pytest.approx(1717221601.5, abs=0.01)
    import piexif

    ex = piexif.load(str(dest))["Exif"]
    assert ex[piexif.ExifIFD.SubSecTimeOriginal] == b"500"


# ---- B2.5: a real closed-eye photo (CC0) through the real landmarker, no fake -------------------
CLOSED_DIR = Path(__file__).parent / "fixtures" / "closed_eyes_cc0"
CLOSED_SPAN = (2.0, 3.0)  # seconds: open 0-2, closed 2-3, open 3-5


def _closed_eye_video(path: Path, sources: Path) -> None:
    import cv2

    w, h = 640, 960

    def cover(im):
        s = max(w / im.shape[1], h / im.shape[0])
        im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        y, x = (im.shape[0] - h) // 2, (im.shape[1] - w) // 2
        return im[y : y + h, x : x + w]

    open_ = cover(cv2.imread(str(sources / "portrait2.jpg")))
    closed = cover(cv2.imread(str(CLOSED_DIR / "closed_eyes.jpg")))
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    for i in range(5 * FPS):
        vw.write(closed if CLOSED_SPAN[0] <= i / FPS < CLOSED_SPAN[1] else open_)
    vw.release()


def _frame_reasons(rgb, settings):
    from shotsieve.features.record import build_vector, compute_norm
    from shotsieve.scoring import quality

    raw = pv._analyse(rgb, settings)
    assert raw.faces, "the real detector must find the face"
    fv = build_vector(
        raw, compute_norm([raw]), eyes_thresh=settings.eyes_closed_thresh,
        eyes_both_thresh=settings.eyes_closed_both_thresh,
    )  # fmt: skip
    return quality.assess(fv, settings, pv._flags(raw, settings)).hard_fails


@pytest.mark.models
@pytest.mark.usefixtures("models_ready")
def test_real_closed_eyes_in_video_are_flagged_and_never_picked(tmp_path, sources, monkeypatch):
    import hashlib

    import cv2

    assert hashlib.sha256((CLOSED_DIR / "closed_eyes.jpg").read_bytes()).hexdigest() == (
        "2fe2bbb576a55e2655ff814751ab3b125bc95062ceaad66ebc73b0d55f71820d"
    )  # pinned, see SOURCE.txt
    settings = NO_AUDIT
    clip = tmp_path / "closed.mp4"
    _closed_eye_video(clip, sources)
    # the open-eye source is a different person, so the cut would isolate the closed clip into
    # its own scene segment (and force one pick there); this test is about the eye penalty
    monkeypatch.setattr(pv, "CUT_CHI2", 1.01)

    # the still photo itself must be read as closed by the real model (else pick another photo)
    still = cv2.cvtColor(cv2.imread(str(CLOSED_DIR / "closed_eyes.jpg")), cv2.COLOR_BGR2RGB)
    assert "eyes_closed" in _frame_reasons(still, settings)

    with iov.VideoReader(clip, 960) as vr:
        for t in (2.25, 2.5, 2.75):
            assert "eyes_closed" in _frame_reasons(vr.frame_at(t), settings), t
        for t in (0.5, 1.5, 3.5, 4.5):
            assert "eyes_closed" not in _frame_reasons(vr.frame_at(t), settings), t

    picks, _ = pv.scan_video(clip, settings)
    assert picks
    for d in picks:
        assert not (CLOSED_SPAN[0] - 1e-6 <= d["t"] < CLOSED_SPAN[1]), d["t"]

    dest = tmp_path / "shoot"
    dest.mkdir()
    shutil.copy(clip, dest / "closed.mp4")
    res = run_cull(dest, settings)
    vids = [split_virtual(r.master_path)[1] for r in res.images if split_virtual(r.master_path)]
    assert vids and all(not (CLOSED_SPAN[0] <= t < CLOSED_SPAN[1]) for t in vids)


@pytest.mark.models
@pytest.mark.usefixtures("models_ready")
def test_closed_eye_clip_in_its_own_scene_segment_is_never_a_clean_keep(tmp_path, sources):
    """m4, unpatched: the different-person closed clip forms its own scene segment, so one pick
    is forced there. Pipeline rule: it carries `eyes_closed` and the cull never marks it keep."""
    settings = NO_AUDIT
    clip = tmp_path / "closed.mp4"
    _closed_eye_video(clip, sources)
    picks, _ = pv.scan_video(clip, settings)
    inside = [d for d in picks if CLOSED_SPAN[0] - 1e-6 <= d["t"] < CLOSED_SPAN[1]]
    assert inside, "the closed clip is its own segment: one pick is forced there"
    dest = tmp_path / "shoot"
    dest.mkdir()
    shutil.copy(clip, dest / "closed.mp4")
    res = run_cull(dest, settings)
    seen = 0
    for r in res.images:
        v = split_virtual(r.master_path)
        if v and CLOSED_SPAN[0] <= v[1] < CLOSED_SPAN[1]:
            seen += 1
            assert "eyes_closed" in r.reasons and r.decision != "keep", (v[1], r.decision)
            assert r.decision == "review"  # forced pick is surfaced for a human, not rejected
    assert seen == len(inside)
