"""Multi-camera clock offset: algorithm (synthetic embeddings), persistence, API, CLI, pipeline."""

import shutil
import threading
import time
from pathlib import Path

import numpy as np
import piexif
import psutil
import pytest
from fastapi.testclient import TestClient

from shotsieve.config import Settings
from shotsieve.features.record import RawFeatures
from shotsieve.grouping import clock
from shotsieve.grouping.clock import ClockImage
from shotsieve.io.exif import ImageMeta
from shotsieve.pipeline.cache import Cache, CachedImage
from shotsieve.server.app import create_app

T0 = 1_700_000_000.0
DT = 3725.0  # camera B's clock runs ahead of A by this much


def unit(rng, n, d=384):
    v = rng.normal(size=(n, d)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def noisy(rng, v, cos=0.9):
    z = unit(rng, len(v), v.shape[1])
    out = cos * v + np.sqrt(1 - cos**2) * z
    return (out / np.linalg.norm(out, axis=1, keepdims=True)).astype(np.float32)


def two_cams(rng, n=60, matched=40, delta=DT, jitter=0.3, tz=(None, None), extra=10):
    """A: n frames over 5 min. B: `matched` copies of A's scenes (clock +delta) + `extra` others
    (fewer frames than A, so A is the reference)."""
    ts = np.sort(rng.uniform(0, 300, n))
    ea = unit(rng, n)
    pick = rng.choice(n, matched, replace=False)
    eb = np.vstack([noisy(rng, ea[pick]), unit(rng, extra)])
    tb = np.concatenate([ts[pick] + delta + rng.uniform(-jitter, jitter, matched),
                         rng.uniform(0, 300, extra) + delta])  # fmt: skip
    imgs = [ClockImage(f"a{i}.jpg", "A", T0 + t, tz_offset_s=tz[0]) for i, t in enumerate(ts)]
    imgs += [ClockImage(f"b{i}.jpg", "B", T0 + t, tz_offset_s=tz[1]) for i, t in enumerate(tb)]
    return imgs, np.vstack([ea, eb])


def test_recovers_offset_within_one_second():
    imgs, emb = two_cams(np.random.default_rng(0))
    (s,) = clock.suggest_offsets(imgs, emb)
    assert s.serial == "B" and s.ref_serial == "A"
    # B's clock is DT ahead, so the shift that aligns it is -DT (t_adj = t + offset)
    assert abs(s.offset_s + DT) <= 1.0
    assert s.confidence >= 0.5 and s.support >= 10 and s.auto_apply
    assert s.pair is not None and s.pair.cos >= 0.65 and abs(s.pair.delta_s + DT) <= 3.5


def test_unrelated_scenes_give_no_offset():
    rng = np.random.default_rng(1)
    imgs = [ClockImage(f"a{i}", "A", T0 + i * 5) for i in range(50)]
    imgs += [ClockImage(f"b{i}", "B", T0 + DT + i * 5) for i in range(40)]
    (s,) = clock.suggest_offsets(imgs, unit(rng, 90))
    assert s.offset_s is None and not s.auto_apply and s.alternatives == []


def test_similar_scenes_at_inconsistent_times_offer_top5_pairs_only():
    rng = np.random.default_rng(2)
    base = unit(rng, 30)
    emb = np.vstack([base, noisy(rng, base)])
    ts = rng.uniform(0, 36000, 60)  # same scenes, but times that agree on no offset
    imgs = [ClockImage(f"a{i}", "A", T0 + ts[i]) for i in range(30)]
    imgs += [ClockImage(f"b{i}", "B", T0 + ts[30 + i]) for i in range(30)]
    (s,) = clock.suggest_offsets(imgs, emb)
    assert s.offset_s is None and len(s.alternatives) == 5
    assert [p.cos for p in s.alternatives] == sorted((p.cos for p in s.alternatives), reverse=True)


def test_mtime_and_single_camera_are_ignored():
    rng = np.random.default_rng(3)
    imgs, emb = two_cams(rng)
    mt = [ClockImage(i.path, i.serial, i.ts, "mtime") for i in imgs if i.serial == "B"]
    only_a = [i for i in imgs if i.serial == "A"] + mt
    assert clock.suggest_offsets(only_a, emb) == []
    assert clock.suggest_offsets([i for i in imgs if i.serial == "A"], emb[:60]) == []


def test_time_zone_prior_is_honoured_even_with_thin_support():
    rng = np.random.default_rng(4)
    n = 30
    ea, ts = unit(rng, n), np.sort(rng.uniform(0, 600, n))
    eb = np.vstack([noisy(rng, ea[:2]), unit(rng, 8)])  # only two matching frames
    tb = np.concatenate([ts[:2] + 3600 + 0.2, rng.uniform(0, 600, 8) + 3600])
    imgs = [ClockImage(f"a{i}", "A", T0 + t, tz_offset_s=0.0) for i, t in enumerate(ts)]
    imgs += [ClockImage(f"b{i}", "B", T0 + t, tz_offset_s=3600.0) for i, t in enumerate(tb)]
    (s,) = clock.suggest_offsets(imgs, np.vstack([ea, eb]))
    assert s.source == "tz" and abs(s.offset_s + 3600) <= 1.0 and not s.auto_apply
    # without the prior the same data has no offset
    plain = [ClockImage(i.path, i.serial, i.ts) for i in imgs]
    assert clock.suggest_offsets(plain, np.vstack([ea, eb]))[0].offset_s is None


def test_face_bonus_ranks_pair_with_matching_faces_first():
    rng = np.random.default_rng(5)
    imgs, emb = two_cams(rng, n=20, matched=15, extra=2)
    face = rng.normal(size=(1, 128)).astype(np.float32)
    face /= np.linalg.norm(face)
    faces = [None] * len(imgs)
    inside = [i for i, im in enumerate(imgs) if im.serial == "B"][0]
    faces[inside] = face  # B's first frame and its A twin share a face
    twin = int(np.argmax(emb[:20] @ emb[inside]))
    faces[twin] = face
    (s,) = clock.suggest_offsets(imgs, emb, faces)
    assert s.offset_s is not None
    assert s.pair.path == imgs[inside].path or s.pair.cos >= 0.65


def test_10k_by_10k_is_fast_and_light():
    rng = np.random.default_rng(6)
    a, b = unit(rng, 10_000), unit(rng, 10_000)
    imgs = [ClockImage(f"a{i}", "A", T0 + i) for i in range(10_000)]
    imgs += [ClockImage(f"b{i}", "B", T0 + i) for i in range(10_000)]
    emb = np.vstack([a, b])
    proc, peak, stop = psutil.Process(), [0], threading.Event()
    base = proc.memory_info().rss

    def watch():
        while not stop.is_set():
            peak[0] = max(peak[0], proc.memory_info().rss)
            time.sleep(0.005)

    th = threading.Thread(target=watch)
    th.start()
    t0 = time.perf_counter()
    out = clock.suggest_offsets(imgs, emb)
    dt = time.perf_counter() - t0
    stop.set()
    th.join()
    assert out[0].offset_s is None
    assert dt < 2.0, dt
    assert (peak[0] - base) / 1e6 < 200, (peak[0] - base) / 1e6


def test_mutual_nn_matches_brute_force():
    rng = np.random.default_rng(7)
    a, b = unit(rng, 300, 16), unit(rng, 250, 16)
    ia, ib, cos = clock.mutual_nn(a, b, chunk=64)
    sims = a @ b.T
    nn = sims.argmax(axis=1)
    want = {(i, int(nn[i])) for i in range(300) if sims[:, nn[i]].argmax() == i}
    assert set(zip(ia.tolist(), ib.tolist(), strict=True)) == want
    assert np.allclose(cos, sims[ia, ib])


def test_parsers_and_helpers():
    assert clock.parse_tz("+02:00") == 7200 and clock.parse_tz("-0530") == -19800
    assert clock.parse_tz("") is None and clock.parse_tz("junk") is None
    assert clock.parse_offset("+3725") == 3725 and clock.parse_offset("-1:02:05") == -3725
    assert clock.parse_offset("+0:00:30.5") == 30.5
    assert clock.adjust(100.0, "B", {"B": -5.0}) == 95.0 and clock.adjust(None, "B", {}) is None
    raw = clock.set_offset(None, "B", 10, "manual")
    assert clock.set_offset(raw, "B", 99, "auto")["B"]["offset_s"] == 10  # manual wins
    assert clock.set_offset(raw, "B", 99, "manual")["B"]["offset_s"] == 99


# ---- persistence, API, CLI on a seeded shoot database ---------------------------------------


def seed(folder: Path, rng=None, delta=DT):
    rng = rng or np.random.default_rng(8)
    imgs, emb = two_cams(rng, n=24, matched=20, delta=delta, extra=2)
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    with Cache.for_folder(folder) as c:
        for k, (im, e) in enumerate(zip(imgs, emb, strict=True)):
            p = folder / im.path
            p.write_bytes(b"x")
            meta = ImageMeta(timestamp=im.ts, ts_source="exif", camera_serial=im.serial)
            c.put(str(p), 1, 1, CachedImage(meta, RawFeatures(), e, np.zeros((0, 128), np.float32)))
            rows.append(dict(path=str(p), decision="keep", reasons=[], score=0.9, prior=0.9,
                             group_id=k, burst_id=k, is_best=True, audit=False, audit_prob=None,
                             features={}))  # fmt: skip
        c.commit()
        c.save_decisions(rows)
    return imgs


def test_cache_persistence_manual_wins_and_t_adj(tmp_path):
    f = tmp_path / "s"
    imgs = seed(f)
    with Cache.for_folder(f) as c:
        c.set_clock_offset("B", -DT, "auto", 0.9)
        c.set_clock_offset("B", -1.0, "manual")
        c.set_clock_offset("B", -DT, "auto", 0.9)  # must not replace the manual value
    with Cache.for_folder(f) as c:  # survives a re-open
        assert c.get_clock_offsets()["B"] == {"offset_s": -1.0, "source": "manual"}
        adj = c.adjusted_timestamps()
        b0 = next(i for i in imgs if i.serial == "B")
        assert adj[str(f / b0.path)] == pytest.approx(b0.ts - 1.0)
        c.clear_clock_offset("B")
        assert c.get_clock_offsets() == {}


def test_analyse_auto_applies_confident_offset_and_reruns_keep_it(tmp_path):
    f = tmp_path / "s"
    seed(f)
    with Cache.for_folder(f) as c:
        sugs, offsets = clock.analyse(c, auto_apply=True)
        assert sugs[0].auto_apply and abs(offsets["B"]["offset_s"] + DT) <= 1
        assert offsets["B"]["source"] == "auto"
        c.set_clock_offset("B", 5.0, "manual")
        _, offsets = clock.analyse(c, auto_apply=True)  # a re-run must not clobber manual
        assert offsets["B"]["offset_s"] == 5.0


def test_api_get_post_clock_and_timeline_order(tmp_path):
    f = tmp_path / "s"
    seed(f)
    with TestClient(create_app(Settings(), default_folder=f)) as cl:
        d = cl.get("/api/clock", params={"folder": str(f)}).json()
        assert {c["serial"] for c in d["cameras"]} == {"A", "B"} and d["offsets"] == {}
        (s,) = d["suggestions"]
        assert abs(s["offset_s"] + DT) <= 1 and s["pair"]["ref_id"] != s["pair"]["id"]
        # pick the suggested pair: manual offset = t(other camera's frame) - t(this camera's)
        r = cl.post("/api/clock", json={"folder": str(f), "serial": "B",
                                        "pair": [s["pair"]["ref_id"], s["pair"]["id"]]})
        assert r.status_code == 200 and abs(r.json()["offset_s"] + DT) <= 3.5
        d = cl.get("/api/clock", params={"folder": str(f)}).json()
        assert d["offsets"]["B"]["source"] == "manual"
        # manual typed offset wins over everything and the timeline uses t_adj
        assert cl.post("/api/clock", json={"folder": str(f), "serial": "B",
                                           "offset_s": -DT}).status_code == 200
        shoot = cl.get("/api/shoot", params={"folder": str(f)}).json()
        order = [im["name"] for im in shoot["images"]]
        a_idx = [i for i, n in enumerate(order) if n.startswith("a")]
        b_idx = [i for i, n in enumerate(order) if n.startswith("b")]
        assert max(a_idx) - min(a_idx) > 0 and min(b_idx) < max(a_idx)  # interleaved, not B last
        assert cl.post("/api/clock", json={"folder": str(f), "serial": "B",
                                           "clear": True}).json()["offsets"] == {}
        bad = cl.post("/api/clock", json={"folder": str(f), "serial": "B"})
        assert bad.status_code == 400
        ids = [im["id"] for im in shoot["images"] if im["name"] in ("a0.jpg", "a1.jpg")]
        same = cl.post("/api/clock", json={"folder": str(f), "serial": "B", "pair": ids})
        assert same.status_code == 400  # both frames are from camera A


def test_cli_set_apply_and_write_xmp_time(tmp_path, capsys):
    from shotsieve.cli import main
    from shotsieve.io.xmp import sidecar_path

    f = tmp_path / "s"
    imgs = seed(f)
    assert main(["clock", str(f)]) == 0
    out = capsys.readouterr().out
    assert "suggested" in out and "camera" in out
    assert main(["clock", str(f), "--apply"]) == 0
    with Cache.for_folder(f) as c:
        assert abs(c.get_clock_offsets()["B"]["offset_s"] + DT) <= 1
    assert main(["clock", str(f), "--set", "B=-1:02:05", "--write-xmp-time"]) == 0
    with Cache.for_folder(f) as c:
        assert c.get_clock_offsets()["B"] == {"offset_s": -3725.0, "source": "manual"}
    b0 = next(i for i in imgs if i.serial == "B")
    xmp = sidecar_path(f / b0.path).read_text()
    assert "DateCreated" in xmp
    a0 = next(i for i in imgs if i.serial == "A")
    assert not sidecar_path(f / a0.path).exists()  # only re-clocked bodies are written
    assert main(["clock", str(f), "--set", "oops"]) == 2


# ---- the real pipeline: cam B = re-stamped copies of the fixture shoot --------------------------


@pytest.mark.models
@pytest.mark.usefixtures("models_ready")
def test_pipeline_finds_3725s_offset_between_two_bodies(tmp_path, shoot_dir):
    from datetime import datetime, timedelta

    from shotsieve.pipeline.runner import run_cull

    shoot = tmp_path / "shoot"
    shutil.copytree(shoot_dir, shoot)
    rng = np.random.default_rng(9)
    for src in sorted(shoot_dir.glob("burst*_*.jpg"))[:12]:
        dst = shoot / f"camB_{src.name}"
        shutil.copy(src, dst)
        ex = piexif.load(str(dst))
        raw_t = ex["Exif"][piexif.ExifIFD.DateTimeOriginal].decode()
        t = datetime.strptime(raw_t, "%Y:%m:%d %H:%M:%S")
        t += timedelta(seconds=DT + float(rng.uniform(-0.3, 0.3)))
        ex["Exif"][piexif.ExifIFD.DateTimeOriginal] = t.strftime("%Y:%m:%d %H:%M:%S").encode()
        ex["Exif"][piexif.ExifIFD.SubSecTimeOriginal] = b"0"
        ex["Exif"][piexif.ExifIFD.BodySerialNumber] = b"SN-CAM-B"
        piexif.insert(piexif.dump(ex), str(dst))
    run_cull(shoot, Settings(audit_enabled=False))
    with Cache.for_folder(shoot) as c:
        sugs, _ = clock.analyse(c, auto_apply=False)
        assert sugs and sugs[0].serial == "SN-CAM-B"
        assert abs(sugs[0].offset_s + DT) <= 1.0 and sugs[0].confidence >= 0.5
        # offsets survive a re-run of the cull (persisted in meta)
        c.set_clock_offset("SN-CAM-B", sugs[0].offset_s, "auto", sugs[0].confidence)
    run_cull(shoot, Settings(audit_enabled=False))
    with Cache.for_folder(shoot) as c:
        assert abs(c.get_clock_offsets()["SN-CAM-B"]["offset_s"] + DT) <= 1.0


# ---- Review #4 M2: absolute, serial-keyed, idempotent offsets -------------------------------


def _add_frames(folder: Path, serial: str, n: int, rng):
    with Cache.for_folder(folder) as c:
        for k in range(n):
            p = folder / f"x_{serial}{k}.jpg"
            p.write_bytes(b"x")
            meta = ImageMeta(timestamp=T0 + 10_000 + k, ts_source="exif", camera_serial=serial)
            e = unit(rng, 1)[0]
            c.put(str(p), 1, 1, CachedImage(meta, RawFeatures(), e, np.zeros((0, 128), np.float32)))
        c.commit()


def _rel_error(c: Cache, imgs, f: Path) -> float:
    """Error of B-vs-A alignment after t_adj (true offset: B is DT ahead)."""
    adj = c.adjusted_timestamps()
    a = [i for i in imgs if i.serial == "A"][0]
    b = [i for i in imgs if i.serial == "B"][0]
    shift_a = adj[str(f / a.path)] - a.ts
    shift_b = adj[str(f / b.path)] - b.ts
    return abs((shift_b - shift_a) + DT)


@pytest.mark.parametrize("forget_ref", [False, True])
def test_reference_flip_between_runs_keeps_b_vs_a_alignment(tmp_path, forget_ref):
    f = tmp_path / "s"
    imgs = seed(f)  # A 24 frames, B 22
    with Cache.for_folder(f) as c:
        clock.analyse(c, auto_apply=True)
        assert _rel_error(c, imgs, f) <= 1.0
        if forget_ref:  # e.g. a cache from before the reference was persisted
            c.set_meta("clock_ref", "")
    _add_frames(f, "B", 20, np.random.default_rng(1))  # B now has more frames: ref would flip
    with Cache.for_folder(f) as c:
        clock.analyse(c, auto_apply=True)
        assert _rel_error(c, imgs, f) <= 1.0  # was 3725 s


def test_analyse_is_idempotent_and_offsets_are_absolute(tmp_path):
    f = tmp_path / "s"
    seed(f)
    with Cache.for_folder(f) as c:
        clock.analyse(c, auto_apply=True)
        once = c.get_clock_offsets()
        adj1 = c.adjusted_timestamps()
        for _ in range(3):
            clock.analyse(c, auto_apply=True)
        assert c.get_clock_offsets() == once
        assert c.adjusted_timestamps() == adj1


def test_reference_manual_offset_is_part_of_the_suggestion():
    imgs, emb = two_cams(np.random.default_rng(0))
    (s,) = clock.suggest_offsets(imgs, emb, offsets={"A": 10.0})
    assert abs(s.offset_s - (-DT + 10.0)) <= 1.0  # B must land in A's *adjusted* domain


def test_pair_offset_includes_the_other_bodys_offset(tmp_path):
    f = tmp_path / "s"
    imgs = seed(f)
    a = next(i for i in imgs if i.serial == "A")
    b = next(i for i in imgs if i.serial == "B")
    with Cache.for_folder(f) as c:
        c.set_clock_offset("A", 100.0, "manual")
    with TestClient(create_app(Settings(), default_folder=f)) as cl:
        shoot = cl.get("/api/shoot", params={"folder": str(f)}).json()
        ids = {im["name"]: im["id"] for im in shoot["images"]}
        r = cl.post("/api/clock", json={"folder": str(f), "serial": "B",
                                        "pair": [ids[a.path], ids[b.path]]})
        assert r.status_code == 200, r.text
        assert r.json()["offset_s"] == pytest.approx(a.ts + 100.0 - b.ts)
