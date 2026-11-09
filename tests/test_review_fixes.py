"""Regression tests for the findings in docs/REVIEW.md (ids in the test names)."""

import importlib.util
import math
import shutil
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from shotsieve.config import Settings
from shotsieve.features import sharpness
from shotsieve.features.vector import FEATURE_NAMES, FeatureVector
from shotsieve.grouping.bursts import TimeItem
from shotsieve.grouping.duplicates import MAX_COMPONENT, group_duplicates
from shotsieve.io import xmp
from shotsieve.personal import audit, calibrate, dataset
from shotsieve.pipeline import evaluate
from shotsieve.pipeline.cache import Cache, CachedImage
from shotsieve.scoring import quality, reasons, selection

S = Settings()
ROOT = Path(__file__).resolve().parents[1]


def good(**kw):
    base = dict(
        n_faces=1.0, main_face_area=0.05, sharp_global_norm=0.8, face_sharp_norm_main=0.8,
        face_sharp_min_important=0.8, eyes_closed_main=0.05, eyes_closed_max_important=0.05,
        n_eyes_closed_important=0.0, exp_ev=0.0, exp_mean=110.0, exp_clip_hi=0.01,
        exp_clip_lo=0.01, smile_main=0.2,
    )  # fmt: skip
    base.update(kw)
    return FeatureVector(**base)


def cand(name, group, fv, p, flags=()):
    return selection.Candidate(name, group, fv, quality.assess(fv, S, flags), p)


# ---- B1: scale-invariant face focus ----------------------------------------------------


def test_B1_face_sharpness_scale_invariant(rgb):
    gray = cv2.cvtColor(rgb("portrait2.jpg"), cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    vals, blurred = [], []
    for size in (600, 300, 200, 150, 120, 100):
        s = size / min(h, w)
        g = cv2.resize(gray, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        box = (0, 0, g.shape[1], g.shape[0])
        vals.append(sharpness.face_sharpness(g, box))
        b = cv2.GaussianBlur(g, (0, 0), max(1.0, size / 60))
        blurred.append(sharpness.face_sharpness(b, box))
    assert max(vals) / min(vals) < 1.6  # the old measure varied 2.4x over this range
    assert max(blurred) * 5 < min(vals)  # ... while real blur stays far below
    small = cv2.resize(gray, (60, 73), interpolation=cv2.INTER_AREA)
    assert math.isnan(sharpness.face_sharpness(small, (0, 0, 60, 73)))  # too small to judge


def test_B1_unjudgeable_small_face_is_never_a_focus_reject():
    fv = good(face_sharp_norm_main=math.nan, main_face_area=0.003)
    assert "face_out_of_focus" not in quality.hard_fails(fv, S)
    soft = good(face_sharp_norm_main=0.1, sharp_global_norm=0.6, main_face_area=0.01)
    assert quality.assess(soft, S).tiers["face_out_of_focus"] == quality.SOFT
    d = selection.decide([cand("a", 0, soft, 0.7)], S, 0.6, 0.3)[0]
    assert d.decision == "review"  # small-face focus: review, not a hard reject


def test_B1_smooth_backdrop_is_not_blur(tmp_path):
    from shotsieve.features.record import RawFeatures, compute_norm

    recs = [RawFeatures(lap_var=v) for v in (3000.0, 2500.0, 2000.0, 90.0)]
    ref = compute_norm(recs).lap_ref
    assert ref <= sharpness.LAP_REF_CAP
    assert 90.0 / ref > 0.15  # a sharp frame of smooth content must not read as blurred


@pytest.mark.models
@pytest.mark.usefixtures("models_ready")
def test_B1_in_focus_face_at_five_scales_never_out_of_focus(tmp_path, gen_dir, make_fixtures):
    from shotsieve.pipeline.runner import run_cull

    rgb = make_fixtures.load_rgb
    bg = cv2.resize(rgb(gen_dir / "src" / "building.jpg"), (1600, 1067))
    face = rgb(gen_dir / "src" / "portrait2.jpg")
    folder = tmp_path / "scales"
    folder.mkdir()
    for k, frac in enumerate((0.30, 0.20, 0.12, 0.06, 0.03)):
        fh = int(1067 * frac * 3)
        fw = int(face.shape[1] * fh / face.shape[0])
        patch = cv2.resize(face, (fw, fh), interpolation=cv2.INTER_AREA)
        img = bg.copy()
        x0, y0 = 800 - fw // 2, 530 - fh // 2
        img[y0 : y0 + fh, x0 : x0 + fw] = patch
        ts = 1_700_000_000 + 600 * k
        make_fixtures.save_jpeg(img, folder / f"s{k}.jpg", make_fixtures.exif_bytes(ts, None))
    res = run_cull(folder, Settings(audit_enabled=False))
    for r in res.images:
        assert "face_out_of_focus" not in r.reasons, (r.path.name, r.reasons)
        assert r.decision != "reject", (r.path.name, r.reasons)


def test_B1_benchmark_records_ground_truth(tmp_path):
    spec = importlib.util.spec_from_file_location("bench", ROOT / "scripts" / "benchmark.py")
    bench = importlib.util.module_from_spec(spec)
    sys.modules["bench"] = bench
    spec.loader.exec_module(bench)
    truth = bench.synthesize(24, tmp_path / "b", seed=3)
    assert (tmp_path / "b" / bench.TRUTH_FILE).exists() and len(truth) == 24
    assert {t["degraded"] for t in truth} <= {"", "blur", "dark", "black"}
    imgs = [
        SimpleNamespace(master_path=tmp_path / "b" / t["file"], decision="reject",
                        reasons=["blur"], group_id=i)
        for i, t in enumerate(truth)
    ]  # fmt: skip
    rep = bench.score_truth(imgs, truth)
    assert rep["clean_hard_rejects"] == rep["n_clean"] > 0  # all clean frames rejected -> flagged
    assert rep["moment_no_keep_rate"] == 1.0


# ---- B2 / tiers / M6 / M7 / M8 ---------------------------------------------------------


def test_B2_margins_move_borderline_rejects_into_review():
    best = good()
    sib = good(eyes_closed_main=0.8, eyes_closed_max_important=0.8)  # firm blink
    wink = good(eyes_closed_main=0.8, eyes_closed_max_important=0.8)
    cs = [
        cand("best.jpg", 0, best, 0.90),
        cand("dup.jpg", 0, good(), 0.80),  # duplicate 0.10 below best
        cand("blink.jpg", 0, sib, 0.78),  # failed sibling
        cand("solo.jpg", 1, wink, 0.50),  # firm fail, singleton
        cand("black.jpg", 2, good(exp_mean=2.0, n_faces=0.0), 0.5),  # certain
    ]
    base = selection.decide(cs, S, 0.6, 0.3)
    assert [d.decision for d in base] == ["keep", "reject", "reject", "reject", "reject"]
    cal = selection.decide(cs, S, 0.6, 0.3, hard_margin=0.15, dup_margin=0.15)
    assert [d.decision for d in cal] == ["keep", "review", "review", "review", "reject"]
    assert cal[4].reasons == ["accidental"]  # certain rules never move


def test_B2_calibrate_levers_by_stratum():
    th = {"t_keep": 0.6, "t_reject": 0.3}
    r = calibrate.calibrate([calibrate.AuditSample(0.2, True, "dup")], 20, th)
    assert r.lowered and r.thresholds["dup_margin"] > 0 and r.thresholds["t_reject"] == 0.3
    r = calibrate.calibrate([calibrate.AuditSample(0.2, True, "hard")], 20, th)
    assert r.thresholds["hard_margin"] > 0
    r = calibrate.calibrate([calibrate.AuditSample(0.2, True, "certain")], 20, th)
    assert not r.lowered and r.by_stratum["certain"]["est_missed"] > 0
    assert r.rate_ci[0] <= r.missed_keeper_rate <= r.rate_ci[1]  # m2: a CI is reported


def test_tiers_certain_firm_soft():
    t = lambda fv, flags=(): quality.classify_fails(fv, S, flags)  # noqa: E731
    assert t(good(sharp_global_norm=0.02, face_sharp_norm_main=0.02))["blur"] == quality.CERTAIN
    assert t(good(exp_mean=2.0, n_faces=0.0)) == {"accidental": quality.CERTAIN}
    assert t(good(), ["flash_misfire"])["flash_misfire"] == quality.CERTAIN
    both = good(eyes_closed_main=0.9, eyes_closed_max_important=0.9)
    assert t(both)["eyes_closed"] == quality.FIRM
    assert t(both, ["eyes_single"])["eyes_closed"] == quality.SOFT  # M6: a wink is soft
    assert t(good(exp_clip_hi=0.5))["exposure_bad"] == quality.SOFT  # M7: high key


def test_M7_white_backdrop_with_well_exposed_subject_is_fine():
    fv = good(exp_clip_hi=0.5)
    assert "exposure_bad" not in quality.hard_fails(fv, S, ["subject_well_exposed"])


def test_M8_trained_model_may_override_firm_fail_but_not_certain():
    blink = good(eyes_closed_main=0.9, eyes_closed_max_important=0.9)
    c = [
        cand("a.jpg", 0, blink, 0.8),
        cand("a2.jpg", 0, good(), 0.95),
    ]  # not the best, has a better sibling
    assert selection.decide(c, S, 0.6, 0.3)[0].decision == "reject"
    assert selection.decide(c, S, 0.6, 0.3, trained=True)[0].decision == "review"
    cert = [cand("b.jpg", 0, good(), 0.9, ["flash_misfire"])]
    assert selection.decide(cert, S, 0.6, 0.3, trained=True)[0].decision == "reject"


def test_m4_unknown_eyes_not_open():
    fv = good(eyes_closed_main=math.nan, eyes_closed_max_important=math.nan)
    assert quality.sub_scores(fv)["eyes"] < 1.0


def test_m5_duplicate_never_cites_a_rejected_best():
    best = cand("best.jpg", 0, good(), 0.1)
    sib = cand("sib.jpg", 0, good(), 0.05)
    d = selection.decide([best, sib], S, 0.6, 0.3)
    assert "duplicate" not in d[1].reasons and "duplicate_tie" not in d[1].reasons


def test_m11_long_exposure_dark_frame_is_only_soft():
    fv = good(n_faces=0.0, exp_mean=3.0, shutter_log2=3.0)
    assert quality.classify_fails(fv, S) == {"accidental": quality.SOFT}


# ---- M1: overrides separate from model decisions -------------------------------------------


@pytest.mark.models
@pytest.mark.usefixtures("models_ready")
def test_M1_override_survives_rerun_and_u_restores_model(tmp_path, shoot_dir):
    from fastapi.testclient import TestClient

    from shotsieve.pipeline.runner import run_cull
    from shotsieve.server.app import create_app

    folder = tmp_path / "s"
    shutil.copytree(shoot_dir, folder)
    cfg = Settings(audit_enabled=False)
    first = run_cull(folder, cfg)
    target = next(r for r in first.images if r.model_decision == "reject")
    with Cache.for_folder(folder) as c:
        c.set_override(str(target.master_path), "keep", 1.0, "ui")
    again = run_cull(folder, cfg)
    r = next(x for x in again.images if x.master_path == target.master_path)
    assert r.decision == "keep" and r.overridden and r.model_decision == "reject"
    with Cache.for_folder(folder) as c:
        assert c.load_decisions()[str(target.master_path)]["decision"] == "reject"
        assert str(target.master_path) in c.overrides()
    with TestClient(create_app(cfg, default_folder=folder)) as cl:
        shoot = cl.get("/api/shoot").json()
        iid = next(i["id"] for i in shoot["images"] if i["name"] == target.master_path.name)
        out = cl.post("/api/override", json={"image_id": iid, "decision": "model"}).json()
        assert out["decision"] == "reject" and out["overridden"] is False


def test_m1_audit_pool_skips_overridden():
    def img(name, over):
        return SimpleNamespace(master_path=Path(name), decision="reject", score=0.1, audit=False,
                               reasons=["blur"], reason_texts=[], features=None,
                               overridden=over, audit_prob=None)  # fmt: skip

    imgs = [img(f"{i}.jpg", i < 20) for i in range(60)]
    picked = audit.sample_audit(imgs, 0)
    assert picked and not any(p.overridden for p in picked)


# ---- M2 / M3 / M9: XMP -----------------------------------------------------------------


def test_M2_foreign_rating_and_label_are_never_overwritten(tmp_path):
    m = tmp_path / "a.nef"
    xmp.write_sidecar(m, rating=5, label="Purple", record=False)  # the photographer's own
    out = xmp.WriteOutcome()
    xmp.write_decision(m, S, "reject", ["blur"], 0.1, outcome=out)
    d = xmp.read_sidecar(m)
    assert (d.rating, d.label, d.decision) == (5, "Purple", "reject")
    assert out.rating_kept and out.label_kept
    xmp.write_decision(m, S, "keep", [], 0.9, override={"rating": 4})  # explicit override
    d = xmp.read_sidecar(m)
    assert (d.rating, d.label) == (4, "Green")


def test_M2_own_values_are_updated_on_rerun(tmp_path):
    m = tmp_path / "a.nef"
    xmp.write_decision(m, S, "keep", [], 0.9)
    xmp.write_decision(m, S, "reject", ["blur"], 0.1)
    d = xmp.read_sidecar(m)
    assert d.label == "Red" and d.written_label == "Red"


def test_M9_foreign_ns1_prefix_does_not_crash(tmp_path):
    sp = tmp_path / "a.xmp"
    sp.write_text(
        '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF '
        'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description rdf:about="" xmlns:ns1="http://example.com/ns1/" ns1:Foo="bar"/>'
        "</rdf:RDF></x:xmpmeta>"
    )
    xmp.write_sidecar(tmp_path / "a.jpg", rating=3, label="Green")
    text = sp.read_text()
    assert "http://example.com/ns1/" in text and "bar" in text
    assert xmp.read_sidecar(tmp_path / "a.jpg").rating == 3


def test_M9_api_xmp_write_survives_one_bad_sidecar(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from shotsieve.server import app as appmod

    folder = tmp_path / "s"
    folder.mkdir()
    for n in ("a", "b"):
        (folder / f"{n}.jpg").write_bytes(b"x")
    rows = [
        dict(path=str(folder / f"{n}.jpg"), decision="keep", reasons=[], score=0.9, prior=0.9,
             group_id=i, burst_id=i, is_best=True, audit=False, audit_prob=None, features={})
        for i, n in enumerate("ab")
    ]  # fmt: skip
    with Cache.for_folder(folder) as c:
        c.save_decisions(rows)
    real = appmod.write_decision

    def flaky(path, *a, **k):
        if Path(path).name == "a.jpg":
            raise ValueError("Prefix format reserved for internal use")
        return real(path, *a, **k)

    monkeypatch.setattr(appmod, "write_decision", flaky)
    with TestClient(appmod.create_app(S, default_folder=folder)) as cl:
        r = cl.post("/api/xmp/write", json={"folder": str(folder)})
    assert r.status_code == 200 and r.json()["written"] == 1 and r.json()["failed"] == 1
    assert r.json()["warnings"]  # M11: Lightroom limitation is surfaced


def test_M3_shotsieve_output_is_not_ground_truth(tmp_path):
    for n in "abc":
        (tmp_path / f"{n}.jpg").write_bytes(b"x")
    xmp.write_decision(tmp_path / "a.jpg", S, "keep", [], 0.9)
    xmp.write_decision(tmp_path / "b.jpg", S, "reject", [], 0.1)
    assert dataset.labels_from_xmp_folder(tmp_path) == []
    assert evaluate.truth_from_xmp(tmp_path, S) == {}
    # a legacy sidecar (no provenance fields) is recognised by its decision + default scheme
    xmp.write_sidecar(tmp_path / "c.jpg", rating=3, label="Green", decision="keep", record=False)
    assert dataset.labels_from_xmp_folder(tmp_path) == []
    # the user changing the rating afterwards makes it a real label
    xmp.write_sidecar(tmp_path / "a.jpg", rating=1, override=True, record=False)
    labs = dataset.labels_from_xmp_folder(tmp_path)
    assert [Path(x.path).name for x in labs] == ["a.jpg"] and labs[0].keep


def test_M11_lightroom_warning_and_help(capsys):
    assert xmp.lightroom_warning(["a.jpg", "b.dng"]) and "Lightroom" in xmp.LIGHTROOM_NOTE
    assert xmp.lightroom_warning(["a.cr2", "b.nef"]) is None
    from shotsieve import cli

    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["cull", "--help"])
    assert "Lightroom" in capsys.readouterr().out


# ---- eval: group-level metric ------------------------------------------------------------


def test_eval_group_level_missed_keeper_metric():
    def im(name, dec, g):
        return SimpleNamespace(path=Path(f"/x/{name}"), master_path=Path(f"/x/{name}"),
                               decision=dec, group_id=g)  # fmt: skip

    images = [im("a", "keep", 0), im("b", "reject", 0), im("c", "reject", 1), im("d", "review", 2),
              im("e", "reject", 2), im("f", "keep", 3)]  # fmt: skip
    truth = {f"/x/{n}": True for n in "abcdf"} | {"/x/e": False}
    rep = evaluate.evaluate(images, truth)
    assert rep.missed_keepers == 2  # frame level: b and c
    assert rep.group_missed_keepers == 2 and rep.group_review_rescuable == 1  # c, d
    assert rep.group_missed_keeper_rate == pytest.approx(2 / 5)
    assert rep.n_kept_groups == 4 and rep.moment_misses == 2
    assert rep.moment_review_rescuable == 1


def test_m6_norm_path_case_insensitive_on_mac(tmp_path):
    if sys.platform not in ("darwin", "win32"):
        pytest.skip("case-insensitive volumes only")
    assert evaluate.norm_path("/Tmp/Foo.JPG") == evaluate.norm_path("/tmp/foo.jpg")


# ---- M4 / m3: cache identity -------------------------------------------------------------


def _rec():
    from shotsieve.features.record import RawFeatures
    from shotsieve.io.exif import ImageMeta

    return CachedImage(
        ImageMeta(timestamp=0.0), RawFeatures(), np.ones(384, np.float32) / 20, np.zeros((0, 128))
    )


def test_M4_cache_and_overrides_survive_folder_move(tmp_path):
    a = tmp_path / "old"
    a.mkdir()
    (a / "x.jpg").write_bytes(b"x" * 1000)
    with Cache.for_folder(a) as c:
        c.put(str(a / "x.jpg"), 1000, 5, _rec(), "fp", "ex")
        c.save_decisions([dict(path=str(a / "x.jpg"), decision="reject", reasons=["blur"],
                               score=0.1, prior=0.1, group_id=0, burst_id=0, is_best=True,
                               audit=False, audit_prob=None, features={})])  # fmt: skip
        c.set_override(str(a / "x.jpg"), "keep")
        iid = c.ids_for([str(a / "x.jpg")])[str(a / "x.jpg")]
        c.commit()
    b = tmp_path / "new"
    a.rename(b)
    with Cache.for_folder(b) as c:
        hit = c.lookup_many({str(b / "x.jpg"): (1000, 5)}, "ex")
        assert list(hit) == [str(b / "x.jpg")]
        assert list(c.overrides()) == [str(b / "x.jpg")]
        assert list(c.load_decisions()) == [str(b / "x.jpg")]
        assert c.path_for_id(iid) == str(b / "x.jpg")


def test_M4_content_fingerprint_survives_mtime_reset(tmp_path):
    from shotsieve.pipeline.cache import fingerprint

    f = tmp_path / "x.jpg"
    f.write_bytes(bytes(range(256)) * 2000)
    with Cache.for_folder(tmp_path) as c:
        c.put(str(f), f.stat().st_size, 5, _rec(), fingerprint(f), "ex")
        size = f.stat().st_size
        assert c.lookup_many({str(f): (size, 999)}, "ex") == {}  # no fingerprint -> miss
        hit = c.lookup_many({str(f): (size, 999)}, "ex", lambda p: fingerprint(p))
        assert str(f) in hit
        f.write_bytes(bytes(reversed(f.read_bytes())))  # same size, different content
        assert c.lookup_many({str(f): (size, 999)}, "ex", lambda p: fingerprint(p)) == {}


def test_m3_extractor_fingerprint_invalidates_cache(tmp_path):
    (tmp_path / "x.jpg").write_bytes(b"x")
    with Cache.for_folder(tmp_path) as c:
        c.put(str(tmp_path / "x.jpg"), 1, 5, _rec(), "", "v1")
        st = {str(tmp_path / "x.jpg"): (1, 5)}
        assert c.lookup_many(st, "v1") and c.lookup_many(st, "v2") == {}
    from shotsieve.pipeline.runner import extractor_fingerprint

    assert extractor_fingerprint(Settings()) != extractor_fingerprint(
        Settings(preview_max_edge=640)
    )


# ---- M5: chaining ----------------------------------------------------------------------


def test_M5_long_chain_never_collapses_into_one_group():
    n = MAX_COMPONENT + 200
    rng = np.random.default_rng(0)
    v = rng.normal(size=384)
    step = rng.normal(size=384)
    emb = []
    for i in range(n):  # slow drift: neighbours identical, ends far apart
        x = v + step * (i / n) * 3.0
        emb.append(x / np.linalg.norm(x))
    emb = np.array(emb, np.float32)
    assert 1 - float(emb[0] @ emb[-1]) > 0.5
    items = [TimeItem("a", i * 0.05) for i in range(n)]
    ids = group_duplicates(emb, items, None, window_s=30.0).group_ids
    assert ids[0] != ids[-1] and len(set(ids)) > 3
    sizes = np.bincount(ids)
    assert sizes.max() < n / 2


# ---- M10: ingest ---------------------------------------------------------------------------


def _jpg(path, seed):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.random.default_rng(seed).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_M10_by_date_name_collisions_get_suffix_not_overwrite(tmp_path):
    from shotsieve.ingest import copier

    src = tmp_path / "card"
    _jpg(src / "100CANON" / "IMG_0001.jpg", 1)
    _jpg(src / "101CANON" / "IMG_0001.jpg", 2)
    dest = tmp_path / "out"
    r = copier.ingest(src, dest, by_date=True)
    assert len(r.copied) == 2 and not r.failed
    names = sorted(p.name for p in dest.rglob("*.jpg"))
    assert names == ["IMG_0001.jpg", "IMG_0001_1.jpg"]
    assert not list(dest.rglob("*.part"))
    r2 = copier.ingest(src, dest, by_date=True)  # idempotent
    assert len(r2.skipped) == 2 and not r2.copied and not r2.failed


def test_m10_verify_pass_can_bypass_page_cache(tmp_path):
    from shotsieve.ingest import copier

    f = tmp_path / "x.bin"
    f.write_bytes(b"abc" * 1000)
    assert copier._hash_file(f, nocache=True) == copier._hash_file(f)


# ---- M12 / m7 / m8 / m9 ------------------------------------------------------------------


def test_M12_per_user_reject_bar_and_review_fraction():
    from shotsieve.personal.model import PersonalModel

    rng = np.random.default_rng(0)
    n = 600
    X = rng.normal(size=(n, len(FEATURE_NAMES))).astype(np.float32)
    y = (X[:, 0] + 0.3 * rng.normal(size=n) > 0.85).astype(int)  # a photographer keeping ~20 %
    m = PersonalModel().fit(X, y)
    assert m.trained and 0.02 <= m.thresholds["t_reject"] <= 0.30
    p = m.predict_proba(X)
    keepers = p[y == 1]
    assert (keepers < m.thresholds["t_reject"]).mean() <= 0.03
    review = ((p >= m.thresholds["t_reject"]) & (p < 0.6)).mean()
    assert review <= 0.20


def test_m7_embedded_jpeg_decoded_reduced():
    from shotsieve.io.preview import decode_embedded_jpeg

    img = np.random.default_rng(0).integers(0, 255, (1200, 1800, 3), dtype=np.uint8)
    data = cv2.imencode(".jpg", img)[1].tobytes()
    out = decode_embedded_jpeg(data, 400)
    assert max(out.shape[:2]) == 900 // 1 // 1 or max(out.shape[:2]) >= 400
    assert max(out.shape[:2]) < 1800  # reduced decode (1/2 here), not full size


def test_m8_get_shoot_does_not_change_server_folder_and_calibration_blocks_cull(tmp_path):
    from fastapi.testclient import TestClient

    from shotsieve.server.app import create_app

    a, b = tmp_path / "a", tmp_path / "b"
    for f in (a, b):
        f.mkdir()
        with Cache.for_folder(f) as c:
            c.save_decisions([dict(path=str(f / "x.jpg"), decision="keep", reasons=[], score=1,
                                   prior=1, group_id=0, burst_id=0, is_best=True, audit=False,
                                   audit_prob=None, features={})])  # fmt: skip
    app = create_app(S, default_folder=a)
    with TestClient(app) as cl:
        assert cl.get("/api/shoot", params={"folder": str(b)}).status_code == 200
        assert app.state.shotsieve.current_folder == a
        app.state.shotsieve.calibrating.add(a)
        assert app.state.shotsieve.busy_with(a)
        r = cl.post("/api/audit/complete", json={"folder": str(a)})
        assert r.status_code == 409


def test_m2_audit_complete_refused_while_audits_pending(tmp_path):
    from fastapi.testclient import TestClient

    from shotsieve.server.app import create_app

    f = tmp_path / "s"
    f.mkdir()
    row = dict(path=str(f / "x.jpg"), decision="review", reasons=["audit", "blur"], score=0.1,
               prior=0.1, group_id=0, burst_id=0, is_best=True, audit=True, audit_prob=0.5,
               features={}, klass="hard")  # fmt: skip
    with Cache.for_folder(f) as c:
        c.save_decisions([row])
    with TestClient(create_app(S, default_folder=f)) as cl:
        r = cl.post("/api/audit/complete", json={"folder": str(f)})
    assert r.status_code == 409 and "not judged" in r.json()["detail"]


def test_m9_locked_catalog_falls_back_to_copy(tmp_path, gen_dir):
    from shotsieve.io.lrcat import read_catalog

    cat = tmp_path / "c.lrcat"
    shutil.copy(gen_dir / "catalog.lrcat", cat)
    expected = read_catalog(cat)
    lock = sqlite3.connect(cat, isolation_level=None)
    lock.execute("BEGIN EXCLUSIVE")
    try:
        assert read_catalog(cat) == expected
    finally:
        lock.execute("ROLLBACK")
        lock.close()


# ---- T5: moment safeguard -----------------------------------------------------------------


def test_T5_best_of_group_blink_firm_fail_goes_to_review_not_reject():
    blink = good(eyes_closed_main=0.9, eyes_closed_max_important=0.9)  # firm fail
    c = [cand("a.jpg", 0, blink, 0.5), cand("b.jpg", 0, blink, 0.4)]
    out = selection.decide(c, S, 0.6, 0.3)
    best = [d for d in out if d.is_best]
    assert len(best) == 1 and best[0].decision == "review"
    assert best[0].reasons[0] == "best_available"
    text = reasons.describe("best_available", None, best[0].ctx)
    assert text.startswith("Best available frame of this moment, but eyes are closed")
    assert out[1].decision == "reject"  # the non-best failed sibling stays rejectable


def test_T5_best_of_group_low_score_goes_to_review():
    c = [cand("a.jpg", 0, good(), 0.1), cand("b.jpg", 0, good(), 0.05)]
    out = selection.decide(c, S, 0.6, 0.3)
    assert out[0].is_best and out[0].decision == "review"
    assert "low" in reasons.describe("best_available", None, out[0].ctx)
    assert sum(d.is_best for d in out) == 1


def test_T5_certain_fail_best_may_still_be_rejected():
    c = [cand("a.jpg", 0, good(), 0.9, ["flash_misfire"]), cand("b.jpg", 0, good(), 0.8,
         ["flash_misfire"])]  # fmt: skip
    out = selection.decide(c, S, 0.6, 0.3)
    assert out[0].is_best and out[0].decision == "reject"


def test_T5_singletons_unaffected():
    out = selection.decide([cand("a.jpg", 0, good(), 0.1)], S, 0.6, 0.3)
    assert out[0].decision == "reject"


def test_T5_identical_embeddings_do_not_crash_linkage():
    """Byte-identical frames can give cosine distance ~ -1e-7; clustering must clamp it."""
    rng = np.random.default_rng(0)
    v = (rng.normal(size=384) * 3.0).astype(np.float32)
    v = (v / np.float32(np.linalg.norm(v)) * np.float32(1.0000002)).astype(np.float32)
    emb = np.stack([v, v, v]).astype(np.float32)
    items = [TimeItem("cam", 1000.0 + i, "exif") for i in range(3)]
    res = group_duplicates(emb, items)
    assert len(set(res.group_ids)) == 1


# ---- Review #2: N1-N4 ------------------------------------------------------------------------


def _bench():
    spec = importlib.util.spec_from_file_location("bench", ROOT / "scripts" / "benchmark.py")
    bench = importlib.util.module_from_spec(spec)
    sys.modules["bench"] = bench
    spec.loader.exec_module(bench)
    return bench


def test_N1_dup_lever_is_proportional_bounded_and_fine():
    th = {"t_keep": 0.6, "t_reject": 0.3}
    small = calibrate.calibrate([calibrate.AuditSample(0.5, True, "dup")], 100, th)
    big = calibrate.calibrate([calibrate.AuditSample(0.1, True, "dup")] * 20, 20, th)
    assert 0 < small.thresholds["dup_margin"] <= 0.02  # tiny share -> tiny step (was 0.05)
    assert small.thresholds["dup_margin"] < big.thresholds["dup_margin"] <= 0.05  # proportional
    # the budget caps the growth per round at +10 points, and at the 20 % ceiling in total
    r = calibrate.calibrate(
        [calibrate.AuditSample(0.1, True, "dup")] * 20, 20, th, simulate=lambda t: 0.03
    )
    assert r.thresholds["dup_review_budget"] == pytest.approx(0.10)
    r = calibrate.calibrate(
        [calibrate.AuditSample(0.1, True, "dup")] * 20, 20, th, simulate=lambda t: 0.15
    )
    assert r.thresholds["dup_review_budget"] == pytest.approx(0.05)  # 0.20 - 0.15
    r = calibrate.calibrate(
        [calibrate.AuditSample(0.1, True, "dup")] * 20, 20, th, simulate=lambda t: 0.25
    )
    assert r.thresholds["dup_margin"] == 0.0 and not r.lowered  # already over the ceiling


def test_N1_decide_budget_keeps_only_the_closest_siblings():
    cs = [cand("best.jpg", 0, good(), 0.90)] + [
        cand(f"s{k}.jpg", 0, good(), 0.89 - 0.001 * k) for k in range(8)
    ]
    free = selection.decide(cs, S, 0.6, 0.3, dup_margin=0.05)
    assert sum(d.decision == "review" for d in free) == 7  # the cliff
    capped = selection.decide(cs, S, 0.6, 0.3, dup_margin=0.05, dup_review_budget=3 / 9)
    assert [d.decision for d in capped[2:]] == ["review"] * 3 + ["reject"] * 4
    assert capped[4].klass == "" and capped[5].klass == "dup"


def test_N1_calibration_keeps_review_bounded_on_benchmark_like_shoot(tmp_path):
    from shotsieve.pipeline.runner import run_cull

    folder = tmp_path / "shoot"
    _bench().synthesize(150, folder, seed=7)
    cfg = Settings(audit_enabled=False)
    base = run_cull(folder, cfg).stats
    r0 = base["counts"]["review"] / base["n_images"]
    th = {"t_keep": cfg.t_keep, "t_reject": cfg.t_reject}
    dup_keeps = [calibrate.AuditSample(0.1, True, "dup")] * 20  # strong signal: ask for the max
    with Cache.for_folder(folder) as cache:
        sim = calibrate._simulator(cache, folder, cfg)
        uncapped = calibrate.calibrate(dup_keeps, 20, th)
        overshoot = sim({**th, "dup_margin": uncapped.thresholds["dup_margin"]})
        res = calibrate.calibrate(dup_keeps, 20, th, simulate=sim)
        cache.set_thresholds({k: res.thresholds[k] for k in ("dup_margin", "dup_review_budget")})
    after = run_cull(folder, cfg).stats
    r1 = after["counts"]["review"] / after["n_images"]
    print(f"review {r0:.3f} -> unbounded {overshoot:.3f} -> bounded {r1:.3f}")
    assert overshoot > 0.20  # without the bound the lever overshoots the ceiling
    assert res.thresholds["dup_margin"] > 0 and r1 > r0  # ...and the lever still does work
    assert r1 <= min(r0 + calibrate.MAX_REVIEW_GROWTH, calibrate.MAX_REVIEW) + 1e-9
    assert r1 <= 0.20


def test_N2_eval_gate_uses_group_level_review_as_saved(tmp_path, monkeypatch, capsys):
    from shotsieve import cli
    from shotsieve.pipeline import runner

    def im(name, dec, g, audit=False):
        return SimpleNamespace(path=Path(f"/x/{name}"), master_path=Path(f"/x/{name}"),
                               decision=dec, group_id=g, audit=audit)  # fmt: skip

    imgs = [im("a", "keep", 0), im("b", "review", 1), im("c", "review", 2), im("d", "keep", 3)]
    truth = {f"/x/{n}": True for n in "abcd"}
    rep = evaluate.evaluate(imgs, truth)
    assert rep.group_missed_keeper_rate == pytest.approx(0.5)  # strict: review = miss
    assert rep.group_gate_rate == 0.0  # gate: review = saved
    monkeypatch.setattr(evaluate, "truth_from_xmp", lambda *a, **k: truth)
    monkeypatch.setattr(cli, "_check_models", lambda: None)
    monkeypatch.setattr(runner, "run_cull", lambda *a, **k: SimpleNamespace(images=imgs))
    args = cli.build_parser().parse_args(
        ["eval", str(tmp_path), "--truth", "xmp", "--gate", "0.01"]
    )
    assert cli.cmd_eval(args) == 0  # passes although strict is 50 %
    out = capsys.readouterr().out
    assert "strict" in out
    imgs2 = [im("a", "reject", 0), im("b", "keep", 1)]
    monkeypatch.setattr(runner, "run_cull", lambda *a, **k: SimpleNamespace(images=imgs2))
    monkeypatch.setattr(evaluate, "truth_from_xmp", lambda *a, **k: {"/x/a": True, "/x/b": True})
    assert cli.cmd_eval(args) == 3  # a real miss still fails the gate


def test_N3_audit_review_does_not_rescue_a_group():
    def im(name, dec, g, audit=False):
        return SimpleNamespace(path=Path(f"/x/{name}"), master_path=Path(f"/x/{name}"),
                               decision=dec, group_id=g, audit=audit)  # fmt: skip

    imgs = [im("a", "review", 0, audit=True), im("b", "review", 1)]
    rep = evaluate.evaluate(imgs, {"/x/a": True, "/x/b": True})
    assert rep.group_missed_keepers == 2 and rep.group_review_rescuable == 1
    assert rep.group_gate_rate == pytest.approx(0.5)
    assert rep.moment_review_rescuable == 1
    # the benchmark scorer applies the same rule
    bench = _bench()
    ims = [SimpleNamespace(master_path=Path("/x/a.jpg"), decision="review", audit=True,
                           reasons=[], group_id=0)]  # fmt: skip
    gt = bench.score_truth(ims, [{"file": "a.jpg", "moment": 0, "degraded": ""}])
    assert gt["moment_group_lost"] == 1


def test_N4_firm_fail_singleton_goes_to_review_like_grouped_frames():
    blink = good(eyes_closed_main=0.9, eyes_closed_max_important=0.9)  # firm fail
    near = selection.decide([cand("a.jpg", 0, blink, 0.58)], S, 0.6, 0.3)
    assert near[0].decision == "review"  # p 0.58 >= t_soft 0.55
    far = selection.decide([cand("a.jpg", 0, blink, 0.50)], S, 0.6, 0.3)
    assert far[0].decision == "reject"  # clearly bad stays a reject
    black = good(exp_mean=2.0, n_faces=0.0)
    assert selection.decide([cand("b.jpg", 0, black, 0.5)], S, 0.6, 0.3)[0].decision == "reject"


def test_N5_repeat_audit_complete_idempotent(tmp_path):
    f = tmp_path / "s"
    f.mkdir()

    def row(name, dec, audit=False, reasons=()):
        return dict(path=str(f / name), decision=dec, reasons=list(reasons), score=0.2, prior=0.2,
                    group_id=0, burst_id=0, is_best=True, audit=audit,
                    audit_prob=0.25 if audit else None, features={},
                    klass="score" if audit else None)  # fmt: skip

    with Cache.for_folder(f) as c:
        c.save_decisions([row("a.jpg", "review", True, ["audit"]), row("k.jpg", "keep")])
        c.set_audit_sample({str(f / "a.jpg"): {"p": 0.25, "k": "score"}})
        c.set_override(str(f / "a.jpg"), "keep", 1.0, "ui", None)
    r1, _ = calibrate.calibrate_shoot(f, S, redecide=False)
    assert r1.applied and r1.n_audited == 1 and r1.n_kept == 1
    with Cache.for_folder(f) as c:  # a re-decide: the overridden image no longer carries `audit`
        c.save_decisions([row("a.jpg", "keep"), row("k.jpg", "keep")])
        cal, th = c.get_meta("calibration"), c.get_thresholds()
    r2, _ = calibrate.calibrate_shoot(f, S, redecide=False)
    with Cache.for_folder(f) as c:
        assert c.get_meta("calibration") == cal and c.get_thresholds() == th
    assert r2.to_dict() == r1.to_dict()


def test_N5_zero_samples_leave_calibration_untouched(tmp_path):
    f = tmp_path / "s"
    f.mkdir()
    with Cache.for_folder(f) as c:
        c.set_meta("calibration", {"x": 1})
        c.set_thresholds({"t_reject": 0.2})
    r, cull = calibrate.calibrate_shoot(f, S, redecide=False)
    assert not r.applied and cull is None and r.thresholds["t_reject"] == 0.2
    with Cache.for_folder(f) as c:
        assert c.get_meta("calibration") == {"x": 1}
