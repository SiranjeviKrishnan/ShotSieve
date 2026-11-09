import math

import numpy as np
import pytest

from shotsieve.config import Settings
from shotsieve.features.vector import FeatureVector
from shotsieve.scoring import quality, reasons, selection
from shotsieve.scoring.selection import Candidate, blend, blend_weight, decide

S = Settings()


def good(**kw) -> FeatureVector:
    base = dict(
        sharp_global=200, sharp_global_norm=0.9, tenengrad_norm=0.9, exp_mean=120, exp_clip_hi=0.0,
        exp_clip_lo=0.0, exp_ev=0.0, n_faces=1, main_face_area=0.05, main_face_centrality=0.9,
        face_sharp_main=100, face_sharp_min_important=0.9, face_sharp_norm_main=0.9,
        eyes_closed_main=0.05, eyes_closed_max_important=0.05, n_eyes_closed_important=0,
        smile_main=0.3, face_det_score_main=0.95,
    )  # fmt: skip
    base.update(kw)
    return FeatureVector(**base)


# ---- quality ----------------------------------------------------------------------------


def test_good_frame_scores_high_no_fails():
    q = quality.assess(good(), S)
    assert q.score > 0.85 and q.hard_fails == []
    assert set(q.subs) == {"sharp", "eyes", "exposure", "subject", "expression"}
    assert all(0 <= v <= 1 for v in q.subs.values())
    assert sum(quality.WEIGHTS.values()) == pytest.approx(1.0)


def test_blur_rule():
    q = quality.assess(good(sharp_global_norm=0.05, face_sharp_norm_main=0.05,
                            face_sharp_min_important=0.05), S)  # fmt: skip
    assert "blur" in q.hard_fails and q.score < quality.assess(good(), S).score
    # soft global but a sharp face is not 'blur'
    assert "blur" not in quality.assess(good(sharp_global_norm=0.05), S).hard_fails
    nof = good(n_faces=0, face_sharp_norm_main=math.nan, sharp_global_norm=0.1)
    assert "blur" in quality.assess(nof, S).hard_fails


def test_face_out_of_focus_rule():
    q = quality.assess(good(face_sharp_norm_main=0.1, face_sharp_min_important=0.1,
                            sharp_global_norm=0.8), S)  # fmt: skip
    assert q.hard_fails == ["face_out_of_focus"]


def test_eyes_closed_rules():
    assert "eyes_closed" in quality.assess(good(eyes_closed_main=0.8), S).hard_fails
    assert "eyes_closed" not in quality.assess(good(eyes_closed_main=0.4), S).hard_fails
    group = good(n_faces=4, eyes_closed_main=0.1, n_eyes_closed_important=2)
    assert "eyes_closed" in quality.assess(group, S).hard_fails  # >= 50% of faces
    group = good(n_faces=4, eyes_closed_main=0.1, n_eyes_closed_important=1)
    assert "eyes_closed" not in quality.assess(group, S).hard_fails
    unknown = good(eyes_closed_main=math.nan, eyes_closed_max_important=math.nan)
    q = quality.assess(unknown, S)
    assert "eyes_closed" not in q.hard_fails
    assert q.subs["eyes"] == quality.UNKNOWN_EYES < 1.0  # unknown is not "open"


def test_exposure_rules():
    assert "exposure_bad" in quality.assess(good(exp_ev=-3.0, exp_mean=20), S).hard_fails
    assert "exposure_bad" in quality.assess(good(exp_ev=2.8), S).hard_fails
    assert "exposure_bad" in quality.assess(good(exp_clip_hi=0.3), S).hard_fails
    assert "exposure_bad" in quality.assess(good(exp_clip_lo=0.5), S).hard_fails
    # a night scene: dark mean but real highlights -> not 'bad exposure'
    night = good(exp_ev=-3.0, exp_mean=30, exp_clip_hi=0.03, exp_clip_lo=0.3)
    assert "exposure_bad" not in quality.assess(night, S).hard_fails
    dark_score = quality.assess(good(exp_ev=-2.0), S).subs["exposure"]
    assert dark_score < quality.assess(good(), S).subs["exposure"]


def test_accidental_and_flash_flags():
    black = good(n_faces=0, main_face_area=0, exp_mean=0.0, exp_ev=-17, sharp_global_norm=0.0,
                 face_sharp_norm_main=math.nan, eyes_closed_main=math.nan,
                 eyes_closed_max_important=math.nan)  # fmt: skip
    assert quality.assess(black, S).hard_fails == ["accidental"]
    q = quality.assess(good(), S, flags=["flash_misfire"])
    assert q.hard_fails == ["flash_misfire"]
    # a dark misfire is reported as misfire (+ exposure), never as blur
    dark = good(exp_ev=-3.0, exp_mean=15, sharp_global_norm=0.05, face_sharp_norm_main=0.05)
    fails = quality.assess(dark, S, flags=["flash_misfire"]).hard_fails
    assert "flash_misfire" in fails and "exposure_bad" in fails and "blur" not in fails


def test_no_faces_landscape_is_fine():
    land = good(n_faces=0, main_face_area=0, face_sharp_norm_main=math.nan,
                eyes_closed_main=math.nan, eyes_closed_max_important=math.nan)  # fmt: skip
    q = quality.assess(land, S)
    assert q.hard_fails == [] and q.score > 0.6


def test_weakest_area():
    assert quality.weakest_area({"sharp": 0.2, "eyes": 1, "exposure": 1, "subject": 1,
                                 "expression": 1}) == "sharpness"  # fmt: skip


# ---- reasons ----------------------------------------------------------------------------


def test_every_code_has_plain_text():
    for code in reasons.REASON_LABELS:
        text = reasons.describe(code, good(), {"best": "IMG_1.jpg", "weakest": "sharpness"})
        assert isinstance(text, str) and len(text) > 15 and not text.startswith(code)
    assert "IMG_1.jpg" in reasons.describe("duplicate", None, {"best": "IMG_1.jpg"})
    assert "dark" in reasons.describe("exposure_bad", good(exp_ev=-3.2)).lower()
    assert "bright" in reasons.describe("exposure_bad", good(exp_ev=2.9)).lower()
    assert "EV" in reasons.describe("flash_misfire", None, {"flash_delta_ev": -2.0})
    assert set(reasons.REJECT_CODES) <= set(reasons.REASON_LABELS)
    with pytest.raises(KeyError):
        reasons.describe("nope")
    assert len(reasons.describe_all(["blur", "duplicate"])) == 2


# ---- selection --------------------------------------------------------------------------


def cand(name, gid, p, fv=None, flags=()):
    fv = fv or good()
    return Candidate(name, gid, fv, quality.assess(fv, S, flags), p)


def test_blend_weight_and_blend():
    assert blend_weight(0) == 0 and blend_weight(200) == 0.5
    assert blend_weight(800) == pytest.approx(0.8)
    prior, pers = np.array([0.2, 0.8]), np.array([1.0, 0.0])
    assert np.allclose(blend(prior, pers, 200), [0.6, 0.4])
    assert np.allclose(blend(prior, None, 200), prior) and np.allclose(blend(prior, pers, 0), prior)


class _Model:
    n_train = 100

    def __init__(self, two_col=False):
        self.two_col, self.emb = two_col, None

    def predict_proba(self, X, embeddings=None):
        self.emb = embeddings
        p = np.full(len(X), 0.9)
        return np.stack([1 - p, p], 1) if self.two_col else p


def test_personal_probabilities_shapes_and_protocol():
    X = np.zeros((3, 25), np.float32)
    m = _Model()
    assert isinstance(m, selection.PersonalModelLike)
    assert selection.personal_probabilities(m, X).shape == (3,)
    emb = np.zeros((3, 384), np.float32)
    assert np.allclose(selection.personal_probabilities(_Model(True), X, emb), 0.9)
    m2 = _Model()
    selection.personal_probabilities(m2, X, emb)
    assert m2.emb is emb

    class Plain:
        n_train = 1

        def predict_proba(self, X):
            return np.full(len(X), 0.5)

    assert selection.personal_probabilities(Plain(), X, emb).shape == (3,)


def test_exactly_one_best_per_group_and_duplicates_rejected():
    cands = [cand("a", 0, 0.95), cand("b", 0, 0.80), cand("c", 0, 0.70), cand("s", 1, 0.9)]
    d = decide(cands, S, 0.6, 0.3)
    assert [x.is_best for x in d] == [True, False, False, True]
    assert d[0].decision == "keep" and d[3].decision == "keep"
    assert d[1].decision == d[2].decision == "reject" and d[1].reasons == ["duplicate"]
    assert d[1].ctx["best"] == "a"


def test_tie_is_rejected_unless_best_is_uncertain_and_hard_fail_is_not_best():
    cands = [cand("a", 0, 0.90), cand("b", 0, 0.89)]
    d = decide(cands, S, 0.6, 0.3)
    assert d[0].decision == "keep" and d[1].decision == "reject"
    assert d[1].reasons == ["duplicate_tie"] and d[1].ctx["best"] == "a"
    text = reasons.describe("duplicate_tie", None, d[1].ctx)
    assert text == "Near-identical to the best frame a; kept the sharper/eyes-open one."
    # best frame itself borderline -> the near-identical sibling goes to review
    d = decide([cand("a", 0, 0.50), cand("b", 0, 0.49)], S, 0.6, 0.3)
    assert d[0].decision == "review" and d[1].decision == "review"
    assert d[1].reasons == ["duplicate_tie"] and d[1].ctx["best_uncertain"]
    assert "borderline" in reasons.describe("duplicate_tie", None, d[1].ctx)
    # clearly worse sibling is a plain duplicate
    d = decide([cand("a", 0, 0.90), cand("b", 0, 0.60)], S, 0.6, 0.3)
    assert d[1].reasons == ["duplicate"]

    blurry = good(sharp_global_norm=0.02, face_sharp_norm_main=0.02, face_sharp_min_important=0.02)
    cands = [cand("blur", 0, 0.99, blurry), cand("ok", 0, 0.7)]  # higher score but hard-failed
    d = decide(cands, S, 0.6, 0.3)
    assert d[1].is_best and d[1].decision == "keep"
    assert d[0].decision == "reject" and d[0].reasons == ["blur"] and not d[0].is_best


def test_all_failed_group_still_has_one_best():
    bad = good(exp_ev=-3.5, exp_mean=10)
    d = decide([cand("a", 0, 0.5, bad), cand("b", 0, 0.4, bad)], S, 0.6, 0.3)
    assert sum(x.is_best for x in d) == 1 and all(x.decision == "reject" for x in d)


def test_thresholds_and_reasons_for_score_rejects():
    d = decide([cand("hi", 0, 0.7), cand("mid", 1, 0.45), cand("lo", 2, 0.1)], S, 0.6, 0.3)
    assert [x.decision for x in d] == ["keep", "review", "reject"]
    assert d[2].reasons == ["low_quality"] and d[1].reasons == ["uncertain"]
    nof = good(n_faces=0)
    assert decide([cand("n", 0, 0.1, nof)], S, 0.6, 0.3)[0].reasons == ["no_subject"]


def test_every_reject_has_reason_text_and_keep_per_group():
    cands = [cand(f"{i}", 0, 0.95 - 0.1 * i) for i in range(9)]
    d = decide(cands, S, 0.6, 0.3)
    assert sum(x.decision == "keep" for x in d) == 2  # groups of >= 8 keep two
    assert sum(x.is_best for x in d) == 1
    for x in d:
        if x.decision == "reject":
            assert x.reasons and all(reasons.describe(c, None, x.ctx) for c in x.reasons)
