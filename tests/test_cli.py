"""CLI subcommands, plus train -> cull -> eval end to end on the fixtures."""

from __future__ import annotations

import io
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from shotsieve import cli
from shotsieve.config import Settings, save_settings
from shotsieve.features.vector import stack
from shotsieve.personal.model import PersonalModel
from shotsieve.pipeline.cache import Cache
from shotsieve.pipeline.runner import run_cull

SUBCOMMANDS = [
    ["models"],
    ["models", "download"],
    ["models", "check"],
    ["ingest"],
    ["cull"],
    ["train"],
    ["eval"],
    ["serve"],
]
NO_AUDIT = Settings(audit_enabled=False)


@pytest.mark.parametrize("cmd", SUBCOMMANDS, ids=lambda c: " ".join(c))
def test_help_for_every_subcommand(cmd, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main([*cmd, "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "usage: shotsieve " + cmd[0] in out


def test_top_level_help_lists_all_commands(capsys):
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    for name in ("models", "ingest", "cull", "train", "eval", "serve"):
        assert name in out
    assert cli.main(["--version"]) == 0


def test_flags_documented_in_spec_exist(capsys):
    for cmd, flags in {
        "cull": ["--write-xmp", "--json", "--config"],
        "train": ["--lrcat", "--xmp-folder", "--from-overrides"],
        "eval": ["--truth", "--lrcat"],
        "serve": ["--port"],
        "ingest": ["--backup", "--by-date"],
    }.items():
        with pytest.raises(SystemExit):
            cli.main([cmd, "--help"])
        out = capsys.readouterr().out
        assert all(f in out for f in flags), (cmd, flags)


def test_errors_exit_nonzero(tmp_path, capsys):
    assert cli.main(["train"]) == 2  # no source given
    with pytest.raises(SystemExit) as exc:
        cli.main(["cull", str(tmp_path / "missing")])
    assert exc.value.code == 2
    assert cli.main(["eval", str(tmp_path), "--truth", "lrcat"]) == 2
    assert "error" in capsys.readouterr().err


def test_ingest_command(tmp_path, shoot_dir, capsys):
    dest = tmp_path / "dest"
    args = ["ingest", str(shoot_dir), str(dest), "--backup", str(tmp_path / "bak")]
    assert cli.main(args) == 0
    assert "copied" in capsys.readouterr().out
    assert len(list(dest.glob("*.jpg"))) == 17 and len(list((tmp_path / "bak").glob("*.jpg"))) == 17
    assert cli.main(args) == 0  # re-run skips
    assert "skipped" in capsys.readouterr().out


def test_models_check_exit_code(monkeypatch, capsys):
    monkeypatch.setattr("shotsieve.models.registry.verify_model", lambda *a, **k: False)
    assert cli.main(["models", "check"]) == 1
    assert "MISSING" in capsys.readouterr().out


def test_progress_bar_writes_to_given_stream():
    class Tty(io.StringIO):
        def isatty(self):
            return True

    out = Tty()
    bar = cli.ProgressBar(out, min_interval=0)
    bar("features", 5, 10)
    bar("features", 10, 10)
    text = out.getvalue()
    assert "\r" in text and "5/10" in text and "10/10" in text and "#" in text
    plain = io.StringIO()
    bar = cli.ProgressBar(plain, min_interval=0)
    bar("embed", 3, 10)
    bar("embed", 10, 10)
    assert plain.getvalue().count("\n") == 1 and "embed" in plain.getvalue()


def test_print_table(capsys):
    cli.print_table([("a", 1), ("long name", 22)], ["k", "v"])
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == ["k", "v"] and lines[2].startswith("a ") and "long name" in lines[3]


# ---- with models ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def culled_shoot(tmp_path_factory, shoot_dir):
    dest = tmp_path_factory.mktemp("cli") / "shoot"
    shutil.copytree(shoot_dir, dest)
    return dest


@pytest.mark.models
@pytest.mark.usefixtures("models_ready")
def test_cull_json_on_fixtures(culled_shoot, tmp_path, capsys):
    out = tmp_path / "out.json"
    assert cli.main(["cull", str(culled_shoot), "--json", str(out), "--write-xmp"]) == 0
    captured = capsys.readouterr()
    assert "keep" in captured.out and "reject" in captured.out and "img/s" in captured.out
    assert "features" in captured.err  # progress on stderr, table on stdout
    data = json.loads(out.read_text())
    assert len(data["images"]) == 17
    assert data["thresholds"]["t_keep"] == 0.6 and data["thresholds"]["t_reject"] == 0.3
    first = data["images"][0]
    assert {"path", "decision", "reasons", "reason_texts", "score", "group_id", "audit"} <= set(
        first
    )
    assert (culled_shoot / "black.xmp").exists()
    for im in data["images"]:
        if im["decision"] == "reject":
            assert im["reasons"] and im["reason_texts"]


def _train_world(tmp_path_factory, shoot_dir, make_fixtures, copies=4):
    """`copies` copies of the fixture shoot (68 images) and a catalog labelling all of them."""
    root = tmp_path_factory.mktemp("train")
    manifest = json.loads((shoot_dir.parent / "shoot_manifest.json").read_text())
    entries = []
    for i in range(copies):
        folder = root / f"shoot{i}"
        shutil.copytree(shoot_dir, folder)
        for m in manifest:
            good = m["variant"] == "normal" and m["frame"] == 0
            bad = m["variant"] in {"blur", "underexposed", "black"}
            entries.append(
                {
                    "root": str(root),
                    "rel": f"shoot{i}/",
                    "base": m["file"].rsplit(".", 1)[0],
                    "ext": "jpg",
                    "pick": 1 if good else -1 if bad else 0,
                    "rating": 4 if good else None,
                }
            )
    cat = root / "cat.lrcat"
    make_fixtures.build_lrcat(cat, entries)
    return root, cat


@pytest.fixture(scope="module")
def world(tmp_path_factory, shoot_dir, make_fixtures):
    return _train_world(tmp_path_factory, shoot_dir, make_fixtures)


@pytest.mark.models
@pytest.mark.usefixtures("models_ready")
def test_train_on_synthetic_lrcat_then_cull_and_eval(world, tmp_path, capsys):
    root, cat = world
    model_path = tmp_path / "personal.joblib"
    assert cli.main(["train", "--lrcat", str(cat), "--out", str(model_path)]) == 0
    out = capsys.readouterr().out
    assert "trained" in out and "68" in out and model_path.exists()
    model = PersonalModel.load(model_path)
    assert model.trained and model.n_train == 68 and model.pca.n_components_ == 16
    # personal model in effect: cull with it through a config file
    cfg = tmp_path / "cfg.toml"
    save_settings(Settings(personal_model_path=str(model_path)), cfg)
    out_json = tmp_path / "res.json"
    folder = root / "shoot0"
    assert cli.main(["cull", str(folder), "--config", str(cfg), "--json", str(out_json)]) == 0
    capsys.readouterr()
    res = json.loads(out_json.read_text())
    assert res["stats"]["personal_n_train"] == 68
    # eval against the catalog truth
    assert cli.main(["eval", str(folder), "--truth", "lrcat", "--lrcat", str(cat),
                     "--config", str(cfg)]) == 0  # fmt: skip
    ev = capsys.readouterr().out
    assert "missed-keeper rate" in ev and "keepers in truth" in ev


@pytest.mark.models
@pytest.mark.usefixtures("models_ready")
def test_train_with_too_few_rows_stays_on_prior(shoot_dir, make_fixtures, tmp_path, capsys):
    folder = tmp_path / "s"
    shutil.copytree(shoot_dir, folder)
    cat = tmp_path / "c.lrcat"
    entries = [
        {"root": str(tmp_path), "rel": "s/", "base": f"burst0_{i}", "ext": "jpg",
         "pick": 1 if i == 0 else -1, "rating": 4 if i == 0 else None}
        for i in range(5)
    ]  # fmt: skip
    make_fixtures.build_lrcat(cat, entries)
    out = tmp_path / "m.joblib"
    assert cli.main(["train", "--lrcat", str(cat), "--out", str(out)]) == 0
    assert "NOT trained" in capsys.readouterr().out and not out.exists()


@pytest.mark.models
@pytest.mark.usefixtures("models_ready")
def test_train_from_overrides_and_xmp(culled_shoot, make_fixtures, tmp_path, capsys):
    folder = tmp_path / "s"
    shutil.copytree(culled_shoot, folder)
    run_cull(folder, NO_AUDIT)
    with Cache.for_folder(folder) as c:
        for i, p in enumerate(sorted(folder.glob("*.jpg"))):
            c.set_override(str(p), "keep" if i % 2 else "reject", 0.5 if i % 3 == 0 else 1.0)
    out = tmp_path / "m.joblib"
    assert cli.main(["train", "--from-overrides", str(folder), "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "override" in text and "17" in text


@pytest.mark.models
@pytest.mark.usefixtures("models_ready")
def test_eval_with_xmp_truth(culled_shoot, tmp_path, capsys):
    from shotsieve.io.xmp import write_sidecar

    folder = tmp_path / "s"
    shutil.copytree(culled_shoot, folder)
    for p in folder.glob("*.xmp"):
        p.unlink()
    for p in folder.glob("burst0_*.jpg"):
        write_sidecar(p, rating=4 if p.stem == "burst0_0" else 0)
    assert cli.main(["eval", str(folder), "--truth", "xmp"]) == 0
    out = capsys.readouterr().out
    assert "with ground truth" in out and "5" in out


# ---- the trained model really changes decisions in the runner --------------------------------


@pytest.mark.models
@pytest.mark.usefixtures("models_ready")
def test_trained_personal_model_changes_decisions(culled_shoot, tmp_path):
    folder = tmp_path / "s"
    shutil.copytree(culled_shoot, folder)
    base = run_cull(folder, NO_AUDIT)
    assert base.stats["personal_n_train"] == 0
    X = stack([r.features for r in base.images])
    rng = np.random.default_rng(0)
    Xt = np.vstack([X + rng.normal(0, 0.01, X.shape).astype(np.float32) for _ in range(20)])
    y = np.zeros(len(Xt), int)
    y[::40] = 1  # a very picky user: almost nothing is a keeper
    model = PersonalModel().fit(Xt, y)
    assert model.trained and model.n_train == len(Xt)
    path = model.save(tmp_path / "picky.joblib")
    picky = run_cull(folder, Settings(audit_enabled=False, personal_model_path=str(path)))
    assert picky.stats["personal_n_train"] == model.n_train
    # blend: score = w * personal + (1 - w) * prior
    w = model.n_train / (model.n_train + 200)
    with Cache.for_folder(folder) as c:
        scene = c.scene_embeddings()
    pp = model.predict_proba(
        stack([r.features for r in picky.images]),
        embeddings=np.stack([scene[str(r.master_path)] for r in picky.images]),
    )
    for r, p in zip(picky.images, pp, strict=True):
        assert r.score == pytest.approx(w * p + (1 - w) * r.prior, abs=1e-6)
    before = {r.path.name: r.decision for r in base.images}
    after = {r.path.name: r.decision for r in picky.images}
    assert before != after
    assert sum(d == "keep" for d in after.values()) < sum(d == "keep" for d in before.values())
    assert picky.stats["counts"]["keep"] < base.stats["counts"]["keep"]
    assert Path(path).exists()


def test_cli_search_filter_only_json(tmp_path, shoot_dir, capsys):
    import json
    import shutil

    from shotsieve.cli import main
    from shotsieve.features.vector import FeatureVector
    from shotsieve.pipeline.cache import Cache

    d = tmp_path / "s"
    d.mkdir()
    shutil.copy(shoot_dir / "landscape.jpg", d / "landscape.jpg")
    row = dict(path=str(d / "landscape.jpg"), decision="keep", reasons=[], score=0.5, prior=0.5,
               group_id=0, burst_id=0, is_best=True, audit=False,
               features=FeatureVector().to_dict())  # fmt: skip
    with Cache.for_folder(d) as c:
        c.save_decisions([row])
    assert main(["search", str(d), "is:keep", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert [h["path"].rsplit("/", 1)[-1] for h in out] == ["landscape.jpg"]
    assert main(["search", str(d), 'person:"Ana"']) == 1  # people search needs consent


def test_train_table_reports_video_frame_labels(monkeypatch, capsys, tmp_path):
    import numpy as np

    from shotsieve.personal import dataset

    monkeypatch.setattr(cli, "_check_models", lambda: None)
    ts = dataset.TrainingSet(
        np.zeros((0, 25), np.float32),
        np.zeros(0, int),
        np.zeros(0),
        np.zeros((0, 384), np.float32),
        n_video=3,
    )
    monkeypatch.setattr(dataset, "build_training_set", lambda *a, **k: ts)
    assert cli.main(["train", "--from-overrides", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "video frame labels (not used for training)" in out
    assert [ln for ln in out.splitlines() if "video frame labels" in ln][0].split()[-1] == "3"
