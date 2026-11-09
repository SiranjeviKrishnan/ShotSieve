"""Release-bundle behaviour that does not need the built app: where state lives when the code runs
from inside `ShotSieve.app`, the read-only bundled model folder, built-in shot-list templates."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import shotsieve.config as config
from shotsieve.models import registry

FAKE = "/Applications/ShotSieve.app/Contents/Resources/python/lib/python3.11/site-packages"


@pytest.fixture
def in_app(monkeypatch, tmp_path):
    """Pretend the package lives inside an .app bundle, with HOME in tmp."""
    monkeypatch.setattr(config, "__file__", f"{FAKE}/shotsieve/config.py")
    monkeypatch.delenv("SHOTSIEVE_DATA_DIR", raising=False)
    monkeypatch.delenv("SHOTSIEVE_LOG_DIR", raising=False)
    monkeypatch.delenv("SHOTSIEVE_MODELS_DIR", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    return tmp_path / "home"


def test_checkout_is_not_a_bundle():
    assert config.bundle_resources() is None and not config.in_bundle()


def test_bundle_state_goes_to_library(in_app):
    assert config.bundle_resources() == Path("/Applications/ShotSieve.app/Contents/Resources")
    assert config.data_dir() == in_app / "Library" / "Application Support" / "ShotSieve"
    assert config.logs_dir() == in_app / "Library" / "Logs" / "ShotSieve"
    assert config.data_dir().is_dir() and config.logs_dir().is_dir()


def test_bundle_env_overrides(in_app, monkeypatch, tmp_path):
    monkeypatch.setenv("SHOTSIEVE_DATA_DIR", str(tmp_path / "d"))
    assert config.data_dir() == tmp_path / "d"
    assert config.logs_dir() == tmp_path / "d" / "logs"  # data dir override carries the logs
    monkeypatch.setenv("SHOTSIEVE_LOG_DIR", str(tmp_path / "l"))
    assert config.logs_dir() == tmp_path / "l"


def test_bundle_models_dir_is_inside_the_app(in_app):
    assert registry.models_dir() == Path("/Applications/ShotSieve.app/Contents/Resources/models")


def make_bundle(folder: Path, drop: tuple[str, ...] = ("siglip2_vision_int8",)) -> None:
    """A fake complete bundle: every spec's file (tiny, size patched) + bundle.json."""
    folder.mkdir(parents=True)
    manifest = {}
    for name, spec in registry.MODEL_SPECS.items():
        if name in drop:
            continue
        data = name.encode()
        (folder / spec.filename).write_bytes(data)
        manifest[name] = {"filename": spec.filename, "size": len(data), "sha256": "x" * 64}
    (folder / registry.BUNDLE_MANIFEST).write_text(json.dumps(manifest))


def test_bundled_folder_is_complete_and_never_downloads(tmp_models_dir, monkeypatch):
    make_bundle(tmp_models_dir)
    registry._bundle_manifest.cache_clear()
    assert registry.bundled()
    assert registry.verify_model("yunet") and registry.verify_model("nafnet_sidd")
    assert registry.model_path("dinov2_small").read_bytes() == b"dinov2_small"
    st = registry.models_status()
    assert "search-int8" not in st and all(g["installed"] for g in st.values())

    def boom(*a, **k):
        raise AssertionError("network access in a bundle")

    monkeypatch.setattr(registry, "_download", boom)
    monkeypatch.setattr(registry.urllib.request, "urlopen", boom)
    out = registry.ensure_models(groups=("core", "search", "ocr", "crop", "denoise"))
    assert set(out) == set(registry.group_names("core", "search", "ocr", "crop", "denoise"))
    assert not list(tmp_models_dir.glob("*.part"))
    assert not (tmp_models_dir / "manifest.json").exists()  # read-only: nothing rewritten


def test_bundle_variant_replaces_the_spec_file(tmp_models_dir):
    """nafnet ships as the repaired copy under bundle.json's name, not the registry filename."""
    tmp_models_dir.mkdir()
    (tmp_models_dir / "nafnet_sidd_width32_coreml.onnx").write_bytes(b"abc")
    (tmp_models_dir / registry.BUNDLE_MANIFEST).write_text(json.dumps(
        {"nafnet_sidd": {"filename": "nafnet_sidd_width32_coreml.onnx", "size": 3, "sha256": ""}}))
    registry._bundle_manifest.cache_clear()
    assert registry.verify_model("nafnet_sidd")
    assert registry.model_path("nafnet_sidd").name == "nafnet_sidd_width32_coreml.onnx"
    from shotsieve.models import denoise

    assert denoise._coreml_ready_path().endswith("nafnet_sidd_width32_coreml.onnx")


def test_dev_folder_is_not_bundled(tmp_models_dir):
    tmp_models_dir.mkdir()
    registry._bundle_manifest.cache_clear()
    assert not registry.bundled()


def test_onboarding_reports_bundled(tmp_models_dir):
    from fastapi.testclient import TestClient

    from shotsieve.server.app import create_app

    make_bundle(tmp_models_dir)
    registry._bundle_manifest.cache_clear()
    c = TestClient(create_app())
    ob = c.get("/api/onboarding").json()
    assert ob["bundled"] is True and ob["models_ready"] is True


def test_builtin_shotlist_templates_ship_in_the_package(tmp_path, monkeypatch):
    from shotsieve.pipeline import shotlist

    monkeypatch.setenv("SHOTSIEVE_DATA_DIR", str(tmp_path))  # no user templates
    assert {"wedding", "sports", "corporate"} <= set(shotlist.template_names())
    assert len(shotlist.load_template("wedding")) >= 15
    user = tmp_path / "shotlists"
    user.mkdir()
    (user / "wedding.txt").write_text("!only this shot\n")
    (user / "mine.txt").write_text("a dog\n")
    assert [i.text for i in shotlist.load_template("wedding")] == ["only this shot"]
    assert "mine" in shotlist.template_names()
