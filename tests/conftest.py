"""Shared fixtures. Generated assets come from scripts/make_fixtures.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "tests" / "fixtures" / "generated"


def _load_make_fixtures():
    spec = importlib.util.spec_from_file_location(
        "make_fixtures", ROOT / "scripts" / "make_fixtures.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_fixtures"] = mod
    spec.loader.exec_module(mod)
    return mod


def pytest_configure(config):
    """Isolate the user's real data dir (recent folders, onboarding, caches) for the whole run.

    Server subprocesses started by tests inherit os.environ, so they are isolated too; a test that
    wants its own data dir still sets SHOTSIEVE_DATA_DIR itself (monkeypatch)."""
    import os
    import tempfile

    config._shotsieve_data = tempfile.mkdtemp(prefix="shotsieve_test_data_")
    os.environ["SHOTSIEVE_DATA_DIR"] = config._shotsieve_data


def pytest_unconfigure(config):
    import shutil

    shutil.rmtree(getattr(config, "_shotsieve_data", ""), ignore_errors=True)


@pytest.fixture(autouse=True)
def _testserver_host(monkeypatch):
    """TestClient talks to Host `testserver`; production allows loopback names only (m5)."""
    monkeypatch.setenv("SHOTSIEVE_EXTRA_HOSTS", "testserver")


@pytest.fixture(scope="session")
def make_fixtures():
    return _load_make_fixtures()


@pytest.fixture(scope="session")
def gen_dir(make_fixtures) -> Path:
    """Generated fixture directory (built on first use; downloads need network once)."""
    needed = [GEN / "shoot_manifest.json", GEN / "src" / "group_rgb.jpg", GEN / "catalog.lrcat"]
    if not all(p.exists() for p in needed):
        assert make_fixtures.main([]) == 0
    return GEN


@pytest.fixture(scope="session")
def shoot_dir(gen_dir) -> Path:
    return gen_dir / "shoot"


@pytest.fixture(scope="session")
def manifest(gen_dir) -> list[dict]:
    return json.loads((gen_dir / "shoot_manifest.json").read_text())


@pytest.fixture(scope="session")
def rgb(gen_dir):
    """Loader for source images by file name, as RGB arrays."""

    def load(name: str) -> np.ndarray:
        return cv2.cvtColor(cv2.imread(str(gen_dir / "src" / name)), cv2.COLOR_BGR2RGB)

    return load


@pytest.fixture(scope="session")
def models_ready():
    from shotsieve.models.registry import group_names, verify_model

    missing = [n for n in group_names("core") if not verify_model(n)]
    if missing:
        pytest.skip(f"models not downloaded: {missing} (run scripts/download_models.py)")


@pytest.fixture
def tmp_models_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("SHOTSIEVE_MODELS_DIR", str(tmp_path / "models"))
    return tmp_path / "models"


@pytest.fixture(scope="session")
def search_ready():
    from shotsieve.models.registry import group_names, verify_model

    missing = [n for n in group_names("search") if not verify_model(n)]
    if missing:
        pytest.skip(f"search models not downloaded: {missing} (models download --extra search)")


@pytest.fixture(scope="session")
def search_images(make_fixtures) -> dict[str, np.ndarray]:
    """The 13 search subjects (RGB), downloaded once into fixtures/generated/search."""
    folder = make_fixtures.fetch_search_images()
    return {
        p.stem: cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        for p in sorted(folder.glob("*.jpg"))
    }


@pytest.fixture(scope="session")
def edit_styles(models_ready):
    """tests/fixtures/edit_styles/ (T15 synthetic photographer styles), built on first use."""
    spec = importlib.util.spec_from_file_location(
        "make_edit_styles", ROOT / "scripts" / "make_edit_styles.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_edit_styles"] = mod
    spec.loader.exec_module(mod)
    return mod.build()


@pytest.fixture(scope="session")
def edit_sets(edit_styles):
    """{style: (train EditSet, held-out EditSet)} split by base image (every 5th base held out);
    features come from (and are cached in) the fixture folder's features.db."""
    from shotsieve.personal.edit_dataset import FeatureStore, build_edit_set
    from shotsieve.personal.edit_profile import EditSet

    def subset(S: EditSet, m: np.ndarray) -> EditSet:
        return EditSet(
            S.stats[m], S.emb[m], S.fmt[m], {k: v[m] for k, v in S.y.items()}, S.groups[m],
            S.carry, [p for p, x in zip(S.paths, m, strict=True) if x], S.sources, S.notes,
        )

    store = FeatureStore(edit_styles / "features.db")
    out = {}
    for style in "ABC":
        S = build_edit_set(
            edit_styles / f"style{style}.lrcat", [], None, None, store=store,
            group_of=lambda p: Path(p).parent.name,
        )
        base = np.array([int(Path(p).parent.name[1:]) for p in S.paths])
        out[style] = (subset(S, base % 5 != 0), subset(S, base % 5 == 0), S)
    store.close()
    return out


@pytest.fixture(scope="session")
def edit_profile_file(edit_sets, tmp_path_factory) -> Path:
    """An editing profile trained on style A (saved once per session)."""
    from shotsieve.personal.edit_profile import EditProfile

    path = tmp_path_factory.mktemp("edit_profile") / "edit_profile.joblib"
    EditProfile().fit(edit_sets["A"][0]).save(path)
    return path


@pytest.fixture
def edit_env(tmp_path, monkeypatch, edit_profile_file) -> Path:
    """An isolated data dir that already holds the trained profile (no shared feature cache)."""
    import shutil

    d = tmp_path / "data"
    d.mkdir()
    shutil.copy(edit_profile_file, d / "edit_profile.joblib")
    monkeypatch.setenv("SHOTSIEVE_DATA_DIR", str(d))
    return d


@pytest.fixture(scope="session")
def browser():
    """One headless Chromium for every real-browser test (test_ui.py, test_onboarding_ui.py)."""
    import os

    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(ROOT / ".playwright"))
    sync_api = pytest.importorskip("playwright.sync_api")
    pw = sync_api.sync_playwright().start()
    try:
        b = pw.chromium.launch()
    except Exception as exc:  # noqa: BLE001 - browser binary missing
        pw.stop()
        pytest.skip(f"Chromium not available ({str(exc).splitlines()[0]})")
    yield b
    b.close()
    pw.stop()


def pytest_collection_modifyitems(config, items):
    """`install` / `release` tests (sandbox installs, launcher, the .app bundle) run only with
    `-m install` or `-m release` (the release tests also need dist/ShotSieve.app)."""
    expr = config.getoption("-m") or ""
    if "install" in expr or "release" in expr:
        return
    skip = pytest.mark.skip(reason="install/release tests run only with -m install or -m release")
    for item in items:
        if "install" in item.keywords:
            item.add_marker(skip)
