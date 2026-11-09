"""The release bundle (scripts/build_release.sh): dist/ShotSieve.app must work away from the repo.

Skipped unless `dist/ShotSieve.app` exists (build it with `scripts/build_release.sh`). The app is
copied to a scratch "Applications" folder outside the repo and started with a temporary HOME, a
PATH without uv or the .venv, and (when sandbox-exec is available) with outbound network denied, so
a model download attempt would fail loudly. It must report every model ready, cull the CC0 crop
fixtures end to end, serve denoise / crop / search from the bundled models, and write nothing
inside the (signed, read-only) bundle.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

pytestmark = [pytest.mark.release, pytest.mark.install]

ROOT = Path(__file__).resolve().parents[1]
DIST_APP = ROOT / "dist" / "ShotSieve.app"
FIXTURES = ROOT / "tests" / "fixtures" / "crop" / "real"  # CC0, see ../LICENSES.md
SANDBOX_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"
NONET_PROFILE = """(version 1)
(allow default)
(deny network-outbound)
(allow network-outbound (remote ip "localhost:*"))
(allow network-outbound (remote unix-socket))
"""

skip_no_dist = pytest.mark.skipif(not DIST_APP.is_dir(), reason="dist/ShotSieve.app not built")


def _get(port: int, path: str, timeout: float = 60):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=timeout) as r:
        return r.status, r.read()


def _post(port: int, path: str, body: dict | None = None, timeout: float = 60):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=json.dumps(body or {}).encode(), method="POST",
        headers={"Content-Type": "application/json", "Origin": f"http://127.0.0.1:{port}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def _wait(pred, timeout: float, step: float = 0.5):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        v = pred()
        if v:
            return v
        time.sleep(step)
    return None


@pytest.fixture(scope="module")
def applications(tmp_path_factory) -> Path:
    """A copy of the app in a scratch Applications folder (clone, no extra disk)."""
    d = tmp_path_factory.mktemp("rel") / "Applications"
    d.mkdir()
    subprocess.run(["/bin/cp", "-Rc", str(DIST_APP), str(d / "ShotSieve.app")], check=True)
    return d


def _launch(app: Path, home: Path, extra_env: dict | None = None, offline: bool = True):
    env = {"HOME": str(home), "PATH": SANDBOX_PATH, "SHOTSIEVE_APP_AUTOCLOSE_S": "420",
           **(extra_env or {})}
    exe = str(app / "Contents" / "MacOS" / "ShotSieve")
    cmd = [exe]
    if offline and Path("/usr/bin/sandbox-exec").exists():
        prof = home.parent / "nonet.sb"
        prof.write_text(NONET_PROFILE)
        cmd = ["/usr/bin/sandbox-exec", "-f", str(prof), exe]
    return subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            start_new_session=True)


def _server(data: Path) -> dict | None:
    try:
        return json.loads((data / "server.json").read_text())
    except (OSError, ValueError):
        return None


@pytest.fixture(scope="module")
def running(applications, tmp_path_factory):
    home = tmp_path_factory.mktemp("home") / "home"
    home.mkdir()
    app = applications / "ShotSieve.app"
    proc = _launch(app, home)
    data = home / "Library" / "Application Support" / "ShotSieve"
    info = _wait(lambda: _server(data), 90)
    assert info, "the app did not write server.json"
    port = info["port"]
    assert _wait(lambda: _get(port, "/api/health")[0] == 200, 60)
    yield {"proc": proc, "port": port, "home": home, "data": data, "app": app}
    try:
        _post(port, "/api/quit")
    except Exception:  # noqa: BLE001
        pass
    if _wait(lambda: proc.poll() is not None, 15) is None:
        os.killpg(proc.pid, signal.SIGKILL)


@skip_no_dist
def test_bundle_is_self_contained(applications):
    app = applications / "ShotSieve.app"
    res = app / "Contents" / "Resources"
    assert (res / "python" / "bin" / "python3.11").is_file()
    assert (res / "models" / "bundle.json").is_file()
    assert (res / "LICENSE").is_file() and (res / "NOTICE").is_file()
    assert (res / "THIRD_PARTY_NOTICES.md").is_file()
    plist = (app / "Contents" / "Info.plist").read_text()
    import re
    bid = re.search(r"<key>CFBundleIdentifier</key>\s*<string>([^<]+)</string>", plist).group(1)
    assert re.fullmatch(r"app\.shotsieve\.desktop|io\.github\.[a-z0-9-]+\.shotsieve", bid), bid
    assert "LSArchitecturePriority" in plist
    # licence texts shipped in the bundle (GPL/MPL/permissive libraries, embedded Python)
    lic = res / "licenses"
    for f in ("GPL-2.0.txt", "GPL-3.0.txt", "LGPL-2.1.txt", "LGPL-3.0.txt", "MPL-2.0.txt"):
        assert (lic / f).is_file(), f
    assert (lic / "python-build-standalone" / "LICENSE.cpython.txt").is_file()
    assert (lic / "python-build-standalone" / "LICENSE.openssl-3.txt").is_file()
    assert (lic / "dylibs" / "dav1d.txt").is_file()
    notices = (res / "THIRD_PARTY_NOTICES.md").read_text()
    assert "gpl-sources.tar.gz" in notices and "Written offer" in notices
    # every shared library of the bundle is named in the notices
    site = res / "python" / "lib" / "python3.11" / "site-packages"
    missing = [p.name for p in site.rglob("*.dylib") if p.name not in notices]
    assert not missing, missing
    assert "local.shotsieve" not in plist
    r = subprocess.run(["/usr/bin/codesign", "--verify", "--deep", "--strict", str(app)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    # nothing in the models folder but the verified bundle (no int8 duplicate, no raw NAFNet fp16)
    names = {p.name for p in (res / "models").iterdir()}
    assert "siglip2_vision_int8.onnx" not in names
    assert "nafnet_sidd_width32_fp16.onnx" not in names
    assert "nafnet_sidd_width32_coreml.onnx" in names and "siglip2_vision_fp32.onnx" in names


@skip_no_dist
def test_models_match_pinned_sha256(applications):
    """Every shipped file matches bundle.json AND (when not derived) the registry's pinned sha."""
    import hashlib

    from shotsieve.models.registry import MODEL_SPECS

    mdir = applications / "ShotSieve.app" / "Contents" / "Resources" / "models"
    manifest = json.loads((mdir / "bundle.json").read_text())
    assert set(manifest) == set(MODEL_SPECS) - {"siglip2_vision_int8"}
    for name, ent in manifest.items():
        h = hashlib.sha256((mdir / ent["filename"]).read_bytes()).hexdigest()
        assert h == ent["sha256"], name
        if "derived_from_sha256" not in ent:
            assert h == MODEL_SPECS[name].sha256, name
        else:
            assert ent["derived_from_sha256"] == MODEL_SPECS[name].sha256


@skip_no_dist
def test_app_runs_offline_from_a_scratch_folder(running, tmp_path):
    port, home, data, app = running["port"], running["home"], running["data"], running["app"]
    health = json.loads(_get(port, "/api/health")[1])
    assert health["ok"] and health["models_ready"] and all(health["models"].values())
    assert health["search_ready"]

    ob = json.loads(_get(port, "/api/onboarding")[1])
    assert ob["bundled"] and ob["models_ready"]  # the wizard skips the download step
    groups = json.loads(_get(port, "/api/models")[1])["groups"]
    assert all(g["installed"] for g in groups.values()) and "search-int8" not in groups
    # a download request is a no-op, never a network call
    job = _post(port, "/api/models/download", {"groups": ["core", "search", "crop", "denoise"]})
    st = _wait(lambda: (lambda j: j if j["state"] in ("done", "error") else None)(
        json.loads(_get(port, f"/api/jobs/{job['job_id']}")[1])), 30)
    assert st and st["state"] == "done", st

    # state lives in ~/Library, not in the bundle
    assert (home / "Library" / "Logs" / "ShotSieve" / "serve.log").is_file()
    assert (data / "app.lock").exists() and not (app / "Contents" / "Resources" / "data").exists()

    # cull the CC0 fixtures end to end
    shoot = tmp_path / "shoot"
    shoot.mkdir()
    for f in sorted(FIXTURES.glob("*.jpg"))[:16]:
        shutil.copy(f, shoot / f.name)
    job = _post(port, "/api/cull", {"folder": str(shoot)})["job_id"]
    done = _wait(lambda: (lambda j: j if j["state"] in ("done", "error") else None)(
        json.loads(_get(port, f"/api/jobs/{job}")[1])), 240, 1)
    assert done and done["state"] == "done", done
    shoot_info = json.loads(_get(port, f"/api/shoot?folder={shoot}")[1])
    assert len(shoot_info["images"]) == 16
    assert {i["decision"] for i in shoot_info["images"]} <= {"keep", "review", "reject"}
    assert (shoot / ".shotsieve").is_dir()  # shoot data stays with the shoot
    first = shoot_info["images"][0]["id"]

    # denoise (bundled, CoreML-repaired NAFNet) and crop (bundled YOLOX)
    ds = json.loads(_get(port, "/api/denoise/status")[1])
    assert ds["model"]["installed"]
    code, body = _get(port, f"/api/denoise/{first}/region?x0=0.2&y0=0.2&x1=0.45&y1=0.45"
                            f"&level=medium&folder={shoot}", 180)
    assert code == 200 and body[:2] == b"\xff\xd8"
    sub = json.loads(_get(port, f"/api/crop/{first}/subjects?folder={shoot}", 120)[1])
    assert "subjects" in sub

    # natural-language search from the bundled SigLIP2
    _post(port, "/api/index/start", {"folder": str(shoot), "people": False, "ocr": False})
    ok = _wait(lambda: (lambda s: s if s.get("indexed", 0) >= 16 else None)(
        json.loads(_get(port, f"/api/index/status?folder={shoot}")[1])), 240, 2)
    assert ok, "the indexer did not finish"
    res = json.loads(_get(port, f"/api/search?folder={shoot}&q=a+dog&k=3", 120)[1])
    assert res["results"] if "results" in res else res

    # no model download was attempted: no .part file, no registry download in the logs
    mdir = app / "Contents" / "Resources" / "models"
    assert not list(mdir.glob("*.part"))
    logs = ""
    for p in (home / "Library" / "Logs" / "ShotSieve").glob("*.log*"):
        logs += p.read_text(errors="replace")
    assert "downloading http" not in logs and "huggingface.co" not in logs
    assert "Operation not permitted" not in logs  # the no-network sandbox never had to refuse

    # the signed bundle was not modified by running
    r = subprocess.run(["/usr/bin/codesign", "--verify", "--deep", "--strict", str(app)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


@skip_no_dist
def test_open_via_launchservices(applications, tmp_path):
    """`open -a <app>`: the way Finder starts it (LaunchServices env is not our test HOME, so the
    data folder is redirected with --env)."""
    app = applications / "ShotSieve.app"
    data = tmp_path / "ls-data"
    r = subprocess.run(
        ["/usr/bin/open", "-n", "-a", str(app), "--env", f"SHOTSIEVE_DATA_DIR={data}",
         "--env", "SHOTSIEVE_APP_AUTOCLOSE_S=240"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    info = _wait(lambda: _server(data), 90)
    try:
        assert info, "LaunchServices launch did not start the server"
        health = json.loads(_get(info["port"], "/api/health")[1])
        assert health["ok"] and health["models_ready"]
        assert (data / "logs" / "serve.log").is_file()
    finally:
        if info:
            try:
                _post(info["port"], "/api/quit")
            except (urllib.error.URLError, OSError):
                pass
            _wait(lambda: _server(data) is None, 15)
