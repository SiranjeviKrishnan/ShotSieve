"""T18: install.sh, build_app.sh, launcher.sh, doctor. Run with `-m install` (skipped otherwise).

Sandbox installs run with HOME=<tmp>/home and a minimal PATH and prove the real home tree is
untouched. The offline install clones `.uv-cache`/`.uv-python` from the dev checkout and uses the
local uv tarball (`data/cache/`, fetched once) and the local models.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

pytestmark = pytest.mark.install

ROOT = Path(__file__).resolve().parents[1]
UV_VERSION = "0.11.4"
UV_TARBALL = ROOT / "data" / "cache" / f"uv-{UV_VERSION}-aarch64.tar.gz"
SANDBOX_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"
EXCLUDES = ["/.venv", "/.uv-cache", "/.uv-python", "/models", "/data", "/.tools", "/ShotSieve.app",
            "/.pytest_tmp*", "/.playwright", "/.ruff_cache", "/.pytest_cache", "__pycache__",
            "/tests/fixtures/generated", "/.git", "/.app-build.*", "/.tools.*"]


# ---- helpers ----------------------------------------------------------------------------------


def _sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def lint_targets():
    return [ROOT / "install.sh", ROOT / "Install ShotSieve.command",
            ROOT / "scripts" / "build_app.sh", ROOT / "scripts" / "launcher.sh"]


def make_fake_root(tmp: Path) -> Path:
    """A throwaway ROOT: real scripts, `.venv/bin` wrappers around the dev venv."""
    root = tmp / "fakeroot"
    (root / "scripts").mkdir(parents=True)
    for f in ("build_app.sh", "launcher.sh", "make_icon.py"):
        shutil.copy(ROOT / "scripts" / f, root / "scripts" / f)
    (root / ".venv" / "bin").mkdir(parents=True)
    for name, real in (("python", "python"), ("shotsieve", "shotsieve")):
        w = root / ".venv" / "bin" / name
        w.write_text(f'#!/bin/bash\nexec "{ROOT}/.venv/bin/{real}" "$@"\n')
        w.chmod(0o755)
    return root


def build_app(root: Path):
    r = _sh(["/bin/bash", str(root / "scripts" / "build_app.sh")], env={**os.environ})
    assert r.returncode == 0, r.stderr
    return root / "ShotSieve.app" / "Contents" / "MacOS" / "ShotSieve"


def free_port_range_ok() -> bool:
    for p in range(8765, 8776):
        with socket.socket() as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", p))
            except OSError:
                return False
    return True


def health(port: int):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=3) as r:
        return json.load(r)


def data_pid(data: Path):
    try:
        return json.loads((data / "server.json").read_text())["pid"]
    except (OSError, ValueError, KeyError):
        return None


def alive(pid) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def stop_servers(data: Path, root: Path = None):
    """Quit (then kill) only the app whose pid is in THIS data dir's server.json."""
    sj = data / "server.json"
    pid = data_pid(data)
    if sj.exists():
        try:
            port = json.loads(sj.read_text())["port"]
            urllib.request.urlopen(
                urllib.request.Request(f"http://127.0.0.1:{port}/api/quit", method="POST"),
                timeout=3)
        except Exception:  # noqa: BLE001
            pass
    t0 = time.time()
    while alive(pid) and time.time() - t0 < 6:
        time.sleep(0.2)
    if alive(pid):
        os.kill(pid, signal.SIGKILL)


def fake_env(data: Path, **extra):
    """Display-less run: tests/fake_webview.py stands in for pywebview."""
    return {**os.environ, "SHOTSIEVE_DATA_DIR": str(data), "SHOTSIEVE_DIALOG": "echo",
            "SHOTSIEVE_WEBVIEW_MODULE": "fake_webview",
            "PYTHONPATH": str(ROOT / "tests"), **extra}


def start_launcher(launcher: Path, data: Path, wait_s: float = 40, **extra):
    """Run the launcher like LaunchServices would (no wrapper shell): returns (proc, port)."""
    p = subprocess.Popen([str(launcher)], env=fake_env(data, **extra), stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True)
    t0 = time.time()
    while time.time() - t0 < wait_s:
        if p.poll() is not None:
            raise AssertionError(f"launcher exited {p.returncode}: {p.stdout.read()}")
        sj = data / "server.json"
        if sj.exists():
            try:
                port = json.loads(sj.read_text())["port"]
                if health(port)["app"] == "shotsieve":
                    return p, port
            except Exception:  # noqa: BLE001
                pass
        time.sleep(0.2)
    p.kill()
    raise AssertionError("app did not come up")


def run_launcher(launcher: Path, data: Path, **extra):
    t0 = time.time()
    r = subprocess.run([str(launcher)], env=fake_env(data, **extra), capture_output=True,
                       text=True, timeout=60)
    return r, time.time() - t0


def tree_snapshot(path: Path):
    snap = {}
    if not path.exists():
        return snap
    for dp, dn, fn in os.walk(path):
        for n in dn + fn:
            p = Path(dp) / n
            try:
                st = p.lstat()
                snap[str(p)] = (st.st_size, st.st_mtime_ns)
            except OSError:
                pass
    return snap


# ---- lint and structure -----------------------------------------------------------------------


def test_scripts_lint_and_exec_bits():
    for f in lint_targets():
        assert f.exists(), f
        assert _sh(["/bin/bash", "-n", str(f)]).returncode == 0, f
        assert os.access(f, os.X_OK), f"{f} is not executable"


def test_install_ignores_user_uv_config_and_never_uses_system_tmp():
    """Review #10 m6: UV_NO_CONFIG=1 is exported before any uv call, and the temp dir for the uv
    download is only ever created inside $ROOT (no `mktemp -d` fallback to the system temp)."""
    text = (ROOT / "install.sh").read_text()
    assert "export UV_NO_CONFIG=1" in text
    assert text.index("export UV_NO_CONFIG=1") < text.index('"$UV_BIN" python install')
    for line in text.splitlines():
        if "mktemp" in line and not line.lstrip().startswith("#"):
            assert '"$ROOT/' in line and "||  mktemp" not in line and "|| mktemp" not in line, line


def test_build_app_structure_and_root_substitution(tmp_path):
    root = make_fake_root(tmp_path)
    launcher = build_app(root)
    plist = root / "ShotSieve.app" / "Contents" / "Info.plist"
    assert _sh(["/usr/bin/plutil", "-lint", str(plist)]).returncode == 0
    keys = _sh(["/usr/bin/plutil", "-convert", "json", "-o", "-", str(plist)]).stdout
    info = json.loads(keys)
    assert info["CFBundleExecutable"] == "ShotSieve" and info["CFBundlePackageType"] == "APPL"
    assert info["CFBundleIdentifier"] == "local.shotsieve.app"
    assert info["LSMinimumSystemVersion"] == "14.0"
    from shotsieve import __version__

    assert info["CFBundleShortVersionString"] == __version__
    text = launcher.read_text()
    assert f"ROOT={root.resolve()}" in text and "@ROOT@" not in text
    assert os.access(launcher, os.X_OK)
    assert (root / "ShotSieve.app/Contents/Resources/ShotSieve.icns").exists()
    # rebuilding replaces the bundle (idempotent), nothing written outside the root
    build_app(root)
    assert not list(root.glob(".app-build.*"))


# ---- launcher ---------------------------------------------------------------------------------


@pytest.fixture()
def launched(tmp_path):
    root = make_fake_root(tmp_path)
    launcher = build_app(root)
    data = tmp_path / "data"
    data.mkdir()
    yield root, launcher, data
    stop_servers(data)


def test_launcher_execs_the_app_process(launched):
    """The bundle executable IS the app process (regression: LaunchServices killed a detached
    server when the launcher script exited)."""
    root, launcher, data = launched
    p, port = start_launcher(launcher, data)
    try:
        assert p.poll() is None
        assert p.pid == data_pid(data)  # no child, no wrapper: the launcher's pid serves
        assert health(port)["pid"] == p.pid
        # second launch: no second server, returns quickly with 0
        r2, dt2 = run_launcher(launcher, data)
        assert r2.returncode == 0 and dt2 <= 15, (r2.stdout, r2.stderr, dt2)
        assert data_pid(data) == p.pid
        # quit: process exits <= 5 s, server.json gone
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/quit", method="POST")
        assert json.load(urllib.request.urlopen(req, timeout=3))["ok"]
        assert p.wait(timeout=10) == 0
        assert not (data / "server.json").exists()
    finally:
        if p.poll() is None:
            p.kill()


def test_launcher_port_busy_uses_next_and_replaces_stale_json(launched):
    root, launcher, data = launched
    (data / "server.json").write_text(
        json.dumps({"pid": 999999, "port": 8765, "version": "0", "started": 0}))  # dead pid
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", 8765))
        except OSError:
            pytest.skip("8765 is busy on this machine")
        s.listen()
        p, port = start_launcher(launcher, data)
        try:
            assert port != 8765 and data_pid(data) == p.pid
        finally:
            p.kill()


def test_launcher_rotates_big_log(launched):
    root, launcher, data = launched
    (data / "logs").mkdir()
    (data / "logs" / "serve.log").write_bytes(b"x" * (5 * 1024 * 1024 + 10))
    p, _ = start_launcher(launcher, data)
    try:
        assert (data / "logs" / "serve.log.1").exists()
        assert (data / "logs" / "serve.log").stat().st_size < 1024 * 1024
    finally:
        p.kill()


def test_bundle_uses_exec_and_plist_is_complete(tmp_path):
    root = make_fake_root(tmp_path)
    launcher = build_app(root)
    text = launcher.read_text()
    assert 'exec "$BIN" app' in text and "nohup" not in text and "disown" not in text
    info = json.loads(_sh(["/usr/bin/plutil", "-convert", "json", "-o", "-",
                           str(launcher.parent.parent / "Info.plist")]).stdout)
    assert info["CFBundleName"] == "ShotSieve" and info["CFBundleDisplayName"] == "ShotSieve"
    assert info["CFBundleIdentifier"] == "local.shotsieve.app"
    assert info["CFBundleIconFile"] == "ShotSieve" and info["NSHighResolutionCapable"] is True
    icns = launcher.parent.parent / "Resources" / "ShotSieve.icns"
    assert icns.read_bytes()[:4] == b"icns"


def test_open_via_launchservices_stays_up_and_quits(tmp_path):
    """T23a regression: `open` (LaunchServices, like a Finder double-click) with a REAL window.
    The server must stay healthy well past the ~15 s at which it used to die, then quit cleanly."""
    root = make_fake_root(tmp_path)
    build_app(root)
    app = root / "ShotSieve.app"
    data = root / "data"  # the launcher's default data dir for this fake root
    data.mkdir()
    r = _sh(["/usr/bin/open", "-n", "-a", str(app)])
    assert r.returncode == 0, r.stderr
    pid = None
    try:
        t0 = time.time()
        port = None
        while time.time() - t0 < 60:
            if (data / "server.json").exists():
                try:
                    port = json.loads((data / "server.json").read_text())["port"]
                    health(port)
                    break
                except Exception:  # noqa: BLE001
                    pass
            time.sleep(0.5)
        assert port, "app did not start via open: " + (
            (data / "logs/serve.log").read_text()[-800:] if (data / "logs/serve.log").exists()
            else "no log")
        pid = data_pid(data)
        t1 = time.time()
        while time.time() - t1 < 32:
            assert alive(pid), f"app died {time.time() - t1:.0f} s after launch"
            assert health(port)["app"] == "shotsieve"
            time.sleep(3)
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/quit", method="POST")
        assert json.load(urllib.request.urlopen(req, timeout=3))["ok"]
        t2 = time.time()
        while alive(pid) and time.time() - t2 < 15:
            time.sleep(0.3)
        assert not alive(pid), "app process still alive 15 s after /api/quit"
        assert not (data / "server.json").exists()
    finally:
        if pid and alive(pid):
            os.kill(pid, signal.SIGKILL)


def test_open_via_launchservices_window_close_ends_process(tmp_path):
    """Field bug 'ShotSieve is not launching': closing the window left a windowless ghost process
    that every Dock click merely re-activated. Real window via `open`; a second `open -a` while it
    runs must not start another process; closing the window (test knob) must end the process."""
    root = make_fake_root(tmp_path)
    build_app(root)
    app = root / "ShotSieve.app"
    data = root / "data"
    data.mkdir()
    r = _sh(["/usr/bin/open", "-n", "--env", "SHOTSIEVE_APP_AUTOCLOSE_S=20", "-a", str(app)])
    assert r.returncode == 0, r.stderr
    pid = None
    try:
        t0 = time.time()
        port = None
        while time.time() - t0 < 60:
            try:
                port = json.loads((data / "server.json").read_text())["port"]
                health(port)
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
        assert port, "app did not start via open"
        pid = data_pid(data)
        t_up = time.time()
        # Dock/Finder click on the running app: same process, still healthy
        assert _sh(["/usr/bin/open", "-a", str(app)]).returncode == 0
        time.sleep(2)
        assert alive(pid) and data_pid(data) == pid and health(port)["app"] == "shotsieve"
        # the window closes (knob, ~20 s after the GUI loop started): the process must be gone
        deadline = max(t_up, time.time()) + 20 + 10
        while alive(pid) and time.time() < deadline:
            time.sleep(0.3)
        assert not alive(pid), "process still alive 10 s after the window closed"
        assert not (data / "server.json").exists()
    finally:
        if pid and alive(pid):
            os.kill(pid, signal.SIGKILL)


def test_launcher_moved_root_shows_dialog_and_exits_1(tmp_path):
    root = make_fake_root(tmp_path)
    launcher = build_app(root)
    moved = tmp_path / "moved"
    root.rename(moved)
    r, _ = run_launcher(moved / "ShotSieve.app/Contents/MacOS/ShotSieve", tmp_path / "d")
    assert r.returncode == 1
    assert "folder moved or is incomplete" in r.stdout and "Install ShotSieve.command" in r.stdout
    assert launcher.parent.parent.parent.parent == root  # (the original path is gone)


# ---- doctor -----------------------------------------------------------------------------------


def _doctor(env_extra, tmp_path):
    env = {**os.environ, **env_extra}
    return subprocess.run([sys.executable, "-m", "shotsieve.cli", "doctor"], env=env,
                          capture_output=True, text=True, cwd=tmp_path)


@pytest.mark.models
def test_doctor_ok(models_ready, tmp_path):
    r = _doctor({"SHOTSIEVE_DATA_DIR": str(tmp_path / "d")}, tmp_path)
    assert r.returncode == 0, r.stdout
    assert "ready" in r.stdout


def test_doctor_reports_each_failure(tmp_path):
    # models missing
    r = _doctor({"SHOTSIEVE_DATA_DIR": str(tmp_path / "d"),
                 "SHOTSIEVE_MODELS_DIR": str(tmp_path / "nomodels")}, tmp_path)
    assert r.returncode == 1 and "core models are missing" in r.stdout
    # unwritable data dir
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o500)
    try:
        r = _doctor({"SHOTSIEVE_DATA_DIR": str(ro),
                     "SHOTSIEVE_MODELS_DIR": str(tmp_path / "nomodels")}, tmp_path)
        assert r.returncode == 1 and "cannot write to its data folder" in r.stdout
    finally:
        ro.chmod(0o700)
    # port range busy
    socks = []
    try:
        for p in range(8765, 8776):
            s = socket.socket()
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", p))
            s.listen()
            socks.append(s)
    except OSError:
        pytest.skip("cannot occupy 8765-8775 here")
    try:
        r = _doctor({"SHOTSIEVE_DATA_DIR": str(tmp_path / "d2"),
                     "SHOTSIEVE_MODELS_DIR": str(tmp_path / "nomodels")}, tmp_path)
        assert r.returncode == 1 and "No free port between 8765 and 8775" in r.stdout
        assert r.stdout.count("\n") >= 2  # one sentence per failure
    finally:
        for s in socks:
            s.close()


# ---- sandbox installs -------------------------------------------------------------------------


def clone_repo(dest: Path, with_caches: bool) -> Path:
    dest.mkdir(parents=True)
    cmd = ["/usr/bin/rsync", "-a"]
    for e in EXCLUDES:
        cmd += ["--exclude", e]
    assert _sh(cmd + [f"{ROOT}/", f"{dest}/"]).returncode == 0
    if with_caches:
        for d in (".uv-cache", ".uv-python"):
            assert _sh(["/bin/cp", "-c", "-R", str(ROOT / d), str(dest / d)]).returncode == 0
    return dest


POISON_UV_TOML = (
    'required-version = ">=99"\nindex-url = "http://127.0.0.1:9/nope"\nno-binary = true\n'
)


def sandbox_env(tmp: Path, **extra):
    home = tmp / "home"
    home.mkdir(exist_ok=True)
    (home / "marker.txt").write_text("home")
    return {"HOME": str(home), "PATH": SANDBOX_PATH, "TMPDIR": str(tmp / "tmp"), **extra}


def installed_size_mb(root: Path) -> float:
    total = 0
    for dp, dn, fn in os.walk(root):
        if Path(dp) == root and ".uv-cache" in dn:
            dn.remove(".uv-cache")
        for n in fn:
            try:
                total += (Path(dp) / n).lstat().st_size
            except OSError:
                pass
    return total / 1e6


def system_footprint():
    return {
        "shared": sorted(p.name for p in Path("/Users/Shared").iterdir())
        if Path("/Users/Shared").exists() else [],
        "library": tree_snapshot(Path.home() / "Library" / "Application Support" / "shotsieve"),
        "caches": tree_snapshot(Path.home() / "Library" / "Caches" / "uv"),
    }


@pytest.fixture(scope="module")
def uv_tarball():
    if not UV_TARBALL.exists():
        UV_TARBALL.parent.mkdir(parents=True, exist_ok=True)
        r = _sh(["/usr/bin/curl", "-fL", "--retry", "3", "-o", str(UV_TARBALL),
                 f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/"
                 "uv-aarch64-apple-darwin.tar.gz"])
        if r.returncode:
            pytest.skip("uv tarball unavailable offline (data/cache/)")
    return UV_TARBALL


@pytest.fixture(scope="module")
def offline_install(tmp_path_factory, uv_tarball):
    """One offline install shared by the acceptance 2-4 tests (they run in file order)."""
    for need in (ROOT / ".uv-cache", ROOT / ".uv-python", ROOT / "models"):
        if not need.exists():
            pytest.skip(f"{need.name} missing in the dev checkout")
    tmp = tmp_path_factory.mktemp("offline")
    (tmp / "tmp").mkdir()
    # prime the dev wheel cache once (needs network only when a wheel is not cached yet)
    uv = shutil.which("uv") or str(ROOT / ".tools" / "uv" / "uv")
    prime = subprocess.run(
        [uv, "sync", "--frozen", "--no-dev", "--inexact", "--project", str(ROOT)],
        env={**os.environ, "UV_CACHE_DIR": str(ROOT / ".uv-cache"),
             "UV_PROJECT_ENVIRONMENT": str(tmp / "primevenv"),
             "UV_PYTHON_INSTALL_DIR": str(ROOT / ".uv-python")},
        capture_output=True, text=True)
    if prime.returncode:
        pytest.skip("cannot prime the wheel cache: " + prime.stderr[-300:])
    shutil.rmtree(tmp / "primevenv", ignore_errors=True)
    root = clone_repo(tmp / "ShotSieve", with_caches=True)
    env = sandbox_env(tmp, SHOTSIEVE_UV_TARBALL=str(uv_tarball),
                      SHOTSIEVE_MODELS_SRC=str(ROOT / "models"), UV_OFFLINE="1")
    # n3: a broken user uv config in the sandbox HOME must be ignored (UV_NO_CONFIG=1)
    poison = Path(env["HOME"]) / ".config" / "uv" / "uv.toml"
    poison.parent.mkdir(parents=True)
    poison.write_text(POISON_UV_TOML)
    home_before = tree_snapshot(Path(env["HOME"]))
    foot_before = system_footprint()
    t0 = time.time()
    r = subprocess.run(["/bin/bash", str(root / "install.sh")], env=env, cwd=tmp,
                       capture_output=True, text=True, timeout=600)
    dt = time.time() - t0
    info = dict(root=root, env=env, tmp=tmp, result=r, seconds=dt,
                home_before=home_before, foot_before=foot_before)
    yield info
    stop_servers(root / "data", root)


def test_offline_install(offline_install):
    i = offline_install
    # the poison really breaks uv when the config is honoured (so passing below proves the fix)
    uv = shutil.which("uv") or str(ROOT / ".tools" / "uv" / "uv")
    penv = {**i["env"], "UV_CACHE_DIR": str(i["tmp"] / "probe-cache")}  # keep HOME untouched
    cmd = [uv, "python", "list", "--offline"]
    probe = subprocess.run(cmd, env=penv, capture_output=True, text=True)
    assert probe.returncode != 0, "poisoned uv.toml has no effect: the n3 test would prove nothing"
    probe = subprocess.run(
        cmd, env={**penv, "UV_NO_CONFIG": "1"}, capture_output=True, text=True
    )
    assert probe.returncode == 0
    r, root = i["result"], i["root"]
    log = (root / "data/logs/install.log").read_text()[-2000:]
    assert r.returncode == 0, r.stdout + r.stderr + log
    assert "Done. Open ShotSieve.app" in r.stdout
    assert i["seconds"] <= 120, i["seconds"]
    for p in (".tools/uv/uv", ".venv/bin/shotsieve", "ShotSieve.app/Contents/Info.plist",
              "models/face_detection_yunet_2023mar.onnx"):
        assert (root / p).exists(), p
    assert tree_snapshot(Path(i["env"]["HOME"])) == i["home_before"]
    assert system_footprint() == i["foot_before"]
    assert installed_size_mb(root) <= 1500


def test_installed_launcher_starts_and_survives_cache_removal(offline_install):
    i = offline_install
    root = i["root"]
    shutil.rmtree(root / ".uv-cache")
    data = i["tmp"] / "rundata"
    data.mkdir()
    t0 = time.time()
    p, port = start_launcher(root / "ShotSieve.app/Contents/MacOS/ShotSieve", data,
                             HOME=i["env"]["HOME"])
    try:
        assert time.time() - t0 <= 20
        assert health(port)["models_ready"] is True
    finally:
        stop_servers(data, root)
        if p.poll() is None:
            p.kill()


def test_rerun_is_offline_quick_and_downloads_nothing(offline_install):
    i = offline_install
    root, env = i["root"], dict(i["env"])
    env.pop("SHOTSIEVE_MODELS_SRC")  # models are on disk: no source needed, nothing fetched
    env.pop("SHOTSIEVE_UV_TARBALL")  # uv is on disk
    env["UV_OFFLINE"] = "1"
    (root / ".uv-cache").mkdir(exist_ok=True)  # removed by the previous test; offline sync
    before = tree_snapshot(root / "models")
    t0 = time.time()
    r = subprocess.run(["/bin/bash", str(root / "install.sh")], env=env, cwd=i["tmp"],
                       capture_output=True, text=True, timeout=300)
    dt = time.time() - t0
    assert r.returncode == 0, r.stdout + r.stderr
    assert dt <= 60, dt
    assert tree_snapshot(root / "models") == before | {
        k: v for k, v in tree_snapshot(root / "models").items() if k.endswith("manifest.json")}
    assert "downloading" not in (root / "data/logs/install.log").read_text().split(
        "=== install")[-1]


def test_corrupt_model_is_repaired(offline_install):
    i = offline_install
    root, env = i["root"], dict(i["env"])
    env.pop("SHOTSIEVE_UV_TARBALL")
    f = root / "models" / "face_recognition_sface_2021dec.onnx"
    good = f.read_bytes()
    bad = bytearray(good)
    bad[1000] ^= 0xFF
    f.write_bytes(bytes(bad))
    r = subprocess.run(["/bin/bash", str(root / "install.sh")], env=env, cwd=i["tmp"],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    assert f.read_bytes() == good


def test_wrong_uv_tarball_sha_fails_and_extracts_nothing(tmp_path, uv_tarball):
    (tmp_path / "tmp").mkdir()
    root = clone_repo(tmp_path / "ShotSieve", with_caches=False)
    bad = tmp_path / "bad.tar.gz"
    data = bytearray(uv_tarball.read_bytes())
    data[200] ^= 0xFF
    bad.write_bytes(bytes(data))
    env = sandbox_env(tmp_path, SHOTSIEVE_UV_TARBALL=str(bad))
    r = subprocess.run(["/bin/bash", str(root / "install.sh")], env=env, cwd=tmp_path,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode != 0 and "checksum" in r.stderr
    assert not (root / ".tools" / "uv").exists()
    assert not (root / ".venv").exists()


def test_install_preflight_exit_codes(tmp_path):
    (tmp_path / "tmp").mkdir()
    root = clone_repo(tmp_path / "ShotSieve", with_caches=False)
    env = sandbox_env(tmp_path)
    r = subprocess.run(["/bin/bash", str(root / "install.sh")], env={**env,
                       "SHOTSIEVE_TEST_UNAME_S": "Linux"}, capture_output=True, text=True)
    assert r.returncode == 2 and "macOS only" in r.stderr
    r = subprocess.run(["/bin/bash", str(root / "install.sh")], env={**env,
                       "SHOTSIEVE_TEST_OS_VERSION": "12.6"}, capture_output=True, text=True)
    assert r.returncode == 2 and "macOS 14" in r.stderr
    r = subprocess.run(["/bin/bash", str(root / "install.sh")], env={**env,
                       "SHOTSIEVE_MIN_FREE_KB": str(10**12)}, capture_output=True, text=True)
    assert r.returncode == 3 and "MB free" in r.stderr
    r = subprocess.run(["/bin/bash", str(root / "install.sh"), "--bogus"], env=env,
                       capture_output=True, text=True)
    assert r.returncode == 2


@pytest.mark.network
def test_full_install_sandbox(tmp_path):
    """Acceptance 1: real downloads (uv, Python, wheels, models) in a clean sandbox HOME."""
    (tmp_path / "tmp").mkdir()
    root = clone_repo(tmp_path / "ShotSieve", with_caches=False)
    env = sandbox_env(tmp_path)
    # n3: a broken user uv config in the sandbox HOME must be ignored (UV_NO_CONFIG=1)
    poison = Path(env["HOME"]) / ".config" / "uv" / "uv.toml"
    poison.parent.mkdir(parents=True)
    poison.write_text(POISON_UV_TOML)
    home_before = tree_snapshot(Path(env["HOME"]))
    foot_before = system_footprint()
    t0 = time.time()
    r = subprocess.run(["/bin/bash", str(root / "install.sh")], env=env, cwd=tmp_path,
                       capture_output=True, text=True, timeout=1500)
    dt = time.time() - t0
    assert r.returncode == 0, r.stdout + r.stderr
    cache_mb = sum(p.stat().st_size for p in (root / ".uv-cache").rglob("*") if p.is_file()) / 1e6
    print(f"full install: {dt:.0f} s, {installed_size_mb(root):.0f} MB excl cache, "
          f"cache {cache_mb:.0f} MB")
    try:
        data = tmp_path / "rundata"
        data.mkdir()
        shutil.rmtree(root / ".uv-cache")
        p, port = start_launcher(root / "ShotSieve.app/Contents/MacOS/ShotSieve", data,
                                 HOME=env["HOME"])
        assert health(port)["app"] == "shotsieve"
    finally:
        stop_servers(tmp_path / "rundata", root)
    assert not alive(data_pid(tmp_path / "rundata"))
    assert tree_snapshot(Path(env["HOME"])) == home_before
    assert system_footprint() == foot_before
    assert installed_size_mb(root) <= 1500
