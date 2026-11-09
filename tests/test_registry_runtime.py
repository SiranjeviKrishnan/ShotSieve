
import numpy as np
import pytest

from shotsieve.models import runtime
from shotsieve.models.registry import (
    MODEL_SPECS,
    ModelMissingError,
    ensure_models,
    model_path,
    sha256_file,
    verify_model,
)


def test_specs():
    from shotsieve.models.registry import group_names

    assert set(group_names("core")) == {"yunet", "sface", "face_landmarker", "dinov2_small"}
    assert set(group_names("search")) == {"siglip2_vision", "siglip2_text", "siglip2_tokenizer"}
    assert group_names("search-int8") == ["siglip2_vision_int8"]
    assert sum(MODEL_SPECS[n].size for n in group_names("search")) <= 720_000_000
    assert [MODEL_SPECS[k].size for k in ("yunet", "sface", "face_landmarker", "dinov2_small")] == [
        232589,
        38696353,
        3758596,
        88532934,
    ]


def test_missing_model_error(tmp_models_dir):
    with pytest.raises(ModelMissingError, match="download_models"):
        model_path("yunet")
    with pytest.raises(KeyError):
        model_path("nope")
    with pytest.raises(ModelMissingError):
        ensure_models(["yunet"], download=False)


def test_wrong_size_rejected(tmp_models_dir):
    tmp_models_dir.mkdir()
    (tmp_models_dir / MODEL_SPECS["yunet"].filename).write_bytes(b"short")
    assert not verify_model("yunet")


@pytest.mark.models
def test_real_models_verified(models_ready):
    paths = ensure_models(download=False)
    assert len(paths) == 4
    from shotsieve.models.registry import read_manifest

    manifest = read_manifest()
    for name, p in paths.items():
        assert p.stat().st_size == MODEL_SPECS[name].size
        assert manifest[name]["sha256"] == sha256_file(p)


@pytest.mark.models
def test_sha_mismatch_detected(models_ready, tmp_models_dir, monkeypatch):
    import shutil

    from shotsieve.config import project_root

    real = project_root() / "models" / MODEL_SPECS["yunet"].filename
    tmp_models_dir.mkdir()
    shutil.copy(real, tmp_models_dir / real.name)
    data = bytearray((tmp_models_dir / real.name).read_bytes())
    data[100] ^= 0xFF  # flip a byte: size unchanged, pinned sha256 differs
    (tmp_models_dir / real.name).write_bytes(bytes(data))
    assert verify_model("yunet") and not verify_model("yunet", check_hash=True)


def test_np_dtype():
    assert runtime.np_dtype("tensor(float16)") == np.float16
    assert runtime.np_dtype("tensor(float)") == np.float32
    with pytest.raises(ValueError):
        runtime.np_dtype("tensor(string)")


@pytest.mark.models
def test_session_provider_and_fallback(models_ready, monkeypatch):
    path = model_path("dinov2_small")
    sess, prov = runtime.create_session(path)
    assert prov in {runtime.COREML, runtime.CPU}
    assert sess.get_inputs()[0].name == "pixel_values"

    def boom(_s):
        if not state["cpu"]:
            raise RuntimeError("coreml warmup failed")

    state = {"cpu": False}
    calls = []

    def warmup(s):
        calls.append(s.get_providers()[0])
        if s.get_providers()[0] == runtime.COREML:
            raise RuntimeError("simulated CoreML failure")

    _, prov2 = runtime.create_session(path, warmup=warmup)
    assert prov2 == runtime.CPU
    _, prov3 = runtime.create_session(path, use_coreml=False)
    assert prov3 == runtime.CPU


# ---- T18: pinned hashes, local source, resume ----------------------------------------------


def _fake_spec(monkeypatch, name, payload):
    import hashlib
    from dataclasses import replace

    from shotsieve.models import registry

    spec = replace(
        MODEL_SPECS[name], size=len(payload), sha256=hashlib.sha256(payload).hexdigest()
    )
    monkeypatch.setitem(registry.MODEL_SPECS, name, spec)
    return spec


def test_every_spec_has_sha256_and_pinned_urls():
    for spec in MODEL_SPECS.values():
        assert spec.sha256 and len(spec.sha256) == 64, spec.name
        assert "/raw/main/" not in spec.url and "/resolve/main/" not in spec.url, spec.name


def test_models_src_copy_and_sha_check(tmp_models_dir, tmp_path, monkeypatch):
    from shotsieve.models import registry

    payload = b"x" * 5000
    spec = _fake_spec(monkeypatch, "yunet", payload)
    src = tmp_path / "src"
    src.mkdir()
    (src / spec.filename).write_bytes(payload)
    monkeypatch.setenv("SHOTSIEVE_MODELS_SRC", str(src))
    monkeypatch.setattr(registry, "_download", lambda *a, **k: pytest.fail("no network"))
    got = ensure_models(["yunet"])
    assert got["yunet"].read_bytes() == payload
    # same size, wrong bytes: rejected and deleted, never accepted
    (src / spec.filename).write_bytes(b"y" * 5000)
    got["yunet"].unlink()
    with pytest.raises(ModelMissingError, match="corrupted"):
        ensure_models(["yunet"])
    assert not (tmp_models_dir / spec.filename).exists()


def test_corrupt_model_is_refetched(tmp_models_dir, monkeypatch):
    from shotsieve.models import registry

    payload = b"good-bytes" * 100
    spec = _fake_spec(monkeypatch, "yunet", payload)
    tmp_models_dir.mkdir()
    (tmp_models_dir / spec.filename).write_bytes(b"b" * len(payload))  # flipped, same size
    calls = []

    def fake(url, dest, progress=None):
        calls.append(url)
        dest.write_bytes(payload)

    monkeypatch.setattr(registry, "_download", fake)
    ensure_models(["yunet"])
    assert len(calls) == 1 and (tmp_models_dir / spec.filename).read_bytes() == payload
    ensure_models(["yunet"])  # complete now: nothing fetched
    assert len(calls) == 1


def test_download_resumes_part_with_range(tmp_path, monkeypatch):
    import io
    import urllib.request

    from shotsieve.models import registry

    payload = bytes(range(256)) * 20
    dest = tmp_path / "m.bin"
    part = tmp_path / "m.bin.part"
    part.write_bytes(payload[:1000])
    seen = {}

    class Resp(io.BytesIO):
        status = 206
        headers = {"Content-Length": str(len(payload) - 1000)}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_open(req, timeout=0):
        seen["range"] = req.headers.get("Range")
        return Resp(payload[1000:])

    monkeypatch.setattr(urllib.request, "urlopen", fake_open)
    ticks = []
    registry._download("http://x/m", dest, lambda d, t: ticks.append((d, t)))
    assert seen["range"] == "bytes=1000-"
    assert dest.read_bytes() == payload and not part.exists()
    assert ticks[-1] == (len(payload), len(payload))


def test_progress_events_and_status(tmp_models_dir, monkeypatch):
    from shotsieve.models import registry

    payload = b"z" * 3000
    spec = _fake_spec(monkeypatch, "yunet", payload)

    def fake(url, dest, progress=None):
        for k in (1000, 2000, 3000):
            progress(k, 3000)
        dest.write_bytes(payload)

    monkeypatch.setattr(registry, "_download", fake)
    ev = []
    ensure_models(["yunet"], progress=ev.append)
    assert [e["done"] for e in ev if e["file_done"] < 3000] == [1000, 2000]
    assert ev[-1]["done"] == ev[-1]["total"] == spec.size
    st = registry.models_status()
    assert st["core"]["bytes"] == registry.group_bytes("core") and not st["core"]["installed"]


def test_onnxruntime_import_does_not_touch_home(tmp_path):
    """T18 footprint: ORT 1.31 writes ~/Library/.../DeveloperTools on import; we redirect it."""
    import os
    import subprocess
    import sys

    home, data = tmp_path / "home", tmp_path / "data"
    home.mkdir()
    env = {**os.environ, "HOME": str(home), "SHOTSIEVE_DATA_DIR": str(data)}
    code = ("import shotsieve.models.runtime as r, os; r.coreml_available(); "
            "print(os.environ['HOME'])")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == str(home)  # HOME restored
    assert list(home.iterdir()) == []  # nothing written to the user's home


@pytest.mark.models
def test_cull_writes_nothing_to_home(models_ready, tmp_path, shoot_dir):
    """T18 footprint: a cull touches only the shoot folder and the data dir, never $HOME."""
    import os
    import shutil
    import subprocess
    import sys

    home, data, shoot = tmp_path / "home", tmp_path / "data", tmp_path / "shoot"
    home.mkdir()
    shoot.mkdir()
    for p in sorted(shoot_dir.glob("burst0_*.jpg")):
        shutil.copy(p, shoot / p.name)
    env = {**os.environ, "HOME": str(home), "SHOTSIEVE_DATA_DIR": str(data)}
    env.pop("MPLCONFIGDIR", None)
    r = subprocess.run([sys.executable, "-m", "shotsieve.cli", "cull", str(shoot)], env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert list(home.iterdir()) == []


def test_user_home_is_immune_to_the_ort_home_swap(tmp_path):
    """m4: while `import onnxruntime` runs with HOME redirected, other threads still see the real
    HOME through `user_home()`; HOME is restored afterwards."""
    import os
    import subprocess
    import sys
    import textwrap

    fake = tmp_path / "fake"
    fake.mkdir()
    (fake / "onnxruntime.py").write_text(textwrap.dedent("""
        import os, threading
        from shotsieve._ort import user_home
        seen = {}
        seen["env_home"] = os.environ["HOME"]
        t = threading.Thread(target=lambda: seen.update(thread=str(user_home())))
        t.start(); t.join()
        def disable_telemetry_events(): pass
    """))
    home = tmp_path / "home"
    home.mkdir()
    code = textwrap.dedent("""
        import os, sys
        sys.path.insert(0, {fake!r})
        from shotsieve import _ort
        ort = _ort.import_onnxruntime()
        print(ort.seen["env_home"]); print(ort.seen["thread"]); print(os.environ["HOME"])
        print(_ort.user_home())
    """).format(fake=str(fake))
    env = {**os.environ, "HOME": str(home), "SHOTSIEVE_DATA_DIR": str(tmp_path / "d")}
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
    )
    env_home, thread_home, after, after2 = out.stdout.split()
    assert env_home.endswith("ort-home")
    assert thread_home == str(home) == after == after2


def test_download_hashes_part_before_the_final_name(tmp_models_dir, monkeypatch):
    """m8: wrong bytes of the right size never appear under the final name."""
    import io
    import urllib.request

    from shotsieve.models import registry

    good = b"g" * 3000
    spec = _fake_spec(monkeypatch, "yunet", good)
    dest = registry.models_dir() / spec.filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    final_seen = []

    class Resp(io.BytesIO):
        status = 200
        headers = {"Content-Length": "3000"}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=0: Resp(b"b" * 3000))
    orig_replace = type(dest).replace

    def spy_replace(self, target):
        final_seen.append(str(target))
        return orig_replace(self, target)

    monkeypatch.setattr(type(dest), "replace", spy_replace)
    with pytest.raises(registry.ModelMissingError):
        registry._download(spec.url, dest)
    assert final_seen == [] and not dest.exists()
    assert not dest.with_name(dest.name + ".part").exists()  # the bad part is deleted
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=0: Resp(good))
    registry._download(spec.url, dest)
    assert dest.read_bytes() == good and final_seen == [str(dest)]
