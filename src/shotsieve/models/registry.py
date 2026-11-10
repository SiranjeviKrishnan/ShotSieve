"""Model manifest: specs, locations, download and verification."""

from __future__ import annotations

import functools
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from shotsieve.config import bundle_resources, project_root
from shotsieve.log import get_logger

log = get_logger("registry")


class ModelMissingError(FileNotFoundError):
    """A required model file is absent or corrupt."""


@dataclass(frozen=True)
class ModelSpec:
    name: str
    filename: str
    url: str
    sha256: str | None
    size: int
    license: str
    group: str = "core"  # core | rawdenoise | search | search-int8 | ocr | crop | denoise


# pinned to a commit (T18); bytes verified against the sha256 below
_ZOO_SHA = "48b10fa505aeb51fbd127fc60a5c660040e823b7"
_ZOO = f"https://github.com/opencv/opencv_zoo/raw/{_ZOO_SHA}/models"

MODEL_SPECS: dict[str, ModelSpec] = {
    "yunet": ModelSpec(
        "yunet",
        "face_detection_yunet_2023mar.onnx",
        f"{_ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        None,
        232589,
        "MIT",
    ),
    "sface": ModelSpec(
        "sface",
        "face_recognition_sface_2021dec.onnx",
        f"{_ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        None,
        38696353,
        "Apache-2.0",
    ),
    "face_landmarker": ModelSpec(
        "face_landmarker",
        "face_landmarker.task",
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/"
        "float16/1/face_landmarker.task",
        None,
        3758596,
        "Apache-2.0",
    ),
    "dinov2_small": ModelSpec(
        "dinov2_small",
        "dinov2_small_fp32.onnx",
        "https://huggingface.co/onnx-community/dinov2-small/resolve/"
        "8b1f705a3a7f6f062f6bdd21986c1583d3ef105d/onnx/model.onnx",
        None,
        88532934,
        "Apache-2.0",
    ),
}

_SIGLIP_SHA = "ba1f3b0843f24bc5417d38e19c37b287d719b2f4"
_SIGLIP = (
    f"https://huggingface.co/onnx-community/siglip2-base-patch16-224-ONNX/resolve/{_SIGLIP_SHA}"
)
for _name, _file, _remote, _size in (
    ("siglip2_vision", "siglip2_vision_fp32.onnx", "onnx/vision_model.onnx", 371807752),
    ("siglip2_vision_int8", "siglip2_vision_int8.onnx", "onnx/vision_model_quantized.onnx",
     94553333),  # group "search-int8": gate alternative, failed the gate (docs/02_MODELS.md)
    ("siglip2_text", "siglip2_text_int8.onnx", "onnx/text_model_quantized.onnx", 283438275),
    ("siglip2_tokenizer", "siglip2_tokenizer.json", "tokenizer.json", 34363039),
):  # fmt: skip
    MODEL_SPECS[_name] = ModelSpec(
        _name, _file, f"{_SIGLIP}/{_remote}", None, _size, "Apache-2.0",
        "search-int8" if _name.endswith("int8") and "vision" in _name else "search",
    )

# monkt/paddleocr-onnx: PaddleOCR PP-OCRv3 det + PP-OCRv5 en rec (Apache-2.0)
_PPOCR_SHA = "7b02d0a30a07ba2b92ad1ff5a8941ae2c633de65"
_PPOCR = f"https://huggingface.co/monkt/paddleocr-onnx/resolve/{_PPOCR_SHA}"
for _name, _file, _remote, _size in (
    ("ppocr_det", "ppocr_det_v3.onnx", "detection/v3/det.onnx", 2429873),
    ("ppocr_rec", "ppocr_rec_en.onnx", "languages/english/rec.onnx", 7830888),
    ("ppocr_dict", "ppocr_en_dict.txt", "languages/english/dict.txt", 1416),
):  # fmt: skip
    MODEL_SPECS[_name] = ModelSpec(
        _name, _file, f"{_PPOCR}/{_remote}", None, _size, "Apache-2.0", "ocr"
    )

# T22 crop: YOLOX-s (Megvii, Apache-2.0) from the same pinned opencv_zoo commit as YuNet/SFace
MODEL_SPECS["yolox_s"] = ModelSpec(
    "yolox_s",
    "yolox_s_2022nov.onnx",
    f"{_ZOO}/object_detection_yolox/object_detection_yolox_2022nov.onnx",
    None,
    35858002,
    "Apache-2.0",
    "crop",
)

# T23f denoise (view only): NAFNet-SIDD-width32 (Chen et al., MIT), ONNX by tf2onnx from the
# litert-community TFLite (MIT), pinned to a repo commit
MODEL_SPECS["nafnet_sidd"] = ModelSpec(
    "nafnet_sidd",
    "nafnet_sidd_width32_fp16.onnx",
    "https://huggingface.co/mlboydaisuke/NAFNet-SIDD-width32-OpenVINO/resolve/"
    "ccea65867e9e54460335054c81eea25524edac4d/nafnet_sidd_width32_fp16.onnx",
    None,
    123972868,
    "MIT",
    "denoise",
)

# T26 RawNIND UtNet2 (Bayer RAW denoiser, bundled with every install, not an optional download):
# the author's checkpoint (weights (c) Benoit Brummer, Christophe De Vleeschouwer, CC-BY-4.0
# option of their GPL-3.0 / CC-BY-4.0 dual licence) converted to a static 512 ONNX by
# scripts/export_rawnind.py (torch 2.2.2, opset 17; checkpoint sha256 c1f7de3d...), bit-identical
# to darktable's ONNX of the same checkpoint on real tiles. Hosted as an asset of a ShotSieve
# GitHub release (models-v1, uploaded 2026-11-10). See docs/02_MODELS.md ("RawNIND: hosting").
RAWNIND_RELEASE_URL = (
    "https://github.com/SiranjeviKrishnan/ShotSieve/releases/download/models-v1/"
    "rawnind_utnet2_bayer_512.onnx"
)
MODEL_SPECS["rawnind_utnet2"] = ModelSpec(
    "rawnind_utnet2",
    "rawnind_utnet2_bayer_512.onnx",
    RAWNIND_RELEASE_URL,
    None,
    31054248,
    "CC-BY-4.0",
    "rawdenoise",
)

PINNED_SHA256: dict[str, str] = {
    "rawnind_utnet2": (
        "a7978fd669618d373cf7f1f72010f43776991d31f4bd9d10e83f2c0e52236c17"
    ),
    "nafnet_sidd": (
        "a9ca266868fad47ce1d19ae2677809e8e69a3c131afbe54ba24c431fe0dc069a"
    ),
    "yolox_s": (
        "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063"
    ),
    "dinov2_small": (
        "f22797eabf810a75e41de68d378541ebea372122b25c4ce3ef25ff618250c20a"
    ),
    "face_landmarker": (
        "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
    ),
    "ppocr_det": (
        "ee40e80071ba3a320d4efda75f3e22047a7d049e9bf7bcaaf9daea23fc21b935"
    ),
    "ppocr_dict": (
        "e025a66d31f327ba0c232e03f407ae8d105e1e709e7ccb3f408aa778c24e70d6"
    ),
    "ppocr_rec": (
        "4e16deb22c4da6468bdca539b2cd3c8687825538b67109177c47d359ab994cd7"
    ),
    "sface": (
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
    ),
    "siglip2_text": (
        "3a0603d3a00c05a80a6ded4743c16aaac7b1e62cdcc7e362e7ce418659b96400"
    ),
    "siglip2_tokenizer": (
        "cb9140fae3ac5122c972d37adf83e1248471a38147ad76f8215c8872c6fd8322"
    ),
    "siglip2_vision": (
        "c0573e3f4140c3a7c4e9cc5912bd6b26a033b46a6a8e8af26cbea262b163bcad"
    ),
    "siglip2_vision_int8": (
        "5f2b401c1a4fc095702a5d45348e17ad46c4f87064085365b43c6e8eaa5c0070"
    ),
    "yunet": (
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
    ),
}

# T18: every spec carries its sha256 (no "trust on first use")
MODEL_SPECS = {n: replace(spec, sha256=PINNED_SHA256[n]) for n, spec in MODEL_SPECS.items()}

GROUPS = ("core", "rawdenoise", "search", "search-int8", "ocr", "crop", "denoise")


def group_names(*groups: str) -> list[str]:
    return [n for n, s in MODEL_SPECS.items() if s.group in groups]


def models_dir() -> Path:
    """Model folder: SHOTSIEVE_MODELS_DIR, else Contents/Resources/models inside a built app
    (read-only, shipped with the app), else <project>/models."""
    env = os.environ.get("SHOTSIEVE_MODELS_DIR")
    if env:
        return Path(env)
    res = bundle_resources()
    return res / "models" if res else project_root() / "models"


BUNDLE_MANIFEST = "bundle.json"


@functools.lru_cache(maxsize=4)
def _bundle_manifest(folder: str) -> dict[str, dict]:
    try:
        return json.loads((Path(folder) / BUNDLE_MANIFEST).read_text())
    except (OSError, ValueError):
        return {}


def bundled() -> bool:
    """True when the model folder is a release bundle (it has `bundle.json`, written by
    scripts/build_release.sh after verifying every file against the pinned sha256). Such a
    folder is complete and read-only: nothing is ever downloaded into it."""
    return bool(_bundle_manifest(str(models_dir())))


def _effective(spec: ModelSpec) -> tuple[str, int, str | None]:
    """(filename, size, sha256) of the file that is actually used. A bundle may ship a derived
    variant of a model (e.g. the repaired NAFNet) under the same name."""
    ent = _bundle_manifest(str(models_dir())).get(spec.name)
    if ent:
        return ent["filename"], int(ent["size"]), ent.get("sha256")
    return spec.filename, spec.size, spec.sha256


def _manifest_path() -> Path:
    return models_dir() / "manifest.json"


def read_manifest() -> dict[str, dict]:
    try:
        return json.loads(_manifest_path().read_text())
    except (OSError, ValueError):
        return {}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _expected_sha(spec: ModelSpec) -> str | None:
    return spec.sha256 or read_manifest().get(spec.name, {}).get("sha256")


def verify_model(name: str, check_hash: bool = False) -> bool:
    """True if the file exists with the expected size (and sha256 when `check_hash`)."""
    spec = MODEL_SPECS[name]
    filename, size, sha = _effective(spec)
    path = models_dir() / filename
    if not path.is_file() or path.stat().st_size != size:
        return False
    if check_hash:
        expected = sha or _expected_sha(spec)
        return expected is None or sha256_file(path) == expected
    return True


def model_path(name: str) -> Path:
    """Path to a model file; raises ModelMissingError if absent or wrong size."""
    if name not in MODEL_SPECS:
        raise KeyError(f"unknown model {name!r}; known: {sorted(MODEL_SPECS)}")
    path = models_dir() / _effective(MODEL_SPECS[name])[0]
    if not verify_model(name):
        raise ModelMissingError(
            f"model {name!r} missing or corrupt at {path}; run `python scripts/download_models.py`"
        )
    return path


def group_bytes(*groups: str) -> int:
    return sum(MODEL_SPECS[n].size for n in group_names(*groups))


def models_status() -> dict[str, dict]:
    """Per group: bytes, installed (size check only, cheap), missing names. Listing only.
    In a release bundle the groups the bundle does not carry (search-int8) are omitted."""
    out: dict[str, dict] = {}
    for g in GROUPS:
        names = group_names(g)
        if bundled():
            names = [n for n in names if n in _bundle_manifest(str(models_dir()))]
            if not names:
                continue
        missing = [n for n in names if not verify_model(n)]
        out[g] = {
            "bytes": group_bytes(g),
            "installed": not missing,
            "missing": missing,
            "files": len(names),
        }
    return out


ProgressCb = Callable[[dict], None]


def _verify_part(spec: ModelSpec | None, part: Path) -> None:
    """Size + sha256 of a finished `.part` BEFORE it gets its final name (deleted on mismatch)."""
    if spec is not None:
        _check_fetched(spec, part)


def _spec_for_url(url: str) -> ModelSpec | None:
    return next((s for s in MODEL_SPECS.values() if s.url == url), None)


def _download(url: str, dest: Path, progress: Callable[[int, int], None] | None = None) -> None:
    """Fetch `url` to `dest` via `dest.part`, resuming a partial file with an HTTP Range request.

    `progress(done, total)` is called per chunk with bytes of this file. A failure leaves the
    `.part` so the next call resumes it.
    """
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": "shotsieve/0.1"}
    if have:
        headers["Range"] = f"bytes={have}-"
    req = urllib.request.Request(url, headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=120)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and have:  # the .part is already complete (or stale)
            _verify_part(_spec_for_url(url), part)
            part.replace(dest)
            return
        raise
    with resp:
        status = getattr(resp, "status", 200)
        length = int(resp.headers.get("Content-Length") or 0)
        if have and status == 206:
            mode, done, total = "ab", have, have + length
        else:  # server ignored Range (or fresh start): restart the file
            mode, done, total = "wb", 0, length
        with open(part, mode) as out:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)
    _verify_part(_spec_for_url(url), part)  # a wrong file never sits under the final name
    part.replace(dest)


def _fetch_local(src_dir: Path, spec: ModelSpec, dest: Path) -> bool:
    """SHOTSIEVE_MODELS_SRC: clone/copy from a local folder (offline installs, tests)."""
    src = src_dir / spec.filename
    if not src.is_file():
        return False
    part = dest.with_name(dest.name + ".part")
    part.unlink(missing_ok=True)
    cloned = False
    if sys.platform == "darwin":
        cloned = subprocess.run(
            ["/bin/cp", "-c", str(src), str(part)], capture_output=True
        ).returncode == 0
    if not cloned:
        shutil.copyfile(src, part)
    _verify_part(spec, part)
    part.replace(dest)
    return True


def _check_fetched(spec: ModelSpec, path: Path) -> None:
    """Size + sha256 of a freshly fetched file; a bad file is deleted."""
    problem = None
    if path.stat().st_size != spec.size:
        problem = f"size {path.stat().st_size} != expected {spec.size}"
    elif spec.sha256 and sha256_file(path) != spec.sha256:
        problem = "sha256 mismatch"
    if problem:
        path.unlink(missing_ok=True)
        raise ModelMissingError(
            f"{spec.name}: download corrupted ({problem}), run the install again"
        )


def _file_progress(emit: Callable, name: str, idx: int, size: int, done: int, total: int) -> None:
    emit(name, done, total or size, idx)


def ensure_models(
    names: list[str] | None = None,
    download: bool = True,
    groups: tuple[str, ...] | None = None,
    progress: ProgressCb | None = None,
) -> dict[str, Path]:
    """Make sure the named models are present and verified (default: the core group only;
    `groups=("search",)` selects an extra group).

    Present files are verified against the pinned sha256; missing/corrupt ones are copied from
    `SHOTSIEVE_MODELS_SRC` (when set) or downloaded to `.part` (resumable), then size- and
    hash-checked (a mismatch deletes the file). `progress` receives dicts
    `{name, file_done, file_total, done, total, files_done, files}` (bytes).
    """
    out: dict[str, Path] = {}
    mdir = models_dir()
    if bundled():  # a release bundle: read-only and complete, never download or rewrite
        for name in names or group_names(*(groups or ("core",))):
            out[name] = model_path(name)
        return out
    mdir.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest()
    todo = names or group_names(*(groups or ("core",)))
    total = sum(MODEL_SPECS[n].size for n in todo)
    done_before = 0
    src_env = os.environ.get("SHOTSIEVE_MODELS_SRC")

    def emit(name: str, fdone: int, ftotal: int, nfiles: int) -> None:
        if progress:
            progress({"name": name, "file_done": fdone, "file_total": ftotal,
                      "done": done_before + fdone, "total": total,
                      "files_done": nfiles, "files": len(todo)})

    for i, name in enumerate(todo):
        spec = MODEL_SPECS[name]
        path = mdir / spec.filename
        if path.is_file() and verify_model(name, check_hash=True):
            out[name] = path
        else:
            if not download:
                raise ModelMissingError(f"model {name!r} missing or corrupt at {path}")
            path.unlink(missing_ok=True)  # corrupt: replace
            if not (src_env and _fetch_local(Path(src_env), spec, path)):
                log.info("downloading %s", spec.url)
                cb = functools.partial(_file_progress, emit, name, i, spec.size)
                _download(spec.url, path, cb)
            _check_fetched(spec, path)
            out[name] = path
        manifest[name] = {
            "filename": spec.filename,
            "size": spec.size,
            "sha256": spec.sha256 or sha256_file(path),
            "url": spec.url,
            "license": spec.license,
        }
        emit(name, spec.size, spec.size, i + 1)
        done_before += spec.size
    tmp = _manifest_path().with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    tmp.replace(_manifest_path())
    return out
