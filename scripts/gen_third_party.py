# ruff: noqa: E501
"""Regenerate THIRD_PARTY_NOTICES.md from the lockfile and the model registry.

    uv run python scripts/gen_third_party.py            # rewrite THIRD_PARTY_NOTICES.md
    uv run python scripts/gen_third_party.py --check    # exit 1 when it is out of date

The Python package list is `uv export --frozen --no-dev` (exactly what scripts/build_release.sh
installs into the app) with markers evaluated for macOS arm64; licence names come from the table
below (taken from each package's metadata, verified by hand: a package that is missing from the
table fails the run so a new dependency cannot slip in unreviewed).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

from packaging.requirements import Requirement

from shotsieve.models.registry import MODEL_SPECS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from third_party_dylibs import CV2_DYLIBS, OTHER_DYLIBS, PYTHON_COMPONENTS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "THIRD_PARTY_NOTICES.md"
MAC = {"sys_platform": "darwin", "platform_machine": "arm64", "os_name": "posix",
       "platform_system": "Darwin", "python_version": "3.11", "implementation_name": "cpython",
       "platform_python_implementation": "CPython", "python_full_version": "3.11.15"}

BSD3, MIT, APACHE = "BSD-3-Clause", "MIT", "Apache-2.0"
LICENSES: dict[str, str] = {
    "absl-py": APACHE, "annotated-doc": MIT, "annotated-types": MIT, "anyio": MIT, "bottle": MIT,
    "certifi": "MPL-2.0", "cffi": "MIT-0", "click": BSD3, "cloudpickle": BSD3,
    "contourpy": BSD3, "cycler": BSD3, "exifread": BSD3, "fastapi": MIT, "filelock": MIT,
    "flatbuffers": APACHE, "fonttools": MIT, "fsspec": BSD3, "h11": MIT, "hf-xet": APACHE,
    "httpcore": BSD3, "httpcore2": BSD3, "httpx": BSD3, "httpx2": BSD3,
    "huggingface-hub": APACHE, "idna": BSD3, "joblib": BSD3, "kiwisolver": BSD3,
    "matplotlib": "Matplotlib licence (PSF-based, BSD-compatible)", "mediapipe": APACHE,
    "narwhals": MIT, "numpy": "BSD-3-Clause (+ bundled 0BSD, MIT, Zlib, CC0-1.0 parts)",
    "onnxruntime": MIT,
    "opencv-contrib-python": "Apache-2.0 (OpenCV); the wheel bundles FFmpeg and codecs, some GPL: see below",
    "opentelemetry-api": APACHE, "packaging": "Apache-2.0 OR BSD-2-Clause", "piexif": MIT,
    "pillow": "MIT-CMU (HPND)",
    "pillow-heif": "BSD-3-Clause; the binary wheel bundles libheif, libde265 (LGPL-3.0) and x265 (GPL-2.0): see below",
    "protobuf": BSD3, "proxy-tools": MIT, "psutil": BSD3, "pycparser": BSD3, "pydantic": MIT,
    "pydantic-core": MIT, "pyobjc-core": MIT, "pyobjc-framework-cocoa": MIT,
    "pyobjc-framework-quartz": MIT, "pyobjc-framework-security": MIT,
    "pyobjc-framework-uniformtypeidentifiers": MIT, "pyobjc-framework-webkit": MIT,
    "pyparsing": MIT, "python-dateutil": "BSD-3-Clause / Apache-2.0", "python-multipart": APACHE,
    "pywebview": BSD3, "pyyaml": MIT,
    "rawpy": "MIT; bundles LibRaw 0.22.2 (LGPL-2.1 or CDDL-1.0, dynamically linked) and JasPer, libjpeg-turbo, Little CMS: see below",
    "scikit-learn": BSD3,
    "scipy": "BSD-3-Clause; bundles libgfortran / libgcc_s (GPL-3.0 with GCC Runtime Library Exception) and libquadmath (LGPL-2.1)",
    "sounddevice": MIT, "sqlite-vec": "MIT OR Apache-2.0", "starlette": BSD3,
    "threadpoolctl": BSD3, "tokenizers": APACHE, "tomli-w": MIT, "tqdm": "MPL-2.0 AND MIT",
    "truststore": MIT, "typing-extensions": "PSF-2.0", "typing-inspection": MIT, "tzdata": APACHE,
    "uvicorn": BSD3,
}  # fmt: skip

# name -> (task, upstream credit, citation)
MODEL_INFO: dict[str, tuple[str, str, str]] = {
    "yunet": ("Face detection", "Shiqi Yu et al. (libfacedetection / YuNet), via OpenCV Zoo",
              "Wu, W., Peng, H., Yu, S. YuNet: A Tiny Millisecond-level Face Detector. Machine Intelligence Research, 2023."),
    "sface": ("Face recognition (embeddings)", "Zhong et al., via OpenCV Zoo",
              "Zhong, Y., Deng, W., et al. SFace: Sigmoid-Constrained Hypersphere Loss for Robust Face Recognition. IEEE TIP, 2021."),
    "face_landmarker": ("Eye / face landmarks (blink detection)", "Google MediaPipe",
                        "Kartynnik, Y. et al. Real-time Facial Surface Geometry from Monocular Video on Mobile GPUs. 2019."),
    "dinov2_small": ("Image embeddings (near-duplicates, grouping)", "Meta AI; ONNX export by onnx-community",
                     "Oquab, M. et al. DINOv2: Learning Robust Visual Features without Supervision. TMLR, 2024."),
    "siglip2_vision": ("Natural-language search: image encoder", "Google; ONNX conversion by onnx-community",
                       "Tschannen, M. et al. SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features. arXiv:2502.14786, 2025."),
    "siglip2_text": ("Natural-language search: text encoder (int8)", "Google; ONNX conversion by onnx-community",
                     "as above (SigLIP 2)"),
    "siglip2_tokenizer": ("Natural-language search: tokenizer", "Google (SentencePiece vocabulary shipped in the SigLIP 2 repository)",
                          "as above (SigLIP 2)"),
    "siglip2_vision_int8": ("Alternative int8 image encoder (NOT shipped)", "Google; onnx-community",
                            "as above (SigLIP 2)"),
    "ppocr_det": ("Jersey-number OCR: text detection (PP-OCRv3)", "PaddlePaddle / PaddleOCR; ONNX by monkt",
                  "Li, C. et al. PP-OCRv3: More Attempts for the Improvement of Ultra Lightweight OCR System. arXiv:2206.03001, 2022."),
    "ppocr_rec": ("Jersey-number OCR: text recognition (English)", "PaddlePaddle / PaddleOCR; ONNX by monkt",
                  "Cui, C. et al. PaddleOCR 3.0 Technical Report. arXiv:2507.05595, 2025."),
    "ppocr_dict": ("Jersey-number OCR: character dictionary", "PaddlePaddle / PaddleOCR; monkt", "as above"),
    "yolox_s": ("Crop suggestions: subject detection (COCO classes)", "Megvii (YOLOX), via OpenCV Zoo",
                "Ge, Z. et al. YOLOX: Exceeding YOLO Series in 2021. arXiv:2107.08430, 2021."),
    "nafnet_sidd": ("Denoise (view only)", "Megvii (NAFNet); TFLite by litert-community; ONNX by mlboydaisuke",
                    "Chen, L., Chu, X., Zhang, X., Sun, J. Simple Baselines for Image Restoration. ECCV, 2022 (trained on SIDD: Abdelhamed et al., CVPR 2018)."),
}

HEADER = """\
# Third-party notices

ShotSieve's own source code is licensed under the Apache License 2.0 (see `LICENSE` and `NOTICE`).
The release app (`ShotSieve.app`, the `.dmg`) is a bundle that also contains third-party software and
machine-learning models, each under its own licence, **including GPL-2.0 / GPL-3.0 and LGPL parts
(FFmpeg, x264, x265, ...): the `.dmg` as a whole is therefore not purely Apache-2.0.** This file is
generated by `scripts/gen_third_party.py` from `uv.lock`, the model registry and
`scripts/third_party_dylibs.py`; do not edit it by hand.

## Summary of what needs attention

| Component | Licence | What it means for redistributing the `.dmg` |
|---|---|---|
| **FFmpeg 7.1.1 (Homebrew 7.1.1_3) incl. libpostproc, x264 r3108 (31e19f9), x265 4.1+1-1d117be, Rubber Band 4.0.0, vid.stab 1.1.1, inside the `opencv-contrib-python` 5.0.0.93 wheel (`cv2/.dylibs`)** | **GPL-3.0-or-later** as a whole (FFmpeg is configured `--enable-gpl --enable-version3`; x264, Rubber Band and vid.stab are GPL-2.0-or-later, x265 GPL-2.0) | **Not permissive and not LGPL.** The `.dmg` carries GPL code, so redistributing it carries the GPL obligations for those components (licence text, corresponding source, no further restrictions). ShotSieve's own source stays Apache-2.0 (Apache-2.0 code may be combined into a GPL-3.0 work). Source: see "Corresponding source" below. |
| **x265 4.3+1-e9b8812 inside the `pillow-heif` 1.8.1 wheel** (`pillow_heif/.dylibs/libx265.217.dylib`) | **GPL-2.0** | Same as above. libheif 1.23.6 and libde265 1.1.3 in the same wheel are LGPL-3.0. |
| LGPL libraries in `cv2/.dylibs`: GnuTLS, Nettle, GMP, libunistring, libidn2, libssh, libsoxr, LAME, libbluray, aribb24, libtasn1, GLib, gettext (libintl), FriBidi (+ Graphite2, tri-licensed) | LGPL-2.0 / 2.1 / 3.0 (+ or later) | Dynamically linked shared libraries, replaceable by the user (see "Replacing an LGPL library"); source in the sources archive. |
| LibRaw 0.22.2 inside `rawpy` | LGPL-2.1 **or** CDDL-1.0 | Same as the LGPL row. |
| libgfortran / libgcc_s inside `scipy` | GPL-3.0 with GCC Runtime Library Exception | The exception permits redistribution with any program. libquadmath is LGPL-2.1. |
| `certifi`, `tqdm`; **libzmq 4.3.5, libsrt 1.5.4** (`cv2/.dylibs`) | MPL-2.0 | File-level copyleft: unmodified, shipped as built by their upstream builders; the source is in the sources archive (MPL-2.0 s.3.2). Text: `licenses/MPL-2.0.txt`. |
| everything else | MIT, BSD, ISC, Apache-2.0, PSF, HPND, Zlib, ... | Keep the notices (inside each package's `*.dist-info`, `licenses/dylibs/`, `licenses/python-build-standalone/`). |

All models below allow redistribution in binary form (MIT or Apache-2.0); none is non-commercial or
research-only.

## Corresponding source and written offer (GPL / LGPL / MPL components)

**Where the source is.** Each GitHub Release that carries a `.dmg` also carries
`ShotSieve-<version>-gpl-sources.tar.gz` (and its `.sha256`) next to it: the exact upstream source
releases of every GPL, LGPL and MPL component listed here, the Homebrew formulae and patches that built them
(homebrew-core commit `a671b0c7ae230e99a07f541b471a974e3dbfd403`), the opencv-python build scripts (tag
`93`), the pillow-heif 1.8.1 and rawpy 0.28.0 sdists, and a README mapping each source to the binary it
produced. That is equivalent access from the same place as the binary (GPL-2.0 s.3, GPL-3.0 s.6(d)).
`scripts/make_gpl_sources.sh` rebuilds the archive from the official upstreams and verifies every
checksum. The identification of the versions (strings, library versions, wheel metadata) is in the
archive's README.

**Written offer (valid for at least three years from the date of the release, GPL-2.0 s.3(b), GPL-3.0
s.6(b)).** If the archive is no longer available from the release page, ask for the complete
corresponding source of the GPL/LGPL/MPL components in the `.dmg` by contacting the maintainer through
the channel given in `SECURITY.md` (GitHub private vulnerability report:
https://github.com/SiranjeviKrishnan/shotsieve/security/advisories/new, or an issue at
https://github.com/SiranjeviKrishnan/shotsieve/issues). The source is sent on request for no more
than the cost of physically performing the distribution. Maintainer contact: open an issue titled "GPL source request" at
https://github.com/SiranjeviKrishnan/shotsieve/issues.

## Components under the GPL / LGPL

### GPL components (via the OpenCV and pillow-heif wheels)

Identified from the bundled binaries (`strings`, `otool -L`, wheel metadata):

* **FFmpeg 7.1.1**: the Homebrew bottle `ffmpeg 7.1.1_3` (`--prefix=/opt/homebrew/Cellar/ffmpeg/7.1.1_3
  --enable-gpl --enable-version3 ...`) as delocated into `opencv-contrib-python` 5.0.0.93
  (`libavcodec.61`, `libavformat.61`, `libavutil.59`, `libavfilter.10`, `libavdevice.61`, `libswscale.8`,
  `libswresample.5`, `libpostproc.58`), with **x264 r3108 (31e19f9)** (`libx264.164`),
  **x265 4.1+1-1d117be** (`libx265.215`), **Rubber Band 4.0.0** (`librubberband.3`) and **vid.stab
  1.1.1** (`libvidstab.1.2`). ShotSieve uses it only to open video files for frame grabs (HEVC goes
  through AVFoundation first); it encodes nothing. Upstream: https://ffmpeg.org/releases/
  (ffmpeg-7.1.1.tar.xz), https://code.videolan.org/videolan/x264 (commit 31e19f92f00c7003fa115047ce50978bc98c3a0d),
  https://bitbucket.org/multicoreware/x265_git (4.1), https://breakfastquay.com/rubberband/,
  https://github.com/georgmartius/vid.stab (v1.1.1), the wheel's build scripts
  https://github.com/opencv/opencv-python (tag `93`).
* **x265 4.3+1-e9b8812, libde265 1.1.3, libheif 1.23.6** inside `pillow-heif` 1.8.1 (the wheel's
  `LICENSES_bundled.txt` is kept in `pillow_heif-*.dist-info/licenses/` and says x265 and the wheel
  as a whole are GPLv2; its text claims "x265 4.3": the binary reports `4.3+1-e9b8812`, which is the
  x265 4.3 release tarball). Upstream: https://github.com/Multicorewareinc/x265/releases/tag/4.3,
  https://github.com/strukturag/libde265, https://github.com/strukturag/libheif, build script
  `libheif/build_libs.py` of https://pypi.org/project/pillow-heif/1.8.1/.
* Texts: `licenses/GPL-2.0.txt`, `licenses/GPL-3.0.txt`, `licenses/LGPL-2.1.txt`,
  `licenses/LGPL-3.0.txt`, `licenses/MPL-2.0.txt`.

If you need an Apache-only/permissive-only build (no GPL code), build from source without the
`opencv-contrib-python` wheel's FFmpeg (a headless OpenCV built with `-DWITH_FFMPEG=OFF`) and
without `pillow-heif`; ShotSieve then reads video through AVFoundation only and has no HEIC support.

### LGPL: LibRaw (rawpy) and the other LGPL libraries

`rawpy` bundles LibRaw 0.22.2 (https://www.libraw.org/, dual-licensed LGPL-2.1 / CDDL-1.0) as the
shared library `libraw_r.25.0.0.dylib` (`.25` is its soname), loaded dynamically. LibRaw source:
https://github.com/LibRaw/LibRaw (tag 0.22.2, commit b93f6e45c194f5df9b02a43b1af9a54b4f41f33f). The
LGPL libraries of `cv2/.dylibs` and `pillow_heif/.dylibs` are listed in the table below.

### Replacing an LGPL library

Every LGPL library in the app is a separate shared library (`.dylib`) loaded at run time, so you can
use a modified version: build it from the source in the sources archive, put the result in
`ShotSieve.app/Contents/Resources/python/lib/python3.11/site-packages/<folder>/` (`cv2/.dylibs/`,
`pillow_heif/.dylibs/`, `rawpy/`) under the same file name, and re-sign the app
(`codesign --force --deep -s - ShotSieve.app`; the app is ad-hoc signed). This applies to LibRaw, libheif,
libde265, GnuTLS, Nettle, GMP, libunistring, libidn2, libssh, libsoxr, LAME, libbluray, aribb24,
libtasn1, GLib, libintl, FriBidi and libquadmath.

"""


def packages() -> list[tuple[str, str]]:
    uv = shutil.which("uv") or str(ROOT / ".tools" / "uv" / "uv")
    text = subprocess.run(
        [uv, "export", "--frozen", "--no-dev", "--no-emit-project", "--no-hashes", "-q"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout
    out: dict[str, str] = {}
    for line in text.splitlines():
        if not re.match(r"^[A-Za-z0-9_.-]+==", line):
            continue
        req = Requirement(line.split(" \\")[0].strip())
        if req.marker is not None and not req.marker.evaluate(MAC):
            continue
        out[req.name.lower()] = str(req.specifier).lstrip("=")
    return sorted(out.items())


# onnx-community conversions declare no licence of their own: covered by the base model's licence
ONNX_COMMUNITY_BASE = {
    "dinov2_small": "facebook/dinov2-small",
    "siglip2_vision": "google/siglip2-base-patch16-224",
    "siglip2_text": "google/siglip2-base-patch16-224",
    "siglip2_vision_int8": "google/siglip2-base-patch16-224",
}


def dylib_sections() -> str:
    kind_name = {"gpl": "GPL", "lgpl": "LGPL", "mpl": "MPL-2.0", "permissive": "permissive"}
    rows = []
    for proj, ver, lic, kind, txt, files in CV2_DYLIBS:
        where = f"`licenses/dylibs/{txt}.txt`" if txt else {
            "gpl": "`licenses/GPL-2.0.txt`, `licenses/GPL-3.0.txt`",
            "lgpl": "`licenses/LGPL-2.1.txt`, `licenses/LGPL-3.0.txt`",
            "mpl": "`licenses/MPL-2.0.txt`"}[kind]
        rows.append(f"| {proj} | {ver} | {lic} | {kind_name[kind]} | {', '.join(f'`{f}`' for f in files)} | {where} |")
    orows = []
    for loc, proj, ver, lic, text, files in OTHER_DYLIBS:
        orows.append(f"| `{loc}/` | {proj} | {ver} | {lic} | {', '.join(f'`{f}`' for f in files)} | {text} |")
    prows = [f"| {n} | {v} | {lic} | `licenses/python-build-standalone/{f}` |" for n, v, lic, f in PYTHON_COMPONENTS]
    return (
        "## Shared libraries vendored in the wheels\n\n"
        "Every `.dylib` in the app is listed here (the build fails when one is missing from this file). "
        "Versions were read from the binaries (version strings, `/opt/homebrew/Cellar/<name>/<version>` "
        "paths, library version numbers) or, where a binary carries none, taken from the Homebrew formula "
        "revision named above; \"x\" marks a patch level that is not recorded in the binary.\n\n"
        "### `cv2/.dylibs` (opencv-contrib-python 5.0.0.93, Homebrew ffmpeg 7.1.1_3 and its dependencies)\n\n"
        "| Project | Version | Licence | Kind | Files | Licence / notice text in the app |\n|---|---|---|---|---|---|\n"
        + "\n".join(rows)
        + "\n\nThe permissive licence texts in `licenses/dylibs/` were taken from the Homebrew bottles of "
        "these projects (`scripts/fetch_dylib_licenses.py`); the copyleft ones are in `licenses/`.\n\n"
        "### Other vendored libraries\n\n"
        "| Folder in `Contents/Resources/python/lib/python3.11/site-packages` | Project | Version | Licence | Files | Text |\n"
        "|---|---|---|---|---|---|\n"
        + "\n".join(orows)
        + "\n\n## Embedded Python\n\n"
        "`Contents/Resources/python` is the python-build-standalone release **20260325** "
        "(https://github.com/astral-sh/python-build-standalone) of **CPython 3.11.15**. These libraries "
        "are linked statically into it; the licence files of python-build-standalone are copied into "
        "`Contents/Resources/licenses/python-build-standalone/`. zlib, libedit (readline) and "
        "ncurses are linked from macOS and are not shipped.\n\n"
        "| Component | Version | Licence | Text in the app |\n|---|---|---|---|\n"
        + "\n".join(prows)
        + "\n\n"
    )


def render() -> str:
    rows = []
    for name, ver in packages():
        if name not in LICENSES:
            sys.exit(f"gen_third_party: no reviewed licence for {name}=={ver}: add it to LICENSES")
        rows.append(f"| {name} | {ver} | {LICENSES[name]} | https://pypi.org/project/{name}/{ver}/ |")
    mrows, cites = [], []
    for name, spec in MODEL_SPECS.items():
        task, credit, cite = MODEL_INFO[name]
        mb = f"{spec.size / 1e6:.1f} MB"
        shipped = "no" if name == "siglip2_vision_int8" else "yes"
        lic = spec.license
        if name in ONNX_COMMUNITY_BASE:
            lic = (f"{lic} (base model {ONNX_COMMUNITY_BASE[name]}; the onnx-community conversion declares "
                   "no licence of its own and is covered by the base model's)")
        mrows.append(f"| {name} | {task} | {lic} | {mb} | {shipped} | {credit} | {spec.url} |")
        if cite != "as above" and not cite.startswith("as above"):
            cites.append(f"* **{name}**: {cite}")
    return (
        HEADER
        + "## Python packages in the app\n\nExactly the packages `scripts/build_release.sh` installs "
        "(`uv export --frozen --no-dev`).\n\n| Package | Version | Licence | Source |\n|---|---|---|---|\n"
        + "\n".join(rows)
        + "\n\n"
        + dylib_sections()
        + "## Models in the app\n\nEvery file is pinned to a commit and verified against the "
        "sha256 in `src/shotsieve/models/registry.py` at build time. `nafnet_sidd` ships as a "
        "structurally repaired copy of the listed file (same weights; see "
        "`src/shotsieve/models/onnx_conv_fix.py`).\n\n"
        "| Model | Task | Licence | Size | In the app | Upstream | Download (pinned) |\n"
        "|---|---|---|---|---|---|---|\n"
        + "\n".join(mrows)
        + "\n\nThe SigLIP 2 tokenizer is the Gemma SentencePiece vocabulary, redistributed by Google "
        "inside the Apache-2.0 SigLIP 2 repository; the Gemma model weights (and the Gemma Terms of "
        "Use) are not part of ShotSieve.\n\n## Citations\n\n"
        + "\n".join(cites)
        + "\n\nThanks to every author above.\n"
    )


def main() -> int:
    text = render()
    if "--check" in sys.argv:
        return 0 if OUT.exists() and OUT.read_text() == text else 1
    OUT.write_text(text)
    print(f"wrote {OUT} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
