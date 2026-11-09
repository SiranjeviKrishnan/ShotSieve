# ruff: noqa: E501
"""Collect the licence / notice texts of the permissive libraries delocated into cv2/.dylibs.

    python3 scripts/fetch_dylib_licenses.py          # rewrites licenses/dylibs/*.txt

The libraries in `opencv-contrib-python`'s cv2/.dylibs are Homebrew bottles. For each project this
downloads the Homebrew bottle (arm64, pinned by the sha256 in the formula of homebrew-core commit
a671b0c7ae23, the revision of the ffmpeg 7.1.1_3 bottle) and keeps the licence files that the keg
contains (LICENSE*, COPYING*, COPYRIGHT*, NOTICE*, AUTHORS-style legal files) in one text file per
project, with a header naming the project, the bottle version and the file. Run only by a
maintainer when the library set changes; the output is committed (licenses/dylibs/). Network needed.
Standard library only.
"""

from __future__ import annotations

import io
import re
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "licenses" / "dylibs"
REV = "a671b0c7ae230e99a07f541b471a974e3dbfd403"
RAW = f"https://raw.githubusercontent.com/Homebrew/homebrew-core/{REV}/Formula"
# Homebrew formula name -> formula path (Formula/<path>.rb)
FORMULAE = {
    "aom": "a/aom", "libarchive": "lib/libarchive", "libass": "lib/libass", "libavif": "lib/libavif",
    "brotli": "b/brotli", "cjson": "c/cjson", "openssl@3": "o/openssl@3", "dav1d": "d/dav1d",
    "libdeflate": "lib/libdeflate", "fontconfig": "f/fontconfig", "freetype": "f/freetype",
    "giflib": "g/giflib", "harfbuzz": "h/harfbuzz", "highway": "h/highway", "openexr": "o/openexr",
    "imath": "i/imath", "jpeg-turbo": "j/jpeg-turbo", "jpeg-xl": "j/jpeg-xl",
    "little-cms2": "l/little-cms2", "lz4": "l/lz4", "xz": "x/xz",
    "mbedtls": "m/mbedtls", "libogg": "lib/libogg", "opencore-amr": "o/opencore-amr",
    "openjpeg": "o/openjpeg", "opus": "o/opus", "p11-kit": "p/p11-kit", "pcre2": "p/pcre2",
    "libpng": "lib/libpng", "rav1e": "r/rav1e", "librist": "lib/librist",
    "libsamplerate": "lib/libsamplerate", "sdl2": "s/sdl2", "webp": "w/webp", "snappy": "s/snappy",
    "libsodium": "lib/libsodium", "speex": "s/speex", "svt-av1": "s/svt-av1",
    "tesseract": "t/tesseract", "theora": "t/theora", "libtiff": "lib/libtiff",
    "libunibreak": "lib/libunibreak", "libvmaf": "lib/libvmaf", "libvorbis": "lib/libvorbis",
    "libvpx": "lib/libvpx", "libx11": "lib/libx11", "libxau": "lib/libxau", "libxcb": "lib/libxcb",
    "libxdmcp": "lib/libxdmcp", "zimg": "z/zimg", "zstd": "z/zstd", "libb2": "lib/libb2",
}
# projects whose Homebrew bottle has no licence file, or that are not Homebrew libraries:
# name -> (version, [(label, url)]) written to licenses/dylibs/<name>.txt
EXTRA = {
    "leptonica": ("1.85.0", [("leptonica-license.txt", "https://raw.githubusercontent.com/DanBloomberg/leptonica/1.85.0/leptonica-license.txt")]),
    "jasper": ("4.x (rawpy/.dylibs/libjasper.7)", [("LICENSE.txt", "https://raw.githubusercontent.com/jasper-software/jasper/version-4.2.4/LICENSE.txt")]),
    "portaudio": ("19.7.0-devel (_sounddevice_data/portaudio-binaries)", [("LICENSE.txt", "https://raw.githubusercontent.com/PortAudio/portaudio/v19.7.0/LICENSE.txt")]),
    "sqlite-vec": ("0.1.9 (sqlite_vec/vec0.dylib)", [("LICENSE-MIT", "https://raw.githubusercontent.com/asg017/sqlite-vec/v0.1.9/LICENSE-MIT"),
                                                        ("LICENSE-APACHE", "https://raw.githubusercontent.com/asg017/sqlite-vec/v0.1.9/LICENSE-APACHE")]),
}
LEGAL = re.compile(r"(^|/)(LICEN[CS]E|COPYING|COPYRIGHT|NOTICE|PATENTS|LICENSE-[A-Za-z0-9.-]+|COPYING\.[A-Za-z0-9.-]+)(\.(txt|md|rst))?$", re.I)


def get(url: str, headers: dict | None = None) -> bytes:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310 (fixed https hosts)
        return r.read()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    failed = []
    for name, path in sorted(FORMULAE.items()):
        try:
            rb = get(f"{RAW}/{path}.rb").decode()
            m = (re.search(r'arm64_sequoia:\s*"([0-9a-f]{64})"', rb) or re.search(r'arm64_sonoma:\s*"([0-9a-f]{64})"', rb)
                 or re.search(r'arm64_ventura:\s*"([0-9a-f]{64})"', rb))
            if not m:
                raise RuntimeError("no arm64 bottle in formula")
            blob = get(f"https://ghcr.io/v2/homebrew/core/{name.replace('@', '/')}/blobs/sha256:{m.group(1)}",
                       {"Authorization": "Bearer QQ=="})
            parts, version = [], "?"
            with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
                for ti in tf:
                    segs = ti.name.split("/")  # <name>/<version>/<file...>
                    if len(segs) >= 3 and version == "?":
                        version = segs[1]
                    if not ti.isfile() or len(segs) > 5 or "/test" in ti.name or "/.brew/" in ti.name:
                        continue
                    if LEGAL.search(ti.name) and ti.size < 200_000:
                        text = tf.extractfile(ti).read().decode("utf-8", "replace")  # type: ignore[union-attr]
                        parts.append((("/".join(segs[2:])), text))
            if not parts:
                raise RuntimeError("bottle contains no licence file")
            body = "".join(
                f"===== {name} {version}: {fn} =====\n{txt.rstrip()}\n\n" for fn, txt in sorted(set(parts)))
            header = (f"Licence / notice files of {name} {version}, taken from the Homebrew bottle (arm64) that "
                      f"the cv2/.dylibs of opencv-contrib-python were built from.\n\n")
            (OUT / f"{name.replace('@', '-')}.txt").write_text(header + body)
            print(f"ok   {name} {version} ({len(parts)} file(s))")
        except Exception as e:  # noqa: BLE001
            failed.append(name)
            print(f"FAIL {name}: {e}", file=sys.stderr)
    for name, (version, files) in sorted(EXTRA.items()):
        try:
            body = "".join(f"===== {name} {version}: {label} =====\n{get(url).decode('utf-8', 'replace').rstrip()}\n\n"
                           for label, url in files)
            (OUT / f"{name}.txt").write_text(f"Licence texts of {name} {version}, from the upstream repository.\n\n" + body)
            print(f"ok   {name} {version}")
        except Exception as e:  # noqa: BLE001
            failed.append(name)
            print(f"FAIL {name}: {e}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
