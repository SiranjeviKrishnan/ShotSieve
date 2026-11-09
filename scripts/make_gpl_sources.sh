#!/bin/bash
# Build the corresponding-source archive for the GPL / LGPL / MPL components inside the ShotSieve .dmg.
#
#   scripts/make_gpl_sources.sh        -> dist/ShotSieve-<version>-gpl-sources.tar.gz (+ .sha256)
#
# Downloads, from the official upstreams and verifies by sha256 (or by the commit id stored in a
# git archive), the exact sources of the binaries that the locked wheels bundle:
#   opencv-contrib-python 5.0.0.93  -> cv2/.dylibs   = Homebrew ffmpeg 7.1.1_3 and its dependencies
#                                     (FFmpeg 7.1.1 incl. libpostproc, x264 r3108 31e19f9,
#                                      x265 4.1+1-1d117be, librubberband 4.0.0, libvidstab 1.1.1,
#                                      LGPL / MPL libraries) built from Homebrew formula revision
#                                      a671b0c7ae23 ("ffmpeg: update 7.1.1_3 bottle.")
#   pillow-heif 1.8.1               -> pillow_heif/.dylibs = x265 4.3+1-e9b8812, libde265 1.1.3, libheif 1.23.6
# plus the build scripts (Homebrew formulae and patches, opencv-python tag 93, the pillow-heif sdist).
# The versions were read from the bundled binaries (strings / otool -L / wheel metadata); see the
# README written into the archive. Needs network access, curl, git (only `git get-tar-commit-id`, no
# repository is touched), shasum, tar.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
cd "$ROOT"
VERSION="$(sed -n 's/^version *= *"\(.*\)"/\1/p' pyproject.toml | head -1)"
[ -n "$VERSION" ] || { echo "no version in pyproject.toml" >&2; exit 2; }
DIST="$ROOT/dist"
CACHE="$DIST/.gpl-sources-cache"
NAME="ShotSieve-$VERSION-gpl-sources"
OUT="$DIST/$NAME.tar.gz"
mkdir -p "$DIST" "$CACHE"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/shotsieve-gpl.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
PKG="$WORK/$NAME"
mkdir -p "$PKG"

BREW_REV=a671b0c7ae230e99a07f541b471a974e3dbfd403   # homebrew-core: "ffmpeg: update 7.1.1_3 bottle."
BREW_RAW="https://raw.githubusercontent.com/Homebrew/homebrew-core/$BREW_REV"

sha_of() { shasum -a 256 "$1" | awk '{print $1}'; }

# fetch URL SHA256 DEST: download (cached), verify the checksum, copy into the package
fetch() {
  local url="$1" sha="$2" dest="$3" c
  c="$CACHE/$sha"
  if [ ! -f "$c" ] || [ "$(sha_of "$c")" != "$sha" ]; then
    echo "   get $url"
    curl -fL --retry 3 --retry-delay 3 -sS -o "$c.tmp" "$url"
    mv "$c.tmp" "$c"
  fi
  [ "$(sha_of "$c")" = "$sha" ] || { echo "CHECKSUM MISMATCH: $url" >&2; rm -f "$c"; exit 1; }
  mkdir -p "$PKG/$(dirname "$dest")"
  cp "$c" "$PKG/$dest"
}

# fetch_git URL COMMIT DEST: a `git archive` tarball; its embedded commit id must match
fetch_git() {
  local url="$1" commit="$2" dest="$3" c id
  c="$CACHE/git-$commit.tar.gz"
  if [ ! -f "$c" ]; then
    echo "   get $url"
    curl -fL --retry 3 --retry-delay 3 -sS -o "$c.tmp" "$url"
    mv "$c.tmp" "$c"
  fi
  id="$(gzip -dc "$c" | git get-tar-commit-id 2>/dev/null || true)"
  [ "$id" = "$commit" ] || { echo "COMMIT MISMATCH for $url: got '$id', want $commit" >&2; rm -f "$c"; exit 1; }
  mkdir -p "$PKG/$(dirname "$dest")"
  cp "$c" "$PKG/$dest"
}

# fetch_formula NAME PATH: Homebrew formula at the pinned revision
fetch_formula() {
  local name="$1" path="$2" c="$CACHE/formula-$BREW_REV-$1.rb"
  [ -f "$c" ] || curl -fL --retry 3 -sS -o "$c" "$BREW_RAW/$path"
  mkdir -p "$PKG/homebrew-formulae"
  cp "$c" "$PKG/homebrew-formulae/$name.rb"
}

# check that a sha256 given in this script is also the one stated in the pinned Homebrew formula
formula_says() { grep -q "$2" "$PKG/homebrew-formulae/$1.rb" || { echo "formula $1 does not state $2" >&2; exit 1; }; }

echo "== Homebrew formulae at $BREW_REV"
for f in "ffmpeg f/ffmpeg" "x264 x/x264" "x265 x/x265" "rubberband r/rubberband" "libvidstab lib/libvidstab" \
         "gnutls g/gnutls" "nettle n/nettle" "gmp g/gmp" "libunistring lib/libunistring" "libidn2 lib/libidn2" \
         "libssh lib/libssh" "libsoxr lib/libsoxr" "lame l/lame" "libbluray lib/libbluray" "aribb24 a/aribb24" \
         "libtasn1 lib/libtasn1" "srt s/srt" "zeromq z/zeromq" "glib g/glib" "gettext g/gettext" \
         "fribidi f/fribidi" "graphite2 g/graphite2"; do
  set -- $f; fetch_formula "$1" "Formula/$2.rb"
done

echo "== GPL: cv2 (Homebrew ffmpeg 7.1.1_3)"
S=ffmpeg
fetch https://ffmpeg.org/releases/ffmpeg-7.1.1.tar.xz 733984395e0dbbe5c046abda2dc49a5544e7e0e1e2366bba849222ae9e3a03b1 $S/ffmpeg-7.1.1.tar.xz
formula_says ffmpeg 733984395e0dbbe5c046abda2dc49a5544e7e0e1e2366bba849222ae9e3a03b1
fetch "https://github.com/FFmpeg/FFmpeg/commit/d1ed5c06e3edc5f2b5f3664c80121fa55b0baa95.patch?full_index=1" \
  0eb23ab90c0e5904590731dd3b81c86a4127785bc2b367267d77723990fb94a2 $S/patches/d1ed5c06-ffmpeg.patch
fetch "https://gitlab.archlinux.org/archlinux/packaging/packages/ffmpeg/-/raw/5670ccd86d3b816f49ebc18cab878125eca2f81f/add-av_stream_get_first_dts-for-chromium.patch" \
  57e26caced5a1382cb639235f9555fc50e45e7bf8333f7c9ae3d49b3241d3f77 $S/patches/add-av_stream_get_first_dts-for-chromium.patch
fetch_git https://code.videolan.org/videolan/x264/-/archive/31e19f92f00c7003fa115047ce50978bc98c3a0d/x264-31e19f92f00c7003fa115047ce50978bc98c3a0d.tar.gz \
  31e19f92f00c7003fa115047ce50978bc98c3a0d x264/x264-r3108-31e19f92.tar.gz
fetch https://bitbucket.org/multicoreware/x265_git/downloads/x265_4.1.tar.gz \
  a31699c6a89806b74b0151e5e6a7df65de4b49050482fe5ebf8a4379d7af8f29 x265/x265_4.1.tar.gz
formula_says x265 a31699c6a89806b74b0151e5e6a7df65de4b49050482fe5ebf8a4379d7af8f29
fetch "https://api.bitbucket.org/2.0/repositories/multicoreware/x265_git/diff/b354c009a60bcd6d7fc04014e200a1ee9c45c167" \
  f7d3ce261c4b0cd461b55ad00de38ffa6a7cc2fa13ae6f034b3e46d8bb3cb6a8 x265/patches/b354c009.diff
fetch "https://api.bitbucket.org/2.0/repositories/multicoreware/x265_git/diff/51ae8e922bcc4586ad4710812072289af91492a8" \
  56c78f60cbaac61a44cb6e9889ece3380f9b60d32a4b704e274d9a636a16379d x265/patches/51ae8e92.diff
fetch https://breakfastquay.com/files/releases/rubberband-4.0.0.tar.bz2 \
  af050313ee63bc18b35b2e064e5dce05b276aaf6d1aa2b8a82ced1fe2f8028e9 rubberband/rubberband-4.0.0.tar.bz2
formula_says rubberband af050313ee63bc18b35b2e064e5dce05b276aaf6d1aa2b8a82ced1fe2f8028e9
fetch https://github.com/georgmartius/vid.stab/archive/refs/tags/v1.1.1.tar.gz \
  9001b6df73933555e56deac19a0f225aae152abbc0e97dc70034814a1943f3d4 vidstab/vid.stab-1.1.1.tar.gz
formula_says libvidstab 9001b6df73933555e56deac19a0f225aae152abbc0e97dc70034814a1943f3d4
# the x265 4.1 tarball must carry the changeset that the binary reports (4.1+1-1d117be)
meta="$(tar xzf "$PKG/x265/x265_4.1.tar.gz" -O --include '*/x265Version.txt' 2>/dev/null || true)"
case "$meta" in *1d117be*) ;; *) echo "x265_4.1.tar.gz does not identify changeset 1d117be" >&2; exit 1;; esac

echo "== LGPL / MPL libraries delocated into cv2/.dylibs"
S=lgpl-mpl-libs
fetch https://www.gnupg.org/ftp/gcrypt/gnutls/v3.8/gnutls-3.8.9.tar.xz 69e113d802d1670c4d5ac1b99040b1f2d5c7c05daec5003813c049b5184820ed $S/gnutls-3.8.9.tar.xz
fetch https://ftp.gnu.org/gnu/nettle/nettle-3.10.1.tar.gz b0fcdd7fc0cdea6e80dcf1dd85ba794af0d5b4a57e26397eee3bc193272d9132 $S/nettle-3.10.1.tar.gz
fetch https://ftp.gnu.org/gnu/gmp/gmp-6.3.0.tar.xz a3c2b80201b89e68616f4ad30bc66aee4927c3ce50e33929ca819d5c43538898 $S/gmp-6.3.0.tar.xz
fetch https://ftp.gnu.org/gnu/libunistring/libunistring-1.3.tar.gz 8ea8ccf86c09dd801c8cac19878e804e54f707cf69884371130d20bde68386b7 $S/libunistring-1.3.tar.gz
fetch https://ftp.gnu.org/gnu/libidn/libidn2-2.3.8.tar.gz f557911bf6171621e1f72ff35f5b1825bb35b52ed45325dcdee931e5d3c0787a $S/libidn2-2.3.8.tar.gz
fetch https://www.libssh.org/files/0.11/libssh-0.11.1.tar.xz 14b7dcc72e91e08151c58b981a7b570ab2663f630e7d2837645d5a9c612c1b79 $S/libssh-0.11.1.tar.xz
fetch https://downloads.sourceforge.net/project/soxr/soxr-0.1.3-Source.tar.xz b111c15fdc8c029989330ff559184198c161100a59312f5dc19ddeb9b5a15889 $S/soxr-0.1.3-Source.tar.xz
fetch https://raw.githubusercontent.com/Homebrew/formula-patches/76868b36263be42440501d3692fd3a258f507d82/libsoxr/arm64_defines.patch 9df5737a21b9ce70cc136c302e195fad9f9f6c14418566ad021f14bb34bb022c $S/patches/soxr-arm64_defines.patch
fetch https://downloads.sourceforge.net/project/lame/lame/3.100/lame-3.100.tar.gz ddfe36cab873794038ae2c1210557ad34857a4b6bdc515785d1da9e175b1da1e $S/lame-3.100.tar.gz
fetch https://download.videolan.org/videolan/libbluray/1.3.4/libbluray-1.3.4.tar.bz2 478ffd68a0f5dde8ef6ca989b7f035b5a0a22c599142e5cd3ff7b03bbebe5f2b $S/libbluray-1.3.4.tar.bz2
fetch https://code.videolan.org/jeeb/aribb24/-/archive/v1.0.4/aribb24-v1.0.4.tar.bz2 88b58dd760609372701087e25557ada9f7c6d973306c017067c5dcaf9e2c9710 $S/aribb24-v1.0.4.tar.bz2
fetch https://ftp.gnu.org/gnu/libtasn1/libtasn1-4.20.0.tar.gz 92e0e3bd4c02d4aeee76036b2ddd83f0c732ba4cda5cb71d583272b23587a76c $S/libtasn1-4.20.0.tar.gz
fetch https://github.com/Haivision/srt/archive/refs/tags/v1.5.4.tar.gz d0a8b600fe1b4eaaf6277530e3cfc8f15b8ce4035f16af4a5eb5d4b123640cdd $S/srt-1.5.4.tar.gz
fetch "https://github.com/Haivision/srt/commit/7962936829e016295e5c570539eb2520b326da4c.patch?full_index=1" e4489630886bf8b26f63a23c8b1aec549f7280f07713ded07fce281e542725f7 $S/patches/srt-7962936.patch
fetch https://github.com/zeromq/libzmq/releases/download/v4.3.5/zeromq-4.3.5.tar.gz 6653ef5910f17954861fe72332e68b03ca6e4d9c7160eb3a8de5a5a913bfab43 $S/zeromq-4.3.5.tar.gz
fetch https://download.gnome.org/sources/glib/2.84/glib-2.84.3.tar.xz aa4f87c3225bf57ca85f320888f7484901a17934ca37023c3bd8435a72db863e $S/glib-2.84.3.tar.xz
fetch https://ftp.gnu.org/gnu/gettext/gettext-0.25.tar.gz aee02dab79d9138fdcc7226b67ec985121bce6007edebe30d0e39d42f69a340e $S/gettext-0.25.tar.gz
fetch https://github.com/fribidi/fribidi/releases/download/v1.0.16/fribidi-1.0.16.tar.xz 1b1cde5b235d40479e91be2f0e88a309e3214c8ab470ec8a2744d82a5a9ea05c $S/fribidi-1.0.16.tar.xz
fetch https://github.com/silnrsi/graphite/releases/download/1.3.14/graphite2-1.3.14.tgz f99d1c13aa5fa296898a181dff9b82fb25f6cc0933dbaa7a475d8109bd54209d $S/graphite2-1.3.14.tgz
formula_says gnutls 69e113d802d1670c4d5ac1b99040b1f2d5c7c05daec5003813c049b5184820ed
formula_says zeromq 6653ef5910f17954861fe72332e68b03ca6e4d9c7160eb3a8de5a5a913bfab43
formula_says srt d0a8b600fe1b4eaaf6277530e3cfc8f15b8ce4035f16af4a5eb5d4b123640cdd

echo "== opencv-contrib-python build (tag 93)"
# tag 93 is an annotated tag (27b15e5) of commit b83046cd; the archive does not contain the OpenCV
# submodules (Apache-2.0, fetched by the build from github.com/opencv/opencv and opencv_contrib, branch 5.x)
fetch_git https://github.com/opencv/opencv-python/archive/refs/tags/93.tar.gz \
  b83046cda41133f1bf2e73e99dba16a1248f103a opencv-python/opencv-python-93.tar.gz

echo "== pillow-heif 1.8.1 (x265 4.3, libde265 1.1.3, libheif 1.23.6)"
fetch https://files.pythonhosted.org/packages/44/c1/82145984920ca055675af2c2795bd30da6f7461215c41f3c1eacb3d66353/pillow_heif-1.8.1.tar.gz \
  521ebffb8a181d56c3904e5a61f20903edee0d9d3275967b8fb345f866215c06 pillow-heif/pillow_heif-1.8.1.tar.gz
fetch https://github.com/Multicorewareinc/x265/releases/download/4.3/x265_4.3.tar.gz \
  83c53e4c8bbb8f1e33ed59e10a7d621d1d7801ca853910c3eb41f038b8ffb121 pillow-heif/x265_4.3.tar.gz
fetch https://github.com/strukturag/libde265/releases/download/v1.1.3/libde265-1.1.3.tar.gz \
  554228bd17788c99a7e63b37ab5634722190e6e2bf60c1dcb01cef328e133905 pillow-heif/libde265-1.1.3.tar.gz
fetch https://github.com/strukturag/libheif/releases/download/v1.23.6/libheif-1.23.6.tar.gz \
  4484346dc5995319dbc11e3a1c35d0a2ec46511ce370900869337fd2c7033125 pillow-heif/libheif-1.23.6.tar.gz
# the binary reports "4.3+1-e9b8812": the 4.3 release tarball is release tag 4.3, distance 1, changeset e9b8812
meta="$(tar xzf "$PKG/pillow-heif/x265_4.3.tar.gz" -O --include '*/x265Version.txt' 2>/dev/null || true)"
case "$meta" in *e9b8812*) ;; *) echo "x265_4.3.tar.gz does not identify changeset e9b8812" >&2; exit 1;; esac

echo "== rawpy 0.28.0 (LibRaw 0.22.2, LGPL-2.1 or CDDL-1.0)"
fetch https://files.pythonhosted.org/packages/84/17/d9b34cdf64b1377e4f9d070375674276a5f68954741849592eda81c67557/rawpy-0.28.0.tar.gz \
  3176c865b786dfec4e54251759f98e5a00733035f1f256e8c09909c59b5353d2 rawpy/rawpy-0.28.0.tar.gz
fetch_git https://github.com/LibRaw/LibRaw/archive/refs/tags/0.22.2.tar.gz \
  b93f6e45c194f5df9b02a43b1af9a54b4f41f33f rawpy/LibRaw-0.22.2.tar.gz

echo "== licence texts"
mkdir -p "$PKG/licenses"
cp licenses/*.txt "$PKG/licenses/"

cat > "$PKG/README.md" <<EOF
# ShotSieve $VERSION: corresponding source for the GPL / LGPL / MPL components

This archive is the *complete corresponding source offer* for the copyleft libraries that are
bundled, as binaries, inside \`ShotSieve.app\` (the \`.dmg\`): the build scripts, patches and the
exact upstream source releases. It is published on the same GitHub Release as the \`.dmg\` (GPLv2 s.3,
GPLv3 s.6(d)). ShotSieve's own source code (Apache-2.0) is in the repository at the release tag.
Every file's sha256 is in \`SHA256SUMS\`. The sources were verified against the checksums of the
upstream release or the Homebrew formula (or, for git archives, the commit id stored in the archive).

Where the binaries are: \`ShotSieve.app/Contents/Resources/python/lib/python3.11/site-packages/\`
\`cv2/.dylibs/\` (from the \`opencv-contrib-python\` wheel) and \`pillow_heif/.dylibs/\` (from the
\`pillow-heif\` wheel). How the versions below were identified is noted for each part.

## 1. \`cv2/.dylibs\`: opencv-contrib-python 5.0.0.93 = Homebrew ffmpeg 7.1.1_3

The wheel \`opencv_contrib_python-5.0.0.93-cp37-abi3-macosx_13_0_arm64\` (built by the opencv-python
project, \`delocate\`d) contains the shared libraries of the Homebrew bottle \`ffmpeg 7.1.1_3\`
and of its dependencies. The configure line stored in \`libavcodec.61.19.101.dylib\` is
\`--prefix=/opt/homebrew/Cellar/ffmpeg/7.1.1_3 --enable-shared --enable-version3 --enable-gpl ...\`
(\`strings libavcodec*.dylib | grep prefix\`). The GPL-licensed ones:

| Binary in the bundle | Version | How identified | Source in this archive |
|---|---|---|---|
| libavcodec.61, libavformat.61, libavutil.59, libavfilter.10, libavdevice.61, libswscale.8, libswresample.5, libpostproc.58 | FFmpeg 7.1.1 (built \`--enable-gpl --enable-version3\`: GPL-3.0-or-later as a whole; libpostproc: "GPL version 3 or later") | "FFmpeg version 7.1.1" and the configure line inside the dylibs | \`ffmpeg/ffmpeg-7.1.1.tar.xz\` + 2 Homebrew patches \`ffmpeg/patches/\` |
| libx264.164 | x264 core 164 **r3108 31e19f9** (GPL-2.0-or-later) | string "r3108 31e19f9" in the dylib | \`x264/x264-r3108-31e19f92.tar.gz\` (git archive of commit 31e19f92f00c7003fa115047ce50978bc98c3a0d) |
| libx265.215 | x265 **4.1+1-1d117be** (GPL-2.0) | version string in the dylib | \`x265/x265_4.1.tar.gz\` (changeset 1d117be) + 2 Homebrew patches |
| librubberband.3 | Rubber Band 4.0.0 (GPL-2.0-or-later) | soname 3, formula at the Homebrew revision | \`rubberband/rubberband-4.0.0.tar.bz2\` |
| libvidstab.1.2 | vid.stab 1.1.1 (GPL-2.0-or-later) | soname 1.2, formula | \`vidstab/vid.stab-1.1.1.tar.gz\` |

The exact build recipe of each (configure flags, patches) is the Homebrew formula in
\`homebrew-formulae/\`, taken from homebrew-core commit \`$BREW_REV\`
("ffmpeg: update 7.1.1_3 bottle."), the revision that published the ffmpeg 7.1.1_3 bottle that
the wheel contains. Not every dependency in the wheel is at that revision's version (the wheel
builder's runner had other bottles installed: e.g. OpenSSL 3.6.0, libpng 1.6.49, libvpx 1.15.2,
libarchive 3.8.1, glib 2.84.3 are newer or different), so for the copyleft libraries the version was
read from the binary itself (version string, the \`/opt/homebrew/Cellar/<name>/<version>\` path
stored in the dylib, or the library's current-version number) and that exact release is included.
None of the libraries whose version differs from the formula revision is copyleft except glib (the
formula at that revision is 2.84.2; the binary is 2.84.3, whose tarball is included).

LGPL / MPL / copyleft-by-choice libraries in the same folder (\`lgpl-mpl-libs/\`; formulae in
\`homebrew-formulae/\`): GnuTLS 3.8.9 (libgnutls.30), Nettle 3.10.1 (libnettle.8, libhogweed.6),
GMP 6.3.0 (libgmp.10), libunistring 1.3 (libunistring.5), libidn2 2.3.8 (libidn2.0), libssh 0.11.1
(libssh.4), libsoxr 0.1.3 (libsoxr.0.1.2), LAME 3.100 (libmp3lame.0), libbluray 1.3.4
(libbluray.2), aribb24 1.0.4 (libaribb24.0), libtasn1 4.20.0 (libtasn1.6), GLib 2.84.3
(libglib-2.0.0), gettext 0.25 (libintl.8), FriBidi 1.0.16 (libfribidi.0), Graphite2 1.3.14
(libgraphite2.3, multi-licensed MPL / LGPL / GPL), SRT 1.5.4 (libsrt.1.5.4, MPL-2.0), ZeroMQ 4.3.5
(libzmq.5, MPL-2.0). Versions read directly from the binaries: GnuTLS, GMP, libidn2, libssh, LAME,
libtasn1, SRT, GLib, gettext, FriBidi (strings or Cellar path); libbluray, Nettle, ZeroMQ, libsoxr,
libunistring, aribb24, Graphite2 were matched by their library version number (e.g. libzmq current
version 8.5.0 = ZeroMQ 4.3.5, libbluray 7.3.0 = 1.3.4, libgraphite2 3.2.1 = 1.3.14) to the single
upstream release that produces it.
The dylibs of the other projects (BSD/MIT/ISC/Apache etc.) are not copyleft; their notices are in
\`licenses/\` in the app and in the repository.

The opencv-python build scripts (how the wheel was assembled, the Homebrew FFmpeg step, delocate):
\`opencv-python/opencv-python-93.tar.gz\` = tag \`93\` of https://github.com/opencv/opencv-python
(commit b83046cda41133f1bf2e73e99dba16a1248f103a; wheel version 5.0.0.93). OpenCV itself (Apache-2.0,
\`cv2.abi3.so\`) comes from the submodules \`opencv\` / \`opencv_contrib\` (branch 5.x) of that tag
and is not copyleft.

## 2. \`pillow_heif/.dylibs\`: pillow-heif 1.8.1

The wheel \`pillow_heif-1.8.1-cp311-cp311-macosx_11_0_arm64\` ships \`libheif.1.23.6.dylib\`
(LGPL-3.0), \`libde265.0.2.3.dylib\` (libde265 1.1.3, LGPL-3.0) and \`libx265.217.dylib\`
(**x265 4.3+1-e9b8812, GPL-2.0**; the wheel's own \`LICENSES_bundled.txt\` says GPLv2).
They are built by \`libheif/build_libs.py\` of the pillow-heif sdist from the three release tarballs
in \`pillow-heif/\`: \`x265_4.3.tar.gz\` (release tag 4.3 plus one changeset, e9b8812, as the binary
reports), \`libde265-1.1.3.tar.gz\`, \`libheif-1.23.6.tar.gz\`, and the sdist
\`pillow_heif-1.8.1.tar.gz\` (build scripts, PyPI).

## 3. \`rawpy/\`: rawpy 0.28.0 (LibRaw 0.22.2)

\`rawpy/libraw_r.25.0.0.dylib\` reports "0.22.2-Release" (\`rawpy.libraw_version\` = (0, 22, 2); the
\`.25\` is the library's soname, not its version). LibRaw is LGPL-2.1 or CDDL-1.0 and loaded
dynamically. Source: \`rawpy/LibRaw-0.22.2.tar.gz\` (git archive of tag 0.22.2, commit
b93f6e45c194f5df9b02a43b1af9a54b4f41f33f) and the rawpy sdist \`rawpy/rawpy-0.28.0.tar.gz\` (the
wheel's build scripts). rawpy's \`rawpy/.dylibs/\` also carry JasPer, libjpeg-turbo and Little CMS
(permissive, notices in \`licenses/\`).

## 4. Not included (and why)

* \`scipy/.dylibs/libgcc_s.1.1.dylib\`, \`libgfortran.5.dylib\` (GPL-3.0-or-later with the GCC Runtime
  Library Exception, which permits redistribution with any program) and \`libquadmath.0.dylib\`
  (LGPL-2.1-or-later): GCC runtime libraries from the scipy wheel build. The binaries are stripped and
  carry no GCC version, so the exact GCC release cannot be established from them; the sources are
  the libgcc / libgfortran / libquadmath directories of https://gcc.gnu.org/git/gcc.git, and scipy's
  wheel build is documented at https://github.com/scipy/scipy. They are covered by the written
  offer in \`THIRD_PARTY_NOTICES.md\`.
* Everything under the embedded CPython (python-build-standalone 20260325): permissive licences,
  texts in \`licenses/python-build-standalone/\`.

## How to rebuild or replace a library

The LGPL libraries are shared libraries loaded dynamically: build a modified version from the source
here, replace the \`.dylib\` of the same name inside the app (same folder, same file name), and
re-sign with \`codesign --force -s - ShotSieve.app\` (the app is ad-hoc signed). The GPL
libraries must be redistributed under the GPL together with their source (this archive).

Licence texts: \`licenses/\` (GPL-2.0, GPL-3.0, LGPL-2.1, LGPL-3.0, MPL-2.0).
EOF

echo "== packing"
( cd "$PKG" && find . -type f ! -name SHA256SUMS | sort | sed 's|^\./||' | while IFS= read -r f; do shasum -a 256 "$f"; done > SHA256SUMS )
rm -f "$OUT" "$OUT.sha256"
# reproducible-ish: sorted names, no extended attributes
( cd "$WORK" && COPYFILE_DISABLE=1 tar --no-xattrs -czf "$OUT" "$NAME" 2>/dev/null || COPYFILE_DISABLE=1 tar -czf "$OUT" "$NAME" )
( cd "$DIST" && shasum -a 256 "$(basename "$OUT")" > "$(basename "$OUT").sha256" )
echo "   $OUT: $(( $(stat -f %z "$OUT") / 1048576 )) MB ($(stat -f %z "$OUT") bytes)"
cat "$OUT.sha256"
echo "== done =="
