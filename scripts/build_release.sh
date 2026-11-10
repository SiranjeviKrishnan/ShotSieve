#!/bin/bash
# Build the self-contained, relocatable ShotSieve.app and a .dmg (Apple Silicon).
#
#   scripts/build_release.sh            -> dist/ShotSieve.app
#                                          dist/ShotSieve-<version>-arm64.dmg (+ .sha256)
#
# What goes into the bundle (nothing refers to this machine):
#   Contents/MacOS/ShotSieve          small native launcher (scripts/launcher.c)
#   Contents/Resources/python/        python-build-standalone 3.11 (relocatable) + the locked,
#                                     non-dev dependencies (uv export --frozen) + shotsieve itself
#   Contents/Resources/models/        every model, sha256-verified, + bundle.json
#   Contents/Resources/{LICENSE,NOTICE,THIRD_PARTY_NOTICES.md,licenses/}
# The app is ad-hoc code-signed (`codesign -s -`, required on arm64); there is no Developer ID, so
# macOS shows the "unidentified developer" prompt on first open (see README).
#
# Environment: SHOTSIEVE_MODELS_SRC (default ./models; fetched with scripts/fetch_models.py when
# missing), SKIP_DMG=1, SHOTSIEVE_BUILD_UV (uv binary), DMG_FORMAT (default UDZO).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
cd "$ROOT"
DIST="$ROOT/dist"
[ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ] || { echo "build_release.sh needs an Apple Silicon Mac." >&2; exit 2; }

# ---- tools ------------------------------------------------------------------------------------
UV="${SHOTSIEVE_BUILD_UV:-}"
[ -n "$UV" ] || { [ -x "$ROOT/.tools/uv/uv" ] && UV="$ROOT/.tools/uv/uv"; } || true
[ -n "$UV" ] || UV="$(command -v uv || true)"
[ -x "$UV" ] || { echo "uv not found (install uv, or run ./install.sh --dev)." >&2; exit 2; }
export UV_PYTHON_PREFERENCE=only-managed
[ -d "$ROOT/.uv-python" ] && export UV_PYTHON_INSTALL_DIR="$ROOT/.uv-python"
[ -d "$ROOT/.uv-cache" ] && export UV_CACHE_DIR="$ROOT/.uv-cache"
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV

VERSION="$(sed -n 's/^version *= *"\(.*\)"/\1/p' pyproject.toml | head -1)"
[ -n "$VERSION" ] || { echo "no version in pyproject.toml" >&2; exit 2; }
PKG_VERSION="$(sed -n 's/^__version__ *= *"\(.*\)"/\1/p' src/shotsieve/__init__.py)"
[ "$VERSION" = "$PKG_VERSION" ] || { echo "version mismatch: pyproject $VERSION vs __init__ $PKG_VERSION" >&2; exit 2; }
# Bundle id: io.github.SiranjeviKrishnan.shotsieve once scripts/prepare_repo.sh replaced the placeholder
# below with the GitHub username; without it (a plain checkout) a neutral id.
GITHUB_USER="SiranjeviKrishnan"
case "$GITHUB_USER" in
  "<"*) BUNDLE_ID="app.shotsieve.desktop" ;;
  *)    BUNDLE_ID="io.github.$(printf '%s' "$GITHUB_USER" | tr '[:upper:]' '[:lower:]').shotsieve" ;;
esac
echo "== ShotSieve $VERSION release build (bundle id $BUNDLE_ID) =="

WORK="$(mktemp -d "${TMPDIR:-/tmp}/shotsieve-release.XXXXXX")"
WORK="$(cd "$WORK" && pwd -P)"   # real path (no //, no /var -> /private/var alias)
trap 'rm -rf "$WORK"' EXIT
export PYTHONDONTWRITEBYTECODE=1  # the embedded Python must not leave .pyc files naming $WORK
APP="$WORK/ShotSieve.app"
C="$APP/Contents"
R="$C/Resources"
mkdir -p "$C/MacOS" "$R"

# ---- 1. embedded Python -----------------------------------------------------------------------
echo "-- embedded Python 3.11"
"$UV" python install 3.11 --no-bin >/dev/null 2>&1 || true
BASE_PY="$("$UV" python find --managed-python 3.11 2>/dev/null || "$UV" python find 3.11)"
PREFIX="$("$BASE_PY" -c 'import sys; print(sys.base_prefix)')"
[ -f "$PREFIX/lib/libpython3.11.dylib" ] || { echo "not a python-build-standalone prefix: $PREFIX" >&2; exit 2; }
mkdir -p "$R/python"
/usr/bin/rsync -a --exclude '/include' --exclude '/share' --exclude '/lib/pkgconfig' \
  --exclude '/lib/tcl*' --exclude '/lib/tk*' --exclude '/lib/itcl*' --exclude '/lib/thread*' \
  --exclude '/lib/libtcl*' --exclude '/lib/libtk*' --exclude '/build' \
  --exclude 'lib/python3.11/test' --exclude 'lib/python3.11/idlelib' \
  --exclude 'lib/python3.11/tkinter' --exclude 'lib/python3.11/turtledemo' \
  --exclude 'lib/python3.11/lib2to3' --exclude 'lib/python3.11/ensurepip' \
  --exclude 'lib/python3.11/config-3.11-darwin' --exclude 'lib/python3.11/site-packages/*' \
  --exclude 'lib-dynload/_tkinter*' --exclude 'lib-dynload/_test*' --exclude 'lib-dynload/_xx*' \
  "$PREFIX/" "$R/python/"
# uv patches the managed Python it installs: libpython's install name and the sysconfig data
# carry the absolute prefix. Restore the relocatable form (python-build-standalone uses /install).
install_name_tool -id @rpath/libpython3.11.dylib "$R/python/lib/libpython3.11.dylib"
grep -rIlF "$PREFIX" "$R/python" 2>/dev/null | while IFS= read -r f; do
  sed -i '' "s|$PREFIX|/install|g" "$f"
done
# console scripts carry the build machine's path in their shebang: keep only the interpreter
find "$R/python/bin" -mindepth 1 ! -name 'python3.11' -delete
ln -s python3.11 "$R/python/bin/python3"
PY="$R/python/bin/python3.11"
SITE="$R/python/lib/python3.11/site-packages"
mkdir -p "$SITE"

# ---- 2. dependencies (locked) + shotsieve ----------------------------------------------------
echo "-- installing locked dependencies"
"$UV" export --frozen --no-dev --no-emit-project -q -o "$WORK/requirements.txt"
"$UV" pip install --python "$PY" --break-system-packages --no-deps --require-hashes \
  --link-mode copy -q -r "$WORK/requirements.txt"
"$UV" build --wheel --out-dir "$WORK/wheel" -q
"$UV" pip install --python "$PY" --break-system-packages --no-deps --link-mode copy -q \
  "$WORK"/wheel/shotsieve-*.whl
# direct_url.json records where the shotsieve wheel was built (a temp path on this machine)
find "$SITE" -name direct_url.json -delete
# uv writes console-script wrappers with an absolute shebang into bin/: not needed
find "$R/python/bin" -mindepth 1 ! -name 'python3*' -delete

# ---- 3. strip ---------------------------------------------------------------------------------
echo "-- stripping tests, caches and unused data"
find "$R/python" -type d \( -name tests -o -name test -o -name PyObjCTest -o -name sample_data \) \
  -prune -exec rm -rf {} + 2>/dev/null || true
find "$R/python" -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
rm -rf "$SITE"/{pip,setuptools,pkg_resources,_distutils_hack} "$SITE"/distutils-precedence.pth \
  "$SITE"/pip-* "$SITE"/setuptools-* 2>/dev/null || true
find "$SITE" -name '*.dSYM' -prune -exec rm -rf {} + 2>/dev/null || true
find "$SITE" -name '*.pyi' -delete
find "$SITE" -name '*.pxd' -delete
# rawpy ships the same libraw three times (hard copies); keep the real file, link the aliases
if [ -f "$SITE/rawpy/libraw_r.25.0.0.dylib" ]; then
  for alias in libraw_r.25.dylib libraw_r.dylib; do
    if [ -f "$SITE/rawpy/$alias" ] && cmp -s "$SITE/rawpy/$alias" "$SITE/rawpy/libraw_r.25.0.0.dylib"; then
      rm "$SITE/rawpy/$alias"; ln -s libraw_r.25.0.0.dylib "$SITE/rawpy/$alias"
    fi
  done
fi

# thin the few universal2 extension modules (pyobjc) to arm64
while IFS= read -r f; do
  if /usr/bin/lipo -info "$f" 2>/dev/null | grep -q 'Architectures in the fat file'; then
    /usr/bin/lipo -thin arm64 "$f" -output "$f.thin" && mv "$f.thin" "$f"
  fi
done < <(find "$R/python" -type f \( -name '*.so' -o -name '*.dylib' \) -print)

# wheels built on CI runners keep stale absolute LC_RPATHs (/Users/runner/..., /opt/homebrew/...):
# harmless but they could pick up a user's Homebrew libraries, so drop every non-system one
while IFS= read -r f; do
  /usr/bin/file -b "$f" | grep -q 'Mach-O' || continue
  otool -l "$f" 2>/dev/null | awk '/cmd LC_RPATH/{r=1} r&&/^ +path /{print $2; r=0}' \
    | grep -E '^/' | grep -vE '^(/usr/lib|/System/Library)' | while IFS= read -r rp; do
        install_name_tool -delete_rpath "$rp" "$f" 2>/dev/null || true
      done || true
done < <(find "$R/python" -type f \( -name '*.so' -o -name '*.dylib' \))

# editing load commands invalidated those signatures (arm64 kills such a binary): re-sign now so
# the embedded Python can still run during the rest of the build
sign_binaries() {
  find "$R/python" -type f \( -name '*.so' -o -name '*.dylib' -o -path '*/python/bin/python3.11' \) -print0 \
    | xargs -0 -n 40 codesign --force -s - >/dev/null 2>&1
}
sign_binaries

# ---- 4. models --------------------------------------------------------------------------------
echo "-- models"
MODELS_SRC="${SHOTSIEVE_MODELS_SRC:-$ROOT/models}"
if [ ! -f "$MODELS_SRC/yolox_s_2022nov.onnx" ] || [ ! -f "$MODELS_SRC/siglip2_vision_fp32.onnx" ] \
   || [ ! -f "$MODELS_SRC/nafnet_sidd_width32_fp16.onnx" ] \
   || [ ! -f "$MODELS_SRC/rawnind_utnet2_bayer_512.onnx" ]; then
  echo "   some models are missing: fetching (sha256-verified) into $MODELS_SRC"
  SHOTSIEVE_MODELS_DIR="$MODELS_SRC" PYTHONPATH="$SITE" "$PY" "$ROOT/scripts/fetch_models.py"
fi
"$PY" "$ROOT/scripts/bundle_models.py" "$MODELS_SRC" "$R/models"

# ---- 5. launcher, icon, Info.plist, legal files -------------------------------------------------
echo "-- launcher, icon, Info.plist"
# LSMinimumSystemVersion = the highest minos of any shipped Mach-O (numpy, onnxruntime, mediapipe
# are built for 14.0), so macOS never lets the app start where a library cannot load.
MINOS="$(find "$R/python" -type f \( -name '*.so' -o -name '*.dylib' -o -name 'python3.11' \) -print0 \
  | xargs -0 -n 50 sh -c 'for f; do otool -l "$f" 2>/dev/null | awk "/LC_BUILD_VERSION/{b=1} b&&/minos/{print \$2; b=0}"; done' sh \
  | sort -t. -k1,1n -k2,2n | tail -1)"
MINOS="${MINOS:-14.0}"
echo "   highest minos of shipped binaries: $MINOS"
cc -arch arm64 -O2 -Wall -mmacosx-version-min="$MINOS" -o "$C/MacOS/ShotSieve" scripts/launcher.c

if "$PY" scripts/make_icon.py "$WORK/icon.png" 2>/dev/null; then
  SET="$WORK/ShotSieve.iconset"; mkdir -p "$SET"
  for sz in 16 32 128 256 512; do
    /usr/bin/sips -z $sz $sz "$WORK/icon.png" --out "$SET/icon_${sz}x${sz}.png" >/dev/null
    /usr/bin/sips -z $((sz * 2)) $((sz * 2)) "$WORK/icon.png" --out "$SET/icon_${sz}x${sz}@2x.png" >/dev/null
  done
  /usr/bin/iconutil -c icns "$SET" -o "$R/ShotSieve.icns" || echo "Note: icon not created (cosmetic)." >&2
fi

cat > "$C/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>ShotSieve</string>
  <key>CFBundleDisplayName</key><string>ShotSieve</string>
  <key>CFBundleExecutable</key><string>ShotSieve</string>
  <key>CFBundleIdentifier</key><string>$BUNDLE_ID</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleIconFile</key><string>ShotSieve</string>
  <key>CFBundleShortVersionString</key><string>$VERSION</string>
  <key>CFBundleVersion</key><string>$VERSION</string>
  <key>LSMinimumSystemVersion</key><string>$MINOS</string>
  <key>LSArchitecturePriority</key><array><string>arm64</string></array>
  <key>LSApplicationCategoryType</key><string>public.app-category.photography</string>
  <key>NSHumanReadableCopyright</key><string>Copyright 2026 The ShotSieve Authors. Licensed under the Apache License, Version 2.0.</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSSupportsAutomaticGraphicsSwitching</key><true/>
</dict>
</plist>
PLIST
/usr/bin/plutil -lint "$C/Info.plist" >/dev/null

cp LICENSE NOTICE THIRD_PARTY_NOTICES.md "$R/"
[ -d licenses ] && cp -R licenses "$R/licenses"

# ---- 6. compile bytecode (the bundle is read-only and signed: nothing may be written later) ----
echo "-- compiling bytecode"
find "$R/python" -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
# -s/-p: record the install location (/Applications/...) in the .pyc, not this machine's temp dir
"$PY" -m compileall -q -j 0 --invalidation-mode unchecked-hash \
  -s "$R/python" -p /Applications/ShotSieve.app/Contents/Resources/python \
  -f "$R/python/lib/python3.11" >/dev/null 2>&1 || true

# ---- 7. relocation checks: nothing may reference this machine ------------------------------
echo "-- relocation checks"
BAD=0
# (a) linked libraries: only system libs and @rpath/@loader_path/@executable_path
while IFS= read -r f; do
  /usr/bin/file -b "$f" | grep -q 'Mach-O' || continue
  # LC_LOAD_DYLIB / LC_RPATH only (a library's own install name is not a dependency)
  otool -l "$f" 2>/dev/null | awk '/cmd LC_(LOAD|LOAD_WEAK|REEXPORT)_DYLIB/{g=1} g&&/^ +name /{print $2; g=0} /cmd LC_RPATH/{r=1} r&&/^ +path /{print $2; r=0}' \
    | grep -vE '^(/usr/lib/|/System/Library/|@rpath|@loader_path|@executable_path)' | sed "s|^|   $f -> |" | grep . && BAD=1 || true
done < <(find "$APP" -type f \( -name '*.so' -o -name '*.dylib' -o -perm -u+x \))
# (b) text and binary files naming this machine
# (wheels built on CI runners legitimately carry their own /Users/runner/... strings: only THIS
# machine's user, repo and temp folders count)
PATTERN="$ROOT|$WORK|${WORK#/private}|shotsieve-release\\."
# The build user's home only counts on a personal machine. On a CI runner (GitHub Actions sets
# CI=true and the user is "runner") every third-party wheel already contains /Users/runner/...
# from its own build, so that generic home would flag the whole bundle.
if [ "${CI:-}" != "true" ] || [ "$(id -un)" != "runner" ]; then
  PATTERN="$PATTERN|/Users/$(id -un)|$HOME"
fi
HITS="$(grep -rIlE "$PATTERN" "$APP" 2>/dev/null || true)"
if [ -n "$HITS" ]; then echo "$HITS" | sed 's/^/   path in text: /'; BAD=1; fi
HITS="$(grep -rlaE "$PATTERN" "$APP" --include='*.so' --include='*.dylib' --include='python3.11' --include='ShotSieve' --include='*.pyc' 2>/dev/null || true)"
if [ -n "$HITS" ]; then echo "$HITS" | sed 's/^/   path in binary: /'; BAD=1; fi
[ "$BAD" = 0 ] || { echo "FAILED: the bundle still refers to this machine (see above)." >&2; exit 1; }
echo "   clean"

# (c) every shared library shipped in the app must be named in THIRD_PARTY_NOTICES.md (licence coverage)
MISSING=0
while IFS= read -r lib; do
  b="$(basename "$lib")"
  case "$b" in libpython3.11.dylib) continue ;; esac
  grep -qF "$b" THIRD_PARTY_NOTICES.md || { echo "   not in THIRD_PARTY_NOTICES.md: ${lib#$APP/}"; MISSING=1; }
done < <(find "$R/python/lib/python3.11/site-packages" \( -name '*.dylib' \) | sort)
[ "$MISSING" = 0 ] || { echo "FAILED: a shipped library has no licence entry; update scripts/third_party_dylibs.py." >&2; exit 1; }
[ -f "$R/licenses/GPL-2.0.txt" ] && [ -f "$R/licenses/MPL-2.0.txt" ] && [ -d "$R/licenses/python-build-standalone" ] \
  && [ -d "$R/licenses/dylibs" ] || { echo "FAILED: licence texts missing from the bundle." >&2; exit 1; }
echo "   every shipped library is listed in THIRD_PARTY_NOTICES.md"

# ---- 8. ad-hoc signing --------------------------------------------------------------------------
echo "-- ad-hoc code signing"
sign_binaries
codesign --force --deep -s - "$APP"
codesign --verify --deep --strict "$APP"
echo "   signature valid (ad-hoc)"

# ---- 9. publish to dist/ ---------------------------------------------------------------------
mkdir -p "$DIST"
rm -rf "$DIST/ShotSieve.app"
/bin/cp -R "$APP" "$DIST/ShotSieve.app"
APP_KB="$(du -sk "$DIST/ShotSieve.app" | awk '{print $1}')"
echo "   dist/ShotSieve.app: $((APP_KB / 1024)) MB"

# ---- 10. dmg ---------------------------------------------------------------------------------
if [ "${SKIP_DMG:-0}" = 1 ]; then echo "SKIP_DMG=1: done."; exit 0; fi
echo "-- dmg"
DMG="$DIST/ShotSieve-$VERSION-arm64.dmg"
rm -f "$DMG" "$DMG.sha256"
STAGE="$WORK/dmg"
mkdir -p "$STAGE/.background"
/bin/cp -R "$APP" "$STAGE/ShotSieve.app"
ln -s /Applications "$STAGE/Applications"
"$PY" "$ROOT/scripts/make_dmg_background.py" "$STAGE/.background/background.png"
RW="$WORK/rw.dmg"
hdiutil create -quiet -srcfolder "$STAGE" -volname ShotSieve -fs HFS+ -format UDRW -ov "$RW"
# Mount at the standard, neutral /Volumes/ShotSieve: Finder stores the volume path of the background
# picture in .DS_Store, so a mount point under this build's temp folder would leak that folder.
[ ! -e /Volumes/ShotSieve ] || { echo "FAILED: /Volumes/ShotSieve is in use; eject it and run the build again." >&2; exit 1; }
hdiutil attach -quiet -readwrite -noverify -noautoopen -mountpoint /Volumes/ShotSieve "$RW"
# window layout: written with Finder when a GUI session allows it (best effort, 20 s limit)
(
  /usr/bin/osascript <<OSA >/dev/null 2>&1 &
tell application "Finder"
  tell disk "ShotSieve"
    open
    set current view of container window to icon view
    set toolbar visible of container window to false
    set statusbar visible of container window to false
    set the bounds of container window to {200, 150, 860, 550}
    set o to the icon view options of container window
    set arrangement of o to not arranged
    set icon size of o to 112
    set text size of o to 13
    set background picture of o to file ".background:background.png"
    set position of item "ShotSieve.app" of container window to {170, 190}
    set position of item "Applications" of container window to {490, 190}
    close
  end tell
end tell
OSA
  OSA_PID=$!
  for _ in $(seq 1 20); do kill -0 "$OSA_PID" 2>/dev/null || break; sleep 1; done
  kill "$OSA_PID" 2>/dev/null || true
) || true
sync; sleep 1
hdiutil detach -quiet /Volumes/ShotSieve || hdiutil detach -quiet -force /Volumes/ShotSieve
# Finder writes .DS_Store again at detach and records the path of the backing image (this build's
# temp folder) in it: re-attach without Finder, overwrite that path in place with a neutral string of
# the same length (ASCII and UTF-16 forms) so the record stays valid, and detach.
SCRUB="$WORK/scrub"; mkdir -p "$SCRUB"
hdiutil attach -quiet -readwrite -noverify -noautoopen -nobrowse -mountpoint "$SCRUB" "$RW"
if [ -f "$SCRUB/.DS_Store" ]; then
  "$PY" - "$SCRUB/.DS_Store" "$WORK" <<'PYS'
import sys
f, work = sys.argv[1], sys.argv[2]
data = open(f, "rb").read()
import os
base = os.path.basename(work)
for path in (work, work.removeprefix("/private"), base):
    for enc in ("ascii", "utf-16-be", "utf-16-le"):
        old = path.encode(enc)
        new = (("/tmp/" if path != base else "") + "x" * len(path)).encode(enc)[: len(old)]
        data = data.replace(old, new)
open(f, "wb").write(data)
PYS
fi
sync
hdiutil detach -quiet "$SCRUB" || hdiutil detach -quiet -force "$SCRUB"
hdiutil convert -quiet "$RW" -format "${DMG_FORMAT:-UDZO}" -imagekey zlib-level=9 -ov -o "$DMG"

# ---- 11. leak check of the whole .dmg (the app AND .DS_Store, .background, hidden files) --------
echo "-- scanning the whole .dmg for this machine's identity"
CHK="$WORK/dmgchk"; mkdir -p "$CHK"
hdiutil attach -quiet -readonly -noverify -noautoopen -nobrowse -mountpoint "$CHK" "$DMG"
HOSTS=""
for h in "$(scutil --get LocalHostName 2>/dev/null || true)" "$(scutil --get ComputerName 2>/dev/null || true)" "$(hostname -s 2>/dev/null || true)"; do
  [ "${#h}" -ge 4 ] && HOSTS="$HOSTS|$h"
done
DMG_PATTERN="$PATTERN$HOSTS"
BAD=0
HITS="$(grep -rlaE -- "$DMG_PATTERN" "$CHK" 2>/dev/null || true)"
if [ -n "$HITS" ]; then
  echo "$HITS" | sed "s|^$CHK/|   in dmg: |"
  echo "$HITS" | head -5 | while IFS= read -r h; do grep -aoE -- "$DMG_PATTERN" "$h" | sort -u | head -3 | sed 's/^/      matched: /'; done
  BAD=1
fi
# outside the app: no temp-folder path, no build-user folder at all
HITS="$(find "$CHK" -path "$CHK/ShotSieve.app" -prune -o -type f -print0 | xargs -0 grep -laE -- "/var/folders/|shotsieve-release|$DMG_PATTERN" 2>/dev/null || true)"
if [ -n "$HITS" ]; then echo "$HITS" | sed "s|^$CHK/|   in dmg (outside the app): |"; BAD=1; fi
hdiutil detach -quiet "$CHK" || hdiutil detach -quiet -force "$CHK"
[ "$BAD" = 0 ] || { rm -f "$DMG"; echo "FAILED: the .dmg refers to this machine (see above); it was deleted." >&2; exit 1; }
echo "   clean"
( cd "$DIST" && shasum -a 256 "$(basename "$DMG")" > "$(basename "$DMG").sha256" )
DMG_MB=$(( $(stat -f %z "$DMG") / 1048576 ))
echo "   $DMG: $DMG_MB MB"
cat "$DMG.sha256"
[ "$DMG_MB" -lt 2000 ] || { echo "WARNING: the dmg is over the 2 GB GitHub release asset limit." >&2; exit 1; }
echo "== done =="
