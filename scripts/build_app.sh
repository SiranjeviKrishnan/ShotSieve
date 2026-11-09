#!/bin/bash
# Build ShotSieve.app next to this repo: a plain, unsigned bundle whose launcher has ROOT baked in
# and exec()s `shotsieve app` (a native pywebview window).
# Local build => no quarantine attribute. Writes only $ROOT/ShotSieve.app.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
APP="$ROOT/ShotSieve.app"
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || { echo "Run install.sh first (no .venv at $ROOT)." >&2; exit 1; }

VERSION="$("$PY" -c 'import shotsieve; print(shotsieve.__version__)')"
STAGE="$(mktemp -d "$ROOT/.app-build.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT
B="$STAGE/ShotSieve.app/Contents"
mkdir -p "$B/MacOS" "$B/Resources"

cat > "$B/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>ShotSieve</string>
  <key>CFBundleDisplayName</key><string>ShotSieve</string>
  <key>CFBundleExecutable</key><string>ShotSieve</string>
  <key>CFBundleIdentifier</key><string>local.shotsieve.app</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleIconFile</key><string>ShotSieve</string>
  <key>CFBundleShortVersionString</key><string>$VERSION</string>
  <key>CFBundleVersion</key><string>$VERSION</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>LSApplicationCategoryType</key><string>public.app-category.photography</string>
  <key>NSSupportsAutomaticGraphicsSwitching</key><true/>
</dict>
</plist>
PLIST

# launcher with the absolute ROOT substituted for the placeholder line
"$PY" - "$ROOT" "$ROOT/scripts/launcher.sh" "$B/MacOS/ShotSieve" <<'PYEOF'
import shlex, sys
root, src, dst = sys.argv[1:]
text = open(src).read()
assert "ROOT='@ROOT@'" in text
open(dst, "w").write(text.replace("ROOT='@ROOT@'", "ROOT=" + shlex.quote(root), 1))
PYEOF
chmod +x "$B/MacOS/ShotSieve"
bash -n "$B/MacOS/ShotSieve"

# icon: Pillow -> PNG -> iconset (sips) -> icns (iconutil); the icon is cosmetic, never fatal
ICON_OK=0
if "$PY" "$ROOT/scripts/make_icon.py" "$STAGE/icon.png" 2>/dev/null; then
  SET="$STAGE/ShotSieve.iconset"; mkdir -p "$SET"
  ok=1
  for sz in 16 32 128 256 512; do
    /usr/bin/sips -z $sz $sz "$STAGE/icon.png" --out "$SET/icon_${sz}x${sz}.png" >/dev/null 2>&1 || ok=0
    /usr/bin/sips -z $((sz * 2)) $((sz * 2)) "$STAGE/icon.png" --out "$SET/icon_${sz}x${sz}@2x.png" >/dev/null 2>&1 || ok=0
  done
  if [ "$ok" = 1 ] && /usr/bin/iconutil -c icns "$SET" -o "$B/Resources/ShotSieve.icns" 2>/dev/null; then
    ICON_OK=1
  fi
fi
[ "$ICON_OK" = 1 ] || echo "Note: app icon not created (cosmetic)." >&2

/usr/bin/plutil -lint "$B/Info.plist" >/dev/null
rm -rf "$APP"
mv "$STAGE/ShotSieve.app" "$APP"
echo "Built $APP"
